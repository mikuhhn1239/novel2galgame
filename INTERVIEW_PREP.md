# 面试准备：长篇小说 RAG 知识增强系统

> 项目定位：RAG 驱动的长文本 LLM 内容生成系统。核心卖点是 RAG 设计 + 评测体系 + 模型训练。

---

## 一、30 秒电梯演讲

> "我独立开发了一个 RAG 驱动的长篇内容生成系统。核心挑战是 90 章百万字小说无法装入单次 LLM 上下文窗口——Agent 逐章处理时，第 50 章需要第 1 章的角色信息。我设计了一套增量 RAG 知识管线，Agent 每章处理后自动提取知识按六种语义维度分块入库，知识库随管线推进增量生长。检索端做了 ChromaDB 向量、BM25 稀疏、元数据精确匹配三路召回加 RRF 融合和 Cross-Encoder 精排。建立了 30 条 × 5 类评测集，Hit@1 70%、Hit@5 90%、MRR 0.77。另外在 8×A800 上训练了 Qwen3-8B 和 3 个 LoRA，Attribution 准确率 86.7%."

---

## 二、项目全景图

### 核心流程

```
小说 .txt
  │
  ├─ Structure Agent (L0 规则) → 章节识别
  ├─ Narrative Agent (L2 LLM)  → 段落分类
  │     └─ RAG 注入: 已知角色列表
  ├─ Attribution Agent (L2 LLM/LoRA) → 谁说了什么
  │     └─ RAG 注入: 角色知识检索 → RAG 写入: 提取的角色知识入库
  ├─ Segmentation Agent (L2 LLM/LoRA) → 场景边界
  │     └─ RAG 注入: 场景模式检索 → RAG 写入: 场景结构入库
  ├─ VN Mapping Agent → 文本转 VN 脚本
  ├─ Visual Prompt Agent → 角色/背景 prompt
  │     └─ RAG 注入: 角色外观检索
  └─ Fidelity Agent → 忠实度审查
```

每个 Agent 处理时通过 RAG 实时检索前序章节的知识，同时将新知识写入知识库。

### 技术栈

| 类别 | 选型 |
|------|------|
| 编排 | LangGraph StateGraph + checkpoint |
| 向量存储 | ChromaDB HNSW 索引 |
| 嵌入 | bge-small-zh-v1.5 (512-dim) |
| 多路召回 | 稠密向量 + 稀疏 BM25 + 元数据精确匹配 → RRF 融合 |
| 精排 | bge-reranker-large Cross-Encoder → LLM 终排 (top-3) |
| 分块 | 层次化分块 (6 种子块 + 父文档召回) |
| 模型 | Qwen3-8B SFT + 3×LoRA (8×A800, 669 本小说) |
| 后端 | Node.js + Express + SQLite |
| 前端 | React 19 + Vite 6 + Tailwind CSS |

---

## 三、RAG 设计全流程（核心亮点）

### 3.1 问题定义

90 章长篇小说百万字，单次 LLM 上下文窗口只能装 1-2 章。Agent 逐章处理时：
- 第 50 章的 Attribution Agent 需要第 1 章的角色外貌
- 第 3 章的角色只透露了外貌，第 10 章才透露关系和性格
- 第 50 章不能"偷看"第 80 章的角色信息（时间穿越式标签泄露）

### 3.2 数据库选型

| 方案 | 向量搜索 | 元数据过滤 | 零依赖 | 结论 |
|------|---------|-----------|--------|------|
| ChromaDB | HNSW | ✅ where 子句 | 独立进程 | **首选** |
| LanceDB | IVF-PQ | ✅ SQL | 嵌入式 | Node.js 绑定不成熟 |
| FAISS | 极致性能 | ❌ 不支持 | native addon | 不适合 |
| Qdrant | HNSW | payload filter | 需 Docker | 太重 |

选择 ChromaDB：pnpm add chromadb 即用，支持 where 元数据过滤，HNSW 索引毫秒级检索。

### 3.3 分块策略

采用层次化分块，子块独立嵌入、父文档完整召回：

| chunkType | 内容 | 用途 |
|-----------|------|------|
| identity | 角色名、别名、首次出现章 | 别名查询、角色发现 |
| appearance | 发型、服装、瞳色等外貌特征 | Attribution Agent 外观匹配 |
| personality | 性格特点、行为模式 | 性格一致性检查 |
| relationship | 与其他角色的关系 | 关系推理、对话归因 |
| quote | 该角色的典型台词和口吻 | 说话风格识别 |
| summary | 跨章节聚合的角色全貌 | 长文本压缩表示 |

每个子块独立嵌入存入 ChromaDB，检索命中子块后通过 parentDocId 返回完整父文档。

### 3.4 检索管线（四阶段）

```
查询 "长发的女生"
  │
  ├─ 阶段 1: 三路并行召回
  │   ├─ 路 1: ChromaDB HNSW 稠密向量 (nResults=20)
  │   ├─ 路 2: BM25 稀疏关键词 (全文搜索)
  │   └─ 路 3: 元数据精确匹配 (where chunkType=appearance)
  │
  ├─ 阶段 2: RRF 融合排序
  │   rrfScore = Σ 1/(k + rank_i) , k=60
  │
  ├─ 阶段 3: Cross-Encoder 精排
  │   bge-reranker-large 对 (query, document) 对打分
  │
  └─ 阶段 4: LLM 终排 (仅 top-3)
      粗筛 20 → RRF 融合 → CE 精筛 10 → LLM 终筛 3
```

### 3.5 评测体系

**评测集设计**：30 条查询 × 5 类场景

| 类别 | 示例查询 | 条数 | 测试目标 |
|------|---------|------|---------|
| exact | "长发的女生" | 8 条 | 精确关键词匹配 |
| semantic | "送伞的人"（chunk 中是 "送伞送早餐"） | 8 条 | 语义等价检索 |
| relational | "林秋和苏雨晴是什么关系" | 6 条 | 关系推理检索 |
| cross_chapter | "苏雨晴的全部外貌描述" | 5 条 | 跨章节信息聚合 |
| edge | "那个人"、"不存在的角色张三" | 3 条 | 边界/空结果处理 |

**Ground Truth**：人工标注每条查询应当返回的精确 chunk ID 列表。

**消融实验**：keyword-only vs bigram-enhanced 对比。

**评测结果**：

| 指标 | 数值 |
|------|------|
| Hit@1 | 70.0% |
| Hit@5 | 90.0% |
| MRR | 0.7678 |
| exact 类别 Hit@5 | 100% |
| semantic 类别 Hit@5 | 100% |
| relational 类别 Hit@5 | 83% |
| cross_chapter 类别 Hit@5 | 100% |
| edge 类别 Hit@5 | 33%（预期：2 条应返回空） |

### 3.6 遇到的问题与解决方案

| 问题 | 现象 | 根因 | 解决方案 | 效果 |
|------|------|------|---------|------|
| 语义查询 keyword 失效 | "送伞的人" 匹配不到 chunk "送伞送早餐" | 单字匹配将双字词组 "送伞" 拆散为 "送"+"伞" | bge-large-zh 向量检索 | 语义维度全覆盖 |
| bigram 引入噪声 | MRR 从 0.7678 仅提升至 0.7733 | 跨词 bigram "善但" "了男" 误匹配无关 chunk | Cross-Encoder 精排过滤 | 消除噪声 |
| summary 被抢 rank0 | "苏雨晴的全部外貌描述" → identity chunk 排第一 | 所有 chunkType 权重均等 | chunkType 加权 (summary > relationship > 其他) | summary 排名提升 |
| 时序标签泄露 | 第 3 章检索到第 8 章写入的角色信息 | 无时序约束 | $lte $ne 元数据过滤 | 杜绝时间穿越 |
| 评测数据难以构造 | 需要人工标注每条查询的期望 chunk | ground truth 依赖对人判断 | 分 5 类设计查询，涵盖精确到边界的完整场景 | 30 条标准化评测集 |
| 消融实验结论模糊 | keyword 和 bigram 的 MRR 几乎相同 | 两种方法对 exact 查询都有效 | 按类别分组统计，定位 semantic/relational 为真正短板 | 精确的改进方向 |

---

## 四、其他技术亮点

### 4.1 Pipeline 编排与韧性

LangGraph StateGraph 编排 7 Agent 工作流，条件边 + checkpoint 断点续跑。SHA256 语义缓存省 80% API 调用，指数退避重试 (3 次) + 三级失败策略 (Soft/Recoverable/Hard) + AbortController 取消。Agent 全链路可观测 (duration/token/retry)。

### 4.2 领域模型微调

669 本小说构建训练集 (7200 万字)，8×A800 完成 Qwen3-8B 全参微调 (DeepSpeed ZeRO-2, seq_len=2048, ~9h) + 3 个 LoRA (r=64 α=128)。核心洞察：短 prompt 比长 prompt 好 8pp；reasoning 链让小模型套模板；8B+SFT 场景边界天花板 ~30%。Attribution 86.7%，HuggingFace 发布，bitsandbytes 4-bit 量化 RTX 4060 部署。

---

## 五、模拟面试问答

### Q1: 为什么你的项目需要 RAG？直接用更大的上下文窗口不行吗？

> "90 章百万字远超任何模型的上下文窗口。即使未来有百万 token 窗口，把所有章节塞进去也不是好方案——长上下文会导致注意力稀释，LLM 对中间位置的信息 recall 显著下降。RAG 不是临时补丁，是这个场景的架构选择：按需检索相关信息，让 Agent 只关注当前推理需要的知识。"

### Q2: 你的 RAG 和普通的"切文档 + 调 embedding API"有什么不同？

> "三个核心区别。第一，增量写入——不是一次性灌入所有文档，Agent 每章处理后写一次，知识库随管线生长。第二，语义分块——不是按字数切，是按语义维度切，appearance 和 personality 独立嵌入。第三，时序元数据过滤——必须防止 Agent 在处理第 3 章时检索到第 8 章的信息。"

### Q3: 为什么选 ChromaDB 而不是 Pinecone / Milvus / Weaviate？

> "约束决定的。产品是本地工作台，不能依赖外部服务。ChromaDB 是唯一同时满足'本地嵌入式运行 + HNSW 索引 + 元数据过滤 + Node.js 绑定'四个条件的选择。Pinecone 是 SaaS，Milvus 要 Docker，FAISS 不支持元数据过滤。"

### Q4: 你怎么评测 RAG 的效果？不是只测了 retrieval 的指标吗？

> "两层评测。检索层测 recall——30 条 × 5 类查询统计 Hit@K 和 MRR。业务层测实际影响——对比有 RAG 和无 RAG 的管线输出质量，场景切分 67% → 73%，这个 +7% 是 RAG 的实际业务价值。两层指标互补：检索指标告诉我哪里有问题，业务指标告诉我问题有多严重。"

### Q5: 评测数据怎么构造的？Ground truth 怎么保证客观？

> "五类查询覆盖不同难度。精确特征查询的 ground truth 是明确的——包含关键词的 chunk 就是正确答案。语义查询的 ground truth 由我根据推理标注——这个有主观性，但 30 条的规模可以通过多次标注取多数来去偏。消融实验的对照组设计是客观的——同样的查询、同样的 chunk、不同的检索策略，差异归因清晰。"

### Q6: 你的分块策略为什么选 6 种语义类型？多了还是少了？

> "6 种是根据 Agent 的实际需求反推的。Attribution Agent 需要 appearance 确认外貌、personality 确认性格、relationship 确认关系——三个维度。Visual Prompt Agent 需要 summary 获取角色全貌。quote 帮 Attribution 通过说话风格辅助判断。identity 解决别名查询。少于 6 个会漏需求，多于 6 个会稀释每个 chunk 的语义密度。"

### Q7: RRF 和加权平均有什么区别？为什么选 RRF？

> "加权平均需要调 weight 参数，而且 vector score 和 keyword score 的量纲不同——一个 0.95 的向量分和一个 0.3 的关键词分怎么加权全靠猜。RRF 只关心排名，不需要归一化。业界标准做法，我用的 k=60。"

### Q8: 遇到的最难问题是什么？怎么解决的？

> "时序标签泄露最隐蔽。测试时发现第 3 章的归因结果莫名其妙地准确——原来 RAG 检索到了第 8 章写入的完整角色人设。这是严重的数据泄露。修复方案是在检索时加 $lte 和 $ne 元数据过滤，只检索 chapterId ≤ 当前章 且 ≠ 当前章的信息。这个修复是后来加的评测才发现的——没有回归评测根本不会注意到。"

### Q9: 如果重新设计 RAG，你会改什么？

> "两个改进。第一，用 bge-large-zh 替换 bge-small-zh——512-dim 到 1024-dim 的质量提升值得额外的计算开销。第二，加一个检索质量监控 dashboard，每次查询记录 latency、召回数、top-3 score，发现异常自动告警。目前的效果评估是靠离线评测，online 监控是缺失的。"

### Q10: 面试官问 "你做了什么 RAG" 时的最佳回答框架

> "两分钟结构化回答：问题（太长装不下）→ 设计（增量写入 + 语义分块 + 时序过滤）→ 检索（三路召回 + RRF + CE 精排）→ 评测（30 条 × 5 类，Hit@5 90%）→ 遇到最难的（时序泄露，加元数据过滤解决）。这五个点讲完，面试官就不会再问'你做的 RAG 跟别人有什么不同'。"

---

## 六、简历项目经历

### 版本 A：大模型应用开发 / AI Engineer

### 长篇小说智能处理与 RAG 知识增强系统
*独立开发 | 2026.06 — 2026.07 | LangGraph / TypeScript / ChromaDB / Python*

**项目简介**：面向长篇小说篇幅过长、单次 LLM 上下文窗口仅能容纳 1-2 章内容、跨章节知识遗忘严重的问题，设计并实现一套基于 RAG 的跨章节知识管理系统。支持逐章增量知识抽取与入库、多路混合检索、语义重排与检索效果回归评测。

**技术栈**：LangGraph、ChromaDB、bge-large-zh-v1.5、BM25、RRF、bge-reranker-large、Qwen3-8B SFT + LoRA、TypeScript、Node.js、React

- 为解决逐章处理时 Agent 无法访问已处理章节信息的问题，设计增量 RAG 知识管线。Attribution Agent 每章推理完成后自动提取角色知识，按 identity、appearance、personality、relationship、quote、summary 六种语义维度分块入库。Agent 既是 RAG 的消费者也是生产者，知识库随管线推进增量生长。

- 为按语义维度精确召回信息，采用层次化分块策略：子块独立嵌入用于高精度检索，父文档用于完整上下文注入。不同 Agent 检索不同维度的知识互不干扰。增量写入时同一角色的新信息追加为新子块而非覆盖旧数据，保留角色信息在全书中渐进揭露的时序特征。

- 为提升检索精度并防止时间穿越式信息泄露，实现四阶段检索管线：稠密向量 ChromaDB HNSW 索引、稀疏 BM25 关键词、元数据精确匹配三路并行召回，RRF 融合排序，bge-reranker-large Cross-Encoder 精排，LLM 对 top-3 候选做终排。元数据过滤增加时序约束，确保 Agent 处理早期章节时不会检索到后期章节才写入的角色信息。

- 建立多指标评测体系。构建 30 条查询 × 5 类的评测集，覆盖精确特征、语义推理、关系查询、跨章节聚合和边缘边界五种检索场景。统计 Hit@K、MRR、NDCG@K 指标，通过消融实验对比不同检索策略。Hit@1 70.0%，Hit@5 90.0%，MRR 0.7678，定位 semantic 和 relational 类别为主要短板，分析 Cross-Encoder 精排和 chunkType 加权的改进方向。

- 669 本中文小说构建领域训练集，总计 7200 万字符，8×A800 完成 Qwen3-8B 全参微调及 3 个 LoRA Adapter 训练，Attribution 准确率 86.7%。将 SFT 模型部署为前 3 个 Agent 的本地推理后端，bitsandbytes 4-bit 量化部署至 RTX 4060，HuggingFace 公开发布。

- LangGraph StateGraph 编排七步工作流，SHA256 语义缓存节省 80% API 调用，指数退避重试与三级失败策略保证管线韧性，90 章长篇小说端到端稳定运行。

---

### 版本 B：精简版（A4 一页用）

### 长篇小说 RAG 知识增强系统
*独立开发 | 2026.06 — 2026.07 | LangGraph / TypeScript / ChromaDB / Python*

- 面向 90 章百万字小说无法装入单次 LLM 上下文窗口的问题，设计增量 RAG 系统。Agent 每章处理后自动提取角色知识按六种语义维度分块入库，知识库随管线推进增量生长。三路召回并行，RRF 融合，Cross-Encoder 精排。建立 30 条 × 5 类评测集对比消融实验，Hit@1 70%、Hit@5 90%、MRR 0.77。

- LangGraph 编排 7 Agent 工作流，3 Agent 支持本地 LoRA 与云端 API 混合调用。SHA256 语义缓存、三级失败策略、90 章管线稳定运行。

- 669 本小说于 8×A800 完成 Qwen3-8B 全参微调及 3 个 LoRA，Attribution 准确率 86.7%，HuggingFace 发布。TypeScript Monorepo，11 packages，IR 多端架构。

---

## 七、数字速查

| 指标 | 数值 |
|------|------|
| RAG Hit@1 | 70.0% |
| RAG Hit@5 | 90.0% |
| RAG MRR | 0.7678 |
| 评测集规模 | 30 条 × 5 类 × 26 chunk |
| 场景切分提升 | +7% (67% → 73%) |
| LoRA Attribution | 86.7% |
| LoRA Narrative | 72.8% |
| 训练数据 | 669 本, 7200 万字, 8×A800 |
| 缓存节省 | 80% API 调用 |
| 代码规模 | 11 packages, TypeScript Monorepo |

## 八、加分金句

1. "RAG 不是锦上添花——90 章百万字，上下文窗口装不下，这是架构选择。"
2. "Agent 既是 RAG 消费者也是生产者——知识库随管线推进增量生长。"
3. "四阶段检索：三路召回 → RRF 融合 → CE 精排 → LLM 终排。不是'调个 embedding API'。"
4. "时序元数据过滤——处理第 3 章时不能看第 8 章的信息，这是评测中发现的隐蔽 bug。"
5. "30 条 × 5 类评测集，按类别分组统计才能知道真正短板在哪——relational 查询是弱项。"
