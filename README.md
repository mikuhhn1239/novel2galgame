# All Novel Can Be Galgame

将中文恋爱向 txt 小说一键转化为可玩视觉小说 (Galgame) 的本地 AI 工作台。

> 不是创意改写工具，是忠实的叙事转译器——保留原文剧情、对话、人物关系与情感基调。

## 核心管线

```
txt 小说 → Structure → Narrative Parsing → Attribution → Scene Segmentation
    → VN Mapping + Visual Prompt (并行) → Fidelity Review → Consistency Review → 可玩预览
```

9 个 Agent 组成的流水线，每章自动执行：

| 阶段 | 能力层级 | 说明 |
|------|---------|------|
| Structure | L0 规则 | 章节识别、文本清洗 |
| Narrative Parsing | L2 语义 | 叙事单元分类 (对话/叙述/心理/动作/场景) |
| Attribution | L2 语义 | 角色归属标注 |
| Scene Segmentation | L2 语义 | 场景切分 |
| VN Mapping | L2 语义 | 叙事单元 → VN 脚本步骤 |
| Visual Prompt | L2 语义 | 生成立绘/背景提示词 |
| Fidelity Review | L2 语义 | 忠实度审核 |
| Consistency Review | L2 语义 | 跨章节一致性检查 |

## 技术栈

- **Monorepo:** pnpm workspaces + Turborepo
- **后端:** Node.js + Express + SQLite (better-sqlite3)
- **前端:** React 19 + Vite 6 + Tailwind CSS 4 + Zustand + TanStack Query
- **VN 引擎:** 自研步骤执行器，8 种步骤类型
- **LLM:** 兼容 OpenAI API 的任意模型 (已验证: NVIDIA/Kimi, MiMo v2.5, Agnes AI)

## 项目结构

```
apps/
  api/          Node.js REST API
  workbench/    React SPA 工作台
packages/
  core/         领域模型与 TypeScript 接口
  agents/       9 个 AI Agent 实现
  runtime/      VN 播放引擎
  providers/    LLM + 图像生成 Provider
  storage/      SQLite 索引 + 文件系统存储
  evaluation/   评测框架
docs/           设计文档 (中文)
data/           项目数据、评测数据集
```

## 快速开始

```bash
# 安装依赖
pnpm install

# 构建
pnpm build

# 配置 LLM API
cp apps/api/.env.example apps/api/.env
# 编辑 .env，填入 API Key、Base URL、模型名

# 启动 API
cd apps/api && npx tsx src/index.ts

# 启动前端
cd apps/workbench && pnpm dev
```

## API 示例

```bash
# 创建项目
curl -X POST http://localhost:3001/projects \
  -H "Content-Type: application/json" \
  -d '{"title": "我的小说"}'

# 导入小说
curl -X POST http://localhost:3001/projects/{id}/import \
  -F "file=@novel.txt"

# 运行结构识别
curl -X POST http://localhost:3001/projects/{id}/structure/run

# 运行章节管线 (完整管线: 解析→归属→切分→VN映射→忠实度审核)
curl -X POST http://localhost:3001/projects/{id}/chapters/{chapterId}/run
```

## VN 脚本格式

每章生成固定 8 种步骤类型的 VN 脚本：

| 步骤类型 | 说明 | 关键字段 |
|---------|------|---------|
| `bg` | 背景切换 | backgroundId, backgroundLabel |
| `show` | 显示角色 | characterId, expression, position |
| `hide` | 隐藏角色 | characterId |
| `narration` | 旁白 | text |
| `say` | 角色对话 | characterId, displayName, text |
| `thought` | 内心独白 | characterId, displayName, text |
| `pause` | 暂停 | durationMs |
| `transition` | 转场 | name (fade/cut/dissolve) |

## 设计约束

- 对话保留率 >= 95%
- 非原文添加量 <= 5%
- 三级状态机: Project (9态) / Chapter (8态) / Scene (6态)
- 混合存储: SQLite 索引 + 文件系统内容

## 设计文档

详见 [`docs/`](docs/) 目录，包含 10 份中文设计规格文档。

## License

Private / TBD
