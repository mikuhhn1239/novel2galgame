# All Novel Can Be Galgame - 开发进度

## 项目概览

将中文恋爱向 txt 小说转换为可玩的视觉小说 (Galgame) 的本地 AI 工作台。

**技术栈:** TypeScript / pnpm monorepo / Turborepo / React / Node.js / SQLite + 文件系统

---

## 开发阶段

### Phase 1: 文本主链路

#### 1.1 packages/core - 领域模型与数据结构 ✅

**状态:** 已完成  
**日期:** 2026-06-02  

**已完成内容:**

| 模块 | 文件 | 内容 |
|------|------|------|
| monorepo 根配置 | `package.json`, `pnpm-workspace.yaml`, `turbo.json`, `.gitignore` | pnpm + Turborepo 工作区 |
| core 包配置 | `packages/core/package.json`, `tsconfig.json` | Zod 依赖, ES2022 target |
| 领域类型 | `src/domain/` (12 文件) | 全部 TypeScript 接口定义 |
| Zod Schema | `src/schemas/` (10 文件) | 运行时校验 schema |
| 常量 | `src/constants/` (3 文件) | ID 生成、文件名、状态枚举、管线顺序 |

**领域类型清单:**

- `project.ts` - ProjectConfig, ProjectState, ProjectManifest (含 9 种 ProjectStatus)
- `structure.ts` - ChapterMeta, StructureResult
- `chapter.ts` - ChapterState (含 8 种 ChapterStatus), ChapterSource
- `narrative.ts` - NarrativeUnit (5 种类型: dialogue/narration/thought/action/scene_description), NarrativeParsingResult
- `attribution.ts` - CharacterRef, AttributionInfo, AttributedNarrativeUnit, AttributionResult
- `scene.ts` - Scene (含 6 种边界原因), SceneState (含 6 种 SceneStatus), SegmentationResult
- `vn-script.ts` - 8 种 VNStep (bg/show/hide/narration/say/thought/pause/transition), VNScript, UnitToStepMap
- `fidelity.ts` - FidelityIssue (6 种类型 x 3 种严重度), FidelityReport
- `visual-prompt.ts` - VisualEvidence, CharacterPromptPack, BackgroundPromptPack, VisualPromptResult
- `consistency.ts` - ConsistencyIssue (5 种类型), ConsistencyReport
- `task.ts` - TaskRecord (8 种 TaskType x 5 种 TaskStatus), CacheKey, CacheEntry
- `store.ts` - AppStore (前端状态管理)

**构建验证:**

- `tsc --noEmit` 通过, 0 错误
- `tsc` 构建成功, 输出 48 个 `.d.ts` + 48 个 `.js` + source maps
- Zod schemas 编译正常

---

### Phase 1: 文本主链路 ✅

**整体状态:** 已完成  
**日期:** 2026-06-02 ~ 2026-06-03  

#### 1.2 packages/storage - 本地存储与缓存层 ✅

**状态:** 已完成  
**日期:** 2026-06-03  

**已完成内容:**

| 模块 | 文件 | 内容 |
|------|------|------|
| 包配置 | `package.json`, `tsconfig.json` | better-sqlite3 依赖, workspace 引用 @novel2gal/core |
| SQLite 数据库 | `src/db/database.ts` | 4 表 (projects/chapters/scenes/tasks) + schema_meta, WAL 模式, 索引 |
| 项目仓储 | `src/repositories/project-repo.ts` | CRUD, 按状态/章节数更新 |
| 章节仓储 | `src/repositories/chapter-repo.ts` | CRUD, 按项目列表, 状态/scene数更新 |
| Scene 仓储 | `src/repositories/scene-repo.ts` | CRUD, 按章节/项目列表, 多字段状态更新 |
| 任务仓储 | `src/repositories/task-repo.ts` | CRUD, markRunning, 按项目/章节列表 |
| 文件系统层 | `src/filesystem/project-fs.ts` | 项目目录初始化, JSON 读写, 类型化便捷写入 (Narrative/Attribution/Segmentation/VNScript/Fidelity/VisualPrompt) |
| 缓存层 | `src/cache/cache.ts` | SHA256 输入哈希, CacheKey 构建, 读/写/查找, hitCount 自增 |

**测试结果:**

```
Test dir: C:\Users\...\novel2gal-test-VFXHpt
Project created: 测试小说 - status: created
Project list count: 1
FS roundtrip: OK
Cache roundtrip: OK
All smoke tests passed!
```

- SQLite CRUD: 创建/查询/列表 通过
- 文件系统: 目录初始化 + JSON 序列化/反序列化 通过
- 缓存层: 哈希计算 + 写入/读取 通过
- `tsc --noEmit` 通过, 0 错误

#### 1.3 packages/agents/structure - Structure Agent ✅

**状态:** 已完成  
**日期:** 2026-06-03  

**已完成内容:**

| 模块 | 文件 | 内容 |
|------|------|------|
| 包配置 | `package.json`, `tsconfig.json` | 依赖 @novel2gal/core, @types/node |
| 编码检测 | `src/structure/encoding.ts` | UTF-8 BOM/UTF-16/GBK/GB18030 自动检测, CJK 含量启发式判断 |
| 文本清洗 | `src/structure/cleaner.ts` | 编码统一, 广告/平台尾部逐行过滤, 引号规范化, 空白压缩 |
| 章节检测 | `src/structure/chapter-detector.ts` | 6 种模式 (第X章/特殊标题/序号/数字/Chapter/纯数字), 多模式合并, 元数据过滤 |
| Agent 入口 | `src/structure/structure-agent.ts` | 编码->清洗->检测->置信度评估, Buffer/string 双输入, 低置信度软失败 |
| Agent 类型 | `src/shared/agent-types.ts` | AgentResult, AgentContext, AgentFailureLevel |

**章节模式支持:**
- 标准: 第X章/节/卷/回/篇/集/部 (中/阿数字) - 0.95 置信度
- 特殊: 楔子/序章/序言/前言/引子/番外/后记/尾声/终章 - 0.85 置信度
- 序号: 一、二、三、- 0.8 置信度
- 数字: 1. 2. 3. - 0.7 置信度
- 英文: Chapter X - 0.85 置信度

**测试结果:**

合成小说测试:
```
Chapters: 6 (楔子 + 3章 + 番外 + 后记)
Confidence: 0.90
Special: extra=true, afterword=true, authorNote=true
Ad removal: OK
Empty input: hard failure
```

真实小说测试 (《AI恋人》GBK 649KB):
```
Chapters: 90 (第1章 ~ 第88章)
Confidence: 0.944
Encoding: GBK auto-detected and converted
Book title: 《AI恋人》作者：妄初
```

**待实现 (L2 回退):**
- 低置信度时调用 LLM 解析章节结构 (需先实现 packages/providers)

#### 1.4 ~ 1.8 L2 Agents ✅

**状态:** 已完成  
**日期:** 2026-06-03  

| Agent | 文件 | LLM 依赖 | 输入 | 输出 |
|-------|------|----------|------|------|
| Narrative Parsing | `src/narrative-parsing/` | JSON mode | chapterId + chapterText | NarrativeParsingResult |
| Attribution | `src/attribution/` | JSON mode | chapterId + units[] | AttributionResult |
| Scene Segmentation | `src/scene-segmentation/` | JSON mode | chapterId + units[] | SegmentationResult |
| VN Mapping | `src/vn-mapping/` | JSON mode | sceneId + scene + units[] | VNScript |
| Fidelity Review | `src/fidelity-review/` | JSON mode | sceneId + vnScript + originalUnits | FidelityReport |

**共通模式:** prompt template -> LLM JSON call -> schema validation -> AgentResult

**关键特性:**
- Narrative Parsing: 长章节自动分段处理 (8000 字/段)
- Attribution: 已知角色列表传递, 不确定归属标记
- VN Mapping: 非原文添加量监控 (>5% 则标记 suspiciousExpansions)
- Fidelity Review: 6 种 issue 类型 (dialogue_rewrite/content_omission/wrong_attribution/order_changed/unsupported_addition/semantic_drift)

**packages/providers:**
- `src/interfaces/llm.ts` - LLMProvider 接口 (chat + chatJson)
- `src/llm/openai/openai-provider.ts` - OpenAI 适配器 (支持自定义 baseUrl)

#### 1.9 apps/api - 最小后端 API ✅

**状态:** 已完成  
**日期:** 2026-06-03  

**API 路由:**

| 方法 | 路径 | 功能 |
|------|------|------|
| POST | `/projects` | 创建项目 |
| GET | `/projects` | 项目列表 |
| GET | `/projects/:id` | 项目详情 |
| DELETE | `/projects/:id` | 删除项目 |
| POST | `/projects/:id/import` | 导入 txt 文件 (multer) |
| POST | `/projects/:id/structure/run` | 运行 Structure Agent |
| GET | `/projects/:id/structure` | 获取结构解析结果 |
| GET | `/projects/:id/chapters` | 章节列表 |
| POST | `/projects/:id/chapters/:id/run` | 运行章节全管线 |
| GET | `/projects/:id/tasks` | 任务列表 |
| GET | `/health` | 健康检查 |

**章节管线 (orchestrator):** Narrative -> Attribution -> Scene -> VN Mapping -> Fidelity Review

#### 1.10 管线集成测试 ✅

**测试结果:**
```
1. Health: status=200
2. Create project: status=201
3. List projects: count=1
4. Get project: title=测试小说, status=created
5. Copied test novel (634KB)
6. Structure: 90 chapters, confidence=0.944, title=《AI恋人》
7. Get structure: chapters=90
8. List chapters: count=90
9. Tasks: count=0
=== All integration tests passed! ===
```

使用真实 GBK 编码小说《AI恋人》(649KB) 端到端验证:
- 编码自动检测: GBK -> UTF-8
- 章节识别: 90 章, 置信度 0.944
- API CRUD: 创建/查询/列表/删除 全部通过
- 目录结构: normalized/ + chapters/ 自动生成

---

### Phase 2: 工作流与本地工作台 ✅

**整体状态:** 已完成
**日期:** 2026-06-04
**分支:** `phase2-workbench`

#### 2.1 API 增强 ✅

| 路由 | 功能 |
|------|------|
| GET/POST `/config/models` | 模型配置读写 |
| POST `/config/test-connection` | 连接测试 |
| GET `/projects/:id/scenes/:sceneId/script` | VN 脚本获取 |
| GET `/projects/:id/scenes/:sceneId/fidelity` | 忠实性报告 |
| GET `/projects/:id/chapters/:chapterId/narrative` | 解析结果 |
| GET `/projects/:id/chapters/:chapterId/attribution` | 归因结果 |
| GET `/projects/:id/chapters/:chapterId/segmentation` | 切分结果 |
| GET `/projects/:id/scenes/:sceneId/visual-prompt` | 视觉提示词 |
| POST `/projects/:id/scenes/:sceneId/visual-prompt/run` | 运行 Visual Prompt Agent |
| POST `/images/generate` | 图像生成 |
| GET `/images/providers` | 列出图像 provider |
| SSE `/progress` | 进度推送 |

#### 2.2 apps/workbench - React SPA 工作台 ✅

| 页面 | 路由 | 状态 |
|------|------|------|
| 项目列表 | `/` | P0 完成 |
| 新建项目 (3步向导) | `/projects/new` | P0 完成 |
| 项目总览 | `/projects/:id/overview` | P0 完成 |
| 章节管理 (三栏) | `/projects/:id/chapters` | P0 完成 |
| 场景工作区 (三栏) | `/projects/:id/scenes/:chapterId` | P0 完成 |
| 模型配置 | `/config` | P0 完成 |
| 任务日志 | `/projects/:id/tasks` | P1 完成 |
| VN 脚本 | `/projects/:id/script/:sceneId` | P1 占位 |
| 视觉提示 | `/projects/:id/prompts` | P1 完成 (Phase 3) |
| 预览播放 | `/projects/:id/preview` | P1 完成 (Phase 3) |
| 项目设置 | `/projects/:id/settings` | P1 占位 |

**技术栈:** React 19 + Vite 6 + Tailwind CSS 4 + Zustand + TanStack Query + lucide-react

**构建产物:** 355KB JS + 23.5KB CSS, 无 warning

---

### Phase 3: 预览播放器与视觉层 ✅

**整体状态:** 已完成
**日期:** 2026-06-04
**分支:** `phase2-workbench`

#### 3.1 packages/runtime - VN 播放引擎 ✅

| 模块 | 文件 | 内容 |
|------|------|------|
| 步骤执行器 | `src/step-engine/execute-step.ts` | 8 种 VNStep → RenderAction 映射 |
| 渲染动作类型 | `src/step-engine/step-types.ts` | RenderAction 联合类型 (8 种) |
| 播放控制器 | `src/player/player-controller.ts` | PlayerController 类 (play/pause/next/back/goto/loadScript) |
| 播放状态机 | `src/player/player-state.ts` | PlayerState (idle/playing/paused/waiting/ended) |
| 场景导航 | `src/player/navigation.ts` | getSceneList/findScriptByScene/findScriptByChapter |
| 背景渲染 | `src/renderer/background-renderer.ts` | BackgroundState 解析 |
| 角色渲染 | `src/renderer/character-renderer.ts` | show/hide 角色状态管理 |
| 文本渲染 | `src/renderer/text-renderer.ts` | narration/dialogue/thought 三种文本模式 |
| 转场渲染 | `src/renderer/transition-renderer.ts` | fade/cut/dissolve 效果 |

**输出格式:** ESM (package.json `"type": "module"`)

#### 3.2 Visual Prompt Agent ✅

| 文件 | 内容 |
|------|------|
| `packages/agents/src/visual-prompt/visual-prompt-agent.ts` | 从叙事单元提取视觉证据, 生成 CharacterPromptPack + BackgroundPromptPack |
| 支持风格模板 | school-romance-anime, urban-romance, fresh-japanese |
| 证据校验 | 验证 quote 是否原文精确引用, 未匹配标记 `[unverified]` |
| 输出 | VisualPromptResult (角色 prompt[] + 背景 prompt + 风格模板) |

#### 3.3 图像生成 Provider ✅

| Provider | 模型 | API 格式 |
|----------|------|----------|
| OpenAIImageProvider | gpt-image-1 | OpenAI Images API |
| ZhipuImageProvider | cogview-4-250304 | OpenAI 兼容 (open.bigmodel.cn) |
| SiliconFlowImageProvider | FLUX.1-schnell / SD3.5 | OpenAI 兼容 (api.siliconflow.cn) |

**接口:** `ImageProvider` (generateImage + getSupportedModels + getDefaultSize)
**自动保存:** 图像可自动保存到 `data/projects/{id}/preview/{sceneId}/`

#### 3.4 Preview Player 页面 ✅

- 16:9 VN 播放区域
- 点击推进 / 自动播放 (可配置延迟)
- 后退 / 下一步 / 跳转
- 章节/场景导航 (左侧面板)
- 角色立绘区域 (left/center/right 定位)
- 文本框 (narration 斜体 / dialogue 带名称 / thought 紫色斜体)
- Debug 模式 (显示当前 action JSON + 角色状态)
- 信息面板 (步骤数、状态、场景 ID)

#### 3.5 Visual Prompt 页面 ✅

- 章节/场景选择 (左侧导航)
- 角色提示词卡片 (可展开: 证据 → 保守补全 → 最终 Prompt)
- 背景提示词展示
- 风格模板选择 (3 种)
- 一键生成/重新生成
- Mutation 自动刷新缓存

**构建验证:**
- packages/runtime: tsc ✅
- packages/providers: tsc ✅
- packages/agents: tsc ✅
- apps/api: tsc ✅ (含新路由)
- apps/workbench: vite build ✅ (355KB JS, 23.5KB CSS)
- 集成测试: 9/9 通过 (Health/CRUD/Structure/Chapters/Tasks)

---

### Phase 4: MVP 收敛与验收 (待开发)

- [ ] 评测框架集成
- [ ] 性能优化
- [ ] MVP 验收指标达标

---

## MVP 验收指标

| Agent | 指标 | 目标 |
|-------|------|------|
| Structure | 章节识别 F1 | >= 0.95 |
| Narrative Parsing | 宏 F1 | >= 0.86 |
| Attribution | 说话人归属准确率 | >= 0.87 |
| Scene Segmentation | 边界 F1 | >= 0.78 |
| VN Mapping | 对话保留率 | >= 95% |
| VN Mapping | 非原文添加量 | <= 5% |
| Fidelity Review | 严重问题召回率 | >= 0.92 |
| 系统 | 章节完成率 | >= 85% |
| 系统 | 预览可用率 | >= 90% |
