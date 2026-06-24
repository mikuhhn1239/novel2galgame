# 训练数据优化方案

当前问题：scene-boundary F1 28.6%（v2 最好时 53.3%），narrative-type JSON 解析 2.6%。

---

## 1. Scene Boundary Detection — 恢复 v2 输出格式

### 问题根因
v3 数据只让模型输出 `{"boundaries": [7]}`，太简单，模型失去了"为什么要切"的语义信号。v2 要求同时输出原因（location_change/time_change 等），F1 达到 53.3%。

### 优化方案
在现有 v3 数据（350 条，`datasets/training/v3/scene-boundary-detection/`）基础上，**补全 reasons 字段**。

### 当前格式（v3）
```json
// assistant content
{"boundaries": [7]}
```

### 目标格式（v4）
```json
// assistant content  
{
  "boundaries": [7],
  "reasons": [
    {"after_paragraph": 7, "reason": "location_change"}
  ]
}
```

### 六种原因类型
| reason | 含义 | 示例 |
|--------|------|------|
| `location_change` | 地点变化 | 教室→校门外 |
| `time_change` | 时间跳跃 | 当天下午→第二天早晨 |
| `event_shift` | 新事件开始 | 吃饭→打架 |
| `focus_shift` | 叙事焦点切换 | 角色A→角色B的视角 |
| `flashback_shift` | 进入/退出回忆 | "她想起十年前..." |
| `unknown` | 难以归类 | 边界存在但原因模糊 |

### 处理步骤
1. 读取 `datasets/training/v3/scene-boundary-detection/` 下全部 3 个文件
2. 对每条数据的 `messages[2]["content"]`（即 assistant 输出），从 `{"boundaries": [7]}` 改写为：
```json
{
  "boundaries": [7],
  "reasons": [
    {"after_paragraph": 7, "reason": "location_change"}
  ]
}
```
3. 原因需要根据对应段落的上下文判断，每个 boundary 位置对应一条 reason
4. 如果判断不了原因，填 `"unknown"`
5. 输出到新目录 `datasets/training/v4/scene-boundary-detection/`

### System Prompt 也需要更新
```
你是一个中文小说 scene 边界检测助手。判断段落中哪些边界应切换 scene。为每个边界给出原因。
原因类型：location_change, time_change, event_shift, focus_shift, flashback_shift, unknown
只输出 JSON，格式：{"boundaries": [N, ...], "reasons": [{"after_paragraph": N, "reason": "xxx"}, ...]}
```

---

## 2. Narrative Type Classification — 修复 JSON 输出

### 问题根因
模型 JSON 解析率 2.6%。推断原因：输入文本中的原始中文双引号 `""` 与 JSON 的双引号冲突，模型混淆了文本内容和 JSON 结构。

### 优化方案
预处理输入数据，把 unit 文本中的双引号替换掉。

### 当前格式（v3）
```
units:
[1] "我说你这穿的什么玩意儿？
[2] 这破布料，这土掉渣的款式，你确定这是衣服？
[3] "
叶桉一边嫌弃地捏着苏柚一的衣服，一边吐槽道。
```

### 目标格式（v4）
```
units:
[1] 「我说你这穿的什么玩意儿？
[2] 这破布料，这土掉渣的款式，你确定这是衣服？
[3] 」
叶桉一边嫌弃地捏着苏柚一的衣服，一边吐槽道。
```

### 处理步骤
1. 读取 `datasets/training/v3/narrative-type-classification/` 下全部 3 个文件
2. 对每条数据的 `messages[1]["content"]`（user 输入），将：
   - 中文双引号 `""` → `「」`（Unicode U+300C / U+300D）
   - 仅替换 unit 引用[N]行内的引号，保持其他部分不变
3. 如果 unit 文本中不含 `""`，则跳过
4. `messages[0]["content"]`（system prompt）和 `messages[2]["content"]`（assistant 输出）**不变**
5. 输出到新目录 `datasets/training/v4/narrative-type-classification/`

### 注意事项
- 不能替换 JSON 结构中的引号（`"labels"`, `"unit_id"`, `"type"` 等）
- 只替换用户输入中 unit 文本内容里的引号
- 可以在 system prompt 末尾加一句："输入中的对话引号已统一为 「」，你的 JSON 输出仍使用标准双引号。"

---

## 3. Attribution Best Candidate — 保持不动

v3.1 已达 43.3% acc，JSON 100%，只需复制到 v4：

```
cp -r datasets/training/v3/attribution-best-candidate datasets/training/v4/attribution-best-candidate
```

---

## 输出要求

最终产出目录结构：
```
datasets/training/v4/
├── narrative-type-classification/
│   ├── train.jsonl  (310条，输入中引号已处理)
│   ├── val.jsonl    (39条)
│   └── test.jsonl   (39条)
├── scene-boundary-detection/
│   ├── train.jsonl  (280条，输出已补全 reasons)
│   ├── val.jsonl    (35条)
│   └── test.jsonl   (35条)
└── attribution-best-candidate/
    ├── train.jsonl  (240条，从 v3 直接复制)
    ├── val.jsonl    (30条)
    └── test.jsonl   (30条)
```

所有文件的 messages 结构保持 `[system, user, assistant]` ChatML 格式不变。
