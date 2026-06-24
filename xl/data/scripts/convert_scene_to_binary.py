#!/usr/bin/env python3
"""将 v2 + v3.2 的 scene-boundary 数据转换为二元决策格式 v3.3"""

import json, os, re

SRC = [
    # (路径, 标签)
    ('/workspace/project-nas-1000073/已移除-用户名/data/datasets/training/scene-boundary-v2', 'v2'),
    ('/workspace/project-nas-1000073/已移除-用户名/data/datasets/training/v3.2/scene-boundary-detection', 'v3.2'),
]
OUT = '/workspace/project-nas-1000073/已移除-用户名/data/datasets/training/v3.3/scene-boundary-binary'
os.makedirs(OUT, exist_ok=True)

SYSTEM_PROMPT = """你是一个中文小说 scene 边界检测助手。

你的任务是：对输入段落的每一对相邻段落，判断它们之间是否发生场景切换。

【切分原则】
- 只有在存在足够明显的叙事边界变化时才判定 change=true
- 宁可略保守，也不要过度切分
- 同一角色的连续叙述、内心独白延续、同一想法分多段表述 → change=false
- 地点变化（室内→室外）、时间跳跃（第二天、晚上）、事件切换（新情节）、焦点转移（换视角）→ change=true
- 作者注、章节标题等元信息结束恢复正文 → change=true

【输出格式】
只输出 JSON：
{
  "decisions": [
    {"after": 1, "change": false},
    {"after": 2, "change": false},
    {"after": 5, "change": true},
    ...
  ]
}

- after=N 表示段落 P_N 和 P_{N+1} 之间的边界
- 必须为每一对相邻段落都输出一个 decision，不能跳过
- change=true 表示这里切换 scene，change=false 表示不切换
- 不要输出任何 JSON 以外的解释文字"""


def count_paras(user_text):
    return len(re.findall(r'\[P\d+\]', user_text))


def convert_sample(sample, src_label):
    msgs = sample['messages']
    user_text = msgs[1]['content']
    n_paras = count_paras(user_text)
    gold = json.loads(msgs[2]['content'])
    boundaries = set(gold.get('boundaries', []))

    # Build decisions for ALL adjacent pairs
    decisions = []
    for after in range(1, n_paras):
        if after in boundaries:
            decisions.append({"after": after, "change": True})
        else:
            decisions.append({"after": after, "change": False})

    new_asst = json.dumps({"decisions": decisions}, ensure_ascii=False)

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": new_asst},
        ]
    }


# ─── 合并所有数据源 ───
combined = {"train": [], "val": [], "test": []}

for src_dir, label in SRC:
    for split in ['train', 'val', 'test']:
        path = os.path.join(src_dir, f'{split}.jsonl')
        if not os.path.exists(path):
            continue
        samples = [json.loads(l) for l in open(path, encoding='utf-8') if l.strip()]
        converted = [convert_sample(s, label) for s in samples]
        combined[split].extend(converted)
        print(f"  [{label}] {split}: {len(samples)} → {len(converted)} samples")

# ─── 去重（按 user content） ───
for split in ['train', 'val', 'test']:
    seen = set()
    deduped = []
    for s in combined[split]:
        key = s['messages'][1]['content']
        if key not in seen:
            seen.add(key)
            deduped.append(s)
    combined[split] = deduped

print(f"\n  After dedup:")
for split in ['train', 'val', 'test']:
    print(f"    {split}: {len(combined[split])} samples")

# ─── 验证 ───
for split in ['train', 'val', 'test']:
    samples = combined[split]
    if not samples:
        continue

    errors = 0
    for i, s in enumerate(samples):
        n_paras = count_paras(s['messages'][1]['content'])
        decisions = json.loads(s['messages'][2]['content'])['decisions']

        if len(decisions) != n_paras - 1:
            print(f"  ❌ [{split}:{i}] paras={n_paras} but decisions={len(decisions)}")
            errors += 1

        # Verify after numbers are sequential
        for j, d in enumerate(decisions):
            if d['after'] != j + 1:
                print(f"  ❌ [{split}:{i}] decision[{j}].after={d['after']}, expected {j+1}")
                errors += 1

    if errors == 0:
        print(f"  ✅ {split}: all {len(samples)} samples valid")
    else:
        print(f"  ⚠️  {split}: {errors} errors")

# ─── 写入 ───
for split in ['train', 'val', 'test']:
    path = os.path.join(OUT, f'{split}.jsonl')
    with open(path, 'w', encoding='utf-8') as f:
        for s in combined[split]:
            f.write(json.dumps(s, ensure_ascii=False) + '\n')
    print(f"  💾 {path} ({len(combined[split])} samples)")

# ─── 统计 ───
print(f"\n{'='*50}")
print(f"  v3.3 数据统计")
print(f"{'='*50}")
for split in ['train', 'val', 'test']:
    samples = combined[split]
    if not samples:
        continue
    para_cnts = []
    pos_cnts = []
    for s in samples:
        decisions = json.loads(s['messages'][2]['content'])['decisions']
        para_cnts.append(len(decisions) + 1)
        pos_cnts.append(sum(1 for d in decisions if d['change']))
    print(f"  {split}: {len(samples)} samples, "
          f"paras {min(para_cnts)}-{max(para_cnts)} (avg {sum(para_cnts)/len(para_cnts):.1f}), "
          f"boundaries {min(pos_cnts)}-{max(pos_cnts)} (avg {sum(pos_cnts)/len(pos_cnts):.1f}), "
          f"ratio {sum(pos_cnts)/sum(p-1 for p in para_cnts)*100:.1f}%")
