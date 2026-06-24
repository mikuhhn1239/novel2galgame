#!/usr/bin/env python3
"""v3.4: 场景边界检测 → 滑动窗口二元分类
每个样本只判断一对相邻段落是否切换场景
"""

import json, os, re

SRC = '/workspace/project-nas-1000073/linyupeng/data/datasets/training/v3.3/scene-boundary-binary'
OUT = '/workspace/project-nas-1000073/linyupeng/data/datasets/training/v3.4/scene-boundary-pairwise'
os.makedirs(OUT, exist_ok=True)

SYSTEM = """你是一个中文小说 scene 边界检测助手。

你的任务：判断下面上下文中【标记的段落对】之间是否发生场景切换。

【输出格式】
只输出 JSON：
{"boundary": true}  或  {"boundary": false}
不要输出任何 JSON 以外的内容。"""


def convert_sample(sample, pair_idx):
    """从原始样本生成第 pair_idx 对（Pi 和 Pi+1 之间）的 pairwise 样本"""
    msgs = sample['messages']
    user_text = msgs[1]['content']
    gold = json.loads(msgs[2]['content'])

    # 提取所有段落
    paras = re.findall(r'\[P\d+\]\s*(.+?)(?=\[P\d+\]|\Z)', user_text, re.DOTALL)
    n = len(paras)

    if pair_idx >= n - 1 or pair_idx < 0:
        return None

    # 滑动窗口：±2 段落上下文
    start = max(0, pair_idx - 1)
    end = min(n, pair_idx + 3)  # +3 = pair_idx+1 + 2 more

    # 构建 user prompt
    context_lines = []
    for j in range(start, end):
        context_lines.append(f"[P{j+1}] {paras[j].strip()}")

    user_prompt = f"""上下文段落:
{chr(10).join(context_lines)}

请判断: [P{pair_idx+1}] 和 [P{pair_idx+2}] 之间是否切换场景？"""

    # Gold label
    is_boundary = (pair_idx + 1) in set(
        d['after'] for d in gold['decisions'] if d.get('change')
    )
    assistant = json.dumps({"boundary": is_boundary}, ensure_ascii=False)

    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant},
        ]
    }


# ─── 转换所有 split ───
for split in ['train', 'val', 'test']:
    src_path = os.path.join(SRC, f'{split}.jsonl')
    samples = [json.loads(l) for l in open(src_path, encoding='utf-8') if l.strip()]

    converted = []
    total_pairs = 0
    pos_pairs = 0

    for s in samples:
        user_text = s['messages'][1]['content']
        n_paras = len(re.findall(r'\[P\d+\]', user_text))

        for pair in range(n_paras - 1):
            c = convert_sample(s, pair)
            if c:
                converted.append(c)
                total_pairs += 1
                if json.loads(c['messages'][2]['content'])['boundary']:
                    pos_pairs += 1

    # Write
    out_path = os.path.join(OUT, f'{split}.jsonl')
    with open(out_path, 'w', encoding='utf-8') as f:
        for c in converted:
            f.write(json.dumps(c, ensure_ascii=False) + '\n')

    neg_pairs = total_pairs - pos_pairs
    print(f"  {split}: {len(samples)} passages → {len(converted)} pairs "
          f"(+{pos_pairs} / -{neg_pairs}, ratio={pos_pairs/total_pairs*100:.1f}%)")

# ─── 抽查 ───
print(f"\n=== 抽查样本 ===")
split = 'train'
with open(os.path.join(OUT, f'{split}.jsonl')) as f:
    pairs = [json.loads(l) for l in f if l.strip()]

for idx in [0, 1, 100, 500, 1000, 2000, 3000]:
    if idx >= len(pairs):
        break
    p = pairs[idx]
    label = json.loads(p['messages'][2]['content'])['boundary']
    print(f"  [{idx}] label={'TRUE ' if label else 'FALSE'} | user={p['messages'][1]['content'][:120]}...")
