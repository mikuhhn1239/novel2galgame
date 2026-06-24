#!/usr/bin/env python3
"""三 Agent 评估可视化"""
import json, os, re, torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE = '/workspace/project-nas-1000073/linyupeng/data/checkpoints/stage1-base-sft/final'
MODELS = {
    "narrative-type": '/workspace/project-nas-1000073/linyupeng/data/checkpoints/stage2-v3.2/narrative-type-classification/final',
    "scene-boundary": '/workspace/project-nas-1000073/linyupeng/data/checkpoints/stage2-v3.1/scene-boundary-detection/final',
    "attribution-best": '/workspace/project-nas-1000073/linyupeng/data/checkpoints/stage2-v3.2/attribution-best-candidate/final',
}
DATA = '/workspace/project-nas-1000073/linyupeng/data/datasets/training/v3.2'

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

def load_model(lora_path):
    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    m = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16 if device=="cuda" else torch.float32,
                                              device_map='auto' if device=="cuda" else None, trust_remote_code=True),
        lora_path)
    m.eval()
    return m, tok

def generate(model, tok, msgs, max_new=1024):
    text = ''.join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in msgs)
    text += '<|im_start|>assistant\n'
    inputs = tok(text, return_tensors='pt').to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    return tok.decode(out[0][len(inputs.input_ids[0]):], skip_special_tokens=True)

def extract_json(text):
    text = text.strip()
    try: return json.loads(text)
    except: pass
    for pat in [r'\[.*\]', r'\{.*\}']:
        m = re.search(pat, text, re.DOTALL)
        if m:
            raw = m.group()
            try: return json.loads(raw)
            except:
                for i in range(len(raw), 0, -1):
                    for suffix in ['}]}', ']}]}', raw.rstrip(',')+'}]}']:
                        try:
                            r = json.loads(raw[:i] + suffix)
                            if isinstance(r, dict) and r: return r
                        except: pass
    return None

# ─── 评估 + 收集逐样本数据 ───
all_data = {}

for name, lora_path in MODELS.items():
    print(f"\n{'='*40}\n  {name}\n{'='*40}")
    m, tok = load_model(lora_path)

    task_dir = {"narrative-type": "narrative-type-classification",
                "scene-boundary": "scene-boundary-detection",
                "attribution-best": "attribution-best-candidate"}[name]

    test_path = os.path.join(DATA, task_dir, "test.jsonl")
    samples = [json.loads(l) for l in open(test_path) if l.strip()]
    per_sample = []

    for i, s in enumerate(samples):
        msgs = s['messages']
        gold = json.loads(msgs[2]['content'])

        # 推理
        resp = generate(m, tok, msgs[:-1])
        pred = extract_json(resp)

        record = {"sample_id": i, "parsed": pred is not None and isinstance(pred, dict) and bool(pred)}

        if name == "narrative-type":
            gl = {u['unit_id']: u['type'] for u in gold.get('labels', [])}
            if pred and isinstance(pred, dict):
                pl = {u.get('unit_id', ''): u.get('type', '?') for u in pred.get('labels', [])}
                ct = sum(1 for uid in gl if pl.get(uid) == gl[uid])
                record["correct"] = ct
                record["total"] = len(gl)
                record["accuracy"] = ct / len(gl) if gl else 0

                # 按 type 统计
                for uid in gl:
                    gt = gl[uid]; pt = pl.get(uid, '?')
                    record.setdefault('type_stats', []).append({"unit_id": uid, "gold": gt, "pred": pt, "match": gt==pt})
            else:
                record["correct"] = 0
                record["total"] = len(gl)
                record["accuracy"] = 0

        elif name == "scene-boundary":
            g_set = set(gold.get("boundaries", []))
            p_set = set(pred.get("boundaries", []) if isinstance(pred, dict) else [])
            tp = len(g_set & p_set)
            fp = len(p_set - g_set)
            fn = len(g_set - p_set)
            record["tp"] = tp; record["fp"] = fp; record["fn"] = fn
            record["gold_boundaries"] = list(g_set)
            record["pred_boundaries"] = list(p_set)

        elif name == "attribution-best":
            record["gold_best"] = gold.get("best_candidate", "")
            record["pred_best"] = pred.get("best_candidate", "") if isinstance(pred, dict) else ""
            record["gold_uncertain"] = gold.get("uncertain", False)
            record["pred_uncertain"] = pred.get("uncertain", False) if isinstance(pred, dict) else False
            record["correct"] = record["gold_best"] == record["pred_best"]

        per_sample.append(record)
        if i % 10 == 0: print(f"  [{i}/{len(samples)}]", flush=True)

    all_data[name] = per_sample
    del m; torch.cuda.empty_cache()

# ─── 保存原始数据 ───
with open('/workspace/project-nas-1000073/linyupeng/data/viz_data.json', 'w') as f:
    json.dump(all_data, f, ensure_ascii=False, indent=2)

# ─── 可视化 ───
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
colors = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12', '#9b59b6']
VALID_TYPES = ['dialogue', 'narration', 'thought', 'action', 'scene_description']

# 1. narrative-type: 类型准确率柱状图
ax = axes[0, 0]
d = all_data["narrative-type"]
if d and d[0].get("type_stats"):
    type_stats = {}
    for r in d:
        for ts in r.get("type_stats", []):
            t = ts["gold"]
            type_stats.setdefault(t, {"correct": 0, "total": 0})
            type_stats[t]["total"] += 1
            if ts["match"]: type_stats[t]["correct"] += 1

    types = [t for t in VALID_TYPES if t in type_stats]
    accs = [type_stats[t]["correct"]/type_stats[t]["total"]*100 for t in types]
    counts = [type_stats[t]["total"] for t in types]
    bars = ax.bar(types, accs, color=colors[:len(types)])
    for bar, acc, cnt in zip(bars, accs, counts):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1, f'{acc:.0f}%\n({cnt})',
                ha='center', va='bottom', fontsize=8)
    ax.set_ylim(0, 110)
ax.set_title(f'Narrative Type Accuracy (overall: {np.mean([r["accuracy"] for r in d])*100:.0f}%)', fontsize=11)
ax.set_ylabel('Accuracy (%)')

# 2. narrative-type: 逐样本准确率
ax = axes[0, 1]
accs = [r["accuracy"]*100 for r in d]
ax.bar(range(len(accs)), accs, color=['#2ecc71' if a>=70 else '#e74c3c' if a<40 else '#f39c12' for a in accs])
ax.axhline(y=np.mean(accs), color='red', linestyle='--', label=f'Mean: {np.mean(accs):.0f}%')
ax.set_title(f'Narrative — Per-Sample Accuracy', fontsize=11)
ax.set_xlabel('Sample ID'); ax.set_ylabel('Accuracy (%)')
ax.legend()

# 3. narrative-type: 混淆矩阵
ax = axes[0, 2]
cm = {}
for t in VALID_TYPES + ['?']: cm[t] = {t2: 0 for t2 in VALID_TYPES + ['?']}
for r in d:
    for ts in r.get("type_stats", []):
        cm[ts["gold"]][ts["pred"]] += 1

all_t = [t for t in VALID_TYPES if sum(cm[t].values()) > 0]
mat = np.array([[cm[t1][t2] for t2 in all_t] for t1 in all_t])
mat_norm = mat / mat.sum(axis=1, keepdims=True) * 100
im = ax.imshow(mat_norm, cmap='YlGn', vmin=0, vmax=100)
for i in range(len(all_t)):
    for j in range(len(all_t)):
        val = mat[i][j]
        ax.text(j, i, f'{mat_norm[i][j]:.0f}%\n({val})', ha='center', va='center', fontsize=7)
ax.set_xticks(range(len(all_t))); ax.set_xticklabels(all_t, rotation=45, ha='right', fontsize=8)
ax.set_yticks(range(len(all_t))); ax.set_yticklabels(all_t, fontsize=8)
ax.set_title('Narrative — Type Confusion Matrix', fontsize=11)
plt.colorbar(im, ax=ax, shrink=0.8)

# 4. scene-boundary: F1 逐样本
ax = axes[1, 0]
d = all_data["scene-boundary"]
f1s = []
for r in d:
    tp, fp, fn = r.get("tp", 0), r.get("fp", 0), r.get("fn", 0)
    prec = tp/(tp+fp) if (tp+fp) else 0
    rec = tp/(tp+fn) if (tp+fn) else 0
    f1s.append(2*prec*rec/(prec+rec) if (prec+rec) else 0)
colors_s = ['#2ecc71' if f>=50 else '#e74c3c' if f==0 else '#f39c12' for f in f1s]
ax.bar(range(len(f1s)), [f*100 for f in f1s], color=colors_s)
ax.axhline(y=np.mean(f1s)*100, color='red', linestyle='--', label=f'Mean: {np.mean(f1s)*100:.0f}%')
ax.set_title(f'Scene Boundary — Per-Sample F1 (mean={np.mean(f1s)*100:.0f}%)', fontsize=11)
ax.set_xlabel('Sample ID'); ax.set_ylabel('F1 (%)')
ax.legend()

# 5. scene-boundary: TP/FP/FN 统计
ax = axes[1, 1]
tp = sum(r.get("tp",0) for r in d)
fp = sum(r.get("fp",0) for r in d)
fn = sum(r.get("fn",0) for r in d)
ax.bar(['TP', 'FP', 'FN'], [tp, fp, fn], color=['#2ecc71', '#e74c3c', '#f39c12'])
for i, v in enumerate([tp, fp, fn]):
    ax.text(i, v+0.5, str(v), ha='center', fontsize=12, fontweight='bold')
prec = tp/(tp+fp) if (tp+fp) else 0
rec = tp/(tp+fn) if (tp+fn) else 0
ax.set_title(f'Scene Boundary — P={prec:.1%} R={rec:.1%} F1={2*prec*rec/(prec+rec) if (prec+rec) else 0:.1%}', fontsize=11)

# 6. attribution-best: 逐样本对错
ax = axes[1, 2]
d = all_data["attribution-best"]
correct = [r["correct"] for r in d]
colors_a = ['#2ecc71' if c else '#e74c3c' for c in correct]
ax.bar(range(len(correct)), [1]*len(correct), color=colors_a, width=1)
acc = sum(correct)/len(correct)
ax.set_title(f'Attribution Best — Per-Sample ({acc*100:.0f}% correct {sum(correct)}/{len(correct)})', fontsize=11)
ax.set_xlabel('Sample ID'); ax.set_yticks([])

# uncertain 分布
unc_correct = sum(1 for r in d if r["gold_uncertain"] and r["pred_uncertain"])
unc_total = sum(1 for r in d if r["gold_uncertain"])
if unc_total > 0:
    ax.text(0.5, -0.25, f'Uncertain recall: {unc_correct}/{unc_total} ({unc_correct/unc_total:.0%})',
            transform=ax.transAxes, ha='center', fontsize=9, color='gray')

plt.tight_layout()
plt.savefig('/workspace/project-nas-1000073/linyupeng/data/eval_visualization.png', dpi=150, bbox_inches='tight')
print(f"\n✅ Saved to eval_visualization.png")
print(f"✅ Raw data saved to viz_data.json")
