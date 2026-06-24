#!/usr/bin/env python3
"""v3.4 pairwise 评估：对每条 passage 逐对推理后汇总边界"""
import json, os, re, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE = '/workspace/project-nas-1000073/已移除-用户名/data/checkpoints/stage1-base-sft/final'
LORA = '/workspace/project-nas-1000073/已移除-用户名/data/checkpoints/stage2-v3.4/scene-boundary-pairwise/final'
TEST_PASSAGES = '/workspace/project-nas-1000073/已移除-用户名/data/datasets/training/v3.3/scene-boundary-binary/test.jsonl'
TEST_PAIRS    = '/workspace/project-nas-1000073/已移除-用户名/data/datasets/training/v3.4/scene-boundary-pairwise/test.jsonl'

print("Loading model...", flush=True)
tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token
m = PeftModel.from_pretrained(
    AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map='auto', trust_remote_code=True),
    LORA)
m.eval()
print("Model loaded", flush=True)

passages = [json.loads(l) for l in open(TEST_PASSAGES, encoding='utf-8') if l.strip()]
pairs = [json.loads(l) for l in open(TEST_PAIRS, encoding='utf-8') if l.strip()]

def generate(msgs):
    text = ''.join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in msgs)
    text += '<|im_start|>assistant\n'
    inputs = tok(text, return_tensors='pt').to(m.device)
    with torch.no_grad():
        out = m.generate(**inputs, max_new_tokens=32, do_sample=False,
                         pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    return tok.decode(out[0][len(inputs.input_ids[0]):], skip_special_tokens=True)

def extract_boundary(resp):
    resp = resp.strip().lower()
    if 'true' in resp: return True
    if 'false' in resp: return False
    # Try JSON
    try:
        r = json.loads(re.search(r'\{.*\}', resp, re.DOTALL).group())
        return r.get('boundary', False)
    except:
        return False

# ─── 逐 passage 评估 ───
import re as _re
tp = fp = fn = parsed = pair_correct = pair_total = 0

pair_idx = 0  # tracks index in pairs array
for p_idx, p in enumerate(passages):
    gold_decisions = json.loads(p['messages'][2]['content'])['decisions']
    g_set = set(d['after'] for d in gold_decisions if d.get('change'))
    g_all = {d['after']: d.get('change', False) for d in gold_decisions}

    n_paras = len(g_all) + 1  # N-1 decisions → N paragraphs
    pred_set = set()

    for after in range(1, n_paras):
        pair_sample = pairs[pair_idx]
        resp = generate(pair_sample['messages'][:-1])
        is_boundary = extract_boundary(resp)

        if is_boundary:
            pred_set.add(after)

        # Pair accuracy
        pair_total += 1
        if is_boundary == g_all[after]:
            pair_correct += 1

        pair_idx += 1

    # Accumulate
    tp += len(g_set & pred_set)
    fp += len(pred_set - g_set)
    fn += len(g_set - pred_set)

    if p_idx % 10 == 0:
        print(f"  [{p_idx}/{len(passages)}] gold={sorted(g_set)} pred={sorted(pred_set)}", flush=True)

prec = tp/(tp+fp) if (tp+fp) else 0
rec = tp/(tp+fn) if (tp+fn) else 0
f1 = 2*prec*rec/(prec+rec) if (prec+rec) else 0
pair_acc = pair_correct/pair_total if pair_total else 0

print(f"\n{'='*50}", flush=True)
print(f"v3.4 Scene Boundary Pairwise 测试结果", flush=True)
print(f"{'='*50}", flush=True)
print(f"  Test passages: {len(passages)}", flush=True)
print(f"  Test pairs:    {pair_total}", flush=True)
print(f"  TP={tp} FP={fp} FN={fn}", flush=True)
print(f"  Precision: {prec:.1%}", flush=True)
print(f"  Recall:    {rec:.1%}", flush=True)
print(f"  F1:        {f1:.1%}", flush=True)
print(f"  Pair Acc:  {pair_acc:.1%} ({pair_correct}/{pair_total})", flush=True)
print(f"\n  对比:", flush=True)
print(f"    v3.2 (列表):   F1=20.0%", flush=True)
print(f"    v3.3 (二元):   F1=15.4%", flush=True)
print(f"    v3.4 (pairwise): F1={f1:.1%}", flush=True)

del m; torch.cuda.empty_cache()
