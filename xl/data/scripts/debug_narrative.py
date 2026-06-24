#!/usr/bin/env python3
"""Debug narrative-type model output"""
import json, torch, re, time
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE = '/workspace/project-nas-1000073/已移除-用户名/data/checkpoints/stage1-base-sft/final'
LORA = '/workspace/project-nas-1000073/已移除-用户名/data/checkpoints/stage2-v3.2/narrative-type-classification/final'
TEST = '/workspace/project-nas-1000073/已移除-用户名/data/datasets/training/v3.2/narrative-type-classification/test.jsonl'

print("Loading model...")
t0 = time.time()
tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token
m = PeftModel.from_pretrained(
    AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map='auto', trust_remote_code=True),
    LORA)
m.eval()
print(f"Loaded in {time.time()-t0:.0f}s")

with open(TEST) as f:
    samples = [json.loads(l) for l in f if l.strip()]

for i in [0, 1, 2]:
    msgs = samples[i]['messages']
    text = ''.join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in msgs[:-1])
    text += '<|im_start|>assistant\n'
    inputs = tok(text, return_tensors='pt').to(m.device)

    t0 = time.time()
    print(f"\n[{i}] Generating (input={inputs.input_ids.shape[1]} tokens)...", flush=True)
    with torch.no_grad():
        out = m.generate(**inputs, max_new_tokens=50, do_sample=False,
                         pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)

    resp = tok.decode(out[0][len(inputs.input_ids[0]):], skip_special_tokens=False)
    n_tok = out.shape[1] - inputs.input_ids.shape[1]
    print(f"  Done in {time.time()-t0:.0f}s, {n_tok} tokens", flush=True)
    print(f"  RAW: {repr(resp[:300])}", flush=True)

    # Try to extract JSON
    clean = tok.decode(out[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
    import re as _re
    mj = _re.search(r'\{.*\}', clean, re.DOTALL)
    if mj:
        try:
            p = json.loads(mj.group())
            print(f"  JSON OK: labels={len(p.get('labels',[]))}", flush=True)
        except:
            print(f"  JSON BAD: {mj.group()[:200]}", flush=True)
    else:
        print(f"  No JSON found", flush=True)

    gold = json.loads(msgs[2]['content'])
    print(f"  Gold: {len(gold['labels'])} units", flush=True)
