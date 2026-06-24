#!/usr/bin/env python3
"""v3.6 Scene Boundary — v2 简洁格式 + v3.2 数据量
v2 成功要素：95字短 prompt + 纯边界输出（无 reasons）
v3.2 贡献：384 条训练数据（vs v2 的 280）
"""

import json, os, torch, re
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType

BASE = '/workspace/project-nas-1000073/已移除-用户名/data/checkpoints/stage1-base-sft/final'
DATA = '/workspace/project-nas-1000073/已移除-用户名/data/datasets/training/v3.2/scene-boundary-detection'
OUT  = '/workspace/project-nas-1000073/已移除-用户名/data/checkpoints/stage2-v3.6'

# ─── v2 风格短 system prompt ───
SYSTEM_PROMPT = "你是一个中文小说 scene 边界检测助手。判断输入段落中哪些边界应切换 scene。只有在明显边界变化时才切。不要因情绪变化就切。只输出 boundaries，不输出原因。输出 JSON。"

def load_jsonl(p):
    with open(p) as f: return [json.loads(l) for l in f]

def fmt(msgs):
    return ''.join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in msgs)

# ─── 转换数据：去掉 reasons，用短 system prompt ───
for split in ['train', 'val']:
    src = load_jsonl(f'{DATA}/{split}.jsonl')
    out_samples = []
    for s in src:
        msgs = s['messages']
        gold = json.loads(msgs[2]['content'])
        # 纯边界，无 reasons
        new_asst = json.dumps({"boundaries": gold.get('boundaries', [])}, ensure_ascii=False)
        out_samples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                msgs[1],  # user (paragraphs)
                {"role": "assistant", "content": new_asst},
            ]
        })
    out_path = f'{OUT}/{split}.jsonl'
    os.makedirs(OUT, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        for s in out_samples:
            f.write(json.dumps(s, ensure_ascii=False) + '\n')
    print(f'  {split}: {len(out_samples)} samples → {out_path}')

# ─── 训练 ───
TASK = 'scene-boundary'
print(f"\n{'='*50}\n  v3.6 {TASK} (v2-style short prompt)\n{'='*50}")

td = Dataset.from_list([{'messages': d['messages']} for d in load_jsonl(f'{OUT}/train.jsonl')])
vd = Dataset.from_list([{'messages': d['messages']} for d in load_jsonl(f'{OUT}/val.jsonl')])
print(f'  train={len(td)} val={len(vd)}')

tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token

m = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map='auto', trust_remote_code=True)
m = get_peft_model(m, LoraConfig(task_type=TaskType.CAUSAL_LM, r=64, lora_alpha=128, lora_dropout=0.05,
    target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']))
m.print_trainable_parameters()

td = td.map(lambda x: {'text': fmt(x['messages'])}, remove_columns=['messages'])
vd = vd.map(lambda x: {'text': fmt(x['messages'])}, remove_columns=['messages'])
td = td.map(lambda x: tok(x['text'], truncation=True, max_length=4096), batched=True, remove_columns=['text'])
vd = vd.map(lambda x: tok(x['text'], truncation=True, max_length=4096), batched=True, remove_columns=['text'])

args = TrainingArguments(
    output_dir=f'{OUT}/{TASK}',
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    num_train_epochs=5,    # 同 v3.1
    learning_rate=1e-4,    # 同 v3.1
    lr_scheduler_type='cosine',
    warmup_ratio=0.05,
    weight_decay=0.01,
    bf16=True,
    logging_steps=10,
    save_strategy='epoch',
    eval_strategy='epoch',
    save_total_limit=2,
    remove_unused_columns=True,
    report_to='none',
)

trainer = Trainer(model=m, args=args, train_dataset=td, eval_dataset=vd,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tok, mlm=False))
trainer.train()

m.save_pretrained(f'{OUT}/{TASK}/final')
tok.save_pretrained(f'{OUT}/{TASK}/final')
print(f'\n  ✅ v3.6 saved to {OUT}/{TASK}/final')
