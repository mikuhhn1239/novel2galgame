#!/usr/bin/env python3
"""v3 串行训练三个 Agent，单卡稳定运行"""
import os, json, torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType

BASE = '/workspace/project-nas-1000073/已移除-用户名/data/checkpoints/stage1-base-sft/final'
DATA = '/workspace/project-nas-1000073/已移除-用户名/data/datasets/training/v3.2'
OUT = '/workspace/project-nas-1000073/已移除-用户名/data/checkpoints/stage2-v3.2'
TASKS = ['narrative-type-classification', 'scene-boundary-detection', 'attribution-best-candidate']

def load_jsonl(p):
    with open(p) as f: return [json.loads(l) for l in f]

def fmt(msgs):
    return ''.join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in msgs)

for task in TASKS:
    print(f"\n{'='*50}\n  {task}\n{'='*50}")
    td = Dataset.from_list([{'messages': d['messages']} for d in load_jsonl(f'{DATA}/{task}/train.jsonl')])
    vd = Dataset.from_list([{'messages': d['messages']} for d in load_jsonl(f'{DATA}/{task}/val.jsonl')])
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

    args = TrainingArguments(output_dir=f'{OUT}/{task}', per_device_train_batch_size=1, gradient_accumulation_steps=16,
        num_train_epochs=5, learning_rate=1e-4, lr_scheduler_type='cosine', warmup_ratio=0.05, weight_decay=0.01,
        bf16=True, logging_steps=10, save_strategy='epoch', eval_strategy='epoch',
        save_total_limit=2, remove_unused_columns=True, report_to='none')

    trainer = Trainer(model=m, args=args, train_dataset=td, eval_dataset=vd,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tok, mlm=False))
    trainer.train()
    m.save_pretrained(f'{OUT}/{task}/final'); tok.save_pretrained(f'{OUT}/{task}/final')
    print(f'  ✅ {task} saved')
    del m, trainer; torch.cuda.empty_cache()

print('\n✅ All done!')
for t in TASKS: print(f'  {OUT}/{t}/final')
