#!/usr/bin/env python3
"""
Qwen3-8B v3 LoRA 训练 — 三卡并行版本
GPU 2 → narrative-type, GPU 4 → scene-boundary, GPU 6 → attribution-best
"""

import os, json, torch, sys
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType
import torch.multiprocessing as mp

# ─── 配置 ───
BASE_MODEL = "/workspace/project-nas-1000073/linyupeng/data/checkpoints/stage1-base-sft/final"
DATA_ROOT = "/workspace/project-nas-1000073/linyupeng/data/datasets/training/v3"
OUTPUT_ROOT = "/workspace/project-nas-1000073/linyupeng/data/checkpoints/stage2-v3"
BATCH_SIZE = 2
GRAD_ACCUM = 8
EPOCHS = 3
CUTOFF_LEN = 4096
LR = 2e-4

TASKS = [
    ("narrative-type-classification", 2),
    ("scene-boundary-detection", 4),
    ("attribution-best-candidate", 6),
]


def load_jsonl(path):
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def format_qwen3(messages):
    text = ""
    for m in messages:
        text += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
    return text


def train_task(task_name, gpu_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = "cuda:0"  # CUDA_VISIBLE_DEVICES 映射后就是 cuda:0
    print(f"\n{'='*60}")
    print(f"  GPU {gpu_id} → 训练: {task_name}")
    print(f"{'='*60}")

    # 加载数据
    train_path = os.path.join(DATA_ROOT, task_name, "train.jsonl")
    val_path = os.path.join(DATA_ROOT, task_name, "val.jsonl")
    train_data = load_jsonl(train_path)
    val_data = load_jsonl(val_path)
    print(f"  train={len(train_data)} val={len(val_data)}")

    train_ds = Dataset.from_list([{"messages": d["messages"]} for d in train_data])
    val_ds = Dataset.from_list([{"messages": d["messages"]} for d in val_data])

    # 加载模型
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )

    # LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=64, lora_alpha=128, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Tokenize（ChatML 文本 → token ids）
    train_ds = train_ds.map(lambda x: {"text": format_qwen3(x["messages"])}, remove_columns=["messages"])
    val_ds = val_ds.map(lambda x: {"text": format_qwen3(x["messages"])}, remove_columns=["messages"])

    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=CUTOFF_LEN)

    train_ds = train_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
    val_ds = val_ds.map(tokenize_fn, batched=True, remove_columns=["text"])

    # 训练
    output_dir = os.path.join(OUTPUT_ROOT, task_name)
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        bf16=True,
        logging_steps=5,
        save_strategy="epoch",
        eval_strategy="epoch",
        save_total_limit=2,
        remove_unused_columns=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    trainer.train()

    # 保存
    final_dir = os.path.join(output_dir, "final")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"  ✅ {task_name} 保存到: {final_dir}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    processes = []
    for task_name, gpu_id in TASKS:
        p = mp.Process(target=train_task, args=(task_name, gpu_id))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("\n✅ 全部完成!")
    for task_name, _ in TASKS:
        print(f"  {OUTPUT_ROOT}/{task_name}/final")
