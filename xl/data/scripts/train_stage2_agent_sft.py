#!/usr/bin/env python3
"""
Stage 2: Agent SFT — 从 Stage 1 基座分叉训练三个独立 Agent

  Agent 1: narrative-parsing   → 叙事单元切分 + 类型标注
  Agent 2: scene-segmentation  → 场景边界识别
  Agent 3: attribution-assist  → 角色归因

用法：
  python3 train_stage2_agent_sft.py all              # 串行训练全部三个
  python3 train_stage2_agent_sft.py narrative-parsing  # 只训练一个
"""

import os
import sys
import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

# ============================================================
# 配置
# ============================================================
BASE_MODEL_DIR = "/workspace/project-nas-1000073/已移除-用户名/data/checkpoints/stage1-base-sft/final"
DATA_DIR = "/workspace/project-nas-1000073/已移除-用户名/data/datasets/training"
OUTPUT_BASE_DIR = "/workspace/project-nas-1000073/已移除-用户名/data/checkpoints/stage2"

NUM_EPOCHS = 10
BATCH_SIZE = 2
LEARNING_RATE = 1e-5
MAX_LENGTH = 4096
SAVE_STEPS = 20

AGENT_CONFIGS = {
    "narrative-parsing": "叙事单元切分 + 类型标注",
    "scene-segmentation": "场景边界识别",
    "attribution-assist": "角色归因",
}


def load_agent_data(task_name):
    task_dir = os.path.join(DATA_DIR, task_name)
    samples = []
    for split_file in ["train.jsonl", "val.jsonl", "test.jsonl"]:
        filepath = os.path.join(task_dir, split_file)
        if not os.path.exists(filepath):
            continue
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                samples.append(json.loads(line))
    print(f"  Loaded {len(samples)} samples for {task_name}")
    return Dataset.from_list(samples)


def format_chatml(sample):
    text = ""
    for msg in sample["messages"]:
        text += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
    return {"text": text}


def train_agent(task_name):
    output_dir = os.path.join(OUTPUT_BASE_DIR, task_name)
    desc = AGENT_CONFIGS[task_name]

    print(f"\n{'='*60}")
    print(f"  Training: {task_name} — {desc}")
    print(f"{'='*60}")

    # 1. 加载数据
    dataset = load_agent_data(task_name)
    dataset = dataset.map(format_chatml)
    dataset = dataset.train_test_split(test_size=0.15, seed=42)
    print(f"  Train: {len(dataset['train'])}, Val: {len(dataset['test'])}")

    # 2. 加载模型
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_DIR,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )

    # 3. Tokenize
    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=MAX_LENGTH, padding=False)

    train_ds = dataset["train"].map(tokenize_fn, batched=True, remove_columns=dataset["train"].column_names)
    val_ds = dataset["test"].map(tokenize_fn, batched=True, remove_columns=dataset["test"].column_names)

    # 4. 训练
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=2,
        learning_rate=LEARNING_RATE,
        warmup_ratio=0.1,
        logging_steps=5,
        save_steps=SAVE_STEPS,
        eval_strategy="steps",
        eval_steps=SAVE_STEPS,
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    trainer.train()

    # 5. 保存
    final_dir = os.path.join(output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"  Saved to {final_dir}")

    del model, tokenizer, trainer
    torch.cuda.empty_cache()


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 train_stage2_agent_sft.py <task|all>")
        sys.exit(1)

    arg = sys.argv[1]
    tasks = list(AGENT_CONFIGS.keys()) if arg == "all" else [arg]

    for t in tasks:
        train_agent(t)

    print("\nDone. Models at:")
    for t in tasks:
        print(f"  {OUTPUT_BASE_DIR}/{t}/final/")


if __name__ == "__main__":
    main()
