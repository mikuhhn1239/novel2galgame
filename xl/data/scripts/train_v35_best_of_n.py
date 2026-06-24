#!/usr/bin/env python3
"""v3.5 Scene Boundary — Best-of-N 迭代训练 (简化 GRPO)
每轮：生成 K 个候选 → F1 评分 → 选最佳 → SFT 1 epoch → 重复
"""

import json, os, re, torch, sys, random
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import PeftModel, LoraConfig, get_peft_model, TaskType

# ─── 配置 ───
BASE_MODEL  = '/workspace/project-nas-1000073/linyupeng/data/checkpoints/stage1-base-sft/final'
LORA_INIT   = '/workspace/project-nas-1000073/linyupeng/data/checkpoints/stage2-v3.2/scene-boundary-detection/final'  # v3.2 格式一致
TRAIN_DATA  = '/workspace/project-nas-1000073/linyupeng/data/datasets/training/v3.2/scene-boundary-detection/train.jsonl'
TEST_DATA   = '/workspace/project-nas-1000073/linyupeng/data/datasets/training/v3.2/scene-boundary-detection/test.jsonl'
OUT_DIR     = '/workspace/project-nas-1000073/linyupeng/data/checkpoints/stage2-v3.5'
N_CANDIDATES = 3      # 每个样本生成几个候选
N_ITERATIONS = 3      # 迭代轮数
TEMPERATURE  = 0.8    # 生成多样性

os.makedirs(OUT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token

# ─── 加载数据 ───
samples = [json.loads(l) for l in open(TRAIN_DATA, encoding='utf-8') if l.strip()]
test_samples = [json.loads(l) for l in open(TEST_DATA, encoding='utf-8') if l.strip()]
print(f"Train: {len(samples)}, Test: {len(test_samples)}")

def load_lora(lora_path):
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16,
                                                 device_map='auto', trust_remote_code=True)
    if lora_path:
        m = PeftModel.from_pretrained(base, lora_path)
    else:
        m = base
    m.eval()
    return m

def generate_candidates(model, msgs, n=N_CANDIDATES, temp=TEMPERATURE):
    """生成 n 个不同候选"""
    text = ''.join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in msgs[:-1])
    text += '<|im_start|>assistant\n'
    inputs = tok(text, return_tensors='pt').to(model.device)

    candidates = []
    do_sample = temp > 0.0
    gen_kwargs = dict(max_new_tokens=256,
                      pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    if do_sample:
        gen_kwargs.update(do_sample=True, temperature=temp, top_p=0.9)
    else:
        gen_kwargs.update(do_sample=False)

    with torch.no_grad():
        for _ in range(n):
            out = model.generate(**inputs, **gen_kwargs)
            resp = tok.decode(out[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
            candidates.append(resp)
    return candidates

def extract_json(text):
    text = text.strip()
    try: return json.loads(text)
    except: pass
    m_re = re.search(r'\{.*\}', text, re.DOTALL)
    if m_re:
        raw = m_re.group()
        for i in range(len(raw), 0, -1):
            for suffix in ['}]}', ']}]}']:
                try:
                    r = json.loads(raw[:i] + suffix)
                    if isinstance(r, dict) and r: return r
                except: pass
    return None

def jaccard_score(gold_set, pred_set):
    """Jaccard 相似度，正确处理空集"""
    if not gold_set and not pred_set:
        return 1.0  # 两者都空，完美匹配
    intersection = len(gold_set & pred_set)
    union = len(gold_set | pred_set)
    return intersection / union if union > 0 else 0.0

def f1_score(gold_set, pred_set):
    tp = len(gold_set & pred_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec  = tp / (tp + fn) if (tp + fn) else 0
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0

def score_prediction(pred_text, gold):
    """解析预测并计算 Jaccard 分数"""
    pred_json = extract_json(pred_text)
    if not pred_json or not isinstance(pred_json, dict):
        return 0.0, set()

    gold_set = set(gold.get('boundaries', []))
    pred_set = set(pred_json.get('boundaries', []))
    # 使用 F1 作为奖励（对业务指标直接对齐）
    score = f1_score(gold_set, pred_set)
    return score, pred_set

def evaluate_model(model):
    """快速测试集评估"""
    tp = fp = fn = 0
    for s in test_samples:
        msgs = s['messages']
        candidates = generate_candidates(model, msgs, n=1, temp=0.0)  # greedy
        score, _ = score_prediction(candidates[0], json.loads(msgs[2]['content']))
        # Re-extract for counting
        pred_json = extract_json(candidates[0])
        gold_json = json.loads(msgs[2]['content'])
        if pred_json and isinstance(pred_json, dict):
            gs = set(gold_json.get('boundaries', []))
            ps = set(pred_json.get('boundaries', []))
            tp += len(gs & ps)
            fp += len(ps - gs)
            fn += len(gs - ps)
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec  = tp / (tp + fn) if (tp + fn) else 0
    f1  = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    return f1, prec, rec, tp, fp, fn

# ─── 主循环 ───
current_lora = LORA_INIT

for iteration in range(N_ITERATIONS):
    print(f"\n{'='*60}")
    print(f"  Iteration {iteration+1}/{N_ITERATIONS}")
    print(f"{'='*60}")

    # 1. 加载当前模型
    print(f"Loading model from {current_lora}...", flush=True)
    model = load_lora(current_lora)

    # 2. 评估当前模型
    f1, prec, rec, tp, fp, fn = evaluate_model(model)
    print(f"  📊 Before iter {iteration+1}: F1={f1:.3f} P={prec:.3f} R={rec:.3f} TP={tp} FP={fp} FN={fn}", flush=True)

    # 3. 为每个训练样本生成候选 + 选最佳
    print(f"  Generating {N_CANDIDATES} candidates × {len(samples)} samples...", flush=True)
    best_samples = []
    total_score = 0.0

    for i, s in enumerate(samples):
        msgs = s['messages']
        gold = json.loads(msgs[2]['content'])
        gold_set = set(gold.get('boundaries', []))

        # 生成候选
        candidates = generate_candidates(model, msgs, n=N_CANDIDATES)

        # 评分
        best_score = -1.0
        best_text = None
        for c in candidates:
            sc, _ = score_prediction(c, gold)
            if sc > best_score:
                best_score = sc
                best_text = c

        total_score += best_score

        # 用最佳候选（或 gold，如果 gold 更好）构建训练样本
        # 策略：如果 best_score >= 0.8 用候选，否则用 gold（防止退化）
        if best_score >= 0.8:
            # 保持候选的原始格式
            target_content = extract_json(best_text)
            if target_content:
                new_asst = json.dumps(target_content, ensure_ascii=False)
            else:
                new_asst = msgs[2]['content']
        else:
            new_asst = msgs[2]['content']  # 回退到 gold

        best_samples.append({
            "messages": [
                msgs[0],  # system
                msgs[1],  # user
                {"role": "assistant", "content": new_asst},
            ]
        })

        if i % 20 == 0:
            print(f"    [{i}/{len(samples)}] avg_score={total_score/(i+1):.3f}", flush=True)

    avg_score = total_score / len(samples)
    print(f"  📊 Avg best score: {avg_score:.3f}", flush=True)
    del model; torch.cuda.empty_cache()

    # 4. 用最佳候选 SFT 1 epoch
    print(f"  Fine-tuning on {len(best_samples)} samples for 1 epoch...", flush=True)

    td = Dataset.from_list([{'messages': d['messages']} for d in best_samples])

    def fmt(msgs):
        return ''.join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in msgs)

    td = td.map(lambda x: {'text': fmt(x['messages'])}, remove_columns=['messages'])
    td = td.map(lambda x: tok(x['text'], truncation=True, max_length=4096), batched=True, remove_columns=['text'])

    m = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16,
                                              device_map='auto', trust_remote_code=True)
    # 从上一轮的 LoRA 继续
    if current_lora:
        m = PeftModel.from_pretrained(m, current_lora, is_trainable=True)
    else:
        m = get_peft_model(m, LoraConfig(task_type=TaskType.CAUSAL_LM, r=64, lora_alpha=128,
            lora_dropout=0.05, target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']))

    out_iter = f'{OUT_DIR}/iter{iteration+1}'
    args = TrainingArguments(
        output_dir=out_iter,
        per_device_train_batch_size=1, gradient_accumulation_steps=16,
        num_train_epochs=1, learning_rate=5e-5,  # 更小的 LR 用于迭代训练
        lr_scheduler_type='cosine', warmup_ratio=0.1, weight_decay=0.01,
        bf16=True, logging_steps=5, save_strategy='no',
        remove_unused_columns=True, report_to='none',
    )
    trainer = Trainer(model=m, args=args, train_dataset=td,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tok, mlm=False))
    trainer.train()

    current_lora = f'{out_iter}/final'
    m.save_pretrained(current_lora)
    tok.save_pretrained(current_lora)
    del m; del trainer; torch.cuda.empty_cache()
    print(f"  💾 Saved to {current_lora}")

# ─── 最终评估 ───
print(f"\n{'='*60}")
print(f"  Final Evaluation")
print(f"{'='*60}")
model = load_lora(current_lora)
f1, prec, rec, tp, fp, fn = evaluate_model(model)
print(f"  ✅ v3.5 Final: F1={f1:.3f} P={prec:.3f} R={rec:.3f} TP={tp} FP={fp} FN={fn}")
print(f"  Baseline v3.1: F1=0.286")

del model; torch.cuda.empty_cache()
