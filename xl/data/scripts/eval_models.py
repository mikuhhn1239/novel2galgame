#!/usr/bin/env python3
"""
三 Agent 模型自动化指标评估

用法:
  python eval_models.py                          # 评估全部 3 个 Agent
  python eval_models.py narrative-parsing        # 只评估 1 个
  python eval_models.py --verbose               # 打印每个样本的细节
"""

import json, os, sys, re, argparse
import torch
from pathlib import Path
from collections import Counter

CHECKPOINT_DIR = Path("/workspace/project-nas-1000073/已移除-用户名/data/checkpoints")
TEST_DIR = Path("/workspace/project-nas-1000073/已移除-用户名/data/datasets/training_opt")
STAGE1_PATH = CHECKPOINT_DIR / "stage1-base-sft" / "final"

# LLM 输出可能含多余文字，需要提取 JSON
def extract_json(text: str):
    """从 LLM 输出中提取 JSON 数组或对象"""
    text = text.strip()
    # 尝试直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # 尝试提取 JSON 数组
    m = re.search(r'\[.*\]', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    # 尝试提取 JSON 对象
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return None


# ============================================================
# 模型加载
# ============================================================

def load_agent_model(agent_name: str):
    """加载 Stage 1 基座 + Stage 2 LoRA adapter"""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  加载基座: {STAGE1_PATH}")
    print(f"  加载 LoRA: {CHECKPOINT_DIR / 'stage2' / agent_name / 'final'}")
    print(f"  设备: {device}")

    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(
        str(STAGE1_PATH), trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 直接加载 LoRA 模型，无需 merge_and_unload（省时间）
    lora_path = CHECKPOINT_DIR / "stage2" / agent_name / "final"
    model = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(
            str(STAGE1_PATH),
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            trust_remote_code=True,
            device_map="auto" if device == "cuda" else None,
        ),
        str(lora_path),
    )
    model.eval()

    return model, tokenizer


def generate(model, tokenizer, messages, max_new_tokens=1024):
    """用 Qwen3 ChatML 格式推理"""
    text = ""
    for msg in messages:
        text += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
    text += "<|im_start|>assistant\n"

    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)


# ============================================================
# 评估函数
# ============================================================

VALID_TYPES = {"dialogue", "narration", "thought", "action", "scene_description"}
VALID_REASONS = {"location_change", "time_change", "event_shift", "focus_shift", "flashback_shift", "unknown"}


def eval_narrative_parsing(model, tokenizer, verbose=False):
    """评估叙事解析: 比较每个 unit 的 type"""
    test_file = TEST_DIR / "narrative-parsing" / "test.jsonl"
    samples = []
    for line in open(test_file, 'r', encoding='utf-8'):
        samples.append(json.loads(line))

    total_units = 0
    correct_type = 0
    json_parse_ok = 0
    results = []

    for i, sample in enumerate(samples):
        msgs = sample["messages"]
        response = generate(model, tokenizer, msgs[:-1])
        pred = extract_json(response)
        gold = json.loads(msgs[-1]["content"])

        parsed_ok = pred is not None
        if parsed_ok:
            json_parse_ok += 1
            gold_types = [u["type"] for u in gold]
            pred_types = [u.get("type", "") for u in pred]

            # 对齐比较（按位置配对，取 min 长度）
            n = min(len(gold_types), len(pred_types))
            for j in range(n):
                total_units += 1
                if pred_types[j] == gold_types[j]:
                    correct_type += 1

            if verbose:
                print(f"\n  Sample {i+1}:")
                print(f"    Gold: {[(u['type'], u['text'][:30]) for u in gold]}")
                print(f"    Pred: {[(u.get('type','?'), u.get('text','')[:30]) for u in pred]}")

        results.append({
            "sample_id": i,
            "parsed_ok": parsed_ok,
            "gold_units": len(gold),
            "pred_units": len(pred) if pred else 0,
        })

    acc = correct_type / total_units if total_units > 0 else 0
    return {
        "samples": len(samples),
        "json_parse_ok": json_parse_ok,
        "json_parse_rate": json_parse_ok / len(samples),
        "total_units": total_units,
        "type_accuracy": acc,
        "type_accuracy_pct": f"{acc*100:.1f}%",
        "per_sample": results,
    }


def eval_scene_segmentation(model, tokenizer, verbose=False):
    """评估场景切分: boundary 的 Precision/Recall/F1"""
    test_file = TEST_DIR / "scene-segmentation" / "test.jsonl"
    samples = []
    for line in open(test_file, 'r', encoding='utf-8'):
        samples.append(json.loads(line))

    tp_boundary = 0
    fp_boundary = 0
    fn_boundary = 0
    json_parse_ok = 0
    results = []

    for i, sample in enumerate(samples):
        msgs = sample["messages"]
        response = generate(model, tokenizer, msgs[:-1])
        pred = extract_json(response)
        gold = extract_json(msgs[-1]["content"])

        parsed_ok = pred is not None and gold is not None
        if parsed_ok:
            json_parse_ok += 1
            gold_set = set(gold.get("boundaries", []))
            pred_set = set(pred.get("boundaries", []))
            tp_boundary += len(gold_set & pred_set)
            fp_boundary += len(pred_set - gold_set)
            fn_boundary += len(gold_set - pred_set)

            if verbose:
                print(f"\n  Sample {i+1}:")
                print(f"    Gold boundaries: {list(gold_set)}")
                print(f"    Pred boundaries: {list(pred_set)}")

        results.append({
            "sample_id": i,
            "parsed_ok": parsed_ok,
            "gold_boundaries": list(gold.get("boundaries", [])) if gold else [],
            "pred_boundaries": list(pred.get("boundaries", [])) if pred else [],
        })

    prec = tp_boundary / (tp_boundary + fp_boundary) if (tp_boundary + fp_boundary) > 0 else 0
    rec = tp_boundary / (tp_boundary + fn_boundary) if (tp_boundary + fn_boundary) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

    return {
        "samples": len(samples),
        "json_parse_ok": json_parse_ok,
        "json_parse_rate": json_parse_ok / len(samples),
        "boundary_precision": prec,
        "boundary_recall": rec,
        "boundary_f1": f1,
        "tp": tp_boundary, "fp": fp_boundary, "fn": fn_boundary,
        "per_sample": results,
    }


def eval_attribution_assist(model, tokenizer, verbose=False):
    """评估角色归因: best_candidate 准确率, uncertain 一致率"""
    test_file = TEST_DIR / "attribution-assist" / "test.jsonl"
    samples = []
    for line in open(test_file, 'r', encoding='utf-8'):
        samples.append(json.loads(line))

    correct_best = 0
    uncertain_match = 0
    uncertain_gold = 0
    json_parse_ok = 0
    results = []

    for i, sample in enumerate(samples):
        msgs = sample["messages"]
        response = generate(model, tokenizer, msgs[:-1])
        pred = extract_json(response)
        gold = extract_json(msgs[-1]["content"])

        parsed_ok = pred is not None and gold is not None
        if parsed_ok:
            json_parse_ok += 1
            gold_best = gold.get("best_candidate", "")
            pred_best = pred.get("best_candidate", "")
            gold_uncertain = gold.get("uncertain", False)
            pred_uncertain = pred.get("uncertain", False)

            if gold_best and pred_best == gold_best:
                correct_best += 1

            if gold_uncertain:
                uncertain_gold += 1
                if pred_uncertain:
                    uncertain_match += 1

            if verbose:
                print(f"\n  Sample {i+1}:")
                print(f"    Gold: best={gold_best}, uncertain={gold_uncertain}")
                print(f"    Pred: best={pred_best}, uncertain={pred_uncertain}")

        results.append({
            "sample_id": i,
            "parsed_ok": parsed_ok,
            "gold_best": gold.get("best_candidate", "") if gold else "",
            "pred_best": pred.get("best_candidate", "") if pred else "",
            "gold_uncertain": gold.get("uncertain", False) if gold else None,
            "pred_uncertain": pred.get("uncertain", False) if pred else None,
        })

    best_acc = correct_best / json_parse_ok if json_parse_ok > 0 else 0
    uncertain_recall = uncertain_match / uncertain_gold if uncertain_gold > 0 else 0

    return {
        "samples": len(samples),
        "json_parse_ok": json_parse_ok,
        "json_parse_rate": json_parse_ok / len(samples),
        "best_candidate_accuracy": best_acc,
        "best_candidate_accuracy_pct": f"{best_acc*100:.1f}%",
        "correct_best": correct_best,
        "uncertain_recall": uncertain_recall,
        "uncertain_match": uncertain_match, "uncertain_gold": uncertain_gold,
        "per_sample": results,
    }


# ============================================================
# 主入口
# ============================================================

EVAL_FUNCS = {
    "narrative-parsing": eval_narrative_parsing,
    "scene-segmentation": eval_scene_segmentation,
    "attribution-assist": eval_attribution_assist,
}

METRIC_NAMES = {
    "narrative-parsing": [
        ("JSON解析率", "json_parse_rate", "{:.1%}"),
        ("类型准确率", "type_accuracy", "{:.1%}"),
        ("比对单元数", "total_units", "{}"),
    ],
    "scene-segmentation": [
        ("JSON解析率", "json_parse_rate", "{:.1%}"),
        ("边界Precision", "boundary_precision", "{:.1%}"),
        ("边界Recall", "boundary_recall", "{:.1%}"),
        ("边界F1", "boundary_f1", "{:.1%}"),
        ("TP/FP/FN", None, None),  # special
    ],
    "attribution-assist": [
        ("JSON解析率", "json_parse_rate", "{:.1%}"),
        ("最佳候选准确率", "best_candidate_accuracy", "{:.1%}"),
        ("不确定召回率", "uncertain_recall", "{:.1%}"),
    ],
}


def main():
    parser = argparse.ArgumentParser(description="三 Agent 模型自动化评估")
    parser.add_argument("agent", nargs="?", default="all",
                        choices=["all", "narrative-parsing", "scene-segmentation", "attribution-assist"])
    parser.add_argument("--verbose", "-v", action="store_true", help="打印每个样本细节")
    args = parser.parse_args()

    targets = list(EVAL_FUNCS.keys()) if args.agent == "all" else [args.agent]

    all_results = {}

    for agent in targets:
        print(f"\n{'='*60}")
        print(f"  评估: {agent}")
        print(f"{'='*60}")

        model, tokenizer = load_agent_model(agent)
        result = EVAL_FUNCS[agent](model, tokenizer, verbose=args.verbose)
        all_results[agent] = result

        print(f"\n  ── 指标 ──")
        for label, key, fmt in METRIC_NAMES[agent]:
            if key == "tp":
                print(f"  TP={result['tp']}  FP={result['fp']}  FN={result['fn']}")
            elif key:
                val = result[key]
                print(f"  {label}: {fmt.format(val)}")

        # 清理显存
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 汇总
    print(f"\n{'='*60}")
    print(f"  汇总")
    print(f"{'='*60}")
    for agent in targets:
        r = all_results[agent]
        print(f"\n  {agent}:")
        for label, key, fmt in METRIC_NAMES[agent]:
            if key == "tp":
                print(f"    {label}: TP={r['tp']} FP={r['fp']} FN={r['fn']}")
            elif key:
                print(f"    {label}: {fmt.format(r[key])}")

    # 保存详细结果
    out_path = Path(__file__).parent / "eval_results.json"
    # 清理 per_sample 避免 JSON 太大
    for r in all_results.values():
        r.pop("per_sample", None)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n  详细结果已保存: {out_path}")


if __name__ == "__main__":
    main()
