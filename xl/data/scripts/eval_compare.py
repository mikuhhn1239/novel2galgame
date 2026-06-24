#!/usr/bin/env python3
"""
三模型对比评估：零基座 vs +Stage1 vs +Stage2 v3

用法: python3 eval_compare.py
"""

import json, os, sys, re, torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ─── 路径配置 ───
TEST_DATA = {
    "narrative-type-classification": "/workspace/project-nas-1000073/已移除-用户名/data/datasets/training/v3.2/narrative-type-classification/test.jsonl",
    "scene-boundary-detection": "/workspace/project-nas-1000073/已移除-用户名/data/datasets/training/v3.2/scene-boundary-detection/test.jsonl",
    "attribution-best-candidate": "/workspace/project-nas-1000073/已移除-用户名/data/datasets/training/v3.2/attribution-best-candidate/test.jsonl",
    # v3.3 binary format — same test content, different output format
    "scene-boundary-binary": "/workspace/project-nas-1000073/已移除-用户名/data/datasets/training/v3.3/scene-boundary-binary/test.jsonl",
}

MODELS = {
    "+Stage2 v3.2": {
        "base": "/workspace/project-nas-1000073/已移除-用户名/data/checkpoints/stage1-base-sft/final",
        "lora_base": "/workspace/project-nas-1000073/已移除-用户名/data/checkpoints/stage2-v3.2",
    },
    "+Stage2 v3.3 (Scene Binary)": {
        "base": "/workspace/project-nas-1000073/已移除-用户名/data/checkpoints/stage1-base-sft/final",
        "lora_base": "/workspace/project-nas-1000073/已移除-用户名/data/checkpoints/stage2-v3.3",
    },
}


def extract_json(text: str):
    text = text.strip()
    try: return json.loads(text)
    except: pass
    for pat in [r'\[.*\]', r'\{.*\}']:
        m = re.search(pat, text, re.DOTALL)
        if m:
            try: return json.loads(m.group())
            except:
                # 截断修复：逐字符回退找到最后一个有效 JSON 前缀
                raw = m.group()
                for i in range(len(raw), 0, -1):
                    try:
                        prefix = raw[:i]
                        # 补齐缺失的闭合符号
                        candidates = [
                            prefix + '}]}',
                            prefix + ']}]}',
                            prefix.rstrip(',') + '}]}',
                            prefix.rstrip(',') + ']}]}',
                            prefix + '"}',
                        ]
                        for c in candidates:
                            try:
                                result = json.loads(c)
                                if isinstance(result, dict) and result:
                                    return result
                            except: pass
                    except: pass
    return None


def load_model(model_cfg, agent_name, device):
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["base"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        model_cfg["base"],
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        trust_remote_code=True,
        device_map="auto" if device == "cuda" else None,
    )

    if model_cfg.get("lora_base"):
        lora_path = os.path.join(model_cfg["lora_base"], agent_name, "final")
        model = PeftModel.from_pretrained(base, lora_path)
    else:
        model = base
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, messages, max_new_tokens=1024):
    text = ""
    for m in messages:
        text += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
    text += "<|im_start|>assistant\n"
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                 do_sample=False,  # greedy，更快
                                 pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    return tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)


def eval_narrative_type(model, tok, test_path):
    samples = [json.loads(l) for l in open(test_path, encoding="utf-8")]
    correct, total, parsed = 0, 0, 0
    for s in samples:
        msgs = s["messages"]
        resp = generate(model, tok, msgs[:-1])
        pred = extract_json(resp)
        gold = extract_json(msgs[-1]["content"])
        if not isinstance(gold, dict): gold = {}
        if not isinstance(pred, dict): pred = {}
        if pred and gold:
            parsed += 1
            gold_labels = {u.get("unit_id"): u["type"] for u in gold.get("labels", gold if isinstance(gold, list) else [])}
            pred_labels = {u.get("unit_id"): u["type"] for u in pred.get("labels", pred if isinstance(pred, list) else [])}
            for uid in gold_labels:
                total += 1
                if pred_labels.get(uid) == gold_labels[uid]:
                    correct += 1
    acc = correct / total if total else 0
    return {"json_ok": parsed, "json_rate": parsed/len(samples) if samples else 0, "accuracy": acc, "total_units": total}


def eval_scene_boundary(model, tok, test_path):
    samples = [json.loads(l) for l in open(test_path, encoding="utf-8")]
    tp, fp, fn, parsed, pair_correct, pair_total = 0, 0, 0, 0, 0, 0
    is_binary_format = False  # auto-detect
    for s in samples:
        msgs = s["messages"]
        resp = generate(model, tok, msgs[:-1])
        pred = extract_json(resp)
        gold = extract_json(msgs[-1]["content"])
        if not isinstance(gold, dict): gold = {}
        if not isinstance(pred, dict): pred = {}
        if pred and gold:
            parsed += 1
            # Support both old (boundaries) and new (decisions) format
            if "decisions" in gold:
                is_binary_format = True
                g_set = set(d["after"] for d in gold["decisions"] if d.get("change"))
                g_all = {d["after"]: d.get("change", False) for d in gold["decisions"]}
            else:
                g_set = set(gold.get("boundaries", []))
            if "decisions" in pred:
                p_set = set(d["after"] for d in pred["decisions"] if d.get("change"))
                # Pair accuracy
                if is_binary_format:
                    p_all = {d["after"]: d.get("change", False) for d in pred["decisions"]}
                    for after in g_all:
                        pair_total += 1
                        if p_all.get(after) == g_all[after]:
                            pair_correct += 1
            else:
                p_set = set(pred.get("boundaries", []))
            tp += len(g_set & p_set)
            fp += len(p_set - g_set)
            fn += len(g_set - p_set)
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    result = {"json_ok": parsed, "json_rate": parsed/len(samples) if samples else 0,
              "precision": prec, "recall": rec, "f1": f1}
    if pair_total > 0:
        result["pair_accuracy"] = pair_correct / pair_total
    return result


def eval_attr_best(model, tok, test_path):
    samples = [json.loads(l) for l in open(test_path, encoding="utf-8")]
    correct, parsed = 0, 0
    for s in samples:
        msgs = s["messages"]
        resp = generate(model, tok, msgs[:-1])
        pred = extract_json(resp)
        gold = extract_json(msgs[-1]["content"])
        if not isinstance(gold, dict): gold = {}
        if not isinstance(pred, dict): pred = {}
        if pred and gold:
            parsed += 1
            if pred.get("best_candidate") == gold.get("best_candidate"):
                correct += 1
    acc = correct / parsed if parsed else 0
    return {"json_ok": parsed, "json_rate": parsed/len(samples) if samples else 0, "best_accuracy": acc}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    EVALS = {
        "narrative-type-classification": eval_narrative_type,
        "scene-boundary-detection": eval_scene_boundary,
        "attribution-best-candidate": eval_attr_best,
        "scene-boundary-binary": eval_scene_boundary,
    }

    all_results = {}

    for model_name, cfg in MODELS.items():
        print(f"\n{'='*60}")
        print(f"  模型: {model_name}")
        print(f"{'='*60}")
        all_results[model_name] = {}

        for agent, eval_fn in EVALS.items():
            print(f"  → {agent}")
            try:
                model, tok = load_model(cfg, agent, device)
                r = eval_fn(model, tok, TEST_DATA[agent])
                all_results[model_name][agent] = r
                print(f"    JSON解析={r['json_rate']:.1%}  {list(r.keys())[-1]}={list(r.values())[-1]:.3f}")
                del model; torch.cuda.empty_cache()
            except Exception as e:
                print(f"    ⚠️ 跳过: {e}")

    # ─── 汇总表格 ───
    print(f"\n{'='*70}")
    print(f"  指标对比")
    print(f"{'='*70}")
    for agent_label, metric_key, metric_name in [
        ("narrative-type-classification", "accuracy", "类型准确率"),
        ("scene-boundary-detection", "f1", "边界F1(旧)"),
        ("scene-boundary-binary", "f1", "边界F1(二元)"),
        ("scene-boundary-binary", "pair_accuracy", "Pair准确率"),
        ("attribution-best-candidate", "best_accuracy", "最佳候选准确率"),
    ]:
        print(f"\n  {metric_name}:")
        for m in MODELS:
            if agent_label not in all_results.get(m, {}):
                continue
            r = all_results[m][agent_label]
            if metric_key not in r:
                continue
            print(f"    {m:12s}  {r[metric_key]:.1%}  (JSON={r['json_rate']:.1%})")

    # 保存
    out_path = Path(__file__).parent / "eval_compare_results.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n  结果保存: {out_path}")


if __name__ == "__main__":
    main()
