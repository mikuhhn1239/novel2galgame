#!/usr/bin/env python3
"""
本地测试微调后的模型（不需要改路径，命令行指定）

用法：
  python3 test_model_local.py ./checkpoints/stage2

  或者指定单个 Agent：
  python3 test_model_local.py ./checkpoints/stage2 narrative-parsing

如果没有 GPU，会自动用 CPU（较慢但能跑）。
"""

import sys
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_model(model_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map="auto" if device == "cuda" else None,
    )
    if device == "cpu":
        model = model.float()
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, messages, max_new_tokens=512):
    text = ""
    for msg in messages:
        text += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
    text += "<|im_start|>assistant\n"

    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)


# ── 测试用例 ──────────────────────────────────────────────

def test_narrative_parsing(path):
    model, tok = load_model(path)
    prompt = """请处理下面的小说片段，完成叙事单元切分与类型标注。

文本：
"你怎么来了？"她愣了一下。
我站在门口，雨水顺着伞沿滴落，连鞋边都湿透了。
其实我也不知道自己为什么会来。
便利店的灯光很亮，映得她的脸色有些苍白。"""

    system = """你是一个中文小说叙事解析助手。你的唯一任务是：
1. 将输入的小说片段切分为叙事单元（narrative units）
2. 为每个叙事单元标注类型

你只能使用以下五种类型：dialogue, narration, thought, action, scene_description

输出必须是 JSON 数组。每个元素必须包含 unit_id, text, type。"""

    return generate(model, tok, [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ])


def test_scene_segmentation(path):
    model, tok = load_model(path)
    prompt = """请判断下面小说片段中哪些段落边界应该切换 scene。

段落：
[P1] 下课铃响的时候，教室里一下子热闹起来。
[P2] 她低头收拾书包，像是没有注意到我站在门口。
[P3] 我犹豫了一下，还是走过去叫住了她。
[P4] 十分钟后，我们并肩走在校门外的街道上，风有点冷。
[P5] 她把围巾往上拉了拉，没有说话。"""

    return generate(model, tok, [
        {"role": "system", "content": "你是一个中文小说 scene 切分助手。判断段落边界是否应该切换 scene。"},
        {"role": "user", "content": prompt},
    ])


def test_attribution_assist(path):
    model, tok = load_model(path)
    prompt = """请根据上下文判断目标对话最可能的说话人候选。

候选角色：林秋、苏晚、陈遥

上下文：
[1] "你今天又迟到了。"苏晚把笔放下，看了我一眼。
[2] 我把书包丢到桌上，没接话。
[3] 陈遥坐在后排，像是在憋笑。
[4] "路上堵车。"【目标对话】
[5] 她轻轻哼了一声，把练习册推了过来。"""

    return generate(model, tok, [
        {"role": "system", "content": "你是一个中文小说归因辅助助手。输出最可能的说话人候选排序。"},
        {"role": "user", "content": prompt},
    ])


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 test_model_local.py <checkpoints/stage2> [agent_name]")
        print("  agent_name: narrative-parsing | scene-segmentation | attribution-assist")
        sys.exit(1)

    base = sys.argv[1]
    agents = {
        "narrative-parsing":  test_narrative_parsing,
        "scene-segmentation": test_scene_segmentation,
        "attribution-assist": test_attribution_assist,
    }

    if len(sys.argv) > 2:
        targets = [sys.argv[2]]
    else:
        targets = list(agents.keys())

    for name in targets:
        path = os.path.join(base, name, "final")
        if not os.path.exists(path):
            print(f"Not found: {path}")
            continue
        print(f"\n{'='*60}")
        print(f"  {name}  →  {path}")
        print(f"{'='*60}")
        try:
            result = agents[name](path)
            print(result)
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
