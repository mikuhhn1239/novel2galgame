#!/usr/bin/env python3
"""
快速测试微调后的模型

用法：
  # 测试全部三个 Agent
  python test_model.py

  # 测试指定的 Agent
  python test_model.py narrative-parsing
  python test_model.py scene-segmentation
  python test_model.py attribution-assist
"""

import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 模型路径
MODEL_PATHS = {
    "narrative-parsing": "/workspace/project-nas-1000073/已移除-用户名/data/checkpoints/stage2/narrative-parsing/final",
    "scene-segmentation": "/workspace/project-nas-1000073/已移除-用户名/data/checkpoints/stage2/scene-segmentation/final",
    "attribution-assist": "/workspace/project-nas-1000073/已移除-用户名/data/checkpoints/stage2/attribution-assist/final",
}


def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, messages, max_new_tokens=1024):
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


# ============================================================
# 测试用例
# ============================================================

def test_narrative_parsing(model_path):
    """Agent 1: 叙事单元切分 + 类型标注"""
    print(f"\n{'='*60}")
    print(f"  Agent 1: Narrative Parsing — 叙事单元切分")
    print(f"  Model: {model_path}")
    print(f"{'='*60}")

    model, tokenizer = load_model(model_path)

    prompt = """请处理下面的小说片段，完成叙事单元切分与类型标注。

章节标题：第 37 章

文本：
"你怎么来了？"她愣了一下。
我站在门口，雨水顺着伞沿滴落，连鞋边都湿透了。
其实我也不知道自己为什么会来，只是走到这里的时候，已经没有回头路了。
便利店的灯光很亮，映得她的脸色有些苍白。"""

    messages = [
        {"role": "system", "content": "你是一个中文小说叙事解析助手..."},
        {"role": "user", "content": prompt},
    ]
    # 用数据里真实的 system prompt
    sample_system = """你是一个中文小说叙事解析助手。你的唯一任务是：

1. 将输入的小说片段切分为叙事单元（narrative units）
2. 为每个叙事单元标注类型

你只能使用以下五种类型：
- dialogue
- narration
- thought
- action
- scene_description

你必须严格遵守以下规则：

【任务边界】
- 你只负责叙事单元切分和类型标注
- 你不负责 speaker attribution
- 你不负责角色识别
- 你不负责 scene 切分

【切分原则】
- 按叙事功能切分，而不是机械按句号切分
- 如果一句中包含明显不同的叙事功能，可以拆成多个单元

【输出要求】
输出必须是 JSON 数组。每个元素必须包含 unit_id, text, type。"""

    messages[0]["content"] = sample_system
    result = generate(model, tokenizer, messages)
    print(result)


def test_scene_segmentation(model_path):
    """Agent 2: 场景边界识别"""
    print(f"\n{'='*60}")
    print(f"  Agent 2: Scene Segmentation — 场景边界识别")
    print(f"  Model: {model_path}")
    print(f"{'='*60}")

    model, tokenizer = load_model(model_path)

    prompt = """请判断下面小说片段中哪些段落边界应该切换 scene。

段落：
[P1] 下课铃响的时候，教室里一下子热闹起来。
[P2] 她低头收拾书包，像是没有注意到我站在门口。
[P3] 我犹豫了一下，还是走过去叫住了她。
[P4] 十分钟后，我们并肩走在校门外的街道上，风有点冷。
[P5] 她把围巾往上拉了拉，没有说话。"""

    messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": prompt},
    ]
    result = generate(model, tokenizer, messages)
    print(result)


def test_attribution_assist(model_path):
    """Agent 3: 角色归因"""
    print(f"\n{'='*60}")
    print(f"  Agent 3: Attribution Assist — 角色归因")
    print(f"  Model: {model_path}")
    print(f"{'='*60}")

    model, tokenizer = load_model(model_path)

    prompt = """请根据上下文判断目标对话最可能的说话人候选。

候选角色：
- 林秋
- 苏晚
- 陈遥

上下文：
[1] "你今天又迟到了。"苏晚把笔放下，看了我一眼。
[2] 我把书包丢到桌上，没接话。
[3] 陈遥坐在后排，像是在憋笑。
[4] "路上堵车。"【目标对话】
[5] 她轻轻哼了一声，把练习册推了过来。"""

    messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": prompt},
    ]
    result = generate(model, tokenizer, messages)
    print(result)


def main():
    if len(sys.argv) > 1:
        task = sys.argv[1]
        if task not in MODEL_PATHS:
            print(f"Unknown agent: {task}")
            print(f"Available: {list(MODEL_PATHS.keys())}")
            sys.exit(1)
        tasks = [task]
    else:
        tasks = list(MODEL_PATHS.keys())

    test_fns = {
        "narrative-parsing": test_narrative_parsing,
        "scene-segmentation": test_scene_segmentation,
        "attribution-assist": test_attribution_assist,
    }

    for task in tasks:
        model_path = MODEL_PATHS[task]
        try:
            test_fns[task](model_path)
        except Exception as e:
            print(f"Error testing {task}: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
