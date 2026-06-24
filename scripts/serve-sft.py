"""
Lightweight OpenAI-compatible API server using transformers + bitsandbytes.
For WSL2 where vLLM bitsandbytes UVA is not available.

Usage:
    pip install transformers flask bitsandbytes accelerate torch
    python3 serve-sft.py

API: http://localhost:8000/v1/chat/completions
"""

import os, json, time, uuid, torch, signal, threading
from flask import Flask, request, jsonify

MODEL_PATH = os.environ.get("MODEL_PATH", "/mnt/d/Project/novel2glagame/model/qwen3-8b-novel-base-sft")
PORT = int(os.environ.get("PORT", "8000"))

print(f"Loading model: {MODEL_PATH}")
print("Using 4-bit quantization (NF4)")

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Free memory: {torch.cuda.mem_get_info(0)[0] / 1024**3:.1f} GB")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map={"": 0},  # Force all layers to GPU 0
    trust_remote_code=True,
    torch_dtype=torch.float16,
)
print(f"Model loaded. Device: {next(model.parameters()).device}")
print(f"GPU memory after load: {torch.cuda.memory_allocated(0) / 1024**3:.1f} GB allocated, {torch.cuda.memory_reserved(0) / 1024**3:.1f} GB reserved")

def generate_with_timeout(model, inputs, timeout_seconds=120, **kwargs):
    """Run model.generate in a thread with timeout."""
    result = [None]
    error = [None]

    def _run():
        try:
            with torch.no_grad():
                result[0] = model.generate(**inputs, **kwargs)
        except Exception as e:
            error[0] = e

    t = threading.Thread(target=_run)
    t.start()
    t.join(timeout_seconds)
    if t.is_alive():
        raise TimeoutError(f"Generation timed out after {timeout_seconds}s")
    if error[0]:
        raise error[0]
    return result[0]


app = Flask(__name__)

@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    data = request.json
    messages = data.get("messages", [])
    max_tokens = data.get("max_tokens", 512)
    temperature = data.get("temperature", 0.3)

    # Apply chat template (keep thinking tags - model was trained with them)
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        try:
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
                top_p=data.get("top_p", 0.95),
            )
        except Exception as e:
            return jsonify({"error": f"Generation failed: {str(e)}"}), 500

    generated = outputs[0][inputs["input_ids"].shape[-1]:]
    content = tokenizer.decode(generated, skip_special_tokens=True).strip()

    response_msg = {"role": "assistant", "content": content}

    return jsonify({
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "qwen3-8b-sft",
        "choices": [{
            "index": 0,
            "message": response_msg,
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": inputs["input_ids"].shape[-1],
            "completion_tokens": len(generated),
            "total_tokens": inputs["input_ids"].shape[-1] + len(generated),
        },
    })

@app.route("/v1/models", methods=["GET"])
def list_models():
    return jsonify({"data": [{"id": "qwen3-8b-sft", "object": "model"}]})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

print(f"Starting server on port {PORT}...")
app.run(host="0.0.0.0", port=PORT, threaded=True)
