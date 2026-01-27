import numpy as np
import onnxruntime as ort
from transformers import GPT2Tokenizer

# ============================================================
# CONFIG
# ============================================================

ONNX_PATH = "gpt2_dynamic_kv.onnx"
MAX_NEW_TOKENS = 50
EOS_TOKEN_ID = 50256

TEMPERATURE = 0.8
TOP_K = 40

# ============================================================
# TOKENIZER
# ============================================================

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# ============================================================
# LOAD ONNX MODEL
# ============================================================

sess = ort.InferenceSession(
    ONNX_PATH,
    providers=["CPUExecutionProvider"]
)

input_names = [i.name for i in sess.get_inputs()]
output_names = [o.name for o in sess.get_outputs()]

print("Inputs:", input_names)
print("Outputs:", output_names)

# ============================================================
# HELPER
# ============================================================

def sample_next_token(logits, temperature=1.0, top_k=40):
    logits = logits / temperature
    top_k = min(top_k, logits.shape[-1])
    idx = np.argpartition(logits, -top_k)[-top_k:]
    probs = np.exp(logits[idx] - np.max(logits[idx]))
    probs = probs / np.sum(probs)
    return int(np.random.choice(idx, p=probs))

# ============================================================
# PROMPT
# ============================================================

prompt = "The future of AI is"
input_ids = tokenizer.encode(prompt, return_tensors="np")

generated_tokens = input_ids[0].tolist()

# ============================================================
# 🔹 PHASE 1: FULL PROMPT (NO KV INPUTS)
# ============================================================

inputs = {
    "input_ids": input_ids.astype(np.int64)
}

outputs = sess.run(None, inputs)

logits = outputs[0]
past_kv = outputs[1:]   # flattened KV tensors

# ============================================================
# 🔹 PHASE 2: TOKEN-BY-TOKEN DECODE
# ============================================================

for _ in range(MAX_NEW_TOKENS):

    next_token = sample_next_token(
        logits[0, -1],
        temperature=TEMPERATURE,
        top_k=TOP_K
    )

    if next_token == EOS_TOKEN_ID:
        break

    generated_tokens.append(next_token)

    input_ids = np.array([[next_token]], dtype=np.int64)

    inputs = {
        "input_ids": input_ids
    }

    # Feed ALL KV inputs (must be ≥1 length now)
    for name, kv in zip(input_names[1:], past_kv):
        inputs[name] = kv

    outputs = sess.run(None, inputs)

    logits = outputs[0]
    past_kv = outputs[1:]

# ============================================================
# OUTPUT
# ============================================================

text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

print("\n📝 Generated text:\n")
print(text)
