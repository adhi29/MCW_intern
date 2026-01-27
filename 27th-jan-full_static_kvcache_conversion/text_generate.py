import numpy as np
import onnxruntime as ort
from transformers import GPT2Tokenizer



BATCH_SIZE = 1
PROMPT_LEN = 8         # must match export
MAX_SEQ_LEN = 128
EOS_TOKEN_ID = 50256

# Sampling hyperparameters (GPT-2 friendly)
TEMPERATURE = 0.8
TOP_K = 40
TOP_P = 0.9
REPETITION_PENALTY = 1.05
MIN_TOKENS_BEFORE_EOS = 8


tokenizer = GPT2Tokenizer.from_pretrained("gpt2")


init_sess = ort.InferenceSession(
    "gpt2_init_static.onnx",
    providers=["CPUExecutionProvider"]
)

decode_sess = ort.InferenceSession(
    "gpt2_decode_static.onnx",
    providers=["CPUExecutionProvider"]
)


def build_attention_mask(past_seq_len, max_seq_len):
    """
    Mask valid tokens only.
    Shape: [1, max_seq_len]
    """
    mask = np.zeros((1, max_seq_len), dtype=np.int64)
    mask[:, :past_seq_len + 1] = 1
    return mask


def apply_repetition_penalty(logits, generated_tokens, penalty):
    """
    Penalize older tokens only (not the most recent ones).
    """
    for token in set(generated_tokens[:-5]):
        logits[token] /= penalty
    return logits


def top_p_filtering(logits, top_p):
    """
    Nucleus (top-p) filtering.
    """
    sorted_idx = np.argsort(logits)[::-1]
    sorted_logits = logits[sorted_idx]

    probs = np.exp(sorted_logits - np.max(sorted_logits))
    probs = probs / np.sum(probs)
    cumulative_probs = np.cumsum(probs)

    sorted_logits[cumulative_probs > top_p] = -1e10

    filtered_logits = np.full_like(logits, -1e10)
    filtered_logits[sorted_idx] = sorted_logits
    return filtered_logits


def sample_next_token(logits, temperature, top_k):
    """
    Temperature + Top-K sampling.
    """
    logits = logits / temperature

    top_k = min(top_k, logits.shape[-1])
    indices = np.argpartition(logits, -top_k)[-top_k:]
    filtered_logits = logits[indices]

    probs = np.exp(filtered_logits - np.max(filtered_logits))
    probs = probs / np.sum(probs)

    return int(np.random.choice(indices, p=probs))



prompt = "Cristiano Ronaldo is"

input_ids = tokenizer.encode(prompt, return_tensors="np")

if input_ids.shape[1] > PROMPT_LEN:
    raise ValueError(
        f"Prompt too long ({input_ids.shape[1]}) for PROMPT_LEN={PROMPT_LEN}"
    )



pad_len = PROMPT_LEN - input_ids.shape[1]

# 🔴 IMPORTANT: pad with 0, NOT EOS
input_ids = np.pad(
    input_ids,
    ((0, 0), (0, pad_len)),
    constant_values=0
)

attention_mask_init = np.zeros_like(input_ids, dtype=np.int64)
attention_mask_init[:, :PROMPT_LEN - pad_len] = 1



logits, K, V = init_sess.run(
    None,
    {
        "input_ids": input_ids.astype(np.int64),
        "attention_mask": attention_mask_init.astype(np.int64),
    }
)


past_seq_len = PROMPT_LEN - pad_len
current_token = input_ids[:, past_seq_len - 1:past_seq_len]

generated_tokens = input_ids[0, :past_seq_len].tolist()



while past_seq_len < MAX_SEQ_LEN:

    attention_mask = build_attention_mask(past_seq_len, MAX_SEQ_LEN)

    logits, K, V = decode_sess.run(
        None,
        {
            "input_ids": current_token.astype(np.int64),
            "past_key": K,
            "past_value": V,
            "past_seq_len": np.array(past_seq_len, dtype=np.int64),
            "attention_mask": attention_mask,
        }
    )

    step_logits = logits[0, -1]

    # repetition penalty
    step_logits = apply_repetition_penalty(
        step_logits,
        generated_tokens,
        REPETITION_PENALTY
    )

    # nucleus sampling
    step_logits = top_p_filtering(step_logits, TOP_P)

    # sample
    next_token = sample_next_token(
        step_logits,
        TEMPERATURE,
        TOP_K
    )

    # EOS handling
    if next_token == EOS_TOKEN_ID and len(generated_tokens) > MIN_TOKENS_BEFORE_EOS:
        break

    generated_tokens.append(next_token)
    current_token = np.array([[next_token]], dtype=np.int64)
    past_seq_len += 1



output_text = tokenizer.decode(
    generated_tokens,
    skip_special_tokens=True
)

print("\n Generated text:\n")
print(output_text)
