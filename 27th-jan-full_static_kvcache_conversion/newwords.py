import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer



BATCH_SIZE = 1
MAX_SEQ_LEN = 128
NUM_LAYERS = 12
NUM_HEADS = 12
HEAD_DIM = 64
HIDDEN_SIZE = 768
OPSET = 13


class GPT2Init(nn.Module):
    """
    Runs the full prompt once.
    Produces:
      - logits
      - fully allocated KV cache (static size)
    """

    def __init__(self, model: GPT2LMHeadModel):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True
        )

        past = outputs.past_key_values  # tuple of (K, V)

        # Preallocate STATIC KV cache
        K = torch.zeros(
            NUM_LAYERS,
            BATCH_SIZE,
            NUM_HEADS,
            MAX_SEQ_LEN,
            HEAD_DIM,
            device=input_ids.device
        )
        V = torch.zeros_like(K)

        seq_len = input_ids.shape[1]

        # Copy prompt KV into static buffer
        for layer in range(NUM_LAYERS):
            k, v = past[layer]  # [1, heads, seq, head_dim]
            K[layer, :, :, :seq_len, :] = k
            V[layer, :, :, :seq_len, :] = v

        return outputs.logits, K, V


# ============================================================
# 3. DECODE MODEL (ONE TOKEN STEP, STATIC KV + MASK)
# ============================================================

class GPT2Decode(nn.Module):
    """
    Decodes ONE token at a time using STATIC KV cache.
    Attention mask is REQUIRED for correct causal behavior.
    """

    def __init__(self, model: GPT2LMHeadModel):
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids,
        past_key,
        past_value,
        past_seq_len,
        attention_mask
    ):
        """
        input_ids:      [1, 1]
        past_key:       [layers, 1, heads, max_seq, head_dim]
        past_value:     same
        past_seq_len:   scalar int
        attention_mask: [1, max_seq_len]
        """

        # Rebuild past_key_values tuple expected by GPT-2
        past = []
        for layer in range(NUM_LAYERS):
            k = past_key[layer, :, :, :past_seq_len, :]
            v = past_value[layer, :, :, :past_seq_len, :]
            past.append((k, v))
        
        attn_mask = attention_mask[:, :past_seq_len + 1]

        outputs = self.model(
            input_ids=input_ids,
            past_key_values=tuple(past),
            attention_mask=attn_mask,
            use_cache=True,
            return_dict=True
        )

        new_past = outputs.past_key_values

        # Write NEW KV at FIXED index
        for layer in range(NUM_LAYERS):
            new_k, new_v = new_past[layer]
            past_key[layer, :, :, past_seq_len, :] = new_k[:, :, -1, :]
            past_value[layer, :, :, past_seq_len, :] = new_v[:, :, -1, :]

        return outputs.logits, past_key, past_value


# ============================================================
# 4. LOAD MODEL
# ============================================================

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

init_model = GPT2Init(model)
decode_model = GPT2Decode(model)


# ============================================================
# 5. DUMMY STATIC INPUTS (DEFINE ONNX SHAPES)
# ============================================================

PROMPT_LEN = 8  # must match inference padding

dummy_input_ids_init = torch.zeros(
    BATCH_SIZE, PROMPT_LEN, dtype=torch.long
)
dummy_attention_mask_init = torch.ones_like(dummy_input_ids_init)

dummy_input_ids_decode = torch.zeros(
    BATCH_SIZE, 1, dtype=torch.long
)

dummy_K = torch.zeros(
    NUM_LAYERS,
    BATCH_SIZE,
    NUM_HEADS,
    MAX_SEQ_LEN,
    HEAD_DIM
)
dummy_V = torch.zeros_like(dummy_K)

dummy_past_seq_len = torch.tensor(1, dtype=torch.long)

dummy_attention_mask_decode = torch.ones(
    BATCH_SIZE, MAX_SEQ_LEN, dtype=torch.long
)


# ============================================================
# 6. EXPORT INIT ONNX
# ============================================================

torch.onnx.export(
    init_model,
    (dummy_input_ids_init, dummy_attention_mask_init),
    "gpt2_init_static.onnx",
    opset_version=OPSET,
    input_names=["input_ids", "attention_mask"],
    output_names=["logits", "past_key", "past_value"],
    dynamic_axes=None
)

print("✅ Exported gpt2_init_static.onnx")


# ============================================================
# 7. EXPORT DECODE ONNX (STATIC KV + MASK)
# ============================================================

torch.onnx.export(
    decode_model,
    (
        dummy_input_ids_decode,
        dummy_K,
        dummy_V,
        dummy_past_seq_len,
        dummy_attention_mask_decode
    ),
    "gpt2_decode_static.onnx",
    opset_version=OPSET,
    input_names=[
        "input_ids",
        "past_key",
        "past_value",
        "past_seq_len",
        "attention_mask"
    ],
    output_names=[
        "logits",
        "updated_key",
        "updated_value"
    ],
    dynamic_axes=None
)

print(" Exported gpt2_decode_static.onnx")
