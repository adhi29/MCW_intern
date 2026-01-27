

import torch

from torch import nn

from transformers import AutoModelForCausalLM, AutoTokenizer
 
MODEL_NAME = "gpt2"

OUTPUT_ONNX = "gpt2_kvcache.onnx"

OPSET = 17
 
print(f"🔄 Loading model: {MODEL_NAME}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)

model.eval()
 
batch_size = 1

seq_len = 4

dummy_inputs = tokenizer("Hello world!", return_tensors="pt")

input_ids = dummy_inputs["input_ids"]

attention_mask = dummy_inputs["attention_mask"]
 
# -----------------------------

# WRAPPER

# -----------------------------

class Qwen3Wrapper(nn.Module):

    def __init__(self, model):

        super().__init__()

        self.model = model
 
    def forward(self, input_ids, attention_mask):

        # For first ONNX export, past_key_values=None

        outputs = self.model(

            input_ids=input_ids,

            attention_mask=attention_mask,

            past_key_values=None,  # important

            use_cache=True

        )

        logits = outputs.logits

        # You can choose to ignore past_key_values for tracing

        return logits
 
wrapper = Qwen3Wrapper(model)
 
# -----------------------------

# EXPORT

# -----------------------------

dynamic_axes = {

    "input_ids": {0: "batch", 1: "seq_len"},

    "attention_mask": {0: "batch", 1: "seq_len"},

}

input_names = ["input_ids", "attention_mask"]

output_names = ["logits"]
 
print(f"🚀 Exporting model to ONNX: {OUTPUT_ONNX}")

torch.onnx.export(

    wrapper,

    (input_ids, attention_mask),

    OUTPUT_ONNX,

    input_names=input_names,

    output_names=output_names,

    dynamic_axes=dynamic_axes,

    opset_version=OPSET,

    do_constant_folding=True,

)
 
print(f" Export complete: {OUTPUT_ONNX}")

 