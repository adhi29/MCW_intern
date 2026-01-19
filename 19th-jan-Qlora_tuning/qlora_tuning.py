# ============================================================
# QLoRA Fine-Tuning of GPT-2 on Tiny Shakespeare (With Attention Mask)
# ============================================================

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    GPT2TokenizerFast,
    BitsAndBytesConfig,
)
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
MODEL_NAME = "gpt2"
BLOCK_SIZE = 256
BATCH_SIZE = 4
NUM_EPOCHS = 3
LEARNING_RATE = 3e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------------------------------------
# Step 1: Load Dataset
# ------------------------------------------------------------
dataset = load_dataset(
    "text",
    data_files="/Users/adhi/Desktop/Multicoreware/fine_tuning/data/tiny_shakespeare.txt"
)

dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)

# ------------------------------------------------------------
# Step 2: Tokenizer
# ------------------------------------------------------------
tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# ------------------------------------------------------------
# Step 3: Tokenization (keep attention_mask)
# ------------------------------------------------------------
def tokenize_fn(examples):
    return tokenizer(
        examples["text"],
        add_special_tokens=False,
    )

tokenized = dataset.map(
    tokenize_fn,
    batched=True,
    remove_columns=["text"]
)

# ------------------------------------------------------------
# Step 4: Chunk input_ids and attention_mask
# ------------------------------------------------------------
def group_texts(examples):
    concatenated_ids = sum(examples["input_ids"], [])
    concatenated_mask = sum(examples["attention_mask"], [])

    total_length = (len(concatenated_ids) // BLOCK_SIZE) * BLOCK_SIZE

    input_ids = [
        concatenated_ids[i:i + BLOCK_SIZE]
        for i in range(0, total_length, BLOCK_SIZE)
    ]

    attention_mask = [
        concatenated_mask[i:i + BLOCK_SIZE]
        for i in range(0, total_length, BLOCK_SIZE)
    ]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": input_ids.copy(),
    }

lm_dataset = tokenized.map(group_texts, batched=True)

# ------------------------------------------------------------
# Step 5: DataLoader with collate function
# ------------------------------------------------------------
def collate_fn(batch):
    return {
        "input_ids": torch.tensor([x["input_ids"] for x in batch]),
        "attention_mask": torch.tensor([x["attention_mask"] for x in batch]),
        "labels": torch.tensor([x["labels"] for x in batch]),
    }

train_dataloader = DataLoader(
    lm_dataset["train"],
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn
)

# ------------------------------------------------------------
# Step 6: Load GPT-2 in 4-bit (QLoRA base)
# ------------------------------------------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto"
)

# ------------------------------------------------------------
# Step 7: LoRA Configuration
# ------------------------------------------------------------
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["c_attn"],
    bias="none",
)

# Apply LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ------------------------------------------------------------
# Step 8: Optimizer
# ------------------------------------------------------------
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

# ------------------------------------------------------------
# Step 9: Training Loop
# ------------------------------------------------------------
model.train()

for epoch in range(NUM_EPOCHS):
    total_loss = 0.0
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")

    for batch in tqdm(train_dataloader):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_dataloader)
    print(f"Average training loss: {avg_loss:.4f}")

# ------------------------------------------------------------
# Step 10: Save QLoRA Adapters
# ------------------------------------------------------------
model.save_pretrained("qlora-gpt2-shakespeare")
tokenizer.save_pretrained("qlora-gpt2-shakespeare")

# ------------------------------------------------------------
# Step 11: Text Generation Test
# ------------------------------------------------------------
model.eval()

prompt = "ROMEO:"
inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

with torch.no_grad():
    generated = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=120,
        temperature=0.8,
        do_sample=True,
        top_p=0.95,
    )

print("\nGenerated Text:")
print(tokenizer.decode(generated[0], skip_special_tokens=True))


