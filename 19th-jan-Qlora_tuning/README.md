# QLoRA Fine-tuning Experiments

This folder has my work on QLoRA (Quantized LoRA) fine-tuning for GPT-2.

## Files

- **qlora_tuning.py** - Python script for QLoRA fine-tuning on Tiny Shakespeare dataset
- **Qlora_with_evaluation.ipynb** - Notebook version with evaluation metrics

## What I Learned

QLoRA is basically LoRA but with 4-bit quantization, which makes it even more memory efficient. The model is loaded in 4-bit format and then LoRA adapters are added on top.

Tried fine-tuning GPT-2 on the Shakespeare dataset to see if it can generate text in that style. The script uses the BitsAndBytes library for quantization.

## Key Points

- Uses 4-bit quantization (NF4 format)
- Only trains LoRA adapters, not the full model
- Much lower memory usage compared to regular fine-tuning
- Tested text generation with different prompts

## Results

The model generates Shakespeare-style text after training. Still experimenting with hyperparameters like learning rate and LoRA rank to get better results.

---
*Experiments done on 19th January*
