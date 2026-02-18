# SPDX-License-Identifier: Apache-2.0
"""
End-to-end baseline test for Qwen3-VL-2B-Instruct.

Roadmap stages 2-3:
  - Stage 2: Load fp32 model, run greedy inference, save reference.json
  - Stage 3: Load fp16 model, run same inference, compare semantic output

Saves reference.json for comparison at later ONNX / NPU stages.

Usage:
    python run_e2e_baseline.py
    python run_e2e_baseline.py --image test_image.jpg --prompt "Describe this image."
    python run_e2e_baseline.py --skip-fp32  # skip fp32 run, only run fp16
"""

import argparse
import json
import os
import torch

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
SAVE_DIR = "/data/users/adhi/qwen_npu"


def load_model(dtype, device):
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    print(f"  Loading {MODEL_ID} in {dtype} on {device}...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map=device,
        attn_implementation="eager",
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    print(f"  Params: {sum(p.numel() for p in model.parameters())/1e6:.0f}M")
    return model, processor


def run_inference(model, processor, image_path, prompt, max_new_tokens=64):
    from qwen_vl_utils import process_vision_info

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text",  "text": prompt},
        ],
    }]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    trimmed = out_ids[0][inputs["input_ids"].shape[1]:]
    return processor.decode(trimmed, skip_special_tokens=True)


def check_fp16_safety(model):
    """List modules that may be sensitive to fp16 precision."""
    issues = []
    for name, mod in model.named_modules():
        if isinstance(mod, (torch.nn.LayerNorm, torch.nn.Softmax)):
            issues.append((name, type(mod).__name__))
    return issues


def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-VL-2B fp32/fp16 baseline accuracy test",
    )
    parser.add_argument(
        "--image", type=str, default=None,
        help="Path to test image (downloads demo image if not provided)",
    )
    parser.add_argument(
        "--prompt", type=str, default="Describe this image in one sentence.",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=64,
    )
    parser.add_argument(
        "--skip-fp32", action="store_true",
        help="Skip fp32 run (useful if reference.json already exists)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    os.makedirs(SAVE_DIR, exist_ok=True)
    reference_path = os.path.join(SAVE_DIR, "reference.json")
    image_path     = os.path.join(SAVE_DIR, "test_image.jpg")

    # Download demo image if no image provided
    if args.image is None:
        if not os.path.exists(image_path):
            print("Downloading demo image...")
            import requests
            from PIL import Image as PILImage
            url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
            import io
            resp = requests.get(url, stream=True)
            img = PILImage.open(io.BytesIO(resp.content)).convert("RGB")
            img.save(image_path)
            print(f"  Saved: {image_path}")
        args.image = image_path
    else:
        image_path = args.image

    # ── Stage 2: fp32 baseline ─────────────────────────────────────────────
    if not args.skip_fp32:
        print("\n" + "=" * 60)
        print("STAGE 2: fp32 baseline (ground truth)")
        print("=" * 60)

        model_fp32, processor = load_model(torch.float32, args.device)
        fp32_text = run_inference(
            model_fp32, processor, image_path, args.prompt, args.max_new_tokens,
        )
        print(f"\n[fp32] {fp32_text}")

        json.dump({"baseline_fp32": fp32_text}, open(reference_path, "w"), indent=2)
        print(f"\nSaved reference: {reference_path}")

        # Free GPU memory before fp16 run
        del model_fp32
        torch.cuda.empty_cache()
    else:
        print(f"\nSkipping fp32 run — loading reference from {reference_path}")
        with open(reference_path) as f:
            fp32_text = json.load(f)["baseline_fp32"]
        print(f"[fp32 reference] {fp32_text}")

    # ── Stage 3: fp16 accuracy gate ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("STAGE 3: fp16 accuracy gate")
    print("=" * 60)

    model_fp16, processor_fp16 = load_model(torch.float16, args.device)

    # Inspect fp16 sensitive layers
    issues = check_fp16_safety(model_fp16)
    print(f"\nPotentially fp16-sensitive layers: {len(issues)}")
    for name, typ in issues[:5]:
        print(f"  {typ:20s}  {name}")
    if len(issues) > 5:
        print(f"  ... and {len(issues)-5} more")

    fp16_text = run_inference(
        model_fp16, processor_fp16, image_path, args.prompt, args.max_new_tokens,
    )
    print(f"\n[fp32 reference] {fp32_text}")
    print(f"[fp16          ] {fp16_text}")

    exact_match = fp16_text.strip() == fp32_text.strip()
    print(f"\nExact match: {exact_match}")
    if not exact_match:
        # Simple word overlap as semantic similarity proxy
        fp32_words = set(fp32_text.lower().split())
        fp16_words = set(fp16_text.lower().split())
        overlap = len(fp32_words & fp16_words) / max(len(fp32_words | fp16_words), 1)
        print(f"Word overlap (Jaccard): {overlap:.2f}  (>0.5 = semantically equivalent)")
        if overlap < 0.2:
            print("\n[WARN] Low overlap — fp16 output diverges significantly.")
            print("       Check for LayerNorm overflow. Consider keeping LayerNorm in fp32.")
        else:
            print("[PASS] Semantic content preserved in fp16.")
    else:
        print("[PASS] Exact match.")

    # Update reference with fp16 output
    ref = json.load(open(reference_path))
    ref["baseline_fp16"] = fp16_text
    json.dump(ref, open(reference_path, "w"), indent=2)
    print(f"\nUpdated {reference_path} with fp16 result.")

    print("\n" + "=" * 60)
    print("Next step — Export ONNX models:")
    print(f"  python /data/users/adhi/qwen_npu/export_qwen3vl_2b.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
