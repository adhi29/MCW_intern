# SPDX-License-Identifier: Apache-2.0
"""
Convert float32 ONNX models to float16 or bfloat16.

Usage:
    # Convert a single model to fp16:
    python convert_onnx_precision.py --input model_fp32.onnx --output model_fp16.onnx --precision fp16

    # Convert a single model to bf16:
    python convert_onnx_precision.py --input model_fp32.onnx --output model_fp16.onnx --precision bf16

    # Convert all VLM ONNX models in onnx_models/ to bf16:
    python convert_onnx_precision.py --all --precision bf16

Conversion approach:
  - fp16: Uses onnxruntime float16 converter (well-tested, handles op compatibility)
  - bf16: Manually converts float32 weight initializers to bfloat16 tensor type.
          Keeps compute graph inputs/outputs and certain ops in float32 for compatibility.
"""

import os
import argparse
import numpy as np
import onnx
from onnx import TensorProto, numpy_helper
from onnx.external_data_helper import (
    convert_model_to_external_data,
    load_external_data_for_model,
)


def float32_to_bfloat16_bytes(fp32_data: np.ndarray) -> bytes:
    """Convert float32 numpy array to bfloat16 raw bytes.

    bfloat16 is the upper 16 bits of float32.
    """
    fp32_bytes = fp32_data.astype(np.float32).tobytes()
    bf16_values = []
    for i in range(0, len(fp32_bytes), 4):
        # Take upper 2 bytes of each float32 (big-endian truncation)
        bf16_values.append(fp32_bytes[i + 2:i + 4])
    return b"".join(bf16_values)


def _has_external_data(initializer):
    """Check if an initializer uses external data storage."""
    for entry in initializer.external_data:
        if entry.key == "location":
            return True
    return False


def convert_to_bf16(model_path: str, output_path: str, keep_io_fp32: bool = True):
    """Convert float32 ONNX model to bfloat16.

    Converts weight initializers from float32 to bfloat16.
    Handles models with external data (weights stored as separate files).
    Optionally keeps graph inputs/outputs in float32 for easy numpy interop.
    """
    print(f"  Loading {model_path}...")
    model_dir = os.path.dirname(os.path.abspath(model_path))

    # Load model without external data first (just the graph structure)
    model = onnx.load(model_path, load_external_data=False)

    # Check if any initializer uses external data
    uses_external = any(_has_external_data(init) for init in model.graph.initializer)
    if uses_external:
        print("  Model uses external data storage, loading weights...")
        load_external_data_for_model(model, model_dir)

    converted_count = 0
    skipped_count = 0
    already_half_count = 0

    # Convert initializers (model weights)
    for initializer in model.graph.initializer:
        if initializer.data_type == TensorProto.FLOAT:
            num_elements = 1
            for d in initializer.dims:
                num_elements *= d

            raw_len = len(initializer.raw_data) if initializer.raw_data else 0
            expected_fp32_bytes = num_elements * 4
            expected_bf16_bytes = num_elements * 2

            # Case 1: raw_data matches bfloat16 size — already half-precision
            # (exported from bfloat16 model, metadata incorrectly says FLOAT)
            if raw_len == expected_bf16_bytes and raw_len != expected_fp32_bytes:
                initializer.data_type = TensorProto.BFLOAT16
                # raw_data is already the right bf16 bytes, just fix the type
                del initializer.external_data[:]
                already_half_count += 1
                continue

            # Case 2: normal float32 data — convert to bfloat16
            try:
                fp32_data = numpy_helper.to_array(initializer)
                bf16_bytes = float32_to_bfloat16_bytes(fp32_data)

                initializer.data_type = TensorProto.BFLOAT16
                initializer.raw_data = bf16_bytes
                # Clear float_data if present
                del initializer.float_data[:]
                # Clear external data references since data is now inline
                del initializer.external_data[:]
                converted_count += 1
            except Exception as e:
                print(f"    WARN: Could not convert {initializer.name}: {e}")
                skipped_count += 1

    # Update value_info types for internal tensors
    if not keep_io_fp32:
        for vi in model.graph.input:
            if vi.type.tensor_type.elem_type == TensorProto.FLOAT:
                vi.type.tensor_type.elem_type = TensorProto.BFLOAT16
        for vi in model.graph.output:
            if vi.type.tensor_type.elem_type == TensorProto.FLOAT:
                vi.type.tensor_type.elem_type = TensorProto.BFLOAT16

    parts = [f"  Converted {converted_count} initializers to bfloat16"]
    if already_half_count:
        parts.append(f"{already_half_count} already half-precision (type fixed)")
    if skipped_count:
        parts.append(f"{skipped_count} skipped")
    print(parts[0] + (" (" + ", ".join(parts[1:]) + ")" if len(parts) > 1 else ""))

    # Check total model size — if >2GB, must use external data
    output_dir = os.path.dirname(os.path.abspath(output_path))
    output_basename = os.path.basename(output_path)

    # Try to estimate model size
    total_size = 0
    for initializer in model.graph.initializer:
        total_size += len(initializer.raw_data) if initializer.raw_data else 0
        total_size += len(initializer.float_data) * 4

    if total_size > 1.5 * 1024 * 1024 * 1024:  # >1.5GB, use external data
        print(f"  Model is {total_size / 1024 / 1024 / 1024:.1f} GB, using external data storage...")
        data_file = output_basename.replace(".onnx", "_data.bin")
        convert_model_to_external_data(
            model,
            all_tensors_to_one_file=True,
            location=data_file,
            size_threshold=1024,
        )
        onnx.save(model, output_path)
        data_path = os.path.join(output_dir, data_file)
        data_size = os.path.getsize(data_path) / 1024 / 1024
        print(f"  Saved to: {output_path} + {data_file} ({data_size:.1f} MB)")
    else:
        print(f"  Saving to {output_path}...")
        onnx.save(model, output_path)
        print(f"  Done. Size: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")


def convert_to_fp16(model_path: str, output_path: str):
    """Convert float32 ONNX model to float16 using onnxruntime's graph-aware converter.

    Uses onnxruntime.transformers.float16 which properly handles type propagation
    through the entire graph (initializers, constants, intermediate tensors, ops).
    """
    from onnxruntime.transformers.float16 import convert_float_to_float16

    print(f"  Loading {model_path}...")
    model_dir = os.path.dirname(os.path.abspath(model_path))

    # Load model with external data
    model = onnx.load(model_path, load_external_data=False)
    uses_external = any(_has_external_data(init) for init in model.graph.initializer)
    if uses_external:
        print("  Model uses external data storage, loading weights...")
        load_external_data_for_model(model, model_dir)

    print("  Converting to float16 (graph-aware)...")
    model_fp16 = convert_float_to_float16(
        model,
        keep_io_types=False,
        disable_shape_infer=True,
        force_fp16_initializers=True,
    )

    # Save with external data if needed to avoid 2GB protobuf limit
    output_dir = os.path.dirname(os.path.abspath(output_path))
    output_basename = os.path.basename(output_path)

    total_size = 0
    for initializer in model_fp16.graph.initializer:
        total_size += len(initializer.raw_data) if initializer.raw_data else 0

    if total_size > 1.5 * 1024 * 1024 * 1024:  # >1.5GB, use external data
        data_file = output_basename.replace(".onnx", "_data.bin")
        print(f"  Model is {total_size / 1024 / 1024 / 1024:.1f} GB, using external data storage...")
        convert_model_to_external_data(
            model_fp16,
            all_tensors_to_one_file=True,
            location=data_file,
            size_threshold=1024,
        )
        onnx.save(model_fp16, output_path)
        data_path = os.path.join(output_dir, data_file)
        data_size = os.path.getsize(data_path) / 1024 / 1024
        onnx_size = os.path.getsize(output_path) / 1024 / 1024
        print(f"  Done. Graph: {onnx_size:.1f} MB, Weights: {data_size:.1f} MB")
    else:
        print(f"  Saving to {output_path}...")
        onnx.save(model_fp16, output_path)
        fsize = os.path.getsize(output_path) / 1024 / 1024
        print(f"  Done. Size: {fsize:.1f} MB")


VLM_MODELS = [
    "vlm_embed_tokens.onnx",
    "vlm_lm_head.onnx",
    "vlm_vision_encoder.onnx",
    "vlm_decoder_prefill.onnx",
    "vlm_decoder_decode.onnx",
]


def main():
    parser = argparse.ArgumentParser(description="Convert ONNX models to fp16/bf16")
    parser.add_argument("--input", type=str, help="Input ONNX model path")
    parser.add_argument("--output", type=str, help="Output ONNX model path")
    parser.add_argument(
        "--precision", type=str, choices=["fp16", "bf16"], default="bf16",
        help="Target precision (default: bf16)"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Convert all VLM ONNX models in onnx_models/"
    )
    parser.add_argument(
        "--onnx-dir", type=str, default="./onnx_models",
        help="Directory containing ONNX models (for --all)"
    )
    parser.add_argument(
        "--suffix", type=str, default=None,
        help="Suffix for output files (e.g., '_bf16'). Default: overwrite in place"
    )
    args = parser.parse_args()

    if args.all:
        for model_name in VLM_MODELS:
            input_path = os.path.join(args.onnx_dir, model_name)
            if not os.path.exists(input_path):
                print(f"\nSKIPPED: {model_name} not found")
                continue

            if args.suffix:
                base, ext = os.path.splitext(model_name)
                output_name = f"{base}{args.suffix}{ext}"
            else:
                output_name = model_name

            output_path = os.path.join(args.onnx_dir, output_name)
            print(f"\n{'='*60}")
            print(f"Converting {model_name} to {args.precision}...")

            if args.precision == "bf16":
                convert_to_bf16(input_path, output_path)
            else:
                convert_to_fp16(input_path, output_path)

        print(f"\n{'='*60}")
        print("All conversions complete.")
    elif args.input:
        if not args.output:
            base, ext = os.path.splitext(args.input)
            args.output = f"{base}_{args.precision}{ext}"

        if args.precision == "bf16":
            convert_to_bf16(args.input, args.output)
        else:
            convert_to_fp16(args.input, args.output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
