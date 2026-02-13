#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Export all VLM ONNX components and optionally convert to bf16.
#
# Usage:
#   bash export_all_vlm.sh              # Export fp32 only
#   bash export_all_vlm.sh --bf16       # Export fp32 then convert to bf16

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

CONVERT_BF16=false
if [[ "$1" == "--bf16" ]]; then
    CONVERT_BF16=true
fi

echo "============================================================"
echo "Exporting VLM embed_tokens + lm_head..."
echo "============================================================"
python export_vlm_head.py

echo ""
echo "============================================================"
echo "Exporting VLM vision encoder..."
echo "============================================================"
python export_vlm_vision.py

echo ""
echo "============================================================"
echo "Exporting VLM decoder (prefill + decode)..."
echo "============================================================"
python export_vlm_decoder.py

if $CONVERT_BF16; then
    echo ""
    echo "============================================================"
    echo "Converting all VLM models to bfloat16..."
    echo "============================================================"
    python convert_onnx_precision.py --all --precision bf16
fi

echo ""
echo "============================================================"
echo "All VLM ONNX exports complete!"
echo "============================================================"
ls -lh onnx_models/vlm_*.onnx 2>/dev/null || echo "No VLM ONNX files found"
