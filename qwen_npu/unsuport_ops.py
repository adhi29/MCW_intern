import onnx
from collections import Counter


def analyze_onnx_ops(model_path):
    model = onnx.load(model_path)

    # Extract all operators
    ops = []
    for node in model.graph.node:
        ops.append(node.op_type)

    # Count occurrences
    op_counts = Counter(ops)

    print("Operators in your model:")
    print("-" * 50)
    for op, count in sorted(op_counts.items()):
        print(f"{op}: {count}")

    return op_counts


# Use it
model_path = "/data/users/adhi/qwen_npu/onnx_models/common/vlm_vision_encoder.onnx"
ops = analyze_onnx_ops(model_path)
