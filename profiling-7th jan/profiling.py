import onnxruntime as ort
import numpy as np
import os

# Get the current script's directory (onnx_quantization&profiling folder)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Create session with profiling enabled
session_options = ort.SessionOptions()
session_options.enable_profiling = True

# Set the profile file to save in the current directory
profile_path = os.path.join(current_dir, "model_profile")
session_options.profile_file_prefix = profile_path

# Create inference session
session = ort.InferenceSession("/Users/adhi/Desktop/Multicoreware/profiling/inception_v3_int8.onnx", session_options)

# Prepare dummy input (adjust shape and dtype to match your model)
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape

# Convert symbolic dimensions to actual numbers
# Replace string dimensions with concrete values
concrete_shape = []
for dim in input_shape:
    if isinstance(dim, str):
        concrete_shape.append(1)  # Use 1 for batch size or unknown dimensions
    else:
        concrete_shape.append(dim)

print(f"Input name: {input_name}")
print(f"Original shape: {input_shape}")
print(f"Concrete shape: {concrete_shape}")

#dummy_input = np.random.randn(*concrete_shape).astype(np.float16)
dummy_input = np.random.randint(0, 255, size=concrete_shape, dtype=np.uint8)

# Run inference (this generates profiling data)
outputs = session.run(None, {input_name: dummy_input})

# End profiling and get the profile file path
profile_file = session.end_profiling()
print(f"Profiling data saved to: {profile_file}")