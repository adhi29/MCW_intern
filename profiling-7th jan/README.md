# Profiling - 7th Jan

Testing ONNX Runtime profiling on Inception V3 models to compare FP16 vs INT8 performance.

## What's in here

- `inception_v3_fp16.onnx` - FP16 model (45.5 MB)
- `inception_v3_int8.onnx` - INT8 quantized version (22.9 MB) 
- `profiling.py` - script to run profiling
- `int8.json` - session timing data
- `model_profile_2026-01-07_14-58-13.json` - detailed profiling output

## Quick stats

INT8 model is about half the size of FP16 (22.9 MB vs 45.5 MB).

Session initialization times from int8.json:
- Model loading: 27ms
- Session init: 83ms

## Running it

```bash
python profiling.py
```

The script loads the INT8 model, runs inference with a random input (1x3x299x299), and dumps profiling data to a JSON file.

## Profiling output

The big JSON file has node-by-node execution times, memory usage, and thread scheduling info. You can open it in chrome://tracing to visualize where time is being spent.

Main operations profiled:
- Conv layers
- Cast operations (for precision conversion)
- ReLU activations

