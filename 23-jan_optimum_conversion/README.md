# Converting GPT-2 to ONNX with Optimum

So I spent some time exploring Hugging Face's Optimum library for converting GPT-2 to ONNX format, and honestly, it's a game changer compared to doing it manually. This folder has a straightforward Python script that shows how simple the process can be.

## The Story Behind This

Remember in the previous project (22-jan_full_converssion) where I manually converted GPT-2 to ONNX? That was a learning experience for sure - dealing with static KV caches, managing all 12 transformer layers, handling padding and truncation... it was a lot. The code worked, but it was complex.

Then I discovered Optimum. It's basically Hugging Face's answer to "how do we make model optimization and export actually easy?" And it delivers. What took me dozens of lines of careful wrapper code before now takes literally three lines with Optimum.

## Why I Switched to Optimum

Here's what sold me on it:

**Simplicity** - The API is so clean. You basically just say "export this model" and it handles everything.

**Smart Defaults** - It automatically enables KV caching, applies graph optimizations, and picks the right ONNX opset version. No guessing.

**Battle-Tested** - This isn't some experimental library. Hugging Face uses this internally for their own production deployments.

**Flexibility** - Works with pretty much any transformer model, not just GPT-2. BERT, T5, you name it.

## What's Inside

I wrote a script with four main parts:

### Converting the Model
The `convert_model_to_onnx()` function does the heavy lifting. It downloads GPT-2 from Hugging Face, converts it to ONNX with all the optimizations, and saves everything to a folder. The whole process takes maybe 20-30 seconds.

### Benchmarking
Once we have the ONNX model, the `run_benchmark()` function tests how fast it is. It runs the model 20 times (with warmup runs first) and gives you average, min, and max inference times. This helps you see the real performance, not just a single lucky run.

### Text Generation
The `generate_some_text()` function is where you actually see the model work. Give it a prompt, and it generates text using the ONNX model. It also tells you how many tokens per second it's generating, which is a nice metric to track.

### PyTorch Comparison
This is my favorite part - `compare_with_pytorch()` runs the same inference on both the original PyTorch model and the ONNX version, then shows you the speedup. Spoiler: ONNX is usually 3-4x faster on CPU.

## Running It

Super simple:

```bash
# Install what you need
pip install optimum[exporters] onnx onnxruntime transformers torch

# Run the script
python gpt2_optimum_conversion.py
```

The script walks through all four steps automatically. You'll see the conversion happen, benchmark results, some generated text, and the PyTorch comparison.

## What I Learned

### The Performance Gains Are Real
On my machine, PyTorch inference was taking around 350ms per forward pass. The ONNX version? About 90ms. That's almost 4x faster, which is huge if you're doing lots of inference.

### KV Caching Just Works
In the manual implementation, I had to carefully manage the KV cache with fixed shapes and padding. Optimum handles all of that automatically with dynamic shapes. It's smarter about memory too.

### The Export Process is Robust
I tried exporting different GPT-2 variants (base, medium, large) and it worked smoothly every time. No weird errors or version conflicts. The library just handles it.

## Comparing Approaches

**Manual Export (Previous Project)**
- Full control over every detail
- Great for learning how ONNX works
- Lots of code to maintain
- Easy to make mistakes

**Optimum (This Project)**
- Super quick to implement
- Production-ready out of the box
- Less code = fewer bugs
- Automatic optimizations

Both approaches have their place. If you're learning or need very specific control, go manual. If you want to ship something that works well, use Optimum.

## Some Gotchas I Hit

**Installation Issues** - Make sure you install `optimum[exporters]` with the brackets. Just `optimum` doesn't include the export tools.

**Model Size** - The ONNX files are pretty big (500+ MB for base GPT-2). Make sure you have space.

**First Run is Slow** - The first time you run it, it downloads the model from Hugging Face. After that, it's cached locally.

## What You Could Do Next

I've got some ideas for extending this:

**Try Bigger Models** - The script works with gpt2-medium, gpt2-large, and gpt2-xl too. Just change the model name. Would be interesting to see how the speedup scales.

**Add Quantization** - Optimum supports INT8 quantization which can make models even faster and smaller. That's probably my next experiment.

**GPU Testing** - I ran everything on CPU, but you can use `onnxruntime-gpu` for CUDA acceleration. The speedup should be even more dramatic.

**Deployment** - Package this up with FastAPI or something similar for a production inference server.


--