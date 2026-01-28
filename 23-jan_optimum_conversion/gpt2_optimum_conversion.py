

import os
import time
import numpy as np
from pathlib import Path
from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM
import onnxruntime as ort


def convert_model_to_onnx(model_name="gpt2", save_folder="./gpt2_onnx"):
 
    print(f"Let's convert {model_name} to ONNX format!")
   
    
    # Make sure we have a place to save the model
    output_folder = Path(save_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading the {model_name} model from Hugging Face...")
    start = time.time()
    
  
    model = ORTModelForCausalLM.from_pretrained(
        model_name,
        export=True,       
        use_cache=True,     
    )
    
    # Save everything to disk
    model.save_pretrained(save_folder)
    
    elapsed = time.time() - start
    print(f"Done! Took {elapsed:.2f} seconds")
    print(f"Saved to: {output_folder.absolute()}\n")
    
    # Let's see what files we created
    onnx_files = list(output_folder.glob("*.onnx"))
    if onnx_files:
        print("Here's what we exported:")
        total_mb = 0
        for file in onnx_files:
            size_mb = file.stat().st_size / (1024 * 1024)
            total_mb += size_mb
            print(f"  • {file.name}: {size_mb:.2f} MB")
        print(f"  Total: {total_mb:.2f} MB")
    
    return str(output_folder.absolute())


def run_benchmark(model_folder, iterations=20):
   
    
    # Load up the model we just converted
    print("Loading the ONNX model...")
    model = ORTModelForCausalLM.from_pretrained(model_folder)
    tokenizer = AutoTokenizer.from_pretrained(model_folder)
    
    # Let's use a simple test sentence
    test_text = "Artificial intelligence is"
    inputs = tokenizer(test_text, return_tensors="pt")
    
    print(f"Test input: '{test_text}'")
    print(f"Token count: {inputs.input_ids.shape[1]}\n")
    
    # First, warm up the model (this helps get consistent timings)
    print("Warming up the model (5 runs)...")
    for _ in range(5):
        _ = model(**inputs)
    
    # Now let's do the actual benchmark
    print(f"Running {iterations} timed iterations...")
    timings = []
    
    for i in range(iterations):
        start = time.time()
        outputs = model(**inputs)
        timings.append(time.time() - start)
    
    # Calculate some statistics
    avg_ms = np.mean(timings) * 1000
    std_ms = np.std(timings) * 1000
    min_ms = np.min(timings) * 1000
    max_ms = np.max(timings) * 1000
    
    print(f"\nResults:")
    print(f"  Average: {avg_ms:.2f} ms (± {std_ms:.2f} ms)")
    print(f"  Fastest: {min_ms:.2f} ms")
    print(f"  Slowest: {max_ms:.2f} ms")
    print(f"  Output shape: {outputs.logits.shape}")


def generate_some_text(model_folder, prompt="Once upon a time", max_tokens=50):
 
   
    
    # Load the model and tokenizer
    print("Loading model...")
    model = ORTModelForCausalLM.from_pretrained(model_folder)
    tokenizer = AutoTokenizer.from_pretrained(model_folder)
    
    print(f"Starting prompt: '{prompt}'\n")
    print("Generating...")
    
    # Tokenize the input
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Generate text
    start = time.time()
    output_tokens = model.generate(
        **inputs,
        max_length=max_tokens,
        num_beams=1,              # Simple greedy decoding
        do_sample=True,           # Add some randomness
        temperature=0.7,          # Controls randomness (lower = more focused)
        top_p=0.9,               # Nucleus sampling
        pad_token_id=tokenizer.eos_token_id
    )
    elapsed = time.time() - start
    
    # Decode the output
    generated_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    
    
    print(generated_text)
   
    
    tokens_generated = len(output_tokens[0]) - len(inputs.input_ids[0])
    print(f"Generated {tokens_generated} new tokens in {elapsed:.2f} seconds")
    print(f"That's about {tokens_generated/elapsed:.1f} tokens per second")


def compare_with_pytorch(model_name="gpt2", onnx_folder="./gpt2_onnx"):
    
    
    # First, load the original PyTorch model
    print("Loading PyTorch model...")
    pytorch_model = AutoModelForCausalLM.from_pretrained(model_name)
    pytorch_model.eval()  # Put it in evaluation mode
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Now load our ONNX version
    print("Loading ONNX model...")
    onnx_model = ORTModelForCausalLM.from_pretrained(onnx_folder)
    
    # Prepare a test input
    test_text = "The future of technology"
    inputs = tokenizer(test_text, return_tensors="pt")
    
    print(f"\nTest input: '{test_text}'")
    print("Running benchmarks...\n")
    
    # Benchmark PyTorch
    print("Testing PyTorch (with warmup)...")
    # Warmup
    for _ in range(5):
        import torch
        with torch.no_grad():
            _ = pytorch_model(**inputs)
    
    # Actual timing
    pytorch_times = []
    for _ in range(20):
        start = time.time()
        with torch.no_grad():
            _ = pytorch_model(**inputs)
        pytorch_times.append(time.time() - start)
    
    # Benchmark ONNX
    print("Testing ONNX (with warmup)...")
    # Warmup
    for _ in range(5):
        _ = onnx_model(**inputs)
    
    # Actual timing
    onnx_times = []
    for _ in range(20):
        start = time.time()
        _ = onnx_model(**inputs)
        onnx_times.append(time.time() - start)
    
    # Calculate the results
    pytorch_avg = np.mean(pytorch_times) * 1000
    pytorch_std = np.std(pytorch_times) * 1000
    onnx_avg = np.mean(onnx_times) * 1000
    onnx_std = np.std(onnx_times) * 1000
    
    speedup = pytorch_avg / onnx_avg
    
    # Show the results
   
    print(f"PyTorch:  {pytorch_avg:.2f} ms ± {pytorch_std:.2f} ms")
    print(f"ONNX:     {onnx_avg:.2f} ms ± {onnx_std:.2f} ms")
  
    print(f"\nONNX is {speedup:.2f}x faster! 🚀")
    
    # Calculate percentage improvement
    improvement = ((pytorch_avg - onnx_avg) / pytorch_avg) * 100
    print(f"That's a {improvement:.1f}% speed improvement")


# Main execution
if __name__ == "__main__":
   
    print("GPT-2 to ONNX Conversion - Using Optimum")
    
    
    # Configuration - change these if you want
    MODEL_NAME = "gpt2"           # Which model to use
    OUTPUT_FOLDER = "./gpt2_onnx"  # Where to save it
    
    # Step 1: Convert the model
    print("\n[Step 1] Converting model to ONNX...")
    model_path = convert_model_to_onnx(
        model_name=MODEL_NAME,
        save_folder=OUTPUT_FOLDER
    )
    
    # Step 2: Benchmark it
    print("\n[Step 2] Benchmarking the ONNX model...")
    run_benchmark(model_path, iterations=20)
    
    # Step 3: Generate some text
    print("\n[Step 3] Testing text generation...")
    generate_some_text(
        model_path,
        prompt="The impact of artificial intelligence on society",
        max_tokens=100
    )
    
    # Step 4: Compare with PyTorch (if available)
    print("\n[Step 4] Comparing with PyTorch...")
    try:
        import torch
        compare_with_pytorch(MODEL_NAME, model_path)
    except ImportError:
        print("PyTorch not installed, skipping comparison")
        print("(Install with: pip install torch)")
    

   
    
    print(f"You can find it in: {OUTPUT_FOLDER}")
    
