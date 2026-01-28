#!/usr/bin/env python3
"""
Gemma ONNX Conversion with Static KV Cache
Converts Gemma model from Hugging Face to ONNX format and compares with PyTorch reference.
"""

import argparse
import json
import time
import numpy as np
import torch
import onnx
import onnxruntime as ort
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast


class GemmaPrefillWrapper(torch.nn.Module):
    """Wrapper for Gemma prefill phase with static shapes."""
    
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.config = config
        
    def forward(self, input_ids, attention_mask):
        """
        Prefill forward pass.
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            
        Returns:
            logits: [batch_size, seq_len, vocab_size]
            past_key_values: List of (key, value) tuples for each layer
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True
        )
        
        # Flatten past_key_values for ONNX export
        # Convert from list of tuples to separate outputs
        logits = outputs.logits
        past_kvs = outputs.past_key_values
        
        # Return logits and flattened KV cache
        kv_outputs = []
        for layer_past in past_kvs:
            kv_outputs.append(layer_past[0])  # key
            kv_outputs.append(layer_past[1])  # value
            
        return (logits, *kv_outputs)


class GemmaDecodeWrapper(torch.nn.Module):
    """Wrapper for Gemma decode phase with static KV cache."""
    
    def __init__(self, model, config, num_layers):
        super().__init__()
        self.model = model
        self.config = config
        self.num_layers = num_layers
        
    def forward(self, input_ids, *past_key_values_flat):
        """
        Decode forward pass with KV cache.
        
        Args:
            input_ids: [batch_size, 1]
            past_key_values_flat: Flattened list of keys and values
            
        Returns:
            logits: [batch_size, 1, vocab_size]
            updated_past_key_values: Flattened list of updated keys and values
        """
        # Reconstruct past_key_values from flat inputs
        past_kvs = []
        for i in range(0, len(past_key_values_flat), 2):
            key = past_key_values_flat[i]
            value = past_key_values_flat[i + 1]
            past_kvs.append((key, value))
        
        outputs = self.model(
            input_ids=input_ids,
            past_key_values=past_kvs,
            use_cache=True,
            return_dict=True
        )
        
        # Flatten updated KV cache
        logits = outputs.logits
        updated_kvs = outputs.past_key_values
        
        kv_outputs = []
        for layer_past in updated_kvs:
            kv_outputs.append(layer_past[0])  # key
            kv_outputs.append(layer_past[1])  # value
            
        return (logits, *kv_outputs)


class GemmaONNXConverter:
    """Main converter class for Gemma to ONNX."""
    
    def __init__(self, model_name: str, hf_token: str, output_dir: str = "./gemma_onnx"):
        self.model_name = model_name
        self.hf_token = hf_token
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.tokenizer = None
        self.model = None
        self.config = None
        
        # Static shapes
        self.batch_size = 1
        self.prefill_seq_len = 32
        self.decode_seq_len = 1
        
    def load_model(self):
        """Load Gemma model from Hugging Face."""
        print(f"Loading model: {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            token=self.hf_token
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            token=self.hf_token,
            torch_dtype=torch.float32,  # Use FP32 for ONNX export
            device_map="cpu"  # Keep on CPU for export
        )
        
        self.model.eval()
        self.config = self.model.config
        
        print(f"✓ Model loaded successfully")
        print(f"  Config: {self.config.num_hidden_layers} layers, "
              f"{self.config.num_attention_heads} heads, "
              f"{self.config.hidden_size} hidden size")
        
    def export_prefill_model(self) -> str:
        """Export prefill model to ONNX."""
        print("\n" + "="*60)
        print("Exporting Prefill Model")
        print("="*60)
        
        prefill_wrapper = GemmaPrefillWrapper(self.model, self.config)
        
        # Create dummy inputs
        dummy_input_ids = torch.randint(
            0, self.config.vocab_size,
            (self.batch_size, self.prefill_seq_len),
            dtype=torch.long
        )
        dummy_attention_mask = torch.ones(
            (self.batch_size, self.prefill_seq_len),
            dtype=torch.long
        )
        
        # Define input names
        input_names = ["input_ids", "attention_mask"]
        
        # Define output names
        output_names = ["logits"]
        num_layers = self.config.num_hidden_layers
        for i in range(num_layers):
            output_names.append(f"past_key_{i}")
            output_names.append(f"past_value_{i}")
        
        # Dynamic axes for potential flexibility
        dynamic_axes = {
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size", 1: "sequence_length"}
        }
        
        # Add dynamic axes for KV cache
        for i in range(num_layers):
            dynamic_axes[f"past_key_{i}"] = {0: "batch_size", 2: "sequence_length"}
            dynamic_axes[f"past_value_{i}"] = {0: "batch_size", 2: "sequence_length"}
        
        output_path = self.output_dir / "gemma_prefill.onnx"
        
        print(f"Exporting to: {output_path}")
        print(f"Input shapes: input_ids={dummy_input_ids.shape}, "
              f"attention_mask={dummy_attention_mask.shape}")
        
        with torch.no_grad():
            # Use legacy exporter for better compatibility
            torch.onnx._export(
                prefill_wrapper,
                (dummy_input_ids, dummy_attention_mask),
                str(output_path),
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                opset_version=14,
                do_constant_folding=True,
                export_params=True,
                verbose=False,
                export_modules_as_functions=False
            )
        
        print(f"✓ Prefill model exported successfully")
        
        # Validate ONNX model
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        print(f"✓ ONNX model validation passed")
        
        return str(output_path)
    
    def export_decode_model(self) -> str:
        """Export decode model to ONNX."""
        print("\n" + "="*60)
        print("Exporting Decode Model")
        print("="*60)
        
        num_layers = self.config.num_hidden_layers
        num_heads = self.config.num_key_value_heads if hasattr(self.config, 'num_key_value_heads') else self.config.num_attention_heads
        head_dim = self.config.head_dim if hasattr(self.config, 'head_dim') else self.config.hidden_size // self.config.num_attention_heads
        
        decode_wrapper = GemmaDecodeWrapper(self.model, self.config, num_layers)
        
        # Create dummy inputs
        dummy_input_ids = torch.randint(
            0, self.config.vocab_size,
            (self.batch_size, self.decode_seq_len),
            dtype=torch.long
        )
        
        # Create dummy past_key_values
        dummy_past_kvs = []
        for _ in range(num_layers):
            # Key: [batch, num_heads, seq_len, head_dim]
            key = torch.randn(
                self.batch_size, num_heads, self.prefill_seq_len, head_dim,
                dtype=torch.float32
            )
            value = torch.randn(
                self.batch_size, num_heads, self.prefill_seq_len, head_dim,
                dtype=torch.float32
            )
            dummy_past_kvs.extend([key, value])
        
        # Define input names
        input_names = ["input_ids"]
        for i in range(num_layers):
            input_names.append(f"past_key_{i}")
            input_names.append(f"past_value_{i}")
        
        # Define output names
        output_names = ["logits"]
        for i in range(num_layers):
            output_names.append(f"present_key_{i}")
            output_names.append(f"present_value_{i}")
        
        # Dynamic axes
        dynamic_axes = {
            "input_ids": {0: "batch_size"},
            "logits": {0: "batch_size"}
        }
        
        # Add dynamic axes for KV cache
        for i in range(num_layers):
            dynamic_axes[f"past_key_{i}"] = {0: "batch_size", 2: "past_sequence_length"}
            dynamic_axes[f"past_value_{i}"] = {0: "batch_size", 2: "past_sequence_length"}
            dynamic_axes[f"present_key_{i}"] = {0: "batch_size", 2: "sequence_length"}
            dynamic_axes[f"present_value_{i}"] = {0: "batch_size", 2: "sequence_length"}
        
        output_path = self.output_dir / "gemma_decode.onnx"
        
        print(f"Exporting to: {output_path}")
        print(f"Input shapes: input_ids={dummy_input_ids.shape}, "
              f"KV cache={dummy_past_kvs[0].shape}")
        
        with torch.no_grad():
            # Use legacy exporter for better compatibility
            torch.onnx._export(
                decode_wrapper,
                (dummy_input_ids, *dummy_past_kvs),
                str(output_path),
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                opset_version=14,
                do_constant_folding=True,
                export_params=True,
                verbose=False,
                export_modules_as_functions=False
            )
        
        print(f"✓ Decode model exported successfully")
        
        # Validate ONNX model
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        print(f"✓ ONNX model validation passed")
        
        return str(output_path)
    
    def compare_with_reference(self, reference_file: str = "gemma_reference_outputs.npz"):
        """Compare ONNX outputs with PyTorch reference."""
        print("\n" + "="*60)
        print("Comparing ONNX with PyTorch Reference")
        print("="*60)
        
        # Load reference outputs
        ref_path = Path(reference_file)
        if not ref_path.exists():
            print(f"⚠ Reference file not found: {reference_file}")
            print("  Skipping comparison. Run gemma_phase1.ipynb first to generate reference.")
            return None
        
        ref_data = np.load(reference_file, allow_pickle=True)
        print(f"✓ Loaded reference data from {reference_file}")
        
        # Load ONNX models
        prefill_session = ort.InferenceSession(
            str(self.output_dir / "gemma_prefill.onnx"),
            providers=['CPUExecutionProvider']
        )
        decode_session = ort.InferenceSession(
            str(self.output_dir / "gemma_decode.onnx"),
            providers=['CPUExecutionProvider']
        )
        
        results = {}
        
        # Compare prefill phase
        print("\n--- Prefill Phase Comparison ---")
        input_ids = ref_data['input_ids']
        attention_mask = ref_data['attention_mask']
        ref_prefill_logits = ref_data['prefill_logits']
        
        # Run ONNX prefill
        prefill_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
        
        start_time = time.time()
        prefill_outputs = prefill_session.run(None, prefill_inputs)
        prefill_time = time.time() - start_time
        
        onnx_prefill_logits = prefill_outputs[0]
        onnx_past_kvs = prefill_outputs[1:]
        
        # Compare logits
        logits_mse = np.mean((ref_prefill_logits - onnx_prefill_logits) ** 2)
        logits_max_diff = np.max(np.abs(ref_prefill_logits - onnx_prefill_logits))
        
        # Compare next token prediction
        ref_next_token = ref_data['next_token']
        onnx_next_token = np.argmax(onnx_prefill_logits[:, -1, :], axis=-1)
        tokens_match = np.array_equal(ref_next_token, onnx_next_token)
        
        print(f"  Logits MSE: {logits_mse:.2e}")
        print(f"  Logits Max Diff: {logits_max_diff:.2e}")
        print(f"  Next Token Match: {tokens_match}")
        print(f"  PyTorch next token: {ref_next_token}")
        print(f"  ONNX next token: {onnx_next_token}")
        print(f"  ONNX Prefill Time: {prefill_time*1000:.2f} ms")
        
        results['prefill'] = {
            'logits_mse': float(logits_mse),
            'logits_max_diff': float(logits_max_diff),
            'tokens_match': bool(tokens_match),
            'pytorch_next_token': int(ref_next_token.item()) if hasattr(ref_next_token, 'item') else int(ref_next_token),
            'onnx_next_token': int(onnx_next_token.item()) if hasattr(onnx_next_token, 'item') else int(onnx_next_token),
            'inference_time_ms': float(prefill_time * 1000)
        }
        
        # Compare decode phase
        print("\n--- Decode Phase Comparison ---")
        decode_input_ids = ref_data['decode_input_ids']
        ref_decode_logits = ref_data['decode_logits']
        
        # Prepare KV cache inputs for decode
        decode_inputs = {"input_ids": decode_input_ids}
        num_layers = len(onnx_past_kvs) // 2
        
        for i in range(num_layers):
            decode_inputs[f"past_key_{i}"] = onnx_past_kvs[i * 2]
            decode_inputs[f"past_value_{i}"] = onnx_past_kvs[i * 2 + 1]
        
        # Run ONNX decode
        start_time = time.time()
        decode_outputs = decode_session.run(None, decode_inputs)
        decode_time = time.time() - start_time
        
        onnx_decode_logits = decode_outputs[0]
        
        # Compare logits
        decode_logits_mse = np.mean((ref_decode_logits - onnx_decode_logits) ** 2)
        decode_logits_max_diff = np.max(np.abs(ref_decode_logits - onnx_decode_logits))
        
        print(f"  Logits MSE: {decode_logits_mse:.2e}")
        print(f"  Logits Max Diff: {decode_logits_max_diff:.2e}")
        print(f"  ONNX Decode Time: {decode_time*1000:.2f} ms")
        
        results['decode'] = {
            'logits_mse': float(decode_logits_mse),
            'logits_max_diff': float(decode_logits_max_diff),
            'inference_time_ms': float(decode_time * 1000)
        }
        
        # Overall summary
        print("\n" + "="*60)
        print("Comparison Summary")
        print("="*60)
        
        accuracy_threshold = 1e-3
        prefill_accurate = logits_max_diff < accuracy_threshold
        decode_accurate = decode_logits_max_diff < accuracy_threshold
        
        print(f"Prefill Accuracy: {'✓ PASS' if prefill_accurate else '✗ FAIL'} "
              f"(max diff: {logits_max_diff:.2e}, threshold: {accuracy_threshold:.2e})")
        print(f"Decode Accuracy: {'✓ PASS' if decode_accurate else '✗ FAIL'} "
              f"(max diff: {decode_logits_max_diff:.2e}, threshold: {accuracy_threshold:.2e})")
        print(f"Token Prediction: {'✓ MATCH' if tokens_match else '✗ MISMATCH'}")
        
        results['summary'] = {
            'prefill_accurate': prefill_accurate,
            'decode_accurate': decode_accurate,
            'tokens_match': tokens_match,
            'accuracy_threshold': accuracy_threshold
        }
        
        # Save results
        results_path = self.output_dir / "onnx_comparison_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Results saved to {results_path}")
        
        return results
    
    def benchmark_onnx(self, num_iterations: int = 10):
        """Benchmark ONNX model performance."""
        print("\n" + "="*60)
        print("Benchmarking ONNX Models")
        print("="*60)
        
        # Load ONNX models
        prefill_session = ort.InferenceSession(
            str(self.output_dir / "gemma_prefill.onnx"),
            providers=['CPUExecutionProvider']
        )
        decode_session = ort.InferenceSession(
            str(self.output_dir / "gemma_decode.onnx"),
            providers=['CPUExecutionProvider']
        )
        
        # Create dummy inputs
        input_ids = np.random.randint(
            0, self.config.vocab_size,
            (self.batch_size, self.prefill_seq_len),
            dtype=np.int64
        )
        attention_mask = np.ones(
            (self.batch_size, self.prefill_seq_len),
            dtype=np.int64
        )
        
        # Benchmark prefill
        print(f"\nBenchmarking Prefill ({num_iterations} iterations)...")
        prefill_times = []
        
        for i in range(num_iterations):
            start = time.time()
            outputs = prefill_session.run(
                None,
                {"input_ids": input_ids, "attention_mask": attention_mask}
            )
            prefill_times.append(time.time() - start)
        
        prefill_mean = np.mean(prefill_times) * 1000
        prefill_std = np.std(prefill_times) * 1000
        
        print(f"  Mean: {prefill_mean:.2f} ms")
        print(f"  Std: {prefill_std:.2f} ms")
        
        # Prepare decode inputs
        decode_input_ids = np.random.randint(
            0, self.config.vocab_size,
            (self.batch_size, self.decode_seq_len),
            dtype=np.int64
        )
        
        past_kvs = outputs[1:]
        decode_inputs = {"input_ids": decode_input_ids}
        num_layers = len(past_kvs) // 2
        
        for i in range(num_layers):
            decode_inputs[f"past_key_{i}"] = past_kvs[i * 2]
            decode_inputs[f"past_value_{i}"] = past_kvs[i * 2 + 1]
        
        # Benchmark decode
        print(f"\nBenchmarking Decode ({num_iterations} iterations)...")
        decode_times = []
        
        for i in range(num_iterations):
            start = time.time()
            _ = decode_session.run(None, decode_inputs)
            decode_times.append(time.time() - start)
        
        decode_mean = np.mean(decode_times) * 1000
        decode_std = np.std(decode_times) * 1000
        
        print(f"  Mean: {decode_mean:.2f} ms")
        print(f"  Std: {decode_std:.2f} ms")
        
        print(f"\nTokens/sec (decode): {1000/decode_mean:.2f}")
        
        return {
            'prefill_mean_ms': prefill_mean,
            'prefill_std_ms': prefill_std,
            'decode_mean_ms': decode_mean,
            'decode_std_ms': decode_std,
            'tokens_per_sec': 1000 / decode_mean
        }


def main():
    parser = argparse.ArgumentParser(description="Convert Gemma model to ONNX with static KV cache")
    parser.add_argument("--model", type=str, default="google/gemma-1.1-2b-it",
                        help="Hugging Face model name")
    parser.add_argument("--token", type=str, required=True,
                        help="Hugging Face access token")
    parser.add_argument("--output-dir", type=str, default="./gemma_onnx",
                        help="Output directory for ONNX models")
    parser.add_argument("--reference", type=str, default="gemma_reference_outputs.npz",
                        help="Path to PyTorch reference outputs")
    parser.add_argument("--skip-export", action="store_true",
                        help="Skip ONNX export (use existing models)")
    parser.add_argument("--skip-comparison", action="store_true",
                        help="Skip comparison with reference")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run performance benchmark")
    
    args = parser.parse_args()
    
    print("="*60)
    print("Gemma ONNX Conversion with Static KV Cache")
    print("="*60)
    
    converter = GemmaONNXConverter(
        model_name=args.model,
        hf_token=args.token,
        output_dir=args.output_dir
    )
    
    if not args.skip_export:
        converter.load_model()
        converter.export_prefill_model()
        converter.export_decode_model()
    else:
        print("\n⚠ Skipping ONNX export (using existing models)")
        converter.load_model()
    
    if not args.skip_comparison:
        converter.compare_with_reference(args.reference)
    
    if args.benchmark:
        converter.benchmark_onnx()
    
    print("\n" + "="*60)
    print("✓ Conversion Complete!")
    print("="*60)
    print(f"Output directory: {converter.output_dir}")
    print(f"  - gemma_prefill.onnx")
    print(f"  - gemma_decode.onnx")
    print(f"  - onnx_comparison_results.json")


if __name__ == "__main__":
    main()
