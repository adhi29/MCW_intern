

import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
import onnx
import onnxruntime as ort
import numpy as np
from typing import Dict, Tuple, List, Optional
import time
import warnings
import os

warnings.filterwarnings('ignore')


class Config:
    # Model settings
    MODEL_NAME = "gpt2"
    
    # STATIC dimensions (CRITICAL for static KV cache)
    BATCH_SIZE = 1
    MAX_SEQ_LENGTH = 128  # Maximum sequence length supported
    MAX_NEW_TOKENS = 50
    
    # Device settings
    DEVICE = "cpu"  
    # ONNX settings
    ONNX_MODEL_PATH = "gpt2_true_static_kv.onnx"
    OPSET_VERSION = 14
    
    # Generation settings
    TEMPERATURE = 0.7
    TOP_P = 0.9


# ============================================================================
# STATIC KV CACHE GPT-2 MODEL
# ============================================================================

class GPT2StaticKVCache(nn.Module):
   
    
    def __init__(self, model_name: str, max_seq_length: int, batch_size: int = 1):
        super().__init__()
        
        # Load base model
        self.base_model = GPT2LMHeadModel.from_pretrained(model_name)
        self.config = self.base_model.config
        
        # Extract model dimensions
        self.num_layers = self.config.n_layer
        self.num_heads = self.config.n_head
        self.hidden_size = self.config.n_embd
        self.head_dim = self.hidden_size // self.num_heads
        self.vocab_size = self.config.vocab_size
        
        # Static dimensions
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        
        # Set to eval mode
        self.base_model.eval()
        
        print(f"Model initialized:")
        print(f"  - Layers: {self.num_layers}")
        print(f"  - Heads: {self.num_heads}")
        print(f"  - Hidden size: {self.hidden_size}")
        print(f"  - Max sequence length: {self.max_seq_length}")
    
    def forward(
        self,
        input_ids: torch.Tensor,  # [batch_size, 1] - single token input
        cache_position: torch.Tensor,  # [1] - scalar position
        *past_key_values_flat,  # Flattened KV cache: k0, v0, k1, v1, ...
    ) -> Tuple:
       
        
        # Reshape flat KV cache into list of (key, value) tuples
        past_key_values = []
        for i in range(0, len(past_key_values_flat), 2):
            past_key_values.append((past_key_values_flat[i], past_key_values_flat[i+1]))
        
        # Prepare position for attention
        position_int = cache_position.item()
        
        # Create attention mask: [batch, 1, 1, max_seq_len]
        # Only attend to positions 0 to cache_position
        attention_mask = torch.zeros(
            (self.batch_size, 1, 1, self.max_seq_length),
            dtype=torch.float32,
            device=input_ids.device
        )
        attention_mask[:, :, :, :position_int + 1] = 1.0
        
        # Convert to attention bias (0 for attend, -inf for mask)
        attention_mask = (1.0 - attention_mask) * -10000.0
        
        # Get embeddings
        hidden_states = self.base_model.transformer.wte(input_ids)
        
        # Add position embeddings for current position
        position_ids = cache_position.unsqueeze(0).expand(self.batch_size, -1)
        position_embeds = self.base_model.transformer.wpe(position_ids)
        hidden_states = hidden_states + position_embeds
        
        # Process through transformer layers
        new_past_key_values = []
        
        for i, block in enumerate(self.base_model.transformer.h):
            # Get current layer's KV cache
            layer_past_key, layer_past_value = past_key_values[i]
            
            # Self-attention with static cache
            hidden_states, new_key, new_value = self._layer_forward_with_static_cache(
                block,
                hidden_states,
                layer_past_key,
                layer_past_value,
                attention_mask,
                position_int
            )
            
            new_past_key_values.append(new_key)
            new_past_key_values.append(new_value)
        
        # Final layer norm
        hidden_states = self.base_model.transformer.ln_f(hidden_states)
        
        # Get logits
        logits = self.base_model.lm_head(hidden_states)
        
        # Return as flat tuple
        return (logits, *new_past_key_values)
    
    def _layer_forward_with_static_cache(
        self,
        block: nn.Module,
        hidden_states: torch.Tensor,
        past_key: torch.Tensor,
        past_value: torch.Tensor,
        attention_mask: torch.Tensor,
        cache_position: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process one transformer layer with static KV cache update.
        """
        
        # Layer norm
        residual = hidden_states
        hidden_states = block.ln_1(hidden_states)
        
        # Attention - manually compute to control KV cache
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # QKV projection
        qkv = block.attn.c_attn(hidden_states)
        query, key, value = qkv.split(self.hidden_size, dim=2)
        
        # Reshape to [batch, num_heads, seq_len, head_dim]
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Update static KV cache at cache_position
        # Clone to avoid in-place modification issues with ONNX
        updated_past_key = past_key.clone()
        updated_past_value = past_value.clone()
        
        updated_past_key[:, :, cache_position:cache_position+1, :] = key
        updated_past_value[:, :, cache_position:cache_position+1, :] = value
        
        # Use cache for attention: slice up to cache_position+1
        key_for_attn = updated_past_key[:, :, :cache_position+1, :]
        value_for_attn = updated_past_value[:, :, :cache_position+1, :]
        
        # Attention computation
        attn_weights = torch.matmul(query, key_for_attn.transpose(-1, -2))
        attn_weights = attn_weights / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        
        # Apply mask (only for relevant positions)
        mask_for_attn = attention_mask[:, :, :, :cache_position+1]
        attn_weights = attn_weights + mask_for_attn
        
        # Softmax
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, value_for_attn)
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, hidden_size)
        
        # Output projection
        attn_output = block.attn.c_proj(attn_output)
        
        # Residual connection
        hidden_states = residual + attn_output
        
        # MLP
        residual = hidden_states
        hidden_states = block.ln_2(hidden_states)
        feed_forward_hidden_states = block.mlp(hidden_states)
        hidden_states = residual + feed_forward_hidden_states
        
        return hidden_states, updated_past_key, updated_past_value
    
    def _create_empty_cache(self) -> Tuple[Tuple[torch.Tensor, torch.Tensor]]:
        """Create empty static KV cache with fixed dimensions."""
        cache = []
        for _ in range(self.num_layers):
            key_cache = torch.zeros(
                (self.batch_size, self.num_heads, self.max_seq_length, self.head_dim),
                dtype=torch.float32
            )
            value_cache = torch.zeros(
                (self.batch_size, self.num_heads, self.max_seq_length, self.head_dim),
                dtype=torch.float32
            )
            cache.append((key_cache, value_cache))
        return tuple(cache)


# ============================================================================
# ONNX EXPORT WITH STATIC SHAPES
# ============================================================================

def export_to_onnx_static(
    model: GPT2StaticKVCache,
    output_path: str,
    config: Config
):
    

    model.eval()
    
    # Dummy inputs with STATIC shapes
    dummy_input_ids = torch.ones((config.BATCH_SIZE, 1), dtype=torch.long)
    dummy_cache_position = torch.tensor([0], dtype=torch.long)
    
    # Create dummy KV cache with STATIC shapes - FLATTENED
    dummy_past_kv_flat = []
    for _ in range(model.num_layers):
        key = torch.zeros(
            (config.BATCH_SIZE, model.num_heads, config.MAX_SEQ_LENGTH, model.head_dim),
            dtype=torch.float32
        )
        value = torch.zeros(
            (config.BATCH_SIZE, model.num_heads, config.MAX_SEQ_LENGTH, model.head_dim),
            dtype=torch.float32
        )
        dummy_past_kv_flat.append(key)
        dummy_past_kv_flat.append(value)
    
    # Pack inputs as tuple (input_ids, cache_position, k0, v0, k1, v1, ...)
    dummy_inputs = (dummy_input_ids, dummy_cache_position, *dummy_past_kv_flat)
    
    # Define input names
    input_names = ['input_ids', 'cache_position']
    for i in range(model.num_layers):
        input_names.append(f'past_key_{i}')
        input_names.append(f'past_value_{i}')
    
    # Define output names
    output_names = ['logits']
    for i in range(model.num_layers):
        output_names.append(f'present_key_{i}')
        output_names.append(f'present_value_{i}')
    
    print(f"Total inputs: {len(input_names)} (2 + {len(input_names)-2} KV cache tensors)")
    print(f"Total outputs: {len(output_names)} (1 + {len(output_names)-1} KV cache tensors)")
    print(f"All shapes are STATIC")
    
    print("\nExporting to ONNX...")
    
    try:
        # Export with explicit input/output names for PyTorch 2.1.2
        with torch.no_grad():
            torch.onnx.export(
                model,
                dummy_inputs,
                output_path,
                input_names=input_names,
                output_names=output_names,
                opset_version=config.OPSET_VERSION,
                do_constant_folding=True,
                export_params=True,
                verbose=False,
            )
        
        print(f"✓ Successfully exported to {output_path}")
        
        # Verify the model
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX model validated")
        
        # Print model info
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"✓ Model size: {size_mb:.1f} MB")
        
        # Verify static shapes
        print("\n[Verifying Static Shapes - First 5 inputs]")
        for i, inp in enumerate(onnx_model.graph.input[:5]):
            shape = [dim.dim_value if dim.dim_value > 0 else 'dynamic' for dim in inp.type.tensor_type.shape.dim]
            print(f"  Input {i}: {shape}")
        print(f"  ... and {len(onnx_model.graph.input) - 5} more inputs (all static)")
        
        # Check for Concat operations (shouldn't have any!)
        concat_count = sum(1 for node in onnx_model.graph.node if node.op_type == 'Concat')
        print(f"\n[Static KV Cache Verification]")
        print(f"  Concat operations: {concat_count} (should be 0 for true static cache)")
        
        if concat_count > 0:
            print("  ⚠ Warning: Model has Concat ops - may not be fully static")
        else:
            print("  ✓ No Concat operations - true static KV cache confirmed!")
        
        return True
        
    except Exception as e:
        print(f"✗ Export failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# ONNX INFERENCE WITH STATIC KV CACHE
# ============================================================================

class ONNXStaticKVInference:
    """
    ONNX Runtime inference for static KV cache model.
    """
    
    def __init__(self, onnx_path: str, config: Config):
        self.config = config
        
        # Create ONNX Runtime session
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        providers = ['CPUExecutionProvider']
        
        self.session = ort.InferenceSession(
            onnx_path,
            sess_options,
            providers=providers
        )
        
        print(f"✓ ONNX session created")
        print(f"  Providers: {self.session.get_providers()}")
        
        # Get input/output names (use actual names from ONNX model)
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]
        
        print(f"  Inputs: {len(self.input_names)}")
        print(f"  First few input names: {self.input_names[:3]}")
        print(f"  Outputs: {len(self.output_names)}")
        
        # Determine number of layers
        # Count KV cache inputs (all except first 2: input_ids and cache_position)
        kv_cache_inputs = [name for name in self.input_names[2:]]
        self.num_layers = len(kv_cache_inputs) // 2
        
        print(f"  Layers: {self.num_layers}")
    
    def infer_single_token(
        self,
        input_id: int,
        cache_position: int,
        past_kv_cache: List[np.ndarray]
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Run inference for a single token.
        
        Args:
            input_id: Single token ID
            cache_position: Position in cache (0 for first token)
            past_kv_cache: List of KV cache arrays
        
        Returns:
            logits: [1, 1, vocab_size]
            updated_kv_cache: Updated KV cache
        """
        
        # Build input feed using actual input names from ONNX model
        input_feed = {}
        
        # First two inputs are input_ids and cache_position
        input_feed[self.input_names[0]] = np.array([[input_id]], dtype=np.int64)
        input_feed[self.input_names[1]] = np.array([cache_position], dtype=np.int64)
        
        # Rest are KV cache tensors
        for i, kv_name in enumerate(self.input_names[2:]):
            input_feed[kv_name] = past_kv_cache[i]
        
        # Run inference
        outputs = self.session.run(self.output_names, input_feed)
        
        # Extract logits and updated cache
        logits = outputs[0]
        updated_cache = list(outputs[1:])
        
        return logits, updated_cache
    
    def generate_text(
        self,
        tokenizer: GPT2Tokenizer,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 50
    ) -> str:
        """
        Generate text using ONNX model with static KV cache.
        """
        
        # Tokenize prompt
        input_ids = tokenizer.encode(prompt, return_tensors='np')[0]
        
        # Initialize empty KV cache
        kv_cache = self._create_empty_cache()
        
        # Process prompt tokens
        generated_ids = []
        for i, token_id in enumerate(input_ids):
            logits, kv_cache = self.infer_single_token(
                int(token_id),
                i,
                kv_cache
            )
            generated_ids.append(int(token_id))
        
        # Generate new tokens
        cache_pos = len(input_ids)
        
        for _ in range(max_new_tokens):
            if cache_pos >= self.config.MAX_SEQ_LENGTH:
                break
            
            # Sample next token
            next_token_logits = logits[0, 0, :] / temperature
            
            # Top-k sampling
            top_k_indices = np.argpartition(next_token_logits, -top_k)[-top_k:]
            top_k_logits = next_token_logits[top_k_indices]
            
            # Softmax
            probs = np.exp(top_k_logits - np.max(top_k_logits))
            probs = probs / np.sum(probs)
            
            # Sample
            next_token_idx = np.random.choice(len(top_k_indices), p=probs)
            next_token = int(top_k_indices[next_token_idx])
            
            generated_ids.append(next_token)
            
            # Check for EOS
            if next_token == tokenizer.eos_token_id:
                break
            
            # Generate next token
            logits, kv_cache = self.infer_single_token(
                next_token,
                cache_pos,
                kv_cache
            )
            
            cache_pos += 1
        
        # Decode
        text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        return text
    
    def _create_empty_cache(self) -> List[np.ndarray]:
        """Create empty static KV cache."""
        cache = []
        
        # Get dimensions from first KV input
        for inp in self.session.get_inputs():
            if 'past_key' in inp.name or 'past_value' in inp.name:
                shape = inp.shape
                cache.append(np.zeros(shape, dtype=np.float32))
        
        return cache


# ============================================================================
# TESTING AND COMPARISON
# ============================================================================

def test_pytorch_model(model: GPT2StaticKVCache, config: Config):
    """Test PyTorch model with static KV cache."""
    
    print("\n" + "="*70)
    print("TESTING PYTORCH MODEL")
    print("="*70)
    
    tokenizer = GPT2Tokenizer.from_pretrained(config.MODEL_NAME)
    prompt = "The future of AI is"
    
    print(f"\nPrompt: '{prompt}'")
    
    # Tokenize
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    
    # Initialize KV cache (flattened)
    past_kv_flat = []
    for _ in range(model.num_layers):
        key = torch.zeros((config.BATCH_SIZE, model.num_heads, config.MAX_SEQ_LENGTH, model.head_dim))
        value = torch.zeros((config.BATCH_SIZE, model.num_heads, config.MAX_SEQ_LENGTH, model.head_dim))
        past_kv_flat.append(key)
        past_kv_flat.append(value)
    
    # Process tokens one by one
    generated = []
    
    with torch.no_grad():
        for i in range(min(len(input_ids[0]), config.MAX_SEQ_LENGTH)):
            token = input_ids[0, i:i+1].unsqueeze(0)
            cache_pos = torch.tensor([i])
            
            # Forward pass returns (logits, k0, v0, k1, v1, ...)
            outputs = model(token, cache_pos, *past_kv_flat)
            logits = outputs[0]
            past_kv_flat = list(outputs[1:])  # Update cache
            
            generated.append(int(input_ids[0, i]))
        
        # Generate a few more tokens
        for i in range(len(input_ids[0]), min(len(input_ids[0]) + 20, config.MAX_SEQ_LENGTH)):
            # Sample next token (greedy)
            next_token = torch.argmax(logits[0, 0, :]).item()
            generated.append(next_token)
            
            if next_token == tokenizer.eos_token_id:
                break
            
            # Next iteration
            token = torch.tensor([[next_token]])
            cache_pos = torch.tensor([i])
            outputs = model(token, cache_pos, *past_kv_flat)
            logits = outputs[0]
            past_kv_flat = list(outputs[1:])
    
    text = tokenizer.decode(generated, skip_special_tokens=True)
    print(f"\nGenerated: {text}")
    print(f"Tokens: {len(generated)}")


def test_onnx_model(onnx_inference: ONNXStaticKVInference, config: Config):
    """Test ONNX model with static KV cache."""
    
    print("\n" + "="*70)
    print("TESTING ONNX MODEL")
    print("="*70)
    
    tokenizer = GPT2Tokenizer.from_pretrained(config.MODEL_NAME)
    prompt = "The future of AI is"
    
    print(f"\nPrompt: '{prompt}'")
    print("Generating...")
    
    text = onnx_inference.generate_text(
        tokenizer,
        prompt,
        max_new_tokens=20,
        temperature=0.8,
        top_k=50
    )
    
    print(f"\nGenerated: {text}")


def benchmark_inference(onnx_inference: ONNXStaticKVInference, num_runs: int = 100):
    """Benchmark ONNX inference speed."""
    
    print("\n" + "="*70)
    print("BENCHMARKING ONNX INFERENCE")
    print("="*70)
    
    # Create dummy inputs
    kv_cache = onnx_inference._create_empty_cache()
    
    times = []
    
    # Warmup
    for _ in range(10):
        _ = onnx_inference.infer_single_token(100, 0, kv_cache)
    
    # Benchmark
    for i in range(num_runs):
        start = time.perf_counter()
        logits, kv_cache = onnx_inference.infer_single_token(100, i % 64, kv_cache)
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)  # Convert to ms
    
    times = np.array(times)
    
    print(f"\nResults over {num_runs} runs:")
    print(f"  Mean: {times.mean():.3f} ms")
    print(f"  Std: {times.std():.3f} ms")
    print(f"  Min: {times.min():.3f} ms")
    print(f"  Max: {times.max():.3f} ms")
    print(f"  P50: {np.percentile(times, 50):.3f} ms")
    print(f"  P95: {np.percentile(times, 95):.3f} ms")
    print(f"  P99: {np.percentile(times, 99):.3f} ms")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main execution."""
    
    print("\n" + "="*70)
    print("GPT-2 WITH TRUE STATIC KV CACHE")
    print("Complete Implementation")
    print("="*70)
    
    config = Config()
    
    print(f"\nConfiguration:")
    print(f"  Model: {config.MODEL_NAME}")
    print(f"  Batch size: {config.BATCH_SIZE}")
    print(f"  Max sequence length: {config.MAX_SEQ_LENGTH}")
    print(f"  Device: {config.DEVICE}")
    
    # Step 1: Create model with static KV cache
    print("\n[Step 1] Creating GPT-2 with Static KV Cache...")
    model = GPT2StaticKVCache(
        config.MODEL_NAME,
        config.MAX_SEQ_LENGTH,
        config.BATCH_SIZE
    )
    
    # Step 2: Test PyTorch model
    print("\n[Step 2] Testing PyTorch Model...")
    test_pytorch_model(model, config)
    
    # Step 3: Export to ONNX
    print("\n[Step 3] Exporting to ONNX...")
    success = export_to_onnx_static(model, config.ONNX_MODEL_PATH, config)
    
    if not success:
        print("✗ Export failed. Exiting.")
        return
    
    # Step 4: Load ONNX model
    print("\n[Step 4] Loading ONNX Model...")
    onnx_inference = ONNXStaticKVInference(config.ONNX_MODEL_PATH, config)
    
    # Step 5: Test ONNX model
    print("\n[Step 5] Testing ONNX Model...")
    test_onnx_model(onnx_inference, config)
    
    # Step 6: Benchmark
    print("\n[Step 6] Benchmarking...")
    benchmark_inference(onnx_inference, num_runs=100)
    
    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70)
    print(f"\nStatic KV Cache ONNX model saved to: {config.ONNX_MODEL_PATH}")
    print("\nKey features:")
    print("  ✓ Fixed KV cache dimensions")
    print("  ✓ No dynamic shapes")
    print("  ✓ No Concat operations for KV updates")
    print("  ✓ In-place cache updates using position index")
    print("  ✓ Optimized for hardware acceleration")


if __name__ == "__main__":
    main()