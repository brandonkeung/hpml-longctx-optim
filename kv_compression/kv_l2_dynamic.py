"""
Simple manual decode loop with L2 KV cache compression.
Strategy: prefill -> compress cache once -> decode with compressed cache
"""

import torch
from .cache import l2_compress


def generate_with_l2_compress(
    model,
    tokenizer,
    inputs,
    max_new_tokens=20,
    keep_ratio=0.9,
    prune_after=2048,
    skip_layers=None,
    **generate_kwargs
):
    """
    Generate text with L2-based KV cache compression.
    
    Args:
        model: The language model
        tokenizer: The tokenizer (not used, for compatibility)
        inputs: Dict with 'input_ids' and optionally 'attention_mask'
        max_new_tokens: Number of tokens to generate
        keep_ratio: Fraction of cache to keep (0.5 = keep 50%)
        prune_after: Start compression when seq_len >= this value
        skip_layers: List of layer indices to skip compression
        **generate_kwargs: Additional generation arguments (do_sample, etc.)
    
    Returns:
        generated_ids: Tensor of shape (batch_size, input_len + generated_len)
        ttft_ms: Time to first token in milliseconds
    """
    if skip_layers is None:
        skip_layers = []
    
    input_ids = inputs['input_ids']
    attention_mask = inputs.get('attention_mask', None)
    batch_size = input_ids.shape[0]
    device = input_ids.device
    
    # Generation settings
    do_sample = generate_kwargs.get('do_sample', False)
    pad_token_id = generate_kwargs.get('pad_token_id', None)
    eos_token_id = generate_kwargs.get('eos_token_id', None)
    
    with torch.no_grad():
        # Step 1: Prefill - process entire input prompt
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )
        
        past_key_values = outputs.past_key_values
        
        # Step 2: Compress KV cache if prompt is long enough
        if past_key_values is not None:
            # Convert DynamicCache to tuple if needed
            if hasattr(past_key_values, 'to_legacy_cache'):
                legacy_cache = past_key_values.to_legacy_cache()
            else:
                legacy_cache = past_key_values
            
            print(f"compressing with keep_ratio={keep_ratio}, prune_after={prune_after}, skip_layers={skip_layers}")
            # Apply L2 compression
            compressed_cache = l2_compress(
                legacy_cache,
                keep_ratio=keep_ratio,
                prune_after=prune_after,
                skip_layers=skip_layers,
            )
            
            # Convert back to DynamicCache if needed
            if hasattr(past_key_values, 'from_legacy_cache'):
                from transformers import DynamicCache
                past_key_values = DynamicCache.from_legacy_cache(compressed_cache)
            else:
                past_key_values = compressed_cache
        
        # Step 3: Manual decode loop
        generated = []
        ttft_ms = 0.0
        
        for step in range(max_new_tokens):
            # Get next token logits
            logits = outputs.logits[:, -1, :]
            
            # Sample or greedy decode
            if do_sample:
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            
            generated.append(next_token)
            
            # Check for EOS
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
            
            # Prepare for next iteration
            if step < max_new_tokens - 1:
                # Update attention mask
                if attention_mask is not None:
                    attention_mask = torch.cat([
                        attention_mask,
                        torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=device)
                    ], dim=-1)
                
                # Measure TTFT for first decode step
                if step == 0:
                    torch.cuda.synchronize()
                    ttft_start = torch.cuda.Event(enable_timing=True)
                    ttft_end = torch.cuda.Event(enable_timing=True)
                    ttft_start.record()
                
                # Forward pass with cached keys/values
                outputs = model(
                    input_ids=next_token,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
                past_key_values = outputs.past_key_values
                
                # Record TTFT after first decode step
                if step == 0:
                    ttft_end.record()
                    torch.cuda.synchronize()
                    ttft_ms = ttft_start.elapsed_time(ttft_end)
        
        # Concatenate input and generated tokens
        if generated:
            output_ids = torch.cat([input_ids] + generated, dim=-1)
        else:
            output_ids = input_ids
    
    return output_ids, ttft_ms
