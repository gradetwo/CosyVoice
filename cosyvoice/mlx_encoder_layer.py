import mlx.core as mx
import mlx.nn as nn
from typing import Tuple, Optional
import sys
import os

# Add cosyvoice root to sys.path to allow finding the cosyvoice package
# This assumes the script is run from the 'tools/' directory or similar context
# where 'cosyvoice' is a sibling or in PYTHONPATH.
# For robustness, check if parent is already in path to avoid multiple additions if script is re-run in same session
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Attempt to import MLX modules
try:
    from cosyvoice.mlx_attention import MLXRelPositionMultiHeadedAttention
    from cosyvoice.utils.mlx_layers import MLXPositionwiseFeedForward, MLXConvolutionModule
except ImportError as e:
    # This fallback might be hit if running standalone and cosyvoice isn't structured as a package
    # or if there's an issue with how the environment is set up.
    # For the agent's execution, it assumes these will be found.
    print(f"Initial import attempt failed: {e}. Trying relative imports for standalone execution if applicable.")
    from .mlx_attention import MLXRelPositionMultiHeadedAttention
    from .utils.mlx_layers import MLXPositionwiseFeedForward, MLXConvolutionModule


class MLXConformerEncoderLayer(nn.Module):
    def __init__(
        self,
        size: int, # Dimensionality of input and output features
        mlx_self_attn_module: nn.Module, # Should be an instance of MLXRelPositionMultiHeadedAttention
        mlx_feed_forward_module: Optional[nn.Module] = None,
        mlx_feed_forward_macaron_module: Optional[nn.Module] = None,
        mlx_conv_module: Optional[nn.Module] = None,
        dropout_rate: float = 0.1,
        normalize_before: bool = True,
        layer_norm_eps: float = 1e-12, # Matching PyTorch default if specified like this
    ):
        super().__init__()
        self.size = size
        self.self_attn = mlx_self_attn_module
        self.feed_forward = mlx_feed_forward_module
        self.feed_forward_macaron = mlx_feed_forward_macaron_module
        self.conv_module = mlx_conv_module
        self.normalize_before = normalize_before

        self.norm_ff = nn.LayerNorm(dims=size, eps=layer_norm_eps) if mlx_feed_forward_module else None
        self.norm_mha = nn.LayerNorm(dims=size, eps=layer_norm_eps)
        
        if mlx_feed_forward_macaron_module is not None:
            self.norm_ff_macaron = nn.LayerNorm(dims=size, eps=layer_norm_eps)
            self.ff_scale = 0.5
        else:
            self.norm_ff_macaron = None # To avoid errors if not used
            self.ff_scale = 1.0
        
        if mlx_conv_module is not None:
            self.norm_conv = nn.LayerNorm(dims=size, eps=layer_norm_eps)
            self.norm_final = nn.LayerNorm(dims=size, eps=layer_norm_eps)
        else:
            self.norm_conv = None
            self.norm_final = None
            
        self.dropout = nn.Dropout(p=dropout_rate)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array, # Attention mask
        pos_emb: mx.array, # Positional embedding
        mask_pad: Optional[mx.array] = None, # Padding mask for conv (B, 1, T)
        att_cache: Optional[mx.array] = None, 
        cnn_cache: Optional[mx.array] = None,
    ) -> Tuple[mx.array, mx.array, mx.array, mx.array]:

        # Initialize caches if None (for non-streaming or first chunk)
        # Using example's logic for creating zero-sequence-length caches
        if att_cache is None:
            att_cache = mx.zeros((x.shape[0], self.self_attn.h, 0, self.self_attn.d_k * 2), dtype=x.dtype)

        if cnn_cache is None and self.conv_module is not None and self.conv_module.causal:
            # Only initialize cnn_cache if conv_module is causal and expects one
            cnn_cache = mx.zeros((x.shape[0], self.size, 0), dtype=x.dtype) 
        elif cnn_cache is None: # If no conv_module or not causal, use a dummy empty array
            cnn_cache = mx.array([], dtype=x.dtype) # Ensure it's an mx.array

        # Macaron FeedForward
        if self.feed_forward_macaron is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_ff_macaron(x)
            x = residual + self.ff_scale * self.dropout(self.feed_forward_macaron(x))
            if not self.normalize_before:
                x = self.norm_ff_macaron(x)

        # Multi-Headed Self-Attention
        residual = x
        if self.normalize_before:
            x = self.norm_mha(x)
        
        x_att, new_att_cache = self.self_attn(x, x, x, mask, pos_emb=pos_emb, cache=att_cache)
        x = residual + self.dropout(x_att)
        if not self.normalize_before:
            x = self.norm_mha(x)

        # Convolution Module
        new_cnn_cache_out = cnn_cache # Pass through if no conv_module or if not causal and no cache needed
        if self.conv_module is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_conv(x)
            
            if mask_pad is None: # Default mask_pad if not provided
                 mask_pad = mx.ones((x.shape[0], 1, x.shape[1]), dtype=mx.bool_)

            x_conv, new_cnn_cache_out = self.conv_module(x, mask_pad=mask_pad, cache=cnn_cache)
            x = residual + self.dropout(x_conv)
            if not self.normalize_before:
                x = self.norm_conv(x)
        
        # FeedForward Module
        if self.feed_forward is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_ff(x) # norm_ff could be None if feed_forward is None
            x = residual + self.ff_scale * self.dropout(self.feed_forward(x))
            if not self.normalize_before and self.norm_ff is not None:
                x = self.norm_ff(x)

        # Final LayerNorm if convolution is used
        if self.conv_module is not None and self.norm_final is not None:
            x = self.norm_final(x)
            
        return x, mask, new_att_cache, new_cnn_cache_out

# Example Usage (can be kept for testing or removed for production code):
if __name__ == '__main__':
    print("--- MLXConformerEncoderLayer Smoke Test ---")

    # Parameters for testing
    batch_size = 2
    seq_len = 50
    feat_size = 64 # Must be divisible by num_heads
    num_heads = 4
    ffn_hidden_units = 128
    conv_kernel_size = 7
    dropout = 0.1
    layer_norm_eps_test = 1e-5 # MLX default

    # Dummy sub-modules
    # Positional Encoding for Relative MHA
    # T_rel for pos_emb is 2 * seq_len - 1
    pos_emb_dummy = mx.random.normal((1, 2 * seq_len - 1, feat_size)) 

    # Self-Attention Module (Relative)
    mha_module = MLXRelPositionMultiHeadedAttention(n_head=num_heads, n_feat=feat_size, dropout_rate=dropout)
    
    # FeedForward Module (main)
    ffn_module = MLXPositionwiseFeedForward(idim=feat_size, hidden_units=ffn_hidden_units, dropout_rate=dropout, activation_module=nn.ReLU())
    
    # FeedForward Module (macaron)
    ffn_macaron_module = MLXPositionwiseFeedForward(idim=feat_size, hidden_units=ffn_hidden_units, dropout_rate=dropout, activation_module=nn.ReLU())
    
    # Convolution Module
    conv_module_inst = MLXConvolutionModule(channels=feat_size, kernel_size=conv_kernel_size, activation_module=nn.ReLU(), norm="batch_norm", causal=False)

    # Test cases for normalize_before True/False and with/without optional modules

    def run_test(normalize_before_val, use_macaron, use_conv):
        print(f"\n--- Testing with normalize_before={normalize_before_val}, macaron={use_macaron}, conv={use_conv} ---")
        
        current_ffn_macaron = ffn_macaron_module if use_macaron else None
        current_conv_module = conv_module_inst if use_conv else None

        encoder_layer = MLXConformerEncoderLayer(
            size=feat_size,
            mlx_self_attn_module=mha_module,
            mlx_feed_forward_module=ffn_module,
            mlx_feed_forward_macaron_module=current_ffn_macaron,
            mlx_conv_module=current_conv_module,
            dropout_rate=dropout,
            normalize_before=normalize_before_val,
            layer_norm_eps=layer_norm_eps_test
        )
        mx.eval(encoder_layer.parameters()) # Initialize parameters

        # Dummy inputs
        x_input = mx.random.normal((batch_size, seq_len, feat_size))
        # Attention mask: (B, 1, T_q, T_kv) or (B, 1, 1, T_kv) for self-attention
        # For self-attention, T_q = T_kv = seq_len.
        # A common mask is (B, 1, seq_len) where 0 means masked.
        # MLXMultiHeadedAttention expects mask where 1 means valid.
        att_mask_input = mx.ones((batch_size, 1, 1, seq_len)) 
        if seq_len > 10: # Mask out some future positions for testing
            att_mask_input[:, :, :, seq_len-5:] = 0 
        
        # Convolution padding mask: (B, 1, T) where 1 means valid.
        conv_mask_pad_input = mx.ones((batch_size, 1, seq_len))
        if seq_len > 10:
            conv_mask_pad_input[:, :, seq_len-3:] = 0 # Mask some trailing elements for conv

        # Dummy caches
        # att_cache: (B, num_heads, cache_seq_len, d_k * 2)
        # For this test, let's simulate an empty initial cache by passing None or zero-seq-len cache.
        # The layer's __call__ handles None caches.
        initial_att_cache = None # Or mx.zeros((batch_size, num_heads, 0, (feat_size//num_heads)*2))
        
        # cnn_cache: (B, channels, cache_t_conv)
        initial_cnn_cache = None # Or mx.zeros((batch_size, feat_size, 0)) if conv_module_inst.lorder > 0 else None

        # Run forward pass
        output_x, _, new_att_cache_out, new_cnn_cache_out = encoder_layer(
            x_input, 
            att_mask_input, 
            pos_emb_dummy, 
            mask_pad=conv_mask_pad_input if use_conv else None,
            att_cache=initial_att_cache,
            cnn_cache=initial_cnn_cache
        )
        mx.eval(output_x, new_att_cache_out, new_cnn_cache_out)

        print(f"Input x shape: {x_input.shape}")
        print(f"Output x shape: {output_x.shape}")
        print(f"New attention cache shape: {new_att_cache_out.shape}")
        print(f"New CNN cache shape: {new_cnn_cache_out.shape}")

        assert output_x.shape == (batch_size, seq_len, feat_size)
        # New att cache should have seq_len from input key/value
        assert new_att_cache_out.shape == (batch_size, num_heads, seq_len, (feat_size//num_heads)*2)
        if use_conv and conv_module_inst.causal: # Only if causal conv module is used and has lorder
            assert new_cnn_cache_out.shape == (batch_size, feat_size, conv_module_inst.lorder)
        elif use_conv and not conv_module_inst.causal:
            assert new_cnn_cache_out.size == 0 # No cache for non-causal conv
        else: # No conv module
             assert new_cnn_cache_out.size == 0


    # Run tests for different configurations
    run_test(normalize_before_val=True, use_macaron=True, use_conv=True)
    run_test(normalize_before_val=False, use_macaron=True, use_conv=True)
    run_test(normalize_before_val=True, use_macaron=False, use_conv=True)
    run_test(normalize_before_val=True, use_macaron=True, use_conv=False)
    run_test(normalize_before_val=False, use_macaron=False, use_conv=False)
    
    print("\nMLXConformerEncoderLayer smoke tests passed.")
    print("\n--- End of Tests ---")
