import mlx.core as mx
import mlx.nn as nn
import numpy as np 
import math # For SinusoidalPosEmb
from typing import List, Tuple, Optional
import sys 
import os

# Add cosyvoice root to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
cosyvoice_root_dir = os.path.abspath(os.path.join(script_dir, os.pardir)) 
if cosyvoice_root_dir not in sys.path:
    sys.path.insert(0, cosyvoice_root_dir)

try:
    from cosyvoice.mlx_attention import MLXDiffusersAttention
except ImportError:
    try:
        from mlx_attention import MLXDiffusersAttention
    except ImportError as e:
        print(f"MLXDiffusersAttention import failed: {e}. Using placeholder.")
        class MLXDiffusersAttention(nn.Module): 
            def __init__(self, query_dim, heads, dim_head, dropout, bias, cross_attention_dim=None): super().__init__() # type: ignore
            def __call__(self, hidden_states, encoder_hidden_states=None, attention_mask=None): return hidden_states

# Helper functions
def mlx_make_pad_mask(lengths: mx.array, max_len: Optional[int] = None) -> mx.array:
    if max_len is None:
        max_len_val = lengths.max().item()
        if isinstance(max_len_val, mx.array): max_len_val = max_len_val.item()
    else:
        max_len_val = max_len
    if not isinstance(max_len_val, int): max_len_val = int(max_len_val)
    seq_range = mx.arange(max_len_val)
    return seq_range[None, :] < lengths[:, None] # (B, max_len), True for valid

def mask_to_bias_mlx(mask: mx.array, dtype: mx.Dtype = mx.float32) -> mx.array:
    # Input mask: (B, ..., T_key), True for valid positions
    # Output bias: (B, ..., T_key), 0.0 for valid, -inf for masked
    return mx.where(mask, mx.array(0.0, dtype=dtype), mx.array(-mx.inf, dtype=dtype))

# --- Previously defined components (shortened for brevity in this diff view) ---
class MLXInterpolateRegulator(nn.Module): 
    def __init__(self, channels: int, sampling_ratios: List[int], out_channels: Optional[int]=None, groups: int=1):
        super().__init__(); # Simplified for diff
        self.model = nn.Identity() 
    def _interpolate_linear_1d(self, x,output_size): return x # Simplified
    def forward(self, x, ylens): return x, ylens # Simplified
    def inference(self, x1, x2, ml1, ml2, ifr=50): return x1, ml1+ml2 # Simplified
    def __call__(self, x, ylens): return self.forward(x, ylens)


class MLXTranspose(nn.Module):
    def __init__(self, *dims: int): super().__init__(); self.dims = dims
    def __call__(self, x: mx.array) -> mx.array: return x.transpose(*self.dims)

class MLXSinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int, max_positions: int = 10000):
        super().__init__(); self.dim = dim; self.max_positions = max_positions
        embedding = mx.zeros((max_positions, dim))
        position = mx.arange(0, max_positions, dtype=mx.float32).expand_dims(1)
        div_term = mx.exp(mx.arange(0, dim, 2, dtype=mx.float32) * (-math.log(10000.0) / dim))
        embedding[:, 0::2] = mx.sin(position * div_term)
        embedding[:, 1::2] = mx.cos(position * div_term)
        self.embedding = embedding
    def __call__(self, x: mx.array) -> mx.array:
        x_indices = x.astype(mx.int32); out_emb = self.embedding[x_indices]
        if x.ndim == 0 and out_emb.ndim == 1: return out_emb.expand_dims(0)
        return out_emb

class MLXTimestepEmbedding(nn.Module):
    def __init__(self, in_channels: int, time_embed_dim: int, act_fn_name: str = "silu"):
        super().__init__()
        if act_fn_name == "silu": act_fn = nn.SiLU()
        elif act_fn_name == "mish": act_fn = nn.Mish()
        elif act_fn_name == "relu": act_fn = nn.ReLU()
        else: raise ValueError(f"Unsupported activation: {act_fn_name}")
        self.mlp = nn.Sequential(nn.Linear(in_channels, time_embed_dim), act_fn, nn.Linear(time_embed_dim, time_embed_dim))
    def __call__(self, t: mx.array) -> mx.array: return self.mlp(t)

class MLXBlock1D(nn.Module):
    def __init__(self, dim: int, dim_out: int):
        super().__init__(); self.conv = nn.Conv1d(dim, dim_out, kernel_size=3, padding=1)
        self.norm = nn.LayerNorm(dims=dim_out); self.act = nn.Mish()
    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        x = self.conv(x); 
        if mask is not None: x = mx.where(mask.broadcast_to(x.shape), x, mx.zeros_like(x))
        x = x.transpose(0, 2, 1); x = self.norm(x); x = x.transpose(0, 2, 1)
        x = self.act(x)
        if mask is not None: x = mx.where(mask.broadcast_to(x.shape), x, mx.zeros_like(x))
        return x

class MLXResnetBlock1D(nn.Module):
    def __init__(self, dim: int, dim_out: int, time_emb_dim: Optional[int] = None):
        super().__init__(); self.block1 = MLXBlock1D(dim, dim_out)
        if time_emb_dim is not None:
            self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out))
        else: self.time_mlp = None
        self.block2 = MLXBlock1D(dim_out, dim_out)
        self.res_conv = nn.Conv1d(dim, dim_out, kernel_size=1) if dim != dim_out else nn.Identity()
    def __call__(self, x: mx.array, mask: Optional[mx.array] = None, time_emb: Optional[mx.array] = None) -> mx.array:
        h = self.block1(x, mask)
        if self.time_mlp is not None and time_emb is not None:
            time_condition = self.time_mlp(time_emb)
            h = h + time_condition.expand_dims(-1) 
        h = self.block2(h, mask)
        residual_input = self.res_conv(x)
        if mask is not None: residual_input = mx.where(mask.broadcast_to(residual_input.shape), residual_input, mx.zeros_like(residual_input))
        return h + residual_input

class MLXDownsample1D(nn.Module):
    def __init__(self, dim: int): super().__init__(); self.conv = nn.Conv1d(dim, dim, kernel_size=3, stride=2, padding=1)
    def __call__(self, x: mx.array) -> mx.array: return self.conv(x)

class MLXUpsample1D(nn.Module):
    def __init__(self, dim: int, use_conv_transpose: bool = True):
        super().__init__()
        if use_conv_transpose: self.module = nn.ConvTranspose1d(dim, dim, kernel_size=4, stride=2, padding=1)
        else: self.module = nn.Sequential(nn.Upsample(scale_factor=2.0), nn.Conv1d(dim, dim, kernel_size=3, padding=1))
    def __call__(self, x: mx.array) -> mx.array: return self.module(x)

class MLXMatchaFeedForward(nn.Module):
    def __init__(self, dim: int, dim_out: Optional[int]=None, mult: int=4, dropout: float=0.0, activation_fn: str="geglu", final_dropout: bool=False):
        super().__init__(); inner_dim = int(dim*mult); actual_dim_out = dim_out if dim_out is not None else dim
        self._activation_fn_name = activation_fn
        if activation_fn == "gelu": self.proj_in = nn.Linear(dim, inner_dim); self.activation = nn.GELU()
        elif activation_fn == "geglu": self.proj_in_geglu = nn.Linear(dim, inner_dim*2); self.activation = nn.GELU()
        else: raise ValueError(f"Unsupported activation_fn: {activation_fn}")
        self.dropout_layer = nn.Dropout(dropout); self.proj_out = nn.Linear(inner_dim, actual_dim_out)
        self.final_dropout_layer = nn.Dropout(dropout) if final_dropout else None
    def __call__(self, hidden_states: mx.array) -> mx.array:
        if self._activation_fn_name == "gelu": x = self.activation(self.proj_in(hidden_states))
        elif self._activation_fn_name == "geglu": x_proj = self.proj_in_geglu(hidden_states); gate, up = mx.split(x_proj, 2, axis=-1); x = gate * self.activation(up)
        else: raise ValueError(f"Unsupported activation_fn: {self._activation_fn_name}")
        x = self.dropout_layer(x); x = self.proj_out(x)
        if self.final_dropout_layer is not None: x = self.final_dropout_layer(x)
        return x

class MLXBasicTransformerBlock(nn.Module):
    def __init__(self, dim: int, num_attention_heads: int, attention_head_dim: int, dropout: float=0.0, cross_attention_dim: Optional[int]=None, activation_fn: str="gelu", attention_bias: bool=False, only_cross_attention: bool=False, norm_elementwise_affine: bool=True):
        super().__init__(); self.only_cross_attention = only_cross_attention
        self.norm1 = nn.LayerNorm(dims=dim, affine=norm_elementwise_affine)
        self.attn1 = MLXDiffusersAttention(query_dim=dim, heads=num_attention_heads, dim_head=attention_head_dim, dropout=dropout, bias=attention_bias, cross_attention_dim=cross_attention_dim if only_cross_attention else None)
        self.norm2 = None; self.attn2 = None
        if cross_attention_dim is not None and not only_cross_attention:
            self.norm2 = nn.LayerNorm(dims=dim, affine=norm_elementwise_affine)
            self.attn2 = MLXDiffusersAttention(query_dim=dim, cross_attention_dim=cross_attention_dim, heads=num_attention_heads, dim_head=attention_head_dim, dropout=dropout, bias=attention_bias)
        self.norm3 = nn.LayerNorm(dims=dim, affine=norm_elementwise_affine)
        self.ff = MLXMatchaFeedForward(dim=dim, dropout=dropout, activation_fn=activation_fn)
    def __call__(self, hidden_states: mx.array, attention_mask: Optional[mx.array]=None, encoder_hidden_states: Optional[mx.array]=None, encoder_attention_mask: Optional[mx.array]=None) -> mx.array:
        norm_h = self.norm1(hidden_states)
        attn_out = self.attn1(norm_h, encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None, attention_mask=encoder_attention_mask if self.only_cross_attention else attention_mask)
        hidden_states = attn_out + hidden_states
        if self.attn2 is not None:
            norm_h = self.norm2(hidden_states)
            attn_out = self.attn2(norm_h, encoder_hidden_states=encoder_hidden_states, attention_mask=encoder_attention_mask)
            hidden_states = attn_out + hidden_states
        norm_h = self.norm3(hidden_states)
        ff_out = self.ff(norm_h)
        hidden_states = ff_out + hidden_states
        return hidden_states

class MLXConditionalDecoder(nn.Module): # Shortened for brevity
    def __init__(self, in_channels: int, out_channels: int, channels: Tuple[int, ...], dropout: float, attention_head_dim: int, n_blocks: int, num_mid_blocks: int, num_heads: int, act_fn: str):
        super().__init__(); # Simplified
        self.packed_input_channels = in_channels; self.out_channels = out_channels
        time_emb_input_dim = channels[0]; self.time_embeddings = MLXSinusoidalPosEmb(dim=time_emb_input_dim)
        time_embed_dim_mlp_out = time_emb_input_dim * 4
        self.time_mlp = MLXTimestepEmbedding(in_channels=time_emb_input_dim,time_embed_dim=time_embed_dim_mlp_out,act_fn_name="silu")
        self.down_blocks_resnet = nn.ModuleList(); self.down_blocks_transformer_sequences = nn.ModuleList(); self.down_blocks_downsampler = nn.ModuleList()
        self.mid_blocks_resnet = nn.ModuleList(); self.mid_blocks_transformer_sequences = nn.ModuleList()
        self.up_blocks_resnet = nn.ModuleList(); self.up_blocks_transformer_sequences = nn.ModuleList(); self.up_blocks_upsampler = nn.ModuleList()
        current_channel = self.packed_input_channels
        for i, stage_channels_out in enumerate(channels): # Down Blocks
            self.down_blocks_resnet.append(MLXResnetBlock1D(dim=current_channel, dim_out=stage_channels_out, time_emb_dim=time_embed_dim_mlp_out))
            self.down_blocks_transformer_sequences.append(nn.Sequential(*[MLXBasicTransformerBlock(dim=stage_channels_out, num_attention_heads=num_heads, attention_head_dim=attention_head_dim,dropout=dropout, activation_fn=act_fn) for _ in range(n_blocks)]))
            self.down_blocks_downsampler.append(MLXDownsample1D(stage_channels_out) if i < len(channels) - 1 else nn.Conv1d(stage_channels_out, stage_channels_out, kernel_size=3, padding=1))
            current_channel = stage_channels_out
        for _ in range(num_mid_blocks): # Mid Blocks
            self.mid_blocks_resnet.append(MLXResnetBlock1D(dim=current_channel, dim_out=current_channel, time_emb_dim=time_embed_dim_mlp_out))
            self.mid_blocks_transformer_sequences.append(nn.Sequential(*[MLXBasicTransformerBlock(dim=current_channel, num_attention_heads=num_heads, attention_head_dim=attention_head_dim,dropout=dropout, activation_fn=act_fn) for _ in range(n_blocks)]))
        reversed_stage_channels = list(channels[::-1])
        for i, skip_connection_channel in enumerate(reversed_stage_channels): # Up Blocks
            input_channel_stage = current_channel + skip_connection_channel; output_channel_stage = skip_connection_channel
            self.up_blocks_resnet.append(MLXResnetBlock1D(dim=input_channel_stage, dim_out=output_channel_stage, time_emb_dim=time_embed_dim_mlp_out))
            self.up_blocks_transformer_sequences.append(nn.Sequential(*[MLXBasicTransformerBlock(dim=output_channel_stage, num_attention_heads=num_heads, attention_head_dim=attention_head_dim,dropout=dropout, activation_fn=act_fn) for _ in range(n_blocks)]))
            self.up_blocks_upsampler.append(MLXUpsample1D(output_channel_stage) if i < len(channels) - 1 else nn.Conv1d(output_channel_stage, output_channel_stage, kernel_size=3, padding=1))
            current_channel = output_channel_stage
        self.final_block = MLXBlock1D(current_channel, current_channel); self.final_proj = nn.Conv1d(current_channel, self.out_channels, kernel_size=1)
    def __call__(self, x_noise, mask, mu, t, spks=None, cond=None, streaming=False): # Simplified body for diff
        return x_noise # Placeholder

# --- Causal Convolutional Components Start Here ---
class MLXCausalConv1d(nn.Conv1d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, dilation: int = 1, groups: int = 1, bias: bool = True):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding=0, 
                         dilation=dilation, groups=groups, bias=bias)
        # Effective padding for causal is (kernel_size - 1) * dilation on the left.
        self.causal_padding_amount = (kernel_size - 1) * dilation

    def __call__(self, x: mx.array, cache: Optional[mx.array] = None) -> Tuple[mx.array, mx.array]:
        # x: (B, C, T_new)
        # cache: (B, C, self.causal_padding_amount)
        if self.causal_padding_amount > 0:
            if cache is not None and cache.size > 0:
                if cache.shape[2] != self.causal_padding_amount:
                    raise ValueError(f"Cache length {cache.shape[2]} != causal_padding_amount {self.causal_padding_amount}")
                if cache.shape[0] != x.shape[0] or cache.shape[1] != x.shape[1]: # Check B, C dims
                     raise ValueError(f"Cache shape {cache.shape} incompatible with input x shape {x.shape}")
                x_padded = mx.concatenate([cache, x], axis=2)
            else: # No valid cache, pad with zeros
                x_padded = mx.pad(x, ((0,0), (0,0), (self.causal_padding_amount, 0)))
            
            new_cache = x_padded[..., -self.causal_padding_amount:]
        else: # No causal padding needed (e.g. kernel_size=1)
            x_padded = x
            new_cache = mx.zeros((x.shape[0], x.shape[1], 0), dtype=x.dtype)

        conv_output = super().__call__(x_padded)
        return conv_output, new_cache

class MLXCausalBlock1D(nn.Module):
    def __init__(self, dim: int, dim_out: int):
        super().__init__()
        self.conv = MLXCausalConv1d(dim, dim_out, kernel_size=3) # Default stride=1, dilation=1
        self.norm = nn.LayerNorm(dims=dim_out)
        self.act = nn.Mish()

    def __call__(self, x: mx.array, mask: Optional[mx.array], 
                 cache: Optional[mx.array] = None) -> Tuple[mx.array, mx.array]:
        # x: (B, C, T), mask: (B, 1, T)
        x_masked = mx.where(mask, x, mx.zeros_like(x)) if mask is not None else x
        
        h, new_conv_cache = self.conv(x_masked, cache)
        
        h_norm = self.norm(h.transpose(0, 2, 1)).transpose(0, 2, 1) 
        h_act = self.act(h_norm)
        
        output = mx.where(mask, h_act, mx.zeros_like(h_act)) if mask is not None else h_act
        return output, new_conv_cache

class MLXCausalResnetBlock1D(nn.Module):
    def __init__(self, dim: int, dim_out: int, time_emb_dim: Optional[int] = None):
        super().__init__()
        self.block1 = MLXCausalBlock1D(dim, dim_out)
        
        if time_emb_dim is not None:
            self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out))
        else:
            self.time_mlp = None

        self.block2 = MLXCausalBlock1D(dim_out, dim_out)
        
        # 1x1 conv for residual, no causal padding needed, no cache
        self.res_conv = nn.Conv1d(dim, dim_out, kernel_size=1) if dim != dim_out else nn.Identity()

    def __call__(self, x: mx.array, mask: Optional[mx.array], time_emb: Optional[mx.array], 
                 cache1: Optional[mx.array] = None, 
                 cache2: Optional[mx.array] = None) -> Tuple[mx.array, mx.array, mx.array]:
        # x: (B, C, T), mask: (B, 1, T), time_emb: (B, time_emb_dim)
        
        h, new_cache1 = self.block1(x, mask, cache1)
        
        if self.time_mlp is not None and time_emb is not None:
            time_condition = self.time_mlp(time_emb) 
            h = h + time_condition.expand_dims(-1) 
            
        h, new_cache2 = self.block2(h, mask, cache2)
        
        x_masked_for_res = mx.where(mask, x, mx.zeros_like(x)) if mask is not None else x
        residual = self.res_conv(x_masked_for_res)

        output = h + residual
        return output, new_cache1, new_cache2


# Example Usage (Extending existing main block)
if __name__ == '__main__':
    # ... (Placeholder for previous tests, then new tests) ...
    print("\n--- End of MLXConditionalDecoder Smoke Test ---") # Marker for where previous tests ended

    print("\n--- MLX Causal Convolutional Components Tests ---")
    batch_size_c, channels_c, length_c = 2, 16, 32
    kernel_size_c = 3
    dilation_c = 1
    causal_pad_amount = (kernel_size_c - 1) * dilation_c

    # Test MLXCausalConv1d
    print("\nTesting MLXCausalConv1d...")
    causal_conv = MLXCausalConv1d(channels_c, channels_c * 2, kernel_size_c, dilation=dilation_c)
    mx.eval(causal_conv.parameters())
    
    test_x_cc = mx.random.normal((batch_size_c, channels_c, length_c))
    
    # Test without cache
    output_cc_nocache, new_cache_cc1 = causal_conv(test_x_cc)
    mx.eval(output_cc_nocache, new_cache_cc1)
    print(f"CausalConv1d (no cache) - Input: {test_x_cc.shape}, Output: {output_cc_nocache.shape}, NewCache: {new_cache_cc1.shape}")
    assert output_cc_nocache.shape == (batch_size_c, channels_c * 2, length_c) # Causal conv maintains length
    assert new_cache_cc1.shape == (batch_size_c, channels_c, causal_pad_amount)

    # Test with cache
    dummy_cache_cc = mx.random.normal((batch_size_c, channels_c, causal_pad_amount))
    output_cc_cache, new_cache_cc2 = causal_conv(test_x_cc, cache=dummy_cache_cc)
    mx.eval(output_cc_cache, new_cache_cc2)
    print(f"CausalConv1d (with cache) - Output: {output_cc_cache.shape}, NewCache: {new_cache_cc2.shape}")
    assert output_cc_cache.shape == (batch_size_c, channels_c * 2, length_c)
    assert new_cache_cc2.shape == (batch_size_c, channels_c, causal_pad_amount)
    print("MLXCausalConv1d test passed.")

    # Test MLXCausalBlock1D
    print("\nTesting MLXCausalBlock1D...")
    causal_block = MLXCausalBlock1D(dim=channels_c, dim_out=channels_c * 2)
    mx.eval(causal_block.parameters())
    test_x_cb = mx.random.normal((batch_size_c, channels_c, length_c))
    mask_cb = mx.ones((batch_size_c, 1, length_c), dtype=mx.bool_)
    if length_c > 3: mask_cb[:, :, -3:] = False

    output_cb, new_cache_cb = causal_block(test_x_cb, mask_cb, cache=None) # Test no cache first
    mx.eval(output_cb, new_cache_cb)
    print(f"CausalBlock1D (no cache) - Input: {test_x_cb.shape}, Output: {output_cb.shape}, NewCache: {new_cache_cb.shape}")
    assert output_cb.shape == (batch_size_c, channels_c * 2, length_c)
    assert new_cache_cb.shape == (batch_size_c, channels_c, causal_pad_amount) # From its CausalConv1d
    if length_c > 3:
        masked_sum_cb = mx.sum(mx.abs(output_cb) * (~mask_cb).astype(output_cb.dtype)).item()
        assert np.isclose(masked_sum_cb, 0.0), f"Masking failed in CausalBlock1D, sum: {masked_sum_cb}"
    print("MLXCausalBlock1D test passed.")

    # Test MLXCausalResnetBlock1D
    print("\nTesting MLXCausalResnetBlock1D...")
    time_emb_dim_crb = 64
    causal_res_block = MLXCausalResnetBlock1D(dim=channels_c, dim_out=channels_c*2, time_emb_dim=time_emb_dim_crb)
    mx.eval(causal_res_block.parameters())
    test_x_crb = mx.random.normal((batch_size_c, channels_c, length_c))
    mask_crb = mx.ones((batch_size_c, 1, length_c), dtype=mx.bool_)
    time_emb_crb = mx.random.normal((batch_size_c, time_emb_dim_crb))
    
    # Test no cache
    output_crb, nc1_crb, nc2_crb = causal_res_block(test_x_crb, mask_crb, time_emb_crb, cache1=None, cache2=None)
    mx.eval(output_crb, nc1_crb, nc2_crb)
    print(f"CausalResnetBlock1D (no cache) - Out: {output_crb.shape}, NC1: {nc1_crb.shape}, NC2: {nc2_crb.shape}")
    assert output_crb.shape == (batch_size_c, channels_c*2, length_c)
    assert nc1_crb.shape == (batch_size_c, channels_c, causal_pad_amount)
    assert nc2_crb.shape == (batch_size_c, channels_c*2, causal_pad_amount)
    print("MLXCausalResnetBlock1D test passed.")

    print("\n--- End of All Flow Components Tests (including Causal) ---")
