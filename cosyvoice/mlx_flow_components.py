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

# --- Previously defined components ---
class MLXInterpolateRegulator(nn.Module): # Shortened for brevity, assume correct from prev step
    def __init__(self, channels: int, sampling_ratios: List[int], out_channels: Optional[int]=None, groups: int=1):
        super().__init__(); self.model = nn.Identity() # Placeholder
    def __call__(self, x, ylens): return x, ylens # Placeholder
    def inference(self, x1, x2, ml1, ml2, ifr=50): return x1, ml1+ml2 # Placeholder

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
            h = h + time_condition.expand_dims(-1) # Broadcast time_condition to h's shape
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

# --- MLXConditionalDecoder Starts Here ---
class MLXConditionalDecoder(nn.Module):
    def __init__(
        self,
        in_channels: int, 
        out_channels: int,
        channels: Tuple[int, ...] = (256, 256, 256, 256), 
        dropout: float = 0.05,
        attention_head_dim: int = 64,
        n_blocks: int = 1,
        num_mid_blocks: int = 2,
        num_heads: int = 8,
        act_fn: str = "gelu",
    ):
        super().__init__()
        self.packed_input_channels = in_channels # This is the channel dim *after* packing x,mu,spk,cond
        self.out_channels = out_channels
        
        time_emb_input_dim = channels[0] 
        self.time_embeddings = MLXSinusoidalPosEmb(dim=time_emb_input_dim)
        time_embed_dim_mlp_out = time_emb_input_dim * 4
        self.time_mlp = MLXTimestepEmbedding(
            in_channels=time_emb_input_dim,
            time_embed_dim=time_embed_dim_mlp_out,
            act_fn_name="silu"
        )

        self.down_blocks_resnet = nn.ModuleList()
        self.down_blocks_transformer = nn.ModuleList()
        self.down_blocks_downsample = nn.ModuleList()
        
        self.mid_blocks_resnet = nn.ModuleList()
        self.mid_blocks_transformer = nn.ModuleList()
        
        self.up_blocks_resnet = nn.ModuleList()
        self.up_blocks_transformer = nn.ModuleList()
        self.up_blocks_upsample = nn.ModuleList()

        current_channel = self.packed_input_channels
        
        # Down Blocks
        for i, stage_channels_out in enumerate(channels):
            self.down_blocks_resnet.append(MLXResnetBlock1D(dim=current_channel, dim_out=stage_channels_out, time_emb_dim=time_embed_dim_mlp_out))
            self.down_blocks_transformer.append(
                nn.Sequential(*[
                    MLXBasicTransformerBlock(
                        dim=stage_channels_out, num_attention_heads=num_heads, attention_head_dim=attention_head_dim,
                        dropout=dropout, activation_fn=act_fn
                    ) for _ in range(n_blocks)
                ])
            )
            if i < len(channels) - 1:
                self.down_blocks_downsample.append(MLXDownsample1D(stage_channels_out))
            else: # Last down_block before mid, use Conv1d as per PyTorch
                self.down_blocks_downsample.append(nn.Conv1d(stage_channels_out, stage_channels_out, kernel_size=3, padding=1))
            current_channel = stage_channels_out

        # Mid Blocks
        for _ in range(num_mid_blocks):
            self.mid_blocks_resnet.append(MLXResnetBlock1D(dim=current_channel, dim_out=current_channel, time_emb_dim=time_embed_dim_mlp_out))
            self.mid_blocks_transformer.append(
                 nn.Sequential(*[
                    MLXBasicTransformerBlock(
                        dim=current_channel, num_attention_heads=num_heads, attention_head_dim=attention_head_dim,
                        dropout=dropout, activation_fn=act_fn
                    ) for _ in range(n_blocks)
                ])
            )
        
        # Up Blocks
        reversed_stage_channels = list(channels[::-1]) 
        for i, stage_channels_out_rev in enumerate(reversed_stage_channels):
            # Input to up-block resnet is current_channel (from prev upsample or mid) + skip_channel
            # skip_channel is also stage_channels_out_rev (output of corresponding down_block)
            input_channel_stage = current_channel + stage_channels_out_rev
            
            self.up_blocks_resnet.append(MLXResnetBlock1D(dim=input_channel_stage, dim_out=stage_channels_out_rev, time_emb_dim=time_embed_dim_mlp_out))
            self.up_blocks_transformer.append(
                nn.Sequential(*[
                    MLXBasicTransformerBlock(
                        dim=stage_channels_out_rev, num_attention_heads=num_heads, attention_head_dim=attention_head_dim,
                        dropout=dropout, activation_fn=act_fn
                    ) for _ in range(n_blocks)
                ])
            )
            if i < len(channels) - 1: # Not the last upsample stage (which leads to final_block)
                self.up_blocks_upsample.append(MLXUpsample1D(stage_channels_out_rev))
            else: # Last upsample, use Conv1d as per PyTorch
                self.up_blocks_upsample.append(nn.Conv1d(stage_channels_out_rev, stage_channels_out_rev, kernel_size=3, padding=1))
            current_channel = stage_channels_out_rev

        self.final_block = MLXBlock1D(current_channel, current_channel) 
        self.final_proj = nn.Conv1d(current_channel, self.out_channels, kernel_size=1)

    def __call__(self, x_noise, mask, mu, t, spks=None, cond=None, streaming=False):
        # x_noise, mu: (B, C_data_in, T_data)
        # spks: (B, C_spk) -> needs expand_dims(2).broadcast_to(..., T_data)
        # cond: (B, C_cond, T_data)
        # mask: (B, 1, T_data) boolean, True for valid positions
        
        time_emb = self.time_embeddings(t.astype(mx.int32))
        time_emb = self.time_mlp(time_emb)

        packed_inputs = [x_noise, mu]
        if spks is not None:
            spks_expanded = spks.expand_dims(2).broadcast_to((*spks.shape, x_noise.shape[2]))
            packed_inputs.append(spks_expanded)
        if cond is not None:
            packed_inputs.append(cond)
        x = mx.concatenate(packed_inputs, axis=1) # (B, self.packed_input_channels, T_data)

        skip_connections = []
        down_masks = [] # Store masks at each downsample level

        current_data_mask = mask # (B, 1, T_data)
        down_masks.append(current_data_mask)

        # Down Blocks
        for i in range(len(self.down_blocks_resnet)):
            x = self.down_blocks_resnet[i](x, current_data_mask, time_emb)
            x_for_tf = x.transpose(0, 2, 1)
            # For self-attention, mask should be (B, T, T) or (B, 1, T, T)
            # Simplified: pass None, BasicTransformerBlock handles None as no mask.
            x_for_tf = self.down_blocks_transformer[i](x_for_tf, attention_mask=None) 
            x = x_for_tf.transpose(0, 2, 1)
            
            skip_connections.append(x)
            x = self.down_blocks_downsample[i](x * current_data_mask) # Apply mask before downsample
            
            # Downsample the mask for the next stage, unless it's the last "downsample" (which is a Conv1d)
            if i < len(self.down_blocks_resnet) - 1: 
                 current_data_mask = current_data_mask[:, :, ::2] 
            down_masks.append(current_data_mask)


        # Mid Blocks
        for i in range(len(self.mid_blocks_resnet)):
            x = self.mid_blocks_resnet[i](x, current_data_mask, time_emb)
            x_for_tf = x.transpose(0, 2, 1)
            x_for_tf = self.mid_blocks_transformer[i](x_for_tf, attention_mask=None)
            x = x_for_tf.transpose(0, 2, 1)

        # Up Blocks
        for i in range(len(self.up_blocks_resnet)):
            skip = skip_connections.pop()
            # Corresponding mask for skip connection
            current_skip_mask = down_masks[len(self.down_blocks_resnet) - 1 - i] 
            
            # Ensure skip and x have same time dimension before concat
            if x.shape[2] != skip.shape[2]:
                # This usually means upsample output length slightly differs from downsample input length
                # Pad x to match skip, as skip is from earlier, more reliable length.
                # Or, pad/slice based on which is larger/smaller.
                # For PoC, let's assume upsampler gives compatible length or small mismatch.
                # If x is shorter, pad x. If x is longer, slice x.
                if x.shape[2] < skip.shape[2]:
                    pad_len = skip.shape[2] - x.shape[2]
                    x = mx.pad(x, ((0,0),(0,0),(0, pad_len)))
                    current_data_mask = mx.pad(current_data_mask, ((0,0),(0,0),(0, pad_len)), constant_values=False) # Pad mask too
                elif x.shape[2] > skip.shape[2]:
                    x = x[:,:,:skip.shape[2]]
                    current_data_mask = current_data_mask[:,:,:skip.shape[2]]


            x = mx.concatenate((x, skip), axis=1)
            
            x = self.up_blocks_resnet[i](x, current_skip_mask, time_emb) # Use mask from skip connection
            x_for_tf = x.transpose(0, 2, 1)
            x_for_tf = self.up_blocks_transformer[i](x_for_tf, attention_mask=None)
            x = x_for_tf.transpose(0, 2, 1)
            
            x = self.up_blocks_upsample[i](x * current_skip_mask) # Apply mask before upsample
            # Upsample mask for next stage (if not last upsample)
            if i < len(self.up_blocks_resnet) -1:
                 current_data_mask = mx.repeat(current_skip_mask, 2, axis=2)[:,:,:x.shape[2]] # Upsample and trim/pad to match x


        x = self.final_block(x, current_data_mask) 
        output = self.final_proj(x)
        
        return output * mask # Apply original full-resolution input mask to final output

# Example Usage
if __name__ == '__main__':
    # ... (Keep previous tests for InterpolateRegulator, basic Flow Components, MatchaFF, BasicTransformerBlock) ...
    print("\n--- End of BasicTransformerBlock Tests ---") 

    print("\n--- MLXConditionalDecoder Smoke Test ---")
    cd_in_channels_packed = 320 
    cd_out_channels = 80 
    cd_channels_stages = (128, 256) 
    cd_dropout = 0.05
    cd_att_head_dim = 64
    cd_n_blocks = 1 
    cd_num_mid_blocks = 1
    cd_num_heads = 4
    cd_act_fn = "gelu"

    decoder = MLXConditionalDecoder(
        in_channels=cd_in_channels_packed, # Total packed channels
        out_channels=cd_out_channels,
        channels=cd_channels_stages,
        dropout=cd_dropout,
        attention_head_dim=cd_att_head_dim,
        n_blocks=cd_n_blocks,
        num_mid_blocks=cd_num_mid_blocks,
        num_heads=cd_num_heads,
        act_fn=cd_act_fn
    )
    mx.eval(decoder.parameters())
    print("MLXConditionalDecoder instantiated.")

    batch_s = 1
    data_c_x_mu = 80 # Channels for x_noise and mu
    spk_c = 80
    cond_c = 80
    seq_t = 64 

    dummy_x_noise = mx.random.normal((batch_s, data_c_x_mu, seq_t))
    dummy_mask = mlx_make_pad_mask(mx.array([seq_t]*batch_s), seq_t).expand_dims(1) # (B,1,T) bool
    dummy_mu = mx.random.normal((batch_s, data_c_x_mu, seq_t))
    dummy_t_steps = mx.array([10]) 
    dummy_spks = mx.random.normal((batch_s, spk_c))
    dummy_cond = mx.random.normal((batch_s, cond_c, seq_t))

    # Calculate actual in_channels for the decoder based on what's passed
    actual_decoder_in_channels = data_c_x_mu * 2 + spk_c + cond_c
    assert actual_decoder_in_channels == cd_in_channels_packed, "Mismatch in calculated in_channels"

    print(f"Input x_noise shape: {dummy_x_noise.shape}")
    
    try:
        output_decoder = decoder(dummy_x_noise, dummy_mask, dummy_mu, dummy_t_steps, spks=dummy_spks, cond=dummy_cond)
        mx.eval(output_decoder)
        print(f"Decoder output shape: {output_decoder.shape}")
        assert output_decoder.shape == (batch_s, cd_out_channels, seq_t)
        print("MLXConditionalDecoder __call__ smoke test passed.")
    except Exception as e:
        print(f"MLXConditionalDecoder __call__ smoke test failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n--- End of All Flow Components Tests ---")
