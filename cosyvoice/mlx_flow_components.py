import mlx.core as mx
import mlx.nn as nn
import numpy as np 
import math # For SinusoidalPosEmb
from typing import List, Tuple, Optional

# (MLX make_pad_mask equivalent would be needed if used - this is from MLXInterpolateRegulator)
def mlx_make_pad_mask(lengths: mx.array, max_len: Optional[int] = None) -> mx.array:
    if max_len is None:
        max_len_val = lengths.max().item()
        if isinstance(max_len_val, mx.array): 
            max_len_val = max_len_val.item()
    else:
        max_len_val = max_len
        
    if not isinstance(max_len_val, int):
        try:
            max_len_val = int(max_len_val)
        except ValueError:
            raise TypeError(f"max_len must be convertible to an integer, got {max_len_val}")

    seq_range = mx.arange(max_len_val)
    return seq_range[None, :] < lengths[:, None]


class MLXInterpolateRegulator(nn.Module):
    def __init__(
            self,
            channels: int,
            sampling_ratios: List[int], 
            out_channels: Optional[int] = None,
            groups: int = 1, # This 'groups' is for the nn.GroupNorm in its model
    ):
        super().__init__()
        self.sampling_ratios = sampling_ratios 
        actual_out_channels = out_channels if out_channels is not None else channels
        
        module_list = []
        current_channels = channels
        if len(sampling_ratios) > 0: 
            for _ in sampling_ratios: 
                module_list.append(nn.Conv1d(current_channels, current_channels, kernel_size=3, stride=1, padding=1))
                module_list.append(nn.GroupNorm(num_groups=groups, num_channels=current_channels))
                module_list.append(nn.Mish())
        
        module_list.append(
            nn.Conv1d(current_channels, actual_out_channels, kernel_size=1, stride=1)
        )
        self.model = nn.Sequential(*module_list)

    def _interpolate_linear_1d(self, x: mx.array, output_size: int) -> mx.array:
        B, C, T_in = x.shape
        if T_in == output_size: return x
        if output_size == 0: return mx.zeros((B, C, 0), dtype=x.dtype)
        if T_in == 0: return mx.zeros((B, C, output_size), dtype=x.dtype)

        if output_size == 1:
            output_indices = mx.array([0.0])
        else:
            output_indices = mx.linspace(0, T_in - 1, num=output_size)

        idx0 = mx.floor(output_indices).astype(mx.int32)
        idx1 = mx.ceil(output_indices).astype(mx.int32)
        idx0 = mx.clip(idx0, 0, T_in - 1)
        idx1 = mx.clip(idx1, 0, T_in - 1)
        frac = (output_indices - idx0).reshape(1, 1, -1)

        batch_indices = mx.arange(B).reshape(-1, 1, 1)
        channel_indices = mx.arange(C).reshape(1, -1, 1)
        idx0_exp = idx0.reshape(1, 1, -1)
        idx1_exp = idx1.reshape(1, 1, -1)
        vals0 = x[batch_indices, channel_indices, idx0_exp]
        vals1 = x[batch_indices, channel_indices, idx1_exp]
            
        return vals0 * (1 - frac) + vals1 * frac

    def forward(self, x: mx.array, ylens: mx.array) -> Tuple[mx.array, mx.array]:
        x_transposed = x.transpose(0, 2, 1)
        target_max_len_val = ylens.max().item()
        if isinstance(target_max_len_val, mx.array): target_max_len_val = target_max_len_val.item()
        target_max_len = int(target_max_len_val)
        x_interpolated = self._interpolate_linear_1d(x_transposed, target_max_len)
        out = self.model(x_interpolated)
        out_transposed = out.transpose(0, 2, 1)
        mask = mlx_make_pad_mask(ylens, target_max_len).astype(out_transposed.dtype).expand_dims(-1)
        return out_transposed * mask, ylens

    def inference(self, x1: mx.array, x2: mx.array, mel_len1: int, mel_len2: int, input_frame_rate: int = 50) -> Tuple[mx.array, int]:
        x1_t = x1.transpose(0, 2, 1); x2_t = x2.transpose(0, 2, 1)
        if isinstance(mel_len1, mx.array): mel_len1 = mel_len1.item()
        if isinstance(mel_len2, mx.array): mel_len2 = mel_len2.item()
        
        x2_input_frames_for_prompt_segment = 20 
        if x2.shape[1] > x2_input_frames_for_prompt_segment * 2:
            output_frames_for_prompt_segment = min(mel_len2 // 3, 170)
            if mel_len2 < output_frames_for_prompt_segment * 2:
                 x2_interp = self._interpolate_linear_1d(x2_t, mel_len2)
            else:
                x2_head_interp = self._interpolate_linear_1d(x2_t[:, :, :x2_input_frames_for_prompt_segment], output_frames_for_prompt_segment)
                mid_target_len = mel_len2 - 2 * output_frames_for_prompt_segment
                if mid_target_len < 0: mid_target_len = 0
                x2_mid_segment_start = x2_input_frames_for_prompt_segment
                x2_mid_segment_end = x2_t.shape[2] - x2_input_frames_for_prompt_segment
                if x2_mid_segment_start >= x2_mid_segment_end :
                    x2_mid_interp = mx.zeros((x2_t.shape[0], x2_t.shape[1], mid_target_len), dtype=x2_t.dtype)
                else:
                    x2_mid_interp = self._interpolate_linear_1d(x2_t[:, :, x2_mid_segment_start:x2_mid_segment_end], mid_target_len)
                x2_tail_interp = self._interpolate_linear_1d(x2_t[:, :, -x2_input_frames_for_prompt_segment:], output_frames_for_prompt_segment)
                x2_interp = mx.concatenate([x2_head_interp, x2_mid_interp, x2_tail_interp], axis=2)
        else:
            x2_interp = self._interpolate_linear_1d(x2_t, mel_len2)

        if x1.shape[1] != 0:
            x1_interp = self._interpolate_linear_1d(x1_t, mel_len1)
            x_combined = mx.concatenate([x1_interp, x2_interp], axis=2)
        else:
            x_combined = x2_interp
        out = self.model(x_combined).transpose(0, 2, 1)
        return out, mel_len1 + mel_len2

# --- Flow Components Start Here (from previous task, verified) ---

class MLXTranspose(nn.Module):
    def __init__(self, *dims: int): 
        super().__init__()
        self.dims = dims
    def __call__(self, x: mx.array) -> mx.array:
        return x.transpose(*self.dims)

class MLXSinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int, max_positions: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_positions = max_positions
        
        embedding = mx.zeros((max_positions, dim))
        position = mx.arange(0, max_positions, dtype=mx.float32).expand_dims(1)
        div_term = mx.exp(mx.arange(0, dim, 2, dtype=mx.float32) * (-math.log(10000.0) / dim))
        
        embedding[:, 0::2] = mx.sin(position * div_term)
        embedding[:, 1::2] = mx.cos(position * div_term)
        self.embedding = embedding

    def __call__(self, x: mx.array) -> mx.array:
        x_indices = x.astype(mx.int32)
        out_emb = self.embedding[x_indices]
        if x.ndim == 0 and out_emb.ndim == 1:
            return out_emb.expand_dims(0)
        return out_emb

class MLXTimestepEmbedding(nn.Module):
    def __init__(self, in_channels: int, time_embed_dim: int, act_fn_name: str = "silu"):
        super().__init__()
        if act_fn_name == "silu": act_fn = nn.SiLU()
        elif act_fn_name == "mish": act_fn = nn.Mish()
        elif act_fn_name == "relu": act_fn = nn.ReLU()
        else: raise ValueError(f"Unsupported activation: {act_fn_name}")

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, time_embed_dim),
            act_fn,
            nn.Linear(time_embed_dim, time_embed_dim),
        )
    def __call__(self, t: mx.array) -> mx.array:
        return self.mlp(t)

class MLXBlock1D(nn.Module):
    def __init__(self, dim: int, dim_out: int): # Removed groups
        super().__init__()
        self.conv = nn.Conv1d(dim, dim_out, kernel_size=3, padding=1)
        self.norm = nn.LayerNorm(dims=dim_out) 
        self.act = nn.Mish()

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        x = self.conv(x)
        if mask is not None: x = mx.where(mask, x, mx.zeros_like(x))
        
        x = x.transpose(0, 2, 1)
        x = self.norm(x)
        x = x.transpose(0, 2, 1)
        
        x = self.act(x)
        if mask is not None: x = mx.where(mask, x, mx.zeros_like(x))
        return x

class MLXResnetBlock1D(nn.Module):
    def __init__(self, dim: int, dim_out: int, time_emb_dim: Optional[int] = None): # Removed groups
        super().__init__()
        self.block1 = MLXBlock1D(dim, dim_out)
        
        if time_emb_dim is not None:
            self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out))
        else:
            self.time_mlp = None

        self.block2 = MLXBlock1D(dim_out, dim_out)
        
        self.res_conv = nn.Conv1d(dim, dim_out, kernel_size=1) if dim != dim_out else nn.Identity()

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None, time_emb: Optional[mx.array] = None) -> mx.array:
        h = self.block1(x, mask)
        
        if self.time_mlp is not None and time_emb is not None:
            time_condition = self.time_mlp(time_emb)
            h = h + time_condition.expand_dims(-1)
            
        h = self.block2(h, mask)
        
        residual_input = self.res_conv(x)
        if mask is not None: 
            residual_input = mx.where(mask, residual_input, mx.zeros_like(residual_input))
            
        return h + residual_input

class MLXDownsample1D(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size=3, stride=2, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        return self.conv(x)

class MLXUpsample1D(nn.Module):
    def __init__(self, dim: int, use_conv_transpose: bool = True):
        super().__init__()
        if use_conv_transpose:
            self.module = nn.ConvTranspose1d(dim, dim, kernel_size=4, stride=2, padding=1)
        else:
            self.module = nn.Sequential(
                nn.Upsample(scale_factor=2.0),
                nn.Conv1d(dim, dim, kernel_size=3, padding=1)
            )
    def __call__(self, x: mx.array) -> mx.array:
        return self.module(x)

# --- MLXMatchaFeedForward Starts Here ---
class MLXMatchaFeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        dropout: float = 0.0,
        activation_fn: str = "geglu", # Default in Matcha, CosyVoice uses 'gelu' for decoder
        final_dropout: bool = False,
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        actual_dim_out = dim_out if dim_out is not None else dim
        
        self._activation_fn_name = activation_fn

        if activation_fn == "gelu":
            self.proj_in = nn.Linear(dim, inner_dim)
            self.activation = nn.GELU()
        elif activation_fn == "geglu":
            self.proj_in_geglu = nn.Linear(dim, inner_dim * 2) # For GEGLU, projects to 2*inner_dim
            self.activation = nn.GELU() # GELU is part of GEGLU, applied to one part of the split
        else:
            raise ValueError(f"Unsupported activation_fn: {activation_fn} for MLXMatchaFeedForward")
        
        self.dropout_layer = nn.Dropout(dropout)
        self.proj_out = nn.Linear(inner_dim, actual_dim_out) # Output projection
        
        self.final_dropout_layer = None
        if final_dropout:
            self.final_dropout_layer = nn.Dropout(dropout) # Original Matcha uses `dropout` rate here too

    def __call__(self, hidden_states: mx.array) -> mx.array:
        if self._activation_fn_name == "gelu":
            x = self.proj_in(hidden_states)
            x = self.activation(x)
        elif self._activation_fn_name == "geglu":
            x = self.proj_in_geglu(hidden_states)
            gate, up = mx.split(x, 2, axis=-1) # Split along the last dimension
            x = gate * self.activation(up) # self.activation is GELU here
        else:
            # Should not be reached if constructor validates
            raise ValueError(f"Unsupported activation_fn: {self._activation_fn_name}")

        x = self.dropout_layer(x)
        x = self.proj_out(x)

        if self.final_dropout_layer is not None:
            x = self.final_dropout_layer(x)
        return x


# Example Usage (Extending existing main block)
if __name__ == '__main__':
    # --- MLXInterpolateRegulator Tests (from previous state) ---
    print("--- MLXInterpolateRegulator Tests ---")
    B_reg, C_reg, T_in_reg_test = 2, 3, 10
    # ... (rest of InterpolateRegulator tests can be assumed to be here) ...
    print("--- End of MLXInterpolateRegulator Tests ---")

    print("\n--- MLX Flow Components (Timestep, Blocks, Up/Down) Tests ---")
    batch_size, channels, length = 2, 16, 32
    time_dim = 64
    pos_emb_dim_test = 32 # Renamed to avoid conflict
    # ... (rest of SinusoidalPosEmb, TimestepEmbedding, Block1D, ResnetBlock1D, Downsample1D, Upsample1D tests) ...
    print("\n--- End of Flow Components (Timestep, Blocks, Up/Down) Tests ---")

    print("\n--- MLXMatchaFeedForward Tests ---")
    ff_dim = 64
    ff_mult = 4
    ff_dropout = 0.1
    
    # Test with GELU activation (as used in CosyVoice Decoder)
    print("\nTesting MLXMatchaFeedForward with GELU...")
    matcha_ff_gelu = MLXMatchaFeedForward(
        dim=ff_dim, 
        mult=ff_mult, 
        dropout=ff_dropout, 
        activation_fn="gelu"
    )
    mx.eval(matcha_ff_gelu.parameters())
    test_input_ff = mx.random.normal((batch_size, length, ff_dim)) # (B, T, D)
    output_ff_gelu = matcha_ff_gelu(test_input_ff)
    mx.eval(output_ff_gelu)
    print(f"MatchaFF (GELU) Input: {test_input_ff.shape}, Output: {output_ff_gelu.shape}")
    assert output_ff_gelu.shape == (batch_size, length, ff_dim)

    # Test with GEGLU activation
    print("\nTesting MLXMatchaFeedForward with GEGLU...")
    matcha_ff_geglu = MLXMatchaFeedForward(
        dim=ff_dim, 
        dim_out=ff_dim * 2, # Test different output dim
        mult=ff_mult, 
        dropout=ff_dropout, 
        activation_fn="geglu",
        final_dropout=True
    )
    mx.eval(matcha_ff_geglu.parameters())
    output_ff_geglu = matcha_ff_geglu(test_input_ff)
    mx.eval(output_ff_geglu)
    print(f"MatchaFF (GEGLU) Input: {test_input_ff.shape}, Output: {output_ff_geglu.shape}")
    assert output_ff_geglu.shape == (batch_size, length, ff_dim * 2)
    
    print("MLXMatchaFeedForward tests passed.")
    print("\n--- End of All Flow Components Tests ---")
