import mlx.core as mx
import mlx.nn as nn
from typing import List, Tuple, Optional # Added Optional
import numpy as np # Added numpy

class MLXSnake(nn.Module):
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def __call__(self, x: mx.array) -> mx.array:
        # Ensure alpha is not zero to avoid division by zero or unexpected behavior.
        # The original paper specifies alpha > 0.
        # If alpha is 0, sin^2(alpha*x) / alpha = (alpha*x - (alpha*x)^3/6 + ...)^2 / alpha
        # = (alpha^2*x^2 - ...) / alpha = alpha*x^2 - ...
        # As alpha -> 0, this term goes to 0. So, Snake(x) -> x.
        # This behavior (output x when alpha is 0) is consistent with
        # the limit of (x + alpha*x^2) as alpha -> 0.
        if self.alpha == 0.0:
            # raise ValueError("Alpha cannot be zero in Snake activation if strict adherence to the formula (1/alpha) is required.")
            # Adopting the limit behavior: f(x) = x for alpha = 0
            return x
        
        return x + (1.0 / self.alpha) * mx.power(mx.sin(self.alpha * x), 2)

class MLXResBlock(nn.Module):
    def __init__(self, channels: int = 512, kernel_size: int = 3, dilations: List[int] = [1, 3, 5], snake_alpha: float = 1.0):
        super().__init__()
        
        self.convs1 = []
        self.convs2 = []
        self.activations1 = []
        self.activations2 = []

        for dilation_rate in dilations: # Renamed dilation to dilation_rate to avoid conflict with Conv1d arg
            # Calculate padding for 'same' output length with stride 1
            # padding = (kernel_size * dilation_rate - dilation_rate) // 2 
            # More general formula: padding = (kernel_size - 1) * dilation_rate // 2
            # For nn.Conv1d, padding is applied to both sides, so total added length is 2 * padding.
            # If input length is L_in, output length L_out = L_in + 2*padding - dilation_rate*(kernel_size-1)
            # For L_out = L_in (same padding), we need 2*padding = dilation_rate*(kernel_size-1)
            # So, padding = dilation_rate * (kernel_size - 1) // 2
            
            padding1 = dilation_rate * (kernel_size - 1) // 2
            self.convs1.append(
                nn.Conv1d(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    stride=1,
                    dilation=dilation_rate, # Use the dilation from the list
                    padding=padding1
                )
            )
            
            # The second convolution in the pair always has dilation = 1
            padding2 = (kernel_size - 1) // 2 
            self.convs2.append(
                nn.Conv1d(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    stride=1,
                    dilation=1, # Dilation is 1 for the second conv in each pair
                    padding=padding2
                )
            )
            self.activations1.append(MLXSnake(alpha=snake_alpha))
            self.activations2.append(MLXSnake(alpha=snake_alpha))

    def __call__(self, x: mx.array) -> mx.array:
        for i in range(len(self.convs1)):
            xt = self.activations1[i](x)
            xt = self.convs1[i](xt)
            xt = self.activations2[i](xt)
            xt = self.convs2[i](xt)
            x = xt + x
        return x

class MLXSineGen(nn.Module):
    def __init__(self, samp_rate: int, harmonic_num: int = 0,
                 sine_amp: float = 0.1, noise_std: float = 0.003,
                 voiced_threshold: float = 0.0):
        super().__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold

    def _f02uv(self, f0: mx.array) -> mx.array:
        uv = (f0 > self.voiced_threshold).astype(mx.float32)
        return uv

    def __call__(self, f0: mx.array) -> Tuple[mx.array, mx.array, mx.array]:
        B, _, T_len = f0.shape
        
        harmonics_list = []
        for i in range(self.harmonic_num + 1):
            harmonics_list.append(f0 * (i + 1) / self.sampling_rate)
        F_mat = mx.stack(harmonics_list, axis=1)

        cumsum_F_mat = mx.cumsum(F_mat, axis=2) 
        theta_mat = 2 * np.pi * (cumsum_F_mat % 1.0)
        
        phase_zeroth = mx.zeros((B, 1, 1))
        if self.harmonic_num > 0:
            phase_rest = mx.random.uniform(low=-np.pi, high=np.pi, shape=(B, self.harmonic_num, 1))
            phase_vec = mx.concatenate((phase_zeroth, phase_rest), axis=1)
        else:
            phase_vec = phase_zeroth

        sine_waves = self.sine_amp * mx.sin(theta_mat + phase_vec)
        uv = self._f02uv(f0) 
        noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3.0
        noise = noise_amp * mx.random.normal(sine_waves.shape)
        processed_sine_waves = sine_waves * uv + noise
        
        return processed_sine_waves, uv, noise

class MLXSourceModuleHnNSF(nn.Module):
    def __init__(self, sampling_rate: int, upsample_scale: int, 
                 harmonic_num: int = 0, sine_amp: float = 0.1,
                 add_noise_std: float = 0.003, voiced_threshold: float = 0.0):
        super().__init__()
        self.sine_amp = sine_amp
        self.l_sin_gen = MLXSineGen(sampling_rate, harmonic_num,
                                     sine_amp, add_noise_std, voiced_threshold)
        self.l_linear = nn.Linear(input_dims=harmonic_num + 1, output_dims=1)

    def __call__(self, x: mx.array) -> Tuple[mx.array, mx.array, mx.array]:
        f0_transposed = x.transpose(0, 2, 1)
        sine_wavs_harmonics, uv_details, _ = self.l_sin_gen(f0_transposed)
        sine_wavs_transposed = sine_wavs_harmonics.transpose(0, 2, 1)
        linear_out = self.l_linear(sine_wavs_transposed)
        sine_merge = mx.tanh(linear_out)
        uv_transposed = uv_details.transpose(0, 2, 1)
        noise_source = mx.random.normal(uv_transposed.shape) * self.sine_amp / 3.0
        return sine_merge, noise_source, uv_transposed

class MLXPositionwiseFeedForward(nn.Module):
    def __init__(self,
                 idim: int,
                 hidden_units: int,
                 dropout_rate: float,
                 activation_module: nn.Module): # Pass an instantiated activation module
        super().__init__()
        self.w_1 = nn.Linear(idim, hidden_units)
        self.activation = activation_module
        self.dropout = nn.Dropout(dropout_rate)
        self.w_2 = nn.Linear(hidden_units, idim)

    def __call__(self, xs: mx.array) -> mx.array:
        return self.w_2(self.dropout(self.activation(self.w_1(xs))))

class MLXConvolutionModule(nn.Module):
    def __init__(self,
                 channels: int,
                 kernel_size: int = 15,
                 activation_module: nn.Module = nn.ReLU(), # Pass instance
                 norm: str = "batch_norm",
                 causal: bool = False,
                 bias: bool = True):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.activation = activation_module
        self.norm_type = norm
        self.causal = causal

        self.pointwise_conv1 = nn.Conv1d(
            in_channels=channels,
            out_channels=2 * channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )

        if causal:
            self.padding = 0 # Causal padding handled by F.pad or cache concat
            self.lorder = kernel_size - 1
        else:
            assert (kernel_size - 1) % 2 == 0, "Kernel size must be odd for non-causal symmetric padding"
            self.padding = (kernel_size - 1) // 2
            self.lorder = 0
        
        self.depthwise_conv = nn.Conv1d(
            in_channels=channels, # GLU output has `channels`
            out_channels=channels,
            kernel_size=kernel_size,
            stride=1,
            padding=self.padding, # Use pre-calculated padding for non-causal
            groups=channels,
            bias=bias,
        )

        if norm == "batch_norm":
            self.norm = nn.BatchNorm(num_features=channels)
        elif norm == "layer_norm":
            self.norm = nn.LayerNorm(dims=channels) # MLX LayerNorm uses 'dims'
        else:
            raise ValueError("norm must be 'batch_norm' or 'layer_norm'")

        self.pointwise_conv2 = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )

    def __call__(self, x: mx.array,
                 mask_pad: Optional[mx.array] = None, # Shape (B, 1, T)
                 cache: Optional[mx.array] = None    # Shape (B, C, cache_t)
                 ) -> Tuple[mx.array, mx.array]:
        
        x = x.transpose(0, 2, 1)  # (B, C, T_in)

        if mask_pad is not None and mask_pad.size > 0:
            # Ensure mask_pad is broadcastable (B, 1, T_in)
            x = mx.where(mask_pad, x, mx.zeros_like(x))

        if self.lorder > 0: # Causal convolution
            if cache is None or cache.size == 0:
                # Pad current input x on the left for causal convolution for pointwise_conv1
                x_input_to_pw1 = mx.pad(x, ((0,0), (0,0), (self.lorder, 0)), constant_values=0.0)
            else:
                # Ensure cache has correct batch and channel size
                if cache.shape[0] != x.shape[0] or cache.shape[1] != x.shape[1]:
                     raise ValueError(f"Cache shape {cache.shape} incompatible with input x shape {x.shape}")
                if cache.shape[2] != self.lorder: # Cache must have length `lorder`
                    raise ValueError(f"Cache time dimension {cache.shape[2]} does not match expected lorder {self.lorder}")
                
                x_input_to_pw1 = mx.concatenate((cache, x), axis=2)
            
            # The new cache for the next step is the tail of the current effective input to the convolutions
            # This input (x_input_to_pw1) already includes the cache from previous step or initial padding.
            # Its length is T_cache + T_new_x or T_lorder + T_new_x.
            # The depthwise_conv is causal (padding=0), so its effective input length determines output length.
            # new_cache should be the last `self.lorder` part of what goes into depthwise_conv's effective region.
            # Since pointwise_conv1 doesn't change length, new_cache is tail of x_input_to_pw1.
            if x_input_to_pw1.shape[2] >= self.lorder:
                new_cache = x_input_to_pw1[:, :, -self.lorder:]
            else: 
                # This case implies that (cache + x) or (padded x) is still shorter than lorder.
                # This should only happen if the initial x itself is extremely short.
                new_cache = mx.pad(x_input_to_pw1, ((0,0),(0,0),(self.lorder - x_input_to_pw1.shape[2], 0)))
            
        else: # Non-causal
            x_input_to_pw1 = x
            new_cache = mx.array([]) # No cache for non-causal

        x = self.pointwise_conv1(x_input_to_pw1)  # (B, 2*C, T_effective_pw1_out)
                                                # T_effective_pw1_out = T_in (non-causal) or T_in+lorder (causal initial) or T_cache+T_in (causal with cache)
        
        # GLU activation
        # Split along channel dimension (axis=1) into two equal parts
        x_a, x_b = mx.split(x, 2, axis=1) 
        x = x_a * nn.sigmoid(x_b)    # (B, C, T_effective_pw1_out)

        # Depthwise convolution
        # If causal (self.padding=0), input length T_effective_pw1_out, output length T_effective_pw1_out - K + 1.
        # If non-causal (self.padding=(K-1)//2), output length T_effective_pw1_out.
        # For causal case: T_out_depthwise = (T_in + lorder) - K + 1 = (T_in + K - 1) - K + 1 = T_in.
        # So, after depthwise_conv, length is T_in for causal, or T_effective_pw1_out (which is T_in) for non-causal.
        # Thus, output length of depthwise_conv is always T_in (original time dimension of x).
        x = self.depthwise_conv(x) # (B, C, T_in)
        
        if self.norm_type == "layer_norm":
            x = x.transpose(0, 2, 1) # (B, T_effective, C)
            x = self.norm(x)
            x = self.activation(x)
            x = x.transpose(0, 2, 1) # (B, C, T_effective)
        else: # BatchNorm
            x = self.norm(x) # BatchNorm expects (N,C,L) or (C,L) if affine=False and only C provided to init. MLX BatchNorm expects (N,L,C) or (L,C) if not 2D. For 1D, (N,C,L) is fine.
            x = self.activation(x)
            
        x = self.pointwise_conv2(x) # (B, C, T_effective)

        if mask_pad is not None and mask_pad.size > 0:
             # Ensure mask_pad matches the time dimension of x after convolutions
             # If causal padding was added to x_conv_input, x's time dim is T_in.
             # If non-causal, depthwise_conv with padding='same' (effectively) keeps T_in.
            x = mx.where(mask_pad, x, mx.zeros_like(x))
        
        return x.transpose(0, 2, 1), new_cache # (B, T_out, C), new_cache (B, C, lorder)

# Example Usage (can be kept for testing or removed for production code):
if __name__ == '__main__':
    print("--- MLXSnake Activation Function Tests ---")
    # ... (previous tests for Snake, ResBlock, SineGen, SourceModule, FFN) ...
    # (Assuming the print statements for previous tests are here to keep the full context)
    print("\n--- End of SineGen & SourceModuleHnNSF Tests ---")
    print("\n--- MLXPositionwiseFeedForward Tests ---")
    ffn_idim = 256; ffn_hidden = 1024; ffn_dropout = 0.1; mlx_relu = nn.ReLU()
    ffn_layer = MLXPositionwiseFeedForward(ffn_idim, ffn_hidden, ffn_dropout, mlx_relu)
    mx.eval(ffn_layer.parameters())
    test_input_ffn = mx.random.normal(shape=(2, 10, ffn_idim))
    output_ffn = ffn_layer(test_input_ffn)
    mx.eval(output_ffn)
    assert output_ffn.shape == test_input_ffn.shape
    print("MLXPositionwiseFeedForward test completed.")
    print("\n--- End of PositionwiseFeedForward Tests ---") # Marker for where previous tests ended

    print("\n--- MLXConvolutionModule Tests ---")
    channels_conv = 64
    kernel_size_conv = 7
    batch_size_conv = 2
    seq_len_conv = 50
    
    # Test with BatchNorm (default)
    print("\nTesting with BatchNorm...")
    conv_module_bn = MLXConvolutionModule(channels=channels_conv, kernel_size=kernel_size_conv, activation_module=nn.ReLU(), norm="batch_norm", causal=False)
    mx.eval(conv_module_bn.parameters())
    
    test_x_conv = mx.random.normal((batch_size_conv, seq_len_conv, channels_conv))
    mask_pad_conv = mx.ones((batch_size_conv, 1, seq_len_conv)) # All valid
    
    output_conv_bn, cache_bn = conv_module_bn(test_x_conv, mask_pad_conv, None)
    mx.eval(output_conv_bn, cache_bn)
    print(f"BatchNorm - Input shape: {test_x_conv.shape}, Output shape: {output_conv_bn.shape}, Cache shape: {cache_bn.shape}")
    assert output_conv_bn.shape == (batch_size_conv, seq_len_conv, channels_conv)
    assert cache_bn.size == 0 # Non-causal, empty cache expected based on my impl.

    # Test with LayerNorm
    print("\nTesting with LayerNorm...")
    conv_module_ln = MLXConvolutionModule(channels=channels_conv, kernel_size=kernel_size_conv, activation_module=nn.ReLU(), norm="layer_norm", causal=False)
    mx.eval(conv_module_ln.parameters())
    output_conv_ln, cache_ln = conv_module_ln(test_x_conv, mask_pad_conv, None)
    mx.eval(output_conv_ln, cache_ln)
    print(f"LayerNorm - Input shape: {test_x_conv.shape}, Output shape: {output_conv_ln.shape}, Cache shape: {cache_ln.shape}")
    assert output_conv_ln.shape == (batch_size_conv, seq_len_conv, channels_conv)

    # Test Causal Convolution with BatchNorm
    print("\nTesting Causal Convolution with BatchNorm...")
    conv_module_causal = MLXConvolutionModule(channels=channels_conv, kernel_size=kernel_size_conv, activation_module=nn.ReLU(), norm="batch_norm", causal=True)
    mx.eval(conv_module_causal.parameters())
    
    # Test 1: No initial cache
    output_causal1, cache_out1 = conv_module_causal(test_x_conv, mask_pad_conv, None)
    mx.eval(output_causal1, cache_out1)
    print(f"Causal (no cache) - Output shape: {output_causal1.shape}, New Cache shape: {cache_out1.shape}")
    assert output_causal1.shape == (batch_size_conv, seq_len_conv, channels_conv)
    assert cache_out1.shape == (batch_size_conv, channels_conv, conv_module_causal.lorder)

    # Test 2: With initial cache
    # Create a dummy cache that matches expected shape (B, C, lorder)
    dummy_cache_in = mx.random.normal((batch_size_conv, channels_conv, conv_module_causal.lorder))
    output_causal2, cache_out2 = conv_module_causal(test_x_conv, mask_pad_conv, cache=dummy_cache_in)
    mx.eval(output_causal2, cache_out2)
    print(f"Causal (with cache) - Output shape: {output_causal2.shape}, New Cache shape: {cache_out2.shape}")
    assert output_causal2.shape == (batch_size_conv, seq_len_conv, channels_conv)
    assert cache_out2.shape == (batch_size_conv, channels_conv, conv_module_causal.lorder)
    
    # Test with a shorter input than lorder (for cache creation)
    print("\nTesting Causal Convolution with input shorter than lorder...")
    short_seq_len = conv_module_causal.lorder // 2
    test_x_short_conv = mx.random.normal((batch_size_conv, short_seq_len, channels_conv))
    mask_pad_short_conv = mx.ones((batch_size_conv, 1, short_seq_len))
    output_short_causal, cache_short_out = conv_module_causal(test_x_short_conv, mask_pad_short_conv, None)
    mx.eval(output_short_causal, cache_short_out)
    print(f"Causal (short input) - Output shape: {output_short_causal.shape}, New Cache shape: {cache_short_out.shape}")
    assert output_short_causal.shape == (batch_size_conv, short_seq_len, channels_conv)
    assert cache_short_out.shape == (batch_size_conv, channels_conv, conv_module_causal.lorder) # Cache should still be lorder

    print("MLXConvolutionModule tests passed.")
    print("\n--- End of All Tests in mlx_layers.py ---")
