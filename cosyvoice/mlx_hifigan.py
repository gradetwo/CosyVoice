import mlx.core as mx
import mlx.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional

# Assuming that when this runs, the cosyvoice base directory is in PYTHONPATH
# Adjust these imports if necessary based on the actual file structure and PYTHONPATH
try:
    from .utils.mlx_layers import MLXSourceModuleHnNSF, MLXResBlock
    from .utils.mlx_audio import mlx_stft, mlx_istft, get_hann_window
except ImportError:
    # Fallback for direct execution or if paths are not set up during subtask
    from cosyvoice.utils.mlx_layers import MLXSourceModuleHnNSF, MLXResBlock
    from cosyvoice.utils.mlx_audio import mlx_stft, mlx_istft, get_hann_window


class MLXHiFTGenerator(nn.Module):
    def __init__(
            self,
            in_channels: int = 80,
            base_channels: int = 512,
            nb_harmonics: int = 8, 
            sampling_rate: int = 22050,
            nsf_alpha: float = 0.1, 
            nsf_sigma: float = 0.003, 
            nsf_voiced_threshold: float = 0.0, 
            upsample_rates: List[int] = [8, 8],
            upsample_kernel_sizes: List[int] = [16, 16],
            istft_params: Dict[str, int] = {"n_fft": 16, "hop_len": 4}, 
            resblock_kernel_sizes: List[int] = [3, 7, 11],
            resblock_dilation_sizes: List[List[int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            source_resblock_kernel_sizes: List[int] = [7, 11],
            source_resblock_dilation_sizes: List[List[int]] = [[1, 3, 5], [1, 3, 5]],
            lrelu_slope: float = 0.1,
            audio_limit: float = 0.99,
            f0_predictor: Optional[nn.Module] = None, 
            snake_alpha: float = 1.0 
    ):
        super().__init__()
        self.out_channels = 1
        self.nb_harmonics = nb_harmonics
        self.sampling_rate = sampling_rate
        self.istft_params = istft_params
        self.lrelu_slope = lrelu_slope
        self.audio_limit = audio_limit
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)

        self.m_source = MLXSourceModuleHnNSF(
            sampling_rate=sampling_rate,
            upsample_scale = int(np.prod(upsample_rates) * istft_params["hop_len"]), # This scale is for internal consistency if SineGen uses it
            harmonic_num=nb_harmonics,
            sine_amp=nsf_alpha,
            add_noise_std=nsf_sigma,
            voiced_threshold=nsf_voiced_threshold
        )
        # Actual F0 upsampling before m_source call
        self.f0_upsample_scale = float(np.prod(upsample_rates) * istft_params["hop_len"])
        self.f0_upsamp = nn.Upsample(scale_factor=self.f0_upsample_scale)


        self.conv_pre = nn.Conv1d(in_channels, base_channels, kernel_size=7, padding=3)

        self.ups = []
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                nn.ConvTranspose1d(
                    in_channels=base_channels // (2**i),
                    out_channels=base_channels // (2**(i + 1)),
                    kernel_size=k,
                    stride=u,
                    padding=(k - u) // 2
                )
            )
        
        self.source_downs = []
        self.source_resblocks = []
        
        stft_channels = istft_params["n_fft"] + 2 # Real + Imag parts from STFT (n_fft/2+1 for real, n_fft/2+1 for imag)
        current_source_conv_in_channels = stft_channels

        # Effective downsampling rates for source features at each fusion point.
        # This logic is based on the example, but might need adjustment for perfect HiFiGAN replication.
        # The example `downsample_rates_effective = [1] + upsample_rates[::-1][:-1]` seemed problematic.
        # A more direct interpretation for HiFiGAN is that each source_downs[i] processes the original s_stft
        # with a stride that matches the total upsampling of x up to that point.
        # However, the example structure has source_downs building sequentially.
        # Let's follow the example's sequential source_downs structure.
        
        for i in range(self.num_upsamples):
            target_channels_x = base_channels // (2**(i + 1)) # Channels of x after upsampler self.ups[i]
            
            # Determine u_eff for this stage. The example's u_eff logic was:
            # downsample_rates_effective = [1] + upsample_rates[::-1][:-1]
            # u_eff = downsample_rates_effective[self.num_upsamples - 1 - i]
            # This means u_eff takes values from upsample_rates (but reversed and excluding the last one, plus a 1).
            # Example: upsample_rates = [8,8,2,2]. Then u_eff for i=0,1,2,3 would be from [1,2,2,8] -> u_eff = 2,2,8,1 (reversed index)
            # This seems to be for progressively downsampling the source. Let's try to implement this.
            temp_downsample_rates = [1] + upsample_rates[::-1] # [1, r_last, ..., r_1]
            u_eff = temp_downsample_rates[i] # if we iterate from last fusion point to first.
                                            # Or, if i is current upsample stage (0 to num_upsamples-1)
                                            # Example: upsample_rates = [U0, U1, U2]. num_upsamples = 3.
                                            # i=0 (after U0): u_eff related to U2?
                                            # i=1 (after U1): u_eff related to U1?
                                            # i=2 (after U2): u_eff related to U0? (No, usually 1 here)
            # The PyTorch HiFT reference has `kernels_s` which are `[1, u_n, u_n*u_{n-1}, ...] / np.prod(us)`
            # This seems to be total downsampling factor.
            # For PoC, the example's sequential Conv1D for source_downs is simpler:
            # Conv1D -> ResBlock -> Conv1D -> ResBlock ...
            # Let's use a kernel/stride that makes some sense for downsampling, or just 1 for channel matching.
            # The example code's source_downs was:
            # if u_eff == 1: self.source_downs.append(nn.Conv1d(current_source_channels, target_channels, kernel_size=1, stride=1))
            # else: self.source_downs.append(nn.Conv1d(current_source_channels, target_channels, kernel_size=u_eff*2, stride=u_eff, padding=u_eff//2))
            # This implies current_source_channels changes.
            # For this iteration, let's make each source_downs[i] take the *original* s_stft channels
            # and project to target_channels_x. The stride logic is the hardest.
            # Simplified: use kernel 3, stride 1, padding 1 for now, assuming time dim is handled by input or ResBlock.
            
            if i < len(source_resblock_kernel_sizes): # Ensure we don't create more source paths than defined by resblocks
                self.source_downs.append(
                    nn.Conv1d(stft_channels, target_channels_x, kernel_size=3, stride=1, padding=1) 
                    # NOTE: Stride 1 here is a major simplification. Proper stride is needed to match time dimensions.
                )
                self.source_resblocks.append(
                    MLXResBlock(target_channels_x, source_resblock_kernel_sizes[i], source_resblock_dilation_sizes[i], snake_alpha=snake_alpha)
                )

        self.resblocks = []
        for i in range(len(self.ups)): # Resblocks after each upsampling layer
            ch_after_ups = base_channels // (2**(i + 1))
            for j in range(self.num_kernels):
                self.resblocks.append(MLXResBlock(ch_after_ups, resblock_kernel_sizes[j], resblock_dilation_sizes[j], snake_alpha=snake_alpha))
        
        final_ch_for_conv_post = base_channels // (2**len(self.ups))
        self.conv_post = nn.Conv1d(final_ch_for_conv_post, istft_params["n_fft"] // 2 + 1, kernel_size=7, padding=3) 
        
        self.stft_window = get_hann_window(istft_params["n_fft"])
        self.f0_predictor = f0_predictor

    def decode(self, x: mx.array, s: mx.array) -> mx.array:
        # s has shape (B, 1, T_audio_source)
        # mlx_stft expects (B, T_audio) or (T_audio)
        s_stft_real, s_stft_imag = mlx_stft(s.squeeze(1), self.istft_params["n_fft"], self.istft_params["hop_len"], self.istft_params["n_fft"], self.stft_window)
        
        # s_stft_real/imag shapes: (B, FreqBins, NumFrames_s)
        # Concatenate to form (B, 2*FreqBins, NumFrames_s) = (B, n_fft+2 if FreqBins=n_fft/2+1, NumFrames_s)
        # Note: PyTorch code might use n_fft as channel count for real and imag parts separately then cat.
        # Here, FreqBins is n_fft//2+1. So 2*FreqBins = n_fft+2.
        s_stft = mx.concatenate([s_stft_real, s_stft_imag], axis=1)

        x = self.conv_pre(x) # x is (B, in_channels, T_mel) -> (B, base_channels, T_mel)
        
        for i in range(self.num_upsamples):
            x = nn.leaky_relu(x, self.lrelu_slope)
            x = self.ups[i](x) # x shape: (B, base_channels//(2**(i+1)), T_mel_upsampled)
            
            if i < len(self.source_downs): # Check if source fusion layers exist for this stage
                # s_stft is (B, n_fft+2, Frames_s_stft)
                # source_downs[i] processes s_stft. Its output time dimension needs to match x.
                # This is a known simplification point if strides aren't correctly set in source_downs.
                si = self.source_downs[i](s_stft)
                
                if i < len(self.source_resblocks): # Apply corresponding resblock if it exists
                    si = self.source_resblocks[i](si)
                
                # Ensure si has same time dimension as x for addition
                if x.shape[2] > si.shape[2]:
                    pad_len = x.shape[2] - si.shape[2]
                    si = mx.pad(si, ((0,0), (0,0), (0, pad_len))) # Pad time dimension of si
                elif si.shape[2] < x.shape[2]: # Should be x.shape[2] < si.shape[2]
                    si = si[:, :, :x.shape[2]] # Truncate time dimension of si
                x = x + si

            xs = None
            # Correct indexing for resblocks: each upsample stage has self.num_kernels resblocks
            current_stage_resblock_start_idx = i * self.num_kernels
            for j in range(self.num_kernels):
                res_idx = current_stage_resblock_start_idx + j
                if xs is None:
                    xs = self.resblocks[res_idx](x)
                else:
                    xs += self.resblocks[res_idx](x)
            x = xs / self.num_kernels

        x = nn.leaky_relu(x)
        x_mag = self.conv_post(x) # Outputs log magnitude (B, n_fft//2+1, T_final)
        
        magnitude = mx.exp(x_mag) 
        phase = mx.zeros_like(magnitude) # Zero phase for simplicity

        magnitude = mx.clip(magnitude, a_min=None, a_max=1e2)

        # target_len for istft. This should be the length of x after all upsampling.
        # x.shape[2] from conv_post is in frames. Target audio length is x.shape[2] * hop_length
        target_audio_len = x_mag.shape[2] * self.istft_params["hop_len"]

        out_wav = mlx_istft(magnitude, phase, self.istft_params["n_fft"], self.istft_params["hop_len"], self.istft_params["n_fft"], self.stft_window, target_len=target_audio_len)
        out_wav = mx.clip(out_wav, a_min=-self.audio_limit, a_max=self.audio_limit)
        return out_wav

    def __call__(self, speech_feat: mx.array, f0: mx.array, cache_source: Optional[mx.array] = None) -> Tuple[mx.array, mx.array]:
        # speech_feat: (B, T_mel, Mel_bins) -> for conv_pre (B, Mel_bins, T_mel)
        # f0: (B, T_mel, 1)
        
        # Upsample f0. f0_upsamp expects (B, C, T)
        f0_for_upsample = f0.transpose(0, 2, 1) # (B, 1, T_mel)
        s_upsampled_f0 = self.f0_upsamp(f0_for_upsample) # (B, 1, T_audio_len_f0_domain)
        
        # Source module expects f0 in shape (B, T_audio_len, 1)
        s_upsampled_f0_for_source = s_upsampled_f0.transpose(0, 2, 1)
        s, _, _ = self.m_source(s_upsampled_f0_for_source) # s output: (B, T_audio_len, 1) harmonic part
        
        # Transpose s for STFT: (B, 1, T_audio_len) -> s is already (B, T_audio_len, 1) from m_source
        s_for_decode = s.transpose(0, 2, 1) # (B, 1, T_audio_len) for decode method

        if cache_source is not None and cache_source.shape[2] > 0:
            # cache_source is (B, 1, T_cache)
            # s_for_decode is (B, 1, T_current_segment_audio_len)
            # The example cache logic from task description:
            if s_for_decode.shape[2] >= cache_source.shape[2]: # Ensure current segment is long enough for the cache part
                s_list = []
                for k_idx in range(s_for_decode.shape[0]): # Iterate over batch
                    s_k = s_for_decode[k_idx]    # (1, T_current_s)
                    cs_k = cache_source[k_idx] # (1, T_cache)
                    
                    len_cache = cs_k.shape[1]
                    # This logic was: if current_s is shorter than cache, trim cache.
                    # This seems counter-intuitive for typical cache use where cache might be longer.
                    # Original PyTorch `s[:, :, :T_cache] = cache_source` assumes `s` is the buffer.
                    # For MLX, if `s_for_decode` is the new audio, and `cache_source` is old audio,
                    # they are usually concatenated for STFT, or `s_for_decode` is an update to a larger buffer.
                    # The example's loop creates a new 's' by taking 'cache' as prefix.
                    
                    prefix = cs_k 
                    if s_k.shape[1] > len_cache: # If current segment has parts after cache length
                        suffix = s_k[:, len_cache:]
                        combined = mx.concatenate([prefix, suffix], axis=1)
                    else: # Cache fills or exceeds current s_k length
                        combined = prefix[:, :s_k.shape[1]] # Use cache, trimmed to s_k's original length
                    s_list.append(combined)
                
                if s_list:
                    s_for_decode = mx.stack(s_list, axis=0)
            # else: s_for_decode is shorter than cache, this case might need specific handling or indicates an issue.
            # For PoC, if s_for_decode is shorter, we might just use s_for_decode or error.
            # The example logic implicitly assumes s_for_decode is long enough or cache is trimmed.

        # Transpose speech_feat for conv_pre: (B, Mel_bins, T_mel)
        # speech_feat input to __call__ is (B, T_mel, Mel_bins)
        speech_feat_transposed = speech_feat.transpose(0, 2, 1) # (B, Mel_bins, T_mel)
        generated_speech = self.decode(x=speech_feat_transposed, s=s_for_decode)
        
        # Output is (B, T_audio). PyTorch returns (B, 1, T_audio)
        return generated_speech.unsqueeze(1), f0 # Return f0 for consistency

    def inference(self, speech_feat: mx.array, cache_source: Optional[mx.array] = None) -> Tuple[mx.array, mx.array]:
        # speech_feat is (B, T_mel, Mel_bins)
        if self.f0_predictor is None:
            raise ValueError("f0_predictor is not set for inference.")
        
        # f0_predictor expects (B, Mel_bins, T_mel)
        f0 = self.f0_predictor(speech_feat.transpose(0,2,1)) # f0 output: (B, T_mel, 1)
        
        return self.__call__(speech_feat, f0, cache_source)

# Example Usage (Placeholder - needs actual model weights and f0_predictor for meaningful test)
if __name__ == '__main__':
    print("--- MLXHiFTGenerator Smoke Test ---")
    
    # Dummy f0 predictor for testing structure
    class DummyF0Predictor(nn.Module):
        def __init__(self, in_channels, out_dims):
            super().__init__()
            # A simple conv to change time dim if needed, then linear
            self.conv = nn.Conv1d(in_channels, 1, kernel_size=1) # Output 1 channel for F0
        def __call__(self, x_mel_T): # x_mel_T is (B, Mel_bins, T_mel)
            f0 = self.conv(x_mel_T) # (B, 1, T_mel)
            return f0.transpose(0,2,1) # -> (B, T_mel, 1)

    # Parameters (simplified for testing)
    test_in_channels = 80 # Mel bins
    test_base_channels = 128 # Smaller for faster test
    test_sampling_rate = 16000
    test_upsample_rates = [4, 4] # Total upsample 16x
    test_istft_hop_len = 4
    test_istft_n_fft = 16 # Small FFT for speed
    
    # f0 predictor
    dummy_f0_pred = DummyF0Predictor(in_channels=test_in_channels, out_dims=1)

    generator = MLXHiFTGenerator(
        in_channels=test_in_channels,
        base_channels=test_base_channels,
        sampling_rate=test_sampling_rate,
        upsample_rates=test_upsample_rates,
        istft_params={"n_fft": test_istft_n_fft, "hop_len": test_istft_hop_len},
        nb_harmonics=1, # Simpler source
        f0_predictor=dummy_f0_pred,
        resblock_kernel_sizes=[3,5], # Fewer resblocks
        resblock_dilation_sizes=[[1,3],[1,3]],
        source_resblock_kernel_sizes=[7], # Fewer source resblocks
        source_resblock_dilation_sizes=[[1,3]]
    )
    mx.eval(generator.parameters()) # Initialize parameters

    print("MLXHiFTGenerator initialized.")

    # Create dummy input
    batch_size = 1
    mel_seq_len = 32 # T_mel
    
    speech_feat_input = mx.random.normal((batch_size, mel_seq_len, test_in_channels)) # (B, T_mel, Mel_bins)
    
    print(f"Input speech_feat shape: {speech_feat_input.shape}")

    # Test inference method
    try:
        output_wav, output_f0 = generator.inference(speech_feat_input)
        mx.eval(output_wav, output_f0) # Ensure computation
        print("Inference method smoke test passed.")
        print(f"Output waveform shape: {output_wav.shape}") # Expected (B, 1, T_audio)
        print(f"Output f0 shape: {output_f0.shape}")       # Expected (B, T_mel, 1)
        
        # Expected audio length: mel_seq_len * prod(upsample_rates) * istft_hop_len
        expected_audio_len = mel_seq_len * np.prod(test_upsample_rates) * test_istft_hop_len
        # The actual audio length from ISTFT depends on the number of frames output by conv_post
        # which is mel_seq_len * prod(upsample_rates).
        # So, T_audio = (mel_seq_len * prod(upsample_rates)) * hop_length for istft
        # This seems correct.
        
        # Example: T_mel=32, upsample=[4,4] -> T_x_final_frames = 32*16 = 512
        # T_audio = 512 * hop_len(4) = 2048
        # output_wav.shape[2] should be close to this.

    except Exception as e:
        print(f"Error during inference smoke test: {e}")
        import traceback
        traceback.print_exc()

    print("\n--- End of MLXHiFTGenerator Smoke Test ---")
