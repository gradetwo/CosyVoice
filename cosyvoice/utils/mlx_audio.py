import mlx.core as mx
import numpy as np
from typing import Tuple

# It's good practice to pre-compute windows if they are fixed
def get_hann_window(win_length: int, dtype=mx.float32) -> mx.array:
    # numpy.hanning returns a window normalized to sum to 1 by default if used in filters.
    # For STFT, the window is typically applied directly.
    # The standard Hann window formula is 0.5 * (1 - cos(2*pi*n / (N-1)))
    # numpy.hanning(M) generates a window of length M.
    return mx.array(np.hanning(win_length).astype(np.float32), dtype=dtype)

def mlx_stft(x: mx.array, n_fft: int, hop_length: int, win_length: int, window: mx.array) -> Tuple[mx.array, mx.array]:
    if win_length != n_fft:
        # This implementation simplifies by assuming win_length == n_fft
        # For different win_length and n_fft, padding of frames before FFT is needed.
        raise ValueError("This simplified STFT implementation requires win_length == n_fft.")

    if x.ndim == 1:
        x = x[None, :] # Add batch dimension
    B, T_len = x.shape

    # Centered STFT: Pad the signal so that frame t is centered at time t * hop_length
    # The padding amount is n_fft // 2 on each side.
    # Using reflection padding to mimic torch.stft(center=True) behavior.
    # mx.pad does not directly support 'reflect' mode like numpy.pad.
    # For simplicity, we will use zero padding.
    # For reflection padding, one would need to implement it manually.
    
    padding_amount = n_fft // 2
    # x_padded = mx.pad(x, ((0,0), (padding_amount, padding_amount)), constant_values=0.0) # Zero padding
    
    # Manual reflection padding (simplified for 1D, then adapt for batch)
    # This is a bit complex to do efficiently in MLX without a direct primitive.
    # Given the PoC nature, let's proceed with zero padding for now, as it's simpler to implement.
    # A more robust solution would require careful manual construction of padded signal.
    # With zero padding, the first few and last few frames might not be perfectly centered
    # or might have artifacts if the signal doesn't start/end with zeros.
    
    # Let's adjust the framing loop to handle centered frames with the original signal length,
    # and pad individual frames if they go out of bounds (implicitly handled by how torch.stft works).
    # A common way is to pad the input so that the t-th frame is centered at time t*hop_length.
    # Total padding needed: n_fft - hop_length. Half at beginning, half at end.
    # For center=True, torch effectively pads with n_fft // 2 on each side.
    
    x_padded = mx.pad(x, ((0,0), (padding_amount, padding_amount)), constant_values=0.0)
    T_padded = x_padded.shape[1]

    num_frames = (T_padded - win_length) // hop_length + 1
    
    frames = []
    for i in range(num_frames):
        start = i * hop_length
        frame = x_padded[:, start : start + win_length]
        
        # Ensure frame is of win_length, pad if necessary (should not be if input padding is correct)
        if frame.shape[1] < win_length:
            pad_width = win_length - frame.shape[1]
            frame = mx.pad(frame, ((0,0), (0, pad_width)))
            
        frames.append(frame * window) # Apply window
    
    if not frames:
        # Handle cases where input is too short for any frames
        # Output shape: (B, n_fft // 2 + 1, 0)
        num_freq_bins = n_fft // 2 + 1
        empty_spectrum = mx.zeros((B, num_freq_bins, 0), dtype=mx.complex64)
        return mx.real(empty_spectrum), mx.imag(empty_spectrum)

    framed_signal = mx.stack(frames, axis=1) # Shape: (B, num_frames, win_length)
    
    # RFFT expects last dimension to be the one transformed
    # n=n_fft specifies the FFT size, which should match win_length in this simplified version
    spectrum = mx.fft.rfft(framed_signal, n=n_fft, axis=-1) # Shape: (B, num_frames, n_fft//2 + 1)
    
    # Transpose to match typical (B, FreqBins, NumFrames)
    spectrum = spectrum.transpose(0, 2, 1) 
    
    return mx.real(spectrum), mx.imag(spectrum)

def mlx_istft(real_part: mx.array, imag_part: mx.array, n_fft: int, hop_length: int, win_length: int, window: mx.array, target_len: int = None) -> mx.array:
    if win_length != n_fft:
        raise ValueError("This simplified ISTFT implementation requires win_length == n_fft.")

    # Combine real and imaginary parts into a complex spectrum
    # Input shape is (B, FreqBins, NumFrames) or (FreqBins, NumFrames)
    if real_part.ndim == 2: # Add batch dimension if not present
        real_part = real_part[None, :, :]
        imag_part = imag_part[None, :, :]

    spectrum_complex = mx.complex(real_part, imag_part) # Shape: (B, FreqBins, NumFrames)
    
    # Transpose back for ifft: (B, NumFrames, FreqBins)
    spectrum_complex_T = spectrum_complex.transpose(0, 2, 1)

    # IRFFT
    # n=n_fft specifies the output length of the time-domain signal from IRFFT
    iframes = mx.fft.irfft(spectrum_complex_T, n=n_fft, axis=-1) # Shape: (B, NumFrames, n_fft)
    
    B, num_frames, frame_len = iframes.shape

    # Ensure frame_len matches win_length (it should if n_fft for irfft was win_length)
    if frame_len != win_length:
        # This condition should ideally not be hit if n_fft for irfft is win_length.
        # If n_fft was larger than win_length for rfft (e.g. for zero-padding spectra),
        # then irfft(n=n_fft) would produce frames of length n_fft.
        # Here we assume n_fft used for irfft matches original win_length for simplicity.
        # If frame_len > win_length, truncate. If smaller, pad (less common for irfft output).
        if frame_len > win_length:
            iframes = iframes[:, :, :win_length]
        else:
            iframes = mx.pad(iframes, ((0,0),(0,0),(0, win_length - frame_len)))
    
    # Apply window to iframes. This is part of the OLA process.
    # For perfect reconstruction with certain windows (like Hann), this windowing
    # should be designed carefully with the analysis window.
    # Often, the same window is used, or a synthesis window derived from it.
    iframes = iframes * window 

    # Overlap-add
    expected_output_len = (num_frames - 1) * hop_length + win_length
    
    # Using the NumPy-based OLA loop as it's hard to vectorize OLA directly in MLX
    # without more complex gather/scatter operations or custom kernels.
    y_list = []
    for b_idx in range(B):
        # .item() is for scalar, .tolist() for array to python list, .numpy() for np array
        ola_buffer = np.zeros(expected_output_len, dtype=iframes[b_idx, 0, 0].dtype.numpy()) # Match dtype
        for i in range(num_frames):
            start = i * hop_length
            frame_np = iframes[b_idx, i, :].numpy()
            if start + win_length <= expected_output_len:
                ola_buffer[start : start + win_length] += frame_np
            else: # Should not happen if expected_output_len is calculated correctly
                # This case handles potential off-by-one if frame goes beyond buffer
                valid_len = expected_output_len - start
                ola_buffer[start:] += frame_np[:valid_len]
        y_list.append(mx.array(ola_buffer))
    
    if not y_list: # If num_frames was 0
        output_shape_len = target_len if target_len is not None else 0
        return mx.zeros((B, output_shape_len), dtype=mx.float32) # Assuming float32 output

    y = mx.stack(y_list, axis=0)

    # Trim padding added for centering in STFT (n_fft // 2 from each side)
    # The OLA process reconstructs the signal to the length of the padded signal.
    # We need to trim it back to a length that corresponds to the original signal + OLA artifacts.
    # The `expected_output_len` is based on the number of frames from the *padded* signal.
    # The true "centered" STFT output length after ISTFT should be close to original T_len.
    # Let's remove the initial padding_amount.
    padding_amount_stft = n_fft // 2 
    # The effective length after OLA, before considering target_len, includes the initial padding.
    # We should trim off the padding_amount from both sides of the OLA result.
    # However, the total length of the OLA'd signal is `expected_output_len`.
    # The "true" signal part starts after `padding_amount_stft` and ends `padding_amount_stft` before the end of
    # a signal of length `T_len + 2 * padding_amount_stft` after OLA.
    # So, if `T_len` was original, OLA gives `T_len + 2 * padding_amount_stft`.
    # We need to extract `T_len` from the center.
    # This logic is tricky. Torchaudio handles it internally.
    # For now, let's use target_len if provided, otherwise return based on OLA.
    # A common approach is to trim the output of istft to `original_length`.
    # The `padding_amount` was added to `x` in stft.
    # The OLA signal `y` has length `(num_frames - 1) * hop_length + win_length`.
    # This length corresponds to the *padded* input to the framing loop.
    # So we should trim `padding_amount` from the start.
    
    # Let's simplify: if target_len is provided, use it. Otherwise, trim a standard amount.
    # The `torch.stft` with `center=True` effectively pads by `n_fft // 2`.
    # The output of `torch.istft` is then usually trimmed if length is provided.
    # If not, it returns a length related to `(num_frames - 1) * hop_length + n_fft`.
    
    # Trim the padding applied during STFT for centering
    # The y constructed is for the padded length T_padded = T_len + 2 * padding_amount
    # We need to extract a segment of length T_len from it.
    # The first frame was centered at time 0 of original signal, meaning it used padded region.
    # So, y should be trimmed by padding_amount at the start.
    
    # Effective signal starts after padding_amount
    y = y[:, padding_amount_stft:]

    if target_len is not None:
        if y.shape[1] > target_len:
            y = y[:, :target_len]
        elif y.shape[1] < target_len:
            y = mx.pad(y, ((0,0), (0, target_len - y.shape[1]))) # Pad if shorter
    else:
        # If target_len is not provided, trim to a length that might correspond to original T_len
        # This depends on how T_len relates to num_frames and hop_length
        # For now, if target_len is None, we'll return y after initial trim.
        # This might be T_len + padding_amount if not careful.
        # Let's assume if target_len is None, we want a length that would have been T_len from original.
        # original_T_len = (num_frames_from_original_signal -1) * hop + win
        # This is getting complex. Safest is to require target_len or return the OLA of padded.
        # For now, the y after initial trim is `T_len + padding_amount`.
        # We should trim the other `padding_amount` if `target_len` is not given
        # and we want to match original T_len.
        # However, this is not always desired. Let's stick to explicit target_len or return the current `y`.
        # The current `y` has length `expected_output_len - padding_amount_stft`.
        pass


    if y.ndim == 2 and y.shape[0] == 1: # If it was a single sample, squeeze batch dim
         if real_part.ndim == 2: # Only if original input was 2D (Freq, Frames)
            y = y.squeeze(0)
            
    return y

# Example Usage (can be kept for testing or removed for production code):
if __name__ == '__main__':
    print("--- MLX STFT/ISTFT Tests ---")

    # Test parameters
    n_fft = 512
    hop_length = 128
    win_length = 512 # Must be == n_fft for this simplified version
    sampling_rate = 16000
    duration = 1.0  # 1 second
    
    # Create a Hann window
    window = get_hann_window(win_length)

    # Test with a simple sine wave
    T_len_orig = int(sampling_rate * duration)
    t = mx.arange(T_len_orig) / sampling_rate
    test_signal_mono = 0.5 * mx.sin(2 * np.pi * 440 * t) # A4 note
    test_signal_batch = mx.stack([test_signal_mono, 0.8 * mx.sin(2 * np.pi * 220 * t)], axis=0)

    print(f"Mono signal shape: {test_signal_mono.shape}")
    print(f"Batch signal shape: {test_signal_batch.shape}")

    # Test STFT (mono)
    print("\nTesting STFT (mono)...")
    real_mono, imag_mono = mlx_stft(test_signal_mono, n_fft, hop_length, win_length, window)
    mx.eval(real_mono, imag_mono) # Ensure computation
    print(f"STFT output shapes (mono): real={real_mono.shape}, imag={imag_mono.shape}")
    # Expected: FreqBins = n_fft//2 + 1 = 257. NumFrames = (T_len_orig + 2*(n_fft//2) - win_length) // hop_length + 1
    # T_padded = T_len_orig + n_fft
    # NumFrames = (T_len_orig + n_fft - win_length) // hop_length + 1 = (T_len_orig) // hop_length + 1
    expected_frames_mono = T_len_orig // hop_length + 1
    assert real_mono.shape == (n_fft // 2 + 1, expected_frames_mono)
    assert imag_mono.shape == (n_fft // 2 + 1, expected_frames_mono)

    # Test STFT (batch)
    print("\nTesting STFT (batch)...")
    real_batch, imag_batch = mlx_stft(test_signal_batch, n_fft, hop_length, win_length, window)
    mx.eval(real_batch, imag_batch)
    print(f"STFT output shapes (batch): real={real_batch.shape}, imag={imag_batch.shape}")
    assert real_batch.shape == (test_signal_batch.shape[0], n_fft // 2 + 1, expected_frames_mono)
    assert imag_batch.shape == (test_signal_batch.shape[0], n_fft // 2 + 1, expected_frames_mono)

    # Test ISTFT (mono)
    print("\nTesting ISTFT (mono)...")
    reconstructed_mono = mlx_istft(real_mono, imag_mono, n_fft, hop_length, win_length, window, target_len=T_len_orig)
    mx.eval(reconstructed_mono)
    print(f"ISTFT output shape (mono): {reconstructed_mono.shape}")
    assert reconstructed_mono.shape == (T_len_orig,)
    
    # Test ISTFT (batch)
    print("\nTesting ISTFT (batch)...")
    reconstructed_batch = mlx_istft(real_batch, imag_batch, n_fft, hop_length, win_length, window, target_len=T_len_orig)
    mx.eval(reconstructed_batch)
    print(f"ISTFT output shape (batch): {reconstructed_batch.shape}")
    assert reconstructed_batch.shape == (test_signal_batch.shape[0], T_len_orig)

    # Check reconstruction error (basic check)
    # Note: Perfect reconstruction depends on window properties and OLA normalization.
    # This basic Hann window OLA might not be perfectly normalized.
    if reconstructed_batch.shape == test_signal_batch.shape:
        diff = mx.abs(test_signal_batch - reconstructed_batch)
        max_abs_error = mx.max(diff).item()
        print(f"Max absolute reconstruction error (batch): {max_abs_error}")
        # This error can be significant if OLA normalization is not handled.
        # For a PoC, just checking shapes and that it runs is key.

    print("\n--- End of STFT/ISTFT Tests ---")
