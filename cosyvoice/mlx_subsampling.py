import mlx.core as mx
import mlx.nn as nn
from typing import Tuple, Union, Type, Optional # Added Optional
import sys
import os

# Add cosyvoice root to sys.path to allow finding the cosyvoice package
# This assumes the script is run from the 'tools/' directory or similar context
# where 'cosyvoice' is a sibling or in PYTHONPATH.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Attempt to import MLX positional encoding classes
try:
    from .mlx_embedding import MLXPositionalEncoding, MLXEspnetRelPositionalEncoding
except ImportError:
    # Fallback for direct execution or if paths are not set up during subtask
    # This helps in testing the file standalone if cosyvoice is in the Python path
    from cosyvoice.mlx_embedding import MLXPositionalEncoding, MLXEspnetRelPositionalEncoding


class MLXBaseSubsampling(nn.Module):
    def __init__(self):
        super().__init__()
        self.right_context = 0
        self.subsampling_rate = 1
        self.pos_enc: Optional[nn.Module] = None # To be set by subclasses

    def position_encoding(self, offset: Union[int, mx.array], size: int) -> mx.array:
        if self.pos_enc is None:
            raise ValueError("Positional encoding module not set in MLXBaseSubsampling.")
        
        # Assuming pos_enc module has a 'position_encoding' method or is callable in a way
        # that returns just the positional embedding based on offset and size.
        # The typical __call__ of our PE modules returns (x_with_pe, pe_itself).
        # We need a method that directly gives the PE slice.
        if hasattr(self.pos_enc, 'position_encoding'):
            # Assuming apply_dropout=False as typically PE is added before final dropout of the layer
            return self.pos_enc.position_encoding(offset, size, apply_dropout=False) 
        else:
            # Fallback if position_encoding method is not directly available,
            # try to use __call__ with dummy zeros just to get the pos_emb.
            # This is less ideal. The PE modules should ideally have a direct way to get PE.
            # For MLXPositionalEncoding and MLXEspnetRelPositionalEncoding, they do have a position_encoding method.
            dummy_x_for_pe = mx.zeros((1, size, self.pos_enc.d_model if hasattr(self.pos_enc, 'd_model') else 0), dtype=mx.float32)
            _, pos_emb = self.pos_enc(dummy_x_for_pe, offset) # Get (x, pos_emb)
            return pos_emb # Return only the PE part


class MLXLinearNoSubsampling(MLXBaseSubsampling):
    def __init__(self, idim: int, odim: int, dropout_rate: float,
                 mlx_pos_enc_module: nn.Module):
        super().__init__()
        self.out = nn.Sequential(
            nn.Linear(idim, odim),
            nn.LayerNorm(odim),
            nn.Dropout(dropout_rate),
        )
        self.pos_enc = mlx_pos_enc_module
        self.right_context = 0
        self.subsampling_rate = 1

    def __call__(self, x: mx.array, x_mask: mx.array, offset: Union[int, mx.array] = 0
    ) -> Tuple[mx.array, mx.array, mx.array]:
        x = self.out(x)
        x_with_pe, pos_emb = self.pos_enc(x, offset=offset) # pos_enc __call__ expects offset
        return x_with_pe, pos_emb, x_mask


class MLXConv2dSubsampling4(MLXBaseSubsampling):
    def __init__(self, idim: int, odim: int, dropout_rate: float, # dropout_rate for pos_enc
                 mlx_pos_enc_module: nn.Module):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=odim, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=odim, out_channels=odim, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
        )
        
        # Calculate output dim of convs for the linear layer
        # Based on PyTorch Conv2d: output_size = (input_size - kernel_size + 2*padding) / stride + 1
        # Conv1: F_out1 = (idim - 3 + 2*0) // 2 + 1 = (idim - 3)//2 + 1
        # Conv2: F_out2 = (F_out1 - 3 + 2*0) // 2 + 1 = (F_out1 - 3)//2 + 1
        # This is the formula (((idim - 1) // 2 - 1) // 2) if idim means "number of points" and K=3, P=0, S=2
        # For example, if idim = 80 (e.g. 80 mel bins)
        # F_out1 = (80 - 3)//2 + 1 = 77//2 + 1 = 38 + 1 = 39
        # F_out2 = (39 - 3)//2 + 1 = 36//2 + 1 = 18 + 1 = 19
        # So, feature_dim_after_conv = odim * 19
        # The formula `odim * (((idim - 1) // 2 - 1) // 2)` is from ESPnet, let's use it.
        # If idim = 80: odim * (((79)//2 - 1)//2) = odim * ((39-1)//2) = odim * (38//2) = odim * 19. Matches.
        feature_dim_after_conv = odim * (((idim - 1) // 2 - 1) // 2)

        self.linear_out = nn.Linear(feature_dim_after_conv, odim)
        self.pos_enc = mlx_pos_enc_module # This will use its own dropout_rate
        self.subsampling_rate = 4
        # Context from PyTorch version: (K-1)*dilation for each conv, summed for total context.
        # Here, K=3, D=1, S=2.
        # Conv1 adds (3-1)*1 = 2 context. Effective field scaled by stride 2.
        # Conv2 adds (3-1)*1 = 2 context. Effective field scaled by stride 2*2 = 4.
        # Right context calculation in ESPnet for this stack:
        # ((kernel_size - 1) // 2) * num_layers_implicit_dilation_1 + (kernel_size-1) for last layer?
        # Or simply: total_stride * ( (K1-1)/2 + (K2-1)/2 / S1 + ... ) -> this is complex.
        # The ESPnet value of 6 for (K=3,S=2)*2 layers is standard.
        self.right_context = 6 

    def __call__(self, x: mx.array, x_mask: mx.array, offset: Union[int, mx.array] = 0
    ) -> Tuple[mx.array, mx.array, mx.array]:
        # x: (B, T_in, idim)
        x = x.expand_dims(1)  # (B, 1, T_in, idim) -> (B, C_in, H, W) where C_in=1, H=T_in, W=idim
        
        x = self.conv(x) # Output: (B, C_out=odim, T_out, F_out)
        
        B, C_conv_out, T_conv_out, F_conv_out = x.shape
        
        # Reshape for linear layer: (B, T_out, C_out * F_out)
        # PyTorch: x.transpose(1, 2).contiguous().view(b, t_out, c_out * f_out)
        # MLX: transpose(0, 2, 1, 3) means (B, T_out, C_out, F_out)
        # Then reshape to (B, T_out, C_out * F_out)
        x = x.transpose(0, 2, 1, 3).reshape(B, T_conv_out, C_conv_out * F_conv_out)
        
        x = self.linear_out(x) # Output: (B, T_out, odim)
        x_with_pe, pos_emb = self.pos_enc(x, offset=offset)
        
        # Subsample the mask: x_mask is (B, 1, T_in)
        # PyTorch code: subsampled_mask = x_mask[:, :, :-2:2][:, :, :-2:2]
        # This is equivalent to x_mask[:, :, 2::2][:, :, 2::2] if length allows.
        # Or more simply x_mask[:, :, ::4] if total stride is 4.
        # The PyTorch code `x_mask[:, :, i_1::s_1][:, :, i_2::s_2]` is often simplified to `x_mask[:,:,::total_stride]`
        # For two layers of stride 2, total stride is 4.
        if x_mask is not None and x_mask.shape[2] > 0 :
             subsampled_mask = x_mask[:, :, ::self.subsampling_rate]
             # To match the output length of convs more precisely:
             # T_out_conv1 = (T_in - K)/S + 1 = (T_in - 3)//2 + 1
             # T_out_conv2 = (T_out_conv1 - K)/S + 1 = (T_out_conv1 - 3)//2 + 1
             # If T_out_final = x_with_pe.shape[1], then mask should be (B, 1, T_out_final)
             # However, the simple ::self.subsampling_rate is common.
             # Let's ensure the mask length matches the data length T_conv_out (which is x_with_pe.shape[1])
             if subsampled_mask.shape[2] > T_conv_out :
                 subsampled_mask = subsampled_mask[:,:,:T_conv_out]
             elif subsampled_mask.shape[2] < T_conv_out: # Should not happen if T_in is large enough
                 # This indicates an issue or very short input. Pad mask.
                 pad_len_mask = T_conv_out - subsampled_mask.shape[2]
                 subsampled_mask = mx.pad(subsampled_mask, ((0,0),(0,0),(0,pad_len_mask)), constant_values=0)

        else: # If input mask is None or empty
             subsampled_mask = x_mask

        return x_with_pe, pos_emb, subsampled_mask

# Example Usage (can be kept for testing or removed for production code):
if __name__ == '__main__':
    print("--- MLX Subsampling Modules Tests ---")
    
    # Common test parameters
    d_model = 8 # odim for subsampling layers
    idim_test = 80 # input feature dimension (e.g., mel bins)
    dropout_rate_test = 0.1
    batch_size_test = 2
    seq_len_test = 100 # T_in
    
    # Initialize a dummy MLXPositionalEncoding module for testing
    # max_len for PE module should be sufficient for subsampled sequence length
    pe_module = MLXPositionalEncoding(d_model=d_model, dropout_rate=dropout_rate_test, max_len=seq_len_test) 
    mx.eval(pe_module.parameters())

    # --- Test MLXLinearNoSubsampling ---
    print("\nTesting MLXLinearNoSubsampling...")
    linear_nosub = MLXLinearNoSubsampling(idim=idim_test, odim=d_model, dropout_rate=dropout_rate_test, mlx_pos_enc_module=pe_module)
    mx.eval(linear_nosub.parameters())

    test_x_linear = mx.random.normal((batch_size_test, seq_len_test, idim_test))
    test_x_mask_linear = mx.ones((batch_size_test, 1, seq_len_test)) # Dummy mask (all valid)

    out_linear, pos_emb_linear, out_mask_linear = linear_nosub(test_x_linear, test_x_mask_linear, offset=0)
    mx.eval(out_linear, pos_emb_linear, out_mask_linear)

    print(f"Input x shape: {test_x_linear.shape}")
    print(f"Output x shape: {out_linear.shape}")
    print(f"Positional embedding shape: {pos_emb_linear.shape}")
    print(f"Output mask shape: {out_mask_linear.shape}")

    assert out_linear.shape == (batch_size_test, seq_len_test, d_model)
    assert pos_emb_linear.shape == (1, seq_len_test, d_model) # PE module returns (1, size, D)
    assert mx.array_equal(out_mask_linear, test_x_mask_linear) # Mask unchanged
    assert linear_nosub.subsampling_rate == 1
    print("MLXLinearNoSubsampling test passed.")

    # --- Test MLXConv2dSubsampling4 ---
    print("\nTesting MLXConv2dSubsampling4...")
    # max_len for PE module for Conv2dSubsampling4 should be seq_len_test // 4
    pe_module_conv = MLXPositionalEncoding(d_model=d_model, dropout_rate=dropout_rate_test, max_len=seq_len_test // 4 + 1)
    mx.eval(pe_module_conv.parameters())

    conv2d_sub4 = MLXConv2dSubsampling4(idim=idim_test, odim=d_model, dropout_rate=dropout_rate_test, mlx_pos_enc_module=pe_module_conv)
    mx.eval(conv2d_sub4.parameters())
    
    test_x_conv = mx.random.normal((batch_size_test, seq_len_test, idim_test))
    test_x_mask_conv = mx.ones((batch_size_test, 1, seq_len_test)) # Dummy mask

    out_conv, pos_emb_conv, out_mask_conv = conv2d_sub4(test_x_conv, test_x_mask_conv, offset=0)
    mx.eval(out_conv, pos_emb_conv, out_mask_conv)

    print(f"Input x shape: {test_x_conv.shape}")
    print(f"Output x shape: {out_conv.shape}")
    print(f"Positional embedding shape: {pos_emb_conv.shape}")
    print(f"Output mask shape: {out_mask_conv.shape}")
    
    # Expected output time dimension: T_out = ((T_in - K1)/S1 + 1 - K2)/S2 + 1
    # T_out1 = (seq_len_test - 3)//2 + 1
    # T_out_final = (T_out1 - 3)//2 + 1
    expected_T_out = ((seq_len_test - 3)//2 + 1 - 3)//2 + 1
    
    assert out_conv.shape == (batch_size_test, expected_T_out, d_model)
    assert pos_emb_conv.shape == (1, expected_T_out, d_model)
    # Expected mask length: T_in / 4 (simplified) or more accurately based on conv output
    # The current mask subsampling is `::self.subsampling_rate`
    expected_mask_len = (seq_len_test + conv2d_sub4.subsampling_rate -1) // conv2d_sub4.subsampling_rate # Simplified ceil(T_in/S)
    # If using `::S`, then length is `(T_in - 1)//S + 1` if S is total stride.
    # Or more simply, `ceil(T_in / S)`.
    # The code has `subsampled_mask = x_mask[:, :, ::self.subsampling_rate]`
    # which gives `(T_in - 1) // S + 1` length if S is `self.subsampling_rate`.
    # Let's verify against `out_conv.shape[1]`
    assert out_mask_conv.shape == (batch_size_test, 1, expected_T_out)
    assert conv2d_sub4.subsampling_rate == 4
    print("MLXConv2dSubsampling4 test passed.")
    
    print("\n--- End of Tests ---")
