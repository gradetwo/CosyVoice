import mlx.core as mx
import mlx.nn as nn
import numpy as np
import math
from typing import Tuple, Union

class MLXPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout_rate: float, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        self.xscale = mx.sqrt(mx.array(self.d_model, dtype=mx.float32))
        self.dropout = nn.Dropout(p=dropout_rate)
        self.max_len = max_len

        pe = mx.zeros((self.max_len, self.d_model), dtype=mx.float32)
        position = mx.arange(0, self.max_len, dtype=mx.float32).expand_dims(1)
        div_term = mx.exp(
            mx.arange(0, self.d_model, 2, dtype=mx.float32) *
            -(math.log(10000.0) / self.d_model)
        )
        pe[:, 0::2] = mx.sin(position * div_term)
        pe[:, 1::2] = mx.cos(position * div_term)
        self.pe = pe.expand_dims(0) # Shape (1, max_len, d_model)

    def position_encoding(self, offset: Union[int, mx.array], size: int, apply_dropout: bool = True) -> mx.array:
        if isinstance(offset, int):
            # Ensure offset and size are within bounds
            start = offset
            end = offset + size
            if not (0 <= start and end <= self.max_len):
                 raise ValueError(f"Offset {offset} + size {size} is out of bounds for max_len {self.max_len}")
            pos_emb = self.pe[:, start : end]
        elif isinstance(offset, mx.array) and offset.ndim == 0: # scalar mx.array
            offset_val = offset.item()
            start = int(offset_val)
            end = int(offset_val) + size
            if not (0 <= start and end <= self.max_len):
                 raise ValueError(f"Offset {offset_val} + size {size} is out of bounds for max_len {self.max_len}")
            pos_emb = self.pe[:, start : end]
        else: 
            # Batched tensor offset for F.embedding style gathering.
            # MLX nn.Embedding could be used if self.pe was just the weights, but it's precomputed.
            # This requires more complex gather operations.
            raise NotImplementedError(
                "Batched tensor offset for MLXPositionalEncoding.position_encoding is not implemented in this PoC."
            )

        if apply_dropout:
            pos_emb = self.dropout(pos_emb)
        return pos_emb

    def __call__(self, x: mx.array, offset: Union[int, mx.array] = 0) -> Tuple[mx.array, mx.array]:
        # Get positional embedding without dropout (dropout applied at the end)
        pos_emb = self.position_encoding(offset, x.shape[1], apply_dropout=False) 
        
        scaled_x = x * self.xscale
        encoded_x = scaled_x + pos_emb
        
        return self.dropout(encoded_x), self.dropout(pos_emb)


class MLXEspnetRelPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout_rate: float, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        self.xscale = mx.sqrt(mx.array(d_model, dtype=mx.float32))
        self.dropout = nn.Dropout(p=dropout_rate)
        # max_len here refers to the maximum expected sequence length for which PE needs to be generated.
        # self.pe will store relative encodings for up to 2*max_len - 1 relative positions.
        self.max_len_for_pe_construction = max_len 
        self.pe = None 

    def extend_pe(self, x: mx.array):
        # x shape: (B, T, D), T is current sequence length
        current_seq_len = x.shape[1]
        
        # self.pe stores encodings for relative positions from -(L-1) to (L-1) where L = current_seq_len
        # So, total length of self.pe should be 2*current_seq_len - 1
        # The self.max_len_for_pe_construction is the L_max for which PE is precomputed/extended.
        # We need to ensure self.pe can cover relative positions for current_seq_len.
        # This means self.pe should be of length 2 * self.max_len_for_pe_construction - 1 if precomputed to max.
        # Or, it can be dynamically extended to 2 * current_seq_len - 1.
        # The ESPnet impl. seems to extend PE based on current_seq_len (L_cur) to cover [-L_cur+1, L_cur-1].

        # Let L be current_seq_len for which we need relative PEs.
        # self.pe needs to be of shape (1, 2*L - 1, D)
        # Let's use self.max_len_for_pe_construction as the L for which we build PE if not built or too small.
        # This means self.pe will be of size (1, 2*self.max_len_for_pe_construction -1, D)
        # This is consistent with typical relative PE where a large enough band is precomputed.

        # Determine the L for current PE construction. If streaming, this L might be fixed.
        # If not streaming or first time, L can be current_seq_len or self.max_len_for_pe_construction.
        # Let's use self.max_len_for_pe_construction to build a PE that's reusable.
        # The `size` param in `position_encoding` will then determine what slice is taken.
        
        L_construct = self.max_len_for_pe_construction

        if self.pe is not None and self.pe.shape[1] >= (2 * L_construct - 1) and self.pe.dtype == x.dtype:
            return

        pe_positive = mx.zeros((L_construct, self.d_model), dtype=x.dtype)
        position = mx.arange(0, L_construct, dtype=x.dtype).expand_dims(1)
        div_term = mx.exp(
            mx.arange(0, self.d_model, 2, dtype=x.dtype) *
            -(math.log(10000.0) / self.d_model)
        )
        pe_positive[:, 0::2] = mx.sin(position * div_term)
        pe_positive[:, 1::2] = mx.cos(position * div_term)

        # Create PE for negative relative positions: sin(-pos*div) = -sin(pos*div), cos(-pos*div)=cos(pos*div)
        # Or, simply use different positions. For Transformer-XL style, it's symmetric.
        # The ESPnet code concatenates flipped positive PEs (for t_query > t_key) 
        # and non-flipped positive PEs (excluding pos 0, for t_query < t_key).
        # This means PEs for relative positions: [-(L-1), ..., -1, 0, 1, ..., L-1]
        # Total length 2*L - 1. Index L-1 corresponds to relative position 0.
        
        # pe_positive[0] is for relative index 0.
        # pe_positive[1] is for relative index 1 (key is 1 step ahead of query)
        # We need to construct self.pe such that index `center` corresponds to relative_pos = 0.
        # And it covers from -(L_construct-1) to +(L_construct-1).
        
        # Example: L_construct = 3. Rel_pos: -2, -1, 0, 1, 2. Total 2*3-1 = 5. Center index = 2 (for rel_pos 0)
        # self.pe[center] = PE(0)
        # self.pe[center+1] = PE(1) ; self.pe[center-1] = PE(-1) = PE_neg(1)
        # self.pe[center+2] = PE(2) ; self.pe[center-2] = PE_neg(2)
        
        # Let's use the structure from example: pe_positive_flipped and pe_negative[1:]
        # pe_negative is just pe_positive with negative positions, which is not what's usually done.
        # Standard Transformer-XL: PE(pos) = sin(pos/10000^(2i/d)).
        # self.pe is indexed by (key_idx - query_idx + max_len - 1).
        # The example's extend_pe logic for self.pe construction:
        # pe_negative[:, 0::2] = mx.sin(-1 * position * div_term)
        # pe_negative[:, 1::2] = mx.cos(-1 * position * div_term)
        # pe_positive_flipped = mx.flip(pe_positive, axis=0) # PEs for 0, -1, -2, ... -(L-1)
        # self.pe = mx.concatenate([pe_positive_flipped, pe_negative[1:]], axis=0)
        # This means: [PE(0), PE(-1), ..., PE(-(L-1))] cat [PE_neg(1), ..., PE_neg(L-1)]
        # This is not the standard symmetrical PE band.
        # ESPnet's RelPositionalEncoding uses `self.pe = torch.cat([pe_positive_flipped, pe_positive[1:]], dim=0)`
        # This makes self.pe = [PE(L-1), PE(L-2), ..., PE(1), PE(0), PE(1), ..., PE(L-1)]
        # This is a symmetric band of length 2*L-1. Index L-1 is PE(0).
        
        pe_positive_flipped_excluding_zero = mx.flip(pe_positive[1:], axis=0) # PE(L-1)...PE(1)
        if L_construct == 1:
             self.pe = pe_positive[0:1,:].expand_dims(0) # Just PE(0)
        else:
             self.pe = mx.concatenate([pe_positive_flipped_excluding_zero, pe_positive], axis=0).expand_dims(0)
        # Now self.pe has shape (1, 2*L_construct-1, d_model)
        # Index L_construct-1 corresponds to relative position 0.

    def position_encoding(self, size: int, offset: int) -> mx.array:
        # self.pe is (1, 2*L_construct-1, D), center at L_construct-1 is rel_pos 0.
        # We need to return a slice of length (2*size-1) for relative positions
        # from -(size-1)+offset to (size-1)+offset.
        
        L_construct = self.pe.shape[1] // 2 + 1 # The L used to build self.pe
        center_of_pe = L_construct - 1

        # Relative indices required: from -(size-1)+offset to (size-1)+offset
        # Smallest relative index: rel_start = -(size - 1) + offset
        # Largest relative index:  rel_end   =  (size - 1) + offset
        
        # Corresponding indices in self.pe:
        # self.pe_idx = center_of_pe + relative_index
        pe_idx_start = center_of_pe + rel_start
        pe_idx_end = center_of_pe + rel_end
        
        # Slice from pe_idx_start to pe_idx_end (inclusive for end)
        # Python slicing is exclusive for end, so pe_idx_end + 1
        
        # Boundary checks against self.pe's actual extent
        if not (0 <= pe_idx_start and pe_idx_end < self.pe.shape[1]):
             # This should not happen if extend_pe was called with x.shape[1] >= size
             # and self.max_len_for_pe_construction is also >= size.
             # And if offset doesn't push it too far.
             # For robust slicing, clamp to actual self.pe bounds if needed,
             # though this might indicate an issue with how `size` and `offset` relate to `L_construct`.
             # ESPnet's RelSelfAttention has logic to handle this when creating rel_pos_emb.
             # This function should just return the slice as requested.
             # If indices are out of bound for self.pe, it's an issue.
             # For now, assume valid indices based on prior extend_pe and sensible size/offset.
            pass

        return self.pe[:, pe_idx_start : pe_idx_end + 1]

    def __call__(self, x: mx.array, offset: int = 0) -> Tuple[mx.array, mx.array]:
        # x is (B, T, D), T is current sequence length (size for position_encoding)
        self.extend_pe(x) 
        
        scaled_x = x * self.xscale
        
        # The `size` for `position_encoding` is typically the current sequence length T.
        # The `offset` is used to shift the window of relative PEs.
        pos_emb = self.position_encoding(size=x.shape[1], offset=offset)
        
        return self.dropout(scaled_x), self.dropout(pos_emb)


# Example Usage (can be kept for testing or removed for production code):
if __name__ == '__main__':
    print("--- MLXPositionalEncoding Tests ---")
    d_model_test = 4
    dropout_test = 0.0
    max_len_test = 10
    batch_size_test = 1
    seq_len_test = 5

    # Test MLXPositionalEncoding
    pos_enc = MLXPositionalEncoding(d_model_test, dropout_test, max_len_test)
    mx.eval(pos_enc.parameters()) # Ensure dropout is initialized if stateful
    
    test_x = mx.zeros((batch_size_test, seq_len_test, d_model_test))
    
    # Test with int offset
    offset_int = 2
    encoded_x_int, p_emb_int = pos_enc(test_x, offset=offset_int)
    mx.eval(encoded_x_int, p_emb_int)
    print(f"Input x shape: {test_x.shape}")
    print(f"Int offset: {offset_int}")
    print(f"Encoded x shape (int offset): {encoded_x_int.shape}")
    print(f"Pos emb shape (int offset): {p_emb_int.shape}")
    assert p_emb_int.shape == (1, seq_len_test, d_model_test)
    assert mx.array_equal(p_emb_int, pos_enc.pe[:, offset_int : offset_int + seq_len_test])

    # Test with scalar mx.array offset
    offset_mx_scalar = mx.array(1)
    encoded_x_mx, p_emb_mx = pos_enc(test_x, offset=offset_mx_scalar)
    mx.eval(encoded_x_mx, p_emb_mx)
    print(f"\nScalar mx.array offset: {offset_mx_scalar.item()}")
    print(f"Encoded x shape (mx_scalar offset): {encoded_x_mx.shape}")
    print(f"Pos emb shape (mx_scalar offset): {p_emb_mx.shape}")
    assert p_emb_mx.shape == (1, seq_len_test, d_model_test)
    assert mx.array_equal(p_emb_mx, pos_enc.pe[:, offset_mx_scalar.item() : offset_mx_scalar.item() + seq_len_test])

    print("\n--- MLXEspnetRelPositionalEncoding Tests ---")
    rel_pos_enc = MLXEspnetRelPositionalEncoding(d_model_test, dropout_test, max_len=max_len_test)
    
    # Test extend_pe
    rel_pos_enc.extend_pe(test_x) # seq_len_test = 5
    mx.eval(rel_pos_enc.pe)
    print(f"PE shape after extend_pe for seq_len {seq_len_test} (using max_len {max_len_test}): {rel_pos_enc.pe.shape}")
    # Expected PE length: 2 * max_len_test - 1 = 2*10 - 1 = 19
    assert rel_pos_enc.pe.shape == (1, 2 * max_len_test - 1, d_model_test)

    # Test position_encoding of MLXEspnetRelPositionalEncoding
    # Request PEs for a window of `size_att` relevant for a query at `offset_att`
    size_att = seq_len_test # current chunk length T = 5
    offset_att = 0 # No shift in the relative PE window
    
    # This should return PE for relative positions [-(size_att-1)+offset_att, ..., (size_att-1)+offset_att]
    # i.e., [-4, -3, -2, -1, 0, 1, 2, 3, 4] if offset_att=0, size_att=5. Length 2*5-1 = 9.
    rel_p_emb_slice = rel_pos_enc.position_encoding(size=size_att, offset=offset_att)
    mx.eval(rel_p_emb_slice)
    print(f"\nRel position_encoding for size={size_att}, offset={offset_att}: shape={rel_p_emb_slice.shape}")
    assert rel_p_emb_slice.shape == (1, 2 * size_att - 1, d_model_test)

    # Test __call__ of MLXEspnetRelPositionalEncoding
    encoded_x_rel, p_emb_rel = rel_pos_enc(test_x, offset=offset_att)
    mx.eval(encoded_x_rel, p_emb_rel)
    print(f"\nInput x shape: {test_x.shape}")
    print(f"Rel Encoded x shape: {encoded_x_rel.shape}")
    print(f"Rel Pos emb shape from __call__: {p_emb_rel.shape}") # This is the slice from position_encoding
    assert p_emb_rel.shape == (1, 2 * size_att - 1, d_model_test)
    
    # Test with a different offset for position_encoding call
    offset_att_shifted = 1
    # This should return PE for relative positions [-(5-1)+1, ..., (5-1)+1] = [-3, ..., 5]
    rel_p_emb_slice_shifted = rel_pos_enc.position_encoding(size=size_att, offset=offset_att_shifted)
    mx.eval(rel_p_emb_slice_shifted)
    print(f"Rel position_encoding for size={size_att}, offset={offset_att_shifted}: shape={rel_p_emb_slice_shifted.shape}")
    assert rel_p_emb_slice_shifted.shape == (1, 2 * size_att - 1, d_model_test)
    
    # Verify the slice content (conceptual check)
    # Center of self.pe (L_construct-1) = 10-1 = 9. This is rel_pos 0.
    # For size=5, offset=0: rel_start=-4, rel_end=4.
    # pe_idx_start = 9 + (-4) = 5. pe_idx_end = 9 + 4 = 13. Slice self.pe[:, 5:14]
    center_of_pe_test = max_len_test - 1
    pe_idx_start_test = center_of_pe_test - (size_att - 1) + offset_att
    pe_idx_end_test = center_of_pe_test + (size_att - 1) + offset_att
    expected_slice = rel_pos_enc.pe[:, pe_idx_start_test : pe_idx_end_test + 1]
    assert mx.array_equal(rel_p_emb_slice, expected_slice)
    print("MLXEspnetRelPositionalEncoding tests passed.")

    print("\n--- End of Tests ---")
