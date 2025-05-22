import mlx.core as mx
import mlx.nn as nn
import math
from typing import Tuple, Optional 

class MLXMultiHeadedAttention(nn.Module):
    def __init__(self, n_head: int, n_feat: int, dropout_rate: float, key_bias: bool = True):
        super().__init__()
        assert n_feat % n_head == 0
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat, bias=key_bias)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward_qkv(self, query: mx.array, key: mx.array, value: mx.array
    ) -> Tuple[mx.array, mx.array, mx.array]:
        B = query.shape[0]
        q = self.linear_q(query).reshape(B, -1, self.h, self.d_k)
        k = self.linear_k(key).reshape(B, -1, self.h, self.d_k)
        v = self.linear_v(value).reshape(B, -1, self.h, self.d_k)
        q = q.transpose(0, 2, 1, 3)  # (B, h, T1, d_k)
        k = k.transpose(0, 2, 1, 3)  # (B, h, T2, d_k)
        v = v.transpose(0, 2, 1, 3)  # (B, h, T2, d_k)
        return q, k, v

    def forward_attention(self, value: mx.array, scores: mx.array, mask: mx.array
    ) -> mx.array:
        B = value.shape[0]
        # mask shape is (B, 1, T1_or_1, T2) or (B, T2) or (B, T1, T2)
        # scores shape is (B, h, T1, T2)
        if mask.size > 0: 
            # Expand mask to be broadcastable with scores
            # Common mask shapes:
            # (B, T2) -> (B, 1, 1, T2) for self-attention encoder mask
            # (B, T1, T2) -> (B, 1, T1, T2) for decoder cross-attention or full mask
            if mask.ndim == 2: # (B, T2)
                mask_expanded = mask[:, None, None, :] 
            elif mask.ndim == 3: # (B, T1, T2)
                mask_expanded = mask[:, None, :, :]
            else: # Already (B, 1, T1_or_1, T2) or similar
                mask_expanded = mask
            
            # The condition for masking is where mask is False (0)
            # So, mx.where(mask_condition_is_true, use_scores, use_fill_value)
            # If mask represents padding (0 for pad, 1 for non-pad), then we want to fill where mask is 0.
            # The example had `mask.expand_dims(1) == 0`.
            # If `mask` is already boolean where True means "keep" and False means "mask out":
            # then `mx.where(mask_expanded, scores, fill_value)`
            # If `mask` is 0/1 where 0 means "mask out":
            mask_condition = (mask_expanded != 0) # True where not masked
            
            # Ensure mask_condition is broadcastable with scores.
            # It should be (B, 1, T1_or_1, T2) -> broadcasts over h.
            scores = mx.where(mask_condition, scores, mx.array(-1e9, dtype=scores.dtype))

        attn = nn.softmax(scores, axis=-1)
        p_attn = self.dropout(attn)
        x = mx.matmul(p_attn, value)  # (B, h, T1, d_k)
        x = x.transpose(0, 2, 1, 3).reshape(B, -1, self.h * self.d_k) # (B, T1, n_feat)
        return self.linear_out(x)

    def __call__(self, query: mx.array, key: mx.array, value: mx.array,
                 mask: mx.array, 
                 pos_emb: Optional[mx.array] = None, 
                 cache: Optional[mx.array] = None
    ) -> Tuple[mx.array, mx.array]:
        q, k, v = self.forward_qkv(query, key, value)

        if cache is not None and cache.size > 0 : 
            # cache is (B, h, T_cache, d_k*2) or similar structure
            # For this MHA, cache should store key and value
            # Assuming cache stores concatenated k and v: (B, h, T_cache, 2*d_k)
            # Or it could be a tuple (key_cache, value_cache)
            # Based on `mx.concatenate((k,v), axis=-1)` for new_cache, let's assume concatenated.
            
            # If cache is (B, h, T_cache, d_k*2)
            # We need to ensure d_k matches.
            # If it's (B, T_cache, h, d_k*2) after some reshape, need to be careful.
            # Let's assume cache is (B, h, T_cache, d_k_concat) where d_k_concat = 2*d_k
            
            num_cached_dims = k.shape[-1] # d_k
            key_cache, value_cache = mx.split(cache, [num_cached_dims], axis=-1) # Split into two parts along last axis

            k = mx.concatenate([key_cache, k], axis=2) # Concat along time dim T_cache + T_new_k
            v = mx.concatenate([value_cache, v], axis=2)
        
        new_cache = mx.concatenate((k, v), axis=-1) # Store k and v concatenated
        
        scores = mx.matmul(q, k.transpose(0, 1, 3, 2)) / math.sqrt(self.d_k) # (B,h,T1,d_k) @ (B,h,d_k,T2) -> (B,h,T1,T2)
        return self.forward_attention(v, scores, mask), new_cache


class MLXRelPositionMultiHeadedAttention(MLXMultiHeadedAttention):
    def __init__(self, n_head: int, n_feat: int, dropout_rate: float, key_bias: bool = True):
        super().__init__(n_head, n_feat, dropout_rate, key_bias)
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        # These are treated as buffers/non-parameters in PyTorch, loaded from checkpoint.
        # In MLX, if they are not nn.Parameter, update() won't affect them.
        # For inference, direct mx.array attributes are fine if weights are set manually after init.
        # If these were meant to be learnable parameters from scratch:
        # self.pos_bias_u = nn.Parameter(mx.zeros((self.h, self.d_k))) 
        # self.pos_bias_v = nn.Parameter(mx.zeros((self.h, self.d_k)))
        # For now, as per example, they are mx.zeros and will be overwritten by loaded weights.
        self.pos_bias_u = mx.zeros((self.h, self.d_k))
        self.pos_bias_v = mx.zeros((self.h, self.d_k))

    def rel_shift(self, x: mx.array) -> mx.array:
        # x: (B, H, T1, T_rel) where T_rel is typically 2*T1-1 or 2*T2-1
        B, H, T1, T_rel = x.shape
        
        # Pad with one column of zeros on the left (dim=-1, which is T_rel)
        # padding = [(0,0)] * (x.ndim - 1) + [(1,0)] # This pads the last dimension
        # Correct padding for MLX: list of (before, after) pairs for each dim
        padding_config = [(0,0) for _ in range(x.ndim -1)] + [(1,0)] # Pads last dim on left
        x_padded = mx.pad(x, padding_config, constant_values=0.0)
        
        # Reshape and slice to achieve the shift
        # x_padded: (B, H, T1, T_rel + 1)
        # View as (B, H, T_rel + 1, T1) according to ESPnet code.
        x_padded = x_padded.reshape(B, H, T_rel + 1, T1)
        
        # Slice to remove the first element of the T_rel+1 dimension, then reshape back
        # ESPnet: x_padded[:, :, 1:].view_as(x) -> (B,H,T1,T_rel)
        # This means: x_padded.slice[:,:,1:,:].reshape(B,H,T1,T_rel)
        x_shifted = x_padded[:, :, 1:, :].reshape(B, H, T1, T_rel) 
        
        # Slice to the final desired size, typically (B, H, T1, T1) if T_key=T1
        # The original code has `x = x[:, :, :, :T1]` if `T_rel` was `2*T1-1`
        # This assumes T_key (used to determine T_rel) equals T_query (T1).
        # If T_key can be different, this slice needs to be `T_key`.
        # For self-attention, T_query = T_key = T1.
        # So, if T_rel = 2*T1 - 1, then this slice will be x_shifted[:,:,:, :T1]
        # The example code's rel_shift has `return x[:, :, :, :T1]`
        # This assumes T1 is the query length.
        # The length of the relevant part of the relative PE matrix is T1 (query length).
        return x_shifted[:, :, :, :T1]


    def __call__(self, query: mx.array, key: mx.array, value: mx.array,
                 mask: mx.array, 
                 pos_emb: mx.array, # (B_pos, T_rel, D_feat)
                 cache: Optional[mx.array] = None
    ) -> Tuple[mx.array, mx.array]: 
        q_raw, k_raw, v_raw = self.forward_qkv(query, key, value) # q,k,v are (B, h, T_q/k/v, d_k)

        if cache is not None and cache.size > 0:
            num_cached_dims = k_raw.shape[-1]
            key_cache, value_cache = mx.split(cache, [num_cached_dims], axis=-1)
            k = mx.concatenate([key_cache, k_raw], axis=2)
            v = mx.concatenate([value_cache, v_raw], axis=2)
        else:
            k, v = k_raw, v_raw
            
        new_cache = mx.concatenate((k, v), axis=-1)

        # q_raw is (B, h, T1, d_k). For score calculation with biases, need (B, T1, h, d_k)
        q_for_bias_calc = q_raw.transpose(0, 2, 1, 3) 

        # Add biases. pos_bias_u/v are (h, d_k). Reshape for broadcasting.
        bias_u_reshaped = self.pos_bias_u.reshape(1, 1, self.h, self.d_k)
        bias_v_reshaped = self.pos_bias_v.reshape(1, 1, self.h, self.d_k)
        
        q_with_bias_u = q_for_bias_calc + bias_u_reshaped
        q_with_bias_v = q_for_bias_calc + bias_v_reshaped

        # Transpose qs back for matmul: (B, h, T1, d_k)
        q_u_final = q_with_bias_u.transpose(0, 2, 1, 3)
        q_v_final = q_with_bias_v.transpose(0, 2, 1, 3)

        # Project positional embedding: pos_emb is (B_pos, T_rel, D_feat)
        # B_pos can be 1 if pos_emb is shared across batch.
        p_proj = self.linear_pos(pos_emb) # (B_pos, T_rel, n_feat)
        p_reshaped = p_proj.reshape(p_proj.shape[0], -1, self.h, self.d_k) # (B_pos, T_rel, h, d_k)
        p = p_reshaped.transpose(0, 2, 1, 3)  # (B_pos, h, T_rel, d_k)

        # AC term: (q + u) @ k.T
        # q_u_final: (B, h, T1, d_k), k: (B, h, T2, d_k) -> k.T: (B, h, d_k, T2)
        matrix_ac = mx.matmul(q_u_final, k.transpose(0, 1, 3, 2))

        # BD term: (q + v) @ p.T (relative part)
        # q_v_final: (B, h, T1, d_k), p: (B_pos, h, T_rel, d_k) -> p.T: (B_pos, h, d_k, T_rel)
        # If B_pos=1 and B>1, p will broadcast.
        matrix_bd = mx.matmul(q_v_final, p.transpose(0, 1, 3, 2)) # (B, h, T1, T_rel)
        
        # Apply relative shift to BD term
        # rel_shift expects (B, H, T1, T_rel) and outputs (B, H, T1, T1_slice_from_T_rel)
        matrix_bd_shifted = self.rel_shift(matrix_bd) 
        
        scores = (matrix_ac + matrix_bd_shifted) / math.sqrt(self.d_k)
        
        return self.forward_attention(v, scores, mask), new_cache

class MLXDiffusersAttention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        self.inner_dim = dim_head * heads
        self.query_dim = query_dim
        self.cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head**-0.5

        self.to_q = nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_k = nn.Linear(self.cross_attention_dim, self.inner_dim, bias=bias)
        self.to_v = nn.Linear(self.cross_attention_dim, self.inner_dim, bias=bias)

        to_out_layers = [nn.Linear(self.inner_dim, query_dim)]
        if dropout > 0.0:
            to_out_layers.append(nn.Dropout(dropout))
        self.to_out = nn.Sequential(*to_out_layers)
        
    def _prepare_qkv(self, query: mx.array, key_value: mx.array) -> Tuple[mx.array, mx.array, mx.array]:
        B_q, T_q, _ = query.shape # query is hidden_states
        B_kv, T_kv, _ = key_value.shape 

        q = self.to_q(query)
        k = self.to_k(key_value)
        v = self.to_v(key_value)

        q = q.reshape(B_q, T_q, self.heads, self.dim_head).transpose(0, 2, 1, 3) # (B, H, T_q, Dh)
        k = k.reshape(B_kv, T_kv, self.heads, self.dim_head).transpose(0, 2, 1, 3) # (B, H, T_kv, Dh)
        v = v.reshape(B_kv, T_kv, self.heads, self.dim_head).transpose(0, 2, 1, 3) # (B, H, T_kv, Dh)
        
        return q, k, v

    def __call__(
        self,
        hidden_states: mx.array, 
        encoder_hidden_states: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
        **kwargs 
    ) -> mx.array:
        
        key_value_source = hidden_states if encoder_hidden_states is None else encoder_hidden_states
        
        q, k, v = self._prepare_qkv(hidden_states, key_value_source)
        
        scores = mx.matmul(q, k.transpose(0, 1, 3, 2)) * self.scale
        
        if attention_mask is not None:
            # Mask is expected to be boolean: True for valid, False for masked.
            # Shape can be (B, T_q, T_kv) or (B, 1, T_q, T_kv) or (B, H, T_q, T_kv)
            if attention_mask.ndim == 3: # (B, T_q, T_kv)
                attention_mask = attention_mask.expand_dims(1) # -> (B, 1, T_q, T_kv) for broadcasting over heads
            # `mx.where` expects condition, x, y. If condition is False, use -1e9 (or -mx.inf).
            scores = mx.where(attention_mask, scores, mx.array(-1e9 if scores.dtype == mx.float32 else -1e4, dtype=scores.dtype))

        attn_probs = nn.softmax(scores, axis=-1)
        
        # Diffusers Attention does not apply dropout to attention_probs here.
        # Dropout is part of self.to_out if configured.

        context = mx.matmul(attn_probs, v) 
        context = context.transpose(0, 2, 1, 3).reshape(hidden_states.shape[0], hidden_states.shape[1], self.inner_dim)
        
        return self.to_out(context)


# Example Usage (can be kept for testing or removed for production code):
if __name__ == '__main__':
    print("--- MLX Attention Modules Tests ---")
    
    batch_size, seq_len_q, seq_len_kv = 2, 10, 12
    n_feat_test = 16 
    n_head_test = 4
    dropout_test = 0.0

    query_tensor = mx.random.normal((batch_size, seq_len_q, n_feat_test))
    key_tensor = mx.random.normal((batch_size, seq_len_kv, n_feat_test))
    value_tensor = mx.random.normal((batch_size, seq_len_kv, n_feat_test))
    mask_tensor = mx.ones((batch_size, 1, seq_len_kv), dtype=mx.bool_) # Boolean mask
    if seq_len_kv > 0 : mask_tensor[:, :, -seq_len_kv//2:] = False

    # Test MLXMultiHeadedAttention
    print("\nTesting MLXMultiHeadedAttention...")
    mha = MLXMultiHeadedAttention(n_head_test, n_feat_test, dropout_test)
    mx.eval(mha.parameters())
    output_mha, new_cache_mha = mha(query_tensor, key_tensor, value_tensor, mask_tensor.astype(mx.float32)) # MHA expects float mask
    mx.eval(output_mha, new_cache_mha)
    assert output_mha.shape == (batch_size, seq_len_q, n_feat_test)
    print("MLXMultiHeadedAttention basic test passed.")

    # Test MLXRelPositionMultiHeadedAttention
    print("\nTesting MLXRelPositionMultiHeadedAttention...")
    seq_len_rel = 10
    query_rel = mx.random.normal((batch_size, seq_len_rel, n_feat_test))
    T_rel = 2 * seq_len_rel - 1
    pos_emb_tensor = mx.random.normal((1, T_rel, n_feat_test))
    mask_rel_float = mx.ones((batch_size, 1, seq_len_rel), dtype=mx.float32)

    rel_mha = MLXRelPositionMultiHeadedAttention(n_head_test, n_feat_test, dropout_test)
    rel_mha.pos_bias_u = mx.random.normal((n_head_test, n_feat_test // n_head_test)) * 0.02
    rel_mha.pos_bias_v = mx.random.normal((n_head_test, n_feat_test // n_head_test)) * 0.02
    mx.eval(rel_mha.parameters(), rel_mha.pos_bias_u, rel_mha.pos_bias_v)
    output_rel_mha, _ = rel_mha(query_rel, query_rel, query_rel, mask_rel_float, pos_emb_tensor)
    mx.eval(output_rel_mha)
    assert output_rel_mha.shape == (batch_size, seq_len_rel, n_feat_test)
    print("MLXRelPositionMultiHeadedAttention basic test passed.")

    # Test MLXDiffusersAttention
    print("\nTesting MLXDiffusersAttention (Self-Attention)...")
    diffusers_attn_self = MLXDiffusersAttention(query_dim=n_feat_test, heads=n_head_test, dim_head=n_feat_test//n_head_test, dropout=dropout_test)
    mx.eval(diffusers_attn_self.parameters())
    
    # Mask for DiffusersAttention (B, T_q, T_kv) boolean
    self_attn_mask = mx.ones((batch_size, seq_len_q, seq_len_q), dtype=mx.bool_)
    if seq_len_q > 1: self_attn_mask[:, :, -1] = False # Mask one element for testing

    output_diff_self = diffusers_attn_self(hidden_states=query_tensor, attention_mask=self_attn_mask)
    mx.eval(output_diff_self)
    print(f"Diffusers Self-Attention Output shape: {output_diff_self.shape}")
    assert output_diff_self.shape == (batch_size, seq_len_q, n_feat_test)

    print("\nTesting MLXDiffusersAttention (Cross-Attention)...")
    cross_attention_dim_test = n_feat_test * 2 # Example different dim
    encoder_hidden_states_test = mx.random.normal((batch_size, seq_len_kv, cross_attention_dim_test))
    # Mask for cross-attention (B, T_q, T_kv)
    cross_attn_mask = mx.ones((batch_size, seq_len_q, seq_len_kv), dtype=mx.bool_)
    if seq_len_kv > 1: cross_attn_mask[:, :, -1] = False

    diffusers_attn_cross = MLXDiffusersAttention(
        query_dim=n_feat_test, 
        cross_attention_dim=cross_attention_dim_test,
        heads=n_head_test, 
        dim_head=n_feat_test//n_head_test, 
        dropout=dropout_test
    )
    mx.eval(diffusers_attn_cross.parameters())

    output_diff_cross = diffusers_attn_cross(
        hidden_states=query_tensor, 
        encoder_hidden_states=encoder_hidden_states_test, 
        attention_mask=cross_attn_mask
    )
    mx.eval(output_diff_cross)
    print(f"Diffusers Cross-Attention Output shape: {output_diff_cross.shape}")
    assert output_diff_cross.shape == (batch_size, seq_len_q, n_feat_test)
    print("MLXDiffusersAttention tests passed.")

    print("\n--- End of Tests ---")
