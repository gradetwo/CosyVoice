import mlx.core as mx
import mlx.nn as nn
from typing import Tuple, List, Optional, Union
import sys
import os

# Add cosyvoice root to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import MLX modules
try:
    # Assuming these are in the expected locations relative to 'cosyvoice' root
    from cosyvoice.mlx_embedding import MLXEspnetRelPositionalEncoding, MLXPositionalEncoding # Added MLXPositionalEncoding for completeness
    from cosyvoice.mlx_subsampling import MLXLinearNoSubsampling, MLXConv2dSubsampling4 # Added MLXConv2dSubsampling4
    from cosyvoice.mlx_encoder_layer import MLXConformerEncoderLayer
    from cosyvoice.mlx_attention import MLXRelPositionMultiHeadedAttention
    from cosyvoice.utils.mlx_layers import MLXPositionwiseFeedForward, MLXConvolutionModule, MLXSnake # Mish can be nn.Mish
except ImportError as e:
    print(f"Error importing MLX modules: {e}. Ensure all dependencies are correctly placed and named.")
    # Fallbacks for worker, assuming direct paths if structured differently during execution
    from mlx_embedding import MLXEspnetRelPositionalEncoding, MLXPositionalEncoding
    from mlx_subsampling import MLXLinearNoSubsampling, MLXConv2dSubsampling4
    from mlx_encoder_layer import MLXConformerEncoderLayer
    from mlx_attention import MLXRelPositionMultiHeadedAttention
    from utils.mlx_layers import MLXPositionwiseFeedForward, MLXConvolutionModule, MLXSnake


# Helper functions (from example structure)
def make_pad_mask_mlx(lengths: mx.array, max_len: Optional[int] = None) -> mx.array:
    if max_len is None:
        max_len_val = lengths.max().item()
        if isinstance(max_len_val, mx.array): max_len_val = max_len_val.item()
    else:
        max_len_val = max_len
    
    if not isinstance(max_len_val, int): max_len_val = int(max_len_val) # Ensure int

    seq_range = mx.arange(max_len_val)
    return seq_range[None, :] < lengths[:, None] # (B, max_len)


def add_optional_chunk_mask_mlx(xs_shape: Tuple[int, int, int], # B, T, D
                                masks: mx.array, # (B, T_sub, T_sub) or (B, 1, T_sub)
                                use_dynamic_chunk: bool,
                                use_dynamic_left_chunk: bool,
                                decoding_chunk_size: int,
                                static_chunk_size: int,
                                num_decoding_left_chunks: int):
    # Simplified for PoC: just return original mask if not doing dynamic chunking.
    # This function in ESPnet is complex and creates specific chunk masks for attention.
    # For a non-streaming forward pass, the existing `masks` (derived from padding)
    # already define the valid attention context.
    # If `masks` is (B, T_sub, T_sub), it's a full attention mask.
    # If `masks` is (B, 1, T_sub), it's a padding mask to be broadcasted.
    # For this PoC, we will assume `masks` is already the attention mask to be used by layers.
    
    # If no chunking is applied, the mask should allow attending to all valid positions.
    # If `masks` is (B, 1, T_sub) from padding, convert to (B, T_sub, T_sub) square mask.
    if masks.ndim == 3 and masks.shape[1] == 1: # (B, 1, T_sub) padding mask
        # Create a square attention mask: M[i,j] is true if both key j and query i are valid.
        # Also, for self-attention, it's often combined with a causal mask if needed (not typical for Conformer encoder).
        attention_mask = masks.transpose(0, 2, 1) * masks # (B, T_sub, T_sub)
        return attention_mask
    elif masks.ndim == 3 and masks.shape[1] == masks.shape[2]: # Already a square attention mask
        return masks
    else:
        # Fallback or error for unexpected mask shape
        print(f"Warning: Unexpected mask shape {masks.shape} in add_optional_chunk_mask_mlx. Returning as is.")
        return masks


class MLXBaseEncoder(nn.Module):
    def __init__(
        self,
        input_size: int, 
        output_size: int,
        subsampling_module: nn.Module, 
        pos_enc_module: nn.Module, # Passed but primarily used by the subsampling_module
        normalize_before: bool = True,
        # For PoC, streaming params are stored but not fully implemented in forward_chunk
        static_chunk_size: int = -1,
        use_dynamic_chunk: bool = False,
        use_dynamic_left_chunk: bool = False,
    ):
        super().__init__()
        self._output_size = output_size
        self.embed = subsampling_module 
        self.pos_enc_main = pos_enc_module # Keep a reference if needed separately, though embed should handle it.
        self.normalize_before = normalize_before
        self.after_norm = nn.LayerNorm(dims=output_size, eps=1e-5) 
        
        self.static_chunk_size = static_chunk_size
        self.use_dynamic_chunk = use_dynamic_chunk
        self.use_dynamic_left_chunk = use_dynamic_left_chunk

    def output_size(self) -> int:
        return self._output_size

    def forward_layers(self, xs: mx.array, chunk_masks: mx.array, 
                       pos_emb: mx.array, mask_pad: mx.array, # mask_pad is for conv module
                       att_cache: Optional[mx.array] = None, # For streaming
                       cnn_cache: Optional[mx.array] = None  # For streaming
    ) -> Tuple[mx.array, Optional[mx.array], Optional[mx.array]]: # Return updated caches
        raise NotImplementedError("forward_layers must be implemented by subclasses")

    def __call__(
        self,
        xs: mx.array, 
        xs_lens: mx.array, 
        decoding_chunk_size: int = 0, 
        num_decoding_left_chunks: int = -1, 
    ) -> Tuple[mx.array, mx.array]:
        
        input_max_len = xs.shape[1]
        # Create boolean mask (True for valid, False for pad)
        initial_input_mask_bool = make_pad_mask_mlx(xs_lens, input_max_len) 
        initial_input_mask = initial_input_mask_bool.expand_dims(1) # (B, 1, T_in)

        # self.embed is the subsampling module (e.g., MLXLinearNoSubsampling, MLXConv2dSubsampling4)
        # It should apply positional encoding internally using its own pos_enc module.
        # __call__ of subsampling modules: (x, x_mask, offset) -> (x_out, pos_emb_out, mask_out)
        xs_embed, pos_emb, subsampled_mask_pad = self.embed(xs, initial_input_mask, offset=0)
        
        # subsampled_mask_pad is (B, 1, T_subsampled), True for valid.
        # chunk_masks for Conformer layers (attention mask).
        # For non-streaming, full attention within padding.
        # add_optional_chunk_mask_mlx will convert padding mask to square attention mask.
        chunk_masks_for_attn = add_optional_chunk_mask_mlx(
            xs_shape=xs_embed.shape, # Not directly used by simplified version
            masks=subsampled_mask_pad, 
            use_dynamic_chunk=self.use_dynamic_chunk, 
            use_dynamic_left_chunk=self.use_dynamic_left_chunk,
            decoding_chunk_size=decoding_chunk_size, 
            static_chunk_size=self.static_chunk_size,
            num_decoding_left_chunks=num_decoding_left_chunks
        )
        
        # forward_layers expects mask_pad for conv module, which is subsampled_mask_pad
        xs_encoded, _, _ = self.forward_layers(xs_embed, chunk_masks_for_attn, pos_emb, subsampled_mask_pad,
                                               att_cache=None, cnn_cache=None) # No cache for full forward
        
        if self.normalize_before:
            xs_final = self.after_norm(xs_encoded)
        else: # If post-norm, after_norm is applied inside layers or not at all at the end
            xs_final = xs_encoded 
        
        return xs_final, subsampled_mask_pad # Return final features and their padding mask

    def forward_chunk(self, xs: mx.array, offset: int, required_cache_size: int,
                      att_cache: mx.array, cnn_cache: mx.array
    ) -> Tuple[mx.array, mx.array, mx.array]:
        # xs: (B, T_chunk, D_in)
        # offset: current chunk's start time index in global sequence
        # required_cache_size: how much of the new cache to keep
        
        # Create mask for the current chunk (assuming all valid within chunk)
        chunk_mask_for_embed = mx.ones((xs.shape[0], 1, xs.shape[1]), dtype=mx.bool_)
        
        xs_embed, pos_emb, subsampled_mask_pad_chunk = self.embed(xs, chunk_mask_for_embed, offset=offset)
        
        # For streaming, chunk_masks for attention needs to be causal or specific to lookback.
        # This is a complex part. For PoC, a simplified causal mask for the chunk might be:
        # T_chunk_subsampled = xs_embed.shape[1]
        # chunk_att_mask = mx.tril(mx.ones((T_chunk_subsampled, T_chunk_subsampled), dtype=mx.bool_)).expand_dims(0).expand_dims(0)
        # This doesn't account for cache yet.
        # The add_optional_chunk_mask_mlx should ideally handle this based on decoding_chunk_size etc.
        # For this stub, we'll pass the subsampled_mask_pad_chunk, assuming layer handles full attention over cache+chunk.
        
        # The self-attention layer in ConformerEncoderLayer will handle cache concatenation.
        # The mask passed to it should be for the combined (cache + current_chunk) sequence.
        # This requires careful construction of chunk_masks.
        # For PoC, let's assume the layer's internal masking logic can handle a basic padding mask
        # over the concatenated sequence, or that the provided `mask` to the layer is a full one.
        # This part is highly dependent on how MLXRelPositionMultiHeadedAttention handles masks with cache.
        
        # For now, let's pass a mask relevant to the current chunk's processing (potentially simplified)
        # This will need to be refined for correct streaming behavior.
        # A placeholder: use subsampled_mask_pad_chunk converted to a square mask for the current chunk.
        # True streaming mask would be more complex.
        chunk_att_mask = add_optional_chunk_mask_mlx(None, subsampled_mask_pad_chunk, False, False, 0,0,0)


        xs_encoded, new_att_cache, new_cnn_cache = self.forward_layers(
            xs_embed, chunk_att_mask, pos_emb, subsampled_mask_pad_chunk,
            att_cache=att_cache, cnn_cache=cnn_cache
        )

        if self.normalize_before:
            xs_final = self.after_norm(xs_encoded)
        else:
            xs_final = xs_encoded
            
        # Slice new_att_cache and new_cnn_cache to required_cache_size if it's positive
        if required_cache_size > 0:
            if new_att_cache is not None and new_att_cache.size > 0:
                new_att_cache = new_att_cache[:, :, -required_cache_size:, :]
            if new_cnn_cache is not None and new_cnn_cache.size > 0: # cnn_cache is (B, C, T_cache)
                new_cnn_cache = new_cnn_cache[:, :, -required_cache_size:] # Assuming T_cache is last dim for CNN
        elif required_cache_size == 0: # Reset cache
             new_att_cache = mx.zeros_like(new_att_cache) if new_att_cache is not None else mx.array([])
             new_cnn_cache = mx.zeros_like(new_cnn_cache) if new_cnn_cache is not None else mx.array([])


        return xs_final, new_att_cache, new_cnn_cache


class MLXConformerEncoder(MLXBaseEncoder):
    def __init__(
        self,
        input_size: int, 
        output_size: int = 256, 
        attention_heads: int = 4,
        linear_units: int = 2048, 
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1, 
        attention_dropout_rate: float = 0.0, 
        # Pre-instantiated MLX modules for subsampling and positional encoding
        subsampling_module: nn.Module, 
        pos_enc_module: nn.Module, # This is used by subsampling_module
        normalize_before: bool = True,
        # Conformer specific args
        macaron_style: bool = True,
        activation_type: str = "swish", 
        use_cnn_module: bool = True,
        cnn_module_kernel: int = 15,
        causal_cnn: bool = False, 
        key_bias_attention: bool = True, 
        layer_norm_eps: float = 1e-5, # Default for MLX LayerNorm, PyTorch Conformer uses 1e-12
        # Streaming related, passed to BaseEncoder
        static_chunk_size: int = -1,
        use_dynamic_chunk: bool = False,
        use_dynamic_left_chunk: bool = False,
    ):
        super().__init__(input_size=input_size, output_size=output_size, 
                         subsampling_module=subsampling_module,
                         pos_enc_module=pos_enc_module, # Base stores it, but embed (subsampling) uses it
                         normalize_before=normalize_before,
                         static_chunk_size=static_chunk_size,
                         use_dynamic_chunk=use_dynamic_chunk,
                         use_dynamic_left_chunk=use_dynamic_left_chunk)

        # activation_module mapping
        if activation_type == "swish":
            activation_module_instance = nn.SiLU()
        elif activation_type == "relu":
            activation_module_instance = nn.ReLU()
        elif activation_type == "mish":
            activation_module_instance = nn.Mish() 
        elif activation_type == "snake": # Assuming MLXSnake takes a default alpha or is configured elsewhere
            activation_module_instance = MLXSnake() 
        else:
            raise ValueError(f"Unsupported activation type: {activation_type}")

        self.encoders = []
        for i in range(num_blocks):
            attn_module = MLXRelPositionMultiHeadedAttention(
                n_head=attention_heads, n_feat=output_size, 
                dropout_rate=attention_dropout_rate, key_bias=key_bias_attention
            )
            
            ffn_module = MLXPositionwiseFeedForward(
                idim=output_size, hidden_units=linear_units, 
                dropout_rate=dropout_rate, activation_module=activation_module_instance
            )
            
            macaron_ffn_module = None
            if macaron_style:
                macaron_ffn_module = MLXPositionwiseFeedForward(
                    idim=output_size, hidden_units=linear_units, 
                    dropout_rate=dropout_rate, activation_module=activation_module_instance
                )
            
            conv_module_instance = None
            if use_cnn_module:
                conv_module_instance = MLXConvolutionModule(
                    channels=output_size, kernel_size=cnn_module_kernel,
                    activation_module=activation_module_instance, 
                    norm="batch_norm", # Default in ESPnet ConformerEncoderLayer
                    causal=causal_cnn,
                    bias=True 
                )

            layer = MLXConformerEncoderLayer(
                size=output_size,
                mlx_self_attn_module=attn_module,
                mlx_feed_forward_module=ffn_module,
                mlx_feed_forward_macaron_module=macaron_ffn_module,
                mlx_conv_module=conv_module_instance,
                dropout_rate=dropout_rate,
                normalize_before=normalize_before,
                layer_norm_eps=layer_norm_eps
            )
            self.encoders.append(layer)

    def forward_layers(self, xs: mx.array, chunk_masks: mx.array, 
                       pos_emb: mx.array, mask_pad: mx.array,
                       att_cache: Optional[mx.array] = None, # For streaming
                       cnn_cache: Optional[mx.array] = None  # For streaming
    ) -> Tuple[mx.array, Optional[mx.array], Optional[mx.array]]:
        
        current_att_cache_list = [None] * len(self.encoders)
        current_cnn_cache_list = [None] * len(self.encoders)

        if att_cache is not None and att_cache.size > 0:
            # att_cache for multiple layers: (num_layers, B, H, T_cache, D_k*2)
            # Or a list of caches. Assuming it's a list for MLX.
            if isinstance(att_cache, list) and len(att_cache) == len(self.encoders):
                current_att_cache_list = att_cache
            # else: error or handle single tensor cache split across layers if that's the format

        if cnn_cache is not None and cnn_cache.size > 0 :
            if isinstance(cnn_cache, list) and len(cnn_cache) == len(self.encoders):
                current_cnn_cache_list = cnn_cache

        next_att_cache_list = []
        next_cnn_cache_list = []

        for i, layer in enumerate(self.encoders):
            xs, _, layer_att_cache, layer_cnn_cache = layer(
                x=xs, 
                mask=chunk_masks, 
                pos_emb=pos_emb,  
                mask_pad=mask_pad, 
                att_cache=current_att_cache_list[i], 
                cnn_cache=current_cnn_cache_list[i]  
            )
            next_att_cache_list.append(layer_att_cache)
            next_cnn_cache_list.append(layer_cnn_cache)
        
        # For PoC, how to return list of caches? Or a stacked tensor if uniform shape?
        # Let's return lists of caches for now.
        return xs, next_att_cache_list, next_cnn_cache_list


# Example Usage (can be kept for testing or removed for production code):
if __name__ == '__main__':
    print("--- MLX Encoder Modules Smoke Test ---")

    # Common parameters
    input_dim = 80
    output_dim_encoder = 64 # Conformer hidden size
    attention_heads_enc = 4
    ffn_linear_units_enc = 128
    num_blocks_enc = 2 # Small number of blocks for testing
    dropout_enc = 0.1
    pos_dropout_enc = 0.1
    attn_dropout_enc = 0.0
    cnn_kernel_enc = 7

    # 1. Create Positional Encoding module instance
    # Max len for PE should be large enough for subsampled sequence
    # If subsampling by 4, max_len_pe = max_input_seq_len / 4
    max_input_seq_len = 200
    # For MLXEspnetRelPositionalEncoding, d_model is output_dim_encoder
    pe_module_instance = MLXEspnetRelPositionalEncoding(d_model=output_dim_encoder, dropout_rate=pos_dropout_enc, max_len=max_input_seq_len // 1 + 10) # Adjusted max_len

    # 2. Create Subsampling module instance (using LinearNoSubsampling for this test)
    # It needs the PE module if it's designed to use it internally (ESPnet style)
    # Our MLXLinearNoSubsampling takes the PE module and applies it.
    subsampling_module_instance = MLXLinearNoSubsampling(
        idim=input_dim, odim=output_dim_encoder, 
        dropout_rate=dropout_enc, mlx_pos_enc_module=pe_module_instance
    )
    mx.eval(subsampling_module_instance.parameters())


    # 3. Instantiate MLXConformerEncoder
    print("\nInstantiating MLXConformerEncoder...")
    conformer_encoder = MLXConformerEncoder(
        input_size=input_dim, # This is input to subsampling
        output_size=output_dim_encoder,
        attention_heads=attention_heads_enc,
        linear_units=ffn_linear_units_enc,
        num_blocks=num_blocks_enc,
        dropout_rate=dropout_enc,
        positional_dropout_rate=pos_dropout_enc, # Used by PE module
        attention_dropout_rate=attn_dropout_enc,
        subsampling_module=subsampling_module_instance,
        pos_enc_module=pe_module_instance, # Passed to BaseEncoder, but used by subsampling_module
        normalize_before=True,
        macaron_style=True,
        activation_type="swish",
        use_cnn_module=True,
        cnn_module_kernel=cnn_kernel_enc,
        causal_cnn=False,
        key_bias_attention=True,
        layer_norm_eps=1e-5,
        static_chunk_size=-1,
        use_dynamic_chunk=False,
        use_dynamic_left_chunk=False
    )
    mx.eval(conformer_encoder.parameters())
    print("MLXConformerEncoder instantiated.")

    # 4. Create dummy input for forward pass
    batch_size_test_enc = 2
    current_seq_len_test_enc = 60 # T_in
    input_xs = mx.random.normal((batch_size_test_enc, current_seq_len_test_enc, input_dim))
    input_xs_lens = mx.array([current_seq_len_test_enc, current_seq_len_test_enc - 10]) # Example lengths

    print(f"\nInput xs shape: {input_xs.shape}, xs_lens: {input_xs_lens.tolist()}")

    # 5. Test __call__ (non-streaming forward)
    output_xs, output_mask = conformer_encoder(input_xs, input_xs_lens)
    mx.eval(output_xs, output_mask)
    
    print(f"Output xs shape from __call__: {output_xs.shape}")
    print(f"Output mask shape from __call__: {output_mask.shape}")

    # Check output shapes
    # Subsampling (LinearNoSubsampling) does not change sequence length.
    # So, output seq len should be same as input seq len for this subsampling type.
    # If MLXConv2dSubsampling4 was used, seq len would be reduced by factor of 4.
    expected_seq_len_out = current_seq_len_test_enc # For LinearNoSubsampling
    assert output_xs.shape == (batch_size_test_enc, expected_seq_len_out, output_dim_encoder)
    assert output_mask.shape == (batch_size_test_enc, 1, expected_seq_len_out)
    print("MLXConformerEncoder __call__ test passed.")

    # 6. Test forward_chunk (stubbed, should raise NotImplementedError)
    print("\nTesting forward_chunk (expecting NotImplementedError)...")
    try:
        # Dummy args for forward_chunk
        chunk_xs = mx.random.normal((batch_size_test_enc, 16, input_dim)) # e.g. chunk size 16
        offset_val = 0
        req_cache = 0
        # Dummy caches (list of Nones for num_blocks)
        att_cache_list_dummy = [None] * num_blocks_enc
        cnn_cache_list_dummy = [None] * num_blocks_enc
        
        _ = conformer_encoder.forward_chunk(chunk_xs, offset_val, req_cache, att_cache_list_dummy, cnn_cache_list_dummy)
    except NotImplementedError:
        print("forward_chunk correctly raised NotImplementedError.")
    except Exception as e:
        print(f"forward_chunk test failed with unexpected error: {e}")


    print("\n--- End of MLX Encoder Modules Smoke Test ---")
