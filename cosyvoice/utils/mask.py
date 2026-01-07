# Copyright (c) 2019 Shigeki Karita
#               2020 Mobvoi Inc (Binbin Zhang)
#               2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import mlx.core as mx
import numpy as np

def subsequent_mask(
        size: int,
        device: str = "cpu", # ignored in MLX
) -> mx.array:
    """Create mask for subsequent steps (size, size).
    """
    arange = mx.arange(size)
    # mask = arange.expand(size, size) # MLX broadcast automatically or need explicit broadcast_to
    # arange: (size,)
    # arange.unsqueeze(-1) -> (size, 1)

    # We want mask[i, j] = True if j <= i

    # row indices: i (size, 1)
    # col indices: j (1, size)

    i = arange[:, None]
    j = arange[None, :]

    mask = j <= i
    return mask


def subsequent_chunk_mask(
        size: int,
        chunk_size: int,
        num_left_chunks: int = -1,
        device: str = "cpu",
) -> mx.array:
    """Create mask for subsequent steps (size, size) with chunk size,
       this is for streaming encoder
    """
    # NOTE this modified implementation meets onnx export requirements, but it doesn't support num_left_chunks
    # Porting exactly as is from PyTorch version in the file.

    pos_idx = mx.arange(size)
    # block_value = (torch.div(pos_idx, chunk_size, rounding_mode='trunc') + 1) * chunk_size
    block_value = ((pos_idx // chunk_size) + 1) * chunk_size

    # ret = pos_idx.unsqueeze(0) < block_value.unsqueeze(1)
    # pos_idx (1, size) < block_value (size, 1)

    ret = pos_idx[None, :] < block_value[:, None]
    return ret


def add_optional_chunk_mask(xs: mx.array,
                            masks: mx.array,
                            use_dynamic_chunk: bool,
                            use_dynamic_left_chunk: bool,
                            decoding_chunk_size: int,
                            static_chunk_size: int,
                            num_decoding_left_chunks: int,
                            enable_full_context: bool = True):
    """ Apply optional mask for encoder.
    """
    # Whether to use chunk mask or not
    if use_dynamic_chunk:
        max_len = xs.shape[1]
        if decoding_chunk_size < 0:
            chunk_size = max_len
            num_left_chunks = -1
        elif decoding_chunk_size > 0:
            chunk_size = decoding_chunk_size
            num_left_chunks = num_decoding_left_chunks
        else:
            # Random chunk size logic
            # Use numpy for random int generation as it's scalar logic
            rand_int = int(np.random.randint(1, max_len))
            chunk_size = rand_int
            num_left_chunks = -1
            if chunk_size > max_len // 2 and enable_full_context:
                chunk_size = max_len
            else:
                chunk_size = chunk_size % 25 + 1
                if use_dynamic_left_chunk:
                    max_left_chunks = (max_len - 1) // chunk_size
                    if max_left_chunks > 0:
                        num_left_chunks = int(np.random.randint(0, max_left_chunks))
                    else:
                        num_left_chunks = 0

        chunk_masks = subsequent_chunk_mask(xs.shape[1], chunk_size,
                                            num_left_chunks)  # (L, L)
        chunk_masks = chunk_masks[None, :, :]  # (1, L, L)

        # masks is (B, 1, L)
        # chunk_masks (1, L, L)
        # result (B, L, L)

        # MLX broadcast and
        # masks (B, 1, L) -> broadcast to (B, L, L)

        chunk_masks = masks & chunk_masks

    elif static_chunk_size > 0:
        num_left_chunks = num_decoding_left_chunks
        chunk_masks = subsequent_chunk_mask(xs.shape[1], static_chunk_size,
                                            num_left_chunks)  # (L, L)
        chunk_masks = chunk_masks[None, :, :]  # (1, L, L)
        chunk_masks = masks & chunk_masks  # (B, L, L)
    else:
        chunk_masks = masks

    # assert chunk_masks.dtype == mx.bool_ # or check

    # Sanity check logic
    # if (chunk_masks.sum(dim=-1) == 0).sum().item() != 0:
    #     print(...)
    #     chunk_masks[chunk_masks.sum(dim=-1) == 0] = True

    row_sums = mx.sum(chunk_masks, axis=-1)
    zero_rows_mask = (row_sums == 0)
    if mx.sum(zero_rows_mask).item() != 0:
        print('get chunk_masks all false at some timestep, force set to true, make sure they are masked in futuer computation!')
        # Force set diagonal to true where row sum is 0?
        # Original code: chunk_masks[chunk_masks.sum(dim=-1) == 0] = True
        # Sets entire row to True? Yes.

        # In MLX:
        # We want to set chunk_masks to True where zero_rows_mask is True.
        # chunk_masks = mx.where(zero_rows_mask[..., None], True, chunk_masks)

        # zero_rows_mask is (B, L). Expand to (B, L, L) (last dim broadcast)
        chunk_masks = mx.where(zero_rows_mask[..., None], True, chunk_masks)

    return chunk_masks


def make_pad_mask(lengths: mx.array, max_len: int = 0) -> mx.array:
    """Make mask tensor containing indices of padded part.
    Args:
        lengths (mx.array): Batch of lengths (B,).
    Returns:
        mx.array: Mask tensor containing indices of padded part. (B, T)
        True where padded.
    """
    batch_size = lengths.shape[0]
    if max_len == 0:
        max_len = int(lengths.max().item())

    seq_range = mx.arange(0, max_len, dtype=mx.int64)
    # seq_range (T,)

    # Expand
    seq_range_expand = seq_range[None, :] # (1, T)
    seq_length_expand = lengths[:, None] # (B, 1)

    mask = seq_range_expand >= seq_length_expand
    return mask
