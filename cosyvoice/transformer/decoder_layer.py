# Copyright (c) 2019 Shigeki Karita
#               2020 Mobvoi Inc (Binbin Zhang)
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
"""Decoder self-attention layer definition."""
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


class DecoderLayer(nn.Module):
    """Single decoder layer module.

    Args:
        size (int): Input dimension.
        self_attn (nn.Module): Self-attention module instance.
            `MultiHeadedAttention` instance can be used as the argument.
        src_attn (nn.Module): Inter-attention module instance.
            `MultiHeadedAttention` instance can be used as the argument.
            If `None` is passed, Inter-attention is not used, such as
            CIF, GPT, and other decoder only model.
        feed_forward (nn.Module): Feed-forward module instance.
            `PositionwiseFeedForward` instance can be used as the argument.
        dropout_rate (float): Dropout rate.
        normalize_before (bool):
            True: use layer_norm before each sub-block.
            False: to use layer_norm after each sub-block.
    """

    def __init__(
        self,
        size: int,
        self_attn: nn.Module,
        src_attn: Optional[nn.Module],
        feed_forward: nn.Module,
        dropout_rate: float,
        normalize_before: bool = True,
    ):
        """Construct an DecoderLayer object."""
        super().__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.norm1 = nn.LayerNorm(size, eps=1e-5)
        self.norm2 = nn.LayerNorm(size, eps=1e-5)
        self.norm3 = nn.LayerNorm(size, eps=1e-5)
        self.dropout = nn.Dropout(dropout_rate)
        self.normalize_before = normalize_before

    def __call__(
        self,
        tgt: mx.array,
        tgt_mask: mx.array,
        memory: mx.array,
        memory_mask: mx.array,
        cache: Optional[mx.array] = None
    ) -> Tuple[mx.array, mx.array, mx.array, mx.array]:
        """Compute decoded features.

        Args:
            tgt (mx.array): Input tensor (#batch, maxlen_out, size).
            tgt_mask (mx.array): Mask for input tensor
                (#batch, maxlen_out).
            memory (mx.array): Encoded memory
                (#batch, maxlen_in, size).
            memory_mask (mx.array): Encoded memory mask
                (#batch, maxlen_in).
            cache (mx.array): cached tensors.
                (#batch, maxlen_out - 1, size).

        Returns:
            mx.array: Output tensor (#batch, maxlen_out, size).
            mx.array: Mask for output tensor (#batch, maxlen_out).
            mx.array: Encoded memory (#batch, maxlen_in, size).
            mx.array: Encoded memory mask (#batch, maxlen_in).

        """
        residual = tgt
        if self.normalize_before:
            tgt = self.norm1(tgt)

        if cache is None:
            tgt_q = tgt
            tgt_q_mask = tgt_mask
        else:
            # compute only the last frame query keeping dim: max_time_out -> 1
            # assert cache.shape == (
            #     tgt.shape[0],
            #     tgt.shape[1] - 1,
            #     self.size,
            # ), "{cache.shape} == {(tgt.shape[0], tgt.shape[1] - 1, self.size)}"
            tgt_q = tgt[:, -1:, :]
            residual = residual[:, -1:, :]
            if tgt_mask is not None:
                tgt_q_mask = tgt_mask[:, -1:, :]
            else:
                tgt_q_mask = None

        # self_attn returns (output, cache) usually.
        # But wait, original code calls `self.self_attn(..., cache=att_cache)[0]`?
        # The DecoderLayer code in torch:
        # x = residual + self.dropout(self.self_attn(tgt_q, tgt, tgt, tgt_q_mask)[0])

        # self_attn here is MultiHeadedAttention.
        # It returns (output, new_cache). We take [0].

        # Also note self_attn expects (query, key, value, mask).

        x_att, _ = self.self_attn(tgt_q, tgt, tgt, tgt_q_mask)
        x = residual + self.dropout(x_att)
        if not self.normalize_before:
            x = self.norm1(x)

        if self.src_attn is not None:
            residual = x
            if self.normalize_before:
                x = self.norm2(x)

            x_att, _ = self.src_attn(x, memory, memory, memory_mask)
            x = residual + self.dropout(x_att)

            if not self.normalize_before:
                x = self.norm2(x)

        residual = x
        if self.normalize_before:
            x = self.norm3(x)
        x = residual + self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm3(x)

        if cache is not None:
            x = mx.concatenate([cache, x], axis=1)

        return x, tgt_mask, memory, memory_mask
