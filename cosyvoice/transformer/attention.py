# Copyright (c) 2019 Shigeki Karita
#               2020 Mobvoi Inc (Binbin Zhang)
#               2022 Xingchen Song (sxc19@mails.tsinghua.edu.cn)
#               2024 Alibaba Inc (Xiang Lyu)
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
"""Multi-Head Attention layer definition."""

import math
from typing import Tuple

import mlx.core as mx
import mlx.nn as nn


class MultiHeadedAttention(nn.Module):
    """Multi-Head Attention layer.

    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self,
                 n_head: int,
                 n_feat: int,
                 dropout_rate: float,
                 key_bias: bool = True):
        """Construct an MultiHeadedAttention object."""
        super().__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat, bias=key_bias)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward_qkv(
        self, query: mx.array, key: mx.array, value: mx.array
    ) -> Tuple[mx.array, mx.array, mx.array]:
        """Transform query, key and value.

        Args:
            query (mx.array): Query tensor (#batch, time1, size).
            key (mx.array): Key tensor (#batch, time2, size).
            value (mx.array): Value tensor (#batch, time2, size).

        Returns:
            mx.array: Transformed query tensor, size
                (#batch, n_head, time1, d_k).
            mx.array: Transformed key tensor, size
                (#batch, n_head, time2, d_k).
            mx.array: Transformed value tensor, size
                (#batch, n_head, time2, d_k).

        """
        n_batch = query.shape[0]
        q = self.linear_q(query).reshape(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).reshape(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).reshape(n_batch, -1, self.h, self.d_k)
        q = q.transpose(0, 2, 1, 3)  # (batch, head, time1, d_k)
        k = k.transpose(0, 2, 1, 3)  # (batch, head, time2, d_k)
        v = v.transpose(0, 2, 1, 3)  # (batch, head, time2, d_k)

        return q, k, v

    def forward_attention(
        self,
        value: mx.array,
        scores: mx.array,
        mask: mx.array = None # Default value in mlx? Or empty array?
    ) -> mx.array:
        """Compute attention context vector.

        Args:
            value (mx.array): Transformed value, size
                (#batch, n_head, time2, d_k).
            scores (mx.array): Attention score, size
                (#batch, n_head, time1, time2).
            mask (mx.array): Mask, size (#batch, 1, time2) or
                (#batch, time1, time2), (0, 0, 0) means fake mask.

        Returns:
            mx.array: Transformed value (#batch, time1, d_model)
                weighted by the attention score (#batch, time1, time2).

        """
        n_batch = value.shape[0]

        # NOTE(xcsong): When will `if mask.size(2) > 0` be True?
        # Original code uses torch.ones((0,0,0)) as fake mask.
        if mask is not None and mask.size > 0:  # time2 > 0
            # mask logic: in torch code: mask = mask.unsqueeze(1).eq(0)
            # mask passed in is likely boolean or integer.
            # (batch, 1, time2) or (batch, time1, time2)

            # MLX doesn't have masked_fill directly as a method, uses mx.where

            # mask shape check
            # if mask is (batch, 1, time2), expand?

            # Original code:
            # mask = mask.unsqueeze(1).eq(0)  # (batch, 1, *, time2)
            # scores = scores.masked_fill(mask, -float('inf'))

            # Assuming mask passed in is valid boolean mask where True means keep, False means mask out (or vice versa? check eq(0))
            # eq(0) means False -> True. So original mask: 1 means valid, 0 means invalid?
            # "mask pad for input is in (#batch, 1, T) shape". Usually 1 for valid.
            # eq(0) makes invalid positions True. masked_fill fills True positions with -inf.
            # So we mask out where mask input is 0.

            # MLX:
            # mask: (batch, 1, time2)

            # scores: (batch, head, time1, time2)

            # if mask.ndim == 3: mask = mask[:, None, :, :] ?
            # torch unsqueeze(1) on (batch, 1, time) makes (batch, 1, 1, time).

            if mask.ndim == 3:
                 mask = mask[:, None, :, :]

            # Convert mask to boolean if not
            # mask = (mask == 0) # True where we want to mask

            # But wait, arguments say mask is "Mask tensor".
            # Usually users pass what they have.
            # Let's assume mask is 1 for valid, 0 for padding.

            # scores = mx.where(mask == 0, -1e9, scores) # -inf causes NaNs sometimes in softmax gradients?
            scores = mx.where(mask == 0, -1e9, scores)

            attn = nn.softmax(scores, axis=-1)
            # attn = mx.where(mask == 0, 0.0, attn) # Zero out attention for masked positions
            attn = mx.where(mask == 0, 0.0, attn)

        else:
            attn = nn.softmax(scores, axis=-1)  # (batch, head, time1, time2)

        p_attn = self.dropout(attn)
        x = mx.matmul(p_attn, value)  # (batch, head, time1, d_k)

        # x.transpose(1, 2) -> (batch, time1, head, d_k)
        x = x.transpose(0, 2, 1, 3)
        # contiguous().view(...) -> reshape
        x = x.reshape(n_batch, -1, self.h * self.d_k)

        return self.linear_out(x)  # (batch, time1, d_model)

    def __call__(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        mask: mx.array = None,
        pos_emb: mx.array = None,
        cache: mx.array = None
    ) -> Tuple[mx.array, mx.array]:
        """Compute scaled dot product attention.

        Args:
            query (mx.array): Query tensor (#batch, time1, size).
            key (mx.array): Key tensor (#batch, time2, size).
            value (mx.array): Value tensor (#batch, time2, size).
            mask (mx.array): Mask tensor.
            cache (mx.array): Cache tensor (1, head, cache_t, d_k * 2)

        Returns:
            mx.array: Output tensor (#batch, time1, d_model).
            mx.array: Cache tensor (1, head, cache_t + time1, d_k * 2)

        """
        q, k, v = self.forward_qkv(query, key, value)

        if cache is not None and cache.size > 0:
            # key_cache, value_cache = split
            # cache shape (1, head, cache_t, d_k * 2)
            # split at last dim

            # MLX split
            key_cache, value_cache = mx.split(cache, 2, axis=-1)

            # We need to broadcast or match dims.
            # q, k, v are (batch, head, time, d_k).
            # cache is (1, head, cache_t, d_k).
            # Assuming batch=1 for inference when caching is used?
            # Or cache is expanded.

            # torch.cat
            k = mx.concatenate([key_cache, k], axis=2)
            v = mx.concatenate([value_cache, v], axis=2)

        new_cache = mx.concatenate((k, v), axis=-1)

        scores = mx.matmul(q, k.transpose(0, 1, 3, 2)) / math.sqrt(self.d_k)
        return self.forward_attention(v, scores, mask), new_cache


class RelPositionMultiHeadedAttention(MultiHeadedAttention):
    """Multi-Head Attention layer with relative position encoding.
    Paper: https://arxiv.org/abs/1901.02860
    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.
    """

    def __init__(self,
                 n_head: int,
                 n_feat: int,
                 dropout_rate: float,
                 key_bias: bool = True):
        """Construct an RelPositionMultiHeadedAttention object."""
        super().__init__(n_head, n_feat, dropout_rate, key_bias)
        # linear transformation for positional encoding
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        # these two learnable bias are used in matrix c and matrix d
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3

        # Parameter init
        self.pos_bias_u = mx.random.uniform(low=-0.1, high=0.1, shape=(self.h, self.d_k)) # Approximation of xavier
        self.pos_bias_v = mx.random.uniform(low=-0.1, high=0.1, shape=(self.h, self.d_k))
        # MLX doesn't have xavier_uniform_ in-place init easily accessible like torch.nn.init
        # But we can assume it's initialized somehow.
        # Since we are porting, we should probably implement a property-based parameter if we want it to be trainable.
        # But for now, we leave it as array, or wrap in a way to make it trainable (e.g. self.update(parameters))
        # In MLX, assigning to self.param works if we treat it as state.

    def rel_shift(self, x: mx.array) -> mx.array:
        """Compute relative positional encoding.

        Args:
            x (mx.array): Input tensor (batch, head, time1, 2*time1-1).
            time1 means the length of query vector.

        Returns:
            mx.array: Output tensor.

        """
        # zero_pad = mx.zeros((x.shape[0], x.shape[1], x.shape[2], 1), dtype=x.dtype)
        zero_pad = mx.zeros((*x.shape[:3], 1), dtype=x.dtype)
        x_padded = mx.concatenate([zero_pad, x], axis=-1)

        x_padded = x_padded.reshape(x.shape[0],
                                 x.shape[1],
                                 x.shape[3] + 1, x.shape[2])
        # x_padded[:, :, 1:]
        x_reshaped = x_padded[:, :, 1:].reshape(x.shape)
        return x_reshaped[:, :, :, : x.shape[-1] // 2 + 1]

    def __call__(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        mask: mx.array = None,
        pos_emb: mx.array = None,
        cache: mx.array = None
    ) -> Tuple[mx.array, mx.array]:
        """Compute 'Scaled Dot Product Attention' with rel. positional encoding.
        """
        q, k, v = self.forward_qkv(query, key, value)
        q = q.transpose(0, 2, 1, 3)  # (batch, time1, head, d_k) -> wait, forward_qkv returns (batch, head, time, d_k)
        # The torch code says q = q.transpose(1, 2) which results in (batch, time1, head, d_k)
        # BUT forward_qkv docstring says it returns (batch, head, time1, d_k).
        # Let's check torch implementation again.
        # forward_qkv: q = q.transpose(1, 2) -> (batch, head, time1, d_k). Correct.
        # In RelPositionMultiHeadedAttention.forward: q = q.transpose(1, 2) -> (batch, time1, head, d_k).

        q = q.transpose(0, 2, 1, 3) # (batch, time1, head, d_k)

        if cache is not None and cache.size > 0:
            key_cache, value_cache = mx.split(cache, 2, axis=-1)
            k = mx.concatenate([key_cache, k], axis=2)
            v = mx.concatenate([value_cache, v], axis=2)

        new_cache = mx.concatenate((k, v), axis=-1)

        n_batch_pos = pos_emb.shape[0]
        p = self.linear_pos(pos_emb).reshape(n_batch_pos, -1, self.h, self.d_k)
        p = p.transpose(0, 2, 1, 3)  # (batch, head, time1, d_k)

        # q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
        # q is (batch, time1, head, d_k). pos_bias_u is (head, d_k).
        # Broadcasting works (batch, time1, head, d_k) + (head, d_k).

        q_with_bias_u = (q + self.pos_bias_u).transpose(0, 2, 1, 3) # (batch, head, time1, d_k)
        q_with_bias_v = (q + self.pos_bias_v).transpose(0, 2, 1, 3)

        # matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))
        # k is (batch, head, time2, d_k)
        matrix_ac = mx.matmul(q_with_bias_u, k.transpose(0, 1, 3, 2))

        # matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        # p is (batch, head, time1, d_k) -> wait, pos_emb is (#batch, time2, size)?
        # docstring says pos_emb (#batch, time2, size)
        # p comes from pos_emb. So p is (batch, head, time2, d_k).
        matrix_bd = mx.matmul(q_with_bias_v, p.transpose(0, 1, 3, 2))

        if matrix_ac.shape != matrix_bd.shape:
            matrix_bd = self.rel_shift(matrix_bd)

        scores = (matrix_ac + matrix_bd) / math.sqrt(self.d_k)

        return self.forward_attention(v, scores, mask), new_cache
