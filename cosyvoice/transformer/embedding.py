# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang, Di Wu)
#               2024 Alibaba Inc (Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified from ESPnet(https://github.com/espnet/espnet)
"""Positonal Encoding Module."""

import math
from typing import Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np


class PositionalEncoding(nn.Module):
    """Positional encoding.

    :param int d_model: embedding dim
    :param float dropout_rate: dropout rate
    :param int max_len: maximum input length

    PE(pos, 2i)   = sin(pos/(10000^(2i/dmodel)))
    PE(pos, 2i+1) = cos(pos/(10000^(2i/dmodel)))
    """

    def __init__(self,
                 d_model: int,
                 dropout_rate: float,
                 max_len: int = 5000,
                 reverse: bool = False):
        """Construct an PositionalEncoding object."""
        super().__init__()
        self.d_model = d_model
        self.xscale = math.sqrt(self.d_model)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.max_len = max_len

        pe = mx.zeros((self.max_len, self.d_model))
        position = mx.arange(0, self.max_len, dtype=mx.float32)[:, None]
        div_term = mx.exp(
            mx.arange(0, self.d_model, 2, dtype=mx.float32) *
            -(math.log(10000.0) / self.d_model))

        # MLX slicing and assignment
        # pe[:, 0::2] = mx.sin(position * div_term)
        # pe[:, 1::2] = mx.cos(position * div_term)
        # Using concat instead of slicing assignment which might be cleaner/more MLX idiomatic if full tensor replacement is needed
        # But slicing assignment works in MLX

        sin_part = mx.sin(position * div_term)
        cos_part = mx.cos(position * div_term)

        # Interleave
        # This part is tricky to do efficiently without mutation.
        # But we can just use numpy for initialization since it is constant.
        import numpy as np
        pe_np = np.zeros((self.max_len, self.d_model), dtype=np.float32)
        position_np = np.arange(0, self.max_len, dtype=np.float32)[:, None]
        div_term_np = np.exp(
            np.arange(0, self.d_model, 2, dtype=np.float32) *
            -(math.log(10000.0) / self.d_model))
        pe_np[:, 0::2] = np.sin(position_np * div_term_np)
        pe_np[:, 1::2] = np.cos(position_np * div_term_np)

        self._pe = mx.array(pe_np)[None, :, :] # (1, max_len, d_model)

    def __call__(self,
                x: mx.array,
                offset: Union[int, mx.array] = 0) \
            -> Tuple[mx.array, mx.array]:
        """Add positional encoding.

        Args:
            x (mx.array): Input. Its shape is (batch, time, ...)
            offset (int, mx.array): position offset

        Returns:
            mx.array: Encoded tensor. Its shape is (batch, time, ...)
            mx.array: for compatibility to RelPositionalEncoding
        """

        pos_emb = self.position_encoding(offset, x.shape[1], False)
        x = x * self.xscale + pos_emb
        return self.dropout(x), self.dropout(pos_emb)

    def position_encoding(self,
                          offset: Union[int, mx.array],
                          size: int,
                          apply_dropout: bool = True) -> mx.array:
        """ For getting encoding in a streaming fashion

        Attention!!!!!
        we apply dropout only once at the whole utterance level in a none
        streaming way, but will call this function several times with
        increasing input size in a streaming scenario, so the dropout will
        be applied several times.

        Args:
            offset (int or mx.array): start offset
            size (int): required size of position encoding

        Returns:
            mx.array: Corresponding encoding
        """
        if isinstance(offset, int):
            assert offset + size <= self.max_len
            pos_emb = self._pe[:, offset:offset + size]
        elif isinstance(offset, mx.array) and offset.ndim == 0:  # scalar
            offset_val = int(offset.item())
            assert offset_val + size <= self.max_len
            pos_emb = self._pe[:, offset_val:offset_val + size]
        else:  # for batched streaming decoding
            # MLX doesn't support advanced indexing like PyTorch yet in all cases,
            # but basic gathering might work.
            # However, offset is usually a scalar in inference or a tensor of start indices.

            # offset: (B,)
            # We need (B, T, D)

            # index = offset.unsqueeze(1) + torch.arange(0, size)
            # F.embedding(index, self.pe[0])

            # In MLX, we can use take/gather.
            # self.pe[0] is (max_len, d_model)

            # B = offset.shape[0]
            # indices = offset[:, None] + mx.arange(0, size)[None, :] # (B, T)

            # We need to clamp or mask negative offsets?
            # Original code: index = index * flag (flag = index > 0)

            indices = offset[:, None] + mx.arange(0, size)[None, :] # (B, T)
            flag = indices > 0
            indices = indices * flag

            # Gather
            # self._pe[0] shape (max_len, d_model)
            # indices shape (B, T)
            # output (B, T, d_model)
            pos_emb = self._pe[0][indices] # MLX supports this kind of indexing? Yes.

        if apply_dropout:
            pos_emb = self.dropout(pos_emb)
        return pos_emb


class RelPositionalEncoding(PositionalEncoding):
    """Relative positional encoding module.
    See : Appendix B in https://arxiv.org/abs/1901.02860
    Args:
        d_model (int): Embedding dimension.
        dropout_rate (float): Dropout rate.
        max_len (int): Maximum input length.
    """

    def __init__(self, d_model: int, dropout_rate: float, max_len: int = 5000):
        """Initialize class."""
        super().__init__(d_model, dropout_rate, max_len, reverse=True)

    def __call__(self,
                x: mx.array,
                offset: Union[int, mx.array] = 0) \
            -> Tuple[mx.array, mx.array]:
        """Compute positional encoding.
        Args:
            x (mx.array): Input tensor (batch, time, `*`).
        Returns:
            mx.array: Encoded tensor (batch, time, `*`).
            mx.array: Positional embedding tensor (1, time, `*`).
        """
        x = x * self.xscale
        pos_emb = self.position_encoding(offset, x.shape[1], False)
        return self.dropout(x), self.dropout(pos_emb)


class WhisperPositionalEncoding(PositionalEncoding):
    """ Sinusoids position encoding used in openai-whisper.encoder
    """

    def __init__(self, d_model: int, dropout_rate: float, max_len: int = 1500):
        super().__init__(d_model, dropout_rate, max_len)
        self.xscale = 1.0
        log_timescale_increment = np.log(10000) / (d_model // 2 - 1)
        inv_timescales = np.exp(-log_timescale_increment *
                                   np.arange(d_model // 2))
        scaled_time = np.arange(max_len)[:, np.newaxis] * \
            inv_timescales[np.newaxis, :]
        pe = np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=1)

        self._pe = mx.array(pe)[None, :, :]


class LearnablePositionalEncoding(PositionalEncoding):
    """ Learnable position encoding used in openai-whisper.decoder
    """

    def __init__(self, d_model: int, dropout_rate: float, max_len: int = 448):
        super().__init__(d_model, dropout_rate, max_len)
        # NOTE(xcsong): overwrite self.pe & self.xscale
        self.pe = mx.zeros((1, max_len, d_model)) # Parameter usually, but let's initialize it.
        # In MLX, we need to declare it as a parameter if it is trainable.
        # Since it is learnable, we should.
        # But PositionalEncoding parent class doesn't use self.pe as a parameter in this way.

        # We can just assign it.
        # self.pe = nn.Parameter(...)
        # Wait, parent uses self._pe (renamed to avoid conflict if I used self.pe)
        # Original code used self.pe which was a buffer (non-trainable) in parent, but Parameter here.

        self._pe = mx.zeros((1, max_len, d_model)) # We will treat this as a parameter.
        self.xscale = 1.0


class NoPositionalEncoding(nn.Module):
    """ No position encoding
    """

    def __init__(self, d_model: int, dropout_rate: float):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout_rate)

    def __call__(self,
                x: mx.array,
                offset: Union[int, mx.array] = 0) \
            -> Tuple[mx.array, mx.array]:
        """ Just return zero vector for interface compatibility
        """
        pos_emb = mx.zeros((1, x.shape[1], self.d_model))
        return self.dropout(x), pos_emb

    def position_encoding(self, offset: Union[int, mx.array],
                          size: int) -> mx.array:
        return mx.zeros((1, size, self.d_model))


class EspnetRelPositionalEncoding(nn.Module):
    """Relative positional encoding module (new implementation).

    Details can be found in https://github.com/espnet/espnet/pull/2816.

    See : Appendix B in https://arxiv.org/abs/1901.02860

    Args:
        d_model (int): Embedding dimension.
        dropout_rate (float): Dropout rate.
        max_len (int): Maximum input length.

    """

    def __init__(self, d_model: int, dropout_rate: float, max_len: int = 5000):
        """Construct an PositionalEncoding object."""
        super(EspnetRelPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.xscale = math.sqrt(self.d_model)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.pe = None
        self.extend_pe(mx.array([0.0]).broadcast_to((1, max_len)))

    def extend_pe(self, x: mx.array):
        """Reset the positional encodings."""
        if self.pe is not None:
            # self.pe contains both positive and negative parts
            # the length of self.pe is 2 * input_len - 1
            if self.pe.shape[1] >= x.shape[1] * 2 - 1:
                return
        # Suppose `i` means to the position of query vecotr and `j` means the
        # position of key vector. We use position relative positions when keys
        # are to the left (i>j) and negative relative positions otherwise (i<j).

        import numpy as np
        length = x.shape[1]
        pe_positive = np.zeros((length, self.d_model), dtype=np.float32)
        pe_negative = np.zeros((length, self.d_model), dtype=np.float32)
        position = np.arange(0, length, dtype=np.float32)[:, None]
        div_term = np.exp(
            np.arange(0, self.d_model, 2, dtype=np.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        pe_positive[:, 0::2] = np.sin(position * div_term)
        pe_positive[:, 1::2] = np.cos(position * div_term)
        pe_negative[:, 0::2] = np.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = np.cos(-1 * position * div_term)

        # Reserve the order of positive indices and concat both positive and
        # negative indices. This is used to support the shifting trick
        # as in https://arxiv.org/abs/1901.02860
        pe_positive = np.flip(pe_positive, axis=0)[None, :, :]
        pe_negative = pe_negative[1:][None, :, :]
        pe = np.concatenate([pe_positive, pe_negative], axis=1)
        self.pe = mx.array(pe)

    def __call__(self, x: mx.array, offset: Union[int, mx.array] = 0) \
            -> Tuple[mx.array, mx.array]:
        """Add positional encoding.

        Args:
            x (mx.array): Input tensor (batch, time, `*`).

        Returns:
            mx.array: Encoded tensor (batch, time, `*`).

        """
        self.extend_pe(x)
        x = x * self.xscale
        pos_emb = self.position_encoding(size=x.shape[1], offset=offset)
        return self.dropout(x), self.dropout(pos_emb)

    def position_encoding(self,
                          offset: Union[int, mx.array],
                          size: int) -> mx.array:
        """ For getting encoding in a streaming fashion

        Attention!!!!!
        we apply dropout only once at the whole utterance level in a none
        streaming way, but will call this function several times with
        increasing input size in a streaming scenario, so the dropout will
        be applied several times.

        Args:
            offset (int or mx.array): start offset
            size (int): required size of position encoding

        Returns:
            mx.array: Corresponding encoding
        """
        if isinstance(offset, int):
            center = self.pe.shape[1] // 2
            start = center - size - offset + 1
            end = center + size + offset
            pos_emb = self.pe[:, start:end]
        elif isinstance(offset, mx.array):
             # This part seems to assume scalar offset or similar handling.
             # In PyTorch code it did slicing.
             # "elif isinstance(offset, torch.Tensor):" logic in original code was exactly same as int case.
             # It implies offset is scalar tensor.
             offset_val = int(offset.item())
             center = self.pe.shape[1] // 2
             start = center - size - offset_val + 1
             end = center + size + offset_val
             pos_emb = self.pe[:, start:end]
        return pos_emb
