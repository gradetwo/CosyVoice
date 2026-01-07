# Copyright (c) 2021 Mobvoi Inc (Binbin Zhang, Di Wu)
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
# Modified from ESPnet(https://github.com/espnet/espnet)
"""Subsampling layer definition."""

from typing import Tuple, Union

import mlx.core as mx
import mlx.nn as nn


class BaseSubsampling(nn.Module):

    def __init__(self):
        super().__init__()
        self.right_context = 0
        self.subsampling_rate = 1

    def position_encoding(self, offset: Union[int, mx.array],
                          size: int) -> mx.array:
        return self.pos_enc.position_encoding(offset, size)


class EmbedinigNoSubsampling(BaseSubsampling):
    """Embedding input without subsampling
    """

    def __init__(self, idim: int, odim: int, dropout_rate: float,
                 pos_enc_class: nn.Module):
        super().__init__()
        self.embed = nn.Embedding(idim, odim)
        self.pos_enc = pos_enc_class

    def __call__(
        self,
        x: mx.array,
        x_mask: mx.array,
        offset: Union[int, mx.array] = 0
    ) -> Tuple[mx.array, mx.array, mx.array]:
        """Input x.

        Args:
            x (mx.array): Input tensor (#batch, time, idim).
            x_mask (mx.array): Input mask (#batch, 1, time).

        Returns:
            mx.array: linear input tensor (#batch, time', odim),
                where time' = time .
            mx.array: linear input mask (#batch, 1, time'),
                where time' = time .

        """
        x = self.embed(x)
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask


class LinearNoSubsampling(BaseSubsampling):
    """Linear transform the input without subsampling

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, idim: int, odim: int, dropout_rate: float,
                 pos_enc_class: nn.Module):
        """Construct an linear object."""
        super().__init__()
        self.out = nn.Sequential(
            nn.Linear(idim, odim),
            nn.LayerNorm(odim, eps=1e-5),
            nn.Dropout(dropout_rate),
        )
        self.pos_enc = pos_enc_class
        self.right_context = 0
        self.subsampling_rate = 1

    def __call__(
        self,
        x: mx.array,
        x_mask: mx.array,
        offset: Union[int, mx.array] = 0
    ) -> Tuple[mx.array, mx.array, mx.array]:
        """Input x.

        Args:
            x (mx.array): Input tensor (#batch, time, idim).
            x_mask (mx.array): Input mask (#batch, 1, time).

        Returns:
            mx.array: linear input tensor (#batch, time', odim),
                where time' = time .
            mx.array: linear input mask (#batch, 1, time'),
                where time' = time .

        """
        x = self.out(x)
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask


class Conv1dSubsampling2(BaseSubsampling):
    """Convolutional 1D subsampling (to 1/2 length).
       It is designed for Whisper, ref:
       https://github.com/openai/whisper/blob/main/whisper/model.py

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, idim: int, odim: int, dropout_rate: float,
                 pos_enc_class: nn.Module):
        """Construct an Conv1dSubsampling2 object."""
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(idim, odim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(odim, odim, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )
        self.pos_enc = pos_enc_class
        self.subsampling_rate = 2
        self.right_context = 4

    def __call__(
        self,
        x: mx.array,
        x_mask: mx.array,
        offset: Union[int, mx.array] = 0
    ) -> Tuple[mx.array, mx.array, mx.array]:
        """Subsample x.

        Args:
            x (mx.array): Input tensor (#batch, time, idim).
            x_mask (mx.array): Input mask (#batch, 1, time).

        Returns:
            mx.array: Subsampled tensor (#batch, time', odim),
                where time' = time // 2.
            mx.array: Subsampled mask (#batch, 1, time'),
                where time' = time // 2.
            mx.array: positional encoding

        """
        time = x.shape[1]
        # x = x.transpose(1, 2)  # (b, f, t)
        # MLX Conv1d expects (N, T, C). No transpose needed if input is (N, T, D).
        # But wait, original code:
        # x input is (batch, time, idim).
        # x.transpose(1, 2) -> (batch, idim, time).
        # nn.Conv1d(idim, odim, ...) -> input channels=idim.

        # In MLX Conv1d: input (N, L, C_in).
        # x is (N, T, idim). So it matches.

        x = self.conv(x)

        # x = x.transpose(1, 2)  # (b, t, f) -> Not needed in MLX as output is (N, L, C_out).

        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask[:, :, (time + 1) % 2::2]


class Conv2dSubsampling4(BaseSubsampling):
    """Convolutional 2D subsampling (to 1/4 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, idim: int, odim: int, dropout_rate: float,
                 pos_enc_class: nn.Module):
        """Construct an Conv2dSubsampling4 object."""
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, odim, 3, stride=2), # stride=(2,2)? torch default is stride=1? No, 3, 2 means kernel=3, stride=2.
            nn.ReLU(),
            nn.Conv2d(odim, odim, 3, stride=2),
            nn.ReLU(),
        )
        # Output dim calc:
        # Input (B, 1, T, idim).
        # Conv1: stride 2.
        # Conv2: stride 2.
        # Total stride 4.
        # Feature dim (idim) also reduced by 4 (approx).
        # We flatten feature dim.

        self.out = nn.Sequential(
            nn.Linear(odim * (((idim - 1) // 2 - 1) // 2), odim))

        self.pos_enc = pos_enc_class
        self.subsampling_rate = 4
        self.right_context = 6

    def __call__(
        self,
        x: mx.array,
        x_mask: mx.array,
        offset: Union[int, mx.array] = 0
    ) -> Tuple[mx.array, mx.array, mx.array]:
        """Subsample x.
        """
        # x: (B, T, idim)
        # MLX Conv2d expects (N, H, W, C).
        # Torch Conv2d expects (N, C, H, W).

        # Torch code: x = x.unsqueeze(1) -> (B, 1, T, idim) (N, C, H, W) where H=T, W=idim.

        # MLX: (B, T, idim, 1).
        x = x[:, :, :, None]

        # We need to map torch logic to MLX logic.
        # Torch: Conv2d(1, odim, 3, 2).
        # Input channels 1. Output channels odim.

        # MLX: Conv2d(in_channels, out_channels, kernel_size, stride).
        # Input should be (N, H, W, C).

        # If we treat T as H and idim as W. C=1.

        x = self.conv(x)
        # Output x: (N, H', W', C').
        # H' = T // 4.
        # W' = idim // 4.
        # C' = odim.

        b, t, f, c = x.shape
        # Note: MLX output order matches input spatial dims order + channels at end.

        # We want (B, T', D).
        # We flatten f and c.

        x = x.reshape(b, t, c * f) # (B, T', odim * reduced_idim)

        x = self.out(x)
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask[:, :, 2::2][:, :, 2::2]


class Conv2dSubsampling6(BaseSubsampling):
    """Convolutional 2D subsampling (to 1/6 length).
    """

    def __init__(self, idim: int, odim: int, dropout_rate: float,
                 pos_enc_class: nn.Module):
        """Construct an Conv2dSubsampling6 object."""
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, odim, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(odim, odim, 5, stride=3),
            nn.ReLU(),
        )
        self.linear = nn.Linear(odim * (((idim - 1) // 2 - 2) // 3),
                                      odim)
        self.pos_enc = pos_enc_class
        self.subsampling_rate = 6
        self.right_context = 10

    def __call__(
        self,
        x: mx.array,
        x_mask: mx.array,
        offset: Union[int, mx.array] = 0
    ) -> Tuple[mx.array, mx.array, mx.array]:
        """Subsample x.
        """
        x = x[:, :, :, None] # (B, T, F, 1)
        x = self.conv(x)
        b, t, f, c = x.shape
        x = self.linear(x.reshape(b, t, c * f))
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask[:, :, 2::2][:, :, 4::3]


class Conv2dSubsampling8(BaseSubsampling):
    """Convolutional 2D subsampling (to 1/8 length).
    """

    def __init__(self, idim: int, odim: int, dropout_rate: float,
                 pos_enc_class: nn.Module):
        """Construct an Conv2dSubsampling8 object."""
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, odim, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(odim, odim, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(odim, odim, 3, stride=2),
            nn.ReLU(),
        )
        self.linear = nn.Linear(
            odim * ((((idim - 1) // 2 - 1) // 2 - 1) // 2), odim)
        self.pos_enc = pos_enc_class
        self.subsampling_rate = 8
        self.right_context = 14

    def __call__(
        self,
        x: mx.array,
        x_mask: mx.array,
        offset: Union[int, mx.array] = 0
    ) -> Tuple[mx.array, mx.array, mx.array]:
        """Subsample x.
        """
        x = x[:, :, :, None]
        x = self.conv(x)
        b, t, f, c = x.shape
        x = self.linear(x.reshape(b, t, c * f))
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask[:, :, 2::2][:, :, 2::2][:, :, 2::2]


class LegacyLinearNoSubsampling(BaseSubsampling):
    """Linear transform the input without subsampling
    """

    def __init__(self, idim: int, odim: int, dropout_rate: float,
                 pos_enc_class: nn.Module):
        """Construct an linear object."""
        super().__init__()
        self.out = nn.Sequential(
            nn.Linear(idim, odim),
            nn.LayerNorm(odim, eps=1e-5),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
        )
        self.pos_enc = pos_enc_class
        self.right_context = 0
        self.subsampling_rate = 1

    def __call__(
        self,
        x: mx.array,
        x_mask: mx.array,
        offset: Union[int, mx.array] = 0
    ) -> Tuple[mx.array, mx.array, mx.array]:
        """Input x.
        """
        x = self.out(x)
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask
