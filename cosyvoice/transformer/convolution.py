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
"""ConvolutionModule definition."""

from typing import Tuple

import mlx.core as mx
import mlx.nn as nn

class DepthwiseConv1d(nn.Module):
    """
    Depthwise convolution implemented using loop or group workaround.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        # In depthwise, groups == in_channels == out_channels usually.
        # We need independent filters for each channel.
        # MLX nn.Conv1d does not support groups.
        # We can implement it by running Conv1d with in_channels=1, out_channels=1 for each channel.
        # This is slow but functional.
        # Alternatively, reshape to (Batch, T, C) -> (Batch, T, 1, C)? No.

        # We will use a ModuleList of 1-channel convolutions.

        assert in_channels == out_channels, "Depthwise conv assumes in==out channels here"
        self.channels = in_channels
        self.convs = [
            nn.Conv1d(1, 1, kernel_size, stride=stride, padding=padding, bias=bias)
            for _ in range(in_channels)
        ]
        # Register as list of modules if possible, but list comprehension doesn't register params automatically in some frameworks.
        # In MLX, list of modules assigned to self works? Yes.

    def __call__(self, x):
        # x: (N, T, C)
        # Split channels

        # If we loop over channels:
        outs = []
        for i in range(self.channels):
            # Extract channel i: (N, T, 1)
            xi = x[:, :, i:i+1]
            out_i = self.convs[i](xi)
            outs.append(out_i)

        return mx.concatenate(outs, axis=2)


class ConvolutionModule(nn.Module):
    """ConvolutionModule in Conformer model."""

    def __init__(self,
                 channels: int,
                 kernel_size: int = 15,
                 activation: nn.Module = nn.ReLU(),
                 norm: str = "batch_norm",
                 causal: bool = False,
                 bias: bool = True):
        """Construct an ConvolutionModule object.
        """
        super().__init__()

        self.pointwise_conv1 = nn.Conv1d(
            channels,
            2 * channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        if causal:
            padding = 0
            self.lorder = kernel_size - 1
        else:
            assert (kernel_size - 1) % 2 == 0
            padding = (kernel_size - 1) // 2
            self.lorder = 0

        # self.depthwise_conv = nn.Conv1d(..., groups=channels, ...) -> Replaced
        self.depthwise_conv = DepthwiseConv1d(
            channels,
            channels,
            kernel_size,
            stride=1,
            padding=padding,
            bias=bias,
        )

        assert norm in ['batch_norm', 'layer_norm']
        if norm == "batch_norm":
            self.use_layer_norm = False
            self.norm = nn.BatchNorm(channels)
        else:
            self.use_layer_norm = True
            self.norm = nn.LayerNorm(channels)

        self.pointwise_conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        self.activation = activation

    def __call__(
        self,
        x: mx.array,
        mask_pad: mx.array = None,
        cache: mx.array = None,
    ) -> Tuple[mx.array, mx.array]:
        """Compute convolution module.
        """
        # x is (N, T, C).

        # mask batch padding
        if mask_pad is not None and mask_pad.size > 0:  # time > 0
            # mask_pad (N, 1, T).
            mask = mask_pad.transpose(0, 2, 1)
            x = mx.where(mask == 0, 0.0, x) # Broadcasts over C

        if self.lorder > 0:
            # Causal padding
            if cache is None or cache.size == 0:  # cache_t == 0
                pad_width = [(0, 0), (self.lorder, 0), (0, 0)]
                x = mx.pad(x, pad_width)
            else:
                x = mx.concatenate((cache, x), axis=1)

            new_cache = x[:, -self.lorder:, :]
        else:
            new_cache = mx.zeros((0, 0, 0))

        # GLU mechanism
        # x is (N, T, C)
        x = self.pointwise_conv1(x)  # (N, T, 2*C)
        x = nn.glu(x, axis=2)  # (N, T, C)

        # 1D Depthwise Conv
        x = self.depthwise_conv(x)

        if self.use_layer_norm:
            pass
        else:
            pass

        x = self.norm(x)
        x = self.activation(x)

        x = self.pointwise_conv2(x)

        # mask batch padding
        if mask_pad is not None and mask_pad.size > 0:  # time > 0
             mask = mask_pad.transpose(0, 2, 1)
             x = mx.where(mask == 0, 0.0, x)

        return x, new_cache


# NOTE(Xiang Lyu) causal conv module used in convolution-based vocoder
class CausalConv1d(nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        causal_type: str = 'left',
        device=None,
        dtype=None
    ) -> None:
        if groups > 1:
             # If groups > 1, we might need manual implementation similar to DepthwiseConv1d
             # if mlx doesn't support it.
             # Assuming simple case for now or failing gracefully.
             # If groups == in_channels, use DepthwiseConv1d logic?
             pass

        super().__init__(in_channels, out_channels,
                                           kernel_size, stride=1,
                                           padding=0, dilation=dilation,
                                           # groups=groups, # Ignored/Not supported
                                           bias=bias)
        assert stride == 1
        self.causal_padding = int((kernel_size * dilation - dilation) / 2) * 2 + (kernel_size + 1) % 2
        assert causal_type in ['left', 'right']
        self.causal_type = causal_type

    def __call__(self, x: mx.array, cache: mx.array = None) -> Tuple[mx.array]:
        # x: (N, T, C)
        input_timestep = x.shape[1]

        if cache is None or cache.size == 0:
            # cache should be (N, padding, C).
            cache = mx.zeros((x.shape[0], self.causal_padding, x.shape[2]))

        if self.causal_type == 'left':
            x = mx.concatenate([cache, x], axis=1)
        else:
            x = mx.concatenate([x, cache], axis=1)

        x = super().__call__(x)
        return x


class CausalConv1dDownSample(nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        device=None,
        dtype=None
    ) -> None:
        super().__init__(in_channels, out_channels,
                                                     kernel_size, stride,
                                                     padding=0, dilation=dilation,
                                                     # groups=groups,
                                                     bias=bias)
        assert stride != 1 and dilation == 1
        assert kernel_size % stride == 0
        self.causal_padding = stride - 1

    def __call__(self, x: mx.array, cache: mx.array = None) -> Tuple[mx.array, mx.array]:
        # x: (N, T, C)
        if cache is None or cache.size == 0:
            pad_width = [(0, 0), (self.causal_padding, 0), (0, 0)]
            x = mx.pad(x, pad_width)
        else:
            x = mx.concatenate([cache, x], axis=1)

        x = super().__call__(x)
        return x


class CausalConv1dUpsample(nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        device=None,
        dtype=None
    ) -> None:
        super().__init__(in_channels, out_channels,
                                                   kernel_size, 1,
                                                   padding=0, dilation=dilation,
                                                   # groups=groups,
                                                   bias=bias)
        assert dilation == 1
        self.causal_padding = kernel_size - 1
        self.upsample_scale = stride

    def __call__(self, x: mx.array, cache: mx.array = None) -> Tuple[mx.array, mx.array]:
        # x: (N, T, C)
        # Upsample on T dim.
        if self.upsample_scale > 1:
            x = x.repeat(self.upsample_scale, axis=1) # Nearest neighbor upsample on time

        input_timestep = x.shape[1]
        if cache is None or cache.size == 0:
            pad_width = [(0, 0), (self.causal_padding, 0), (0, 0)]
            x = mx.pad(x, pad_width)
        else:
            x = mx.concatenate([cache, x], axis=1)

        x = super().__call__(x)
        return x
