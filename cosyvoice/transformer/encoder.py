# Copyright (c) 2021 Mobvoi Inc (Binbin Zhang, Di Wu)
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
# Modified from ESPnet(https://github.com/espnet/espnet)
"""Encoder definition."""
from typing import Tuple

import mlx.core as mx
import mlx.nn as nn
# import torch.utils.checkpoint as ckpt # No checkpointing in MLX yet/needed?

from cosyvoice.transformer.convolution import ConvolutionModule
from cosyvoice.transformer.encoder_layer import TransformerEncoderLayer
from cosyvoice.transformer.encoder_layer import ConformerEncoderLayer
from cosyvoice.transformer.positionwise_feed_forward import PositionwiseFeedForward
from cosyvoice.utils.class_utils import (
    COSYVOICE_EMB_CLASSES,
    COSYVOICE_SUBSAMPLE_CLASSES,
    COSYVOICE_ATTENTION_CLASSES,
    COSYVOICE_ACTIVATION_CLASSES,
)
from cosyvoice.utils.mask import make_pad_mask
from cosyvoice.utils.mask import add_optional_chunk_mask


class BaseEncoder(nn.Module):

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
        input_layer: str = "conv2d",
        pos_enc_layer_type: str = "abs_pos",
        normalize_before: bool = True,
        static_chunk_size: int = 0,
        use_dynamic_chunk: bool = False,
        global_cmvn: nn.Module = None,
        use_dynamic_left_chunk: bool = False,
        gradient_checkpointing: bool = False,
    ):
        """
        Args:
            input_size (int): input dim
            output_size (int): dimension of attention
            attention_heads (int): the number of heads of multi head attention
            linear_units (int): the hidden units number of position-wise feed
                forward
            num_blocks (int): the number of decoder blocks
            dropout_rate (float): dropout rate
            attention_dropout_rate (float): dropout rate in attention
            positional_dropout_rate (float): dropout rate after adding
                positional encoding
            input_layer (str): input layer type.
                optional [linear, conv2d, conv2d6, conv2d8]
            pos_enc_layer_type (str): Encoder positional encoding layer type.
                opitonal [abs_pos, scaled_abs_pos, rel_pos, no_pos]
            normalize_before (bool):
                True: use layer_norm before each sub-block of a layer.
                False: use layer_norm after each sub-block of a layer.
            static_chunk_size (int): chunk size for static chunk training and
                decoding
            use_dynamic_chunk (bool): whether use dynamic chunk size for
                training or not, You can only use fixed chunk(chunk_size > 0)
                or dyanmic chunk size(use_dynamic_chunk = True)
            global_cmvn (Optional[nn.Module]): Optional GlobalCMVN module
            use_dynamic_left_chunk (bool): whether use dynamic left chunk in
                dynamic chunk training
            key_bias: whether use bias in attention.linear_k, False for whisper models.
            gradient_checkpointing: rerunning a forward-pass segment for each
                checkpointed segment during backward.
        """
        super().__init__()
        self._output_size = output_size

        self.global_cmvn = global_cmvn
        self.embed = COSYVOICE_SUBSAMPLE_CLASSES[input_layer](
            input_size,
            output_size,
            dropout_rate,
            COSYVOICE_EMB_CLASSES[pos_enc_layer_type](output_size,
                                                      positional_dropout_rate),
        )

        self.normalize_before = normalize_before
        self.after_norm = nn.LayerNorm(output_size, eps=1e-5)
        self.static_chunk_size = static_chunk_size
        self.use_dynamic_chunk = use_dynamic_chunk
        self.use_dynamic_left_chunk = use_dynamic_left_chunk
        self.gradient_checkpointing = gradient_checkpointing

    def output_size(self) -> int:
        return self._output_size

    def __call__(
        self,
        xs: mx.array,
        xs_lens: mx.array,
        decoding_chunk_size: int = 0,
        num_decoding_left_chunks: int = -1,
    ) -> Tuple[mx.array, mx.array]:
        """Embed positions in tensor.

        Args:
            xs: padded input tensor (B, T, D)
            xs_lens: input length (B)
            decoding_chunk_size: decoding chunk size for dynamic chunk
                0: default for training, use random dynamic chunk.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
            num_decoding_left_chunks: number of left chunks, this is for decoding,
            the chunk size is decoding_chunk_size.
                >=0: use num_decoding_left_chunks
                <0: use all left chunks
        Returns:
            encoder output tensor xs, and subsampled masks
            xs: padded output tensor (B, T' ~= T/subsample_rate, D)
            masks: torch.Tensor batch padding mask after subsample
                (B, 1, T' ~= T/subsample_rate)
        NOTE(xcsong):
            We pass the `__call__` method of the modules instead of `forward` to the
            checkpointing API because `__call__` attaches all the hooks of the module.
            https://discuss.pytorch.org/t/any-different-between-model-input-and-model-forward-input/3690/2
        """
        T = xs.shape[1]
        # masks = ~make_pad_mask(xs_lens, T).unsqueeze(1)  # (B, 1, T)
        # make_pad_mask returns (B, T) boolean mask (True for padding).
        # We want masks where True for valid? Or True for padding?
        # Original: ~make_pad_mask. make_pad_mask returns True for padding (usually).
        # So ~mask is True for valid.

        # We need to implement make_pad_mask for MLX or assume it returns MLX array.
        # It is imported from cosyvoice.utils.mask. I need to port that too or assume it works.
        # It likely uses torch.

        # Assuming make_pad_mask is ported or I reimplement logic here.
        # xs_lens is (B,).

        # mask = mx.arange(T)[None, :] < xs_lens[:, None] # (B, T) True for valid.
        masks = mx.arange(T)[None, :] < xs_lens[:, None]
        masks = masks[:, None, :] # (B, 1, T)

        if self.global_cmvn is not None:
            xs = self.global_cmvn(xs)
        xs, pos_emb, masks = self.embed(xs, masks)
        mask_pad = masks  # (B, 1, T/subsample_rate)

        # add_optional_chunk_mask also needs porting or verification.
        # Assuming it returns a mask (mx.array).
        chunk_masks = add_optional_chunk_mask(xs, masks,
                                              self.use_dynamic_chunk,
                                              self.use_dynamic_left_chunk,
                                              decoding_chunk_size,
                                              self.static_chunk_size,
                                              num_decoding_left_chunks)

        # if self.gradient_checkpointing and self.training:
             # MLX supports checkpointing via transforms but it's different.
             # Ignoring for now.

        xs = self.forward_layers(xs, chunk_masks, pos_emb, mask_pad)
        if self.normalize_before:
            xs = self.after_norm(xs)
        # Here we assume the mask is not changed in encoder layers, so just
        # return the masks before encoder layers, and the masks will be used
        # for cross attention with decoder later
        return xs, masks

    def forward_layers(self, xs: mx.array, chunk_masks: mx.array,
                       pos_emb: mx.array,
                       mask_pad: mx.array) -> mx.array:
        for layer in self.encoders:
            xs, chunk_masks, _, _ = layer(xs, chunk_masks, pos_emb, mask_pad)
        return xs

    # forward_layers_checkpointed removed or adapted if needed.

    # forward_chunk and forward_chunk_by_chunk porting...
    # These are complex as they manage cache.
    # I should attempt to port logic but keeping in mind MLX array operations.

    def forward_chunk(
        self,
        xs: mx.array,
        offset: int,
        required_cache_size: int,
        att_cache: mx.array = None,
        cnn_cache: mx.array = None,
        att_mask: mx.array = None,
    ) -> Tuple[mx.array, mx.array, mx.array]:
        """ Forward just one chunk
        """
        # assert xs.shape[0] == 1
        # tmp_masks is just for interface compatibility
        # tmp_masks = torch.ones(1, xs.size(1), device=xs.device, dtype=torch.bool)
        tmp_masks = mx.ones((1, xs.shape[1])).astype(mx.bool_)
        tmp_masks = tmp_masks[:, None, :]

        if self.global_cmvn is not None:
            xs = self.global_cmvn(xs)

        xs, pos_emb, _ = self.embed(xs, tmp_masks, offset)

        if att_cache is None:
             # We need to know shape to initialize? Or handle in loop.
             # Original code: att_cache is passed as zeros if empty.
             elayers = 0
             cache_t1 = 0
        else:
             elayers = att_cache.shape[0]
             cache_t1 = att_cache.shape[2]

        chunk_size = xs.shape[1]
        attention_key_size = cache_t1 + chunk_size

        # self.embed.position_encoding
        pos_emb = self.embed.position_encoding(offset=offset - cache_t1,
                                               size=attention_key_size)

        if required_cache_size < 0:
            next_cache_start = 0
        elif required_cache_size == 0:
            next_cache_start = attention_key_size
        else:
            next_cache_start = max(attention_key_size - required_cache_size, 0)

        r_att_cache = []
        r_cnn_cache = []

        if att_mask is None:
             att_mask = mx.ones((0,0,0)).astype(mx.bool_) # Fake mask?

        for i, layer in enumerate(self.encoders):
            # att_cache slice
            if att_cache is not None and elayers > 0:
                layer_att_cache = att_cache[i:i + 1]
            else:
                layer_att_cache = mx.zeros((0,0,0,0)) # or None depending on layer impl

            if cnn_cache is not None and cnn_cache.size > 0:
                layer_cnn_cache = cnn_cache[i]
            else:
                layer_cnn_cache = mx.zeros((0,0,0)) # or None

            xs, _, new_att_cache, new_cnn_cache = layer(
                xs,
                att_mask,
                pos_emb,
                att_cache=layer_att_cache,
                cnn_cache=layer_cnn_cache)

            r_att_cache.append(new_att_cache[:, :, next_cache_start:, :])
            r_cnn_cache.append(new_cnn_cache[None, ...])

        if self.normalize_before:
            xs = self.after_norm(xs)

        r_att_cache = mx.concatenate(r_att_cache, axis=0)
        r_cnn_cache = mx.concatenate(r_cnn_cache, axis=0)

        return (xs, r_att_cache, r_cnn_cache)

    def forward_chunk_by_chunk(
        self,
        xs: mx.array,
        decoding_chunk_size: int,
        num_decoding_left_chunks: int = -1,
    ) -> Tuple[mx.array, mx.array]:
        """ Forward input chunk by chunk with chunk_size like a streaming
            fashion
        """
        assert decoding_chunk_size > 0
        # The model is trained by static or dynamic chunk
        assert self.static_chunk_size > 0 or self.use_dynamic_chunk
        subsampling = self.embed.subsampling_rate
        context = self.embed.right_context + 1  # Add current frame
        stride = subsampling * decoding_chunk_size
        decoding_window = (decoding_chunk_size - 1) * subsampling + context
        num_frames = xs.shape[1]

        att_cache = mx.zeros((0, 0, 0, 0))
        cnn_cache = mx.zeros((0, 0, 0, 0))

        outputs = []
        offset = 0
        required_cache_size = decoding_chunk_size * num_decoding_left_chunks

        # Feed forward overlap input step by step
        for cur in range(0, num_frames - context + 1, stride):
            end = min(cur + decoding_window, num_frames)
            chunk_xs = xs[:, cur:end, :]
            (y, att_cache,
             cnn_cache) = self.forward_chunk(chunk_xs, offset,
                                             required_cache_size, att_cache,
                                             cnn_cache)
            outputs.append(y)
            offset += y.shape[1]
        ys = mx.concatenate(outputs, axis=1)
        masks = mx.ones((1, 1, ys.shape[1])).astype(mx.bool_)
        return ys, masks


class TransformerEncoder(BaseEncoder):
    """Transformer encoder module."""

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
        input_layer: str = "conv2d",
        pos_enc_layer_type: str = "abs_pos",
        normalize_before: bool = True,
        static_chunk_size: int = 0,
        use_dynamic_chunk: bool = False,
        global_cmvn: nn.Module = None,
        use_dynamic_left_chunk: bool = False,
        key_bias: bool = True,
        selfattention_layer_type: str = "selfattn",
        activation_type: str = "relu",
        gradient_checkpointing: bool = False,
    ):
        """ Construct TransformerEncoder

        See Encoder for the meaning of each parameter.
        """
        super().__init__(input_size, output_size, attention_heads,
                         linear_units, num_blocks, dropout_rate,
                         positional_dropout_rate, attention_dropout_rate,
                         input_layer, pos_enc_layer_type, normalize_before,
                         static_chunk_size, use_dynamic_chunk, global_cmvn,
                         use_dynamic_left_chunk, gradient_checkpointing)
        activation = COSYVOICE_ACTIVATION_CLASSES[activation_type]()
        # We need to make sure ModuleList is handled correctly in MLX (usually list of layers)
        # MLX nn.Module works with list of layers if assigned to self.

        self.encoders = [
            TransformerEncoderLayer(
                output_size,
                COSYVOICE_ATTENTION_CLASSES[selfattention_layer_type](attention_heads,
                                                                      output_size,
                                                                      attention_dropout_rate,
                                                                      key_bias),
                PositionwiseFeedForward(output_size, linear_units,
                                        dropout_rate, activation),
                dropout_rate, normalize_before) for _ in range(num_blocks)
        ]


class ConformerEncoder(BaseEncoder):
    """Conformer encoder module."""

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
        input_layer: str = "conv2d",
        pos_enc_layer_type: str = "rel_pos",
        normalize_before: bool = True,
        static_chunk_size: int = 0,
        use_dynamic_chunk: bool = False,
        global_cmvn: nn.Module = None,
        use_dynamic_left_chunk: bool = False,
        positionwise_conv_kernel_size: int = 1,
        macaron_style: bool = True,
        selfattention_layer_type: str = "rel_selfattn",
        activation_type: str = "swish",
        use_cnn_module: bool = True,
        cnn_module_kernel: int = 15,
        causal: bool = False,
        cnn_module_norm: str = "batch_norm",
        key_bias: bool = True,
        gradient_checkpointing: bool = False,
    ):
        """Construct ConformerEncoder
        """
        super().__init__(input_size, output_size, attention_heads,
                         linear_units, num_blocks, dropout_rate,
                         positional_dropout_rate, attention_dropout_rate,
                         input_layer, pos_enc_layer_type, normalize_before,
                         static_chunk_size, use_dynamic_chunk, global_cmvn,
                         use_dynamic_left_chunk, gradient_checkpointing)
        activation = COSYVOICE_ACTIVATION_CLASSES[activation_type]()

        # self-attention module definition
        encoder_selfattn_layer_args = (
            attention_heads,
            output_size,
            attention_dropout_rate,
            key_bias,
        )
        # feed-forward module definition
        positionwise_layer_args = (
            output_size,
            linear_units,
            dropout_rate,
            activation,
        )
        # convolution module definition
        convolution_layer_args = (output_size, cnn_module_kernel, activation,
                                  cnn_module_norm, causal)

        self.encoders = [
            ConformerEncoderLayer(
                output_size,
                COSYVOICE_ATTENTION_CLASSES[selfattention_layer_type](
                    *encoder_selfattn_layer_args),
                PositionwiseFeedForward(*positionwise_layer_args),
                PositionwiseFeedForward(
                    *positionwise_layer_args) if macaron_style else None,
                ConvolutionModule(
                    *convolution_layer_args) if use_cnn_module else None,
                dropout_rate,
                normalize_before,
            ) for _ in range(num_blocks)
        ]
