# Copyright (c) 2021 Mobvoi Inc. (authors: Binbin Zhang, Di Wu)
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
"""Decoder definition."""
from typing import Tuple, List, Optional

import mlx.core as mx
import mlx.nn as nn
import logging

from cosyvoice.transformer.decoder_layer import DecoderLayer
from cosyvoice.transformer.positionwise_feed_forward import PositionwiseFeedForward
from cosyvoice.utils.class_utils import (
    COSYVOICE_EMB_CLASSES,
    COSYVOICE_ATTENTION_CLASSES,
    COSYVOICE_ACTIVATION_CLASSES,
)
from cosyvoice.utils.mask import (subsequent_mask, make_pad_mask)


class TransformerDecoder(nn.Module):
    """Base class of Transfomer decoder module.
    """

    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        self_attention_dropout_rate: float = 0.0,
        src_attention_dropout_rate: float = 0.0,
        input_layer: str = "embed",
        use_output_layer: bool = True,
        normalize_before: bool = True,
        src_attention: bool = True,
        key_bias: bool = True,
        activation_type: str = "relu",
        gradient_checkpointing: bool = False,
        tie_word_embedding: bool = False,
    ):
        super().__init__()
        attention_dim = encoder_output_size
        activation = COSYVOICE_ACTIVATION_CLASSES[activation_type]()

        if input_layer == "no_pos":
             embed_layer = nn.Identity()
        else:
             embed_layer = nn.Embedding(vocab_size, attention_dim)

        self.embed = nn.Sequential(
            embed_layer,
            COSYVOICE_EMB_CLASSES[input_layer](attention_dim,
                                               positional_dropout_rate),
        )

        self.normalize_before = normalize_before
        self.after_norm = nn.LayerNorm(attention_dim, eps=1e-5)
        self.use_output_layer = use_output_layer
        if use_output_layer:
            self.output_layer = nn.Linear(attention_dim, vocab_size)
        else:
            self.output_layer = nn.Identity()
        self.num_blocks = num_blocks
        self.decoders = [
            DecoderLayer(
                attention_dim,
                COSYVOICE_ATTENTION_CLASSES["selfattn"](
                    attention_heads, attention_dim,
                    self_attention_dropout_rate, key_bias),
                COSYVOICE_ATTENTION_CLASSES["selfattn"](
                    attention_heads, attention_dim, src_attention_dropout_rate,
                    key_bias) if src_attention else None,
                PositionwiseFeedForward(attention_dim, linear_units,
                                        dropout_rate, activation),
                dropout_rate,
                normalize_before,
            ) for _ in range(self.num_blocks)
        ]

        self.gradient_checkpointing = gradient_checkpointing
        self.tie_word_embedding = tie_word_embedding

    def __call__(
        self,
        memory: mx.array,
        memory_mask: mx.array,
        ys_in_pad: mx.array,
        ys_in_lens: mx.array,
        r_ys_in_pad: mx.array = None,
        reverse_weight: float = 0.0,
    ) -> Tuple[mx.array, mx.array, mx.array]:
        """Forward decoder.
        """
        tgt = ys_in_pad
        maxlen = tgt.shape[1]

        # tgt_mask: (B, 1, L)
        # make_pad_mask returns True for padding
        # We need ~mask (valid)
        tgt_mask = make_pad_mask(ys_in_lens, maxlen)[:, None, :]
        tgt_mask = (tgt_mask == 0) # Flip logic if make_pad_mask returns 1 for pad

        # m: (1, L, L)
        m = subsequent_mask(tgt_mask.shape[-1])[None, :, :]
        # tgt_mask: (B, L, L)
        tgt_mask = tgt_mask & m

        x, _ = self.embed(tgt)

        x = self.forward_layers(x, tgt_mask, memory, memory_mask)

        if self.normalize_before:
            x = self.after_norm(x)
        if self.use_output_layer:
            x = self.output_layer(x)

        olens = mx.sum(tgt_mask, axis=1) # This sum might be tricky if mask is broadcasted?
        # tgt_mask is (B, L, L). We want length of output.
        # ys_in_lens is passed in. We can return that?
        # Original: olens = tgt_mask.sum(1).
        # This sums over the L (query) dimension? No, sum(1) sums over L (rows)?
        # If mask is (B, L, L), sum(1) -> (B, L).
        # Actually olens return in original code seems unused or just informational.
        # But `ys_in_lens` is available.

        return x, mx.array(0.0), ys_in_lens

    def forward_layers(self, x: mx.array, tgt_mask: mx.array,
                       memory: mx.array,
                       memory_mask: mx.array) -> mx.array:
        for layer in self.decoders:
            x, tgt_mask, memory, memory_mask = layer(x, tgt_mask, memory,
                                                     memory_mask)
        return x

    def forward_one_step(
        self,
        memory: mx.array,
        memory_mask: mx.array,
        tgt: mx.array,
        tgt_mask: mx.array,
        cache: Optional[List[mx.array]] = None,
    ) -> Tuple[mx.array, List[mx.array]]:
        """Forward one step.
        """
        x, _ = self.embed(tgt)
        new_cache = []
        for i, decoder in enumerate(self.decoders):
            if cache is None:
                c = None
            else:
                c = cache[i]
            x, tgt_mask, memory, memory_mask = decoder(x,
                                                       tgt_mask,
                                                       memory,
                                                       memory_mask,
                                                       cache=c)
            new_cache.append(x)
        if self.normalize_before:
            y = self.after_norm(x[:, -1])
        else:
            y = x[:, -1]
        if self.use_output_layer:
            y = nn.log_softmax(self.output_layer(y), axis=-1)
        return y, new_cache

    def tie_or_clone_weights(self, jit_mode: bool = True):
        """Tie or clone module weights (between word_emb and output_layer)
        """
        if not self.use_output_layer:
            return
        # In MLX, tying weights is done by assigning parameters.
        # self.output_layer.weight = self.embed[0].weight

        # embed[0] is Identity or Embedding.
        if isinstance(self.embed.layers[0], nn.Embedding):
             self.output_layer.weight = self.embed.layers[0].weight

        # Bias pad logic: MLX doesn't usually pad bias like that,
        # but if output dim > vocab size for some reason?
        # Usually they match.


class BiTransformerDecoder(nn.Module):
    """Base class of Transfomer decoder module.
    """

    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        r_num_blocks: int = 0,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        self_attention_dropout_rate: float = 0.0,
        src_attention_dropout_rate: float = 0.0,
        input_layer: str = "embed",
        use_output_layer: bool = True,
        normalize_before: bool = True,
        key_bias: bool = True,
        gradient_checkpointing: bool = False,
        tie_word_embedding: bool = False,
    ):

        super().__init__()
        self.tie_word_embedding = tie_word_embedding
        self.left_decoder = TransformerDecoder(
            vocab_size,
            encoder_output_size,
            attention_heads,
            linear_units,
            num_blocks,
            dropout_rate,
            positional_dropout_rate,
            self_attention_dropout_rate,
            src_attention_dropout_rate,
            input_layer,
            use_output_layer,
            normalize_before,
            key_bias=key_bias,
            gradient_checkpointing=gradient_checkpointing,
            tie_word_embedding=tie_word_embedding)

        self.right_decoder = TransformerDecoder(
            vocab_size,
            encoder_output_size,
            attention_heads,
            linear_units,
            r_num_blocks,
            dropout_rate,
            positional_dropout_rate,
            self_attention_dropout_rate,
            src_attention_dropout_rate,
            input_layer,
            use_output_layer,
            normalize_before,
            key_bias=key_bias,
            gradient_checkpointing=gradient_checkpointing,
            tie_word_embedding=tie_word_embedding)

    def __call__(
        self,
        memory: mx.array,
        memory_mask: mx.array,
        ys_in_pad: mx.array,
        ys_in_lens: mx.array,
        r_ys_in_pad: mx.array,
        reverse_weight: float = 0.0,
    ) -> Tuple[mx.array, mx.array, mx.array]:
        """Forward decoder.
        """
        l_x, _, olens = self.left_decoder(memory, memory_mask, ys_in_pad,
                                          ys_in_lens)
        r_x = mx.array(0.0)
        if reverse_weight > 0.0:
            r_x, _, olens = self.right_decoder(memory, memory_mask,
                                               r_ys_in_pad, ys_in_lens)
        return l_x, r_x, olens

    def forward_one_step(
        self,
        memory: mx.array,
        memory_mask: mx.array,
        tgt: mx.array,
        tgt_mask: mx.array,
        cache: Optional[List[mx.array]] = None,
    ) -> Tuple[mx.array, List[mx.array]]:
        """Forward one step.
        """
        return self.left_decoder.forward_one_step(memory, memory_mask, tgt,
                                                  tgt_mask, cache)

    def tie_or_clone_weights(self, jit_mode: bool = True):
        """Tie or clone module weights (between word_emb and output_layer)
        """
        self.left_decoder.tie_or_clone_weights(jit_mode)
        self.right_decoder.tie_or_clone_weights(jit_mode)
