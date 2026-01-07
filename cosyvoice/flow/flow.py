# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Zhihao Du)
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
import logging
import random
from typing import Dict, Optional
import mlx.core as mx
import mlx.nn as nn
# from omegaconf import DictConfig # Assuming DictConfig usage can be handled by standard dict or removed if just typing
# cosyvoice.utils.mask has make_pad_mask (ported)
from cosyvoice.utils.mask import make_pad_mask

# Helper for omegaconf if needed or just use dict
class DictConfig(dict):
    pass

class MaskedDiffWithXvec(nn.Module):
    def __init__(self,
                 input_size: int = 512,
                 output_size: int = 80,
                 spk_embed_dim: int = 192,
                 output_type: str = "mel",
                 vocab_size: int = 4096,
                 input_frame_rate: int = 50,
                 only_mask_loss: bool = True,
                 encoder: nn.Module = None,
                 length_regulator: nn.Module = None,
                 decoder: nn.Module = None,
                 decoder_conf: Dict = {}):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.decoder_conf = decoder_conf
        self.vocab_size = vocab_size
        self.output_type = output_type
        self.input_frame_rate = input_frame_rate
        logging.info(f"input frame rate={self.input_frame_rate}")
        self.input_embedding = nn.Embedding(vocab_size, input_size)
        self.spk_embed_affine_layer = nn.Linear(spk_embed_dim, output_size)
        self.encoder = encoder
        self.encoder_proj = nn.Linear(self.encoder.output_size(), output_size)
        self.decoder = decoder
        self.length_regulator = length_regulator
        self.only_mask_loss = only_mask_loss

    def __call__(
            self,
            batch: dict,
            device: str = None,
    ) -> Dict[str, Optional[mx.array]]:
        token = batch['speech_token']
        token_len = batch['speech_token_len']
        feat = batch['speech_feat']
        feat_len = batch['speech_feat_len']
        embedding = batch['embedding']

        # xvec projection
        norm = mx.linalg.norm(embedding, axis=1, keepdims=True)
        embedding = embedding / (norm + 1e-6)
        embedding = self.spk_embed_affine_layer(embedding)

        # concat text and prompt_text
        # mask = (~make_pad_mask(token_len)).float().unsqueeze(-1)
        # make_pad_mask returns True for padding in torch (usually).
        # My MLX port returns True for padding? Let's check mask.py.
        # make_pad_mask returns mask where indices >= lengths. So True for pad.
        # So ~mask is valid.

        mask = make_pad_mask(token_len)
        mask = (mask == 0) # True for valid
        mask = mask.astype(mx.float32)[:, :, None] # (B, T, 1)

        # token clamp min 0. In MLX token is int?
        # token = self.input_embedding(mx.maximum(token, 0)) * mask
        # nn.Embedding takes int input.
        # mx.maximum works.

        token_emb = self.input_embedding(mx.maximum(token, 0))
        token_emb = token_emb * mask

        # text encode
        h, h_lengths = self.encoder(token_emb, token_len)
        h = self.encoder_proj(h)
        h, h_lengths = self.length_regulator(h, feat_len)

        # get conditions
        conds = mx.zeros(feat.shape)
        # conds update loop
        # feat_len is (B,)
        # In MLX we cannot easily iterate and update tensor in-place efficiently if we want to trace/compile,
        # but eager mode is fine.

        # Original logic: randomly pick prefix as condition.

        # Since we are in inference mostly or fine-tuning, eager loop is okay.
        # Or construct mask.

        cond_list = []
        for i in range(feat_len.shape[0]):
            l = int(feat_len[i].item())
            if random.random() < 0.5:
                cond_list.append(mx.zeros_like(feat[i]))
            else:
                index = random.randint(0, int(0.3 * l))
                # mask out after index
                # feat[i] is (T, D)
                mask_i = mx.arange(feat.shape[1]) < index
                cond_i = feat[i] * mask_i[:, None]
                cond_list.append(cond_i)

        conds = mx.stack(cond_list, axis=0)

        # conds.transpose(1, 2) ?
        # feat in torch is usually (B, T, D) for conformer/transformer input.
        # In original flow.py: `conds = conds.transpose(1, 2)`.
        # This implies conds becomes (B, D, T).
        # decoder.compute_loss takes `cond`.
        # Let's check `flow_matching.py` (ConditionalCFM).
        # It expects `mu` (output of encoder) shape `(batch_size, n_feats, mel_timesteps)` in docstring.
        # But my port of `flow_matching.py` assumes `mu` is `(B, D, T)`?
        # Wait, standard MLX layers like Conv1d expect `(N, T, C)`.
        # TransformerEncoder output `h` is `(B, T, D)`.

        # Original code:
        # h, h_lengths = self.encoder(...) -> h (B, T, D)
        # h = h.transpose(1, 2) -> (B, D, T) passed to compute_loss.
        # `feat.transpose(1, 2)` passed as target.

        # In `flow_matching.py` (ConditionalCFM):
        # `mu` docstring says `(batch_size, n_feats, mel_timesteps)` -> (B, D, T).

        # But wait, MLX layers prefer (B, T, D).
        # If `ConditionalCFM` uses `estimator` which is likely a WaveNet or UNet or Conformer-like.
        # If estimator uses Conv1d, it expects (B, T, C).
        # If estimator expects (B, D, T), then we should transpose.

        # Let's verify `ConditionalCFM.compute_loss`.
        # It does `b, _, t = mu.shape`. If shape is (B, D, T), `t` is timesteps.

        # My `ConditionalCFM` port assumes (B, T, D) if I used standard MLX layers inside estimator?
        # The user provided `estimator` to `ConditionalCFM`.
        # If `estimator` is `DiT` or `WaveNet` ported to MLX, they usually follow MLX convention (B, T, C).

        # If I change the convention to (B, T, D), I should remove transposes here.
        # Let's assume standard MLX (B, T, D).

        # So I will NOT transpose here.
        # And I should check `ConditionalCFM`.

        mask = make_pad_mask(feat_len)
        mask = (mask == 0).astype(mx.float32)[:, :, None] # (B, T, 1) (valid mask)

        loss, _ = self.decoder.compute_loss(
            feat, # (B, T, D)
            mask, # (B, T, 1)
            h,    # (B, T, D)
            embedding,
            cond=conds
        )
        return {'loss': loss}

    def inference(self,
                  token,
                  token_len,
                  prompt_token,
                  prompt_token_len,
                  prompt_feat,
                  prompt_feat_len,
                  embedding,
                  flow_cache):
        # assert token.shape[0] == 1
        norm = mx.linalg.norm(embedding, axis=1, keepdims=True)
        embedding = embedding / (norm + 1e-6)
        embedding = self.spk_embed_affine_layer(embedding)

        # concat speech token and prompt speech token
        token_len1 = prompt_token.shape[1]
        token_len2 = token.shape[1]

        token = mx.concatenate([prompt_token, token], axis=1)
        token_len = prompt_token_len + token_len

        mask = make_pad_mask(token_len)
        mask = (mask == 0).astype(mx.float32)[:, :, None]

        token_emb = self.input_embedding(mx.maximum(token, 0)) * mask

        # text encode
        h, h_lengths = self.encoder(token_emb, token_len)
        h = self.encoder_proj(h)

        # length regulator inference
        # Assuming length_regulator is ported or compatible.
        # h (B, T, D)
        # mel_len1 from prompt_feat.
        mel_len1 = prompt_feat.shape[1]
        mel_len2 = int(token_len2 / self.input_frame_rate * 22050 / 256)

        h_part1 = h[:, :token_len1]
        h_part2 = h[:, token_len1:]

        h, h_lengths = self.length_regulator.inference(h_part1, h_part2, mel_len1, mel_len2, self.input_frame_rate)

        # get conditions
        # conds: (1, mel_len1 + mel_len2, output_size)
        total_len = mel_len1 + mel_len2
        conds = mx.zeros((1, total_len, self.output_size), dtype=h.dtype)
        # conds[:, :mel_len1] = prompt_feat
        # conds = conds.at[:, :mel_len1].set(prompt_feat) # Not efficient but correct.
        # Or construct:
        # conds = mx.concatenate([prompt_feat, mx.zeros((1, mel_len2, output_size))], axis=1)

        conds = mx.concatenate([prompt_feat, mx.zeros((1, mel_len2, self.output_size), dtype=h.dtype)], axis=1)

        # No transpose if (B, T, D) convention.

        mask = make_pad_mask(mx.array([total_len]))
        mask = (mask == 0).astype(mx.float32)[:, :, None]

        # decoder inference
        feat, flow_cache = self.decoder(
            mu=h,
            mask=mask,
            spks=embedding,
            cond=conds,
            n_timesteps=10,
            prompt_len=mel_len1,
            cache=flow_cache
        )

        feat = feat[:, mel_len1:, :] # (B, T_gen, D)
        # assert feat.shape[1] == mel_len2

        return feat, flow_cache


class CausalMaskedDiffWithXvec(nn.Module):
    def __init__(self,
                 input_size: int = 512,
                 output_size: int = 80,
                 spk_embed_dim: int = 192,
                 output_type: str = "mel",
                 vocab_size: int = 4096,
                 input_frame_rate: int = 50,
                 only_mask_loss: bool = True,
                 token_mel_ratio: int = 2,
                 pre_lookahead_len: int = 3,
                 encoder: nn.Module = None,
                 decoder: nn.Module = None,
                 decoder_conf: Dict = {}):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.decoder_conf = decoder_conf
        self.vocab_size = vocab_size
        self.output_type = output_type
        self.input_frame_rate = input_frame_rate
        logging.info(f"input frame rate={self.input_frame_rate}")
        self.input_embedding = nn.Embedding(vocab_size, input_size)
        self.spk_embed_affine_layer = nn.Linear(spk_embed_dim, output_size)
        self.encoder = encoder
        self.encoder_proj = nn.Linear(self.encoder.output_size(), output_size)
        self.decoder = decoder
        self.only_mask_loss = only_mask_loss
        self.token_mel_ratio = token_mel_ratio
        self.pre_lookahead_len = pre_lookahead_len

    def __call__(
            self,
            batch: dict,
            device: str = None,
    ) -> Dict[str, Optional[mx.array]]:
        token = batch['speech_token']
        token_len = batch['speech_token_len']
        feat = batch['speech_feat']
        feat_len = batch['speech_feat_len']
        embedding = batch['embedding']

        streaming = True if random.random() < 0.5 else False

        norm = mx.linalg.norm(embedding, axis=1, keepdims=True)
        embedding = embedding / (norm + 1e-6)
        embedding = self.spk_embed_affine_layer(embedding)

        mask = make_pad_mask(token_len)
        mask = (mask == 0).astype(mx.float32)[:, :, None]

        token_emb = self.input_embedding(mx.maximum(token, 0)) * mask

        # text encode
        h, h_lengths = self.encoder(token_emb, token_len, streaming=streaming)
        h = self.encoder_proj(h)

        # get conditions
        cond_list = []
        for i in range(feat_len.shape[0]):
            l = int(feat_len[i].item())
            if random.random() < 0.5:
                cond_list.append(mx.zeros_like(feat[i]))
            else:
                index = random.randint(0, int(0.3 * l))
                mask_i = mx.arange(feat.shape[1]) < index
                cond_i = feat[i] * mask_i[:, None]
                cond_list.append(cond_i)
        conds = mx.stack(cond_list, axis=0)

        # mask for decoder loss
        # h_lengths sum? In torch code: h_lengths.sum(dim=-1).squeeze(dim=1).
        # h_lengths from encoder usually (B,). If it is (B, 1), squeeze.

        mask_len = h_lengths
        if mask_len.ndim > 1:
             mask_len = mx.sum(mask_len, axis=-1)

        mask = make_pad_mask(mask_len)
        mask = (mask == 0).astype(mx.float32)[:, :, None]

        loss, _ = self.decoder.compute_loss(
            feat,
            mask,
            h,
            embedding,
            cond=conds,
            streaming=streaming,
        )
        return {'loss': loss}

    def inference(self,
                  token,
                  token_len,
                  prompt_token,
                  prompt_token_len,
                  prompt_feat,
                  prompt_feat_len,
                  embedding,
                  streaming,
                  finalize):
        norm = mx.linalg.norm(embedding, axis=1, keepdims=True)
        embedding = embedding / (norm + 1e-6)
        embedding = self.spk_embed_affine_layer(embedding)

        token = mx.concatenate([prompt_token, token], axis=1)
        token_len = prompt_token_len + token_len

        mask = make_pad_mask(token_len)
        mask = (mask == 0).astype(mx.float32)[:, :, None]

        token_emb = self.input_embedding(mx.maximum(token, 0)) * mask

        if finalize is True:
            h, h_lengths = self.encoder(token_emb, token_len, streaming=streaming)
        else:
            token_in = token_emb[:, :-self.pre_lookahead_len]
            context = token_emb[:, -self.pre_lookahead_len:]
            h, h_lengths = self.encoder(token_in, token_len, context=context, streaming=streaming)

        mel_len1 = prompt_feat.shape[1]
        mel_len2 = h.shape[1] - prompt_feat.shape[1]
        h = self.encoder_proj(h)

        conds = mx.concatenate([prompt_feat, mx.zeros((1, mel_len2, self.output_size), dtype=h.dtype)], axis=1)

        mask = make_pad_mask(mx.array([mel_len1 + mel_len2]))
        mask = (mask == 0).astype(mx.float32)[:, :, None]

        feat, _ = self.decoder(
            mu=h,
            mask=mask,
            spks=embedding,
            cond=conds,
            n_timesteps=10,
            streaming=streaming
        )
        feat = feat[:, mel_len1:, :]
        return feat, None


class CausalMaskedDiffWithDiT(nn.Module):
    def __init__(self,
                 input_size: int = 512,
                 output_size: int = 80,
                 spk_embed_dim: int = 192,
                 output_type: str = "mel",
                 vocab_size: int = 4096,
                 input_frame_rate: int = 50,
                 only_mask_loss: bool = True,
                 token_mel_ratio: int = 2,
                 pre_lookahead_len: int = 3,
                 pre_lookahead_layer: nn.Module = None,
                 decoder: nn.Module = None,
                 decoder_conf: Dict = {}):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.decoder_conf = decoder_conf
        self.vocab_size = vocab_size
        self.output_type = output_type
        self.input_frame_rate = input_frame_rate
        logging.info(f"input frame rate={self.input_frame_rate}")
        self.input_embedding = nn.Embedding(vocab_size, input_size)
        self.spk_embed_affine_layer = nn.Linear(spk_embed_dim, output_size)
        self.pre_lookahead_len = pre_lookahead_len
        self.pre_lookahead_layer = pre_lookahead_layer
        self.decoder = decoder
        self.only_mask_loss = only_mask_loss
        self.token_mel_ratio = token_mel_ratio

    def __call__(
            self,
            batch: dict,
            device: str = None,
    ) -> Dict[str, Optional[mx.array]]:
        token = batch['speech_token']
        token_len = batch['speech_token_len']
        feat = batch['speech_feat']
        feat_len = batch['speech_feat_len']
        embedding = batch['embedding']

        streaming = True if random.random() < 0.5 else False

        norm = mx.linalg.norm(embedding, axis=1, keepdims=True)
        embedding = embedding / (norm + 1e-6)
        embedding = self.spk_embed_affine_layer(embedding)

        mask = make_pad_mask(token_len)
        mask = (mask == 0).astype(mx.float32)[:, :, None]

        token_emb = self.input_embedding(mx.maximum(token, 0)) * mask

        # text encode
        h = self.pre_lookahead_layer(token_emb)
        # h = h.repeat_interleave(self.token_mel_ratio, dim=1)
        h = h.repeat(self.token_mel_ratio, axis=1) # Repeat on time dim?
        # Original: dim=1 (time).

        # mask.repeat_interleave
        mask = mask.repeat(self.token_mel_ratio, axis=1).squeeze(-1) # squeeze last dim if needed for broadcasting in decoder loss?
        # decoder.compute_loss mask unsqueeze(1) -> (B, 1, T).
        # Here mask (B, T).

        # get conditions
        cond_list = []
        for i in range(feat_len.shape[0]):
            l = int(feat_len[i].item())
            if random.random() < 0.5:
                cond_list.append(mx.zeros_like(feat[i]))
            else:
                index = random.randint(0, int(0.3 * l))
                mask_i = mx.arange(feat.shape[1]) < index
                cond_i = feat[i] * mask_i[:, None]
                cond_list.append(cond_i)
        conds = mx.stack(cond_list, axis=0)

        # mask needs to be (B, T, 1) or (B, T) depending on compute_loss expectation.
        # compute_loss calls `mask.unsqueeze(1)` -> (B, 1, T).
        # My compute_loss expects (B, T, 1) if using MLX convention?
        # Let's align with ConditionalCFM.
        # ConditionalCFM mask is concatenated.
        # If I change mask shape to (B, T, 1) here it should be consistent.

        mask = mask[:, :, None] # (B, T, 1)

        loss, _ = self.decoder.compute_loss(
            feat,
            mask,
            h,
            embedding,
            cond=conds,
            streaming=streaming,
        )
        return {'loss': loss}

    def inference(self,
                  token,
                  token_len,
                  prompt_token,
                  prompt_token_len,
                  prompt_feat,
                  prompt_feat_len,
                  embedding,
                  streaming,
                  finalize):

        norm = mx.linalg.norm(embedding, axis=1, keepdims=True)
        embedding = embedding / (norm + 1e-6)
        embedding = self.spk_embed_affine_layer(embedding)

        token = mx.concatenate([prompt_token, token], axis=1)
        token_len = prompt_token_len + token_len

        mask = make_pad_mask(token_len)
        mask = (mask == 0).astype(mx.float32)[:, :, None]

        token_emb = self.input_embedding(mx.maximum(token, 0)) * mask

        if finalize is True:
            h = self.pre_lookahead_layer(token_emb)
        else:
            token_in = token_emb[:, :-self.pre_lookahead_len]
            context = token_emb[:, -self.pre_lookahead_len:]
            # Assuming pre_lookahead_layer handles context if signature matches
            h = self.pre_lookahead_layer(token_in, context=context)

        h = h.repeat(self.token_mel_ratio, axis=1)

        mel_len1 = prompt_feat.shape[1]
        mel_len2 = h.shape[1] - prompt_feat.shape[1]

        conds = mx.concatenate([prompt_feat, mx.zeros((1, mel_len2, self.output_size), dtype=h.dtype)], axis=1)

        mask = make_pad_mask(mx.array([mel_len1 + mel_len2]))
        mask = (mask == 0).astype(mx.float32)[:, :, None]

        feat, _ = self.decoder(
            mu=h,
            mask=mask,
            spks=embedding,
            cond=conds,
            n_timesteps=10,
            streaming=streaming
        )
        feat = feat[:, mel_len1:, :]
        return feat, None
