# Copyright (c) 2020 Mobvoi Inc (Binbin Zhang)
#               2024 Alibaba Inc (authors: Xiang Lyu)
#               2025 Alibaba Inc (authors: Xiang Lyu, Bofan Zhou)
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
"""Unility functions for Transformer."""

import queue
import random
from typing import List

import numpy as np
import mlx.core as mx
import mlx.nn as nn

IGNORE_ID = -1

instruct_list = ["You are a helpful assistant. 请用广东话表达。<|endofprompt|>",
                 "You are a helpful assistant. 请用东北话表达。<|endofprompt|>",
                 "You are a helpful assistant. 请用甘肃话表达。<|endofprompt|>",
                 "You are a helpful assistant. 请用贵州话表达。<|endofprompt|>",
                 "You are a helpful assistant. 请用河南话表达。<|endofprompt|>",
                 "You are a helpful assistant. 请用湖北话表达。<|endofprompt|>",
                 "You are a helpful assistant. 请用湖南话表达。<|endofprompt|>",
                 "You are a helpful assistant. 请用江西话表达。<|endofprompt|>",
                 "You are a helpful assistant. 请用闽南话表达。<|endofprompt|>",
                 "You are a helpful assistant. 请用宁夏话表达。<|endofprompt|>",
                 "You are a helpful assistant. 请用山西话表达。<|endofprompt|>",
                 "You are a helpful assistant. 请用陕西话表达。<|endofprompt|>",
                 "You are a helpful assistant. 请用山东话表达。<|endofprompt|>",
                 "You are a helpful assistant. 请用上海话表达。<|endofprompt|>",
                 "You are a helpful assistant. 请用四川话表达。<|endofprompt|>",
                 "You are a helpful assistant. 请用天津话表达。<|endofprompt|>",
                 "You are a helpful assistant. 请用云南话表达。<|endofprompt|>",
                 "You are a helpful assistant. Please say a sentence as loudly as possible.<|endofprompt|>",
                 "You are a helpful assistant. Please say a sentence in a very soft voice.<|endofprompt|>",
                 "You are a helpful assistant. 请用尽可能慢地语速说一句话。<|endofprompt|>",
                 "You are a helpful assistant. 请用尽可能快地语速说一句话。<|endofprompt|>",
                 "You are a helpful assistant. 请非常开心地说一句话。<|endofprompt|>",
                 "You are a helpful assistant. 请非常伤心地说一句话。<|endofprompt|>",
                 "You are a helpful assistant. 请非常生气地说一句话。<|endofprompt|>",
                 "You are a helpful assistant. 我想体验一下小猪佩奇风格，可以吗？<|endofprompt|>",
                 "You are a helpful assistant. 你可以尝试用机器人的方式解答吗？<|endofprompt|>"]


def pad_list(xs: List[mx.array], pad_value: int):
    """Perform padding for the list of tensors.
    """
    max_len = max([item.shape[0] for item in xs])
    batchs = len(xs)
    ndim = xs[0].ndim

    pad_shape = [batchs, max_len] + list(xs[0].shape[1:])

    # MLX doesn't have .fill_ inplace?
    # Create zeros or fill with pad_value
    # pad_res = mx.zeros(pad_shape, dtype=xs[0].dtype)
    pad_res = mx.full(pad_shape, pad_value, dtype=xs[0].dtype)

    # Since MLX arrays are immutable, we cannot do pad_res[i, :len] = xs[i].
    # We must construct list and stack.

    padded_list = []
    for x in xs:
        pad_len = max_len - x.shape[0]
        if pad_len > 0:
            # Pad at end of dim 0
            pad_shape_local = list(x.shape)
            pad_shape_local[0] = pad_len
            pad = mx.full(pad_shape_local, pad_value, dtype=x.dtype)
            x_padded = mx.concatenate([x, pad], axis=0)
        else:
            x_padded = x
        padded_list.append(x_padded)

    return mx.stack(padded_list, axis=0)


def th_accuracy(pad_outputs: mx.array, pad_targets: mx.array,
                ignore_label: int) -> mx.array:
    """Calculate accuracy.
    """
    pad_pred = pad_outputs.reshape(pad_targets.shape[0], pad_targets.shape[1],
                                pad_outputs.shape[1]).argmax(2)
    mask = pad_targets != ignore_label

    # numerator = torch.sum(pad_pred.masked_select(mask) == pad_targets.masked_select(mask))

    correct = (pad_pred == pad_targets) & mask
    numerator = mx.sum(correct)
    denominator = mx.sum(mask)
    return (numerator / denominator)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


def init_weights(m, mean=0.0, std=0.01):
    # MLX initialization is usually done at construction or by replacing params.
    # In MLX, models are stateful in parameters but `m` here is likely an `nn.Module`.
    # `nn.Module` in MLX holds parameters in `self`.
    # This function is meant to be used with `apply`.
    # MLX `nn.Module` does not have `apply`.
    # But for porting purposes, this function might be called explicitly if we were iterating.
    # If the user code calls `model.apply(init_weights)`, it will fail in MLX.
    # However, in `hifigan/generator.py`, we removed `apply` calls or should have.
    # Let's check `generator.py` port.
    # I commented out `apply` in `generator.py` in my previous step.
    # So this function might not be called.
    # I will leave it empty or implement a warning.
    pass


# Repetition Aware Sampling in VALL-E 2
def ras_sampling(weighted_scores, decoded_tokens, sampling, top_p=0.8, top_k=25, win_size=10, tau_r=0.1):
    top_ids = nucleus_sampling(weighted_scores, top_p=top_p, top_k=top_k)
    # Check repetition
    # decoded_tokens is list of ints
    if len(decoded_tokens) >= win_size:
        recent_tokens = mx.array(decoded_tokens[-win_size:])
        rep_num = mx.sum(recent_tokens == top_ids).item()
        if rep_num >= win_size * tau_r:
            top_ids = random_sampling(weighted_scores, decoded_tokens, sampling)
    return top_ids


def nucleus_sampling(weighted_scores, top_p=0.8, top_k=25):
    # weighted_scores: (vocab_size,)
    probs = nn.softmax(weighted_scores, axis=0)

    # Sort
    sorted_indices = mx.argsort(probs, axis=0)[::-1]
    sorted_probs = probs[sorted_indices]

    cum_probs = mx.cumsum(sorted_probs, axis=0)

    # Cutoff
    # We need to find index where cum_probs >= top_p
    # mask = cum_probs < top_p
    # But we want to include the first one that crosses top_p?
    # Or strict?
    # Torch code: if cum_prob < top_p and len(prob) < top_k: append.

    # We can do this in numpy for scalar logic ease if we want 1:1,
    # or implement vectorized if possible.
    # Since this is sampling (one step), efficiency isn't critical.

    probs_np = np.array(probs)
    sorted_idx_np = np.array(sorted_indices)

    selected_probs = []
    selected_indices = []
    cum = 0.0
    for i in range(len(sorted_idx_np)):
        if cum < top_p and len(selected_probs) < top_k:
            p = probs_np[sorted_idx_np[i]]
            cum += p
            selected_probs.append(p)
            selected_indices.append(sorted_idx_np[i])
        else:
            break

    # Normalize selected probs
    selected_probs = np.array(selected_probs)
    selected_probs /= selected_probs.sum()

    # Sample
    chosen_idx_in_selected = np.random.choice(len(selected_probs), p=selected_probs)
    top_id = selected_indices[chosen_idx_in_selected]
    return int(top_id)


def random_sampling(weighted_scores, decoded_tokens, sampling):
    # weighted_scores: (vocab_size,)
    probs = nn.softmax(weighted_scores, axis=0)
    probs_np = np.array(probs)
    probs_np /= probs_np.sum()
    top_id = np.random.choice(len(probs_np), p=probs_np)
    return int(top_id)


def fade_in_out(fade_in_mel, fade_out_mel, window):
    # fade_in_mel, fade_out_mel: (..., T) or (..., T, D)?
    # window: (win_len,)
    # mel_overlap_len = win_len / 2

    # In MLX, arrays are immutable.
    # Construct result.

    mel_overlap_len = int(window.shape[0] / 2)

    # Slicing
    # fade_in_mel[..., :mel_overlap_len]

    part1 = fade_in_mel[..., :mel_overlap_len] * window[:mel_overlap_len] + \
            fade_out_mel[..., -mel_overlap_len:] * window[mel_overlap_len:]

    # Concatenate: part1 + fade_in_mel[..., mel_overlap_len:]
    res = mx.concatenate([part1, fade_in_mel[..., mel_overlap_len:]], axis=-1)
    return res


def set_all_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    mx.random.seed(seed)


def mask_to_bias(mask: mx.array, dtype=mx.float32) -> mx.array:
    # mask: bool
    mask = mask.astype(dtype)
    # (1.0 - mask) * -1.0e+10
    mask = (1.0 - mask) * -1.0e+10
    return mask


class TrtContextWrapper:
    # TensorRT wrapper. Not applicable for MLX usually unless using TRT-LLM on side?
    # But usually restricted to Nvidia GPUs.
    # MLX is for Apple Silicon.
    # We can probably remove this or keep dummy.
    def __init__(self, trt_engine, trt_concurrent=1, device='cuda:0'):
        pass

    def acquire_estimator(self):
        return None, None

    def release_estimator(self, context, stream):
        pass
