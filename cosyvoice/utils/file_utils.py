# Copyright (c) 2021 Mobvoi Inc. (authors: Binbin Zhang)
#               2024 Alibaba Inc (authors: Xiang Lyu, Zetao Hu)
#               2025 Alibaba Inc (authors: Xiang Lyu, Yabin Li)
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

import os
import json
import logging
import soundfile as sf
import mlx.core as mx
# import torch
# import torchaudio

logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s')


def read_lists(list_file):
    lists = []
    with open(list_file, 'r', encoding='utf8') as fin:
        for line in fin:
            lists.append(line.strip())
    return lists


def read_json_lists(list_file):
    lists = read_lists(list_file)
    results = {}
    for fn in lists:
        with open(fn, 'r', encoding='utf8') as fin:
            results.update(json.load(fin))
    return results


def load_wav(wav, target_sr, min_sr=16000):
    speech, sample_rate = sf.read(wav)
    # speech is (T, C) or (T,).
    if speech.ndim == 1:
        speech = speech[:, None] # (T, 1)

    # Transpose to (1, T) to match original torchaudio convention which usually returns (C, T)
    # But torchaudio.load returns (C, T).
    # Let's keep it (1, T) for mono.
    speech = speech.T # (C, T)

    # Mix to mono if needed
    if speech.shape[0] > 1:
        speech = speech.mean(axis=0, keepdims=True)

    if sample_rate != target_sr:
        assert sample_rate >= min_sr, 'wav sample rate {} must be greater than {}'.format(sample_rate, target_sr)
        # Resample logic
        # For simplicity in MLX/numpy without heavy resampling libs, we might skip or use simple interpolation.
        # But robust resampling is complex.
        # Assuming librosa or scipy is available if high quality needed.
        # But we want to avoid too many deps.
        # Let's implement simple linear interpolation or use scipy.signal.resample if scipy installed (it is).
        from scipy import signal
        num_samples = int(speech.shape[1] * target_sr / sample_rate)
        # scipy.signal.resample operates on last axis by default
        speech_resampled = signal.resample(speech, num_samples, axis=1)
        speech = speech_resampled

    return mx.array(speech)


def convert_onnx_to_trt(trt_model, trt_kwargs, onnx_model, fp16):
    # TensorRT logic removed
    pass


# NOTE do not support bistream inference as only speech token embedding/head is kept
def export_cosyvoice2_vllm(model, model_path, device):
    # vllm export logic removed
    pass
