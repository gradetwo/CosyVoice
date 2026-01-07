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
"""Label smoothing module."""

import mlx.core as mx
import mlx.nn as nn


class LabelSmoothingLoss(nn.Module):
    """Label-smoothing loss.
    """

    def __init__(self,
                 size: int,
                 padding_idx: int,
                 smoothing: float,
                 normalize_length: bool = False):
        """Construct an LabelSmoothingLoss object."""
        super(LabelSmoothingLoss, self).__init__()
        # self.criterion = nn.KLDivLoss(reduction="none") # MLX doesn't have KLDivLoss module?
        # But we can compute it manually.
        # KL(P || Q) = sum(P(x) * log(P(x) / Q(x))) = sum(P(x) * (log P(x) - log Q(x)))
        # Here x is log_softmax output (log_probs). true_dist is P.
        # criterion(log_probs, target_probs) expects input, target.
        # PyTorch KLDivLoss: l(x, y) = y * (log y - x)

        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.normalize_length = normalize_length

    def __call__(self, x: mx.array, target: mx.array) -> mx.array:
        """Compute loss between x and target.

        Args:
            x (mx.array): prediction (batch, seqlen, class)
            target (mx.array):
                target signal masked with self.padding_id (batch, seqlen)
        Returns:
            loss (mx.array) : The KL loss, scalar float value
        """
        assert x.shape[2] == self.size
        batch_size = x.shape[0]
        x = x.reshape(-1, self.size)
        target = target.reshape(-1)

        # true_dist construction
        # true_dist = mx.zeros_like(x) # MLX doesn't have zeros_like? mx.zeros(x.shape, dtype=x.dtype)
        # true_dist.fill_(...) -> MLX arrays are immutable.

        # true_dist = mx.full(x.shape, self.smoothing / (self.size - 1), dtype=x.dtype)

        # ignore = target == self.padding_idx
        # total = len(target) - ignore.sum().item()

        # target = target.masked_fill(ignore, 0) -> target = mx.where(ignore, 0, target)

        # true_dist.scatter_(...) ->
        # true_dist[mx.arange(len(target)), target] = self.confidence

        # Let's do it efficiently.
        # We want to compute KL divergence.
        # Loss = -sum(true_dist * log_probs)
        # true_dist has value `confidence` at target index, and `smoothing/(size-1)` elsewhere.

        # log_probs = log_softmax(x)
        log_probs = nn.log_softmax(x, axis=1)

        # Loss = - [ confidence * log_probs[target] + sum_{i!=target} (smoothing/(size-1)) * log_probs[i] ]
        #      = - [ confidence * log_probs[target] + (smoothing/(size-1)) * (sum(log_probs) - log_probs[target]) ]

        # We need to handle padding_idx.

        ignore = (target == self.padding_idx)
        # total = (1 - ignore).sum() # count valid

        # Mask target for safe indexing
        safe_target = mx.where(ignore, 0, target)

        # log_probs_target = log_probs[mx.arange(len(target)), safe_target]
        # In MLX, gathering like this needs explicit indices or take_along_axis?
        # log_probs is (N, C). safe_target is (N,).

        # use pick/take logic
        # row_indices = mx.arange(len(target))
        # log_probs_target = log_probs[row_indices, safe_target]
        log_probs_target = mx.take_along_axis(log_probs, safe_target[:, None], axis=1).squeeze(1)

        log_probs_sum = mx.sum(log_probs, axis=1)

        loss_per_sample = - (self.confidence * log_probs_target +
                             (self.smoothing / (self.size - 1)) * (log_probs_sum - log_probs_target))

        # Mask out ignored
        loss_per_sample = mx.where(ignore, 0.0, loss_per_sample)

        total_loss = mx.sum(loss_per_sample)

        if self.normalize_length:
            denom = mx.sum(1 - ignore) # total valid tokens
        else:
            denom = batch_size

        return total_loss / denom
