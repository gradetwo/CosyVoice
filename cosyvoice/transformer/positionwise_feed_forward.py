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
"""Positionwise feed forward layer definition."""

import mlx.core as mx
import mlx.nn as nn


class PositionwiseFeedForward(nn.Module):
    """Positionwise feed forward layer.

    FeedForward are appied on each position of the sequence.
    The output dim is same with the input dim.

    Args:
        idim (int): Input dimenstion.
        hidden_units (int): The number of hidden units.
        dropout_rate (float): Dropout rate.
        activation (nn.Module): Activation function
    """

    def __init__(
            self,
            idim: int,
            hidden_units: int,
            dropout_rate: float,
            activation: nn.Module = nn.ReLU(),
    ):
        """Construct a PositionwiseFeedForward object."""
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(idim, hidden_units)
        self.activation = activation
        self.dropout = nn.Dropout(dropout_rate)
        self.w_2 = nn.Linear(hidden_units, idim)

    def __call__(self, xs: mx.array) -> mx.array:
        """Forward function.

        Args:
            xs: input tensor (B, L, D)
        Returns:
            output tensor, (B, L, D)
        """
        return self.w_2(self.dropout(self.activation(self.w_1(xs))))


class MoEFFNLayer(nn.Module):
    """
    Mixture of expert with Positionwise feed forward layer
    See also figure 1 in https://arxiv.org/pdf/2305.15663.pdf
    The output dim is same with the input dim.

    Modified from https://github.com/Lightning-AI/lit-gpt/pull/823
                  https://github.com/mistralai/mistral-src/blob/b46d6/moe_one_file_ref.py#L203-L219
    Args:
        n_expert: number of expert.
        n_expert_per_token: The actual number of experts used for each frame
        idim (int): Input dimenstion.
        hidden_units (int): The number of hidden units.
        dropout_rate (float): Dropout rate.
        activation (nn.Module): Activation function
    """

    def __init__(
            self,
            n_expert: int,
            n_expert_per_token: int,
            idim: int,
            hidden_units: int,
            dropout_rate: float,
            activation: nn.Module = nn.ReLU(),
    ):
        super(MoEFFNLayer, self).__init__()
        self.gate = nn.Linear(idim, n_expert, bias=False)
        self.experts = [
            PositionwiseFeedForward(idim, hidden_units, dropout_rate,
                                    activation) for _ in range(n_expert)
        ]
        self.n_expert_per_token = n_expert_per_token

    def __call__(self, xs: mx.array) -> mx.array:
        """Foward function.
        Args:
            xs: input tensor (B, L, D)
        Returns:
            output tensor, (B, L, D)

        """
        B, L, D = xs.shape
        # batch size, sequence length, embedding dimension (idim)
        xs_reshaped = xs.reshape(-1, D)  # (B*L, D)
        router = self.gate(xs_reshaped)  # (B*L, n_expert)

        # MLX doesn't have topk with indices yet?
        # Actually it does: mx.topk
        # But wait, mx.topk returns only values? No, checking docs or usage.
        # It usually returns values. Indices can be got via argsort.

        # Using argpartition or argsort.
        # indices = mx.argpartition(router, -self.n_expert_per_token, axis=1)[:, -self.n_expert_per_token:]
        # This gives indices of top K elements but not necessarily sorted.

        # Or I can use sort and take the last K.
        # indices = mx.argsort(router, axis=1)[:, -self.n_expert_per_token:]

        # logits = mx.take_along_axis(router, indices, axis=1)

        # Since n_expert_per_token is small, sorting is fine.

        all_indices = mx.argsort(router, axis=1)
        indices = all_indices[:, -self.n_expert_per_token:] # (B*L, n_expert_per_token)

        # We need logits corresponding to these indices
        # row_indices = mx.arange(indices.shape[0])[:, None]
        # logits = router[row_indices, indices] # This works in numpy, maybe in MLX?

        # In MLX, we can use take_along_axis equivalent if it exists, or fancy indexing.
        # router is (N, E). indices is (N, K).
        # We want (N, K) output.

        # router: (B*L, n_expert)
        # indices: (B*L, n_expert_per_token)

        # Create offsets
        # offsets = mx.arange(0, router.size, router.shape[1])
        # flat_indices = indices + offsets[:, None]
        # logits = router.flatten()[flat_indices]

        # Simplified:
        row_indices = mx.arange(router.shape[0])[:, None]
        # In MLX currently we might need to do:
        # logits = router[row_indices, indices] # This should work.

        logits = mx.take_along_axis(router, indices, axis=1)

        weights = nn.softmax(logits, axis=1) # (B*L, n_expert_per_token)

        output = mx.zeros_like(xs_reshaped)  # (B*L, D)

        # Iterate over experts
        # In PyTorch:
        # for i, expert in enumerate(self.experts):
        #     mask = indices == i
        #     batch_idx, ith_expert = torch.where(mask)
        #     output[batch_idx] += weights[batch_idx, ith_expert, None] * expert(xs[batch_idx])

        # This relies on dynamic control flow and masking which is slow but fine for now.
        # MLX graph compilation might struggle with dynamic loops if not careful, but eager mode is fine.

        for i, expert in enumerate(self.experts):
            # mask = indices == i # (B*L, n_expert_per_token)

            # Since MLX doesn't support torch.where(condition) returning indices directly same way?
            # mx.where(cond, x, y) is elementwise.
            # To get indices where condition is true:
            # indices_of_mask = mx.array(np.where(np.array(mask)))

            # This requires converting to numpy which syncs GPU/CPU.
            # Ideally we want to stay in MLX.

            # Alternative: Weighted sum of all experts?
            # But MoE is sparse.

            # Loop over experts.
            # For expert i, we want to find which tokens routed to it.

            # Let's try to stick to the logic.
            mask = (indices == i) # (B*L, K)

            # If mask is all false, skip
            if not mask.any():
                continue

            # Flatten mask to find which rows in xs_reshaped actuall use this expert
            # A token might select expert i in one of its K slots.

            # rows_using_expert = mask.any(axis=1) # (B*L,)

            # This is tricky in MLX without `torch.where` equivalent for indices.
            # But we can assume eager execution for now.

            # Let's use numpy for the indices calculation part to be safe and simple,
            # as long as we don't break the graph too much (params are in MLX).

            # But wait, we need to backprop through weights.

            # Let's try to implement a dense version (multiplying by zero) if K is small?
            # Or use scatter_add? MLX doesn't have scatter_add easily?

            # Actually, `output[batch_idx] += ...` is scatter add.

            # Let's look at how we can implement this.

            # indices has shape (N, K). Values in [0, n_expert-1].
            # weights has shape (N, K).

            # We can construct a sparse matrix or just loop.

            # Let's use a loop over tokens? No, too slow.

            # Loop over experts is correct (N_experts is small, e.g. 8 or 16).

            # For expert i:
            # We want to identify tokens that selected i.
            # mask = (indices == i) # (N, K)

            # rows, cols = np.where(mask)
            # We can use numpy for indices, assuming `indices` (the array) is not diff-able (it's from argmax/argsort).

            import numpy as np
            indices_np = np.array(indices)
            mask_np = (indices_np == i)
            if not np.any(mask_np):
                continue

            batch_idx_np, ith_expert_np = np.where(mask_np)

            batch_idx = mx.array(batch_idx_np)
            ith_expert = mx.array(ith_expert_np)

            # Input to expert: xs_reshaped[batch_idx]
            selected_xs = xs_reshaped[batch_idx]

            expert_out = expert(selected_xs)

            # Weighting
            # weights[batch_idx, ith_expert] -> (M,)
            w = weights[batch_idx, ith_expert][:, None] # (M, 1)

            weighted_expert_out = w * expert_out

            # output[batch_idx] += weighted_expert_out
            # MLX arrays are immutable in-place?
            # output[indices] = val is not supported for in-place modification if it's not a variable?
            # But we can update `output`.

            # scatter_add is what we need.
            # output = output + scatter(weighted_expert_out, batch_idx)

            # MLX has index_add?
            # output.at[batch_idx].add(weighted_expert_out)
            output = output.at[batch_idx].add(weighted_expert_out)

        return output.reshape(B, L, D)
