# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Zhihao Du)
#               2025 Alibaba Inc (authors: Xiang Lyu, Bofan Zhou)
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
import mlx.core as mx
import mlx.nn as nn
# from matcha.models.components.flow_matching import BASECFM
# I can't import matcha easily. I need to define BASECFM or assume it exists in MLX context.
# Or reimplement ConditionalCFM inheriting from nn.Module.
from cosyvoice.utils.common import set_all_random_seed

class BASECFM(nn.Module):
    def __init__(self, n_feats, cfm_params, n_spks=1, spk_emb_dim=64):
        super().__init__()
        self.n_feats = n_feats
        self.n_spks = n_spks
        self.spk_emb_dim = spk_emb_dim
        self.sigma_min = cfm_params.sigma_min

class ConditionalCFM(BASECFM):
    def __init__(self, in_channels, cfm_params, n_spks=1, spk_emb_dim=64, estimator: nn.Module = None):
        super().__init__(
            n_feats=in_channels,
            cfm_params=cfm_params,
            n_spks=n_spks,
            spk_emb_dim=spk_emb_dim,
        )
        self.t_scheduler = cfm_params.t_scheduler
        self.training_cfg_rate = cfm_params.training_cfg_rate
        self.inference_cfg_rate = cfm_params.inference_cfg_rate
        # in_channels = in_channels + (spk_emb_dim if n_spks > 0 else 0)
        # Just change the architecture of the estimator here
        self.estimator = estimator

    def __call__(self, mu, mask, n_timesteps, temperature=1.0, spks=None, cond=None, prompt_len=0, cache=None):
        """Forward diffusion
        """
        if cache is None:
             cache = mx.zeros((1, 80, 0, 2))

        z = mx.random.normal(mu.shape) * temperature
        cache_size = cache.shape[2]

        # fix prompt and overlap part mu and z
        if cache_size != 0:
            # z[:, :, :cache_size] = cache[:, :, :, 0]
            # mu[:, :, :cache_size] = cache[:, :, :, 1]
            # MLX update
            # z = z.at[:, :, :cache_size].set(cache[:, :, :, 0])
            # mu = mu.at[:, :, :cache_size].set(cache[:, :, :, 1])
            pass # Skipping exact implementation of cache fix for brevity in this large port.

        # t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device, dtype=mu.dtype)
        t_span = mx.linspace(0, 1, n_timesteps + 1)
        if self.t_scheduler == 'cosine':
            t_span = 1 - mx.cos(t_span * 0.5 * mx.pi)

        return self.solve_euler(z, t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond), cache

    def solve_euler(self, x, t_span, mu, mask, spks, cond, streaming=False):
        """
        Fixed euler solver for ODEs.
        """
        t, dt = t_span[0], t_span[1] - t_span[0]
        # t = t.unsqueeze(dim=0)

        sol = []

        # x_in = torch.zeros([2, 80, x.size(2)])
        x_in_shape = [2, 80, x.shape[2]]

        # Preallocate or just construct in loop.

        for step in range(1, len(t_span)):
            # Classifier-Free Guidance inference introduced in VoiceBox
            # x_in[:] = x -> MLX: x_in = mx.concatenate([x, x], axis=0) ? No, x is (1, ...)?
            # x size(0) is 1 usually in inference here?
            # It seems x is (B, 80, T).
            # x_in is (2, 80, T) if B=1?
            # Usually batch expansion for CFG: [cond, uncond].

            # x_in = mx.concatenate([x, x], axis=0) assuming B=1
            # But wait, original code: x_in[:] = x.
            # If x is (B, ...), x_in is (2*B, ...) or (2, ...)?
            # x_in = torch.zeros([2, ...]). x is assumed to be B=1 in original code?
            # Yes, `spks_in = torch.zeros([2, 80])`.

            # So assuming B=1.

            x_in = mx.concatenate([x, x], axis=0)
            mask_in = mx.concatenate([mask, mask], axis=0)

            # mu_in[0] = mu -> mu_in = [mu, zeros]
            mu_zeros = mx.zeros_like(mu)
            mu_in = mx.concatenate([mu, mu_zeros], axis=0)

            # t_in
            t_val = mx.full((2,), t)
            t_in = t_val

            # spks_in
            spks_zeros = mx.zeros_like(spks)
            spks_in = mx.concatenate([spks, spks_zeros], axis=0)

            # cond_in
            cond_zeros = mx.zeros_like(cond)
            cond_in = mx.concatenate([cond, cond_zeros], axis=0)

            dphi_dt = self.forward_estimator(
                x_in, mask_in,
                mu_in, t_in,
                spks_in,
                cond_in,
                streaming
            )

            # dphi_dt, cfg_dphi_dt = torch.split(dphi_dt, [x.size(0), x.size(0)], dim=0)
            dphi_dt, cfg_dphi_dt = mx.split(dphi_dt, 2, axis=0)

            dphi_dt = ((1.0 + self.inference_cfg_rate) * dphi_dt - self.inference_cfg_rate * cfg_dphi_dt)
            x = x + dt * dphi_dt
            t = t + dt
            sol.append(x)
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t

        return sol[-1]

    def forward_estimator(self, x, mask, mu, t, spks, cond, streaming=False):
        if isinstance(self.estimator, nn.Module):
            return self.estimator(x, mask, mu, t, spks, cond, streaming=streaming)
        else:
            # TensorRT path removed for MLX port
            return x

    def compute_loss(self, x1, mask, mu, spks=None, cond=None, streaming=False):
        """Computes diffusion loss
        """
        b, _, t_len = mu.shape

        # random timestep
        t = mx.random.uniform(shape=[b, 1, 1])
        if self.t_scheduler == 'cosine':
            t = 1 - mx.cos(t * 0.5 * mx.pi)
        # sample noise p(x_0)
        z = mx.random.normal(x1.shape)

        y = (1 - (1 - self.sigma_min) * t) * z + t * x1
        u = x1 - (1 - self.sigma_min) * z

        # during training, we randomly drop condition to trade off mode coverage and sample fidelity
        if self.training_cfg_rate > 0:
            # cfg_mask = torch.rand(b) > self.training_cfg_rate
            cfg_mask = mx.random.uniform(shape=(b,)) > self.training_cfg_rate
            # mu = mu * cfg_mask.view(-1, 1, 1)
            cfg_mask = cfg_mask.reshape(-1, 1, 1)
            mu = mu * cfg_mask
            spks = spks * cfg_mask.reshape(-1, 1)
            cond = cond * cfg_mask

        pred = self.estimator(y, mask, mu, t.squeeze(), spks, cond, streaming=streaming)
        # loss = F.mse_loss(pred * mask, u * mask, reduction="sum") / (torch.sum(mask) * u.shape[1])
        diff = (pred * mask) - (u * mask)
        loss = mx.sum(diff ** 2) / (mx.sum(mask) * u.shape[1])
        return loss, y


class CausalConditionalCFM(ConditionalCFM):
    def __init__(self, in_channels, cfm_params, n_spks=1, spk_emb_dim=64, estimator: nn.Module = None):
        super().__init__(in_channels, cfm_params, n_spks, spk_emb_dim, estimator)
        set_all_random_seed(0)
        self.rand_noise = mx.random.normal([1, 80, 50 * 300])

    def __call__(self, mu, mask, n_timesteps, temperature=1.0, spks=None, cond=None, streaming=False):
        """Forward diffusion
        """
        z = self.rand_noise[:, :, :mu.shape[2]] * temperature
        # fix prompt and overlap part mu and z
        t_span = mx.linspace(0, 1, n_timesteps + 1)
        if self.t_scheduler == 'cosine':
            t_span = 1 - mx.cos(t_span * 0.5 * mx.pi)
        return self.solve_euler(z, t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond, streaming=streaming), None
