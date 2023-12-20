from typing import List

import torch
import torch.nn.functional as F


class DumbExperimentProbe(torch.nn.Module):
    def __init__(
            self,
            resolution: int,
            n_timesteps: int
        ):
        super(DumbExperimentProbe, self).__init__()

        self.res = resolution
        self.n_timesteps = n_timesteps
        self.sigmoid = torch.nn.Sigmoid()

        self.timestep_weights = torch.nn.Parameter(
            torch.randn(n_timesteps),
            requires_grad=True
        )

    def forward(self, x: torch.Tensor, gt: torch.Tensor):
        # Shape: (time steps, n_pixels, resolution, resolution)
        assert x.shape == (self.n_timesteps, self.res**2, self.res, self.res)
        assert gt.shape == (self.res, self.res)

        # Multiply each time step with its corresponding scalar
        x = x * self.sigmoid(self.timestep_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))

        # Sum across time steps
        x = x.sum(dim=0)

        # Select only self-attention maps / channels that correspond to
        # salient object pixels in the ground truth
        object_mask = (gt > 0.5).flatten()

        x = x[object_mask, ...]

        # Sum across channels/pixels
        x = x.sum(dim=0)

        return x


class LinearProbe(torch.nn.Module):
    def __init__(
            self,
            n_timesteps: int = 10,
            n_channels: int = 77,
            res_combinations: torch.Tensor =torch.tensor([1, 1, 1, 1]),
            resolutions: List[int] = [8, 16, 32, 64]):
        super().__init__()
        self.n_timesteps      = n_timesteps
        self.n_channels       = n_channels
        self.res_combinations = res_combinations # Selects which resolutions to use
        self.resolutions      = resolutions

        resolutions = [8, 16, 32, 64]
        self.ts_weights = torch.nn.ParameterDict({f'{res}': self._init_weights(n_timesteps, True) for res in resolutions})
        self.ch_weights = torch.nn.ParameterDict({f'{res}': self._init_weights(n_channels) for res in resolutions})
        self.res_weights = self._init_weights(4, True)

    def forward(self, *cross_attn_maps):
        batch_size = cross_attn_maps[0].size(0)
        assert all(map(lambda x: x.size(0) == batch_size, cross_attn_maps)), "All inputs must have the same batch size"
        assert len(cross_attn_maps) == 4  # For 4 resolutions: 8, 16, 32, 64

        result = torch.zeros(batch_size, 64, 64, device=cross_attn_maps[0].device)
        for i, cross_attn in enumerate(cross_attn_maps):
            resolution = self.resolutions[i]  # 8, 16, 32, 64
            ts_weight  = self.ts_weights[str(resolution)]
            ch_weight  = self.ch_weights[str(resolution)]

            # Weighting and summing across time steps and channels
            t_sum  = (cross_attn * ts_weight[None, :, None, None, None]).sum(dim=1)
            ch_sum = (t_sum * ch_weight[None, :, None, None]).sum(dim=1)

            # Upscaling to 64x64
            if resolution != 64:
                ch_sum = self._upscale_to_64(ch_sum)
            
            # Weighting and summing across resolutions
            result += ch_sum * self.res_weights[i] * self.res_combinations[i]

        return result.sigmoid()

    def _init_weights(self, n_weights, non_negative=False):
        weight_tensor = torch.rand(n_weights) if non_negative else torch.randn(n_weights)
        return torch.nn.Parameter(weight_tensor, requires_grad=True)

    def _upscale_to_64(self, x):
        return F.interpolate(x.unsqueeze(1), size=(64, 64), mode='bicubic').squeeze(1)
