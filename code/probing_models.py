import torch


class DumbExperimentProbe(torch.nn.Module):
    def __init__(self, resolution: int, n_timesteps: int):
        super(DumbExperimentProbe, self).__init__()

        self.res = resolution
        self.n_timesteps = n_timesteps
        self.sigmoid = torch.nn.Sigmoid()

        self.timestep_weights = torch.nn.Parameter(
            torch.randn(n_timesteps), requires_grad=True
        )

    def forward(self, x: torch.Tensor, gt: torch.Tensor):
        # Shape: (time steps, n_pixels, resolution, resolution)
        assert x.shape == (self.n_timesteps, self.res**2, self.res, self.res)
        assert gt.shape == (self.res, self.res)

        # Multiply each time step with its corresponding scalar
        x = x * self.sigmoid(
            self.timestep_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        )

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
    def __init__(self, n_timesteps: int = 10):
        super().__init__()

        # self.instance_norm = torch.nn.InstanceNorm2d(77)

        self.n_timesteps = n_timesteps
        # self.sigmoid = torch.nn.Identity()#torch.nn.Sigmoid()

        # Define timestep weights for each resolution
        # Each timestep weight is a scalar that gets multiplied with
        # the (77, res, res) cross-attention maps for that timestep
        self.ts_weights_8 = self._init_weights(n_timesteps)
        self.ts_weights_16 = self._init_weights(n_timesteps)
        self.ts_weights_32 = self._init_weights(n_timesteps)
        self.ts_weights_64 = self._init_weights(n_timesteps)

        # Define channel weights for each resolution
        # Each channel weight is a scalar that gets multiplied with
        # the (res, res) cross-attention map for that channel
        self.ch_weights_8 = self._init_weights(77)
        self.ch_weights_16 = self._init_weights(77)
        self.ch_weights_32 = self._init_weights(77)
        self.ch_weights_64 = self._init_weights(77)

        # Define resolution weights for each resolution
        # Each resolution weight is a scalar that gets multiplied with the
        # (64, 64) interpolated cross-attention map for that resolution
        self.res_weights = self._init_weights(4)

    def forward(
        self,
        cross_attn_8: torch.Tensor,
        cross_attn_16: torch.Tensor,
        cross_attn_32: torch.Tensor,
        cross_attn_64: torch.Tensor,
    ) -> torch.Tensor:
        assert cross_attn_8.shape == (self.n_timesteps, 77, 8, 8)
        assert cross_attn_16.shape == (self.n_timesteps, 77, 16, 16)
        assert cross_attn_32.shape == (self.n_timesteps, 77, 32, 32)
        assert cross_attn_64.shape == (self.n_timesteps, 77, 64, 64)

        # Normalize cross-attention maps
        # cross_attn_8 = self.instance_norm(cross_attn_8)
        # cross_attn_16 = self.instance_norm(cross_attn_16)
        # cross_attn_32 = self.instance_norm(cross_attn_32)
        # cross_attn_64 = self.instance_norm(cross_attn_64)

        # Multiply each time step with its corresponding scalar
        t_weighted_8 = cross_attn_8 * self.ts_weights_8[:, None, None, None]
        t_weighted_16 = cross_attn_16 * self.ts_weights_16[:, None, None, None]
        t_weighted_32 = cross_attn_32 * self.ts_weights_32[:, None, None, None]
        t_weighted_64 = cross_attn_64 * self.ts_weights_64[:, None, None, None]

        # Compute weighted sum across time steps for each resolution
        t_sum_8 = t_weighted_8.sum(dim=0)
        t_sum_16 = t_weighted_16.sum(dim=0)
        t_sum_32 = t_weighted_32.sum(dim=0)
        t_sum_64 = t_weighted_64.sum(dim=0)

        # Multiply each channel with its corresponding scalar
        ch_weighted_8 = t_sum_8 * self.ch_weights_8[:, None, None]
        ch_weighted_16 = t_sum_16 * self.ch_weights_16[:, None, None]
        ch_weighted_32 = t_sum_32 * self.ch_weights_32[:, None, None]
        ch_weighted_64 = t_sum_64 * self.ch_weights_64[:, None, None]

        # Compute weighted sum across channels for each resolution
        ch_sum_8 = ch_weighted_8.sum(dim=0)
        ch_sum_16 = ch_weighted_16.sum(dim=0)
        ch_sum_32 = ch_weighted_32.sum(dim=0)
        ch_sum_64 = ch_weighted_64.sum(dim=0)

        # Interpolate cross-attention maps to (64, 64) resolution
        ch_sum_8 = self._upscale_to_64(ch_sum_8)
        ch_sum_16 = self._upscale_to_64(ch_sum_16)
        ch_sum_32 = self._upscale_to_64(ch_sum_32)

        # Multiply each resolution with its corresponding scalar
        res_weighted_8 = ch_sum_8 * self.res_weights[0]
        res_weighted_16 = ch_sum_16 * self.res_weights[1]
        res_weighted_32 = ch_sum_32 * self.res_weights[2]
        res_weighted_64 = ch_sum_64 * self.res_weights[3]

        # Compute weighted sum across resolutions
        result = res_weighted_8 + res_weighted_16 + res_weighted_32 + res_weighted_64

        assert result.shape == (64, 64)

        return result.sigmoid()

    def _init_weights(self, n_weights: int) -> torch.nn.Parameter:
        return torch.nn.Parameter(torch.randn(n_weights), requires_grad=True)

    def _upscale_to_64(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.interpolate(
            x[None, None, :, :], size=(64, 64), mode="bicubic"
        ).squeeze()
