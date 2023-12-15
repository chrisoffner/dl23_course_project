import torch


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
