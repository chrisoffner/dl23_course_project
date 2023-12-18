import torch
import torch.nn as nn


class SelfAttnExperiment(torch.nn.Module):
    def __init__(
            self,
            resolution: int,
            n_timesteps: int,
            dim: int
        ):
        super(SelfAttnExperiment, self).__init__()

        self.res = resolution
        self.n_timesteps = n_timesteps
        self.sigmoid = nn.Sigmoid()

        self.timestep_weights = torch.nn.Parameter(
            torch.randn(n_timesteps),
            requires_grad=True
        )
        
        self.layers = nn.Sequential(
        nn.Linear(dim, 128),
        nn.ReLU(),
        nn.BatchNorm1d(num_features=128),
        nn.Linear(128, 32),
        nn.ReLU(),
        nn.BatchNorm1d(num_features=32),
        nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor):
        # Shape: (batch_size, time steps, n_pixels, resolution, resolution)
        assert x.shape == (self.n_timesteps, self.res**2, self.res, self.res)

        # Multiply each time step with its corresponding scalar
        x = x * self.sigmoid(self.timestep_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)) ## why sigmoid? bc positive values only?
        
        # Sum across time steps
        x = x.sum(dim=0) # (4096,64,64)
        
        x = x.view(self.res**2,self.res**2) # (4096,4096)
        x = self.layers(x) # (4096,1)
        

        return x