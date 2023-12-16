import torch
import torch.nn.functional as F


class TimeStepWeightsNet(torch.nn.Module):
    def __init__(
            self,
            resolution: int,
            n_timesteps: int
        ):
        super(TimeStepWeightsNet, self).__init__()

        self.res = resolution
        self.n_timesteps = n_timesteps
        self.sigmoid = torch.nn.Sigmoid()

        # These are the weights that we want to learn for each time step
        self.timestep_weights = torch.nn.Parameter(
            torch.randn(n_timesteps),
            requires_grad=True
        )

        self.linear_layer = torch.nn.Linear(in_features=self.res**2, out_features=1)

    def forward(self, x: torch.Tensor):
        # Shape: (time steps, n_pixels, resolution, resolution)
        assert x.shape == (self.n_timesteps, self.res**2, self.res, self.res)
        # assert gt.shape == (self.res, self.res)

        # Multiply each time step with its corresponding scalar
        x = x * self.sigmoid(self.timestep_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))

        # Sum across time steps, after this the size should be (res**2, res, res)
        x = x.sum(dim=0)
        
        # reshape to (4096, 4096)
        x = x.reshape((self.res**2, self.res**2)).T
        
        # Linear layer that somehow gives me (4096, 1) 
        x = self.sigmoid(self.linear_layer(x))
        
        # Return the output
        return x

