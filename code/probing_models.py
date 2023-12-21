import torch
from typing import List


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


class LinearProbe2(torch.nn.Module):
    """
    With batches.

    """

    def __init__(
        self,
        n_timesteps: int = 10,
        n_channels: int = 77,
        resolutions: List[int] = [8, 16, 32, 64],
    ):
        super().__init__()
        self.n_timesteps = n_timesteps
        self.n_channels = n_channels
        self.resolutions = resolutions
        self.n_resolutions = len(self.resolutions)


        self.ts_weights = self._init_weights(self.n_timesteps)
        self.ch_weights = self._init_weights(self.n_channels)
        self.res_weights = self._init_weights(self.n_resolutions)
        self.scale_weights = self._init_weights(1)

    def scheme_tcr(self, cross_attn_maps):
        # order of collapsing dimensions: timestep, channel, resolution
        
        x = cross_attn_maps # (b,4,10,77,64,64)

        x = x * self.ts_weights[None, None, :, None, None, None]
        x = x.sum(dim=2)  # sum across timesteps --> (b,4,77,64,64)

        x = x * self.ch_weights[None, None, :, None, None]
        x = x.sum(dim=2)  # sum across channels --> (b,4,64,64)

        x = x * self.res_weights[None, :, None, None]
        x = x.sum(dim=1) # sum across resolutions --> (b,64,64)
        
        # res = x.shape[-1]
        # x = x.reshape(-1,res*res)
        # x = x * self.scale_weights[None,:]
        # x = x.reshape(-1,res,res)

        return x

    def forward(self, cross_attn_maps):
        
        ca8,ca16,ca32,ca64 = cross_attn_maps
        assert ca8.ndim==ca16.ndim==ca32.ndim==ca64.ndim
        if ca8.ndim ==4:
            # happens if batch size is 1, otherwise ndim is 5
            ca8 = ca8.unsqueeze(0)
            ca16 = ca16.unsqueeze(0)
            ca32 = ca32.unsqueeze(0)
            ca64 = ca64.unsqueeze(0)

        ca8 = self.upsample(ca8)   # (b,10,77,64,64)    
        ca16 = self.upsample(ca16) # (b,10,77,64,64)
        ca32 = self.upsample(ca32) # (b,10,77,64,64)
        cross_attn_maps = (ca8,ca16,ca32,ca64)

        maps = torch.stack(cross_attn_maps, dim=1) # --> (b,4,10,77,64,64)

        result = self.scheme_tcr(cross_attn_maps=maps)
        return result.sigmoid() 

    def _init_weights(self, n_weights):
        weight_tensor = torch.randn(n_weights)
        return torch.nn.Parameter(weight_tensor, requires_grad=True)


    def upsample(self, x, size=64):
        # x: 3D, *4D*, 5D tensor --> dims:  mini-batch x channels x [optional depth] x [optional height] x width.
        # https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
        
        
        assert x.ndim == 5, f"Input must be 5D tensor (batch, ts, ch, res, res), got {x.shape}" # (b,10,77,res,res)
        assert x.shape[1]==self.n_timesteps
        assert x.shape[2]==self.n_channels
        assert x.shape[3]==x.shape[4], "Input must be square"

        x = x.reshape(-1, self.n_timesteps * self.n_channels, x.shape[-2], x.shape[-1]) # (b,770, res,res)
        x = torch.nn.functional.interpolate(x, size=(size,size), mode="bicubic") 
        x = x.reshape(-1, self.n_timesteps, self.n_channels, size, size) # (b,10,77,64,64)
        return x