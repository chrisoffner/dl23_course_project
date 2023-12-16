from typing import Dict, List, Tuple
from pathlib import Path

import torch

class CrossAttentionDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            cross_attn_maps: List[Dict[int, Dict[int, torch.Tensor]]],
            gt_segmentations: List[torch.Tensor]
        ):
        assert len(cross_attn_maps) == len(gt_segmentations)
        assert all([len(img) == 10 for img in cross_attn_maps])
        assert all(gt.shape == (64, 64) for gt in gt_segmentations)

        self.cross_attn_8  = []
        self.cross_attn_16 = []
        self.cross_attn_32 = []
        self.cross_attn_64 = []
        self.gt = gt_segmentations
        self.timesteps: List[int] = sorted(cross_attn_maps[0].keys())

        # Preprocess and reshape data
        for img in cross_attn_maps:
            list_8  = [img[t][ 8].reshape( 8,  8, -1) for t in self.timesteps]
            list_16 = [img[t][16].reshape(16, 16, -1) for t in self.timesteps]
            list_32 = [img[t][32].reshape(32, 32, -1) for t in self.timesteps]
            list_64 = [img[t][64].reshape(64, 64, -1) for t in self.timesteps]

            attn_8  = torch.stack(list_8).permute(0, 3, 1, 2)
            attn_16 = torch.stack(list_16).permute(0, 3, 1, 2)
            attn_32 = torch.stack(list_32).permute(0, 3, 1, 2)
            attn_64 = torch.stack(list_64).permute(0, 3, 1, 2)

            self.cross_attn_8.append(attn_8)
            self.cross_attn_16.append(attn_16)
            self.cross_attn_32.append(attn_32)
            self.cross_attn_64.append(attn_64)

        assert len(self.cross_attn_8)  == len(self.cross_attn_16) == \
               len(self.cross_attn_32) == len(self.cross_attn_64)

    def __len__(self):
        return len(self.cross_attn_8)

    def __getitem__(
            self,
            idx: int
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.cross_attn_8[idx],  self.cross_attn_16[idx], \
               self.cross_attn_32[idx], self.cross_attn_64[idx], self.gt[idx]
