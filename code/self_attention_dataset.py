import os
import torch
from torch.utils.data import Dataset, DataLoader
from my_utils import dict_from_disk
from PIL import Image
from torchvision import transforms

class CustomDataset(Dataset):
    def __init__(self, mask_dir,attn_maps_dir,idx_txt_file):
        self.mask_dir = mask_dir
        self.attn_maps_dir = attn_maps_dir
        with open(idx_txt_file, 'r') as f:
            self.idx_list = [line.strip() for line in f]
        

    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, idx):
        index = self.idx_list[idx]
        attn_map_dict = dict_from_disk(f"{self.attn_maps_dir}/{index}_self.h5")
        
        res = 64
        x = torch.empty(0, res**2, res, res)
        for t, self_attn_maps in attn_map_dict.items():
            self_attn = torch.from_numpy(self_attn_maps[res].reshape(-1, res, res)).unsqueeze(0)
            x = torch.cat((x, self_attn), dim=0)
        
        attn_map_tensor = x # (n_timesteps,res**2,res,res)
        
        mask = Image.open(f"{self.mask_dir}/{index}.png")
        mask = mask.resize((64,64),resample=Image.BILINEAR)

        threshold_value = 128
        transform = transforms.ToTensor()
        
        # Apply binary thresholding using the point method
        mask = mask.point(lambda x: 0 if x < threshold_value else 255)
        mask_tensor = transform(mask).squeeze(0)
        mask_tensor = mask_tensor.view(-1,1) # (4096,1)
        
        # added only for visualization
        image_dir = self.mask_dir.replace("gt","img")
        image = Image.open(f"{image_dir}/{index}.jpg")
        image_tensor = transform(image).squeeze(0)
        
        return attn_map_tensor, mask_tensor, image_tensor