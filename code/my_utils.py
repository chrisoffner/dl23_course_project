from typing import Dict
import json

import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import h5py
from pycocotools import mask as coco_mask


def dict_to_disk(
        attn_dict: Dict[int, Dict[int, torch.Tensor]],
        file_path: str
    ):
    assert file_path.endswith('.h5'), "File path must end with .h5"

    # Write to file
    with h5py.File(file_path, 'w') as file:
        for t_step, res_dict in attn_dict.items():
            for res, attn_map in res_dict.items():
                dataset_name = f'{t_step}/{res}'
                file.create_dataset(dataset_name, data=attn_map)


def dict_from_disk(file_path: str) -> Dict[int, Dict[int, torch.Tensor]]:
    assert file_path.endswith('.h5'), "Filename must end with .h5"

    # Read from file
    attn_dict = {}
    with h5py.File(file_path, 'r') as file:
        for timestep in file.keys():
            attn_dict[int(timestep)] = {}
            for res in file[timestep].keys():
                attn_dict[int(timestep)][int(res)] = file[timestep][res][:]

    return attn_dict


class DPLoss(torch.nn.Module):
    def forward(self, x: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        return (x - gt).norm()


def load_image_as_tensor(image_path: str) -> torch.Tensor:
    with Image.open(image_path) as img:
        img = img.convert('RGB')
        tensor = transforms.ToTensor()(img)

        return tensor


def resize_image_tensor(
        image_tensor: torch.Tensor,
        new_size=(512, 512)
    ) -> torch.Tensor:
    # Define the resize transformation
    resize_transform = transforms.Resize(new_size, antialias=True)

    # Check if the tensor has a channel dimension, if not add one
    if len(image_tensor.shape) == 2:
        # Add a channel dimension (assumes grayscale so only 1 channel)
        image_tensor = image_tensor.unsqueeze(0)

    # Apply the resize transformation
    resized_tensor = resize_transform(image_tensor)

    if resized_tensor.size(0) == 1:
        resized_tensor = resized_tensor.squeeze(0)

    return resized_tensor


def decode_masks_from_json(json_path: str) -> torch.Tensor:
    with open(json_path, 'r') as file:
        data = json.load(file)

    # image_height = data['image']['height']
    # image_width = data['image']['width']
    annotations = data['annotations']

    masks = []
    for annotation in annotations:
        rle = annotation['segmentation']
        binary_mask = coco_mask.decode(rle)
        masks.append(binary_mask)

    # Stack all masks into a single multi-channel array
    stacked_masks = np.stack(masks, axis=0)
    
    # Convert to a PyTorch tensor
    return torch.tensor(stacked_masks, dtype=torch.float32)
