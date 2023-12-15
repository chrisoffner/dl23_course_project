from typing import Dict
import torch

import h5py


def dict_to_disk(
        attn_dict: Dict[int, Dict[int, torch.Tensor]],
        filename: str
    ):
    # Write to file
    with h5py.File(f'{filename}.h5', 'w') as file:
        for t_step, res_dict in attn_dict.items():
            for res, attn_map in res_dict.items():
                dataset_name = f'{t_step}/{res}'
                file.create_dataset(dataset_name, data=attn_map)


def dict_from_disk(filename: str) -> Dict[int, Dict[int, torch.Tensor]]:
    # Read from file
    self_attn_dict = {}
    with h5py.File(f'{filename}.h5', 'r') as file:
        for t_step in file.keys():
            self_attn_dict[int(t_step)] = {}
            for res in file[t_step].keys():
                self_attn_dict[int(t_step)][int(res)] = file[t_step][res][:]

    return self_attn_dict
