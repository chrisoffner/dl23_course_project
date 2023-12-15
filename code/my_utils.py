from typing import Dict
import torch

import h5py


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
