from pathlib import Path
import json

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from my_utils import dict_from_disk, load_image_as_tensor
from cross_attention_dataset import CrossAttentionDataset
from probing_models import LinearProbe

# LOAD CROSS-ATTENTION MAPS FROM DISK

# Set the path to the directory containing the cross-attention maps
FEATURE_DIR = Path(
    "/Users/chrisoffner3d/Library/Mobile Documents/com~apple~CloudDocs/DL_project/ECSSD_resized/features/cross_attn"
)
GT_DIR = Path("../data/ECSSD_resized/gt")

# Filter files in directory for the cross-attention maps
cross_attn_filenames = sorted(
    [f for f in FEATURE_DIR.glob("*.h5") if f.stem.endswith("_cross")]
)

# Load the cross-attention maps
cross_attn_maps = [dict_from_disk(str(f)) for f in tqdm(cross_attn_filenames)]

# Load the ground truth masks for the cross-attention maps as (64, 64) tensors
base_names = map(lambda path: path.stem, cross_attn_filenames)
gt_paths = sorted(
    [GT_DIR / f"{base_name.split('_')[0]}.png" for base_name in base_names]
)
gt_segmentations = [load_image_as_tensor(path, True) for path in gt_paths]

# CREATE DATASET AND DATA LOADER
dataset = CrossAttentionDataset(cross_attn_maps, gt_segmentations)
data_loader = DataLoader(dataset, shuffle=True)

# COMBINATIONS OF RESOLUTIONS
res_combinations = torch.tensor(
    [
        # [1, 0, 0, 0], # only one resolution
        # [0, 1, 0, 0],
        # [0, 0, 1, 0],
        # [0, 0, 0, 1],
        # [1, 1, 0, 0], # two resolutions
        # [1, 0, 1, 0],
        # [1, 0, 0, 1],
        # [0, 1, 1, 0],
        # [0, 1, 0, 1],
        # [0, 0, 1, 1],
        [1, 1, 1, 0],  # three resolutions
        [1, 1, 0, 1],
        [1, 0, 1, 1],
        [0, 1, 1, 1],
        [1, 1, 1, 1],  # all four resolutions
    ]  # 8 16 32 64
)  

# TRAIN PROBE
N_EPOCHS = 60  # Number of epochs
N_RUNS = 3  # Number of runs per resolution combination

results_dict = {
    ",".join(str(num) for num in res_combination.tolist()): []
    for res_combination in res_combinations
}

for res_combination in tqdm(res_combinations, desc="Resolution combinations"):
    for i in range(N_RUNS):
        # CREATE PROBE
        model = LinearProbe(res_combinations=res_combination)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        criterion = torch.nn.BCELoss()

        epoch_losses = []
        for epoch in tqdm(range(N_EPOCHS), desc=f"Run {i}/{N_RUNS} Epochs"):
            epoch_loss = 0

            for cross_attn_maps, gt in data_loader:
                optimizer.zero_grad()

                # Forward pass
                output = model(*cross_attn_maps)
                loss = criterion(output, gt)

                epoch_loss += loss.item()

                # Backward pass
                loss.backward()
                optimizer.step()

            epoch_losses.append(epoch_loss / len(data_loader))

        combination_str = ",".join(str(num) for num in res_combination.tolist())
        results_dict[combination_str].append(epoch_losses)


# Write results to disk
with open("../data/60ep_three_four_res_losses.json", "w") as write_file:
    json.dump(results_dict, write_file)
