from pathlib import Path
import json

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from my_utils import dict_from_disk, load_image_as_tensor
from cross_attention_dataset import CrossAttentionDataset
from probing_models import LinearProbe

DEVICE = torch.device("cpu")

# LOAD CROSS-ATTENTION MAPS FROM DISK

# Set the path to the directory containing the cross-attention maps
FEATURE_DIR = Path("/Users/chrisoffner3d/Library/Mobile Documents/com~apple~CloudDocs/DL_project/ECSSD_resized/features/cross_attn")
GT_DIR = Path("/Users/chrisoffner3d/Documents/Dev/ETH/DL Code/dl_project/DL Project/data/ECSSD_resized/gt")

# Filter files in directory for the cross-attention maps
cross_attn_filenames = sorted([f for f in FEATURE_DIR.glob("*.h5") if f.stem.endswith("_cross")])

# Load the cross-attention maps
cross_attn_maps = [dict_from_disk(str(f)) for f in tqdm(cross_attn_filenames)]

# Load the ground truth masks for the cross-attention maps as (64, 64) tensors
base_names = map(lambda path: path.stem, cross_attn_filenames)
gt_paths = sorted([GT_DIR / f"{base_name.split('_')[0]}.png" for base_name in base_names])
gt_segmentations = [load_image_as_tensor(path, True) for path in gt_paths]

# CREATE DATASET AND DATA LOADER
dataset = CrossAttentionDataset(cross_attn_maps, gt_segmentations)
data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

# COMBINATIONS OF RESOLUTIONS
res_combinations = torch.tensor([
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
    [1, 1, 1, 0], # three resolutions
    [1, 1, 0, 1],
    [1, 0, 1, 1],
    [0, 1, 1, 1],
    [1, 1, 1, 1], # all four resolutions
])#  8 16 32 64

# TRAIN PROBE
n_epochs = 60      # Number of epochs
# lambda_l1 = 0 # 5e-4   # L1 regularisation strength

results_dict = { ','.join(str(num) for num in res_combination.tolist()) : [] for res_combination in res_combinations }

for res_combination in tqdm(res_combinations):
    for i in tqdm(range(3)):

        # CREATE PROBE
        model = LinearProbe(res_combinations=res_combination).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        criterion = torch.nn.BCELoss()

        epoch_losses = []
        for epoch in range(n_epochs):
            epoch_loss = 0
            # with tqdm(data_loader, desc=f"Epoch {epoch}") as tepoch:
            for cross_attn_8, cross_attn_16, cross_attn_32, cross_attn_64, gt in data_loader:
                cross_attn_8  = cross_attn_8.squeeze().to(DEVICE)
                cross_attn_16 = cross_attn_16.squeeze().to(DEVICE)
                cross_attn_32 = cross_attn_32.squeeze().to(DEVICE)
                cross_attn_64 = cross_attn_64.squeeze().to(DEVICE)
                gt = gt.to(DEVICE)

                optimizer.zero_grad()

                # Forward pass
                output = model(cross_attn_8, cross_attn_16, cross_attn_32, cross_attn_64).unsqueeze(0)
                loss = criterion(output, gt)

                # L1 regularisation
                # l1_loss = 0
                # for p in model.parameters():
                #     l1_loss += p.abs().sum()

                # loss += lambda_l1 * l1_loss
                epoch_loss += loss.item() / len(data_loader)

                # Backward pass
                loss.backward()
                optimizer.step()

                # Update the progress bar description
                # tepoch.set_description(f"Epoch {epoch} Loss: {epoch_loss:.4f}\t")
            
            epoch_losses.append(epoch_loss)

        combination_str = ','.join(str(num) for num in res_combination.tolist())
        results_dict[combination_str].append(epoch_losses)


# Write results to disk
with open("../data/60ep_three_four_res_losses.json", "w") as write_file:
    json.dump(results_dict, write_file)
