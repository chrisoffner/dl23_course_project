from time import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from my_utils import dict_from_disk, load_image_as_tensor
from cross_attention_dataset import CrossAttentionDataset
from probing_models import LinearProbe

start_time = time()

# ========================= HYPERPARAMETER DEFINITIONS =========================


N_EPOCHS   = 200  # Number of epochs
N_RUNS     = 25   # Number of runs


# ======================== PREPARATION OF TRAINING DATA ========================

# Path to the directory containing cross-attention maps and ground truth masks
FEATURE_DIR = Path("/Users/chrisoffner3d/Downloads/custom_datasets/small/cross_attn_maps")
GT_DIR = Path("/Users/chrisoffner3d/Documents/Dev/ETH/DL Code/dl_project/DL Project/data/custom_datasets/small/gt")

# Filter files in directory for the cross-attention maps
cross_attn_filenames = sorted([f for f in FEATURE_DIR.glob("*.h5") if f.stem.endswith("_cross")])

# Load the cross-attention maps
cross_attn_maps = [dict_from_disk(str(f)) for f in tqdm(cross_attn_filenames, desc="Loading Cross-Attention Maps")]

# Load the ground truth masks for the cross-attention maps as (64, 64) tensors
filenames = map(lambda path: path.stem, cross_attn_filenames)
gt_paths = sorted([GT_DIR / f"{name.split('_cross')[0]}.png" for name in filenames])
gt_segmentations = [load_image_as_tensor(path, True) for path in gt_paths]

# Create dataset with batch size
dataset = CrossAttentionDataset(cross_attn_maps, gt_segmentations)

# Split dataset into training and validation
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create data loaders
train_loader = DataLoader(train_dataset, shuffle=True)
val_loader   = DataLoader(val_dataset,   shuffle=False)


# ========================== INITIALISATION OF RECORDS =========================


n_measurements_per_run = N_EPOCHS * len(train_loader) + 1
resolutions = [8, 16, 32, 64]

# Final of each record dict: (n_runs, n_measurements_per_run, n_weights)
ts_weights_record  = { res : torch.empty(0, n_measurements_per_run, 10) for res in resolutions }
ch_weights_record  = { res : torch.empty(0, n_measurements_per_run, 77) for res in resolutions }
res_weights_record = torch.empty(0, n_measurements_per_run,  4)

# Initialise losses
train_losses = torch.empty(N_RUNS, N_EPOCHS)
val_losses   = torch.empty(N_RUNS, N_EPOCHS)


# ================================== TRAINING ==================================


for run in tqdm(range(N_RUNS), desc="Runs"):
    # Create model, optimizer and loss function
    model = LinearProbe()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = torch.nn.BCELoss()

    # Extract initial weights
    ts_weights  = { res : model.ts_weights[str(res)].unsqueeze(0).detach() for res in resolutions }
    ch_weights  = { res : model.ch_weights[str(res)].unsqueeze(0).detach() for res in resolutions }
    res_weights = model.res_weights.unsqueeze(0).detach()

    for epoch in range(N_EPOCHS):
        # Training phase
        model.train()
        train_loss = 0
        for cross_attn_8, cross_attn_16, cross_attn_32, cross_attn_64, gt in train_loader:
            optimizer.zero_grad()

            # Forward pass
            output = model(cross_attn_8, cross_attn_16, cross_attn_32, cross_attn_64)
            loss = criterion(output, gt)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Record the resolution weights for later analysis
            for res in resolutions:
                ts_weights[res] = torch.cat((ts_weights[res], model.ts_weights[str(res)].unsqueeze(0).detach()), dim=0)
                ch_weights[res] = torch.cat((ch_weights[res], model.ch_weights[str(res)].unsqueeze(0).detach()), dim=0)

            res_weights = torch.cat((res_weights, model.res_weights.unsqueeze(0).detach()), dim=0)

            # Update the progress bar description
            train_loss += loss.item() / len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for cross_attn_8, cross_attn_16, cross_attn_32, cross_attn_64, gt in val_loader:
                # Calculate and accumulate validation loss
                output = model(cross_attn_8, cross_attn_16, cross_attn_32, cross_attn_64)
                loss = criterion(output, gt)
                val_loss += loss.item() / len(val_loader)

        # Record the losses for this epoch
        train_losses[run, epoch] = train_loss
        val_losses[run, epoch] = val_loss

    # Record the weights for this run
    for res in resolutions:
        ts_weights_record[res] = torch.cat((ts_weights_record[res], ts_weights[res].unsqueeze(0)), dim=0)
        ch_weights_record[res] = torch.cat((ch_weights_record[res], ch_weights[res].unsqueeze(0)), dim=0)
    res_weights_record = torch.cat((res_weights_record, res_weights.unsqueeze(0)), dim=0)


# =========================== WRITING RECORDS TO DISK ==========================

DATASET = "small" # "allsizes", "small", "medium", "large"

# Save the time step and channel weights for each resolution
for res in resolutions:
    torch.save(ts_weights_record[res], f"{DATASET}_{N_RUNS}runs_{N_EPOCHS}epochs_ts_weights_{res}.pt")
    torch.save(ch_weights_record[res], f"{DATASET}_{N_RUNS}runs_{N_EPOCHS}epochs_ch_weights_{res}.pt")

# Save the resolution weights
torch.save(res_weights_record, f"{DATASET}_{N_RUNS}runs_{N_EPOCHS}epochs_res_weights.pt")

# Save the losses
torch.save(train_losses, f"{DATASET}_{N_RUNS}runs_{N_EPOCHS}epochs_train_losses.pt")
torch.save(val_losses, f"{DATASET}_{N_RUNS}runs_{N_EPOCHS}epochs_val_losses.pt")


# ========================= PRINTING EXECUTION DURATION ========================


end_time = time()

# Calculate total duration
total_time = end_time - start_time

# Convert to hours, minutes, and seconds
hours   = int(total_time // 3600)
minutes = int((total_time % 3600) // 60)
seconds = total_time % 60

print(f"Execution Time: {hours} hours, {minutes} minutes, and {seconds:.2f} seconds")
