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


N_EPOCHS = 100  # Number of epochs
N_SUBSETS = 10  # Number of training subsets
VAL_SIZE = 50  # Size of validation set
DATASET = "small"  # "allsizes", "small", "medium", "large"


# ======================== PREPARATION OF TRAINING DATA ========================


# Path to the directory containing cross-attention maps and ground truth masks
FEATURE_DIR = Path(
    "/Users/chrisoffner3d/Downloads/custom_datasets/small/cross_attn_maps"
)
GT_DIR = Path("../data/custom_datasets/small/gt")

# Filter files in directory for the cross-attention maps
cross_attn_filenames = sorted(
    [f for f in FEATURE_DIR.glob("*.h5") if f.stem.endswith("_cross")]
)

# Load the cross-attention maps
cross_attn_maps = [
    dict_from_disk(str(f))
    for f in tqdm(cross_attn_filenames, desc="Loading Cross-Attention Maps")
]

# Load the ground truth masks for the cross-attention maps as (64, 64) tensors
filenames = map(lambda path: path.stem, cross_attn_filenames)
gt_paths = sorted([GT_DIR / f"{name.split('_cross')[0]}.png" for name in filenames])
gt_segmentations = [load_image_as_tensor(path, True) for path in gt_paths]

# Create dataset with batch size
dataset = CrossAttentionDataset(cross_attn_maps, gt_segmentations)

assert len(dataset) > VAL_SIZE, "Validation set size is larger than dataset size"

# Split dataset into training and validation
train_size = len(dataset) - VAL_SIZE
train_dataset, val_dataset = random_split(dataset, [train_size, VAL_SIZE])

# Further split training dataset into five subsets
subset_size = train_size // N_SUBSETS
subset_sizes = [subset_size] * N_SUBSETS

# Handle the case where the dataset size is not evenly divisible
remainder = train_size - subset_size * N_SUBSETS
for i in range(remainder):
    subset_sizes[i] += 1

# Create disjoint training subsets
training_subsets = random_split(train_dataset, subset_sizes)

# Create data loaders
val_loader = DataLoader(val_dataset, shuffle=False)
train_loaders = [DataLoader(subset, shuffle=True) for subset in training_subsets]


# ========================== INITIALISATION OF RECORDS =========================


train_losses = torch.empty(N_SUBSETS, N_EPOCHS)
val_losses = torch.empty(N_SUBSETS, N_EPOCHS)
trained_models = [None] * N_SUBSETS


# ================================== TRAINING ==================================


for subset_idx, train_loader in enumerate(train_loaders):
    # Create model, optimizer and loss function
    model = LinearProbe()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = torch.nn.BCELoss()

    for epoch in tqdm(range(N_EPOCHS), desc=f"Model {subset_idx+1}/{N_SUBSETS}"):
        # Training phase
        model.train()
        train_loss = 0
        for cross_attn_maps, gt in train_loader:
            optimizer.zero_grad()

            # Forward pass
            output = model(*cross_attn_maps)
            loss = criterion(output, gt)
            train_loss += loss.item()

            # Backward pass
            loss.backward()
            optimizer.step()

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for cross_attn_maps, gt in val_loader:
                # Calculate and accumulate validation loss
                output = model(*cross_attn_maps)
                loss = criterion(output, gt)
                val_loss += loss.item()

        # Record the losses for this epoch
        train_losses[subset_idx, epoch] = train_loss / len(train_loader)
        val_losses[subset_idx, epoch] = val_loss / len(val_loader)

    # Record the trained model
    trained_models[subset_idx] = model


# =========================== WRITING RECORDS TO DISK ==========================


PROJECT_DIR   = Path.cwd().parent
RESULTS_DIR   = PROJECT_DIR / "exp_results" / f"subset_training_{DATASET}"

# Save the losses
torch.save(train_losses, RESULTS_DIR / f"train_losses_{N_EPOCHS}_epochs.pt")
torch.save(val_losses, RESULTS_DIR / f"val_losses_{N_EPOCHS}_epochs.pt")

# Save the models
for idx, model in enumerate(trained_models):
    torch.save(model, RESULTS_DIR / f"model_{idx}.pt")


# ========================= PRINTING EXECUTION DURATION ========================


end_time = time()

# Calculate total duration
total_time = end_time - start_time

# Convert to hours, minutes, and seconds
hours = int(total_time // 3600)
minutes = int((total_time % 3600) // 60)
seconds = total_time % 60

print(f"Execution Time: {hours} hours, {minutes} minutes, and {seconds:.2f} seconds")
