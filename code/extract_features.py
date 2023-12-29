import os
from tqdm import tqdm
from typing import Dict
import h5py

import torch
import tensorflow as tf
from keras_cv.src.models.stable_diffusion.image_encoder import ImageEncoder
from stable_diffusion import StableDiffusion

from utils import process_image, augmenter
from my_utils import dict_to_disk

"""
Usage:
- set `IMG_DIR` to the directory path of the RGB images
- set `FEATURE_DIR` to where the extracted features should be saved
"""

DEVICE = "cpu"
# print(f"GPUs available: ", tf.config.experimental.list_physical_devices('GPU'))
# DEVICE = tf.test.gpu_device_name()
# print(f"Using device: {DEVICE}")

# Path of the context vector file
CONTEXT_PATH = "../data/context.h5"

# Load random but fixed context vector
with h5py.File(CONTEXT_PATH, 'r') as file:
    context = file["context"][:]
    
print("\n=== Initializing Stable Diffusion Model ===")
with tf.device(DEVICE):
    image_encoder = ImageEncoder()
    vae = tf.keras.Model(
        image_encoder.input,
        image_encoder.layers[-1].output,
    )
    model = StableDiffusion(img_width=512, img_height=512, use_pretrained_weights=True)


for object_size in ["small", "medium", "large"]:
    print(f"\n=== Extracting features for {object_size} objects ===")

    # This is where the RGB images are located
    IMG_DIR = f"../data/custom_datasets/{object_size}/img"

    # This is where the extracted features will be saved
    FEATURE_DIR = f"/Users/chrisoffner3d/Downloads/custom_datasets/{object_size}/cross_attn_maps"

    assert os.path.exists(IMG_DIR), f"Source directory {IMG_DIR} does not exist"
    assert os.path.exists(FEATURE_DIR), f"Target directory {FEATURE_DIR} does not exist"

    # Get list of filenames in DIRECTORY, filter out .DS_Store if necessary
    file_paths = sorted(os.listdir(IMG_DIR))
    if ".DS_Store" in file_paths:
        file_paths.remove(".DS_Store")

    print("\n=== Extracting self-attention maps and cross-attention maps ===")
    for image in tqdm(file_paths):
        # Construct file paths for self- and cross-attention maps
        filename = os.path.splitext(image)[0]
        self_attn_path = f"{FEATURE_DIR}/{filename}_self.h5"
        cross_attn_path = f"{FEATURE_DIR}/{filename}_cross.h5"

        # Skip image if .h5 feature files for image already exist
        if os.path.exists(cross_attn_path):
            print(f"Skipping {image}. Features already exist.")
            continue

        # Dictionary of structure { timestep : { resolution : self-attention map } }
        # self_attn_dict: Dict[int, Dict[int, torch.Tensor]] = {}
        cross_attn_dict: Dict[int, Dict[int, torch.Tensor]] = {}

        # Load image, preprocess it, and run it through the VAE encoder
        with tf.device(DEVICE):
            image_path = f"{IMG_DIR}/{image}"
            image = process_image(image_path)
            image = augmenter(image)
            latent = vae(tf.expand_dims(image, axis=0), training=False)

        n_timesteps = 10
        timesteps = torch.linspace(
            start=0,
            end=999,
            steps=n_timesteps,
            dtype=torch.int32
        ).tolist()

        for timestep in timesteps:
            with tf.device(DEVICE):
                # Extract all self-attention and cross-attention maps
                self_attn_64,  self_attn_32,  self_attn_16,  self_attn_8, \
                cross_attn_64, cross_attn_32, cross_attn_16, cross_attn_8 \
                = model.generate_image(
                    encoded_text=context,
                    latent=latent,
                    timestep=timestep
                )

                # Average over attention heads and store attention maps for
                # current time step in dictionary with half-precision (float16)
                # self_attn_dict[timestep] = {
                #     8:  torch.from_numpy(self_attn_8.mean(axis=(0,1))).half(),
                #     16: torch.from_numpy(self_attn_16.mean(axis=(0,1))).half(),
                #     32: torch.from_numpy(self_attn_32.mean(axis=(0,1))).half(),
                #     64: torch.from_numpy(self_attn_64.mean(axis=(0,1))).half()
                # }

                cross_attn_dict[timestep] = {
                    8:  torch.from_numpy(cross_attn_8.mean(axis=(0,1))).half(),
                    16: torch.from_numpy(cross_attn_16.mean(axis=(0,1))).half(),
                    32: torch.from_numpy(cross_attn_32.mean(axis=(0,1))).half(),
                    64: torch.from_numpy(cross_attn_64.mean(axis=(0,1))).half()
                }

        # Save dictionaries to disk
        # dict_to_disk(attn_dict=self_attn_dict,  file_path=self_attn_path)
        dict_to_disk(attn_dict=cross_attn_dict, file_path=cross_attn_path)

    print("\n=== Feature extraction complete ===")
