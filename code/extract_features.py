from typing import Dict
import os
from tqdm import tqdm

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

# This is where the RGB images are located
IMG_DIR = "../data/ECSSD_resized/img"

# This is where the extracted features will be saved
FEATURE_DIR = "~/Downloads/ECSSD_resized/features"

assert os.path.exists(IMG_DIR), f"Source directory {IMG_DIR} does not exist"
assert os.path.exists(FEATURE_DIR), f"Target directory {FEATURE_DIR} does not exist"


def main():
    if not os.path.exists(FEATURE_DIR):
        os.makedirs(FEATURE_DIR)

    print(f"GPUs available: ", tf.config.experimental.list_physical_devices('GPU'))
    device = tf.test.gpu_device_name()
    print(tf.test.gpu_device_name())

    print("\n=== Initializing Stable Diffusion Model ===")
    with tf.device(device):
        image_encoder = ImageEncoder()
        vae = tf.keras.Model(
            image_encoder.input,
            image_encoder.layers[-1].output,
        )
        model = StableDiffusion(img_width=512, img_height=512)

    # Get list of filenames in DIRECTORY, filter out .DS_Store if necessary
    files = sorted(os.listdir(IMG_DIR))
    if ".DS_Store" in files:
        files.remove(".DS_Store")

    print("\n=== Extracting self-attention maps and cross-attention maps ===")
    for image in tqdm(files):
        # Constructing paths
        filename = os.path.splitext(image)[0]
        self_attn_path = f"{FEATURE_DIR}/{filename}_self.h5"
        cross_attn_path = f"{FEATURE_DIR}/{filename}_cross.h5"

        # If .h5 feature files for image already exist, skip image  
        if os.path.exists(self_attn_path) and os.path.exists(cross_attn_path):
            print(f"Skipping {image} because features already exist")
            continue

        # Dictionary of structure { timestep : { resolution : self-attention map } }
        self_attn_dict: Dict[int, Dict[int, torch.Tensor]] = { }
        cross_attn_dict: Dict[int, Dict[int, torch.Tensor]] = { }

        # Load image, preprocess it, and run it through the VAE encoder
        with tf.device(device):
            image_path = f"{IMG_DIR}/{image}"
            image = process_image(image_path)
            image = augmenter(image)
            latent = vae(tf.expand_dims(image, axis=0), training=False)

        num_timesteps = 10
        for timestep in torch.linspace(0, 999, num_timesteps, dtype=torch.int32).tolist():
            with tf.device(device):
                # Extract all self-attention and cross-attention maps
                self_attn_64,  self_attn_32,  self_attn_16,  self_attn_8, \
                cross_attn_64, cross_attn_32, cross_attn_16, cross_attn_8 \
                = model.generate_image(latent, timestep,)

                # Average over attention heads and store attention maps for
                # current time step in dictionary with half-precision (float16)
                self_attn_dict[timestep] = {
                    8:  torch.from_numpy(self_attn_8.mean(axis=(0,1))).half(),
                    16: torch.from_numpy(self_attn_16.mean(axis=(0,1))).half(),
                    32: torch.from_numpy(self_attn_32.mean(axis=(0,1))).half(),
                    64: torch.from_numpy(self_attn_64.mean(axis=(0,1))).half()
                }

                cross_attn_dict[timestep] = {
                    8:  torch.from_numpy(cross_attn_8.mean(axis=(0,1))).half(),
                    16: torch.from_numpy(cross_attn_16.mean(axis=(0,1))).half(),
                    32: torch.from_numpy(cross_attn_32.mean(axis=(0,1))).half(),
                    64: torch.from_numpy(cross_attn_64.mean(axis=(0,1))).half()
                }
        
        # Save dictionaries to disk
        dict_to_disk(attn_dict=self_attn_dict,  filename=self_attn_path)
        dict_to_disk(attn_dict=cross_attn_dict, filename=cross_attn_path)


if __name__ == "__main__":
    main()