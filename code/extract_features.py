from typing import Dict
import os
from tqdm import tqdm

import torch
import tensorflow as tf
from keras_cv.src.models.stable_diffusion.image_encoder import ImageEncoder
from stable_diffusion import StableDiffusion

from utils import process_image, augmenter
from my_utils import dict_to_disk

'''
Usage:
- set `DIRECTORY` variable to the right path (path to the RGB images)
- for the "self_attn_dict", decide how many self-attention maps you want to add
  by un-/commenting (only 64x64 selected now)
'''

# DIRECTORY = "C:/Datasets/Resized_MSRA10K_Imgs_GT/Resized_images"
DIRECTORY = "../data/ECSSD_resized/img"
print(os.listdir(DIRECTORY))


def main():
    # Get the parent directory of DIRECTORY
    parent_dir = os.path.dirname(DIRECTORY)

    # Combine the parent directory with the new feature directory name
    FEATURE_DIR = os.path.join(parent_dir, "features")
    print(f"Features will be saved to {FEATURE_DIR}/")

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


    # Run image through VAE encoder
    print("\n=== Extracting self-attention maps and cross-attention maps ===")
    for file in tqdm(os.listdir(DIRECTORY)):
        # If .h5 feature files for file already exists, skip it
        self_attn_path = f"{FEATURE_DIR}/{file.split('.jpg')[0]}" + "_cross"
        cross_attn_path = f"{FEATURE_DIR}/{file.split('.jpg')[0]}" + "_cross"
        if os.path.exists(self_attn_path + ".h5") and os.path.exists(cross_attn_path + ".h5"):
            continue

        # Dictionary of structure { timestep : { resolution : self-attention map } }
        self_attn_dict: Dict[int, Dict[int, torch.Tensor]] = { }
        cross_attn_dict: Dict[int, Dict[int, torch.Tensor]] = { }

        # Load image, preprocess it, and run it through the VAE encoder
        with tf.device(device):
            image_path = f"{DIRECTORY}/{file}"
            image = process_image(image_path)
            image = augmenter(image)
            latent = vae(tf.expand_dims(image, axis=0), training=False)

        num_timesteps = 10
        for timestep in torch.linspace(0, 999, num_timesteps, dtype=torch.int32).tolist():
            with tf.device(device):
                # Extract all self-attention and cross-attention maps
                self_attn_64,  self_attn_32,  self_attn_16,  self_attn_8, \
                cross_attn_64, cross_attn_32, cross_attn_16, cross_attn_8 = model.generate_image(
                    batch_size=1,
                    latent=latent,
                    timestep=timestep,
                )

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