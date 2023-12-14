from typing import Dict
import tensorflow as tf
import numpy as np
import os
import time

from keras_cv.src.models.stable_diffusion.image_encoder import ImageEncoder
from diffusion_models.stable_diffusion import StableDiffusion
from utils import process_image, augmenter
from my_utils import dict_to_disk


def main():
    print(f"GPUs available: ", tf.config.experimental.list_physical_devices('GPU'))
    device = tf.test.gpu_device_name()
    print(tf.test.gpu_device_name())

    print("\n---Initialize Stable Diffusion Model----")
    # Inialize Stable Diffusion Model on GPU:0
    with tf.device(device):
        image_encoder = ImageEncoder()
        vae = tf.keras.Model(
            image_encoder.input,
            image_encoder.layers[-1].output,
        )
        model = StableDiffusion(img_width=512, img_height=512)



    # Run image through VAE encoder
    print("\nRun image through VAE encoder")
    dir = 'C:/Datasets/Resized_MSRA10K_Imgs_GT/Resized_images'
    for i, file in enumerate (os.listdir(dir)):
        with tf.device(device):
            print()
            print(file)
            print(f"Image number {i}/10 000")
            print(f"Progress: {i/10000.0}%")
            image_path = f"{dir}/{file}"
            vae_start = time.time()
            image = process_image(image_path)
            image = augmenter(image)
            latent = vae(tf.expand_dims(image, axis=0), training=False)
            # save ram
            image = 0
        print("VAE Latent extracted")
        vae_end = time.time()
        print(f"VAE time: {vae_end - vae_start} s")    
        
        # Dictionary of structure { timestep : { resolution : self-attention map } }
        self_attn_dict: Dict[int, Dict[int, np.ndarray]] = { }

        print("Get Self-Attention Maps")
        diff_start = time.time()
        num_timesteps = 10
        for timestep in np.arange(0, 1000, 1000 // num_timesteps):
            with tf.device(device):
                weight_64, weight_32, weight_16, weight_8 = model.generate_image(
                    batch_size=1,
                    latent=latent,
                    timestep=timestep,
                )

                # Average over attention heads and store self-attention maps for
                # current time step in dictionary
                self_attn_dict[timestep] = {
                    # 8:  weight_8.mean(axis=(0,1)),
                    # 16: weight_16.mean(axis=(0,1)),
                    # 32: weight_32.mean(axis=(0,1)),
                    64: weight_64.mean(axis=(0,1))
                }          
                print(f"Self-attention map from {timestep} extracted")
        diff_end = time.time()
        print(f"Diffusion model time: {diff_end - diff_start} s")
        # Save self-attention maps to disk
        print("Save dict to disk...")
        dict_to_disk(
            self_attn_dict=self_attn_dict,
            filename= "self_attn_maps/" + file.replace(".jpg","")
        )
        print("Saved to disk")
        print(f"Total iteration time: {diff_end - vae_start} s")

        
if __name__ == "__main__":
    main()