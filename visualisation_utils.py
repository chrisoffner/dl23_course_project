# ----------------------------------------------------------------------------
# Author:   Chris Offner
# Email:    chrisoffner@pm.me
#
# Copyright 2032 Chris Offner. All rights reserved.
# License:  GPL License
# ----------------------------------------------------------------------------
# This file contains utility functions for visualising attention maps in
# diffusion models.
# ----------------------------------------------------------------------------


from typing import Dict, List, Tuple

import numpy as np
from matplotlib.figure import Figure
from matplotlib.artist import Artist
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.ndimage import zoom
from tqdm import tqdm

def convert_channel_idx(orig_channel_idx: int, orig_res: int, new_res: int) -> int:
    """
    Takes a channel index for a 64 x 64 attention map and converts it to the
    channel index for anattention map of a different resolution.

    Parameters
    ----------
    orig_channel : Integer specifying a channel index in a 64 x 64 attention map
    orig_res     : Integer specifying the resolution w.r.t. which orig_channel
                   has been specified
    new_res      : Integer specifying the resolution for which the new channel
                   index is to be computed

    Returns
    -------
    new_channel  : Integer of the channel index (in the attention map of
                   resolution new_res x new_res) that corresponds to
                   orig_channel in the orig_res x orig_res attention map
    """

    assert orig_channel_idx <= 64**2

    # Determine 2d pixel coordinate for original channel
    row1, col1 = divmod(orig_channel_idx, orig_res)

    # Calculate scaling factor
    scale = new_res / orig_res

    # Convert 2d coordinate back to single channel integer
    new_channel = int(row1 * scale) * new_res + int(col1 * scale)

    return new_channel


def channel_to_rgb(channel: np.ndarray, colormap=plt.cm.viridis) -> np.ndarray:
    """
    Converts a single channel image to the RGB representation using the Viridis
    colormap.
    """
    # Normalize the single channel data to be in the range [0, 1]
    normalized_data = (channel - channel.min()) / (channel.max() - channel.min())

    # Apply the colormap
    mapped_data = colormap(normalized_data)

    # Extract the first three channels (RGB) and convert to desired format
    rgb = np.delete(mapped_data, 3, 2)  # Removes the alpha channel

    return rgb


def plot_attention_location(
        res2weights: Dict[int, np.ndarray],
        orig_channel_idx: int = 2355,
        orig_res: int = 64,
        interpolate: bool = False,
        timestep: int = 300,
        fig: Figure = None,     # Used when rendering animations
        axs: np.ndarray = None  # Used when rendering animations
    ) -> List[Artist]:
    """
    Plots a 2 x 8 grid of attention map subplots where each column shows the
    A, B variants for one attention map resolution for a given image location.
    The location is determined by a channel index that is specified relative to
    an original resolution.

    For example, orig_channel_idx==2355 and orig_res==64 plots all attention
    maps that correspond to channel 2355 in the 64 x 64 attention map. For maps
    with lower resolution, the channel index will be converted accordingly.

    Parameters
    ----------
    res2weights      : Dictionary containing { resolution: attention_map } pairs
    orig_channel_idx : Channel index specifying a location in the orig_res map 
    orig_res         : Resolution of the attention map in which orig_channel_idx
                       has been chosen
    interpolate      : Boolean deciding whether to render with bicubic upscaling

    Returns
    -------
    artists          : A list of artist objects that represent subplots of fig
    """

    resolutions = [8, 16, 32, 64]
    artists = []

    if fig is None and axs is None:
        # Used when rendering a still plot. When rendering an animation, fig and
        # axs are passed by the frame generation function.
        fig, axs = plt.subplots(2, 4, figsize=(20, 10))
        fig.tight_layout()
        fig.suptitle(f"Self-attention maps for timestep {timestep}", y=1.05)

    for i, res in enumerate(resolutions):
        # Compute parameters used for plotting channel i at resolution res
        A, B        = _attention_interpretations(res2weights, res)
        ch          = convert_channel_idx(orig_channel_idx, orig_res, res)
        row, col    = divmod(ch, res)
        zoom_factor = resolutions[-i-1] if interpolate else 1
        offset      = zoom_factor / 2 if interpolate else 0

        # Create scatter objects and set their offsets
        scatter_artist1 = axs[0, i].scatter([], [], color='red', s=30)
        scatter_artist2 = axs[1, i].scatter([], [], color='red', s=30)
        scatter_artist1.set_offsets([col * zoom_factor + offset, row * zoom_factor + offset])
        scatter_artist2.set_offsets([col * zoom_factor + offset, row * zoom_factor + offset])

        # Set subplot titles
        axs[0, i].set_title(f'A[{ch}, :, :] ({res} x {res})')
        axs[1, i].set_title(f'B[:, :, {ch}] ({res} x {res})')

        # Add the artist objects created by imshow and scatter to the list
        artists.append(axs[0, i].imshow(zoom(A[ch, :, :], zoom_factor, order=2) if interpolate else A[ch, :, :]))
        artists.append(axs[1, i].imshow(zoom(B[:, :, ch], zoom_factor, order=2) if interpolate else B[:, :, ch]))
        artists.append(scatter_artist1)
        artists.append(scatter_artist2)

    return artists


def animate_locations(
        res2weights: Dict[int, np.ndarray],
        num_frames: int = 4096,
        fps: int = 1,
        interpolate: bool = False,
        filename: str = "attention_animation"
    ) -> None:
    fig, axs = plt.subplots(2, 4, figsize=(20, 10))  # Create the figure and axes
    fig.tight_layout()

    def generate_frame(
            channel_idx: int,
            fig: Figure,
            axs: np.ndarray
        ) -> List[Artist]:
        # Clear the axes for each frame
        for ax in axs.ravel():
            ax.clear()

        # Plot the new frame's content on the existing figure
        artists = plot_attention_location(
            res2weights,
            orig_channel_idx=channel_idx,
            orig_res=64, 
            interpolate=interpolate,
            fig=fig,
            axs=axs
        )

        return artists

    # Create the animation
    ani = animation.FuncAnimation(
        fig,
        generate_frame,
        frames=range(num_frames),
        fargs=(fig, axs),
        blit=True
    )

    # Show a progress bar while rendering the animation, then save file to disk
    with tqdm(total=num_frames, desc="Saving animation") as pbar:
        ani.save(
            f'./animations/{filename}.mp4',
            writer='ffmpeg',
            fps=fps,
            progress_callback=lambda i, n: pbar.update()
        )


def _attention_interpretations(
        res2weights: Dict[int, np.ndarray],
        res: int
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Helper function that reshapes attention maps in two different ways to allow
    visual comparison. Sums attention maps across all 8 heads.
    Example for 64 x 64 self-attention maps:

    A: (1, 8, 4096, 4096) -> (1, 8, 4096, 64,   64) -> (4096, 64,   64)
    B: (1, 8, 4096, 4096) -> (1, 8,   64  64, 4096) -> (64,   64, 4096)
    """

    assert res in res2weights.keys()
    assert res2weights[res].shape[0] == res**2
    assert res2weights[res].shape[0] == res**2

    # Reshape 64 x 64 self-attention maps and sum them across heads for visualisation
    A = res2weights[res].reshape(res*res, res, res)
    B = res2weights[res].reshape(res, res, res*res)

    # Normalise values for visualisation
    A -= A.min()
    B -= B.min()
    A /= A.max()
    B /= B.max()

    assert A.shape == (res**2, res, res)
    assert B.shape == (res, res, res**2)

    return A, B


from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def render_attention_animation(
        timestep_range,
        orig_channel_idx=2355,
        orig_res=64,
        interpolate=False,
        filename='attention_animation',
        fps: int = 5
    ):
    """
    Renders an animation of attention maps over a range of timesteps.

    Parameters
    ----------
    res2weights      : Dictionary containing { resolution: attention_map } pairs
    timestep_range   : Tuple or list defining the start and end of the timestep range
    orig_channel_idx : Channel index specifying a location in the orig_res map
    orig_res         : Resolution of the attention map in which orig_channel_idx
                       has been chosen
    interpolate      : Boolean deciding whether to render with bicubic upscaling
    save_path        : Path to save the animation

    Returns
    -------
    None
    """
    fig, axs = plt.subplots(2, 4, figsize=(20, 10))
    title = plt.suptitle(t='', fontsize = 20)

    def update_frame(timestep):
        # Clear both subplots
        axs[0, 0].cla()  
        axs[1, 0].cla()

        # Set title
        title.set_text(f"Self-attention maps for timestep {timestep}")

        image_path = "./images/img1.jpeg"  # Specify the path to your image

        # Run inference to obtain self-attention maps for current time step
        with tf.device(device):
            images = process_image(image_path)
            images = augmenter(images)
            latent = vae(tf.expand_dims(images, axis=0), training=False)
            _, _, weight_64, weight_32, weight_16, weight_8, _, _, _, _ = model.text_to_image(
                batch_size=1,
                latent=latent,
                timestep=timestep
            )

        # Store self-attention maps in a dictionary for resolution-specific access
        res2weights = { 8: weight_8, 16: weight_16, 32: weight_32, 64: weight_64 }

        artists = plot_attention_location(
            res2weights,
            orig_channel_idx=orig_channel_idx,
            orig_res=orig_res,
            interpolate=interpolate,
            timestep=timestep,
            fig=fig,
            axs=axs
        )
        return artists

    ani = FuncAnimation(fig, update_frame, frames=timestep_range, blit=True)

    # Show a progress bar while rendering the animation, then save file to disk
    with tqdm(total=len(timestep_range), desc="Saving animation") as pbar:
        ani.save(
            f"animations/{filename}.mp4",
            writer="ffmpeg",
            fps=fps,
            progress_callback=lambda i, n: pbar.update()
        )
