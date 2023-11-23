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


from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.ndimage import zoom

def convert_channel_idx(orig_channel: int, orig_res: int, new_res: int) -> int:
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

    assert orig_channel <= 64**2

    # Determine 2d pixel coordinate for original channel
    row1, col1 = divmod(orig_channel, orig_res)

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


def plot_attention(
        res2weights: Dict[int, np.ndarray],
        res: int = 16,
        channel: int = 0
    ) -> None:
    """
    Plots a 4 x 4 grid of attention map subplots where the first row shows the
    attention map in the original resolution and the second row shows the same
    maps bicubically upscaled to 512 x 512.

    Parameters
    ----------
    res2weights : Dictionary containing { resolution: attention_map } pairs.
    res         : Resolution of attention map to use from res2weights.
    channel     : Integer specifying which channel to plot. Maximum is res**2.

    Returns
    -------
    None
    """

    A, B = _attention_interpretations(res2weights, res)

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    scale = 512 / res # Scaling factor for upscaling from res x res to 512 x 512
    axs[0, 0].imshow(A[channel, :, :])
    axs[0, 1].imshow(B[:, :, channel])
    axs[1, 0].imshow(zoom(A[channel, :, :], scale, order=2))
    axs[1, 1].imshow(zoom(B[:, :, channel], scale, order=2))

    # Calculate pixel position corresponding to the channel number
    row = channel // res
    col = channel % res

    # Update the scatter plot to highlight the pixel in red
    axs[0, 0].scatter([], [], color='red', s=30).set_offsets([col, row])
    axs[0, 1].scatter([], [], color='red', s=30).set_offsets([col, row])
    axs[1, 0].scatter([], [], color='red', s=30).set_offsets([col*scale + scale/2, row*scale + scale/2])
    axs[1, 1].scatter([], [], color='red', s=30).set_offsets([col*scale + scale/2, row*scale + scale/2])

    #  Set subplot titles
    axs[0, 0].set_title(f'A[{channel}, :, :] ({res} x {res})')
    axs[0, 1].set_title(f'B[:, :, {channel}] ({res} x {res})')
    axs[1, 0].set_title(f'A[{channel}, :, :] ({res} x {res}) upscaled to (512 x 512)')
    axs[1, 1].set_title(f'B[:, :, {channel}] ({res} x {res}) upscaled to (512 x 512)')

    fig.tight_layout()
    fig.suptitle(f"Self-attention {res}x{res}", y=1.05)
    plt.show()


def animate_attention(
        res2weights: Dict[int, np.ndarray],
        res: int = 16,
        num_frames: int = 100,
        fps: int = 1
    ) -> None:
    """
    Renders a 4 x 4 grid of attention map plots where the first row shows the
    attention map in the original resolution and the second row shows the same
    maps bicubically upscaled to 512 x 512. Each frame corresponds to one
    channel of the attention map.

    Parameters
    ----------
    res2weights : Dictionary containing { resolution: attention_map } pairs.
    res         : Resolution of attention map to use from res2weights.
    num_frames  : Number of frames to render. Maximum is res**2.
    fps         : Frame rate of animation. Maximum is 60.

    Returns
    -------
    None
    """

    assert num_frames <= res**2, "num_frames must be at most res**2"
    assert res in res2weights.keys()
    assert fps <= 60

    A, B = _attention_interpretations(res2weights, res)

    # Create a figure and axes
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    fig.tight_layout()

    # Adjust the spacing between the rows of subplots
    fig.subplots_adjust(hspace=0.1)   # Increase the value to add more space

    # Initial scatter plots marking 'active pixel'
    scatters = [
        axs[0, 0].scatter([], [], color='red', s=30),
        axs[0, 1].scatter([], [], color='red', s=30),
        axs[1, 0].scatter([], [], color='red', s=30),
        axs[1, 1].scatter([], [], color='red', s=30)
    ]

    # Create the animation
    ani = animation.FuncAnimation(
        fig,
        _update_plot,
        frames=np.linspace(0, res**2, num=num_frames, endpoint=False).round().astype(int),
        fargs=(A, B, axs, scatters, res),
        blit=False
    )

    # Save the animation to disk
    ani.save(f'./animations/self_attn_{res}.mp4', writer='ffmpeg', fps=fps)
    print(f'Saved animation as "self_attn_{res}.mp4"')


# Helper function to update the plots for each frame
def _update_plot(channel, A, B, axs, scatters, res):
    scale = 512 / res   # Scaling factor

    # Calculate red 'active pixel' position
    row, col = divmod(channel, res)

    # Move red 'active pixel' dot marker to new position
    scatters[0].set_offsets([col, row])
    scatters[1].set_offsets([col, row])
    scatters[2].set_offsets([col*scale + scale/2, row*scale + scale/2])
    scatters[3].set_offsets([col*scale + scale/2, row*scale + scale/2])

    # Update self-attention map plots
    axs[0, 0].imshow(A[channel, :, :])
    axs[0, 1].imshow(B[:, :, channel])
    axs[1, 0].imshow(zoom(A[channel, :, :], scale, order=2)) # Bicubic upscale
    axs[1, 1].imshow(zoom(B[:, :, channel], scale, order=2)) # Bicubic upscale

    # Update subplot titles
    axs[0, 0].set_title(f'A[{channel}, :, :] ({res} x {res})')
    axs[0, 1].set_title(f'B[:, :, {channel}] ({res} x {res})')
    axs[1, 0].set_title(f'A[{channel}, :, :] ({res} x {res}) upscaled to (512 x 512)')
    axs[1, 1].set_title(f'B[:, :, {channel}] ({res} x {res}) upscaled to (512 x 512)')


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
    assert res2weights[res].shape[1] == 8      # Number of attention heads
    assert res2weights[res].shape[2] == res**2
    assert res2weights[res].shape[3] == res**2

    # Reshape 64 x 64 self-attention maps and sum them across heads for visualisation
    A = res2weights[res].reshape(1, 8, res*res, res, res).squeeze(0).sum(axis=0)
    B = res2weights[res].reshape(1, 8, res, res, res*res).squeeze(0).sum(axis=0)

    # Normalise values for visualisation
    A -= A.min()
    B -= B.min()
    A /= A.max()
    B /= B.max()

    assert A.shape == (res**2, res, res)
    assert B.shape == (res, res, res**2)

    return A, B
