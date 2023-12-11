# The following code is modifed from 
# https://github.com/keras-team/keras-cv/blob/master/keras_cv/models/stable_diffusion/stable_diffusion.py

"""Keras implementation of StableDiffusion.

Credits:

- Original implementation:
  https://github.com/CompVis/stable-diffusion
- Initial TF/Keras port:
  https://github.com/divamgupta/stable-diffusion-tensorflow

The current implementation is a rewrite of the initial TF/Keras port by
Divam Gupta.
"""

import math

from keras_cv.src.models.stable_diffusion.constants import _UNCONDITIONAL_TOKENS
from keras_cv.src.models.stable_diffusion.decoder import Decoder
from .diffusion_model import DiffusionModel
from keras_cv.src.models.stable_diffusion.text_encoder import TextEncoder
import numpy as np
import tensorflow as tf

MAX_PROMPT_LENGTH = 77

class StableDiffusionBase:
  """Base class for stable diffusion and stable diffusion v2 model."""

  def __init__(
      self,
      img_height=512,
      img_width=512,
      jit_compile=False,
  ):
    # UNet requires multiples of 2**7 = 128
    img_height = round(img_height / 128) * 128
    img_width = round(img_width / 128) * 128
    self.img_height = img_height
    self.img_width = img_width

    # lazy initialize the component models and the tokenizer
    self._image_encoder = None
    self._text_encoder = None
    self._diffusion_model = None
    self._decoder = None

    self.jit_compile = jit_compile

  def generate_image(
      self,
      batch_size=1,
      latent=None,
      timestep=None,
  ):
    unconditional_context = tf.repeat(
          self._get_unconditional_context(),
          batch_size,
          axis=0
    )

    t_emb = self._get_timestep_embedding(timestep, batch_size)

    # ====================== Extract attention maps ======================

    unconditional_latent, weight_64, weight_32, weight_16, weight_8 \
        = self.diffusion_model.predict_on_batch([latent, t_emb, unconditional_context])
    
    # ====================================================================

    # Decoding
    # output_image = self.decoder.predict_on_batch(unconditional_latent)
    # output_image = ((output_image + 1) / 2) * 255
    # output_image = np.clip(output_image, 0, 255).astype("uint8")

    return weight_64, weight_32, weight_16, weight_8

  def _get_unconditional_context(self):
    unconditional_tokens = tf.convert_to_tensor(
        [_UNCONDITIONAL_TOKENS],
        dtype=tf.int32
    )
    unconditional_context = self.text_encoder.predict_on_batch(
        [unconditional_tokens, self._get_pos_ids()]
    )

    return unconditional_context

  @property
  def diffusion_model(self):
    pass

  @property
  def decoder(self):
    """decoder returns the diffusion image decoder model with pretrained

    weights. Can be overriden for tasks where the decoder needs to be
    modified.
    """
    if self._decoder is None:
      self._decoder = Decoder(self.img_height, self.img_width)
      if self.jit_compile:
        self._decoder.compile(jit_compile=True)
    return self._decoder

  def _get_timestep_embedding(
      self,
      timestep,
      batch_size,
      dim=320,
      max_period=10000
  ):
    half = dim // 2
    freqs = tf.math.exp(
        -math.log(max_period) * tf.range(0, half, dtype=tf.float32) / half
    )
    args = tf.convert_to_tensor([timestep], dtype=tf.float32) * freqs
    embedding = tf.concat([tf.math.cos(args), tf.math.sin(args)], 0)
    embedding = tf.reshape(embedding, [1, -1])
    return tf.repeat(embedding, batch_size, axis=0)

  @staticmethod
  def _get_pos_ids():
    return tf.convert_to_tensor(
        [list(range(MAX_PROMPT_LENGTH))], dtype=tf.int32
    )


class StableDiffusion(StableDiffusionBase):
  """Keras implementation of Stable Diffusion.

  Note that the StableDiffusion API, as well as the APIs of the sub-components
  of StableDiffusion (e.g. ImageEncoder, DiffusionModel) should be considered
  unstable at this point. We do not guarantee backwards compatability for
  future changes to these APIs.

  Stable Diffusion is a powerful image generation model that can be used,
  among other things, to generate pictures according to a short text
  description (called a "prompt").

  Arguments:
      img_height: int, height of the images to generate, in pixel. Note that
        only multiples of 128 are supported; the value provided will be rounded
        to the nearest valid value. Defaults to 512.
      img_width: int, width of the images to generate, in pixel. Note that only
        multiples of 128 are supported; the value provided will be rounded to
        the nearest valid value. Defaults to 512.
      jit_compile: bool, whether to compile the underlying models to XLA. This
        can lead to a significant speedup on some systems. Defaults to False.

  Example:

  ```python
  from keras_cv.models import StableDiffusion
  from PIL import Image

  model = StableDiffusion(img_height=512, img_width=512, jit_compile=True)
  img = model.text_to_image(
      prompt="A beautiful horse running through a field",
      batch_size=1,  # How many images to generate at once
      num_steps=25,  # Number of iterations (controls image quality)
      seed=123,  # Set this to always get the same image from the same prompt
  )
  Image.fromarray(img[0]).save("horse.png")
  print("saved at horse.png")
  ```

  References:
  - [About Stable
    Diffusion](https://stability.ai/blog/stable-diffusion-announcement)
  - [Original implementation](https://github.com/CompVis/stable-diffusion)
  """  # noqa: E501

  def __init__(
      self,
      img_height=512,
      img_width=512,
      jit_compile=False,
  ):
    super().__init__(img_height, img_width, jit_compile)
    print(
        "By using this model checkpoint, you acknowledge that its usage is "
        "subject to the terms of the CreativeML Open RAIL-M license at "
        "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/LICENSE"  # noqa: E501
    )

  @property
  def text_encoder(self):
    """text_encoder returns the text encoder with pretrained weights.

    Can be overriden for tasks like textual inversion where the text encoder
    needs to be modified.
    """
    if self._text_encoder is None:
      self._text_encoder = TextEncoder(MAX_PROMPT_LENGTH)
      if self.jit_compile:
        self._text_encoder.compile(jit_compile=True)
    return self._text_encoder

  @property
  def diffusion_model(self):
    """diffusion_model returns the diffusion model with pretrained weights.

    Can be overriden for tasks where the diffusion model needs to be modified.
    """
    if self._diffusion_model is None:
      self._diffusion_model = DiffusionModel(
          self.img_height,
          self.img_width,
          MAX_PROMPT_LENGTH
      )
      if self.jit_compile:
        self._diffusion_model.compile(jit_compile=True)
    return self._diffusion_model