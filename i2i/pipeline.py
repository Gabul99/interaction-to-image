# Copyright 2024 PixArt-Alpha Authors and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import html
import copy
import inspect
import re
import os
import json
import argparse
from functools import partial
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import urllib.parse as ul
from typing import Callable, List, Optional, Tuple, Union
from datetime import datetime
import numpy as np

import torch
import torch.nn.functional as F
from torch import nn
from transformers import T5EncoderModel, T5Tokenizer

from diffusers.image_processor import VaeImageProcessor
from diffusers.models import AutoencoderKL
from diffusers.schedulers import DPMSolverMultistepScheduler
from diffusers.utils import (
    # BACKENDS_MAPPING,
    deprecate,
    is_bs4_available,
    is_ftfy_available,
    logging,
    replace_example_docstring,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput


import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from i2i.transformer_2d import Transformer2DModel
from i2i.global_update_functions import *
from i2i.utils import *
from i2i.guidance.clip.base_clip import CLIPEncoder, SigLIPEncoder

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

if is_bs4_available():
    from bs4 import BeautifulSoup

if is_ftfy_available():
    import ftfy

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import PixArtAlphaPipeline

        >>> # You can replace the checkpoint id with "PixArt-alpha/PixArt-XL-2-512x512" too.
        >>> pipe = PixArtAlphaPipeline.from_pretrained("PixArt-alpha/PixArt-XL-2-1024-MS", torch_dtype=torch.float16)
        >>> # Enable memory optimizations.
        >>> pipe.enable_model_cpu_offload()

        >>> prompt = "A small cactus with a happy face in the Sahara desert."
        >>> image = pipe(prompt).images[0]
        ```
"""

ASPECT_RATIO_1024_BIN = {
    "0.25": [512.0, 2048.0],
    "0.28": [512.0, 1856.0],
    "0.32": [576.0, 1792.0],
    "0.33": [576.0, 1728.0],
    "0.35": [576.0, 1664.0],
    "0.4": [640.0, 1600.0],
    "0.42": [640.0, 1536.0],
    "0.48": [704.0, 1472.0],
    "0.5": [704.0, 1408.0],
    "0.52": [704.0, 1344.0],
    "0.57": [768.0, 1344.0],
    "0.6": [768.0, 1280.0],
    "0.68": [832.0, 1216.0],
    "0.72": [832.0, 1152.0],
    "0.78": [896.0, 1152.0],
    "0.82": [896.0, 1088.0],
    "0.88": [960.0, 1088.0],
    "0.94": [960.0, 1024.0],
    "1.0": [1024.0, 1024.0],
    "1.07": [1024.0, 960.0],
    "1.13": [1088.0, 960.0],
    "1.21": [1088.0, 896.0],
    "1.29": [1152.0, 896.0],
    "1.38": [1152.0, 832.0],
    "1.46": [1216.0, 832.0],
    "1.67": [1280.0, 768.0],
    "1.75": [1344.0, 768.0],
    "2.0": [1408.0, 704.0],
    "2.09": [1472.0, 704.0],
    "2.4": [1536.0, 640.0],
    "2.5": [1600.0, 640.0],
    "3.0": [1728.0, 576.0],
    "4.0": [2048.0, 512.0],
}

ASPECT_RATIO_512_BIN = {
    "0.25": [256.0, 1024.0],
    "0.28": [256.0, 928.0],
    "0.32": [288.0, 896.0],
    "0.33": [288.0, 864.0],
    "0.35": [288.0, 832.0],
    "0.4": [320.0, 800.0],
    "0.42": [320.0, 768.0],
    "0.48": [352.0, 736.0],
    "0.5": [352.0, 704.0],
    "0.52": [352.0, 672.0],
    "0.57": [384.0, 672.0],
    "0.6": [384.0, 640.0],
    "0.68": [416.0, 608.0],
    "0.72": [416.0, 576.0],
    "0.78": [448.0, 576.0],
    "0.82": [448.0, 544.0],
    "0.88": [480.0, 544.0],
    "0.94": [480.0, 512.0],
    "1.0": [512.0, 512.0],
    "1.07": [512.0, 480.0],
    "1.13": [544.0, 480.0],
    "1.21": [544.0, 448.0],
    "1.29": [576.0, 448.0],
    "1.38": [576.0, 416.0],
    "1.46": [608.0, 416.0],
    "1.67": [640.0, 384.0],
    "1.75": [672.0, 384.0],
    "2.0": [704.0, 352.0],
    "2.09": [736.0, 352.0],
    "2.4": [768.0, 320.0],
    "2.5": [800.0, 320.0],
    "3.0": [864.0, 288.0],
    "4.0": [1024.0, 256.0],
}

ASPECT_RATIO_256_BIN = {
    "0.25": [128.0, 512.0],
    "0.28": [128.0, 464.0],
    "0.32": [144.0, 448.0],
    "0.33": [144.0, 432.0],
    "0.35": [144.0, 416.0],
    "0.4": [160.0, 400.0],
    "0.42": [160.0, 384.0],
    "0.48": [176.0, 368.0],
    "0.5": [176.0, 352.0],
    "0.52": [176.0, 336.0],
    "0.57": [192.0, 336.0],
    "0.6": [192.0, 320.0],
    "0.68": [208.0, 304.0],
    "0.72": [208.0, 288.0],
    "0.78": [224.0, 288.0],
    "0.82": [224.0, 272.0],
    "0.88": [240.0, 272.0],
    "0.94": [240.0, 256.0],
    "1.0": [256.0, 256.0],
    "1.07": [256.0, 240.0],
    "1.13": [272.0, 240.0],
    "1.21": [272.0, 224.0],
    "1.29": [288.0, 224.0],
    "1.38": [288.0, 208.0],
    "1.46": [304.0, 208.0],
    "1.67": [320.0, 192.0],
    "1.75": [336.0, 192.0],
    "2.0": [352.0, 176.0],
    "2.09": [368.0, 176.0],
    "2.4": [384.0, 160.0],
    "2.5": [400.0, 160.0],
    "3.0": [432.0, 144.0],
    "4.0": [512.0, 128.0],
}

def linear_schedule(total_steps, current_step, end_weight, start_weight=1.0):
    if current_step >= total_steps:
        return 0.0
    else:
        return (end_weight - start_weight) / float(total_steps) * current_step + start_weight

# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
                Custom timesteps used to support arbitrary spacing between timesteps. If `None`, then the default
                timestep spacing strategy of the scheduler is used. If `timesteps` is passed, `num_inference_steps`
                must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())

        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps

@torch.no_grad()
def MainBranch(prompt_args, misc_args, latents, pipe):
    """
    Main branch for processing latents with a noise prediction model and optional guidance.

    Args:
        prompt_args (dict): Contains prompt-related arguments such as embeddings and attention masks.
        misc_args (dict): Miscellaneous arguments such as guidance scale, timesteps, and configuration.
        latents (torch.Tensor): Latent tensor to process.
        pipe: The pipeline object containing the scheduler and transformer.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (next latents x_{t-1}, predicted clean latents E[x_0 | x_t])
    """
    # Extract arguments from dictionaries
    prompt_embeds = prompt_args["prompt_embeds"]
    prompt_attention_mask = prompt_args["prompt_attention_mask"]
    added_cond_kwargs = prompt_args["added_cond_kwargs"]

    do_cfg = misc_args["do_cfg"]
    guidance_scale = misc_args["guidance_scale"]
    extra_step_kwargs = misc_args["extra_step_kwargs"]
    t = misc_args["t"]
    latent_channels = misc_args["latent_channels"]

    # Prepare latent model input
    latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
    latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

    # Process the current timestep
    current_timestep = pipe._timestep_process(
        t, 
        latent_model_input.device, 
        latent_model_input.shape[0]
    )

    # Predict noise model output
    noise_pred, _, _ = pipe.transformer(
        latent_model_input,
        encoder_hidden_states=prompt_embeds,
        encoder_attention_mask=prompt_attention_mask,
        timestep=current_timestep,
        added_cond_kwargs=added_cond_kwargs,
        cross_attention_kwargs=misc_args.get("cross_attention_kwargs", None),
        return_dict=False,
        return_attn_maps=False,  # GrounDiT flag
        is_object_branch=False,  # GrounDiT flag
    )

    # Perform classifier-free guidance (CFG)
    if do_cfg:
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

    # Adjust noise prediction for learned sigma
    if pipe.transformer.config.out_channels // 2 == latent_channels:
        noise_pred = noise_pred.chunk(2, dim=1)[0]

    # Compute previous latents (x_t -> x_{t-1}) and the predicted clean latents E[x_0 | x_t]
#    print(pipe.step_index)
    next_latents, pred_x0 = pipe.scheduler.step(
        noise_pred, t, latents,
        **extra_step_kwargs,
        return_dict=False
    )

    return next_latents, pred_x0

class PixArtAlphaPipeline(DiffusionPipeline):
    r"""
    Pipeline for text-to-image generation using PixArt-Alpha.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`T5EncoderModel`]):
            Frozen text-encoder. PixArt-Alpha uses
            [T5](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5EncoderModel), specifically the
            [t5-v1_1-xxl](https://huggingface.co/PixArt-alpha/PixArt-alpha/tree/main/t5-v1_1-xxl) variant.
        tokenizer (`T5Tokenizer`):
            Tokenizer of class
            [T5Tokenizer](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5Tokenizer).
        transformer ([`Transformer2DModel`]):
            A text conditioned `Transformer2DModel` to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
    """

    bad_punct_regex = re.compile(
        r"["
        + "#®•©™&@·º½¾¿¡§~"
        + r"\)"
        + r"\("
        + r"\]"
        + r"\["
        + r"\}"
        + r"\{"
        + r"\|"
        + "\\"
        + r"\/"
        + r"\*"
        + r"]{1,}"
    )  # noqa

    _optional_components = ["tokenizer", "text_encoder"]
    model_cpu_offload_seq = "text_encoder->transformer->vae"

    def __init__(
        self,
        tokenizer: T5Tokenizer,
        text_encoder: T5EncoderModel,
        vae: AutoencoderKL,
        transformer: Transformer2DModel,
        scheduler: DPMSolverMultistepScheduler,
    ):
        super().__init__()

        self.register_modules(tokenizer = tokenizer, text_encoder = text_encoder, vae = vae, transformer = transformer,scheduler = scheduler)

        # ================== GrounDiT ================== #
        # Originally from https://github.com/StevenShaw1999/RnB
        # Used in calculating "aggregated grounding loss" in the Global Update. See Sec. 5.1 of the paper.
        self.smth_3 = GaussianSmoothing(sigma = 3.0)
        sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32)
        
        sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0],[-1, -2, -1]], dtype=torch.float32)
        sobel_x = sobel_x.view(1, 1, 3, 3)
        sobel_y = sobel_y.view(1, 1, 3, 3)

        self.sobel_conv_x = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.sobel_conv_y = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)

        self.sobel_conv_x.weight = nn.Parameter(sobel_x)
        self.sobel_conv_y.weight = nn.Parameter(sobel_y)
        # ================== GrounDiT ================== #

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

    # Adapted from https://github.com/PixArt-alpha/PixArt-alpha/blob/master/diffusion/model/utils.py
    def mask_text_embeddings(self, emb, mask):
        if emb.shape[0] == 1:
            keep_index = mask.sum().item()
            return emb[:, :, :keep_index, :], keep_index
        else:
            masked_feature = emb * mask[:, None, :, None]
            return masked_feature, emb.shape[2]

    # Adapted from diffusers.pipelines.deepfloyd_if.pipeline_if.encode_prompt
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        do_classifier_free_guidance: bool = True,
        negative_prompt: str = "",
        num_images_per_prompt: int = 1,
        device: Optional[torch.device] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        prompt_attention_mask: Optional[torch.FloatTensor] = None,
        negative_prompt_attention_mask: Optional[torch.FloatTensor] = None,
        clean_caption: bool = False,
        max_sequence_length: int = 120,
        **kwargs,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt not to guide the image generation. If not defined, one has to pass `negative_prompt_embeds`
                instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`). For
                PixArt-Alpha, this should be "".
            do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
                whether to use classifier free guidance or not
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                number of images that should be generated per prompt
            device: (`torch.device`, *optional*):
                torch device to place the resulting embeddings on
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. For PixArt-Alpha, it's should be the embeddings of the ""
                string.
            clean_caption (`bool`, defaults to `False`):
                If `True`, the function will preprocess and clean the provided caption before encoding.
            max_sequence_length (`int`, defaults to 120): Maximum sequence length to use for the prompt.
        """

        if "mask_feature" in kwargs:
            deprecation_message = "The use of `mask_feature` is deprecated. It is no longer used in any computation and that doesn't affect the end results. It will be removed in a future version."
            deprecate("mask_feature", "1.0.0", deprecation_message, standard_warn=False)

        if device is None:
            device = self._execution_device

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # See Section 3.1. of the paper.
        max_length = max_sequence_length

        if prompt_embeds is None:
            prompt = self._text_preprocessing(prompt, clean_caption=clean_caption)
            
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(untruncated_ids[:, max_length - 1 : -1])
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {max_length} tokens: {removed_text}"
                )

            prompt_attention_mask = text_inputs.attention_mask
            prompt_attention_mask = prompt_attention_mask.to(device)

            prompt_embeds = self.text_encoder(text_input_ids.to(device), attention_mask=prompt_attention_mask)
            prompt_embeds = prompt_embeds[0]

        if self.text_encoder is not None:
            dtype = self.text_encoder.dtype
        elif self.transformer is not None:
            dtype = self.transformer.dtype
        else:
            dtype = None

        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)
        prompt_attention_mask = prompt_attention_mask.view(bs_embed, -1)
        prompt_attention_mask = prompt_attention_mask.repeat(num_images_per_prompt, 1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens = [negative_prompt] * batch_size
            uncond_tokens = self._text_preprocessing(uncond_tokens, clean_caption=clean_caption)
            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_attention_mask=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            negative_prompt_attention_mask = uncond_input.attention_mask
            negative_prompt_attention_mask = negative_prompt_attention_mask.to(device)

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device), attention_mask=negative_prompt_attention_mask
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

            negative_prompt_attention_mask = negative_prompt_attention_mask.view(bs_embed, -1)
            negative_prompt_attention_mask = negative_prompt_attention_mask.repeat(num_images_per_prompt, 1)
        else:
            negative_prompt_embeds = None
            negative_prompt_attention_mask = None

        return prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(
        self,
        prompt,
        height,
        width,
        negative_prompt,
        callback_steps,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        prompt_attention_mask=None,
        negative_prompt_attention_mask=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and prompt_attention_mask is None:
            raise ValueError("Must provide `prompt_attention_mask` when specifying `prompt_embeds`.")

        if negative_prompt_embeds is not None and negative_prompt_attention_mask is None:
            raise ValueError("Must provide `negative_prompt_attention_mask` when specifying `negative_prompt_embeds`.")

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )
            if prompt_attention_mask.shape != negative_prompt_attention_mask.shape:
                raise ValueError(
                    "`prompt_attention_mask` and `negative_prompt_attention_mask` must have the same shape when passed directly, but"
                    f" got: `prompt_attention_mask` {prompt_attention_mask.shape} != `negative_prompt_attention_mask`"
                    f" {negative_prompt_attention_mask.shape}."
                )

    # Copied from diffusers.pipelines.deepfloyd_if.pipeline_if.IFPipeline._text_preprocessing
    def _text_preprocessing(self, text, clean_caption=False):
        if clean_caption and not is_bs4_available():
            clean_caption = False

        if clean_caption and not is_ftfy_available():
            clean_caption = False

        if not isinstance(text, (tuple, list)):
            text = [text]

        def process(text: str):
            if clean_caption:
                text = self._clean_caption(text)
                text = self._clean_caption(text)
            else:
                text = text.lower().strip()
            return text

        return [process(t) for t in text]

    # Copied from diffusers.pipelines.deepfloyd_if.pipeline_if.IFPipeline._clean_caption
    def _clean_caption(self, caption):
        caption = str(caption)
        caption = ul.unquote_plus(caption)
        caption = caption.strip().lower()
        caption = re.sub("<person>", "person", caption)
        # urls:
        caption = re.sub(
            r"\b((?:https?:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))",  # noqa
            "",
            caption,
        )  # regex for urls
        caption = re.sub(
            r"\b((?:www:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))",  # noqa
            "",
            caption,
        )  # regex for urls
        # html:
        caption = BeautifulSoup(caption, features="html.parser").text

        # @<nickname>
        caption = re.sub(r"@[\w\d]+\b", "", caption)

        # 31C0—31EF CJK Strokes
        # 31F0—31FF Katakana Phonetic Extensions
        # 3200—32FF Enclosed CJK Letters and Months
        # 3300—33FF CJK Compatibility
        # 3400—4DBF CJK Unified Ideographs Extension A
        # 4DC0—4DFF Yijing Hexagram Symbols
        # 4E00—9FFF CJK Unified Ideographs
        caption = re.sub(r"[\u31c0-\u31ef]+", "", caption)
        caption = re.sub(r"[\u31f0-\u31ff]+", "", caption)
        caption = re.sub(r"[\u3200-\u32ff]+", "", caption)
        caption = re.sub(r"[\u3300-\u33ff]+", "", caption)
        caption = re.sub(r"[\u3400-\u4dbf]+", "", caption)
        caption = re.sub(r"[\u4dc0-\u4dff]+", "", caption)
        caption = re.sub(r"[\u4e00-\u9fff]+", "", caption)
        #######################################################

        # все виды тире / all types of dash --> "-"
        caption = re.sub(
            r"[\u002D\u058A\u05BE\u1400\u1806\u2010-\u2015\u2E17\u2E1A\u2E3A\u2E3B\u2E40\u301C\u3030\u30A0\uFE31\uFE32\uFE58\uFE63\uFF0D]+",  # noqa
            "-",
            caption,
        )

        # кавычки к одному стандарту
        caption = re.sub(r"[`´«»“”¨]", '"', caption)
        caption = re.sub(r"[‘’]", "'", caption)

        # &quot;
        caption = re.sub(r"&quot;?", "", caption)
        # &amp
        caption = re.sub(r"&amp", "", caption)

        # ip adresses:
        caption = re.sub(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", " ", caption)

        # article ids:
        caption = re.sub(r"\d:\d\d\s+$", "", caption)

        # \n
        caption = re.sub(r"\\n", " ", caption)

        # "#123"
        caption = re.sub(r"#\d{1,3}\b", "", caption)
        # "#12345.."
        caption = re.sub(r"#\d{5,}\b", "", caption)
        # "123456.."
        caption = re.sub(r"\b\d{6,}\b", "", caption)
        # filenames:
        caption = re.sub(r"[\S]+\.(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)", "", caption)

        #
        caption = re.sub(r"[\"\']{2,}", r'"', caption)  # """AUSVERKAUFT"""
        caption = re.sub(r"[\.]{2,}", r" ", caption)  # """AUSVERKAUFT"""

        caption = re.sub(self.bad_punct_regex, r" ", caption)  # ***AUSVERKAUFT***, #AUSVERKAUFT
        caption = re.sub(r"\s+\.\s+", r" ", caption)  # " . "

        # this-is-my-cute-cat / this_is_my_cute_cat
        regex2 = re.compile(r"(?:\-|\_)")
        if len(re.findall(regex2, caption)) > 3:
            caption = re.sub(regex2, " ", caption)

        caption = ftfy.fix_text(caption)
        caption = html.unescape(html.unescape(caption))

        caption = re.sub(r"\b[a-zA-Z]{1,3}\d{3,15}\b", "", caption)  # jc6640
        caption = re.sub(r"\b[a-zA-Z]+\d+[a-zA-Z]+\b", "", caption)  # jc6640vc
        caption = re.sub(r"\b\d+[a-zA-Z]+\d+\b", "", caption)  # 6640vc231

        caption = re.sub(r"(worldwide\s+)?(free\s+)?shipping", "", caption)
        caption = re.sub(r"(free\s)?download(\sfree)?", "", caption)
        caption = re.sub(r"\bclick\b\s(?:for|on)\s\w+", "", caption)
        caption = re.sub(r"\b(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)(\simage[s]?)?", "", caption)
        caption = re.sub(r"\bpage\s+\d+\b", "", caption)

        caption = re.sub(r"\b\d*[a-zA-Z]+\d+[a-zA-Z]+\d+[a-zA-Z\d]*\b", r" ", caption)  # j2d1a2a...

        caption = re.sub(r"\b\d+\.?\d*[xх×]\d+\.?\d*\b", "", caption)

        caption = re.sub(r"\b\s+\:\s+", r": ", caption)
        caption = re.sub(r"(\D[,\./])\b", r"\1 ", caption)
        caption = re.sub(r"\s+", " ", caption)

        caption.strip()

        caption = re.sub(r"^[\"\']([\w\W]+)[\"\']$", r"\1", caption)
        caption = re.sub(r"^[\'\_,\-\:;]", r"", caption)
        caption = re.sub(r"[\'\_,\-\:\-\+]$", r"", caption)
        caption = re.sub(r"^\.\S+$", "", caption)

        return caption.strip()

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    @staticmethod
    def classify_height_width_bin(height: int, width: int, ratios: dict) -> Tuple[int, int]:
        """Returns binned height and width."""
        ar = float(height / width)
        closest_ratio = min(ratios.keys(), key=lambda ratio: abs(float(ratio) - ar))
        default_hw = ratios[closest_ratio]
        return int(default_hw[0]), int(default_hw[1])

    @staticmethod
    def resize_and_crop_tensor(samples: torch.Tensor, new_width: int, new_height: int) -> torch.Tensor:
        orig_height, orig_width = samples.shape[2], samples.shape[3]

        # Check if resizing is needed
        if orig_height != new_height or orig_width != new_width:
            ratio = max(new_height / orig_height, new_width / orig_width)
            resized_width = int(orig_width * ratio)
            resized_height = int(orig_height * ratio)

            # Resize
            samples = F.interpolate(
                samples, size=(resized_height, resized_width), mode="bilinear", align_corners=False
            )

            # Center Crop
            start_x = (resized_width - new_width) // 2
            end_x = start_x + new_width
            start_y = (resized_height - new_height) // 2
            end_y = start_y + new_height
            samples = samples[:, :, start_y:end_y, start_x:end_x]
        return samples

    # ================== GrounDiT ================== # 
    @staticmethod
    def classify_aspect_ratio_bin(aspect_ratio: float, ratios: dict) -> Tuple[int, int]:
        """Returns binned height and width."""
        closest_ratio = min(ratios.keys(), key=lambda ratio: abs(float(ratio) - aspect_ratio))
        default_hw = ratios[closest_ratio]
        return int(default_hw[0]), int(default_hw[1])

    @staticmethod
    def _timestep_process(current_timestep, latent_device, batch_size):
        """
        This function was originally inside the denoising loop in the PixArtAlphePipeline. 
        But to reduce unnecessary duplication in the code, we moved it outside the loop.
        """
        if not torch.is_tensor(current_timestep):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = latent_device.type == "mps"
            if isinstance(current_timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            current_timestep = torch.tensor([current_timestep], dtype=dtype, device=latent_device)
        elif len(current_timestep.shape) == 0:
            current_timestep = current_timestep[None].to(latent_device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        current_timestep = current_timestep.expand(batch_size)
        return current_timestep
    # ================== GrounDiT ================== #

    def predict_x0_noadvance(self, noise_pred, timestep, latents):
        """
        Predict clean latents x0 from x_t and noise prediction without mutating the scheduler state.
        Tries to use a cloned scheduler for robust behavior across schedulers; falls back to closed-form when needed.
        """
        try:
            import copy as _copy
            cloned_scheduler = _copy.deepcopy(self.scheduler)
            # scheduler.step returns (prev_sample, pred_original_sample)
            _, pred_x0 = cloned_scheduler.step(noise_pred, timestep, latents, return_dict=False)
            return pred_x0
        except Exception:
            pass

        # Fallback: compute using alphas_cumprod when available
        alphas_cumprod = getattr(self.scheduler, "alphas_cumprod", None)
        prediction_type = getattr(getattr(self.scheduler, "config", None), "prediction_type", "epsilon")
        if alphas_cumprod is None:
            # Last-resort: avoid crashing; return input latents (no guidance)
            return latents

        if isinstance(timestep, torch.Tensor):
            # Use scalar value (assume same t across batch) for indexing
            t_idx = int(timestep.flatten()[0].item())
        else:
            t_idx = int(timestep)

        alpha_prod_t = alphas_cumprod[t_idx].to(device=latents.device, dtype=latents.dtype)
        sqrt_alpha_prod_t = torch.sqrt(alpha_prod_t)
        sqrt_one_minus_alpha_prod_t = torch.sqrt(1 - alpha_prod_t)

        if prediction_type in ["epsilon", "eps"]:
            pred_x0 = (latents - sqrt_one_minus_alpha_prod_t * noise_pred) / sqrt_alpha_prod_t
        elif prediction_type in ["v_prediction", "v"]:
            # x_t = sqrt(alpha_bar) * x0 + sqrt(1 - alpha_bar) * eps
            # v = sqrt(alpha_bar) * eps - sqrt(1 - alpha_bar) * x0
            # => x0 = sqrt(alpha_bar) * x_t - sqrt(1 - alpha_bar) * v
            pred_x0 = sqrt_alpha_prod_t * latents - sqrt_one_minus_alpha_prod_t * noise_pred
        else:
            pred_x0 = (latents - sqrt_one_minus_alpha_prod_t * noise_pred) / sqrt_alpha_prod_t

        return pred_x0

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: str = "",
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        guidance_scale: float = 4.5,
        num_images_per_prompt: Optional[int] = 1,
        height: Optional[int] = None,
        width: Optional[int] = None,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        prompt_attention_mask: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_attention_mask: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        clean_caption: bool = True,
        use_resolution_binning: bool = True,
        max_sequence_length: int = 120,

        # (1) General Arguments
        bbox_list = None, 
        phrases = None,
        phrases_idx = None,
        # (2) Layout guidance
        loss_scale = 10,
        loss_threshold = 0.00001,
        gradient_weight = 5.0,
        global_update_max_iter_per_step = 3,

        # (3) Reference Image Guidance (CLIP)
        ref_image_paths: Optional[List[str]] = ["/home/jaesang/i2i/images/pattern.png"],
        clip_guidance_scale: float = 2.0,
        # Region for CLIP style guidance (normalized [x0,y0,x1,y1] in [0,1])
        clip_guidance_regions: Optional[List[List[float]]] = [(0.15, 0.10, 0.40, 0.3)],

        # (4) Text Guidance (CLIP)
        text_guidance_scale: float = 2.0,
        text_guidance_text: Optional[Union[str, List[str]]] = ["Eyes closed"],
        # Regions for text guidance (normalized [x0,y0,x1,y1] in [0,1]); either single or per-text list
        text_guidance_regions: Optional[List[List[float]]] = [(0.15, 0.10, 0.40, 1.0)],

        # (5) Enable/disable guidance types
        enable_grounding_guidance: bool = False,
        enable_clip_guidance: bool = True,
        enable_text_guidance: bool = False,
        enable_edge_guidance: bool = False,

        # (6) Guidance application intervals (step index inclusive ranges)
        layout_guidance_intervals: Optional[List[Tuple[int, int]]] = [(0, 25)], # best
        clip_guidance_intervals: Optional[List[Tuple[int, int]]] = [(35, 50)],
        text_guidance_intervals: Optional[List[Tuple[int, int]]] = [(35, 50)],
        edge_guidance_intervals: Optional[List[Tuple[int, int]]] = [(0, 30)],
        # (7) Save intermediates
        save_intermediates_dir: Optional[str] = None,
        save_edge_maps_dir: Optional[str] = "/home/jaesang/i2i/edge_maps",
        save_target_edge_maps_dir: Optional[str] = "/home/jaesang/i2i/target_edge_maps",

        # (8) Edge guidance inputs
        edge_guidance_images: Optional[List[Union[str, torch.Tensor]]] = ["/home/jaesang/i2i/sketch/dog_left.png", "/home/jaesang/i2i/sketch/cat_right.png"],
        edge_guidance_scale: float = 2.0,
        # Preprocess hard line drawings into soft edges before Sobel
        # Options: "auto" | "hed" | "canny" | "none"
        edge_preprocessor: str = "hed",

        **kwargs,
    ) -> Union[ImagePipelineOutput, Tuple]:
        """
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            num_inference_steps (`int`, *optional*, defaults to 100):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process. If not defined, equal spaced `num_inference_steps`
                timesteps are used. Must be in descending order.
            guidance_scale (`float`, *optional*, defaults to 4.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            height (`int`, *optional*, defaults to self.unet.config.sample_size):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size):
                The width in pixels of the generated image.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            prompt_attention_mask (`torch.FloatTensor`, *optional*): Pre-generated attention mask for text embeddings.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. For PixArt-Alpha this negative prompt should be "". If not
                provided, negative_prompt_embeds will be generated from `negative_prompt` input argument.
            negative_prompt_attention_mask (`torch.FloatTensor`, *optional*):
                Pre-generated attention mask for negative text embeddings.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.IFPipelineOutput`] instead of a plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            clean_caption (`bool`, *optional*, defaults to `True`):
                Whether or not to clean the caption before creating embeddings. Requires `beautifulsoup4` and `ftfy` to
                be installed. If the dependencies are not installed, the embeddings will be created from the raw
                prompt.
            use_resolution_binning (`bool` defaults to `True`):
                If set to `True`, the requested height and width are first mapped to the closest resolutions using
                `ASPECT_RATIO_1024_BIN`. After the produced latents are decoded into images, they are resized back to
                the requested resolution. Useful for generating non-square images.
            max_sequence_length (`int` defaults to 120): Maximum sequence length to use with the `prompt`.
            save_intermediates_dir (`str`, *optional*):
                If provided, saves the decoded predicted image x0 at every denoising step into this directory as
                files named like `step_XXXX_img_YY.png`. Decoding follows the same postprocess/cropping used for final
                output when resolution binning is enabled.

        Examples:

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images
        """
        if "mask_feature" in kwargs:
            deprecation_message = "The use of `mask_feature` is deprecated. It is no longer used in any computation and that doesn't affect the end results. It will be removed in a future version."
            deprecate("mask_feature", "1.0.0", deprecation_message, standard_warn=False)
        # 1. Check inputs. Raise error if not correct
        height = height or self.transformer.config.sample_size * self.vae_scale_factor
        width = width or self.transformer.config.sample_size * self.vae_scale_factor
        if use_resolution_binning:
            if self.transformer.config.sample_size == 128:
                aspect_ratio_bin = ASPECT_RATIO_1024_BIN
            elif self.transformer.config.sample_size == 64:
                aspect_ratio_bin = ASPECT_RATIO_512_BIN
            elif self.transformer.config.sample_size == 32:
                aspect_ratio_bin = ASPECT_RATIO_256_BIN
            else:
                raise ValueError("Invalid sample size")
            orig_height, orig_width = height, width
            height, width = self.classify_height_width_bin(height, width, ratios=aspect_ratio_bin)

        self.check_inputs(
            prompt,
            height,
            width,
            negative_prompt,
            callback_steps,
            prompt_embeds,
            negative_prompt_embeds,
            prompt_attention_mask,
            negative_prompt_attention_mask,
        )

        # 2. Default height and width to transformer
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        ) = self.encode_prompt(
            prompt,
            do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            device=device,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            clean_caption=clean_caption,
            max_sequence_length=max_sequence_length,
        )
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)

        # Encode phrases
        phrase_embeds_list = []
        phrase_attention_mask_list = []

        for phrase in phrases:
            (
                phrase_embeds,
                phrase_attention_mask,
                negative_prompt_embeds_obj,
                negative_prompt_attention_mask_obj,
            ) = self.encode_prompt(
                phrase,
                do_classifier_free_guidance,
                negative_prompt=negative_prompt,
                num_images_per_prompt=num_images_per_prompt,
                device=device,
                prompt_embeds=None,
                negative_prompt_embeds=None,
                prompt_attention_mask=None,
                negative_prompt_attention_mask=None,
                clean_caption=clean_caption,
                max_sequence_length=max_sequence_length,
            )
            if do_classifier_free_guidance:
                phrase_embeds = torch.cat([negative_prompt_embeds_obj, phrase_embeds], dim=0)
                phrase_attention_mask = torch.cat([negative_prompt_attention_mask_obj, phrase_attention_mask], dim=0)
            
            phrase_embeds_list.append(phrase_embeds)
            phrase_attention_mask_list.append(phrase_attention_mask)

       # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)

        # 5. Prepare latents.
        latent_channels = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            latent_channels,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 6.1 Prepare micro-conditions.
        added_cond_kwargs = {
            "resolution": None, 
            "aspect_ratio": None
        }
        
        if self.transformer.config.sample_size == 128:
            # Actual resolution of the image
            resolution = torch.tensor([height, width])
            resolution = resolution.repeat(batch_size * num_images_per_prompt, 1)
            resolution = resolution.to(dtype=prompt_embeds.dtype, device=device)

            # Aspect ratio of the image
            aspect_ratio = torch.tensor([float(height / width)])
            aspect_ratio = aspect_ratio.repeat(batch_size * num_images_per_prompt, 1)
            aspect_ratio = aspect_ratio.to(dtype=prompt_embeds.dtype, device=device)

            if do_classifier_free_guidance:
                resolution = torch.cat([resolution, resolution], dim=0)
                aspect_ratio = torch.cat([aspect_ratio, aspect_ratio], dim=0)

            added_cond_kwargs = {
                "resolution": resolution, 
                "aspect_ratio": aspect_ratio
            }
        

        # Prepare CLIP image encoder(s) if reference guidance requested and enabled
        if enable_clip_guidance and clip_guidance_scale is not None and clip_guidance_scale > 0.0:
            # Multiple reference images path list takes precedence when provided
            if ref_image_paths is not None and isinstance(ref_image_paths, (list, tuple)) and len(ref_image_paths) > 0:
                cached = getattr(self, "_clip_ref_image_paths", None)
                if not hasattr(self, "clip_image_encoders") or cached is None or tuple(ref_image_paths) != tuple(cached):
                    encoders: List[CLIPEncoder] = []
                    for _p in ref_image_paths:
                        try:
                            enc = CLIPEncoder(need_ref=True, ref_path=_p).to(device)
                            if hasattr(enc, "ref"):
                                enc.ref = enc.ref.to(device)
                            encoders.append(enc)
                        except Exception:
                            continue
                    if len(encoders) > 0:
                        self.clip_image_encoders = encoders
                        self._clip_ref_image_paths = tuple(ref_image_paths)
                        # Back-compat: also expose first encoder at legacy attribute
                        self.clip_image_encoder = self.clip_image_encoders[0]
            # Fallback to single-path behavior
            elif ref_image_path is not None:
                if not hasattr(self, "clip_image_encoder") or getattr(self, "_clip_ref_image_path", None) != ref_image_path:
                    try:
                        self.clip_image_encoder = CLIPEncoder(need_ref=True, ref_path=ref_image_path).to(device)
                        if hasattr(self.clip_image_encoder, "ref"):
                            self.clip_image_encoder.ref = self.clip_image_encoder.ref.to(device)
                        self._clip_ref_image_path = ref_image_path
                        # Keep a list form for unified downstream handling
                        self.clip_image_encoders = [self.clip_image_encoder]
                        self._clip_ref_image_paths = (ref_image_path,)
                    except Exception:
                        pass

        # Prepare CLIP text encoder if text guidance is enabled
        if enable_text_guidance and text_guidance_scale is not None and text_guidance_scale > 0.0:
            if not hasattr(self, "clip_text_encoder"):
                # self.clip_text_encoder = CLIPEncoder().to(device)
                self.clip_text_encoder = SigLIPEncoder().to(device)
            # except Exception:
            #     self.clip_text_encoder = None
            #     text_guidance_scale = 0.0


        # 7. Denoising loop
        self.set_progress_bar_config(leave=False)
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        # Prepare directory for saving intermediates, if requested
        save_intermediates = save_intermediates_dir is not None and str(save_intermediates_dir) != ""
        if save_intermediates:
            try:
                os.makedirs(save_intermediates_dir, exist_ok=True)
            except Exception:
                save_intermediates = False
        
        # Prepare directory for saving attention-based edge maps, if requested
        save_edge_maps = save_edge_maps_dir is not None and str(save_edge_maps_dir) != ""
        if save_edge_maps:
            try:
                os.makedirs(save_edge_maps_dir, exist_ok=True)
            except Exception:
                save_edge_maps = False
        # Prepare directory for saving target edge maps, if requested
        save_target_edge_maps = save_target_edge_maps_dir is not None and str(save_target_edge_maps_dir) != ""
        if save_target_edge_maps:
            try:
                os.makedirs(save_target_edge_maps_dir, exist_ok=True)
            except Exception:
                save_target_edge_maps = False
        
        denoiser_args = {
            "prompt_embeds": prompt_embeds, 
            "prompt_attention_mask": prompt_attention_mask, 
            "added_cond_kwargs": added_cond_kwargs,
            "do_cfg": do_classifier_free_guidance,
            "guidance_scale": guidance_scale,
        }
        grounding_update_args = {
            "loss_scale": loss_scale, 
            "gradient_weight": gradient_weight, 
            'bbox_list': bbox_list, 
            'phrases_idx': phrases_idx, 
            'height': height, 
            'width': width,
        }
        clip_guidance_args = {
            'clip_guidance_scale': clip_guidance_scale,
            'gradient_weight': gradient_weight,
            'regions': clip_guidance_regions,
            'ref_image_paths': ref_image_paths,
        }
        # Edge guidance args (precompute per-object target edge maps at latent attention resolution)
        edge_guidance_args = {
            'edge_guidance_scale': edge_guidance_scale,
            'gradient_weight': gradient_weight,
            'loss_scale': loss_scale,
            'bbox_list': bbox_list,
            'phrases_idx': phrases_idx,
            'height': height,
            'width': width,
            'object_edge_maps': None,
            'save_edge_maps_dir': save_edge_maps_dir if save_edge_maps else None,
        }
        if enable_edge_guidance:
            num_objects = len(bbox_list) if bbox_list is not None else 0
            valid_imgs = isinstance(edge_guidance_images, (list, tuple)) and len(edge_guidance_images) >= num_objects and num_objects > 0
            if not valid_imgs:
                enable_edge_guidance = False
            else:
                latent_h_small = height // (self.vae_scale_factor * 2)
                latent_w_small = width // (self.vae_scale_factor * 2)

                def _to_gray_float_2d(img_like):
                    if isinstance(img_like, str):
                        im = Image.open(img_like).convert("L")
                        arr = np.array(im, dtype=np.float32) / 255.0
                        return torch.from_numpy(arr)
                    elif isinstance(img_like, torch.Tensor):
                        t = img_like.float()
                        if t.ndim == 3:
                            # [C,H,W] -> grayscale
                            if t.shape[0] == 3:
                                t = t.mean(dim=0)
                            elif t.shape[0] == 1:
                                t = t[0]
                            else:
                                t = t.mean(dim=0)
                        elif t.ndim == 2:
                            pass
                        else:
                            # Unsupported, try flatten
                            t = t.view(-1).mean().unsqueeze(0)
                        # Normalize to [0,1]
                        t_min, t_max = t.min(), t.max()
                        if (t_max - t_min) > 1e-6:
                            t = (t - t_min) / (t_max - t_min)
                        return t
                    else:
                        # Unsupported type
                        return None
                def _to_pil_rgb(img_like):
                    if isinstance(img_like, str):
                        im = Image.open(img_like).convert("RGB")
                        return im.resize((width, height))
                    elif isinstance(img_like, torch.Tensor):
                        t = img_like.detach().cpu().float()
                        if t.ndim == 2:
                            # [H,W] grayscale -> RGB
                            arr = (t.clamp(0, 1).numpy() * 255.0).astype(np.uint8)
                            im = Image.fromarray(arr, mode="L").convert("RGB")
                            return im.resize((width, height))
                        elif t.ndim == 3:
                            # [C,H,W]
                            if t.shape[0] in (1, 3):
                                if t.shape[0] == 1:
                                    arr = (t[0].clamp(0, 1).numpy() * 255.0).astype(np.uint8)
                                    im = Image.fromarray(arr, mode="L").convert("RGB")
                                else:
                                    arr = (t.clamp(0, 1).permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
                                    im = Image.fromarray(arr, mode="RGB")
                                return im.resize((width, height))
                            else:
                                # Collapse channels if unusual
                                arr = (t.mean(dim=0).clamp(0, 1).numpy() * 255.0).astype(np.uint8)
                                im = Image.fromarray(arr, mode="L").convert("RGB")
                                return im.resize((width, height))
                    elif isinstance(img_like, np.ndarray):
                        arr = img_like
                        if arr.ndim == 2:
                            im = Image.fromarray(arr.astype(np.uint8), mode="L").convert("RGB")
                        elif arr.ndim == 3 and arr.shape[2] in (1, 3, 4):
                            if arr.dtype != np.uint8:
                                arr = np.clip(arr, 0, 255).astype(np.uint8)
                            if arr.shape[2] == 1:
                                im = Image.fromarray(arr[:, :, 0], mode="L").convert("RGB")
                            elif arr.shape[2] == 3:
                                im = Image.fromarray(arr, mode="RGB")
                            else:
                                im = Image.fromarray(arr[:, :, :3], mode="RGB")
                        else:
                            im = Image.fromarray(arr.astype(np.uint8))
                        return im.resize((width, height))
                    return None
                def _pil_to_gray_float_2d(pil_im: "Image.Image"):
                    im = pil_im.convert("L")
                    arr = np.array(im, dtype=np.float32) / 255.0
                    return torch.from_numpy(arr)
                def _is_binary_like_gray(x_2d: torch.Tensor) -> bool:
                    if x_2d is None:
                        return False
                    x = x_2d.detach().float()
                    if x.numel() == 0:
                        return False
                    # Fraction of pixels near extremes
                    extreme = ((x < 0.1) | (x > 0.9)).float().mean().item()
                    # Count of unique 8-bit levels (small => likely binary/line art)
                    try:
                        uniq = torch.unique((x * 255).to(torch.uint8)).numel()
                    except Exception:
                        uniq = 256
                    return (extreme > 0.95) or (uniq <= 8 and extreme > 0.8)
                def _preprocess_to_soft_edges(img_like, method: str) -> Optional[torch.Tensor]:
                    """
                    Use controlnet_aux Processor to convert to soft edge maps.
                    Returns a 2D torch tensor in [0,1] if successful, else None.
                    """
                    pil_rgb = _to_pil_rgb(img_like)
                    if pil_rgb is None:
                        return None
                    proc_id = "softedge_hed" if method in ("hed", "softedge_hed") else ("canny" if method == "canny" else "softedge_hed")
                    from controlnet_aux.processor import Processor
                    processor = Processor(proc_id)
                    pil_out = processor(pil_rgb, to_pil=True)
                    if isinstance(pil_out, Image.Image):
                        return _pil_to_gray_float_2d(pil_out)
                    # Some processors may return numpy arrays
                    if isinstance(pil_out, np.ndarray):
                        if pil_out.ndim == 2:
                            arr = pil_out.astype(np.float32)
                        else:
                            arr = pil_out[..., 0:3].mean(axis=-1).astype(np.float32)
                        arr = arr / (255.0 if arr.max() > 1.0 else (arr.max() + 1e-6))
                        return torch.from_numpy(arr)

                    # Fallback: approximate with simple edge filter + blur to soften
                    try:
                        from PIL import ImageFilter
                        edge = pil_rgb.convert("L").filter(ImageFilter.FIND_EDGES).filter(ImageFilter.GaussianBlur(radius=1.5))
                        return _pil_to_gray_float_2d(edge)
                    except Exception:
                        return None
                conv_dtype = self.smth_3.weight.dtype if hasattr(self.smth_3, "weight") else self.sobel_conv_x.weight.dtype
                def _compute_sobel_edge_2d(x_2d: torch.Tensor):
                    # resize to latent attention resolution and match module dtype/device
                    conv_dtype = self.smth_3.weight.dtype if hasattr(self.smth_3, "weight") else self.sobel_conv_x.weight.dtype
                    x = x_2d.to(device=device, dtype=conv_dtype)[None, None, :, :]
                    x = F.interpolate(x, size=(latent_h_small, latent_w_small), mode="bilinear", align_corners=False)
                    # reflect pad -> smooth -> sobel
                    x = F.pad(x, (1, 1, 1, 1), mode="reflect")
                    x = self.smth_3(x)
                    sx = self.sobel_conv_x(x)
                    sy = self.sobel_conv_y(x)
                    mag = torch.sqrt(sx.pow(2) + sy.pow(2)).squeeze(0).squeeze(0)
                    mag = mag / (mag.max() + 1e-6) if mag.max() > 0 else mag
                    return mag

                object_edge_maps: List[torch.Tensor] = []
                for obj_idx in range(num_objects):
                    # Load raw grayscale
                    g_raw = _to_gray_float_2d(edge_guidance_images[obj_idx])
                    if g_raw is None:
                        enable_edge_guidance = False
                        break
                    # Optionally convert hard binary/line drawings to soft edges via HED/Canny
                    g_for_edge: torch.Tensor
                    if edge_preprocessor in ("hed", "softedge_hed", "canny"):
                        g_soft = _preprocess_to_soft_edges(edge_guidance_images[obj_idx], "hed" if edge_preprocessor in ("hed", "softedge_hed") else "canny")
                        g_for_edge = g_soft if g_soft is not None else g_raw
                    elif edge_preprocessor == "auto":
                        if _is_binary_like_gray(g_raw):
                            g_soft = _preprocess_to_soft_edges(edge_guidance_images[obj_idx], "hed")
                            g_for_edge = g_soft if g_soft is not None else g_raw
                        else:
                            g_for_edge = g_raw
                    else:
                        g_for_edge = g_raw
                    # Compute Sobel magnitude on the (optionally) softened edge map
                    target_edge = _compute_sobel_edge_2d(g_for_edge)
                    target_edge = target_edge.to(device=device, dtype=conv_dtype)
                    if save_target_edge_maps:
                        arr = (target_edge.clamp(0, 1).detach().cpu().numpy() * 255.0).astype(np.uint8)
                        im = Image.fromarray(arr, mode="L")
                        im.save(os.path.join(save_target_edge_maps_dir, f"target_obj_{obj_idx:02d}.png"))
                    object_edge_maps.append(target_edge)

                if enable_edge_guidance:
                    edge_guidance_args['object_edge_maps'] = object_edge_maps
        # Text guidance args
        # Use provided text or fallback to prompt
        if text_guidance_text is None:
            if isinstance(prompt, list):
                _text_list = [str(p) for p in prompt]
                _text_prompt = " ".join(_text_list)
            else:
                _t = str(prompt) if prompt is not None else ""
                _text_list = [_t]
                _text_prompt = _t
        else:
            if isinstance(text_guidance_text, list):
                _text_list = [str(p) for p in text_guidance_text]
                _text_prompt = " ".join(_text_list)
            else:
                _t = str(text_guidance_text)
                _text_list = [_t]
                _text_prompt = _t
        text_guidance_args = {
            'text_guidance_scale': text_guidance_scale,
            'text': _text_prompt,
            'texts': _text_list,
            'regions': text_guidance_regions,
        }

        # Guidance Update functions
        GroundingUpdate = partial(GroundingUpdateFunc, denoiser_args=denoiser_args, loss_args=grounding_update_args, enabled=enable_grounding_guidance)
        ClipUpdate = partial(ClipGuidanceUpdateFunc, denoiser_args=denoiser_args, clip_args=clip_guidance_args, enabled=enable_clip_guidance)
        TextUpdate = partial(TextGuidanceUpdateFunc, denoiser_args=denoiser_args, text_args=text_guidance_args, enabled=enable_text_guidance)
        EdgeUpdate = partial(EdgeGuidanceUpdateFunc, denoiser_args=denoiser_args, edge_args=edge_guidance_args, enabled=enable_edge_guidance)

        # Normalize default intervals
        if not enable_grounding_guidance:
            layout_guidance_intervals = []
        if not enable_clip_guidance:
            clip_guidance_intervals = []
        elif clip_guidance_intervals is None:
            clip_guidance_intervals = list(layout_guidance_intervals) if clip_guidance_scale and clip_guidance_scale > 0.0 else []
        if not enable_text_guidance:
            text_guidance_intervals = []
        elif text_guidance_intervals is None:
            # If None, mirror clip intervals when provided; otherwise use entire range
            text_guidance_intervals = list(clip_guidance_intervals) if len(clip_guidance_intervals or []) > 0 else [(0, num_inference_steps - 1)]
        if not enable_edge_guidance:
            edge_guidance_intervals = []
        elif edge_guidance_intervals is None:
            edge_guidance_intervals = list(layout_guidance_intervals)

        def _in_intervals(step_idx: int, intervals: List[Tuple[int, int]]) -> bool:
            for s, e in intervals:
                if s <= step_idx <= e:
                    return True
            return False

        with torch.enable_grad():
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    # =========== Grounding Guidance Update =========== #
                    if enable_grounding_guidance and _in_intervals(i, layout_guidance_intervals):
                        _iter_cnt = 0
                        while _iter_cnt < global_update_max_iter_per_step:
                            latents, loss = GroundingUpdate(self, latents, t, i)
                            if loss is None or loss <= loss_threshold:
                                break
                            _iter_cnt += 1

                    # =========== Edge Guidance Update =========== #
                    if enable_edge_guidance and _in_intervals(i, edge_guidance_intervals):
                        _iter_cnt = 0
                        while _iter_cnt < global_update_max_iter_per_step:
                            latents, _edge_loss_val = EdgeUpdate(self, latents, t, i)
                            if _edge_loss_val is None or _edge_loss_val <= loss_threshold:
                                break
                            _iter_cnt += 1

                    # =========== CLIP Image Guidance Update =========== #
                    if enable_clip_guidance and _in_intervals(i, clip_guidance_intervals):
                        _iter_cnt = 0
                        while _iter_cnt < global_update_max_iter_per_step:
                            latents, _clip_loss_val = ClipUpdate(self, latents, t, i)
                            # No threshold break required, but keep parity with loop control
                            if _clip_loss_val is None or _clip_loss_val <= loss_threshold:
                                break
                            _iter_cnt += 1

                    # =========== CLIP Text Guidance Update =========== #
                    if enable_text_guidance and _in_intervals(i, text_guidance_intervals):
                        _iter_cnt = 0
                        while _iter_cnt < global_update_max_iter_per_step:
                            latents, _text_loss_val = TextUpdate(self, latents, t, i)
                            if _text_loss_val is None or _text_loss_val <= loss_threshold:
                                break
                            _iter_cnt += 1

                    # =========== Edge Map Save (if requested and not already saved via EdgeGuidance) =========== #
                    if save_edge_maps and not (enable_edge_guidance and _in_intervals(i, edge_guidance_intervals)):
                        try:
                            with torch.no_grad():
                                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                                current_timestep = self._timestep_process(t, latent_model_input.device, latent_model_input.shape[0])
                                _, attn_maps, _ = self.transformer(
                                    latent_model_input,
                                    encoder_hidden_states=prompt_embeds,
                                    encoder_attention_mask=prompt_attention_mask,
                                    timestep=current_timestep,
                                    added_cond_kwargs=added_cond_kwargs,
                                    return_dict=False,
                                    return_attn_maps=True,
                                    is_object_branch=False,
                                )
                                latent_h = height // (self.vae_scale_factor * 2)
                                latent_w = width // (self.vae_scale_factor * 2)
                                save_attention_edge_maps(
                                    attn_maps=attn_maps,
                                    object_positions=phrases_idx,
                                    bbox_list=bbox_list,
                                    latent_h=latent_h,
                                    latent_w=latent_w,
                                    pipe=self,
                                    save_dir=save_edge_maps_dir,
                                    step_idx=i,
                                )
                                del attn_maps
                                torch.cuda.empty_cache()
                        except Exception:
                            pass

                    with torch.no_grad():
                        misc_args = {
                            "do_cfg": do_classifier_free_guidance, 
                            "guidance_scale": guidance_scale, 
                            "t": t, 
                            "latents": latents, 
                            "extra_step_kwargs": extra_step_kwargs, 
                            "latent_channels": latent_channels
                        }

                        # =========== Main Branch =========== #
                        # Main Branch Arguments
                        prompt_args = {
                            "prompt_embeds": prompt_embeds, 
                            "prompt_attention_mask": prompt_attention_mask, 
                            "added_cond_kwargs": added_cond_kwargs
                        }
                        
                        # Main Branch Denoising
                        latents, x0_pred = MainBranch(
                            prompt_args, 
                            misc_args, 
                            latents, 
                            self
                        )
                        # =========== Main Branch =========== #
                        
                        # Optionally save intermediate predicted x0 images for each step
                        if save_intermediates:
                            try:
                                z_mid = (x0_pred.to(dtype=self.vae.dtype)) / self.vae.config.scaling_factor
                                img_mid = self.vae.decode(z_mid, return_dict=False)[0]
                                if use_resolution_binning:
                                    img_mid = self.resize_and_crop_tensor(img_mid, orig_width, orig_height)
                                images_mid = self.image_processor.postprocess(img_mid, output_type="pil")
                                for b_idx, im_mid in enumerate(images_mid):
                                    im_mid.save(os.path.join(save_intermediates_dir, f"step_{i:04d}_img_{b_idx:02d}.png"))
                            except Exception:
                                pass


                        # call the callback, if provided 
                        if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                            progress_bar.update()
                            if callback is not None and i % callback_steps == 0:
                                step_idx = i // getattr(self.scheduler, "order", 1)
                                callback(step_idx, t, x0_pred)

        # Save trajectory images if callback was used
        if not output_type == "latent":
            z = (latents.to(dtype=self.vae.dtype)) / self.vae.config.scaling_factor
            image = self.vae.decode(z, return_dict=False)[0]
            if use_resolution_binning:
                image = self.resize_and_crop_tensor(image, orig_width, orig_height)
        else:
            image = latents

        if not output_type == "latent":
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Save denoise trajectory grid if available
        try:
            if len(trajectory_images_per_latent) > 0 and all(len(row) > 0 for row in trajectory_images_per_latent):
                num_rows = len(trajectory_images_per_latent)
                num_cols = max(len(r) for r in trajectory_images_per_latent)
                fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 2.0, num_rows * 2.0))
                if num_rows == 1:
                    axes = __import__("numpy").array(axes).reshape(1, -1)
                for r in range(num_rows):
                    for c in range(num_cols):
                        if c < len(trajectory_images_per_latent[r]):
                            axes[r, c].imshow(trajectory_images_per_latent[r][c])
                        axes[r, c].axis("off")
                        if r == 0 and c < len(trajectory_step_indices):
                            axes[r, c].set_title(f"step {trajectory_step_indices[c]}")
                    if num_cols > 0:
                        axes[r, 0].set_ylabel(f"L{r}")
                fig.tight_layout()
                plt.savefig(os.path.join(sample_dir, "denoise_trajectory.png"), dpi=150, bbox_inches="tight")
                plt.close(fig)
        except Exception:
            pass

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image, )

        return ImagePipelineOutput(images=image)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="results")
    parser.add_argument("--model_version", choices=["512", "1024"], default="512")
    parser.add_argument("--input_config_path", type=str, default="config/config.json")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=67)
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of inference steps.")
    # parser.add_argument("--ref_image_path", type=str, default="/home/jaesang/i2i/images/pattern.png", help="Path to reference image for CLIP guidance.")
    # parser.add_argument("--clip_guidance_scale", type=float, default=5.0, help="Scale for CLIP image guidance loss.")

    args = parser.parse_args()

    def _parse_intervals(interval_str_or_list):
        """
        Parse a string like '0-25,30-40' or '10,20-30' into a list of (start, end) integer tuples.
        If already a list of lists/tuples, coerce to list of tuples[int,int].
        """
        if interval_str_or_list is None:
            return None
        if isinstance(interval_str_or_list, (list, tuple)):
            out = []
            for it in interval_str_or_list:
                if isinstance(it, (list, tuple)) and len(it) == 2:
                    out.append((int(it[0]), int(it[1])))
            return out if len(out) > 0 else None
        s = str(interval_str_or_list).strip()
        if len(s) == 0:
            return None
        intervals = []
        for part in s.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                start, end = part.split("-", 1)
            elif ":" in part:
                start, end = part.split(":", 1)
            else:
                start = end = part
            intervals.append((int(start), int(end)))
        return intervals if len(intervals) > 0 else None

    def _as_list(x):
        if x is None:
            return None
        return x if isinstance(x, (list, tuple)) else [x]

    def _expand_regions(regions, needed_len, default_region=(0.0, 0.0, 1.0, 1.0)):
        """
        Ensure 'regions' is a list with length 'needed_len', padding/truncating as required.
        """
        if needed_len is None or needed_len <= 0:
            return None
        if regions is None:
            return [list(default_region) for _ in range(needed_len)]
        reg_list = list(regions) if isinstance(regions, (list, tuple)) else [regions]
        # normalize each region to list[4]
        norm_regs = []
        for r in reg_list:
            if isinstance(r, (list, tuple)) and len(r) == 4:
                norm_regs.append([float(r[0]), float(r[1]), float(r[2]), float(r[3])])
        if len(norm_regs) >= needed_len:
            return norm_regs[:needed_len]
        # pad
        while len(norm_regs) < needed_len:
            norm_regs.append(list(default_region))
        return norm_regs
    # Set seed and device
    device = torch.device(f"cuda:{args.gpu_id}")
    seed_everything(args.seed)
    generator = torch.Generator(device=device).manual_seed(args.seed)

    # Data loading
    with open(args.input_config_path, "r") as f:
        total_data_input = json.load(f)
    # Support optional global guidance config under "guidance" key
    guidance_config = total_data_input.get("guidance", {}) if isinstance(total_data_input, dict) else {}
    # Count only numeric keys as samples
    sample_keys = []
    if isinstance(total_data_input, dict):
        for k in total_data_input.keys():
            if str(k).isdigit():
                sample_keys.append(str(k))
        sample_keys = sorted(sample_keys, key=lambda x: int(x))
        num_total_data = len(sample_keys)
    else:
        num_total_data = 0
    print(f"Will process {num_total_data} data.")
    # Create timestamped root directory: results/{timestamp}_seed{seed}
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_root = os.path.join(args.save_dir, f"{ts}_seed{args.seed}")
    os.makedirs(save_root, exist_ok=True)

    # Load model
    if args.model_version == "512":
        hw_bin = ASPECT_RATIO_512_BIN
        model_id = "PixArt-alpha/PixArt-XL-2-512x512"
    elif args.model_version == "1024":
        # WARNING: 1024 version is not tested due to GPU memory limitation !!!
        hw_bin = ASPECT_RATIO_1024_BIN
        model_id = "PixArt-alpha/PixArt-XL-2-1024-MS"
    else:
        raise ValueError(f"Invalid model version: {args.model_version}. Choose either 512 or 1024.")
    
    # pipe, tokenizer = load_groundit_model(model_id, device)

    # Load base GroundDiT components and build dedicated dual pipeline
    pipe, tokenizer = load_i2i_model(model_id, device)


    # Generate samples for each data input 
    for idx in tqdm(range(num_total_data), desc="Generating Samples: "):
        # Create a directory and construct the path for saving the samples
        sample_dir = os.path.join(save_root, str(idx))
        os.makedirs(sample_dir, exist_ok=True)
        img_path = os.path.join(sample_dir, "img.png")
        img_with_bbox_path = os.path.join(sample_dir, "img_with_bbox.png")

        # Fetch data from the input config
        data = total_data_input[str(idx)]
        prompt = data["prompt"]
        phrases = data["phrases"]
        bboxes = data["bboxes"]

        bbox_list = sanity_check(bboxes, phrases)
        
        # Find the location of phrase indices in the prompt after tokenization, do it for all pharse in phrases list.
        phrases_idx = get_phrases_idx_in_prompt(prompt, phrases, tokenizer)

        # Convert bbox coordinates to pixel & latent & patch space. Get the indices of the patches in the patch space that are covered by the bounding box.
        # Here boundnig box region corresponds to "Local Patch" in the paper. See Figure 2.
        if 'height' in data and 'width' in data:
            original_height, original_width = data['height'], data['width']
            target_height, target_width = pipe.classify_height_width_bin(original_height, original_width, hw_bin)
        elif 'aspect_ratio' in data:
            target_height, target_width = pipe.classify_aspect_ratio_bin(data['aspect_ratio'], hw_bin)
            original_height, original_width = target_height, target_width
        else:
            raise ValueError("Invalid data format. Need to provide either height/width or aspect_ratio.")
            
        latent_height, latent_width = target_height // pipe.vae_scale_factor, target_width // pipe.vae_scale_factor
        # all_bbox_coord_in_pixel_space = get_bbox_coord_in_pixel_space(bbox_list, target_height, target_width)
        # all_bbox_coord_in_latent_space = get_bbox_coord_in_latent_space(bbox_list, latent_height, latent_width)

        # # Get the "Object Image" height and width in pixel space, where "Object Image" is introduced in the paper. See Figure 2.
        # object_image_hw_in_pixel_space = get_bbox_region_hw(
        #     all_bbox_coord_in_pixel_space, 
        #     hw_bin_classify_func=partial(pipe.classify_height_width_bin, ratios=hw_bin)
        # )

        # Main image latent
        latent_shape = (1, 4, latent_height, latent_width)
        latent = randn_tensor(
            latent_shape, generator=generator, 
            device=device, dtype=torch.float16
        ) * pipe.scheduler.init_noise_sigma

        # # Run
        # # Denoise trajectory capture (per-sample)
        trajectory_images_per_latent = [[]]
        trajectory_step_indices = []

        def _traj_callback(step_idx, t, latents_cb):
            try:
                with torch.no_grad():
                    img_tensor = pipe.vae.decode(latents_cb / pipe.vae.config.scaling_factor, return_dict=False)[0]
                    # Match original requested resolution if binning was used
                    img_tensor = pipe.resize_and_crop_tensor(img_tensor, original_width, original_height)
                    imgs = pipe.image_processor.postprocess(img_tensor, output_type="pil")
                    max_side = 256
                    for idx_img in range(min(1, len(imgs))):
                        im = imgs[idx_img].copy()
                        im.thumbnail((max_side, max_side))
                        trajectory_images_per_latent[idx_img].append(im)
                    trajectory_step_indices.append(int(step_idx))
            except Exception:
                pass

        # Merge global guidance config with per-sample override
        per_sample_guidance = data.get("guidance", {}) if isinstance(data, dict) else {}
        gconf = {}
        if isinstance(guidance_config, dict):
            gconf.update(guidance_config)
        if isinstance(per_sample_guidance, dict):
            gconf.update(per_sample_guidance)

        # Resolve enable flags
        enable_grounding_guidance = bool(gconf.get("enable_grounding_guidance", False))
        enable_clip_guidance = bool(gconf.get("enable_clip_guidance", True))
        enable_text_guidance = bool(gconf.get("enable_text_guidance", False))
        enable_edge_guidance = bool(gconf.get("enable_edge_guidance", False))

        # Layout/grounding params
        loss_scale = float(gconf.get("loss_scale", 10.0))
        loss_threshold = float(gconf.get("loss_threshold", 0.00001))
        gradient_weight = float(gconf.get("gradient_weight", 5.0))
        global_update_max_iter_per_step = int(gconf.get("global_update_max_iter_per_step", 3))
        layout_guidance_intervals = _parse_intervals(gconf.get("layout_guidance_intervals", None))

        # CLIP/style guidance params
        clip_guidance_scale = float(gconf.get("clip_guidance_scale", 5.0))
        ref_image_paths = gconf.get("ref_image_paths", None)
        ref_image_paths = _as_list(ref_image_paths) if ref_image_paths is not None else None
        clip_guidance_regions = gconf.get("clip_guidance_regions", None)
        clip_guidance_regions = _expand_regions(clip_guidance_regions, len(ref_image_paths))
        clip_guidance_intervals = _parse_intervals(gconf.get("clip_guidance_intervals", None))

        # Text guidance params
        text_guidance_scale = float(gconf.get("text_guidance_scale", 2.0))
        text_guidance_text = gconf.get("text_guidance_text", None)
        text_list = _as_list(text_guidance_text) if text_guidance_text is not None else None
        text_guidance_regions = gconf.get("text_guidance_regions", None)
        text_guidance_regions = _expand_regions(text_guidance_regions, len(text_list) if text_list is not None else 0)
        text_guidance_intervals = _parse_intervals(gconf.get("text_guidance_intervals", None))

        # Edge guidance params
        edge_guidance_scale = float(gconf.get("edge_guidance_scale", 2.0))
        edge_preprocessor = str(gconf.get("edge_preprocessor", "hed"))
        edge_guidance_images = gconf.get("edge_guidance_images", None)
        edge_guidance_images = list(edge_guidance_images) if isinstance(edge_guidance_images, (list, tuple)) else edge_guidance_images
        edge_guidance_intervals = _parse_intervals(gconf.get("edge_guidance_intervals", None))

        # Save directories (per-sample subdirectories if provided)
        save_intermediates_dir = gconf.get("save_intermediates_dir", None)
        if isinstance(save_intermediates_dir, str) and len(save_intermediates_dir) > 0:
            save_intermediates_dir = os.path.join(save_intermediates_dir, str(idx))
        else:
            save_intermediates_dir = None
        save_edge_maps_dir = gconf.get("save_edge_maps_dir", os.path.join(sample_dir, "edge_maps"))
        if isinstance(save_edge_maps_dir, str) and len(save_edge_maps_dir) > 0:
            save_edge_maps_dir = os.path.join(save_edge_maps_dir, str(idx)) if not save_edge_maps_dir.startswith(sample_dir) else save_edge_maps_dir
        save_target_edge_maps_dir = gconf.get("save_target_edge_maps_dir", os.path.join(sample_dir, "target_edge_maps"))
        if isinstance(save_target_edge_maps_dir, str) and len(save_target_edge_maps_dir) > 0:
            save_target_edge_maps_dir = os.path.join(save_target_edge_maps_dir, str(idx)) if not save_target_edge_maps_dir.startswith(sample_dir) else save_target_edge_maps_dir

        # Generate sample with guidance from config
        pipe_kwargs = dict(
            prompt=prompt, 
            width=original_width, 
            height=original_height, 
            latents=latent, 
            num_inference_steps=args.num_inference_steps,
            # General Arguments
            bbox_list=bbox_list, 
            phrases=phrases, 
            phrases_idx=phrases_idx,
            # Enable flags
            enable_grounding_guidance=enable_grounding_guidance,
            enable_clip_guidance=enable_clip_guidance,
            enable_text_guidance=enable_text_guidance,
            enable_edge_guidance=enable_edge_guidance,
            # Layout params
            loss_scale=loss_scale,
            loss_threshold=loss_threshold,
            gradient_weight=gradient_weight,
            global_update_max_iter_per_step=global_update_max_iter_per_step,
            # CLIP/style params
            ref_image_paths=ref_image_paths,
            clip_guidance_scale=clip_guidance_scale,
            clip_guidance_regions=clip_guidance_regions,
            # Text params
            text_guidance_text=text_list,
            text_guidance_scale=text_guidance_scale,
            text_guidance_regions=text_guidance_regions,
            # Edge params
            edge_guidance_images=edge_guidance_images,
            edge_guidance_scale=edge_guidance_scale,
            edge_preprocessor=edge_preprocessor,
            # Save dirs
            save_intermediates_dir=save_intermediates_dir,
            save_edge_maps_dir=save_edge_maps_dir,
            save_target_edge_maps_dir=save_target_edge_maps_dir,
            # Callback for trajectory thumbs
            callback=_traj_callback,
            callback_steps=1,
        )
        # Only override default intervals when explicitly provided
        if layout_guidance_intervals is not None:
            pipe_kwargs["layout_guidance_intervals"] = layout_guidance_intervals
        if clip_guidance_intervals is not None:
            pipe_kwargs["clip_guidance_intervals"] = clip_guidance_intervals
        if text_guidance_intervals is not None:
            pipe_kwargs["text_guidance_intervals"] = text_guidance_intervals
        if edge_guidance_intervals is not None:
            pipe_kwargs["edge_guidance_intervals"] = edge_guidance_intervals

        out = pipe(**pipe_kwargs)

        # Save the generated samples
        image = out.images[0]
        image.save(img_path)
        draw_box(image, bbox_list, ";".join(phrases), original_height, original_width)
        image.save(img_with_bbox_path)

        # Save denoise trajectory grid if available
        try:
            if len(trajectory_images_per_latent) > 0 and all(len(row) > 0 for row in trajectory_images_per_latent):
                num_rows = len(trajectory_images_per_latent)
                num_cols = max(len(r) for r in trajectory_images_per_latent)
                fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 2.0, num_rows * 2.0))
                if num_rows == 1:
                    axes = __import__("numpy").array(axes).reshape(1, -1)
                for r in range(num_rows):
                    for c in range(num_cols):
                        if c < len(trajectory_images_per_latent[r]):
                            axes[r, c].imshow(trajectory_images_per_latent[r][c])
                        axes[r, c].axis("off")
                        if r == 0 and c < len(trajectory_step_indices):
                            axes[r, c].set_title(f"step {trajectory_step_indices[c]}")
                    if num_cols > 0:
                        axes[r, 0].set_ylabel(f"L{r}")
                fig.tight_layout()
                plt.savefig(os.path.join(sample_dir, "denoise_trajectory.png"), dpi=150, bbox_inches="tight")
                plt.close(fig)
        except Exception:
            pass