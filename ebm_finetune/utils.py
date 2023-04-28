## glide_util.py
# Utilities for tokenizing, padding, and batching data and sampling from GLIDE.

import os
from typing import Tuple

import PIL
import numpy as np
import torch as th

from composable_diffusion.model_creation import (
    Sampler_create_gaussian_diffusion,
    create_gaussian_diffusion,
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    model_and_diffusion_defaults_upsampler,
)
# from composable_diffusion.tokenizer.bpe import Encoder

MODEL_TYPES = ["base", "upsample", "base-inpaint", "upsample-inpaint"]



def load_model(
    is_master: bool,
    energy_mode: bool,
    learn_sigma: bool,
    num_classes: str = "",
    model_type: str = "base",
    noise_schedule: str = "squaredcos_cap_v2",
):
    assert model_type in MODEL_TYPES, f"Model must be one of {MODEL_TYPES}. Exiting."
    if model_type in ["base", "base-inpaint"]:
        options = model_and_diffusion_defaults()
    elif model_type in ["upsample", "upsample-inpaint"]:
        options = model_and_diffusion_defaults_upsampler()
    if "inpaint" in model_type:
        options["inpaint"] = True


    options["noise_schedule"]= noise_schedule
    options["learn_sigma"] = learn_sigma
    options["use_fp16"] = False
    options["num_classes"] = None if num_classes == "" else num_classes
    options["dataset"] = "clevr_norel"
    options["image_size"] =  64 
    options["num_channels"] = 128
    options["num_res_blocks"] = 3
    options["energy_mode"] = energy_mode

    print("Using Energy Based Model?: " , energy_mode)

    if is_master:
        print(options)


    model, diffusion = create_model_and_diffusion(**options)



    return model, diffusion, options

def read_image(path: str, shape: Tuple[int, int]):
    pil_img = PIL.Image.open(path).convert('RGB')
    pil_img = pil_img.resize(shape, resample=PIL.Image.BICUBIC)
    img = np.array(pil_img)
    return th.from_numpy(img)[None].permute(0, 3, 1, 2).float() / 127.5 - 1


@th.inference_mode()
def sample(
    model,
    options,
    uncond=False,
    batch_size=1,
    guidance_scale=4,
    device="cpu",
    prediction_respacing="100",
):
    eval_diffusion = create_gaussian_diffusion(
        steps=options["diffusion_steps"],
        noise_schedule=options["noise_schedule"],
        timestep_respacing=prediction_respacing,
        learn_sigma=options["learn_sigma"],
    )

    # Change if you want to sample from different class
    size = 64
    labels = th.tensor([[2]]).long()

    full_batch_size = batch_size * (len(labels) + 1)
    masks = [True] * len(labels) + [False]
    labels = th.cat(([labels] + [th.zeros_like(labels)]), dim=0)
      


    if uncond:
        model_kwargs = dict(
        y=None,
        masks=None
            )
        full_batch_size = batch_size
        def model_fn(x, t, **kwargs):
            return model(x, t, y=None,masks=None)
    else:
        model_kwargs = dict(
        y=labels.clone().detach().to(device),
        masks=th.tensor(masks, dtype=th.bool, device=device)
            )

        def model_fn(x_t, ts, **kwargs):
            half = x_t[: len(x_t) // 2]
            combined = th.cat([half, half], dim=0)
            eps = model(combined, ts, **kwargs)
            masks = kwargs.get('masks')
            cond_eps, uncond_eps = eps[masks], eps[~masks]
            half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
            eps = th.cat([half_eps, half_eps], dim=0)
            return eps



    samples = eval_diffusion.p_sample_loop(
        model_fn,
        (full_batch_size, 3, size, size),  # only thing that's changed
        device=device,
        clip_denoised=True,
        progress=False,
        model_kwargs=model_kwargs,
        cond_fn=None,
    )[:batch_size]

    return samples



# @th.inference_mode()
def energy_sample(
    model,
    options,
    uncond=False,
    batch_size=1,
    guidance_scale=4,
    device="cpu",
    prediction_respacing="100",
):
    eval_diffusion = Sampler_create_gaussian_diffusion(
        steps=options["diffusion_steps"],
        noise_schedule=options["noise_schedule"],
        timestep_respacing=prediction_respacing,
        learn_sigma=options["learn_sigma"],
    )

    model.eval()


    # Change if you want to sample from different class
    size = 64
    labels = th.tensor([[2]]).long()



    full_batch_size = batch_size * (len(labels) + 1)
    masks = [True] * len(labels) + [False]
    labels = th.cat(([labels] + [th.zeros_like(labels)]), dim=0)


    if uncond:
        model_kwargs = dict(
        y=None,
        masks=None
            )
        full_batch_size = batch_size
        def model_fn(x, t, **kwargs):
            return model(x, t, y=None,masks=None)
    else:
        model_kwargs = dict(
        y=labels.clone().detach().to(device),
        masks=th.tensor(masks, dtype=th.bool, device=device)
            )

        def model_fn(x_t, ts, **kwargs):
            half = x_t[: len(x_t) // 2]
            combined = th.cat([half, half], dim=0)
            eps = model(combined, ts,eval=True, **kwargs)
            masks = kwargs.get('masks')
            cond_eps, uncond_eps = eps[masks], eps[~masks]
            half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
            eps = th.cat([half_eps, half_eps], dim=0)
            return eps



    samples = eval_diffusion.p_sample_loop(
        None,
        model_fn,
        (full_batch_size, 3, size, size),  # only thing that's changed
        device=device,
        clip_denoised=True,
        progress=False,
        model_kwargs=model_kwargs,
        cond_fn=None,
    )[:batch_size]
    
    model.train()
    return samples