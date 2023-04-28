import os
from typing import Tuple

import numpy as np
import PIL
import torch as th
import wandb
from tqdm import tqdm


def save_model(
    model: th.nn.Module, checkpoints_dir: str, train_idx: int, epoch: int,ddp=True
):
    if ddp:
        th.save(
            model.module.state_dict(),
            os.path.join(checkpoints_dir, f"model-{epoch}x{train_idx}.pt"),
        )
    else:
        th.save(
            model.state_dict(),
            os.path.join(checkpoints_dir, f"model-{epoch}x{train_idx}.pt"),
        )
   
    tqdm.write(
        f"Saved checkpoint {train_idx} to {checkpoints_dir}/model-{epoch}x{train_idx}.pt"
    )


def pred_to_pil(pred: th.Tensor) -> PIL.Image:
    scaled = ((pred + 1) * 127.5).round().clamp(0, 255).to(th.uint8).cpu()
    reshaped = scaled.permute(2, 0, 3, 1).reshape([pred.shape[2], -1, 3])
    return PIL.Image.fromarray(reshaped.numpy())


def pil_image_to_norm_tensor(pil_image):
    """
    Convert a PIL image to a PyTorch tensor normalized to [-1, 1] with shape [B, C, H, W].
    """
    return th.from_numpy(np.asarray(pil_image)).float().permute(2, 0, 1) / 127.5 - 1.0


def resize_for_upsample(
    original, low_res_x, low_res_y, upscale_factor: int = 4
) -> Tuple[th.Tensor, th.Tensor]:
    """
    Resize/Crop an image to the size of the low resolution image. This is useful for upsampling.

    Args:
        original: A PIL.Image object to be cropped.
        low_res_x: The width of the low resolution image.
        low_res_y: The height of the low resolution image.
        upscale_factor: The factor by which to upsample the image.

    Returns:
        The downsampled image and the corresponding upscaled version cropped according to upscale_factor.
    """
    high_res_x, high_res_y = low_res_x * upscale_factor, low_res_y * upscale_factor
    high_res_image = original.resize((high_res_x, high_res_y), PIL.Image.LANCZOS)
    high_res_tensor = pil_image_to_norm_tensor(pil_image=high_res_image)
    low_res_image = high_res_image.resize(
        (low_res_x, low_res_y), resample=PIL.Image.BICUBIC
    )
    low_res_tensor = pil_image_to_norm_tensor(pil_image=low_res_image)
    return low_res_tensor, high_res_tensor


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def wandb_setup(
    learn_sigma: bool,
    batch_size: int,
    learning_rate: float,
    device: str,
    base_dir: str,
    project_name: str = "ebm-text2im-finetune",
):
    return wandb.init(
        project=project_name,
        dir = base_dir,
        config={
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "device": device,
            "base_dir": base_dir,
            "learn_sigma": learn_sigma,
        },
    )
