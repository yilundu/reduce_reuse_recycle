import os
from typing import Tuple

import torch as th
from composable_diffusion.respace import SpacedDiffusion
from composable_diffusion.unet import UNetModel
from wandb import wandb

from ebm_finetune import train_util, utils


def base_train_step(
    device,
    model,
    diffusion: SpacedDiffusion,
    batch: Tuple[th.Tensor, th.Tensor, th.Tensor],
    energy_mode = False,
    uncond = False,
    learn_sigma=False,
):
    """
    Perform a single training step.

        Args:
            model: The model to train.
            diffusion: The diffusion to use.
            batch: A tuple of (tokens, masks, reals) where tokens is a tensor of shape (batch_size, seq_len), masks is a tensor of shape (batch_size, seq_len) and reals is a tensor of shape (batch_size, 3, side_x, side_y) normalized to [-1, 1].
            device: The device to use for getting model outputs and computing loss.
        Returns:
            The loss.
    """
    tokens, masks, reals = [x.to(device) for x in batch]
    
   
    timesteps = th.randint(
        0, len(diffusion.betas) - 1, (reals.shape[0],), device=device
    )
    noise = th.randn_like(reals, device=device)
    x_t = diffusion.q_sample(reals, timesteps, noise=noise).to(device)
    _, C = x_t.shape[:2]


    if uncond:
        model_output = model(
            x_t.to(device),
            timesteps.to(device),
            y=None,
            masks=None,
        )
    else:
        model_output = model(
            x_t.to(device),
            timesteps.to(device),
            y=tokens.to(device),
            masks=masks.to(device),
        )    

    if not learn_sigma:
        epsilon = model_output    
    else:
        epsilon, _ = th.split(model_output, C, dim=1)
    return th.nn.functional.mse_loss(epsilon, noise.to(device).detach())



def upsample_train_step(
    model: UNetModel,
    diffusion: SpacedDiffusion,
    batch: Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor],
    device: str,
):
    """
    Perform a single training step.

        Args:
            model: The model to train.
            diffusion: The diffusion to use.
            batch: A tuple of (tokens, masks, low_res, high_res) where 
                - tokens is a tensor of shape (batch_size, seq_len), 
                - masks is a tensor of shape (batch_size, seq_len) with dtype torch.bool
                - low_res is a tensor of shape (batch_size, 3, base_x, base_y), normalized to [-1, 1]
                - high_res is a tensor of shape (batch_size, 3, base_x*4, base_y*4), normalized to [-1, 1]
            device: The device to use for getting model outputs and computing loss.
        Returns:
            The loss.
    """
    tokens, masks, low_res_image, high_res_image = [ x.to(device) for x in batch ]
    timesteps = th.randint(0, len(diffusion.betas) - 1, (low_res_image.shape[0],), device=device)
    noise = th.randn_like(high_res_image, device=device) # Noise should be shape of output i think
    noised_high_res_image = diffusion.q_sample(high_res_image, timesteps, noise=noise).to(device)
    _, C = noised_high_res_image.shape[:2]
    model_output = model(
        noised_high_res_image.to(device),
        timesteps.to(device),
        low_res=low_res_image.to(device),
        tokens=tokens.to(device),
        mask=masks.to(device))
    epsilon, _ = th.split(model_output, C, dim=1)
    return th.nn.functional.mse_loss(epsilon, noise.to(device).detach())


def run_ebm_finetune_epoch(
    device,
    is_master:bool,
    uncond : bool,
    model,
    diffusion: SpacedDiffusion,
    options: dict,
    dataloader: th.utils.data.DataLoader,
    optimizer: th.optim.Optimizer,
    prompt,  # prompt for inference, not training
    sample_bs: int,  # batch size for inference
    sample_gs: float = 10.0,  # guidance scale for inference
    sample_respacing: str = '100', # respacing for inference
    outputs_dir: str = "./outputs",
    checkpoints_dir: str = "./finetune_checkpoints",
    log_frequency: int = 100,
    wandb_run=None,
    gradient_accumualation_steps=1,
    epoch: int = 0,
    train_upsample: bool = False,
    upsample_factor=4,
    image_to_upsample='low_res_face.png',
    energy_mode = False,
):
    if train_upsample: train_step = upsample_train_step
    else: train_step = base_train_step

    model.to(device)
    model.train()
    log = {}
    for train_idx, batch in enumerate(dataloader):

        accumulated_loss = train_step(
            energy_mode= energy_mode,
            model=model,
            diffusion=diffusion,
            batch=batch,
            device=device,
            uncond = uncond,
            learn_sigma=options["learn_sigma"],
        )
        accumulated_loss.backward()
        optimizer.step()
        model.zero_grad()
        
        if is_master:
            log = {"iter": train_idx,"loss": accumulated_loss.item() / gradient_accumualation_steps}
        
        # Sample from the model
      
        if is_master and train_idx > 0 and train_idx % log_frequency == 0:
            
            if is_master:
                print(f"loss: {accumulated_loss.item():.4f}")
                print(f"Sampling from model at iteration {train_idx}")
            
            
            if energy_mode:
                sampler = utils.energy_sample
            else:
                sampler = utils.sample
            
            samples =sampler(
                uncond = uncond,
                model=model,
                options=options,
                batch_size=sample_bs,
                guidance_scale=sample_gs, 
                device=device,
                prediction_respacing=sample_respacing,
            )
            sample_save_path = os.path.join(outputs_dir, f"{train_idx}.png")
            train_util.pred_to_pil(samples).save(sample_save_path)
            if uncond:
                 if is_master:
                    wandb_run.log(
                        {
                            **log,
                            "iter": train_idx,
                            "samples": wandb.Image(sample_save_path, caption="Unconditional"),
                        }
                    )
            else:
                if is_master:
                    wandb_run.log(
                        {
                            **log,
                            "samples": wandb.Image(sample_save_path,caption=prompt["caption"]),
                        }
                 )
            print(f"Saved sample {sample_save_path}")
        if is_master and train_idx % 5000 == 0 and train_idx > 0:
            train_util.save_model(model, checkpoints_dir, train_idx, epoch)
            print(
                f"Saved checkpoint {train_idx} to {checkpoints_dir}/glide-ft-{train_idx}.pt"
            )
        
        if is_master:
            wandb_run.log(log)

    if is_master:
     
        print(f"Finished training, saving final checkpoint")
        train_util.save_model(model, checkpoints_dir, train_idx, epoch)
