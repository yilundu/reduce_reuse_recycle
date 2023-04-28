
import torch as th
import matplotlib.pyplot as plt

import numpy as np 
import os 
from composable_diffusion.model_creation import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    create_gaussian_diffusion,
    Sampler_create_gaussian_diffusion,
)

from composable_diffusion.model_creation import create_model_and_diffusion,model_and_diffusion_defaults,create_gaussian_diffusion

from anneal_samplers import AnnealedMALASampler, AnnealedCHASampler, AnnealedUHASampler,AnnealedULASampler
import argparse
def pil_image_to_norm_tensor(pil_image):
    """
    Convert a PIL image to a PyTorch tensor normalized to [-1, 1] with shape [B, C, H, W].
    """
    return th.from_numpy(np.asarray(pil_image)).float().permute(2, 0, 1) / 127.5 - 1.0


def convert_images(batch: th.Tensor):
    scaled = ((batch + 1)*127.5).round().clamp(0,255).to(th.uint8).permute(0, 2, 3, 1)

    return scaled


def get_caption_simple(label):
    colors_to_idx = {"gray": 0, "red": 1, "blue": 2, "green": 3, "brown": 4, "purple": 5, "cyan": 6, "yellow": 7, "none": 8}
    
    shapes_to_idx = {"cube": 0, "sphere": 1,"cylinder":2,"none": 3}#{"cube": 0, "boot": 1, "sphere": 2,"cylinder":3,"none": 4}#{"cube": 0, "boot": 1, "sphere": 2,"truck":3,"cylinder":4,"none": 5}#

    materials_to_idx = {"rubber": 0, "metal": 1, "none": 2}
    sizes_to_idx = {"small": 0, "large": 1, "none": 2}
    relations_to_idx = {"left": 0, "right": 1, "front": 2, "behind": 3, "none": 4}#{"right": 0,"behind":1,"none": 2} #{"left": 0, "right": 1, "front": 2, "behind": 3,"none": 4}
    
    label_description = {
            "left": "to the left of",
            "right": "to the right of",
            "behind": "behind",
            "front": "in front of",
            "above": "above",
            "below": "below"
        }
    list(colors_to_idx.keys())
    shapes = list(shapes_to_idx.keys())
    list(materials_to_idx.keys())
    list(sizes_to_idx.keys())
    list(relations_to_idx.keys())

    return f'A {shapes[label[0]]}'


parser = argparse.ArgumentParser()


parser.add_argument('--ckpt_path', required=True)
parser.add_argument('--sampler', type=str, default="mala",choices=["MALA", "HMC", "UHMC", "ULA","Rev_Diff"])
args = parser.parse_args()



has_cuda = th.cuda.is_available()
device = th.device('cpu' if not has_cuda else 'cuda')






options = model_and_diffusion_defaults()

#64x64
model_path1= args.ckpt_path # 
options["noise_schedule"]= "linear"
options["learn_sigma"] = False
options["use_fp16"] = False
options["num_classes"] = "3,"  # "4,"
options["dataset"] = "clevr_norel"
options["image_size"] =   64#  128 , 3 
options["num_channels"] = 128 #192 
options["num_res_blocks"] = 3 #2
options["energy_mode"] = True

base_timestep_respacing = '100' 


if options['energy_mode']:
    print("Using energy mode")
    diffusion = Sampler_create_gaussian_diffusion(
    steps=100,#1000,
    learn_sigma=options['learn_sigma'],
    noise_schedule=options['noise_schedule'],
    timestep_respacing=base_timestep_respacing,
    )
else:
    diffusion = create_gaussian_diffusion(
    steps=100,
    learn_sigma=options['learn_sigma'],
    noise_schedule=options['noise_schedule'],
    timestep_respacing=base_timestep_respacing,
    )

if len(model_path1) > 0:
    assert os.path.exists(
        model_path1
    ), f"Failed to resume from {model_path1}, file does not exist."
    weights = th.load(model_path1, map_location="cpu")
    model1,_ = create_model_and_diffusion(**options)
    model1.load_state_dict(weights)


model1 = model1.to("cuda")
model1.eval()



guidance_scale = 4
batch_size = 1


labels = th.tensor([[ [0], [2] ]]).long() # Compose Cube And Cylinder Labels
# labels = th.tensor([[ [1], [2] ]]).long() # # Compose Sphere And Cylinder Labels

[print(get_caption_simple(lab.numpy())) for lab in labels[0]]
print(labels)
print(labels.shape)



labels = [x.squeeze(dim=1) for x in th.chunk(labels, labels.shape[1], dim=1)]
full_batch_size = batch_size * (len(labels) + 1)

masks = [True] * len(labels) * batch_size + [False] * batch_size

labels = th.cat((labels + [th.zeros_like(labels[0])]), dim=0)

model_kwargs = dict(
    y=labels.clone().detach().to(device),
    masks=th.tensor(masks, dtype=th.bool, device=device)
)
print(labels)
print(masks)
print(labels.shape)



def model_fn_t(x_t, ts, **kwargs):
    cond_eps = model1(x_t, ts,eval=True, **kwargs)
    kwargs['y'] = th.zeros(kwargs['y'].shape, dtype=th.long,device = "cuda")
    kwargs['masks'] = th.tensor([False] * batch_size, dtype=th.bool, device=device)
    uncond_eps = model1(x_t, ts,eval=True, **kwargs)

    eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)

    return eps


def cfg_model_fn(x_t, ts, **kwargs):
    half = x_t[:1]
    combined = th.cat([half] * kwargs['y'].size(0), dim=0)
    eps = model1(combined, ts,eval=True, **kwargs)
    # eps, rest = model_out[:, :3], model_out[:, 3:]

    cond_eps, uncond_eps = eps[:-1], eps[-1:]
    # assume weights are equal to guidance scale
    # print(cond_eps.shape,uncond_eps.shape)
    half_eps = uncond_eps + guidance_scale*(cond_eps - uncond_eps).sum(dim=0, keepdim=True)
    eps = th.cat([half_eps] * x_t.size(0), dim=0)
    return eps

def cfg_model_fn_noen(x_t, ts, **kwargs):
    half = x_t[:1]
    combined = th.cat([half] * kwargs['y'].size(0), dim=0)
    eps = model1(combined, ts,**kwargs)
    # eps, rest = model_out[:, :3], model_out[:, 3:]
    
    cond_eps, uncond_eps = eps[:-1], eps[-1:]
    # assume weights are equal to guidance scale
    half_eps = uncond_eps + guidance_scale*(cond_eps - uncond_eps).sum(dim=0, keepdim=True)
    eps = th.cat([half_eps] * x_t.size(0), dim=0)
    return eps





alphas = 1 - diffusion.betas
alphas_cumprod = np.cumprod(alphas)
scalar = np.sqrt(1 / (1 - alphas_cumprod))

def gradient(x_t, ts, **kwargs):
    half = x_t[:1]
    combined = th.cat([half] * kwargs['y'].size(0), dim=0)
    eps = model1(combined, ts, eval=True,**kwargs)

    cond_eps, uncond_eps = eps[:-1], eps[-1:]
    # assume weights are equal to guidance scale
    half_eps = uncond_eps + guidance_scale*(cond_eps - uncond_eps).sum(dim=0, keepdim=True)
    eps = th.cat([half_eps] * x_t.size(0), dim=0)  
    # Need to scale the gradients by coefficient to properly account for normalization in DSM loss + data contraction
    # print(ts)
    scale = scalar[ts[0]]
    return -1*scale*eps

# Hypeprparameters For Samplers : Need to be tuned carefully for generating good proposals

num_steps = 100

#ULA 
# increase the number of Langevin MCMC steps run to sample between intermediate distributions
# more steps improves sampling quality
la_steps = 20
la_step_sizes = diffusion.betas * 2


#HMC / UHMC SAMPLER
ha_steps = 10#2 # Hamiltonian steps to run
num_leapfrog_steps = 3 # Steps to run in leapfrog
damping_coeff = 0.7#0.9
mass_diag_sqrt = diffusion.betas
ha_step_sizes = (diffusion.betas) * 0.1 #0.1


# MALA SAMPLER 
la_steps = 20
la_step_sizes = diffusion.betas * 0.035

def gradient_cha(x_t, ts, **kwargs):
    half = x_t[:1]
    combined = th.cat([half] * kwargs['y'].size(0), dim=0)
    energy_norm,eps = model1(combined, ts, mala_sampler=True,**kwargs)

    cond_energy,uncond_energy = energy_norm[:-1], energy_norm[-1:]
    total_energy = uncond_energy.sum() + guidance_scale*(cond_energy.sum() - uncond_energy.sum())
    
    
    cond_eps, uncond_eps = eps[:-1], eps[-1:]
    # assume weights are equal to guidance scale
    half_eps = uncond_eps +guidance_scale* (cond_eps - uncond_eps).sum(dim=0, keepdim=True)
    eps = th.cat([half_eps] * x_t.size(0), dim=0)  

    # Need to scale the gradients by coefficient to properly account for normalization in DSM loss + data contraction
    # print(ts)
    scale = scalar[ts[0]]
    return -scale*total_energy,-1*scale*eps


if args.sampler == 'MALA':
    sampler = AnnealedMALASampler(num_steps, la_steps, la_step_sizes, gradient_cha)
elif args.sampler == 'ULA':
    sampler = AnnealedULASampler(num_steps, la_steps, la_step_sizes, gradient)
elif args.sampler == 'UHMC':
    sampler = AnnealedUHASampler(num_steps,
                ha_steps,
                ha_step_sizes,
                damping_coeff,
                mass_diag_sqrt,
                num_leapfrog_steps,
                gradient,
                )
elif args.sampler == 'HMC':
    sampler = AnnealedCHASampler(num_steps,
                ha_steps,
                ha_step_sizes,
                damping_coeff,
                mass_diag_sqrt,
                num_leapfrog_steps,
                gradient_cha,
                None)

elif args.sampler == 'Rev_Diff':
    print("Using Reverse Diffusion Sampling only")
    sampler = None

print("Using Sampler: ",args.sampler)
all_samp = []


for k in range(4):
    if options['energy_mode']:
        samples = diffusion.p_sample_loop(
            sampler,
            cfg_model_fn,
            (full_batch_size, 3, 64, 64),
            device=device,
            clip_denoised=True,
            progress=False,
            model_kwargs=model_kwargs,
            cond_fn=None,
        )[:batch_size]
    else:
        samples = diffusion.p_sample_loop(
            cfg_model_fn_noen,
            (full_batch_size, 3, 128, 128),
            device=device,
            clip_denoised=True,
            progress=True,
            model_kwargs=model_kwargs,
            cond_fn=None,
        )[:batch_size]
    sample = samples.contiguous()

    sample = convert_images(sample)

    show_img = sample.cpu().detach().numpy()
    all_samp.append(show_img)

    
fig ,ax = plt.subplots(figsize=(32,32))

arr = np.concatenate(all_samp, axis=0)
show_img = th.tensor(arr)
show_img = show_img.permute(0, 3, 1, 2) # N C H W


w = 4
h = 4
fig = plt.figure(figsize=(32,32))
columns = 2
rows = 2
for i in range(1, columns*rows +1):
    img = show_img[i-1].permute(1,2,0).numpy()
    fig.add_subplot(rows, columns, i)
    plt.imshow(img)
    plt.axis("off")
    cap = "A Sphere And A Cylinder"
    # z = cap.split(" ")
    # z.insert(len(z)//2,'\n')
    # cap = " ".join(z)

    plt.title(cap,fontsize=25)

plt.savefig(f"Energy_Object_{cap}_CHA_{guidance_scale}.png")
