import argparse
from glob import glob
import os

import numpy as np
import torch as th
import torchvision.transforms as T
from tqdm import trange
import random
from ebm_finetune.finetune import run_ebm_finetune_epoch
from ebm_finetune.utils import load_model
from ebm_finetune.loader import blender_64
from ebm_finetune.train_util import wandb_setup

import torch 
import utils

def run_ebm_finetune(
    data_dir,
    world_size,
    dist_url,
    learn_sigma: bool,
    uncond = False,
    noise_schedule="squaredcos_cap_v2",
    batch_size=1,
    learning_rate=1e-5,
    resume_ckpt="",
    checkpoints_dir="./finetune_checkpoints",
    device="cpu",
    project_name="ebm_finetune",
    num_epochs=100,
    log_frequency=100,
    sample_bs=1,
    sample_gs=8.0,
    enable_upsample=False,
    outputs_dir = "./outputs",
    num_classes="",
    energy_mode=False, 
    buffer=False,
   

):

    is_master = (utils.get_rank()==0)
    # Setup distributed training
    device = torch.device("cuda")


    # Start wandb logging
    if is_master:
        wandb_run = wandb_setup(
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=device,
            base_dir=checkpoints_dir,
            project_name=project_name,
            learn_sigma = learn_sigma
        )
        print("Wandb setup.")
    else:
        wandb_run = None

   
    # Model setup
    model, diffusion, options = load_model(
        is_master=is_master,
        energy_mode=energy_mode,
        noise_schedule=noise_schedule,
        learn_sigma=learn_sigma,
        num_classes=num_classes,
        model_type="base" if not enable_upsample else "upsample",
    )

    model.to(device)
    ddp_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    n_parameters = sum(p.numel() for p in ddp_model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    

    if is_master == 0:
        number_of_params = sum(x.numel() for x in model.parameters())
        print(f"Number of parameters: {number_of_params}")
        number_of_trainable_params = sum(
            x.numel() for x in model.parameters() if x.requires_grad
        )
        print(f"Trainable parameters: {number_of_trainable_params}")
        
        # Watch the model for 0 rank 
        # wandb_run.watch(model, log="all")

    optimizer = th.optim.AdamW(ddp_model.parameters(), lr=learning_rate, weight_decay = 0.0)

    # Data setup

    if is_master == 0:
        print("buffer status: ", buffer)


    dataset = blender_64(data_dir)
    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()

    sampler_train = torch.utils.data.DistributedSampler(
            dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )


    dataloader = torch.utils.data.DataLoader(
        dataset, sampler=sampler_train,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )

    

    test_prompt_lab= dataset.get_test_sample()
    
    for epoch in trange(num_epochs):
        if is_master:
            print(f"Starting epoch {epoch}")

        dataloader.sampler.set_epoch(epoch)
        run_ebm_finetune_epoch(
            is_master = is_master,
            uncond = uncond,
            device = device,
            model=ddp_model,
            diffusion=diffusion,
            options=options,
            optimizer=optimizer,
            dataloader=dataloader,
            prompt=test_prompt_lab,
            sample_bs=sample_bs,
            sample_gs=sample_gs,
            checkpoints_dir=checkpoints_dir,
            outputs_dir=outputs_dir,
            wandb_run=wandb_run,
            log_frequency=log_frequency,
            epoch=epoch,
            gradient_accumualation_steps=1,
            train_upsample=enable_upsample,
            energy_mode= energy_mode
        )



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", "-data", type=str, default="./data")
    parser.add_argument("--data_name", type=str, default=None)
    parser.add_argument("--batch_size", "-bs", type=int, default=1)
    parser.add_argument("--learning_rate", "-lr", type=float, default=2e-5)
    parser.add_argument("--adam_weight_decay", "-adam_wd", type=float, default=0.0)
    parser.add_argument(
        "--uncond_p",
        "-p",
        type=float,
        default=0.2,
        help="Probability of using the empty/unconditional token instead of a caption. OpenAI used 0.2 for their finetune.",
    )
    parser.add_argument(
        "--train_upsample",
        "-upsample",
        action="store_true",
        help="Train the upsampling type of the model instead of the base model.",
    )
    parser.add_argument(
        "--resume_ckpt",
        "-resume",
        type=str,
        default="",
        help="Checkpoint to resume from",
    )
    parser.add_argument(
        "--checkpoints_dir", "-ckpt", type=str, default="./glide_checkpoints/"
    )
    parser.add_argument(
        "--outputs_dir",  type=str, default="./glide_outs/"
    )
    parser.add_argument(
        "--num_classes",  type=str, default=""
    )
    parser.add_argument("--use_fp16", "-fp16", action="store_true")
    parser.add_argument("--device", "-dev", type=str, default="")
    parser.add_argument("--log_frequency", "-freq", type=int, default=100)
    parser.add_argument("--project_name", "-name", type=str, default="glide-finetune")
    parser.add_argument("--use_captions", "-txt", action="store_true")
    parser.add_argument("--learn_sigma", action="store_true")
    parser.add_argument("--uncond", action="store_true")
    parser.add_argument("--buffer", action="store_true")
    parser.add_argument("--epochs", "-epochs", type=int, default=60)
    parser.add_argument(
        "--test_prompt",
        "-prompt",
        type=str,
        default="a group of skiers are preparing to ski down a mountain.",
    )
    parser.add_argument(
        "--test_batch_size",
        "-tbs",
        type=int,
        default=1,
        help="Batch size used for model eval, not training.",
    )
    parser.add_argument(
        "--test_guidance_scale",
        "-tgs",
        type=float,
        default=4.0,
        help="Guidance scale used during model eval, not training.",
    )

    parser.add_argument(
        "--energy_mode",
        action="store_true",
        help="Energy_mode",
    )
    
    parser.add_argument("--seed", "-seed", type=int, default=0)
    parser.add_argument(
        "--cudnn_benchmark",
        "-cudnn",
        action="store_true",
        help="Enable cudnn benchmarking. May improve performance. (may not)",
    )
    parser.add_argument(
        "--upscale_factor", "-upscale", type=int, default=4, help="Upscale factor for training the upsampling model only"
    )
    parser.add_argument(
        "--buffer_size",type=int, default=1000, help="Buffer(Replay) Size"
    )
    parser.add_argument(
        "--noise_schedule",  type=str, default="squaredcos_cap_v2",choices=["squaredcos_cap_v2","linear"]
    )
    parser.add_argument("--image_to_upsample", "-lowres", type=str, default="low_res_face.png")
   
    parser.add_argument(
        "--world_size",type=int, default=3, help="number of states"
    )
    parser.add_argument(
        "--dist_url",  type=str, default="env://"
    )
    args = parser.parse_args()

    return args

def setup_seed(seed):
    # print("setup random seed = {}".format(seed))
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # th.backends.cudnn.deterministic = True


if __name__ == "__main__":
    # Distributed setup
    args = parse_args()

    utils.init_distributed_mode(args)

    seeds= args.seed + utils.get_rank() + 10
    print("Setting the Seed to ", seeds)
    setup_seed(seed=seeds)
    # # th.manual_seed(args.seed)
    # # np.random.seed(args.seed)
    # th.backends.cudnn.benchmark = args.cudnn_benchmark

    for arg in vars(args):
        print(f"--{arg} {getattr(args, arg)}")

    isExist = os.path.exists(args.outputs_dir)
    
    if not isExist:
        os.makedirs(args.outputs_dir)

    isExist_ckpt = os.path.exists(args.checkpoints_dir)
    
    if not isExist_ckpt:
        os.makedirs(args.checkpoints_dir)
   
    data_dir = args.data_dir
        
    run_ebm_finetune(
        data_dir=data_dir,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        resume_ckpt=args.resume_ckpt,
        checkpoints_dir=args.checkpoints_dir,
        log_frequency=args.log_frequency,
        project_name=args.project_name,
        num_epochs=args.epochs,
        sample_bs=args.test_batch_size,
        sample_gs=args.test_guidance_scale,
        enable_upsample=args.train_upsample,
        outputs_dir =args.outputs_dir,
        num_classes = args.num_classes,
        buffer = args.buffer,
        learn_sigma = args.learn_sigma,
        noise_schedule = args.noise_schedule,
        uncond = args.uncond,
        energy_mode = args.energy_mode,
        world_size = args.world_size,
        dist_url = args.dist_url
    )
