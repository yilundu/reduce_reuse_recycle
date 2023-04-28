#!/bin/bash

wandb online 

python -m torch.distributed.launch --nproc_per_node=2 --master_port=5672  --use_env train_ddp.py \
  --project_name 'blender-EBM' \
  --data_dir './data/blender_64' \
  --batch_size 16  \
  --energy_mode \
  --noise_schedule 'squaredcos_cap_v2' \
  --seed 42 \
  --test_guidance_scale 8 \
  --epochs 50 \
  --log_frequency 50 \
  --num_classes "3," \
  --learning_rate 1e-04 \
  --checkpoints_dir './checkpoints/Energy_object_chkpt' \
  --outputs_dir './checkpoints/Energy_object_chkpt'
