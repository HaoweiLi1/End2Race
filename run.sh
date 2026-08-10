#!/usr/bin/env bash
set -euo pipefail

END2RACE_PYTHON=/home/haowei/miniconda3/envs/end2race/bin/python

$END2RACE_PYTHON train_ppo.py --pretrained_model_path pretrained/end2race.pth --output_dir post-trained/ppo_global_temporal_speed_noise_hold10steps --seed 42 --num_updates 30 --speed_noise_hold_steps 10
mkdir post-trained/ppo_global_temporal_speed_noise_hold10steps/update30
ln post-trained/ppo_global_temporal_speed_noise_hold10steps/checkpoints/actor_u0030.pth post-trained/ppo_global_temporal_speed_noise_hold10steps/update30/actor.pth
PYTHON=$END2RACE_PYTHON MODEL_PATH=post-trained/ppo_global_temporal_speed_noise_hold10steps/update30/actor.pth MAP_NAME=Austin SAVE_TRACE=true bash evaluate.sh
PYTHON=$END2RACE_PYTHON MODEL_PATH=post-trained/ppo_global_temporal_speed_noise_hold10steps/update30/actor.pth MAP_NAME=Hockenheim SAVE_TRACE=true bash evaluate.sh
PYTHON=$END2RACE_PYTHON MODEL_PATH=post-trained/ppo_global_temporal_speed_noise_hold10steps/update30/actor.pth MAP_NAME=MoscowRaceway SAVE_TRACE=true bash evaluate.sh
PYTHON=$END2RACE_PYTHON MODEL_PATH=post-trained/ppo_global_temporal_speed_noise_hold10steps/update30/actor.pth MAP_NAME=Nuerburgring SAVE_TRACE=true bash evaluate.sh

$END2RACE_PYTHON train_ppo.py --pretrained_model_path pretrained/end2race.pth --output_dir post-trained/ppo_global_hold10_front_corridor_hold50_speed_noise --seed 42 --num_updates 30 --speed_noise_hold_steps 10 --front_corridor_speed_noise_hold_steps 50
mkdir post-trained/ppo_global_hold10_front_corridor_hold50_speed_noise/update30
ln post-trained/ppo_global_hold10_front_corridor_hold50_speed_noise/checkpoints/actor_u0030.pth post-trained/ppo_global_hold10_front_corridor_hold50_speed_noise/update30/actor.pth
PYTHON=$END2RACE_PYTHON MODEL_PATH=post-trained/ppo_global_hold10_front_corridor_hold50_speed_noise/update30/actor.pth MAP_NAME=Austin SAVE_TRACE=true bash evaluate.sh
PYTHON=$END2RACE_PYTHON MODEL_PATH=post-trained/ppo_global_hold10_front_corridor_hold50_speed_noise/update30/actor.pth MAP_NAME=Hockenheim SAVE_TRACE=true bash evaluate.sh
PYTHON=$END2RACE_PYTHON MODEL_PATH=post-trained/ppo_global_hold10_front_corridor_hold50_speed_noise/update30/actor.pth MAP_NAME=MoscowRaceway SAVE_TRACE=true bash evaluate.sh
PYTHON=$END2RACE_PYTHON MODEL_PATH=post-trained/ppo_global_hold10_front_corridor_hold50_speed_noise/update30/actor.pth MAP_NAME=Nuerburgring SAVE_TRACE=true bash evaluate.sh

$END2RACE_PYTHON train_ppo.py --pretrained_model_path pretrained/end2race.pth --output_dir post-trained/ppo_loss_sample_stride10 --seed 42 --num_updates 30 --ppo_loss_sample_stride 10
mkdir post-trained/ppo_loss_sample_stride10/update30
ln post-trained/ppo_loss_sample_stride10/checkpoints/actor_u0030.pth post-trained/ppo_loss_sample_stride10/update30/actor.pth
PYTHON=$END2RACE_PYTHON MODEL_PATH=post-trained/ppo_loss_sample_stride10/update30/actor.pth MAP_NAME=Austin SAVE_TRACE=true bash evaluate.sh
PYTHON=$END2RACE_PYTHON MODEL_PATH=post-trained/ppo_loss_sample_stride10/update30/actor.pth MAP_NAME=Hockenheim SAVE_TRACE=true bash evaluate.sh
PYTHON=$END2RACE_PYTHON MODEL_PATH=post-trained/ppo_loss_sample_stride10/update30/actor.pth MAP_NAME=MoscowRaceway SAVE_TRACE=true bash evaluate.sh
PYTHON=$END2RACE_PYTHON MODEL_PATH=post-trained/ppo_loss_sample_stride10/update30/actor.pth MAP_NAME=Nuerburgring SAVE_TRACE=true bash evaluate.sh

$END2RACE_PYTHON train_ppo.py --pretrained_model_path pretrained/end2race.pth --output_dir post-trained/ppo_online_same_state_branched_return --seed 42 --num_updates 45 --online_same_state_branch_ppo
mkdir post-trained/ppo_online_same_state_branched_return/update45
ln post-trained/ppo_online_same_state_branched_return/checkpoints/actor_u0045.pth post-trained/ppo_online_same_state_branched_return/update45/actor.pth
PYTHON=$END2RACE_PYTHON MODEL_PATH=post-trained/ppo_online_same_state_branched_return/update45/actor.pth MAP_NAME=Austin SAVE_TRACE=true bash evaluate.sh
PYTHON=$END2RACE_PYTHON MODEL_PATH=post-trained/ppo_online_same_state_branched_return/update45/actor.pth MAP_NAME=Hockenheim SAVE_TRACE=true bash evaluate.sh
PYTHON=$END2RACE_PYTHON MODEL_PATH=post-trained/ppo_online_same_state_branched_return/update45/actor.pth MAP_NAME=MoscowRaceway SAVE_TRACE=true bash evaluate.sh
PYTHON=$END2RACE_PYTHON MODEL_PATH=post-trained/ppo_online_same_state_branched_return/update45/actor.pth MAP_NAME=Nuerburgring SAVE_TRACE=true bash evaluate.sh

$END2RACE_PYTHON train_ppo.py --pretrained_model_path pretrained/end2race.pth --output_dir post-trained/ppo_collision_prefix_one_second_branched_return --seed 42 --num_updates 30 --collision_prefix_branch_ppo
mkdir post-trained/ppo_collision_prefix_one_second_branched_return/update30
ln post-trained/ppo_collision_prefix_one_second_branched_return/checkpoints/actor_u0030.pth post-trained/ppo_collision_prefix_one_second_branched_return/update30/actor.pth
PYTHON=$END2RACE_PYTHON MODEL_PATH=post-trained/ppo_collision_prefix_one_second_branched_return/update30/actor.pth MAP_NAME=Austin SAVE_TRACE=true bash evaluate.sh
PYTHON=$END2RACE_PYTHON MODEL_PATH=post-trained/ppo_collision_prefix_one_second_branched_return/update30/actor.pth MAP_NAME=Hockenheim SAVE_TRACE=true bash evaluate.sh
PYTHON=$END2RACE_PYTHON MODEL_PATH=post-trained/ppo_collision_prefix_one_second_branched_return/update30/actor.pth MAP_NAME=MoscowRaceway SAVE_TRACE=true bash evaluate.sh
PYTHON=$END2RACE_PYTHON MODEL_PATH=post-trained/ppo_collision_prefix_one_second_branched_return/update30/actor.pth MAP_NAME=Nuerburgring SAVE_TRACE=true bash evaluate.sh
