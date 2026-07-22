#!/bin/bash
set -eo pipefail

PYTHON=/home/haowei/miniconda3/envs/end2race/bin/python

evaluate_run() {
    run_name=$1
    shift
    run_dir="post-trained/$run_name"

    for update in "$@"; do
        model_name="${run_name}_u${update}"
        model_path="$run_dir/${model_name}.pth"
        if [[ ! -e "$model_path" && ! -L "$model_path" ]]; then
            ln -s "checkpoints/actor_u${update}.pth" "$model_path"
        fi
        PYTHON=$PYTHON MODEL_PATH="$model_path" COLLISION_SCOPE=ego RENDER=false SAVE_TRACE=true bash evaluate.sh
    done
}

# Common controls: 20 updates, actor epochs 2, critic epochs 5,
# steering std 0.03, speed std 0.15. Evaluate U1/U5/U10/U15/U20(final).

# End2Race Baseline eval


# Group 1: critic comparison. Baseline batch size 12800 and clip range 0.10.
$PYTHON train_ppo.py --critic independent_gru --num_updates 20 --actor_epochs 2 --critic_epochs 5 --batch_size 12800 --steering_latent_std 0.03 --speed_physical_std 0.15 --clip_range 0.15 --output_dir post-trained/ppo_independent_gru_0721_base
evaluate_run ppo_independent_gru_0721_base 0001 0005 0010 0015 0020

$PYTHON train_ppo.py --critic priviledge_mlp --num_updates 20 --actor_epochs 2 --critic_epochs 5 --batch_size 12800 --steering_latent_std 0.03 --speed_physical_std 0.15 --clip_range 0.15 --output_dir post-trained/ppo_privilege_mlp_0721_base
evaluate_run ppo_privilege_mlp_0721_base 0001 0005 0010 0015 0020

$PYTHON train_ppo.py --critic privilege_gru --num_updates 20 --actor_epochs 2 --critic_epochs 5 --batch_size 12800 --steering_latent_std 0.03 --speed_physical_std 0.15 --clip_range 0.15 --output_dir post-trained/ppo_privilege_gru_0721_base
evaluate_run ppo_privilege_gru_0721_base 0001 0005 0010 0015 0020

# Group 2: privilege_gru batch-size comparison. The 12800 baseline is the
# privilege_gru run above, so only the two additional legal batch sizes rerun.
$PYTHON train_ppo.py --critic privilege_gru --num_updates 20 --actor_epochs 2 --critic_epochs 5 --batch_size 25600 --steering_latent_std 0.03 --speed_physical_std 0.15 --clip_range 0.15 --output_dir post-trained/ppo_privilege_gru_0721_bs25600
evaluate_run ppo_privilege_gru_0721_bs25600 0001 0005 0010 0015 0020

$PYTHON train_ppo.py --critic privilege_gru --num_updates 20 --actor_epochs 2 --critic_epochs 5 --batch_size 51200 --steering_latent_std 0.03 --speed_physical_std 0.15 --clip_range 0.15 --output_dir post-trained/ppo_privilege_gru_0721_bs51200
evaluate_run ppo_privilege_gru_0721_bs51200 0001 0005 0010 0015 0020

# Group 3: privilege_gru clip-range comparison. Baseline batch size is 12800.
$PYTHON train_ppo.py --critic privilege_gru --num_updates 20 --actor_epochs 2 --critic_epochs 5 --batch_size 12800 --steering_latent_std 0.03 --speed_physical_std 0.15 --clip_range 0.10 --output_dir post-trained/ppo_privilege_gru_0721_clip010
evaluate_run ppo_privilege_gru_0721_clip010 0001 0005 0010 0015 0020

$PYTHON train_ppo.py --critic privilege_gru --num_updates 20 --actor_epochs 2 --critic_epochs 5 --batch_size 12800 --steering_latent_std 0.03 --speed_physical_std 0.15 --clip_range 0.20 --output_dir post-trained/ppo_privilege_gru_0721_clip020
evaluate_run ppo_privilege_gru_0721_clip020 0001 0005 0010 0015 0020

# Group 4: Lr comparision

# Group 5：privilege_gru long-run clip-range comparison.

