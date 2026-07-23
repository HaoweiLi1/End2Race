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

# Shared controls unless a group overrides them: privilege_gru, batch size
# 12800, actor/critic epochs 2/5, steering/speed std 0.03/0.15, clip 0.15.

# BC baseline: exact Austin600 evaluation with traces for all later comparisons.
# PYTHON=$PYTHON MODEL_PATH=pretrained/end2race.pth COLLISION_SCOPE=ego RENDER=false SAVE_TRACE=true bash evaluate.sh

# Groups 1-3 were completed on 0721. Keep them commented as the exact experiment
# record; do not rerun them into the existing non-empty output directories.
#
# Group 1: critic comparison at batch 12800 and clip 0.15.
# $PYTHON train_ppo.py --critic independent_gru --num_updates 20 --actor_epochs 2 --critic_epochs 5 --batch_size 12800 --steering_latent_std 0.03 --speed_physical_std 0.15 --clip_range 0.15 --output_dir post-trained/ppo_independent_gru_0721_base
# evaluate_run ppo_independent_gru_0721_base 0001 0005 0010 0015 0020
# $PYTHON train_ppo.py --critic priviledge_mlp --num_updates 20 --actor_epochs 2 --critic_epochs 5 --batch_size 12800 --steering_latent_std 0.03 --speed_physical_std 0.15 --clip_range 0.15 --output_dir post-trained/ppo_privilege_mlp_0721_base
# evaluate_run ppo_privilege_mlp_0721_base 0001 0005 0010 0015 0020
# $PYTHON train_ppo.py --critic privilege_gru --num_updates 20 --actor_epochs 2 --critic_epochs 5 --batch_size 12800 --steering_latent_std 0.03 --speed_physical_std 0.15 --clip_range 0.15 --output_dir post-trained/ppo_privilege_gru_0721_base
# evaluate_run ppo_privilege_gru_0721_base 0001 0005 0010 0015 0020
#
# Group 2: privilege_gru batch-size comparison; 12800 baseline is Group 1.
# $PYTHON train_ppo.py --critic privilege_gru --num_updates 20 --actor_epochs 2 --critic_epochs 5 --batch_size 25600 --steering_latent_std 0.03 --speed_physical_std 0.15 --clip_range 0.15 --output_dir post-trained/ppo_privilege_gru_0721_bs25600
# evaluate_run ppo_privilege_gru_0721_bs25600 0001 0005 0010 0015 0020
# $PYTHON train_ppo.py --critic privilege_gru --num_updates 20 --actor_epochs 2 --critic_epochs 5 --batch_size 51200 --steering_latent_std 0.03 --speed_physical_std 0.15 --clip_range 0.15 --output_dir post-trained/ppo_privilege_gru_0721_bs51200
# evaluate_run ppo_privilege_gru_0721_bs51200 0001 0005 0010 0015 0020
#
# Group 3: privilege_gru clip-range comparison; clip 0.15 baseline is Group 1.
# $PYTHON train_ppo.py --critic privilege_gru --num_updates 20 --actor_epochs 2 --critic_epochs 5 --batch_size 12800 --steering_latent_std 0.03 --speed_physical_std 0.15 --clip_range 0.10 --output_dir post-trained/ppo_privilege_gru_0721_clip010
# evaluate_run ppo_privilege_gru_0721_clip010 0001 0005 0010 0015 0020
# $PYTHON train_ppo.py --critic privilege_gru --num_updates 20 --actor_epochs 2 --critic_epochs 5 --batch_size 12800 --steering_latent_std 0.03 --speed_physical_std 0.15 --clip_range 0.20 --output_dir post-trained/ppo_privilege_gru_0721_clip020
# evaluate_run ppo_privilege_gru_0721_clip020 0001 0005 0010 0015 0020

# Group 4: target-KL off; compare actor LR levels 1/3/5. The completed
# ppo_privilege_gru_0721_base run is the middle level (GRU/head 3e-6/3e-5);
# run the 1e-6/1e-5 and 5e-6/5e-5 levels here. Critic LR stays fixed at 3e-4.
# Evaluate U1/U5/U10/U15/U20(final).
# $PYTHON train_ppo.py --critic privilege_gru --num_updates 20 --actor_epochs 2 --critic_epochs 5 --batch_size 12800 --gru_learning_rate 1e-6 --head_learning_rate 1e-5 --critic_learning_rate 3e-4 --steering_latent_std 0.03 --speed_physical_std 0.15 --clip_range 0.15 --output_dir post-trained/ppo_privilege_gru_0722_lr1_tkloff
# evaluate_run ppo_privilege_gru_0722_lr1_tkloff 0001 0005 0010 0015 0020

# $PYTHON train_ppo.py --critic privilege_gru --num_updates 20 --actor_epochs 2 --critic_epochs 5 --batch_size 12800 --gru_learning_rate 5e-6 --head_learning_rate 5e-5 --critic_learning_rate 3e-4 --steering_latent_std 0.03 --speed_physical_std 0.15 --clip_range 0.15 --output_dir post-trained/ppo_privilege_gru_0722_lr5_tkloff
# evaluate_run ppo_privilege_gru_0722_lr5_tkloff 0001 0005 0010 0015 0020

# # Group 5: target-KL off; 30-update privilege_gru long runs comparing only
# # clip range 0.15 vs 0.20. Evaluate every five updates plus U1 and U30(final).
# $PYTHON train_ppo.py --critic privilege_gru --num_updates 30 --actor_epochs 2 --critic_epochs 5 --batch_size 12800 --gru_learning_rate 3e-6 --head_learning_rate 3e-5 --critic_learning_rate 3e-4 --steering_latent_std 0.03 --speed_physical_std 0.15 --clip_range 0.15 --output_dir post-trained/ppo_privilege_gru_0722_long_clip015
# evaluate_run ppo_privilege_gru_0722_long_clip015 0001 0005 0010 0015 0020 0025 0030

# $PYTHON train_ppo.py --critic privilege_gru --num_updates 30 --actor_epochs 2 --critic_epochs 5 --batch_size 12800 --gru_learning_rate 3e-6 --head_learning_rate 3e-5 --critic_learning_rate 3e-4 --steering_latent_std 0.03 --speed_physical_std 0.15 --clip_range 0.20 --output_dir post-trained/ppo_privilege_gru_0722_long_clip020
# evaluate_run ppo_privilege_gru_0722_long_clip020 0001 0005 0010 0015 0020 0025 0030

# Group 6 was completed on 0722. Keep it commented as the exact target-KL
# experiment record; target-KL is not used by the new runs below.
# $PYTHON train_ppo.py --critic privilege_gru --num_updates 20 --actor_epochs 2 --critic_epochs 5 --batch_size 12800 --gru_learning_rate 3e-6 --head_learning_rate 3e-5 --critic_learning_rate 3e-4 --steering_latent_std 0.03 --speed_physical_std 0.15 --clip_range 0.15 --target_kl 0.02 --output_dir post-trained/ppo_privilege_gru_0722_clip015_tkl002
# evaluate_run ppo_privilege_gru_0722_clip015_tkl002 0001 0005 0010 0015 0020

# $PYTHON train_ppo.py --critic privilege_gru --num_updates 20 --actor_epochs 2 --critic_epochs 5 --batch_size 12800 --gru_learning_rate 3e-6 --head_learning_rate 3e-5 --critic_learning_rate 3e-4 --steering_latent_std 0.03 --speed_physical_std 0.15 --clip_range 0.15 --target_kl 0.04 --output_dir post-trained/ppo_privilege_gru_0722_clip015_tkl004
# evaluate_run ppo_privilege_gru_0722_clip015_tkl004 0001 0005 0010 0015 0020

# Group 7: extend the current clip-0.20 winner to 45 updates. With the same
# seed, 12 env workers, source, and cache, U1-U30 must reproduce the completed
# long_clip020 run before its existing evaluations are reused. Evaluate only
# the new convergence checkpoints here.
# $PYTHON train_ppo.py --critic privilege_gru --num_updates 45 --actor_epochs 2 --critic_epochs 5 --batch_size 12800 --gru_learning_rate 3e-6 --head_learning_rate 3e-5 --critic_learning_rate 3e-4 --steering_latent_std 0.03 --speed_physical_std 0.15 --clip_range 0.20 --output_dir post-trained/ppo_privilege_gru_0722_long45_clip020
# evaluate_run ppo_privilege_gru_0722_long45_clip020 0035 0040 0045

# Group 8: clip-0.25 boundary check under the same 45-update controls. This is
# a new trajectory, so retain U1 and every-five-update evaluations.
# $PYTHON train_ppo.py --critic privilege_gru --num_updates 45 --actor_epochs 2 --critic_epochs 5 --batch_size 12800 --gru_learning_rate 3e-6 --head_learning_rate 3e-5 --critic_learning_rate 3e-4 --steering_latent_std 0.03 --speed_physical_std 0.15 --clip_range 0.25 --output_dir post-trained/ppo_privilege_gru_0722_long45_clip025
# evaluate_run ppo_privilege_gru_0722_long45_clip025 0001 0005 0010 0015 0020 0025 0030 0035 0040 0045

# Group 9 was the completed uniform merged-pool hard-neighbor run. Its realized
# hard share was 326 / (479 + 326) = 40.5% of collision episode resets.
# $PYTHON train_ppo.py --critic privilege_gru --num_updates 45 --actor_epochs 2 --critic_epochs 5 --batch_size 12800 --gru_learning_rate 3e-6 --head_learning_rate 3e-5 --critic_learning_rate 3e-4 --steering_latent_std 0.03 --speed_physical_std 0.15 --clip_range 0.20 --hard_neighbors --hard_neighbor_cache_dir post-trained/collision-cache/boundary-aware-v1 --output_dir post-trained/ppo_privilege_gru_0722_long45_clip020_hard
# evaluate_run ppo_privilege_gru_0722_long45_clip020_hard 0001 0005 0010 0015 0020 0025 0030 0035 0040 0045

# Groups 10-11: keep all 326 boundary-aware cases available, but draw them from
# an independent queue at an exact fraction of collision episode resets. The
# ordinary/collision role schedule and every PPO hyperparameter remain fixed.
#
# Group 10: 20% hard-neighbor within collision resets.
$PYTHON train_ppo.py --critic privilege_gru --num_updates 45 --actor_epochs 2 --critic_epochs 5 --batch_size 12800 --gru_learning_rate 3e-6 --head_learning_rate 3e-5 --critic_learning_rate 3e-4 --steering_latent_std 0.03 --speed_physical_std 0.15 --clip_range 0.20 --hard_neighbors --hard_neighbor_fraction 0.20 --hard_neighbor_cache_dir post-trained/collision-cache/boundary-aware-v1 --output_dir post-trained/ppo_privilege_gru_0723_long45_clip020_hard020
evaluate_run ppo_privilege_gru_0723_long45_clip020_hard020 0001 0005 0010 0015 0020 0025 0030 0035 0040 0045

# Group 11: 10% hard-neighbor within collision resets.
# $PYTHON train_ppo.py --critic privilege_gru --num_updates 45 --actor_epochs 2 --critic_epochs 5 --batch_size 12800 --gru_learning_rate 3e-6 --head_learning_rate 3e-5 --critic_learning_rate 3e-4 --steering_latent_std 0.03 --speed_physical_std 0.15 --clip_range 0.20 --hard_neighbors --hard_neighbor_fraction 0.10 --hard_neighbor_cache_dir post-trained/collision-cache/boundary-aware-v1 --output_dir post-trained/ppo_privilege_gru_0723_long45_clip020_hard010
# evaluate_run ppo_privilege_gru_0723_long45_clip020_hard010 0001 0005 0010 0015 0020 0025 0030 0035 0040 0045
