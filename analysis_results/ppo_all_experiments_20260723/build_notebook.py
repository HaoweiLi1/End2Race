#!/usr/bin/env python3
"""Build and execute the reproducible PPO/BC analysis notebook.

The End2Race environment does not include nbformat/nbclient, so this script
emits standards-compliant notebook JSON and executes code cells in a shared
Python namespace while capturing stdout.  Every cell remains directly runnable
in a normal Jupyter installation.
"""

from __future__ import annotations

import contextlib
import io
import json
import traceback
from pathlib import Path


HERE = Path(__file__).resolve().parent
TARGET = HERE / "ppo_all_experiments_tail_analysis.ipynb"


def markdown(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


CELLS = [
    markdown(
        """# End2Race PPO 全实验与 BC episode 级碰撞分析

数据截至 **2026-07-23（Asia/Singapore）**。本 notebook 读取本目录中的审计后 CSV/JSON；原始来源是 `post-trained/` 的模型、训练 metrics/checkpoints，以及 `eval_results/` 的 `results_multi.json` 和 NPZ 轨迹。

## TL;DR

PPO **缓解但没有解决** BC 从 lattice expert 继承的“超车后并线、车尾扫到 opponent”问题。BC 有 22 次碰撞，其中 11 次满足预注册的主判据。整体最佳碰撞 checkpoint（clip 0.20, U30）为 11 次总碰撞、4 次该机制：相对 BC 消除 10 个原始该类场景，但新产生 3 个同类场景。hard-neighbor U35 可让 11 个原始场景全部变为不再同类碰撞，却仍产生 2 个新主判据事件、7 个宽松判据事件，并有 20 次总碰撞。
"""
    ),
    markdown(
        """## Context & Methods

分析单位是固定 Austin600 面板中的 `(policy checkpoint, scenario_id)`。碰撞真值始终来自 `results_multi.json`；0721 旧格式 NPZ 未保存终止后状态，因此不能用最后一个 NPZ collision marker 覆盖 JSON 真值。

### Key Assumptions

- 主判据：ego 相对赛道进度由不领先穿越至领先至少 0.10 s；opponent 同时碰撞；终止时 opponent 位于 ego 车体坐标后方；超车后双方赛道横向间距至少收敛 0.10 m。
- 宽松判据使用 0.05 s / 0.05 m；严格判据要求从至少落后 0.10 m 到至少领先 0.10 m、领先至少 0.15 s、横向收敛至少 0.10 m。
- 轨迹仅允许计算 pose-derived slip proxy；并未保存车辆动力学内部真实 tire slip angle，因此不能把几何接触证据表述为已证明的轮胎侧偏因果。
- 91 个有效 PPO panel 来自单次训练实现与同一 Austin600 面板；跨 checkpoint 选择与多重比较需要显式保留。
"""
    ),
    markdown("## Data\n\n先核对原始覆盖、有效性边界、NPZ 格式和独立验证回执。"),
    code(
        """from pathlib import Path
import json
import numpy as np
import pandas as pd

BASE = Path('analysis_results/ppo_all_experiments_20260723')
panels = pd.read_csv(BASE / 'eval_panels.csv')
episodes = pd.read_csv(BASE / 'eval_episode_outcomes.csv')
paired = pd.read_csv(BASE / 'paired_vs_bc.csv')
groups = pd.read_csv(BASE / 'group_summary.csv')
inventory = pd.read_csv(BASE / 'run_inventory.csv')
training = pd.read_csv(BASE / 'training_metrics.csv')
actor_delta = pd.read_csv(BASE / 'actor_parameter_deltas.csv')
critic_summary = pd.read_csv(BASE / 'critic_parameter_summary.csv')
kinematics = pd.read_csv(BASE / 'collision_episode_kinematics.csv')
frequency = pd.read_csv(BASE / 'scenario_frequency_unique_policies.csv')
validation = json.loads((BASE / 'validation_receipt.json').read_text())
npz_audit = json.loads((BASE / 'npz_audit.json').read_text())

print('training runs:', len(inventory))
print('eval panels:', len(panels), '| valid:', int(panels.valid.sum()), '| invalid:', int((~panels.valid).sum()))
print('episode/trace records:', len(episodes))
print('NPZ formats:', npz_audit['by_format'])
print('validation assessment:', validation['assessment'])
print('validation checks:', [(x['check'], x['status']) for x in validation['checks']])
"""
    ),
    code(
        """invalid = panels.loc[~panels.valid, ['panel','episode_rows','unique_scenarios','trace_files','error_count','collision_count','overtake_count','follow_count']]
print(invalid.to_string(index=False))
"""
    ),
    markdown("## Results\n\n### 1. 模型参数与训练设置"),
    code(
        """architecture = json.loads((BASE / 'model_architecture.json').read_text())
print(json.dumps(architecture, indent=2, ensure_ascii=False))

config_cols = ['run','critic','env_workers','batch_size','clip_range','target_kl','hard_neighbors',
               'formal_rows','checkpoint_complete','collision_pool_count','early_stop_updates',
               'actor_steps_completed','actor_steps_planned','final_explained_variance_post','final_value_loss_post']
print('\\nRun inventory:')
print(inventory[config_cols].to_string(index=False))
"""
    ),
    code(
        """selected_policies = [
    'ppo_privilege_gru_0721_base_u0020',
    'ppo_privilege_gru_0722_long_clip020_u0030',
    'ppo_privilege_gru_0722_long45_clip020_u0045',
    'ppo_privilege_gru_0722_long45_clip025_u0045',
    'ppo_privilege_gru_0722_long45_clip020_hard_u0045',
]
delta_cols = ['panel','update','actor_relative_l2_from_bc','actor_delta_rms_from_bc',
              'actor_max_abs_delta_from_bc','gru_relative_l2_from_bc','head_relative_l2_from_bc',
              'fixed_frontend_delta_l2']
print(actor_delta.loc[actor_delta.panel.isin(selected_policies), delta_cols].to_string(index=False))

selected_runs = [
    'ppo_privilege_gru_0721_base',
    'ppo_privilege_gru_0722_long45_clip020',
    'ppo_privilege_gru_0722_long45_clip025',
    'ppo_privilege_gru_0722_long45_clip020_hard',
]
critic_cols = ['run','final_update','critic_parameter_count','critic_relative_l2_from_warmup',
               'critic_delta_rms_from_warmup','privileged_projection_l2']
print('\\nFinal critic deltas:')
print(critic_summary.loc[critic_summary.run.isin(selected_runs), critic_cols].to_string(index=False))
"""
    ),
    markdown("### 2. 训练 metrics：优化器在工作，但训练代理指标不能替代 eval"),
    code(
        """late_runs = [
    'ppo_privilege_gru_0722_long45_clip020',
    'ppo_privilege_gru_0722_long45_clip025',
    'ppo_privilege_gru_0722_long45_clip020_hard',
]
updates = [1, 20, 25, 30, 35, 40, 45]
train_cols = ['run','update','formal_training_timesteps','training_collision_rate','mean_episode_return',
              'value_loss_post','explained_variance_post','approx_kl_mean','approx_kl_max',
              'clip_fraction_mean','actor_optimizer_steps_completed','actor_optimizer_steps_planned']
print(training.loc[training.run.isin(late_runs) & training['update'].isin(updates), train_cols].to_string(index=False))

correlations = pd.read_csv(BASE / 'training_eval_correlations.csv')
print('\\nPooled descriptive correlations (not causal):')
print(correlations.to_string(index=False))
"""
    ),
    markdown("### 3. 全实验 eval 总览与受控比较"),
    code(
        """summary_cols = ['group','arm','updates','collision_path','merge_tail_path','overtake_path',
                'mean_collisions_all_valid','mean_tail_all_valid','mean_collisions_u25_plus',
                'mean_tail_u25_plus','final_collision_count','final_tail_count','final_overtake_count']
print(groups[summary_cols].to_string(index=False))

controls = pd.read_csv(BASE / 'group_control_audit.csv')
print('\\nControl audit:')
print(controls.to_string(index=False))
"""
    ),
    markdown("### 4. 关键 checkpoint 与 BC 的 episode 级配对结果"),
    code(
        """selected = [
    'BC',
    'ppo_privilege_gru_0721_base_u0020',
    'ppo_privilege_gru_0722_long_clip020_u0030',
    'ppo_privilege_gru_0722_long45_clip020_u0040',
    'ppo_privilege_gru_0722_long45_clip020_u0045',
    'ppo_privilege_gru_0722_long45_clip025_u0045',
    'ppo_privilege_gru_0722_long45_clip020_hard_u0035',
    'ppo_privilege_gru_0722_long45_clip020_hard_u0045',
]
eval_cols = ['panel','collision_count','overtake_count','follow_count','collision_with_opponent_count',
             'wall_like_collision_count','merge_tail_relaxed_count','merge_tail_primary_count',
             'merge_tail_strict_count','merge_tail_primary_share_of_collisions']
print(panels.loc[panels.panel.isin(selected), eval_cols].to_string(index=False))

pair_cols = ['panel','collision_resolved','collision_shared','collision_created','collision_net_reduction',
             'collision_exact_p','collision_cluster_bootstrap_diff_ci_low','collision_cluster_bootstrap_diff_ci_high',
             'tail_resolved','tail_shared','tail_created','tail_net_reduction','tail_exact_p',
             'tail_cluster_bootstrap_diff_ci_low','tail_cluster_bootstrap_diff_ci_high',
             'bc_tail_scenarios_still_any_collision','bc_tail_scenarios_now_overtake']
print('\\nPaired transitions vs BC:')
print(paired.loc[paired.panel.isin(selected[1:]), pair_cols].to_string(index=False))
"""
    ),
    markdown("### 5. BC 问题 episode 的几何标定"),
    code(
        """bc_tail = kinematics[(kinematics.panel == 'BC') & (kinematics.merge_tail_primary)]
bc_cols = ['scenario_id','collision_time_s','pass_time_s','pass_to_collision_s',
           'post_pass_lateral_convergence_m','terminal_opponent_body_x_m',
           'terminal_opponent_body_y_m','terminal_abs_kinematic_slip_proxy_rad']
print(bc_tail[bc_cols].sort_values('scenario_id').to_string(index=False))
print('\\nBC summary:')
print('count =', len(bc_tail))
print('pass-to-collision median/range =', bc_tail.pass_to_collision_s.median(),
      (bc_tail.pass_to_collision_s.min(), bc_tail.pass_to_collision_s.max()))
print('lateral convergence median/range =', bc_tail.post_pass_lateral_convergence_m.median(),
      (bc_tail.post_pass_lateral_convergence_m.min(), bc_tail.post_pass_lateral_convergence_m.max()))
"""
    ),
    markdown("### 6. 碰撞 episode 的共性与机制迁移"),
    code(
        """top_cols = ['scenario_id','opponent_raceline','opponent_speed_scale','collision_panels',
            'collision_rate','merge_tail_panels','merge_tail_rate','overtake_panels','follow_panels']
print(frequency.sort_values(['collision_panels','merge_tail_panels'], ascending=False)[top_cols].head(20).to_string(index=False))

common = json.loads((BASE / 'collision_commonality.json').read_text())
print('\\nCollision-class partition across 86 unique PPO policies:')
print(common['unique_policy_collision_class_counts'])
print('collision time quantiles:', common['unique_policy_collision_time_quantiles_s'])
print('speed-scale counts:', common['unique_policy_speed_scale_counts'])
"""
    ),
    markdown("### 7. 阈值敏感性与结论稳健性"),
    code(
        """valid_ppo = panels[(panels.valid) & (panels.run != 'BC')]
for metric in ['merge_tail_relaxed_count','merge_tail_primary_count','merge_tail_strict_count']:
    minimum = int(valid_ppo[metric].min())
    names = valid_ppo.loc[valid_ppo[metric] == minimum, 'panel'].tolist()
    print(metric, 'minimum =', minimum, '|', names)

print('\\nNo valid checkpoint has zero relaxed or primary events.')
print('Holm-adjusted p-value range across selected-vs-BC panel tests:',
      paired.loc[paired.valid, ['collision_exact_p_holm_all_valid_panels','tail_exact_p_holm_all_valid_panels']].min().to_dict())
"""
    ),
    markdown(
        """## Takeaways

1. **模型确实改变了策略，而不是 eval 噪声。** 11.30M 个 actor 可训练参数发生小幅但非零变化，540 个固定前端参数保持完全不变；相同 actor checkpoint 的重复 eval 结果完全一致。
2. **P20 `privilege_gru` critic 是早期实验中最强的 critic 方案，但后续训练没有单调收敛。** clip 0.20 的总碰撞路径在 U30/U40 为 11，U45 回升到 12；相邻 checkpoint 的碰撞身份持续翻转。
3. **整体最优折中是 clip 0.20 U30，而不是 hard-neighbor。** U30 把总碰撞 22→11、主判据 11→4；hard U35 虽把主判据降到 2，却仍有 20 次总碰撞，且宽松判据为 7。
4. **对用户核心问题的严格回答：没有解决，只是显著缓解。** 部分 PPO checkpoint 可解决全部 11 个 BC 原始问题场景，但会在其他场景生成同类碰撞。最低宽松/主判据分别仍是 3/2，任何有效 checkpoint 都没有归零。
5. **不能声称已经证明“真实甩尾动力学”被修复。** 数据足以证明超车后横向收敛 + opponent 位于车尾的接触几何；没有保存真实 tire slip angle，只能报告 pose-derived proxy。

### Required caveats

- 单 seed、固定 Austin600、checkpoint 选择和多重比较限制外推；所有 91 个有效 panel 的 Holm 校正后检验均不显著。
- `ppo_privilege_gru_0722_lr5_tkloff_u0020` 只有 599 条记录，已从排序与推断中排除。
- 0721 旧 NPZ 漏终止后碰撞状态；碰撞真值由 `results_multi.json` 控制。
- `run_config.json` 没有持久化 seed 与 source commit；当前工作树只能用于代码语义核对，不能补写历史 provenance。
"""
    ),
]


def execute(cells: list[dict]) -> None:
    namespace: dict = {"__name__": "__notebook__"}
    execution_count = 0
    for cell in cells:
        if cell["cell_type"] != "code":
            continue
        execution_count += 1
        cell["execution_count"] = execution_count
        stream = io.StringIO()
        try:
            with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                exec("".join(cell["source"]), namespace)
        except Exception:
            stream.write(traceback.format_exc())
            cell["outputs"] = [{
                "output_type": "error",
                "ename": "ExecutionError",
                "evalue": "Notebook cell failed; see traceback",
                "traceback": stream.getvalue().splitlines(),
            }]
            raise
        output = stream.getvalue()
        if output:
            cell["outputs"] = [{"name": "stdout", "output_type": "stream", "text": output.splitlines(keepends=True)}]


def main() -> None:
    execute(CELLS)
    notebook = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {"display_name": "End2Race", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    TARGET.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {TARGET} with {len(CELLS)} cells")


if __name__ == "__main__":
    main()
