#!/usr/bin/env python3
"""Build the executable companion notebook for the all-experiment PPO audit."""

from __future__ import annotations

import base64
import contextlib
import io
import json
import traceback
from pathlib import Path

OUT = Path(__file__).resolve().parent


def md(text: str):
    return {"cell_type": "markdown", "metadata": {}, "source": text}


def code(text: str):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text,
    }


nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "version": "3"},
    },
}

nb["cells"] = [
    md(
        """# End2Race PPO 全实验审计：BC 至 Group 9

## TL;DR

- 固定 Austin600 面板上，最低碰撞为 **11/600**：G5 clip 0.20 U30 与其逐位复现后延长得到的 G7 U40。
- PPO **显著缓解但未消除** BC 继承的“超车后并线、尾部高侧滑撞对手”问题。BC 严格判据 8 例；G5 U30 为 3 例、G7 U40 为 5 例、G9 hard-neighbor U45 为 1 例。
- hard-neighbor U45 对该特定机制最强，但总体碰撞为 17，且相对 BC 新增 10 个碰撞场景；它把失败转移到其他车车接触，不能称为总体解决。
- 这是单 seed、固定选择面板的描述性证据；全部 checkpoint 的多重比较校正后，没有单个结果达到常规显著性门槛。
"""
    ),
    md(
        """## Context & Methods

本 notebook 读取 `analyze_all_experiments.py` 生成的有界 CSV/JSON。原始输入包括 16 个 run 的配置、415 条 formal-update metrics、92 个 PPO eval 面板（另有 1 个 BC 面板）、55,200 条 PPO-vs-BC 逐 episode 配对记录，以及 600 条 BC 自参考行和对应 NPZ trace。

“甩尾”采用可复核的复合判据：车车碰撞；参考帧前相对赛道进度曾越过 0；碰撞时对手位于 ego 后方且更接近车尾；前 0.5 秒横向间距至少闭合 0.10 m；由位置差分得到的最大绝对侧滑角达到阈值。主阈值为 5°，同时报告 3°/8°敏感性。trace 未保存模拟器真实 `slip_angle` 或接触点，因此这是机制级代理，不是接触动力学的直接观测。
"""
    ),
    code(
        """from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt

OUT = Path.cwd()
if not (OUT / 'analysis_summary.json').exists():
    OUT = Path('analysis_results/ppo_all_experiments_0723')

summary = json.loads((OUT / 'analysis_summary.json').read_text())
configs = pd.read_csv(OUT / 'run_config_matrix.csv')
train = pd.read_csv(OUT / 'training_updates.csv')
train_summary = pd.read_csv(OUT / 'training_run_summary.csv')
panels = pd.read_csv(OUT / 'eval_panels.csv')
episodes = pd.read_csv(OUT / 'eval_episode_comparison.csv')
tail = pd.read_csv(OUT / 'tail_issue_by_panel.csv')
features = pd.read_csv(OUT / 'collision_episode_features.csv')
frequency = pd.read_csv(OUT / 'collision_scenario_frequency.csv')
paired = pd.read_csv(OUT / 'selected_paired_comparisons.csv')
logical = pd.read_csv(OUT / 'logical_clip020_45u_path.csv')

print({
    'runs': len(configs),
    'formal_updates': len(train),
    'eval_panels': len(panels),
    'valid_panels': int(panels.summary_valid.sum()),
    'episode_pairs': len(episodes),
    'collision_traces': len(features),
})
"""
    ),
    md("## Data quality"),
    code(
        """quality = pd.DataFrame([
    ['完整 eval 面板', f\"{int(panels.summary_valid.sum())}/{len(panels)}\", '高 LR U20 缺 1 episode'],
    ['scenario IDs / panel', int(panels.unique_scenario_ids.min()), '固定 ID 可逐场景配对'],
    ['唯一物理初态 / panel', int(panels.unique_physical_initial_conditions.min()), '闭环端点导致 8 组重复'],
    ['训练 update', len(train), '16 个 run 均达到配置的 update 数'],
    ['碰撞 trace 缺失', int(tail.missing_collision_traces.sum()), '相对各 JSON 记录为 0'],
], columns=['检查', '结果', '解释'])
display(quality)
"""
    ),
    md("## Model parameters and experiment matrix"),
    code(
        """display(configs[[
    'group','arm','critic','env_workers','batch_size','num_updates',
    'actor_epochs','critic_epochs','gru_learning_rate','head_learning_rate',
    'critic_learning_rate','gamma','gae_lambda','clip_range','target_kl',
    'steering_latent_std','speed_physical_std','hard_neighbors',
    'collision_pool_count','actor_parameter_count','critic_parameter_count'
]].sort_values(['group','arm']))
"""
    ),
    md("## Training metrics"),
    code(
        """display(train_summary[[
    'group','arm','formal_metric_rows','mean_rollout_collision_rate',
    'final_rollout_collision_rate','mean_episode_return','final_episode_return',
    'median_approx_kl_mean','max_approx_kl_max','mean_clip_fraction',
    'median_actor_grad_norm','max_actor_grad_norm','final_explained_variance',
    'early_stop_updates','actor_steps_completed','actor_steps_planned'
]].sort_values(['group','arm']))
"""
    ),
    md(
        """训练指标与 eval 的时间语义不同：metrics 的 update `k` rollout 由 checkpoint `k-1` 产生，而 eval U`k` 使用更新后的 checkpoint `k`；因此不能把同编号两列当作同一 actor 的直接泛化差。G8/G9 的训练 rollout 指标好于 G7，而后期 eval 更差，也证明训练 collision/return 不能替代固定面板评估。"""
    ),
    md("## Evaluation paths"),
    code(
        """valid = panels[(panels.group != 'BC') & panels.summary_valid].copy()
paths = valid.pivot_table(index=['group','arm','run'], columns='update', values='collision_count', aggfunc='first')
display(paths)

fig, ax = plt.subplots(figsize=(10, 4.8))
for label, frame in [
    ('clip 0.20 logical', logical),
    ('clip 0.25', valid[valid.group.eq('G8')]),
    ('hard-neighbor', valid[valid.group.eq('G9')]),
]:
    frame = frame.sort_values('update')
    ax.plot(frame['update'], frame.collision_count, marker='o', label=label)
ax.axhline(summary['bc_collision_count'], color='black', linestyle='--', linewidth=1, label='BC')
ax.set(xlabel='formal update', ylabel='collision episodes / 600', title='45-update candidate paths')
ax.legend()
ax.grid(alpha=.2)
plt.show()
"""
    ),
    md("## Exact episode-paired comparison against BC"),
    code(
        """display(paired)

transition = (episodes.groupby(['group','run','update','collision_transition'])
              .size().rename('episodes').reset_index())
display(transition.sort_values(['group','run','update','collision_transition']))
"""
    ),
    md("## Collision commonality"),
    code(
        """top = frequency.sort_values(['ppo_collision_panel_count','scenario_id'], ascending=[False, True]).head(15)
display(top)

fig, ax = plt.subplots(figsize=(10, 5))
plot = top.sort_values('ppo_collision_panel_rate')
ax.barh(plot.scenario_id.str.replace('evaluation-', '', regex=False), plot.ppo_collision_panel_rate)
ax.set(xlabel='fraction of 92 PPO panels with collision', title='Most persistent collision scenarios')
ax.grid(axis='x', alpha=.2)
plt.show()
"""
    ),
    md(
        """两个场景在 92/92 PPO 面板均碰撞：`sp17-ego727-raceline2-v0.5` 与 `sp35-ego1497-raceline1-v0.5`，它们主要是稳定的早期/前向接触，并非甩尾。最持续的甩尾型场景是 `sp5-ego213-raceline0-v0.7`：79/92 面板碰撞，且发生碰撞时 79/79 都满足结构性尾部接触和严格 5° 判据。"""
    ),
    md("## Did PPO solve the inherited post-overtake tail-swing failure?"),
    code(
        """wanted = {
    ('BC', 0): 'BC',
    ('ppo_privilege_gru_0722_long_clip020', 30): 'G5 clip.20 U30',
    ('ppo_privilege_gru_0722_long45_clip020', 40): 'G7 clip.20 U40',
    ('ppo_privilege_gru_0722_long45_clip020', 45): 'G7 clip.20 U45',
    ('ppo_privilege_gru_0722_long45_clip025', 45): 'G8 clip.25 U45',
    ('ppo_privilege_gru_0722_long45_clip020_hard', 20): 'G9 hard U20',
    ('ppo_privilege_gru_0722_long45_clip020_hard', 45): 'G9 hard U45',
}
rows=[]
for (run, update), label in wanted.items():
    row = panels[(panels.run == run) & (panels['update'] == update)].iloc[0]
    rows.append({
        'checkpoint': label,
        'collisions': row.collision_count,
        'vehicle': row.vehicle_collision_count,
        'wall': row.wall_only_collision_count,
        'structural_tail': row.post_overtake_rear_contact_count,
        'strict_tail_3deg': row.high_sideslip_tail_3deg_count,
        'strict_tail_5deg': row.high_sideslip_tail_5deg_count,
        'strict_tail_8deg': row.high_sideslip_tail_8deg_count,
        'BC_strict_resolved': row.bc_strict_5deg_tail_scenarios_resolved,
        'new_strict_vs_BC': row.new_strict_5deg_tail_scenarios_vs_bc,
    })
tail_selected = pd.DataFrame(rows)
display(tail_selected)

ax = tail_selected.set_index('checkpoint')[['collisions','strict_tail_5deg']].plot.bar(figsize=(10,4.8))
ax.set(ylabel='episode count / 600', title='Total collisions vs strict post-overtake tail-swing proxy')
ax.grid(axis='y', alpha=.2)
plt.xticks(rotation=25, ha='right')
plt.show()
"""
    ),
    code(
        """bc_strict = pd.read_csv(OUT / 'bc_tail_scenario_outcomes.csv')
selected = bc_strict[bc_strict.run.isin([
    'BC',
    'ppo_privilege_gru_0722_long_clip020',
    'ppo_privilege_gru_0722_long45_clip020',
    'ppo_privilege_gru_0722_long45_clip025',
    'ppo_privilege_gru_0722_long45_clip020_hard',
])]
selected = selected[
    ((selected.run == 'BC') & (selected['update'] == 0)) |
    ((selected.run.str.endswith('long_clip020')) & (selected['update'] == 30)) |
    ((selected.run.str.endswith('long45_clip020')) & selected['update'].isin([40,45])) |
    ((selected.run.str.endswith('long45_clip025')) & (selected['update'] == 45)) |
    ((selected.run.str.endswith('long45_clip020_hard')) & selected['update'].isin([20,45]))
]
display(selected.pivot_table(index='scenario_id', columns=['run','update'], values='high_sideslip_tail_5deg', aggfunc='first'))
"""
    ),
    md(
        """## Takeaways

1. **总体安全候选仍是普通 clip 0.20。** G5 U30 与 G7 U40 都是 11/600；G7 U45 为 12/600。45U 没有形成继续单调下降，而是进入 11–14 的波动平台。
2. **clip 0.25 没有继续改善。** U45 为 16/600，整条 45U 路径也没有优于 clip 0.20 的 11。
3. **PPO 对 BC 甩尾是部分修复。** G5 U30 解决 BC 8 个严格甩尾中的 7 个，但又产生 2 个新严格甩尾；G7 U40 同样解决 7 个，却新增 4 个。
4. **hard-neighbor 是机制定向改善，不是总体 winner。** U45 严格甩尾只剩 1 个、没有新增严格甩尾，但总碰撞 17，并新增 10 个 BC 未碰撞场景，说明失败被转移。
5. **不要用单点或训练 metrics 宣称收敛/泛化。** 所有结果来自单 seed 和同一固定面板；Austin600 还含 8 组重复物理初态。应先修正面板生成，再做预注册 checkpoint 的 holdout 复评。
"""
    ),
]

path = OUT / "ppo_all_experiments_analysis.ipynb"
execution_count = 0
namespace: dict = {}
active_outputs: list[dict] = []


def display(value):
    if hasattr(value, "to_html"):
        data = {"text/html": value.to_html(), "text/plain": repr(value)}
    else:
        data = {"text/plain": repr(value)}
    active_outputs.append({"output_type": "display_data", "data": data, "metadata": {}})


namespace["display"] = display
for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    execution_count += 1
    cell["execution_count"] = execution_count
    active_outputs = cell["outputs"]
    stdout = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stdout):
            exec(compile(cell["source"], f"notebook-cell-{execution_count}", "exec"), namespace)
        text_output = stdout.getvalue()
        if text_output:
            cell["outputs"].insert(0, {"output_type": "stream", "name": "stdout", "text": text_output})
        plt = namespace.get("plt")
        if plt is not None:
            for number in plt.get_fignums():
                figure = plt.figure(number)
                buffer = io.BytesIO()
                figure.savefig(buffer, format="png", dpi=120, bbox_inches="tight")
                active_outputs.append(
                    {
                        "output_type": "display_data",
                        "data": {"image/png": base64.b64encode(buffer.getvalue()).decode("ascii")},
                        "metadata": {},
                    }
                )
            plt.close("all")
    except Exception as exc:
        cell["outputs"].append(
            {
                "output_type": "error",
                "ename": type(exc).__name__,
                "evalue": str(exc),
                "traceback": traceback.format_exc().splitlines(),
            }
        )
        path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n")
        raise

path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n")
print(path)
