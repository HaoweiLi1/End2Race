# End2Race 代码风格

本文从用户亲自编写的 `train.py`、`eval_multiagent.py`、`model.py`、`utils.py` 归纳，
用于指导后续重构与新代码。**目标是让新代码读起来像这几个文件写的，而不是像库代码。**

判据不是"哪种风格更好"，而是"与既有代码一致"。下面每条都标了实测依据。

---

## 0. 总原则：脚本风格，不是库风格

用户的文件是**可直接运行的脚本**：顶层 import、几个函数、末尾一个
`if __name__ == "__main__":` 把流程串起来。不要引入库工程化的层层抽象。

实测对比（用户文件 vs 现有 `ppo/` 模块）：

| 特征 | 用户文件 | 现有 `ppo/` | 应采用 |
|---|---|---|---|
| `from __future__ import annotations` | **0 / 6 文件** | 11 / 11 文件 | **不写** |
| 模块级 docstring | **无** | 有 | **不写** |
| 函数类型注解 | `eval_multiagent` 0/6、`utils` 0/28 | `scenarios` 22/24 | **默认不注解** |
| `add_argument` 单行 | **14/14、7/7 全单行** | 有多行写法 | **强制单行** |
| 最长行 | 154 / 105 / 128 字符 | 117 / 160 | **不为换行而换行** |

---

## 1. Import

**写法**：一行一个，按"用到就加"的顺序，不分组、不排序、不加空行分隔。

```python
import argparse
import sys
import gym
import numpy as np
import torch
import os
import gc
import imageio
from f110_gym.envs.base_classes import Integrator
from model import End2Race
from latticeplanner.utils import project_point_to_centerline, obsDict2oppoArray
from demonstration import setup_opp_planner
from ppo.reward import ProgressProjector
from utils import *
```

注意 `sys` 在 `gym` 前、`os`/`gc` 在 `torch` 后——**标准库与三方库交错是正常的**，
不要按 isort 重排。

规则：

- **不写 `from __future__ import annotations`**。用户 6 个文件里一次都没有。
- 项目自己的工具模块可以 `from utils import *`；三方库不要星号导入。
- `import torch.nn as nn`、`import torch.optim as optim` 这类惯用缩写照用。
- 只在真正需要时才用局部延迟 import（如子进程初始化、避免循环导入）。

---

## 2. CLI

**`add_argument` 永远占一条物理行**，不管多长。用户的 21 条参数无一例外。

```python
def parse_arguments():
    parser = argparse.ArgumentParser(description='Evaluate model on segment with opponent')

    # Model parameters
    parser.add_argument("--model_path", type=str, default='pretrained/end2race.pth')
    parser.add_argument("--hidden_scale", type=int, default=4)
    parser.add_argument("--noise", type=float, default=0.0)

    # Segment parameters
    parser.add_argument("--map_name", type=str, default="Austin")
    parser.add_argument("--ego_idx", type=int, default=0)
    parser.add_argument("--render", action='store_true')

    return parser.parse_args()
```

规则：

- 函数名固定 `parse_arguments()`，直接 `return parser.parse_args()`。
- 用 `#` 注释把参数分组（`# Model parameters`、`# Training configuration`、
  `# Data and model paths`），组间空一行。
- 每个参数都给 `type=` 和 `default=`；`store_true` 不给 default。
- **不写 `help=`**。参数名本身就是说明；加 help 会逼出多行写法。
- 长默认值也不换行，例如：

```python
parser.add_argument("--collision_cache_dir", type=str, default="post-trained/collision-cache/pretrained_end2race_austin_collision_pool_479")
```

---

## 3. 注释

只有两种，都很短。

**(a) 段落标记**：说明下面这块在做什么，不说为什么。

```python
    # Setup environment
    # Initialize model state
    # Load centerline
    # Reset environment
    # Main simulation loop
    # Model inference for ego
    # Apply noise
    # Step environment
    # Track metrics
    # Check collision
    # Calculate final metrics
    # Print results
```

特征：句首大写、无句号、2-4 个词、动词开头。它们是**代码的目录**，让人扫一眼就能
定位。密度约每 10-20 行一个。

**(b) 行尾澄清**：解释一个不自明的值或形状。

```python
tracker_steps = 10  # Default tracker steps
visited_points = [[], []]  # [ego_points, opp_points]
speed_prev = action_data[:-1, 1:2]  # Take desired_speed column and keep 2D shape
```

**不要写**：

- 模块级 docstring；
- 重复代码字面意思的注释（`# increment counter`）；
- 多行块注释解释设计动机——那属于 `.agents/` 文档，不属于源码；
- 参数逐条说明的 docstring。

**函数 docstring**：只写一行，动词开头，能省则省。

```python
def evaluate_segment(...):
    """Evaluate a single segment with model against lattice planner opponent"""

def parse_arguments():
    """Parse command line arguments"""

def _load_episodes(self, data_path: str):
    """Load CSV files and create sequences."""
```

句号有无不统一，不必强求。私有小函数可以完全不写。

---

## 4. 类型注解

**默认不写。** 只在两处用：

1. `Dataset` 这类有明确数据契约的类方法（`train.py` 的 8 个 def 中 6 个有）；
2. 返回结构复杂、不标注就读不懂时（`-> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]`）。

`eval_multiagent.py` 6 个函数、`utils.py` 28 个函数**一个注解都没有**——普通脚本函数
就该这样。不要给每个参数都加 `: float | None = None`。

---

## 5. 行宽与换行

**不设硬性行宽。** 用户文件最长 154 字符，且是有意的：

```python
env = gym.make("f110-v0", map=f"f1tenth_racetracks/{map_name}/{map_name}_map", map_ext=".png", num_agents=2, timestep=0.01, integrator=Integrator.RK4)
```

一次调用写一行，比拆成 7 行更好读。

**该换行的情况**：参数在语义上分组时，按组换行并对齐：

```python
    result = evaluate_segment(
        model, device, args.noise,
        args.map_name, args.ego_idx, args.interval_idx,
        args.ego_raceline, args.opp_raceline, args.opp_speedscale,
        args.sim_duration, args.render, args.save_trace, args.model_path,
        args.metrics_out,
        args.collision_scope,
    )
```

注意是**按语义分组换行**（模型/场景/时长/输出），不是每行一个参数。

**不要**为了凑 79/88/100 字符把一个表达式拆成三层临时变量。

---

## 6. 命名

全用 snake_case，**写完整词**，不做激进缩写：

```text
opp_speedscale   relative_unwrapped   tracker_count    ego_raw_lidar_history
initial_speeds   centerline_total_length              observation_finite
```

领域缩写沿用既有的：`ego` / `opp` / `idx` / `lidar` / `obs`。
私有辅助函数前缀 `_`（`_load_episodes`、`_create_sequences`）。

---

## 7. 程序结构

```python
import ...

def parse_arguments():
    ...

def helper_a(...):
    ...

def main_work(...):
    """One line."""
    ...

if __name__ == "__main__":
    args = parse_arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ...
    print(f"...")
```

规则：

- **不写 `def main()`**。编排逻辑直接放在 `if __name__ == "__main__":` 里。
- `parse_arguments()` 放最前面或靠前。
- 输出用 f-string 的 `print`。`train.py` 用 `print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.5f}")`，
  `eval_multiagent.py` 用 `print(f"STATE={result['state']}")` 这种机器可读格式。
- 错误处理：脚本入口用 `try/except` 打印到 `stderr` 并 `sys.exit(码)`，
  不在库函数里吞异常。

---

## 8. 数值与数据

- 显式 `float()` / `int()` / `bool()` 转换后再放进 dict 或 JSON——
  `eval_multiagent.py` 的 `episode_metrics` 每个字段都这么做，避免 numpy 标量泄漏。
- numpy 用 `np.asarray(..., dtype=...)` 明确 dtype。
- 字典字面量直接展开写，需要合并时用 `**`：

```python
    episode_metrics = {
        "episode_key": key,
        "outcome": outcome,
        "avg_speed": float(avg_speed),
        **proximity_quality,
        "ego_min_lidar": float(np.min(ego_lidar_minima)),
    }
```

---

## 9. 重构时的额外约束

- **一次只搬一个模块**，搬完立即更新 import 并删旧入口，不留两份同名实现。
- 搬迁不改数值、不改默认值、不改公式。风格调整和行为调整**不要放在同一个 diff**。
- 合并文件时保留原有的段落注释，不要顺手"整理"成 docstring。
- 新文件顶部**不加**模块 docstring；文件职责写在 `.agents/` 文档里。

---

## 10. 自检

提交前对照：

1. 有没有 `from __future__ import annotations`？删掉。
2. 有没有模块级 docstring？删掉。
3. 每个 `add_argument` 是不是一行？有没有混进 `help=`？
4. 普通脚本函数是不是加了不必要的类型注解？
5. 有没有为了行宽把一个调用拆成多行临时变量？
6. 注释是不是"段落标记 + 行尾澄清"两类之外的东西？
7. 有没有写 `def main()` 而不是直接放在 `__main__` 块里？
