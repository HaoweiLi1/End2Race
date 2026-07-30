# Evaluation panel inputs

此目录只保存可复用的固定评估/训练输入，不保存某次actor的评估输出、日志或分析报告。

## `heldout_hard_v1/`

使用冻结B/U30在Austin的21,600个候选场景上预先分类得到。候选来自200个新起点，
按20个物理区块预先分成100个train和100个held-out起点；选择过程没有查看后续
treatment actor。

| 文件 | 条数 | 用途 |
|---|---:|---|
| `candidate_scenarios.json` | 21,600 | 完整候选网格 |
| `candidate_labels.jsonl` | 21,600 | 冻结U30逐候选结果 |
| `train_collision_scenarios.json` | 468 | train碰撞场景 |
| `train_near_miss_scenarios.json` | 400 | train近失场景 |
| `train_difficult_scenarios.json` | 868 | 上述两者并集 |
| `heldout_eval_collision_scenarios.json` | 334 | held-out hard334 |
| `heldout_eval_near_miss_scenarios.json` | 400 | held-out near400 |
| `heldout_eval_difficult_scenarios.json` | 734 | hard334与near400并集 |
| `heldout_eval_interval15_collision_scenarios.json` | 73 | hard73诊断面板 |
| `train_interval15_difficult_pool.json` | 229 | 合规fixed collision-role训练池 |
| `design_manifest.json` | — | 候选和起点设计 |
| `classification_summary.json` | — | 分类计数和选择完整性摘要 |

`train_interval15_difficult_pool.json`由103个ego-collision和126个near-miss场景组成，
全部为interval 15。near-miss阈值为最小OBB clearance不高于0.1m。

配对身份使用：

```text
(map, opponent_raceline, ego_idx, opp_idx, opponent_speed_scale)
```

跨raceline时不得用`(opp_idx - ego_idx) mod waypoint_count`反推interval。

评估输出继续写入`eval_results/<EXPERIMENT_ID>/<PANEL_ID>/<UPDATE>/`；不要向本目录写
actor结果或临时分析文件。
