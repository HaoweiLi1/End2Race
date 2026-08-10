# Fixed training inputs

此目录只保存当前仍被训练合同直接读取、且不能由正式入口即时生成的固定输入。模型评测输出、日志和临时分析不得写入这里。

## 正式四图600评测不保存重复panel

正式模型评测统一由根目录`evaluate.sh`生成场景：每张地图50个circular start、3条opponent raceline、4个speed scale，共600个episode。Austin、Hockenheim、MoscowRaceway和Nuerburgring分别通过`MAP_NAME`运行；当前默认`COLLISION_SCOPE=ego`。

原`standard_multiagent_600_v1/`四份JSON与`evaluate.sh`按当前raceline和`get_circular_startpoints()`生成的场景逐条完全一致，且没有当前代码消费者，已作为重复资产删除。若racetrack数据或场景生成函数改变，应把它视为评测合同变化并重新建立对照，而不是静默恢复一份旧JSON副本。

## `first_action_preference_v1/`

Simulator-return-filtered first-action preference训练所用的固定同状态反事实数据集。`manifest.json`登记65个episode，其中46个target、19个matched control；目录同时保留snapshot、noop branch、candidate branch、状态前缀和构建Gate结果，以便复核第一动作标签及训练输入。

该目录是成功训练臂的输入，不是模型eval panel。`ppo/rollout.py`中的`FirstActionPreferenceDataset`和已完成运行的`run_config.json`都使用`first_action_preference_v1`作为dataset ID，因此不得仅为缩短名称而改名。

## 命名与输出边界

- 这里只保留manifest中的canonical训练数据ID；不维护`PJTE`、`heldout_hard`或`failed_*`别名。
- actor评测输出写入`eval_results/<experiment_id>/...`，不得写入本目录。
- 失败方法的科学结论保留在`.agents/HANDOFF.md`与`.agents/ANALYSIS.md`；已删除二进制不作为结论的唯一载体。
