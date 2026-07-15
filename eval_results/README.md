# eval_results — 评估输出暂存区

这是 `eval_multiagent.py` / `eval_singleagent.py` / `evaluate.sh` 的**默认写入目录**（路径硬编码在这些文件里），不是实验归档地。

**评估跑完，把输出目录移进它所属的实验目录**：

```bash
mv eval_results/<tag>_<map>  Experiments/B2_ppo_pilot/eval_results/
```

历史上的 73 个废弃轨道输出（D1 / 旧 D2 / D4a / D4c / anchor / end2race_*，共 33 GB）已归入
`Experiments/_archive/eval_results/`。它们是**本地唯一副本**——远端没有备份，因为它们产生于
2026-07-08「实验只在远端跑」政策之前。未经项目所有者批准不要删除。

实验编号与索引见 `Experiments/INDEX.md`。
