# robot_data_hub

- [AgiBot World 2026](https://huggingface.co/datasets/agibot-world/AgiBotWorld2026)
- [ABC-130k](https://huggingface.co/datasets/XDOF/ABC-130k)

## ABC-130k 数据选取配置

- 数据范围：YAM 双臂真实数据，覆盖 `data/train/` 下全部任务目录（当前 201 个），优先保证任务多样性。
- 抽样方式：每个任务目录随机选取 30 条完整轨迹，不足 30 条则全部保留；固定随机种子为 `42`。
- 预计规模：最多 6,030 条轨迹，约占原数据集 130,703 条轨迹的 4.6%（按轨迹数量计算）。

## AgiBot World 2026 数据选取配置

- 数据范围：`ImitationLearning/CommercialSpaces`（商业场景），当前 138 条轨迹。
- 抽样方式：按文件体积取最小的 5 条轨迹（便于快速采样）；也可用 `scripts/download_agibotworld_subset.py select --strategy random --seed 42` 随机抽样。
- 已下载：5 条轨迹，共约 28 GiB，保存于 `dataset/raw/AgiBotWorld2026/`（LeRobot 格式 tar.gz）。

