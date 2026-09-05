# robot_data_hub

- [RoboCOIN](https://huggingface.co/RoboCOIN)
- [ABC-130k](https://huggingface.co/datasets/XDOF/ABC-130k)

## ABC-130k 数据选取配置

- 数据范围：YAM 双臂真实数据，覆盖 `data/train/` 下全部任务目录（当前 201 个），优先保证任务多样性。
- 抽样方式：每个任务目录随机选取 30 条完整轨迹，不足 30 条则全部保留；固定随机种子为 `42`。
- 预计规模：最多 6,030 条轨迹，约占原数据集 130,703 条轨迹的 4.6%（按轨迹数量计算）。
