# robot_data_hub

- [RoboCOIN](https://huggingface.co/RoboCOIN)
- [ABC-130k](https://huggingface.co/datasets/XDOF/ABC-130k)

## ABC-130k 数据选取配置

- 数据范围：YAM 双臂真实数据，覆盖 `data/train/` 下全部任务目录（当前 201 个），优先保证任务多样性。
- 抽样方式：每个任务目录随机选取 30 条完整轨迹，不足 30 条则全部保留；固定随机种子为 `42`。
- 预计规模：最多 6,030 条轨迹，约占原数据集 130,703 条轨迹的 4.6%（按轨迹数量计算）。

### 选择与下载

使用 `rdh` 环境运行；需先在 Hugging Face 上获得 ABC-130k 的访问权限并完成 `hf auth login`。脚本固定访问官方站点 `https://huggingface.co`。

```bash
conda activate rdh

# 只读取远端目录元数据，生成抽样清单并统计下载体积
python scripts/download_abc_subset.py select

# 按清单下载，保留原始目录结构
python scripts/download_abc_subset.py download --output-dir data/ABC-130k-subset
```

- 默认清单：`manifests/abc130k-train-30-seed42.json`，记录仓库 commit、选中的轨迹和文件大小。已有清单不会被覆盖。
- 每个任务独立使用种子 `42:<任务目录名>` 对排序后的 episode 列表抽样；`--per-task`、`--seed` 可调整配置。
- 下载选中轨迹的 `episode.mcap`、存在的 `annotation.mcap`，以及仓库说明、许可证和 `meta/`、`models/`、`docs/` 中的公共文件。
- 默认并发数为 4，临时网络错误最多额外重试 3 次，可通过 `--workers`、`--retries` 调整。重复运行下载命令会复用已完成的文件，失败记录保存在输出目录的 `download_failures.json`。
- 下载保持 MCAP 原始格式；随附的原始统计元数据仍描述完整数据集，子集范围以 `subset_manifest.json` 为准。
