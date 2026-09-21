# robot_data_hub

- [AgiBot World 2026](https://huggingface.co/datasets/agibot-world/AgiBotWorld2026)
- [ABC-130k](https://huggingface.co/datasets/XDOF/ABC-130k)
- [Galaxea Open-World Dataset](https://huggingface.co/datasets/OpenGalaxea/Galaxea-Open-World-Dataset)

## 项目组织原则

数据清理工作拆成四类相互独立的产物，各司其职，互不混用：

1. **目标格式数据契约（唯一）**：定义清理后数据的统一格式与存储结构，整个项目只有一份，见 [`docs/export-contract.md`](docs/export-contract.md)。所有源数据都向它对齐，任何格式调整只改这一处。
2. **源数据集数据契约（每数据集一份）**：描述单个源数据集的原始字段、坐标系、时间戳等语义，每个数据集维护一份（如各数据集自带的格式说明），只讲“这个数据源长什么样”。
3. **转换代码（每数据集一份，共用函数）**：把每个源数据集转换成目标格式的脚本，一个数据集一个转换入口；跨数据集共用的逻辑（写 Parquet、编码视频、坐标变换等）下沉到共用函数，避免复制。
4. **目标格式 dataset viewer（人工检查用）**：读取目标格式数据并可视化，供人工核对转换结果是否与源数据语义一致；只依赖目标契约，不依赖任何单个源数据集。

这四点把“格式定义”与“实现”分开、“各数据源的差异”与“公共逻辑”分开、“转换”与“检查”分开。新增数据源时，只需新增一份源契约和一个转换脚本。

## ABC-130k 数据选取配置

- 数据范围：YAM 双臂真实数据，覆盖 `data/train/` 下全部任务目录（当前 201 个），优先保证任务多样性。
- 抽样方式：每个任务目录随机选取 30 条完整轨迹，不足 30 条则全部保留；固定随机种子为 `42`。
- 预计规模：最多 6,030 条轨迹，约占原数据集 130,703 条轨迹的 4.6%（按轨迹数量计算）。

## AgiBot World 2026 数据选取配置

- 数据范围：`ImitationLearning/CommercialSpaces`（商业场景），当前 138 条轨迹。
- 抽样方式：按文件体积取最小的 5 条轨迹（便于快速采样）；也可用 `scripts/download_agibotworld_subset.py select --strategy random --seed 42` 随机抽样。
- 已下载：5 条轨迹，共约 28 GiB，保存于 `dataset/raw/AgiBotWorld2026/`（LeRobot 格式 tar.gz）。

## Galaxea Open-World Dataset 数据选取配置

- 数据范围：`lerobot/` 下全部 227 个任务归档，每个归档是一个自包含的 LeRobot v2.1 数据集，内含多条轨迹（episode）。
- 抽样方式：按归档体积取最小的 5 个任务归档（便于快速采样）；也可用 `scripts/download_galaxea_subset.py select --strategy random --seed 42` 随机抽样。
- 说明：该数据集以任务归档为最小下载单元，无法单独下载单条轨迹，因此抽样单元是任务而非轨迹；5 个归档共含 244 条轨迹。
- 已下载：5 个任务归档，共约 5.00 GiB / 244 条轨迹，保存于 `dataset/raw/Galaxea-Open-World-Dataset/`（tar.gz 及 `extract` 解压后的 LeRobot 数据集）。

