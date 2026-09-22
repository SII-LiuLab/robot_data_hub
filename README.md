# robot_data_hub

- [AgiBot World 2026](https://huggingface.co/datasets/agibot-world/AgiBotWorld2026)
- [ABC-130k](https://huggingface.co/datasets/XDOF/ABC-130k)
- [Galaxea Open-World Dataset](https://huggingface.co/datasets/OpenGalaxea/Galaxea-Open-World-Dataset)
- [HiFi-UMI-2K](https://huggingface.co/datasets/simple-world-lab/HiFi-UMI-2K)
- [MolmoAct2-BimanualYAM](https://huggingface.co/datasets/allenai/MolmoAct2-BimanualYAM-Dataset)

## 项目组织原则

数据清理工作拆成四类相互独立的产物，各司其职，互不混用：

1. **目标格式数据契约（唯一）**：定义清理后数据的统一格式与存储结构，整个项目只有一份，见 [`docs/export-contract.md`](docs/export-contract.md)。所有源数据都向它对齐，任何格式调整只改这一处。
2. **源数据集数据契约（每数据集一份）**：描述单个源数据集的原始字段、坐标系、时间戳等语义，每个数据集维护一份（如各数据集自带的格式说明），只讲“这个数据源长什么样”。
3. **转换代码（每数据集一份，共用函数）**：把每个源数据集转换成目标格式的脚本，一个数据集一个转换入口；跨数据集共用的逻辑（写 Parquet、编码视频、坐标变换等）下沉到共用函数，避免复制。
4. **目标格式 dataset viewer（人工检查用）**：读取目标格式数据并可视化，供人工核对转换结果是否与源数据语义一致；只依赖目标契约，不依赖任何单个源数据集。

这四点把“格式定义”与“实现”分开、“各数据源的差异”与“公共逻辑”分开、“转换”与“检查”分开。新增数据源时，只需新增一份源契约和一个转换脚本。

## ABC-130k 数据选取配置

- 源格式说明：[`docs/abc130k-source.md`](docs/abc130k-source.md)。
- 转换到统一导出格式：`python scripts/convert_abc130k.py`；EEF 使用 YAM 模型与实测关节角 FK。用法与 TCP 定义见 [`docs/abc130k-conversion.md`](docs/abc130k-conversion.md)。
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

## HiFi-UMI-2K 数据选取配置

- 数据范围：全部 398 个 shard（`chunk-XXXX/part-0000`），每个 shard 是一个自包含的 LeRobot v3 数据集，内含约 140 条 episode（轨迹）。
- 抽样方式：先按体积取最小的 1 个 shard（当前为 `chunk-0397`），再在该 shard 内用固定随机种子 `42` 随机抽取 5 条轨迹；也可用 `scripts/download_hifi_umi_subset.py select --strategy smallest` 取最短的 5 条。
- 说明：该数据集把同一 shard 内全部 episode 的 6 路相机视频分别拼接成一路一个 MP4，单条轨迹不是独立文件，因此下载时对远端 MP4 发起 HTTP range 请求，按 episode 的 `from/to_timestamp` 精确截取并逐帧重新编码为 H.264（CRF 18，码率与源相当）。源 shard 的 `meta` 及各表保存在 `source/`。
- 已下载：5 条轨迹（`episode_000013`、`000024`、`000084`、`000091`、`000122`），共 8,395 帧 / 约 0.75 GiB，保存于 `dataset/raw/HiFi-UMI-2K/`；每个 episode 含 6 路相机 MP4、逐帧 `data.parquet` 与 `episode.json`。

## MolmoAct2-BimanualYAM 数据选取配置

- 数据范围：单个合并后的 LeRobot v3 数据集，机器人 `bi_yam_follower`，30 fps，共 32,246 条轨迹（episode）、76,046,658 帧，覆盖 34 个任务。
- 抽样方式：用固定随机种子 `42` 在整个数据集上随机抽取 5 条轨迹（`scripts/download_molmoact2_subset.py select --strategy random --seed 42`）；也可用 `--strategy smallest` 取最短的 5 条。
- 说明：该数据集把同一视频文件内多条 episode 的 3 路相机视频（`top`/`left`/`right`，AV1）分别拼接成一路 MP4，单条轨迹不是独立文件，因此下载时对远端 MP4 发起 HTTP range 请求，按 episode 的 `from/to_timestamp` 精确截取并逐帧重新编码为 H.264（CRF 18，码率与源相当）。episode 的逐帧表从 `data/chunk-*/file-*.parquet` 中按 `episode_index` 过滤得到；`meta/tasks_annotated.parquet` 提供逐 episode 的语言标注。源 `meta` 及各数据表保存在 `source/`。
- 已下载：5 条轨迹（`episode_006848`、`014787`、`022490`、`030268`、`032144`），共 12,103 帧 / 约 0.62 GiB，保存于 `dataset/raw/MolmoAct2-BimanualYAM/`；每个 episode 含 3 路相机 MP4、逐帧 `data.parquet` 与 `episode.json`。

## 统一机器人模型

ABC-130K / MolmoAct2 对应 YAM，AgiBotWorld2026 对应 G2，Galaxea 对应 R1 Lite。
三套自包含 URDF（本体、双臂、夹爪、视觉/碰撞网格、材质和清单）的入口见
[`assets/robot_models/README.md`](assets/robot_models/README.md)。
`assets/` 不纳入 Git，完整资产公开保存在 ModelScope：
[`BingqianWu/RobotDataHub-Assets`](https://modelscope.cn/models/BingqianWu/RobotDataHub-Assets)。
下载该仓库后，将其中的 `assets/` 目录放回本项目根目录即可。
可直接用于可视化和 FK；数据集 TCP、零位及安装外参尚未逐帧标定，状态在清单中明确记录。
