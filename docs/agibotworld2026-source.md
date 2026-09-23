# AgiBot World 2026 源数据契约

本项目下载的 `ImitationLearning/CommercialSpaces/task_*/<起止帧>.tar.gz` 是各自独立的 LeRobot v2.1 数据集。以下记录的是已下载版本的实际结构与转换所用字段；源数据版本见 `dataset/raw/AgiBotWorld2026/subset_manifest.json`。

- `data/meta/info.json`：`robot_type=g2a`，视频相机特征、`observation.state` 向量的 `field_descriptions[*].indices`，以及可选的 `instruction_segments`。各归档的状态向量长度不同，字段索引必须从本归档的 `info.json` 读取。
- `data/meta/episodes.jsonl`：每集任务文本和长度。
- `data/data/chunk-*/episode_*.parquet`：逐帧 `observation.state`、`timestamp`（相对本集起点的秒数，源列为 float32）、`frame_index`、`episode_index`。状态与相机按帧对应；没有每个相机独立的采集时间列。源秒数乘 `1e9` 后取整，不按 30 fps 重造时间戳；由于源为 float32，后段时间存在微秒级量化误差。
- `data/videos/chunk-*/observation.images.*/episode_*.mp4`：逐相机视频。深度通道由 `video.is_depth_map=true` 标注；RGB 通道为 AV1 MP4。每个视频按显示顺序对应同集 Parquet 的每一行。

转换读取 `state/left_effector/position`、`state/right_effector/position` 的实测夹爪值，`state/joint/position` 的双臂 14 关节角，`state/waist/position` 的 5 个腰部关节，以及存在时的 `state/robot/position` 和 `state/robot/orientation`（xyzw）。不把 `action` 写入目标格式。

本地 `task_3401/399093_399454.tar.gz` 将底盘位置与姿态声明为 `dimensions: 0, indices: []`，即整个归档的 14 集都没有底盘位姿记录。该归档共 38,345 帧的 `action/robot/velocity` 两维全部为 0；结合这一归档为原地操作的确认，转换时将底盘视为不移动，使用固定底盘参考系。脚本逐集校验速度命令确实为零；其它有底盘位姿的归档照常使用实测位姿。若未来遇到缺少位姿且底盘命令非零的归档，转换会报错。

本地 5 个归档的实测夹爪值在约 `[-0.91, 0]`：G2 OmniPicker URDF 的主动内指关节向负角运动时开口增大，故转换采用 `openness = clip(-state/0.91, 0, 1)`；`0` 对应闭合，`-0.91` 对应张开。`0.91` 是已下载真实样本的开端饱和值，不是 URDF 关节下限（模型下限约 `-0.7854`）；这是源数据到 `[0,1]` 的显式归一化约定，跨数据版本使用前应重新核对。

逐集指令优先选 `instruction_segments` 的 `default` track：其区间是 `[start_frame_index, end_frame_index)`；其它 track 可能交叠，不并入单一指令时间轴。`default` 的空白区间与没有 `default` track 的整集使用 `episodes.jsonl.tasks[0]`。末尾 `end_frame_index == episode_length` 映射到最后一帧的时间戳，以满足目标契约的整集终点定义。

源格式说明：[AgiBot World 2026 官方数据页](https://huggingface.co/datasets/agibot-world/AgiBotWorld2026)。
