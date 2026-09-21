# 目标格式数据契约

本文是清理后数据的唯一目标格式契约，只定义数据内容、表示方式和存储结构。源数据的字段解释、转换方法及人工检查工具分别维护。

## 1. 数据内容

数据按 episode（一次连续操作轨迹）组织，每个 episode 包含：

| 内容 | 记录形式 |
|---|---|
| 左臂、右臂实际 state | 各自的 EEF 绝对位姿 |
| 左、右夹爪实际 state | 各自的开合程度，范围 `[0,1]`；`0` 完全闭合，`1` 完全张开 |
| RGB | 一个或多个相机的彩色视频及逐帧时间戳 |
| instruction | 带起止时间的任务指令文本列表 |

不记录 action、关节状态、速度、深度、触觉、力或音频。

## 2. 时间模型

**同一 episode 的所有数据流共用一个 clock，各流按原生节奏独立、异步记录。** 左右臂位姿、左右夹爪、各相机均有自己的时间戳序列，不要求频率或样本数相同。

- 每个样本携带 `timestamp_ns`：`int64`，单位纳秒，表示相对该 episode 共同时间原点的采样时刻。
- episode 元数据记录 `clock_id` 及 `time_origin`（共同时间原点的定义）。各流不能分别以自己的首样本归零。
- 保留原生采样时刻与间隔；不插值、重采样或补帧，不通过行号或 FPS 推算时间。
- 每个流内按时间非递减排列。行号只表示该流内的样本顺序，不用于跨流配对。
- instruction 按共享 clock 的起止时间标注，不作为周期采样流。

## 3. State 表示

### 坐标系

左右臂 EEF 位姿均相对于同一个、在 episode 内固定的参考坐标系 `reference_frame`，采用右手系，X 向前、Y 向左、Z 向上。元数据明确参考系的物理原点及“前、左、上”的依据。

每个 EEF 坐标系原点为对应夹爪的工具中心点（TCP）；X 为工具接近方向，Y 沿夹爪开合轴，Z=X×Y。元数据明确左右 TCP 的物理位置和工具 Y 轴正向所指的夹指，不能仅用 `left`、`right` 等名称代替定义。

pose 表示从 EEF 坐标系到参考坐标系的变换：

```text
p_reference = R @ p_eef + t
```

### 位姿与夹爪字段

左右臂、左右夹爪分别存为独立 Parquet 表，每行一个原生样本：

| 数据流 | 字段 | 类型 | 含义 |
|---|---|---|---|
| 所有 state 流 | `timestamp_ns` | `int64` | 共享 clock 下的采样时刻 |
| 左/右 EEF | `pose` | `float64[9]` | 位置 3 维 + rotation6D 6 维 |
| 左/右夹爪 | `openness` | `float32` | 实际开合程度，`[0,1]` |

`pose` 的固定顺序为：

```text
[x, y, z, R00, R10, R20, R01, R11, R21]
```

`[x,y,z]` 是 TCP 在参考系中的位置，单位米；后 6 维是旋转矩阵 `R` 的第一列和第二列，第三列为第一列叉乘第二列。有效旋转的前两列须为正交单位向量。

`openness` 的端点表示实际闭合和张开端点，不是该 episode 内的观测最小值和最大值。

## 4. RGB 与 instruction

每个相机独立保存为无音轨的 H.264 MP4，并配套逐帧 Parquet 索引：

| 字段 | 类型 | 含义 |
|---|---|---|
| `frame_index` | `int64` | 视频按显示顺序解码后的帧序号，从 0 开始 |
| `timestamp_ns` | `int64` | 该帧在共享 clock 下的采集时刻 |

索引与视频帧一一对应。保留原生帧序和采样间隔，包括可变帧率；采集时间以索引中的 `timestamp_ns` 为准。元数据记录各相机的 ID、视角（如左腕、右腕、外部）及图像宽高。

`instruction` 在 episode 元数据中保存为列表 `instructions`，每条记录一段生效区间：

| 字段 | 类型 | 含义 |
|---|---|---|
| `start_ns` | `int64` | 该指令生效起点，相对 `time_origin` |
| `end_ns` | `int64` | 该指令生效终点，须大于 `start_ns` |
| `text` | `string` | 非空 UTF-8 任务指令文本 |

至少一条；整集仅一句指令时，写一条覆盖全轨迹时间范围的记录即可。

## 5. 存储结构

```text
dataset/
├── dataset.json
└── episodes/<episode_id>/
    ├── episode.json
    ├── state/
    │   ├── left_eef.parquet
    │   ├── right_eef.parquet
    │   ├── left_gripper.parquet
    │   └── right_gripper.parquet
    └── rgb/
        ├── <camera_id>.mp4
        └── <camera_id>.parquet
```

`dataset.json` 记录 `format_version` 和 episode ID 列表。

`episode.json` 记录 `episode_id`、`instructions`、`clock_id`、`time_origin`、`reference_frame`、左右 `eef_frames` 及 `cameras`。坐标系信息按第 3 节记录物理定义，相机信息按第 4 节记录。
