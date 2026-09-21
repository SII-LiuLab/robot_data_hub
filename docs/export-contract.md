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
- 共同时间原点为本 episode 各流首样本中的最早者。各流不能分别以自己的首样本归零。
- 时间只在 episode 内可比；不记录 clock 标识或绝对时刻，不承诺跨 episode 的时间对齐。
- 保留原生采样时刻与间隔；不插值、重采样或补帧，不通过行号或 FPS 推算时间。
- 每个流内按时间非递减排列。行号只表示该流内的样本顺序，不用于跨流配对。
- instruction 按共享 clock 的起止时间标注，不作为周期采样流。

## 3. State 表示

### 坐标系

左右臂 EEF 位姿各自相对于一个 episode 内**时不变**的参考坐标系：该系相对外部世界保持恒定，不得使用随时间运动的 link（如人形躯干 link）。**不要求左右臂使用同一参考系**（臂间关系不在本契约消费范围内，UMI 等源也无法提供）。除时不变外，不规定其原点与轴向：各源可沿用各自天然的固定系（如机器人 base、odom 系、或 episode 起始工具位姿）。`info.json` 记录左右参考系与工具系的身份，使其可复现、可审计。

> 注：下游由 state 导出的 delta 采用 body-frame 定义 `Δ = T_k⁻¹ · T_{k+1}`，参考系的原点与轴向不影响 delta，这是上文不规定其轴向的前提。

每个 EEF 坐标系原点为对应夹爪的工具中心点（TCP）；X 为工具接近方向，Y 沿夹爪开合轴，Z=X×Y。`info.json` 说明 TCP 的物理定义（如两指闭合中点）及工具 Y 轴正向所指的夹指，不能仅用 `left`、`right` 等名称代替定义。

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

## 4. RGB

每个相机独立保存为无音轨的 H.264 MP4，并配套逐帧 Parquet 索引：

| 字段 | 类型 | 含义 |
|---|---|---|
| `frame_index` | `int64` | 视频按显示顺序解码后的帧序号，从 0 开始 |
| `timestamp_ns` | `int64` | 该帧在共享 clock 下的采集时刻 |

索引与视频帧一一对应。保留原生帧序和采样间隔，包括可变帧率；采集时间以索引中的 `timestamp_ns` 为准。`info.json` 记录各相机的 ID、视角（如左腕、右腕、外部）及图像宽高。

## 5. instruction

`instruction` 在 episode 元数据中保存为列表 `instructions`，每条记录一段生效区间：

| 字段 | 类型 | 含义 |
|---|---|---|
| `start_ns` | `int64` | 该指令生效起点，相对共同时间原点 |
| `end_ns` | `int64` | 该指令生效终点，须大于 `start_ns` |
| `text` | `string` | 非空 UTF-8 任务指令文本 |

至少一条；整集仅一句指令时，写一条覆盖全轨迹时间范围的记录即可。

## 6. 存储结构

```text
dataset/
├── info.json
├── episodes.jsonl
└── episodes/<episode_id>/
    ├── state/
    │   ├── left_eef.parquet
    │   ├── right_eef.parquet
    │   ├── left_gripper.parquet
    │   └── right_gripper.parquet
    └── rgb/
        ├── <camera_id>.mp4
        └── <camera_id>.parquet
```

`info.json` 记录 `format_version` 及数据集级约定：参考系、工具系、EEF 流绑定、相机。

`episodes.jsonl` 每行一个 episode，记录 `episode_id`、本集相机及指令区间。所有 episode 级信息集中在这一处，不再有 per-episode JSON 文件。各字段的具体定义见第 7 节。

## 7. 元数据 schema

两处元数据：数据集级 `info.json` 与逐 episode 的 `episodes.jsonl`。以下为契约字段定义；未列出的字段不属于契约，实现应忽略。所有 `*_ns` 字段均为 `int64` 纳秒，位置单位为米。

### 7.1 info.json

| 字段 | 类型 | 必填 | 含义 |
|---|---|---|---|
| `format_version` | string | 是 | 契约版本号，`MAJOR.MINOR`；不兼容变更递增 MAJOR |
| `frames` | object | 是 | 参考系与工具系约定注册表，见 7.2 |
| `streams` | object | 是 | EEF 流到参考系/工具系的绑定，见 7.3 |
| `cameras` | object | 是 | 相机注册表，见 7.4 |

```json
{
  "format_version": "2.0",
  "frames": {
    "references": {
      "base_link": { "origin": "机器人底座中心", "axes": "X 前, Y 左, Z 上" }
    },
    "tools": {
      "yam_gripper": {
        "tcp": "两指闭合中点",
        "x": "接近方向",
        "y": "开合轴，正向指向<上指>",
        "z": "X×Y"
      }
    }
  },
  "streams": {
    "left_eef":  { "reference": "base_link", "tool": "yam_gripper" },
    "right_eef": { "reference": "base_link", "tool": "yam_gripper" }
  },
  "cameras": {
    "left_wrist": { "view": "left_wrist", "width": 640, "height": 480 }
  }
}
```

### 7.2 frames

`frames.references` 与 `frames.tools` 是 ID 到文本定义的映射，供 7.3 引用。约定写在此处一次，不在 episode 级重复。

`frames.references.<id>`：

| 字段 | 类型 | 必填 | 含义 |
|---|---|---|---|
| `origin` | string | 是 | 参考系原点的物理定义 |
| `axes` | string | 是 | 参考系轴向的物理定义，须可据此确认其在 episode 内恒定 |

`frames.tools.<id>`：

| 字段 | 类型 | 必填 | 含义 |
|---|---|---|---|
| `tcp` | string | 是 | TCP 的物理定义（如“两指闭合中点”） |
| `x` | string | 是 | 工具 X 轴的物理定义（接近方向） |
| `y` | string | 是 | 工具 Y 轴的物理定义，须指明正向所指的夹指，不能仅写 `left`/`right` |
| `z` | string | 是 | 工具 Z 轴的物理定义，固定为 X×Y |

EEF 原点与轴向固定为第 3 节规定（原点 = TCP，X = 接近方向，Y = 开合轴，Z = X×Y）。

### 7.3 streams

`streams` 把 EEF 状态流绑定到 7.2 中的参考系与工具系；仅列出需要帧绑定的流（左右夹爪流无帧字段，故不列出）。每个值为：

| 字段 | 类型 | 必填 | 含义 |
|---|---|---|---|
| `reference` | string | 是 | `frames.references` 中的键 |
| `tool` | string | 是 | `frames.tools` 中的键 |

左右可指向相同或不同的参考系/工具系，契约不要求相同。

### 7.4 cameras

`cameras` 以 `camera_id` 为键：

| 字段 | 类型 | 必填 | 含义 |
|---|---|---|---|
| `view` | string | 是 | 视角，建议取值 `left_wrist`、`right_wrist`、`exterior`、`other` |
| `width` | int32 | 是 | 图像宽（像素） |
| `height` | int32 | 是 | 图像高（像素） |

每个键须存在对应的 `rgb/<camera_id>.mp4` 与 `rgb/<camera_id>.parquet`。

### 7.5 episodes.jsonl

UTF-8 JSONL，每行一个 episode 对象，行顺序不限：

| 字段 | 类型 | 必填 | 含义 |
|---|---|---|---|
| `episode_id` | string | 是 | 等于 `episodes/<episode_id>/` 目录名 |
| `cameras` | string[] | 是 | 本集包含的相机，须为 `info.json.cameras` 的键，且与 `rgb/` 内容一一对应 |
| `instructions` | object[] | 是 | 指令区间列表，字段见第 5 节，至少一条 |

全体 `episode_id` 须与 `episodes/` 下的子目录一一对应。

```json
{"episode_id":"episode_0001","cameras":["left_wrist"],"instructions":[{"start_ns":0,"end_ns":12000000000,"text":"pick up the cup"}]}
{"episode_id":"episode_0002","cameras":["left_wrist"],"instructions":[{"start_ns":0,"end_ns":8000000000,"text":"open the drawer"}]}
```
