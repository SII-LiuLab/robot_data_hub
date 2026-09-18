# 目标格式数据契约

本文是唯一的目标格式数据契约。适用真实 robot、UMI数据；

**“必须”表示交付要求；“条件要求”仅在对应输出类型、来源或功能启用时适用。缺证据不能记为通过。**

## 1. 输出类型与适用范围

| 输出 | 内容 | 必须满足 |
|---|---|---|
| LeRobot v3 实体数据集 | 数值 Parquet、元数据；可选 RGB MP4 | 第2–6节 |
| 原生异频引用包 | selection、原生 token 引用、clock/profile、checksum；payload 留在源位置 | 第2、7、8节 |
| 引用演练包、prepared 数值、兼容性输出 | 调试、准备或对照产物 | 明示范围与限制，不得仅凭生成成功标为正式训练数据 |

LeRobot 当前采用 `lerobot-v3-export-v1`、`codebase_version=v3.0`，位姿为 `state-action-ee-pose-v2`；RGB 为 `video-serving-v1`；原生引用包为 `native-temporal-package-v1`。

`numeric-only` 是合法的数值输出范围，不含视频；不得从 raw 临时补视频后声称它是已导出的媒体。是否可发布仍由目标用途要求的证据决定。

## 2. 所有输出的共同条件

### 2.1 时间、缺测与真实样本

- 每个 vision/proprio/state/action 流保留 native frequency、真实 source timestamp 和 clock 身份。未知时间单位或 clock 关系必须显式未知，不凭数量级、行号、同 FPS 或相近时间猜测。
- query time 只作 anchor。禁止插值、重采样、升降频、复制/drop frame 凑频率，或无限 hold-last 制造同步；选择删除片段不属于凑频率，但保留窗口必须有真实范围且不跨删除边界。
- 观测关联只允许同 episode、已验证 clock 下的真实历史样本，并受明确容差约束；返回实际时间及 age/gap。无历史记录、超容差、原生无效或 clock 未绑定时为缺测，不跳过坏点寻找更旧有效值。
- mask 必须为明确的布尔值并匹配对应分量/末端/通道；零占位不能作为有效监督，真实零值也不能当作缺测。无效 pose 占位仍须有限，不能用 NaN/Inf 隐藏错误。
- 稀疏流使用自己的 index/timestamp，不能按 dense row 下标配对。整数纳秒时间不得经过浮点秒重建；浏览器传输 epoch 纳秒使用字符串以保留精度。
- 时间倒退进入复核，不能自动修复；允许的确定性 duplicate epsilon shift/frame resequence 只能作为保留原时间和证据的独立 overlay，应用后重新验证。

## 3. LeRobot 数值字段

设 `N=len(ee_ids)`，`S` 为主 state 宽度，`A` 为主 action 宽度。EE 顺序由元数据明确给出，不交换左右手，不强制所有来源具有同样维度。

### 3.1 必需字段与表示

每个 EE 的绝对及相对 pose 块均为9维：

```text
[x, y, z, R00, R10, R20, R01, R11, R21]
  位置3维       旋转矩阵第一列      第二列
```

位置单位为米；rotation6D 是前两列，不是 Euler 或前两行。有效旋转须能构成合法 SE(3)，列向量正交、单位长度且无镜像。

| 字段 | 形状/类型 | 含义与约束 |
|---|---|---|
| `observation.state` | `[S]`，数值类型由 feature 声明 | 原生 state 前缀加绝对 EE pose；前缀可按已声明夹爪/转轴规则转换 |
| `observation.native_state` | 原生 state 宽度 | 转换前副本，不冒充已标准化 state |
| `action` | `[A]`，数值类型由 feature 声明 | 相对 EE 的 Δpos＋rotation6D，另含对应夹爪/手部目标 |
| `observation.ee_pose`、`action.ee_pose` | `[9N]`，`float64` | 分别为绝对观测 pose、绝对目标 pose |
| `observation.ee_pose_valid`、`action.ee_pose_valid` | `[N]`，`bool` | 每个 EE 的有效性 |
| `observation.ee_pose_origin`、`action.ee_pose_origin` | `[N]`，`int64` | `0`不可用、`1`原生记录、`2` FK；无效 pose 必须为0 |
| `observation.ee_pose_timestamp_ns`、`action.ee_pose_timestamp_ns` | `[N]`，`int64` | 各自真实源时刻；有效值非负且不得倒退 |
| `observation.state_valid_mask`、`action_valid_mask` | `[S]`、`[A]`，`bool` | 与主向量等宽，pose 切片有效性与对应 EE 一致 |
| `timestamp` | `[1]`，`float32` | episode 内 LeRobot 相对秒；不能替代整数源时间戳 |
| `frame_index`、`episode_index`、`index`、`task_index` | `[1]`，`int64` | 分别为 episode 内帧号、episode 编号、全局行号、任务索引 |

shape `[1]` 在实际 Parquet 中可为标量。源 anchor 的整数时间及其他监督 mask 必须作为显式 feature 保留，不能在列选择时丢弃。

`cedf.ee_pose_spec`（prepared Parquet metadata）及最终 `meta/info.json` 的 pose feature 契约必须包含 EE 顺序、frame、来源、mapping revision、所用模型 revision/SHA-256，以及 `state_ee_pose_slice`、`relative_action_pose_slices` 和主向量宽度。
主 state 的 pose 切片必须等于独立 `observation.ee_pose`；主 action 切片必须符合声明的相对动作语义。必需 pose/来源/mask/时间字段由 exporter 保留，不能因 request 漏列而省略。

每个导出 episode 的每个 EE 都须至少有有效 state pose 和有效 action pose；部分缺测可 mask，整个 EE 任一侧完全无有效 pose 时阻塞。旧矩阵 pose 或缺目标 pose 的 prepared 数据必须重新生成，不能只改元数据。

### 3.2 Pose 来源、相对动作与坐标

- 优先使用有效记录 pose；无记录 pose 时，只能由对应侧关节通过已验证 FK 推导。state joints 生成 state pose，action joints 生成 action pose，不互相补值。已记录但无效的 pose 不能标成有效监督。
- joint→EE 路线要求 enabled mapping、正确 joint 顺序/单位/极性、固定模型 revision/SHA-256、base/tip 和适用范围。解除标定 blocker 还须独立 observed pose 对照，满足配置的最小样本量、平移 p95、旋转 p95 门槛；FK 结果不能自证。
- `origin` 决定读取各 EE 的 recorded 或 FK frame。相同 `base/world` 字符串不证明共同物理坐标系；未验证变换不能跨臂/跨 frame 相减。设备中心、wrist、工具点和物理 TCP 不得互换。
- 主 action 通常用 `inverse(T_state) × T_target`；声明为 `world_delta` 的路线只将平移改为相应参考系的位置差。沿用已验证 mapping，不因统一字段而更改约定。

### 3.3 动作标签模式

| `action_source` | 标签来源 | 条件 |
|---|---|---|
| `native_action`（默认） | 已有 action mapping | 保留真实源语义；HiFi 原 action 仍是 `derived_target`，不称为 executed command |
| `next_state` | 下一条真实观测的 pose 与夹爪/手部状态 | 明确 `derived_target`；主 action 仍为相对动作，绝对目标写入 `action.ee_pose` |

`next_state` 必须同时满足：

- 只取同一选中 episode/window 的下一条记录，不越界、不跳过无效点；下一条 anchor 的历史 state 匹配若仍引用同一时刻，该 EE 目标无效。
- 目标时差必须为正，并满足显式配置的 `max_gap_ns`；未配置仅表示没有额外 gap 上限。末行 `action_is_pad=true`，目标及 action mask=false。
- 保留 `action.target_row_index`（末尾/边界为-1）、`action.target_horizon_ns`、`action.ee_pose_timestamp_ns`、`action.effector_timestamp_ns`（无效为-1）。未来数据只作 target，不回填观测。
- `action_target_policy` 记录模式、算法、单位转换、终点策略、容差和原 action 语义；若存在原 action，保留 `action.native` 及 `action.native_valid_mask`，其含义是原 mapping 后的 action，不是 raw 全字段副本。
- GenRobot 只允许已验证 Ego+Finger profile 的 `next_state`，无原 command 不生成伪造的 `action.native`；pose 为设备 camera-center/base_link，不是机器人 TCP。

切换标签、速度或坐标模式必须重新 prepare 并生成 numeric revision；export request 只能核对一致性，不能替旧数据换标签。

## 4. 条件字段与转换

### 4.1 标量夹爪

已绑定的标量夹爪 state 归一化到 `[0,1]`，保留 `observation.gripper`、`observation.gripper_source`、`observation.gripper_valid_mask`、`observation.gripper_timestamp_ns`。

- 默认 `selection_minmax`：同 request、同 source 的全部选中行逐夹爪共用范围，`clip((x-min)/(max-min),0,1)`；不能逐 episode 拟合。0/1仅表示范围最小/最大，不自动表示闭/开。
- 常量夹爪输出0、保留原有效性并记录常量提示；缺测为 false mask。`fixed` 必须有有限且递增的范围及证据；缺该 EE 的范围则 mask。拟合范围和字段绑定写入 metadata；跨批次可比须复用固定校准，避免在验证/测试集重新拟合。
- 已知 packed state 夹爪切片同步替换，`observation.native_state` 不变；主 state 无此切片则不扩展前缀。`next_state` action 用下一条归一化观测；`native_action` 不改原指令值。
- 自定义 effector 绑定须声明字段/切片、scale/offset、证据及实际时间/有效性来源；宽度匹配 action 切片。省略独立时间仅在已确认与 state pose 共钟时允许。指定 validity 字段但字段缺失时必须 mask。

### 4.2 ActionNet 灵巧手

保留每手6维原生关节与单位，顺序为 `pinky, ring, middle, index, thumb_pitch, thumb_yaw`，不当标量夹爪归一化。
该来源 LeRobot 数值准备须绑定 hand-closure 校准配置、revision 和 SHA-256，并导出：

```text
observation.left_hand_closure    observation.right_hand_closure
action.left_hand_closure         action.right_hand_closure
以及上述四字段各自的 <字段名>_valid
```

closure 为 `float32` 标量，valid 为非空 bool；按固定端点计算 `z_j=clip((q_j-open_j)/(closed_j-open_j),0,1)`，再加权求和。
有效结果有限且位于 `[0,1]`；无效结果零占位并 mask。非零权重分量缺失即无效，零权重分量缺失不影响结果。
closure 只表示该校准下的开放/闭合程度，不替代6维动作或物理开口；不同校准不能混入同 family。`next_state` closure 同样使用下一条观测。

### 4.3 原生速度（默认关闭）

启用 `export_velocity` 后，仅导出所选标签来源实际存在的速度，字段为 `action.velocity`、`action.velocity_valid_mask`、`action.velocity_timestamp_ns`；metadata 的 `velocity_export.channels` 声明原字段/切片、单位、表示、frame、证据及按通道排列的时间。

`native_action` 用原 action 速度；`next_state` 用对应下一条 state 速度，继承末行/重复/gap mask。禁止有限差分、跨来源补值、将 joint velocity 当 EE twist、将底盘速度当手臂速度或随夹爪位置范围缩放。
全批无可用通道则省略字段并记录原因；部分 episode 缺测则保留 schema 并 mask。源单位/frame 未核实标为 `source_native_unspecified`；稀疏速度无明确 native index 时不得按行配对。

### 4.4 统一轴方向（默认保留源坐标）

`coordinate_system=canonical_axes` 启用 `pose-axes-flu-tool-x-v1`：参考系右手 X前/Y左/Z上；工具 X为接近方向、Y为有证据的夹爪正侧、Z=X×Y。保留原点与 pose 位置点，不补 TCP/shared-world 外参。

- profile 必须 `verified=true` 且 SHA-256 固定，绑定 source/revision、mapping、schema/硬件/工具和 episode 适用范围；每个 section/EE/origin 有源 frame、parent/point 身份及证据。模板改成 true 不是标定证据。
- 仅允许固定正交且 det=+1 的3×3旋转 `A,B`：`p_out=A p_source`、`R_out=A R_source B`。不接受镜像、平移、缩放或动态外参；state/target 的原点和位置点必须可兼容。
- profile 中 `state_poses`、`state_vectors`、`state_passthrough_indices` 必须无重叠地覆盖完整原生 state 前缀；未知空间字段不能冒充 passthrough。
- 同步转换独立 pose、主 state 和相对 action，保留 `observation.ee_pose_source_axes`、`action.ee_pose_source_axes`、`observation.state_source_axes`、`action_source_axes` 及对应 state mask；原生副本保持原语义。
- Cartesian 速度仅接受单位明确的3维 linear/angular 通道及 `vector_rotations`，保留 `action.velocity_source_axes`；部分分量缺测时整个3维向量无效。joint/gripper velocity 不按 XYZ 旋转。

## 5. 媒体条件（包含媒体时适用）

### 5.1 RGB

| 项目 | `video-serving-v1` 要求 |
|---|---|
| 容器/编码 | MP4、H.264 High、8-bit `yuv420p`、faststart；level 按分辨率/FPS确定并记录 |
| 默认 encoder | `libx264`、preset `fast`；wrist/hand CRF18，external/ego/head CRF20 |
| GOP | `g=max(1,round(native_fps))`，`keyint_min=g`，closed GOP，关闭 scenecut/open-gop |
| B帧 | 2，`b-pyramid=none` |
| 时间 | native rational FPS、passthrough；track timescale 90000，记录实际 time base 和源→输出 PTS 映射 |
| 分辨率 | 保持宽高比、不裁剪、不上采样、偶数宽高；长边≤1280保持原尺寸，否则按比例缩至1280，记录 Lanczos 等实际 filter/version |
| 色彩 | 保留/记录 color range、space、primaries、transfer；未知时规范默认 bt709/tv，不能将默认值称为观测证据 |
| 音轨 | camera MP4 不含音轨；音频不导出，见 5.2 |

合规输入优先 remux；不合规输入只能生成独立 derived 媒体。硬件 encoder 或不同 tuning 必须有独立 encoder revision、参数语义和完整质量/seek/decoder 验证，不能把 CQ/QP 写成 CRF。

- 只物化真实连续 frame window，保留 source FPS、frame count 和每帧时间对应。VFR 保留真实时间并标 `fps_mode=vfr`；目标格式不支持则留在原生 reader，不能强转 CFR。
- 完整解码检查 codec/profile/pix_fmt、FPS、分辨率、PTS 单调/重复/gap、帧数、duration、GOP/B帧、faststart、无音轨、decode error 和随机 seek。截断 decode 的结果不能成为最终路由或发布证据。
- 同一 camera feature 的拼接输入须一致：codec、profile、pix_fmt、分辨率、rational FPS、color metadata。LeRobot concat 只 copy packet，不再有损编码；拼接后再次核对帧数、offset 和解码。
- 共享视频保留 full-asset gate 和各 episode window reference；不能只验证一个窗口却为整个 shard 背书。MCAP 视频只接受已验证的 `foxglove.CompressedVideo` H.264/H.265 payload。
- 每个 asset/shard 记录源 URI/id/revision/checksum/size/mtime、episode/camera/role/attached EE、标定引用及已知范围、源编码/色彩/尺寸/FPS/time base/PTS/帧数；记录 encoder/build/config、缩放策略、job、plan/family/shard revision、输出 checksum、时间残差和验证结果。许可信息仅可选 provenance。

### 5.2 Depth、tactile、audio

Depth、tactile、audio 均不保存，任何输出都不包含这三类模态。

## 6. LeRobot family、文件与结构

### 6.1 Family 与计划

- 一次 export 只消费一个 homogeneous family：字段/dtype/shape、EE/手部布局、标签模式、frame/model family、夹爪/closure校准、坐标 profile、相机集合、各流 native FPS 与媒体兼容参数一致。license 不参与 family 划分。
- 当前实体 exporter 的 `fps` 为正整数；numeric `timestamp[i]` 须已满足 native anchor `i/fps`（容差0.0001秒），不是由 exporter 重采样得到。带视频时各 camera 的帧率/episode 帧数须与该 family 匹配。
- 因此异频 numeric/video、非整数 FPS 或不受支持 VFR 不能通过改 `info.fps`、复制/drop 帧强塞进当前实体路线；使用原生引用包或已验证的专用路线。action-anchor 中关联的 state 不代表独立 state 流的全量样本。
- Candidate 逐 episode 登记 source/revision、schema family、numeric bytes/rows、每 camera bytes/frames/native frequency，以及 mapping/repair/window revision。request 中 source/episode 不重复、task 非空、实际行数等于声明值，camera URI 集合与 features 完全相同。
- 当前实体 exporter 输入为绝对本地 `file://` URI；包含视频时绑定实际 media validation revision/evidence。必需输入均可读且 checksum 与证据一致。
- 使用分片计划时 request 的 dataset/view/plan/family 和 episode 集合须精确匹配；严格执行 planner 的 shard path、row/frame offset，不得改顺序或遗漏。分片 byte/row/frame 预算由 plan 声明，不是所有输出固定尺寸。

### 6.2 交付布局与一致性

```text
<family>/
├── data/chunk-XXX/file-XXX.parquet
├── videos/<video_key>/chunk-XXX/file-XXX.mp4   # 含视频时
├── meta/episodes/chunk-XXX/file-XXX.parquet
├── meta/tasks.parquet
├── meta/info.json
├── meta/stats.json
└── export_manifest.json
```

- `info.json` 声明 `codebase_version=v3.0`、真实 FPS、features 的 dtype/shape/语义、数据路径、episode/frame/task 总数和 split。当前空 `stats.json` 不代表已计算归一化统计；`train` 标签不代表无泄漏或专家认证。
- episode metadata 保存 length、数据 shard/全局起止范围、camera shard/起止时间，以及 source/mapping/normalization/repair/window/media lineage。
- 全局 `index` 连续，`frame_index` 在每 episode 从0连续编号；episode 编号连续。数据与视频范围不重叠、无遗漏，metadata 范围恰好覆盖真实 rows/frames，不跨 episode 边界。
- manifest 记录 dataset/view/plan/family/official revision、episode/frame 总数、video keys、实际文件 SHA-256、export revision，及 `raw_modified=false`、`interpolation_applied=false`、`resampling_applied=false`。不能写与实际执行不符的标记。
- 普通/分片导出均须保留所有已启用语义字段和 metadata；schema 不一致、pose/mask/时间异常、checksum 或 planner lineage 不匹配时拒绝交付，不覆盖旧输出。

## 7. 原生异频引用包

原生包使用已有完整 token index，保留每流独立频率，不生成稠密100/30Hz payload。

| 必需输入 | 条件 |
|---|---|
| Training View selection 与 gate state | 当前 revision、eligible；action_control 另有对应用途资格 |
| accepted intervals 与编译 summary | checksum/revision 匹配；每个窗口完全落在所有请求流的已接受区间 |
| native token index | 每流按原生时间有序、整数索引、布尔 mask，绑定唯一 source revision；覆盖范围明确 |
| stream evidence | catalog/token-index revision；source/episode/stream/clock 绑定；实际资产 URI/SHA-256 |
| training profile | 每流 role/kind、lookup、clock、时间网格和容差明确 |

输出为 `selection.parquet`、去重 `native_tokens.parquet`、`config.json`、`package.json`；manifest 绑定 inputs、gate/token-index/accepted-selection revision、文件 checksum、package revision 和实际覆盖范围，标明 `media_storage=external_reference`、`resampled=false`、`training_approval_created=false`。
当前 compiler 输入预算为 selection 最多10000行、token index 最多200万行、gate ledger 最多10万行；每窗口最多2000查询槽位，超限先分区。选择窗口按半开区间 `[start,end)` 保留真实 token。包依赖原 payload 可访问，不能称为脱离源挂载的媒体包。

reader 必须验证 package identity、预期 gate revision、文件和所引用 payload checksum；直接 Parquet 和明确 `kind=rgb_video` 的资产才走对应通用 resolver。archive member、未知媒体使用已验证专用 reader，不走 RGB 解码。

## 8. 训练读取条件

本节适用于引用包以及 LeRobot 导出之上的训练查询适配，不改变落盘频率。

- profile 可用默认数值100/1Hz、视觉30/1Hz查询网格，以整数纳秒有理数计算；它们是请求频率，不是实际观测频率。
- `exact` 仅命中同一时刻；`nearest_before_with_tolerance` 仅命中真实历史样本，`max_age_ns` 与 `max_gap_ns` 共同限制年龄，不能自动放宽。
- 返回 `query_time_ns`、`source_timestamp_ns`、`token_id`、`age_ns`、`valid`、`is_reused`、`fresh_mask`、`supervision_mask`；freshness 相对同一网格上一槽位定义，与 shuffle/worker/访问顺序无关。
- target 只有真实 token 首次命中的槽位可有监督；最终 loss mask 还须与 component mask 相交。重复引用不是新增观测；跨重叠窗口采样权重由训练配置负责。
- observation 不越过 cutoff，target 不跨 selection/window；相对 action 的 base-state 和真实 horizon 保留，不能将100ms标签解释成10ms目标。
- batch 不混合不兼容 export/字段 schema，padding 为 false mask。token-level valid 不替代原 payload 的逐分量 mask、动作来源、frame 或 horizon。
