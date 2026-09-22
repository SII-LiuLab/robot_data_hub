# ABC-130K 转换

入口：`scripts/convert/abc130k.py`。输出遵循 [目标契约 v2.0](export-contract.md)。

```bash
pip install -e .
python scripts/convert/abc130k.py
# 先转换一条；输出目录必须尚不存在
python scripts/convert/abc130k.py --limit 1 --output-dir dataset/export/ABC-130K-preview
```

默认递归读取 `dataset/raw/ABC-130K/` 内的 `episode.mcap`，输出到
`dataset/export/ABC-130K/`。可用 `--input-dir` 指定源根目录或单个 episode 目录，
`--model-dir` 指定 YAM 模型包（默认 `assets/robot_models/yam/`）。
模型资产的下载方式见项目 README。每条 episode 名称必须唯一。

CLI 默认使用 `--video-decoder nvdec --video-encoder nvenc`，需要可访问的
NVIDIA GPU、驱动和 PyAV 18.1 以上；初始化失败时直接报错，不静默回退。
NVDEC 输出保留在显存（CUDA frame），NVENC 直接使用第一帧携带的硬件帧
上下文；每帧检查 CUDA 格式，不执行主存下载、像素重排或重新上传。
MCAP 读取、时间戳索引与 MP4 封装仍在 CPU 上执行。

NVENC 输出 H.264，使用 preset p4、VBR、CQ 18、自动目标码率，关闭 B 帧、
lookahead 和额外输出延迟。CQ 18 与 x264 CRF 18 不是等价质量参数，文件大小
和图像质量可能不同。保留两条对照通路：

```bash
# NVDEC 解码后下载到主存，再用 libx264 编码
python scripts/convert/abc130k.py --video-decoder nvdec --video-encoder libx264
# 全 CPU
python scripts/convert/abc130k.py --video-decoder cpu --video-encoder libx264
```

NVENC 通路要求 NVDEC，不支持静默改为 CPU 解码后上传。

## 关节 FK 与工具坐标系

仅使用 `/{left,right}-arm-state.position` 六维实测关节角，依次映射到模型
`arms[side].joints`，以弧度输入 URDF FK；不读取原始 `pose`，不使用 action。
参考系为模型的固定 `base_link`；左右臂在自己的原生时间戳分别计算，无需同步另一臂。

TCP 使用模型 `arm_{side}_grasp` 的位置，即 `link6` 的 `[0,0,0.1347]` 米。
该模型的同名 `tcp` site 位于腕部原点，因此这里选择抓取中心 `grasp`。
导出 X 为 `+Z_link6`（工具接近方向），Y 为 `+X_link6`（朝 finger2 一侧，沿夹爪开合轴），
Z 为 `+Y_link6`，满足 X×Y=Z。模型 grasp site 的旋转右乘：

```text
[ 0  0 -1 ]
[ 0  1  0 ]
[ 1  0  0 ]
```

结果写为位置三维加旋转矩阵前两列。以上采用模型几何与关节定义，
并不表示已完成实物零位标定；不需要提供从原始 pose 到 TCP 的变换配置。

## 时间、视频与指令

- 四路 state 和所有相机共享最早采样时刻为零点；不插值、不重采样。
- 夹爪实测开合裁剪到 `[0,1]`：小于 0 置为 0，大于 1 置为 1，范围内保持原值。
  NaN、正负无穷等非有限值仍在视频转码前报错。本地样本中存在少量超过 1 的夹爪读数
  （例如 `episode_e18a509f-5940-49ca-815b-a27ebfdcd4b1` 右夹爪最大
  `1.0023205498338077`），导出时按上述规则裁剪，不修改原始数据。
- 按各相机消息的 `format` 解码 H.264/H.265，逐帧编码成无音轨 H.264 MP4。
  用解码 PTS 将输出显示帧映射回输入消息；校验每个输入样本恰好对应一帧。
  MP4 使用 30 fps 容器时序，实际采集时刻只取 Parquet 索引。
- 无子任务时，整集任务覆盖 `[0, 最后采样时刻]`。
  有 `annotation.mcap` 时，子任务从标注时刻开始生效并替代整集指令，
  第一条子任务前保留整集任务；区间裁剪到采样范围，同一时刻最后一条标注生效。
- 校验 Protobuf 时间戳与 MCAP `log_time` 一致。缺失 state、相机或任务文本时报错。
- 在输出目录旁暂存，全部成功后才发布整个数据集；失败自动清理暂存目录。
  不覆盖已有输出目录。暂存需要容纳本次完整导出的空间。

公共 Parquet/视频写出逻辑位于 `scripts/convert/export_common.py`；FK 复用
`scripts/robot/kinematics.py`，源字段映射与 YAM 工具坐标系约定位于转换入口。
