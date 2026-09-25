# MolmoAct2-BimanualYAM 转换

入口：`scripts/convert/molmoact2.py`。读取本项目下载器生成的 episode 子集，
写出[统一导出结构](export-contract.md)。字段依据见[源说明](molmoact2-source.md)。

```bash
pip install -e .
python scripts/convert/molmoact2.py
# 先转换一条；输出目录必须尚不存在
python scripts/convert/molmoact2.py --limit 1 --output-dir dataset/processed/MolmoAct2-preview
```

默认输入 `dataset/raw/MolmoAct2-BimanualYAM/`，默认输出
`dataset/processed/MolmoAct2-BimanualYAM/`。
`--input-dir` 指向包含 `source/meta/` 和 `episodes/` 的下载根目录；
`--model-dir` 默认 `assets/robot_models/yam/`，资产安装见项目 README。
也可运行 `python -m scripts.convert.molmoact2`。
本入口不处理原仓库尚未分割的多 episode AV1 视频。

## 转换规则

- 仅使用 `observation.state`；不读取 action，也不输出关节、速度等额外数据。
- 六个实测关节角按 rad 输入 YAM FK；左右臂分别以自身固定基座为参考系，
  不套用 ABC 的双臂安装外参。工具中心和轴向与 ABC 共用
  `scripts/robot/yam.py` 的实现，按源说明的固定变换生成位置加列式 rotation6D。
- 夹爪已有归一化：有限值裁剪到 `[0,1]`，范围内透传；NaN、无穷时报错。
- 直接将表内 `timestamp` 四舍五入成 int64 纳秒，减去共同首时刻。
  四路 state 和三相机索引沿用同一记录时间序列，不插值、补帧、排序或按 FPS 重建。
  **源仅保留 LeRobot 记录时间，无法恢复硬件采集时刻及各传感器异步节奏**；
  输出保留源所能提供的时间精度，不能视作独立硬件时间戳的测量结果。
- `top → top`、`left → left_wrist`、`right → right_wrist`。
  每路本地 MP4 必须为单路无音轨 H.264，逐帧解码确认分辨率稳定、帧数与表行数一致，
  然后按字节复制，无需再次有损编码或 GPU。容器 PTS 和拼接片段起点不参与输出时间计算。
- 逐 episode 有效详细标注优先，覆盖 `[0, 最后记录时刻]`；缺失、null、空白标注回退到
  标准任务表，按逐行 `task_index` 的文本变化生成不重叠区间。
  相同时间最后一次变化生效，末采样时刻的零长度区间不写出。
  以 `source/meta/` 的任务表为准，不使用 `episode.json.task` 缓存覆盖它。

## 校验和发布

校验本体、版本、14 维状态字段名称及顺序、三路视频特征；
校验 episode 目录 ID、元数据长度、全局索引区间、逐行 episode ID、连续帧序和索引。
时间非有限、倒序、负值或整集零时长，状态维度错误、非有限数，任务索引未知、
任务表重复 ID、相机缺失或视频帧数不符均报错。

不覆盖已有输出。先在输出目录旁暂存，所有 episode 成功后才整体发布；
失败清理暂存，不留下已发布的半成品或缺少元数据的 episode。

```bash
python -m unittest discover -s tests -p 'test_convert_molmoact2.py'
python scripts/viewer/export_viewer.py dataset/processed/MolmoAct2-BimanualYAM
```

测试覆盖非零共同原点和不规则时间间隔、左右臂 FK/工具轴、夹爪裁剪、
标注优先和任务回退、视频复制与帧序、输入损坏及失败清理；YAM 几何测试需安装资产。
