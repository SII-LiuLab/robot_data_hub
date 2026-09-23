# AgiBot World 2026 转换

入口：`scripts/convert/agibotworld2026.py`。输出遵循[目标契约](export-contract.md)；源字段解释见[源数据契约](agibotworld2026-source.md)。

```bash
# 转换前 3 条轨迹到 processed；输出路径须尚不存在
python scripts/convert/agibotworld2026.py \
  --limit-episodes 3 \
  --output-dir dataset/processed/AgiBotWorld2026-preview

# 全部已下载归档
python scripts/convert/agibotworld2026.py
```

默认递归读取 `dataset/raw/AgiBotWorld2026/` 下全部 `.tar.gz`，输出到 `dataset/processed/AgiBotWorld2026/`。`--limit-episodes N` 限制跨归档的前 N 条轨迹；`--limit N` 限制前 N 个排序后的归档。`--model-dir` 默认 `assets/robot_models/agibot_g2/`。需要 `pyproject.toml` 中的 PyAV、PyArrow、NumPy 依赖以及 G2 模型资产。

EEF 通过 G2 URDF 对每一帧实测双臂 14 关节角与腰部 5 关节角做 FK。有底盘位姿时，结果再左乘源 `state/robot/position` 与 `state/robot/orientation` 给出的变换。`task_3401/399093_399454.tar.gz` 的整个归档缺少底盘位姿，按[源数据契约](agibotworld2026-source.md)中的静止底盘约定使用固定底盘参考系，并逐集核验底盘速度命令为零；缺位姿且命令非零时拒绝转换。模型是公开 G2 仿真 URDF，未完成对真实机器的 TCP/零位标定，因此绝对位姿仍受模型误差影响。

TCP 取模型 `arm_{side}_gripper_base_link` 坐标下 `[0, 0, 0.10547]` 米：这是两侧夹指闭合时末端 link 原点的中点。规范 EEF 三轴在该 link 下定义为 `+X=+Z_link`（接近方向）、`+Z=+X_link`（掌背方向）、`+Y=-Y_link`。左右手同一局部约定；姿态写为位置加旋转矩阵前两列。

所有 state 与 RGB 均使用源 Parquet 的 `timestamp` 列，转为纳秒并共享本集共同零点，不按视频 FPS 推算。每路 RGB 视频逐帧解码后用 libx264 编成无音轨 H.264；深度视频不导出。源视频每集每路必须与 Parquet 行数一致。`default` 指令分段按源帧边界映射到源时间戳；缺口用整集任务文本填充。

输出在目标目录旁暂存，全部成功后才发布，不覆盖已有目录。大归档转换会多次顺序读取压缩包；暂存目录需要容纳本次输出和单路源 MP4 的空间。
