# 导出数据异常检测

入口 [`../../scripts/check/anomaly_detection.py`](../../scripts/check/anomaly_detection.py) 只消费
[目标格式契约](../contract/export-contract.md)，实现 [三条异常规则](../contract/anomaly_detection.md)。

在项目根目录运行：

```bash
python scripts/check/anomaly_detection.py dataset/processed/ABC-130K
# 等价的模块入口；可指定输出位置及阈值
python -m scripts.check.anomaly_detection dataset/processed/ABC-130K \
  --output /tmp/abc-anomalies.jsonl \
  --max-boundary-idle-s 4 \
  --max-internal-idle-s 12
```

默认输出 `<dataset>/anomalies.jsonl`，每行只有 `index`、`stream`、`start_ns`、`end_ns`、`reason`。
`index` 是 `episodes.jsonl` 的物理行号（从 0 开始；跳过空行但仍计入行号）。
没有异常时输出空文件。重复运行替换报告；全量检测成功才原子替换，失败保留之前的报告并返回非零退出码。
输出必须是 `.jsonl`，不能覆盖 `episodes.jsonl` 或写入 `episodes/`。

逐 episode 读取四路 state 和相机 Parquet 索引，不解码视频。各流保留原生时间戳与采样数；
episode 起点为 0，终点取四路 state 和所有相机索引的最大时间戳，instruction 不参与确定终点。
无效数值、空流、负时间戳、时间倒退或无效 rotation6D 会报错；重复时间戳合法，突变规则跳过相邻时间差为 0 的样本。
本工具不替代导出格式完整性校验，例如不会校验 MP4 内容。

记录按 episode 行号、规则顺序（首尾静止、突变、中段静止）输出，同一规则按流与时间顺序输出，
中段静止按区间再按流输出。流顺序固定为 `left_eef`、`right_eef`、`left_gripper`、`right_gripper`；
首尾静止若多路并列，取此顺序中的第一路。整段静止且超时仍按契约输出两条首尾记录。
不同规则的记录不去重；位置与朝向同时突变的同一个样本只输出一条 `state_jump`。

所有阈值都可通过命令行覆盖：

| 参数 | 默认值 | 单位 / 含义 |
|---|---|---|
| `--position-tolerance-m` | 0.01 | 米，首尾与中段静止共用 |
| `--orientation-tolerance-deg` | 5 | 度，首尾与中段静止共用 |
| `--openness-tolerance` | 0.05 | 开合量，首尾与中段静止共用 |
| `--max-boundary-idle-s` | 3 | 秒 |
| `--position-acceleration-limit` | 100 | m/s² |
| `--orientation-acceleration-limit` | 20000 | °/s² |
| `--openness-acceleration-limit` | 500 | 1/s² |
| `--max-internal-idle-s` | 10 | 秒 |

所有阈值必须有限且非负；规则采用严格大于，等于阈值不报异常。
无需额外依赖，使用项目已有的 NumPy 与 PyArrow。

测试：

```bash
python -m unittest discover -s tests -p 'test_anomaly_detection.py'
```
