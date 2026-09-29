# 并行转换

将各数据集的转换脚本（见 [`../sources/`](../sources/)）通过 Slurm 在单机或多机上执行。采用 **固定任务清单 + 静态分配 + episode 独立提交 + 自动 resume**。不引入 `torch.distributed`、动态队列、租约或在线抢占。

本文是待实现方案。现有转换 CLI 尚不支持这里的并行与 resume 协议。

## 1. 使用约定

- **重新提交同一个作业脚本即可 resume**，无需手动清理、合并或指定恢复位置。
- 每次运行可以改变节点数、worker 数和 CPU 预算；完成状态按 episode 保存，不与 worker 编号绑定。
- **同一输出同时只允许一个作业。** 重跑前，上一轮所有 worker 必须已经退出；不接管仍在运行的作业。
- 源数据保持只读且不变。resume 使用原任务清单、转换版本、模型和转换参数；不兼容的变更须使用新输出目录。
- 最终输出只在整集完成后出现；失败时保留工作目录。已完成的输出经核对属于同一计划后，再次执行直接成功返回；不接受无关的已有输出。

## 2. 固定任务清单与分配

作业启动时，由单个准备进程生成 `<output>.work/plan.json`，通过临时文件加原子 rename 发布；成功后才启动 workers。resume 读取并校验已有 plan，不重新发现任务或改动顺序。

plan 保存源数据身份、转换版本、模型与转换参数，以及有固定顺序的任务列表。每个任务记录输入位置和预期 episode ID；episode ID 必须在整个计划中唯一，且只能属于一个任务。筛选和数量限制在准备阶段确定，不由每个 worker 各自应用。节点数、worker 数和 CPU 预算不参与计划兼容性判断。

Slurm 每个 task 启动一个 worker，worker 读取：

- `SLURM_PROCID`：当前 worker 编号。
- `SLURM_NTASKS`：本轮 worker 总数。
- `SLURM_CPUS_PER_TASK`：当前 worker 的 CPU 预算。

按固定任务序号分配，不能先各自过滤未完成任务再重新编号：

```python
for index, task in enumerate(plan.tasks):
    if index % worker_count != worker_id:
        continue
    process_unfinished_episodes(task)
```

例如第一次使用 32 个 worker，第二次改为 8 个，则第二次按 `index % 8` 分配同一份清单，并跳过已提交的 episode。任务可以换 worker，完成状态不变。单机与多机使用相同流程。

静态分配可能因任务耗时不均而出现尾部等待，第一版接受这一代价。

## 3. 调度单位与提交单位

提交单位始终是 episode；调度单位根据源数据的读取方式确定：

| 数据源 | 调度单位 | resume 行为 |
|---|---|---|
| ABC-130K、MolmoAct2、HiFi-UMI | episode | 跳过已提交 episode |
| Galaxea | episode | 跳过已提交或已明确合法跳过的 episode |
| AgiBotWorld2026 | archive | 全部 episode 已完成则跳过 archive；否则重新扫描，跳过已提交 episode 的转换 |

AgiBot 的计划可从压缩包元数据枚举预期 episode。转换仍按包顺序读取，避免不同 worker 为同一个包反复解压；适配器须能在单个 episode 完整且校验通过后立即提交，而不是等待整个包完成。未完成的包在 resume 时可能重新扫描，但不重新转换已提交 episode。

复用现有 episode 转换逻辑，并由新的调度层负责 staging、提交和恢复；不能直接并发调用目前会拒绝已有输出目录的整集转换入口。

## 4. 工作目录与提交协议

工作目录、staging 与最终输出必须位于支持所需原子 rename 语义的同一共享文件系统：

```text
<output>.work/
├── plan.json
├── records/<episode_id>.json   # 内部恢复记录，包含 manifest 所需元数据或跳过原因
├── errors/<episode_id>.err     # 最近一次转换失败信息，仅用于诊断
├── staging/<attempt_id>/      # 每次尝试独立，不能当作已完成结果
└── dataset/
    └── episodes/<episode_id>/ # 已提交的完整 episode

<output>/                     # 整集完成后才发布
├── episodes.jsonl
└── episodes/<episode_id>/
```

records 属于工作目录，不进入最终数据集。最终结构遵守[目标格式契约](../contract/export-contract.md)，不增加 per-episode JSON。

每个 episode 按以下顺序处理：

1. 检查已有提交状态；已提交或已合法跳过则直接返回。
2. 在独立 staging 中完成转换、关闭所有输出文件，并校验 episode 内容及 manifest record。
3. 将含 `outcome: exported` 和完整 manifest record 的恢复记录写入临时文件，关闭后原子 rename 到 `records/<id>.json`。
4. 将 staging 中的完整 episode 目录原子 rename 到 `dataset/episodes/<id>`。**这一步是成功导出的提交点。** 不覆盖已提交目录。

不再写额外的 done 标记，也不再追加 worker manifest 分片。**有效的 exported record 与正式 episode 目录同时存在，才表示导出已提交。** 记录先于目录发布，保证已提交目录有对应元数据。

合法跳过采用 `outcome: skipped` 的 record，并记录原因；原子发布该 record 即完成跳过，不创建 episode 目录。转换异常不能作为合法跳过。

本协议处理进程退出、作业取消及节点故障后的续跑；依赖共享文件系统保留已完成的写入。存储系统故障导致的数据丢失不通过“目录存在”掩盖。

## 5. 自动 resume 与失败处理

准备阶段在确认上一轮 worker 已全部退出后清理遗留 staging 和未发布的临时记录；保留 plan、正式 records 和已提交 episode。每个 episode 按下表恢复：

| 状态 | 处理 |
|---|---|
| 有效 exported record + 正式 episode 目录 | 已提交，跳过转换 |
| 只有 exported record，无正式目录 | 未提交，重新转换；成功后更新 record 并提交目录 |
| 无 record、无正式目录 | 未开始或转换中断，重新转换 |
| 有效 skipped record，无正式目录 | 已合法跳过，无需再处理 |
| 正式目录缺少 record、record 损坏、skipped 却有目录等矛盾状态 | 报告状态损坏，停止，不静默跳过或覆盖 |

失败时记录 episode ID、输入位置和 traceback，继续处理本 worker 的后续任务；worker 结束时若有失败，返回非零退出码。失败记录不参与完成判断，下次运行自动重试未提交 episode；成功后清除对应旧错误记录。同一轮不无限重试。

已提交结果保持不变，未提交的 episode 从头转换，不做视频帧级断点恢复。普通进程中断最多损失当时尚未提交的转换工作。

## 6. 作业流程与最终发布

作业脚本自动串联三个阶段：

```text
prepare（单进程）
  → srun workers（K 个进程）
  → finalize（单进程）
```

prepare 失败则不启动 workers；srun 非零退出则作业失败，保留工作目录供下次 resume。srun 正常结束后才执行 finalize，不在 worker 之间设置 barrier。作业在任意阶段被取消后，都可重新提交同一脚本。

finalize 必须：

1. 按 plan 中的全部预期 episode 对账，每项必须是成功提交或合法跳过；发现未完成、矛盾或计划之外的记录与产物则失败。
2. 确认成功 records 与 episode 目录一一对应。内容校验在提交前完成；收尾不重复全量视频解码，也不预设其耗时为秒级。
3. 按固定顺序从成功 records 生成 `episodes.jsonl`，通过临时文件加原子 rename 写入 `dataset/`，每个成功 episode 恰好一行；合法跳过不写入 manifest。
4. 将整个 `<output>.work/dataset` 原子 rename 为 `<output>`，发布符合契约的最终数据集。

finalize 不移动单独的 episode，也不消费或删除 records，因此中途退出后可以重新执行。最终目录 rename 成功即表示发布完成；即使来不及打印成功信息，下次也能识别已完成结果。

发布后保留 plan 和 records，用于核对重复运行；不自动清理恢复依据。最终输出已存在时，prepare 核对计划、manifest 与目录的一致性，匹配则整次运行直接成功返回，不再启动 workers 或重新创建待发布数据集。

## 7. CPU 与内存预算

- 每个 worker 固定编解码线程配置，并设置 `OMP_NUM_THREADS`；不能仅依靠该环境变量限制视频编解码线程。
- `--cpus-per-task` 是整个 worker 的预算。ABC 会同时持有多个相机的编解码上下文，不能给每个上下文都无条件分配全部预算。
- 初始 worker 数按“节点数 × 每节点可用核数 / 每 worker CPU 预算”选择，并受峰值内存和共享盘吞吐约束。
- 从单机保守并发开始，观察总吞吐、峰值内存及共享盘负载，再扩到多机。转码型源与解码校验后直接复制视频的源分别压测。

Slurm 可从 `--nodes=2 --ntasks=32 --cpus-per-task=4` 起步，并使用 `srun --cpu-bind=cores`。这只是资源配置示例，实际并发度按节点资源和测量结果确定；resume 不要求沿用这些数值。

## 8. 验收

- 单 worker 与多 worker 得到相同的 episode 集合及符合契约的内容。
- 在 record 发布前后、episode 目录发布前后终止 worker，重跑后无遗漏、无重复 manifest 行，已提交 episode 不重新转换。
- 用 32 个 worker 中断后改为 8 个，再改为其他数量，仍能完成同一计划。
- 验证转换失败后重试、合法跳过、损坏状态报错以及计划不兼容时拒绝 resume。
- 在 manifest 生成及整集目录发布前后中断，重跑后正确收尾；已发布输出再次运行直接成功。
- 在实际共享文件系统上验证多节点可见性、提交操作和资源预算，再扩大任务规模。
