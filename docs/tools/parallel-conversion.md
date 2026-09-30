# 并行转换

将各数据集的转换脚本（见 [`../sources/`](../sources/)）通过 Slurm 在单机或多机上执行。采用 **固定任务清单 + 静态分配 + episode 独立提交 + 自动 resume**。不引入 `torch.distributed`、动态队列、租约或在线抢占。

统一入口是 [`scripts/convert/parallel.py`](../../scripts/convert/parallel.py)。原有各源转换 CLI 保留单次整集转换行为；需要并行和 resume 时使用本入口。

## 运行

在项目根目录激活已安装项目依赖的 Python 环境。单机运行：

```bash
python -m scripts.convert.parallel run \
  --dataset abc130k \
  --input-dir dataset/raw/ABC-130K \
  --output-dir dataset/export/ABC-130K \
  --workers 8
```

中断后执行同一命令，可将 `--workers 8` 改为其他数量。无需额外的 `--resume` 开关。程序会自动准备计划、启动子进程、对账并发布最终输出。

Slurm 调度由调用者控制。Python 提供三个独立阶段，自己不调用 `srun`，也不接收 Slurm 资源参数。按顺序执行以下三步：

**第 1 步：`prepare`，单独运行一次（单进程）。**

```bash
python -m scripts.convert.parallel prepare \
  --dataset abc130k \
  --input-dir dataset/raw/ABC-130K \
  --output-dir dataset/export/ABC-130K
```

**第 2 步：`worker`，由 `srun` 一次拉起全部 worker。** 

```bash
mkdir -p dataset/export/ABC-130K.work/logs
srun --output='dataset/export/ABC-130K.work/logs/slurm-%J-%t.out' \
  --error='dataset/export/ABC-130K.work/logs/slurm-%J-%t.err' \
  python -u -X faulthandler -m scripts.convert.parallel worker \
  --output-dir dataset/export/ABC-130K
```

保留原有 `srun` 资源参数；`%J` 是 job/step ID，`%t` 是 task ID，避免多个 worker 共写一个文件。日志目录必须提前建立且所有节点可访问。`srun` 默认不会单独生成 `.err`；若未显式设置 `--error`，stderr 会随标准输出走，使用 `sbatch` 时通常在 `slurm-<jobid>.out` 中。启动器自身的报错仍在调用 `srun` 的终端或批处理日志里。参见 [srun 输出选项](https://slurm.schedmd.com/srun.html)。

**第 3 步：`finalize`，单独运行一次（单进程）。**

```bash
python -m scripts.convert.parallel finalize \
  --output-dir dataset/export/ABC-130K
```

`prepare` 和 `finalize` 各执行一次，都是直接执行的单进程命令；`worker` 不单独运行，由 `srun` 按 `--ntasks` 拉起，每个 worker 读取 `SLURM_PROCID` 和 `SLURM_NTASKS` 决定分工。三步按顺序执行，前一步失败就不要执行下一步（也可以把三条命令用 `&&` 串起来，让 shell 保证这一点）。

各节点须能访问同一仓库、Python 环境、源数据和输出盘。上次所有 worker 退出后，重新执行这三步即可 resume；可更改 `srun` 的 worker 数。三个阶段都支持对已发布结果重复执行，不重新转换数据。单机 `run --workers N` 只是本地便利入口，不用于 Slurm。

`--dataset` 支持 `abc130k`、`molmoact2`、`hifi_umi`、`galaxea`、`agibotworld2026`。`--model-dir` 默认为仓库中对应的模型；Galaxea 的该参数指向包含两个本体模型包的父目录。

抽样参数在首次准备时固定，resume 不能更改：

- `--limit N`：前 N 个调度任务；AgiBot 为 archive，其余为 episode。
- `--limit-episodes N`：全局前 N 个候选 episode，包含合法跳过的 episode。它不同于原 Galaxea CLI 的“前 N 个成功导出”语义。
- `--robot-type` 仅用于 Galaxea；TCP 和夹爪参数见 `python -m scripts.convert.parallel run --help`。

转换失败返回非零退出码，episode 诊断保存在 `<output>.work/errors/`，worker 级诊断保存在 `<output>.work/logs/`。共享源数据必须保持不变；修复原始数据或修改转换配置后，使用新输出目录。

## 1. 使用约定

- **重新执行 prepare → worker → finalize 即可 resume**，无需手动清理文件或指定恢复位置；单机也可重跑 `run` 命令。
- 每次运行可以改变节点数、worker 数和 CPU 预算；完成状态按 episode 保存，不与 worker 编号绑定。
- **同一输出同时只允许一个作业。** 重跑前，上一轮所有 worker 必须已经退出；不接管仍在运行的作业。
- 源数据保持只读且不变。resume 使用原任务清单和转换参数；不兼容的配置变更须使用新输出目录。工具不校验转换代码与模型版本，改代码/模型后可以继续 resume，但需自行保证同一数据集由同一版转换逻辑产出。
- 最终输出只在整集完成后出现；失败时保留工作目录。已完成的输出经核对属于同一计划后，再次执行直接成功返回；不接受无关的已有输出。

## 2. 固定任务清单与分配

作业启动时，由单个准备进程生成 `<output>.work/plan.json`，通过临时文件加原子 rename 发布；成功后才启动 workers。resume 读取并校验已有 plan，不重新发现任务或改动顺序。

plan 保存源数据身份、转换参数，以及有固定顺序的任务列表。每个任务记录输入位置和预期 episode ID；episode ID 必须在整个计划中唯一，且只能属于一个任务。筛选和数量限制在准备阶段确定，不由每个 worker 各自应用。节点数、worker 数和 CPU 预算不参与计划兼容性判断。对源文件检查大小和修改时间，不全量散列视频。新增源 episode 不自动加入已有计划。

Slurm 每个 task 启动一个 worker。分配使用前两项环境变量，CPU 预算由 Slurm 绑定：

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
├── logs/worker-<attempt_id>.err # 每次 CLI worker 启动的诊断，重跑保留
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

不再写额外的 done 标记，也不再追加 worker manifest 分片。**有效的 exported record 与正式 episode 目录同时存在，且文件清单及大小匹配，才表示导出已提交。** 记录先于目录发布，保证已提交目录有对应元数据。

合法跳过采用 `outcome: skipped` 的 record，并记录原因；原子发布该 record 即完成跳过，不创建 episode 目录。转换异常不能作为合法跳过。

本协议处理进程退出、作业取消及节点故障后的续跑；依赖共享文件系统保留已完成的写入。存储系统故障导致的数据丢失不通过“目录存在”掩盖。

## 5. 自动 resume 与失败处理

在上一轮 worker 已全部退出的前提下，准备阶段清理遗留 staging 和未发布的临时记录；保留 plan、正式 records 和已提交 episode。每个 episode 按下表恢复：

| 状态 | 处理 |
|---|---|
| 有效 exported record + 正式 episode 目录 | 已提交，跳过转换 |
| 只有 exported record，无正式目录 | 未提交，重新转换；成功后更新 record 并提交目录 |
| 无 record、无正式目录 | 未开始或转换中断，重新转换 |
| 有效 skipped record，无正式目录 | 已合法跳过，无需再处理 |
| 正式目录缺少 record、record 损坏、skipped 却有目录等矛盾状态 | 报告状态损坏，停止，不静默跳过或覆盖 |

失败时记录 episode ID、输入位置和 traceback，继续处理本 worker 的后续任务；worker 结束时若有失败，返回非零退出码。失败记录不参与完成判断，下次运行自动重试未提交 episode；成功后清除对应旧错误记录。同一轮不无限重试。

CLI worker 在解析 Slurm 环境变量、加载依赖和 plan 之前建立独立日志，记录主机、PID、job/step/rank、当前阶段和任务输入。启动失败、输入变化、恢复状态损坏和 Python 中断均保留完整 traceback；正常返回时记录退出码。即使运行成功也保留此日志，`.err` 文件存在本身不代表失败。原生崩溃（如 SIGSEGV、SIGABRT）由 `faulthandler` 尽可能保存 Python 调用栈。

`errors/` 为空不能说明 worker 没有失败：其中仅记录 episode 转换异常。SIGKILL（包括部分 OOM 终止）、节点故障，以及 Python 启动前的错误无法靠 Python 异常处理写 traceback；SIGKILL 前已刷新的 worker 日志可帮助定位最后一个任务。若日志目录本身不可写，也只能从 stderr 或 Slurm 日志排查。Slurm 作业可在提交环境查询：

```bash
sacct -j JOB_ID --format=JobID,State,ExitCode,Elapsed,MaxRSS,NodeList
```

`ExitCode` 格式为 `退出码:信号`，结合 `OUT_OF_MEMORY`、`TIMEOUT`、`CANCELLED` 等状态判断；不要只凭一个非零码推断 OOM。参见 [Slurm 退出码](https://slurm.schedmd.com/job_exit_code.html)和 [sacct](https://slurm.schedmd.com/sacct.html)。

已提交结果保持不变，未提交的 episode 从头转换，不做视频帧级断点恢复。普通进程中断最多损失当时尚未提交的转换工作。

## 6. 作业流程与最终发布

三个阶段按顺序执行：**prepare 和 finalize 各自单独运行一次，都是直接执行的单进程 Python 命令；worker 不单独运行，而是由调用者用 `srun` 一次拉起 K 个进程。** 三者的调度与顺序都由调用者控制，Python 本身不调用 `srun`：

```text
1. prepare    单独运行一次（单进程，直接执行 Python 命令）
2. workers    用 srun 一次拉起 K 个进程（每个 Slurm task 一个 worker）
3. finalize   单独运行一次（单进程，直接执行 Python 命令）
```

prepare 失败则不启动 workers；srun 非零退出则作业失败，保留工作目录供下次 resume。srun 正常结束后才执行 finalize，不在 worker 之间设置 barrier。任意阶段被取消后，都可重新执行这三步。

finalize 必须：

1. 按 plan 中的全部预期 episode 对账，每项必须是成功提交或合法跳过；发现未完成、矛盾或计划之外的记录与产物则失败。
2. 确认成功 records 与 episode 目录一一对应。内容校验在提交前完成；收尾不重复全量视频解码，也不预设其耗时为秒级。
3. 按固定顺序从成功 records 生成 `episodes.jsonl`，通过临时文件加原子 rename 写入 `dataset/`，每个成功 episode 恰好一行；合法跳过不写入 manifest。
4. 将整个 `<output>.work/dataset` 原子 rename 为 `<output>`，发布符合契约的最终数据集。

finalize 不移动单独的 episode，也不消费或删除 records，因此中途退出后可以重新执行。最终目录 rename 成功即表示发布完成；即使来不及打印成功信息，下次也能识别已完成结果。

发布后保留 plan 和 records，用于核对重复运行；不自动清理恢复依据。最终输出已存在时，prepare 核对计划、manifest 与目录的一致性，匹配则成功返回，不重新创建待发布数据集；随后执行的 worker 和 finalize 也直接成功返回。

## 7. CPU 与内存预算

- 当前 worker 显式使用单线程编解码器，关闭 x264 异步 lookahead，并将 OpenMP、BLAS、Arrow 线程池限制为 1；吞吐主要通过 worker 数调节。原有单次转换 CLI 的线程默认值不变。
- `--cpus-per-task` 是整个 worker 的预算。ABC 会同时持有多个相机的编解码上下文，不能给每个上下文都无条件分配全部预算。
- 初始 worker 数按“节点数 × 每节点可用核数 / 每 worker CPU 预算”选择，并受峰值内存和共享盘吞吐约束。
- 从单机保守并发开始，观察总吞吐、峰值内存及共享盘负载，再扩到多机。转码型源与解码校验后直接复制视频的源分别压测。

Slurm 可从 `--nodes=2 --ntasks=32 --cpus-per-task=1` 起步，并使用 `srun --cpu-bind=cores`。这只是资源配置示例，实际并发度按节点资源和测量结果确定；resume 不要求沿用这些数值。

## 8. 验收

- 单 worker 与多 worker 得到相同的 episode 集合及符合契约的内容。
- 在 record 发布前后、episode 目录发布前后终止 worker，重跑后无遗漏、无重复 manifest 行，已提交 episode 不重新转换。
- 用 32 个 worker 中断后改为 8 个，再改为其他数量，仍能完成同一计划。
- 验证转换失败后重试、合法跳过、损坏状态报错以及计划不兼容时拒绝 resume。
- 在 manifest 生成及整集目录发布前后中断，重跑后正确收尾；已发布输出再次运行直接成功。
- 在实际共享文件系统上验证多节点可见性、提交操作和资源预算，再扩大任务规模。
