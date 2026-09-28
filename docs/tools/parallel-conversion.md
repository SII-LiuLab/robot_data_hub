# 并行转换

把各数据集的转换脚本（见 [`../sources/`](../sources/)）在多机上并行执行。转换本身是 **episode 级互相独立**的任务，没有通信，所以并行只做两件事：把 episode 分给多个进程，控制每台机器的 CPU 不超订。不引入 `torch.distributed` / NCCL / DDP，纯 CPU。

## 1. Slurm 给什么

作业通过 `sbatch` 提交，任务通过 `srun` 拉起。Slurm 负责把进程分布到多个节点，worker 只需要知道自己的全局编号：

- **`SLURM_PROCID`**：全局任务号（`0 .. SLURM_NTASKS-1`），即 worker id。worker 只依赖它。
- `SLURM_CPUS_PER_TASK`：每个任务的核数，在作业脚本里用它设 `OMP_NUM_THREADS`。
- 其余（`SLURM_LOCALID`、`SLURM_NODEID`、`SLURM_NNODES`、`SLURM_JOB_NODELIST`）本设计不需要。

## 2. 并行模型

Slurm 直接给到"每 worker 一个进程"，只有一层扇出：

```text
sbatch job.sh
  └── srun --ntasks=K            # Slurm 把 K 个任务分布到分配到的节点
        └── 每个 task = 一个 worker（用 SLURM_PROCID 区分）
              └── 从共享 FS 队列领取 episode → 转换 → 原子发布
```

- worker 之间没有主从关系，全部从同一个共享队列取任务，谁快谁多干。
- 全部 worker 的产物写到同一个共享输出目录。

## 3. 任务队列（共享文件系统）

不依赖任何中间件，队列就是输出目录下的一组文件：

```text
<output>/
├── queue/
│   ├── pending/<id>        # 待领取（启动时按 episode 清单生成）
│   ├── claims/<id>/         # 原子 mkdir 成功 = 已领取；内含 worker 与租约时间
│   ├── done/<id>            # 成功后写入 = 已完成，重跑时跳过
│   └── failed/<id>.err      # 失败记录（含 traceback），不中断其它 worker
├── episodes/<id>/           # 成品，原子发布
└── _shards/episodes.<procid>.jsonl   # 每个 worker 一份 manifest 片段
```

领取与发布规则：

- **领取**：对 `claims/<id>` 执行原子 `mkdir`，成功者独占该 episode。
- **续跑**：已存在 `done/<id>` 直接跳过；失败写 `failed/<id>.err` 后继续下一条。
- **掉线恢复**：claim 里写时间戳作为租约；超过租约仍无 `done` 的 claim 可被其它 worker 抢占。因为转换确定（同输入同输出），重复执行只会浪费算力，不会写坏结果。
- **发布**：worker 先写自己的 staging 目录，成功后 `rename` 到 `episodes/<id>`（同一共享盘上原子）。绝不发布半成品。

## 4. CPU 并发度

纯 CPU 转换的主要成本是视频解码与 libx264 编码，而 libx264 默认会按机器核数自动开线程。如果每个 worker 都让它自动开线程，会严重超订。做法是：

- **每个 worker 固定编码线程数**，与 Slurm 的 `--cpus-per-task` 对齐（例如 `4`），并设置 `OMP_NUM_THREADS`。
- 据此决定 worker 总数：`--ntasks ≈ 节点数 × 每节点核数 / 每任务线程数`。
- 核数多、单 episode 小的时候，小线程、多任务的吞吐通常更好；从保守值开始压测，观察总吞吐而不是单个 worker 的延迟。
- 共享盘上大量并发写 MP4 会争带宽；worker 数过高时先确认是 CPU 瓶颈还是 I/O 瓶颈。

一个示例作业：

```bash
#!/bin/bash
#SBATCH --job-name=convert-abc130k
#SBATCH -o /hpc_logs/slurm-%j.out
#SBATCH -e /hpc_logs/slurm-%j.err
#SBATCH --nodes=2
#SBATCH --ntasks=32            # 总 worker 数 = 节点数 × 每节点 worker 数
#SBATCH --cpus-per-task=4      # 每个 worker 的线程数
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
srun --cpu-bind=cores python -m scripts.convert.worker \
  --dataset abc130k --output /inspire/.../dataset/export/ABC-130K
```

## 5. 输出与收尾

- 源数据只读；输出只增。
- 所有 worker 结束后单独执行一次合并：校验 `queue/done` 与 `episodes/<id>` 一一对应、数量一致、符合[目标格式契约](../contract/export-contract.md)，把 `_shards/*.jsonl` 合并为唯一的 `episodes.jsonl`，然后清理 `queue/`。
- 合并是单进程、秒级操作，不要在作业里用跨进程 barrier 收口（任一 worker 挂掉会卡死）。

## 6. 规模无关

这套设计不关心节点数：单机就是 `--nodes=1`，多机就是 `--nodes>1`。节点数变化或作业重跑时，队列 + `done` 标记保证不漏不重。因此可以先用单机把整套流程验证通过，再直接扩到多机，代码不变。
