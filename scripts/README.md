# scripts 目录分类

脚本按“产物类型”分成三类，互不混用，与项目 README 的四条组织原则对应：

```
scripts/
├── download/   # 源数据选取与下载（每个源数据集一个入口）
│   ├── abc_subset.py
│   ├── agibotworld_subset.py
│   ├── galaxea_subset.py
│   ├── hifi_umi_subset.py
│   └── molmoact2_subset.py
├── convert/    # 转换到目标格式数据契约（每数据集一个入口 + 公共函数）
│   ├── abc130k.py
│   ├── agibotworld2026.py
│   ├── galaxea.py
│   ├── hifi_umi.py
│   ├── molmoact2.py
│   └── export_common.py
├── robot/      # 机器人模型工具（URDF 解析、FK、可视化、包校验）
    ├── urdf_model.py
    ├── kinematics.py
    ├── yam.py
    ├── fk.py
    ├── show_models.py
    └── check_packages.py
└── viewer/     # 仅消费目标格式契约的本地检查工具
```

- `download/`：只依赖各源数据集自身的契约，产出 `dataset/raw/` 与 `dataset/manifests/`。
- `convert/`：把源数据对齐到唯一的目标格式契约（见 [`../docs/export-contract.md`](../docs/export-contract.md)）。`export_common.py` 是跨数据集共用的 Parquet/视频写出逻辑。
- `robot/`：面向成品 URDF 模型包的工具。`urdf_model.py` 与 `kinematics.py` 是库，`fk.py`、`show_models.py`、`check_packages.py` 是命令行入口；`convert/` 也复用前两者。
- `viewer/`：读取导出数据契约的本地 viewer，不读取原始数据或机器人模型。用法见 [`../docs/export-viewer.md`](../docs/export-viewer.md)。

## 运行方式

入口脚本支持两种方式，均以项目根目录为工作目录：

```bash
python scripts/convert/abc130k.py
python -m scripts.convert.abc130k      # 等价
```

`scripts/` 及其子目录是命名空间包（无 `__init__.py`），子包之间用
`from scripts.robot.kinematics import fk_poses` 之类的绝对导入；入口脚本在直接执行时会
自行把项目根目录加入 `sys.path`，因此两种方式都可用。
