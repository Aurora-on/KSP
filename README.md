# KSP
World Models for Lunar Landing in Kerbal Space Program

世界模型课程项目，目标是在 **Mun 动力下降着陆**任务中研究预训练、想象中的策略更新和物理参数泛化。当前仓库记录 final proposal 的任务约定与实验协议，并完成环境准备和诊断工具整理；尚未实现着陆训练环境或训练算法。

## Proposal 对应的研究问题

- **RQ1：数据效率。** 预训练能否减少达到固定安全着陆率所需的 KSP 交互数据？
- **RQ2：想象更新。** 在完全相同的 KSP 数据集上，imagined policy updates 是否带来额外收益？
- **RQ3：泛化。** 在不追加训练的条件下，如何应对质量、惯量、推力和重力变化？

指标分别记录安全着陆率、精度、目标位置误差和推进剂消耗。详见 [proposal 解读](docs/proposal.md)、[任务契约](docs/task-spec.md)、[实验协议](docs/experiment-protocol.md) 与 [环境要求](docs/environment.md)。

## 已有内容

- KSP 1.12.5 + kRPC 0.7.0 的只读诊断、遥测记录及显式灯光控制测试。
- Python 3.12–3.14 环境配置，固定版本的 NumPy、SciPy、Gymnasium、PyTorch、Matplotlib。
- 完全离线的依赖、CPU/MPS 运算和本地插件文件检查。
- Mun 任务定性契约：固定 nominal 着陆器、重力和平坦站点，随机初始位置/速度/姿态，RCS + 主发动机控制，5 s 稳定性窗口；数值阈值保持空缺。
- 2×2 预训练/想象实验协议、RQ1–RQ3 评估口径、OOD 与数据预算规范。
- GitHub Actions 运行不依赖游戏的诊断规则测试与语法检查。

2026-11-06 已在本机验证发射台 RPC、连续遥测和可恢复灯光控制。2026-11-17 审阅 final proposal，并记录任务定性决策与实验协议。这些证据只覆盖程序控制链路和设计约定。**Mun 初始下降场景、可重复 reset/step、着陆判据、预训练与 RQ1–RQ3 实验仍待实现。** [验证记录](docs/validation.md) 分开列出本次与历史结果。

## 快速开始

先独立安装 KSP 和匹配版本的 kRPC，步骤见 [环境文档](docs/environment.md)。然后：

```sh
git clone git@github.com:Aurora-on/KSP.git
cd KSP
bash scripts/bootstrap.sh
```

单独检查科学计算环境，不启动游戏、不发送控制命令：

```sh
.venv/bin/python scripts/check_environment.py --output reports/environment.json
```

手动启动 KSP 并进入存档，在 kRPC 窗口确认 `127.0.0.1:50000/50001` 已启动：

```sh
.venv/bin/python scripts/doctor.py --output reports/live-check.json
```

飞船进入 Flight 场景（发射台或跑道也算）后，可读遥测，或显式运行可恢复控制测试：

```sh
.venv/bin/python scripts/telemetry.py --seconds 10 --output reports/telemetry.csv
.venv/bin/python scripts/doctor.py --control-test --output reports/control-check.json
```

`--control-test` 仅切换灯光并恢复原值。CSV 记录最近收到的流值，不提供同步训练步进。采集期间保持同一场景和飞船。

## 仓库结构

```text
configs/       Mun 任务契约；定性决策已记录，未确定的数值保持 TBD
docs/          proposal 对照、任务契约、实验协议、环境要求、验证记录与开发顺序
requirements/  直接依赖、开发依赖、本机完整版本快照
scripts/       环境安装、离线检查、游戏连接诊断、遥测
tests/         不依赖 KSP 的检查规则测试
```

生成的 `reports/`、数据、模型权重、虚拟环境和游戏文件均由 `.gitignore` 排除。原 proposal 与 final proposal 的 PPTX 留在本地 `downloads/proposals/`（已被 gitignore），只记录文字需求及源文件摘要。仓库不附带 KSP 游戏或第三方模组二进制。

## 团队分工（初始）

以下为初始分工，按里程碑调整；任务定义和接口以 [任务契约](docs/task-spec.md)、[实验协议](docs/experiment-protocol.md) 和 [开发顺序](docs/roadmap.md) 为准。

| 角色 | 负责范围 | 主要交付物 |
| --- | --- | --- |
| 环境与集成 | 任务契约、KSP 适配、reset/step、脚本着陆、数据采集 | `configs/mun_landing.toml`、KSP 环境、transition bank |
| 模拟器 | 6DOF 动力学、变质量/推力/重力、与 KSP 对齐 | 预训练模拟器、对齐报告 |
| 学习与评估 | 数据与指标、评估 harness、2×2 实验、世界模型与策略 | 数据管线、评估结果、训练与实验 |
| 研究记录与可视化 | 文档一致性、图表、实验日志、报告与答辩材料 | 图表、实验记录、报告/PPT |

协作约定：

- KSP 实时操作由环境与集成负责人统一安排，避免并发控制。
- 观测、动作或数据格式变更时，先更新契约文档，再改代码。
- 每两周按开发顺序中的验收节点做一次集成检查。

下一步是补齐任务契约中的 TBD 数值、固定着陆器和 Mun 下降快照，再验收 `reset/step`；详见 [任务契约](docs/task-spec.md) 与 [开发顺序](docs/roadmap.md)。仓库尚未指定开源许可证。
