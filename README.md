# KSP
World Models for Lunar Landing in Kerbal Space Program

世界模型课程项目，目标是在 **Mun 动力下降着陆**任务中研究预训练、想象中的策略更新和物理参数泛化。仓库现已包含可独立运行的六自由度预训练模拟器、任务约定和 KSP 诊断工具；KSP 着陆环境对齐及世界模型/策略训练仍待完成。

## Proposal 对应的研究问题

- **RQ1：数据效率。** 预训练能否减少达到固定安全着陆率所需的 KSP 交互数据？
- **RQ2：想象更新。** 在完全相同的 KSP 数据集上，imagined policy updates 是否带来额外收益？
- **RQ3：泛化。** 在不追加训练的条件下，如何应对质量、惯量、推力和重力变化？

指标分别记录安全着陆率、精度、目标位置误差和推进剂消耗。详见 [proposal 解读](docs/proposal.md)、[任务契约](docs/task-spec.md)、[实验协议](docs/experiment-protocol.md) 与 [环境要求](docs/environment.md)。

## 已有内容

- 智能体支持模拟器演示采集、回合回放、RSSM 离线训练与开环诊断。[安装与使用说明](agent/README.md)
- [六自由度着陆模拟器](simulator/README.md)：变质量动力学、RCS 姿态控制、Gymnasium 接口、脚本控制器、可复现初态与参数扫描；仓库版稳定窗口为 5 s。
- KSP 1.12.5 + kRPC 0.7.0 的只读诊断、遥测记录及显式灯光控制测试。
- `ksp_env/` 可导入适配器骨架：连接、配置、诊断状态读取；任务观测、reset/step 仍受 TBD 决策阻塞，详见 [KSP 环境接口](docs/ksp-env.md)。
- Python 3.12–3.14 环境配置，固定版本的 NumPy、SciPy、Gymnasium、PyTorch、Matplotlib。
- 完全离线的依赖、CPU/MPS 运算和本地插件文件检查。
- Mun 任务定性契约：固定 nominal 着陆器、重力和平坦站点，随机初始位置/速度/姿态，RCS + 主发动机控制，5 s 稳定性窗口；数值阈值保持空缺。
- 2×2 预训练/想象实验协议、RQ1–RQ3 评估口径、OOD 与数据预算规范。
- GitHub Actions 在 Python 3.12 和 3.14 上检查语法、离线诊断和模拟器，并用固定版本的 CPU PyTorch 验证智能体安装与测试。

2026-11-06 已在本机验证发射台 RPC、连续遥测和可恢复灯光控制；2026-11-17 记录 final proposal 的任务定性决策与实验协议。模拟器的独立验收见 [模拟器验证说明](simulator/docs/VALIDATION.md)，既有 KSP 证据见 [验证记录](docs/validation.md)。**Mun 初始下降场景、KSP 可重复 reset/step、实船安全阈值、两端接口校准、预训练与 RQ1–RQ3 实验仍待完成。** 模拟器的 pilot 数值没有填入共享任务配置中的 TBD；动作顺序、轴映射和精度指标差异见 [仓库接入说明](simulator/docs/REPOSITORY_INTEGRATION.md)。

## 快速开始

模拟器可以离线运行，无需启动 KSP：

```sh
git clone git@github.com:Aurora-on/KSP.git
cd KSP/simulator
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev,demo]'
.venv/bin/python -m pytest -q
.venv/bin/mun-lander demo --seed 7 --output artifacts/demo
```

运行 `.venv/bin/python examples/reproduce_acceptance.py` 可在本地重建验收数据与轨迹；这些输出写入被忽略的 `simulator/artifacts/`。仓库提供 [着陆动画](simulator/docs/assets/landing.gif)、[轨迹图](simulator/docs/assets/trajectory.png) 和 [验证摘要](simulator/docs/validation-summary.json)，不存放原始数据集、轨迹或机器测试日志。

连接游戏前，先独立安装 KSP 和匹配版本的 kRPC，步骤见 [环境文档](docs/environment.md)。在仓库根目录执行：

```sh
bash scripts/bootstrap.sh
```

当前 Windows + WSL2 主机没有 `ensurepip`，改用 `uv` 建立 KSP 接口环境：

```sh
uv venv --python 3.14 .venv
uv pip install --python .venv/bin/python \
  krpc==0.7.0 protobuf==7.36.2 numpy==2.5.4 gymnasium==1.4.0 matplotlib==3.11.2
```

如需完整科学计算栈（含 SciPy/PyTorch），使用 `requirements/dev.txt`。不跑训练时不需要 PyTorch。

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
agent/         模拟器演示采集、回放、RSSM 拟合与开环诊断
configs/       Mun 任务契约（数值 TBD）与 adapter 测试配置（非任务）
docs/          proposal 对照、任务契约、实验协议、环境要求、验证记录与开发顺序
ksp_env/       可导入的 KSP 环境接口（连接、观测、动作、reset/step 骨架）
requirements/  直接依赖、开发依赖、本机完整版本快照
scripts/       环境安装、离线检查、游戏连接诊断、遥测、控制响应探测
tests/         不依赖 KSP 的检查规则测试
simulator/     可独立安装的 6DOF 预训练模拟器、测试、接口文档与验收复现脚本
```

生成的 `reports/`、`simulator/artifacts/`、数据、模型权重、虚拟环境和游戏文件均由 `.gitignore` 排除。原 proposal 与 final proposal 的 PPTX 留在本地 `downloads/proposals/`（已被 gitignore），只记录文字需求及源文件摘要。仓库不附带 KSP 游戏或第三方模组二进制。

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

下一步是固定实船和 Mun 下降快照，记录 TBD 数值的决策，验收 KSP `reset/step`，并按共同转移数据校准模拟器；详见 [任务契约](docs/task-spec.md)、[开发顺序](docs/roadmap.md) 与 [仓库接入说明](simulator/docs/REPOSITORY_INTEGRATION.md)。仓库尚未指定开源许可证。

