# KSP
World Models for Lunar Landing in Kerbal Space Program

世界模型课程项目，目标是在 **Mun 动力下降着陆**任务中研究预训练、想象中的策略更新和物理参数泛化。当前仓库完成环境准备与诊断工具整理，尚未实现着陆训练环境或训练算法。

## Proposal 对应的研究问题

- **RQ1：数据效率。** 预训练能否减少达到固定安全着陆率所需的 KSP 交互数据？
- **RQ2：想象更新。** 在完全相同的 KSP 数据集上，imagined policy updates 是否带来额外收益？
- **RQ3：泛化。** 在不追加训练的条件下，如何应对质量、惯量、推力和重力变化？

指标分别记录安全着陆率、目标位置误差和推进剂消耗。详见 [proposal 解读](docs/proposal.md) 与 [环境要求](docs/environment.md)。

## 已有内容

- KSP 1.12.5 + kRPC 0.7.0 的只读诊断、遥测记录及显式灯光控制测试。
- Python 3.12–3.14 环境配置，固定版本的 NumPy、SciPy、Gymnasium、PyTorch、Matplotlib。
- 完全离线的依赖、CPU/MPS 运算和本地插件文件检查。
- Mun 任务配置草案、reset/step 需求、OOD 与数据预算规范。
- GitHub Actions 运行不依赖游戏的诊断规则测试与语法检查。

2026-11-06 已在本机验证发射台 RPC、连续遥测和可恢复灯光控制。这些证据只覆盖程序控制链路。**Mun 初始下降场景、可重复 reset/step、着陆判据、预训练与 RQ1–RQ3 实验仍待实现。** [验证记录](docs/validation.md) 分开列出本次与历史结果。

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
configs/       Mun 任务草案；未确定的数值选择保持空缺
docs/          proposal 对照、环境要求、验证记录与开发顺序
requirements/  直接依赖、开发依赖、本机完整版本快照
scripts/       环境安装、离线检查、游戏连接诊断、遥测
tests/         不依赖 KSP 的检查规则测试
```

生成的 `reports/`、数据、模型权重、虚拟环境和游戏文件均由 `.gitignore` 排除。原 proposal 含图片的 PPTX 留在本地，只记录文字需求及源文件摘要。仓库不附带 KSP 游戏或第三方模组二进制。

下一步是固定着陆器和 Mun 下降快照，再验收 `reset/step`；详见 [开发顺序](docs/roadmap.md)。仓库尚未指定开源许可证。
