# Mun 着陆智能体

`mun-agent` 用于在六自由度 Mun 着陆模拟器中开发和训练策略。当前可从模拟器采集演示数据、拟合世界模型，并在独立回合上检查多步预测。模拟器的原生输入输出见[接口说明](../simulator/docs/INTERFACES.md)，后续训练采用的约定见[方法说明](docs/METHOD.md)。

## 安装与验证

从仓库根目录使用 Python 3.12–3.14 创建虚拟环境。

```sh
python3 -m venv .venv
source .venv/bin/activate
```

Linux CPU 环境先安装官方 CPU 构建。macOS arm64 可直接进入下一步，由 pip 从 PyPI 安装对应平台的构建。

```sh
python -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch==2.15.0'
```

安装本仓库的模拟器、智能体和测试依赖，然后运行导入测试。约束文件沿用仓库固定版本。

```sh
python -m pip install -c requirements/runtime.txt -e ./simulator -e './agent[dev]'
python -m pip check
python -m pytest agent/tests -q
```

## 采集模拟器回合

在仓库根目录运行以下命令。输出目录由命令创建，包含回放 `replay.npz` 和 `collection.json`。后者记录实际初态清单、哈希、每回合控制器与噪声档位。采集始终在 CPU 上运行。可用 `--episodes` 覆盖配置的回合数。

```sh
mun-agent collect --config agent/configs/debug.json --seed 17 --device cpu --out agent/data/m1-train
```

## 拟合世界模型并检查开环预测

M2 的 `train --data` 从已采集的回放离线训练世界模型。以下命令接续上面的采集示例。用另一个采集种子生成完整留出回合。报告命令会检查留出初态与训练初态是否重叠，并核对清单哈希、回放文件、模型规模和环境接口。

```sh
mun-agent collect --config agent/configs/debug.json --seed 18 --device cpu --out agent/data/m2-heldout
mun-agent train --config agent/configs/debug.json --seed 31 --device cpu --data agent/data/m1-train --out agent/runs/m2-debug
mun-agent wm-report --config agent/configs/debug.json --device cpu --data agent/data/m2-heldout --checkpoint agent/runs/m2-debug/world_model.pt --out agent/runs/m2-debug/wm-report.json
```

`debug.json` 用于 CPU 冒烟测试。`base.json` 使用 256 维确定性状态、16 组各 16 类的潜变量和 16×64 的回放批次。配置中的 `fit.updates` 为离线梯度更新次数，可在训练命令末尾添加 `--set fit.updates=500` 覆盖。模型训练和诊断支持 `auto|cpu|mps|cuda`，`auto` 依次选择 CUDA、MPS、CPU。

运行目录包含完整配置与代码版本、逐次更新的损失、拟合前后在同一训练回放探针批次上的确定性损失，以及可跨设备加载的模型检查点。报告在整份留出回放上使用确定性潜状态，枚举每个可用起点，`--seed` 不影响诊断值。正式策略验证和测试清单留待策略阶段使用。
