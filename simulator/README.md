# Mun 六自由度着陆模拟器

这是 Aurora-on/KSP 仓库中对应小组 final proposal 的**模拟器部分**，供世界模型和策略预训练使用。实现平地动力下降、主发动机耗油与质量变化、机体系 RCS 姿态控制、触地与持续稳定判定，并提供 Gymnasium 接口、可复现初态、脚本控制器和评估命令。

本项目不包含 Dreamer/SAC 训练或真实 KSP/kRPC 连接。默认飞行器参数是可运行的工程起点，尚未依据队伍实际 KSP 飞行器标定；源模拟器结果不代表 KSP 迁移效果。

仓库版本默认连续稳定 **5 s**，与 [共享任务契约](../docs/task-spec.md) 一致。此前本地独立交付版的 2 s 默认及其结果作为历史记录保留；两者不能混用。其他 pilot 数值、原生动作顺序与精度统计尚需对齐，详见 [仓库接入说明](docs/REPOSITORY_INTEGRATION.md)。

## 已完成的验收

仓库版的测试、名义脚本着陆、失败路径、独立参数扫描和性能测量见 [验收说明](docs/VALIDATION.md) 与 [验证摘要](docs/validation-summary.json)。可直接查看 [着陆动画](docs/assets/landing.gif) 和 [轨迹图](docs/assets/trajectory.png)。这些记录只评价本模拟器及其脚本控制器，不是学习算法或 KSP 迁移结果。

## 安装与快速运行

需要 Python 3.12–3.14。从仓库根目录进入 `simulator/` 后执行：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,demo]'
pytest -q
mun-lander demo --seed 7 --output artifacts/demo
```

`demo` 保存轨迹数据、图像、着陆动画和本次结果。模拟器本体只依赖 NumPy 与 Gymnasium；绘图和动画使用可选 `demo` 依赖。

[requirements-tested.txt](requirements-tested.txt) 记录本次通过验证的直接依赖版本，可用 `python -m pip install -r requirements-tested.txt` 安装；它不是包含全部间接依赖的完整锁文件。

最小环境调用：

```python
import numpy as np
from mun_lander import LanderEnv

env = LanderEnv()
observation, info = env.reset(seed=0)
while True:
    action = np.array([0.3, 0.0, 0.0, 0.0], dtype=np.float32)
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        print(info["status"], info["is_success"], info["total_fuel_used"])
        break
env.close()
```

这里的恒定动作仅演示 API；可着陆的工程参考见 `ScriptedController` 和 `demo`。

## 可复现评估

重建完整验收数据：

```bash
python examples/reproduce_acceptance.py
```

该脚本使用当前 5 s 配置生成本地初态清单、逐次评估、参数扫描、轨迹、动画与性能记录，输出位于被 gitignore 的 `artifacts/`。原始数据集、轨迹和机器测试日志不随仓库上传；版本库只保留文档、演示图片/动画和验证摘要。

重建或校验 [验证摘要](docs/validation-summary.json)：

```bash
python examples/build_validation_summary.py           # 重跑摘要并记录源码与初态清单哈希
python examples/build_validation_summary.py --check   # 只校验现有摘要，不写入
```

`--check` 会比对源码哈希，并在 NumPy 版本一致时按记录的 seed、数量与 split 重新生成初态清单、比对清单哈希。

也可分别运行：

```bash
mun-lander starts --split test --count 30 --seed 2026 --output artifacts/test_starts.json
mun-lander evaluate --starts artifacts/test_starts.json --controller scripted --output artifacts/evaluation.json
mun-lander starts --split generalization --count 30 --seed 2026 --output artifacts/generalization_starts.json
mun-lander sweep --starts artifacts/generalization_starts.json --scales 0.8 1.0 1.2 --output artifacts/sweep.json
mun-lander benchmark --steps 2000 --output artifacts/benchmark.json
```

各命令可用 `--config configs/nominal.json` 显式指定配置；完整参数见 `mun-lander --help` 与子命令 `--help`。评估前固定配置与初态文件，比较各方法时复用同一文件。训练、适配、验证、测试和泛化评估使用不同 split；各参数的泛化扫描复用同一组独立 generalization 初态。正式实验还应使用至少三个训练种子。

默认训练只随机初始位置、速度、姿态与小幅角速度。泛化扫描分别改变质量、惯量、推力、重力，保持另外三个参数不变。质量扫描通过改变干质量缩放满燃料初始湿质量，不改变燃料容量。

## 交接给另外两组

| 接收方 | 已提供 | 需要对接的内容 |
| --- | --- | --- |
| 模型和 policy | 14 维状态观测、4 维连续动作、奖励分项、终止语义、固定初态、可保存恢复环境状态 | 接入 replay、世界模型/SAC/Dreamer、训练及学习曲线 |
| KSP 适配 | 单位与坐标契约、共享奖励函数、着陆阈值和评估字段 | 实测飞行器质量/惯量/推力/RCS、遥测变换、动作轴映射、游戏存档重置、损伤判定 |

详细约定见 [接口说明](docs/INTERFACES.md)，方程与接触近似见 [物理模型](docs/PHYSICS.md)。本模拟器 pilot 安全阈值为触地总速度 ≤ 2 m/s、倾角 ≤ 10°、角速度模 ≤ 0.2 rad/s，并连续稳定 5 s。除已确定的 5 s 窗口外，这些数值没有替代共享 KSP 配置中的 TBD。模拟器的 `safe_in_target_rate` 要求安全且**首次触地**位于 5 m 目标范围内；稳定窗口内的地面滑行不会改变这一定义。它和仓库任务契约允许包含不安全试验、且采样时刻另行决定的 `precision_rate` 不同，两者不能互相改名。

## 文件结构

| 路径 | 用途 |
| --- | --- |
| `src/mun_lander/config.py` | 参数验证、JSON 配置、独立参数扰动 |
| `src/mun_lander/physics.py` | 变质量六自由度刚体自由飞行与燃烧耗尽分段 |
| `src/mun_lander/env.py` | Gymnasium 环境、接触/稳定窗口、奖励与终止 |
| `src/mun_lander/controller.py` | 只使用公开观测的脚本控制器 |
| `src/mun_lander/evaluation.py` | 共同初态与结果汇总 |
| `src/mun_lander/wrappers.py` | 显式观测缩放和对称动作封装 |
| `configs/nominal.json` | 默认实验配置 |
| `tests/` | 解析物理检查、状态恢复、终止判定与接口检查 |
| `docs/assets/` | 随仓库提供的着陆动画与轨迹图 |
| `docs/validation-summary.json` | 当前仓库配置的紧凑验证摘要（schema 2，哈希可校验） |
| `examples/reproduce_acceptance.py` | 在本地重建验收轨迹、动画与逐次结果 |
| `examples/build_validation_summary.py` | 重建或校验验证摘要与源码/初态清单哈希 |
| `artifacts/` | 本地生成的数据、评估结果和轨迹；不纳入 Git |

## 已知范围

这是本地平地、常重力、固定主惯量的刚体预训练模型，没有地形、轨道曲率、空气阻力、发动机启动延迟、喷口饱和耦合或真实着陆腿结构。RCS 按等效力矩耗料，储量视为充足且不改变质量。接触损伤由速度/姿态阈值近似，不能替代 KSP 的部件损坏遥测。后续先按实船参数校准，再进行源域训练与 KSP 对比。
