# 模拟器验收与仓库集成

2026 年 11 月 19 日，按 `requirements-tested.txt` 固定的依赖版本重跑全部验收，并改用可重建的 schema 2 摘要（`examples/build_validation_summary.py`）。仓库任务契约已有的 **5 s 稳定窗口**保持不变。该结果使用脚本工程控制器和模拟器试验参数；KSP 实船标定、跨模拟器对齐、世界模型和策略训练仍未完成。原独立交付的 2 s 验收记录不作为此版本证据。

## 实测结果

| 检查 | 结果 |
| --- | --- |
| 模拟器测试 | 62 项通过，0 失败 |
| 仓库原有离线测试 | 8 项通过，0 失败 |
| 100 个固定名义测试初态 | 100/100 安全；100/100 安全且首次落点在 5 m 内 |
| 安全率 Wilson 95% 区间 | 96.30%–100% |
| 安全样本平均落点误差（首次触地） | 1.283 m |
| 平均主发动机 / RCS / 总耗料 | 31.023 / 0.530 / 31.553 kg |
| 零动作、随机动作 | 同一测试清单前 10 个初态，各 10/10 碰撞失败 |
| 独立泛化初态 | 10 个；质量、惯量、推力、重力各自独立取 0.8/1.0/1.2 倍，全部安全且入靶 |
| 性能 | 2000 步实测 2501 transitions/s（约 250 倍仿真实时）；含控制器，不含绘图；随机器负载变化 |
| 演示 seed 7 | 36.238 s 完成；首次触地速度 0.692 m/s；落点误差 0.737 m |

各参数的 1.0 倍复用同一名义结果，因此 12 行参数扫描对应 9 个独立物理条件、90 次运行。控制器始终使用名义参数；泛化和名义测试使用不同 split。零动作/随机动作只检查失败路径，不构成学习算法比较。这里的 `safe_in_target_rate` 要求安全着陆且**首次触地**落在目标半径内，不能替代根任务契约中不同定义的 `precision_rate`。

测试覆盖解析自由落体、火箭方程、燃尽分段、角动量和动能、姿态约定、首次触地、持续稳定、损坏检查点拒绝、数据划分、初态清单与配置一致性、落点采样时刻和向量环境。两条 Gymnasium 提示来自 SI 状态中保留的无限 Box 边界。GitHub Actions 配置 Python 3.12/3.14；本次本地执行 Python 3.14.4、NumPy 2.4.6、Gymnasium 1.4.0。

## 可检查证据与重现

[机器可读验收摘要](validation-summary.json) 保存配置、依赖版本、初态清单的 seed/数量/哈希、源码哈希和指标；[着陆动画](assets/landing.gif) 与 [轨迹图](assets/trajectory.png) 来自本次 5 s 配置。摘要中的初态清单哈希可用记录的 seed/数量/split 重新生成，因此不是一次性的本地快照。

在仓库根目录执行测试：

```bash
python -m pip install -e './simulator[dev,demo]'
python -m pytest simulator/tests -q
python -m unittest discover -s tests -v
```

重建原始评估、演示和摘要：

```bash
cd simulator
python examples/reproduce_acceptance.py     # 轨迹、图像、动画与逐次结果
python examples/build_validation_summary.py # 源码哈希、初态清单与汇总指标
python examples/build_validation_summary.py --check  # 只校验，不写入
```

两个脚本固定样本数量、初态 seed、数据划分、控制器和扰动范围。`reproduce_acceptance.py` 在本地 `simulator/artifacts/` 生成逐次结果、共同初态、CSV/NPZ 轨迹、图像和动画；目录已被 gitignore，原始数据、虚拟环境和机器路径日志不进入仓库。`--check` 校验源码哈希，并在 NumPy 版本一致时重新生成初态清单比对哈希；性能会随机器负载变化。

## 对接范围

[仓库集成说明](REPOSITORY_INTEGRATION.md) 列出动作顺序、RCS 轴映射、观测选择与指标定义差异；[接口说明](INTERFACES.md) 定义当前源模拟器 API；[物理模型](PHYSICS.md) 说明简化接触等适用范围。

根 `configs/mun_landing.toml` 的 TBD 与 readiness=false 保持有效。除已确定的 5 s 稳定窗口之外，本模块的飞行器参数、阈值、10 Hz 控制频率和初态分布都是可运行的试验默认值。600 kg 主推进剂对默认 80–120 m 下降场景较宽松，紧燃料预算需在 pilot 中按实船参数调整并重新冻结。

## 对接范围

[仓库集成说明](REPOSITORY_INTEGRATION.md) 列出动作顺序、RCS 轴映射、观测选择与指标定义差异；[接口说明](INTERFACES.md) 定义当前源模拟器 API；[物理模型](PHYSICS.md) 说明简化接触等适用范围。

根 `configs/mun_landing.toml` 的 TBD 与 readiness=false 保持有效。除已确定的 5 s 稳定窗口之外，本模块的飞行器参数、阈值、10 Hz 控制频率和初态分布都是可运行的试验默认值。600 kg 主推进剂对默认 80–120 m 下降场景较宽松，紧燃料预算需在 pilot 中按实船参数调整并重新冻结。
