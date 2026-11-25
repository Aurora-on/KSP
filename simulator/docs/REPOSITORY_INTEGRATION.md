# 接入 Aurora-on/KSP 的范围与待对齐项

本次将已实现的模拟器作为可独立安装的 `simulator/` 模块纳入仓库，提供 6DOF 动力学、Gymnasium 接口、脚本控制器、共同初态生成、评估与独立参数扫描。KSP 连接诊断、[任务契约](../../docs/task-spec.md)、[实验协议](../../docs/experiment-protocol.md) 和 [共享配置](../../configs/mun_landing.toml) 继续作为团队的已有决策依据。本次没有改写其中的 TBD、实验分组、数据划分或就绪标记，也没有增加真实 KSP 着陆适配器。

## 已统一：连续稳定 5 s

仓库任务明确要求触地后持续稳定 5 s，因此本目录的 `LandingConfig.stability_time` 和 [nominal.json](../configs/nominal.json) 使用 5 s。原本地独立交付版使用 2 s，其代码、文档与原始证据保留在原交付位置；原验收数值不能直接作为仓库 5 s 版本的结果。

仓库版本的实际验证以 [VALIDATION.md](VALIDATION.md) 和 [validation-summary.json](validation-summary.json) 为准。稳定窗口对齐只解决一个任务参数，不代表坐标、实船物理参数或整体迁移已经验证。

## 保留原生 API，显式记录差异

| 项目 | 模拟器原生接口 | 仓库共享约定 / 对接工作 |
| --- | --- | --- |
| 观测 | 14 维：位置 3、速度 3、四元数 4、角速度 3、主燃料 1 | 字段类别一致；KSP 坐标、四元数约定和缩放仍待校准 |
| 坐标 | 目标局部右手系、世界 `+z` 向上、机体 `+z` 推力；`wxyz` 表示 body→world | 不直接使用未验证的 kRPC 轴序、手性或符号；需实船测量变换 |
| 动作 | `[throttle, rcs_x, rcs_y, rcs_z]`，油门优先 | 共享配置为 `[rcs_pitch, rcs_yaw, rcs_roll, main_engine_throttle]`；需明确数组重排和实际轴映射 |
| RCS | 三轴机体系等效力矩比例 `[-1,1]` | 共享配置的 torque/normalized input 语义及增益仍为 TBD |
| 控制周期 | 默认 0.1 s，物理子步 0.02 s | 任务适配器按游戏 UT 保持动作并记录实际 dt，10 Hz 步进仍待 KSP 实测。[步进接口](../../docs/ksp-env.md#stepping-by-game-time) |
| 角速度 | 观测 rad/s，pilot 安全阈值 0.2 rad/s | 共享阈值约 11.4592 deg/s（0.2 rad/s）；对接时须换算单位 |
| 安全窗口 | 5 s | 已与共享任务一致 |
| 质量/惯量观测 | 未放入策略观测；质量只在 info 中用于诊断 | 共享任务已确定质量、惯量仅作诊断 |

顺序变换不等于物理轴映射。不能因名称相似就声明 `rcs_x=rcs_pitch` 等对应关系。先进行静止直立、单轴小转动、单轴短脉冲和竖直点火检查，确认机体/世界变换、控制符号和增益，再记录适配器中的映射与测试。模拟器核心保持原生 API；未来适配层应将双方转换到团队最终确定的统一格式。

## 精度指标必须区分

两者分母都是全部试验，但分子不同：

```text
模拟器 safe_in_target_rate = 安全且首次触地落点在目标半径内的试验数 / 全部试验数
仓库   precision_rate      = 位于目标半径内的试验数 / 全部试验数
```

仓库定义允许入靶但不安全的试验计入 `precision_rate`。不能把模拟器的 `safe_in_target_rate` 直接改名为 `precision_rate`；模拟器以**首次触地**位置判定落点，稳定窗口内的滑行不改变它，而共享评估器仍需依据任务契约最终确定的位置采样时刻另行计算仓库指标，并可同时报告安全入靶率。安全着陆率、成功样本的落点误差、主发动机/RCS/总推进剂耗量继续分别报告。

## Pilot 数值不替代 KSP 决策

团队已采用 S3 初态分布、10 Hz、90 s、触地安全阈值和 5 m 目标半径。飞船质量、惯量、主推力、Isp 和三轴 RCS 力矩仍待 KSP 实测；模拟器车辆默认值仍属 pilot 参数。[2026-11-25 任务决议](../../docs/nominal-task-decisions.md)

正式比较前须选定实船、目标点和下降快照，固定任务数值，完成 KSP reset/step 与损伤判定，并用共同 held-out 转移数据验证一步及短期预测误差。模拟器测试通过和脚本着陆成功只证明源域工程行为，不构成世界模型训练、KSP 迁移或 RQ1–RQ3 结论。已有文档决策与数据划分冻结约定保持有效；后续变更需要作为明确的新决策记录。

## 仓库存放与本地复现

仓库保留代码、测试、文档，以及以下轻量展示证据：

- [着陆动画](assets/landing.gif)
- [轨迹图](assets/trajectory.png)
- [验证摘要](validation-summary.json)

原始数据集、初态清单、逐次轨迹、模型权重和机器测试日志不随本次上传。进入 `simulator/` 安装依赖后运行：

```bash
python -m pytest -q
python examples/reproduce_acceptance.py
```

复现脚本在本地被忽略的 `artifacts/` 生成测试初态、名义与失败基线评估、独立泛化扫描、NPZ/CSV 轨迹、动画及性能记录。保留每次运行的配置、初态清单哈希和代码版本；性能结果会随机器负载变化。
