# 模拟器、KSP 与模型组的接口契约

公共导入为 `from mun_lander import LanderEnv, SimConfig`。使用 `LanderEnv(config)` 创建环境，配置可通过 `SimConfig.from_json(path)` 加载。所有原始观测使用 SI 单位；训练前的缩放必须在源模拟器和 KSP 两端一致。

这里描述的是 `simulator/` 已实现的原生接口。仓库 [共享任务契约](../../docs/task-spec.md) 的 KSP 坐标、动作轴与缩放仍有 TBD，因此两端尚未完成校准。已确定的对齐项及差异见 [仓库接入说明](REPOSITORY_INTEGRATION.md)。

`reset/step` 返回值及 `terminated/truncated` 的标准含义参见 [Gymnasium 官方 Env API](https://gymnasium.farama.org/api/env/)；本任务具体的失败判定与 continuation 约定见下文。

## 坐标与状态

世界坐标系是固定在目标处的局部**右手系**，`+z` 竖直向上，`x,y` 是预先固定的水平轴。位置相对目标；水平目标为 `(0,0)`。`z=0` 是等效地面接触面，表示接触点净空，不是直接照搬 KSP 的质心海拔。机体系也是右手系，主发动机推力沿机体 `+z`。绕正轴的角速度与正 RCS 指令遵循右手法则。

姿态四元数为 `q=(w,x,y,z)`，表示**机体系到世界系**的主动旋转：`vector_world = rotation_matrix(q) @ vector_body`。角速度为机体系分量。四元数单位化；`q` 与 `-q` 表示同一姿态，模型组不能把符号变化解释成翻转。若自行做连续符号处理，训练、评估、KSP 适配必须复用同一处理。

原始观测是 `float32`、形状 `(14,)`：

| 索引 | 名称 | 单位/语义 |
| --- | --- | --- |
| `0:3` | `position` | m；目标相对位置与等效接触净空 |
| `3:6` | `velocity` | m/s；世界系目标相对速度 |
| `6:10` | `quaternion` | `wxyz`；机体系到世界系 |
| `10:13` | `angular_velocity` | rad/s；机体系 |
| `13` | `fuel` | kg；主发动机剩余推进剂 |

质量、重力、惯量等不放入策略观测。`info["mass"]` 是调试和评估数据，使用它作策略输入会改变任务定义，尤其会泄漏泛化测试参数。姿态阈值配置/日志采用度，原始角速度和物理运算采用弧度。

## 动作与时间

动作形状 `(4,)`，顺序是 `[throttle, rcs_x, rcs_y, rcs_z]`。油门范围 `[0,1]`，RCS 三轴范围 `[-1,1]`。三轴表示机体系等效力矩比例；默认无 RCS 平移指令、无反作用轮、无 SAS。有限但越界的动作会裁剪并在 `info["applied_action"]` 与 `info["action_clipped"]` 中报告；NaN、Inf、形状错误会报错。

共享配置 `configs/mun_landing.toml` 列出的顺序是 `[rcs_pitch, rcs_yaw, rcs_roll, main_engine_throttle]`。它与本接口的顺序及轴命名不同；在 kRPC 轴向/符号与增益校准前，不能仅重排数组就认定已经对接，也不能把 `pitch/yaw/roll` 名称直接映射为 `x/y/z`。

默认每次 `step` 保持动作 `control_dt=0.1 s`，内部物理步长不超过 `physics_dt=0.02 s`。触地、燃料耗尽、超时等事件可缩短实际推进时间；使用 `info["dt"]` 和 `info["time"]`，不要用网络调用耗时替代仿真时间。KSP 端需要以游戏模拟时间驱动相同控制周期，并记录延迟/实际步长。

需要对称动作空间时可显式使用 `SymmetricAction`；它把第一维 `[-1,1]` 映射到原始油门 `[0,1]`，后三维保持不变。使用 `NormalizedObservation` 时应保存并共享缩放约定；其默认除数依次为位置 100 m、速度 10 m/s、四元数 1、角速度 0.5 rad/s、燃料为配置容量，结果不裁剪到 `[-1,1]`。原始接口保持 SI 单位。

## Reset、数据与恢复

```python
obs, info = env.reset(seed=123)
obs, info = env.reset(seed=123, options={"initial_state": state_dict})
obs, reward, terminated, truncated, info = env.step(action)

checkpoint = env.get_state()
env.set_state(checkpoint)
```

`state_dict` 包含 `position, velocity, quaternion, angular_velocity, fuel`。这是一个新 episode 的初态，不包含累计耗料、接触历史或随机数状态。要精确恢复中途运行应使用完整 `get_state/set_state` 检查点；恢复环境配置必须兼容。`LanderState.to_dict/from_dict` 可序列化物理状态，初态清单可直接交换给 KSP 适配组。

同 seed/配置/动作序列可复现本模拟器轨迹，但不保证不同库版本、硬件或不同模拟器逐位一致。共同评估应交换实际初态清单，而不是假定 KSP 使用同一整数 seed 就生成相同初态。

## 终止与世界模型训练

| `status` | Gymnasium 结束标志 | 评估结果 |
| --- | --- | --- |
| `flying` / `stabilizing` | 均为 False | 尚未结束 |
| `safe_landing` | `terminated=True` | 安全成功 |
| `crash` / `fuel_exhausted` | `terminated=True` | 失败 |
| `timeout` / `out_of_bounds` | `truncated=True` | 失败 |

安全判定要求第一次触地符合速度/姿态/角速度阈值，随后持续满足稳定条件。仓库默认窗口为 5 s，原本地独立版为 2 s；历史结果必须保留对应配置。目标半径不参与 `is_success`；`safe_in_target` 才同时要求安全和落点达标。其他安全阈值仍是本模拟器 pilot 值，共享 KSP 阈值须另行决策并在正式比较前冻结。

Replay 至少保存 `obs, applied_action, reward, next_obs, terminated, truncated, dt`，并保留终止步的真实 `next_obs`；自动重置后的初态不能替代它。Dreamer 的 `is_last = terminated or truncated`，新 episode 第一个观测 `is_first=True`。按照标准 Gymnasium bootstrap 语义，`is_terminal = terminated`；不要把“成功”等同于 `terminated`，因为碰撞和燃料耗尽同样是 terminal。

向量环境示例见 [vector_rollout.py](../examples/vector_rollout.py)；使用 `SAME_STEP` 自动重置时，终止观测在 `infos["final_obs"]`。本地演示导出的 `trajectory.npz` 包含 N+1 个 `observations` 和 N 个 `actions/rewards/terminated/truncated/dt`，对应转移为 `observations[:-1] → observations[1:]`。运行 [reproduce_acceptance.py](../examples/reproduce_acceptance.py) 会在本地 `artifacts/` 重建这些数据，原始轨迹不纳入 Git。

这里超时和越界在评估中明确算失败，且有失败奖励。如果模型组将任务时间上限视为真正的吸收失败，应显式将对应状态的 continuation/discount 置零，并在全部方法及 KSP 端统一使用这个规则；不能在不同算法中分别隐含处理。四元数、燃料与时间单位也应随数据集保存。

## 结果与奖励字段

每步 `info` 含时间、状态、`is_success, safe_in_target, horizontal_error`、`main_fuel_used, rcs_fuel_used, total_fuel_used`、倾角、角速度、接触状态、累计稳定时间和 `reward_terms`。燃料消耗字段是 episode 累计量，单位 kg。`touchdown` 保存首次触地前速度、倾角、角速度、水平误差和事件时间；不能用碰撞投影/阻尼后的速度替代。`last_touchdown` 保存最近一次触地快照，`touchdown_count` 累计触地次数；再次离地会清零稳定时间，再触地需要重新满足整个窗口，首次触地快照仍保留。

为兼容 Gymnasium 向量环境混合飞行/接触状态，首次触地前 `info` 不含 `touchdown`、`last_touchdown` 键，单环境使用 `info.get("touchdown")` 读取；向量环境同时检查对应的 `_touchdown` 有效性掩码。完整 checkpoint 始终保留这两个字段，未触地时值为 `None`。

终止时另有 `episode` 汇总。`reward_terms` 各项之和等于当前步奖励；共享实现位于 `mun_lander.reward.compute_reward`。KSP 应按同一时间步积分运行代价，按实际耗料记油耗代价，按同一任务状态发放终止奖惩。真实 KSP 的部件损伤是额外安全否决项，不能由这里的接触近似替代。

评估汇总的 `safe_in_target_rate = 安全且入靶的试验数 / 全部试验数`。仓库任务契约的 `precision_rate = 入靶试验数 / 全部试验数` 不要求安全，因此入靶但不安全的试验也可计入其分子。这两个指标不能直接改名或混用；共享评估器需按任务契约单独计算 `precision_rate`，同时保留安全入靶率供参考。

## KSP 适配前必须落实的映射

先选定目标坐标参考系、水平轴以及机体前向/右向/推力轴，测量并测试变换矩阵；不要直接将 kRPC 的 `pitch/yaw/roll` 名字当作本环境 `x/y/z`。本项目没有实船连接测试，因此不声称某一 kRPC 轴号或符号已经验证。

适配器至少用静止直立、单轴小转动、单轴短脉冲和竖直点火四类检查确认：坐标手性、四元数顺序/方向、角速度轴/单位、三个控制符号和推力方向。将质心海拔变换成接触净空；将主推进剂和 RCS 资源按密度换算为 kg；主发动机与 RCS 消耗分别记录。固定构型并关闭额外姿态辅助，实测主发动机推力、Isp、主惯量、RCS 等效力矩后再替换默认配置。
