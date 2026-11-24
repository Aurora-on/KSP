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

安装本仓库的模拟器、智能体和测试依赖，然后运行智能体测试。约束文件沿用仓库固定版本。

```sh
python -m pip install -c requirements/runtime.txt -e ./simulator -e './agent[dev]'
python -m pip check
python -m pytest agent/tests -q
```

## 采集 M2R 回放

以下命令从仓库根目录运行。三份数据分别用于拟合、开发时选方案和最终质量验收。每份都从模拟器的 `train` 划分生成完整回合，采集根种子不同。策略阶段的固定验证与测试清单另行使用。

```sh
mun-agent collect --config agent/configs/base.json --device cpu --seed 170 --out agent/data/m2r-train
mun-agent collect --config agent/configs/base.json --device cpu --seed 180 --episodes 50 --out agent/data/m2r-dev
mun-agent collect --config agent/configs/base.json --device cpu --seed 190 --episodes 50 --out agent/data/m2r-heldout
```

`base.json` 默认采集 200 回合，其中 60% 用硬着陆扰动，5% 用随机动作，其余在脚本动作上添加标准差 0、0.05、0.1、0.2 的高斯噪声。采集在 CPU 上执行。命令打印步数、清单哈希和各结局计数。输出目录包含原生数据 `replay.npz` 和 `collection.json`，后者记录初态清单、扰动参数及种子、每回合的结局、首次触地总速度（m/s）和倾角（°）。未触地回合的两项触地指标为 `null`。已有输出目录会被拒绝覆盖。

## 离线拟合与开发诊断

`train --data` 使用指定回放目录中的全部回合。调整 `collect.episodes` 影响下一次采集。下例复现短拟合 A。CPU 冒烟可改用 `debug.json` 并缩短更新数。

```sh
mun-agent train --config agent/configs/base.json --data agent/data/m2r-train --seed 31 \
  --device cpu --out agent/runs/m2r-short-A --set fit.length=32 --set fit.updates=2000 --set fit.checkpoint_every=1000
mun-agent wm-report --config agent/runs/m2r-short-A/config.json --device cpu --data agent/data/m2r-dev \
  --checkpoint agent/runs/m2r-short-A/world_model.pt --out agent/runs/m2r-short-A/dev.json
```

正式候选 C 增加分类因子与类别数，减小 KL 权重，并使用连续贴地步数。它与 A 使用相同的数据、种子、片段长度和终止转移权重。

```sh
mun-agent train --config agent/configs/base.json --data agent/data/m2r-train --seed 31 \
  --device cpu --out agent/runs/m2r-short-C --set fit.length=32 --set fit.updates=2000 \
  --set fit.checkpoint_every=1000 --set model.stoch=32 --set model.classes=32 \
  --set model.dynamics_scale=0.5 --set model.representation_scale=0.03 \
  --set model.event_weight=1 --set model.ground_history=true
```

另有协议外探索 B。复现其短拟合时，将上述输出改为 `agent/runs/m2r-short-B`，并把 `model.event_weight` 改为 40。B 的长拟合探索在 7,947 次更新、561.01 秒时主动停止。正式长拟合只从 A、C 选择，其终止转移权重均为 1。[选择依据及加权的影响](docs/METHOD.md)

`--set section.key=value` 可多次使用，值采用 JSON 语法。用 `fit.updates` 指定更新预算，用 `fit.max_seconds` 指定墙钟预算（默认 7140 秒，上限 7200 秒）；拟合在达到任一预算后保存 `world_model.pt`。当 `fit.checkpoint_every` 为正数时，还会保存第 0 次及每隔指定更新数的 `step-XXXXXX.pt`，可逐个运行报告以查看误差随训练的变化。`config.json` 记录实际训练回合数、数值配置、设备、代码版本、种子和数据哈希。`metrics.jsonl` 记录每次更新，`summary.json` 给出同一批训练样本在拟合前后的各项探针损失及实际更新数。拟合和报告支持 `auto|cpu|mps|cuda`；`auto` 按 CUDA、MPS、CPU 的顺序选择。

## 复现正式长拟合 A

共享的 50 个开发回合上，A、C 的短拟合均通过 0/5 项门槛。A 的后验位置误差为 3.42 m，C 为 6.49 m。两者最差的物理指标均为该位置误差，分别达到对应上限的 6.83 倍和 12.98 倍，正式长拟合采用 A。以下命令从种子 31 重新初始化，每 4,000 次更新保存一次候选模型，以 300,000 次更新和 7,140 秒墙钟预算中先达到者为限。

```sh
mun-agent train --config agent/configs/base.json --data agent/data/m2r-train --seed 31 \
  --device cpu --out agent/runs/m2r-long-A --set fit.length=32 --set fit.updates=300000 \
  --set fit.checkpoint_every=4000 --set fit.max_seconds=7140
```

## 重建开发集曲线

绘图使用可选依赖 Matplotlib。仓库约束文件固定了该依赖的版本。以下命令逐个读取已有检查点并在同一批 50 个开发回合上生成报告，文件名中的 `dev` 表示采集种子 180。图中包含两种正式短拟合、探索性短拟合 B 以及正式长拟合 A。

```sh
python -m pip install -c requirements/runtime.txt matplotlib
study_dir=agent/runs/m2r-study-reproduced
for name in m2r-short-A m2r-short-C m2r-short-B m2r-long-A; do
  run_dir=agent/runs/$name
  [ -d "$run_dir" ] || continue
  for checkpoint in "$run_dir"/step-*.pt "$run_dir"/world_model.pt; do
    [ -f "$checkpoint" ] || continue
    stem=$(basename "$checkpoint" .pt)
    report_path="$study_dir/$name/$stem-dev.json"
    [ -f "$report_path" ] || mun-agent wm-report --config "$run_dir/config.json" \
      --device cpu --data agent/data/m2r-dev --checkpoint "$checkpoint" --out "$report_path" >/dev/null
  done
done
python agent/scripts/plot_wm_quality.py --study "$study_dir" --long-run A
```

图表保存在 `$study_dir/quality-curve.png` 和 `$study_dir/quality-curve.svg`。横轴为优化器更新次数。开发集用于选方案和检查点，各误差的单位、比较基线和达标比例见[方法说明](docs/METHOD.md)。

## 最终质量验收

正式长拟合 A 在 7,140.028 秒内更新 204,325 次。开发集从 53 个候选中选出第 28,000 次更新的快照。该快照在独立 50 回合上通过 G3，G1、G2、G4、G5 未达标，M2R 停在世界模型阶段。复现本次选定快照的留出报告可运行以下命令。

```sh
run_dir=agent/runs/m2r-long-A
selected_checkpoint="$run_dir/step-028000.pt"
mun-agent wm-report --config "$run_dir/config.json" --device cpu --data agent/data/m2r-heldout \
  --checkpoint "$selected_checkpoint" --out "$run_dir/heldout.json"
```

报告核对模型配置、环境接口、清单和回放哈希。默认模式要求数据与训练初态种子无重叠。若要诊断训练集欠拟合，传入 `--training-diagnostic` 和原始训练目录，该模式核对训练回放 SHA-256。两种模式都可用运行目录的 `config.json`；它保留了实际生效的 `--set model.*`。报告为确定性计算，`--seed` 不改变结果。

JSON 的 `posterior` 和 `horizons` 分别提供后验重建和分时距预测统计，两者各自的 `near_ground` 只取真实端点高度低于 10 m 的样本。`terminal_episodes` 保存逐回合终止奖励与终止前 15 步开环结果。`gates` 给出 G1–G5 明细。M2R 验收使用至少 50 个最终留出回合，要求全部门槛通过。[方法说明](docs/METHOD.md) · [验证记录](../docs/validation.md)

加载器支持 schema 1、2、3，按文件记录的坐标尺度和奖励头读取旧权重。旧回放没有真实结局标签时，相应的 G4 判定为 `null`；旧检查点缺少训练均值时，该基线也为 `null`。
