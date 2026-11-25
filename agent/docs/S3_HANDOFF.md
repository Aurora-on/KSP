# S3 源域策略与固定初态清单

模型组与环境组可以用同一个冻结策略作为名义六自由度月面着陆模拟器的源域基线。正式验证从三个种子中选定 **seed 33，第 22 轮**。它在固定验证和测试清单上均为 100/100 安全入靶，原环境平均回报分别为 93.399 和 93.447。这些成绩来自 `MunLander-v0`。KSP 的实际运行与接口校准记录见[验证记录](../../docs/validation.md)。

从 [GitHub Release](https://github.com/Aurora-on/KSP/releases/tag/s3-source-seed33-r22-v1) 获取 [s3-source-seed33-r22-v1.zip](https://github.com/Aurora-on/KSP/releases/download/s3-source-seed33-r22-v1/s3-source-seed33-r22-v1.zip)。交接版本为 v1，压缩包 109,653 字节。推理代码的已验收版本为 [0e4ff9d](https://github.com/Aurora-on/KSP/tree/0e4ff9daef78bf3ceba0e467288b62a4ffaccae0)。

| 文件或来源 | 内容 | SHA-256 |
|---|---|---|
| `s3-source-seed33-r22-v1.zip` | 完整发布附件 | `01e5dad61253d1910e96b343e8d0dbe7719d8b59f8cab7139e10c003983812f9` |
| `policy.pt` | 82,185 字节，actor 权重、网络规格、输入接口与序列化版本 | `4cb558d7e7fa66e0c9dca317b96c0d34dec58a60deba99802768178e1be57fee` |
| 原始 `best.pt` | 正式报告记录的源训练检查点，供维护者追溯 | `a4451b97f733712c4d7832572d787ff24b816d214bb9f55fc03cdfc0837eae65` |

包中的 `s3-validation.json` 与 `s3-test.json` 是正式实验的原始文件，各含 100 个起点、重置种子和完整初始状态。验证清单用于开发与选优；测试清单留给冻结策略的最终评估。清单内置的内容摘要为验证 `2bf7a0898b9ed4f38dd46f996765f675a97fe703f0c9dcc0fee83727d0e51321`、测试 `c12a88bc1047314fb482bbeec67ba5395e608cdd8be444e7f212f23e8a483f35`。两份 JSON 文件各自的字节级 SHA-256、原报告摘要、包版本、运行库版本和八帧参考动作记录在 `handoff.json` 中。

## 下载后核验

在含本说明的仓库根目录、Python 3.12–3.14 虚拟环境中执行。发布验收使用 Python 3.14.4 和 CPU。

```bash
python -m pip install -e ./simulator -e ./agent
bundle=s3-source-seed33-r22-v1.zip
curl -fL "https://github.com/Aurora-on/KSP/releases/download/s3-source-seed33-r22-v1/$bundle" -o "$bundle"
python -c 'import hashlib, pathlib, sys; p = pathlib.Path(sys.argv[1]); sys.exit(0 if hashlib.sha256(p.read_bytes()).hexdigest() == "01e5dad61253d1910e96b343e8d0dbe7719d8b59f8cab7139e10c003983812f9" else "ZIP SHA-256 mismatch")' "$bundle"
python -m zipfile -e "$bundle" s3-source
python agent/scripts/s3_handoff.py verify s3-source
```

核验脚本检查各文件的字节摘要、两份初态清单的内部一致性，以及 CPU 推理的八个连续四维动作。发布前还在空目录解包并加载，在全部 100 个验证起点的前 12 帧对照源检查点。共 1,200 帧，四个动作分量的最大绝对差为 0。加载本包只需模拟器与 agent 两个本地 Python 包。

## 策略调用

每回合先调用一次 `policy.reset()`，随后按顺序传入每个原始观测。策略内部维护贴地持续步数和回合时钟，模拟器的控制间隔为 0.1 秒。输入是 `float32[14]`，依次为目标系位置（米）、世界系速度（米/秒）、机体到世界系的四元数 `[w,x,y,z]`、机体系角速度（弧度/秒）和剩余主推进剂（千克）。输出是 `[throttle, rcs_x, rcs_y, rcs_z]`，油门取值 [0,1]，三个机体系力矩分量取值 [−1,1]。策略内部完成输入缩放和油门映射。[坐标、观测与动作接口](../../simulator/docs/INTERFACES.md)

```python
from pathlib import Path

from mun_agent.nominal import load_manifest
from mun_agent.policy import load_policy
from mun_lander import LanderEnv

bundle = Path("s3-source")
policy = load_policy(bundle / "policy.pt", device="cpu")
start = load_manifest(bundle / "s3-validation.json", "validation")["starts"][0]
env = LanderEnv()
try:
    obs, _ = env.reset(seed=start["seed"], options={"initial_state": start["initial_state"]})
    policy.reset()
    while True:
        obs, reward, terminated, truncated, info = env.step(policy(obs))
        if terminated or truncated:
            print(start["id"], info["status"])
            break
finally:
    env.close()
```

KSP 适配时需按共有接口对齐目标坐标系、接触高度、动作轴的正负方向和执行时序。跨域比较应分别记录源域与 KSP 结果，并随结果标注策略文件 SHA、清单摘要和环境版本。[环境接入说明](../../simulator/docs/REPOSITORY_INTEGRATION.md)

## 维护者重建

保留正式报告、选中的完整 `best.pt` 和原始两份清单的维护者，可从仓库根目录导出到新路径。导出器先核对正式报告中的检查点、运行标识与每次评估采用的清单，再从原策略生成 CPU 参考动作、剥离训练状态并验证输出。

```bash
python agent/scripts/s3_handoff.py export --out ../handoff/s3-source-seed33-r22-v1.zip
```

默认输入位于 `agent/runs/s3-stable` 和 `agent/data`；用 `--checkpoint`、`--report`、`--starts-dir` 可以指定归档位置。发布时以 Release 附件和上表的整包摘要识别 v1。训练回放、优化器、本地训练目录均留在训练端。
