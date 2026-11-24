# Mun 着陆智能体

`mun-agent` 用于在六自由度 Mun 着陆模拟器中开发和训练策略。M0 提供可编辑安装的 Python 包 `mun_agent` 和 CPU 导入测试。模拟器的原生输入输出见[接口说明](../simulator/docs/INTERFACES.md)，后续训练采用的约定见[方法说明](docs/METHOD.md)。

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
