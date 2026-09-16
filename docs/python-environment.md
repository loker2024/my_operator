# Python 绘图环境

## 适用范围

仓库根目录的 `.venv/` 只用于 `scripts/` 下的 Python 工具，例如将 benchmark CSV
绘制为 PNG。它不参与 CUDA、CMake 或 CUDA Toolkit 的构建环境；不要使用 Conda 的
`activate` 命令来运行仓库脚本。

`.venv/` 已由 `.gitignore` 忽略。可复现的依赖声明保存在
[`scripts/requirements.txt`](../scripts/requirements.txt)：

```text
matplotlib>=3.6
numpy>=1.24
```

## 初始化与更新

在 WSL 的仓库根目录执行。若 `.venv/` 不存在，先创建它；已存在时只更新依赖，避免
覆盖环境中已有的项目工具。

```bash
# 仅首次创建，或在确认需要重建环境时执行。
python3 -m venv .venv

# 安装或更新绘图依赖。
.venv/bin/python -m pip install -r scripts/requirements.txt
```

当前记录（2026-09-16）：`.venv/` 使用 Python 3.14.7，其基础解释器为
`/home/loker1/miniconda3/bin/python3.14`，但它本身是独立的 `venv`
（`include-system-site-packages = false`），无需也不应通过 `conda activate` 使用。
已验证安装 `matplotlib 3.11.2` 与 `numpy 2.5.3`。依赖版本以
`scripts/requirements.txt` 为准；版本升级后应重新执行上述安装命令并更新本记录。

## 使用

始终通过 `.venv/bin/python` 调用脚本，无需激活环境：

```bash
.venv/bin/python scripts/plot_kernel_perf.py \
  --csv operators/gemm/bench/<时间戳>/gemm_bench.csv
```

不传 `--out` 时，PNG 会写到输入 CSV 同目录、同名的 `gemm_bench.png`。
