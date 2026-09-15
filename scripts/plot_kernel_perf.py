#!/usr/bin/env python3
"""性能曲线绘图：把 bench CSV 画成「吞吐 vs 矩阵尺寸」的多内核对比图。

CSV 约定（表头需含这些列，列名可用命令行覆盖）：

    label,size,median_ms,gflops,iters

前两列是内核名（图例）与矩阵边长，绘图默认取 size 与 gflops。GEMM 的
`gemm --bench` 直接产出该格式；其他算子只要提供同样的列名即可复用本脚本。

用法：

    # 1. 多尺寸性能扫描 → CSV（只计时，不做正确性校验）
        ./build/operators/gemm/gemm --bench
    # 2. 画「GFLOP/s vs 矩阵尺寸」对比图 → PNG 与 CSV 同目录、同名
        python3 scripts/plot_kernel_perf.py --csv operators/gemm/bench/20260915141200/gemm_bench.csv

不指定 --out 时，PNG 写在输入 CSV 的同名 .png 处（与 CSV 同目录）。同一次扫描的
(label, size) 必须唯一：出现重复点说明把多次扫描的 CSV 混在了一起（`gemm --bench`
每次扫描各写一个时间戳目录），脚本会直接报错提示只传其中一次。

样式对齐参考图（Simon Boehm 的 CUDA matmul 优化博客）：灰底白网格、等宽字体、
x 轴按尺寸做「分类等距」刻度并旋转 45°，折线末端用同色文字标注内核名。
"""

import argparse
import csv
import os
import sys
from collections import OrderedDict

import matplotlib

matplotlib.use("Agg")  # 只出图不弹窗，便于在无显示环境的 WSL / 服务器上运行
import matplotlib.pyplot as plt

# 默认配色对齐参考图（R ggplot2 的默认色系：粉 → 绿 → 蓝 …），曲线数量多于序列长度时循环取用。
DEFAULT_PALETTE = ("#F8766D", "#00BA38", "#619CFF", "#C77CFF", "#00BFC4", "#7CAE00", "#F564E3",
                   "#9590FF")


def ParseArgs():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--csv", nargs="+", required=True, help="输入 CSV，可给多个（按顺序绘制）")
    parser.add_argument("--out", default=None, help="输出 PNG，默认取第一个 CSV 的同名 .png")
    parser.add_argument("--title", default="Performance of different kernels", help="图标题")
    parser.add_argument("--xlabel", default="Matrix size (square, one side)", help="x 轴标签")
    parser.add_argument("--ylabel", default="GFLOPs/s", help="y 轴标签")
    parser.add_argument("--label-col", default="label", help="内核名列名")
    parser.add_argument("--x-col", default="size", help="x 列列名")
    parser.add_argument("--y-col", default="gflops", help="y 列列名")
    parser.add_argument("--palette", default=",".join(DEFAULT_PALETTE),
                        help="逗号分隔的颜色序列，按曲线顺序取用")
    parser.add_argument("--width", type=float, default=11.0, help="图宽（英寸）")
    parser.add_argument("--height", type=float, default=6.0, help="图高（英寸）")
    parser.add_argument("--dpi", type=int, default=150, help="输出 DPI")
    parser.add_argument("--plain-labels", action="store_true", help="末端标签不带序号前缀")
    return parser.parse_args()


def LoadSeries(paths, label_col, x_col, y_col):
    """读取一个或多个 CSV，返回 {内核名: [(x, y), ...]}，顺序与文件中的首次出现一致。

    (label, x) 重复即视为「把多次扫描的数据混在一起」，直接报错而不是把点叠加成一条线。
    """
    series = OrderedDict()
    seen = {}
    for path in paths:
        try:
            handle = open(path, newline="", encoding="utf-8")
        except OSError as error:
            sys.exit(f"cannot read {path}: {error}")
        with handle:
            reader = csv.DictReader(handle)
            missing = {label_col, x_col, y_col} - set(reader.fieldnames or [])
            if missing:
                sys.exit(f"{path}: missing columns {', '.join(sorted(missing))}")
            for row in reader:
                label = row[label_col]
                x = int(float(row[x_col]))
                y = float(row[y_col])
                if (label, x) in seen:
                    sys.exit(f"duplicate point {label}@{x}: {path} conflicts with {seen[(label, x)]}\n"
                             "point at one single scan CSV; each `gemm --bench` run writes its own "
                             "timestamped directory")
                seen[(label, x)] = path
                series.setdefault(label, []).append((x, y))
    if not series:
        sys.exit("no data rows found in the given CSV file(s)")
    return series


def Plot(series, args):
    # 参考图的风格来自 seaborn 的 darkgrid：浅灰底、白色网格。
    for style in ("seaborn-v0_8-darkgrid", "ggplot"):
        if style in plt.style.available:
            plt.style.use(style)
            break
    plt.rcParams["font.family"] = "monospace"

    # x 轴按分类处理：128/256/512/1024/2048/4096 在参考图里是等距刻度，不用数值轴或对数轴。
    sizes = sorted({x for points in series.values() for x, _ in points})
    index_of = {size: index for index, size in enumerate(sizes)}
    colors = [color.strip() for color in args.palette.split(",") if color.strip()]
    if not colors:
        sys.exit("--palette must contain at least one color")

    fig, ax = plt.subplots(figsize=(args.width, args.height))
    for order, (label, points) in enumerate(series.items()):
        points = sorted(points)
        xs = [index_of[x] for x, _ in points]
        ys = [y for _, y in points]
        color = colors[order % len(colors)]
        ax.plot(xs, ys, marker="o", markersize=3, linewidth=1.2, color=color, label=label)
        # 线末端同色文字，替代图例框（与参考图一致，序号前缀便于和终端输出对照）。
        text = label if args.plain_labels else f"{order}:{label}"
        ax.text(xs[-1] + 0.1, ys[-1], text, color=color, fontsize=8, va="center", ha="left")

    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels([str(size) for size in sizes], rotation=45, fontsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_title(args.title, fontsize=10)
    ax.set_xlabel(args.xlabel, fontsize=9)
    ax.set_ylabel(args.ylabel, fontsize=9)
    ax.set_xlim(-0.3, len(sizes) - 1 + 1.0)  # 右侧留白，避免末端标签被裁掉
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    return fig


def main():
    args = ParseArgs()
    series = LoadSeries(args.csv, args.label_col, args.x_col, args.y_col)
    fig = Plot(series, args)
    out = args.out or os.path.splitext(args.csv[0])[0] + ".png"
    fig.savefig(out, dpi=args.dpi)
    print(f"saved {out}")
    for label, points in series.items():
        print(f"  {label:12s} {len(points)} points, x = {sorted(x for x, _ in points)}")


if __name__ == "__main__":
    main()
