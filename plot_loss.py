"""将实际训练日志画成曲线，留作后续报告素材。"""

import argparse
import csv
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="绘制小模型训练与验证损失")
    parser.add_argument("--log", type=Path, default=Path("runs/small/loss.csv"))
    parser.add_argument("--output", type=Path, default=Path("runs/small/loss.png"))
    args = parser.parse_args()
    # 字体缓存放在项目内，避免依赖用户目录的写权限。
    cache = Path("work/matplotlib").resolve()
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with args.log.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("损失日志为空")
    steps = [int(row["step"]) for row in rows]
    train_losses = [float(row["train_loss"]) for row in rows]
    validation_losses = [float(row["validation_loss"]) for row in rows]
    figure, axis = plt.subplots(figsize=(8, 4.8), dpi=160)
    axis.plot(steps, train_losses, "o-", label="Training (fixed batches)", color="#2463a6")
    axis.plot(steps, validation_losses, "s-", label="Validation (fixed batches)", color="#c56a22")
    axis.set(xlabel="Training step", ylabel="Cross-entropy loss", title="Small decoder language model")
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output)
    figure.savefig(args.output.with_suffix(".svg"))
    plt.close(figure)
    print(f"已保存 {args.output}")


if __name__ == "__main__":
    main()
