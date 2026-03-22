"""Generate a comparison chart for saved model metrics."""
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

matplotlib.use("Agg")

MODELS_DIR = Path("models")
SUMMARY_PATH = Path("docs/accuracy_summary.txt")
CHART_PATH = Path("docs/model_accuracy_comparison.png")


def _display_name(path: Path) -> str:
    name = path.name.removesuffix(".metrics.npz")
    return name.removesuffix(".npz")


def _extract_scalar(data: dict[str, np.ndarray], key: str) -> float | None:
    value = data.get(key)
    if value is None or np.size(value) == 0:
        return None
    return float(np.ravel(value)[0])


def _collect_rows() -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for metrics_path in sorted(MODELS_DIR.glob("*.metrics.npz")):
        data = {key: value for key, value in np.load(metrics_path).items()}
        val_acc = data.get("val_acc")
        test_acc = data.get("test_acc")
        if val_acc is None or test_acc is None or len(val_acc) == 0 or len(test_acc) == 0:
            continue

        best_idx = int(np.nanargmax(val_acc))
        rows.append(
            {
                "model": _display_name(metrics_path),
                "epochs": int(len(val_acc)),
                "best_epoch": best_idx + 1,
                "best_val_acc": float(val_acc[best_idx]),
                "test_at_best": float(test_acc[best_idx]),
                "final_test_acc": _extract_scalar(data, "final_test_acc") or float(test_acc[-1]),
            }
        )
    return sorted(rows, key=lambda row: float(row["test_at_best"]), reverse=True)


def _write_summary(rows: list[dict[str, float | int | str]]) -> None:
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = ["Model, Epochs, Best Epoch, Best Val Acc, Test Acc At Best, Final Test Acc"]
    for row in rows:
        lines.append(
            f"{row['model']}: epochs={row['epochs']}, best_epoch={row['best_epoch']}, "
            f"best_val={row['best_val_acc']:.4f}, test_at_best={row['test_at_best']:.4f}, "
            f"final_test={row['final_test_acc']:.4f}"
        )
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot(rows: list[dict[str, float | int | str]]) -> None:
    labels = [str(row["model"]) for row in rows]
    best_val = [float(row["best_val_acc"]) * 100 for row in rows]
    test_at_best = [float(row["test_at_best"]) * 100 for row in rows]
    fig = Figure(figsize=(10, 5.5), facecolor="white")
    FigureCanvasAgg(fig)

    fig.text(0.5, 0.95, "Model Accuracy Comparison", ha="center", va="top", fontsize=16, weight="bold")
    fig.text(0.08, 0.9, "Accuracy (%)", ha="left", va="top", fontsize=11, color="#333333")

    plot_left = 0.08
    plot_bottom = 0.18
    plot_width = 0.84
    plot_height = 0.64
    group_width = plot_width / max(len(rows), 1)
    bar_width = group_width * 0.24
    bar_gap = group_width * 0.08

    for pct in range(0, 101, 20):
        y = plot_bottom + plot_height * (pct / 100.0)
        fig.add_artist(
            Rectangle(
                (plot_left, y),
                plot_width,
                0.0015,
                transform=fig.transFigure,
                facecolor="#d9d9d9",
                edgecolor="none",
                alpha=0.8,
            )
        )
        fig.text(plot_left - 0.015, y, f"{pct}", ha="right", va="center", fontsize=9, color="#666666")

    for idx, label in enumerate(labels):
        group_center = plot_left + group_width * (idx + 0.5)
        left_x = group_center - (bar_width + bar_gap / 2)
        right_x = group_center + bar_gap / 2
        bars = [
            (left_x, best_val[idx], "#2f6db3", f"{best_val[idx]:.2f}%"),
            (right_x, test_at_best[idx], "#e67e22", f"{test_at_best[idx]:.2f}%"),
        ]
        for x_pos, value, color, text in bars:
            height = plot_height * (value / 100.0)
            fig.add_artist(
                Rectangle(
                    (x_pos, plot_bottom),
                    bar_width,
                    height,
                    transform=fig.transFigure,
                    facecolor=color,
                    edgecolor="none",
                )
            )
            fig.text(
                x_pos + bar_width / 2,
                plot_bottom + height + 0.015,
                text,
                ha="center",
                va="bottom",
                fontsize=9,
                color="#222222",
            )
        fig.text(group_center, 0.12, label, ha="center", va="top", fontsize=10, color="#333333")

    fig.text(0.74, 0.89, "Best validation accuracy", ha="left", va="center", fontsize=10, color="#2f6db3")
    fig.text(0.74, 0.85, "Test accuracy at best epoch", ha="left", va="center", fontsize=10, color="#e67e22")
    CHART_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(CHART_PATH, dpi=160, facecolor=fig.get_facecolor())


def main() -> None:
    rows = _collect_rows()
    if not rows:
        raise SystemExit("No compatible metrics archives found under models/.")
    _write_summary(rows)
    _plot(rows)
    print(f"Wrote summary to {SUMMARY_PATH}")
    print(f"Wrote chart to {CHART_PATH}")


if __name__ == "__main__":
    main()
