"""Summarize saved metrics for key checkpoints."""
from pathlib import Path
import numpy as np

MODEL_METRICS = {
    "initial_model": "models/initial_model.npz.metrics.npz",
    "fedora_chip_test_1": "models/fedora_chip_test_1.npz.metrics.npz",
    "fedora_chip_test_2": "models/fedora_chip_test_2.npz.metrics.npz",
    "kaggle_mnist_full": "models/kaggle_mnist_full.npz.metrics.npz",
}
SUMMARY_DIR = Path("docs/figures")
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_PATH = SUMMARY_DIR / "accuracy_summary.txt"

lines = ["Epochs, Val Acc, Test Acc, Dataset"]
for label, path in MODEL_METRICS.items():
    path_obj = Path(path)
    if not path_obj.exists():
        continue
    data = dict(np.load(path_obj))
    val = data.get("val_acc")
    test = data.get("test_acc")
    if val is None or test is None:
        continue
    best_idx = int(np.nanargmax(val))
    lines.append(
        f"{label}: epochs={len(val)}, best_val={val[best_idx]:.4f}, test_at_best={test[best_idx]:.4f}"
    )
with SUMMARY_PATH.open("w", encoding="utf-8") as fh:
    fh.write("\n".join(lines) + "\n")
print(f"Written accuracy summary to {SUMMARY_PATH}")
