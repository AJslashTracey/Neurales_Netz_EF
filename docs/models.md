# Model Catalogue

This repository keeps three main checkpoints plus their metrics archives. They are all saved under `models/` so you can load whichever version suits your experiment. The list below summarizes where they came from, how they were trained, and what accuracy you can expect:

| File | Dataset | Epochs | Validation Acc | Test Acc | Notes |
| --- | --- | --- | --- | --- | --- |
| `models/initial_model.npz` | `data/Reduced_MNIST_Data` (default split) | 80 | ~0.90 | ~0.91 | Baseline shipped with the original project. Use it for fast prototypes or when you don’t need to retrain. The metrics `models/initial_model.npz.metrics.npz` holds the history for replay.
| `models/fedora_chip_test_1.npz` | Reduced MNIST split with a larger architecture (`784 -> 256 -> 128 -> 64 -> 10`) and data augmentation | 72 | 0.925 | 0.941 | The first “deep” experiment that introduced augmentation, LR decay, and early stopping. Keep it for comparison to see how added structure helps despite the smaller dataset.
| `models/fedora_chip_test_2.npz` | `data/Reduced_MNIST_Data/Reduced_Testing_data` used for both training & testing | 52 | 0.900 | 0.874 | Demonstrates the dataset ceiling when training and testing come from the same small folder—the model fits but generalizes less well. Useful for controlled error analyses.
| `models/kaggle_mnist_full.npz` | `data/kaggle_mnist/mnist_png` (60k train + 10k test PNG MNIST) | 75 | 0.9705 | 0.9736 | Current champion. Uses the PNG-aware loader that stabilizes grayscale/normalization. The metrics file stores per-epoch losses/accuracies + LR schedule. Use this for the Tkinter app or any production inference.

Each checkpoint lives next to a `.metrics.npz` companion (e.g., `models/kaggle_mnist_full.npz.metrics.npz`) that stores arrays such as `train_loss`, `val_acc`, `test_acc`, and `learning_rate`. Run `python scripts/plot_metrics.py` or inspect `docs/figures/accuracy_summary.txt` to compare the training histories without retraining.
