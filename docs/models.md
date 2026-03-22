# Model Catalogue

This repository keeps three main checkpoints plus their metrics archives. They are all saved under `models/` so you can load whichever version suits your experiment. The list below summarizes where they came from, how they were trained, and what accuracy you can expect:

| File | Dataset | Epochs | Validation Acc | Test Acc | Notes |
| --- | --- | --- | --- | --- | --- |
| `models/baseline_reduced_mnist.npz` | `data/Reduced_MNIST_Data` (default split) | 80 | ~0.90 | ~0.91 | Baseline shipped with the original project. Use it for fast prototypes or when you don’t need to retrain. The metrics `models/baseline_reduced_mnist.npz.metrics.npz` hold the history.
| `models/reduced_mnist_augmented.npz` | Reduced MNIST split with a larger architecture (`784 -> 256 -> 128 -> 64 -> 10`) and data augmentation | 72 | 0.925 | 0.941 | Deep architecture plus LR decay and early stopping—keep it for comparison to show how more structure helps.
| `models/reduced_mnist_test_only.npz` | `data/Reduced_MNIST_Data/Reduced_Testing_data` used for both training & testing | 52 | 0.900 | 0.874 | Demonstrates the dataset ceiling when training and testing come from the same small folder.
| `models/kaggle_png_best.npz` | `data/kaggle_mnist/mnist_png` (60k train + 10k test PNG MNIST) | 111 | 0.9788 | 0.9783 | Current champion trained with the PNG-aware loader; restored best epoch 95 after early stopping. Metrics are in `models/kaggle_png_best.npz.metrics.npz`.

Each checkpoint lives next to a `.metrics.npz` companion that stores `train_loss`, `val_acc`, `test_acc`, and `learning_rate` per epoch. Run `python scripts/plot_metrics.py` or inspect `docs/accuracy_summary.txt` to compare the training histories without retraining.
