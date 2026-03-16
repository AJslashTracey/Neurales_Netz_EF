# Neuronales Netz

NumPy-based feed-forward neural network for MNIST digit classification, with a Tkinter app for live digit drawing and prediction.

## Setup

Install dependencies (no `.venv` required):

```bash
pip install -r requirements.txt
```

## Run the app

Start the drawing app directly:

```bash
python app.py --app
```

If no model is loaded at startup, the app lets you choose a `.npz` checkpoint from disk.

## Model quality overview

Current checkpoints are under `models/`:

| Model | Dataset | Train acc | Val acc | Test acc | Notes |
| --- | --- | --- | --- | --- | --- |
| `models/kaggle_mnist_full.npz` | Kaggle MNIST PNG (60k train / 10k test) | 0.9737 | 0.9697 | 0.9743 | Strong general model, good default for inference and demos. |
| `models/M4_pro_showcase_training.npz` | Same Kaggle split/config | 0.9729 | 0.9722 | 0.9725 | Showcase checkpoint used in the app screenshot. |
| `models/initial_model.npz` | Legacy baseline checkpoint | - | - | - | Included for comparison; no `.metrics.npz` archive shipped for this one. |

Training metadata in metrics archives estimates:
- architecture depth: 4 layers (`784 -> 256 -> 128 -> 64 -> 10`)
- parameter count: ~242,762
- estimated dataset ceiling: ~0.98 (with label-noise estimate ~0.02)
- augmentation: enabled

## How the model is trained

Training pipeline (implemented in `train.py` and started via `main.py`/`app.py`):
- mini-batch training with cross-entropy + L2 weight decay
- validation split with early stopping (best checkpoint restored)
- learning-rate step decay
- data augmentation (random shifts, intensity variation, mild thickening, Gaussian noise)
- per-epoch tracking of train/val loss, train/val/test accuracy, confusion matrix, and per-class accuracy

Example full training command:

```bash
python app.py --train --epochs 80 --batch-size 64 --learning-rate 0.005 --hidden-dims 256,128,64 --weight-decay 1e-4 --lr-decay-step 20 --lr-decay-factor 0.5 --patience 12 --model-path models/M4_pro_showcase_training.npz
```

## What you can see in the app

`python app.py --app` opens a UI that includes:
- 28x28 drawing grid with optional snap-to-cell mode
- top-1 prediction plus top-3 ranked probabilities
- probability bar chart across digits 0-9
- processed 28x28 preview of your drawn input
- confidence diagnostics (top confidence, top-2 margin, certainty score)
- confidence trend graph across recent predictions
- model loading and in-app training controls
- training-quality dashboards when a `.metrics.npz` file exists (learning curves, confusion matrix, class coverage, dataset ceiling comparison)

## App showcase

![Digit predictor app showcase](assets/media/app_show_case.png)

## Demos

Training demo:

![Training demo](assets/media/training_model.gif)

Model usage demo:

![Model usage demo](assets/media/showing_model.gif)
