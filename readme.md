# Neuronales Netz

This project is a NumPy neural network that reads handwritten digits (MNIST).  
It also has a Tkinter app where you can draw a digit and see the prediction live.

## Setup

Install the required packages:

```bash
pip install -r requirements.txt
```

## Run the app

```bash
python app.py --app
```

If no model is loaded, the app will ask you to pick a `.npz` model file.

## Model results

Model files are in `models/`.

| Model | Trained on dataset | Train acc | Val acc | Test acc | Simple note |
| --- | --- | --- | --- | --- | --- |
| `models/kaggle_mnist_full.npz` | `data/kaggle_mnist/mnist_png` (60k train / 10k test) | 0.9737 | 0.9697 | 0.9743 | Best all-round model. Great default choice. |
| `models/M4_pro_showcase_training.npz` | `data/kaggle_mnist/mnist_png` (same split/settings) | 0.9729 | 0.9722 | 0.9725 | Very close to best model. Used in showcase image. |
| `models/initial_model.npz` | Older reduced-MNIST workflow (`data/Reduced_MNIST_Data` style) | - | - | - | Older baseline model. No metrics file included. |

Extra training info from metrics files:
- Network shape: `784 -> 256 -> 128 -> 64 -> 10`
- Depth: 4 layers
- Parameters: about 242,762
- Estimated dataset ceiling: about 0.98
- Data augmentation: enabled

## How training works

The training code is in `train.py` and launched from `app.py`/`main.py`.

Main steps:
- mini-batch training
- cross-entropy loss + L2 weight decay
- validation split + early stopping
- learning-rate step decay
- data augmentation (small shifts, brightness/stroke change, mild thickening, noise)
- saves train/val/test metrics every epoch

Example training command:

```bash
python app.py --train --epochs 80 --batch-size 64 --learning-rate 0.005 --hidden-dims 256,128,64 --weight-decay 1e-4 --lr-decay-step 20 --lr-decay-factor 0.5 --patience 12 --model-path models/M4_pro_showcase_training.npz
```

## What the app shows

When you run `python app.py --app`, you can see:
- 28x28 drawing grid
- top prediction + top-3 predictions
- probability chart for digits 0 to 9
- processed 28x28 preview image
- confidence values (top score, top-2 gap, certainty score)
- confidence trend from recent predictions
- buttons to load a model and train from inside the app
- extra training dashboards if a `.metrics.npz` file exists

## App showcase

<img src="assets/media/app_show_case.png" alt="Digit predictor app showcase" width="1200" />

## Demos

Training demo:

<img src="assets/media/training_model.gif" alt="Training demo" width="1200" />

Model usage demo:

<img src="assets/media/showing_model.gif" alt="Model usage demo" width="1200" />
