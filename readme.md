# Neuronales Netz

Simple NumPy-based feed-forward neural network for reduced MNIST digit classification.

*Test change added so you can see a diff.*

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run Training

```bash
python main.py
```

The script loads data from `data/Reduced_MNIST_Data`, trains a model with architecture
`784 -> 128 -> 64 -> 10`, and prints epoch-wise loss and accuracy.

Current default training uses:
- larger architecture: `784 -> 256 -> 128 -> 64 -> 10`
- on-the-fly data augmentation (shift, intensity variation, mild thickening, noise)
- validation split + early stopping
- learning-rate decay
- L2 weight decay

## Command-line flags
Run `python main.py` with the flags below to control its behavior:

- `--train` (default) runs training on the reduced MNIST split and saves the model to `model.npz` unless you override `--model-path`.
- `--app` opens the Tkinter drawing app. Use it together with `--model-path` if you already trained a model.
- `--epochs`, `--batch-size`, `--learning-rate`, `--weight-decay`, `--val-split`, `--hidden-dims` and the learning-rate decay/patience flags let you tweak the training loop.
- `--no-augment` disables the built-in augmentation pipeline, which is on by default.
- `--debug-app` prints extra startup logs when the applet loads, helpful if Tkinter is missing or failing.

## Save Model + Launch Drawing Applet

Train and save a model:

```bash
python main.py --train --model-path model.npz
```

High-quality training example:

```bash
python main.py --train --epochs 80 --hidden-dims 256,128,64 --learning-rate 0.005 --lr-decay-step 20 --lr-decay-factor 0.5 --patience 12 --weight-decay 1e-4 --model-path model.npz
```

Launch the real-time Tkinter drawing app:

```bash
python main.py --app --model-path model.npz
```

Train and open the applet in one command:

```bash
python main.py --train --app --model-path model.npz
```
