# Neuronales Netz

Simple NumPy-based feed-forward neural network for reduced MNIST digit classification.

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

## Save Model + Launch Drawing Applet

Train and save a model:

```bash
python main.py --train --model-path model.npz
```

Launch the real-time Tkinter drawing app:

```bash
python main.py --app --model-path model.npz
```

Train and open the applet in one command:

```bash
python main.py --train --app --model-path model.npz
```