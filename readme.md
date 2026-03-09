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