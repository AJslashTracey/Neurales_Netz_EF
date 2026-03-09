# Neuronales Netz (MNIST Reduced)

Dieses Projekt trainiert ein neuronales Netz (von Grund auf mit `numpy`) zur
Erkennung handgeschriebener Ziffern (0-9) auf einem reduzierten MNIST-Datensatz.

## Projektstruktur

- `data_loader.py` - Laden, Normalisieren, One-Hot-Encoding
- `activations.py` - ReLU, ReLU-Ableitung, Softmax
- `network.py` - Netzwerk-Klasse (Forward, Backward, Update)
- `train.py` - Trainingsloop, Accuracy, Plotly-Visualisierungen
- `main.py` - Einstiegspunkt
- `plots/` - Ausgabeordner fuer HTML-Visualisierungen

## Setup

```bash
pip install -r requirements.txt
```

## Starten

```bash
python main.py
```

Danach werden folgende Dateien erzeugt:

- `plots/network_architecture.html`
- `plots/training_curves.html`
- `plots/confusion_matrix.html`