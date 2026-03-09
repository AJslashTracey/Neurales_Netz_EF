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
- `plots/training_animation.html`
- `plots/confusion_matrix.html`
- `plots/digit_routes/digit_0_route.html` bis `digit_9_route.html`
- `plots/digit_routes_live/digit_0_live.gif` bis `digit_9_live.gif`
![Digit 0 Live Route](plots/digit_routes_live/digit_0_live.gif)
