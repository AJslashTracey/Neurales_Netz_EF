VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

.PHONY: install train-kaggle test-loader plot-metrics clean

install:
	python -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

train-kaggle: ## Train the best Kaggle MNIST checkpoint
	$(PYTHON) main.py --train \
		--train-data-path data/kaggle_mnist/mnist_png/train \
		--test-data-path data/kaggle_mnist/mnist_png/test \
		--model-path models/kaggle_mnist_full.npz

test-loader: ## Smoke-test the data loader
	$(PYTHON) tests/test_loader.py

plot-metrics: docs/accuracy_summary.txt ## Summarize metrics
	$(PYTHON) scripts/plot_metrics.py

clean:
	rm -rf $(VENV)
	rm -f docs/accuracy_summary.txt
