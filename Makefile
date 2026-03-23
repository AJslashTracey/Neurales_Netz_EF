VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

.PHONY: install install-btop start-app train-kaggle test-loader plot-metrics clean

install:
	python -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

install-btop: ## Install btop with an available system package manager
	@if command -v brew >/dev/null 2>&1; then \
		brew install btop; \
	elif command -v apt-get >/dev/null 2>&1; then \
		sudo apt-get update && sudo apt-get install -y btop; \
	else \
		echo "No supported package manager found for installing btop."; \
		exit 1; \
	fi

start-app: ## Start the Tkinter digit app
	$(PYTHON) app.py --app --model-path models/kaggle_png_best.npz

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
