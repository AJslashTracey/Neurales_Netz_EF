import numpy as np
import os

from train import (
    plot_confusion_matrix,
    plot_network_architecture,
    plot_training_curves,
    train_model,
)


def main():
    np.random.seed(42)
    os.makedirs("plots", exist_ok=True)

    model, history, (X_test, y_test) = train_model(
        epochs=20,
        learning_rate=0.01,
        batch_size=64,
        data_path="data/Reduced_MNIST_Data",
    )

    plot_network_architecture("plots/network_architecture.html")
    plot_training_curves(history, "plots/training_curves.html")
    plot_confusion_matrix(model, X_test, y_test, "plots/confusion_matrix.html")

    print("Training complete.")
    print("Visualizations saved in: plots/")


if __name__ == "__main__":
    main()