from data_loader import load_data
from network import NeuralNetwork
from train import train_model


def main() -> None:
    X_train, y_train, X_test, y_test = load_data()
    print(
        f"Loaded data: X_train={X_train.shape}, y_train={y_train.shape}, "
        f"X_test={X_test.shape}, y_test={y_test.shape}"
    )

    model = NeuralNetwork(
        input_dim=X_train.shape[1],
        hidden_dims=(128, 64),
        output_dim=y_train.shape[1],
        learning_rate=0.005,
        seed=42,
    )

    train_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        epochs=10,
        batch_size=64,
    )


if __name__ == "__main__":
    main()
