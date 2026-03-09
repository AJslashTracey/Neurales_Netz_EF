import numpy as np

from network import NeuralNetwork


def accuracy(y_true_one_hot: np.ndarray, y_pred_proba: np.ndarray) -> float:
    y_true = np.argmax(y_true_one_hot, axis=1)
    y_pred = np.argmax(y_pred_proba, axis=1)
    return float(np.mean(y_true == y_pred))


def iterate_minibatches(
    X: np.ndarray, y: np.ndarray, batch_size: int
) -> tuple[np.ndarray, np.ndarray]:
    for start_idx in range(0, len(X), batch_size):
        end_idx = start_idx + batch_size
        yield X[start_idx:end_idx], y[start_idx:end_idx]


def train_model(
    model: NeuralNetwork,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int = 10,
    batch_size: int = 64,
) -> dict[str, list[float]]:
    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_acc": [],
        "test_acc": [],
    }

    for epoch in range(1, epochs + 1):
        indices = np.random.permutation(len(X_train))
        X_epoch = X_train[indices]
        y_epoch = y_train[indices]

        batch_losses: list[float] = []
        for X_batch, y_batch in iterate_minibatches(X_epoch, y_epoch, batch_size):
            loss = model.train_on_batch(X_batch, y_batch)
            batch_losses.append(loss)

        train_pred = model.predict_proba(X_train)
        test_pred = model.predict_proba(X_test)
        avg_loss = float(np.mean(batch_losses))
        train_acc = accuracy(y_train, train_pred)
        test_acc = accuracy(y_test, test_pred)

        history["train_loss"].append(avg_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"loss={avg_loss:.4f} | train_acc={train_acc:.4f} | test_acc={test_acc:.4f}"
        )

    return history
