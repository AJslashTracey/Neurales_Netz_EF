import numpy as np

from network import NeuralNetwork

IMAGE_SIZE = 28
NUM_CLASSES = 10
DEFAULT_LABEL_NOISE_ESTIMATE = 0.02


def accuracy(y_true_one_hot: np.ndarray, y_pred_proba: np.ndarray) -> float:
    y_true = np.argmax(y_true_one_hot, axis=1)
    y_pred = np.argmax(y_pred_proba, axis=1)
    return float(np.mean(y_true == y_pred))


def _confusion_matrix(y_true_one_hot: np.ndarray, y_pred_proba: np.ndarray) -> np.ndarray:
    y_true = np.argmax(y_true_one_hot, axis=1)
    y_pred = np.argmax(y_pred_proba, axis=1)
    matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float32)
    np.add.at(matrix, (y_true, y_pred), 1.0)
    return matrix


def _normalize_confusion(matrix: np.ndarray) -> np.ndarray:
    row_sum = matrix.sum(axis=1, keepdims=True)
    row_sum = np.maximum(row_sum, 1e-8)
    return matrix / row_sum


def _per_class_accuracy(matrix: np.ndarray) -> np.ndarray:
    row_sum = np.maximum(matrix.sum(axis=1), 1e-8)
    return np.diag(matrix) / row_sum


def _class_counts(y_one_hot: np.ndarray) -> np.ndarray:
    return y_one_hot.sum(axis=0).astype(np.float32)


def _estimate_label_noise(
    y_true_one_hot: np.ndarray,
    y_pred_proba: np.ndarray,
    fallback: float = DEFAULT_LABEL_NOISE_ESTIMATE,
) -> float:
    y_true = np.argmax(y_true_one_hot, axis=1)
    true_class_prob = y_pred_proba[np.arange(len(y_true)), y_true]
    suspect_ratio = float(np.mean(true_class_prob < 0.2))
    estimate = max(fallback, suspect_ratio * 0.5)
    return float(np.clip(estimate, 0.0, 0.2))


def iterate_minibatches(
    X: np.ndarray, y: np.ndarray, batch_size: int
) -> tuple[np.ndarray, np.ndarray]:
    for start_idx in range(0, len(X), batch_size):
        end_idx = start_idx + batch_size
        yield X[start_idx:end_idx], y[start_idx:end_idx]


def _shift_image(img: np.ndarray, dx: int, dy: int) -> np.ndarray:
    shifted = np.roll(img, shift=(dy, dx), axis=(0, 1))
    if dy > 0:
        shifted[:dy, :] = 0.0
    elif dy < 0:
        shifted[dy:, :] = 0.0
    if dx > 0:
        shifted[:, :dx] = 0.0
    elif dx < 0:
        shifted[:, dx:] = 0.0
    return shifted


def augment_batch(X_batch: np.ndarray) -> np.ndarray:
    batch = X_batch.reshape(-1, IMAGE_SIZE, IMAGE_SIZE).copy()

    for i in range(batch.shape[0]):
        img = batch[i]

        # Random small translation improves robustness on hand-drawn digits.
        dx = np.random.randint(-2, 3)
        dy = np.random.randint(-2, 3)
        img = _shift_image(img, dx=dx, dy=dy)

        # Random contrast / stroke intensity variation.
        img *= np.random.uniform(0.85, 1.15)

        # Mild random thickening.
        if np.random.rand() < 0.35:
            thick = img.copy()
            thick = np.maximum(thick, np.roll(img, 1, axis=0))
            thick = np.maximum(thick, np.roll(img, -1, axis=0))
            thick = np.maximum(thick, np.roll(img, 1, axis=1))
            thick = np.maximum(thick, np.roll(img, -1, axis=1))
            img = thick

        # Small additive noise to prevent brittle decision boundaries.
        img += np.random.normal(0.0, 0.03, size=img.shape).astype(np.float32)
        batch[i] = np.clip(img, 0.0, 1.0)

    return batch.reshape(X_batch.shape[0], IMAGE_SIZE * IMAGE_SIZE).astype(np.float32)


def split_train_validation(
    X: np.ndarray, y: np.ndarray, val_split: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if val_split <= 0.0:
        return X, y, X.copy(), y.copy()
    val_size = max(1, int(len(X) * val_split))
    return X[val_size:], y[val_size:], X[:val_size], y[:val_size]


def train_model(
    model: NeuralNetwork,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int = 80,
    batch_size: int = 64,
    val_split: float = 0.1,
    use_augmentation: bool = True,
    lr_decay_step: int = 20,
    lr_decay_factor: float = 0.5,
    early_stopping_patience: int = 12,
) -> dict[str, list[float] | list[np.ndarray] | np.ndarray | float]:
    X_core, y_core, X_val, y_val = split_train_validation(X_train, y_train, val_split)

    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "test_acc": [],
        "learning_rate": [],
        "test_confusion_norm": [],
        "test_per_class_acc": [],
    }
    best_val_acc = -1.0
    best_epoch = 0
    best_params = model.get_parameters_copy()
    epochs_without_improvement = 0
    initial_lr = model.learning_rate

    for epoch in range(1, epochs + 1):
        if lr_decay_step > 0:
            decay_power = (epoch - 1) // lr_decay_step
            model.learning_rate = initial_lr * (lr_decay_factor**decay_power)

        indices = np.random.permutation(len(X_core))
        X_epoch = X_core[indices]
        y_epoch = y_core[indices]

        batch_losses: list[float] = []
        for X_batch, y_batch in iterate_minibatches(X_epoch, y_epoch, batch_size):
            if use_augmentation:
                X_batch = augment_batch(X_batch)
            loss = model.train_on_batch(X_batch, y_batch)
            batch_losses.append(loss)

        train_pred = model.predict_proba(X_core)
        val_pred = model.predict_proba(X_val)
        test_pred = model.predict_proba(X_test)
        avg_loss = float(np.mean(batch_losses))
        val_loss = float(model.compute_loss(y_val, val_pred))
        train_acc = accuracy(y_core, train_pred)
        val_acc = accuracy(y_val, val_pred)
        test_acc = accuracy(y_test, test_pred)
        test_conf = _confusion_matrix(y_test, test_pred)
        test_conf_norm = _normalize_confusion(test_conf)
        test_per_class_acc = _per_class_accuracy(test_conf)

        history["train_loss"].append(avg_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["test_acc"].append(test_acc)
        history["learning_rate"].append(model.learning_rate)
        history["test_confusion_norm"].append(test_conf_norm)
        history["test_per_class_acc"].append(test_per_class_acc)

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"lr={model.learning_rate:.5f} | "
            f"train_loss={avg_loss:.4f} | val_loss={val_loss:.4f} | "
            f"train_acc={train_acc:.4f} | val_acc={val_acc:.4f} | test_acc={test_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_params = model.get_parameters_copy()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping_patience:
                print(
                    f"Early stopping at epoch {epoch}. "
                    f"Best val_acc={best_val_acc:.4f} (epoch {best_epoch})."
                )
                break

    model.set_parameters(*best_params)
    train_pred_final = model.predict_proba(X_train)
    test_pred_final = model.predict_proba(X_test)
    final_test_acc = accuracy(y_test, test_pred_final)
    label_noise_estimate = _estimate_label_noise(y_train, train_pred_final)
    dataset_ceiling = float(np.clip(1.0 - label_noise_estimate, 0.0, 1.0))

    history["train_class_counts"] = _class_counts(y_train)
    history["test_class_counts"] = _class_counts(y_test)
    history["label_noise_estimate"] = label_noise_estimate
    history["dataset_ceiling"] = dataset_ceiling
    history["augmentation_flag"] = 1.0 if use_augmentation else 0.0
    history["architecture_depth"] = float(len(model.weights))
    history["parameter_count"] = float(
        sum(int(w.size + b.size) for w, b in zip(model.weights, model.biases))
    )
    history["best_epoch"] = float(best_epoch)
    history["best_val_acc"] = float(best_val_acc)
    history["final_test_acc"] = float(final_test_acc)

    print(
        f"Restored best model from epoch {best_epoch} with val_acc={best_val_acc:.4f}. "
        f"Estimated label noise={label_noise_estimate:.3f}, dataset ceiling~{dataset_ceiling:.3f}."
    )
    return history
