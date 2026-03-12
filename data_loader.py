import os

import matplotlib.pyplot as plt
import numpy as np

SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png")


def load_images_from_folder(folder_path: str) -> tuple[np.ndarray, np.ndarray]:
    images: list[np.ndarray] = []
    labels: list[int] = []

    for digit in range(10):
        digit_path = os.path.join(folder_path, str(digit))
        files = sorted(
            f for f in os.listdir(digit_path) if f.lower().endswith(SUPPORTED_EXTENSIONS)
        )
        for filename in files:
            img = plt.imread(os.path.join(digit_path, filename))
            img = img.astype(np.float32)
            if img.ndim == 3:
                # Keep compatibility with possible RGB/RGBA inputs.
                img = img[..., 0]
            max_val = float(np.max(img))
            if max_val > 1.0:
                img = img / 255.0
            img = np.clip(img, 0.0, 1.0).flatten()
            images.append(img)
            labels.append(digit)

    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int64)


def one_hot_encode(labels: np.ndarray, num_classes: int = 10) -> np.ndarray:
    one_hot = np.zeros((len(labels), num_classes), dtype=np.float32)
    one_hot[np.arange(len(labels)), labels] = 1.0
    return one_hot


def shuffle_data(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    indices = np.random.permutation(len(X))
    return X[indices], y[indices]


def load_data(
    train_dir: str = "data/Reduced_MNIST_Data/Reduced_Trainging_data",
    test_dir: str = "data/Reduced_MNIST_Data/Reduced_Testing_data",
) -> tuple[np.ndarray, ...]:
    X_train, y_train = load_images_from_folder(train_dir)
    X_test, y_test = load_images_from_folder(test_dir)
    if X_train.size == 0 or X_test.size == 0:
        raise ValueError(
            "No images were loaded. Check paths and ensure class folders 0-9 contain "
            f"files with extensions: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    y_train = one_hot_encode(y_train)
    y_test = one_hot_encode(y_test)
    X_train, y_train = shuffle_data(X_train, y_train)

    # Basic shape checks make data errors fail early.
    assert X_train.ndim == 2 and X_test.ndim == 2
    assert y_train.ndim == 2 and y_test.ndim == 2
    assert X_train.shape[1] == X_test.shape[1]
    assert y_train.shape[1] == 10 and y_test.shape[1] == 10

    return X_train, y_train, X_test, y_test
