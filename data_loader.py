import os

import matplotlib.pyplot as plt
import numpy as np


def load_images_from_folder(folder_path: str) -> tuple[np.ndarray, np.ndarray]:
    images: list[np.ndarray] = []
    labels: list[int] = []

    for digit in range(10):
        digit_path = os.path.join(folder_path, str(digit))
        files = sorted(f for f in os.listdir(digit_path) if f.endswith(".jpg"))
        for filename in files:
            img = plt.imread(os.path.join(digit_path, filename))
            img = img.astype(np.float32).flatten() / 255.0
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


def load_data(base_path: str = "data/Reduced_MNIST_Data") -> tuple[np.ndarray, ...]:
    X_train, y_train = load_images_from_folder(
        os.path.join(base_path, "Reduced_Trainging_data")
    )
    X_test, y_test = load_images_from_folder(os.path.join(base_path, "Reduced_Testing_data"))

    y_train = one_hot_encode(y_train)
    y_test = one_hot_encode(y_test)
    X_train, y_train = shuffle_data(X_train, y_train)

    # Basic shape checks make data errors fail early.
    assert X_train.ndim == 2 and X_test.ndim == 2
    assert y_train.ndim == 2 and y_test.ndim == 2
    assert X_train.shape[1] == X_test.shape[1]
    assert y_train.shape[1] == 10 and y_test.shape[1] == 10

    return X_train, y_train, X_test, y_test
