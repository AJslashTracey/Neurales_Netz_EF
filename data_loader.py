import os

import matplotlib.pyplot as plt
import numpy as np


def load_images_from_folder(folder_path):
    images = []
    labels = []

    for digit in range(10):
        digit_path = os.path.join(folder_path, str(digit))
        files = sorted([f for f in os.listdir(digit_path) if f.endswith(".jpg")])

        for filename in files:
            img = plt.imread(os.path.join(digit_path, filename)).astype(np.float64)
            if img.ndim == 3:
                img = img.mean(axis=2)
            if img.max() > 1.0:
                img = img / 255.0
            img = img.flatten()

            images.append(img)
            labels.append(digit)

    return np.array(images, dtype=np.float64), np.array(labels, dtype=np.int64)


def one_hot_encode(labels, num_classes=10):
    one_hot = np.zeros((len(labels), num_classes), dtype=np.float64)
    one_hot[np.arange(len(labels)), labels] = 1.0
    return one_hot


def shuffle_data(X, y):
    indices = np.random.permutation(len(X))
    return X[indices], y[indices]


def load_data(base_path="data/Reduced_MNIST_Data", shuffle_train=True):
    X_train, y_train = load_images_from_folder(
        os.path.join(base_path, "Reduced_Trainging_data")
    )
    X_test, y_test = load_images_from_folder(
        os.path.join(base_path, "Reduced_Testing_data")
    )

    y_train = one_hot_encode(y_train)
    y_test = one_hot_encode(y_test)

    if shuffle_train:
        X_train, y_train = shuffle_data(X_train, y_train)

    return X_train, y_train, X_test, y_test
