"""jurivx© GPLv3 23.02.2026"""

import numpy as np
import matplotlib.pyplot as plt
import os




def load_images_from_folder(folder_path):
    images = []
    labels = []
    

    for digit in range(10):
        digit_path = os.path.join(folder_path, str(digit))
        files = [f for f in os.listdir(digit_path) if f.endswith(".jpg")]
        print("first file:", files[0])
        print("shape:", plt.imread(os.path.join(digit_path, files[0])).shape)
        for filename in files:
            img = plt.imread(os.path.join(digit_path, filename))
            img = img.flatten() / 255.0
            images.append(img)
            labels.append(digit)
    return np.array(images), np.array(labels)

# Return vector of length num_classes with 1 at the index of the number label
def one_hot_encode(labels, num_classes=10):
    # create matrix of length labels with num_classes columns, all values are 0
    one_hot = np.zeros((len(labels), num_classes))
    # set the value at the index of the label to 1
    one_hot[np.arange(len(labels)), labels] = 1.0
    print(one_hot.shape)

    return one_hot



def shuffle_data(X, y):
    indices = np.random.permutation(len(X))
    print(indices)
    return X[indices], y[indices]

def load_data(base_path="data/Reduced_MNIST_Data"):
    X_train, y_train = load_images_from_folder(
        os.path.join(base_path, "Reduced_Trainging_data")
    )
    X_test, y_test = load_images_from_folder(
        os.path.join(base_path, "Reduced_Testing_data")
    )

    y_train = one_hot_encode(y_train)
    y_test = one_hot_encode(y_test)

    X_train, y_train = shuffle_data(X_train, y_train)

    return X_train, y_train, X_test, y_test



data = load_data()
