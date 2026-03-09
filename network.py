import numpy as np

from activations import relu, relu_derivative, softmax


class NeuralNetwork:
    def __init__(self, input_size=784, hidden1_size=128, hidden2_size=64, output_size=10):
        self.W1 = np.random.randn(input_size, hidden1_size).astype(np.float64) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden1_size), dtype=np.float64)

        self.W2 = np.random.randn(hidden1_size, hidden2_size).astype(np.float64) * np.sqrt(2.0 / hidden1_size)
        self.b2 = np.zeros((1, hidden2_size), dtype=np.float64)

        self.W3 = np.random.randn(hidden2_size, output_size).astype(np.float64) * np.sqrt(2.0 / hidden2_size)
        self.b3 = np.zeros((1, output_size), dtype=np.float64)

    def forward(self, X):
        self.X = X
        with np.errstate(divide="ignore", invalid="ignore"):
            self.z1 = X @ self.W1 + self.b1
            self.a1 = relu(self.z1)

            self.z2 = self.a1 @ self.W2 + self.b2
            self.a2 = relu(self.z2)

            self.z3 = self.a2 @ self.W3 + self.b3
        self.y_hat = softmax(self.z3)
        return self.y_hat

    def compute_loss(self, y_hat, y_true):
        eps = 1e-9
        clipped = np.clip(y_hat, eps, 1.0 - eps)
        return -np.mean(np.sum(y_true * np.log(clipped), axis=1))

    def backward(self, y_true):
        batch_size = y_true.shape[0]

        with np.errstate(divide="ignore", invalid="ignore"):
            dz3 = (self.y_hat - y_true) / batch_size
            self.dW3 = self.a2.T @ dz3
            self.db3 = np.sum(dz3, axis=0, keepdims=True)

            da2 = dz3 @ self.W3.T
            dz2 = da2 * relu_derivative(self.z2)
            self.dW2 = self.a1.T @ dz2
            self.db2 = np.sum(dz2, axis=0, keepdims=True)

            da1 = dz2 @ self.W2.T
            dz1 = da1 * relu_derivative(self.z1)
            self.dW1 = self.X.T @ dz1
            self.db1 = np.sum(dz1, axis=0, keepdims=True)

    def update_params(self, learning_rate):
        self.W1 -= learning_rate * self.dW1
        self.b1 -= learning_rate * self.db1
        self.W2 -= learning_rate * self.dW2
        self.b2 -= learning_rate * self.db2
        self.W3 -= learning_rate * self.dW3
        self.b3 -= learning_rate * self.db3

    def predict_proba(self, X):
        return self.forward(X)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)
