import numpy as np

from activations import relu, relu_derivative, softmax


class NeuralNetwork:
    def __init__(
        self,
        input_dim: int = 784,
        hidden_dims: tuple[int, ...] = (128, 64),
        output_dim: int = 10,
        learning_rate: float = 0.01,
        grad_clip_value: float = 5.0,
        seed: int = 42,
    ) -> None:
        np.random.seed(seed)
        self.learning_rate = learning_rate
        self.grad_clip_value = grad_clip_value
        dims = (input_dim, *hidden_dims, output_dim)

        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []

        for idx in range(len(dims) - 1):
            fan_in = dims[idx]
            fan_out = dims[idx + 1]
            # He initialization works well with ReLU layers.
            w = np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in)
            b = np.zeros((1, fan_out), dtype=np.float32)
            self.weights.append(w.astype(np.float32))
            self.biases.append(b)

    def forward(self, X: np.ndarray) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
        activations = [X]
        zs: list[np.ndarray] = []
        a = X

        for layer_idx in range(len(self.weights) - 1):
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                z = a @ self.weights[layer_idx] + self.biases[layer_idx]
            a = relu(z)
            zs.append(z)
            activations.append(a)

        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            z_out = a @ self.weights[-1] + self.biases[-1]
        y_pred = softmax(z_out)
        zs.append(z_out)
        activations.append(y_pred)

        return y_pred, activations, zs

    @staticmethod
    def compute_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_pred = np.clip(y_pred, 1e-12, 1.0)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

    def backward(
        self, y_true: np.ndarray, activations: list[np.ndarray], zs: list[np.ndarray]
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        m = y_true.shape[0]
        grad_w: list[np.ndarray] = [np.zeros_like(w) for w in self.weights]
        grad_b: list[np.ndarray] = [np.zeros_like(b) for b in self.biases]

        dz = activations[-1] - y_true  # softmax + cross-entropy gradient

        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            grad_w[-1] = (activations[-2].T @ dz) / m
        grad_b[-1] = np.sum(dz, axis=0, keepdims=True) / m

        for layer_idx in range(len(self.weights) - 2, -1, -1):
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                dz = (dz @ self.weights[layer_idx + 1].T) * relu_derivative(zs[layer_idx])
                grad_w[layer_idx] = (activations[layer_idx].T @ dz) / m
            grad_b[layer_idx] = np.sum(dz, axis=0, keepdims=True) / m

        return grad_w, grad_b

    def update_parameters(self, grad_w: list[np.ndarray], grad_b: list[np.ndarray]) -> None:
        for idx in range(len(self.weights)):
            np.clip(
                grad_w[idx],
                -self.grad_clip_value,
                self.grad_clip_value,
                out=grad_w[idx],
            )
            np.clip(
                grad_b[idx],
                -self.grad_clip_value,
                self.grad_clip_value,
                out=grad_b[idx],
            )
            self.weights[idx] -= self.learning_rate * grad_w[idx]
            self.biases[idx] -= self.learning_rate * grad_b[idx]

    def train_on_batch(self, X_batch: np.ndarray, y_batch: np.ndarray) -> float:
        y_pred, activations, zs = self.forward(X_batch)
        loss = self.compute_loss(y_batch, y_pred)
        grad_w, grad_b = self.backward(y_batch, activations, zs)
        self.update_parameters(grad_w, grad_b)
        return loss

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        y_pred, _, _ = self.forward(X)
        return y_pred

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)
