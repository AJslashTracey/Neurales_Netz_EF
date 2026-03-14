# `network.py` Documentation

This file defines a fully-connected feed-forward neural network in NumPy with:

- configurable input, hidden, and output layer sizes
- ReLU activations in hidden layers
- softmax output for multi-class classification
- cross-entropy loss
- L2 weight decay (regularization)
- gradient clipping
- model save/load helpers

---

## High-Level Flow

The training step is handled by `train_on_batch()` and follows this pipeline:

1. **Forward pass** (`forward`) computes predictions.
2. **Loss** (`compute_loss`) computes cross-entropy (+ optional L2 regularization).
3. **Backward pass** (`backward`) computes gradients with backpropagation.
4. **Update step** (`update_parameters`) clips gradients and updates weights/biases.

`predict_proba()` and `predict()` use the same forward pass but skip training updates.

---

## Class: `NeuralNetwork`

### `__init__(...)`

Purpose: create and initialize network parameters.

- Sets random seed for reproducibility.
- Stores optimizer-related hyperparameters:
  - `learning_rate`
  - `grad_clip_value`
  - `weight_decay`
- Builds layer dimensions as:
  - `(input_dim, *hidden_dims, output_dim)`
- Initializes:
  - `weights[idx]` with **He initialization**: `N(0, 2/fan_in)`
  - `biases[idx]` as zeros with shape `(1, fan_out)`
- Converts weights and biases to `float32` to reduce memory usage and keep dtype consistency.

---

### `forward(X)`

Purpose: run data through the network and return intermediate values needed for backprop.

Returns:

- `y_pred`: softmax probabilities for each class
- `activations`: list containing input and each layer activation
- `zs`: list of pre-activation linear outputs (`z = aW + b`)

Detailed behavior:

- Starts with `a = X` and stores `X` in `activations`.
- For each hidden layer:
  - computes linear transform: `z = a @ W + b`
  - applies ReLU: `a = relu(z)`
  - stores `z` and `a`
- For output layer:
  - computes `z_out = a @ W_out + b_out`
  - applies softmax: `y_pred = softmax(z_out)`
  - stores final `z_out` and `y_pred`

Note: `np.errstate(...)` is used around matrix operations to suppress runtime warnings (for divide/overflow/invalid) rather than interrupt execution.

---

### `compute_loss(y_true, y_pred)`

Purpose: compute objective value for optimization.

- Clips probabilities to `[1e-12, 1.0]` before log to avoid `log(0)`.
- Computes mean categorical cross-entropy:
  - `-mean(sum(y_true * log(y_pred), axis=1))`
- If `weight_decay > 0`, adds L2 penalty:
  - `0.5 * weight_decay * sum(||W||^2 over all layers)`

So final loss is:

- **cross-entropy only** when `weight_decay <= 0`
- **cross-entropy + L2 regularization** otherwise

---

### `backward(y_true, activations, zs)`

Purpose: compute gradients of loss w.r.t. all weights and biases.

Outputs:

- `grad_w`: list of weight gradients
- `grad_b`: list of bias gradients

Detailed behavior:

- Let `m` be batch size.
- Initializes gradient arrays with zeros matching parameter shapes.
- Output layer gradient uses softmax + cross-entropy simplification:
  - `dz = activations[-1] - y_true`
- Computes output layer gradients:
  - `grad_w[-1] = activations[-2].T @ dz / m`
  - `grad_b[-1] = sum(dz, axis=0) / m`
- For hidden layers (backward loop):
  - propagates gradient: `dz = (dz @ W_next.T) * relu_derivative(z_current)`
  - computes:
    - `grad_w[layer] = activations[layer].T @ dz / m`
    - `grad_b[layer] = sum(dz, axis=0) / m`
- If `weight_decay > 0`, adds L2 term to each `grad_w[layer]`:
  - `grad_w[layer] += weight_decay * W[layer]`

---

### `update_parameters(grad_w, grad_b)`

Purpose: apply one gradient descent update step.

- For each layer:
  - clips `grad_w` and `grad_b` element-wise to:
    - `[-grad_clip_value, +grad_clip_value]`
  - updates parameters:
    - `W -= learning_rate * grad_w`
    - `b -= learning_rate * grad_b`

Gradient clipping helps stabilize training when gradients explode.

---

### `train_on_batch(X_batch, y_batch)`

Convenience method that performs one full training step:

1. forward pass
2. loss computation
3. backward pass
4. parameter update

Returns the batch loss (float).

---

### Inference helpers

- `predict_proba(X)`: returns class probability distribution for each sample.
- `predict(X)`: returns class index with highest probability (`argmax`).

---

### Model persistence

#### `save_model(path)`

Stores all required information into a `.npz` file:

- hyperparameters (`learning_rate`, `grad_clip_value`, `weight_decay`)
- layer count (`num_layers`)
- each layer's `w_i` and `b_i`

#### `load_model(path)` (classmethod)

Reconstructs a `NeuralNetwork` from `.npz`:

- loads all layer parameters
- infers dimensions from loaded weight shapes
- restores saved hyperparameters (with defaults for backward compatibility)
- creates a new model instance and injects loaded weights/biases

---

### Parameter snapshot utilities

- `get_parameters_copy()`: returns deep copies of all weights/biases.
- `set_parameters(weights, biases)`: replaces current parameters with copies of provided ones.

These are useful for checkpointing, rollback, averaging, or federated workflows.

---

## Notes and Assumptions

- The network expects:
  - `X` as shape `(batch_size, input_dim)`
  - `y_true` as one-hot encoded labels with shape `(batch_size, output_dim)`
- Hidden activations are ReLU; output activation is softmax.
- Optimization is plain mini-batch gradient descent (no momentum/Adam in this file).
