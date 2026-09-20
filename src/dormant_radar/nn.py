"""A small dense neural network implemented directly in NumPy.

Deliberately dependency-light: no PyTorch or TensorFlow. The whole
architecture is an autoencoder trained to reconstruct *ordinary* Bitcoin
spends, so the model is small enough that hand-written forward and backward
passes are both fast and fully inspectable. Correctness of the backward pass
is pinned by a finite-difference gradient check in the test suite.

Architecture: input -> [Dense(tanh) ...] -> Dense(tanh) -> Dense(linear)
The final layer is linear because the target is a continuous feature vector.
"""

from __future__ import annotations

import numpy as np


class Dense:
    """A fully connected layer with optional tanh activation."""

    def __init__(self, n_in: int, n_out: int, rng: np.random.Generator, activation: str = "tanh"):
        # Xavier/Glorot scaling keeps activations in a healthy range at init.
        limit = np.sqrt(6.0 / (n_in + n_out))
        self.weight = rng.uniform(-limit, limit, size=(n_in, n_out))
        self.bias = np.zeros(n_out)
        self.activation = activation
        # Cached during forward for the backward pass.
        self._input: np.ndarray | None = None
        self._pre_activation: np.ndarray | None = None
        # Gradients, populated by backward.
        self.grad_weight: np.ndarray | None = None
        self.grad_bias: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._input = x
        z = x @ self.weight + self.bias
        self._pre_activation = z
        if self.activation == "tanh":
            return np.tanh(z)
        if self.activation == "linear":
            return z
        raise ValueError(f"unknown activation: {self.activation}")

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Return the gradient with respect to the input, storing parameter grads."""
        if self.activation == "tanh":
            # d tanh(z) / dz = 1 - tanh(z)^2
            local = 1.0 - np.tanh(self._pre_activation) ** 2
            grad_pre = grad_output * local
        else:
            grad_pre = grad_output
        self.grad_weight = self._input.T @ grad_pre
        self.grad_bias = grad_pre.sum(axis=0)
        return grad_pre @ self.weight.T

    def params(self) -> list[np.ndarray]:
        return [self.weight, self.bias]

    def grads(self) -> list[np.ndarray]:
        return [self.grad_weight, self.grad_bias]


class NeuralNet:
    """A feed-forward network with l2 loss over a continuous target."""

    def __init__(self, sizes: list[int], seed: int = 0):
        if len(sizes) < 2:
            raise ValueError("need at least an input and an output size")
        rng = np.random.default_rng(seed)
        self.layers: list[Dense] = []
        for index in range(len(sizes) - 1):
            last = index == len(sizes) - 2
            self.layers.append(
                Dense(
                    sizes[index],
                    sizes[index + 1],
                    rng,
                    activation="linear" if last else "tanh",
                )
            )
        self.sizes = sizes

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad_output: np.ndarray) -> None:
        grad = grad_output
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def params(self) -> list[np.ndarray]:
        return [p for layer in self.layers for p in layer.params()]

    def grads(self) -> list[np.ndarray]:
        return [g for layer in self.layers for g in layer.grads()]

    def loss_and_grad(self, x: np.ndarray, target: np.ndarray) -> float:
        """Mean squared error over every element, and its parameter gradient.

        The mean is taken across all elements (batch and output dims), so the
        gradient with respect to the prediction is 2 * residual / residual.size.
        Dividing by only the batch size would silently scale gradients by the
        output width; the finite-difference check in the test suite guards this.
        """
        prediction = self.forward(x)
        residual = prediction - target
        loss = float(np.mean(residual**2))
        self.backward(2.0 * residual / residual.size)
        return loss

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


class Adam:
    """Adam optimiser, matching the reference algorithm's defaults."""

    def __init__(self, params: list[np.ndarray], lr: float = 1e-3,
                 beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads: list[np.ndarray]) -> None:
        self.t += 1
        for i, (param, grad) in enumerate(zip(self.params, grads)):
            if grad is None:
                continue
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad * grad
            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self) -> None:
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]
        self.t = 0


def save_network(net: NeuralNet, path: str, extra: dict | None = None) -> None:
    """Persist weights and any normalisation metadata alongside them."""
    payload: dict[str, object] = {"sizes": net.sizes}
    for index, layer in enumerate(net.layers):
        payload[f"W{index}"] = layer.weight
        payload[f"b{index}"] = layer.bias
        payload[f"act{index}"] = layer.activation
    if extra:
        for key, value in extra.items():
            payload[key] = value
    np.savez(path, **payload)


def load_network(path: str) -> tuple[NeuralNet, dict]:
    """Restore a network and any metadata saved with it."""
    data = np.load(path, allow_pickle=False)
    sizes = [int(s) for s in data["sizes"]]
    net = NeuralNet(sizes)
    for index, layer in enumerate(net.layers):
        layer.weight = data[f"W{index}"]
        layer.bias = data[f"b{index}"]
        layer.activation = str(data[f"act{index}"])
    extra = {
        key: data[key]
        for key in data.files
        if not key.startswith(("W", "b", "act")) and key != "sizes"
    }
    return net, extra