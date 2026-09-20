"""Neural network tests.

The gradient check is the important one: hand-written backprop is easy to get
subtly wrong, and a wrong gradient still trains (just badly). Comparing
analytic gradients against finite differences is the standard way to prove the
backward pass matches the forward pass.
"""

import numpy as np
import pytest

from dormant_radar.nn import Adam, Dense, NeuralNet, load_network, save_network


def test_forward_shapes():
    net = NeuralNet([5, 8, 3], seed=1)
    out = net.predict(np.zeros((4, 5)))
    assert out.shape == (4, 3)


def test_forward_is_deterministic_given_seed():
    a = NeuralNet([4, 6, 4], seed=3).predict(np.ones((2, 4)))
    b = NeuralNet([4, 6, 4], seed=3).predict(np.ones((2, 4)))
    assert np.allclose(a, b)


def test_dense_backward_matches_finite_differences():
    """Analytic gradient of a single layer must match numerical gradient."""
    rng = np.random.default_rng(11)
    layer = Dense(4, 3, rng, activation="tanh")
    x = rng.normal(size=(6, 4))
    grad_output = rng.normal(size=(6, 3))

    layer.forward(x)
    layer.backward(grad_output)
    analytic_w = layer.grad_weight.copy()
    analytic_b = layer.grad_bias.copy()

    eps = 1e-6
    # Snapshot the originals: the numeric loop mutates the layer in place, and
    # reusing a perturbed weight while probing the bias would corrupt the probe.
    original_w = layer.weight.copy()
    original_b = layer.bias.copy()

    def loss_with(weight, bias):
        layer.weight = weight
        layer.bias = bias
        return float(np.sum(layer.forward(x) * grad_output))

    numeric_w = np.zeros_like(analytic_w)
    for i in range(analytic_w.shape[0]):
        for j in range(analytic_w.shape[1]):
            plus = original_w.copy()
            plus[i, j] += eps
            minus = original_w.copy()
            minus[i, j] -= eps
            numeric_w[i, j] = (loss_with(plus, original_b) - loss_with(minus, original_b)) / (2 * eps)

    numeric_b = np.zeros_like(analytic_b)
    for j in range(analytic_b.shape[0]):
        plus = original_b.copy()
        plus[j] += eps
        minus = original_b.copy()
        minus[j] -= eps
        numeric_b[j] = (loss_with(original_w, plus) - loss_with(original_w, minus)) / (2 * eps)

    assert np.allclose(analytic_w, numeric_w, atol=1e-6)
    assert np.allclose(analytic_b, numeric_b, atol=1e-6)


def test_network_gradient_matches_finite_differences():
    """Same check for the whole multi-layer network, including the l2 loss."""
    rng = np.random.default_rng(5)
    net = NeuralNet([3, 4, 2], seed=2)
    x = rng.normal(size=(5, 3))
    target = rng.normal(size=(5, 2))

    net.loss_and_grad(x, target)
    analytic = [g.copy() for g in net.grads()]

    eps = 1e-6

    def loss_at():
        prediction = net.forward(x)
        residual = prediction - target
        return float(np.mean(residual**2))

    numeric: list[np.ndarray] = []
    for param in net.params():
        grad = np.zeros_like(param)
        flat = param.reshape(-1)
        for index in range(flat.size):
            original = flat[index]
            flat[index] = original + eps
            plus = loss_at()
            flat[index] = original - eps
            minus = loss_at()
            flat[index] = original
            grad.reshape(-1)[index] = (plus - minus) / (2 * eps)
        numeric.append(grad)

    for a, n in zip(analytic, numeric):
        assert np.allclose(a, n, atol=1e-5), "analytic gradient diverges from numeric"


def test_training_reduces_loss():
    """An autoencoder should reconstruct its training data better after training."""
    rng = np.random.default_rng(0)
    # Low-dimensional structure the network can actually learn.
    latent = rng.normal(size=(200, 2))
    data = latent @ rng.normal(size=(2, 6))

    net = NeuralNet([6, 5, 2, 5, 6], seed=4)
    opt = Adam(net.params(), lr=5e-3)

    first = None
    for _ in range(300):
        loss = net.loss_and_grad(data, data)
        opt.step(net.grads())
        if first is None:
            first = loss
    final = net.loss_and_grad(data, data)

    assert final < first * 0.6, f"loss barely moved: {first} -> {final}"


def test_adam_reduces_a_quadratic():
    """Sanity-check the optimiser on a trivial convex problem."""
    target = np.array([3.0, -2.0])
    param = np.zeros(2)
    opt = Adam([param], lr=0.1)
    for _ in range(500):
        grad = 2 * (param - target)
        opt.step([grad])
    assert np.allclose(param, target, atol=1e-3)


def test_save_and_load_roundtrip(tmp_path):
    net = NeuralNet([5, 6, 5], seed=9)
    x = np.random.default_rng(1).normal(size=(3, 5))
    before = net.predict(x)

    path = str(tmp_path / "model.npz")
    save_network(net, path, extra={"mean": np.arange(5.0), "std": np.ones(5)})

    restored, extra = load_network(path)
    after = restored.predict(x)

    assert np.allclose(before, after)
    assert np.allclose(extra["mean"], np.arange(5.0))
    assert restored.sizes == [5, 6, 5]


def test_unknown_activation_rejected():
    rng = np.random.default_rng(0)
    layer = Dense(2, 2, rng, activation="sigmoid")
    with pytest.raises(ValueError):
        layer.forward(np.zeros((1, 2)))


def test_sizes_must_allow_a_layer():
    with pytest.raises(ValueError):
        NeuralNet([4])