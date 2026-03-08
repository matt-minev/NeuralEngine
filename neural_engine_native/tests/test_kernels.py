import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from neural_engine_native.kernels import (
    activation_backward,
    activation_forward,
    adam_update,
    clip_by_global_norm,
    linear_backward,
    linear_forward,
    sgd_update,
    softmax_cross_entropy,
)


def test_linear_forward_matches_numpy():
    x = np.random.randn(8, 4).astype(np.float32)
    w = np.random.randn(3, 4).astype(np.float32)
    b = np.random.randn(3).astype(np.float32)
    y = linear_forward(x, w, b)
    np.testing.assert_allclose(y, x @ w.T + b, rtol=1e-5, atol=1e-5)


def test_linear_backward_shapes_and_values():
    x = np.random.randn(6, 4).astype(np.float32)
    w = np.random.randn(3, 4).astype(np.float32)
    dz = np.random.randn(6, 3).astype(np.float32)

    d_w, d_b, d_a = linear_backward(dz, x, w)

    np.testing.assert_allclose(d_w, dz.T @ x / 6.0, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(d_b, np.sum(dz, axis=0) / 6.0, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(d_a, dz @ w, rtol=1e-5, atol=1e-5)


def test_activation_forward_backward_relu():
    z = np.array([[-1.0, 0.0, 2.0]], dtype=np.float32)
    a = activation_forward(z, "relu")
    d = activation_backward(np.ones_like(a), z, a, "relu")
    np.testing.assert_array_equal(a, np.array([[0.0, 0.0, 2.0]], dtype=np.float32))
    np.testing.assert_array_equal(d, np.array([[0.0, 0.0, 1.0]], dtype=np.float32))


def test_softmax_cross_entropy_shapes():
    logits = np.random.randn(5, 7).astype(np.float32)
    y = np.zeros((5, 7), dtype=np.float32)
    y[np.arange(5), np.array([0, 1, 2, 3, 4])] = 1.0
    loss, probs, grad = softmax_cross_entropy(logits, y)
    assert np.isfinite(loss)
    assert probs.shape == logits.shape
    assert grad.shape == logits.shape
    np.testing.assert_allclose(np.sum(probs, axis=1), np.ones(5), atol=1e-5)


def test_optimizer_primitives():
    p = np.array([1.0, -2.0, 3.0], dtype=np.float32)
    g = np.array([0.1, -0.2, 0.3], dtype=np.float32)

    p_sgd, v = sgd_update(p, g, 0.01, momentum=0.9, velocity=np.zeros_like(p))
    assert p_sgd.shape == p.shape
    assert v.shape == p.shape

    p_adam, m, vv = adam_update(
        p,
        g,
        np.zeros_like(p),
        np.zeros_like(p),
        learning_rate=0.001,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        step_count=1,
    )
    assert p_adam.shape == p.shape
    assert m.shape == p.shape
    assert vv.shape == p.shape


def test_gradient_clip_by_global_norm():
    grads = [
        np.array([10.0, 0.0], dtype=np.float32),
        np.array([0.0, 10.0], dtype=np.float32),
    ]
    clipped = clip_by_global_norm(grads, max_norm=5.0)
    norm = np.sqrt(sum(np.sum(g**2) for g in clipped))
    assert norm <= 5.0 + 1e-5
