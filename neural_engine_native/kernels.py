"""Native kernel wrappers with NumPy/CuPy fallback dispatch."""

from __future__ import annotations

from typing import Sequence

import numpy as np

try:
    import cupy as cp  # type: ignore
except Exception:  # pragma: no cover
    cp = None

try:
    from . import _kernels as _k  # type: ignore
except Exception:  # pragma: no cover
    _k = None


def is_native_available() -> bool:
    return _k is not None


def _is_cupy(x) -> bool:
    return cp is not None and isinstance(x, cp.ndarray)


def _xp(x):
    return cp if _is_cupy(x) else np


def _as_float32(x, xp):
    return xp.asarray(x, dtype=xp.float32)


def linear_forward(x, w, b):
    xp = _xp(x)
    if xp is np and _k is not None:
        return _k.linear_forward(_as_float32(x, np), _as_float32(w, np), _as_float32(b, np))
    return x @ w.T + b


def linear_backward(d_z, a_prev, w):
    xp = _xp(d_z)
    if xp is np and _k is not None:
        return _k.linear_backward(_as_float32(d_z, np), _as_float32(a_prev, np), _as_float32(w, np))

    batch_size = max(1, int(a_prev.shape[0]))
    d_w = xp.dot(d_z.T, a_prev) / batch_size
    d_b = xp.sum(d_z, axis=0) / batch_size
    d_a_prev = xp.dot(d_z, w)
    return d_w, d_b, d_a_prev


def activation_forward(x, activation: str):
    xp = _xp(x)
    if xp is np and _k is not None:
        return _k.activation_forward(_as_float32(x, np), activation)

    if activation == "relu":
        return xp.maximum(0, x)
    if activation == "leaky_relu":
        return xp.maximum(0.01 * x, x)
    if activation == "elu":
        return xp.where(x > 0, x, xp.exp(x) - 1)
    if activation == "sigmoid":
        return 1 / (1 + xp.exp(-xp.clip(x, -500, 500)))
    if activation == "tanh":
        return xp.tanh(x)
    if activation == "swish":
        s = 1 / (1 + xp.exp(-xp.clip(x, -500, 500)))
        return x * s
    if activation == "gelu":
        return 0.5 * x * (1 + xp.tanh(xp.sqrt(2 / xp.pi) * (x + 0.044715 * x**3)))
    if activation == "softmax":
        shifted = x - xp.max(x, axis=-1, keepdims=True)
        exps = xp.exp(shifted)
        return exps / xp.sum(exps, axis=-1, keepdims=True)
    if activation == "linear":
        return x
    raise ValueError(f"Unsupported activation: {activation}")


def activation_backward(d_a, z, a, activation: str):
    xp = _xp(d_a)
    if xp is np and _k is not None:
        return _k.activation_backward(_as_float32(d_a, np), _as_float32(z, np), _as_float32(a, np), activation)

    if activation == "linear":
        return d_a
    if activation == "relu":
        return d_a * (z > 0)
    if activation == "leaky_relu":
        return d_a * xp.where(z > 0, 1.0, 0.01)
    if activation == "elu":
        return d_a * xp.where(z > 0, 1.0, xp.exp(z))
    if activation == "sigmoid":
        return d_a * a * (1 - a)
    if activation == "tanh":
        return d_a * (1 - a**2)
    if activation == "swish":
        s = 1 / (1 + xp.exp(-xp.clip(z, -500, 500)))
        return d_a * (s + z * s * (1 - s))
    if activation == "gelu":
        k = np.sqrt(2.0 / np.pi)
        u = k * (z + 0.044715 * z**3)
        t = xp.tanh(u)
        sech2 = 1 - t**2
        du = k * (1 + 3 * 0.044715 * z**2)
        return d_a * (0.5 * (1 + t) + 0.5 * z * sech2 * du)
    if activation == "softmax":
        return a * (d_a - xp.sum(d_a * a, axis=1, keepdims=True))
    raise ValueError(f"Unsupported activation: {activation}")


def mse_loss_grad(y_true, y_pred):
    xp = _xp(y_true)
    if xp is np and _k is not None:
        return _k.mse_loss_grad(_as_float32(y_true, np), _as_float32(y_pred, np))

    loss = xp.mean(0.5 * (y_true - y_pred) ** 2)
    grad = (y_pred - y_true) / y_true.size
    return float(loss), grad


def mae_loss_grad(y_true, y_pred):
    xp = _xp(y_true)
    if xp is np and _k is not None:
        return _k.mae_loss_grad(_as_float32(y_true, np), _as_float32(y_pred, np))

    loss = xp.mean(xp.abs(y_true - y_pred))
    grad = xp.sign(y_pred - y_true) / y_true.size
    return float(loss), grad


def softmax_cross_entropy(logits, y_true):
    xp = _xp(logits)
    if xp is np and _k is not None:
        return _k.softmax_cross_entropy(_as_float32(logits, np), _as_float32(y_true, np))

    shifted = logits - xp.max(logits, axis=1, keepdims=True)
    exps = xp.exp(shifted)
    probs = exps / xp.sum(exps, axis=1, keepdims=True)
    loss = -xp.sum(y_true * xp.log(xp.clip(probs, 1e-12, 1.0))) / logits.shape[0]
    grad = (probs - y_true) / logits.shape[0]
    return float(loss), probs, grad


def sgd_update(param, grad, learning_rate: float, momentum: float = 0.0, velocity=None):
    xp = _xp(param)
    velocity = xp.zeros_like(param) if velocity is None else velocity

    if xp is np and _k is not None:
        return _k.sgd_update(
            _as_float32(param, np),
            _as_float32(grad, np),
            float(learning_rate),
            float(momentum),
            _as_float32(velocity, np),
        )

    if momentum > 0:
        velocity = momentum * velocity + (1 - momentum) * grad
        updated = param - learning_rate * velocity
    else:
        updated = param - learning_rate * grad
    return updated, velocity


def adam_update(param, grad, m, v, learning_rate: float, beta1: float, beta2: float, epsilon: float, step_count: int):
    xp = _xp(param)

    if xp is np and _k is not None:
        return _k.adam_update(
            _as_float32(param, np),
            _as_float32(grad, np),
            _as_float32(m, np),
            _as_float32(v, np),
            float(learning_rate),
            float(beta1),
            float(beta2),
            float(epsilon),
            int(step_count),
        )

    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * (grad**2)
    m_hat = m / (1 - beta1**step_count)
    v_hat = v / (1 - beta2**step_count)
    updated = param - learning_rate * m_hat / (xp.sqrt(v_hat) + epsilon)
    return updated, m, v


def global_norm(arrays: Sequence):
    if not arrays:
        return 0.0
    xp = _xp(arrays[0])
    if xp is np and _k is not None:
        return float(_k.global_norm([_as_float32(a, np) for a in arrays]))
    return float(xp.sqrt(sum(xp.sum(a**2) for a in arrays)))


def clip_by_global_norm(arrays: Sequence, max_norm: float):
    if not arrays:
        return list(arrays)
    xp = _xp(arrays[0])
    if xp is np and _k is not None:
        return list(_k.clip_by_global_norm([_as_float32(a, np) for a in arrays], float(max_norm)))

    norm = global_norm(arrays)
    if norm <= max_norm or norm <= 1e-12:
        return list(arrays)
    ratio = max_norm / norm
    return [a * ratio for a in arrays]
