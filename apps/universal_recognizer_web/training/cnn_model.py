"""Trainable CNN model for Universal Recognizer (NumPy/CuPy backends)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

from neural_backend import resolve_backend, to_device, to_cpu


def _to_scalar(x) -> float:
    try:
        return float(to_cpu(x))
    except Exception:
        return float(x)


def _im2col(x, kh, kw, stride=1, pad=0):
    xp = x.__class__.__module__.split('.')[0]
    if xp == 'cupy':
        import cupy as cp  # type: ignore
        xp_mod = cp
    else:
        xp_mod = np

    n, c, h, w = x.shape
    out_h = (h + 2 * pad - kh) // stride + 1
    out_w = (w + 2 * pad - kw) // stride + 1

    x_padded = xp_mod.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
    cols = xp_mod.zeros((n, c, kh, kw, out_h, out_w), dtype=x.dtype)

    for y in range(kh):
        y_max = y + stride * out_h
        for xk in range(kw):
            x_max = xk + stride * out_w
            cols[:, :, y, xk, :, :] = x_padded[:, :, y:y_max:stride, xk:x_max:stride]

    cols = cols.transpose(0, 4, 5, 1, 2, 3).reshape(n * out_h * out_w, -1)
    return cols, out_h, out_w


def _col2im(cols, x_shape, kh, kw, stride=1, pad=0):
    xp_name = cols.__class__.__module__.split('.')[0]
    if xp_name == 'cupy':
        import cupy as cp  # type: ignore
        xp_mod = cp
    else:
        xp_mod = np

    n, c, h, w = x_shape
    out_h = (h + 2 * pad - kh) // stride + 1
    out_w = (w + 2 * pad - kw) // stride + 1

    cols_reshaped = cols.reshape(n, out_h, out_w, c, kh, kw).transpose(0, 3, 4, 5, 1, 2)
    img = xp_mod.zeros((n, c, h + 2 * pad + stride - 1, w + 2 * pad + stride - 1), dtype=cols.dtype)

    for y in range(kh):
        y_max = y + stride * out_h
        for xk in range(kw):
            x_max = xk + stride * out_w
            img[:, :, y:y_max:stride, xk:x_max:stride] += cols_reshaped[:, :, y, xk, :, :]

    if pad == 0:
        return img[:, :, :h, :w]
    return img[:, :, pad:h + pad, pad:w + pad]


@dataclass
class CNNConfig:
    conv1_channels: int = 32
    conv2_channels: int = 64
    fc_hidden: int = 256
    num_classes: int = 62
    dropout: float = 0.25
    input_h: int = 28
    input_w: int = 28


@dataclass
class LayerInfo:
    activation_name: str


class UniversalCNN:
    """Simple CNN: Conv32->Pool->Conv64->Pool->FC256->FC62."""

    def __init__(self, config: CNNConfig | None = None, device: str = 'auto', dtype: str = 'float32'):
        self.config = config or CNNConfig()
        self.dtype = np.dtype(dtype)
        self.xp, self.device, self.backend_name, self.using_gpu = resolve_backend(device)

        # Use numpy for weight initialization (He init), then move to device.
        # This avoids cupy.random.randn which requires curand (may not be available).
        c1, c2 = self.config.conv1_channels, self.config.conv2_channels
        fan1 = 1 * 3 * 3
        fan2 = c1 * 3 * 3

        self.w1 = to_device((np.random.randn(c1, 1, 3, 3) * np.sqrt(2.0 / fan1)).astype(dtype), self.xp, self.dtype)
        self.b1 = self.xp.zeros((c1,), dtype=self.dtype)
        self.w2 = to_device((np.random.randn(c2, c1, 3, 3) * np.sqrt(2.0 / fan2)).astype(dtype), self.xp, self.dtype)
        self.b2 = self.xp.zeros((c2,), dtype=self.dtype)

        flat_dim = c2 * 7 * 7
        self.w3 = to_device((np.random.randn(flat_dim, self.config.fc_hidden) * np.sqrt(2.0 / flat_dim)).astype(dtype), self.xp, self.dtype)
        self.b3 = self.xp.zeros((self.config.fc_hidden,), dtype=self.dtype)
        self.w4 = to_device((np.random.randn(self.config.fc_hidden, self.config.num_classes) * np.sqrt(2.0 / self.config.fc_hidden)).astype(dtype), self.xp, self.dtype)
        self.b4 = self.xp.zeros((self.config.num_classes,), dtype=self.dtype)

        self.training = True
        self._cache: Dict[str, Any] = {}
        self.layer_sizes = [784, self.config.fc_hidden, self.config.num_classes]
        self.layers = [LayerInfo('relu'), LayerInfo('softmax')]

    def __getstate__(self):
        state = self.__dict__.copy()
        state['w1'] = to_cpu(self.w1)
        state['b1'] = to_cpu(self.b1)
        state['w2'] = to_cpu(self.w2)
        state['b2'] = to_cpu(self.b2)
        state['w3'] = to_cpu(self.w3)
        state['b3'] = to_cpu(self.b3)
        state['w4'] = to_cpu(self.w4)
        state['b4'] = to_cpu(self.b4)
        state['xp'] = None
        state['device'] = 'cpu'
        state['backend_name'] = 'numpy'
        state['using_gpu'] = False
        state['_cache'] = {}
        state['training'] = False
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.dtype = np.dtype(getattr(self, 'dtype', np.float32))
        self.xp, self.device, self.backend_name, self.using_gpu = resolve_backend(getattr(self, 'device', 'cpu'))
        self.w1 = to_device(self.w1, self.xp, self.dtype)
        self.b1 = to_device(self.b1, self.xp, self.dtype)
        self.w2 = to_device(self.w2, self.xp, self.dtype)
        self.b2 = to_device(self.b2, self.xp, self.dtype)
        self.w3 = to_device(self.w3, self.xp, self.dtype)
        self.b3 = to_device(self.b3, self.xp, self.dtype)
        self.w4 = to_device(self.w4, self.xp, self.dtype)
        self.b4 = to_device(self.b4, self.xp, self.dtype)
        # Backward compatibility for earlier pickles.
        if not hasattr(self, 'layer_sizes'):
            self.layer_sizes = [784, int(self.w3.shape[1]), int(self.w4.shape[1])]
        if not hasattr(self, 'layers') or not self.layers:
            self.layers = [LayerInfo('relu'), LayerInfo('softmax')]
        self._cache = {}

    def parameters(self) -> List[Any]:
        return [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3, self.w4, self.b4]

    def count_parameters(self) -> int:
        return int(sum(np.prod(to_cpu(p).shape) for p in self.parameters()))

    def _conv_forward(self, x, w, b, stride=1, pad=1):
        n, _, h, ww = x.shape
        fn, _, kh, kw = w.shape

        x_col, out_h, out_w = _im2col(x, kh, kw, stride, pad)
        w_col = w.reshape(fn, -1)
        out = x_col @ w_col.T + b
        out = out.reshape(n, out_h, out_w, fn).transpose(0, 3, 1, 2)
        cache = (x, w, b, stride, pad, x_col, w_col)
        return out, cache

    def _conv_backward(self, dout, cache):
        x, w, b, stride, pad, x_col, w_col = cache
        fn, _, kh, kw = w.shape

        dout_2d = dout.transpose(0, 2, 3, 1).reshape(-1, fn)
        db = self.xp.sum(dout_2d, axis=0)
        dw = (dout_2d.T @ x_col).reshape(w.shape)

        dx_col = dout_2d @ w_col
        dx = _col2im(dx_col, x.shape, kh, kw, stride, pad)
        return dx, dw, db

    def _maxpool_forward(self, x, pool=2, stride=2):
        n, c, h, w = x.shape
        out_h = (h - pool) // stride + 1
        out_w = (w - pool) // stride + 1

        x_reshaped = x.reshape(n * c, 1, h, w)
        x_col, _, _ = _im2col(x_reshaped, pool, pool, stride, 0)
        max_idx = self.xp.argmax(x_col, axis=1)
        out = x_col[self.xp.arange(x_col.shape[0]), max_idx]
        out = out.reshape(n, c, out_h, out_w)

        cache = (x, pool, stride, x_col, max_idx, out_h, out_w)
        return out, cache

    def _maxpool_backward(self, dout, cache):
        x, pool, stride, x_col, max_idx, out_h, out_w = cache
        n, c, h, w = x.shape

        dout_flat = dout.reshape(-1)
        dx_col = self.xp.zeros_like(x_col)
        dx_col[self.xp.arange(dx_col.shape[0]), max_idx] = dout_flat

        dx = _col2im(dx_col, (n * c, 1, h, w), pool, pool, stride, 0)
        dx = dx.reshape(x.shape)
        return dx

    def _relu(self, x):
        return self.xp.maximum(0, x)

    def _relu_backward(self, dout, x):
        return dout * (x > 0)

    def _dropout(self, x, p):
        if not self.training or p <= 0:
            return x, None
        keep = 1.0 - p
        # Use numpy for random generation (cupy.random requires curand which may be unavailable)
        mask_np = (np.random.rand(*x.shape) < keep).astype(np.float32) / keep
        mask = self.xp.asarray(mask_np) if self.xp is not np else mask_np
        return x * mask, mask

    def _softmax(self, logits):
        shifted = logits - self.xp.max(logits, axis=1, keepdims=True)
        exps = self.xp.exp(shifted)
        return exps / self.xp.sum(exps, axis=1, keepdims=True)

    def _prepare_input(self, x):
        x = to_device(x, self.xp, self.dtype)
        if x.ndim == 2 and x.shape[1] == 784:
            x = x.reshape(-1, 1, 28, 28)
        elif x.ndim == 3 and x.shape[1:] == (28, 28):
            x = x.reshape(-1, 1, 28, 28)
        elif x.ndim == 4:
            pass
        else:
            raise ValueError(f'Unsupported input shape for CNN: {x.shape}')
        return x

    def forward(self, x):
        x = self._prepare_input(x)

        z1, c1 = self._conv_forward(x, self.w1, self.b1, stride=1, pad=1)
        a1 = self._relu(z1)
        p1, p1c = self._maxpool_forward(a1, pool=2, stride=2)

        z2, c2 = self._conv_forward(p1, self.w2, self.b2, stride=1, pad=1)
        a2 = self._relu(z2)
        p2, p2c = self._maxpool_forward(a2, pool=2, stride=2)

        n = p2.shape[0]
        flat = p2.reshape(n, -1)
        z3 = flat @ self.w3 + self.b3
        a3 = self._relu(z3)
        d3, drop_mask = self._dropout(a3, self.config.dropout)

        logits = d3 @ self.w4 + self.b4
        probs = self._softmax(logits)

        self._cache = {
            'x': x, 'c1': c1, 'z1': z1, 'p1c': p1c,
            'c2': c2, 'z2': z2, 'p2c': p2c, 'p2_shape': p2.shape,
            'flat': flat, 'z3': z3, 'a3': a3, 'd3': d3, 'drop_mask': drop_mask,
            'logits': logits, 'probs': probs,
        }
        return probs

    def train_step(self, x, y, optimizer):
        probs = self.forward(x)
        y = to_device(y, self.xp, self.dtype)

        n = max(1, y.shape[0])
        loss = -self.xp.sum(y * self.xp.log(self.xp.clip(probs, 1e-12, 1.0))) / n

        dlogits = (probs - y) / n

        grads: Dict[str, Any] = {}
        grads['w4'] = self._cache['d3'].T @ dlogits
        grads['b4'] = self.xp.sum(dlogits, axis=0)

        dd3 = dlogits @ self.w4.T
        if self._cache['drop_mask'] is not None:
            dd3 *= self._cache['drop_mask']
        dz3 = self._relu_backward(dd3, self._cache['z3'])

        grads['w3'] = self._cache['flat'].T @ dz3
        grads['b3'] = self.xp.sum(dz3, axis=0)

        dflat = dz3 @ self.w3.T
        dp2 = dflat.reshape(self._cache['p2_shape'])

        da2 = self._maxpool_backward(dp2, self._cache['p2c'])
        dz2 = self._relu_backward(da2, self._cache['z2'])
        dp1, grads['w2'], grads['b2'] = self._conv_backward(dz2, self._cache['c2'])

        da1 = self._maxpool_backward(dp1, self._cache['p1c'])
        dz1 = self._relu_backward(da1, self._cache['z1'])
        _, grads['w1'], grads['b1'] = self._conv_backward(dz1, self._cache['c1'])

        optimizer.step(self, grads)
        return _to_scalar(loss), probs

    def predict(self, x, on_device=False):
        old = self.training
        self.training = False
        p = self.forward(x)
        self.training = old
        return p if on_device else to_cpu(p)


class AdamCNN:
    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m: Dict[str, Any] = {}
        self.v: Dict[str, Any] = {}

    def step(self, model: UniversalCNN, grads: Dict[str, Any]):
        self.t += 1
        for name in ['w1', 'b1', 'w2', 'b2', 'w3', 'b3', 'w4', 'b4']:
            p = getattr(model, name)
            g = grads[name]
            if name not in self.m:
                self.m[name] = model.xp.zeros_like(p)
                self.v[name] = model.xp.zeros_like(p)

            self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * g
            self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * (g * g)

            m_hat = self.m[name] / (1 - self.beta1 ** self.t)
            v_hat = self.v[name] / (1 - self.beta2 ** self.t)
            p -= self.lr * m_hat / (model.xp.sqrt(v_hat) + self.eps)
            setattr(model, name, p)


def accuracy_from_probs(probs, y_true_onehot) -> float:
    y_pred = np.argmax(to_cpu(probs), axis=1)
    y_true = np.argmax(to_cpu(y_true_onehot), axis=1)
    return float(np.mean(y_pred == y_true) * 100.0)
