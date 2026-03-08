"""
Core neural network implementation.

Includes Layer and NeuralNetwork classes, forward propagation,
and loss functions for training.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from neural_backend import (
    as_numpy,
    array_module,
    default_device,
    default_engine_backend,
    resolve_backend,
    to_cpu,
    to_device,
)

try:
    from neural_engine_native import kernels as native_kernels
except Exception:  # pragma: no cover
    native_kernels = None


class Layer:
    """Single neural network layer: h = activation(W*x + b)"""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        activation: str = "relu",
        init_method: str = "he",
        xp: Any = np,
        dtype: Any = np.float32,
    ):
        self.input_size = input_size
        self.output_size = output_size
        self.activation_name = activation
        self.init_method = init_method
        self.xp = xp
        self.dtype = np.dtype(dtype)

        self.weights = self._initialize_weights(input_size, output_size, init_method).astype(self.dtype)
        self.biases = (self.xp.random.randn(output_size) * 0.01).astype(self.dtype)

        self.activation = self._get_activation_function(activation)

    def _initialize_weights(self, n_in: int, n_out: int, method: str):
        xp = self.xp
        if method == "he":
            return xp.random.randn(n_out, n_in) * xp.sqrt(2.0 / n_in)
        if method == "xavier":
            limit = xp.sqrt(6.0 / (n_in + n_out))
            return xp.random.uniform(-limit, limit, (n_out, n_in))
        if method == "orthogonal":
            w = xp.random.randn(n_out, n_in)
            if n_out >= n_in:
                q, _ = xp.linalg.qr(w)
                return q[:n_out, :n_in]
            q, _ = xp.linalg.qr(w.T)
            return q.T[:n_out, :n_in]
        if method == "uniform":
            limit = xp.sqrt(1.0 / n_in)
            return xp.random.uniform(-limit, limit, (n_out, n_in))
        return xp.random.randn(n_out, n_in) * xp.sqrt(2.0 / n_in)

    def _get_activation_function(self, activation: str) -> Callable:
        xp = self.xp
        activations = {
            "relu": lambda x: xp.maximum(0, x),
            "leaky_relu": lambda x: xp.maximum(0.01 * x, x),
            "elu": lambda x: xp.where(x > 0, x, xp.exp(x) - 1),
            "sigmoid": lambda x: 1 / (1 + xp.exp(-xp.clip(x, -500, 500))),
            "tanh": lambda x: xp.tanh(x),
            "swish": lambda x: x * (1 / (1 + xp.exp(-xp.clip(x, -500, 500)))),
            "gelu": lambda x: 0.5 * x * (1 + xp.tanh(xp.sqrt(2 / xp.pi) * (x + 0.044715 * x**3))),
            "softmax": lambda x: self._softmax(x),
            "linear": lambda x: x,
        }

        if activation not in activations:
            raise ValueError(f"Unknown activation: {activation}. Use: {list(activations.keys())}")

        return activations[activation]

    def _softmax(self, x):
        xp = self.xp
        shifted = x - xp.max(x, axis=-1, keepdims=True)
        exps = xp.exp(shifted)
        return exps / xp.sum(exps, axis=-1, keepdims=True)

    def set_backend(self, xp: Any, dtype: Optional[Any] = None):
        target_dtype = np.dtype(dtype) if dtype is not None else self.dtype
        self.weights = to_device(to_cpu(self.weights), xp, target_dtype)
        self.biases = to_device(to_cpu(self.biases), xp, target_dtype)
        self.xp = xp
        self.dtype = target_dtype
        self.activation = self._get_activation_function(self.activation_name)

    def forward(self, x):
        if x.ndim == 1:
            x = x.reshape(1, -1)
            squeeze = True
        else:
            squeeze = False

        use_native = (
            native_kernels is not None
            and self.xp is np
            and native_kernels.is_native_available()
        )

        if use_native:
            linear_output = native_kernels.linear_forward(x, self.weights, self.biases)
            activated = native_kernels.activation_forward(linear_output, self.activation_name)
        else:
            linear_output = self.xp.dot(x, self.weights.T) + self.biases
            activated = self.activation(linear_output)

        if squeeze:
            return activated.reshape(-1)
        return activated

    def get_parameters(self):
        return self.weights, self.biases

    def set_parameters(self, weights, biases):
        self.weights = weights
        self.biases = biases

    def __getstate__(self):
        state = self.__dict__.copy()
        if "activation" in state:
            del state["activation"]
        state["weights"] = to_cpu(state["weights"])
        state["biases"] = to_cpu(state["biases"])
        state["xp"] = np
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.xp = np
        self.dtype = np.dtype(getattr(self, "dtype", np.float32))
        self.weights = np.asarray(self.weights, dtype=self.dtype)
        self.biases = np.asarray(self.biases, dtype=self.dtype)
        self.activation = self._get_activation_function(self.activation_name)

    def __repr__(self) -> str:
        return f"Layer({self.input_size} -> {self.output_size}, {self.activation_name})"


class NeuralNetwork:
    """Multi-layer neural network for function approximation."""

    def __init__(
        self,
        layer_sizes: List[int],
        activations: Optional[List[str]] = None,
        device: str = "auto",
        dtype: Any = "float32",
    ):
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes) - 1
        self.requested_device = device or default_device()
        self.dtype = np.dtype(dtype)
        self.requested_engine_backend = default_engine_backend()

        self.xp, self.device, self.backend_name, self.using_gpu = resolve_backend(self.requested_device)
        self.execution_backend = self._resolve_execution_backend()

        if activations is None:
            activations = ["relu"] * (self.num_layers - 1) + ["linear"]

        if len(activations) != self.num_layers:
            raise ValueError(f"Need {self.num_layers} activations, got {len(activations)}")

        self.layers: List[Layer] = []
        for i in range(self.num_layers):
            layer = Layer(
                input_size=layer_sizes[i],
                output_size=layer_sizes[i + 1],
                activation=activations[i],
                xp=self.xp,
                dtype=self.dtype,
            )
            self.layers.append(layer)

        print("Neural Network Created:")
        print(f"  Architecture: {' -> '.join(map(str, layer_sizes))}")
        print(f"  Activations: {activations}")
        print(f"  Backend: {self.backend_name} ({self.device})")
        print(f"  Engine backend: {self.execution_backend}")
        print(f"  Total Parameters: {self.count_parameters()}")

    def to_device(self, device: str = "auto"):
        self.xp, self.device, self.backend_name, self.using_gpu = resolve_backend(device)
        self.execution_backend = self._resolve_execution_backend()
        for layer in self.layers:
            layer.set_backend(self.xp, self.dtype)

    def _resolve_execution_backend(self) -> str:
        if self.requested_engine_backend == "python":
            return "python_fallback"
        if self.using_gpu:
            return "cupy_gpu"
        if native_kernels is not None and native_kernels.is_native_available():
            return "native_cpu"
        return "python_fallback"

    def forward(self, x, return_cache: bool = False):
        current_output = to_device(x, self.xp, self.dtype)
        caches: List[Dict[str, Any]] = []

        for i, layer in enumerate(self.layers):
            a_prev = current_output
            if (
                self.execution_backend == "native_cpu"
                and native_kernels is not None
                and native_kernels.is_native_available()
            ):
                a_prev_2d = a_prev.reshape(1, -1) if a_prev.ndim == 1 else a_prev
                z = native_kernels.linear_forward(a_prev_2d, layer.weights, layer.biases)
                current_output = native_kernels.activation_forward(z, layer.activation_name)
                if a_prev.ndim == 1:
                    current_output = current_output.reshape(-1)
                    z = z.reshape(-1)
            else:
                z = (
                    self.xp.dot(a_prev, layer.weights.T) + layer.biases
                    if a_prev.ndim > 1
                    else self.xp.dot(layer.weights, a_prev) + layer.biases
                )
                current_output = layer.activation(z)
            if return_cache:
                caches.append({"a_prev": a_prev, "z": z, "a": current_output, "layer": layer})

            if hasattr(self, "_debug") and self._debug:
                print(f"  Layer {i + 1} output shape: {current_output.shape}")

        if return_cache:
            return current_output, caches
        return current_output

    def predict(self, x, on_device: bool = False):
        preds = self.forward(x)
        if on_device:
            return preds
        return as_numpy(preds)

    def count_parameters(self) -> int:
        total = 0
        for layer in self.layers:
            weights, biases = layer.get_parameters()
            total += int(weights.size + biases.size)
        return total

    def get_all_parameters(self):
        params = []
        for layer in self.layers:
            weights, biases = layer.get_parameters()
            params.extend([weights, biases])
        return params

    def set_all_parameters(self, params):
        param_idx = 0
        for layer in self.layers:
            weights = to_device(params[param_idx], self.xp, self.dtype)
            biases = to_device(params[param_idx + 1], self.xp, self.dtype)
            layer.set_parameters(weights, biases)
            param_idx += 2

    def __getstate__(self):
        state = self.__dict__.copy()
        state["xp"] = np
        state["using_gpu"] = False
        state["backend_name"] = "numpy"
        state["device"] = "cpu"
        state["requested_device"] = "cpu"
        state["requested_engine_backend"] = "native"
        state["execution_backend"] = "native_cpu"
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.dtype = np.dtype(getattr(self, "dtype", np.float32))
        self.requested_engine_backend = getattr(self, "requested_engine_backend", default_engine_backend())
        self.xp, self.device, self.backend_name, self.using_gpu = resolve_backend(getattr(self, "requested_device", "cpu"))
        self.execution_backend = self._resolve_execution_backend()
        for layer in self.layers:
            if not hasattr(layer, "activation"):
                layer.activation = layer._get_activation_function(layer.activation_name)
            layer.set_backend(self.xp, self.dtype)

    def __repr__(self) -> str:
        return f"NeuralNetwork({self.layer_sizes}, {self.count_parameters()} params, backend={self.backend_name})"


def _loss_xp(y_true, y_pred):
    return array_module(y_pred)


# Loss functions
def mean_squared_error(y_true, y_pred):
    xp = _loss_xp(y_true, y_pred)
    y_true_d = to_device(y_true, xp)
    y_pred_d = to_device(y_pred, xp)
    return xp.mean(0.5 * (y_true_d - y_pred_d) ** 2)


def mean_absolute_error(y_true, y_pred):
    xp = _loss_xp(y_true, y_pred)
    y_true_d = to_device(y_true, xp)
    y_pred_d = to_device(y_pred, xp)
    return xp.mean(xp.abs(y_true_d - y_pred_d))


def cross_entropy_loss(y_true, y_pred):
    xp = _loss_xp(y_true, y_pred)
    y_true_d = to_device(y_true, xp)
    y_pred_d = to_device(y_pred, xp)
    epsilon = 1e-15
    y_pred_clipped = xp.clip(y_pred_d, epsilon, 1 - epsilon)
    return -xp.sum(y_true_d * xp.log(y_pred_clipped)) / y_true_d.shape[0]


def categorical_accuracy(y_true, y_pred) -> float:
    xp = _loss_xp(y_true, y_pred)
    y_true_d = to_device(y_true, xp)
    y_pred_d = to_device(y_pred, xp)
    predicted_classes = xp.argmax(y_pred_d, axis=1)
    true_classes = xp.argmax(y_true_d, axis=1)
    accuracy = xp.mean(predicted_classes == true_classes) * 100
    return float(to_cpu(accuracy))


# Utility for testing
def create_sample_data(n_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    x = np.random.randn(n_samples, 2).astype(np.float32)
    y = 2 * x[:, 0] + 3 * x[:, 1] + 1 + 0.1 * np.random.randn(n_samples).astype(np.float32)
    return x, y.reshape(-1, 1).astype(np.float32)


if __name__ == "__main__":
    print("Testing Neural Network Core")
    print("=" * 40)
    x, y = create_sample_data(50)
    print("Sample Data Created:")
    print(f"  Input shape: {x.shape}")
    print(f"  Target shape: {y.shape}")
    print("  Target function: y = 2*x1 + 3*x2 + 1")

    print("\nCreating Neural Network...")
    nn = NeuralNetwork([2, 5, 3, 1], ["relu", "relu", "linear"])

    print("\nTesting Forward Pass...")
    predictions = nn.predict(x)
    print(f"  Prediction shape: {predictions.shape}")
    print(f"  Sample predictions: {predictions[:5].flatten()}")
    print(f"  Sample targets: {y[:5].flatten()}")

    print("\nTesting Loss Computation...")
    mse_loss = mean_squared_error(y, predictions)
    mae_loss = mean_absolute_error(y, predictions)
    print(f"  MSE Loss: {float(mse_loss):.6f}")
    print(f"  MAE Loss: {float(mae_loss):.6f}")

    print("\nTesting Parameter Access...")
    params = nn.get_all_parameters()
    print(f"  Number of parameter arrays: {len(params)}")
    print(f"  Total parameters: {nn.count_parameters()}")

    print("\nAll tests passed!")
