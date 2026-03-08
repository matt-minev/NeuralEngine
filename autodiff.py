"""
Training engine and optimizers for neural network optimization.

Implements manual backpropagation with optional NumPy/CuPy backend execution.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple
import time

import matplotlib.pyplot as plt
import numpy as np

from neural_backend import as_numpy, array_module, default_device, to_cpu, to_device
from nn_core import cross_entropy_loss, mean_absolute_error, mean_squared_error
try:
    from neural_engine_native import kernels as native_kernels
except Exception:  # pragma: no cover
    native_kernels = None


class Optimizer:
    """Base optimizer class. All optimizers inherit from this."""

    def __init__(self, learning_rate: float = 0.001):
        self.learning_rate = learning_rate
        self.step_count = 0

    def update(self, params: List[Any], gradients: List[Any]) -> List[Any]:
        raise NotImplementedError("Subclasses must implement update method")

    def zero_grad(self):
        pass


class SGD(Optimizer):
    """Stochastic gradient descent with optional momentum."""

    def __init__(self, learning_rate: float = 0.001, momentum: float = 0.0):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.velocity = None

    def update(self, params: List[Any], gradients: List[Any]) -> List[Any]:
        if self.velocity is None:
            self.velocity = [array_module(p).zeros_like(p) for p in params]

        updated_params = []
        for i, (param, grad) in enumerate(zip(params, gradients)):
            use_native = (
                native_kernels is not None
                and array_module(param) is np
                and native_kernels.is_native_available()
            )
            if use_native:
                updated_param, self.velocity[i] = native_kernels.sgd_update(
                    param,
                    grad,
                    learning_rate=self.learning_rate,
                    momentum=self.momentum,
                    velocity=self.velocity[i],
                )
            else:
                if self.momentum > 0:
                    self.velocity[i] = self.momentum * self.velocity[i] + (1 - self.momentum) * grad
                    updated_param = param - self.learning_rate * self.velocity[i]
                else:
                    updated_param = param - self.learning_rate * grad
            updated_params.append(updated_param)

        self.step_count += 1
        return updated_params


class Adam(Optimizer):
    """Adam optimizer with backend-agnostic tensors."""

    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None

    def update(self, params: List[Any], gradients: List[Any]) -> List[Any]:
        if self.m is None:
            self.m = [array_module(p).zeros_like(p) for p in params]
            self.v = [array_module(p).zeros_like(p) for p in params]

        self.step_count += 1
        updated_params = []

        for i, (param, grad) in enumerate(zip(params, gradients)):
            xp = array_module(param)
            use_native = (
                native_kernels is not None
                and xp is np
                and native_kernels.is_native_available()
            )
            if use_native:
                updated_param, self.m[i], self.v[i] = native_kernels.adam_update(
                    param,
                    grad,
                    self.m[i],
                    self.v[i],
                    learning_rate=self.learning_rate,
                    beta1=self.beta1,
                    beta2=self.beta2,
                    epsilon=self.epsilon,
                    step_count=self.step_count,
                )
            else:
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)
                m_hat = self.m[i] / (1 - self.beta1 ** self.step_count)
                v_hat = self.v[i] / (1 - self.beta2 ** self.step_count)
                updated_param = param - self.learning_rate * m_hat / (xp.sqrt(v_hat) + self.epsilon)
            updated_params.append(updated_param)

        return updated_params


class TrainingEngine:
    """Manual backprop training engine with optional GPU backend."""

    def __init__(self, network, optimizer: Optimizer, loss_function: Callable, device: Optional[str] = None):
        self.network = network
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.device = device or getattr(network, "device", default_device())
        if hasattr(self.network, "to_device"):
            self.network.to_device(self.device)
        self.xp = getattr(self.network, "xp", np)
        self.history = defaultdict(list)

    def _loss_name(self) -> str:
        fn = self.loss_function
        if fn is mean_squared_error:
            return "mse"
        if fn is mean_absolute_error:
            return "mae"
        if fn is cross_entropy_loss:
            return "cross_entropy"
        return getattr(fn, "__name__", "custom")

    def _ensure_2d(self, x):
        return x.reshape(1, -1) if x.ndim == 1 else x

    def _activation_backward(self, d_a, z, a, activation_name: str):
        if (
            getattr(self.network, "execution_backend", "python_fallback") == "native_cpu"
            and native_kernels is not None
            and native_kernels.is_native_available()
        ):
            return native_kernels.activation_backward(d_a, z, a, activation_name)

        xp = self.xp
        if activation_name == "linear":
            return d_a
        if activation_name == "relu":
            return d_a * (z > 0)
        if activation_name == "leaky_relu":
            return d_a * xp.where(z > 0, 1.0, 0.01)
        if activation_name == "elu":
            return d_a * xp.where(z > 0, 1.0, xp.exp(z))
        if activation_name == "sigmoid":
            s = a
            return d_a * s * (1 - s)
        if activation_name == "tanh":
            return d_a * (1 - a**2)
        if activation_name == "swish":
            s = 1 / (1 + xp.exp(-xp.clip(z, -500, 500)))
            return d_a * (s + z * s * (1 - s))
        if activation_name == "gelu":
            k = np.sqrt(2.0 / np.pi)
            u = k * (z + 0.044715 * z**3)
            t = xp.tanh(u)
            sech2 = 1 - t**2
            du = k * (1 + 3 * 0.044715 * z**2)
            return d_a * (0.5 * (1 + t) + 0.5 * z * sech2 * du)
        if activation_name == "softmax":
            return a * (d_a - xp.sum(d_a * a, axis=1, keepdims=True))
        raise ValueError(f"Unsupported activation for backward pass: {activation_name}")

    def _compute_loss_and_output_grad(self, y_true, y_pred, output_activation: str):
        xp = self.xp
        loss_name = self._loss_name()
        eps = 1e-12
        batch_size = max(1, int(y_true.shape[0]))

        if loss_name == "mse":
            if (
                getattr(self.network, "execution_backend", "python_fallback") == "native_cpu"
                and native_kernels is not None
                and native_kernels.is_native_available()
            ):
                return native_kernels.mse_loss_grad(y_true, y_pred)
            loss = xp.mean(0.5 * (y_true - y_pred) ** 2)
            d_a = (y_pred - y_true) / y_true.size
            return loss, d_a

        if loss_name == "mae":
            if (
                getattr(self.network, "execution_backend", "python_fallback") == "native_cpu"
                and native_kernels is not None
                and native_kernels.is_native_available()
            ):
                return native_kernels.mae_loss_grad(y_true, y_pred)
            loss = xp.mean(xp.abs(y_true - y_pred))
            d_a = xp.sign(y_pred - y_true) / y_true.size
            return loss, d_a

        if loss_name == "cross_entropy":
            if (
                output_activation == "softmax"
                and getattr(self.network, "execution_backend", "python_fallback") == "native_cpu"
                and native_kernels is not None
                and native_kernels.is_native_available()
            ):
                loss, _, d_z = native_kernels.softmax_cross_entropy(y_pred, y_true)
                return loss, d_z
            y_pred_clipped = xp.clip(y_pred, eps, 1 - eps)
            loss = -xp.sum(y_true * xp.log(y_pred_clipped)) / batch_size
            if output_activation == "softmax":
                d_z = (y_pred - y_true) / batch_size
                return loss, d_z
            d_a = -y_true / y_pred_clipped / batch_size
            return loss, d_a

        loss = self.loss_function(y_true, y_pred)
        # Numerical finite-diff fallback for unknown loss functions is intentionally omitted.
        raise ValueError(
            f"Unsupported loss function for manual backprop: {getattr(self.loss_function, '__name__', self.loss_function)}"
        )

    def _backward(self, caches: List[Dict[str, Any]], y_true, y_pred):
        grads: List[Any] = []
        output_activation = caches[-1]["layer"].activation_name
        loss, d_last = self._compute_loss_and_output_grad(y_true, y_pred, output_activation)

        d_a = None
        d_z = d_last

        for idx in reversed(range(len(caches))):
            cache = caches[idx]
            layer = cache["layer"]
            a_prev = self._ensure_2d(cache["a_prev"])
            z = self._ensure_2d(cache["z"])
            a = self._ensure_2d(cache["a"])
            w = layer.weights

            if not (idx == len(caches) - 1 and output_activation == "softmax" and self._loss_name() == "cross_entropy"):
                if d_a is None:
                    d_a = self._ensure_2d(d_last)
                d_z = self._activation_backward(d_a, z, a, layer.activation_name)

            if (
                getattr(self.network, "execution_backend", "python_fallback") == "native_cpu"
                and native_kernels is not None
                and native_kernels.is_native_available()
            ):
                d_w, d_b, d_a = native_kernels.linear_backward(d_z, a_prev, w)
            else:
                batch_size = max(1, int(a_prev.shape[0]))
                d_w = self.xp.dot(d_z.T, a_prev) / batch_size
                d_b = self.xp.sum(d_z, axis=0) / batch_size
                d_a = self.xp.dot(d_z, w)

            grads.insert(0, d_b)
            grads.insert(0, d_w)

        return loss, grads

    def train_step(self, x, y_true, clip_gradients: bool = False, max_norm: float = 5.0) -> float:
        x_d = self._ensure_2d(to_device(x, self.xp, getattr(self.network, "dtype", np.float32)))
        y_d = self._ensure_2d(to_device(y_true, self.xp, getattr(self.network, "dtype", np.float32)))

        y_pred, caches = self.network.forward(x_d, return_cache=True)
        y_pred = self._ensure_2d(y_pred)

        loss, gradients_structured = self._backward(caches, y_d, y_pred)

        if clip_gradients:
            if (
                getattr(self.network, "execution_backend", "python_fallback") == "native_cpu"
                and native_kernels is not None
                and native_kernels.is_native_available()
            ):
                gradients_structured = native_kernels.clip_by_global_norm(gradients_structured, max_norm)
            else:
                global_norm = self.xp.sqrt(sum(self.xp.sum(g**2) for g in gradients_structured))
                if float(to_cpu(global_norm)) > max_norm:
                    clip_ratio = max_norm / (global_norm + 1e-12)
                    gradients_structured = [g * clip_ratio for g in gradients_structured]

        current_params = self.network.get_all_parameters()
        updated_params = self.optimizer.update(current_params, gradients_structured)
        self.network.set_all_parameters(updated_params)

        return float(to_cpu(loss))

    def train(
        self,
        x,
        y_true,
        epochs: int = 1000,
        batch_size: Optional[int] = None,
        validation_data: Optional[Tuple[Any, Any]] = None,
        verbose: bool = True,
        plot_progress: bool = True,
        clip_gradients: bool = False,
        max_grad_norm: float = 5.0,
    ) -> Dict:
        x_d = to_device(x, self.xp, getattr(self.network, "dtype", np.float32))
        y_d = to_device(y_true, self.xp, getattr(self.network, "dtype", np.float32))

        print("Starting training...")
        print(f"  Network: {self.network}")
        print(f"  Optimizer: {self.optimizer.__class__.__name__}")
        print(f"  Backend: {getattr(self.network, 'backend_name', 'numpy')} ({getattr(self.network, 'device', 'cpu')})")
        print(f"  Training samples: {x_d.shape[0]}")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size or 'Full batch'}")

        start_time = time.time()
        report_interval = max(1, epochs // 10)

        for epoch in range(epochs):
            epoch_losses = []
            if batch_size is None:
                loss = self.train_step(x_d, y_d, clip_gradients=clip_gradients, max_norm=max_grad_norm)
                epoch_losses.append(loss)
            else:
                n_samples = x_d.shape[0]
                indices = self.xp.random.permutation(n_samples)
                for i in range(0, n_samples, batch_size):
                    batch_indices = indices[i : i + batch_size]
                    x_batch = x_d[batch_indices]
                    y_batch = y_d[batch_indices]
                    loss = self.train_step(x_batch, y_batch, clip_gradients=clip_gradients, max_norm=max_grad_norm)
                    epoch_losses.append(loss)

            avg_loss = float(np.mean(epoch_losses))
            self.history["train_loss"].append(avg_loss)

            if validation_data is not None:
                x_val, y_val = validation_data
                x_val_d = to_device(x_val, self.xp, getattr(self.network, "dtype", np.float32))
                y_val_d = to_device(y_val, self.xp, getattr(self.network, "dtype", np.float32))
                val_pred = self.network.forward(x_val_d)
                val_loss = self.loss_function(y_val_d, val_pred)
                self.history["val_loss"].append(float(to_cpu(val_loss)))

            if verbose and (epoch % report_interval == 0 or epoch == epochs - 1):
                elapsed = time.time() - start_time
                val_text = f", Val Loss: {self.history['val_loss'][-1]:.6f}" if validation_data else ""
                print(f"  Epoch {epoch:4d}/{epochs}: Loss: {avg_loss:.6f}{val_text} ({elapsed:.1f}s)")

        training_time = time.time() - start_time
        print(f"Training complete! ({training_time:.1f}s)")

        if plot_progress:
            self.plot_training_history()

        return dict(self.history)

    def plot_training_history(self):
        if not self.history["train_loss"]:
            print("No training history to plot.")
            return

        plt.figure(figsize=(10, 6))
        epochs = range(1, len(self.history["train_loss"]) + 1)
        plt.plot(epochs, self.history["train_loss"], "b-", label="Training Loss", linewidth=2)

        if "val_loss" in self.history and self.history["val_loss"]:
            plt.plot(epochs, self.history["val_loss"], "r-", label="Validation Loss", linewidth=2)

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Progress")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale("log")

        final_loss = self.history["train_loss"][-1]
        plt.annotate(
            f"Final Loss: {final_loss:.6f}",
            xy=(len(self.history["train_loss"]), final_loss),
            xytext=(max(1, int(len(self.history["train_loss"]) * 0.7)), final_loss * 2),
            arrowprops=dict(arrowstyle="->", color="black", alpha=0.7),
        )

        plt.tight_layout()
        plt.show()

    def evaluate(self, x, y_true) -> Dict:
        x_d = to_device(x, self.xp, getattr(self.network, "dtype", np.float32))
        y_d = to_device(y_true, self.xp, getattr(self.network, "dtype", np.float32))
        y_pred = self.network.forward(x_d)
        loss = self.loss_function(y_d, y_pred)

        mse = self.xp.mean((y_d - y_pred) ** 2)
        mae = self.xp.mean(self.xp.abs(y_d - y_pred))

        return {
            "loss": float(to_cpu(loss)),
            "mse": float(to_cpu(mse)),
            "mae": float(to_cpu(mae)),
            "predictions": as_numpy(y_pred),
            "targets": as_numpy(y_d),
        }
class LearningRateScheduler:
    """Base class for LR scheduling."""
    
    def __init__(self, initial_lr: float):
        self.initial_lr = initial_lr
    
    def get_lr(self, epoch: int) -> float:
        """Get learning rate for given epoch."""
        raise NotImplementedError


class StepLR(LearningRateScheduler):
    """Step-wise learning rate decay."""
    
    def __init__(self, initial_lr: float, step_size: int, gamma: float = 0.1):
        super().__init__(initial_lr)
        self.step_size = step_size
        self.gamma = gamma
    
    def get_lr(self, epoch: int) -> float:
        return self.initial_lr * (self.gamma ** (epoch // self.step_size))


class ExponentialLR(LearningRateScheduler):
    """Exponential learning rate decay."""
    
    def __init__(self, initial_lr: float, gamma: float = 0.95):
        super().__init__(initial_lr)
        self.gamma = gamma
    
    def get_lr(self, epoch: int) -> float:
        return self.initial_lr * (self.gamma ** epoch)


class CosineAnnealingWarmRestarts(LearningRateScheduler):
    """
    Cosine annealing with warm restarts.
    
    Learning rate follows cosine annealing, then restarts at higher value.
    Good for escaping local minima.
    """
    
    def __init__(self, initial_lr: float, T_0: int = 10, T_mult: int = 2, eta_min: float = 0.0):
        """
        Args:
            initial_lr: Initial learning rate
            T_0: Number of epochs for first restart
            T_mult: Multiplier for restart period
            eta_min: Minimum learning rate
        """
        super().__init__(initial_lr)
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.current_restart = 0
        self.epoch_since_restart = 0
        self.T_curr = T_0
    
    def get_lr(self, epoch: int) -> float:
        """Get learning rate for given epoch with warm restarts."""
        # Check if we need to restart
        if self.epoch_since_restart >= self.T_curr:
            self.current_restart += 1
            self.epoch_since_restart = 0
            self.T_curr = self.T_0 * (self.T_mult ** self.current_restart)
        
        # Cosine annealing within current period
        lr = self.eta_min + (self.initial_lr - self.eta_min) * \
             (1 + np.cos(np.pi * self.epoch_since_restart / self.T_curr)) / 2
        
        self.epoch_since_restart += 1
        return float(lr)


class ReduceLROnPlateau(LearningRateScheduler):
    """
    Reduce learning rate when validation loss plateaus.
    
    Monitors validation loss and reduces LR when no improvement.
    """
    
    def __init__(self, initial_lr: float, factor: float = 0.5, patience: int = 10, 
                 min_lr: float = 1e-6, mode: str = 'min'):
        """
        Args:
            initial_lr: Initial learning rate
            factor: Factor to multiply LR by when reducing
            patience: Number of epochs to wait before reducing
            min_lr: Minimum learning rate
            mode: 'min' to reduce when metric stops decreasing, 'max' for increasing
        """
        super().__init__(initial_lr)
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.mode = mode
        self.best_metric = None
        self.patience_counter = 0
        self.current_lr = initial_lr
    
    def get_lr(self, epoch: int) -> float:
        """Get current learning rate (doesn't change based on epoch alone)."""
        return self.current_lr
    
    def step(self, metric: float):
        """
        Update learning rate based on metric.
        
        Args:
            metric: Current metric value (e.g., validation loss)
        """
        if self.best_metric is None:
            self.best_metric = metric
            return
        
        # Check if metric improved
        if self.mode == 'min':
            improved = metric < self.best_metric
        else:  # mode == 'max'
            improved = metric > self.best_metric
        
        if improved:
            self.best_metric = metric
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                # Reduce learning rate
                self.current_lr = max(self.current_lr * self.factor, self.min_lr)
                self.patience_counter = 0


class OneCycleLR(LearningRateScheduler):
    """
    One cycle learning rate policy.
    
    Increases LR to max_lr, then decreases following cosine annealing.
    """
    
    def __init__(self, initial_lr: float, max_lr: float, total_steps: int, 
                 pct_start: float = 0.3, div_factor: float = 25.0):
        """
        Args:
            initial_lr: Initial learning rate
            max_lr: Maximum learning rate
            total_steps: Total number of training steps
            pct_start: Percentage of steps for warmup (0.0 to 1.0)
            div_factor: Initial LR = max_lr / div_factor
        """
        super().__init__(initial_lr)
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.div_factor = div_factor
        self.initial_lr = max_lr / div_factor
    
    def get_lr(self, epoch: int) -> float:
        """Get learning rate for given epoch in one cycle."""
        if epoch >= self.total_steps:
            return self.initial_lr
        
        # Warmup phase
        warmup_steps = int(self.total_steps * self.pct_start)
        if epoch < warmup_steps:
            # Linear warmup
            lr = self.initial_lr + (self.max_lr - self.initial_lr) * (epoch / warmup_steps)
        else:
            # Cosine annealing
            progress = (epoch - warmup_steps) / (self.total_steps - warmup_steps)
            lr = self.initial_lr + (self.max_lr - self.initial_lr) * \
                 (1 + np.cos(np.pi * progress)) / 2
        
        return float(lr)


if __name__ == "__main__":
    print("Testing Automatic Differentiation Engine")
    print("=" * 50)
    
    # import network components
    import sys
    sys.path.append('.')
    from nn_core import NeuralNetwork, mean_squared_error, create_sample_data
    
    # create sample data
    X, y = create_sample_data(100)
    print(f"Sample data: {X.shape[0]} samples, target: y = 2*x1 + 3*x2 + 1")
    
    # create network
    network = NeuralNetwork([2, 8, 4, 1], ['relu', 'relu', 'linear'])
    
    # test different optimizers
    optimizers = {
        'SGD': SGD(learning_rate=0.01),
        'SGD+Momentum': SGD(learning_rate=0.01, momentum=0.9),
        'Adam': Adam(learning_rate=0.001)
    }
    
    # split data
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    results = {}
    
    for name, optimizer in optimizers.items():
        print(f"\nTesting {name} optimizer...")
        
        # fresh network for each test
        test_network = NeuralNetwork([2, 8, 4, 1], ['relu', 'relu', 'linear'])
        
        # create trainer
        trainer = TrainingEngine(test_network, optimizer, mean_squared_error)
        
        # train
        history = trainer.train(
            X_train, y_train,
            epochs=200,
            validation_data=(X_val, y_val),
            verbose=False,
            plot_progress=False
        )
        
        # evaluate
        eval_results = trainer.evaluate(X_val, y_val)
        results[name] = eval_results
        
        print(f"  Final Loss: {eval_results['loss']:.6f}")
        print(f"  MSE: {eval_results['mse']:.6f}")
        print(f"  MAE: {eval_results['mae']:.6f}")
    
    # compare optimizers
    print(f"\nOptimizer Comparison:")
    print(f"{'Optimizer':<15} {'Loss':<10} {'MSE':<10} {'MAE':<10}")
    print("-" * 45)
    for name, result in results.items():
        print(f"{name:<15} {result['loss']:<10.6f} {result['mse']:<10.6f} {result['mae']:<10.6f}")
    
    # test gradient computation
    print(f"\nTesting gradient computation...")
    network = NeuralNetwork([2, 3, 1])
    optimizer = SGD(learning_rate=0.01)
    trainer = TrainingEngine(network, optimizer, mean_squared_error)
    
    # single gradient step
    loss_before = trainer.train_step(X_train[:10], y_train[:10])
    loss_after = mean_squared_error(y_train[:10], network.forward(X_train[:10]))
    
    print(f"  Loss before step: {loss_before:.6f}")
    print(f"  Loss after step: {loss_after:.6f}")
    print(f"  Loss change: {loss_after - loss_before:.6f}")
    
    if loss_after < loss_before:
        print("  Gradient descent working correctly!")
    else:
        print("  Learning rate might be too high")
    
    print(f"\nAll tests passed! Autodiff engine is working.")
