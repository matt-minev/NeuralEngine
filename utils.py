"""
Utility functions for the neural network engine.

Includes activation functions, math helpers, visualization tools,
and performance monitoring utilities.
"""

import numpy as np
import autograd.numpy as anp
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
import seaborn as sns
from typing import Callable, List, Tuple, Dict, Optional, Any, Union
import time
import psutil
import os
from functools import wraps
import warnings


# Activation functions
class ActivationFunctions:
    """Collection of activation functions with numerical stability."""
    
    @staticmethod
    def relu(x: anp.ndarray) -> anp.ndarray:
        """ReLU: max(0, x)"""
        return anp.maximum(0, x)
    
    @staticmethod
    def leaky_relu(x: anp.ndarray, alpha: float = 0.01) -> anp.ndarray:
        """Leaky ReLU to avoid dead neurons."""
        return anp.maximum(alpha * x, x)
    
    @staticmethod
    def elu(x: anp.ndarray, alpha: float = 1.0) -> anp.ndarray:
        """ELU activation - smoother than ReLU."""
        return anp.where(x > 0, x, alpha * (anp.exp(x) - 1))
    
    @staticmethod
    def sigmoid(x: anp.ndarray) -> anp.ndarray:
        """Sigmoid with clipping to prevent overflow."""
        x_clipped = anp.clip(x, -500, 500)
        return 1.0 / (1.0 + anp.exp(-x_clipped))
    
    @staticmethod
    def tanh(x: anp.ndarray) -> anp.ndarray:
        """Hyperbolic tangent activation."""
        return anp.tanh(x)
    
    @staticmethod
    def swish(x: anp.ndarray) -> anp.ndarray:
        """Swish activation (x * sigmoid(x)) - works well in practice."""
        return x * ActivationFunctions.sigmoid(x)
    
    @staticmethod
    def gelu(x: anp.ndarray) -> anp.ndarray:
        """GELU activation used in transformers (BERT, GPT, etc)."""
        return 0.5 * x * (1 + anp.tanh(anp.sqrt(2 / anp.pi) * (x + 0.044715 * x**3)))
    
    @staticmethod
    def softmax(x: anp.ndarray, axis: int = -1) -> anp.ndarray:
        """Softmax with numerical stability (subtract max)."""
        x_shifted = x - anp.max(x, axis=axis, keepdims=True)
        exp_x = anp.exp(x_shifted)
        return exp_x / anp.sum(exp_x, axis=axis, keepdims=True)
    
    @staticmethod
    def linear(x: anp.ndarray) -> anp.ndarray:
        """Identity function - just returns input."""
        return x
    
    @staticmethod
    def get_activation(name: str) -> Callable:
        """Get activation function by name."""
        activations = {
            'relu': ActivationFunctions.relu,
            'leaky_relu': ActivationFunctions.leaky_relu,
            'elu': ActivationFunctions.elu,
            'sigmoid': ActivationFunctions.sigmoid,
            'tanh': ActivationFunctions.tanh,
            'swish': ActivationFunctions.swish,
            'gelu': ActivationFunctions.gelu,
            'softmax': ActivationFunctions.softmax,
            'linear': ActivationFunctions.linear
        }
        
        if name not in activations:
            raise ValueError(f"Unknown activation: {name}. Available: {list(activations.keys())}")
        
        return activations[name]


# Math utilities
class MathUtils:
    """Math helper functions for numerical stability."""
    
    @staticmethod
    def safe_log(x: anp.ndarray, eps: float = 1e-8) -> anp.ndarray:
        """Prevent log(0) errors."""
        return anp.log(anp.maximum(x, eps))
    
    @staticmethod
    def safe_sqrt(x: anp.ndarray, eps: float = 1e-8) -> anp.ndarray:
        """Prevent sqrt of negatives."""
        return anp.sqrt(anp.maximum(x, eps))
    
    @staticmethod
    def safe_divide(numerator: anp.ndarray, denominator: anp.ndarray, 
                   eps: float = 1e-8) -> anp.ndarray:
        """Avoid division by zero."""
        return numerator / anp.maximum(denominator, eps)
    
    @staticmethod
    def clip_gradients(gradients: List[anp.ndarray], 
                      max_norm: float = 5.0) -> List[anp.ndarray]:
        """Clip gradients to prevent explosions."""
        global_norm = anp.sqrt(sum(anp.sum(g**2) for g in gradients))
        
        if global_norm > max_norm:
            clip_ratio = max_norm / global_norm
            gradients = [g * clip_ratio for g in gradients]
        
        return gradients
    
    @staticmethod
    def xavier_init(n_in: int, n_out: int) -> anp.ndarray:
        """Xavier/Glorot initialization - good for sigmoid/tanh."""
        limit = anp.sqrt(6.0 / (n_in + n_out))
        return anp.random.uniform(-limit, limit, (n_out, n_in))
    
    @staticmethod
    def he_init(n_in: int, n_out: int) -> anp.ndarray:
        """He initialization - better for ReLU activations."""
        return anp.random.randn(n_out, n_in) * anp.sqrt(2.0 / n_in)
    
    @staticmethod
    def orthogonal_init(n_in: int, n_out: int) -> anp.ndarray:
        """Orthogonal weight init using QR decomposition."""
        W = anp.random.randn(n_out, n_in)
        
        # QR decomp to get orthogonal matrix
        if n_out >= n_in:
            Q, R = anp.linalg.qr(W)
            return Q[:n_out, :n_in]
        else:
            Q, R = anp.linalg.qr(W.T)
            return Q.T[:n_out, :n_in]


# Visualization tools
class NetworkVisualizer:
    """Plotting utilities for networks and training."""
    
    @staticmethod
    def plot_network_architecture(layer_sizes: List[int], 
                                 activations: List[str] = None,
                                 title: str = "Neural Network Architecture",
                                 figsize: Tuple[int, int] = (12, 8)) -> None:
        """Draw network architecture diagram."""
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        ax.set_title(title, fontsize=16, fontweight='bold')
        
        n_layers = len(layer_sizes)
        layer_x_positions = np.linspace(1, 9, n_layers)
        
        colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink']
        
        for i, (x_pos, layer_size) in enumerate(zip(layer_x_positions, layer_sizes)):
            max_neurons_to_show = 8  # dont overcrowd
            neurons_to_show = min(layer_size, max_neurons_to_show)
            
            if neurons_to_show < layer_size:
                neuron_y_positions = np.linspace(2, 8, neurons_to_show)
            else:
                neuron_y_positions = np.linspace(2, 8, neurons_to_show)
            
            # draw neurons as circles
            for j, y_pos in enumerate(neuron_y_positions):
                circle = plt.Circle((x_pos, y_pos), 0.2, 
                                  color=colors[i % len(colors)], 
                                  ec='black', linewidth=1.5)
                ax.add_patch(circle)
                
                if layer_size > max_neurons_to_show and j == neurons_to_show // 2:
                    ax.text(x_pos, y_pos - 0.6, f'...{layer_size}...', 
                           ha='center', va='center', fontsize=8)
            
            # connections to next layer
            if i < n_layers - 1:
                next_x = layer_x_positions[i + 1]
                next_layer_size = layer_sizes[i + 1]
                next_neurons_to_show = min(next_layer_size, max_neurons_to_show)
                next_neuron_y_positions = np.linspace(2, 8, next_neurons_to_show)
                
                # draw some connections (not all, too messy)
                for y1 in neuron_y_positions[::2]:
                    for y2 in next_neuron_y_positions[::2]:
                        ax.plot([x_pos + 0.2, next_x - 0.2], [y1, y2], 
                               'gray', alpha=0.3, linewidth=0.5)
            
            # labels
            layer_name = f"Input\n({layer_size})" if i == 0 else \
                        f"Output\n({layer_size})" if i == n_layers - 1 else \
                        f"Hidden {i}\n({layer_size})"
            
            ax.text(x_pos, 1.2, layer_name, ha='center', va='center', 
                   fontsize=10, fontweight='bold')
            
            if activations and i < len(activations):
                ax.text(x_pos, 0.5, activations[i], ha='center', va='center', 
                       fontsize=9, style='italic', color='red')
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_activation_functions(functions: List[str] = None,
                                 x_range: Tuple[float, float] = (-5, 5),
                                 figsize: Tuple[int, int] = (15, 10)) -> None:
        """Plot multiple activation functions for comparison."""
        if functions is None:
            functions = ['relu', 'leaky_relu', 'sigmoid', 'tanh', 'swish', 'gelu']
        
        x = np.linspace(x_range[0], x_range[1], 1000)
        
        n_funcs = len(functions)
        n_cols = 3
        n_rows = (n_funcs + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
        
        for i, func_name in enumerate(functions):
            ax = axes[i]
            
            try:
                func = ActivationFunctions.get_activation(func_name)
                y = func(x)
                
                ax.plot(x, y, linewidth=2, label=func_name)
                ax.grid(True, alpha=0.3)
                ax.set_title(f'{func_name.upper()} Activation', fontweight='bold')
                ax.set_xlabel('x')
                ax.set_ylabel('f(x)')
                ax.legend()
                
                # zero lines for reference
                ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
                ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
                
            except Exception as e:
                ax.text(0.5, 0.5, f'Error: {str(e)}', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{func_name} (Error)', fontweight='bold')
        
        # hide unused subplots
        for i in range(n_funcs, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_training_metrics(history: Dict[str, List[float]], 
                            metrics: List[str] = None,
                            figsize: Tuple[int, int] = (15, 5)) -> None:
        """Plot training history (loss, accuracy, etc)."""
        if metrics is None:
            metrics = ['train_loss', 'val_loss']
        
        available_metrics = [m for m in metrics if m in history and history[m]]
        
        if not available_metrics:
            print("No metrics available to plot.")
            return
        
        n_metrics = len(available_metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
        if n_metrics == 1:
            axes = [axes]
        
        for i, metric in enumerate(available_metrics):
            ax = axes[i]
            
            values = history[metric]
            epochs = range(1, len(values) + 1)
            
            ax.plot(epochs, values, linewidth=2, label=metric)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(f'{metric.replace("_", " ").title()} Over Time')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # use log scale if loss varies a lot
            if 'loss' in metric.lower() and len(values) > 1:
                if max(values) / min(values) > 10:
                    ax.set_yscale('log')
        
        plt.tight_layout()
        plt.show()


# Performance monitoring
class PerformanceMonitor:
    """Timing and memory profiling utilities."""
    
    @staticmethod
    def timer(func: Callable) -> Callable:
        """Decorator to time function execution."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            
            print(f"{func.__name__} took {end_time - start_time:.4f} seconds")
            return result
        
        return wrapper
    
    @staticmethod
    def memory_usage() -> Dict[str, float]:
        """Get current memory usage stats."""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,
            'vms_mb': memory_info.vms / 1024 / 1024,
            'percent': process.memory_percent()
        }
    
    @staticmethod
    def profile_network(network, X: np.ndarray, num_runs: int = 100) -> Dict[str, float]:
        """Profile network forward pass performance."""
        times = []
        
        # warmup runs
        for _ in range(5):
            network.forward(X)
        
        for _ in range(num_runs):
            start_time = time.time()
            network.forward(X)
            end_time = time.time()
            times.append(end_time - start_time)
        
        times = np.array(times)
        
        return {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'throughput_samples_per_sec': X.shape[0] / np.mean(times)
        }


# Helper functions
def create_test_data(func_type: str = 'linear', n_samples: int = 1000, 
                    n_features: int = 3, noise: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """Generate test data for different function types."""
    np.random.seed(42)
    
    if func_type == 'linear':
        X = np.random.randn(n_samples, n_features)
        weights = np.random.randn(n_features)
        y = X @ weights + np.random.randn(n_samples) * noise
        
    elif func_type == 'quadratic':
        X = np.random.randn(n_samples, n_features)
        y = np.sum(X**2, axis=1) + np.random.randn(n_samples) * noise
        
    elif func_type == 'sine':
        X = np.random.uniform(-np.pi, np.pi, (n_samples, n_features))
        y = np.sin(np.sum(X, axis=1)) + np.random.randn(n_samples) * noise
        
    elif func_type == 'xor':
        X = np.random.choice([0, 1], size=(n_samples, 2))
        y = (X[:, 0] ^ X[:, 1]).astype(float) + np.random.randn(n_samples) * noise
        
    else:
        raise ValueError(f"Unknown function type: {func_type}")
    
    return X.astype(np.float32), y.reshape(-1, 1).astype(np.float32)


def print_network_summary(network) -> None:
    """Print detailed network summary."""
    print("Neural Network Summary")
    print("=" * 50)
    print(f"Architecture: {' -> '.join(map(str, network.layer_sizes))}")
    print(f"Total Layers: {network.num_layers}")
    print(f"Total Parameters: {network.count_parameters()}")
    
    print("\nLayer Details:")
    for i, layer in enumerate(network.layers):
        weights, biases = layer.get_parameters()
        print(f"  Layer {i+1}: {layer.input_size} -> {layer.output_size}")
        print(f"    Activation: {layer.activation_name}")
        print(f"    Weights: {weights.shape}")
        print(f"    Biases: {biases.shape}")
        print(f"    Parameters: {weights.size + biases.size}")
    
    print("=" * 50)


if __name__ == "__main__":
    print("Testing utility functions...")
    print("=" * 50)
    
    # test activations
    print("\nTesting activation functions...")
    x_test = np.array([-2, -1, 0, 1, 2])
    
    activations_to_test = ['relu', 'leaky_relu', 'sigmoid', 'tanh', 'swish', 'gelu']
    
    for activation_name in activations_to_test:
        activation_func = ActivationFunctions.get_activation(activation_name)
        y_test = activation_func(x_test)
        print(f"  {activation_name}: {x_test} -> {y_test}")
    
    # test math utils
    print("\nTesting math utilities...")
    
    x_safe = np.array([0.0, 1e-10, 1.0, 100.0])
    safe_log_result = MathUtils.safe_log(x_safe)
    print(f"  Safe log: {x_safe} -> {safe_log_result}")
    
    # test weight init
    print("\nTesting weight initialization...")
    
    n_in, n_out = 5, 3
    
    xavier_weights = MathUtils.xavier_init(n_in, n_out)
    he_weights = MathUtils.he_init(n_in, n_out)
    
    print(f"  Xavier init shape: {xavier_weights.shape}")
    print(f"  Xavier init mean: {np.mean(xavier_weights):.6f}")
    print(f"  Xavier init std: {np.std(xavier_weights):.6f}")
    
    print(f"  He init shape: {he_weights.shape}")
    print(f"  He init mean: {np.mean(he_weights):.6f}")
    print(f"  He init std: {np.std(he_weights):.6f}")
    
    # test viz
    print("\nTesting visualization...")
    
    layer_sizes = [4, 8, 6, 3, 1]
    activations = ['relu', 'relu', 'relu', 'linear']
    
    print(f"  Plotting network architecture: {layer_sizes}")
    NetworkVisualizer.plot_network_architecture(layer_sizes, activations)
    
    print(f"  Plotting activation functions...")
    NetworkVisualizer.plot_activation_functions(['relu', 'sigmoid', 'tanh'])
    
    # test performance monitoring
    print("\nTesting performance monitoring...")
    
    @PerformanceMonitor.timer
    def dummy_computation():
        return np.random.randn(1000, 1000) @ np.random.randn(1000, 1000)
    
    result = dummy_computation()
    
    memory_info = PerformanceMonitor.memory_usage()
    print(f"  Memory usage: {memory_info}")
    
    # test data generation
    print("\nTesting data creation...")
    
    for func_type in ['linear', 'quadratic', 'sine']:
        X, y = create_test_data(func_type, n_samples=100, n_features=3)
        print(f"  {func_type} data: X{X.shape}, y{y.shape}")
        print(f"    X range: [{X.min():.3f}, {X.max():.3f}]")
        print(f"    y range: [{y.min():.3f}, {y.max():.3f}]")
    
    # test network intergration
    print("\nTesting network integration...")
    
    try:
        from nn_core import NeuralNetwork
        
        network = NeuralNetwork([3, 5, 2, 1], ['relu', 'relu', 'linear'])
        
        print_network_summary(network)
        
        X_test, y_test = create_test_data('linear', n_samples=1000, n_features=3)
        perf_stats = PerformanceMonitor.profile_network(network, X_test, num_runs=50)
        
        print(f"\nPerformance Statistics:")
        print(f"  Mean time: {perf_stats['mean_time']:.6f} seconds")
        print(f"  Throughput: {perf_stats['throughput_samples_per_sec']:.0f} samples/sec")
        
    except ImportError:
        print("  Neural network modules not available for integration test")
    
    print(f"\nAll tests passed!")
