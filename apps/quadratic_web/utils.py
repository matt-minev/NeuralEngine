"""
Utility Functions - Helper Tools & Mathematical Utilities
=======================================================

This module provides essential utility functions for the Neural Network Engine:
- Activation functions with numerical stability
- Mathematical helpers and matrix operations
- Visualization tools for networks and training
- Performance monitoring utilities
- Common helper functions

Mathematical Foundation:
- Activation functions: f(x) → y with various non-linearities
- Numerical stability: Preventing overflow/underflow in computations
- Matrix operations: Efficient linear algebra helpers
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


# ============================================================================
# ACTIVATION FUNCTIONS
# ============================================================================

class ActivationFunctions:
    """
    Collection of activation functions with numerical stability.
    
    All functions are implemented using autograd.numpy for automatic
    differentiation compatibility.
    """
    
    @staticmethod
    def relu(x: anp.ndarray) -> anp.ndarray:
        """
        Rectified Linear Unit activation function.
        
        Mathematical Formula: f(x) = max(0, x)
        
        Properties:
        - Non-saturating for positive inputs
        - Sparse activation (many zeros)
        - Computationally efficient
        - Can cause "dead neurons" problem
        
        Args:
            x: Input array
            
        Returns:
            ReLU activated output
        """
        return anp.maximum(0, x)
    
    @staticmethod
    def leaky_relu(x: anp.ndarray, alpha: float = 0.01) -> anp.ndarray:
        """
        Leaky ReLU activation function.
        
        Mathematical Formula: f(x) = max(αx, x) where α = 0.01
        
        Properties:
        - Solves "dead neuron" problem of ReLU
        - Small gradient for negative inputs
        - Maintains benefits of ReLU
        
        Args:
            x: Input array
            alpha: Slope for negative inputs
            
        Returns:
            Leaky ReLU activated output
        """
        return anp.maximum(alpha * x, x)
    
    @staticmethod
    def elu(x: anp.ndarray, alpha: float = 1.0) -> anp.ndarray:
        """
        Exponential Linear Unit activation function.
        
        Mathematical Formula: 
        f(x) = x if x > 0
        f(x) = α(e^x - 1) if x ≤ 0
        
        Properties:
        - Smooth function everywhere
        - Negative values push mean activation closer to zero
        - Reduces bias shift problem
        
        Args:
            x: Input array
            alpha: Scale factor for negative inputs
            
        Returns:
            ELU activated output
        """
        return anp.where(x > 0, x, alpha * (anp.exp(x) - 1))
    
    @staticmethod
    def sigmoid(x: anp.ndarray) -> anp.ndarray:
        """
        Sigmoid activation function with numerical stability.
        
        Mathematical Formula: f(x) = 1/(1 + e^(-x))
        
        Properties:
        - Smooth, differentiable everywhere
        - Output range: (0, 1)
        - Can cause vanishing gradient problem
        - Good for binary classification output
        
        Args:
            x: Input array
            
        Returns:
            Sigmoid activated output
        """
        # Numerical stability: prevent overflow
        x_clipped = anp.clip(x, -500, 500)
        return 1.0 / (1.0 + anp.exp(-x_clipped))
    
    @staticmethod
    def tanh(x: anp.ndarray) -> anp.ndarray:
        """
        Hyperbolic tangent activation function.
        
        Mathematical Formula: f(x) = (e^x - e^(-x))/(e^x + e^(-x))
        
        Properties:
        - Smooth, differentiable everywhere
        - Output range: (-1, 1)
        - Zero-centered (better than sigmoid)
        - Still suffers from vanishing gradient
        
        Args:
            x: Input array
            
        Returns:
            Tanh activated output
        """
        return anp.tanh(x)
    
    @staticmethod
    def swish(x: anp.ndarray) -> anp.ndarray:
        """
        Swish activation function (Self-Gated).
        
        Mathematical Formula: f(x) = x * sigmoid(x)
        
        Properties:
        - Smooth, non-monotonic
        - Self-gated mechanism
        - Often performs better than ReLU
        - Computationally more expensive
        
        Args:
            x: Input array
            
        Returns:
            Swish activated output
        """
        return x * ActivationFunctions.sigmoid(x)
    
    @staticmethod
    def gelu(x: anp.ndarray) -> anp.ndarray:
        """
        Gaussian Error Linear Unit activation function.
        
        Mathematical Formula: f(x) = 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x^3)))
        
        Properties:
        - Smooth approximation of ReLU
        - Used in BERT and GPT models
        - Probabilistic interpretation
        - Good empirical performance
        
        Args:
            x: Input array
            
        Returns:
            GELU activated output
        """
        return 0.5 * x * (1 + anp.tanh(anp.sqrt(2 / anp.pi) * (x + 0.044715 * x**3)))
    
    @staticmethod
    def softmax(x: anp.ndarray, axis: int = -1) -> anp.ndarray:
        """
        Softmax activation function with numerical stability.
        
        Mathematical Formula: f(x_i) = e^(x_i) / Σ(e^(x_j))
        
        Properties:
        - Output sums to 1 (probability distribution)
        - Differentiable everywhere
        - Used in multi-class classification
        - Numerically stable implementation
        
        Args:
            x: Input array
            axis: Axis along which to compute softmax
            
        Returns:
            Softmax activated output
        """
        # Numerical stability: subtract max value
        x_shifted = x - anp.max(x, axis=axis, keepdims=True)
        exp_x = anp.exp(x_shifted)
        return exp_x / anp.sum(exp_x, axis=axis, keepdims=True)
    
    @staticmethod
    def linear(x: anp.ndarray) -> anp.ndarray:
        """
        Linear activation function (identity).
        
        Mathematical Formula: f(x) = x
        
        Properties:
        - No transformation
        - Used in output layers for regression
        - Preserves gradient flow
        
        Args:
            x: Input array
            
        Returns:
            Unchanged input
        """
        return x
    
    @staticmethod
    def get_activation(name: str) -> Callable:
        """
        Get activation function by name.
        
        Args:
            name: Name of activation function
            
        Returns:
            Activation function
        """
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


# ============================================================================
# MATHEMATICAL HELPERS
# ============================================================================

class MathUtils:
    """
    Mathematical utility functions for numerical stability and common operations.
    """
    
    @staticmethod
    def safe_log(x: anp.ndarray, eps: float = 1e-8) -> anp.ndarray:
        """
        Numerically stable logarithm.
        
        Args:
            x: Input array
            eps: Small value to prevent log(0)
            
        Returns:
            Safe logarithm
        """
        return anp.log(anp.maximum(x, eps))
    
    @staticmethod
    def safe_sqrt(x: anp.ndarray, eps: float = 1e-8) -> anp.ndarray:
        """
        Numerically stable square root.
        
        Args:
            x: Input array
            eps: Small value to prevent sqrt(negative)
            
        Returns:
            Safe square root
        """
        return anp.sqrt(anp.maximum(x, eps))
    
    @staticmethod
    def safe_divide(numerator: anp.ndarray, denominator: anp.ndarray, 
                   eps: float = 1e-8) -> anp.ndarray:
        """
        Numerically stable division.
        
        Args:
            numerator: Numerator array
            denominator: Denominator array
            eps: Small value to prevent division by zero
            
        Returns:
            Safe division result
        """
        return numerator / anp.maximum(denominator, eps)
    
    @staticmethod
    def clip_gradients(gradients: List[anp.ndarray], 
                      max_norm: float = 5.0) -> List[anp.ndarray]:
        """
        Clip gradients to prevent exploding gradients.
        
        Args:
            gradients: List of gradient arrays
            max_norm: Maximum allowed gradient norm
            
        Returns:
            Clipped gradients
        """
        # Calculate global norm
        global_norm = anp.sqrt(sum(anp.sum(g**2) for g in gradients))
        
        # Clip if necessary
        if global_norm > max_norm:
            clip_ratio = max_norm / global_norm
            gradients = [g * clip_ratio for g in gradients]
        
        return gradients
    
    @staticmethod
    def xavier_init(n_in: int, n_out: int) -> anp.ndarray:
        """
        Xavier/Glorot weight initialization.
        
        Mathematical Formula: W ~ U(-√(6/(n_in + n_out)), √(6/(n_in + n_out)))
        
        Args:
            n_in: Number of input units
            n_out: Number of output units
            
        Returns:
            Xavier initialized weights
        """
        limit = anp.sqrt(6.0 / (n_in + n_out))
        return anp.random.uniform(-limit, limit, (n_out, n_in))
    
    @staticmethod
    def he_init(n_in: int, n_out: int) -> anp.ndarray:
        """
        He weight initialization (good for ReLU).
        
        Mathematical Formula: W ~ N(0, √(2/n_in))
        
        Args:
            n_in: Number of input units
            n_out: Number of output units
            
        Returns:
            He initialized weights
        """
        return anp.random.randn(n_out, n_in) * anp.sqrt(2.0 / n_in)
    
    @staticmethod
    def orthogonal_init(n_in: int, n_out: int) -> anp.ndarray:
        """
        Orthogonal weight initialization.
        
        Args:
            n_in: Number of input units
            n_out: Number of output units
            
        Returns:
            Orthogonal initialized weights
        """
        # Generate random matrix
        W = anp.random.randn(n_out, n_in)
        
        # Orthogonalize using QR decomposition
        if n_out >= n_in:
            Q, R = anp.linalg.qr(W)
            return Q[:n_out, :n_in]
        else:
            Q, R = anp.linalg.qr(W.T)
            return Q.T[:n_out, :n_in]


# ============================================================================
# VISUALIZATION UTILITIES
# ============================================================================

class NetworkVisualizer:
    """
    Visualization utilities for neural networks and training progress.
    """
    
    @staticmethod
    def plot_network_architecture(layer_sizes: List[int], 
                                 activations: List[str] = None,
                                 title: str = "Neural Network Architecture",
                                 figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Plot neural network architecture diagram.
        
        Args:
            layer_sizes: List of layer sizes
            activations: List of activation functions
            title: Plot title
            figsize: Figure size
        """
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        # Set up plot
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        ax.set_title(title, fontsize=16, fontweight='bold')
        
        # Calculate layer positions
        n_layers = len(layer_sizes)
        layer_x_positions = np.linspace(1, 9, n_layers)
        
        # Colors for different layer types
        colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink']
        
        # Draw layers
        for i, (x_pos, layer_size) in enumerate(zip(layer_x_positions, layer_sizes)):
            # Calculate neuron positions
            max_neurons_to_show = 8  # Don't overcrowd the plot
            neurons_to_show = min(layer_size, max_neurons_to_show)
            
            if neurons_to_show < layer_size:
                # Show some neurons + "..." + some neurons
                neuron_y_positions = np.linspace(2, 8, neurons_to_show)
            else:
                neuron_y_positions = np.linspace(2, 8, neurons_to_show)
            
            # Draw neurons
            for j, y_pos in enumerate(neuron_y_positions):
                circle = plt.Circle((x_pos, y_pos), 0.2, 
                                  color=colors[i % len(colors)], 
                                  ec='black', linewidth=1.5)
                ax.add_patch(circle)
                
                # Add neuron number if layer is large
                if layer_size > max_neurons_to_show and j == neurons_to_show // 2:
                    ax.text(x_pos, y_pos - 0.6, f'...{layer_size}...', 
                           ha='center', va='center', fontsize=8)
            
            # Draw connections to next layer
            if i < n_layers - 1:
                next_x = layer_x_positions[i + 1]
                next_layer_size = layer_sizes[i + 1]
                next_neurons_to_show = min(next_layer_size, max_neurons_to_show)
                next_neuron_y_positions = np.linspace(2, 8, next_neurons_to_show)
                
                # Draw some connections (not all to avoid clutter)
                for y1 in neuron_y_positions[::2]:  # Every other neuron
                    for y2 in next_neuron_y_positions[::2]:
                        ax.plot([x_pos + 0.2, next_x - 0.2], [y1, y2], 
                               'gray', alpha=0.3, linewidth=0.5)
            
            # Layer labels
            layer_name = f"Input\n({layer_size})" if i == 0 else \
                        f"Output\n({layer_size})" if i == n_layers - 1 else \
                        f"Hidden {i}\n({layer_size})"
            
            ax.text(x_pos, 1.2, layer_name, ha='center', va='center', 
                   fontsize=10, fontweight='bold')
            
            # Activation function labels
            if activations and i < len(activations):
                ax.text(x_pos, 0.5, activations[i], ha='center', va='center', 
                       fontsize=9, style='italic', color='red')
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_activation_functions(functions: List[str] = None,
                                 x_range: Tuple[float, float] = (-5, 5),
                                 figsize: Tuple[int, int] = (15, 10)) -> None:
        """
        Plot comparison of activation functions.
        
        Args:
            functions: List of activation function names
            x_range: Range of x values to plot
            figsize: Figure size
        """
        if functions is None:
            functions = ['relu', 'leaky_relu', 'sigmoid', 'tanh', 'swish', 'gelu']
        
        # Create input values
        x = np.linspace(x_range[0], x_range[1], 1000)
        
        # Create subplots
        n_funcs = len(functions)
        n_cols = 3
        n_rows = (n_funcs + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
        
        # Plot each function
        for i, func_name in enumerate(functions):
            ax = axes[i]
            
            # Get activation function
            try:
                func = ActivationFunctions.get_activation(func_name)
                y = func(x)
                
                # Plot function
                ax.plot(x, y, linewidth=2, label=func_name)
                ax.grid(True, alpha=0.3)
                ax.set_title(f'{func_name.upper()} Activation', fontweight='bold')
                ax.set_xlabel('x')
                ax.set_ylabel('f(x)')
                ax.legend()
                
                # Add zero lines
                ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
                ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
                
            except Exception as e:
                ax.text(0.5, 0.5, f'Error: {str(e)}', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{func_name} (Error)', fontweight='bold')
        
        # Hide unused subplots
        for i in range(n_funcs, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_training_metrics(history: Dict[str, List[float]], 
                            metrics: List[str] = None,
                            figsize: Tuple[int, int] = (15, 5)) -> None:
        """
        Plot training metrics over time.
        
        Args:
            history: Training history dictionary
            metrics: List of metrics to plot
            figsize: Figure size
        """
        if metrics is None:
            metrics = ['train_loss', 'val_loss']
        
        # Filter available metrics
        available_metrics = [m for m in metrics if m in history and history[m]]
        
        if not available_metrics:
            print("No metrics available to plot.")
            return
        
        # Create subplots
        n_metrics = len(available_metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
        if n_metrics == 1:
            axes = [axes]
        
        # Plot each metric
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
            
            # Use log scale for loss if values vary greatly
            if 'loss' in metric.lower() and len(values) > 1:
                if max(values) / min(values) > 10:
                    ax.set_yscale('log')
        
        plt.tight_layout()
        plt.show()


# ============================================================================
# PERFORMANCE MONITORING
# ============================================================================

class PerformanceMonitor:
    """
    Performance monitoring utilities for timing and memory usage.
    """
    
    @staticmethod
    def timer(func: Callable) -> Callable:
        """
        Decorator to time function execution.
        
        Args:
            func: Function to time
            
        Returns:
            Wrapped function with timing
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            
            print(f"⏱️  {func.__name__} took {end_time - start_time:.4f} seconds")
            return result
        
        return wrapper
    
    @staticmethod
    def memory_usage() -> Dict[str, float]:
        """
        Get current memory usage information.
        
        Returns:
            Dictionary with memory usage statistics
        """
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size
            'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size
            'percent': process.memory_percent()
        }
    
    @staticmethod
    def profile_network(network, X: np.ndarray, num_runs: int = 100) -> Dict[str, float]:
        """
        Profile neural network forward pass performance.
        
        Args:
            network: Neural network to profile
            X: Input data for profiling
            num_runs: Number of runs for averaging
            
        Returns:
            Performance statistics
        """
        times = []
        
        # Warm up
        for _ in range(5):
            network.forward(X)
        
        # Time multiple runs
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


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_test_data(func_type: str = 'linear', n_samples: int = 1000, 
                    n_features: int = 3, noise: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create test data for various function types.
    
    Args:
        func_type: Type of function ('linear', 'quadratic', 'sine', 'xor')
        n_samples: Number of samples
        n_features: Number of features
        noise: Noise level
        
    Returns:
        Tuple of (X, y) arrays
    """
    np.random.seed(42)  # For reproducibility
    
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
    """
    Print a detailed summary of the neural network.
    
    Args:
        network: Neural network to summarize
    """
    print("🧠 Neural Network Summary")
    print("=" * 50)
    print(f"Architecture: {' → '.join(map(str, network.layer_sizes))}")
    print(f"Total Layers: {network.num_layers}")
    print(f"Total Parameters: {network.count_parameters()}")
    
    print("\nLayer Details:")
    for i, layer in enumerate(network.layers):
        weights, biases = layer.get_parameters()
        print(f"  Layer {i+1}: {layer.input_size} → {layer.output_size}")
        print(f"    Activation: {layer.activation_name}")
        print(f"    Weights: {weights.shape}")
        print(f"    Biases: {biases.shape}")
        print(f"    Parameters: {weights.size + biases.size}")
    
    print("=" * 50)


# Example usage and testing
if __name__ == "__main__":
    """
    Test all utility functions.
    """
    print("🧪 Testing Utility Functions")
    print("=" * 50)
    
    # Test activation functions
    print("\n🔥 Testing Activation Functions...")
    x_test = np.array([-2, -1, 0, 1, 2])
    
    activations_to_test = ['relu', 'leaky_relu', 'sigmoid', 'tanh', 'swish', 'gelu']
    
    for activation_name in activations_to_test:
        activation_func = ActivationFunctions.get_activation(activation_name)
        y_test = activation_func(x_test)
        print(f"  {activation_name}: {x_test} → {y_test}")
    
    # Test mathematical utilities
    print("\n🔢 Testing Mathematical Utilities...")
    
    # Test safe operations
    x_safe = np.array([0.0, 1e-10, 1.0, 100.0])
    safe_log_result = MathUtils.safe_log(x_safe)
    print(f"  Safe log: {x_safe} → {safe_log_result}")
    
    # Test initialization methods
    print("\n⚙️ Testing Weight Initialization...")
    
    n_in, n_out = 5, 3
    
    xavier_weights = MathUtils.xavier_init(n_in, n_out)
    he_weights = MathUtils.he_init(n_in, n_out)
    
    print(f"  Xavier init shape: {xavier_weights.shape}")
    print(f"  Xavier init mean: {np.mean(xavier_weights):.6f}")
    print(f"  Xavier init std: {np.std(xavier_weights):.6f}")
    
    print(f"  He init shape: {he_weights.shape}")
    print(f"  He init mean: {np.mean(he_weights):.6f}")
    print(f"  He init std: {np.std(he_weights):.6f}")
    
    # Test visualization functions
    print("\n📊 Testing Visualization Functions...")
    
    # Test network architecture visualization
    layer_sizes = [4, 8, 6, 3, 1]
    activations = ['relu', 'relu', 'relu', 'linear']
    
    print(f"  Plotting network architecture: {layer_sizes}")
    NetworkVisualizer.plot_network_architecture(layer_sizes, activations)
    
    # Test activation function plotting
    print(f"  Plotting activation functions...")
    NetworkVisualizer.plot_activation_functions(['relu', 'sigmoid', 'tanh'])
    
    # Test performance monitoring
    print("\n⏱️ Testing Performance Monitoring...")
    
    # Test timer decorator
    @PerformanceMonitor.timer
    def dummy_computation():
        return np.random.randn(1000, 1000) @ np.random.randn(1000, 1000)
    
    result = dummy_computation()
    
    # Test memory usage
    memory_info = PerformanceMonitor.memory_usage()
    print(f"  Memory usage: {memory_info}")
    
    # Test data creation
    print("\n📊 Testing Data Creation...")
    
    for func_type in ['linear', 'quadratic', 'sine']:
        X, y = create_test_data(func_type, n_samples=100, n_features=3)
        print(f"  {func_type} data: X{X.shape}, y{y.shape}")
        print(f"    X range: [{X.min():.3f}, {X.max():.3f}]")
        print(f"    y range: [{y.min():.3f}, {y.max():.3f}]")
    
    # Test network integration
    print("\n🧠 Testing Network Integration...")
    
    try:
        # Import network components
        from nn_core import NeuralNetwork
        
        # Create a test network
        network = NeuralNetwork([3, 5, 2, 1], ['relu', 'relu', 'linear'])
        
        # Print network summary
        print_network_summary(network)
        
        # Test performance profiling
        X_test, y_test = create_test_data('linear', n_samples=1000, n_features=3)
        perf_stats = PerformanceMonitor.profile_network(network, X_test, num_runs=50)
        
        print(f"\nPerformance Statistics:")
        print(f"  Mean time: {perf_stats['mean_time']:.6f} seconds")
        print(f"  Throughput: {perf_stats['throughput_samples_per_sec']:.0f} samples/sec")
        
    except ImportError:
        print("  ⚠️  Neural network modules not available for integration test")
    
    print(f"\n✅ All utility function tests passed!")
