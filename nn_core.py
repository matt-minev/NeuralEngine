"""
Core neural network implementation.

Includes Layer and NeuralNetwork classes, forward propagation,
and loss functions for training.
"""

import numpy as np
from typing import List, Tuple, Callable, Optional, Union
import autograd.numpy as anp
from autograd import grad


class Layer:
    """Single neural network layer: h = activation(W*x + b)"""
    
    def __init__(self, input_size: int, output_size: int, activation: str = 'relu'):
        """
        Initialize layer with weights and biases.
        
        Uses He initialization for weights (good for ReLU).
        """
        self.input_size = input_size
        self.output_size = output_size
        self.activation_name = activation
        
        # He initialization - works well with ReLU
        self.weights = anp.random.randn(output_size, input_size) * anp.sqrt(2.0 / input_size)
        self.biases = anp.random.randn(output_size) * 0.01
        
        self.activation = self._get_activation_function(activation)
    
    def _get_activation_function(self, activation: str) -> Callable:
        """Get activation function by name."""
        activations = {
            'relu': lambda x: anp.maximum(0, x),
            'leaky_relu': lambda x: anp.maximum(0.01 * x, x),
            'elu': lambda x: anp.where(x > 0, x, anp.exp(x) - 1),
            'sigmoid': lambda x: 1 / (1 + anp.exp(-anp.clip(x, -500, 500))),
            'tanh': lambda x: anp.tanh(x),
            'swish': lambda x: x * (1 / (1 + anp.exp(-anp.clip(x, -500, 500)))),
            'gelu': lambda x: 0.5 * x * (1 + anp.tanh(anp.sqrt(2 / anp.pi) * (x + 0.044715 * x**3))),
            'softmax': lambda x: anp.exp(x - anp.max(x, axis=-1, keepdims=True)) / anp.sum(anp.exp(x - anp.max(x, axis=-1, keepdims=True)), axis=-1, keepdims=True),
            'linear': lambda x: x
        }
        
        if activation not in activations:
            raise ValueError(f"Unknown activation: {activation}. Use: {list(activations.keys())}")
        
        return activations[activation]
    
    def forward(self, x: anp.ndarray) -> anp.ndarray:
        """Forward pass through layer."""
        # handle both single samples and batches
        if x.ndim == 1:
            linear_output = anp.dot(self.weights, x) + self.biases
        else:
            linear_output = anp.dot(x, self.weights.T) + self.biases
        
        return self.activation(linear_output)
    
    def get_parameters(self) -> Tuple[anp.ndarray, anp.ndarray]:
        """Return weights and biases."""
        return self.weights, self.biases
    
    def set_parameters(self, weights: anp.ndarray, biases: anp.ndarray):
        """Update layer parameters."""
        self.weights = weights
        self.biases = biases
    
    def __getstate__(self):
        """Prepare for pickling - remove unpicklable lambda."""
        state = self.__dict__.copy()
        if 'activation' in state:
            del state['activation']
        return state
    
    def __setstate__(self, state):
        """Restore after unpickling - recreate activation function."""
        self.__dict__.update(state)
        self.activation = self._get_activation_function(self.activation_name)
    
    def __repr__(self) -> str:
        return f"Layer({self.input_size} -> {self.output_size}, {self.activation_name})"


class NeuralNetwork:
    """Multi-layer neural network for function approximation."""
    
    def __init__(self, layer_sizes: List[int], activations: Optional[List[str]] = None):
        """
        Create network with specified architecture.
        
        Args:
            layer_sizes: [input_size, hidden1, hidden2, ..., output_size]
            activations: activation function for each layer
        
        Example:
            nn = NeuralNetwork([2, 5, 3, 1], ['relu', 'relu', 'linear'])
        """
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes) - 1
        
        # default to ReLU for hidden layers, linear for output
        if activations is None:
            activations = ['relu'] * (self.num_layers - 1) + ['linear']
        
        if len(activations) != self.num_layers:
            raise ValueError(f"Need {self.num_layers} activations, got {len(activations)}")
        
        # create all layers
        self.layers = []
        for i in range(self.num_layers):
            layer = Layer(
                input_size=layer_sizes[i],
                output_size=layer_sizes[i + 1],
                activation=activations[i]
            )
            self.layers.append(layer)
        
        print(f"Neural Network Created:")
        print(f"  Architecture: {' -> '.join(map(str, layer_sizes))}")
        print(f"  Activations: {activations}")
        print(f"  Total Parameters: {self.count_parameters()}")
    
    def forward(self, x: anp.ndarray) -> anp.ndarray:
        """
        Forward prop through entire network.
        
        Sequentially applies each layer transformation.
        """
        current_output = x
        
        for i, layer in enumerate(self.layers):
            current_output = layer.forward(current_output)
            
            # debug output if needed
            if hasattr(self, '_debug') and self._debug:
                print(f"  Layer {i+1} output shape: {current_output.shape}")
        
        return current_output
    
    def predict(self, x: anp.ndarray) -> anp.ndarray:
        """Make predictions (alias for forward)."""
        return self.forward(x)
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        total = 0
        for layer in self.layers:
            weights, biases = layer.get_parameters()
            total += weights.size + biases.size
        return total
    
    def get_all_parameters(self) -> List[anp.ndarray]:
        """Get all params as flat list."""
        params = []
        for layer in self.layers:
            weights, biases = layer.get_parameters()
            params.extend([weights, biases])
        return params
    
    def set_all_parameters(self, params: List[anp.ndarray]):
        """Set all params from flat list."""
        param_idx = 0
        for layer in self.layers:
            weights = params[param_idx]
            biases = params[param_idx + 1]
            layer.set_parameters(weights, biases)
            param_idx += 2
    
    def __getstate__(self):
        """Prepare for pickling."""
        state = self.__dict__.copy()
        return state
    
    def __setstate__(self, state):
        """Restore after unpickling."""
        self.__dict__.update(state)
        # make sure all layers have activation functions restored
        for layer in self.layers:
            if not hasattr(layer, 'activation'):
                layer.activation = layer._get_activation_function(layer.activation_name)
    
    def __repr__(self) -> str:
        return f"NeuralNetwork({self.layer_sizes}, {self.count_parameters()} params)"


# Loss functions
def mean_squared_error(y_true: anp.ndarray, y_pred: anp.ndarray) -> anp.ndarray:
    """
    MSE loss: (1/2) * mean((y_true - y_pred)^2)
    
    The 1/2 factor makes derivative cleaner.
    """
    return anp.mean(0.5 * (y_true - y_pred) ** 2)


def mean_absolute_error(y_true: anp.ndarray, y_pred: anp.ndarray) -> anp.ndarray:
    """MAE loss: mean(|y_true - y_pred|)"""
    return anp.mean(anp.abs(y_true - y_pred))


def cross_entropy_loss(y_true: anp.ndarray, y_pred: anp.ndarray) -> anp.ndarray:
    """
    Cross-entropy loss for classification.
    
    Use this with softmax output for digit recogntion.
    
    Args:
        y_true: one-hot encoded labels (batch_size, num_classes)
        y_pred: predicted probabilities from softmax
    """
    # prevent log(0)
    epsilon = 1e-15
    y_pred_clipped = anp.clip(y_pred, epsilon, 1 - epsilon)
    
    loss = -anp.sum(y_true * anp.log(y_pred_clipped)) / y_true.shape[0]
    
    return loss


def categorical_accuracy(y_true: anp.ndarray, y_pred: anp.ndarray) -> float:
    """Calculate classification accuracy as percentage."""
    predicted_classes = anp.argmax(y_pred, axis=1)
    true_classes = anp.argmax(y_true, axis=1)
    accuracy = anp.mean(predicted_classes == true_classes) * 100
    return float(accuracy)


# Utility for testing
def create_sample_data(n_samples: int = 100) -> Tuple[anp.ndarray, anp.ndarray]:
    """
    Generate test data for network.
    
    Creates data for: y = 2*x1 + 3*x2 + 1 + noise
    """
    X = anp.random.randn(n_samples, 2)
    
    # simple linear function with noise
    y = 2 * X[:, 0] + 3 * X[:, 1] + 1 + 0.1 * anp.random.randn(n_samples)
    
    return X, y.reshape(-1, 1)


if __name__ == "__main__":
    print("Testing Neural Network Core")
    print("=" * 40)
    
    # create sample data
    X, y = create_sample_data(50)
    print(f"Sample Data Created:")
    print(f"  Input shape: {X.shape}")
    print(f"  Target shape: {y.shape}")
    print(f"  Target function: y = 2*x1 + 3*x2 + 1")
    
    # create network
    print(f"\nCreating Neural Network...")
    nn = NeuralNetwork([2, 5, 3, 1], ['relu', 'relu', 'linear'])
    
    # test forward pass
    print(f"\nTesting Forward Pass...")
    predictions = nn.predict(X)
    print(f"  Prediction shape: {predictions.shape}")
    print(f"  Sample predictions: {predictions[:5].flatten()}")
    print(f"  Sample targets: {y[:5].flatten()}")
    
    # test loss computation
    print(f"\nTesting Loss Computation...")
    mse_loss = mean_squared_error(y, predictions)
    mae_loss = mean_absolute_error(y, predictions)
    print(f"  MSE Loss: {mse_loss:.6f}")
    print(f"  MAE Loss: {mae_loss:.6f}")
    
    # test parameter acess
    print(f"\nTesting Parameter Access...")
    params = nn.get_all_parameters()
    print(f"  Number of parameter arrays: {len(params)}")
    print(f"  Total parameters: {nn.count_parameters()}")
    
    print(f"\nAll tests passed!")
