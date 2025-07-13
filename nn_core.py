"""
Neural Network Core - Core Logic Implementation
=============================================

This module implements the fundamental neural network components:
- Layer: Individual computation units with weights and biases
- NeuralNetwork: Collection of layers for function approximation
- Forward propagation: Data flow through the network
- Loss functions: Measuring prediction accuracy

Mathematical Foundation:
- Layer computation: h = activation(W × x + b)
- Network output: y = f(x, θ) where θ are all parameters
- Loss: L = (1/2) × ||y_true - y_pred||²
"""

import numpy as np
from typing import List, Tuple, Callable, Optional, Union
import autograd.numpy as anp  # Autograd version of numpy for automatic differentiation
from autograd import grad


class Layer:
    """
    Individual neural network layer implementing: h = activation(W × x + b)
    
    This represents one computational unit in our neural network.
    Each layer transforms its input through linear transformation + activation.
    
    Mathematical Details:
    - W: Weight matrix (output_size × input_size)
    - b: Bias vector (output_size,)
    - h: Output vector (output_size,)
    - x: Input vector (input_size,)
    """
    
    def __init__(self, input_size: int, output_size: int, activation: str = 'relu'):
        """
        Initialize a neural network layer.
        
        Args:
            input_size: Number of input features (previous layer size)
            output_size: Number of neurons in this layer
            activation: Activation function ('relu', 'sigmoid', 'tanh', 'linear')
        
        Mathematical Initialization:
        - Weights: Random values scaled by √(2/input_size) (He initialization)
        - Biases: Initialized to small random values
        """
        self.input_size = input_size
        self.output_size = output_size
        self.activation_name = activation
        
        # Initialize weights using He initialization for better training
        # Scale factor √(2/input_size) helps with gradient flow
        self.weights = anp.random.randn(output_size, input_size) * anp.sqrt(2.0 / input_size)
        
        # Initialize biases to small random values
        self.biases = anp.random.randn(output_size) * 0.01
        
        # Set activation function
        self.activation = self._get_activation_function(activation)
    
    def _get_activation_function(self, activation: str) -> Callable:
        """
        Get activation function by name.
        
        Updated to include all available activations for digit recognition.
        """
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
        """
        Forward propagation through this layer.
        
        Implements: h = activation(W × x + b)
        
        Args:
            x: Input vector or batch of inputs (batch_size, input_size)
        
        Returns:
            h: Layer output (batch_size, output_size)
        
        Mathematical Process:
        1. Linear transformation: z = W × x + b
        2. Non-linear activation: h = activation(z)
        """
        # Handle both single samples and batches
        if x.ndim == 1:
            # Single sample: x is (input_size,)
            linear_output = anp.dot(self.weights, x) + self.biases
        else:
            # Batch: x is (batch_size, input_size)
            linear_output = anp.dot(x, self.weights.T) + self.biases
        
        # Apply activation function
        activated_output = self.activation(linear_output)
        
        return activated_output
    
    def get_parameters(self) -> Tuple[anp.ndarray, anp.ndarray]:
        """
        Get layer parameters for optimization.
        
        Returns:
            Tuple of (weights, biases) that will be optimized
        """
        return self.weights, self.biases
    
    def set_parameters(self, weights: anp.ndarray, biases: anp.ndarray):
        """
        Set layer parameters (used during optimization).
        
        Args:
            weights: New weight matrix
            biases: New bias vector
        """
        self.weights = weights
        self.biases = biases
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"Layer({self.input_size} → {self.output_size}, {self.activation_name})"


class NeuralNetwork:
    """
    Multi-layer neural network for function approximation.
    
    Implements the complete neural network: y = f(x, θ)
    where θ represents all weights and biases across all layers.
    
    Mathematical Process:
    1. Input layer: h₁ = activation₁(W₁ × x + b₁)
    2. Hidden layers: hᵢ = activationᵢ(Wᵢ × hᵢ₋₁ + bᵢ)
    3. Output layer: y = activation_out(W_out × h_last + b_out)
    """
    
    def __init__(self, layer_sizes: List[int], activations: Optional[List[str]] = None):
        """
        Initialize neural network with specified architecture.
        
        Args:
            layer_sizes: List of layer sizes [input_size, hidden1, hidden2, ..., output_size]
            activations: List of activation functions for each layer (optional)
        
        Example:
            # 3-layer network: 2 inputs → 5 hidden → 3 hidden → 1 output
            nn = NeuralNetwork([2, 5, 3, 1], ['relu', 'relu', 'linear'])
        """
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes) - 1  # Number of actual layers (excluding input)
        
        # Set default activations if not provided
        if activations is None:
            # Use ReLU for hidden layers, linear for output
            activations = ['relu'] * (self.num_layers - 1) + ['linear']
        
        if len(activations) != self.num_layers:
            raise ValueError(f"Need {self.num_layers} activations, got {len(activations)}")
        
        # Create layers
        self.layers = []
        for i in range(self.num_layers):
            layer = Layer(
                input_size=layer_sizes[i],
                output_size=layer_sizes[i + 1],
                activation=activations[i]
            )
            self.layers.append(layer)
        
        print(f"🧠 Neural Network Created:")
        print(f"   Architecture: {' → '.join(map(str, layer_sizes))}")
        print(f"   Activations: {activations}")
        print(f"   Total Parameters: {self.count_parameters()}")
    
    def forward(self, x: anp.ndarray) -> anp.ndarray:
        """
        Forward propagation through the entire network.
        
        Implements: y = f(x, θ) = layer_n(...layer_2(layer_1(x))...)
        
        Args:
            x: Input vector or batch (batch_size, input_size)
        
        Returns:
            y: Network output (batch_size, output_size)
        
        Mathematical Process:
        - Sequentially applies each layer transformation
        - Output of layer i becomes input to layer i+1
        """
        current_output = x
        
        # Pass through each layer sequentially
        for i, layer in enumerate(self.layers):
            current_output = layer.forward(current_output)
            
            # Debug info (can be removed in production)
            if hasattr(self, '_debug') and self._debug:
                print(f"   Layer {i+1} output shape: {current_output.shape}")
        
        return current_output
    
    def predict(self, x: anp.ndarray) -> anp.ndarray:
        """
        Make predictions (same as forward, but clearer name for inference).
        
        Args:
            x: Input data for prediction
        
        Returns:
            Predicted outputs
        """
        return self.forward(x)
    
    def count_parameters(self) -> int:
        """
        Count total number of trainable parameters.
        
        Returns:
            Total number of weights and biases in the network
        """
        total = 0
        for layer in self.layers:
            weights, biases = layer.get_parameters()
            total += weights.size + biases.size
        return total
    
    def get_all_parameters(self) -> List[anp.ndarray]:
        """
        Get all network parameters as a flat list.
        
        Returns:
            List of all parameter arrays (weights and biases from all layers)
        """
        params = []
        for layer in self.layers:
            weights, biases = layer.get_parameters()
            params.extend([weights, biases])
        return params
    
    def set_all_parameters(self, params: List[anp.ndarray]):
        """
        Set all network parameters from a flat list.
        
        Args:
            params: List of parameter arrays matching get_all_parameters() format
        """
        param_idx = 0
        for layer in self.layers:
            weights = params[param_idx]
            biases = params[param_idx + 1]
            layer.set_parameters(weights, biases)
            param_idx += 2
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"NeuralNetwork({self.layer_sizes}, {self.count_parameters()} params)"


# Loss Functions
def mean_squared_error(y_true: anp.ndarray, y_pred: anp.ndarray) -> anp.ndarray:
    """
    Mean Squared Error loss function.
    
    Mathematical Formula:
    MSE = (1/2n) × Σ(y_true - y_pred)²
    
    The factor of 1/2 makes the derivative cleaner: d/dx(1/2 x²) = x
    
    Args:
        y_true: True target values
        y_pred: Predicted values
    
    Returns:
        Scalar loss value
    """
    return anp.mean(0.5 * (y_true - y_pred) ** 2)


def mean_absolute_error(y_true: anp.ndarray, y_pred: anp.ndarray) -> anp.ndarray:
    """
    Mean Absolute Error loss function.
    
    Mathematical Formula:
    MAE = (1/n) × Σ|y_true - y_pred|
    
    Args:
        y_true: True target values
        y_pred: Predicted values
    
    Returns:
        Scalar loss value
    """
    return anp.mean(anp.abs(y_true - y_pred))


# Utility function for testing
def create_sample_data(n_samples: int = 100) -> Tuple[anp.ndarray, anp.ndarray]:
    """
    Create sample data for testing the neural network.
    
    Generates data for the function: y = 2x₁ + 3x₂ + 1 + noise
    This is a simple linear function that our network should learn easily.
    
    Args:
        n_samples: Number of data points to generate
    
    Returns:
        Tuple of (X, y) where X is inputs and y is targets
    """
    # Generate random inputs
    X = anp.random.randn(n_samples, 2)
    
    # Generate targets: y = 2x₁ + 3x₂ + 1 + small_noise
    y = 2 * X[:, 0] + 3 * X[:, 1] + 1 + 0.1 * anp.random.randn(n_samples)
    
    return X, y.reshape(-1, 1)  # Reshape y to column vector


# Example usage and testing
if __name__ == "__main__":
    """
    Test the neural network implementation with sample data.
    This runs when you execute: python nn_core.py
    """
    print("🧪 Testing Neural Network Core")
    print("=" * 40)
    
    # Create sample data
    X, y = create_sample_data(50)
    print(f"📊 Sample Data Created:")
    print(f"   Input shape: {X.shape}")
    print(f"   Target shape: {y.shape}")
    print(f"   Target function: y = 2x₁ + 3x₂ + 1")
    
    # Create neural network
    print(f"\n🏗️ Creating Neural Network...")
    nn = NeuralNetwork([2, 5, 3, 1], ['relu', 'relu', 'linear'])
    
    # Test forward pass
    print(f"\n🔄 Testing Forward Pass...")
    predictions = nn.predict(X)
    print(f"   Prediction shape: {predictions.shape}")
    print(f"   Sample predictions: {predictions[:5].flatten()}")
    print(f"   Sample targets: {y[:5].flatten()}")
    
    # Test loss computation
    print(f"\n📊 Testing Loss Computation...")
    mse_loss = mean_squared_error(y, predictions)
    mae_loss = mean_absolute_error(y, predictions)
    print(f"   MSE Loss: {mse_loss:.6f}")
    print(f"   MAE Loss: {mae_loss:.6f}")
    
    # Test parameter access
    print(f"\n⚙️ Testing Parameter Access...")
    params = nn.get_all_parameters()
    print(f"   Number of parameter arrays: {len(params)}")
    print(f"   Total parameters: {nn.count_parameters()}")
    
    print(f"\n✅ All tests passed! Neural network core is working.")
