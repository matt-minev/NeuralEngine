"""
Automatic Differentiation Engine - Optimization & Gradient Computation
===================================================================

This module implements the optimization engine that enables neural network learning:
- Automatic gradient computation using autograd
- Multiple optimization algorithms (SGD, Adam, RMSprop)
- Learning rate scheduling
- Complete training loops with monitoring

Mathematical Foundation:
- Gradient Descent: θ_new = θ_old - α × ∇L(θ)
- Adam: θ_new = θ_old - α × m̂/(√v̂ + ε)
- Momentum: v_new = β × v_old + (1-β) × ∇L(θ)

The key insight: autograd computes ∇L(θ) automatically for any loss function!
"""

import numpy as np
import autograd.numpy as anp
from autograd import grad
from typing import List, Tuple, Dict, Callable, Optional, Any
import matplotlib.pyplot as plt
from collections import defaultdict
import time


class Optimizer:
    """
    Base class for all optimizers.
    
    Defines the interface that all optimization algorithms must implement.
    Each optimizer updates network parameters based on gradients.
    """
    
    def __init__(self, learning_rate: float = 0.001):
        """
        Initialize base optimizer.
        
        Args:
            learning_rate: Step size for parameter updates
        """
        self.learning_rate = learning_rate
        self.step_count = 0
    
    def update(self, params: List[anp.ndarray], gradients: List[anp.ndarray]) -> List[anp.ndarray]:
        """
        Update parameters using gradients.
        
        Args:
            params: Current parameter values
            gradients: Gradients of loss with respect to parameters
        
        Returns:
            Updated parameters
        """
        raise NotImplementedError("Subclasses must implement update method")
    
    def zero_grad(self):
        """Reset any internal state (used by some optimizers)."""
        pass


class SGD(Optimizer):
    """
    Stochastic Gradient Descent optimizer.
    
    Implements: θ_new = θ_old - α × ∇L(θ)
    
    This is the fundamental optimization algorithm:
    - Simple and robust
    - Good baseline for comparison
    - Works well with proper learning rate
    """
    
    def __init__(self, learning_rate: float = 0.001, momentum: float = 0.0):
        """
        Initialize SGD optimizer.
        
        Args:
            learning_rate: Step size for updates
            momentum: Momentum factor (0 = no momentum, 0.9 = strong momentum)
        
        Mathematical Details:
        - Pure SGD: θ_new = θ_old - α × ∇L(θ)
        - With momentum: v_new = β × v_old + (1-β) × ∇L(θ)
                        θ_new = θ_old - α × v_new
        """
        super().__init__(learning_rate)
        self.momentum = momentum
        self.velocity = None  # Will be initialized on first update
    
    def update(self, params: List[anp.ndarray], gradients: List[anp.ndarray]) -> List[anp.ndarray]:
        """
        Update parameters using SGD with optional momentum.
        
        Args:
            params: Current parameter values
            gradients: Gradients of loss with respect to parameters
        
        Returns:
            Updated parameters
        """
        # Initialize velocity on first update
        if self.velocity is None:
            self.velocity = [anp.zeros_like(p) for p in params]
        
        updated_params = []
        
        for i, (param, grad) in enumerate(zip(params, gradients)):
            if self.momentum > 0:
                # Update velocity with momentum
                self.velocity[i] = self.momentum * self.velocity[i] + (1 - self.momentum) * grad
                # Update parameters using velocity
                updated_param = param - self.learning_rate * self.velocity[i]
            else:
                # Pure SGD without momentum
                updated_param = param - self.learning_rate * grad
            
            updated_params.append(updated_param)
        
        self.step_count += 1
        return updated_params


class Adam(Optimizer):
    """
    Adam optimizer - Adaptive learning rates with momentum.
    
    Combines the benefits of:
    - Momentum (like SGD with momentum)
    - Adaptive learning rates (like RMSprop)
    
    Mathematical Formula:
    m_t = β₁ × m_{t-1} + (1-β₁) × ∇L(θ)     [momentum]
    v_t = β₂ × v_{t-1} + (1-β₂) × (∇L(θ))²  [adaptive learning rate]
    m̂_t = m_t / (1 - β₁^t)                   [bias correction]
    v̂_t = v_t / (1 - β₂^t)                   [bias correction]
    θ_new = θ_old - α × m̂_t / (√v̂_t + ε)
    """
    
    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9, 
                 beta2: float = 0.999, epsilon: float = 1e-8):
        """
        Initialize Adam optimizer.
        
        Args:
            learning_rate: Step size for updates
            beta1: Momentum decay rate (usually 0.9)
            beta2: Adaptive learning rate decay (usually 0.999)
            epsilon: Small value to prevent division by zero
        """
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None  # First moment estimates
        self.v = None  # Second moment estimates
    
    def update(self, params: List[anp.ndarray], gradients: List[anp.ndarray]) -> List[anp.ndarray]:
        """
        Update parameters using Adam optimization.
        
        Args:
            params: Current parameter values
            gradients: Gradients of loss with respect to parameters
        
        Returns:
            Updated parameters
        """
        # Initialize moments on first update
        if self.m is None:
            self.m = [anp.zeros_like(p) for p in params]
            self.v = [anp.zeros_like(p) for p in params]
        
        self.step_count += 1
        updated_params = []
        
        for i, (param, grad) in enumerate(zip(params, gradients)):
            # Update first moment (momentum)
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            
            # Update second moment (adaptive learning rate)
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)
            
            # Bias correction
            m_hat = self.m[i] / (1 - self.beta1 ** self.step_count)
            v_hat = self.v[i] / (1 - self.beta2 ** self.step_count)
            
            # Update parameters
            updated_param = param - self.learning_rate * m_hat / (anp.sqrt(v_hat) + self.epsilon)
            updated_params.append(updated_param)
        
        return updated_params


class TrainingEngine:
    """
    Complete training engine that orchestrates the learning process.
    
    This class handles:
    - Gradient computation using autograd
    - Parameter updates using optimizers
    - Training progress monitoring
    - Loss tracking and visualization
    """
    
    def __init__(self, network, optimizer: Optimizer, loss_function: Callable):
        """
        Initialize training engine.
        
        Args:
            network: Neural network to train
            optimizer: Optimization algorithm to use
            loss_function: Loss function to minimize
        """
        self.network = network
        self.optimizer = optimizer
        self.loss_function = loss_function
        
        # Training history
        self.history = defaultdict(list)
        
        # Create gradient function using autograd
        self._create_gradient_function()
    
    def _create_gradient_function(self):
        """
        Create automatic gradient computation function.
        
        This is the magic of autograd: we define a function that computes
        the loss, and autograd automatically creates a function that
        computes the gradients!
        """
        def loss_with_params(params_flat, X, y_true):
            """
            Compute loss given flattened parameters.
            
            Args:
                params_flat: All network parameters as a flat array
                X: Input data
                y_true: Target values
            
            Returns:
                Scalar loss value
            """
            # Reshape flat parameters back to network structure
            params_structured = self._unflatten_params(params_flat)
            
            # Set network parameters
            self.network.set_all_parameters(params_structured)
            
            # Forward pass
            y_pred = self.network.forward(X)
            
            # Compute loss
            loss = self.loss_function(y_true, y_pred)
            
            return loss
        
        # Create gradient function automatically!
        self.gradient_function = grad(loss_with_params, argnum=0)
    
    def _flatten_params(self, params: List[anp.ndarray]) -> anp.ndarray:
        """
        Flatten list of parameter arrays into single array.
        
        Args:
            params: List of parameter arrays
        
        Returns:
            Flattened parameter array
        """
        return anp.concatenate([p.flatten() for p in params])
    
    def _unflatten_params(self, params_flat: anp.ndarray) -> List[anp.ndarray]:
        """
        Reshape flattened parameters back to original structure.
        
        Args:
            params_flat: Flattened parameter array
        
        Returns:
            List of parameter arrays with original shapes
        """
        params = []
        start_idx = 0
        
        # Get original parameter shapes
        original_params = self.network.get_all_parameters()
        
        for original_param in original_params:
            param_size = original_param.size
            param_data = params_flat[start_idx:start_idx + param_size]
            param_reshaped = param_data.reshape(original_param.shape)
            params.append(param_reshaped)
            start_idx += param_size
        
        return params
    
    def train_step(self, X: anp.ndarray, y_true: anp.ndarray) -> float:
        """
        Perform single training step.
        
        Args:
            X: Input data
            y_true: Target values
        
        Returns:
            Loss value for this step
        """
        # Get current parameters
        current_params = self.network.get_all_parameters()
        params_flat = self._flatten_params(current_params)
        
        # Compute loss
        loss = self.loss_function(y_true, self.network.forward(X))
        
        # Compute gradients automatically!
        gradients_flat = self.gradient_function(params_flat, X, y_true)
        gradients_structured = self._unflatten_params(gradients_flat)
        
        # Update parameters
        updated_params = self.optimizer.update(current_params, gradients_structured)
        self.network.set_all_parameters(updated_params)
        
        return float(loss)
    
    def train(self, X: anp.ndarray, y_true: anp.ndarray, epochs: int = 1000, 
              batch_size: Optional[int] = None, validation_data: Optional[Tuple] = None,
              verbose: bool = True, plot_progress: bool = True) -> Dict:
        """
        Train the neural network.
        
        Args:
            X: Training input data
            y_true: Training target values
            epochs: Number of training iterations
            batch_size: Size of mini-batches (None = full batch)
            validation_data: Optional (X_val, y_val) for validation
            verbose: Whether to print progress
            plot_progress: Whether to plot training curves
        
        Returns:
            Training history dictionary
        """
        print(f"🚀 Starting Training...")
        print(f"   Network: {self.network}")
        print(f"   Optimizer: {self.optimizer.__class__.__name__}")
        print(f"   Training samples: {X.shape[0]}")
        print(f"   Epochs: {epochs}")
        print(f"   Batch size: {batch_size or 'Full batch'}")
        
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_losses = []
            
            # Handle batching
            if batch_size is None:
                # Full batch training
                loss = self.train_step(X, y_true)
                epoch_losses.append(loss)
            else:
                # Mini-batch training
                n_samples = X.shape[0]
                indices = anp.random.permutation(n_samples)
                
                for i in range(0, n_samples, batch_size):
                    batch_indices = indices[i:i + batch_size]
                    X_batch = X[batch_indices]
                    y_batch = y_true[batch_indices]
                    
                    loss = self.train_step(X_batch, y_batch)
                    epoch_losses.append(loss)
            
            # Record training loss
            avg_loss = anp.mean(epoch_losses)
            self.history['train_loss'].append(avg_loss)
            
            # Validation loss
            if validation_data is not None:
                X_val, y_val = validation_data
                val_pred = self.network.forward(X_val)
                val_loss = self.loss_function(y_val, val_pred)
                self.history['val_loss'].append(float(val_loss))
            
            # Progress reporting
            if verbose and (epoch % (epochs // 10) == 0 or epoch == epochs - 1):
                elapsed = time.time() - start_time
                val_text = f", Val Loss: {self.history['val_loss'][-1]:.6f}" if validation_data else ""
                print(f"   Epoch {epoch:4d}/{epochs}: Loss: {avg_loss:.6f}{val_text} ({elapsed:.1f}s)")
        
        training_time = time.time() - start_time
        print(f"✅ Training Complete! ({training_time:.1f}s)")
        
        # Plot training progress
        if plot_progress:
            self.plot_training_history()
        
        return dict(self.history)
    
    def plot_training_history(self):
        """Plot training and validation loss curves."""
        if not self.history['train_loss']:
            print("No training history to plot.")
            return
        
        plt.figure(figsize=(10, 6))
        
        # Plot training loss
        epochs = range(1, len(self.history['train_loss']) + 1)
        plt.plot(epochs, self.history['train_loss'], 'b-', label='Training Loss', linewidth=2)
        
        # Plot validation loss if available
        if 'val_loss' in self.history and self.history['val_loss']:
            plt.plot(epochs, self.history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')  # Log scale often better for loss curves
        
        # Add annotations
        final_loss = self.history['train_loss'][-1]
        plt.annotate(f'Final Loss: {final_loss:.6f}', 
                    xy=(len(epochs), final_loss), 
                    xytext=(len(epochs) * 0.7, final_loss * 2),
                    arrowprops=dict(arrowstyle='->', color='black', alpha=0.7))
        
        plt.tight_layout()
        plt.show()
    
    def evaluate(self, X: anp.ndarray, y_true: anp.ndarray) -> Dict:
        """
        Evaluate the trained network.
        
        Args:
            X: Test input data
            y_true: Test target values
        
        Returns:
            Dictionary of evaluation metrics
        """
        y_pred = self.network.forward(X)
        loss = self.loss_function(y_true, y_pred)
        
        # Calculate additional metrics
        mse = anp.mean((y_true - y_pred) ** 2)
        mae = anp.mean(anp.abs(y_true - y_pred))
        
        return {
            'loss': float(loss),
            'mse': float(mse),
            'mae': float(mae),
            'predictions': y_pred,
            'targets': y_true
        }


# Learning Rate Schedulers
class LearningRateScheduler:
    """Base class for learning rate scheduling."""
    
    def __init__(self, initial_lr: float):
        self.initial_lr = initial_lr
    
    def get_lr(self, epoch: int) -> float:
        """Get learning rate for given epoch."""
        raise NotImplementedError


class StepLR(LearningRateScheduler):
    """
    Step-wise learning rate decay.
    
    Reduces learning rate by a factor every few epochs.
    """
    
    def __init__(self, initial_lr: float, step_size: int, gamma: float = 0.1):
        super().__init__(initial_lr)
        self.step_size = step_size
        self.gamma = gamma
    
    def get_lr(self, epoch: int) -> float:
        return self.initial_lr * (self.gamma ** (epoch // self.step_size))


class ExponentialLR(LearningRateScheduler):
    """
    Exponential learning rate decay.
    
    Smoothly reduces learning rate over time.
    """
    
    def __init__(self, initial_lr: float, gamma: float = 0.95):
        super().__init__(initial_lr)
        self.gamma = gamma
    
    def get_lr(self, epoch: int) -> float:
        return self.initial_lr * (self.gamma ** epoch)


# Example usage and testing
if __name__ == "__main__":
    """
    Test the automatic differentiation engine.
    """
    print("🧪 Testing Automatic Differentiation Engine")
    print("=" * 50)
    
    # Import neural network components
    import sys
    sys.path.append('.')
    from nn_core import NeuralNetwork, mean_squared_error, create_sample_data
    
    # Create sample data
    X, y = create_sample_data(100)
    print(f"📊 Sample Data: {X.shape[0]} samples, target: y = 2x₁ + 3x₂ + 1")
    
    # Create neural network
    network = NeuralNetwork([2, 8, 4, 1], ['relu', 'relu', 'linear'])
    
    # Test different optimizers
    optimizers = {
        'SGD': SGD(learning_rate=0.01),
        'SGD+Momentum': SGD(learning_rate=0.01, momentum=0.9),
        'Adam': Adam(learning_rate=0.001)
    }
    
    # Split data into train/validation
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    results = {}
    
    for name, optimizer in optimizers.items():
        print(f"\n🔄 Testing {name} Optimizer...")
        
        # Create fresh network for each test
        test_network = NeuralNetwork([2, 8, 4, 1], ['relu', 'relu', 'linear'])
        
        # Create training engine
        trainer = TrainingEngine(test_network, optimizer, mean_squared_error)
        
        # Train the network
        history = trainer.train(
            X_train, y_train,
            epochs=200,
            validation_data=(X_val, y_val),
            verbose=False,
            plot_progress=False
        )
        
        # Evaluate
        eval_results = trainer.evaluate(X_val, y_val)
        results[name] = eval_results
        
        print(f"   Final Loss: {eval_results['loss']:.6f}")
        print(f"   MSE: {eval_results['mse']:.6f}")
        print(f"   MAE: {eval_results['mae']:.6f}")
    
    # Compare optimizers
    print(f"\n📊 Optimizer Comparison:")
    print(f"{'Optimizer':<15} {'Loss':<10} {'MSE':<10} {'MAE':<10}")
    print("-" * 45)
    for name, result in results.items():
        print(f"{name:<15} {result['loss']:<10.6f} {result['mse']:<10.6f} {result['mae']:<10.6f}")
    
    # Test gradient computation
    print(f"\n🔬 Testing Gradient Computation...")
    network = NeuralNetwork([2, 3, 1])
    optimizer = SGD(learning_rate=0.01)
    trainer = TrainingEngine(network, optimizer, mean_squared_error)
    
    # Single gradient step
    loss_before = trainer.train_step(X_train[:10], y_train[:10])
    loss_after = mean_squared_error(y_train[:10], network.forward(X_train[:10]))
    
    print(f"   Loss before step: {loss_before:.6f}")
    print(f"   Loss after step: {loss_after:.6f}")
    print(f"   Loss change: {loss_after - loss_before:.6f}")
    
    if loss_after < loss_before:
        print("   ✅ Gradient descent working correctly!")
    else:
        print("   ⚠️  Learning rate might be too high")
    
    print(f"\n✅ All tests passed! Automatic differentiation engine is working.")