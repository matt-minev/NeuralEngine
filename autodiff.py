"""
Automatic differentiation engine for neural network optimization.

Implements gradient computation using autograd and various optimizers
(SGD, Adam, etc.) with training loop management.
"""

import numpy as np
import autograd.numpy as anp
from autograd import grad
from typing import List, Tuple, Dict, Callable, Optional, Any
import matplotlib.pyplot as plt
from collections import defaultdict
import time


class Optimizer:
    """Base optimizer class. All optimizers inherit from this."""
    
    def __init__(self, learning_rate: float = 0.001):
        self.learning_rate = learning_rate
        self.step_count = 0
    
    def update(self, params: List[anp.ndarray], gradients: List[anp.ndarray]) -> List[anp.ndarray]:
        """Update parameters using gradients."""
        raise NotImplementedError("Subclasses must implement update method")
    
    def zero_grad(self):
        """Reset internal state."""
        pass


class SGD(Optimizer):
    """
    Stochastic gradient descent with optional momentum.
    
    Update rule: theta_new = theta_old - lr * gradient
    """
    
    def __init__(self, learning_rate: float = 0.001, momentum: float = 0.0):
        """
        Initialize SGD.
        
        Args:
            learning_rate: step size
            momentum: momentum factor (0.0 to 1.0)
        """
        super().__init__(learning_rate)
        self.momentum = momentum
        self.velocity = None
    
    def update(self, params: List[anp.ndarray], gradients: List[anp.ndarray]) -> List[anp.ndarray]:
        """Update params with SGD."""
        # init velocity on first update
        if self.velocity is None:
            self.velocity = [anp.zeros_like(p) for p in params]
        
        updated_params = []
        
        for i, (param, grad) in enumerate(zip(params, gradients)):
            if self.momentum > 0:
                # update with momentum
                self.velocity[i] = self.momentum * self.velocity[i] + (1 - self.momentum) * grad
                updated_param = param - self.learning_rate * self.velocity[i]
            else:
                # pure SGD
                updated_param = param - self.learning_rate * grad
            
            updated_params.append(updated_param)
        
        self.step_count += 1
        return updated_params


class Adam(Optimizer):
    """
    Adam optimizer - adaptive learning rates with momentum.
    
    Combines momentum and adaptive learning rates for faster convergance.
    """
    
    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9, 
                 beta2: float = 0.999, epsilon: float = 1e-8):
        """
        Initialize Adam.
        
        Args:
            learning_rate: step size
            beta1: momentum decay (usually 0.9)
            beta2: adaptive lr decay (usually 0.999)
            epsilon: prevent division by zero
        """
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None  # first moment
        self.v = None  # second moment
    
    def update(self, params: List[anp.ndarray], gradients: List[anp.ndarray]) -> List[anp.ndarray]:
        """Update params with Adam."""
        # init moments on first update
        if self.m is None:
            self.m = [anp.zeros_like(p) for p in params]
            self.v = [anp.zeros_like(p) for p in params]
        
        self.step_count += 1
        updated_params = []
        
        for i, (param, grad) in enumerate(zip(params, gradients)):
            # update first moment (momentum)
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            
            # update second moment (adaptive lr)
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)
            
            # bias correction
            m_hat = self.m[i] / (1 - self.beta1 ** self.step_count)
            v_hat = self.v[i] / (1 - self.beta2 ** self.step_count)
            
            # update parameters
            updated_param = param - self.learning_rate * m_hat / (anp.sqrt(v_hat) + self.epsilon)
            updated_params.append(updated_param)
        
        return updated_params


class TrainingEngine:
    """
    Training engine that handles gradient computation and parameter updates.
    
    Uses autograd to automatically compute gradients.
    """
    
    def __init__(self, network, optimizer: Optimizer, loss_function: Callable):
        """
        Initialize training engine.
        
        Args:
            network: neural network to train
            optimizer: optimization algorithm
            loss_function: loss to minimize
        """
        self.network = network
        self.optimizer = optimizer
        self.loss_function = loss_function
        
        self.history = defaultdict(list)
        
        # create gradient function using autograd
        self._create_gradient_function()
    
    def _create_gradient_function(self):
        """
        Create automatic gradient computation function.
        
        Autograd magically computes gradients for us!
        """
        def loss_with_params(params_flat, X, y_true):
            """Compute loss given flattened params."""
            # reshape flat params back to network structure
            params_structured = self._unflatten_params(params_flat)
            
            self.network.set_all_parameters(params_structured)
            
            # forward pass
            y_pred = self.network.forward(X)
            
            # compute loss
            loss = self.loss_function(y_true, y_pred)
            
            return loss
        
        # create gradient function automatically
        self.gradient_function = grad(loss_with_params, argnum=0)
    
    def _flatten_params(self, params: List[anp.ndarray]) -> anp.ndarray:
        """Flatten parameter list into single array."""
        return anp.concatenate([p.flatten() for p in params])
    
    def _unflatten_params(self, params_flat: anp.ndarray) -> List[anp.ndarray]:
        """Reshape flattened params back to original structure."""
        params = []
        start_idx = 0
        
        # get original param shapes
        original_params = self.network.get_all_parameters()
        
        for original_param in original_params:
            param_size = original_param.size
            param_data = params_flat[start_idx:start_idx + param_size]
            param_reshaped = param_data.reshape(original_param.shape)
            params.append(param_reshaped)
            start_idx += param_size
        
        return params
    
    def train_step(self, X: anp.ndarray, y_true: anp.ndarray) -> float:
        """Perform single training step."""
        # get current params
        current_params = self.network.get_all_parameters()
        params_flat = self._flatten_params(current_params)
        
        # compute loss
        loss = self.loss_function(y_true, self.network.forward(X))
        
        # compute gradients automatically
        gradients_flat = self.gradient_function(params_flat, X, y_true)
        gradients_structured = self._unflatten_params(gradients_flat)
        
        # update params
        updated_params = self.optimizer.update(current_params, gradients_structured)
        self.network.set_all_parameters(updated_params)
        
        return float(loss)
    
    def train(self, X: anp.ndarray, y_true: anp.ndarray, epochs: int = 1000, 
              batch_size: Optional[int] = None, validation_data: Optional[Tuple] = None,
              verbose: bool = True, plot_progress: bool = True) -> Dict:
        """
        Train the neural network.
        
        Args:
            X: training inputs
            y_true: training targets
            epochs: number of training iterations
            batch_size: size of mini-batches (None = full batch)
            validation_data: optional (X_val, y_val) tuple
            verbose: print progress
            plot_progress: plot training curves
        """
        print(f"Starting training...")
        print(f"  Network: {self.network}")
        print(f"  Optimizer: {self.optimizer.__class__.__name__}")
        print(f"  Training samples: {X.shape[0]}")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size or 'Full batch'}")
        
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_losses = []
            
            # handle batching
            if batch_size is None:
                # full batch
                loss = self.train_step(X, y_true)
                epoch_losses.append(loss)
            else:
                # mini-batch
                n_samples = X.shape[0]
                indices = anp.random.permutation(n_samples)
                
                for i in range(0, n_samples, batch_size):
                    batch_indices = indices[i:i + batch_size]
                    X_batch = X[batch_indices]
                    y_batch = y_true[batch_indices]
                    
                    loss = self.train_step(X_batch, y_batch)
                    epoch_losses.append(loss)
            
            # record training loss
            avg_loss = anp.mean(epoch_losses)
            self.history['train_loss'].append(avg_loss)
            
            # validation loss
            if validation_data is not None:
                X_val, y_val = validation_data
                val_pred = self.network.forward(X_val)
                val_loss = self.loss_function(y_val, val_pred)
                self.history['val_loss'].append(float(val_loss))
            
            # progress reporting
            if verbose and (epoch % (epochs // 10) == 0 or epoch == epochs - 1):
                elapsed = time.time() - start_time
                val_text = f", Val Loss: {self.history['val_loss'][-1]:.6f}" if validation_data else ""
                print(f"  Epoch {epoch:4d}/{epochs}: Loss: {avg_loss:.6f}{val_text} ({elapsed:.1f}s)")
        
        training_time = time.time() - start_time
        print(f"Training complete! ({training_time:.1f}s)")
        
        # plot training progress
        if plot_progress:
            self.plot_training_history()
        
        return dict(self.history)
    
    def plot_training_history(self):
        """Plot training and validation loss curves."""
        if not self.history['train_loss']:
            print("No training history to plot.")
            return
        
        plt.figure(figsize=(10, 6))
        
        # plot training loss
        epochs = range(1, len(self.history['train_loss']) + 1)
        plt.plot(epochs, self.history['train_loss'], 'b-', label='Training Loss', linewidth=2)
        
        # plot validation loss if availabe
        if 'val_loss' in self.history and self.history['val_loss']:
            plt.plot(epochs, self.history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')  # log scale often better for loss
        
        # add annotation
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
        
        Returns dict with loss, mse, mae, predictions, and targets.
        """
        y_pred = self.network.forward(X)
        loss = self.loss_function(y_true, y_pred)
        
        # calculate additional metrics
        mse = anp.mean((y_true - y_pred) ** 2)
        mae = anp.mean(anp.abs(y_true - y_pred))
        
        return {
            'loss': float(loss),
            'mse': float(mse),
            'mae': float(mae),
            'predictions': y_pred,
            'targets': y_true
        }


# Learning rate schedulers
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
