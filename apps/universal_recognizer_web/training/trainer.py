"""
Enhanced training engine with state-of-the-art features.

Includes batch processing, learning rate scheduling, early stopping,
gradient clipping, checkpointing, and proper validation monitoring.
"""

import numpy as np
from typing import List, Tuple, Dict, Callable, Optional, Any
import time
import os
import pickle
from collections import defaultdict

# Import Neural Engine components
import sys
# __file__ is apps/universal_recognizer_web/training/trainer.py
# Go up 3 levels to get to NeuralEngine root
_this_file = os.path.abspath(__file__)
base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_this_file))))
if base_path not in sys.path:
    sys.path.insert(0, base_path)

from nn_core import NeuralNetwork, cross_entropy_loss
from autodiff import Optimizer, Adam, TrainingEngine
from neural_backend import to_cpu

# Handle both script and module execution
try:
    from .data_loader import BatchGenerator
except ImportError:
    # Running as script
    training_dir = os.path.dirname(os.path.abspath(__file__))
    if training_dir not in sys.path:
        sys.path.insert(0, training_dir)
    from data_loader import BatchGenerator


class LearningRateScheduler:
    """Learning rate scheduling for training."""
    
    def __init__(self, initial_lr: float, schedule_type: str = 'cosine', **kwargs):
        """
        Initialize scheduler.
        
        Args:
            initial_lr: Starting learning rate
            schedule_type: 'cosine', 'step', 'exponential', or 'constant'
            **kwargs: Schedule-specific parameters
        """
        self.initial_lr = initial_lr
        self.schedule_type = schedule_type
        self.current_epoch = 0
        self.params = kwargs
    
    def get_lr(self, epoch: int) -> float:
        """Get learning rate for given epoch."""
        self.current_epoch = epoch
        
        if self.schedule_type == 'cosine':
            # Cosine annealing
            T_max = self.params.get('T_max', 100)
            eta_min = self.params.get('eta_min', 0.0)
            return eta_min + (self.initial_lr - eta_min) * (1 + np.cos(np.pi * epoch / T_max)) / 2
        
        elif self.schedule_type == 'step':
            # Step decay
            step_size = self.params.get('step_size', 30)
            gamma = self.params.get('gamma', 0.1)
            return self.initial_lr * (gamma ** (epoch // step_size))
        
        elif self.schedule_type == 'exponential':
            # Exponential decay
            gamma = self.params.get('gamma', 0.95)
            return self.initial_lr * (gamma ** epoch)
        
        else:  # constant
            return self.initial_lr
    
    def update_optimizer(self, optimizer: Optimizer, epoch: int):
        """Update optimizer learning rate."""
        new_lr = self.get_lr(epoch)
        optimizer.learning_rate = new_lr
        return new_lr


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'min'):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.counter = 0
        self.best_epoch = 0
        self.stopped = False
    
    def __call__(self, value: float, epoch: int) -> bool:
        """
        Check if training should stop.
        
        Args:
            value: Current metric value
            epoch: Current epoch
        
        Returns:
            True if training should stop
        """
        if self.mode == 'min':
            improved = value < (self.best_value - self.min_delta)
        else:
            improved = value > (self.best_value + self.min_delta)
        
        if improved:
            self.best_value = value
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            self.stopped = True
            return True
        
        return False


def clip_gradients(gradients: List[np.ndarray], max_norm: float = 5.0) -> List[np.ndarray]:
    """
    Clip gradients to prevent explosion.
    
    Args:
        gradients: List of gradient arrays
        max_norm: Maximum gradient norm
    
    Returns:
        Clipped gradients
    """
    # Calculate global norm
    total_norm = 0.0
    for grad in gradients:
        total_norm += np.sum(grad ** 2)
    total_norm = np.sqrt(total_norm)
    
    # Clip if necessary
    if total_norm > max_norm:
        clip_ratio = max_norm / (total_norm + 1e-8)
        gradients = [g * clip_ratio for g in gradients]
    
    return gradients


class EnhancedTrainingEngine:
    """
    Enhanced training engine with state-of-the-art features.
    
    Features:
    - Batch processing
    - Learning rate scheduling
    - Early stopping
    - Gradient clipping
    - Model checkpointing
    - Proper validation monitoring
    """
    
    def __init__(self, network: NeuralNetwork, optimizer: Optimizer, 
                 loss_function: Callable, max_grad_norm: float = 5.0):
        """
        Initialize enhanced training engine.
        
        Args:
            network: Neural network to train
            optimizer: Optimization algorithm
            loss_function: Loss function
            max_grad_norm: Maximum gradient norm for clipping
        """
        self.network = network
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.max_grad_norm = max_grad_norm
        
        self.history = defaultdict(list)
        self.best_val_loss = float('inf')
        self.best_val_accuracy = 0.0
        self.best_model_state = None
        self.base_engine = TrainingEngine(network, optimizer, loss_function)
    
    def train_step(self, X_batch: np.ndarray, y_batch: np.ndarray) -> float:
        """
        Perform single training step on a batch.
        
        Args:
            X_batch: Batch of inputs
            y_batch: Batch of targets
        
        Returns:
            Loss value
        """
        return self.base_engine.train_step(
            X_batch, y_batch, clip_gradients=True, max_norm=self.max_grad_norm
        )
    
    def evaluate(self, X: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model on data.
        
        Args:
            X: Input data
            y_true: True labels (one-hot)
        
        Returns:
            Dictionary with loss, accuracy, and predictions
        """
        y_pred = self.network.forward(X)
        loss = self.loss_function(y_true, y_pred)
        
        # Calculate accuracy
        predicted_classes = np.argmax(to_cpu(y_pred), axis=1)
        true_classes = np.argmax(to_cpu(y_true), axis=1)
        accuracy = np.mean(predicted_classes == true_classes) * 100
        
        return {
            'loss': float(to_cpu(loss)),
            'accuracy': float(accuracy),
            'predictions': y_pred
        }
    
    def train(self, 
              X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
              epochs: int = 100,
              batch_size: int = 64,
              lr_scheduler: Optional[LearningRateScheduler] = None,
              early_stopping: Optional[EarlyStopping] = None,
              checkpoint_dir: Optional[str] = None,
              checkpoint_freq: int = 10,
              augmentation: Optional[Any] = None,
              verbose: bool = True) -> Dict[str, List[float]]:
        """
        Train the network with enhanced features.
        
        Args:
            X_train: Training inputs
            y_train: Training targets
            X_val: Validation inputs
            y_val: Validation targets
            epochs: Number of epochs
            batch_size: Batch size
            lr_scheduler: Learning rate scheduler
            early_stopping: Early stopping callback
            checkpoint_dir: Directory for checkpoints
            checkpoint_freq: Frequency of checkpoints
            augmentation: Data augmentation pipeline
            verbose: Print progress
        
        Returns:
            Training history
        """
        print(f"Starting enhanced training...")
        print(f"  Network: {self.network}")
        print(f"  Optimizer: {self.optimizer.__class__.__name__}")
        print(f"  Training samples: {X_train.shape[0]:,}")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Validation: {'Yes' if X_val is not None else 'No'}")
        
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Update learning rate
            if lr_scheduler:
                new_lr = lr_scheduler.update_optimizer(self.optimizer, epoch)
                if verbose and epoch % 10 == 0:
                    print(f"  Learning rate: {new_lr:.6f}", flush=True)
            
            # Create batch generator
            batch_gen = BatchGenerator(
                X_train, y_train,
                batch_size=batch_size,
                shuffle=True,
                augmentation=augmentation
            )
            
            # Training loop
            epoch_losses = []
            for X_batch, y_batch in batch_gen:
                loss = self.train_step(X_batch, y_batch)
                epoch_losses.append(loss)
            
            avg_train_loss = np.mean(epoch_losses)
            self.history['train_loss'].append(avg_train_loss)
            
            # Validation
            val_loss = None
            val_accuracy = None
            if X_val is not None and y_val is not None:
                val_results = self.evaluate(X_val, y_val)
                val_loss = val_results['loss']
                val_accuracy = val_results['accuracy']
                
                self.history['val_loss'].append(val_loss)
                self.history['val_accuracy'].append(val_accuracy)
                
                # Track best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_val_accuracy = val_accuracy
                    # Save best model state
                    self.best_model_state = [p.copy() for p in self.network.get_all_parameters()]
            
            epoch_time = time.time() - epoch_start
            
            # Progress reporting
            if verbose:
                # Print every epoch for first 5 epochs, then every 10
                if epoch < 5 or epoch % 10 == 0 or epoch == epochs - 1:
                    val_text = f", Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%" if val_loss is not None else ""
                    print(f"  Epoch {epoch:4d}/{epochs}: Train Loss: {avg_train_loss:.4f}{val_text} ({epoch_time:.1f}s)", flush=True)
                elif epoch % 5 == 0:
                    # Quick progress indicator every 5 epochs
                    print(f"  Epoch {epoch:4d}/{epochs}: Training... (Loss: {avg_train_loss:.4f})", flush=True)
            
            # Checkpointing
            if checkpoint_dir and epoch % checkpoint_freq == 0:
                self.save_checkpoint(checkpoint_dir, epoch)
            
            # Early stopping
            if early_stopping and val_loss is not None:
                if early_stopping(val_loss, epoch):
                    print(f"  Early stopping triggered at epoch {epoch}")
                    print(f"  Best validation loss: {self.best_val_loss:.4f} at epoch {early_stopping.best_epoch}")
                    break
        
        training_time = time.time() - start_time
        
        # Restore best model
        if self.best_model_state is not None:
            self.network.set_all_parameters(self.best_model_state)
            print(f"  Restored best model (val_loss: {self.best_val_loss:.4f})")
        
        print(f"Training complete! ({training_time/60:.1f} minutes)")
        
        return dict(self.history)
    
    def save_checkpoint(self, checkpoint_dir: str, epoch: int):
        """Save model checkpoint with error handling."""
        try:
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pkl')
            temp_path = checkpoint_path + '.tmp'
            
            checkpoint = {
                'epoch': epoch,
                'model_state': [to_cpu(p.copy()) for p in self.network.get_all_parameters()],
                'optimizer_state': self.optimizer.__dict__.copy(),
                'history': dict(self.history),
                'best_val_loss': self.best_val_loss,
                'best_val_accuracy': self.best_val_accuracy
            }
            
            # Atomic write
            with open(temp_path, 'wb') as f:
                pickle.dump(checkpoint, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Verify and rename
            if os.path.exists(temp_path) and os.path.getsize(temp_path) > 1000:
                if os.path.exists(checkpoint_path):
                    os.remove(checkpoint_path)
                os.rename(temp_path, checkpoint_path)
            else:
                raise IOError("Checkpoint file verification failed")
        except Exception as e:
            print(f"  Warning: Checkpoint save failed at epoch {epoch}: {e}")
