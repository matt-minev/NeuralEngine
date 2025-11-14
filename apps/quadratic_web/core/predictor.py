import numpy as np
import time
from typing import Tuple, Optional, Dict, Any
import sys
from pathlib import Path

# Add Neural Engine root to path
current_dir = Path(__file__).parent
neural_engine_root = current_dir.parent.parent
sys.path.insert(0, str(neural_engine_root))

from nn_core import NeuralNetwork, mean_squared_error, mean_absolute_error
from autodiff import TrainingEngine, Adam
from config.scenarios import PredictionScenario
from core.data_processor import QuadraticDataProcessor

class QuadraticPredictor:
    """Neural network predictor for quadratic equations"""
    
    def __init__(self, scenario: PredictionScenario, data_processor: QuadraticDataProcessor):
        self.scenario = scenario
        self.data_processor = data_processor
        self.network = None
        self.trainer = None
        self.is_trained = False
        self.training_history = {}
        self.performance_stats = {}
        
    def create_network(self, learning_rate: float = 0.001):
        """Create neural network for this scenario"""
        self.network = NeuralNetwork(
            self.scenario.network_architecture,
            self.scenario.activations
        )
        
        # Create trainer with Adam optimizer and MSE loss function
        optimizer = Adam(learning_rate=learning_rate, beta1=0.9, beta2=0.999)
        self.trainer = TrainingEngine(self.network, optimizer, mean_squared_error)
        
    def train(self, epochs: int = 1000, learning_rate: float = 0.001, verbose: bool = True, 
              use_multi_phase: bool = True, early_stopping_patience: int = 50) -> Dict[str, Any]:
        """
        Train the neural network with multi-phase training, learning rate scheduling, and early stopping
        
        Args:
            epochs: Total number of epochs
            learning_rate: Initial learning rate
            verbose: Print training progress
            use_multi_phase: Use multi-phase training (fast learning -> fine-tuning -> optimization)
            early_stopping_patience: Number of epochs to wait before early stopping
        """
        # Prepare data
        X, y = self.data_processor.prepare_scenario_data(self.scenario, normalize=True)
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.data_processor.split_data(X, y)
        
        start_time = time.time()
        
        if verbose:
            print(f"🚀 Training {self.scenario.name}...")
            print(f"   Input shape: {X_train.shape}")
            print(f"   Target shape: {y_train.shape}")
            print(f"   Network: {self.scenario.network_architecture}")
            print(f"   Learning rate: {learning_rate}")
            print(f"   Multi-phase: {use_multi_phase}")
        
        try:
            if use_multi_phase and epochs >= 300:
                # Multi-phase training
                self.training_history = self._train_multi_phase(
                    X_train, y_train, X_val, y_val,
                    epochs, learning_rate, early_stopping_patience, verbose
                )
            else:
                # Single-phase training with early stopping
                self.create_network(learning_rate)
                self.training_history = self._train_with_early_stopping(
                    X_train, y_train, X_val, y_val,
                    epochs, learning_rate, early_stopping_patience, verbose
                )
            
            training_time = time.time() - start_time
            self.performance_stats['training_time'] = training_time
            self.performance_stats['learning_rate'] = learning_rate
            self.is_trained = True
            
            # Evaluate on test set
            test_results = self.evaluate(X_test, y_test)
            
            if verbose:
                print(f"✅ Training completed in {training_time:.2f}s")
                print(f"   Test R²: {test_results['r2']:.4f}")
                print(f"   Test MSE: {test_results['mse']:.6f}")
                print(f"   Test Accuracy (10%): {test_results['accuracy_10pct']:.2f}%")
                
            return {
                'training_time': training_time,
                'test_results': test_results,
                'training_history': self.training_history
            }
            
        except Exception as e:
            if verbose:
                print(f"❌ Training failed: {str(e)}")
            raise e
    
    def _train_multi_phase(self, X_train, y_train, X_val, y_val, 
                          total_epochs, initial_lr, patience, verbose):
        """Multi-phase training: fast learning -> fine-tuning -> optimization"""
        all_history = {'train_loss': [], 'val_loss': [], 'val_r2': []}
        best_val_loss = float('inf')
        best_network_state = None
        patience_counter = 0
        
        # Phase 1: Fast Learning (40% of epochs, high learning rate)
        phase1_epochs = int(total_epochs * 0.4)
        phase1_lr = initial_lr
        
        if verbose:
            print(f"\n📈 Phase 1: Fast Learning ({phase1_epochs} epochs, LR={phase1_lr:.6f})")
        
        self.create_network(phase1_lr)
        phase1_history = self.trainer.train(
            X_train, y_train,
            epochs=phase1_epochs,
            validation_data=(X_val, y_val),
            verbose=False,
            plot_progress=False
        )
        
        # Track best model
        for i, val_loss in enumerate(phase1_history.get('val_loss', [])):
            all_history['train_loss'].extend(phase1_history.get('train_loss', [])[i:i+1])
            all_history['val_loss'].append(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_network_state = self.network.get_all_parameters()
                patience_counter = 0
            else:
                patience_counter += 1
        
        # Phase 2: Fine-tuning (35% of epochs, reduced learning rate)
        phase2_epochs = int(total_epochs * 0.35)
        phase2_lr = initial_lr * 0.5
        
        if verbose:
            print(f"🔧 Phase 2: Fine-tuning ({phase2_epochs} epochs, LR={phase2_lr:.6f})")
        
        # Create new network with reduced learning rate
        self.create_network(phase2_lr)
        # Copy weights from phase 1
        if best_network_state:
            self.network.set_all_parameters(best_network_state)
        
        phase2_history = self.trainer.train(
            X_train, y_train,
            epochs=phase2_epochs,
            validation_data=(X_val, y_val),
            verbose=False,
            plot_progress=False
        )
        
        # Track best model
        for i, val_loss in enumerate(phase2_history.get('val_loss', [])):
            all_history['train_loss'].extend(phase2_history.get('train_loss', [])[i:i+1])
            all_history['val_loss'].append(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_network_state = self.network.get_all_parameters()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        print(f"⏹️  Early stopping triggered at epoch {len(all_history['val_loss'])}")
                    break
        
        # Phase 3: Optimization (remaining epochs, very low learning rate)
        if patience_counter < patience:
            phase3_epochs = total_epochs - len(all_history['val_loss'])
            phase3_lr = initial_lr * 0.1
            
            if verbose and phase3_epochs > 0:
                print(f"✨ Phase 3: Optimization ({phase3_epochs} epochs, LR={phase3_lr:.6f})")
            
            if phase3_epochs > 0:
                self.create_network(phase3_lr)
                if best_network_state:
                    self.network.set_all_parameters(best_network_state)
                
                phase3_history = self.trainer.train(
                    X_train, y_train,
                    epochs=phase3_epochs,
                    validation_data=(X_val, y_val),
                    verbose=False,
                    plot_progress=False
                )
                
                # Track best model
                for i, val_loss in enumerate(phase3_history.get('val_loss', [])):
                    all_history['train_loss'].extend(phase3_history.get('train_loss', [])[i:i+1])
                    all_history['val_loss'].append(val_loss)
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_network_state = self.network.get_all_parameters()
        
        # Restore best model
        if best_network_state:
            self.network.set_all_parameters(best_network_state)
        
        return all_history
    
    def _train_with_early_stopping(self, X_train, y_train, X_val, y_val,
                                   epochs, learning_rate, patience, verbose):
        """Single-phase training with early stopping and learning rate decay"""
        all_history = {'train_loss': [], 'val_loss': []}
        best_val_loss = float('inf')
        best_network_state = None
        patience_counter = 0
        current_lr = learning_rate
        
        # Train with learning rate decay
        for epoch in range(epochs):
            # Cosine annealing learning rate schedule
            lr = learning_rate * (1 + np.cos(np.pi * epoch / epochs)) / 2
            lr = max(lr, learning_rate * 0.01)  # Minimum learning rate
            
            # Update learning rate if it changed significantly
            if abs(current_lr - lr) / current_lr > 0.1:
                current_lr = lr
                self.create_network(current_lr)
                if best_network_state:
                    self.network.set_all_parameters(best_network_state)
            
            # Train for one epoch
            epoch_history = self.trainer.train(
                X_train, y_train,
                epochs=1,
                validation_data=(X_val, y_val),
                verbose=False,
                plot_progress=False
            )
            
            val_loss = epoch_history.get('val_loss', [float('inf')])[-1]
            train_loss = epoch_history.get('train_loss', [float('inf')])[-1]
            
            all_history['train_loss'].append(train_loss)
            all_history['val_loss'].append(val_loss)
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_network_state = self.network.get_all_parameters()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        print(f"⏹️  Early stopping triggered at epoch {epoch + 1}")
                    break
        
        # Restore best model
        if best_network_state:
            self.network.set_all_parameters(best_network_state)
        
        return all_history

    
    def predict(self, input_data: np.ndarray, return_confidence: bool = True, 
                refine_predictions: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Make predictions with optional confidence estimation and refinement
        
        Args:
            input_data: Input features for prediction
            return_confidence: Whether to return confidence estimates
            refine_predictions: Whether to refine predictions using post-processing
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Transform input data
        X_transformed = self.data_processor.transform_input(self.scenario, input_data)
        
        # Make predictions
        y_pred_transformed = self.network.forward(X_transformed)
        
        # Inverse transform predictions
        y_pred = self.data_processor.inverse_transform_output(self.scenario, y_pred_transformed)
        
        # Refine predictions if requested
        if refine_predictions:
            y_pred = self._refine_predictions(input_data, y_pred)
        
        if return_confidence:
            confidences = self._estimate_confidence(X_transformed)
            return y_pred, confidences
        else:
            return y_pred, None
    
    def _refine_predictions(self, input_data: np.ndarray, predictions: np.ndarray) -> np.ndarray:
        """
        Refine predictions using post-processing:
        - Verify roots satisfy quadratic formula
        - Order roots consistently (x1 <= x2)
        - Handle special cases (repeated roots, near-zero coefficients)
        """
        refined = predictions.copy()
        
        # If predicting roots from coefficients
        if self.scenario.name in ['coeff_to_roots', 'coeff_to_x1', 'coeff_to_x2', 'coeff_to_both_roots']:
            # Ensure predictions are 2D
            if refined.ndim == 1:
                refined = refined.reshape(1, -1)
            
            # For each prediction
            for i in range(refined.shape[0]):
                # Get coefficients if available
                if input_data.ndim == 1:
                    coeffs = input_data
                else:
                    coeffs = input_data[i]
                
                # Extract roots
                if refined.shape[1] >= 2:
                    x1_pred = refined[i, 0]
                    x2_pred = refined[i, 1]
                    
                    # Order roots: x1 <= x2
                    if x1_pred > x2_pred:
                        x1_pred, x2_pred = x2_pred, x1_pred
                        refined[i, 0] = x1_pred
                        refined[i, 1] = x2_pred
                    
                    # If we have coefficients, verify the roots
                    if len(coeffs) >= 3:
                        a, b, c = coeffs[0], coeffs[1], coeffs[2]
                        
                        # Verify roots satisfy ax² + bx + c = 0
                        error1 = abs(a * x1_pred**2 + b * x1_pred + c)
                        error2 = abs(a * x2_pred**2 + b * x2_pred + c)
                        
                        # If error is large, try to refine using iterative method
                        if error1 > 0.1 or error2 > 0.1:
                            # Use Newton's method for refinement (one iteration)
                            if abs(2 * a * x1_pred + b) > 1e-6:
                                x1_refined = x1_pred - (a * x1_pred**2 + b * x1_pred + c) / (2 * a * x1_pred + b)
                                refined[i, 0] = x1_refined
                            
                            if abs(2 * a * x2_pred + b) > 1e-6:
                                x2_refined = x2_pred - (a * x2_pred**2 + b * x2_pred + c) / (2 * a * x2_pred + b)
                                refined[i, 1] = x2_refined
                            
                            # Re-order after refinement
                            if refined[i, 0] > refined[i, 1]:
                                refined[i, 0], refined[i, 1] = refined[i, 1], refined[i, 0]
        
        return refined
    
    def _estimate_confidence(self, X_transformed: np.ndarray, n_samples: int = 50) -> np.ndarray:
        """Estimate prediction confidence using Monte Carlo dropout simulation"""
        predictions = []
        
        # Get current parameters
        original_params = self.network.get_all_parameters()
        
        # Generate multiple predictions with small parameter perturbations
        for _ in range(n_samples):
            # Add small noise to parameters
            perturbed_params = []
            for param in original_params:
                noise = np.random.normal(0, 0.01, param.shape)
                perturbed_params.append(param + noise)
            
            # Set perturbed parameters
            self.network.set_all_parameters(perturbed_params)
            
            # Make prediction
            pred = self.network.forward(X_transformed)
            predictions.append(pred)
        
        # Restore original parameters
        self.network.set_all_parameters(original_params)
        
        # Calculate confidence metrics
        predictions = np.array(predictions)
        std_pred = np.std(predictions, axis=0)
        
        # Confidence as inverse of normalized standard deviation
        confidence = 1.0 / (1.0 + std_pred)
        
        return confidence
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance"""
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        # Make predictions
        y_pred = self.network.forward(X_test)
        
        # Calculate metrics
        mse = np.mean((y_test - y_pred) ** 2)
        mae = np.mean(np.abs(y_test - y_pred))
        rmse = np.sqrt(mse)
        
        # R² score
        ss_res = np.sum((y_test - y_pred) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Accuracy within tolerance
        tolerance = 0.1
        relative_error = np.abs((y_test - y_pred) / (y_test + 1e-8))
        accuracy = np.mean(relative_error < tolerance) * 100
        
        return {
            'mse': float(mse),
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2),
            'accuracy_10pct': float(accuracy)
        }
    
    def get_info(self) -> Dict[str, Any]:
        """Get predictor information"""
        info = {
            'scenario': self.scenario.name,
            'description': self.scenario.description,
            'input_features': self.scenario.input_features,
            'target_features': self.scenario.target_features,
            'network_architecture': self.scenario.network_architecture,
            'is_trained': self.is_trained
        }
        
        if self.is_trained:
            info['training_stats'] = self.performance_stats
            
        return info
