import numpy as np
import time
import os
from typing import Tuple, Optional, Dict, Any
import sys
from pathlib import Path

# Add Neural Engine root to path
current_dir = Path(__file__).parent
neural_engine_root = current_dir.parent.parent
sys.path.insert(0, str(neural_engine_root))

from nn_core import NeuralNetwork, mean_squared_error, mean_absolute_error
from autodiff import TrainingEngine, Adam
from neural_backend import as_numpy
try:
    from config.scenarios import PredictionScenario
except ImportError:
    from apps.quadratic_web.config.scenarios import PredictionScenario
try:
    from core.data_processor import QuadraticDataProcessor
except ImportError:
    from apps.quadratic_web.core.data_processor import QuadraticDataProcessor

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
        # Ensemble support
        self.ensemble_networks = []  # List of trained networks for ensemble
        self.use_ensemble = False
        self.device = os.getenv('NEURAL_ENGINE_DEVICE', 'auto')

    def _as_numpy(self, value):
        """Convert backend tensors (NumPy/CuPy) to NumPy arrays for sklearn/JSON paths."""
        return np.asarray(as_numpy(value))

    def _is_coeff_to_roots_scenario(self) -> bool:
        return (
            set(self.scenario.input_features) == {'a', 'b', 'c'}
            and set(self.scenario.target_features) == {'x1', 'x2'}
        )

    def _is_two_root_target(self) -> bool:
        return set(self.scenario.target_features) == {'x1', 'x2'}

    def _solve_quadratic_roots(self, a: float, b: float, c: float) -> Optional[Tuple[float, float]]:
        """Return sorted real roots for a quadratic, or None when not applicable."""
        if abs(a) < 1e-10:
            return None

        discriminant = b**2 - 4 * a * c
        if discriminant < 0:
            return None

        sqrt_discriminant = float(np.sqrt(discriminant))
        denominator = 2 * a
        if abs(denominator) < 1e-10:
            return None

        roots = sorted((
            float((-b - sqrt_discriminant) / denominator),
            float((-b + sqrt_discriminant) / denominator),
        ))
        return roots[0], roots[1]
        
    def create_network(self, learning_rate: float = 0.001):
        """Create neural network for this scenario"""
        self.network = NeuralNetwork(
            self.scenario.network_architecture,
            self.scenario.activations,
            device=self.device,
        )
        
        # Create trainer with Adam optimizer and MSE loss function
        optimizer = Adam(learning_rate=learning_rate, beta1=0.9, beta2=0.999)
        self.trainer = TrainingEngine(self.network, optimizer, mean_squared_error, device=self.device)
        
    def train(self, epochs: int = 1000, learning_rate: float = 0.001, verbose: bool = True, 
              use_multi_phase: bool = True, early_stopping_patience: int = 50,
              clip_gradients: bool = True, max_grad_norm: float = 5.0,
              lr_scheduler: str = 'onecycle', use_augmentation: bool = False,
              ensemble_size: int = 1) -> Dict[str, Any]:
        """
        Train the neural network with multi-phase training, learning rate scheduling, and early stopping
        
        Args:
            epochs: Total number of epochs
            learning_rate: Initial learning rate
            verbose: Print training progress
            use_multi_phase: Use multi-phase training (fast learning -> fine-tuning -> optimization)
            early_stopping_patience: Number of epochs to wait before early stopping
            clip_gradients: Whether to clip gradients (default: True for better stability)
            max_grad_norm: Maximum gradient norm for clipping (default: 5.0)
            lr_scheduler: Learning rate scheduler ('cosine', 'warm_restarts', 'plateau', 'onecycle')
            use_augmentation: Whether to use data augmentation (default: True)
            ensemble_size: Number of models to train for ensemble (default: 1, no ensemble)
        """
        # Prepare leakage-safe split data (scalers are fit on train only).
        X_train, X_val, X_test, y_train, y_val, y_test = self.data_processor.prepare_scenario_splits(
            self.scenario, normalize=True
        )

        # Apply data augmentation only on training split.
        if use_augmentation:
            X_train, y_train = self._augment_data(X_train, y_train)
        
        start_time = time.time()
        
        if verbose:
            print(f"🚀 Training {self.scenario.name}...")
            print(f"   Input shape: {X_train.shape}")
            print(f"   Target shape: {y_train.shape}")
            print(f"   Network: {self.scenario.network_architecture}")
            print(f"   Learning rate: {learning_rate}")
            print(f"   Multi-phase: {use_multi_phase}")
        
        try:
            # Ensemble training: train multiple models
            if ensemble_size > 1:
                self.use_ensemble = True
                self.ensemble_networks = []
                ensemble_histories = []
                
                if verbose:
                    print(f"🎯 Training ensemble of {ensemble_size} models...")
                
                for model_idx in range(ensemble_size):
                    if verbose:
                        print(f"\n📦 Training model {model_idx + 1}/{ensemble_size}")
                    
                    # Create new network with different initialization
                    self.create_network(learning_rate)
                    
                    # Train this model
                    if use_multi_phase and epochs >= 300:
                        model_history = self._train_multi_phase(
                            X_train, y_train, X_val, y_val,
                            epochs, learning_rate, early_stopping_patience, False,  # verbose=False for ensemble
                            clip_gradients, max_grad_norm
                        )
                    else:
                        model_history = self._train_with_early_stopping(
                            X_train, y_train, X_val, y_val,
                            epochs, learning_rate, early_stopping_patience, False,  # verbose=False for ensemble
                            clip_gradients, max_grad_norm, lr_scheduler
                        )
                    
                    # Store trained network
                    self.ensemble_networks.append(self.network.get_all_parameters())
                    ensemble_histories.append(model_history)
                
                # Use average history
                self.training_history = self._average_histories(ensemble_histories)
                # Keep the last trained network as primary
                self.is_trained = True
                
                if verbose:
                    print(f"\n✅ Ensemble training complete!")
            else:
                # Single model training
                if use_multi_phase and epochs >= 300:
                    # Multi-phase training
                    self.training_history = self._train_multi_phase(
                        X_train, y_train, X_val, y_val,
                        epochs, learning_rate, early_stopping_patience, verbose,
                        clip_gradients, max_grad_norm
                    )
                else:
                    # Single-phase training with early stopping
                    self.create_network(learning_rate)
                    self.training_history = self._train_with_early_stopping(
                        X_train, y_train, X_val, y_val,
                        epochs, learning_rate, early_stopping_patience, verbose,
                        clip_gradients, max_grad_norm, lr_scheduler
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
                          total_epochs, initial_lr, patience, verbose,
                          clip_gradients=True, max_grad_norm=5.0):
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
            plot_progress=False,
            clip_gradients=clip_gradients,
            max_grad_norm=max_grad_norm
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
            plot_progress=False,
            clip_gradients=clip_gradients,
            max_grad_norm=max_grad_norm
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
                    plot_progress=False,
                    clip_gradients=clip_gradients,
                    max_grad_norm=max_grad_norm
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
    
    def _augment_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Augment training data with:
        - Coefficient scaling (multiply all coefficients by constant)
        - Root swapping (swap x1 and x2 labels)
        - Sign flipping (negate all coefficients)
        
        Args:
            X: Input features (coefficients)
            y: Target values (roots)
        
        Returns:
            Augmented X and y arrays
        """
        augmented_X = [X]
        augmented_y = [y]

        # Keep augmentation conservative and scenario-specific.
        # Generic 3->2 augmentation is harmful for non-root tasks.
        if self._is_coeff_to_roots_scenario() and X.shape[1] == 3 and y.shape[1] == 2:
            # Small feature-space jitter improves robustness without changing label semantics.
            noise = np.random.normal(0.0, 0.01, X.shape).astype(np.float32)
            augmented_X.append((X + noise).astype(np.float32))
            augmented_y.append(y.copy())
        
        # Concatenate all augmented data
        X_augmented = np.vstack(augmented_X)
        y_augmented = np.vstack(augmented_y)
        
        # Shuffle augmented data
        indices = np.random.permutation(len(X_augmented))
        X_augmented = X_augmented[indices]
        y_augmented = y_augmented[indices]
        
        return X_augmented, y_augmented
    
    def _train_with_early_stopping(self, X_train, y_train, X_val, y_val,
                                   epochs, learning_rate, patience, verbose,
                                   clip_gradients=True, max_grad_norm=5.0,
                                   lr_scheduler='cosine'):
        """Single-phase training with early stopping and learning rate decay"""
        all_history = {'train_loss': [], 'val_loss': []}
        best_val_loss = float('inf')
        best_network_state = None
        patience_counter = 0
        current_lr = learning_rate
        
        # Initialize learning rate scheduler if needed
        scheduler = None
        if lr_scheduler == 'warm_restarts':
            from autodiff import CosineAnnealingWarmRestarts
            scheduler = CosineAnnealingWarmRestarts(learning_rate, T_0=max(10, epochs//10), T_mult=2)
        elif lr_scheduler == 'plateau':
            from autodiff import ReduceLROnPlateau
            scheduler = ReduceLROnPlateau(learning_rate, factor=0.5, patience=patience//2)
        elif lr_scheduler == 'onecycle':
            from autodiff import OneCycleLR
            scheduler = OneCycleLR(learning_rate, max_lr=learning_rate*10, total_steps=epochs)
        
        # Train with learning rate decay
        for epoch in range(epochs):
            # Get learning rate from scheduler
            if scheduler:
                if lr_scheduler == 'plateau':
                    # For plateau, we'll update after validation
                    lr = scheduler.get_lr(epoch)
                else:
                    lr = scheduler.get_lr(epoch)
            else:
                # Default: Cosine annealing learning rate schedule
                lr = learning_rate * (1 + np.cos(np.pi * epoch / epochs)) / 2
                lr = max(lr, learning_rate * 0.01)  # Minimum learning rate
            
            # Update learning rate if it changed significantly
            if abs(current_lr - lr) / current_lr > 0.1:
                current_lr = lr
                self.trainer.optimizer.learning_rate = float(current_lr)
            
            # Train for one epoch
            epoch_history = self.trainer.train(
                X_train, y_train,
                epochs=1,
                validation_data=(X_val, y_val),
                verbose=False,
                plot_progress=False,
                clip_gradients=clip_gradients,
                max_grad_norm=max_grad_norm
            )
            
            val_loss = epoch_history.get('val_loss', [float('inf')])[-1]
            train_loss = epoch_history.get('train_loss', [float('inf')])[-1]
            
            all_history['train_loss'].append(train_loss)
            all_history['val_loss'].append(val_loss)
            
            # Update learning rate scheduler if using plateau
            if lr_scheduler == 'plateau' and scheduler:
                scheduler.step(val_loss)
                new_lr = scheduler.get_lr(epoch)
                if abs(current_lr - new_lr) / current_lr > 0.1:
                    current_lr = new_lr
                    self.trainer.optimizer.learning_rate = float(current_lr)
            
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
        
        # Make predictions (ensemble or single model)
        if self.use_ensemble and len(self.ensemble_networks) > 0:
            # Ensemble prediction: average predictions from all models
            predictions = []
            for network_params in self.ensemble_networks:
                self.network.set_all_parameters(network_params)
                pred = self.network.forward(X_transformed)
                predictions.append(pred)
            
            # Average predictions
            xp = self.network.xp
            y_pred_transformed = xp.mean(xp.stack(predictions, axis=0), axis=0)
            
            # Restore primary network
            if self.network is not None:
                self.network.set_all_parameters(self.ensemble_networks[-1])
        else:
            # Single model prediction
            y_pred_transformed = self.network.forward(X_transformed)
        
        # Inverse transform predictions
        y_pred_np = self._as_numpy(y_pred_transformed)
        y_pred = self.data_processor.inverse_transform_output(self.scenario, y_pred_np)
        
        # Refine predictions if requested
        if refine_predictions:
            # Calculate confidence if needed
            confidence_vals = None
            if return_confidence:
                # Estimate confidence based on prediction variance or error
                # For now, use a simple heuristic: confidence inversely related to prediction magnitude
                confidence_vals = 1.0 / (1.0 + np.abs(y_pred).mean(axis=1))
            y_pred = self._refine_predictions(input_data, y_pred, confidence_vals)
        
        if return_confidence:
            confidences = self._estimate_confidence(X_transformed)
            return y_pred, confidences
        else:
            return y_pred, None
    
    def _refine_predictions(self, input_data: np.ndarray, predictions: np.ndarray,
                           confidence: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Refine predictions using advanced post-processing:
        - Multi-step Newton's method refinement
        - Confidence-based refinement (aggressive for low confidence)
        - Root verification and correction
        - Handle special cases (repeated roots, near-zero coefficients)
        
        Args:
            input_data: Input features (coefficients)
            predictions: Initial predictions (roots)
            confidence: Optional confidence estimates for each prediction
        """
        refined = predictions.copy()
        
        # If predicting roots from coefficients
        if self._is_coeff_to_roots_scenario():
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
                    
                    # If we have coefficients, verify and refine the roots
                    if len(coeffs) >= 3:
                        a, b, c = coeffs[0], coeffs[1], coeffs[2]
                        
                        # Skip refinement if a is zero or very small
                        if abs(a) < 1e-10:
                            continue
                        
                        # Calculate confidence for this prediction
                        pred_confidence = 0.5
                        if confidence is not None and i < len(confidence):
                            pred_conf_raw = confidence[i]
                            if np.isscalar(pred_conf_raw):
                                pred_confidence = float(pred_conf_raw)
                            else:
                                pred_confidence = float(np.mean(pred_conf_raw))
                        
                        # Determine refinement strategy based on confidence
                        # Low confidence (< 0.7): aggressive refinement (3 iterations)
                        # Medium confidence (0.7-0.9): moderate refinement (2 iterations)
                        # High confidence (>= 0.9): minimal refinement (1 iteration)
                        if pred_confidence < 0.7:
                            max_iterations = 3
                            error_threshold = 0.05
                        elif pred_confidence < 0.9:
                            max_iterations = 2
                            error_threshold = 0.1
                        else:
                            max_iterations = 1
                            error_threshold = 0.2
                        
                        # Multi-step Newton's method refinement for x1
                        x1_refined = x1_pred
                        for iteration in range(max_iterations):
                            error1 = abs(a * x1_refined**2 + b * x1_refined + c)
                            if error1 < error_threshold:
                                break
                            
                            derivative = 2 * a * x1_refined + b
                            if abs(derivative) > 1e-6:
                                x1_new = x1_refined - (a * x1_refined**2 + b * x1_refined + c) / derivative
                                # Check for convergence
                                if abs(x1_new - x1_refined) < 1e-10:
                                    break
                                x1_refined = x1_new
                            else:
                                break
                        
                        # Multi-step Newton's method refinement for x2
                        x2_refined = x2_pred
                        for iteration in range(max_iterations):
                            error2 = abs(a * x2_refined**2 + b * x2_refined + c)
                            if error2 < error_threshold:
                                break
                            
                            derivative = 2 * a * x2_refined + b
                            if abs(derivative) > 1e-6:
                                x2_new = x2_refined - (a * x2_refined**2 + b * x2_refined + c) / derivative
                                # Check for convergence
                                if abs(x2_new - x2_refined) < 1e-10:
                                    break
                                x2_refined = x2_new
                            else:
                                break
                        
                        # Verify refined roots still satisfy equation
                        final_error1 = abs(a * x1_refined**2 + b * x1_refined + c)
                        final_error2 = abs(a * x2_refined**2 + b * x2_refined + c)
                        
                        # Use refined roots if they're better
                        if final_error1 < abs(a * x1_pred**2 + b * x1_pred + c):
                            refined[i, 0] = x1_refined
                        if final_error2 < abs(a * x2_pred**2 + b * x2_pred + c):
                            refined[i, 1] = x2_refined
                        
                        # Special case: repeated roots (discriminant = 0)
                        discriminant = b**2 - 4*a*c
                        if abs(discriminant) < 1e-6:
                            # Both roots should be the same
                            repeated_root = -b / (2 * a)
                            refined[i, 0] = repeated_root
                            refined[i, 1] = repeated_root
                        
                        # Final ordering
                        if refined[i, 0] > refined[i, 1]:
                            refined[i, 0], refined[i, 1] = refined[i, 1], refined[i, 0]

                        # Guard against both outputs collapsing onto the same basin.
                        # When the equation has two distinct real roots, use Vieta's
                        # relation to recover the companion root from the better one.
                        if discriminant > 1e-6 and abs(refined[i, 1] - refined[i, 0]) < 1e-3:
                            current_errors = np.array([
                                abs(a * refined[i, 0]**2 + b * refined[i, 0] + c),
                                abs(a * refined[i, 1]**2 + b * refined[i, 1] + c),
                            ], dtype=np.float64)
                            anchor_index = int(np.argmin(current_errors))
                            anchor_root = float(refined[i, anchor_index])
                            companion_root = float((-b / a) - anchor_root)

                            candidate_pair = np.array(
                                sorted([anchor_root, companion_root]),
                                dtype=refined.dtype,
                            )

                            exact_pair = self._solve_quadratic_roots(a, b, c)
                            if exact_pair is not None:
                                exact_pair_arr = np.array(exact_pair, dtype=np.float64)
                                candidate_distance = float(
                                    np.linalg.norm(candidate_pair.astype(np.float64) - exact_pair_arr)
                                )
                                current_distance = float(
                                    np.linalg.norm(refined[i].astype(np.float64) - exact_pair_arr)
                                )
                                if candidate_distance <= current_distance:
                                    refined[i, :] = candidate_pair
                            else:
                                refined[i, :] = candidate_pair
        
        return refined
    
    def _average_histories(self, histories: list) -> Dict[str, list]:
        """Average training histories from multiple models"""
        if not histories:
            return {}
        
        # Find maximum length
        max_len = max(len(h.get('train_loss', [])) for h in histories)
        
        averaged = {
            'train_loss': [],
            'val_loss': []
        }
        
        for i in range(max_len):
            train_losses = [h.get('train_loss', [0])[i] if i < len(h.get('train_loss', [])) else h.get('train_loss', [0])[-1] for h in histories]
            val_losses = [h.get('val_loss', [0])[i] if i < len(h.get('val_loss', [])) else h.get('val_loss', [0])[-1] for h in histories]
            
            averaged['train_loss'].append(np.mean(train_losses))
            averaged['val_loss'].append(np.mean(val_losses))
        
        return averaged
    
    def _estimate_confidence(self, X_transformed: np.ndarray, n_samples: int = 24) -> np.ndarray:
        """Estimate prediction confidence using Monte Carlo dropout simulation"""
        predictions = []
        
        # Get current parameters
        original_params = self.network.get_all_parameters()
        xp = self.network.xp
        
        # Generate multiple predictions with small parameter perturbations
        for _ in range(n_samples):
            # Add small noise to parameters
            perturbed_params = []
            for param in original_params:
                noise = xp.random.normal(0, 0.01, param.shape)
                perturbed_params.append(param + noise)
            
            # Set perturbed parameters
            self.network.set_all_parameters(perturbed_params)
            
            # Make prediction
            pred = self.network.forward(X_transformed)
            predictions.append(pred)
        
        # Restore original parameters
        self.network.set_all_parameters(original_params)
        
        # Calculate confidence metrics
        predictions = xp.stack(predictions, axis=0)
        std_pred = xp.std(predictions, axis=0)
        
        # Confidence as inverse of normalized standard deviation
        confidence = 1.0 / (1.0 + std_pred)
        
        return self._as_numpy(confidence)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance"""
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        # Make predictions
        y_pred = self._as_numpy(self.network.forward(X_test))
        y_test_np = np.asarray(y_test, dtype=np.float32)

        # Permutation-invariant evaluation for root-pair outputs.
        if self._is_two_root_target() and y_pred.ndim == 2 and y_pred.shape[1] == 2:
            y_pred = np.sort(y_pred, axis=1)
            y_test_np = np.sort(y_test_np, axis=1)
        
        # Calculate metrics
        mse = np.mean((y_test_np - y_pred) ** 2)
        mae = np.mean(np.abs(y_test_np - y_pred))
        rmse = np.sqrt(mse)
        
        # R² score
        ss_res = np.sum((y_test_np - y_pred) ** 2)
        ss_tot = np.sum((y_test_np - np.mean(y_test_np)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Accuracy within tolerance
        tolerance = 0.1
        relative_error = np.abs((y_test_np - y_pred) / (y_test_np + 1e-8))
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
