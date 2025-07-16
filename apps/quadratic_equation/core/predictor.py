import numpy as np
import time
from typing import Tuple, Optional, Dict, Any
import sys
sys.path.append('../..')

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
        
    def create_network(self):
        """Create neural network for this scenario"""
        self.network = NeuralNetwork(
            self.scenario.network_architecture,
            self.scenario.activations
        )
        
        # Create trainer with Adam optimizer and MSE loss function
        optimizer = Adam(learning_rate=0.001, beta1=0.9, beta2=0.999)
        self.trainer = TrainingEngine(self.network, optimizer, mean_squared_error)
        
    def train(self, epochs: int = 1000, verbose: bool = True) -> Dict[str, Any]:
        """Train the neural network"""
        if self.network is None:
            self.create_network()
            
        # Prepare data
        X, y = self.data_processor.prepare_scenario_data(self.scenario, normalize=True)
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.data_processor.split_data(X, y)
        
        # Train the model
        start_time = time.time()
        
        if verbose:
            print(f"🚀 Training {self.scenario.name}...")
            print(f"   Input shape: {X_train.shape}")
            print(f"   Target shape: {y_train.shape}")
            print(f"   Network: {self.scenario.network_architecture}")
        
        try:
            self.training_history = self.trainer.train(
                X_train, y_train,
                epochs=epochs,
                validation_data=(X_val, y_val),
                verbose=verbose,
                plot_progress=False
            )
            
            training_time = time.time() - start_time
            self.performance_stats['training_time'] = training_time
            self.is_trained = True
            
            # Evaluate on test set
            test_results = self.evaluate(X_test, y_test)
            
            if verbose:
                print(f"✅ Training completed in {training_time:.2f}s")
                print(f"   Test R²: {test_results['r2']:.4f}")
                print(f"   Test MSE: {test_results['mse']:.6f}")
                
            return {
                'training_time': training_time,
                'test_results': test_results,
                'training_history': self.training_history
            }
            
        except Exception as e:
            if verbose:
                print(f"❌ Training failed: {str(e)}")
            raise e
    
    def predict(self, input_data: np.ndarray, return_confidence: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Make predictions with optional confidence estimation"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Transform input data
        X_transformed = self.data_processor.transform_input(self.scenario, input_data)
        
        # Make predictions
        y_pred_transformed = self.network.forward(X_transformed)
        
        # Inverse transform predictions
        y_pred = self.data_processor.inverse_transform_output(self.scenario, y_pred_transformed)
        
        if return_confidence:
            confidences = self._estimate_confidence(X_transformed)
            return y_pred, confidences
        else:
            return y_pred, None
    
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
