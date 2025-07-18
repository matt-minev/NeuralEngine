#!/usr/bin/env python3
"""
Quadratic Neural Network Web Application
Neural Engine Integration Layer

Author: Matt
Location: Varna, Bulgaria
Date: July 2025

Integration layer between the web application and the existing Neural Engine
"""

import numpy as np
import json
import time
import threading
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

# Import existing Neural Engine components
try:
    from nn_core import NeuralNetwork, mean_squared_error, mean_absolute_error
    from autodiff import TrainingEngine, Adam, SGD
    from data_utils import DataPreprocessor
    from config.scenarios import PredictionScenario
except ImportError as e:
    print(f"Warning: Neural Engine components not found: {e}")
    print("Make sure the Neural Engine is properly installed")

@dataclass
class TrainingConfig:
    """Configuration for training sessions"""
    epochs: int = 1000
    learning_rate: float = 0.001
    batch_size: int = 32
    validation_split: float = 0.15
    test_split: float = 0.15
    optimizer: str = 'adam'
    loss_function: str = 'mse'
    early_stopping: bool = True
    patience: int = 50
    min_delta: float = 1e-6
    verbose: bool = False

@dataclass
class TrainingProgress:
    """Training progress tracking"""
    current_epoch: int = 0
    total_epochs: int = 0
    current_scenario: str = ""
    scenario_progress: int = 0
    total_scenarios: int = 0
    training_loss: float = 0.0
    validation_loss: float = 0.0
    is_training: bool = False
    start_time: Optional[datetime] = None
    elapsed_time: float = 0.0
    estimated_remaining: float = 0.0

class NeuralEngineIntegrator:
    """Main integration class for the Neural Engine"""
    
    def __init__(self, data_processor, scenarios: Dict[str, PredictionScenario]):
        self.data_processor = data_processor
        self.scenarios = scenarios
        self.predictors = {}
        self.training_configs = {}
        self.training_progress = TrainingProgress()
        self.training_thread = None
        self.stop_training_flag = False
        self.logger = logging.getLogger(__name__)
        
        # Initialize default training configurations
        self._init_default_configs()
    
    def _init_default_configs(self):
        """Initialize default training configurations for each scenario"""
        for scenario_key, scenario in self.scenarios.items():
            self.training_configs[scenario_key] = TrainingConfig(
                epochs=1000,
                learning_rate=0.001,
                batch_size=32,
                optimizer='adam'
            )
    
    def create_neural_network(self, scenario: PredictionScenario) -> NeuralNetwork:
        """Create a neural network for the given scenario"""
        try:
            network = NeuralNetwork(
                scenario.network_architecture,
                scenario.activations
            )
            
            self.logger.info(f"Created neural network for {scenario.name}")
            self.logger.info(f"Architecture: {scenario.network_architecture}")
            self.logger.info(f"Activations: {scenario.activations}")
            
            return network
            
        except Exception as e:
            self.logger.error(f"Failed to create neural network: {e}")
            raise
    
    def create_training_engine(self, network: NeuralNetwork, config: TrainingConfig) -> TrainingEngine:
        """Create a training engine with the specified configuration"""
        try:
            # Select optimizer
            if config.optimizer.lower() == 'adam':
                optimizer = Adam(
                    learning_rate=config.learning_rate,
                    beta1=0.9,
                    beta2=0.999,
                    epsilon=1e-8
                )
            elif config.optimizer.lower() == 'sgd':
                optimizer = SGD(learning_rate=config.learning_rate)
            else:
                raise ValueError(f"Unsupported optimizer: {config.optimizer}")
            
            # Select loss function
            if config.loss_function.lower() == 'mse':
                loss_fn = mean_squared_error
            elif config.loss_function.lower() == 'mae':
                loss_fn = mean_absolute_error
            else:
                raise ValueError(f"Unsupported loss function: {config.loss_function}")
            
            trainer = TrainingEngine(network, optimizer, loss_fn)
            
            self.logger.info(f"Created training engine with {config.optimizer} optimizer")
            return trainer
            
        except Exception as e:
            self.logger.error(f"Failed to create training engine: {e}")
            raise
    
    def prepare_training_data(self, scenario: PredictionScenario) -> Tuple[np.ndarray, ...]:
        """Prepare training data for the given scenario"""
        try:
            # Get data for the scenario
            X, y = self.data_processor.prepare_scenario_data(scenario, normalize=True)
            
            # Split data
            X_train, X_val, X_test, y_train, y_val, y_test = \
                self.data_processor.split_data(X, y)
            
            self.logger.info(f"Data prepared for {scenario.name}")
            self.logger.info(f"Training: {X_train.shape}, Validation: {X_val.shape}, Test: {X_test.shape}")
            
            return X_train, X_val, X_test, y_train, y_val, y_test
            
        except Exception as e:
            self.logger.error(f"Failed to prepare training data: {e}")
            raise
    
    def train_single_model(self, scenario_key: str, config: Optional[TrainingConfig] = None) -> Dict[str, Any]:
        """Train a single model for the given scenario"""
        if config is None:
            config = self.training_configs[scenario_key]
        
        scenario = self.scenarios[scenario_key]
        
        try:
            # Create network and trainer
            network = self.create_neural_network(scenario)
            trainer = self.create_training_engine(network, config)
            
            # Prepare data
            X_train, X_val, X_test, y_train, y_val, y_test = \
                self.prepare_training_data(scenario)
            
            # Train the model
            start_time = time.time()
            
            training_history = trainer.train(
                X_train, y_train,
                epochs=config.epochs,
                validation_data=(X_val, y_val),
                verbose=config.verbose,
                plot_progress=False
            )
            
            training_time = time.time() - start_time
            
            # Evaluate on test set
            test_predictions = network.forward(X_test)
            test_metrics = self.calculate_metrics(y_test, test_predictions)
            
            # Create predictor wrapper
            predictor = NeuralPredictor(
                network=network,
                trainer=trainer,
                scenario=scenario,
                data_processor=self.data_processor,
                training_history=training_history,
                test_metrics=test_metrics,
                training_time=training_time
            )
            
            # Store the predictor
            self.predictors[scenario_key] = predictor
            
            self.logger.info(f"Successfully trained {scenario.name}")
            self.logger.info(f"Test R²: {test_metrics['r2']:.4f}, Time: {training_time:.2f}s")
            
            return {
                'success': True,
                'scenario': scenario.name,
                'training_time': training_time,
                'test_metrics': test_metrics,
                'training_history': training_history
            }
            
        except Exception as e:
            self.logger.error(f"Failed to train {scenario.name}: {e}")
            return {
                'success': False,
                'scenario': scenario.name,
                'error': str(e)
            }
    
    def train_multiple_models(self, scenario_keys: List[str], 
                            configs: Optional[Dict[str, TrainingConfig]] = None,
                            progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """Train multiple models with progress tracking"""
        
        if configs is None:
            configs = {key: self.training_configs[key] for key in scenario_keys}
        
        results = {}
        total_scenarios = len(scenario_keys)
        
        self.training_progress.total_scenarios = total_scenarios
        self.training_progress.is_training = True
        self.training_progress.start_time = datetime.now()
        
        try:
            for i, scenario_key in enumerate(scenario_keys):
                if self.stop_training_flag:
                    break
                
                self.training_progress.current_scenario = self.scenarios[scenario_key].name
                self.training_progress.scenario_progress = i + 1
                
                if progress_callback:
                    progress_callback(self.training_progress)
                
                # Train single model
                result = self.train_single_model(scenario_key, configs.get(scenario_key))
                results[scenario_key] = result
                
                # Update progress
                overall_progress = ((i + 1) / total_scenarios) * 100
                self.training_progress.elapsed_time = (
                    datetime.now() - self.training_progress.start_time
                ).total_seconds()
                
                if i > 0:
                    avg_time_per_scenario = self.training_progress.elapsed_time / (i + 1)
                    remaining_scenarios = total_scenarios - (i + 1)
                    self.training_progress.estimated_remaining = \
                        avg_time_per_scenario * remaining_scenarios
                
                if progress_callback:
                    progress_callback(self.training_progress)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Multi-model training failed: {e}")
            return {'error': str(e)}
            
        finally:
            self.training_progress.is_training = False
            self.stop_training_flag = False
    
    def start_background_training(self, scenario_keys: List[str], 
                                configs: Optional[Dict[str, TrainingConfig]] = None,
                                progress_callback: Optional[callable] = None) -> None:
        """Start training in background thread"""
        
        if self.training_thread and self.training_thread.is_alive():
            raise RuntimeError("Training already in progress")
        
        def training_worker():
            try:
                results = self.train_multiple_models(scenario_keys, configs, progress_callback)
                self.logger.info("Background training completed")
                return results
            except Exception as e:
                self.logger.error(f"Background training failed: {e}")
                raise
        
        self.training_thread = threading.Thread(target=training_worker)
        self.training_thread.daemon = True
        self.training_thread.start()
    
    def stop_training(self):
        """Stop current training session"""
        self.stop_training_flag = True
        self.training_progress.is_training = False
        self.logger.info("Training stop requested")
    
    def make_prediction(self, scenario_key: str, input_data: np.ndarray, 
                       return_confidence: bool = True) -> Dict[str, Any]:
        """Make prediction using trained model"""
        
        if scenario_key not in self.predictors:
            raise ValueError(f"No trained model for scenario: {scenario_key}")
        
        predictor = self.predictors[scenario_key]
        
        try:
            predictions, confidences = predictor.predict(input_data, return_confidence)
            
            return {
                'success': True,
                'predictions': predictions.tolist() if isinstance(predictions, np.ndarray) else predictions,
                'confidences': confidences.tolist() if confidences is not None else None,
                'scenario': predictor.scenario.name,
                'target_features': predictor.scenario.target_features
            }
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics"""
        
        # Ensure proper shapes
        if y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)
        
        # Basic metrics
        mse = np.mean((y_true - y_pred) ** 2)
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(mse)
        
        # R² score
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Accuracy within tolerance
        tolerance = 0.1
        relative_error = np.abs((y_true - y_pred) / (y_true + 1e-8))
        accuracy_10pct = np.mean(relative_error < tolerance) * 100
        
        # Additional metrics
        max_error = np.max(np.abs(y_true - y_pred))
        mean_error = np.mean(y_true - y_pred)  # Bias
        
        return {
            'mse': float(mse),
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2),
            'accuracy_10pct': float(accuracy_10pct),
            'max_error': float(max_error),
            'mean_error': float(mean_error)
        }
    
    def get_model_info(self, scenario_key: str) -> Dict[str, Any]:
        """Get detailed information about a trained model"""
        
        if scenario_key not in self.predictors:
            return {'error': f'No model trained for scenario: {scenario_key}'}
        
        predictor = self.predictors[scenario_key]
        
        try:
            info = {
                'scenario': {
                    'name': predictor.scenario.name,
                    'description': predictor.scenario.description,
                    'input_features': predictor.scenario.input_features,
                    'target_features': predictor.scenario.target_features,
                    'network_architecture': predictor.scenario.network_architecture,
                    'activations': predictor.scenario.activations
                },
                'network': {
                    'parameters': predictor.network.count_parameters(),
                    'layers': len(predictor.scenario.network_architecture),
                    'input_size': predictor.scenario.network_architecture[0],
                    'output_size': predictor.scenario.network_architecture[-1]
                },
                'training': {
                    'training_time': predictor.training_time,
                    'epochs_trained': len(predictor.training_history.get('train_losses', [])),
                    'final_train_loss': predictor.training_history.get('train_losses', [])[-1] if predictor.training_history.get('train_losses') else None,
                    'final_val_loss': predictor.training_history.get('val_losses', [])[-1] if predictor.training_history.get('val_losses') else None
                },
                'performance': predictor.test_metrics
            }
            
            return info
            
        except Exception as e:
            self.logger.error(f"Failed to get model info: {e}")
            return {'error': str(e)}
    
    def export_model(self, scenario_key: str, filepath: str) -> bool:
        """Export trained model to file"""
        
        if scenario_key not in self.predictors:
            return False
        
        predictor = self.predictors[scenario_key]
        
        try:
            # Create export data
            export_data = {
                'scenario': asdict(predictor.scenario),
                'network_parameters': predictor.network.get_all_parameters(),
                'training_history': predictor.training_history,
                'test_metrics': predictor.test_metrics,
                'training_time': predictor.training_time,
                'export_timestamp': datetime.now().isoformat()
            }
            
            # Convert numpy arrays to lists for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_numpy(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj]
                else:
                    return obj
            
            export_data = convert_numpy(export_data)
            
            # Save to file
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            self.logger.info(f"Model exported to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export model: {e}")
            return False
    
    def import_model(self, scenario_key: str, filepath: str) -> bool:
        """Import trained model from file"""
        
        try:
            with open(filepath, 'r') as f:
                import_data = json.load(f)
            
            # Recreate scenario
            scenario_data = import_data['scenario']
            scenario = PredictionScenario(**scenario_data)
            
            # Recreate network
            network = self.create_neural_network(scenario)
            
            # Restore parameters
            parameters = import_data['network_parameters']
            # Convert lists back to numpy arrays
            parameters = [np.array(param) for param in parameters]
            network.set_all_parameters(parameters)
            
            # Create predictor
            predictor = NeuralPredictor(
                network=network,
                trainer=None,
                scenario=scenario,
                data_processor=self.data_processor,
                training_history=import_data['training_history'],
                test_metrics=import_data['test_metrics'],
                training_time=import_data['training_time']
            )
            
            self.predictors[scenario_key] = predictor
            
            self.logger.info(f"Model imported from {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to import model: {e}")
            return False
    
    def get_training_progress(self) -> Dict[str, Any]:
        """Get current training progress"""
        return {
            'is_training': self.training_progress.is_training,
            'current_scenario': self.training_progress.current_scenario,
            'scenario_progress': self.training_progress.scenario_progress,
            'total_scenarios': self.training_progress.total_scenarios,
            'elapsed_time': self.training_progress.elapsed_time,
            'estimated_remaining': self.training_progress.estimated_remaining,
            'start_time': self.training_progress.start_time.isoformat() if self.training_progress.start_time else None
        }

class NeuralPredictor:
    """Wrapper class for trained neural network predictors"""
    
    def __init__(self, network: NeuralNetwork, trainer: TrainingEngine,
                 scenario: PredictionScenario, data_processor,
                 training_history: Dict, test_metrics: Dict, training_time: float):
        self.network = network
        self.trainer = trainer
        self.scenario = scenario
        self.data_processor = data_processor
        self.training_history = training_history
        self.test_metrics = test_metrics
        self.training_time = training_time
    
    def predict(self, input_data: np.ndarray, return_confidence: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Make predictions with optional confidence estimation"""
        
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

# Utility functions for web application integration
def create_integrator(data_processor, scenarios: Dict[str, PredictionScenario]) -> NeuralEngineIntegrator:
    """Factory function to create NeuralEngineIntegrator instance"""
    return NeuralEngineIntegrator(data_processor, scenarios)

def calculate_quadratic_solutions(a: float, b: float, c: float) -> Optional[Tuple[float, float]]:
    """Calculate actual solutions for quadratic equation ax² + bx + c = 0"""
    try:
        if abs(a) < 1e-10:
            if abs(b) < 1e-10:
                return None  # No solution or infinite solutions
            else:
                root = -c / b
                return (root, root)
        
        discriminant = b**2 - 4*a*c
        if discriminant < 0:
            return None  # No real solutions
        
        sqrt_discriminant = np.sqrt(discriminant)
        x1 = (-b + sqrt_discriminant) / (2*a)
        x2 = (-b - sqrt_discriminant) / (2*a)
        
        return (x1, x2)
        
    except Exception:
        return None

def validate_prediction_input(input_data: List[float], scenario: PredictionScenario) -> Tuple[bool, str]:
    """Validate input data for prediction"""
    
    if not input_data:
        return False, "No input data provided"
    
    if len(input_data) != len(scenario.input_features):
        return False, f"Expected {len(scenario.input_features)} inputs, got {len(input_data)}"
    
    # Check for valid numbers
    for i, value in enumerate(input_data):
        if not isinstance(value, (int, float)) or np.isnan(value) or np.isinf(value):
            return False, f"Invalid value for {scenario.input_features[i]}: {value}"
    
    # Scenario-specific validation
    if scenario.name == "Coefficients → Roots":
        a, b, c = input_data
        if abs(a) < 1e-10:
            return False, "Coefficient 'a' cannot be zero for quadratic equations"
    
    return True, "Valid input"

def format_prediction_results(predictions: np.ndarray, confidences: Optional[np.ndarray],
                            scenario: PredictionScenario, actual_solutions: Optional[List[float]] = None) -> Dict[str, Any]:
    """Format prediction results for web application display"""
    
    results = {
        'scenario': scenario.name,
        'description': scenario.description,
        'predictions': {}
    }
    
    for i, (feature, pred) in enumerate(zip(scenario.target_features, predictions)):
        pred_info = {
            'value': float(pred),
            'confidence': float(confidences[i]) if confidences is not None else None,
            'confidence_level': Utils.getConfidenceLevel(confidences[i]) if confidences is not None else None
        }
        
        if actual_solutions and i < len(actual_solutions):
            actual = actual_solutions[i]
            error = abs(pred - actual)
            error_percent = abs(error / (actual + 1e-8)) * 100
            
            pred_info['actual'] = float(actual)
            pred_info['error'] = float(error)
            pred_info['error_percent'] = float(error_percent)
            pred_info['accuracy_assessment'] = _assess_prediction_accuracy(error)
        
        results['predictions'][feature] = pred_info
    
    return results

def _assess_prediction_accuracy(error: float) -> str:
    """Assess prediction accuracy based on error"""
    if error < 0.01:
        return "EXCELLENT"
    elif error < 0.1:
        return "GOOD"
    elif error < 1.0:
        return "MODERATE"
    else:
        return "POOR"

# Configuration for web application
WEB_APP_CONFIG = {
    'max_training_scenarios': 10,
    'default_epochs': 1000,
    'default_learning_rate': 0.001,
    'confidence_estimation_samples': 50,
    'prediction_timeout': 30.0,
    'training_timeout': 3600.0,
    'max_file_size': 50 * 1024 * 1024,  # 50MB
    'supported_file_types': ['.csv', '.json'],
    'export_formats': ['json', 'csv', 'txt']
}

if __name__ == "__main__":
    # Test the integration layer
    print("🧠 Neural Engine Integration Layer")
    print("=" * 50)
    print("Testing integration components...")
    
    # Test configuration
    config = TrainingConfig(epochs=100, learning_rate=0.01)
    print(f"✅ Training configuration: {config}")
    
    # Test progress tracking
    progress = TrainingProgress()
    print(f"✅ Progress tracking: {progress}")
    
    print("🎉 Integration layer ready!")
