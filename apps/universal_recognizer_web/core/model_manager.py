"""
Model loading and management for universal character recognition.
"""

import os
import sys
import pickle
import numpy as np
from .preprocess_contract import load_contract

# Add NeuralEngine to path
base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.insert(0, base_path)


class ModelManager:
    """Manages loading and access to the universal character recognition model."""
    
    def __init__(self, model_path=None):
        """
        Initialize model manager.
        
        Args:
            model_path: Path to model file. If None, uses default location.
        """
        if model_path is None:
            # Default path: apps/universal_recognizer_web/models/universal_character_model.pkl
            # __file__ is apps/universal_recognizer_web/core/model_manager.py
            # Go up to universal_recognizer_web, then into models
            base_dir = os.path.dirname(os.path.dirname(__file__))
            model_path = os.path.join(
                base_dir, 
                'models', 
                'universal_character_model.pkl'
            )
        
        self.model_path = model_path
        self.model = None
        self.model_info = {}
        self._load_model()
    
    def _load_model(self):
        """Load the trained universal character model."""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            print(f"Loading universal character model from {self.model_path}...")
            
            # Add NeuralEngine root to path (for nn_core, autodiff imports)
            # __file__ is apps/universal_recognizer_web/core/model_manager.py
            # Go up 3 levels to get to NeuralEngine root
            neural_engine_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            if neural_engine_root not in sys.path:
                sys.path.insert(0, neural_engine_root)
            
            # Add training directory to path for config import
            # The config module is at apps/universal_recognizer_web/training/config.py
            training_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'training')
            if training_dir not in sys.path:
                sys.path.insert(0, training_dir)
            
            # Load model - config should now be importable
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            contract = load_contract()
            self.model_info = {
                'accuracy': model_data.get('accuracy', 0.0),
                'architecture': model_data.get('architecture', 'universal_character_recognizer'),
                'classes': model_data.get('classes', 62),
                'avg_confidence': model_data.get('avg_confidence', 0.0),
                'training_time': model_data.get('training_time', 0.0),
                'character_type_accuracies': model_data.get('character_type_accuracies', {}),
                'layer_sizes': self.model.layer_sizes,
                'total_parameters': self.model.count_parameters(),
                'activations': [layer.activation_name for layer in self.model.layers],
                'model_version': model_data.get('model_version', 'universal_v1_legacy'),
                'contract_version': model_data.get('contract_version', contract.version),
                'contract_checksum': contract.checksum,
                'calibration': model_data.get('calibration', {'temperature': 1.0}),
                'engine_backend': getattr(self.model, 'execution_backend', 'python_fallback'),
                'device': getattr(self.model, 'device', 'cpu'),
                'backend_name': getattr(self.model, 'backend_name', 'numpy'),
            }
            
            print(f"Model loaded successfully!")
            print(f"  Accuracy: {self.model_info['accuracy']:.2f}%")
            print(f"  Classes: {self.model_info['classes']}")
            print(f"  Architecture: {self.model_info['layer_sizes']}")
            
        except Exception as e:
            print(f"Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def get_model(self):
        """Get the loaded model."""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        return self.model
    
    def get_model_info(self):
        """Get model information."""
        return self.model_info
    
    def is_loaded(self):
        """Check if model is loaded."""
        return self.model is not None


# Global model manager instance
_model_manager = None


def get_model_manager(model_path=None):
    """Get or create the global model manager instance."""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager(model_path)
    return _model_manager
