"""
Comprehensive model evaluation for universal character recognition.
"""

import os
import sys
import numpy as np
import autograd.numpy as anp
from typing import Dict, List, Tuple

# Add paths - need to get to NeuralEngine root
# __file__ is apps/universal_recognizer_web/training/evaluator.py
# Go up 3 levels to get to NeuralEngine root
_this_file = os.path.abspath(__file__)
base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_this_file))))
if base_path not in sys.path:
    sys.path.insert(0, base_path)

from nn_core import cross_entropy_loss

# Handle both script and module execution
try:
    from .data_loader import index_to_character, get_character_type
except ImportError:
    # Running as script
    training_dir = os.path.dirname(os.path.abspath(__file__))
    if training_dir not in sys.path:
        sys.path.insert(0, training_dir)
    from data_loader import index_to_character, get_character_type


def evaluate_model_comprehensive(model, X_test: anp.ndarray, y_test: anp.ndarray) -> Dict:
    """
    Comprehensive model evaluation.
    
    Args:
        model: Trained neural network
        X_test: Test inputs
        y_test: Test labels (one-hot)
    
    Returns:
        Dictionary with comprehensive metrics
    """
    print("Running comprehensive evaluation...")
    
    # Get predictions
    predictions = model.forward(X_test)
    predicted_classes = anp.argmax(predictions, axis=1)
    true_classes = anp.argmax(y_test, axis=1)
    
    # Overall metrics
    accuracy = anp.mean(predicted_classes == true_classes) * 100
    loss = cross_entropy_loss(y_test, predictions)
    confidences = anp.max(predictions, axis=1) * 100
    avg_confidence = anp.mean(confidences)
    
    # Character type analysis
    digit_mask = true_classes < 10
    upper_mask = (true_classes >= 10) & (true_classes < 36)
    lower_mask = true_classes >= 36
    
    digit_accuracy = 0.0
    upper_accuracy = 0.0
    lower_accuracy = 0.0
    
    if anp.any(digit_mask):
        digit_accuracy = anp.mean(predicted_classes[digit_mask] == true_classes[digit_mask]) * 100
    
    if anp.any(upper_mask):
        upper_accuracy = anp.mean(predicted_classes[upper_mask] == true_classes[upper_mask]) * 100
    
    if anp.any(lower_mask):
        lower_accuracy = anp.mean(predicted_classes[lower_mask] == true_classes[lower_mask]) * 100
    
    # Per-character accuracy
    per_char_accuracy = {}
    for char_idx in range(62):
        char_mask = true_classes == char_idx
        if anp.any(char_mask):
            char_acc = anp.mean(predicted_classes[char_mask] == true_classes[char_mask]) * 100
            per_char_accuracy[char_idx] = {
                'character': index_to_character(char_idx),
                'accuracy': float(char_acc),
                'samples': int(anp.sum(char_mask))
            }
    
    return {
        'overall_accuracy': float(accuracy),
        'loss': float(loss),
        'avg_confidence': float(avg_confidence),
        'character_type_accuracies': {
            'digits': float(digit_accuracy),
            'uppercase': float(upper_accuracy),
            'lowercase': float(lower_accuracy)
        },
        'per_character_accuracy': per_char_accuracy,
        'confidences': confidences.tolist(),
        'predictions': predictions,
        'predicted_classes': predicted_classes,
        'true_classes': true_classes
    }

