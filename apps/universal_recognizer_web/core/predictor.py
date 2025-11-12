"""
Prediction logic with mirror detection for universal character recognition.
"""

import numpy as np
from PIL import Image
import io
import base64
import sys
import os

# Add NeuralEngine to path
base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.insert(0, base_path)

# Import model manager - handle both direct and package imports
try:
    from apps.universal_recognizer_web.core.model_manager import get_model_manager
    from apps.universal_recognizer_web.core.preprocessor import preprocess_for_prediction, preprocess_with_metrics
except ImportError:
    # Fallback for direct execution
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from core.model_manager import get_model_manager
    from core.preprocessor import preprocess_for_prediction, preprocess_with_metrics


def index_to_character(index: int) -> str:
    """Convert class index (0-61) to character."""
    if 0 <= index <= 9:
        return str(index)
    elif 10 <= index <= 35:
        return chr(ord('A') + index - 10)
    elif 36 <= index <= 61:
        return chr(ord('a') + index - 36)
    else:
        return '?'


def character_to_index(char: str) -> int:
    """Convert character to class index (0-61)."""
    if char.isdigit():
        return int(char)
    elif char.isupper() and char.isalpha():
        return ord(char) - ord('A') + 10
    elif char.islower() and char.isalpha():
        return ord(char) - ord('a') + 36
    else:
        return -1


def get_character_type(index: int) -> str:
    """Get character type description."""
    if 0 <= index <= 9:
        return "Digit"
    elif 10 <= index <= 35:
        return "Uppercase"
    elif 36 <= index <= 61:
        return "Lowercase"
    else:
        return "Unknown"


def preprocess_image(image_data, normalize=True):
    """
    Preprocess image using advanced preprocessing framework.
    
    This function is kept for backwards compatibility but now uses
    the advanced preprocessor which handles all transformations automatically.
    
    Args:
        image_data: Base64 string or numpy array
        normalize: Ignored (always normalized by preprocessor)
    
    Returns:
        Preprocessed image array (1, 784) or None on error
    """
    try:
        return preprocess_for_prediction(image_data)
    except Exception as e:
        print(f"Image preprocessing error: {e}")
        return None


def predict_character(image_data, return_quality_metrics=False, is_test_image=False, return_debug=False):
    """
    Standard character prediction with advanced preprocessing.
    
    Args:
        image_data: Base64 string or numpy array
        return_quality_metrics: Whether to return quality metrics for display
        is_test_image: If True, skip EMNIST orientation fix (test images already fixed)
        return_debug: Whether to return debug images for visualization
    
    Returns:
        Dictionary with prediction results and optional quality metrics and debug images
    """
    try:
        model_manager = get_model_manager()
        model = model_manager.get_model()
        
        # Preprocess image using advanced preprocessor
        if return_quality_metrics:
            result = preprocess_with_metrics(image_data, is_test_image=is_test_image, return_debug=return_debug)
            if return_debug:
                processed_image, quality_metrics, debug_images = result
            else:
                processed_image, quality_metrics = result
                debug_images = None
        else:
            result = preprocess_for_prediction(image_data, is_test_image=is_test_image, return_debug=return_debug)
            if return_debug:
                processed_image, debug_images = result
            else:
                processed_image = result
            quality_metrics = None
            debug_images = None
        
        if processed_image is None:
            return None
        
        # Make prediction
        predictions = model.forward(processed_image)
        predictions = predictions.flatten()
        
        # Get top prediction
        predicted_index = int(np.argmax(predictions))
        predicted_char = index_to_character(predicted_index)
        confidence = float(predictions[predicted_index]) * 100
        
        # Get top 5 predictions - ensure all values are native Python types
        top_indices = np.argsort(predictions)[::-1][:5]
        top_predictions = [
            {
                'character': index_to_character(int(idx)),
                'index': int(idx),
                'confidence': float(predictions[idx]) * 100.0,  # Ensure native float
                'type': get_character_type(int(idx))
            }
            for idx in top_indices
        ]
        
        result = {
            'predicted_character': predicted_char,
            'predicted_index': predicted_index,
            'confidence': float(confidence),  # Ensure native Python float
            'predictions': [float(x) for x in predictions.tolist()],  # Convert all to native floats
            'top_predictions': top_predictions,
            'character_type': get_character_type(predicted_index)
        }
        
        # Add quality metrics if requested
        if return_quality_metrics and quality_metrics:
            result['quality_metrics'] = quality_metrics
        
        # Add debug images if requested
        if return_debug and debug_images:
            result['debug_images'] = debug_images
        
        return result
    
    except Exception as e:
        print(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return None


def predict_with_mirror_detection(image_data, mirror_threshold=0.15):
    """
    Predict character with mirror detection for accessibility.
    
    Tests both original and horizontally flipped versions using advanced preprocessing.
    Note: Since EMNIST dataset already contains mirrored images, we use a higher
    threshold to avoid false positives.
    
    Args:
        image_data: Base64 string or numpy array
        mirror_threshold: Minimum confidence improvement to flag as mirrored (0.15 = 15%)
    
    Returns:
        Dictionary with both predictions and mirror detection results
    """
    try:
        # Original prediction with quality metrics
        original_result = predict_character(image_data, return_quality_metrics=True)
        if original_result is None:
            return None
        
        # Get preprocessed image for mirroring
        processed_image = preprocess_for_prediction(image_data)
        if processed_image is None:
            return {'original': original_result, 'mirrored': None, 'mirror_detected': False}
        
        # Create mirrored version (flip horizontally)
        img_2d = processed_image.reshape(28, 28)
        img_mirrored = np.flip(img_2d, axis=1)  # Horizontal flip
        
        # Re-normalize mirrored image (it needs to go through EMNIST normalization again)
        # But we already have it in the right format, just need to ensure it's normalized
        img_mirrored_flat = img_mirrored.flatten().reshape(1, -1)
        
        # Predict on mirrored version
        model_manager = get_model_manager()
        model = model_manager.get_model()
        
        predictions_mirrored = model.forward(img_mirrored_flat)
        predictions_mirrored = predictions_mirrored.flatten()
        
        predicted_index_mirrored = int(np.argmax(predictions_mirrored))
        predicted_char_mirrored = index_to_character(predicted_index_mirrored)
        confidence_mirrored = float(predictions_mirrored[predicted_index_mirrored]) * 100
        
        # Determine if mirror improves confidence significantly
        # Use higher threshold since dataset has mirrored images
        confidence_diff = confidence_mirrored - original_result['confidence']
        is_mirrored = confidence_diff > (mirror_threshold * 100)
        
        mirror_result = {
            'predicted_character': predicted_char_mirrored,
            'predicted_index': predicted_index_mirrored,
            'confidence': float(confidence_mirrored),  # Ensure native Python float
            'is_mirrored': is_mirrored,
            'confidence_improvement': float(confidence_diff)  # Ensure native Python float
        }
        
        return {
            'original': original_result,
            'mirrored': mirror_result,
            'mirror_detected': is_mirrored
        }
    
    except Exception as e:
        print(f"Mirror detection error: {e}")
        import traceback
        traceback.print_exc()
        return {'original': original_result, 'mirrored': None, 'mirror_detected': False} if 'original_result' in locals() else None


def analyze_writing_quality(image_data):
    """
    Analyze writing quality metrics using preprocessing framework.
    
    Quality metrics are now calculated during preprocessing and returned
    for display purposes only (not used in prediction).
    
    Args:
        image_data: Base64 string or numpy array
    
    Returns:
        Dictionary with quality metrics
    """
    try:
        # Use preprocessor to get quality metrics
        _, quality_metrics = preprocess_with_metrics(image_data)
        return quality_metrics
    except Exception as e:
        print(f"Quality analysis error: {e}")
        # Return default metrics on error
        return {
            'overall_score': 50.0,
            'clarity_score': 50.0,
            'size_score': 50.0,
            'centering_score': 50.0,
            'stroke_score': 50.0,
            'metrics': {
                'contrast': 0.0,
                'size_ratio': 0.0,
                'center_offset': 0.0,
                'edge_strength': 0.0
            }
        }

