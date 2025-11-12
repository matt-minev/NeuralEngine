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
except ImportError:
    # Fallback for direct execution
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from core.model_manager import get_model_manager


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
    Preprocess image data for Neural Engine prediction.
    
    Matches EMNIST preprocessing: 28x28, normalized to [-1, 1] range.
    
    Args:
        image_data: Base64 string or numpy array
        normalize: Whether to normalize to [-1, 1] range (EMNIST format)
    
    Returns:
        Preprocessed image array (1, 784) or None on error
    """
    try:
        # Handle different input types
        if isinstance(image_data, list):
            # Direct array input
            img_array = np.array(image_data, dtype=np.float32)
            if img_array.max() > 1:
                img_array = img_array / 255.0
        elif isinstance(image_data, str):
            # Base64 string input
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
            
            # Convert to PIL image
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to grayscale and resize to 28x28
            image = image.convert('L')
            image = image.resize((28, 28), Image.Resampling.LANCZOS)
            
            # Convert to numpy array
            img_array = np.array(image, dtype=np.float32)
            img_array = img_array / 255.0
            
            # Invert if background is white (EMNIST is black on white)
            if np.mean(img_array) > 0.5:
                img_array = 1.0 - img_array
        else:
            raise ValueError(f"Unsupported image data type: {type(image_data)}")
        
        # Ensure 28x28 shape
        if img_array.shape != (28, 28):
            img_array = img_array.reshape(28, 28)
        
        # Apply EMNIST normalization: normalize to [-1, 1]
        if normalize:
            # EMNIST preprocessing: contrast enhancement and centering
            # Match the training preprocessing from load_data.py
            contrast_factor = 1.2
            img_array = np.clip(img_array * contrast_factor, 0, 1)
            
            # Center and normalize to [-1, 1]
            mean = np.mean(img_array)
            img_array = img_array - mean
            img_array = img_array * 2.0
            img_array = np.clip(img_array, -1, 1)
        
        # Flatten for neural network
        return img_array.flatten().reshape(1, -1)
    
    except Exception as e:
        print(f"Image preprocessing error: {e}")
        return None


def predict_character(image_data):
    """
    Standard character prediction.
    
    Args:
        image_data: Base64 string or numpy array
    
    Returns:
        Dictionary with prediction results
    """
    try:
        model_manager = get_model_manager()
        model = model_manager.get_model()
        
        # Preprocess image
        processed_image = preprocess_image(image_data)
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
        
        return {
            'predicted_character': predicted_char,
            'predicted_index': predicted_index,
            'confidence': float(confidence),  # Ensure native Python float
            'predictions': [float(x) for x in predictions.tolist()],  # Convert all to native floats
            'top_predictions': top_predictions,
            'character_type': get_character_type(predicted_index)
        }
    
    except Exception as e:
        print(f"Prediction error: {e}")
        return None


def predict_with_mirror_detection(image_data, mirror_threshold=0.1):
    """
    Predict character with mirror detection for accessibility.
    
    Tests both original and horizontally flipped versions.
    
    Args:
        image_data: Base64 string or numpy array
        mirror_threshold: Minimum confidence improvement to flag as mirrored (0.1 = 10%)
    
    Returns:
        Dictionary with both predictions and mirror detection results
    """
    try:
        # Original prediction
        original_result = predict_character(image_data)
        if original_result is None:
            return None
        
        # Preprocess for mirror
        processed_image = preprocess_image(image_data)
        if processed_image is None:
            return original_result
        
        # Create mirrored version
        img_2d = processed_image.reshape(28, 28)
        img_mirrored = np.flip(img_2d, axis=1)  # Horizontal flip
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
        return {'original': original_result, 'mirrored': None, 'mirror_detected': False} if 'original_result' in locals() else None


def analyze_writing_quality(image_data):
    """
    Analyze writing quality metrics.
    
    Args:
        image_data: Base64 string or numpy array
    
    Returns:
        Dictionary with quality metrics
    """
    try:
        processed_image = preprocess_image(image_data, normalize=False)
        if processed_image is None:
            return None
        
        img_2d = processed_image.reshape(28, 28)
        
        # Calculate metrics
        # 1. Stroke clarity (contrast)
        contrast = np.std(img_2d)
        
        # 2. Character size (non-zero pixels)
        non_zero_pixels = np.sum(img_2d > 0.1)
        size_ratio = non_zero_pixels / (28 * 28)
        
        # 3. Centering (center of mass)
        y_coords, x_coords = np.where(img_2d > 0.1)
        if len(x_coords) > 0:
            center_x = np.mean(x_coords)
            center_y = np.mean(y_coords)
            center_offset = np.sqrt((center_x - 14)**2 + (center_y - 14)**2)
        else:
            center_offset = 14.0  # Worst case
        
        # 4. Stroke thickness (edge detection)
        try:
            from scipy import ndimage
            edges = ndimage.sobel(img_2d)
            edge_strength = np.mean(np.abs(edges))
        except ImportError:
            # Fallback if scipy not available
            edge_strength = contrast * 0.5
        
        # Quality scores (0-100) - convert all to native Python floats
        clarity_score = float(min(contrast * 100, 100))
        size_score = float(min(size_ratio * 200, 100))  # Good size is ~30-50% of image
        centering_score = float(max(100 - (center_offset / 14.0 * 100), 0))
        stroke_score = float(min(edge_strength * 100, 100))
        
        overall_score = float((clarity_score + size_score + centering_score + stroke_score) / 4)
        
        return {
            'overall_score': overall_score,
            'clarity_score': clarity_score,
            'size_score': size_score,
            'centering_score': centering_score,
            'stroke_score': stroke_score,
            'metrics': {
                'contrast': float(contrast),
                'size_ratio': float(size_ratio),
                'center_offset': float(center_offset),
                'edge_strength': float(edge_strength)
            }
        }
    
    except Exception as e:
        print(f"Quality analysis error: {e}")
        # Fallback without scipy
        processed_image = preprocess_image(image_data, normalize=False)
        if processed_image is None:
            return None
        
        img_2d = processed_image.reshape(28, 28)
        contrast = np.std(img_2d)
        non_zero_pixels = np.sum(img_2d > 0.1)
        size_ratio = non_zero_pixels / (28 * 28)
        
        return {
            'overall_score': float(min(contrast * 100, 100)),
            'clarity_score': float(min(contrast * 100, 100)),
            'size_score': float(min(size_ratio * 200, 100)),
            'centering_score': 50.0,  # Default
            'stroke_score': 50.0,  # Default
            'metrics': {
                'contrast': float(contrast),
                'size_ratio': float(size_ratio),
                'center_offset': 0.0,
                'edge_strength': 0.0
            }
        }

