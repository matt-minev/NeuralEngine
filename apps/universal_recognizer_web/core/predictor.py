"""
Prediction logic with mirror detection for universal character recognition.
"""

import numpy as np
import sys
import os

# Add NeuralEngine to path
base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.insert(0, base_path)

# Import model manager - handle both direct and package imports
try:
    from apps.universal_recognizer_web.core.model_manager import get_model_manager
    from apps.universal_recognizer_web.core.preprocessor import preprocess_for_prediction, preprocess_with_metrics
    from apps.universal_recognizer_web.core.canonical_preprocessor import CanonicalPreprocessorV2
    from apps.universal_recognizer_web.core.preprocess_contract import load_contract
except ImportError:
    # Fallback for direct execution
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from core.model_manager import get_model_manager
    from core.preprocessor import preprocess_for_prediction, preprocess_with_metrics
    from core.canonical_preprocessor import CanonicalPreprocessorV2
    from core.preprocess_contract import load_contract


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


def _mirror_payload(image_data):
    """Horizontally mirror strict canvas payload for advisory analysis."""
    if not isinstance(image_data, dict):
        return image_data

    payload = dict(image_data)
    canvas = dict(payload.get('canvas', {}) if isinstance(payload.get('canvas', {}), dict) else {})
    width = float(canvas.get('width', 280))
    strokes = payload.get('strokes', [])
    mirrored_strokes = []

    if isinstance(strokes, list):
        for stroke in strokes:
            if not isinstance(stroke, dict):
                continue
            pts = stroke.get('points', [])
            new_pts = []
            for p in pts:
                if isinstance(p, dict):
                    x = p.get('x')
                    y = p.get('y')
                    if x is None or y is None:
                        continue
                    xf = float(x)
                    yf = float(y)
                    if 0.0 <= xf <= 1.0:
                        mx = 1.0 - xf
                    else:
                        mx = max(0.0, width - xf)
                    q = dict(p)
                    q['x'] = mx
                    q['y'] = yf
                    new_pts.append(q)
                elif isinstance(p, (list, tuple)) and len(p) >= 2:
                    xf = float(p[0])
                    yf = float(p[1])
                    if 0.0 <= xf <= 1.0:
                        mx = 1.0 - xf
                    else:
                        mx = max(0.0, width - xf)
                    new_pts.append([mx, yf, *(list(p)[2:])])
            mirrored_strokes.append({'points': new_pts})

    payload['strokes'] = mirrored_strokes
    # Drop raster to force deterministic stroke-based path in strict mode.
    if 'raster' in payload:
        payload.pop('raster')
    if 'image' in payload:
        payload.pop('image')
    payload['canvas'] = canvas
    return payload


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
        model_info = model_manager.get_model_info()
        temperature = float(model_info.get('calibration', {}).get('temperature', 1.0))
        if temperature > 0 and abs(temperature - 1.0) > 1e-6:
            logits = np.log(np.clip(predictions, 1e-12, 1.0))
            logits = logits / temperature
            e = np.exp(logits - np.max(logits))
            predictions = e / np.sum(e)
        
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
            'calibrated_confidence': float(confidence),
            'predictions': [float(x) for x in predictions.tolist()],  # Convert all to native floats
            'top_predictions': top_predictions,
            'character_type': get_character_type(predicted_index),
            'top_k': len(top_predictions),
            'model_version': model_info.get('model_version', 'legacy'),
            'contract_version': model_info.get('contract_version', 'v2'),
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


def predict_with_mirror_detection(image_data, mirror_threshold=0.05):
    """
    Predict character with mirror detection for accessibility.
    
    Tests both original and horizontally flipped versions.
    Uses LOWER threshold (5%) to be more sensitive to mirroring.
    
    Args:
        image_data: Base64 string or numpy array
        mirror_threshold: Minimum confidence improvement to flag as mirrored (0.05 = 5%)
    
    Returns:
        Dictionary with both predictions and mirror detection results
    """
    try:
        # Original prediction with quality metrics
        original_result = predict_character(image_data, return_quality_metrics=True)
        if original_result is None:
            return None
        
        # Build mirrored strict payload.
        preprocessor = CanonicalPreprocessorV2()
        mirrored_payload = _mirror_payload(image_data)
        img_mirrored_processed, _, _ = preprocessor.preprocess(
            mirrored_payload,
            return_metrics=False,
            return_debug=False,
            skip_transform=True,  # Never apply EMNIST transform to canvas input
            already_normalized=False,
            strict_mode=None,
        )
        if img_mirrored_processed is None:
            return {'original': original_result, 'mirrored': None, 'mirror_detected': False}
        
        # Predict on mirrored version
        model_manager = get_model_manager()
        model = model_manager.get_model()
        
        predictions_mirrored = model.forward(img_mirrored_processed)
        predictions_mirrored = predictions_mirrored.flatten()
        
        predicted_index_mirrored = int(np.argmax(predictions_mirrored))
        predicted_char_mirrored = index_to_character(predicted_index_mirrored)
        confidence_mirrored = float(predictions_mirrored[predicted_index_mirrored]) * 100
        
        confidence_diff = confidence_mirrored - original_result['confidence']
        char_different = predicted_char_mirrored != original_result['predicted_character']
        is_mirror_candidate = char_different and confidence_diff > (mirror_threshold * 100)

        mirror_result = {
            'predicted_character': predicted_char_mirrored,
            'predicted_index': predicted_index_mirrored,
            'confidence': float(confidence_mirrored),
            'is_mirrored': is_mirror_candidate,
            'confidence_improvement': float(confidence_diff)
        }
        
        return {
            'original': original_result,
            'mirrored': mirror_result,
            'mirror_detected': is_mirror_candidate
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


def get_confusion_candidates(predictions, top_n=5):
    """Return calibrated confusion candidates for advisory UI."""
    top_indices = np.argsort(predictions)[::-1][:top_n]
    return [
        {
            'character': index_to_character(int(idx)),
            'index': int(idx),
            'confidence': float(predictions[idx]) * 100.0,
            'type': get_character_type(int(idx))
        }
        for idx in top_indices
    ]


def get_preprocess_contract_metadata():
    contract = load_contract()
    return {
        'contract_version': contract.version,
        'contract_checksum': contract.checksum,
        'transform_id': contract.transform_id,
    }
