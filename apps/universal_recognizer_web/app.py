"""
Flask web application for universal character recognition with accessibility features.
"""

import os
import sys
import time
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
from pathlib import Path

# Add NeuralEngine to path
base_path = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, base_path)

# Import core modules
try:
    from apps.universal_recognizer_web.core.model_manager import get_model_manager
    from apps.universal_recognizer_web.core.predictor import (
        predict_character,
        predict_with_mirror_detection,
        analyze_writing_quality,
        index_to_character,
        get_character_type
    )
    from apps.universal_recognizer_web.core.accessibility import format_accessibility_report
    from apps.universal_recognizer_web.core.dataset_tester import get_dataset_tester
except ImportError:
    # Fallback for direct execution
    from core.model_manager import get_model_manager
    from core.predictor import (
        predict_character,
        predict_with_mirror_detection,
        analyze_writing_quality,
        index_to_character,
        get_character_type
    )
    from core.accessibility import format_accessibility_report
    from core.dataset_tester import get_dataset_tester

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Initialize model manager on startup
try:
    model_manager = get_model_manager()
    print("Universal character recognition model loaded successfully!")
except Exception as e:
    print(f"Warning: Failed to load model: {e}")
    model_manager = None


@app.route('/')
def index():
    """Serve the main web page."""
    if model_manager and model_manager.is_loaded():
        model_info = model_manager.get_model_info()
    else:
        model_info = {
            'layer_sizes': [784, 512, 256, 128, 62],
            'total_parameters': 0,
            'accuracy': 0.0
        }
    return render_template('index.html', model_info=model_info)


@app.route('/predict', methods=['POST'])
def predict():
    """Handle standard character prediction requests."""
    try:
        start_time = time.time()
        
        # Get image data from request
        data = request.get_json()
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'error': 'No image data provided'}), 400
        
        if model_manager is None or not model_manager.is_loaded():
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Check if debug is requested
        debug = request.args.get('debug', 'false').lower() == 'true'
        
        # Make prediction with quality metrics and optional debug
        result = predict_character(image_data, return_quality_metrics=True, return_debug=debug)
        
        if result is None:
            return jsonify({'error': 'Failed to process image'}), 400
        
        prediction_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        response = {
            'predicted_character': result['predicted_character'],
            'predicted_index': result['predicted_index'],
            'confidence': result['confidence'],
            'character_type': result['character_type'],
            'predictions': result['predictions'],
            'top_predictions': result['top_predictions'],
            'prediction_time': prediction_time
        }
        
        # Add quality metrics if available (for advanced metrics panel)
        if 'quality_metrics' in result:
            response['quality_metrics'] = result['quality_metrics']
        
        # Add debug images if requested
        if debug and 'debug_images' in result:
            response['debug_images'] = result['debug_images']
        
        return jsonify(response)
    
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


@app.route('/predict/accessibility', methods=['POST'])
def predict_with_accessibility():
    """Handle prediction requests with full accessibility analysis."""
    try:
        start_time = time.time()
        
        # Get image data from request
        data = request.get_json()
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'error': 'No image data provided'}), 400
        
        if model_manager is None or not model_manager.is_loaded():
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Make prediction with mirror detection
        mirror_result = predict_with_mirror_detection(image_data)
        
        if mirror_result is None or mirror_result.get('original') is None:
            return jsonify({'error': 'Failed to process image'}), 400
        
        # Analyze writing quality
        quality_metrics = analyze_writing_quality(image_data)
        
        # Generate accessibility report
        accessibility_report = format_accessibility_report(
            mirror_result['original'],
            mirror_result.get('mirrored'),
            quality_metrics
        )
        
        prediction_time = (time.time() - start_time) * 1000
        
        response = {
            'prediction': mirror_result['original'],
            'mirror_detection': {
                'mirror_detected': mirror_result.get('mirror_detected', False),
                'mirrored_prediction': mirror_result.get('mirrored'),
                'original_prediction': mirror_result['original']
            },
            'quality_metrics': quality_metrics,
            'accessibility': accessibility_report,
            'prediction_time': prediction_time
        }
        
        return jsonify(response)
    
    except Exception as e:
        print(f"Accessibility prediction error: {e}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


@app.route('/api/model/info', methods=['GET'])
def get_model_info():
    """Get detailed model information."""
    try:
        if model_manager is None or not model_manager.is_loaded():
            return jsonify({'error': 'Model not loaded'}), 500
        
        model_info = model_manager.get_model_info()
        
        # Add character mapping info
        model_info['character_mapping'] = {
            'digits': [str(i) for i in range(10)],
            'uppercase': [chr(ord('A') + i) for i in range(26)],
            'lowercase': [chr(ord('a') + i) for i in range(26)]
        }
        
        return jsonify(model_info)
    
    except Exception as e:
        print(f"Model info error: {e}")
        return jsonify({'error': f'Failed to get model info: {str(e)}'}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_manager is not None and model_manager.is_loaded(),
        'model_info': model_manager.get_model_info() if model_manager and model_manager.is_loaded() else None
    })


@app.route('/api/test/random', methods=['GET'])
def get_random_test_samples():
    """Get random test samples from EMNIST dataset."""
    try:
        # Get query parameters
        count = request.args.get('count', '1', type=int)
        character = request.args.get('character', None, type=str)
        
        # Validate count (1, 9, 16, or 25 for grid layouts)
        if count not in [1, 9, 16, 25]:
            count = 1
        
        # Get data directory
        base_dir = os.path.dirname(__file__)
        data_dir = os.path.join(base_dir, 'data')
        
        # Get dataset tester
        tester = get_dataset_tester(data_dir)
        
        # Get random samples
        samples = tester.get_random_samples(count=count, character_filter=character)
        
        return jsonify({
            'samples': samples,
            'count': len(samples)
        })
    
    except Exception as e:
        print(f"Error getting test samples: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Failed to get test samples: {str(e)}'}), 500


@app.route('/api/test/predict', methods=['POST'])
def predict_test_sample():
    """Get prediction for a test sample image."""
    try:
        data = request.get_json()
        image_array = data.get('image_array')
        
        if not image_array:
            return jsonify({'error': 'No image data provided'}), 400
        
        if model_manager is None or not model_manager.is_loaded():
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Convert list to numpy array
        import numpy as np
        image_data = np.array(image_array, dtype=np.float32)
        
        # Make prediction - test images are already in correct orientation
        result = predict_character(image_data, return_quality_metrics=False, is_test_image=True)
        
        if result is None:
            return jsonify({'error': 'Failed to process image'}), 400
        
        return jsonify({
            'prediction': {
                'character': result['predicted_character'],
                'index': result['predicted_index'],
                'confidence': result['confidence'],
                'character_type': result['character_type'],
                'top_predictions': result['top_predictions']
            }
        })
    
    except Exception as e:
        print(f"Error predicting test sample: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


@app.route('/api/characters', methods=['GET'])
def get_characters():
    """Get all supported characters organized by type."""
    return jsonify({
        'digits': [str(i) for i in range(10)],
        'uppercase': [chr(ord('A') + i) for i in range(26)],
        'lowercase': [chr(ord('a') + i) for i in range(26)],
        'total': 62
    })


@app.route('/assets/<path:filename>')
def serve_assets(filename):
    """Serve shared assets from the root assets directory"""
    assets_dir = Path(__file__).parent.parent.parent / 'assets'
    return send_from_directory(assets_dir, filename)


if __name__ == '__main__':
    print("Starting Universal Character Recognition Web Application")
    print("=" * 60)
    
    if model_manager and model_manager.is_loaded():
        info = model_manager.get_model_info()
        print(f"Model: Universal Character Recognizer")
        print(f"  Accuracy: {info['accuracy']:.2f}%")
        print(f"  Classes: {info['classes']} (0-9, A-Z, a-z)")
        print(f"  Architecture: {info['layer_sizes']}")
        print(f"\nStarting web server...")
        print(f"Main app: http://localhost:8003")
        print(f"Model ready for universal character recognition!")
        
        app.run(debug=True, host='0.0.0.0', port=8003)
    else:
        print("ERROR: Model could not be loaded")
        print("Please ensure the model file exists at:")
        print("  apps/universal_recognizer_web/models/universal_character_model.pkl")
        sys.exit(1)

