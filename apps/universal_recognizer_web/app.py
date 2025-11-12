"""
Flask web application for universal character recognition with accessibility features.
"""

import os
import sys
import time
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

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
        
        # Make prediction
        result = predict_character(image_data)
        
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


@app.route('/api/characters', methods=['GET'])
def get_characters():
    """Get all supported characters organized by type."""
    return jsonify({
        'digits': [str(i) for i in range(10)],
        'uppercase': [chr(ord('A') + i) for i in range(26)],
        'lowercase': [chr(ord('a') + i) for i in range(26)],
        'total': 62
    })


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
        print(f"Main app: http://localhost:8000")
        print(f"Model ready for universal character recognition!")
        
        app.run(debug=True, host='0.0.0.0', port=8000)
    else:
        print("ERROR: Model could not be loaded")
        print("Please ensure the model file exists at:")
        print("  apps/universal_recognizer_web/models/universal_character_model.pkl")
        sys.exit(1)

