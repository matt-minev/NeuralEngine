"""
NeuralEngine Digit Recognizer - Web Application Backend
=====================================================

Flask backend serving the trained Neural Engine model for web-based digit recognition.
"""

import os
import sys
import pickle
import numpy as np
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from PIL import Image
import base64
import io
import time

# Add NeuralEngine to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from nn_core import NeuralNetwork

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Global variables for model
neural_network = None
model_accuracy = 0.0
model_info = {}

def load_model():
    """Load the trained Neural Engine model."""
    global neural_network, model_accuracy, model_info
    
    model_path = os.path.join('..', 'digit_recognizer', 'models', 'digit_model_bulletproof.pkl')
    
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        neural_network = model_data['model']
        model_accuracy = model_data.get('accuracy', 0.0)
        model_info = {
            'architecture': neural_network.layer_sizes,
            'parameters': neural_network.count_parameters(),
            'accuracy': model_accuracy,
            'activations': [layer.activation_name for layer in neural_network.layers]
        }
        
        print(f"✅ Neural Engine model loaded successfully!")
        print(f"   Architecture: {model_info['architecture']}")
        print(f"   Parameters: {model_info['parameters']:,}")
        print(f"   Accuracy: {model_info['accuracy']:.2f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False

def preprocess_image(image_data):
    """Preprocess image data for Neural Engine prediction."""
    try:
        # Decode base64 image
        image_data = image_data.split(',')[1]  # Remove data:image/png;base64, prefix
        image_bytes = base64.b64decode(image_data)
        
        # Convert to PIL Image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to grayscale and resize to 28x28
        image = image.convert('L')
        image = image.resize((28, 28), Image.Resampling.LANCZOS)
        
        # Convert to numpy array and normalize
        img_array = np.array(image)
        img_array = img_array.astype(np.float32) / 255.0
        
        # Invert if needed (white background -> black background)
        if np.mean(img_array) > 0.5:
            img_array = 1.0 - img_array
        
        # Flatten for neural network input
        img_flattened = img_array.flatten().reshape(1, -1)
        
        return img_flattened
        
    except Exception as e:
        print(f"Image preprocessing error: {e}")
        return None

@app.route('/')
def index():
    """Serve the main web page."""
    return render_template('index.html', model_info=model_info)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle digit prediction requests."""
    try:
        start_time = time.time()
        
        # Get image data from request
        data = request.get_json()
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Preprocess image
        processed_image = preprocess_image(image_data)
        if processed_image is None:
            return jsonify({'error': 'Failed to process image'}), 400
        
        # Make prediction using Neural Engine
        if neural_network is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        predictions = neural_network.forward(processed_image)
        predictions = predictions.flatten()
        
        # Ensure predictions are properly normalized
        if predictions.max() > 1.0 or abs(predictions.sum() - 1.0) > 0.01:
            predictions = np.exp(predictions) / np.sum(np.exp(predictions))
        
        prediction_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        # Format response
        predicted_digit = int(np.argmax(predictions))
        confidence = float(predictions[predicted_digit]) * 100
        
        response = {
            'predicted_digit': predicted_digit,
            'confidence': confidence,
            'predictions': predictions.tolist(),
            'prediction_time': prediction_time
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/model_info')
def get_model_info():
    """Get information about the loaded model."""
    return jsonify(model_info)

@app.route('/health')
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': neural_network is not None,
        'model_accuracy': model_accuracy
    })

if __name__ == '__main__':
    print("🚀 Starting NeuralEngine Web Application")
    print("=" * 50)
    
    # Load model on startup
    if load_model():
        print(f"🌐 Starting web server...")
        print(f"📱 Access your app at: http://localhost:5000")
        print(f"🎯 Model ready for digit recognition!")
        
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Failed to start - model could not be loaded")
