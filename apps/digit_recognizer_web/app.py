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
import json
import random
import pandas as pd

# Add NeuralEngine to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from nn_core import NeuralNetwork

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Global variables for model
neural_network = None
model_accuracy = 0.0
model_info = {}
test_dataset = None
dataset_samples = []

def load_model(model_name='enhanced_digit_model.pkl'):
    """Load the trained Neural Engine model."""
    global neural_network, model_accuracy, model_info
    model_path = os.path.join('static', 'models', model_name)
    
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
        print(f"   Model: {model_name}")
        print(f"   Architecture: {model_info['architecture']}")
        print(f"   Parameters: {model_info['parameters']:,}")
        print(f"   Accuracy: {model_info['accuracy']:.2f}%")
        return True
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False

def load_test_dataset():
    """Load MNIST test dataset from CSV file."""
    global dataset_samples
    
    try:
        # Load the MNIST CSV file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(script_dir, 'static', 'data', 'mnist_test.csv')
        
        if not os.path.exists(csv_path):
            print(f"⚠️  MNIST CSV file not found at: {csv_path}")
            print("📊 Falling back to synthetic data generation...")
            dataset_samples = generate_synthetic_samples(100)
            return True
        
        print(f"📊 Loading MNIST test dataset from: {csv_path}")
        
        # Read CSV file
        df = pd.read_csv(csv_path)
        print(f"   Dataset shape: {df.shape}")
        
        # Take a subset for performance (first 500 samples)
        subset_size = min(500, len(df))
        df_subset = df.head(subset_size)
        
        dataset_samples = []
        
        for idx, row in df_subset.iterrows():
            # Extract label (first column)
            label = int(row.iloc[0])
            
            # Extract pixel values (remaining 784 columns)
            pixel_values = row.iloc[1:].values.astype(np.uint8)
            
            # Reshape to 28x28 image
            img_array = pixel_values.reshape(28, 28)
            
            # Convert to base64 for web display
            img_base64 = convert_array_to_base64(img_array)
            
            # Create sample entry
            sample = {
                'index': idx,
                'image_data': img_base64,
                'image_array': img_array.tolist(),
                'label': label,
                'metadata': {
                    'source': 'mnist_kaggle',
                    'original_index': idx,
                    'dataset_size': len(df)
                }
            }
            
            dataset_samples.append(sample)
        
        print(f"✅ Loaded {len(dataset_samples)} real MNIST samples")
        print(f"   Label distribution: {get_label_distribution(dataset_samples)}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to load MNIST CSV: {e}")
        print("📊 Falling back to synthetic data generation...")
        dataset_samples = generate_synthetic_samples(100)
        return True

def convert_array_to_base64(img_array):
    """Convert numpy array to base64 image string."""
    try:
        # Ensure proper data type
        if img_array.dtype != np.uint8:
            img_array = img_array.astype(np.uint8)
        
        # Create PIL Image (remove deprecated mode parameter)
        img_pil = Image.fromarray(img_array)  # PIL automatically detects grayscale
        
        # Convert to base64
        buffer = io.BytesIO()
        img_pil.save(buffer, format='PNG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return f'data:image/png;base64,{img_base64}'
        
    except Exception as e:
        print(f"Error converting array to base64: {e}")
        return None

def get_label_distribution(samples):
    """Get distribution of labels in the dataset."""
    labels = [sample['label'] for sample in samples]
    distribution = {}
    for i in range(10):
        distribution[i] = labels.count(i)
    return distribution

def generate_synthetic_samples(num_samples=100):
    """Generate synthetic digit samples for demonstration."""
    samples = []
    
    for i in range(num_samples):
        # Generate random digit
        digit = random.randint(0, 9)
        
        # Create a simple synthetic image
        img_array = np.zeros((28, 28), dtype=np.uint8)
        
        # Add some realistic noise and variation
        noise = np.random.normal(0, 20, (28, 28))
        img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        
        # Create a simple digit-like pattern
        center_x, center_y = 14, 14
        
        # Different patterns for different digits
        if digit == 0:
            # Circle-like pattern
            y, x = np.ogrid[:28, :28]
            mask = (x - center_x)**2 + (y - center_y)**2 <= 64
            img_array[mask] = 255
            inner_mask = (x - center_x)**2 + (y - center_y)**2 <= 25
            img_array[inner_mask] = 0
        elif digit == 1:
            # Vertical line
            img_array[5:23, 12:16] = 255
        elif digit == 2:
            # S-like curve
            img_array[5:10, 8:20] = 255
            img_array[10:15, 15:20] = 255
            img_array[15:20, 8:20] = 255
        else:
            # Generic pattern for other digits
            y, x = np.ogrid[:28, :28]
            mask = (x - center_x)**2 + (y - center_y)**2 <= 49
            img_array[mask] = 255
        
        # Add some random variation
        variation = np.random.uniform(0.7, 1.3)
        img_array = np.clip(img_array * variation, 0, 255).astype(np.uint8)
        
        # Convert to base64
        img_pil = Image.fromarray(img_array)
        buffer = io.BytesIO()
        img_pil.save(buffer, format='PNG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        samples.append({
            'index': i,
            'image_data': f'data:image/png;base64,{img_base64}',
            'image_array': img_array.tolist(),
            'label': digit,
            'metadata': {
                'generated': True,
                'timestamp': time.time()
            }
        })
    
    return samples

# def preprocess_image(image_data):
#     """Preprocess image data for Neural Engine prediction."""
#     try:
#         # Handle both base64 and raw data
#         if isinstance(image_data, str):
#             if image_data.startswith('data:image'):
#                 image_data = image_data.split(',')[1]
#             image_bytes = base64.b64decode(image_data)
#         else:
#             # Handle numpy array input
#             if isinstance(image_data, list):
#                 img_array = np.array(image_data, dtype=np.float32)
#                 img_array = img_array / 255.0 if img_array.max() > 1 else img_array
#                 return img_array.flatten().reshape(1, -1)
        
#         # Convert to PIL Image
#         image = Image.open(io.BytesIO(image_bytes))
        
#         # Convert to grayscale and resize to 28x28
#         image = image.convert('L')
#         image = image.resize((28, 28), Image.Resampling.LANCZOS)
        
#         # Convert to numpy array and normalize
#         img_array = np.array(image)
#         img_array = img_array.astype(np.float32) / 255.0
        
#         # Invert if needed (white background -> black background)
#         if np.mean(img_array) > 0.5:
#             img_array = 1.0 - img_array
        
#         # Flatten for neural network input
#         img_flattened = img_array.flatten().reshape(1, -1)
        
#         return img_flattened
        
#     except Exception as e:
#         print(f"Image preprocessing error: {e}")
#         return None

def preprocess_image(image_data):
    """Preprocess image data for Neural Engine prediction."""
    try:
        # Handle different input types
        if isinstance(image_data, list):
            # Direct array input (from dataset)
            img_array = np.array(image_data, dtype=np.float32)
            # Normalize to 0-1 range
            img_array = img_array / 255.0 if img_array.max() > 1 else img_array
            return img_array.flatten().reshape(1, -1)
        
        elif isinstance(image_data, str):
            # Base64 string input
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
            
            # Convert to PIL Image
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to grayscale and resize to 28x28
            image = image.convert('L')
            image = image.resize((28, 28), Image.Resampling.LANCZOS)
            
            # Convert to numpy array and normalize
            img_array = np.array(image, dtype=np.float32)
            img_array = img_array / 255.0
            
            # For MNIST, pixels are typically white on black
            # If background is white, invert
            if np.mean(img_array) > 0.5:
                img_array = 1.0 - img_array
            
            return img_array.flatten().reshape(1, -1)
        
        else:
            raise ValueError(f"Unsupported image data type: {type(image_data)}")
            
    except Exception as e:
        print(f"Image preprocessing error: {e}")
        return None

@app.route('/')
def index():
    """Serve the main web page."""
    return render_template('index.html', model_info=model_info)

@app.route('/showcase')
def showcase():
    """Serve the dataset showcase page."""
    return render_template('dataset_showcase.html', model_info=model_info)

@app.route('/api/dataset/sample')
def get_dataset_sample():
    """Get a random sample from the test dataset."""
    try:
        if not dataset_samples:
            return jsonify({'error': 'No dataset samples available'}), 404
        
        # Get random sample
        sample = random.choice(dataset_samples)
        
        # Make prediction on this sample
        if neural_network is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Preprocess the sample
        if 'image_array' in sample:
            processed_image = preprocess_image(sample['image_array'])
        else:
            processed_image = preprocess_image(sample['image_data'])
        
        if processed_image is None:
            return jsonify({'error': 'Failed to process sample'}), 400
        
        # Make prediction
        start_time = time.time()
        predictions = neural_network.forward(processed_image)
        predictions = predictions.flatten()
        
        # Normalize predictions
        if predictions.max() > 1.0 or abs(predictions.sum() - 1.0) > 0.01:
            predictions = np.exp(predictions) / np.sum(np.exp(predictions))
        
        prediction_time = (time.time() - start_time) * 1000
        
        # Format response
        predicted_digit = int(np.argmax(predictions))
        confidence = float(predictions[predicted_digit]) * 100
        
        response = {
            'sample': {
                'index': sample['index'],
                'image_data': sample['image_data'],
                'actual_label': sample['label'],
                'metadata': sample.get('metadata', {})
            },
            'prediction': {
                'predicted_digit': predicted_digit,
                'confidence': confidence,
                'predictions': predictions.tolist(),
                'prediction_time': prediction_time,
                'is_correct': predicted_digit == sample['label']
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Dataset sample error: {e}")
        return jsonify({'error': f'Failed to get dataset sample: {str(e)}'}), 500

@app.route('/api/dataset/batch')
def get_dataset_batch():
    """Get a batch of dataset samples."""
    try:
        batch_size = min(int(request.args.get('size', 10)), 50)  # Limit batch size
        
        if not dataset_samples:
            return jsonify({'error': 'No dataset samples available'}), 404
        
        # Get random batch
        batch = random.sample(dataset_samples, min(batch_size, len(dataset_samples)))
        
        # Process batch predictions
        batch_results = []
        for sample in batch:
            if 'image_array' in sample:
                processed_image = preprocess_image(sample['image_array'])
            else:
                processed_image = preprocess_image(sample['image_data'])
            
            if processed_image is not None and neural_network is not None:
                predictions = neural_network.forward(processed_image)
                predictions = predictions.flatten()
                
                # Normalize predictions
                if predictions.max() > 1.0 or abs(predictions.sum() - 1.0) > 0.01:
                    predictions = np.exp(predictions) / np.sum(np.exp(predictions))
                
                predicted_digit = int(np.argmax(predictions))
                confidence = float(predictions[predicted_digit]) * 100
                
                batch_results.append({
                    'sample': sample,
                    'predicted_digit': predicted_digit,
                    'confidence': confidence,
                    'is_correct': predicted_digit == sample['label']
                })
        
        return jsonify({
            'batch': batch_results,
            'batch_size': len(batch_results),
            'accuracy': sum(1 for r in batch_results if r['is_correct']) / len(batch_results) * 100
        })
        
    except Exception as e:
        print(f"Dataset batch error: {e}")
        return jsonify({'error': f'Failed to get dataset batch: {str(e)}'}), 500

@app.route('/api/model/architecture')
def get_model_architecture():
    """Get detailed model architecture information."""
    try:
        if neural_network is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        architecture = {
            'layers': [],
            'total_parameters': neural_network.count_parameters(),
            'input_size': neural_network.layer_sizes[0],
            'output_size': neural_network.layer_sizes[-1],
            'hidden_layers': neural_network.layer_sizes[1:-1],
            'activation_functions': [layer.activation_name for layer in neural_network.layers]
        }
        
        # Add detailed layer information
        for i, layer in enumerate(neural_network.layers):
            layer_info = {
                'index': i,
                'type': 'dense',
                'input_size': layer.input_size,
                'output_size': layer.output_size,
                'activation': layer.activation_name,
                'parameters': layer.input_size * layer.output_size + layer.output_size,
                'weights_shape': [layer.input_size, layer.output_size],
                'bias_shape': [layer.output_size]
            }
            architecture['layers'].append(layer_info)
        
        return jsonify(architecture)
        
    except Exception as e:
        print(f"Architecture error: {e}")
        return jsonify({'error': f'Failed to get architecture: {str(e)}'}), 500

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

@app.route('/api/neural/activations', methods=['POST'])
def get_neural_activations():
    """Get layer-by-layer activations for a specific image."""
    try:
        data = request.get_json()
        image_data = data.get('image_data')
        model_name = data.get('model_name', 'enhanced_digit_model.pkl')
        
        if not image_data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Preprocess image
        processed_image = preprocess_image(image_data)
        if processed_image is None:
            return jsonify({'error': 'Failed to process image'}), 400
        
        if neural_network is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Get activations from each layer
        layer_activations = []
        current_input = processed_image
        
        for i, layer in enumerate(neural_network.layers):
            # Forward pass through this layer
            layer_output = layer.forward(current_input)
            
            # Store activations (limit to first 15 for visualization)
            activations = layer_output.flatten()[:15].tolist()
            layer_activations.append(activations)
            
            current_input = layer_output
        
        return jsonify({
            'layer_activations': layer_activations,
            'final_prediction': int(np.argmax(current_input)),
            'layer_info': [
                {'name': f'Layer {i+1}', 'size': len(acts)} 
                for i, acts in enumerate(layer_activations)
            ]
        })
        
    except Exception as e:
        print(f"Neural activations error: {e}")
        return jsonify({'error': f'Failed to get activations: {str(e)}'}), 500

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
        'model_accuracy': model_accuracy,
        'dataset_samples': len(dataset_samples) if dataset_samples else 0
    })

@app.route('/switch_model', methods=['POST'])
def switch_model():
    """Handle model switching requests."""
    try:
        data = request.get_json()
        model_name = data.get('model_name')
        
        if not model_name:
            return jsonify({'error': 'No model name provided'}), 400
        
        # Update the model path
        model_path = os.path.join('static', 'models', model_name)
        if not os.path.exists(model_path):
            return jsonify({'error': f'Model {model_name} not found'}), 404
        
        # Load the new model
        global neural_network, model_accuracy, model_info
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
        
        print(f"✅ Switched to model: {model_name}")
        print(f"   Architecture: {model_info['architecture']}")
        print(f"   Parameters: {model_info['parameters']:,}")
        print(f"   Accuracy: {model_info['accuracy']:.2f}%")
        
        return jsonify({
            'success': True,
            'model_name': model_name,
            'model_info': model_info
        })
        
    except Exception as e:
        print(f"❌ Model switch error: {e}")
        return jsonify({'error': f'Failed to switch model: {str(e)}'}), 500

if __name__ == '__main__':
    print("🚀 Starting NeuralEngine Web Application")
    print("=" * 50)
    
    # Load model on startup
    if load_model():
        # Load test dataset
        load_test_dataset()
        
        print(f"🌐 Starting web server...")
        print(f"📱 Main app: http://localhost:5000")
        print(f"🎯 Dataset showcase: http://localhost:5000/showcase")
        print(f"🤖 Model ready for digit recognition!")
        
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Failed to start - model could not be loaded")
