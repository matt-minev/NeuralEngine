"""
NeuralEngine digit recognizer web application backend.

Flask backend serving the trained Neural Engine model for web-based digit recognition.
"""

import os
import sys
import pickle
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
import base64
import io
import time
import json
import random
import pandas as pd
import gzip
import struct

# add NeuralEngine to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from nn_core import NeuralNetwork
try:
    from .preprocessing import preprocess_live_payload, preprocess_dataset_image
except ImportError:
    from preprocessing import preprocess_live_payload, preprocess_dataset_image

app = Flask(__name__)
CORS(app)  # enable CORS for cross-origin requests

# global variables for model
neural_network = None
model_accuracy = 0.0
model_info = {}
test_dataset = None
dataset_samples = []
dataset_sources = {
    'mnist': [],
    'emnist_digits': [],
}
current_model_name = 'enhanced_digit_model.pkl'


def load_model(model_name='enhanced_digit_model.pkl'):
    """Load the trained Neural Engine model."""
    global neural_network, model_accuracy, model_info, current_model_name
    # Use absolute path based on script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'static', 'models', model_name)

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
        current_model_name = model_name

        print(f"Neural Engine model loaded succesfully!")
        print(f"  Model: {model_name}")
        print(f"  Architecture: {model_info['architecture']}")
        print(f"  Parameters: {model_info['parameters']:,}")
        print(f"  Accuracy: {model_info['accuracy']:.2f}%")
        return True

    except Exception as e:
        print(f"Failed to load model: {e}")
        return False


def load_mnist_test_dataset():
    """Load MNIST test dataset from the digit recognizer data directory."""
    global dataset_sources

    try:
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        csv_path = os.path.join(repo_root, 'apps', 'digit_recognizer', 'data', 'mnist_test.csv')

        if not os.path.exists(csv_path):
            print(f"MNIST CSV file not found at: {csv_path}")
            return False

        print(f"Loading MNIST test dataset from: {csv_path}")

        # read CSV file
        df = pd.read_csv(csv_path)
        print(f"  Dataset shape: {df.shape}")

        # take a subset for performence (first 500 samples)
        subset_size = min(500, len(df))
        df_subset = df.head(subset_size)

        dataset_samples = []

        for idx, row in df_subset.iterrows():
            # extract label (first column)
            label = int(row.iloc[0])

            # extract pixel values (remaining 784 columns)
            pixel_values = row.iloc[1:].values.astype(np.uint8)

            # reshape to 28x28 image
            img_array = pixel_values.reshape(28, 28)

            # convert to base64 for web display
            img_base64 = convert_array_to_base64(img_array)

            # create sample entry
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

        dataset_sources['mnist'] = dataset_samples
        print(f"Loaded {len(dataset_samples)} real MNIST samples")
        print(f"  Label distribution: {get_label_distribution(dataset_samples)}")
        return True

    except Exception as e:
        print(f"Failed to load MNIST CSV: {e}")
        return False


def load_emnist_digits_test_dataset():
    """Load EMNIST digits test dataset from the extended recognizer data directory."""
    global dataset_sources

    try:
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        data_dir = os.path.join(repo_root, 'apps', 'digit_recognizer_extended', 'data')
        images_path = os.path.join(data_dir, 'emnist-digits-test-images-idx3-ubyte.gz')
        labels_path = os.path.join(data_dir, 'emnist-digits-test-labels-idx1-ubyte.gz')

        if not os.path.exists(images_path) or not os.path.exists(labels_path):
            print("EMNIST digits test files not found.")
            return False

        print(f"Loading EMNIST digits test dataset from: {data_dir}")

        with gzip.open(images_path, 'rb') as image_file:
            magic = struct.unpack('>I', image_file.read(4))[0]
            if magic != 2051:
                raise ValueError(f'Unexpected EMNIST image magic number: {magic}')
            num_images = struct.unpack('>I', image_file.read(4))[0]
            rows = struct.unpack('>I', image_file.read(4))[0]
            cols = struct.unpack('>I', image_file.read(4))[0]
            image_data = np.frombuffer(image_file.read(), dtype=np.uint8).reshape(num_images, rows, cols)

        with gzip.open(labels_path, 'rb') as label_file:
            magic = struct.unpack('>I', label_file.read(4))[0]
            if magic != 2049:
                raise ValueError(f'Unexpected EMNIST label magic number: {magic}')
            num_labels = struct.unpack('>I', label_file.read(4))[0]
            labels = np.frombuffer(label_file.read(), dtype=np.uint8)

        subset_size = min(500, len(labels))
        samples = []

        for idx in range(subset_size):
            # EMNIST requires flip + rotate to match on-screen orientation.
            img_array = np.rot90(np.fliplr(image_data[idx]))
            img_base64 = convert_array_to_base64(img_array)
            samples.append({
                'index': idx,
                'image_data': img_base64,
                'image_array': img_array.tolist(),
                'label': int(labels[idx]),
                'metadata': {
                    'source': 'emnist_digits',
                    'original_index': idx,
                    'dataset_size': int(num_labels),
                }
            })

        dataset_sources['emnist_digits'] = samples
        print(f"Loaded {len(samples)} EMNIST digits samples")
        print(f"  Label distribution: {get_label_distribution(samples)}")
        return True

    except Exception as e:
        print(f"Failed to load EMNIST digits dataset: {e}")
        return False


def get_active_dataset_key(model_name=None):
    """Choose the appropriate dataset for the active model."""
    target_model = model_name or current_model_name
    if target_model == 'enhanced_digit_model.pkl':
        return 'emnist_digits'
    return 'mnist'


def load_test_dataset():
    """Load both supported showcase datasets."""
    global dataset_samples

    mnist_loaded = load_mnist_test_dataset()
    emnist_loaded = load_emnist_digits_test_dataset()

    if not mnist_loaded and not emnist_loaded:
        print("No real datasets loaded. Falling back to synthetic samples.")
        dataset_samples = generate_synthetic_samples(100)
        dataset_sources['mnist'] = dataset_samples
        dataset_sources['emnist_digits'] = dataset_samples
        return True

    fallback_samples = dataset_sources['mnist'] or dataset_sources['emnist_digits']
    if not dataset_sources['mnist']:
        dataset_sources['mnist'] = fallback_samples
    if not dataset_sources['emnist_digits']:
        dataset_sources['emnist_digits'] = fallback_samples

    dataset_samples = dataset_sources[get_active_dataset_key()]
    return True


def convert_array_to_base64(img_array):
    """Convert numpy array to base64 image string."""
    try:
        # ensure proper data type
        if img_array.dtype != np.uint8:
            img_array = img_array.astype(np.uint8)

        # create PIL image (PIL automatically detects grayscale)
        img_pil = Image.fromarray(img_array)

        # convert to base64
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
    """Generate synthetic digit samples for demonstation."""
    samples = []

    for i in range(num_samples):
        # generate random digit
        digit = random.randint(0, 9)

        # create a simple synthetic image
        img_array = np.zeros((28, 28), dtype=np.uint8)

        # add some realistic noise and variation
        noise = np.random.normal(0, 20, (28, 28))
        img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)

        # create a simple digit-like pattern
        center_x, center_y = 14, 14

        # different patterns for different digits
        if digit == 0:
            # circle-like pattern
            y, x = np.ogrid[:28, :28]
            mask = (x - center_x)**2 + (y - center_y)**2 <= 64
            img_array[mask] = 255
            inner_mask = (x - center_x)**2 + (y - center_y)**2 <= 25
            img_array[inner_mask] = 0
        elif digit == 1:
            # vertical line
            img_array[5:23, 12:16] = 255
        elif digit == 2:
            # s-like curve
            img_array[5:10, 8:20] = 255
            img_array[10:15, 15:20] = 255
            img_array[15:20, 8:20] = 255
        else:
            # generic pattern for other digits
            y, x = np.ogrid[:28, :28]
            mask = (x - center_x)**2 + (y - center_y)**2 <= 49
            img_array[mask] = 255

        # add some random variation
        variation = np.random.uniform(0.7, 1.3)
        img_array = np.clip(img_array * variation, 0, 255).astype(np.uint8)

        # convert to base64
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


def preprocess_image(image_data, source='live'):
    """Preprocess image data for Neural Engine prediction."""
    try:
        if source == 'dataset':
            return preprocess_dataset_image(image_data)
        return preprocess_live_payload(image_data)
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


@app.route('/assets/<path:filename>')
def serve_assets(filename):
    """Serve assets from the root assets directory."""
    assets_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'assets', filename)
    if os.path.exists(assets_path):
        return send_from_directory(os.path.dirname(assets_path), filename)
    return "Asset not found", 404


@app.route('/api/dataset/sample')
def get_dataset_sample():
    """Get a random sample from the test dataset."""
    try:
        active_samples = dataset_sources.get(get_active_dataset_key(), dataset_samples)

        if not active_samples:
            return jsonify({'error': 'No dataset samples available'}), 404

        # get random sample
        sample = random.choice(active_samples)

        # make prediction on this sample
        if neural_network is None:
            return jsonify({'error': 'Model not loaded'}), 500

        # preprocess the sample
        if 'image_array' in sample:
            processed_image = preprocess_image(sample['image_array'], source='dataset')
        else:
            processed_image = preprocess_image(sample['image_data'], source='dataset')

        if processed_image is None:
            return jsonify({'error': 'Failed to process sample'}), 400

        # make prediction
        start_time = time.time()
        predictions = neural_network.forward(processed_image)
        predictions = predictions.flatten()

        # normalize predictions
        if predictions.max() > 1.0 or abs(predictions.sum() - 1.0) > 0.01:
            predictions = np.exp(predictions) / np.sum(np.exp(predictions))

        prediction_time = (time.time() - start_time) * 1000

        # format responce
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
        batch_size = min(int(request.args.get('size', 10)), 50)  # limit batch size
        active_samples = dataset_sources.get(get_active_dataset_key(), dataset_samples)

        if not active_samples:
            return jsonify({'error': 'No dataset samples available'}), 404

        # get random batch
        batch = random.sample(active_samples, min(batch_size, len(active_samples)))

        # process batch predictions
        batch_results = []
        for sample in batch:
            if 'image_array' in sample:
                processed_image = preprocess_image(sample['image_array'], source='dataset')
            else:
                processed_image = preprocess_image(sample['image_data'], source='dataset')

            if processed_image is not None and neural_network is not None:
                predictions = neural_network.forward(processed_image)
                predictions = predictions.flatten()

                # normalize predictions
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
    """Get detailed model architecture informaton."""
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

        # add detailed layer information
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

        # get image data from request
        data = request.get_json()
        image_data = data.get('image')

        if not image_data:
            return jsonify({'error': 'No image data provided'}), 400

        # preprocess image
        processed_image = preprocess_image(image_data, source='live')
        if processed_image is None:
            return jsonify({'error': 'Failed to process image'}), 400

        # make prediction using Neural Engine
        if neural_network is None:
            return jsonify({'error': 'Model not loaded'}), 500

        predictions = neural_network.forward(processed_image)
        predictions = predictions.flatten()

        # ensure predictions are properly normalized
        if predictions.max() > 1.0 or abs(predictions.sum() - 1.0) > 0.01:
            predictions = np.exp(predictions) / np.sum(np.exp(predictions))

        prediction_time = (time.time() - start_time) * 1000  # convert to miliseconds

        # format responce
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

        # preprocess image
        processed_image = preprocess_image(image_data, source='live')
        if processed_image is None:
            return jsonify({'error': 'Failed to process image'}), 400

        if neural_network is None:
            return jsonify({'error': 'Model not loaded'}), 500

        # get activations from each layer
        layer_activations = []
        current_input = processed_image

        for i, layer in enumerate(neural_network.layers):
            # forward pass through this layer
            layer_output = layer.forward(current_input)

            # store activations (limit to first 15 for visualisation)
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

        # update the model path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, 'static', 'models', model_name)
        if not os.path.exists(model_path):
            return jsonify({'error': f'Model {model_name} not found'}), 404

        # load the new model
        global neural_network, model_accuracy, model_info, current_model_name, dataset_samples
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
        current_model_name = model_name
        dataset_samples = dataset_sources.get(get_active_dataset_key(model_name), dataset_samples)

        print(f"Switched to model: {model_name}")
        print(f"  Architecture: {model_info['architecture']}")
        print(f"  Parameters: {model_info['parameters']:,}")
        print(f"  Accuracy: {model_info['accuracy']:.2f}%")

        return jsonify({
            'success': True,
            'model_name': model_name,
            'model_info': model_info
        })

    except Exception as e:
        print(f"Model switch error: {e}")
        return jsonify({'error': f'Failed to switch model: {str(e)}'}), 500


if __name__ == '__main__':
    print("Starting NeuralEngine Web Application")
    print("=" * 50)

    # load model on startup
    if load_model():
        # load test dataset
        load_test_dataset()

        print(f"Starting web server...")
        print(f"Main app: http://localhost:8001")
        print(f"Dataset showcase: http://localhost:8001/showcase")
        print(f"Model ready for digit recognition!")

        app.run(debug=True, host='0.0.0.0', port=8001)
    else:
        print("Failed to start - model could not be loaded")
