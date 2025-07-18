#!/usr/bin/env python3
"""
Quadratic Neural Network Web Application
Flask Backend API

Author: Matt
Location: Varna, Bulgaria
Date: 2025

Beautiful Apple-like web interface for quadratic neural network analysis
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import sys
import json
import traceback
from datetime import datetime
import numpy as np
import pandas as pd
from pathlib import Path
import threading
import time

# Add the Neural Engine path
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

# Import application components
try:
    from core.data_processor import QuadraticDataProcessor
    from core.predictor import QuadraticPredictor
    from config.scenarios import get_default_scenarios
    from helpers import format_number, assess_performance, get_confidence_level
except ImportError as e:
    print(f"Error importing components: {e}")
    print("Please ensure all required modules are available.")
    sys.exit(1)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'quadratic-neural-network-varna-2025'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
CORS(app)

# Global application state
app_state = {
    'data_processor': QuadraticDataProcessor(verbose=True),
    'scenarios': get_default_scenarios(),
    'predictors': {},
    'results': {},
    'training_status': {
        'is_training': False,
        'progress': 0,
        'current_scenario': None,
        'logs': []
    }
}

# Ensure upload directory exists
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Routes
@app.route('/')
def index():
    """Main application page"""
    return render_template('index.html')

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'location': 'Varna, Bulgaria'
    })

@app.route('/api/scenarios')
def get_scenarios():
    """Get available prediction scenarios"""
    scenarios_data = {}
    for key, scenario in app_state['scenarios'].items():
        scenarios_data[key] = {
            'name': scenario.name,
            'description': scenario.description,
            'input_features': scenario.input_features,
            'target_features': scenario.target_features,
            'network_architecture': scenario.network_architecture,
            'color': scenario.color
        }
    return jsonify(scenarios_data)

@app.route('/api/data/upload', methods=['POST'])
def upload_data():
    """Upload and process dataset"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not file.filename.endswith('.csv'):
            return jsonify({'error': 'Only CSV files are supported'}), 400
        
        # Save uploaded file
        filename = f"dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Load data
        if app_state['data_processor'].load_data(filepath):
            stats = app_state['data_processor'].get_stats()
            sample_data = app_state['data_processor'].get_sample_data(100)
            
            return jsonify({
                'success': True,
                'message': f'Successfully loaded {len(app_state["data_processor"].data)} equations',
                'filename': filename,
                'stats': stats,
                'sample_data': sample_data.tolist() if sample_data.size > 0 else []
            })
        else:
            return jsonify({'error': 'Failed to load dataset'}), 400
            
    except Exception as e:
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/api/data/info')
def get_data_info():
    """Get current dataset information"""
    if app_state['data_processor'].data is None:
        return jsonify({'loaded': False})
    
    stats = app_state['data_processor'].get_stats()
    sample_data = app_state['data_processor'].get_sample_data(100)
    
    return jsonify({
        'loaded': True,
        'total_equations': len(app_state['data_processor'].data),
        'stats': stats,
        'sample_data': sample_data.tolist() if sample_data.size > 0 else []
    })

@app.route('/api/training/start', methods=['POST'])
def start_training():
    """Start training selected models"""
    try:
        if app_state['data_processor'].data is None:
            return jsonify({'error': 'No dataset loaded'}), 400
        
        if app_state['training_status']['is_training']:
            return jsonify({'error': 'Training already in progress'}), 400
        
        data = request.get_json()
        selected_scenarios = data.get('scenarios', [])
        epochs = data.get('epochs', 1000)
        
        if not selected_scenarios:
            return jsonify({'error': 'No scenarios selected'}), 400
        
        # Start training in background thread
        training_thread = threading.Thread(
            target=_train_models_background,
            args=(selected_scenarios, epochs)
        )
        training_thread.daemon = True
        training_thread.start()
        
        return jsonify({
            'success': True,
            'message': 'Training started',
            'scenarios': selected_scenarios,
            'epochs': epochs
        })
        
    except Exception as e:
        return jsonify({'error': f'Training start failed: {str(e)}'}), 500

@app.route('/api/training/status')
def get_training_status():
    """Get current training status"""
    return jsonify(app_state['training_status'])

@app.route('/api/training/stop', methods=['POST'])
def stop_training():
    """Stop current training"""
    app_state['training_status']['is_training'] = False
    return jsonify({'success': True, 'message': 'Training stopped'})

@app.route('/api/predict', methods=['POST'])
def make_prediction():
    """Make prediction using trained model"""
    try:
        data = request.get_json()
        scenario_key = data.get('scenario')
        input_values = data.get('inputs', [])
        
        if scenario_key not in app_state['predictors']:
            return jsonify({'error': f'Model for scenario not trained'}), 400
        
        if not input_values:
            return jsonify({'error': 'No input values provided'}), 400
        
        # Make prediction
        predictor = app_state['predictors'][scenario_key]
        input_array = np.array(input_values).reshape(1, -1)
        predictions, confidences = predictor.predict(input_array, return_confidence=True)
        
        # Calculate actual solutions if possible
        actual_solutions = None
        if scenario_key == 'coeff_to_roots':
            actual_solutions = _calculate_quadratic_solutions(input_values)
        
        return jsonify({
            'success': True,
            'predictions': predictions[0].tolist(),
            'confidences': confidences[0].tolist(),
            'actual_solutions': actual_solutions,
            'scenario': app_state['scenarios'][scenario_key].name,
            'target_features': app_state['scenarios'][scenario_key].target_features
        })
        
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/api/results')
def get_results():
    """Get training results for all models"""
    results_data = {}
    
    for scenario_key, result in app_state['results'].items():
        scenario = app_state['scenarios'][scenario_key]
        results_data[scenario_key] = {
            'scenario_info': {
                'name': scenario.name,
                'description': scenario.description,
                'color': scenario.color
            },
            'metrics': result
        }
    
    return jsonify(results_data)

@app.route('/api/analysis/performance')
def get_performance_analysis():
    """Get performance analysis data"""
    if not app_state['results']:
        return jsonify({'error': 'No results available'}), 400
    
    analysis_data = {
        'scenarios': list(app_state['results'].keys()),
        'metrics': {
            'r2_scores': [app_state['results'][s]['r2'] for s in app_state['results'].keys()],
            'mse_values': [app_state['results'][s]['mse'] for s in app_state['results'].keys()],
            'mae_values': [app_state['results'][s]['mae'] for s in app_state['results'].keys()],
            'accuracy_values': [app_state['results'][s]['accuracy_10pct'] for s in app_state['results'].keys()]
        },
        'colors': [app_state['scenarios'][s].color for s in app_state['results'].keys()],
        'scenario_names': [app_state['scenarios'][s].name for s in app_state['results'].keys()]
    }
    
    return jsonify(analysis_data)

@app.route('/api/data/random', methods=['GET'])
def get_random_data():
    """Get random sample from dataset for testing"""
    try:
        # Check if data is loaded using the correct state variable
        if app_state['data_processor'].data is None:
            return jsonify({'success': False, 'error': 'No dataset loaded'})
        
        data = app_state['data_processor'].data
        
        # Handle both numpy arrays and pandas DataFrames
        if isinstance(data, np.ndarray):
            # Use numpy random selection for numpy arrays
            random_index = np.random.randint(0, len(data))
            sample_row = data[random_index]
            
            # Convert to dictionary with proper column names
            # Assuming standard quadratic dataset columns: a, b, c, x1, x2
            column_names = ['a', 'b', 'c', 'x1', 'x2']
            sample_dict = {col: float(sample_row[i]) for i, col in enumerate(column_names)}
            
        elif hasattr(data, 'sample'):
            # Use pandas sample method for DataFrames
            sample = data.sample(n=1).iloc[0]
            sample_dict = sample.to_dict()
            
        else:
            # Convert to DataFrame first, then sample
            df = pd.DataFrame(data, columns=['a', 'b', 'c', 'x1', 'x2'])
            sample = df.sample(n=1).iloc[0]
            sample_dict = sample.to_dict()
        
        return jsonify({
            'success': True,
            'data': sample_dict
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

def _train_models_background(selected_scenarios, epochs):
    """Background training function"""
    try:
        app_state['training_status']['is_training'] = True
        app_state['training_status']['logs'] = []
        app_state['predictors'].clear()
        app_state['results'].clear()
        
        _log_training('🎯 Starting training session...')
        _log_training(f'Selected scenarios: {len(selected_scenarios)}')
        
        for i, scenario_key in enumerate(selected_scenarios):
            if not app_state['training_status']['is_training']:
                break
            
            scenario = app_state['scenarios'][scenario_key]
            app_state['training_status']['current_scenario'] = scenario.name
            app_state['training_status']['progress'] = (i / len(selected_scenarios)) * 100
            
            _log_training(f'[{i+1}/{len(selected_scenarios)}] Training {scenario.name}...')
            
            try:
                # Create and train predictor
                predictor = QuadraticPredictor(scenario, app_state['data_processor'])
                training_results = predictor.train(epochs=epochs, verbose=False)
                
                # Store results
                app_state['predictors'][scenario_key] = predictor
                app_state['results'][scenario_key] = training_results['test_results']
                
                # Log success
                test_r2 = training_results['test_results']['r2']
                training_time = training_results['training_time']
                _log_training(f'✅ Completed! R²: {test_r2:.4f}, Time: {training_time:.2f}s')
                
            except Exception as e:
                _log_training(f'❌ Failed: {str(e)}')
        
        app_state['training_status']['progress'] = 100
        _log_training('🎉 Training session completed!')
        
    except Exception as e:
        _log_training(f'❌ Training error: {str(e)}')
    finally:
        app_state['training_status']['is_training'] = False
        app_state['training_status']['current_scenario'] = None

def _log_training(message):
    """Add message to training logs"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_entry = {
        'timestamp': timestamp,
        'message': message
    }
    app_state['training_status']['logs'].append(log_entry)
    
    # Keep only last 100 log entries
    if len(app_state['training_status']['logs']) > 100:
        app_state['training_status']['logs'] = app_state['training_status']['logs'][-100:]

def _calculate_quadratic_solutions(inputs):
    """Calculate actual quadratic solutions"""
    try:
        a, b, c = inputs
        
        if abs(a) < 1e-10:
            if abs(b) < 1e-10:
                return None
            else:
                root = -c / b
                return [root, root]
        
        discriminant = b**2 - 4*a*c
        if discriminant < 0:
            return None
        
        sqrt_discriminant = np.sqrt(discriminant)
        x1 = (-b + sqrt_discriminant) / (2*a)
        x2 = (-b - sqrt_discriminant) / (2*a)
        
        return [x1, x2]
        
    except Exception:
        return None

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("🚀 Quadratic Neural Network Web Application")
    print("=" * 50)
    print("Starting Flask server...")
    print("Location: Varna, Bulgaria 🇧🇬")
    print("Access the app at: http://localhost:5000")
    print("=" * 50)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
