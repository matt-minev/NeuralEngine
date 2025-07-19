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
neural_engine_root = current_dir.parent.parent  # Go up two levels
sys.path.insert(0, str(neural_engine_root))

# Import application components
try:
    from core.data_processor import QuadraticDataProcessor
    from core.predictor import QuadraticPredictor
    from config.scenarios import get_default_scenarios
    from core.helpers import format_number, assess_performance, get_confidence_level
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
    """Main application page with optional dataset loading"""
    load_dataset = request.args.get('load_dataset')
    return render_template('index.html', load_dataset=load_dataset)

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
        learning_rate = data.get('learning_rate', 0.001)
        
        if not selected_scenarios:
            return jsonify({'error': 'No scenarios selected'}), 400
        
        # Start training in background thread
        training_thread = threading.Thread(
            target=_train_models_background,
            args=(selected_scenarios, epochs, learning_rate)
        )
        training_thread.daemon = True
        training_thread.start()
        
        return jsonify({
            'success': True,
            'message': 'Training started',
            'scenarios': selected_scenarios,
            'epochs': epochs,
            'learning_rate': learning_rate
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

@app.route('/api/charts/enhanced-data')
def get_enhanced_chart_data():
    """Get enhanced chart data with improved styling"""
    if not app_state['results']:
        return jsonify({'error': 'No results available'}), 400
    
    chart_data = {
        'colors': {
            'primary': '#007aff',
            'success': '#34c759',
            'warning': '#ff9500',
            'error': '#ff3b30'
        },
        'scenarios': list(app_state['results'].keys()),
        'enhanced_metrics': {}
    }
    
    for scenario_key in app_state['results'].keys():
        result = app_state['results'][scenario_key]
        chart_data['enhanced_metrics'][scenario_key] = {
            'r2': result.get('r2', 0),
            'mse': result.get('mse', 0),
            'mae': result.get('mae', 0),
            'accuracy': result.get('accuracy_10pct', 0),
            'scenario_name': app_state['scenarios'][scenario_key].name
        }
    
    return jsonify(chart_data)

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

@app.route('/dataset-generator')
def dataset_generator():
    """Dataset generator page"""
    return render_template('dataset_generator.html')

@app.route('/api/generate-dataset', methods=['POST'])
def generate_dataset():
    """Generate quadratic equation dataset"""
    try:
        data = request.get_json()
        
        # Extract parameters
        equation_type = data.get('equation_type', 'school_grade')
        num_equations = data.get('num_equations', 1000)
        coefficient_range = data.get('coefficient_range', {'min': -10, 'max': 10})
        root_type = data.get('root_type', 'mixed')  # integers, fractions, mixed
        allow_complex = data.get('allow_complex', False)
        
        # Generate dataset
        dataset = generate_quadratic_dataset(
            equation_type=equation_type,
            num_equations=num_equations,
            coefficient_range=coefficient_range,
            root_type=root_type,
            allow_complex=allow_complex
        )
        
        # Create filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"quadratic_dataset_{equation_type}_{timestamp}.csv"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Save as CSV
        df = pd.DataFrame(dataset, columns=['a', 'b', 'c', 'x1', 'x2'])
        df.to_csv(filepath, index=False)
        
        # Get statistics
        stats = calculate_dataset_stats(df)
        
        return jsonify({
            'success': True,
            'message': f'Generated {len(dataset)} quadratic equations',
            'filename': filename,
            'stats': stats,
            'preview': dataset[:10].tolist()  # First 10 equations for preview
        })
        
    except Exception as e:
        return jsonify({'error': f'Dataset generation failed: {str(e)}'}), 500

def generate_quadratic_dataset(equation_type, num_equations, coefficient_range, root_type, allow_complex):
    """Generate quadratic equation dataset with specified parameters"""
    dataset = []
    
    for _ in range(num_equations):
        if equation_type == 'school_grade':
            equation = generate_school_grade_equation(coefficient_range, root_type)
        elif equation_type == 'random':
            equation = generate_random_equation(coefficient_range, allow_complex)
        elif equation_type == 'integer_solutions':
            equation = generate_integer_solution_equation(coefficient_range)
        elif equation_type == 'fractional_solutions':
            equation = generate_fractional_solution_equation(coefficient_range)
        else:
            equation = generate_school_grade_equation(coefficient_range, root_type)
        
        dataset.append(equation)
    
    return np.array(dataset)

def generate_school_grade_equation(coefficient_range, root_type):
    """Generate school-grade quadratic equations with nice solutions"""
    # Define nice root values
    integer_roots = list(range(-10, 11))
    fractional_roots = [-3/2, -1/2, -1/3, -1/4, 1/4, 1/3, 1/2, 3/2, 2/3, 3/4, 5/4, 4/3, 5/3, 7/4]
    
    if root_type == 'integers':
        possible_roots = integer_roots
    elif root_type == 'fractions':
        possible_roots = fractional_roots
    else:  # mixed
        possible_roots = integer_roots + fractional_roots
    
    # Remove zero to avoid degenerate cases
    possible_roots = [r for r in possible_roots if r != 0]
    
    # Choose two roots
    x1 = np.random.choice(possible_roots)
    x2 = np.random.choice(possible_roots)
    
    # Generate coefficients using Vieta's formulas
    # For ax² + bx + c = 0 with roots x1, x2:
    # x1 + x2 = -b/a
    # x1 * x2 = c/a
    
    # Choose 'a' coefficient (non-zero)
    a_candidates = list(range(coefficient_range['min'], coefficient_range['max'] + 1))
    a_candidates = [a for a in a_candidates if a != 0]
    a = np.random.choice(a_candidates)
    
    # Calculate b and c to ensure integer coefficients when possible
    sum_roots = x1 + x2
    product_roots = x1 * x2
    
    # Find LCM to make coefficients integers
    from fractions import Fraction
    sum_frac = Fraction(sum_roots).limit_denominator()
    prod_frac = Fraction(product_roots).limit_denominator()
    
    # Scale 'a' to make b and c integers
    lcm_denom = np.lcm(sum_frac.denominator, prod_frac.denominator)
    a *= lcm_denom
    
    b = -a * sum_roots
    c = a * product_roots
    
    # Ensure coefficients are in range by scaling if necessary
    max_coeff = max(abs(a), abs(b), abs(c))
    max_allowed = max(abs(coefficient_range['min']), abs(coefficient_range['max']))
    
    if max_coeff > max_allowed:
        scale_factor = max_allowed / max_coeff
        a *= scale_factor
        b *= scale_factor
        c *= scale_factor
    
    # Round to remove floating point errors
    a = round(a)
    b = round(b)
    c = round(c)
    
    # Ensure 'a' is not zero
    if a == 0:
        a = 1
    
    # Recalculate roots with final coefficients to ensure accuracy
    discriminant = b**2 - 4*a*c
    if discriminant >= 0:
        sqrt_disc = np.sqrt(discriminant)
        x1_calc = (-b + sqrt_disc) / (2*a)
        x2_calc = (-b - sqrt_disc) / (2*a)
    else:
        # Fallback to complex roots (shouldn't happen with our method)
        sqrt_disc = np.sqrt(-discriminant)
        x1_calc = (-b + 1j*sqrt_disc) / (2*a)
        x2_calc = (-b - 1j*sqrt_disc) / (2*a)
        x1_calc = x1_calc.real  # Take real part for dataset
        x2_calc = x2_calc.real
    
    return [float(a), float(b), float(c), float(x1_calc), float(x2_calc)]

def generate_random_equation(coefficient_range, allow_complex):
    """Generate random quadratic equations"""
    a = np.random.randint(coefficient_range['min'], coefficient_range['max'] + 1)
    while a == 0:  # Ensure quadratic
        a = np.random.randint(coefficient_range['min'], coefficient_range['max'] + 1)
    
    b = np.random.randint(coefficient_range['min'], coefficient_range['max'] + 1)
    c = np.random.randint(coefficient_range['min'], coefficient_range['max'] + 1)
    
    # Calculate roots
    discriminant = b**2 - 4*a*c
    if discriminant >= 0 or allow_complex:
        if discriminant >= 0:
            sqrt_disc = np.sqrt(discriminant)
            x1 = (-b + sqrt_disc) / (2*a)
            x2 = (-b - sqrt_disc) / (2*a)
        else:
            sqrt_disc = np.sqrt(-discriminant)
            x1 = (-b + 1j*sqrt_disc) / (2*a)
            x2 = (-b - 1j*sqrt_disc) / (2*a)
            x1 = x1.real  # Store real part only
            x2 = x2.real
        
        return [float(a), float(b), float(c), float(x1), float(x2)]
    else:
        # Regenerate if complex roots not allowed
        return generate_random_equation(coefficient_range, allow_complex)

def generate_integer_solution_equation(coefficient_range):
    """Generate equations with integer solutions only"""
    # Choose integer roots
    roots = list(range(-20, 21))
    x1 = np.random.choice(roots)
    x2 = np.random.choice(roots)
    
    # Generate coefficients
    a = np.random.randint(1, 6)  # Keep 'a' small for integer coefficients
    b = -a * (x1 + x2)
    c = a * x1 * x2
    
    return [float(a), float(b), float(c), float(x1), float(x2)]

def generate_fractional_solution_equation(coefficient_range):
    """Generate equations with fractional solutions"""
    from fractions import Fraction
    
    # Choose fractional roots
    numerators = list(range(-10, 11))
    denominators = [2, 3, 4, 5, 6]
    
    x1 = Fraction(np.random.choice(numerators), np.random.choice(denominators))
    x2 = Fraction(np.random.choice(numerators), np.random.choice(denominators))
    
    # Generate integer coefficients
    a = np.random.randint(1, 6)
    sum_roots = x1 + x2
    product_roots = x1 * x2
    
    # Scale to get integer coefficients
    lcm_denom = np.lcm(sum_roots.denominator, product_roots.denominator)
    a *= lcm_denom
    
    b = -a * sum_roots
    c = a * product_roots
    
    return [float(a), float(b), float(c), float(x1), float(x2)]

def calculate_dataset_stats(df):
    """Calculate dataset statistics"""
    stats = {
        'total_equations': len(df),
        'coefficients': {
            'a': {'min': float(df['a'].min()), 'max': float(df['a'].max()), 'mean': float(df['a'].mean())},
            'b': {'min': float(df['b'].min()), 'max': float(df['b'].max()), 'mean': float(df['b'].mean())},
            'c': {'min': float(df['c'].min()), 'max': float(df['c'].max()), 'mean': float(df['c'].mean())}
        },
        'roots': {
            'x1': {'min': float(df['x1'].min()), 'max': float(df['x1'].max()), 'mean': float(df['x1'].mean())},
            'x2': {'min': float(df['x2'].min()), 'max': float(df['x2'].max()), 'mean': float(df['x2'].mean())}
        },
        'quality_metrics': {
            'integer_roots_x1': int(sum(abs(df['x1'] - df['x1'].round()) < 1e-6)),
            'integer_roots_x2': int(sum(abs(df['x2'] - df['x2'].round()) < 1e-6)),
            'integer_coefficients': int(sum((abs(df['a'] - df['a'].round()) < 1e-6) & 
                                         (abs(df['b'] - df['b'].round()) < 1e-6) & 
                                         (abs(df['c'] - df['c'].round()) < 1e-6)))
        }
    }
    return stats

@app.route('/api/download-dataset/<filename>')
def download_dataset(filename):
    """Download generated dataset"""
    try:
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)
    except FileNotFoundError:
        return jsonify({'error': 'File not found'}), 404

@app.route('/api/data/load/<filename>', methods=['POST'])
def load_specific_dataset(filename):
    """Load a specific dataset by filename"""
    try:
        # Security check - only allow files from upload folder
        if '..' in filename or '/' in filename:
            return jsonify({'error': 'Invalid filename'}), 400
        
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Check if file exists
        if not os.path.exists(filepath):
            return jsonify({'error': 'Dataset file not found'}), 404
        
        # Load data using existing data processor
        if app_state['data_processor'].load_data(filepath):
            stats = app_state['data_processor'].get_stats()
            sample_data = app_state['data_processor'].get_sample_data(100)
            
            return jsonify({
                'success': True,
                'message': f'Successfully loaded dataset: {filename}',
                'filename': filename,
                'total_equations': len(app_state['data_processor'].data),
                'stats': stats,
                'sample_data': sample_data.tolist() if sample_data.size > 0 else [],
                'auto_loaded': True
            })
        else:
            return jsonify({'error': 'Failed to load dataset'}), 400
            
    except Exception as e:
        return jsonify({'error': f'Dataset loading failed: {str(e)}'}), 500

@app.route('/api/data/clear', methods=['POST'])
def clear_dataset():
    """Clear the currently loaded dataset"""
    try:
        # Clear the data processor
        app_state['data_processor'].data = None
        
        # Clear any related state
        app_state['predictors'].clear()
        app_state['results'].clear()
        
        # Stop any ongoing training
        app_state['training_status']['is_training'] = False
        app_state['training_status']['current_scenario'] = None
        app_state['training_status']['progress'] = 0
        app_state['training_status']['logs'].clear()
        
        return jsonify({
            'success': True,
            'message': 'Dataset cleared successfully'
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to clear dataset: {str(e)}'}), 500

def _train_models_background(selected_scenarios, epochs, learning_rate):
    """Background training function"""
    try:
        app_state['training_status']['is_training'] = True
        app_state['training_status']['logs'] = []
        app_state['predictors'].clear()
        app_state['results'].clear()
        
        _log_training('🎯 Starting training session...')
        _log_training(f'Selected scenarios: {len(selected_scenarios)}')
        _log_training(f'Training parameters: {epochs} epochs, learning rate: {learning_rate}')
        
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
                training_results = predictor.train(epochs=epochs, learning_rate=learning_rate, verbose=False)
                
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

def validate_dataset_file(filepath):
    """Validate that a dataset file is properly formatted"""
    try:
        df = pd.read_csv(filepath)
        
        # Check required columns
        required_columns = ['a', 'b', 'c', 'x1', 'x2']
        if not all(col in df.columns for col in required_columns):
            return False, f"Missing required columns. Expected: {required_columns}"
        
        # Check data types
        for col in required_columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                return False, f"Column '{col}' must be numeric"
        
        # Check for empty data
        if len(df) == 0:
            return False, "Dataset is empty"
        
        return True, "Valid dataset"
        
    except Exception as e:
        return False, f"File validation error: {str(e)}"

def cleanup_old_uploads():
    """Clean up old uploaded files on startup"""
    try:
        from core.cleanup import cleanup_old_datasets
        deleted_count, total_size = cleanup_old_datasets('uploads', max_age_days=1)
        if deleted_count > 0:
            print(f"🧹 Cleaned up {deleted_count} old dataset files ({total_size:,} bytes)")
    except Exception as e:
        print(f"⚠️ Cleanup failed: {e}")

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
    
    # Clean up old files on startup
    cleanup_old_uploads()
    
    print("Starting Flask server...")
    print("Location: Varna, Bulgaria 🇧🇬")
    print("Access the app at: http://localhost:5000")
    print("=" * 50)
    
    try:
        app.run(debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n" + "=" * 50)
        print("🛑 Server shutdown requested by user")
        print("🧹 Performing cleanup...")
        
        # Stop any ongoing training
        if app_state['training_status']['is_training']:
            app_state['training_status']['is_training'] = False
            print("   ⏹️  Stopped ongoing training sessions")
        
        # Clear predictors and results
        app_state['predictors'].clear()
        app_state['results'].clear()
        print("   🗑️  Cleared cached models and results")
        
        print("✅ Cleanup completed")
        print("👋 Thanks for using Quadratic Neural Network!")
        print("=" * 50)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("🔍 Check the logs for more details")
    finally:
        print("🏁 Application terminated")
