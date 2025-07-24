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
    from core.model_manager import ModelManager
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

# Initialize Model Manager
model_manager = ModelManager()

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
            return jsonify({'error': f'Model for scenario "{scenario_key}" not trained'}), 400
        if not input_values:
            return jsonify({'error': 'No input values provided'}), 400

        # Make prediction
        predictor = app_state['predictors'][scenario_key]
        input_array = np.array(input_values).reshape(1, -1)
        predictions, confidences = predictor.predict(input_array, return_confidence=True)

        # NEW: Get detailed analysis for the frontend
        prediction_details = _get_prediction_details(
            scenario_key, 
            input_values, 
            predictions[0], 
            app_state['scenarios'][scenario_key]
        )

        return jsonify({
            'success': True,
            'predictions': predictions[0].tolist(),
            'confidences': confidences[0].tolist() if confidences is not None else [],
            'scenario': app_state['scenarios'][scenario_key].name,
            'target_features': app_state['scenarios'][scenario_key].target_features,
            'details': prediction_details  # NEW structured data
        })
    except Exception as e:
        traceback.print_exc()
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

# Global state for infinite generation
infinite_generation_state = {
    'active': False,
    'generated_count': 0,
    'current_range': {'min': -2, 'max': 2},
    'batch_size': 100,
    'dataset_buffer': [],
    'config': None
}

@app.route('/api/generate-dataset-infinite/start', methods=['POST'])
def start_infinite_generation():
    """Start infinite dataset generation"""
    try:
        if infinite_generation_state['active']:
            return jsonify({'error': 'Infinite generation already active'}), 400
            
        data = request.get_json()
        
        # Reset state
        infinite_generation_state.update({
            'active': True,
            'generated_count': 0,
            'current_range': {'min': -2, 'max': 2},  # Start small
            'dataset_buffer': [],
            'config': {
                'equation_type': data.get('equation_type', 'school_grade'),
                'root_type': data.get('root_type', 'mixed'),
                'allow_complex': data.get('allow_complex', False)
            }
        })
        
        return jsonify({
            'success': True,
            'message': 'Infinite generation started',
            'initial_range': infinite_generation_state['current_range']
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to start infinite generation: {str(e)}'}), 500

@app.route('/api/generate-dataset-infinite/batch', methods=['POST'])
def generate_infinite_batch():
    """Generate next batch of equations for infinite mode"""
    try:
        if not infinite_generation_state['active']:
            return jsonify({'error': 'Infinite generation not active'}), 400
        
        # Generate batch
        config = infinite_generation_state['config']
        current_range = infinite_generation_state['current_range']
        
        batch_data = []
        for _ in range(infinite_generation_state['batch_size']):
            if config['equation_type'] == 'school_grade':
                equation = generate_school_grade_equation(current_range, config['root_type'])
            elif config['equation_type'] == 'random':
                equation = generate_random_equation(current_range, config['allow_complex'])
            elif config['equation_type'] == 'integer_solutions':
                equation = generate_integer_solution_equation(current_range)
            elif config['equation_type'] == 'fractional_solutions':
                equation = generate_fractional_solution_equation(current_range)
            else:
                equation = generate_school_grade_equation(current_range, config['root_type'])
            
            batch_data.append(equation)
        
        # Add to buffer - KEEP ALL EQUATIONS FOR SAVING
        infinite_generation_state['dataset_buffer'].extend(batch_data)
        infinite_generation_state['generated_count'] += len(batch_data)
        
        # Gradually expand range every 500 equations
        if infinite_generation_state['generated_count'] % 500 == 0:
            expansion = infinite_generation_state['generated_count'] // 500
            new_min = max(-50, -2 - expansion)
            new_max = min(50, 2 + expansion)
            infinite_generation_state['current_range'] = {'min': new_min, 'max': new_max}
        
        # FIXED: Get latest 10 for preview WITHOUT truncating the full dataset
        preview_data = infinite_generation_state['dataset_buffer'][-10:] if infinite_generation_state['dataset_buffer'] else []
        
        return jsonify({
            'success': True,
            'generated_count': infinite_generation_state['generated_count'],
            'current_range': infinite_generation_state['current_range'],
            'batch_size': len(batch_data),
            'preview': preview_data,
            'total_in_buffer': len(infinite_generation_state['dataset_buffer'])  # Added for debugging
        })
        
    except Exception as e:
        return jsonify({'error': f'Batch generation failed: {str(e)}'}), 500

@app.route('/api/generate-dataset-infinite/status')
def get_infinite_status():
    """Get current infinite generation status"""
    return jsonify({
        'active': infinite_generation_state['active'],
        'generated_count': infinite_generation_state['generated_count'],
        'current_range': infinite_generation_state['current_range'],
        'preview': infinite_generation_state['dataset_buffer'][-10:] if infinite_generation_state['dataset_buffer'] else []
    })

@app.route('/api/generate-dataset-infinite/stop', methods=['POST'])
def stop_infinite_generation():
    """Stop infinite generation and save dataset"""
    try:
        if not infinite_generation_state['active']:
            return jsonify({'error': 'No infinite generation active'}), 400
        
        # Mark as inactive
        infinite_generation_state['active'] = False
        
        # Save final dataset
        if infinite_generation_state['dataset_buffer']:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"quadratic_infinite_{infinite_generation_state['config']['equation_type']}_{timestamp}.csv"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # ADDED: Verify we have all equations before saving
            buffer_size = len(infinite_generation_state['dataset_buffer'])
            generated_count = infinite_generation_state['generated_count']
            
            print(f"DEBUG: Stopping infinite generation")
            print(f"DEBUG: Generated count: {generated_count}")
            print(f"DEBUG: Buffer size: {buffer_size}")
            print(f"DEBUG: Saving {buffer_size} equations to {filename}")
            
            # Create DataFrame and save
            df = pd.DataFrame(infinite_generation_state['dataset_buffer'], columns=['a', 'b', 'c', 'x1', 'x2'])
            df.to_csv(filepath, index=False)
            
            # Get final statistics
            stats = calculate_dataset_stats(df)
            
            return jsonify({
                'success': True,
                'message': f'Infinite generation stopped. Generated {generated_count} equations, saved {buffer_size} equations.',
                'filename': filename,
                'final_count': generated_count,
                'saved_count': buffer_size,  # Added to show actual saved count
                'final_range': infinite_generation_state['current_range'],
                'stats': stats
            })
        else:
            return jsonify({
                'success': True,
                'message': 'Infinite generation stopped. No equations generated.',
                'final_count': 0
            })
        
    except Exception as e:
        return jsonify({'error': f'Failed to stop infinite generation: {str(e)}'}), 500

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

def generate_school_grade_equation(coefficient_range, root_type, max_attempts=50):
    """
    ULTIMATE quadratic equation generator for 10M unique equations
    Optimized for neural network training with coefficient range -100 to 100
    """
    import numpy as np
    import math
    from fractions import Fraction
    
    min_coeff = coefficient_range['min']
    max_coeff = coefficient_range['max']
    
    # EXPANDED ROOTS for maximum diversity (targeting 10M equations)
    integer_roots = list(range(-75, 76))  # 151 integer options
    
    # COMPREHENSIVE fractional roots - optimized for uniqueness
    fractional_roots = []
    denominators = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 16, 18, 20, 24, 25, 30, 32, 36, 40]
    for denom in denominators:
        for num in range(-120, 121):  # Extended range for more combinations
            if num != 0:  # Include zero as a root - important for neural network edge cases
                frac = num / denom
                if abs(frac) <= 60:  # Reasonable upper bound
                    fractional_roots.append(frac)
    
    # Add special mathematical constants for edge case training
    special_roots = [0, 0.1, -0.1, 0.25, -0.25, 0.5, -0.5, 0.75, -0.75,
                    math.sqrt(2), -math.sqrt(2), math.sqrt(3), -math.sqrt(3),
                    math.sqrt(5), -math.sqrt(5), math.pi/4, -math.pi/4]
    
    fractional_roots.extend(special_roots)
    fractional_roots = sorted(list(set(fractional_roots)))  # Remove duplicates
    
    # Configure root selection based on type
    if root_type == 'integers':
        possible_roots = integer_roots
    elif root_type == 'fractions':
        possible_roots = fractional_roots
    else:  # mixed - MAXIMUM DIVERSITY
        possible_roots = integer_roots + fractional_roots
    
    # ENHANCED generation algorithm
    max_attempts = max(max_attempts, 300)  # Increase for better success rate
    
    for attempt in range(max_attempts):
        # SMART root selection strategy
        if attempt < max_attempts * 0.4:
            # 40%: Prefer smaller roots for numerical stability
            smaller_roots = [r for r in possible_roots if abs(r) <= 15]
            pool = smaller_roots if smaller_roots and np.random.random() < 0.75 else possible_roots
        elif attempt < max_attempts * 0.8:
            # 40%: Medium-sized roots for good coverage
            medium_roots = [r for r in possible_roots if 5 <= abs(r) <= 35]
            pool = medium_roots if medium_roots and np.random.random() < 0.6 else possible_roots
        else:
            # 20%: Any roots for maximum diversity
            pool = possible_roots
        
        x1 = np.random.choice(pool)
        x2 = np.random.choice(pool)
        
        try:
            # ROBUST fractional arithmetic with overflow protection
            sum_roots = Fraction(x1).limit_denominator(2000) + Fraction(x2).limit_denominator(2000)
            product_roots = Fraction(x1).limit_denominator(2000) * Fraction(x2).limit_denominator(2000)
            
            # Safe LCM calculation
            try:
                lcm_denom = math.lcm(sum_roots.denominator, product_roots.denominator)
            except (ValueError, OverflowError):
                # Fallback for very large denominators
                lcm_denom = sum_roots.denominator * product_roots.denominator
                if lcm_denom > 10000:  # Cap for safety
                    lcm_denom = min(sum_roots.denominator, product_roots.denominator)
            
            # OPTIMAL coefficient range handling for -100 to 100
            max_range = min(abs(max_coeff), abs(min_coeff))
            
            # Intelligent lcm_denom capping
            if lcm_denom > max_range // 3:
                lcm_denom = min(lcm_denom, max_range // 3)
            
            max_a_multiple = max_range // max(lcm_denom, 1)
            if max_a_multiple == 0:
                max_a_multiple = 1
            
            # Generate 'a' candidates with excellent distribution
            a_candidates = []
            for mult in range(1, min(max_a_multiple + 1, 50)):  # Cap for performance
                pos_a = mult * lcm_denom
                neg_a = -mult * lcm_denom
                if abs(pos_a) <= max_range:
                    a_candidates.append(pos_a)
                if abs(neg_a) <= max_range:
                    a_candidates.append(neg_a)
            
            if not a_candidates:
                continue
            
            # WEIGHTED selection optimized for neural network training
            weights = []
            for a_val in a_candidates:
                # Balanced weighting: favor smaller but allow larger
                if abs(a_val) <= 10:
                    weight = 3.0  # High weight for small coefficients
                elif abs(a_val) <= 25:
                    weight = 2.0  # Medium weight
                elif abs(a_val) <= 50:
                    weight = 1.0  # Normal weight
                else:
                    weight = 0.5  # Lower weight for large coefficients
                weights.append(weight)
            
            weights = np.array(weights)
            a = np.random.choice(a_candidates, p=weights / weights.sum())
            
            # Calculate coefficients with proper rounding
            b = int(round(-a * float(sum_roots)))
            c = int(round(a * float(product_roots)))
            
            # STRICT boundary checking
            if (min_coeff <= a <= max_coeff and 
                min_coeff <= b <= max_coeff and 
                min_coeff <= c <= max_coeff):
                
                # FINAL verification and root calculation
                discriminant = b**2 - 4*a*c
                if discriminant >= 0:
                    sqrt_disc = math.sqrt(discriminant)
                    x1_calc = (-b + sqrt_disc) / (2*a)
                    x2_calc = (-b - sqrt_disc) / (2*a)
                else:
                    # For complex cases, return intended roots
                    x1_calc = x1
                    x2_calc = x2
                
                return [float(a), float(b), float(c), float(x1_calc), float(x2_calc)]
                    
        except (ZeroDivisionError, ValueError, OverflowError, TypeError):
            continue
    
    # ENHANCED fallback system
    for _ in range(50):
        a = np.random.choice([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5])
        x1 = np.random.choice([-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6])
        x2 = np.random.choice([-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6])
        
        b = -a * (x1 + x2)
        c = a * x1 * x2
        
        if (min_coeff <= a <= max_coeff and 
            min_coeff <= b <= max_coeff and 
            min_coeff <= c <= max_coeff):
            return [float(a), float(b), float(c), float(x1), float(x2)]
    
    # Ultimate fallback - guaranteed to work
    return [1.0, -1.0, 0.0, 0.0, 1.0]  # x(x-1) = 0

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

@app.route('/api/models/save', methods=['POST'])
def save_model():
    """Save a trained model"""
    try:
        data = request.get_json()
        scenario_key = data.get('scenario_key')
        model_name = data.get('model_name', '').strip()
        
        if not model_name:
            return jsonify({'error': 'Model name is required'}), 400
            
        if len(model_name) > 50:
            return jsonify({'error': 'Model name too long (max 50 characters)'}), 400
            
        if scenario_key not in app_state['predictors']:
            return jsonify({'error': 'No trained model found for this scenario'}), 400
        
        # Get performance metrics from the correct source
        if scenario_key not in app_state['results']:
            return jsonify({'error': 'No performance results found for this scenario'}), 400
            
        performance_metrics = app_state['results'][scenario_key]
        
        # Get dataset info
        dataset_info = {
            'total_equations': len(app_state['data_processor'].data) if app_state['data_processor'].data is not None else 0,
            'stats': app_state['data_processor'].get_stats()
        }
        
        # Pass the correct performance metrics to save_model
        predictor = app_state['predictors'][scenario_key]
        model_id = model_manager.save_model(
            predictor, scenario_key, model_name, dataset_info, performance_metrics
        )
        
        return jsonify({
            'success': True,
            'model_id': model_id,
            'message': f'Model "{model_name}" saved successfully'
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to save model: {str(e)}'}), 500

@app.route('/api/models/save-batch', methods=['POST'])
def save_models_batch():
    """Save all trained models with a common prefix"""
    try:
        data = request.get_json()
        model_prefix = data.get('model_prefix', '').strip()
        
        if not model_prefix:
            return jsonify({'error': 'Model prefix is required'}), 400
            
        if len(model_prefix) > 30:
            return jsonify({'error': 'Model prefix too long (max 30 characters)'}), 400
            
        # Validate prefix (no special characters that could cause file system issues)
        import re
        if not re.match(r'^[a-zA-Z0-9_-]+$', model_prefix):
            return jsonify({'error': 'Model prefix can only contain letters, numbers, underscores, and hyphens'}), 400
        
        # Get all trained scenarios
        trained_scenarios = list(app_state['results'].keys())
        
        if not trained_scenarios:
            return jsonify({'error': 'No trained models available to save'}), 400
            
        # Check if we have predictors for all trained scenarios
        missing_predictors = [key for key in trained_scenarios if key not in app_state['predictors']]
        if missing_predictors:
            return jsonify({'error': f'Missing predictors for scenarios: {", ".join(missing_predictors)}'}), 400
            
        # Get dataset info
        dataset_info = {
            'total_equations': len(app_state['data_processor'].data) if app_state['data_processor'].data is not None else 0,
            'stats': app_state['data_processor'].get_stats()
        }
        
        # Save all models in batch
        saved_models = []
        failed_models = []
        
        for scenario_key in trained_scenarios:
            try:
                # Create model name with prefix
                model_name = f"{model_prefix}_{scenario_key}"
                
                # Get performance metrics
                performance_metrics = app_state['results'][scenario_key]
                predictor = app_state['predictors'][scenario_key]
                
                # Save individual model
                model_id = model_manager.save_models_batch(
                    predictor, 
                    scenario_key, 
                    model_name, 
                    dataset_info, 
                    performance_metrics,
                    model_prefix  # Pass prefix for folder organization
                )
                
                saved_models.append({
                    'scenario_key': scenario_key,
                    'model_name': model_name,
                    'model_id': model_id,
                    'scenario_name': app_state['scenarios'][scenario_key].name
                })
                
            except Exception as model_error:
                failed_models.append({
                    'scenario_key': scenario_key,
                    'error': str(model_error)
                })
                continue
        
        if not saved_models:
            return jsonify({
                'error': 'Failed to save any models',
                'failed_models': failed_models
            }), 500
            
        # Prepare response
        response_data = {
            'success': True,
            'message': f'Batch save completed: {len(saved_models)}/{len(trained_scenarios)} models saved',
            'prefix': model_prefix,
            'saved_models': saved_models,
            'saved_count': len(saved_models),
            'total_count': len(trained_scenarios)
        }
        
        if failed_models:
            response_data['warning'] = f'{len(failed_models)} models failed to save'
            response_data['failed_models'] = failed_models
            
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({'error': f'Batch save failed: {str(e)}'}), 500

@app.route('/api/models/load', methods=['POST'])
def load_models():
    """Load one or multiple saved models"""
    try:
        data = request.get_json()
        model_ids = data.get('model_ids', [])
        
        # Support both single model (backward compatibility) and multiple models
        single_model_id = data.get('model_id')
        if single_model_id and not model_ids:
            model_ids = [single_model_id]
        
        if not model_ids:
            return jsonify({'error': 'Model ID(s) required'}), 400
            
        loaded_models = []
        failed_models = []
        
        for model_id in model_ids:
            try:
                # Load the model
                predictor = model_manager.load_model(
                    model_id, app_state['data_processor'], app_state['scenarios']
                )

                if predictor is None:
                    failed_models.append({
                        'model_id': model_id,
                        'error': 'Model not found or corrupted'
                    })
                    continue
                
                # Get model info
                model_info = model_manager.get_model_info(model_id)
                
                if model_info:
                    scenario_key = model_info['scenario_key']
                    
                    # Store in application state
                    app_state['predictors'][scenario_key] = predictor
                    
                    # Normalize performance stats to match fresh training structure
                    performance = predictor.performance_stats
                    app_state['results'][scenario_key] = {
                        'r2': performance.get('r2', 0),
                        'mse': performance.get('mse', 0),
                        'mae': performance.get('mae', 0),
                        'accuracy_10pct': performance.get('accuracy_10pct', 0)
                    }
                    
                    loaded_models.append({
                        'model_id': model_id,
                        'model_name': model_info['model_name'],
                        'scenario_key': scenario_key,
                        'scenario_name': model_info['scenario_name']
                    })
                else:
                    failed_models.append({
                        'model_id': model_id,
                        'error': 'Model metadata not found'
                    })
                    
            except Exception as model_error:
                failed_models.append({
                    'model_id': model_id,
                    'error': str(model_error)
                })
                continue
        
        if not loaded_models:
            return jsonify({
                'error': 'Failed to load any models',
                'failed_models': failed_models
            }), 500
            
        # Prepare response
        response_data = {
            'success': True,
            'loaded_models': loaded_models,
            'loaded_count': len(loaded_models),
            'total_count': len(model_ids)
        }
        
        if len(loaded_models) == 1:
            # Single model response (backward compatibility)
            response_data['message'] = f'Model "{loaded_models[0]["model_name"]}" loaded successfully'
            response_data['model_info'] = model_manager.get_model_info(loaded_models[0]['model_id'])
            response_data['scenario_key'] = loaded_models[0]['scenario_key']
        else:
            # Multiple models response
            response_data['message'] = f'Batch load completed: {len(loaded_models)}/{len(model_ids)} models loaded'
            
        if failed_models:
            response_data['warning'] = f'{len(failed_models)} models failed to load'
            response_data['failed_models'] = failed_models
            
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({'error': f'Failed to load models: {str(e)}'}), 500

@app.route('/api/models/list')
def list_saved_models():
    """Get list of saved models"""
    try:
        models = model_manager.get_saved_models()
        
        return jsonify({
            'success': True,
            'models': models
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to list models: {str(e)}'}), 500

@app.route('/api/models/delete', methods=['DELETE'])
def delete_saved_model():
    """Delete a saved model"""
    try:
        data = request.get_json()
        model_id = data.get('model_id')
        
        if not model_id:
            return jsonify({'error': 'Model ID is required'}), 400
        
        success = model_manager.delete_model(model_id)
        
        if success:
            return jsonify({
                'success': True,
                'message': 'Model deleted successfully'
            })
        else:
            return jsonify({'error': 'Model not found'}), 404
            
    except Exception as e:
        return jsonify({'error': f'Failed to delete model: {str(e)}'}), 500

@app.route('/api/models/info/<model_id>')
def get_model_info(model_id):
    """Get detailed model information"""
    try:
        model_info = model_manager.get_model_info(model_id)
        
        if model_info:
            return jsonify({
                'success': True,
                'model_info': model_info
            })
        else:
            return jsonify({'error': 'Model not found'}), 404
            
    except Exception as e:
        return jsonify({'error': f'Failed to get model info: {str(e)}'}), 500

def _train_models_background(selected_scenarios, epochs, learning_rate):
    """Background training function"""
    try:
        app_state['training_status']['is_training'] = True
        app_state['training_status']['logs'] = []
        
        # Only clear models that will be retrained, preserve loaded models from other scenarios
        for scenario_key in selected_scenarios:
            if scenario_key in app_state['predictors']:
                del app_state['predictors'][scenario_key]
            if scenario_key in app_state['results']:
                del app_state['results'][scenario_key]
        
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

def _calculate_quadratic_solutions(inputs, return_type=False):
    """
    Calculate actual solutions for quadratic equation ax² + bx + c = 0.
    Now returns a dictionary for richer information.
    """
    a, b, c = inputs
    if abs(a) < 1e-9:
        if abs(b) < 1e-9:
            return {'type': 'invalid', 'roots': None, 'message': 'Not a valid equation (a and b are zero)'}
        root = -c / b
        return {'type': 'linear', 'roots': [root], 'message': 'Linear equation, one root found'}

    discriminant = b**2 - 4*a*c

    if discriminant < 0:
        return {'type': 'complex', 'roots': None, 'message': 'No real solutions (complex roots)'}
    
    if abs(discriminant) < 1e-9:
        root = -b / (2*a)
        return {'type': 'repeated', 'roots': [root], 'message': 'One repeated real root'}

    sqrt_discriminant = np.sqrt(discriminant)
    x1 = (-b + sqrt_discriminant) / (2*a)
    x2 = (-b - sqrt_discriminant) / (2*a)
    
    # Sort roots for consistency
    roots = tuple(sorted((x1, x2)))
    
    if return_type:
      return {'type': 'distinct', 'roots': roots, 'message': 'Two distinct real roots'}
    
    return roots

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

def _get_prediction_details(scenario_key, inputs, predictions, scenario_info):
    """
    Calculates actual values and error metrics for a given prediction scenario.
    This allows the frontend to display a rich comparison for any model type.
    """
    details = {
        'scenario_key': scenario_key,
        'scenario_info': {
            'name': scenario_info.name,
            'description': scenario_info.description,
            'input_features': scenario_info.input_features,
            'target_features': scenario_info.target_features
        },
        'equation_parts': {},
        'predicted_values': {},
        'actual_values': {},
        'error_metrics': {},
        'analysis': {},
        'labels': {
            'predicted': 'Neural Network Prediction',
            'actual': 'Calculated Ground Truth',
            'error': 'Prediction Error'
        }
    }

    try:
        # --- FIX: Renamed 'partial_coeff' to match the key from the error message ---
        if scenario_key == 'partial_coeff_to_missing': 
            # This logic was previously under 'partial_coeff'
            a, b, x1 = inputs
            c_pred, x2_pred = predictions
            
            details['equation_parts'] = {'a': a, 'b': b, 'x₁': x1}
            details['predicted_values'] = {'c': c_pred, 'x₂': x2_pred}
            
            c_actual = -a * (x1**2) - b * x1
            x2_actual = (-b / a) - x1 if a != 0 else 0
            details['actual_values'] = {'c': c_actual, 'x₂': x2_actual}
            
            errors = {'c Error': abs(c_pred - c_actual), 'x₂ Error': abs(x2_pred - x2_actual)}
            errors['Average Error'] = sum(errors.values()) / 2
            details['error_metrics'] = errors

        elif scenario_key == 'coeff_to_roots':
            a, b, c = inputs
            x1_pred, x2_pred = predictions
            details['equation_parts'] = {'a': a, 'b': b, 'c': c}
            
            # FIXED: Sort predicted roots for consistency
            x1_pred_sorted, x2_pred_sorted = sorted([x1_pred, x2_pred])
            details['predicted_values'] = {'x₁': x1_pred_sorted, 'x₂': x2_pred_sorted}
            
            actual_sols_result = _calculate_quadratic_solutions(inputs, return_type=True)
            details['analysis']['actual_solution_type'] = actual_sols_result['type']
            if actual_sols_result['roots']:
                x1_actual, x2_actual = sorted(actual_sols_result['roots'])
                details['actual_values'] = {'x₁': x1_actual, 'x₂': x2_actual}
                
                # Calculate errors with sorted values
                errors = {
                    'x₁ Error': abs(x1_pred_sorted - x1_actual), 
                    'x₂ Error': abs(x2_pred_sorted - x2_actual)
                }
                errors['Average Error'] = sum(errors.values()) / 2
                details['error_metrics'] = errors
            else:
                details['analysis']['actual_solution_message'] = actual_sols_result['message']

        elif scenario_key == 'roots_to_coeff':
            x1, x2 = inputs
            a_pred, b_pred, c_pred = predictions
            
            details['equation_parts'] = {'x₁': x1, 'x₂': x2}
            details['predicted_values'] = {'a': a_pred, 'b': b_pred, 'c': c_pred}
            
            a_actual = 1.0 # Normalize to a=1 for ground truth
            b_actual = -a_actual * (x1 + x2)
            c_actual = a_actual * (x1 * x2)
            
            # Scale ground truth to match predicted 'a' for fair comparison
            scale_factor = a_pred / a_actual if abs(a_actual) > 1e-6 else 0
            details['actual_values'] = {
                'a': a_actual * scale_factor, 
                'b': b_actual * scale_factor, 
                'c': c_actual * scale_factor
            }
            
            errors = {
                'a Error': abs(a_pred - details['actual_values']['a']),
                'b Error': abs(b_pred - details['actual_values']['b']),
                'c Error': abs(c_pred - details['actual_values']['c'])
            }
            errors['Average Error'] = sum(errors.values()) / 3
            details['error_metrics'] = errors

        elif scenario_key == 'single_missing':
            a, b, c, x1 = inputs
            x2_pred, = predictions
            
            details['equation_parts'] = {'a': a, 'b': b, 'c': c, 'x₁': x1}
            details['predicted_values'] = {'x₂': x2_pred}
            
            x2_actual = (-b / a) - x1 if a != 0 else 0
            details['actual_values'] = {'x₂': x2_actual}
            
            errors = {'x₂ Error': abs(x2_pred - x2_actual)}
            errors['Average Error'] = errors['x₂ Error']
            details['error_metrics'] = errors

        elif scenario_key == 'verify_equation':
            a, b, c, x1, x2 = inputs
            error_pred, = predictions
            
            details['display_type'] = 'error_verification'
            details['equation_parts'] = {'a': a, 'b': b, 'c': c, 'x₁': x1, 'x₂': x2}
            details['predicted_values'] = {'Predicted Error': error_pred}
            
            residual1 = abs(a * (x1**2) + b * x1 + c)
            residual2 = abs(a * (x2**2) + b * x2 + c)
            error_actual = (residual1 + residual2) / 2.0
            details['actual_values'] = {'Actual Error': error_actual}
            
            errors = {'Prediction Deviation': abs(error_pred - error_actual)}
            details['error_metrics'] = errors
            details['labels']['predicted'] = 'NN Predicted Error'
            details['labels']['actual'] = 'Calculated Actual Error'

    except Exception as e:
        # traceback.print_exc() # Uncomment for debugging
        return { 'scenario_key': scenario_key, 'display_type': 'error', 'message': str(e) }

    return details

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
