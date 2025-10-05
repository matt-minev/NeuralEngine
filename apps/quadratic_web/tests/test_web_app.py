#!/usr/bin/env python3
"""
Quadratic Neural Network Web Application
Comprehensive Testing Suite

Complete testing suite for the web application functionality
"""

import unittest
import json
import tempfile
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
import time
import threading

# Add the project root to path
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

try:
    from app import app, app_state
    from config import get_config
    from core.data_processor import QuadraticDataProcessor
    from core.predictor import QuadraticPredictor
    from config.scenarios import get_default_scenarios
    from helpers import format_number, assess_performance
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all modules are available")
    sys.exit(1)

class TestFlaskApp(unittest.TestCase):
    """Test Flask application endpoints"""
    
    def setUp(self):
        """Set up test environment"""
        self.app = app
        self.app.config['TESTING'] = True
        self.app.config['WTF_CSRF_ENABLED'] = False
        self.client = self.app.test_client()
        
        # Reset application state
        app_state['data_processor'] = QuadraticDataProcessor(verbose=False)
        app_state['predictors'].clear()
        app_state['results'].clear()
        app_state['training_status']['is_training'] = False
        
        # Create test data
        self.test_data = self.create_test_dataset()
        
    def create_test_dataset(self):
        """Create test dataset for testing"""
        # Generate test quadratic equations
        np.random.seed(42)  # For reproducibility
        n_samples = 1000
        
        # Generate random coefficients
        a = np.random.uniform(-5, 5, n_samples)
        a = np.where(np.abs(a) < 0.1, np.sign(a) * 0.5, a)  # Avoid near-zero
        b = np.random.uniform(-10, 10, n_samples)
        c = np.random.uniform(-10, 10, n_samples)
        
        # Calculate roots using quadratic formula
        discriminant = b**2 - 4*a*c
        sqrt_discriminant = np.sqrt(np.abs(discriminant))
        
        x1 = (-b + sqrt_discriminant) / (2*a)
        x2 = (-b - sqrt_discriminant) / (2*a)
        
        # Handle complex roots (set to NaN for simplicity)
        x1 = np.where(discriminant >= 0, x1, np.nan)
        x2 = np.where(discriminant >= 0, x2, np.nan)
        
        # Filter out NaN values
        mask = ~(np.isnan(x1) | np.isnan(x2))
        
        return pd.DataFrame({
            'a': a[mask],
            'b': b[mask],
            'c': c[mask],
            'x1': x1[mask],
            'x2': x2[mask]
        })
    
    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = self.client.get('/api/health')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'healthy')
        self.assertIn('timestamp', data)
        self.assertEqual(data['location'], 'Varna, Bulgaria')
    
    def test_scenarios_endpoint(self):
        """Test scenarios endpoint"""
        response = self.client.get('/api/scenarios')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertIn('coeff_to_roots', data)
        self.assertIn('name', data['coeff_to_roots'])
        self.assertIn('description', data['coeff_to_roots'])
        self.assertIn('input_features', data['coeff_to_roots'])
        self.assertIn('target_features', data['coeff_to_roots'])
    
    def test_data_upload(self):
        """Test data upload functionality"""
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            self.test_data.to_csv(f.name, index=False)
            temp_file = f.name
        
        try:
            # Test file upload
            with open(temp_file, 'rb') as f:
                response = self.client.post('/api/data/upload', 
                                          data={'file': (f, 'test_data.csv')})
            
            self.assertEqual(response.status_code, 200)
            data = json.loads(response.data)
            self.assertTrue(data['success'])
            self.assertIn('message', data)
            self.assertIn('stats', data)
            self.assertIn('sample_data', data)
            
        finally:
            os.unlink(temp_file)
    
    def test_data_info_endpoint(self):
        """Test data info endpoint"""
        # Test when no data is loaded
        response = self.client.get('/api/data/info')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertFalse(data['loaded'])
        
        # Load test data
        app_state['data_processor'].data = self.test_data.values
        app_state['data_processor']._calculate_stats()
        
        # Test with data loaded
        response = self.client.get('/api/data/info')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertTrue(data['loaded'])
        self.assertIn('total_equations', data)
        self.assertIn('stats', data)
        self.assertIn('sample_data', data)
    
    def test_training_start_without_data(self):
        """Test training start without data"""
        response = self.client.post('/api/training/start',
                                   json={'scenarios': ['coeff_to_roots'], 'epochs': 10})
        
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)
        self.assertIn('No dataset loaded', data['error'])
    
    def test_training_start_with_data(self):
        """Test training start with data"""
        # Load test data
        app_state['data_processor'].data = self.test_data.values
        app_state['data_processor']._calculate_stats()
        
        response = self.client.post('/api/training/start',
                                   json={'scenarios': ['coeff_to_roots'], 'epochs': 10})
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        self.assertEqual(data['scenarios'], ['coeff_to_roots'])
        self.assertEqual(data['epochs'], 10)
        
        # Wait a bit for training to start
        time.sleep(0.1)
        
        # Check training status
        response = self.client.get('/api/training/status')
        self.assertEqual(response.status_code, 200)
        status = json.loads(response.data)
        self.assertIn('is_training', status)
    
    def test_training_status_endpoint(self):
        """Test training status endpoint"""
        response = self.client.get('/api/training/status')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertIn('is_training', data)
        self.assertIn('progress', data)
        self.assertIn('current_scenario', data)
        self.assertIn('logs', data)
    
    def test_training_stop_endpoint(self):
        """Test training stop endpoint"""
        response = self.client.post('/api/training/stop')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        self.assertIn('message', data)
    
    def test_predict_without_model(self):
        """Test prediction without trained model"""
        response = self.client.post('/api/predict',
                                   json={'scenario': 'coeff_to_roots', 'inputs': [1, -3, 2]})
        
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)
        self.assertIn('not trained', data['error'])
    
    def test_results_endpoint(self):
        """Test results endpoint"""
        response = self.client.get('/api/results')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertIsInstance(data, dict)
    
    def test_performance_analysis_endpoint(self):
        """Test performance analysis endpoint"""
        response = self.client.get('/api/analysis/performance')
        
        # Should return error when no results
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)
    
    def test_invalid_endpoints(self):
        """Test invalid endpoints"""
        response = self.client.get('/api/invalid')
        self.assertEqual(response.status_code, 404)
        
        response = self.client.get('/nonexistent')
        self.assertEqual(response.status_code, 404)
    
    def test_main_page(self):
        """Test main page rendering"""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Quadratic Neural Network', response.data)

class TestDataProcessor(unittest.TestCase):
    """Test data processor functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.processor = QuadraticDataProcessor(verbose=False)
        
        # Create test data
        self.test_data = pd.DataFrame({
            'a': [1, 2, 1, -1],
            'b': [-3, -4, 0, 2],
            'c': [2, 2, -1, 1],
            'x1': [1, 1, 1, -1],
            'x2': [2, 1, -1, -1]
        })
    
    def test_load_data_from_dataframe(self):
        """Test loading data from DataFrame"""
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            self.test_data.to_csv(f.name, index=False)
            temp_file = f.name
        
        try:
            # Test loading
            success = self.processor.load_data(temp_file)
            self.assertTrue(success)
            self.assertIsNotNone(self.processor.data)
            self.assertEqual(self.processor.data.shape[0], 4)
            self.assertEqual(self.processor.data.shape[1], 6)  # 5 + error column
            
        finally:
            os.unlink(temp_file)
    
    def test_load_invalid_data(self):
        """Test loading invalid data"""
        # Create invalid data (wrong number of columns)
        invalid_data = pd.DataFrame({
            'a': [1, 2],
            'b': [3, 4]
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            invalid_data.to_csv(f.name, index=False)
            temp_file = f.name
        
        try:
            success = self.processor.load_data(temp_file)
            self.assertFalse(success)
            
        finally:
            os.unlink(temp_file)
    
    def test_calculate_stats(self):
        """Test statistics calculation"""
        # Set data directly
        self.processor.data = self.test_data.values
        self.processor._calculate_stats()
        
        stats = self.processor.get_stats()
        self.assertIsNotNone(stats)
        self.assertIn('total_equations', stats)
        self.assertIn('columns', stats)
        self.assertIn('quality', stats)
        
        # Check column stats
        for col in ['a', 'b', 'c', 'x1', 'x2']:
            self.assertIn(col, stats['columns'])
            self.assertIn('mean', stats['columns'][col])
            self.assertIn('std', stats['columns'][col])
            self.assertIn('min', stats['columns'][col])
            self.assertIn('max', stats['columns'][col])
    
    def test_prepare_scenario_data(self):
        """Test scenario data preparation"""
        # Load test data
        self.processor.data = self.test_data.values
        self.processor._add_error_column()
        
        # Test with a scenario
        scenarios = get_default_scenarios()
        scenario = scenarios['coeff_to_roots']
        
        X, y = self.processor.prepare_scenario_data(scenario, normalize=True)
        
        self.assertEqual(X.shape[1], 3)  # a, b, c
        self.assertEqual(y.shape[1], 2)  # x1, x2
        self.assertEqual(X.shape[0], y.shape[0])
    
    def test_data_splitting(self):
        """Test data splitting"""
        # Create larger dataset for splitting
        data = np.random.rand(100, 5)
        
        X, y = data[:, :3], data[:, 3:]
        
        X_train, X_val, X_test, y_train, y_val, y_test = self.processor.split_data(X, y)
        
        # Check shapes
        self.assertEqual(X_train.shape[0], 70)  # 70% train
        self.assertEqual(X_val.shape[0], 15)    # 15% validation
        self.assertEqual(X_test.shape[0], 15)   # 15% test
        
        # Check consistency
        self.assertEqual(X_train.shape[0], y_train.shape[0])
        self.assertEqual(X_val.shape[0], y_val.shape[0])
        self.assertEqual(X_test.shape[0], y_test.shape[0])
    
    def test_get_sample_data(self):
        """Test sample data retrieval"""
        self.processor.data = self.test_data.values
        
        sample = self.processor.get_sample_data(2)
        self.assertEqual(sample.shape[0], 2)
        self.assertEqual(sample.shape[1], 5)
        
        # Test with more samples than available
        sample = self.processor.get_sample_data(10)
        self.assertEqual(sample.shape[0], 4)  # Only 4 samples available

class TestPredictor(unittest.TestCase):
    """Test predictor functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.data_processor = QuadraticDataProcessor(verbose=False)
        
        # Create test data
        test_data = pd.DataFrame({
            'a': [1, 2, 1, -1, 0.5],
            'b': [-3, -4, 0, 2, -1],
            'c': [2, 2, -1, 1, 0.5],
            'x1': [1, 1, 1, -1, 1],
            'x2': [2, 1, -1, -1, 1]
        })
        
        self.data_processor.data = test_data.values
        self.data_processor._add_error_column()
        self.data_processor._calculate_stats()
        
        self.scenarios = get_default_scenarios()
        self.scenario = self.scenarios['coeff_to_roots']
    
    def test_predictor_creation(self):
        """Test predictor creation"""
        predictor = QuadraticPredictor(self.scenario, self.data_processor)
        
        self.assertEqual(predictor.scenario, self.scenario)
        self.assertEqual(predictor.data_processor, self.data_processor)
        self.assertIsNone(predictor.network)
        self.assertFalse(predictor.is_trained)
    
    def test_network_creation(self):
        """Test network creation"""
        predictor = QuadraticPredictor(self.scenario, self.data_processor)
        predictor.create_network()
        
        self.assertIsNotNone(predictor.network)
        self.assertIsNotNone(predictor.trainer)
    
    def test_training(self):
        """Test model training"""
        predictor = QuadraticPredictor(self.scenario, self.data_processor)
        
        # Test training with very few epochs
        results = predictor.train(epochs=5, verbose=False)
        
        self.assertIsNotNone(results)
        self.assertIn('training_time', results)
        self.assertIn('test_results', results)
        self.assertIn('training_history', results)
        self.assertTrue(predictor.is_trained)
        
        # Check test results
        test_results = results['test_results']
        self.assertIn('r2', test_results)
        self.assertIn('mse', test_results)
        self.assertIn('mae', test_results)
        self.assertIn('rmse', test_results)
        self.assertIn('accuracy_10pct', test_results)
    
    def test_prediction(self):
        """Test making predictions"""
        predictor = QuadraticPredictor(self.scenario, self.data_processor)
        
        # Train first
        predictor.train(epochs=5, verbose=False)
        
        # Test prediction
        input_data = np.array([[1, -3, 2]])  # x^2 - 3x + 2 = 0, roots: 1, 2
        predictions, confidences = predictor.predict(input_data, return_confidence=True)
        
        self.assertIsNotNone(predictions)
        self.assertIsNotNone(confidences)
        self.assertEqual(predictions.shape[1], 2)  # Two roots
        self.assertEqual(confidences.shape[1], 2)  # Two confidences
    
    def test_prediction_without_training(self):
        """Test prediction without training"""
        predictor = QuadraticPredictor(self.scenario, self.data_processor)
        
        input_data = np.array([[1, -3, 2]])
        
        with self.assertRaises(ValueError):
            predictor.predict(input_data)
    
    def test_evaluation(self):
        """Test model evaluation"""
        predictor = QuadraticPredictor(self.scenario, self.data_processor)
        
        # Train first
        predictor.train(epochs=5, verbose=False)
        
        # Prepare test data
        X, y = self.data_processor.prepare_scenario_data(self.scenario, normalize=True)
        
        # Evaluate
        metrics = predictor.evaluate(X, y)
        
        self.assertIn('r2', metrics)
        self.assertIn('mse', metrics)
        self.assertIn('mae', metrics)
        self.assertIn('rmse', metrics)
        self.assertIn('accuracy_10pct', metrics)
    
    def test_get_info(self):
        """Test getting predictor info"""
        predictor = QuadraticPredictor(self.scenario, self.data_processor)
        
        info = predictor.get_info()
        
        self.assertIn('scenario', info)
        self.assertIn('description', info)
        self.assertIn('input_features', info)
        self.assertIn('target_features', info)
        self.assertIn('network_architecture', info)
        self.assertIn('is_trained', info)
        self.assertFalse(info['is_trained'])
        
        # Train and check again
        predictor.train(epochs=5, verbose=False)
        info = predictor.get_info()
        self.assertTrue(info['is_trained'])
        self.assertIn('training_stats', info)

class TestHelpers(unittest.TestCase):
    """Test helper functions"""
    
    def test_format_number(self):
        """Test number formatting"""
        self.assertEqual(format_number(3.14159, 2), "3.14")
        self.assertEqual(format_number(3.14159, 4), "3.1416")
        self.assertEqual(format_number(0, 3), "0.000")
        self.assertEqual(format_number(1e-12, 6), "0.000000")
        self.assertEqual(format_number(-3.14159, 2), "-3.14")
    
    def test_assess_performance(self):
        """Test performance assessment"""
        self.assertEqual(assess_performance(0.95), "EXCELLENT")
        self.assertEqual(assess_performance(0.85), "GOOD")
        self.assertEqual(assess_performance(0.65), "FAIR")
        self.assertEqual(assess_performance(0.35), "POOR")
    
    def test_get_confidence_level(self):
        """Test confidence level assessment"""
        self.assertEqual(get_confidence_level(0.9), "🟢 High")
        self.assertEqual(get_confidence_level(0.7), "🟡 Medium")
        self.assertEqual(get_confidence_level(0.5), "🔴 Low")

class TestScenarios(unittest.TestCase):
    """Test scenario configurations"""
    
    def test_get_default_scenarios(self):
        """Test default scenarios"""
        scenarios = get_default_scenarios()
        
        self.assertIsInstance(scenarios, dict)
        self.assertIn('coeff_to_roots', scenarios)
        self.assertIn('partial_coeff_to_missing', scenarios)
        self.assertIn('roots_to_coeff', scenarios)
        
        # Check scenario structure
        for scenario in scenarios.values():
            self.assertIsNotNone(scenario.name)
            self.assertIsNotNone(scenario.description)
            self.assertIsNotNone(scenario.input_features)
            self.assertIsNotNone(scenario.target_features)
            self.assertIsNotNone(scenario.network_architecture)
            self.assertIsNotNone(scenario.activations)
            self.assertIsNotNone(scenario.color)
    
    def test_scenario_consistency(self):
        """Test scenario consistency"""
        scenarios = get_default_scenarios()
        
        for scenario in scenarios.values():
            # Check that network architecture matches features
            self.assertEqual(scenario.network_architecture[0], len(scenario.input_features))
            self.assertEqual(scenario.network_architecture[-1], len(scenario.target_features))
            
            # Check that activations match layers
            self.assertEqual(len(scenario.activations), len(scenario.network_architecture) - 1)

class TestIntegration(unittest.TestCase):
    """Integration tests"""
    
    def setUp(self):
        """Set up integration test environment"""
        self.app = app
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()
        
        # Create test data
        self.test_data = pd.DataFrame({
            'a': [1, 2, 1, -1, 0.5] * 20,  # More data for training
            'b': [-3, -4, 0, 2, -1] * 20,
            'c': [2, 2, -1, 1, 0.5] * 20,
            'x1': [1, 1, 1, -1, 1] * 20,
            'x2': [2, 1, -1, -1, 1] * 20
        })
    
    def test_full_workflow(self):
        """Test complete workflow from data upload to prediction"""
        # 1. Upload data
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            self.test_data.to_csv(f.name, index=False)
            temp_file = f.name
        
        try:
            with open(temp_file, 'rb') as f:
                response = self.client.post('/api/data/upload', 
                                          data={'file': (f, 'test_data.csv')})
            
            self.assertEqual(response.status_code, 200)
            
            # 2. Check data info
            response = self.client.get('/api/data/info')
            self.assertEqual(response.status_code, 200)
            data = json.loads(response.data)
            self.assertTrue(data['loaded'])
            
            # 3. Start training
            response = self.client.post('/api/training/start',
                                       json={'scenarios': ['coeff_to_roots'], 'epochs': 10})
            
            self.assertEqual(response.status_code, 200)
            
            # 4. Wait for training to complete
            max_wait = 30  # seconds
            wait_time = 0
            while wait_time < max_wait:
                response = self.client.get('/api/training/status')
                status = json.loads(response.data)
                if not status['is_training']:
                    break
                time.sleep(1)
                wait_time += 1
            
            # 5. Check results
            response = self.client.get('/api/results')
            self.assertEqual(response.status_code, 200)
            results = json.loads(response.data)
            
            if 'coeff_to_roots' in results:
                # 6. Make prediction
                response = self.client.post('/api/predict',
                                           json={'scenario': 'coeff_to_roots', 'inputs': [1, -3, 2]})
                
                self.assertEqual(response.status_code, 200)
                prediction = json.loads(response.data)
                self.assertTrue(prediction['success'])
                self.assertIn('predictions', prediction)
                self.assertIn('confidences', prediction)
                
        finally:
            os.unlink(temp_file)

class TestPerformance(unittest.TestCase):
    """Performance tests"""
    
    def test_large_dataset_loading(self):
        """Test loading large datasets"""
        # Create large dataset
        n_samples = 10000
        large_data = pd.DataFrame({
            'a': np.random.uniform(-5, 5, n_samples),
            'b': np.random.uniform(-10, 10, n_samples),
            'c': np.random.uniform(-10, 10, n_samples),
            'x1': np.random.uniform(-5, 5, n_samples),
            'x2': np.random.uniform(-5, 5, n_samples)
        })
        
        processor = QuadraticDataProcessor(verbose=False)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            large_data.to_csv(f.name, index=False)
            temp_file = f.name
        
        try:
            start_time = time.time()
            success = processor.load_data(temp_file)
            load_time = time.time() - start_time
            
            self.assertTrue(success)
            self.assertLess(load_time, 10)  # Should load within 10 seconds
            
        finally:
            os.unlink(temp_file)
    
    def test_prediction_speed(self):
        """Test prediction speed"""
        # Create and train a simple model
        processor = QuadraticDataProcessor(verbose=False)
        processor.data = np.random.rand(1000, 5)
        processor._add_error_column()
        processor._calculate_stats()
        
        scenarios = get_default_scenarios()
        predictor = QuadraticPredictor(scenarios['coeff_to_roots'], processor)
        predictor.train(epochs=5, verbose=False)
        
        # Test prediction speed
        input_data = np.random.rand(100, 3)
        
        start_time = time.time()
        predictions, confidences = predictor.predict(input_data, return_confidence=True)
        prediction_time = time.time() - start_time
        
        self.assertLess(prediction_time, 1)  # Should predict within 1 second
        self.assertEqual(predictions.shape[0], 100)

def run_all_tests():
    """Run all test suites"""
    print("🧪 Running Quadratic Neural Network Web App Tests")
    print("=" * 60)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestFlaskApp,
        TestDataProcessor,
        TestPredictor,
        TestHelpers,
        TestScenarios,
        TestIntegration,
        TestPerformance
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\nOverall result: {'✅ PASSED' if success else '❌ FAILED'}")
    
    return success

if __name__ == '__main__':
    run_all_tests()