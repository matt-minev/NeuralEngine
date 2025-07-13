"""
Neural Network Engine Test Suite
===============================

Comprehensive tests for all components of the Neural Network Engine:
- Unit tests for individual modules
- Integration tests for component interaction
- Performance tests for optimization validation
- End-to-end tests for complete workflows

Run with: pytest tests/test_nn.py -v
"""

import pytest
import numpy as np
import sys
import os

# Add parent directory to path to import our modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our modules
from nn_core import Layer, NeuralNetwork, mean_squared_error, mean_absolute_error, create_sample_data
from autodiff import SGD, Adam, TrainingEngine
from data_utils import DataLoader, DataPreprocessor, DataSplitter, BatchProcessor
from utils import ActivationFunctions, MathUtils, NetworkVisualizer, PerformanceMonitor, create_test_data, print_network_summary


class TestLayer:
    """Test the Layer class functionality."""
    
    def test_layer_initialization(self):
        """Test layer initialization with different parameters."""
        layer = Layer(input_size=3, output_size=5, activation='relu')
        
        assert layer.input_size == 3
        assert layer.output_size == 5
        assert layer.activation_name == 'relu'
        assert layer.weights.shape == (5, 3)
        assert layer.biases.shape == (5,)
    
    def test_layer_forward_pass(self):
        """Test forward propagation through a layer."""
        layer = Layer(input_size=3, output_size=2, activation='linear')
        
        # Set known weights and biases for testing
        layer.weights = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        layer.biases = np.array([0.1, 0.2], dtype=np.float32)
        
        # Test single sample
        x = np.array([1, 2, 3], dtype=np.float32)
        output = layer.forward(x)
        
        expected = np.array([1*1 + 2*2 + 3*3 + 0.1, 1*4 + 2*5 + 3*6 + 0.2])  # [14.1, 32.2]
        np.testing.assert_array_almost_equal(output, expected, decimal=5)
    
    def test_layer_batch_forward(self):
        """Test forward propagation with batch input."""
        layer = Layer(input_size=2, output_size=3, activation='relu')
        
        # Test batch input
        x_batch = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
        output = layer.forward(x_batch)
        
        assert output.shape == (3, 3)  # 3 samples, 3 outputs
        assert np.all(output >= 0)  # ReLU should produce non-negative outputs
    
    def test_activation_functions(self):
        """Test different activation functions."""
        activations = ['relu', 'sigmoid', 'tanh', 'linear']
        
        for activation in activations:
            layer = Layer(input_size=2, output_size=3, activation=activation)
            x = np.array([1, -1], dtype=np.float32)
            output = layer.forward(x)
            
            assert output.shape == (3,)
            assert not np.any(np.isnan(output))
            assert not np.any(np.isinf(output))
    
    def test_parameter_access(self):
        """Test parameter getting and setting."""
        layer = Layer(input_size=2, output_size=3, activation='relu')
        
        # Get parameters
        weights, biases = layer.get_parameters()
        assert weights.shape == (3, 2)
        assert biases.shape == (3,)
        
        # Set new parameters
        new_weights = np.ones((3, 2), dtype=np.float32)
        new_biases = np.zeros((3,), dtype=np.float32)
        layer.set_parameters(new_weights, new_biases)
        
        # Verify parameters were set
        weights, biases = layer.get_parameters()
        np.testing.assert_array_equal(weights, new_weights)
        np.testing.assert_array_equal(biases, new_biases)


class TestNeuralNetwork:
    """Test the NeuralNetwork class functionality."""
    
    def test_network_initialization(self):
        """Test neural network initialization."""
        layer_sizes = [3, 5, 2, 1]
        network = NeuralNetwork(layer_sizes)
        
        assert network.layer_sizes == layer_sizes
        assert network.num_layers == 3
        assert len(network.layers) == 3
        assert network.count_parameters() > 0
    
    def test_network_forward_pass(self):
        """Test forward propagation through entire network."""
        network = NeuralNetwork([2, 3, 1], ['relu', 'linear'])
        
        # Test single sample
        x = np.array([1, 2], dtype=np.float32)
        output = network.forward(x)
        
        assert output.shape == (1,)
        assert not np.isnan(output).any()
        assert not np.isinf(output).any()
    
    def test_network_batch_processing(self):
        """Test network with batch inputs."""
        network = NeuralNetwork([3, 4, 2], ['relu', 'linear'])
        
        # Test batch processing
        x_batch = np.random.randn(10, 3).astype(np.float32)
        output = network.forward(x_batch)
        
        assert output.shape == (10, 2)
        assert not np.isnan(output).any()
        assert not np.isinf(output).any()
    
    def test_parameter_management(self):
        """Test parameter getting and setting."""
        network = NeuralNetwork([2, 3, 1])
        
        # Get all parameters
        params = network.get_all_parameters()
        assert len(params) == 4  # 2 layers × 2 param types (weights, biases)
        
        # Count parameters
        param_count = network.count_parameters()
        manual_count = sum(p.size for p in params)
        assert param_count == manual_count
        
        # Test parameter setting
        network.set_all_parameters(params)  # Should not raise error
    
    def test_network_architectures(self):
        """Test different network architectures."""
        architectures = [
            ([1, 1], ['linear']),
            ([2, 5, 3, 1], ['relu', 'relu', 'linear']),
            ([4, 10, 8, 5, 2], ['relu', 'relu', 'relu', 'linear'])
        ]
        
        for layer_sizes, activations in architectures:
            network = NeuralNetwork(layer_sizes, activations)
            
            # Test forward pass
            x = np.random.randn(5, layer_sizes[0]).astype(np.float32)
            output = network.forward(x)
            
            assert output.shape == (5, layer_sizes[-1])
            assert not np.isnan(output).any()


class TestLossFunctions:
    """Test loss function implementations."""
    
    def test_mean_squared_error(self):
        """Test MSE loss function."""
        y_true = np.array([[1, 2, 3]], dtype=np.float32)
        y_pred = np.array([[1.1, 2.2, 2.8]], dtype=np.float32)
        
        loss = mean_squared_error(y_true, y_pred)
        
        # Manual calculation: 0.5 * mean((diff)^2)
        diff = y_true - y_pred
        expected = 0.5 * np.mean(diff**2)
        
        assert abs(loss - expected) < 1e-6
        assert loss >= 0  # MSE should be non-negative
    
    def test_mean_absolute_error(self):
        """Test MAE loss function."""
        y_true = np.array([[1, 2, 3]], dtype=np.float32)
        y_pred = np.array([[1.1, 2.2, 2.8]], dtype=np.float32)
        
        loss = mean_absolute_error(y_true, y_pred)
        
        # Manual calculation: mean(|diff|)
        diff = y_true - y_pred
        expected = np.mean(np.abs(diff))
        
        assert abs(loss - expected) < 1e-6
        assert loss >= 0  # MAE should be non-negative
    
    def test_loss_function_properties(self):
        """Test loss function mathematical properties."""
        y_true = np.array([[1, 2, 3]], dtype=np.float32)
        
        # Perfect prediction should give zero loss
        mse_perfect = mean_squared_error(y_true, y_true)
        mae_perfect = mean_absolute_error(y_true, y_true)
        
        assert abs(mse_perfect) < 1e-6
        assert abs(mae_perfect) < 1e-6
        
        # Worse prediction should give higher loss
        y_pred_bad = np.array([[2, 3, 4]], dtype=np.float32)
        mse_bad = mean_squared_error(y_true, y_pred_bad)
        mae_bad = mean_absolute_error(y_true, y_pred_bad)
        
        assert mse_bad > mse_perfect
        assert mae_bad > mae_perfect


class TestOptimizers:
    """Test optimizer implementations."""
    
    def test_sgd_optimizer(self):
        """Test SGD optimizer."""
        optimizer = SGD(learning_rate=0.01)
        
        # Test parameter update
        params = [np.array([[1, 2], [3, 4]], dtype=np.float32)]
        gradients = [np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)]
        
        updated_params = optimizer.update(params, gradients)
        
        # Check that parameters were updated in opposite direction of gradients
        expected = params[0] - 0.01 * gradients[0]
        np.testing.assert_array_almost_equal(updated_params[0], expected)
    
    def test_sgd_momentum(self):
        """Test SGD with momentum."""
        optimizer = SGD(learning_rate=0.01, momentum=0.9)
        
        params = [np.array([[1, 2]], dtype=np.float32)]
        gradients = [np.array([[0.1, 0.2]], dtype=np.float32)]
        
        # First update
        updated_params_1 = optimizer.update(params, gradients)
        
        # Second update with same gradient
        updated_params_2 = optimizer.update(updated_params_1, gradients)
        
        # With momentum, second update should be larger
        update_1 = np.abs(updated_params_1[0] - params[0])
        update_2 = np.abs(updated_params_2[0] - updated_params_1[0])
        
        assert np.all(update_2 >= update_1)  # Second update should be at least as large
    
    def test_adam_optimizer(self):
        """Test Adam optimizer."""
        optimizer = Adam(learning_rate=0.001)
        
        params = [np.array([[1, 2], [3, 4]], dtype=np.float32)]
        gradients = [np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)]
        
        updated_params = optimizer.update(params, gradients)
        
        # Check that parameters were updated
        assert not np.array_equal(updated_params[0], params[0])
        assert updated_params[0].shape == params[0].shape
    
    def test_optimizer_step_counting(self):
        """Test optimizer step counting."""
        optimizer = SGD(learning_rate=0.01)
        
        assert optimizer.step_count == 0
        
        params = [np.array([[1]], dtype=np.float32)]
        gradients = [np.array([[0.1]], dtype=np.float32)]
        
        optimizer.update(params, gradients)
        assert optimizer.step_count == 1
        
        optimizer.update(params, gradients)
        assert optimizer.step_count == 2


class TestTrainingEngine:
    """Test the training engine integration."""
    
    def test_training_engine_initialization(self):
        """Test training engine initialization."""
        network = NeuralNetwork([2, 3, 1])
        optimizer = SGD(learning_rate=0.01)
        
        trainer = TrainingEngine(network, optimizer, mean_squared_error)
        
        assert trainer.network == network
        assert trainer.optimizer == optimizer
        assert trainer.loss_function == mean_squared_error
        assert hasattr(trainer, 'gradient_function')
    
    def test_single_training_step(self):
        """Test single training step."""
        network = NeuralNetwork([2, 3, 1])
        optimizer = SGD(learning_rate=0.01)
        trainer = TrainingEngine(network, optimizer, mean_squared_error)
        
        # Create simple data
        X = np.array([[1, 2], [3, 4]], dtype=np.float32)
        y = np.array([[3], [7]], dtype=np.float32)  # y = x1 + x2
        
        # Get initial loss
        initial_loss = trainer.train_step(X, y)
        
        # Take another step
        second_loss = trainer.train_step(X, y)
        
        # Loss should generally decrease (though not guaranteed in one step)
        assert isinstance(initial_loss, float)
        assert isinstance(second_loss, float)
        assert initial_loss >= 0
        assert second_loss >= 0
    
    def test_training_loop(self):
        """Test complete training loop."""
        # Create simple linear problem
        X, y = create_sample_data(100)
        
        # Create network and trainer
        network = NeuralNetwork([2, 5, 1])
        optimizer = SGD(learning_rate=0.01)
        trainer = TrainingEngine(network, optimizer, mean_squared_error)
        
        # Train for a few epochs
        history = trainer.train(X, y, epochs=10, verbose=False, plot_progress=False)
        
        # Check that training ran
        assert 'train_loss' in history
        assert len(history['train_loss']) == 10
        
        # Check that loss generally decreased
        initial_loss = history['train_loss'][0]
        final_loss = history['train_loss'][-1]
        
        # Allow for some fluctuation, but expect overall improvement
        assert final_loss <= initial_loss * 1.1  # Loss shouldn't increase too much
    
    def test_evaluation(self):
        """Test model evaluation."""
        network = NeuralNetwork([2, 3, 1])
        optimizer = SGD(learning_rate=0.01)
        trainer = TrainingEngine(network, optimizer, mean_squared_error)
        
        # Create test data
        X = np.array([[1, 2], [3, 4]], dtype=np.float32)
        y = np.array([[3], [7]], dtype=np.float32)
        
        # Evaluate
        results = trainer.evaluate(X, y)
        
        # Check result format
        assert 'loss' in results
        assert 'mse' in results
        assert 'mae' in results
        assert 'predictions' in results
        assert 'targets' in results
        
        # Check result values
        assert results['loss'] >= 0
        assert results['mse'] >= 0
        assert results['mae'] >= 0
        assert results['predictions'].shape == y.shape
        np.testing.assert_array_equal(results['targets'], y)


class TestDataUtils:
    """Test data utility functions."""
    
    def test_data_preprocessor(self):
        """Test data preprocessing functionality."""
        # Create sample data with different scales
        X = np.array([[1, 100], [2, 200], [3, 300]], dtype=np.float32)
        
        preprocessor = DataPreprocessor(verbose=False)
        
        # Test standard normalization
        X_norm = preprocessor.normalize_features(X, method='standard')
        
        # Check that mean is approximately 0 and std is approximately 1
        assert abs(np.mean(X_norm)) < 1e-6
        assert abs(np.std(X_norm) - 1.0) < 1e-6
        
        # Test inverse transform
        X_recovered = preprocessor.inverse_transform(X_norm, method='standard')
        np.testing.assert_array_almost_equal(X, X_recovered, decimal=5)
    
    def test_data_splitter(self):
        """Test data splitting functionality."""
        # Create sample data
        X = np.random.randn(100, 3).astype(np.float32)
        y = np.random.randn(100, 1).astype(np.float32)
        
        splitter = DataSplitter(verbose=False)
        
        # Test train/val/test split
        X_train, X_val, X_test, y_train, y_val, y_test = splitter.train_val_test_split(
            X, y, train_size=0.7, val_size=0.15, test_size=0.15
        )
        
        # Check shapes
        assert X_train.shape[0] == 70
        assert X_val.shape[0] == 15
        assert X_test.shape[0] == 15
        
        # Check that all samples are accounted for
        assert X_train.shape[0] + X_val.shape[0] + X_test.shape[0] == 100
        
        # Check that features are preserved
        assert X_train.shape[1] == X.shape[1]
        assert y_train.shape[1] == y.shape[1]
    
    def test_batch_processor(self):
        """Test batch processing functionality."""
        # Create sample data
        X = np.random.randn(100, 3).astype(np.float32)
        y = np.random.randn(100, 1).astype(np.float32)
        
        batch_processor = BatchProcessor(batch_size=32, shuffle=False)
        
        # Create batches
        batches = batch_processor.create_batches(X, y)
        
        # Check number of batches
        expected_batches = int(np.ceil(100 / 32))  # 4 batches
        assert len(batches) == expected_batches
        
        # Check batch sizes
        assert batches[0][0].shape[0] == 32  # First batch full
        assert batches[-1][0].shape[0] == 100 % 32 or 32  # Last batch remainder or full
        
        # Check that all data is included
        total_samples = sum(batch[0].shape[0] for batch in batches)
        assert total_samples == 100


class TestUtilities:
    """Test utility functions."""
    
    def test_activation_functions(self):
        """Test activation function implementations."""
        x = np.array([-2, -1, 0, 1, 2], dtype=np.float32)
        
        # Test ReLU
        relu_output = ActivationFunctions.relu(x)
        expected_relu = np.array([0, 0, 0, 1, 2], dtype=np.float32)
        np.testing.assert_array_equal(relu_output, expected_relu)
        
        # Test Sigmoid
        sigmoid_output = ActivationFunctions.sigmoid(x)
        assert np.all(sigmoid_output >= 0) and np.all(sigmoid_output <= 1)
        
        # Test Tanh
        tanh_output = ActivationFunctions.tanh(x)
        assert np.all(tanh_output >= -1) and np.all(tanh_output <= 1)
        
        # Test that all outputs are finite
        for activation_name in ['relu', 'sigmoid', 'tanh', 'linear']:
            activation_func = ActivationFunctions.get_activation(activation_name)
            output = activation_func(x)
            assert np.all(np.isfinite(output))
    
    def test_math_utils(self):
        """Test mathematical utility functions."""
        # Test safe operations
        x = np.array([0, 1e-10, 1, 100], dtype=np.float32)
        
        # Safe log should not produce -inf
        safe_log_result = MathUtils.safe_log(x)
        assert np.all(np.isfinite(safe_log_result))
        
        # Safe sqrt should not produce NaN
        safe_sqrt_result = MathUtils.safe_sqrt(x)
        assert np.all(np.isfinite(safe_sqrt_result))
        assert np.all(safe_sqrt_result >= 0)
    
    def test_weight_initialization(self):
        """Test weight initialization methods."""
        n_in, n_out = 10, 5
        
        # Test Xavier initialization
        xavier_weights = MathUtils.xavier_init(n_in, n_out)
        assert xavier_weights.shape == (n_out, n_in)
        
        # Test He initialization
        he_weights = MathUtils.he_init(n_in, n_out)
        assert he_weights.shape == (n_out, n_in)
        
        # Test that both produce reasonable values
        assert np.all(np.isfinite(xavier_weights))
        assert np.all(np.isfinite(he_weights))
    
    def test_test_data_creation(self):
        """Test test data creation functions."""
        for func_type in ['linear', 'quadratic', 'sine']:
            X, y = create_test_data(func_type, n_samples=100, n_features=3)
            
            assert X.shape == (100, 3)
            assert y.shape == (100, 1)
            assert np.all(np.isfinite(X))
            assert np.all(np.isfinite(y))


class TestIntegration:
    """Integration tests for complete workflows."""
    
    def test_complete_workflow(self):
        """Test complete machine learning workflow."""
        # 1. Create data
        X, y = create_sample_data(200)
        
        # 2. Preprocess data
        preprocessor = DataPreprocessor(verbose=False)
        X_norm = preprocessor.normalize_features(X, method='standard')
        
        # 3. Split data
        splitter = DataSplitter(verbose=False)
        X_train, X_val, X_test, y_train, y_val, y_test = splitter.train_val_test_split(
            X_norm, y, train_size=0.7, val_size=0.15, test_size=0.15
        )
        
        # 4. Create network
        network = NeuralNetwork([2, 8, 4, 1], ['relu', 'relu', 'linear'])
        
        # 5. Create trainer
        optimizer = SGD(learning_rate=0.01)
        trainer = TrainingEngine(network, optimizer, mean_squared_error)
        
        # 6. Train
        history = trainer.train(X_train, y_train, epochs=50, 
                               validation_data=(X_val, y_val),
                               verbose=False, plot_progress=False)
        
        # 7. Evaluate
        results = trainer.evaluate(X_test, y_test)
        
        # Check that everything worked
        assert 'train_loss' in history
        assert len(history['train_loss']) == 50
        assert 'loss' in results
        assert results['loss'] >= 0
        
        # Check that model learned something
        initial_loss = history['train_loss'][0]
        final_loss = history['train_loss'][-1]
        assert final_loss < initial_loss  # Should have improved
    
    def test_different_optimizers(self):
        """Test training with different optimizers."""
        X, y = create_sample_data(100)
        
        optimizers = [
            SGD(learning_rate=0.01),
            Adam(learning_rate=0.001)
        ]
        
        for optimizer in optimizers:
            # Create fresh network for each optimizer
            network = NeuralNetwork([2, 5, 1])
            trainer = TrainingEngine(network, optimizer, mean_squared_error)
            
            # Train
            history = trainer.train(X, y, epochs=20, verbose=False, plot_progress=False)
            
            # Check that training worked
            assert len(history['train_loss']) == 20
            assert all(loss >= 0 for loss in history['train_loss'])
    
    def test_performance_requirements(self):
        """Test that the engine meets performance requirements."""
        # Test forward pass speed
        network = NeuralNetwork([10, 50, 20, 1])
        X = np.random.randn(1000, 10).astype(np.float32)  # Match network input size
        
        import time
        start_time = time.time()
        output = network.forward(X)
        forward_time = time.time() - start_time
        
        # Should be able to process 1000 samples in reasonable time
        assert forward_time < 1.0  # Less than 1 second
        assert output.shape == (1000, 1)
        
        # Test training speed - CREATE DATA WITH MATCHING INPUT SIZE
        X_small = np.random.randn(100, 10).astype(np.float32)  # 10 features to match network
        y_small = (2 * X_small[:, 0] + 3 * X_small[:, 1] + 1 + 
                0.1 * np.random.randn(100)).reshape(-1, 1).astype(np.float32)
        
        trainer = TrainingEngine(network, SGD(learning_rate=0.01), mean_squared_error)
        
        start_time = time.time()
        trainer.train(X_small, y_small, epochs=10, verbose=False, plot_progress=False)
        training_time = time.time() - start_time
        
        # Training should complete in reasonable time
        assert training_time < 10.0  # Less than 10 seconds


# Test configuration
@pytest.fixture
def sample_network():
    """Fixture providing a sample neural network for testing."""
    return NeuralNetwork([3, 5, 2, 1], ['relu', 'relu', 'linear'])


@pytest.fixture
def sample_data():
    """Fixture providing sample data for testing."""
    return create_sample_data(100)


# Performance benchmarks
def test_benchmark_forward_pass(sample_network, sample_data):
    """Benchmark forward pass performance."""
    X, y = sample_data
    
    # CREATE DATA WITH MATCHING INPUT SIZE (3 features to match sample_network)
    X_benchmark = np.random.randn(100, 3).astype(np.float32)
    
    # Warm up
    for _ in range(5):
        sample_network.forward(X_benchmark)
    
    # Benchmark
    import time
    start_time = time.time()
    for _ in range(100):
        sample_network.forward(X_benchmark)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / 100
    throughput = X_benchmark.shape[0] / avg_time
    
    print(f"\nBenchmark Results:")
    print(f"  Average forward pass time: {avg_time:.6f} seconds")
    print(f"  Throughput: {throughput:.0f} samples/second")
    
    # Performance assertions
    assert avg_time < 0.001  # Should be fast
    assert throughput > 10000  # Should process at least 10k samples/sec

if __name__ == "__main__":
    """
    Run tests directly with python test_nn.py
    """
    # Run all tests
    pytest.main([__file__, "-v", "--tb=short"])
