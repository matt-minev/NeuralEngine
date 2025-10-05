"""
Test suite for neural network engine.

Run with: pytest tests/test_nn.py -v
"""

import pytest
import numpy as np
import sys
import os

# add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# import modules
from nn_core import Layer, NeuralNetwork, mean_squared_error, mean_absolute_error, create_sample_data
from autodiff import SGD, Adam, TrainingEngine
from data_utils import DataLoader, DataPreprocessor, DataSplitter, BatchProcessor
from utils import ActivationFunctions, MathUtils, NetworkVisualizer, PerformanceMonitor, create_test_data, print_network_summary


class TestLayer:
    """Test Layer class."""

    def test_layer_initialization(self):
        """Test layer init with different params."""
        layer = Layer(input_size=3, output_size=5, activation='relu')

        assert layer.input_size == 3
        assert layer.output_size == 5
        assert layer.activation_name == 'relu'
        assert layer.weights.shape == (5, 3)
        assert layer.biases.shape == (5,)

    def test_layer_forward_pass(self):
        """Test forward prop through layer."""
        layer = Layer(input_size=3, output_size=2, activation='linear')

        # set known weights/biases for testing
        layer.weights = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        layer.biases = np.array([0.1, 0.2], dtype=np.float32)

        # test single sample
        x = np.array([1, 2, 3], dtype=np.float32)
        output = layer.forward(x)

        expected = np.array([1*1 + 2*2 + 3*3 + 0.1, 1*4 + 2*5 + 3*6 + 0.2])
        np.testing.assert_array_almost_equal(output, expected, decimal=5)

    def test_layer_batch_forward(self):
        """Test forward prop with batch input."""
        layer = Layer(input_size=2, output_size=3, activation='relu')

        x_batch = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
        output = layer.forward(x_batch)

        assert output.shape == (3, 3)
        assert np.all(output >= 0)  # relu should be non-negative

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
        """Test param getting and setting."""
        layer = Layer(input_size=2, output_size=3, activation='relu')

        # get params
        weights, biases = layer.get_parameters()
        assert weights.shape == (3, 2)
        assert biases.shape == (3,)

        # set new params
        new_weights = np.ones((3, 2), dtype=np.float32)
        new_biases = np.zeros((3,), dtype=np.float32)
        layer.set_parameters(new_weights, new_biases)

        # verify params were set
        weights, biases = layer.get_parameters()
        np.testing.assert_array_equal(weights, new_weights)
        np.testing.assert_array_equal(biases, new_biases)


class TestNeuralNetwork:
    """Test NeuralNetwork class."""

    def test_network_initialization(self):
        """Test network init."""
        layer_sizes = [3, 5, 2, 1]
        network = NeuralNetwork(layer_sizes)

        assert network.layer_sizes == layer_sizes
        assert network.num_layers == 3
        assert len(network.layers) == 3
        assert network.count_parameters() > 0

    def test_network_forward_pass(self):
        """Test forward prop through entire network."""
        network = NeuralNetwork([2, 3, 1], ['relu', 'linear'])

        x = np.array([1, 2], dtype=np.float32)
        output = network.forward(x)

        assert output.shape == (1,)
        assert not np.isnan(output).any()
        assert not np.isinf(output).any()

    def test_network_batch_processing(self):
        """Test network with batch inputs."""
        network = NeuralNetwork([3, 4, 2], ['relu', 'linear'])

        x_batch = np.random.randn(10, 3).astype(np.float32)
        output = network.forward(x_batch)

        assert output.shape == (10, 2)
        assert not np.isnan(output).any()
        assert not np.isinf(output).any()

    def test_parameter_management(self):
        """Test param getting and setting."""
        network = NeuralNetwork([2, 3, 1])

        # get all params
        params = network.get_all_parameters()
        assert len(params) == 4  # 2 layers x 2 param types

        # count params
        param_count = network.count_parameters()
        manual_count = sum(p.size for p in params)
        assert param_count == manual_count

        # test param setting
        network.set_all_parameters(params)  # shouldn't raise error

    def test_network_architectures(self):
        """Test different architectures."""
        architectures = [
            ([1, 1], ['linear']),
            ([2, 5, 3, 1], ['relu', 'relu', 'linear']),
            ([4, 10, 8, 5, 2], ['relu', 'relu', 'relu', 'linear'])
        ]

        for layer_sizes, activations in architectures:
            network = NeuralNetwork(layer_sizes, activations)

            x = np.random.randn(5, layer_sizes[0]).astype(np.float32)
            output = network.forward(x)

            assert output.shape == (5, layer_sizes[-1])
            assert not np.isnan(output).any()


class TestLossFunctions:
    """Test loss functions."""

    def test_mean_squared_error(self):
        """Test MSE loss."""
        y_true = np.array([[1, 2, 3]], dtype=np.float32)
        y_pred = np.array([[1.1, 2.2, 2.8]], dtype=np.float32)

        loss = mean_squared_error(y_true, y_pred)

        # manual calculation
        diff = y_true - y_pred
        expected = 0.5 * np.mean(diff**2)

        assert abs(loss - expected) < 1e-6
        assert loss >= 0

    def test_mean_absolute_error(self):
        """Test MAE loss."""
        y_true = np.array([[1, 2, 3]], dtype=np.float32)
        y_pred = np.array([[1.1, 2.2, 2.8]], dtype=np.float32)

        loss = mean_absolute_error(y_true, y_pred)

        # manual calculation
        diff = y_true - y_pred
        expected = np.mean(np.abs(diff))

        assert abs(loss - expected) < 1e-6
        assert loss >= 0

    def test_loss_function_properties(self):
        """Test loss function mathamatical properties."""
        y_true = np.array([[1, 2, 3]], dtype=np.float32)

        # perfect prediction should give zero loss
        mse_perfect = mean_squared_error(y_true, y_true)
        mae_perfect = mean_absolute_error(y_true, y_true)

        assert abs(mse_perfect) < 1e-6
        assert abs(mae_perfect) < 1e-6

        # worse prediction should give higher loss
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

        params = [np.array([[1, 2], [3, 4]], dtype=np.float32)]
        gradients = [np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)]

        updated_params = optimizer.update(params, gradients)

        # check that params moved opposite to gradients
        expected = params[0] - 0.01 * gradients[0]
        np.testing.assert_array_almost_equal(updated_params[0], expected)

    def test_sgd_momentum(self):
        """Test SGD with momentum."""
        optimizer = SGD(learning_rate=0.01, momentum=0.9)

        params = [np.array([[1, 2]], dtype=np.float32)]
        gradients = [np.array([[0.1, 0.2]], dtype=np.float32)]

        # first update
        updated_params_1 = optimizer.update(params, gradients)

        # second update with same gradient
        updated_params_2 = optimizer.update(updated_params_1, gradients)

        # with momentum, second update should be larger
        update_1 = np.abs(updated_params_1[0] - params[0])
        update_2 = np.abs(updated_params_2[0] - updated_params_1[0])

        assert np.all(update_2 >= update_1)

    def test_adam_optimizer(self):
        """Test Adam optimizer."""
        optimizer = Adam(learning_rate=0.001)

        params = [np.array([[1, 2], [3, 4]], dtype=np.float32)]
        gradients = [np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)]

        updated_params = optimizer.update(params, gradients)

        # check that params were updated
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
    """Test training engine integration."""

    def test_training_engine_initialization(self):
        """Test training engine init."""
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

        # create simple data
        X = np.array([[1, 2], [3, 4]], dtype=np.float32)
        y = np.array([[3], [7]], dtype=np.float32)

        # get initial loss
        initial_loss = trainer.train_step(X, y)

        # take another step
        second_loss = trainer.train_step(X, y)

        # loss should be valid
        assert isinstance(initial_loss, float)
        assert isinstance(second_loss, float)
        assert initial_loss >= 0
        assert second_loss >= 0

    def test_training_loop(self):
        """Test complete training loop."""
        X, y = create_sample_data(100)

        network = NeuralNetwork([2, 5, 1])
        optimizer = SGD(learning_rate=0.01)
        trainer = TrainingEngine(network, optimizer, mean_squared_error)

        # train for a few epochs
        history = trainer.train(X, y, epochs=10, verbose=False, plot_progress=False)

        # check that training ran
        assert 'train_loss' in history
        assert len(history['train_loss']) == 10

        # check that loss generally decreased
        initial_loss = history['train_loss'][0]
        final_loss = history['train_loss'][-1]

        # allow for some fluctuation
        assert final_loss <= initial_loss * 1.1

    def test_evaluation(self):
        """Test model evaluation."""
        network = NeuralNetwork([2, 3, 1])
        optimizer = SGD(learning_rate=0.01)
        trainer = TrainingEngine(network, optimizer, mean_squared_error)

        X = np.array([[1, 2], [3, 4]], dtype=np.float32)
        y = np.array([[3], [7]], dtype=np.float32)

        results = trainer.evaluate(X, y)

        # check result format
        assert 'loss' in results
        assert 'mse' in results
        assert 'mae' in results
        assert 'predictions' in results
        assert 'targets' in results

        # check result values
        assert results['loss'] >= 0
        assert results['mse'] >= 0
        assert results['mae'] >= 0
        assert results['predictions'].shape == y.shape
        np.testing.assert_array_equal(results['targets'], y)


class TestDataUtils:
    """Test data utilities."""

    def test_data_preprocessor(self):
        """Test data preprocessing."""
        # create sample data with different scales
        X = np.array([[1, 100], [2, 200], [3, 300]], dtype=np.float32)

        preprocessor = DataPreprocessor(verbose=False)

        # test standard normalization
        X_norm = preprocessor.normalize_features(X, method='standard')

        # check that mean is ~0 and std is ~1
        assert abs(np.mean(X_norm)) < 1e-6
        assert abs(np.std(X_norm) - 1.0) < 1e-6

        # test inverse transform
        X_recovered = preprocessor.inverse_transform(X_norm, method='standard')
        np.testing.assert_array_almost_equal(X, X_recovered, decimal=5)

    def test_data_splitter(self):
        """Test data splitting."""
        X = np.random.randn(100, 3).astype(np.float32)
        y = np.random.randn(100, 1).astype(np.float32)

        splitter = DataSplitter(verbose=False)

        # test train/val/test split
        X_train, X_val, X_test, y_train, y_val, y_test = splitter.train_val_test_split(
            X, y, train_size=0.7, val_size=0.15, test_size=0.15
        )

        # check shapes
        assert X_train.shape[0] == 70
        assert X_val.shape[0] == 15
        assert X_test.shape[0] == 15

        # check that all samples accounted for
        assert X_train.shape[0] + X_val.shape[0] + X_test.shape[0] == 100

        # check that features preserved
        assert X_train.shape[1] == X.shape[1]
        assert y_train.shape[1] == y.shape[1]

    def test_batch_processor(self):
        """Test batch processing."""
        X = np.random.randn(100, 3).astype(np.float32)
        y = np.random.randn(100, 1).astype(np.float32)

        batch_processor = BatchProcessor(batch_size=32, shuffle=False)

        batches = batch_processor.create_batches(X, y)

        # check number of batches
        expected_batches = int(np.ceil(100 / 32))  # 4 batches
        assert len(batches) == expected_batches

        # check batch sizes
        assert batches[0][0].shape[0] == 32  # first batch full
        assert batches[-1][0].shape[0] == 100 % 32 or 32  # last batch

        # check that all data included
        total_samples = sum(batch[0].shape[0] for batch in batches)
        assert total_samples == 100


class TestUtilities:
    """Test utility functions."""

    def test_activation_functions(self):
        """Test activation implementations."""
        x = np.array([-2, -1, 0, 1, 2], dtype=np.float32)

        # test relu
        relu_output = ActivationFunctions.relu(x)
        expected_relu = np.array([0, 0, 0, 1, 2], dtype=np.float32)
        np.testing.assert_array_equal(relu_output, expected_relu)

        # test sigmoid
        sigmoid_output = ActivationFunctions.sigmoid(x)
        assert np.all(sigmoid_output >= 0) and np.all(sigmoid_output <= 1)

        # test tanh
        tanh_output = ActivationFunctions.tanh(x)
        assert np.all(tanh_output >= -1) and np.all(tanh_output <= 1)

        # test that all outputs are finite
        for activation_name in ['relu', 'sigmoid', 'tanh', 'linear']:
            activation_func = ActivationFunctions.get_activation(activation_name)
            output = activation_func(x)
            assert np.all(np.isfinite(output))

    def test_math_utils(self):
        """Test math utilities."""
        x = np.array([0, 1e-10, 1, 100], dtype=np.float32)

        # safe log should not produce -inf
        safe_log_result = MathUtils.safe_log(x)
        assert np.all(np.isfinite(safe_log_result))

        # safe sqrt should not produce nan
        safe_sqrt_result = MathUtils.safe_sqrt(x)
        assert np.all(np.isfinite(safe_sqrt_result))
        assert np.all(safe_sqrt_result >= 0)

    def test_weight_initialization(self):
        """Test weight init methods."""
        n_in, n_out = 10, 5

        # test xavier
        xavier_weights = MathUtils.xavier_init(n_in, n_out)
        assert xavier_weights.shape == (n_out, n_in)

        # test he
        he_weights = MathUtils.he_init(n_in, n_out)
        assert he_weights.shape == (n_out, n_in)

        # test that both produce reasonable values
        assert np.all(np.isfinite(xavier_weights))
        assert np.all(np.isfinite(he_weights))

    def test_test_data_creation(self):
        """Test test data creation."""
        for func_type in ['linear', 'quadratic', 'sine']:
            X, y = create_test_data(func_type, n_samples=100, n_features=3)

            assert X.shape == (100, 3)
            assert y.shape == (100, 1)
            assert np.all(np.isfinite(X))
            assert np.all(np.isfinite(y))


class TestIntegration:
    """Integration tests for complete workflowz."""

    def test_complete_workflow(self):
        """Test complete ML workflow."""
        # create data
        X, y = create_sample_data(200)

        # preprocess
        preprocessor = DataPreprocessor(verbose=False)
        X_norm = preprocessor.normalize_features(X, method='standard')

        # split
        splitter = DataSplitter(verbose=False)
        X_train, X_val, X_test, y_train, y_val, y_test = splitter.train_val_test_split(
            X_norm, y, train_size=0.7, val_size=0.15, test_size=0.15
        )

        # create network
        network = NeuralNetwork([2, 8, 4, 1], ['relu', 'relu', 'linear'])

        # create trainer
        optimizer = SGD(learning_rate=0.01)
        trainer = TrainingEngine(network, optimizer, mean_squared_error)

        # train
        history = trainer.train(X_train, y_train, epochs=50, 
                               validation_data=(X_val, y_val),
                               verbose=False, plot_progress=False)

        # evaluate
        results = trainer.evaluate(X_test, y_test)

        # check that everything worked
        assert 'train_loss' in history
        assert len(history['train_loss']) == 50
        assert 'loss' in results
        assert results['loss'] >= 0

        # check that model learned something
        initial_loss = history['train_loss'][0]
        final_loss = history['train_loss'][-1]
        assert final_loss < initial_loss

    def test_different_optimizers(self):
        """Test training with different optimizers."""
        X, y = create_sample_data(100)

        optimizers = [
            SGD(learning_rate=0.01),
            Adam(learning_rate=0.001)
        ]

        for optimizer in optimizers:
            # fresh network for each optimizer
            network = NeuralNetwork([2, 5, 1])
            trainer = TrainingEngine(network, optimizer, mean_squared_error)

            # train
            history = trainer.train(X, y, epochs=20, verbose=False, plot_progress=False)

            # check that training worked
            assert len(history['train_loss']) == 20
            assert all(loss >= 0 for loss in history['train_loss'])

    def test_performance_requirements(self):
        """Test that engine meets performance requirements."""
        # test forward pass speed
        network = NeuralNetwork([10, 50, 20, 1])
        X = np.random.randn(1000, 10).astype(np.float32)

        import time
        start_time = time.time()
        output = network.forward(X)
        forward_time = time.time() - start_time

        # should process 1000 samples quickly
        assert forward_time < 1.0  # less than 1 second
        assert output.shape == (1000, 1)

        # test training speed
        X_small = np.random.randn(100, 10).astype(np.float32)
        y_small = (2 * X_small[:, 0] + 3 * X_small[:, 1] + 1 + 
                0.1 * np.random.randn(100)).reshape(-1, 1).astype(np.float32)

        trainer = TrainingEngine(network, SGD(learning_rate=0.01), mean_squared_error)

        start_time = time.time()
        trainer.train(X_small, y_small, epochs=10, verbose=False, plot_progress=False)
        training_time = time.time() - start_time

        # training should complete quickly
        assert training_time < 10.0


# fixtures
@pytest.fixture
def sample_network():
    """Fixture providing sample network."""
    return NeuralNetwork([3, 5, 2, 1], ['relu', 'relu', 'linear'])


@pytest.fixture
def sample_data():
    """Fixture providing sample data."""
    return create_sample_data(100)


# performance benchmark
def test_benchmark_forward_pass(sample_network, sample_data):
    """Benchmark forward pass performance."""
    X, y = sample_data

    # create data with matching input size
    X_benchmark = np.random.randn(100, 3).astype(np.float32)

    # warm up
    for _ in range(5):
        sample_network.forward(X_benchmark)

    # benchmark
    import time
    start_time = time.time()
    for _ in range(100):
        sample_network.forward(X_benchmark)
    end_time = time.time()

    avg_time = (end_time - start_time) / 100
    throughput = X_benchmark.shape[0] / avg_time

    print(f"\nBenchmark Results:")
    print(f"  Avg forward pass time: {avg_time:.6f} seconds")
    print(f"  Throughput: {throughput:.0f} samples/second")

    # performance assertions
    assert avg_time < 0.001
    assert throughput > 10000


if __name__ == "__main__":
    """Run tests directly with python test_nn.py"""
    pytest.main([__file__, "-v", "--tb=short"])
