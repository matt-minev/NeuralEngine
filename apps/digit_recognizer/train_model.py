"""
MNIST Training Script - Using YOUR CSV Files & DataLoader
======================================================

Trains using YOUR NeuralEngine with the CSV files you downloaded.
"""

import numpy as np
import os
import sys
import pickle
import time

# Import YOUR complete NeuralEngine
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from nn_core import NeuralNetwork, mean_squared_error
from autodiff import TrainingEngine, Adam
from utils import ActivationFunctions

# Import data loader using YOUR DataLoader
from load_data import prepare_data_splits

def create_digit_model():
    """Create digit recognition model using YOUR NeuralNetwork."""
    print("🧠 Creating NeuralEngine digit recognition model...")
    
    # Use YOUR NeuralNetwork with existing activations
    model = NeuralNetwork(
        layer_sizes=[784, 128, 64, 10],
        activations=['relu', 'relu', 'softmax']  # Using your existing softmax
    )
    
    print(f"   Architecture: 784 → 128 → 64 → 10")
    print(f"   Activations: {[layer.activation_name for layer in model.layers]}")
    print(f"   Total Parameters: {model.count_parameters():,}")
    
    return model

def train_with_csv_data():
    """Train using YOUR CSV files and complete NeuralEngine pipeline."""
    print("🚀 Training NeuralEngine with YOUR MNIST CSV Files")
    print("=" * 55)
    
    # Load data using YOUR DataLoader and CSV files
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = prepare_data_splits()
    
    print(f"\n📊 Dataset Summary:")
    print(f"   Training samples: {X_train.shape[0]:,}")
    print(f"   Validation samples: {X_val.shape[0]:,}")
    print(f"   Test samples: {X_test.shape[0]:,}")
    print(f"   Features per sample: {X_train.shape[1]}")
    print(f"   Classes: {y_train.shape[1]}")
    
    # Create model using YOUR NeuralNetwork
    model = create_digit_model()
    
    # Use YOUR Adam optimizer
    optimizer = Adam(learning_rate=0.001)
    
    # Use YOUR TrainingEngine
    trainer = TrainingEngine(model, optimizer, mean_squared_error)
    
    # Train using YOUR complete pipeline
    print(f"\n🔥 Starting training with YOUR NeuralEngine...")
    start_time = time.time()
    
    history = trainer.train(
        X_train, y_train,
        epochs=20,
        validation_data=(X_val, y_val),
        verbose=True,
        plot_progress=True
    )
    
    training_time = time.time() - start_time
    
    # Evaluate using YOUR evaluation system
    print(f"\n📊 Evaluating model...")
    test_results = trainer.evaluate(X_test, y_test)
    
    # Calculate accuracy
    predictions = model.forward(X_test)
    predicted_digits = np.argmax(predictions, axis=1)
    true_digits = np.argmax(y_test, axis=1)
    accuracy = np.mean(predicted_digits == true_digits) * 100
    
    print(f"\n✅ Training Results:")
    print(f"   🎯 Test Accuracy: {accuracy:.2f}%")
    print(f"   📉 Test Loss: {test_results['loss']:.6f}")
    print(f"   ⏱️  Training Time: {training_time:.1f} seconds")
    print(f"   🔥 Epochs Completed: {len(history['train_loss'])}")
    
    # Save model using YOUR approach
    os.makedirs('models', exist_ok=True)
    model_data = {
        'model': model,
        'accuracy': accuracy,
        'history': history,
        'test_results': test_results,
        'training_time': training_time,
        'data_source': 'mnist_csv',
        'neuralengine_version': '1.0.0'
    }
    
    model_path = 'models/digit_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"\n💾 Model saved to {model_path}")
    print(f"🎉 Training complete using YOUR NeuralEngine!")
    print(f"   • Used YOUR DataLoader for CSV files")
    print(f"   • Used YOUR DataPreprocessor for normalization")
    print(f"   • Used YOUR TrainingEngine for optimization")
    print(f"   • Used YOUR NeuralNetwork architecture")
    
    return model, accuracy

if __name__ == "__main__":
    train_with_csv_data()
