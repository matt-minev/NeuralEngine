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
from nn_core import NeuralNetwork, mean_squared_error, cross_entropy_loss  # Added cross_entropy_loss
from autodiff import TrainingEngine, Adam
from utils import ActivationFunctions


# Import data loader using YOUR DataLoader
from load_data import prepare_data_splits

def create_digit_model():
    """Create HIGH-PERFORMANCE digit recognition model for 95% accuracy."""
    print("🧠 Creating HIGH-PERFORMANCE NeuralEngine digit recognition model...")
    
    # 🎯 OPTIMIZED ARCHITECTURE for 95% accuracy
    model = NeuralNetwork(
        layer_sizes=[784, 256, 128, 64, 10],  # Deeper, wider network
        activations=['relu', 'relu', 'relu', 'softmax']  # More hidden layers
    )
    
    print(f"   🎯 HIGH-PERFORMANCE Architecture: 784 → 256 → 128 → 64 → 10")
    print(f"   Activations: {[layer.activation_name for layer in model.layers]}")
    print(f"   Total Parameters: {model.count_parameters():,}")
    print(f"   Target Accuracy: 95%+")
    
    return model

def train_with_csv_data():
    """Train for 95% accuracy using optimized configuration."""
    print("🚀 Training NeuralEngine for 95% MNIST Accuracy")
    print("=" * 55)
    
    # Load data (your existing code works perfectly)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = prepare_data_splits()
    
    print(f"\n📊 Dataset Summary:")
    print(f"   Training samples: {X_train.shape[0]:,}")
    print(f"   Validation samples: {X_val.shape[0]:,}")
    print(f"   Test samples: {X_test.shape[0]:,}")
    print(f"   Features per sample: {X_train.shape[1]}")
    print(f"   Classes: {y_train.shape[1]}")
    
    # Create OPTIMIZED model
    model = create_digit_model()
    
    # 🎯 OPTIMIZED TRAINING CONFIGURATION
    optimizer = Adam(learning_rate=0.002)  # Increased for better convergence
    trainer = TrainingEngine(model, optimizer, cross_entropy_loss)
    
    print(f"\n🔥 Starting OPTIMIZED training...")
    print(f"   🎯 Target: 95% accuracy")
    print(f"   ⚙️ Learning Rate: 0.002")
    print(f"   🔄 Epochs: 100")
    
    start_time = time.time()
    
    history = trainer.train(
        X_train, y_train,
        epochs=100,  # More epochs for high accuracy
        validation_data=(X_val, y_val),
        verbose=True,
        plot_progress=True
    )
    
    training_time = time.time() - start_time
    
    # Detailed evaluation
    print(f"\n📊 Evaluating model...")
    test_results = trainer.evaluate(X_test, y_test)
    
    predictions = model.forward(X_test)
    predicted_digits = np.argmax(predictions, axis=1)
    true_digits = np.argmax(y_test, axis=1)
    accuracy = np.mean(predicted_digits == true_digits) * 100
    
    # 🔍 DETAILED CONFIDENCE ANALYSIS
    print(f"\n🔍 Detailed Prediction Analysis:")
    confidences = np.max(predictions, axis=1) * 100
    avg_confidence = np.mean(confidences)
    high_confidence_preds = np.sum(confidences > 90)
    
    print(f"   Average Confidence: {avg_confidence:.1f}%")
    print(f"   Predictions >90% confidence: {high_confidence_preds}/{len(predictions)} ({high_confidence_preds/len(predictions)*100:.1f}%)")
    
    # Sample predictions with confidence
    sample_predictions = predictions[:10]
    sample_true = true_digits[:10]
    sample_pred = predicted_digits[:10]
    
    print(f"\n📊 Sample Predictions:")
    for i in range(10):
        confidence = np.max(sample_predictions[i]) * 100
        correct = "✅" if sample_pred[i] == sample_true[i] else "❌"
        print(f"   {correct} Sample {i+1}: True={sample_true[i]}, Pred={sample_pred[i]}, Confidence={confidence:.1f}%")
    
    # Performance summary
    print(f"\n🎯 FINAL RESULTS:")
    print(f"   🎯 Test Accuracy: {accuracy:.2f}%")
    print(f"   📉 Test Loss: {test_results['loss']:.4f}")
    print(f"   🎪 Average Confidence: {avg_confidence:.1f}%")
    print(f"   ⏱️  Training Time: {training_time:.1f} seconds")
    print(f"   🔥 Epochs Completed: {len(history['train_loss'])}")
    
    # Success check
    if accuracy >= 95.0:
        print(f"\n🎉 SUCCESS! Achieved target 95%+ accuracy!")
    elif accuracy >= 90.0:
        print(f"\n🎯 Good progress! {accuracy:.1f}% - Try training longer or adjusting architecture")
    else:
        print(f"\n⚠️  Need improvement! {accuracy:.1f}% - Check architecture and training config")
    
    # Save model
    os.makedirs('models', exist_ok=True)
    model_data = {
        'model': model,
        'accuracy': accuracy,
        'avg_confidence': avg_confidence,
        'history': history,
        'test_results': test_results,
        'training_time': training_time,
        'architecture': 'high_performance',
        'target_accuracy': 95.0
    }
    
    model_path = 'models/digit_model_optimized.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"\n💾 Model saved to {model_path}")
    
    return model, accuracy

if __name__ == "__main__":
    train_with_csv_data()
