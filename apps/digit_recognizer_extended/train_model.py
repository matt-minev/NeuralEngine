"""
Enhanced Digit Recognition Training - EMNIST Digits
==================================================

Trains digit recognition using the extended EMNIST Digits dataset.
Provides 4x more training data for improved accuracy.
"""

import numpy as np
import os
import sys
import pickle
import time

# Import YOUR NeuralEngine
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from nn_core import NeuralNetwork, cross_entropy_loss
from autodiff import TrainingEngine, Adam
from utils import ActivationFunctions

# Import enhanced data loader
from load_data import prepare_emnist_data_splits

def create_enhanced_digit_model():
    """Create enhanced digit recognition model for EMNIST Digits."""
    print("🧠 Creating Enhanced Digit Recognition Model...")
    
    # Optimized architecture for larger dataset
    model = NeuralNetwork(
        layer_sizes=[784, 512, 256, 128, 10],
        activations=['relu', 'relu', 'relu', 'softmax']
    )
    
    print(f"   🎯 Enhanced Architecture: 784 → 512 → 256 → 128 → 10")
    print(f"   Dataset: EMNIST Digits (240K+ samples)")
    print(f"   Parameters: {model.count_parameters():,}")
    print(f"   Target: 97%+ accuracy with enhanced dataset")
    
    return model

def train_enhanced_digit_model():
    """Train enhanced digit recognition model."""
    print("🚀 Training Enhanced Digit Recognizer with EMNIST Digits")
    print("=" * 65)
    
    # Load enhanced dataset
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = prepare_emnist_data_splits()
    
    print(f"\n📊 Enhanced Dataset Summary:")
    print(f"   Training samples: {X_train.shape[0]:,}")
    print(f"   Validation samples: {X_val.shape[0]:,}")
    print(f"   Test samples: {X_test.shape[0]:,}")
    print(f"   Enhancement: {X_train.shape[0]/54000:.1f}x more than standard MNIST")
    
    # Create enhanced model
    model = create_enhanced_digit_model()
    
    # Enhanced training configuration
    print(f"\n🎯 ENHANCED TRAINING CONFIGURATION:")
    print(f"   Strategy: Multi-phase training for maximum accuracy")
    print(f"   Phase 1: Fast learning (0.001 LR)")
    print(f"   Phase 2: Fine-tuning (0.0005 LR)")
    print(f"   Phase 3: Final optimization (0.0001 LR)")
    
    # Phase 1: Fast learning
    print(f"\n🔥 PHASE 1: Fast Learning (Epochs 1-50)")
    optimizer_phase1 = Adam(learning_rate=0.001)
    trainer_phase1 = TrainingEngine(model, optimizer_phase1, cross_entropy_loss)
    
    start_time = time.time()
    
    history_phase1 = trainer_phase1.train(
        X_train, y_train,
        epochs=50,
        validation_data=(X_val, y_val),
        verbose=True,
        plot_progress=False
    )
    
    # Phase 2: Fine-tuning
    print(f"\n🎯 PHASE 2: Fine-tuning (Epochs 51-100)")
    optimizer_phase2 = Adam(learning_rate=0.0005)
    trainer_phase2 = TrainingEngine(model, optimizer_phase2, cross_entropy_loss)
    
    history_phase2 = trainer_phase2.train(
        X_train, y_train,
        epochs=50,
        validation_data=(X_val, y_val),
        verbose=True,
        plot_progress=False
    )
    
    # Phase 3: Final optimization
    print(f"\n🚀 PHASE 3: Final Optimization (Epochs 101-150)")
    optimizer_phase3 = Adam(learning_rate=0.0001)
    trainer_phase3 = TrainingEngine(model, optimizer_phase3, cross_entropy_loss)
    
    history_phase3 = trainer_phase3.train(
        X_train, y_train,
        epochs=50,
        validation_data=(X_val, y_val),
        verbose=True,
        plot_progress=True
    )
    
    training_time = time.time() - start_time
    
    # Combine histories
    history = {
        'train_loss': history_phase1['train_loss'] + history_phase2['train_loss'] + history_phase3['train_loss'],
        'val_loss': history_phase1['val_loss'] + history_phase2['val_loss'] + history_phase3['val_loss']
    }
    
    # Final evaluation
    print(f"\n🔬 FINAL EVALUATION ON ENHANCED DATASET")
    print("=" * 60)
    
    test_results = trainer_phase3.evaluate(X_test, y_test)
    predictions = model.forward(X_test)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_test, axis=1)
    accuracy = np.mean(predicted_classes == true_classes) * 100
    
    # Enhanced performance analysis
    confidences = np.max(predictions, axis=1) * 100
    avg_confidence = np.mean(confidences)
    high_confidence_rate = np.sum(confidences > 85) / len(confidences) * 100
    
    print(f"🏆 ENHANCED DIGIT RECOGNITION RESULTS:")
    print(f"   🎯 Test Accuracy: {accuracy:.2f}%")
    print(f"   📉 Final Loss: {test_results['loss']:.4f}")
    print(f"   🎪 Average Confidence: {avg_confidence:.1f}%")
    print(f"   🔥 High Confidence Rate: {high_confidence_rate:.1f}%")
    print(f"   ⏱️  Training Time: {training_time/60:.1f} minutes")
    print(f"   📚 Total Epochs: {len(history['train_loss'])}")
    print(f"   📊 Dataset: EMNIST Digits ({X_train.shape[0]:,} samples)")
    
    # Compare with typical MNIST performance
    print(f"\n📈 PERFORMANCE COMPARISON:")
    typical_mnist_accuracy = 97.0
    improvement = accuracy - typical_mnist_accuracy
    print(f"   Typical MNIST Accuracy: {typical_mnist_accuracy:.1f}%")
    print(f"   Enhanced Dataset Accuracy: {accuracy:.2f}%")
    print(f"   Improvement: {improvement:+.2f}%")
    
    # Success evaluation
    success_criteria = {
        'accuracy': accuracy >= 97.0,
        'confidence': avg_confidence >= 85.0,
        'high_conf_rate': high_confidence_rate >= 80.0
    }
    
    if all(success_criteria.values()):
        print(f"\n🎉 OUTSTANDING SUCCESS! Enhanced model achieved professional performance!")
        print(f"   ✅ Accuracy: {accuracy:.2f}% (target: 97%+)")
        print(f"   ✅ Confidence: {avg_confidence:.1f}% (target: 85%+)")
        print(f"   ✅ High Confidence Rate: {high_confidence_rate:.1f}% (target: 80%+)")
    else:
        print(f"\n⚠️  PERFORMANCE ANALYSIS:")
        for metric, achieved in success_criteria.items():
            status = "✅" if achieved else "❌"
            print(f"   {status} {metric}: {'PASSED' if achieved else 'NEEDS IMPROVEMENT'}")
    
    # Save enhanced model
    os.makedirs('models', exist_ok=True)
    model_data = {
        'model': model,
        'accuracy': accuracy,
        'avg_confidence': avg_confidence,
        'high_confidence_rate': high_confidence_rate,
        'history': history,
        'test_results': test_results,
        'training_time': training_time,
        'architecture': 'enhanced_digit_recognizer',
        'dataset': 'emnist_digits',
        'dataset_size': X_train.shape[0],
        'success_criteria': success_criteria
    }
    
    model_path = 'models/enhanced_digit_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"\n💾 Enhanced model saved to {model_path}")
    print(f"🎉 Enhanced digit recognition training complete!")
    
    return model, accuracy

if __name__ == "__main__":
    train_enhanced_digit_model()
