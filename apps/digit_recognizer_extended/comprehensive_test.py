"""
Comprehensive EMNIST Digits Test Suite
=====================================

Tests enhanced digit recognition model on EMNIST Digits dataset.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import sys
from sklearn.metrics import classification_report, confusion_matrix
import time

# Import YOUR NeuralEngine
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from nn_core import cross_entropy_loss
from load_data import load_test_data

def load_enhanced_model(model_path='models/enhanced_digit_model.pkl'):
    """Load enhanced digit recognition model."""
    print(f"🔍 Loading enhanced model from {model_path}...")
    
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        model = model_data['model']
        print(f"✅ Enhanced model loaded successfully!")
        print(f"   Training Accuracy: {model_data.get('accuracy', 'N/A'):.2f}%")
        print(f"   Dataset: {model_data.get('dataset', 'unknown')}")
        print(f"   Training Samples: {model_data.get('dataset_size', 'unknown'):,}")
        
        return model, model_data
    
    except FileNotFoundError:
        print(f"❌ Model file not found: {model_path}")
        return None, None

def comprehensive_enhanced_evaluation(model, X_test, y_test):
    """Comprehensive evaluation on enhanced dataset."""
    print(f"\n🔬 COMPREHENSIVE ENHANCED DIGIT RECOGNITION EVALUATION")
    print("=" * 70)
    
    start_time = time.time()
    
    # Get predictions
    predictions = model.forward(X_test)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_test, axis=1)
    
    # Calculate metrics
    accuracy = np.mean(predicted_classes == true_classes) * 100
    test_loss = cross_entropy_loss(y_test, predictions)
    
    # Enhanced confidence analysis
    confidences = np.max(predictions, axis=1) * 100
    avg_confidence = np.mean(confidences)
    
    prediction_time = time.time() - start_time
    
    print(f"📊 ENHANCED DATASET PERFORMANCE:")
    print(f"   🎯 Test Accuracy: {accuracy:.2f}%")
    print(f"   📉 Test Loss: {test_loss:.4f}")
    print(f"   🎪 Average Confidence: {avg_confidence:.1f}%")
    print(f"   📊 Test Samples: {len(X_test):,} (vs 10,000 in MNIST)")
    print(f"   ⏱️  Processing Time: {prediction_time:.3f} seconds")
    
    # Performance comparison
    print(f"\n📈 PERFORMANCE BENCHMARKS:")
    mnist_baseline = 97.0
    performance_ratio = accuracy / mnist_baseline
    print(f"   MNIST Baseline: {mnist_baseline:.1f}%")
    print(f"   Enhanced Performance: {accuracy:.2f}%")
    print(f"   Performance Ratio: {performance_ratio:.3f}")
    
    return {
        'accuracy': accuracy,
        'test_loss': test_loss,
        'avg_confidence': avg_confidence,
        'predictions': predictions,
        'predicted_classes': predicted_classes,
        'true_classes': true_classes,
        'confidences': confidences
    }

def main():
    """Main enhanced testing function."""
    print("🚀 COMPREHENSIVE ENHANCED DIGIT RECOGNITION TEST")
    print("=" * 60)
    
    # Load enhanced model
    model, model_data = load_enhanced_model()
    if model is None:
        return
    
    # Load test data
    X_test, y_test = load_test_data()
    
    # Run evaluation
    results = comprehensive_enhanced_evaluation(model, X_test, y_test)
    
    print(f"\n🎉 ENHANCED TESTING COMPLETE!")
    print(f"   Final Accuracy: {results['accuracy']:.2f}%")
    print(f"   Average Confidence: {results['avg_confidence']:.1f}%")

if __name__ == "__main__":
    main()
