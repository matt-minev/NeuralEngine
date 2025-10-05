"""
MNIST training script using CSV files and DataLoader.

Trains using NeuralEngine with the CSV files downloaded.
"""

import numpy as np
import os
import sys
import pickle
import time

# import complete NeuralEngine
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from nn_core import NeuralNetwork, mean_squared_error, cross_entropy_loss
from autodiff import TrainingEngine, Adam
from utils import ActivationFunctions

# import data loader using DataLoader
from load_data import prepare_data_splits


def create_digit_model():
    """Create bulletproof digit recognition model for high confidence predictions."""
    print("Creating BULLETPROOF NeuralEngine digit recognition model...")
    
    # optimized architecture with better gradient flow
    model = NeuralNetwork(
        layer_sizes=[784, 512, 256, 128, 10],  # wider layers for better capacity
        activations=['relu', 'relu', 'relu', 'softmax']
    )
    
    # improve weight initialization for better gradient flow
    for i, layer in enumerate(model.layers):
        if layer.activation_name == 'relu':
            # he initialization for relu layers (better than default)
            fan_in = layer.input_size
            layer.weights = np.random.randn(layer.output_size, layer.input_size) * np.sqrt(2.0 / fan_in)
            layer.biases = np.zeros(layer.output_size)  # zero init for biases
        elif layer.activation_name == 'softmax':
            # xavier initialization for softmax layer
            fan_in, fan_out = layer.input_size, layer.output_size
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            layer.weights = np.random.uniform(-limit, limit, (layer.output_size, layer.input_size))
            layer.biases = np.zeros(layer.output_size)
    
    print(f"  BULLETPROOF Architecture: 784 -> 512 -> 256 -> 128 -> 10")
    print(f"  Activations: {[layer.activation_name for layer in model.layers]}")
    print(f"  Total Parameters: {model.count_parameters():,}")
    print(f"  Weight Initialization: He (ReLU) + Xavier (Softmax)")
    print(f"  Target: 95%+ accuracy, 85%+ confidence")
    
    return model


def train_with_csv_data():
    """Train for high confidence predictions with advanced techniques."""
    print("Training NeuralEngine for HIGH-CONFIDENCE MNIST Recognition")
    print("=" * 65)
    
    # load data
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = prepare_data_splits()
    
    print(f"\nDataset Summary:")
    print(f"  Training samples: {X_train.shape[0]:,}")
    print(f"  Validation samples: {X_val.shape[0]:,}")
    print(f"  Test samples: {X_test.shape[0]:,}")
    
    # create optimized model
    model = create_digit_model()
    
    # advanced training configration
    print(f"\nADVANCED TRAINING CONFIGURATION:")
    print(f"  Strategy: Multi-phase training for maximum confidence")
    print(f"  Phase 1: Fast learning (higher LR)")
    print(f"  Phase 2: Fine-tuning (lower LR)")
    print(f"  Phase 3: Confidence boosting (very low LR)")
    
    # PHASE 1: fast initial learning
    print(f"\nPHASE 1: Fast Learning (Epochs 1-50)")
    optimizer_phase1 = Adam(learning_rate=0.001)
    trainer_phase1 = TrainingEngine(model, optimizer_phase1, cross_entropy_loss)
    
    start_time = time.time()
    
    history_phase1 = trainer_phase1.train(
        X_train, y_train,
        epochs=50,
        validation_data=(X_val, y_val),
        verbose=True,
        plot_progress=False  # disable plotting for intermediate phases
    )
    
    # PHASE 2: fine-tuning
    print(f"\nPHASE 2: Fine-tuning (Epochs 51-100)")
    optimizer_phase2 = Adam(learning_rate=0.0005)
    trainer_phase2 = TrainingEngine(model, optimizer_phase2, cross_entropy_loss)
    
    history_phase2 = trainer_phase2.train(
        X_train, y_train,
        epochs=50,
        validation_data=(X_val, y_val),
        verbose=True,
        plot_progress=False
    )
    
    # PHASE 3: confidence boosting
    print(f"\nPHASE 3: Confidence Boosting (Epochs 101-150)")
    optimizer_phase3 = Adam(learning_rate=0.0001)
    trainer_phase3 = TrainingEngine(model, optimizer_phase3, cross_entropy_loss)
    
    history_phase3 = trainer_phase3.train(
        X_train, y_train,
        epochs=50,
        validation_data=(X_val, y_val),
        verbose=True,
        plot_progress=True  # show final plot
    )
    
    training_time = time.time() - start_time
    
    # combine training histories
    history = {
        'train_loss': history_phase1['train_loss'] + history_phase2['train_loss'] + history_phase3['train_loss'],
        'val_loss': history_phase1['val_loss'] + history_phase2['val_loss'] + history_phase3['val_loss']
    }
    
    # comprehensive evaluation
    print(f"\nCOMPREHENSIVE MODEL EVALUATION")
    print("=" * 50)
    
    test_results = trainer_phase3.evaluate(X_test, y_test)
    predictions = model.forward(X_test)
    predicted_digits = np.argmax(predictions, axis=1)
    true_digits = np.argmax(y_test, axis=1)
    accuracy = np.mean(predicted_digits == true_digits) * 100
    
    # detailed confidence analisys
    confidences = np.max(predictions, axis=1) * 100
    avg_confidence = np.mean(confidences)
    median_confidence = np.median(confidences)
    
    # confidence buckets
    very_high_conf = np.sum(confidences >= 95)
    high_conf = np.sum(confidences >= 85)
    medium_conf = np.sum(confidences >= 70)
    low_conf = np.sum(confidences < 70)
    
    print(f"\nCONFIDENCE DISTRIBUTION:")
    print(f"  Average Confidence: {avg_confidence:.1f}%")
    print(f"  Median Confidence: {median_confidence:.1f}%")
    print(f"  Very High (95%+): {very_high_conf}/{len(predictions)} ({very_high_conf/len(predictions)*100:.1f}%)")
    print(f"  High (85-94%): {high_conf-very_high_conf}/{len(predictions)} ({(high_conf-very_high_conf)/len(predictions)*100:.1f}%)")
    print(f"  Medium (70-84%): {medium_conf-high_conf}/{len(predictions)} ({(medium_conf-high_conf)/len(predictions)*100:.1f}%)")
    print(f"  Low (<70%): {low_conf}/{len(predictions)} ({low_conf/len(predictions)*100:.1f}%)")
    
    # per-digit analysis
    print(f"\nPER-DIGIT ACCURACY & CONFIDENCE:")
    for digit in range(10):
        digit_mask = (true_digits == digit)
        digit_predictions = predictions[digit_mask]
        digit_predicted = predicted_digits[digit_mask]
        digit_true = true_digits[digit_mask]
        
        if len(digit_true) > 0:
            digit_accuracy = np.mean(digit_predicted == digit_true) * 100
            digit_avg_conf = np.mean(np.max(digit_predictions, axis=1)) * 100
            print(f"  Digit {digit}: {digit_accuracy:.1f}% accuracy, {digit_avg_conf:.1f}% avg confidence")
    
    # sample predictions with high detail
    print(f"\nDETAILED SAMPLE PREDICTIONS:")
    for i in range(15):  # more samples for better insight
        confidence = np.max(predictions[i]) * 100
        correct = "CORRECT" if predicted_digits[i] == true_digits[i] else "WRONG"
        prob_dist = predictions[i]
        top_2_indices = np.argsort(prob_dist)[-2:][::-1]
        
        print(f"  [{correct}] Sample {i+1}: True={true_digits[i]}, Pred={predicted_digits[i]}")
        print(f"    Confidence: {confidence:.1f}% | Top2: {top_2_indices[0]}({prob_dist[top_2_indices[0]]*100:.1f}%), {top_2_indices[1]}({prob_dist[top_2_indices[1]]*100:.1f}%)")
    
    # final performace summary
    print(f"\nFINAL PERFORMANCE SUMMARY:")
    print(f"  Test Accuracy: {accuracy:.2f}%")
    print(f"  Final Loss: {test_results['loss']:.4f}")
    print(f"  Average Confidence: {avg_confidence:.1f}%")
    print(f"  High Confidence Rate: {high_conf/len(predictions)*100:.1f}%")
    print(f"  Total Training Time: {training_time/60:.1f} minutes")
    print(f"  Total Epochs: {len(history['train_loss'])}")
    
    # success evaluation
    success_criteria = {
        'accuracy': accuracy >= 95.0,
        'avg_confidence': avg_confidence >= 85.0,
        'high_confidence_rate': (high_conf/len(predictions)*100) >= 80.0
    }
    
    if all(success_criteria.values()):
        print(f"\nOUTSTANDING SUCCESS! All targets achieved!")
        print(f"  Accuracy: {accuracy:.1f}% (target: 95%+)")
        print(f"  Avg Confidence: {avg_confidence:.1f}% (target: 85%+)")
        print(f"  High Confidence Rate: {high_conf/len(predictions)*100:.1f}% (target: 80%+)")
    else:
        print(f"\nPERFORMANCE ANALYSIS:")
        for metric, achieved in success_criteria.items():
            status = "PASSED" if achieved else "NEEDS IMPROVEMENT"
            print(f"  {metric}: {status}")
    
    # save comprehensive model data
    os.makedirs('models', exist_ok=True)
    model_data = {
        'model': model,
        'accuracy': accuracy,
        'avg_confidence': avg_confidence,
        'confidence_distribution': {
            'very_high': very_high_conf,
            'high': high_conf,
            'medium': medium_conf,
            'low': low_conf
        },
        'history': history,
        'test_results': test_results,
        'training_time': training_time,
        'architecture': 'bulletproof_high_confidence',
        'training_phases': 3,
        'success_criteria': success_criteria
    }
    
    model_path = 'models/digit_model_bulletproof.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"\nModel saved to {model_path}")
    
    return model, accuracy


if __name__ == "__main__":
    train_with_csv_data()
