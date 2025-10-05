"""
Universal character recognition training script.

Trains NeuralEngine for complete character recognition using EMNIST ByClass.
Handles 62 classes (0-9, A-Z, a-z) with proven training pipeline.
"""

import numpy as np
import os
import sys
import pickle
import time

# import complete NeuralEngine
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from nn_core import NeuralNetwork, cross_entropy_loss
from autodiff import TrainingEngine, Adam
from utils import ActivationFunctions

# import data loader
from load_data import prepare_universal_data_splits


def create_universal_model():
    """Create universal character recogntion model using NeuralNetwork."""
    print("Creating UNIVERSAL NeuralEngine character recognition model...")

    # universal architecture for 62 classes
    model = NeuralNetwork(
        layer_sizes=[784, 512, 256, 128, 62],  # 62 classes for complete recognition
        activations=['relu', 'relu', 'relu', 'softmax']
    )

    print(f"  UNIVERSAL Architecture: 784 -> 512 -> 256 -> 128 -> 62")
    print(f"  Classes: 62 total (0-9, A-Z, a-z)")
    print(f"  Activations: {[layer.activation_name for layer in model.layers]}")
    print(f"  Total Parameters: {model.count_parameters():,}")
    print(f"  Target: Professional-grade character recogntion")

    return model


def train_universal_model():
    """Train universal character recogntion model with advanced techniques."""
    print("Training UNIVERSAL NeuralEngine Character Recognizer")
    print("=" * 65)

    # load EMNIST ByClass data
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = prepare_universal_data_splits()

    print(f"\nUniversal Dataset Summary:")
    print(f"  Training samples: {X_train.shape[0]:,}")
    print(f"  Validation samples: {X_val.shape[0]:,}")
    print(f"  Test samples: {X_test.shape[0]:,}")
    print(f"  Features per sample: {X_train.shape[1]}")
    print(f"  Classes: {y_train.shape[1]} (0-9, A-Z, a-z)")

    # create model
    model = create_universal_model()

    # multi-phase training for 62-class optimizaton
    print(f"\nADVANCED TRAINING CONFIGURATION:")
    print(f"  Strategy: Multi-phase training for maximum performence")
    print(f"  Phase 1: Fast learning (epochs 1-50)")
    print(f"  Phase 2: Fine-tuning (epochs 51-100)")
    print(f"  Phase 3: Optimization (epochs 101-150)")

    # phase 1: initial learning
    print(f"\nPHASE 1: Initial Learning (Epochs 1-50)")
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

    # phase 2: fine-tuning
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

    # phase 3: optimization
    print(f"\nPHASE 3: Final Optimization (Epochs 101-150)")
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

    # combine training histories
    history = {
        'train_loss': history_phase1['train_loss'] + history_phase2['train_loss'] + history_phase3['train_loss'],
        'val_loss': history_phase1['val_loss'] + history_phase2['val_loss'] + history_phase3['val_loss']
    }

    # comprehensive evaluation
    print(f"\nCOMPREHENSIVE UNIVERSAL MODEL EVALUATION")
    print("=" * 60)

    test_results = trainer_phase3.evaluate(X_test, y_test)
    predictions = model.forward(X_test)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_test, axis=1)
    accuracy = np.mean(predicted_classes == true_classes) * 100

    # character type analisys
    from load_data import get_character_type, index_to_character

    print(f"\nCHARACTER TYPE PERFORMANCE:")

    # digits analysis (0-9)
    digit_mask = true_classes < 10
    if np.any(digit_mask):
        digit_accuracy = np.mean(predicted_classes[digit_mask] == true_classes[digit_mask]) * 100
        print(f"  Digits (0-9): {digit_accuracy:.1f}% accuracy")

    # uppercase analysis (A-Z)
    upper_mask = (true_classes >= 10) & (true_classes < 36)
    if np.any(upper_mask):
        upper_accuracy = np.mean(predicted_classes[upper_mask] == true_classes[upper_mask]) * 100
        print(f"  Uppercase (A-Z): {upper_accuracy:.1f}% accuracy")

    # lowercase analysis (a-z)
    lower_mask = true_classes >= 36
    if np.any(lower_mask):
        lower_accuracy = np.mean(predicted_classes[lower_mask] == true_classes[lower_mask]) * 100
        print(f"  Lowercase (a-z): {lower_accuracy:.1f}% accuracy")

    # confidence analisys
    confidences = np.max(predictions, axis=1) * 100
    avg_confidence = np.mean(confidences)

    print(f"\nCONFIDENCE ANALYSIS:")
    print(f"  Average Confidence: {avg_confidence:.1f}%")
    print(f"  High Confidence (>80%): {np.sum(confidences > 80)}/{len(confidences)} ({np.sum(confidences > 80)/len(confidences)*100:.1f}%)")

    # sample predictions
    print(f"\nSAMPLE PREDICTIONS:")
    sample_indices = np.random.choice(len(predictions), 10, replace=False)
    for i, idx in enumerate(sample_indices):
        true_char = index_to_character(true_classes[idx])
        pred_char = index_to_character(predicted_classes[idx])
        confidence = confidences[idx]
        correct = "CORRECT" if pred_char == true_char else "WRONG"

        print(f"  {correct} Sample {i+1}: True='{true_char}', Pred='{pred_char}', Confidence={confidence:.1f}%")

    # final results
    print(f"\nUNIVERSAL MODEL PERFORMANCE:")
    print(f"  Overall Accuracy: {accuracy:.2f}%")
    print(f"  Final Loss: {test_results['loss']:.4f}")
    print(f"  Average Confidence: {avg_confidence:.1f}%")
    print(f"  Total Training Time: {training_time/60:.1f} minutes")
    print(f"  Total Epochs: {len(history['train_loss'])}")

    # success evaluation
    success_criteria = {
        'accuracy': accuracy >= 75.0,  # 75%+ is excellent for 62 classes
        'avg_confidence': avg_confidence >= 70.0,
        'training_completed': len(history['train_loss']) == 150
    }

    if all(success_criteria.values()):
        print(f"\nOUTSTANDING SUCCESS! Universal character recognizer achieved professional performence!")
        print(f"  Accuracy: {accuracy:.1f}% (target: 75%+)")
        print(f"  Avg Confidence: {avg_confidence:.1f}% (target: 70%+)")
        print(f"  Training: Complete (150 epochs)")
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
        'character_type_accuracies': {
            'digits': digit_accuracy if np.any(digit_mask) else 0,
            'uppercase': upper_accuracy if np.any(upper_mask) else 0,
            'lowercase': lower_accuracy if np.any(lower_mask) else 0
        },
        'history': history,
        'test_results': test_results,
        'training_time': training_time,
        'architecture': 'universal_character_recognizer',
        'dataset': 'emnist_byclass',
        'classes': 62,
        'success_criteria': success_criteria
    }

    model_path = 'models/universal_character_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)

    print(f"\nUniversal model saved to {model_path}")
    print(f"Training complete! NeuralEngine now recognizes ALL characters!")

    return model, accuracy


if __name__ == "__main__":
    train_universal_model()
