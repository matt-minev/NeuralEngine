"""
Comprehensive universal character recognition test suite.

Tests trained universal character model on EMNIST ByClass test data with detailed statistics,
confidence analysis, and multiple visualization charts for all 62 classes.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import sys
from sklearn.metrics import classification_report, confusion_matrix
import time

# import neural engine
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from nn_core import cross_entropy_loss
from load_data import load_universal_test_data, index_to_character, character_to_index, get_character_type

# setup plotting style
plt.style.use('default')
sns.set_palette("husl")


def load_trained_universal_model(model_path='models/universal_character_model.pkl'):
    """Load the trained universal character model and its metadata."""
    print(f"Loading trained universal model from {model_path}...")

    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)

        model = model_data['model']
        print(f"Universal model loaded succesfully!")
        print(f"  Training Accuracy: {model_data.get('accuracy', 'N/A'):.2f}%")
        print(f"  Architecture: {model.layer_sizes}")
        print(f"  Parameters: {model.count_parameters():,}")
        print(f"  Classes: 62 (0-9, A-Z, a-z)")

        return model, model_data

    except FileNotFoundError:
        print(f"Model file not found: {model_path}")
        print("  Please train a model first by running train_model.py")
        return None, None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None


def comprehensive_universal_test_evaluation(model, X_test, y_test):
    """Perform comprehensive evaluation on universal character test data."""
    print(f"\nCOMPREHENSIVE UNIVERSAL CHARACTER TEST EVALUATION")
    print("=" * 70)

    start_time = time.time()

    # get predictions from model
    predictions = model.forward(X_test)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_test, axis=1)

    # calculate basic metrics
    accuracy = np.mean(predicted_classes == true_classes) * 100
    test_loss = cross_entropy_loss(y_test, predictions)

    # confidence analisys
    confidences = np.max(predictions, axis=1) * 100
    avg_confidence = np.mean(confidences)
    median_confidence = np.median(confidences)

    # confidence buckets for categorization
    very_high_conf = np.sum(confidences >= 95)
    high_conf = np.sum(confidences >= 85)
    medium_conf = np.sum(confidences >= 70)
    low_conf = np.sum(confidences < 70)

    prediction_time = time.time() - start_time

    # print summary stats
    print(f"OVERALL PERFORMANCE:")
    print(f"  Test Accuracy: {accuracy:.2f}%")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Prediction Time: {prediction_time:.3f} seconds")
    print(f"  Throughput: {len(X_test)/prediction_time:.0f} samples/sec")

    print(f"\nCONFIDENCE DISTRIBUTION:")
    print(f"  Average: {avg_confidence:.1f}%")
    print(f"  Median: {median_confidence:.1f}%")
    print(f"  Very High (95%+): {very_high_conf:,} ({very_high_conf/len(predictions)*100:.1f}%)")
    print(f"  High (85-94%): {high_conf-very_high_conf:,} ({(high_conf-very_high_conf)/len(predictions)*100:.1f}%)")
    print(f"  Medium (70-84%): {medium_conf-high_conf:,} ({(medium_conf-high_conf)/len(predictions)*100:.1f}%)")
    print(f"  Low (<70%): {low_conf:,} ({low_conf/len(predictions)*100:.1f}%)")

    # random vs trained performence comparison
    random_accuracy = 100.0 / 62  # ~1.61% for random guessing on 62 classes
    improvement = accuracy / random_accuracy
    print(f"\nPERFORMANCE COMPARISON:")
    print(f"  Random Baseline: {random_accuracy:.2f}%")
    print(f"  Model Performance: {accuracy:.2f}%")
    print(f"  Improvement Factor: {improvement:.1f}x over random")

    return {
        'predictions': predictions,
        'predicted_classes': predicted_classes,
        'true_classes': true_classes,
        'confidences': confidences,
        'accuracy': accuracy,
        'test_loss': test_loss,
        'avg_confidence': avg_confidence,
        'confidence_buckets': {
            'very_high': very_high_conf,
            'high': high_conf - very_high_conf,
            'medium': medium_conf - high_conf,
            'low': low_conf
        }
    }


def character_type_analysis(results):
    """Analyze performance by character type (digits, uppercase, lowercase)."""
    print(f"\nCHARACTER TYPE ANALYSIS:")
    print("-" * 70)

    predictions = results['predictions']
    predicted_classes = results['predicted_classes']
    true_classes = results['true_classes']

    type_stats = {}

    # define character type ranges
    char_types = {
        'Digits (0-9)': (0, 10),
        'Uppercase (A-Z)': (10, 36), 
        'Lowercase (a-z)': (36, 62)
    }

    for type_name, (start_idx, end_idx) in char_types.items():
        # get samples for this character type
        type_mask = (true_classes >= start_idx) & (true_classes < end_idx)

        if np.any(type_mask):
            type_predictions = predictions[type_mask]
            type_predicted = predicted_classes[type_mask]
            type_true = true_classes[type_mask]

            type_accuracy = np.mean(type_predicted == type_true) * 100
            type_count = len(type_true)
            type_correct = np.sum(type_predicted == type_true)
            type_avg_conf = np.mean(np.max(type_predictions, axis=1)) * 100

            print(f"  {type_name}: {type_accuracy:5.1f}% accuracy ({type_correct:4d}/{type_count:4d}) | "
                  f"Avg Conf: {type_avg_conf:5.1f}%")

            type_stats[type_name] = {
                'accuracy': type_accuracy,
                'count': type_count,
                'correct': type_correct,
                'avg_confidence': type_avg_conf
            }

    return type_stats


def per_character_analysis(results, max_chars_to_show=20):
    """Analyze performence for individual characters (showing worst performers)."""
    print(f"\nPER-CHARACTER ANALYSIS (Worst {max_chars_to_show} Performers):")
    print("-" * 80)

    predictions = results['predictions']
    predicted_classes = results['predicted_classes']
    true_classes = results['true_classes']

    char_stats = []

    for char_idx in range(62):
        # get samples for this character
        char_mask = (true_classes == char_idx)

        if np.any(char_mask):
            char_predictions = predictions[char_mask]
            char_predicted = predicted_classes[char_mask]
            char_true = true_classes[char_mask]

            char_accuracy = np.mean(char_predicted == char_true) * 100
            char_count = len(char_true)
            char_correct = np.sum(char_predicted == char_true)
            char_avg_conf = np.mean(np.max(char_predictions, axis=1)) * 100

            character = index_to_character(char_idx)
            char_type = get_character_type(char_idx)

            char_stats.append({
                'character': character,
                'char_type': char_type,
                'index': char_idx,
                'accuracy': char_accuracy,
                'count': char_count,
                'correct': char_correct,
                'avg_confidence': char_avg_conf
            })

    # sort by accuracy (worst first)
    char_stats.sort(key=lambda x: x['accuracy'])

    print("Char | Type      | Accuracy | Samples | Correct | Avg Conf")
    print("-" * 60)
    for stat in char_stats[:max_chars_to_show]:
        print(f" '{stat['character']}' | {stat['char_type']:<9} | {stat['accuracy']:6.1f}% | "
              f"{stat['count']:7d} | {stat['correct']:7d} | {stat['avg_confidence']:6.1f}%")

    return char_stats


def analyze_confusion_patterns(results):
    """Analyze common confusion patterns in predicitons."""
    print(f"\nCONFUSION PATTERN ANALYSIS:")
    print("-" * 50)

    predicted_classes = results['predicted_classes']
    true_classes = results['true_classes']

    # find most common misclassifications
    wrong_mask = predicted_classes != true_classes
    wrong_predictions = predicted_classes[wrong_mask]
    wrong_true = true_classes[wrong_mask]

    confusion_pairs = {}
    for true_char, pred_char in zip(wrong_true, wrong_predictions):
        pair = (true_char, pred_char)
        confusion_pairs[pair] = confusion_pairs.get(pair, 0) + 1

    # sort by frequency
    sorted_confusions = sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)

    print("Top 10 Confusion Pairs:")
    print("True -> Predicted | Count | Characters")
    print("-" * 40)
    for (true_idx, pred_idx), count in sorted_confusions[:10]:
        true_char = index_to_character(true_idx)
        pred_char = index_to_character(pred_idx)
        print(f" {true_idx:2d} -> {pred_idx:2d}        | {count:5d} | '{true_char}' -> '{pred_char}'")


def data_integrity_check(X_test, y_test):
    """Check data integrity and preprocesing quality."""
    print(f"\nDATA INTEGRITY CHECK:")
    print("-" * 40)

    # check data ranges
    print(f"Feature Data:")
    print(f"  Shape: {X_test.shape}")
    print(f"  Range: {X_test.min():.3f} to {X_test.max():.3f}")
    print(f"  Mean: {X_test.mean():.3f}")
    print(f"  Std: {X_test.std():.3f}")

    # check labels
    print(f"\nLabel Data:")
    print(f"  Shape: {y_test.shape}")
    print(f"  Sum check (should be 1.0): {y_test.sum(axis=1)[:5]}")

    # check class distribution
    true_classes = np.argmax(y_test, axis=1)
    print(f"\nClass Distribution:")
    print(f"  Range: {true_classes.min()} to {true_classes.max()}")
    print(f"  Expected: 0 to 61")

    # check for class imbalance
    unique_classes, counts = np.unique(true_classes, return_counts=True)
    min_count, max_count = counts.min(), counts.max()
    imbalance_ratio = max_count / min_count
    print(f"  Class counts range: {min_count} to {max_count}")
    print(f"  Imbalance ratio: {imbalance_ratio:.1f}:1")

    if imbalance_ratio > 10:
        print("  WARNING: Significant class imbalance detected!")


def create_universal_visualizations(results, char_stats, type_stats, model_data):
    """Create comprehensive visualization charts for universal character recogntion."""
    print(f"\nCREATING UNIVERSAL CHARACTER VISUALIZATIONS...")

    # create output directory
    os.makedirs('test_results', exist_ok=True)

    # set up plotting style
    plt.rcParams.update({'font.size': 10, 'figure.dpi': 100})

    # 1. character type performance
    plt.figure(figsize=(15, 6))

    type_names = list(type_stats.keys())
    type_accuracies = [type_stats[name]['accuracy'] for name in type_names]
    type_confidences = [type_stats[name]['avg_confidence'] for name in type_names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    bars1 = ax1.bar(type_names, type_accuracies, color=['skyblue', 'lightgreen', 'lightcoral'], alpha=0.8)
    ax1.set_title('Accuracy by Character Type', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_ylim(0, 100)
    ax1.grid(True, alpha=0.3)

    for bar, acc in zip(bars1, type_accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')

    bars2 = ax2.bar(type_names, type_confidences, color=['orange', 'purple', 'brown'], alpha=0.8)
    ax2.set_title('Average Confidence by Character Type', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Confidence (%)')
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3)

    for bar, conf in zip(bars2, type_confidences):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{conf:.1f}%', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig('test_results/character_type_performance.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. worst performing characters
    plt.figure(figsize=(16, 8))

    worst_chars = char_stats[:20]  # 20 worst performers
    chars = [stat['character'] for stat in worst_chars]
    accuracies = [stat['accuracy'] for stat in worst_chars]
    colors = ['red' if acc < 10 else 'orange' if acc < 30 else 'yellow' for acc in accuracies]

    bars = plt.bar(range(len(chars)), accuracies, color=colors, alpha=0.8)
    plt.title('20 Worst Performing Characters', fontsize=16, fontweight='bold')
    plt.xlabel('Character')
    plt.ylabel('Accuracy (%)')
    plt.xticks(range(len(chars)), [f"'{c}'" for c in chars], rotation=45)
    plt.grid(True, alpha=0.3)

    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig('test_results/worst_characters.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. confidence distribution
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.hist(results['confidences'], bins=50, alpha=0.7, color='green', edgecolor='black')
    plt.title('Overall Confidence Distribution', fontweight='bold')
    plt.xlabel('Confidence (%)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 2)
    confidence_buckets = results['confidence_buckets']
    labels = ['Very High\n(95%+)', 'High\n(85-94%)', 'Medium\n(70-84%)', 'Low\n(<70%)']
    values = [confidence_buckets['very_high'], confidence_buckets['high'], 
              confidence_buckets['medium'], confidence_buckets['low']]
    colors = ['darkgreen', 'green', 'orange', 'red']

    plt.pie(values, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title('Confidence Level Distribution', fontweight='bold')

    plt.subplot(2, 2, 3)
    correct_mask = results['predicted_classes'] == results['true_classes']
    correct_conf = results['confidences'][correct_mask]
    incorrect_conf = results['confidences'][~correct_mask]

    plt.hist(correct_conf, bins=30, alpha=0.7, label='Correct Predictions', color='blue')
    plt.hist(incorrect_conf, bins=30, alpha=0.7, label='Incorrect Predictions', color='red')
    plt.title('Confidence: Correct vs Incorrect', fontweight='bold')
    plt.xlabel('Confidence (%)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 4)
    accuracy_by_type = [type_stats[name]['accuracy'] for name in type_names]
    plt.bar(type_names, accuracy_by_type, color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.8)
    plt.title('Performance Summary by Type', fontweight='bold')
    plt.ylabel('Accuracy (%)')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('test_results/universal_confidence_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Universal character visualizations saved to test_results/ directory")


def generate_universal_report(results, char_stats, type_stats, model_data):
    """Generate a comprehensive text report for universal character recogntion."""
    print(f"\nGENERATING COMPREHENSIVE UNIVERSAL CHARACTER REPORT...")

    report_path = 'test_results/universal_character_test_report.txt'

    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("UNIVERSAL CHARACTER RECOGNITION - COMPREHENSIVE TEST REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write("MODEL INFORMATION:\n")
        f.write("-" * 40 + "\n")
        if 'model' in model_data:
            f.write(f"Architecture: {model_data['model'].layer_sizes}\n")
            f.write(f"Total Parameters: {model_data['model'].count_parameters():,}\n")
        f.write(f"Training Accuracy: {model_data.get('accuracy', 'N/A'):.2f}%\n")
        f.write(f"Training Time: {model_data.get('training_time', 0)/60:.1f} minutes\n")
        f.write(f"Classes: 62 (0-9, A-Z, a-z)\n\n")

        f.write("TEST PERFORMANCE SUMMARY:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Test Accuracy: {results['accuracy']:.2f}%\n")
        f.write(f"Test Loss: {results['test_loss']:.4f}\n")
        f.write(f"Average Confidence: {results['avg_confidence']:.1f}%\n")
        f.write(f"Random Baseline: {100/62:.2f}%\n")
        f.write(f"Improvement over Random: {results['accuracy']/(100/62):.1f}x\n\n")

        f.write("CHARACTER TYPE PERFORMANCE:\n")
        f.write("-" * 40 + "\n")
        for type_name, stats in type_stats.items():
            f.write(f"{type_name}: {stats['accuracy']:.1f}% accuracy, {stats['avg_confidence']:.1f}% confidence\n")
        f.write("\n")

        f.write("WORST 20 PERFORMING CHARACTERS:\n")
        f.write("-" * 40 + "\n")
        f.write("Char | Type      | Accuracy | Samples | Confidence\n")
        f.write("-" * 50 + "\n")
        for stat in char_stats[:20]:
            f.write(f" '{stat['character']}' | {stat['char_type']:<9} | {stat['accuracy']:6.1f}% | "
                   f"{stat['count']:7d} | {stat['avg_confidence']:8.1f}%\n")

    print(f"Comprehensive universal character report saved to {report_path}")


def main():
    """Main universal character testing function."""
    print("COMPREHENSIVE UNIVERSAL CHARACTER RECOGNITION TEST SUITE")
    print("=" * 80)

    # load trained universal model
    model, model_data = load_trained_universal_model()
    if model is None:
        return

    # load test data
    print(f"\nLoading universal character test data...")
    try:
        X_test, y_test = load_universal_test_data()

        print(f"  Test samples: {X_test.shape[0]:,}")
        print(f"  Feature dimensions: {X_test.shape[1]}")
        print(f"  Classes: {y_test.shape[1]}")

        # data integrity check
        data_integrity_check(X_test, y_test)

    except Exception as e:
        print(f"Error loading test data: {e}")
        return

    # run comprehensive evaluation
    results = comprehensive_universal_test_evaluation(model, X_test, y_test)

    # character type analysis
    type_stats = character_type_analysis(results)

    # per-character analysis
    char_stats = per_character_analysis(results)

    # confusion pattern analysis
    analyze_confusion_patterns(results)

    # create visualizations
    create_universal_visualizations(results, char_stats, type_stats, model_data)

    # generate detailed report
    generate_universal_report(results, char_stats, type_stats, model_data)

    print(f"\nCOMPREHENSIVE UNIVERSAL CHARACTER TESTING COMPLETE!")
    print(f"  Check test_results/ directory for all outputs")
    print(f"  Final Test Accuracy: {results['accuracy']:.2f}%")
    print(f"  Average Confidence: {results['avg_confidence']:.1f}%")

    # final diagnosis
    if results['accuracy'] < 10:
        print(f"\nCRITICAL PERFORMANCE ISSUES DETECTED:")
        print(f"  • Accuracy below 10% suggests fundamental training problems")
        print(f"  • Check data preprocessing and label mapping")
        print(f"  • Verify one-hot encoding and class ranges")
    elif results['accuracy'] < 30:
        print(f"\nPOOR PERFORMANCE DETECTED:")
        print(f"  • Model is learning but has significant issues")
        print(f"  • Consider architecture changes or more training")


if __name__ == "__main__":
    main()
