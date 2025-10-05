"""
MNIST test suite with visualizations.

Tests trained model on MNIST test data with detailed statistics,
confidence analysis, and visualization charts.
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
from load_data import load_test_data

# plotting style
plt.style.use('default')
sns.set_palette("husl")


def load_trained_model(model_path='models/digit_model_bulletproof.pkl'):
    """Load trained model and metadata."""
    print(f"Loading model from {model_path}...")
    
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        model = model_data['model']
        print(f"Model loaded successfully!")
        print(f"  Training Accuracy: {model_data.get('accuracy', 'N/A'):.2f}%")
        print(f"  Architecture: {model.layer_sizes}")
        print(f"  Parameters: {model.count_parameters():,}")
        
        return model, model_data
    
    except FileNotFoundError:
        print(f"Model file not found: {model_path}")
        print("Please train a model first by running train_model.py")
        return None, None


def comprehensive_test_evaluation(model, X_test, y_test):
    """Run comprehensive evaluation on test data."""
    print(f"\nCOMPREHENSIVE TEST EVALUATION")
    print("=" * 50)
    
    start_time = time.time()
    
    # get predictions
    predictions = model.forward(X_test)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_test, axis=1)
    
    # basic metrics
    accuracy = np.mean(predicted_classes == true_classes) * 100
    test_loss = cross_entropy_loss(y_test, predictions)
    
    # confidence analysis
    confidences = np.max(predictions, axis=1) * 100
    avg_confidence = np.mean(confidences)
    median_confidence = np.median(confidences)
    
    # confidence buckets
    very_high_conf = np.sum(confidences >= 95)
    high_conf = np.sum(confidences >= 85)
    medium_conf = np.sum(confidences >= 70)
    low_conf = np.sum(confidences < 70)
    
    prediction_time = time.time() - start_time
    
    # print summary
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


def per_digit_analysis(results):
    """Analyze performace for each digit class."""
    print(f"\nPER-DIGIT ANALYSIS:")
    print("-" * 60)
    
    predictions = results['predictions']
    predicted_classes = results['predicted_classes']
    true_classes = results['true_classes']
    
    per_digit_stats = []
    
    for digit in range(10):
        # get samples for this digit
        digit_mask = (true_classes == digit)
        digit_predictions = predictions[digit_mask]
        digit_predicted = predicted_classes[digit_mask]
        digit_true = true_classes[digit_mask]
        
        if len(digit_true) > 0:
            digit_accuracy = np.mean(digit_predicted == digit_true) * 100
            digit_count = len(digit_true)
            digit_correct = np.sum(digit_predicted == digit_true)
            digit_avg_conf = np.mean(np.max(digit_predictions, axis=1)) * 100
            digit_min_conf = np.min(np.max(digit_predictions, axis=1)) * 100
            digit_max_conf = np.max(np.max(digit_predictions, axis=1)) * 100
            
            print(f"  Digit {digit}: {digit_accuracy:5.1f}% accuracy ({digit_correct:3d}/{digit_count:3d}) | "
                  f"Conf: {digit_avg_conf:5.1f}% avg, {digit_min_conf:5.1f}%-{digit_max_conf:5.1f}% range")
            
            per_digit_stats.append({
                'digit': digit,
                'accuracy': digit_accuracy,
                'count': digit_count,
                'correct': digit_correct,
                'avg_confidence': digit_avg_conf,
                'min_confidence': digit_min_conf,
                'max_confidence': digit_max_conf
            })
    
    return per_digit_stats


def create_visualizations(results, per_digit_stats, model_data):
    """Create visualization charts."""
    print(f"\nCreating visualizations...")
    
    # create output dir
    os.makedirs('test_results', exist_ok=True)
    
    # plotting style
    plt.rcParams.update({'font.size': 10, 'figure.dpi': 100})
    
    # 1. confusion matrix
    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(results['true_classes'], results['predicted_classes'])
    
    # normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=range(10), yticklabels=range(10))
    plt.title('MNIST Test Set - Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Digit', fontsize=12)
    plt.ylabel('True Digit', fontsize=12)
    plt.tight_layout()
    plt.savefig('test_results/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. per-digit accuracy bar chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    digits = [stat['digit'] for stat in per_digit_stats]
    accuracies = [stat['accuracy'] for stat in per_digit_stats]
    confidences = [stat['avg_confidence'] for stat in per_digit_stats]
    
    bars1 = ax1.bar(digits, accuracies, color='skyblue', alpha=0.8, edgecolor='navy')
    ax1.set_title('Per-Digit Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Digit')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_ylim(0, 100)
    ax1.grid(True, alpha=0.3)
    
    # add value labels on bars
    for bar, acc in zip(bars1, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    bars2 = ax2.bar(digits, confidences, color='lightcoral', alpha=0.8, edgecolor='darkred')
    ax2.set_title('Per-Digit Average Confidence', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Digit')
    ax2.set_ylabel('Confidence (%)')
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3)
    
    # add value labels
    for bar, conf in zip(bars2, confidences):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{conf:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('test_results/per_digit_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. confidence distribution histogram
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
    sample_counts = [stat['count'] for stat in per_digit_stats]
    plt.bar(digits, sample_counts, color='purple', alpha=0.7)
    plt.title('Test Samples per Digit', fontweight='bold')
    plt.xlabel('Digit')
    plt.ylabel('Number of Samples')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('test_results/confidence_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. training history (if availible)
    if 'history' in model_data and model_data['history']:
        plt.figure(figsize=(15, 5))
        
        history = model_data['history']
        epochs = range(1, len(history['train_loss']) + 1)
        
        plt.subplot(1, 3, 1)
        plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
        if 'val_loss' in history:
            plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        plt.title('Training Progress - Loss', fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 2)
        # convert loss to approximate accuracy for viz
        approx_train_acc = [(2.3 - loss) / 2.3 * 100 for loss in history['train_loss']]
        plt.plot(epochs, approx_train_acc, 'g-', label='Training Accuracy (est.)', linewidth=2)
        plt.title('Training Progress - Accuracy', fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 3)
        final_metrics = {
            'Test Accuracy': results['accuracy'],
            'Avg Confidence': results['avg_confidence'],
            'High Conf Rate': (confidence_buckets['very_high'] + confidence_buckets['high']) / len(results['confidences']) * 100
        }
        
        metrics_names = list(final_metrics.keys())
        metrics_values = list(final_metrics.values())
        colors = ['gold', 'lightblue', 'lightgreen']
        
        bars = plt.bar(metrics_names, metrics_values, color=colors, alpha=0.8)
        plt.title('Final Test Metrics', fontweight='bold')
        plt.ylabel('Percentage (%)')
        plt.xticks(rotation=45)
        
        # add value labels
        for bar, value in zip(bars, metrics_values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('test_results/training_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Visualizations saved to test_results/:")
    print(f"  confusion_matrix.png - confusion matrix")
    print(f"  per_digit_performance.png - accuracy & confidence by digit")
    print(f"  confidence_analysis.png - confidence distribution")
    print(f"  training_summary.png - training progress & final metrics")


def generate_detailed_report(results, per_digit_stats, model_data):
    """Generate comprehensive text report."""
    print(f"\nGenerating detailed report...")
    
    report_path = 'test_results/detailed_test_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("MNIST NEURAL NETWORK - COMPREHENSIVE TEST REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("MODEL INFORMATION:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Architecture: {model_data.get('model', {}).layer_sizes if 'model' in model_data else 'N/A'}\n")
        f.write(f"Total Parameters: {model_data.get('model', {}).count_parameters():,} if 'model' in model_data else 'N/A'\n")
        f.write(f"Training Accuracy: {model_data.get('accuracy', 'N/A'):.2f}%\n")
        f.write(f"Training Time: {model_data.get('training_time', 0)/60:.1f} minutes\n\n")
        
        f.write("TEST PERFORMANCE SUMMARY:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Test Accuracy: {results['accuracy']:.2f}%\n")
        f.write(f"Test Loss: {results['test_loss']:.4f}\n")
        f.write(f"Average Confidence: {results['avg_confidence']:.1f}%\n\n")
        
        f.write("CONFIDENCE DISTRIBUTION:\n")
        f.write("-" * 40 + "\n")
        conf_buckets = results['confidence_buckets']
        total_samples = sum(conf_buckets.values())
        f.write(f"Very High Confidence (95%+): {conf_buckets['very_high']:,} ({conf_buckets['very_high']/total_samples*100:.1f}%)\n")
        f.write(f"High Confidence (85-94%): {conf_buckets['high']:,} ({conf_buckets['high']/total_samples*100:.1f}%)\n")
        f.write(f"Medium Confidence (70-84%): {conf_buckets['medium']:,} ({conf_buckets['medium']/total_samples*100:.1f}%)\n")
        f.write(f"Low Confidence (<70%): {conf_buckets['low']:,} ({conf_buckets['low']/total_samples*100:.1f}%)\n\n")
        
        f.write("PER-DIGIT DETAILED ANALYSIS:\n")
        f.write("-" * 40 + "\n")
        f.write("Digit | Accuracy | Samples | Correct | Avg Conf | Range\n")
        f.write("-" * 55 + "\n")
        for stat in per_digit_stats:
            f.write(f"  {stat['digit']}   |  {stat['accuracy']:5.1f}%  |  {stat['count']:4d}   |  {stat['correct']:4d}   | "
                   f"{stat['avg_confidence']:5.1f}%  | {stat['min_confidence']:4.1f}%-{stat['max_confidence']:4.1f}%\n")
    
    print(f"Detailed report saved to {report_path}")


def main():
    """Main testing function."""
    print("COMPREHENSIVE MNIST TEST SUITE")
    print("=" * 80)
    
    # load trained model
    model, model_data = load_trained_model()
    if model is None:
        return
    
    # load test data
    print(f"\nLoading test data...")
    X_test, y_test = load_test_data()
    
    print(f"  Test samples: {X_test.shape[0]:,}")
    print(f"  Feature dimensions: {X_test.shape[1]}")
    print(f"  Classes: {y_test.shape[1]}")
    
    # run comprehensive evaluation
    results = comprehensive_test_evaluation(model, X_test, y_test)
    
    # per-digit analysis
    per_digit_stats = per_digit_analysis(results)
    
    # create visualizations
    create_visualizations(results, per_digit_stats, model_data)
    
    # generate detailed report
    generate_detailed_report(results, per_digit_stats, model_data)
    
    print(f"\nTESTING COMPLETE!")
    print(f"  Check test_results/ directory for all outputs")
    print(f"  Final Test Accuracy: {results['accuracy']:.2f}%")
    print(f"  Average Confidence: {results['avg_confidence']:.1f}%")


if __name__ == "__main__":
    main()
