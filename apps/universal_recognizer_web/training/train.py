"""
Main training script for universal character recognition.

State-of-the-art training pipeline with all modern techniques.
"""

import os
import sys
import pickle
import time
import numpy as np
from typing import Optional

# Add paths - need to get to NeuralEngine root
# __file__ is apps/universal_recognizer_web/training/train.py
# Go up 3 levels to get to NeuralEngine root
_this_file = os.path.abspath(__file__)
base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_this_file))))
if base_path not in sys.path:
    sys.path.insert(0, base_path)

from nn_core import NeuralNetwork, cross_entropy_loss
from autodiff import Adam

# Handle both script and module execution
try:
    from .data_loader import create_data_splits, index_to_character, get_character_type, check_data_files
    from .data_augmentation import create_augmentation_pipeline
    from .trainer import EnhancedTrainingEngine, LearningRateScheduler, EarlyStopping
    from .config import TrainingConfig, get_default_config, get_high_accuracy_config
    from .evaluator import evaluate_model_comprehensive
except ImportError:
    # Running as script, use absolute imports
    import sys
    training_dir = os.path.dirname(os.path.abspath(__file__))
    if training_dir not in sys.path:
        sys.path.insert(0, training_dir)
    from data_loader import create_data_splits, index_to_character, get_character_type, check_data_files
    from data_augmentation import create_augmentation_pipeline
    from trainer import EnhancedTrainingEngine, LearningRateScheduler, EarlyStopping
    from config import TrainingConfig, get_default_config, get_high_accuracy_config
    from evaluator import evaluate_model_comprehensive


def create_model(config: TrainingConfig) -> NeuralNetwork:
    """Create neural network model."""
    print("Creating universal character recognition model...")
    print(f"  Architecture: {' -> '.join(map(str, config.layer_sizes))}")
    print(f"  Activations: {config.activations}")
    
    model = NeuralNetwork(
        layer_sizes=config.layer_sizes,
        activations=config.activations
    )
    
    print(f"  Total Parameters: {model.count_parameters():,}")
    return model


def train_model(config: TrainingConfig = None, data_dir: Optional[str] = None):
    """
    Train universal character recognition model.
    
    Args:
        config: Training configuration (uses default if None)
        data_dir: Data directory (default: apps/universal_recognizer_web/data)
    """
    if config is None:
        config = get_high_accuracy_config()
    
    print("=" * 70)
    print("STATE-OF-THE-ART UNIVERSAL CHARACTER RECOGNITION TRAINING")
    print("=" * 70)
    
    # Set default data directory
    if data_dir is None:
        # __file__ is training/train.py
        # Go up to universal_recognizer_web, then into data
        base_path = os.path.dirname(os.path.dirname(__file__))
        data_dir = os.path.join(base_path, 'data')
    
    # Check data files exist
    print(f"\n[1/6] Checking data files in: {data_dir}")
    if not check_data_files(data_dir):
        print("\n" + "=" * 70)
        print("ERROR: Dataset files not found!")
        print("=" * 70)
        print("\nPlease download EMNIST ByClass dataset and place files in:")
        print(f"  {data_dir}/")
        print("\nRequired files:")
        print("  - emnist-byclass-train-images-idx3-ubyte.gz")
        print("  - emnist-byclass-train-labels-idx1-ubyte.gz")
        print("  - emnist-byclass-test-images-idx3-ubyte.gz")
        print("  - emnist-byclass-test-labels-idx1-ubyte.gz")
        print("  - emnist-byclass-mapping.txt")
        print("\nSee training/DATASET_SETUP.md for download instructions.")
        print("=" * 70)
        raise FileNotFoundError(f"Dataset files not found in {data_dir}")
    
    # Load data
    print("\n[2/6] Loading and preprocessing data...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = create_data_splits(
        validation_size=config.validation_split,
        data_dir=data_dir
    )
    
    print(f"  Training: {X_train.shape[0]:,} samples")
    print(f"  Validation: {X_val.shape[0]:,} samples")
    print(f"  Test: {X_test.shape[0]:,} samples")
    
    # Create model
    print("\n[3/6] Creating model...")
    model = create_model(config)
    
    # Training phases
    print("\n[4/6] Starting multi-phase training...")
    print(f"  Total phases: {len(config.phases)}")
    
    all_history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
    start_time = time.time()
    global_epoch = 0
    
    for phase_idx, phase_config in enumerate(config.phases):
        print(f"\n{phase_config['name']} (Epochs {global_epoch + 1}-{global_epoch + phase_config['epochs']})")
        print("-" * 70)
        
        # Create optimizer for this phase
        optimizer = Adam(learning_rate=phase_config['learning_rate'])
        
        # Create learning rate scheduler
        lr_scheduler = LearningRateScheduler(
            initial_lr=phase_config['learning_rate'],
            schedule_type=config.lr_schedule_type,
            **config.lr_schedule_params
        )
        
        # Create early stopping (only in later phases)
        early_stopping = None
        if phase_idx >= 2:  # Enable in phases 3-4
            early_stopping = EarlyStopping(
                patience=config.early_stopping_patience,
                min_delta=config.early_stopping_min_delta,
                mode='min'
            )
        
        # Create augmentation
        aug_phase = phase_config.get('augmentation', 'full')
        augmentation = create_augmentation_pipeline(aug_phase)
        print(f"  Augmentation: {aug_phase}")
        
        # Create trainer
        trainer = EnhancedTrainingEngine(
            network=model,
            optimizer=optimizer,
            loss_function=cross_entropy_loss,
            max_grad_norm=config.max_grad_norm
        )
        
        # Train phase
        phase_history = trainer.train(
            X_train, y_train,
            X_val=X_val, y_val=y_val,
            epochs=phase_config['epochs'],
            batch_size=config.batch_size,
            lr_scheduler=lr_scheduler,
            early_stopping=early_stopping,
            checkpoint_dir=config.checkpoint_dir if phase_idx == len(config.phases) - 1 else None,
            checkpoint_freq=config.checkpoint_freq,
            augmentation=augmentation,
            verbose=True
        )
        
        # Merge history
        all_history['train_loss'].extend(phase_history['train_loss'])
        all_history['val_loss'].extend(phase_history['val_loss'])
        all_history['val_accuracy'].extend(phase_history['val_accuracy'])
        
        global_epoch += phase_config['epochs']
        
        # Check if early stopping occurred
        if early_stopping and early_stopping.stopped:
            print(f"  Early stopping triggered, ending training")
            break
    
    training_time = time.time() - start_time
    
    # Evaluation
    print("\n[5/6] Evaluating model...")
    results = evaluate_model_comprehensive(model, X_test, y_test)
    
    # Save model with robust error handling
    print("\n[6/6] Saving model...")
    # Save in universal_recognizer_web/models
    # __file__ is training/train.py
    base_path = os.path.dirname(os.path.dirname(__file__))
    models_dir = os.path.join(base_path, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Save config as dict to avoid import issues
    config_dict = {
        'layer_sizes': config.layer_sizes,
        'activations': config.activations,
        'epochs': config.epochs,
        'batch_size': config.batch_size,
        'learning_rate': config.learning_rate,
        'validation_split': config.validation_split,
    }
    
    model_data = {
        'model': model,
        'accuracy': results['overall_accuracy'],
        'avg_confidence': results['avg_confidence'],
        'character_type_accuracies': results['character_type_accuracies'],
        'history': all_history,
        'training_time': training_time,
        'architecture': config.layer_sizes,
        'dataset': 'emnist_byclass',
        'classes': 62,
        'config': config_dict  # Save as dict instead of object
    }
    
    model_path = os.path.join(models_dir, 'universal_character_model.pkl')
    temp_path = model_path + '.tmp'
    
    try:
        # Atomic write: write to temp file first, then rename
        print(f"  Writing model to temporary file...")
        with open(temp_path, 'wb') as f:
            pickle.dump(model_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Verify the file was written correctly
        if not os.path.exists(temp_path):
            raise IOError("Temporary model file was not created")
        
        file_size = os.path.getsize(temp_path)
        if file_size < 1000:  # Model should be at least 1KB
            raise IOError(f"Model file seems too small ({file_size} bytes)")
        
        # Verify we can load it back
        print(f"  Verifying model file integrity...")
        with open(temp_path, 'rb') as f:
            test_load = pickle.load(f)
            if 'model' not in test_load or 'accuracy' not in test_load:
                raise ValueError("Model file verification failed - missing required keys")
        
        # Atomic rename (only works on same filesystem, but safer)
        if os.path.exists(model_path):
            backup_path = model_path + '.backup'
            if os.path.exists(backup_path):
                os.remove(backup_path)
            os.rename(model_path, backup_path)
        
        os.rename(temp_path, model_path)
        
        # Final verification
        final_size = os.path.getsize(model_path)
        print(f"  ✓ Model saved successfully!")
        print(f"  ✓ File size: {final_size / (1024*1024):.2f} MB")
        print(f"  ✓ Location: {model_path}")
        print(f"  ✓ Verification: PASSED")
        
        # Clean up backup if everything is good
        backup_path = model_path + '.backup'
        if os.path.exists(backup_path):
            os.remove(backup_path)
        
    except Exception as e:
        # Clean up temp file on error
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
        
        print(f"\n" + "=" * 70)
        print(f"ERROR: Model saving failed!")
        print(f"=" * 70)
        print(f"Error: {str(e)}")
        print(f"\nAttempting to save to backup location...")
        
        # Try backup location
        try:
            backup_dir = os.path.join(base_path, 'models', 'backup')
            os.makedirs(backup_dir, exist_ok=True)
            backup_path = os.path.join(backup_dir, f'universal_character_model_backup_{int(time.time())}.pkl')
            with open(backup_path, 'wb') as f:
                pickle.dump(model_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"  Model saved to backup location: {backup_path}")
        except Exception as backup_error:
            print(f"  Backup save also failed: {backup_error}")
            print(f"\nCRITICAL: Model could not be saved!")
            print(f"  However, checkpoints may be available in: {os.path.join(models_dir, 'checkpoints')}")
            raise
    
    # Final report
    print("\nTraining Summary")
    print("=" * 70)
    print(f"Overall Accuracy: {results['overall_accuracy']:.2f}%")
    print(f"  Digits (0-9): {results['character_type_accuracies']['digits']:.1f}%")
    print(f"  Uppercase (A-Z): {results['character_type_accuracies']['uppercase']:.1f}%")
    print(f"  Lowercase (a-z): {results['character_type_accuracies']['lowercase']:.1f}%")
    print(f"Average Confidence: {results['avg_confidence']:.1f}%")
    print(f"Training Time: {training_time/60:.1f} minutes")
    print(f"Total Epochs: {len(all_history['train_loss'])}")
    print("=" * 70)
    
    return model, results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train universal character recognition model')
    parser.add_argument('--config', type=str, choices=['default', 'high_accuracy'], 
                       default='high_accuracy', help='Configuration preset')
    parser.add_argument('--data-dir', type=str, default=None, 
                       help='Data directory (default: apps/universal_recognizer_web/data)')
    
    args = parser.parse_args()
    
    # Select configuration
    if args.config == 'high_accuracy':
        config = get_high_accuracy_config()
    else:
        config = get_default_config()
    
    # Train (data_dir defaults to universal_recognizer_web/data in train_model)
    train_model(config, args.data_dir)

