"""
Enhanced data loader for universal character recognition training.

Self-contained EMNIST ByClass loader - no dependencies on old universal_recognizer.
"""

import os
import sys
import numpy as np
import gzip
from typing import Tuple, Optional, Iterator
from sklearn.model_selection import train_test_split

# Add paths - need to get to NeuralEngine root
# __file__ is apps/universal_recognizer_web/training/data_loader.py
# Go up 3 levels to get to NeuralEngine root
_this_file = os.path.abspath(__file__)
base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_this_file))))
if base_path not in sys.path:
    sys.path.insert(0, base_path)

# Handle both script and module execution
try:
    from .data_augmentation import DataAugmentation, create_augmentation_pipeline
except ImportError:
    # Running as script
    training_dir = os.path.dirname(_this_file)
    if training_dir not in sys.path:
        sys.path.insert(0, training_dir)
    from data_augmentation import DataAugmentation, create_augmentation_pipeline


# Character mapping utilities
def index_to_character(index: int) -> str:
    """Convert class index (0-61) to character."""
    if 0 <= index <= 9:
        return str(index)
    elif 10 <= index <= 35:
        return chr(ord('A') + index - 10)
    elif 36 <= index <= 61:
        return chr(ord('a') + index - 36)
    else:
        return '?'


def character_to_index(char: str) -> int:
    """Convert character to class index (0-61)."""
    if char.isdigit():
        return int(char)
    elif char.isupper() and char.isalpha():
        return ord(char) - ord('A') + 10
    elif char.islower() and char.isalpha():
        return ord(char) - ord('a') + 36
    else:
        return -1


def get_character_type(index: int) -> str:
    """Get character type description."""
    if 0 <= index <= 9:
        return "Digit"
    elif 10 <= index <= 35:
        return "Uppercase"
    elif 36 <= index <= 61:
        return "Lowercase"
    else:
        return "Unknown"


def check_data_files(data_dir: str) -> bool:
    """Check if all required EMNIST data files exist."""
    required_files = [
        'emnist-byclass-train-images-idx3-ubyte.gz',
        'emnist-byclass-train-labels-idx1-ubyte.gz',
        'emnist-byclass-test-images-idx3-ubyte.gz',
        'emnist-byclass-test-labels-idx1-ubyte.gz',
        'emnist-byclass-mapping.txt'
    ]
    
    missing_files = []
    for filename in required_files:
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            missing_files.append(filename)
    
    if missing_files:
        print("Missing required data files:")
        for filename in missing_files:
            print(f"  - {filename}")
        print(f"\nPlease ensure all EMNIST ByClass files are in: {data_dir}")
        return False
    
    return True


def read_idx_images(filename: str) -> np.ndarray:
    """Read EMNIST image data from IDX3 format."""
    print(f"Reading images from {filename}")
    
    with gzip.open(filename, 'rb') as f:
        # Read magic number and dimensions
        magic = int.from_bytes(f.read(4), 'big')
        num_images = int.from_bytes(f.read(4), 'big')
        rows = int.from_bytes(f.read(4), 'big')
        cols = int.from_bytes(f.read(4), 'big')
        
        # Validate magic number
        if magic != 2051:
            raise ValueError(f"Invalid magic number for images: {magic} (expected 2051)")
        
        # Validate dimensions
        if rows != 28 or cols != 28:
            raise ValueError(f"Invalid image dimensions: {rows}x{cols} (expected 28x28)")
        
        print(f"  Loading {num_images:,} images of size {rows}x{cols}")
        
        # Read image data
        buffer = f.read(num_images * rows * cols)
        data = np.frombuffer(buffer, dtype=np.uint8)
        data = data.reshape(num_images, rows, cols)
        
        print(f"  Successfully loaded {data.shape[0]:,} images")
        return data


def read_idx_labels(filename: str) -> np.ndarray:
    """Read EMNIST label data from IDX1 format."""
    print(f"Reading labels from {filename}")
    
    with gzip.open(filename, 'rb') as f:
        magic = int.from_bytes(f.read(4), 'big')
        num_labels = int.from_bytes(f.read(4), 'big')
        
        # Validate magic number
        if magic != 2049:
            raise ValueError(f"Invalid magic number for labels: {magic} (expected 2049)")
        
        print(f"  Loading {num_labels:,} labels")
        
        buffer = f.read(num_labels)
        labels = np.frombuffer(buffer, dtype=np.uint8)
        
        # Validate label range
        if labels.min() < 0 or labels.max() > 61:
            raise ValueError(f"Invalid label range: {labels.min()}-{labels.max()} (expected 0-61)")
        
        print(f"  Successfully loaded {labels.shape[0]:,} labels")
        return labels


def fix_emnist_orientation(images: np.ndarray) -> np.ndarray:
    """
    Fix EMNIST image orientation.
    EMNIST images are rotated 90 degrees CCW and flipped horizontally.
    """
    print("Fixing EMNIST image orientation...")
    
    # Step 1: Flip horizontally
    flipped = np.flip(images, axis=2)
    
    # Step 2: Rotate 90 degrees clockwise
    rotated = np.rot90(flipped, k=-1, axes=(1, 2))
    
    print("  Orientation fixed successfully")
    return rotated


def preprocess_data(X: np.ndarray, normalize: bool = True) -> np.ndarray:
    """
    Preprocess data with proper normalization.
    
    Args:
        X: Input data (samples, 784) or (samples, 28, 28)
        normalize: Whether to normalize to [-1, 1] range
    
    Returns:
        Preprocessed data
    """
    # Ensure 2D for processing
    original_shape = X.shape
    if X.ndim == 3:
        X = X.reshape(X.shape[0], -1)
    
    # Convert to float32
    X = X.astype(np.float32)
    
    # Normalize to [0, 1]
    if X.max() > 1.0:
        X = X / 255.0
    
    if normalize:
        # Center and normalize to [-1, 1]
        mean = np.mean(X, axis=0, keepdims=True)
        std = np.std(X, axis=0, keepdims=True) + 1e-8
        X = (X - mean) / std
        # Scale to [-1, 1] using tanh
        X = np.tanh(X)
    
    return X.reshape(original_shape)


def create_one_hot(labels: np.ndarray, num_classes: int = 62) -> np.ndarray:
    """Create one-hot encoding."""
    print(f"Creating one-hot encoding for {len(labels):,} labels...")
    
    one_hot = np.zeros((labels.shape[0], num_classes), dtype=np.float32)
    one_hot[np.arange(labels.shape[0]), labels] = 1
    
    print(f"  One-hot encoding complete: {one_hot.shape}")
    return one_hot


def load_emnist_data(data_dir: str = None) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Load EMNIST ByClass data with proper preprocessing.
    
    Args:
        data_dir: Data directory (default: apps/universal_recognizer_web/data)
    
    Returns:
        ((X_train, y_train), (X_test, y_test))
    """
    if data_dir is None:
        # Default to universal_recognizer_web/data
        # __file__ is training/data_loader.py
        # Go up to universal_recognizer_web, then into data
        base_path = os.path.dirname(os.path.dirname(__file__))
        data_dir = os.path.join(base_path, 'data')
    
    # Check files exist
    if not check_data_files(data_dir):
        raise FileNotFoundError(f"Required EMNIST data files not found in {data_dir}")
    
    print(f"Loading EMNIST ByClass from: {data_dir}")
    
    # Load training data
    print("\nLoading training data...")
    X_train = read_idx_images(os.path.join(data_dir, 'emnist-byclass-train-images-idx3-ubyte.gz'))
    y_train = read_idx_labels(os.path.join(data_dir, 'emnist-byclass-train-labels-idx1-ubyte.gz'))
    
    # Load test data
    print("\nLoading test data...")
    X_test = read_idx_images(os.path.join(data_dir, 'emnist-byclass-test-images-idx3-ubyte.gz'))
    y_test = read_idx_labels(os.path.join(data_dir, 'emnist-byclass-test-labels-idx1-ubyte.gz'))
    
    print(f"\nRaw data loaded:")
    print(f"  Training: {X_train.shape[0]:,} images")
    print(f"  Test: {X_test.shape[0]:,} images")
    print(f"  Image size: {X_train.shape[1]}x{X_train.shape[2]}")
    print(f"  Classes: {len(np.unique(y_train))}")
    
    # Fix image orientation
    X_train = fix_emnist_orientation(X_train)
    X_test = fix_emnist_orientation(X_test)
    
    # Flatten images for neural network
    print("Flattening images...")
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    
    # Preprocess
    print("Preprocessing data...")
    X_train = preprocess_data(X_train, normalize=True)
    X_test = preprocess_data(X_test, normalize=True)
    
    # Create one-hot encoding
    y_train_onehot = create_one_hot(y_train, 62)
    y_test_onehot = create_one_hot(y_test, 62)
    
    print(f"\nData loaded and preprocessed:")
    print(f"  Training: {X_train.shape[0]:,} samples")
    print(f"  Test: {X_test.shape[0]:,} samples")
    print(f"  Data range: [{X_train.min():.3f}, {X_train.max():.3f}]")
    
    return (X_train, y_train_onehot), (X_test, y_test_onehot)


def create_data_splits(validation_size: float = 0.1, random_state: int = 42,
                       data_dir: str = None) -> Tuple:
    """
    Create train/validation/test splits.
    
    Args:
        validation_size: Fraction of training data for validation
        random_state: Random seed
        data_dir: Data directory (default: apps/universal_recognizer_web/data)
    
    Returns:
        (X_train, y_train), (X_val, y_val), (X_test, y_test)
    """
    # Load data
    (X_train_full, y_train_full), (X_test, y_test) = load_emnist_data(data_dir)
    
    # Create validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=validation_size,
        random_state=random_state,
        stratify=y_train_full.argmax(axis=1)
    )
    
    print(f"\nData splits created:")
    print(f"  Training: {X_train.shape[0]:,} samples")
    print(f"  Validation: {X_val.shape[0]:,} samples")
    print(f"  Test: {X_test.shape[0]:,} samples")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


class BatchGenerator:
    """Generate batches with optional augmentation."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray, batch_size: int = 64,
                 shuffle: bool = True, augmentation: Optional[DataAugmentation] = None):
        """
        Initialize batch generator.
        
        Args:
            X: Features
            y: Labels
            batch_size: Batch size
            shuffle: Whether to shuffle data
            augmentation: Optional augmentation pipeline
        """
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augmentation = augmentation
        self.n_samples = X.shape[0]
        self.n_batches = (self.n_samples + batch_size - 1) // batch_size
        self.indices = np.arange(self.n_samples)
    
    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate batches."""
        if self.shuffle:
            np.random.shuffle(self.indices)
        
        for i in range(self.n_batches):
            start_idx = i * self.batch_size
            end_idx = min(start_idx + self.batch_size, self.n_samples)
            batch_indices = self.indices[start_idx:end_idx]
            
            X_batch = self.X[batch_indices]
            y_batch = self.y[batch_indices]
            
            # Apply augmentation if provided
            if self.augmentation is not None:
                X_batch, y_batch = self.augmentation.augment_batch(X_batch, y_batch)
            
            yield X_batch, y_batch
    
    def __len__(self) -> int:
        """Number of batches."""
        return self.n_batches


# Export utility functions
__all__ = [
    'load_emnist_data',
    'create_data_splits',
    'preprocess_data',
    'BatchGenerator',
    'index_to_character',
    'character_to_index',
    'get_character_type',
    'check_data_files'
]
