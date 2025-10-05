"""
EMNIST digits data loading - official .gz format.

Loads EMNIST digits from official .gz IDX files and prepares for NeuralEngine training.
Handles the official NIST binary format with proper EMNIST transformations.
"""

import numpy as np
import os
import sys
import gzip
import struct
from sklearn.model_selection import train_test_split

# add NeuralEngine to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from data_utils import DataPreprocessor


class EMNISTDigitsLoader:
    """
    Loader for official EMNIST digits .gz files from NIST.
    Handles binary IDX format with proper EMNIST preprocesing.
    """

    def __init__(self, data_dir='data'):
        self.data_dir = data_dir

        # official EMNIST digits .gz files
        self.files = {
            'train_images': 'emnist-digits-train-images-idx3-ubyte.gz',
            'train_labels': 'emnist-digits-train-labels-idx1-ubyte.gz',
            'test_images': 'emnist-digits-test-images-idx3-ubyte.gz',
            'test_labels': 'emnist-digits-test-labels-idx1-ubyte.gz'
        }

    def _read_idx_images(self, filepath):
        """Read IDX3 format image file (official EMNIST format)."""
        print(f"  Reading images from {os.path.basename(filepath)}...")

        with gzip.open(filepath, 'rb') as f:
            # read magic number and dimensions
            magic = struct.unpack('>I', f.read(4))[0]
            if magic != 2051:  # IDX3 magic number
                raise ValueError(f"Invalid magic number for images: {magic}")

            num_images = struct.unpack('>I', f.read(4))[0]
            rows = struct.unpack('>I', f.read(4))[0]
            cols = struct.unpack('>I', f.read(4))[0]

            print(f"    Found {num_images:,} images of size {rows}x{cols}")

            # read image data
            data = np.frombuffer(f.read(), dtype=np.uint8)
            data = data.reshape(num_images, rows, cols)

            # CRITICAL: apply EMNIST transformations
            # EMNIST images need to be rotated 90 degrees counterclockwise and flipped horizontally
            transformed_data = []
            for img in data:
                # rotate 90 degrees counterclockwise and flip horizontally
                transformed = np.rot90(np.fliplr(img))
                transformed_data.append(transformed)

            data = np.array(transformed_data)
            print(f"    Applied EMNIST transformations (rotate + flip)")

        return data

    def _read_idx_labels(self, filepath):
        """Read IDX1 format label file (official EMNIST format)."""
        print(f"  Reading labels from {os.path.basename(filepath)}...")

        with gzip.open(filepath, 'rb') as f:
            # read magic number and dimensions
            magic = struct.unpack('>I', f.read(4))[0]
            if magic != 2049:  # IDX1 magic number
                raise ValueError(f"Invalid magic number for labels: {magic}")

            num_labels = struct.unpack('>I', f.read(4))[0]
            print(f"    Found {num_labels:,} labels")

            # read label data
            data = np.frombuffer(f.read(), dtype=np.uint8)

            # EMNIST digits labels are already 0-9, no conversion needed
            print(f"    Label range: {data.min()} to {data.max()}")

        return data

    def load_emnist_digits(self):
        """Load EMNIST digits dataset from official .gz files."""
        print("Loading EMNIST digits from official .gz files...")

        # check if all files exist
        missing_files = []
        for key, filename in self.files.items():
            filepath = os.path.join(self.data_dir, filename)
            if not os.path.exists(filepath):
                missing_files.append(filename)

        if missing_files:
            raise FileNotFoundError(
                f"Missing EMNIST digits files: {missing_files}\n"
                f"Please ensure these official .gz files are in the {self.data_dir}/ directory:\n"
                f"  - emnist-digits-train-images-idx3-ubyte.gz\n"
                f"  - emnist-digits-train-labels-idx1-ubyte.gz\n"
                f"  - emnist-digits-test-images-idx3-ubyte.gz\n"
                f"  - emnist-digits-test-labels-idx1-ubyte.gz"
            )

        # load training data
        print("\nLoading training data...")
        train_images_path = os.path.join(self.data_dir, self.files['train_images'])
        train_labels_path = os.path.join(self.data_dir, self.files['train_labels'])

        X_train = self._read_idx_images(train_images_path)
        y_train = self._read_idx_labels(train_labels_path)

        # load test data
        print("\nLoading test data...")
        test_images_path = os.path.join(self.data_dir, self.files['test_images'])
        test_labels_path = os.path.join(self.data_dir, self.files['test_labels'])

        X_test = self._read_idx_images(test_images_path)
        y_test = self._read_idx_labels(test_labels_path)

        print(f"\nEMNIST digits loaded succesfully:")
        print(f"  Training: {X_train.shape[0]:,} samples")
        print(f"  Test: {X_test.shape[0]:,} samples")
        print(f"  Image size: {X_train.shape[1]}x{X_train.shape[2]}")
        print(f"  Label range: {y_train.min()}-{y_train.max()}")

        return (X_train, y_train), (X_test, y_test)


def load_and_preprocess_emnist_digits():
    """
    Load and preprocess EMNIST digits data from official .gz files.

    Returns:
        Tuple of ((X_train, y_train), (X_test, y_test))
    """
    print("Loading EMNIST digits with NeuralEngine DataPreprocessor...")

    # initialize EMNIST loader and preprocessor
    emnist_loader = EMNISTDigitsLoader()
    preprocessor = DataPreprocessor()

    # load raw data from .gz files
    (X_train, y_train), (X_test, y_test) = emnist_loader.load_emnist_digits()

    # reshape images to flat arrays (28x28 -> 784)
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    print("\nPreprocessing with NeuralEngine...")

    # normalize pixel values to [0, 1] range
    X_train = preprocessor.normalize_features(X_train.astype(np.float32), method='minmax')
    X_test = preprocessor.normalize_features(X_test.astype(np.float32), method='minmax', fit_scaler=False)

    # convert labels to one-hot encoding
    y_train_onehot = np.zeros((y_train.shape[0], 10))
    y_test_onehot = np.zeros((y_test.shape[0], 10))

    for i in range(y_train.shape[0]):
        y_train_onehot[i, y_train[i]] = 1

    for i in range(y_test.shape[0]):
        y_test_onehot[i, y_test[i]] = 1

    print("EMNIST digits data preprocessing complete!")
    print(f"  Training features: {X_train.shape}")
    print(f"  Training labels: {y_train_onehot.shape}")
    print(f"  Test features: {X_test.shape}")
    print(f"  Test labels: {y_test_onehot.shape}")

    # display enhanced dataset info
    print(f"\nEMNIST digits dataset summary:")
    print(f"  Training samples: {X_train.shape[0]:,} (vs 60,000 in regular MNIST)")
    print(f"  Test samples: {X_test.shape[0]:,} (vs 10,000 in regular MNIST)")
    print(f"  Improvement: {X_train.shape[0]/60000:.1f}x more training data")
    print(f"  Feature range: {X_train.min():.3f} to {X_train.max():.3f}")

    return (X_train, y_train_onehot), (X_test, y_test_onehot)


def prepare_emnist_data_splits():
    """Prepare train/validation splits from loaded EMNIST digits data."""
    print("Creating EMNIST digits data splits...")

    # load the preprocessed data
    (X_train, y_train), (X_test, y_test) = load_and_preprocess_emnist_digits()

    # create train/validation split from training data
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train,
        test_size=0.1,  # 10% for validation
        random_state=42,
        stratify=y_train.argmax(axis=1)  # stratify by digit class
    )

    print(f"EMNIST digits data splits created successfully:")
    print(f"  Training: {X_train_split.shape[0]:,} samples")
    print(f"  Validation: {X_val.shape[0]:,} samples") 
    print(f"  Test: {X_test.shape[0]:,} samples")

    # display enhanced class distribution
    print(f"\nEnhanced dataset class distribution:")
    train_labels = y_train_split.argmax(axis=1)
    for digit in range(10):
        count = np.sum(train_labels == digit)
        old_mnist_approx = 6000  # aproximate MNIST samples per digit
        improvement = count / old_mnist_approx
        print(f"  Digit {digit}: {count:,} samples ({improvement:.1f}x more than MNIST)")

    return (X_train_split, y_train_split), (X_val, y_val), (X_test, y_test)


def load_test_data():
    """
    Load and preprocess the EMNIST digits test data only.

    Used by comprehensive_test.py

    Returns:
        Tuple of (X_test, y_test) where:
        - X_test: normalized test features (samples, 784)
        - y_test: one-hot encoded test labels (samples, 10)
    """
    print("Loading EMNIST digits test data...")

    # initialize EMNIST loader and preprocessor
    emnist_loader = EMNISTDigitsLoader()
    preprocessor = DataPreprocessor()

    # load only test data from .gz files
    (_, _), (X_test, y_test) = emnist_loader.load_emnist_digits()

    # reshape and preprocess
    X_test = X_test.reshape(X_test.shape[0], -1)
    X_test = preprocessor.normalize_features(X_test.astype(np.float32), method='minmax', fit_scaler=False)

    # convert to one-hot encoding
    y_test_onehot = np.zeros((y_test.shape[0], 10))
    for i in range(y_test.shape[0]):
        y_test_onehot[i, y_test[i]] = 1

    print(f"EMNIST digits test data loaded: {X_test.shape[0]:,} samples")
    print(f"  Feature dimensions: {X_test.shape[1]}")
    print(f"  Classes: {y_test_onehot.shape[1]}")

    return X_test, y_test_onehot


# keep backward compatibility with old function name
prepare_data_splits = prepare_emnist_data_splits


if __name__ == "__main__":
    """Test the EMNIST digits .gz data loading pipeline."""
    print("Testing EMNIST digits .gz data loading...")
    print("=" * 50)

    try:
        # test basic loading
        print("\n1. Testing basic EMNIST digits .gz data loading...")
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = prepare_emnist_data_splits()

        print(f"\nEMNIST digits .gz data loading test successful!")
        print(f"  Training set: {X_train.shape}")
        print(f"  Validation set: {X_val.shape}")
        print(f"  Test set: {X_test.shape}")

        # test sample display
        print(f"\n2. Testing sample data quality...")
        sample_labels = y_train.argmax(axis=1)[:10]
        print(f"  Sample training labels: {sample_labels}")
        print(f"  Feature range: {X_train.min():.3f} to {X_train.max():.3f}")
        print(f"  Data type: {X_train.dtype}")

        # compare with MNIST baseline
        print(f"\n3. Dataset comparision with original MNIST...")
        mnist_train_samples = 60000
        mnist_test_samples = 10000
        train_improvement = X_train.shape[0] / mnist_train_samples
        test_improvement = X_test.shape[0] / mnist_test_samples

        print(f"  Training data improvement: {train_improvement:.1f}x")
        print(f"  Test data improvement: {test_improvement:.1f}x")
        print(f"  Total additional samples: {X_train.shape[0] + X_test.shape[0] - 70000:,}")

        print(f"\nAll tests passed! EMNIST digits .gz data loading is ready.")
        print(f"Your model will train on {X_train.shape[0]:,} samples instead of 60,000!")

    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
