# test_data.py
from load_data import prepare_data_splits
import numpy as np

(X_train, y_train), (X_val, y_val), (X_test, y_test) = prepare_data_splits()

print("Data Shape Check:")
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_train range: {X_train.min()} to {X_train.max()}")
print(f"y_train sample: {y_train[0]}")
print(f"y_train sum: {y_train[0].sum()}")

# Check if data is normalized
print(f"\nData should be:")
print(f"- X normalized to 0-1 range: {X_train.min() >= 0 and X_train.max() <= 1}")
print(f"- y one-hot encoded: {np.allclose(y_train.sum(axis=1), 1.0)}")
