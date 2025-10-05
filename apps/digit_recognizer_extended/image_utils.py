"""
Image processing for digit recognition.

Uses NeuralEngine data processing pipeline for consistency.
"""

import numpy as np
import sys
import os

# import data utilities
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from data_utils import DataPreprocessor

# global preprocessor instance to maintain consistancy
_preprocessor = DataPreprocessor(verbose=False)


def preprocess_drawing_for_neural_network(drawing_array: np.ndarray) -> np.ndarray:
    """
    Preprocess drawing for NeuralNetwork using same pipeline as training.
    
    Args:
        drawing_array: 28x28 drawing from UI canvas
        
    Returns:
        Preprocessed array ready for neural network prediction
    """
    # ensure proper shape and type
    processed = drawing_array.copy().astype(np.float32)
    
    # flatten to match training data format (784 features)
    flattened = processed.flatten().reshape(1, -1)
    
    # use DataPreprocessor with same normilization as training
    # ensures consistency between training and prediction
    normalized = _preprocessor.normalize_features(flattened, method='minmax', fit_scaler=False)
    
    return normalized


def enhance_digit_drawing(image: np.ndarray) -> np.ndarray:
    """
    Enhance drawn digit for better recogntion.
    
    Args:
        image: 28x28 drawing array
        
    Returns:
        Enhanced image
    """
    enhanced = image.copy()
    
    # center the digit
    enhanced = center_digit_in_frame(enhanced)
    
    # smooth the edges slightly
    enhanced = smooth_digit_edges(enhanced)
    
    return enhanced


def center_digit_in_frame(image: np.ndarray) -> np.ndarray:
    """Center the digit within the 28x28 frame."""
    # find bounding box of drawn content
    rows = np.any(image, axis=1)
    cols = np.any(image, axis=0)
    
    if not np.any(rows) or not np.any(cols):
        return image  # empty image, return as-is
    
    # get bounding box coordiantes
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    # extract the digit content
    digit_content = image[rmin:rmax+1, cmin:cmax+1]
    
    # create new centered image
    centered = np.zeros((28, 28), dtype=np.float32)
    
    # calculate centering position
    digit_height, digit_width = digit_content.shape
    start_row = max(0, (28 - digit_height) // 2)
    start_col = max(0, (28 - digit_width) // 2)
    
    # place digit in center
    end_row = min(28, start_row + digit_height)
    end_col = min(28, start_col + digit_width)
    
    centered[start_row:end_row, start_col:end_col] = digit_content[:end_row-start_row, :end_col-start_col]
    
    return centered


def smooth_digit_edges(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Apply slight smoothing to digit edges for better recogniton."""
    # simple smoothing using small averaging kernel
    from scipy import ndimage
    
    # apply gentle gaussian smoothing
    smoothed = ndimage.gaussian_filter(image, sigma=0.5)
    
    # maintain original intensity where there was content
    mask = image > 0.1
    result = image.copy()
    result[mask] = smoothed[mask]
    
    return result


# test the image processing
if __name__ == "__main__":
    print("Testing image processing utilities...")
    
    # create a test "drawing"
    test_image = np.zeros((28, 28))
    test_image[10:15, 10:15] = 1.0  # small square "digit"
    
    print(f"  Original image sum: {np.sum(test_image)}")
    
    # test preprocessing
    processed = preprocess_drawing_for_neural_network(test_image)
    print(f"  Processed shape: {processed.shape}")
    print(f"  Processed range: [{processed.min():.3f}, {processed.max():.3f}]")
    
    # test centering
    centered = center_digit_in_frame(test_image)
    print(f"  Centered image sum: {np.sum(centered)}")
    
    print("Image processing utilities working!")
