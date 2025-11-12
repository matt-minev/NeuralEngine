"""
Dataset testing functionality for universal character recognition.

Loads random samples from EMNIST test set for model evaluation.
"""

import numpy as np
import sys
import os
import random
from typing import List, Dict, Tuple, Optional

# Add NeuralEngine to path
base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.insert(0, base_path)

# Import data loader
try:
    from apps.universal_recognizer_web.training.data_loader import (
        load_emnist_data, 
        index_to_character,
        get_character_type
    )
except ImportError:
    # Fallback for direct execution
    training_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'training')
    sys.path.insert(0, training_dir)
    from data_loader import (
        load_emnist_data,
        index_to_character,
        get_character_type
    )


class DatasetTester:
    """
    Manages loading and serving random test samples from EMNIST dataset.
    """
    
    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize dataset tester.
        
        Args:
            data_dir: Path to EMNIST data directory (default: apps/universal_recognizer_web/data)
        """
        self.data_dir = data_dir
        self.test_images = None
        self.test_labels = None
        self.test_indices = None
        self._loaded = False
    
    def load_test_data(self):
        """Load EMNIST test data."""
        if self._loaded:
            return
        
        try:
            print("Loading EMNIST test data for testing mode...")
            
            # Load data
            (_, _), (X_test, y_test_onehot) = load_emnist_data(self.data_dir)
            
            # Convert one-hot back to labels
            y_test = np.argmax(y_test_onehot, axis=1)
            
            # Store test data
            self.test_images = X_test
            self.test_labels = y_test
            self.test_indices = np.arange(len(y_test))
            
            self._loaded = True
            print(f"  Loaded {len(self.test_labels):,} test samples")
            
        except Exception as e:
            print(f"Error loading test data: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def get_random_samples(self, count: int = 1, character_filter: Optional[str] = None) -> List[Dict]:
        """
        Get random test samples.
        
        Args:
            count: Number of samples to return (1, 9, 16, or 25 for grid layouts)
            character_filter: Optional character to filter by (e.g., 'A', '9', 'z')
        
        Returns:
            List of dictionaries with image data, label, and metadata
        """
        if not self._loaded:
            self.load_test_data()
        
        # Filter by character if requested
        if character_filter:
            char_index = None
            # Find character index
            for idx in range(62):
                if index_to_character(idx) == character_filter:
                    char_index = idx
                    break
            
            if char_index is None:
                # Character not found, return random samples
                filtered_indices = self.test_indices
            else:
                # Filter indices by character
                filtered_indices = self.test_indices[self.test_labels == char_index]
                if len(filtered_indices) == 0:
                    # No samples for this character, return random
                    filtered_indices = self.test_indices
        else:
            filtered_indices = self.test_indices
        
        # Sample random indices
        sample_count = min(count, len(filtered_indices))
        sampled_indices = random.sample(list(filtered_indices), sample_count)
        
        # Build results
        samples = []
        for idx in sampled_indices:
            image = self.test_images[idx]
            label = int(self.test_labels[idx])
            character = index_to_character(label)
            
            # Convert image to base64 for frontend
            # Image is already normalized to [-1, 1], convert to [0, 255] for display
            img_display = ((image.reshape(28, 28) + 1) / 2 * 255).astype(np.uint8)
            
            # Convert to base64
            from PIL import Image
            import io
            import base64
            
            pil_img = Image.fromarray(img_display, mode='L')
            buffer = io.BytesIO()
            pil_img.save(buffer, format='PNG')
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            img_data_url = f"data:image/png;base64,{img_base64}"
            
            samples.append({
                'index': int(idx),
                'image_data': img_data_url,
                'image_array': image.tolist(),  # For prediction
                'ground_truth': character,
                'ground_truth_index': label,
                'character_type': get_character_type(label)
            })
        
        return samples
    
    def get_sample_by_index(self, index: int) -> Optional[Dict]:
        """
        Get a specific test sample by index.
        
        Args:
            index: Index in test set
        
        Returns:
            Dictionary with sample data or None if invalid
        """
        if not self._loaded:
            self.load_test_data()
        
        if index < 0 or index >= len(self.test_labels):
            return None
        
        image = self.test_images[index]
        label = int(self.test_labels[index])
        character = index_to_character(label)
        
        # Convert to base64
        img_display = ((image.reshape(28, 28) + 1) / 2 * 255).astype(np.uint8)
        
        from PIL import Image
        import io
        import base64
        
        pil_img = Image.fromarray(img_display, mode='L')
        buffer = io.BytesIO()
        pil_img.save(buffer, format='PNG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        img_data_url = f"data:image/png;base64,{img_base64}"
        
        return {
            'index': int(index),
            'image_data': img_data_url,
            'image_array': image.tolist(),
            'ground_truth': character,
            'ground_truth_index': label,
            'character_type': get_character_type(label)
        }


# Global instance
_dataset_tester = None


def get_dataset_tester(data_dir: Optional[str] = None) -> DatasetTester:
    """
    Get or create global dataset tester instance.
    
    Args:
        data_dir: Path to EMNIST data directory
    
    Returns:
        DatasetTester instance
    """
    global _dataset_tester
    
    if _dataset_tester is None:
        _dataset_tester = DatasetTester(data_dir)
    
    return _dataset_tester

