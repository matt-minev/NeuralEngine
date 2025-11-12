"""
Data augmentation module for universal character recognition.

Provides on-the-fly augmentation for improved generalization and accessibility support.
"""

import numpy as np
from typing import Tuple, Optional
import scipy.ndimage


class DataAugmentation:
    """Data augmentation for character recognition with accessibility support."""
    
    def __init__(self, 
                 mirror_prob: float = 0.5,
                 rotation_range: float = 15.0,
                 scale_range: Tuple[float, float] = (0.9, 1.1),
                 translation_range: int = 2,
                 noise_std: float = 0.05,
                 contrast_range: Tuple[float, float] = (0.8, 1.2),
                 brightness_range: Tuple[float, float] = (0.9, 1.1),
                 elastic_alpha: float = 1.0,
                 elastic_sigma: float = 5.0):
        """
        Initialize augmentation parameters.
        
        Args:
            mirror_prob: Probability of horizontal mirroring (critical for dyslexia)
            rotation_range: Maximum rotation angle in degrees
            scale_range: Scaling range (min, max)
            translation_range: Maximum translation in pixels
            noise_std: Standard deviation for Gaussian noise
            contrast_range: Contrast adjustment range
            brightness_range: Brightness adjustment range
            elastic_alpha: Elastic deformation strength
            elastic_sigma: Elastic deformation smoothness
        """
        self.mirror_prob = mirror_prob
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.translation_range = translation_range
        self.noise_std = noise_std
        self.contrast_range = contrast_range
        self.brightness_range = brightness_range
        self.elastic_alpha = elastic_alpha
        self.elastic_sigma = elastic_sigma
    
    def augment_image(self, image: np.ndarray, apply_all: bool = False) -> np.ndarray:
        """
        Apply random augmentations to a single image.
        
        Args:
            image: 28x28 grayscale image (values in [0, 1] or [-1, 1])
            apply_all: If True, apply all augmentations (for testing)
        
        Returns:
            Augmented image with same shape
        """
        img = image.copy()
        
        # Ensure 2D
        if img.ndim == 1:
            img = img.reshape(28, 28)
        
        # Horizontal mirroring (critical for dyslexia support)
        if apply_all or np.random.random() < self.mirror_prob:
            img = np.flip(img, axis=1)
        
        # Rotation
        if apply_all or np.random.random() < 0.5:
            angle = np.random.uniform(-self.rotation_range, self.rotation_range)
            img = self._rotate_image(img, angle)
        
        # Scaling
        if apply_all or np.random.random() < 0.5:
            scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
            img = self._scale_image(img, scale)
        
        # Translation
        if apply_all or np.random.random() < 0.5:
            tx = np.random.randint(-self.translation_range, self.translation_range + 1)
            ty = np.random.randint(-self.translation_range, self.translation_range + 1)
            img = self._translate_image(img, tx, ty)
        
        # Elastic deformation (handwriting distortion)
        if apply_all or np.random.random() < 0.3:
            img = self._elastic_deform(img)
        
        # Contrast adjustment
        if apply_all or np.random.random() < 0.5:
            contrast = np.random.uniform(self.contrast_range[0], self.contrast_range[1])
            img = self._adjust_contrast(img, contrast)
        
        # Brightness adjustment
        if apply_all or np.random.random() < 0.5:
            brightness = np.random.uniform(self.brightness_range[0], self.brightness_range[1])
            img = self._adjust_brightness(img, brightness)
        
        # Gaussian noise
        if apply_all or np.random.random() < 0.3:
            img = self._add_noise(img)
        
        return img
    
    def _rotate_image(self, img: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image by angle degrees."""
        return scipy.ndimage.rotate(img, angle, reshape=False, order=1, mode='constant', cval=0.0)
    
    def _scale_image(self, img: np.ndarray, scale: float) -> np.ndarray:
        """Scale image by factor."""
        h, w = img.shape
        center = (h // 2, w // 2)
        
        # Create transformation matrix
        M = np.array([[scale, 0, center[0] * (1 - scale)],
                      [0, scale, center[1] * (1 - scale)]])
        
        # Apply affine transformation
        return scipy.ndimage.affine_transform(img, M, output_shape=img.shape, order=1, mode='constant', cval=0.0)
    
    def _translate_image(self, img: np.ndarray, tx: int, ty: int) -> np.ndarray:
        """Translate image by (tx, ty) pixels."""
        M = np.array([[1, 0, -ty],
                      [0, 1, -tx]])
        return scipy.ndimage.affine_transform(img, M, output_shape=img.shape, order=1, mode='constant', cval=0.0)
    
    def _elastic_deform(self, img: np.ndarray) -> np.ndarray:
        """Apply elastic deformation."""
        h, w = img.shape
        
        # Generate random displacement fields
        dx = scipy.ndimage.gaussian_filter(
            (np.random.rand(h, w) * 2 - 1) * self.elastic_alpha,
            self.elastic_sigma, mode='constant'
        )
        dy = scipy.ndimage.gaussian_filter(
            (np.random.rand(h, w) * 2 - 1) * self.elastic_alpha,
            self.elastic_sigma, mode='constant'
        )
        
        # Create coordinate grids
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        x_new = np.clip(x + dx, 0, w - 1)
        y_new = np.clip(y + dy, 0, h - 1)
        
        # Apply deformation
        return scipy.ndimage.map_coordinates(img, [y_new, x_new], order=1, mode='constant', cval=0.0)
    
    def _adjust_contrast(self, img: np.ndarray, factor: float) -> np.ndarray:
        """Adjust image contrast."""
        # Preserve mean
        mean = np.mean(img)
        return np.clip((img - mean) * factor + mean, img.min(), img.max())
    
    def _adjust_brightness(self, img: np.ndarray, factor: float) -> np.ndarray:
        """Adjust image brightness."""
        return np.clip(img * factor, img.min(), img.max())
    
    def _add_noise(self, img: np.ndarray) -> np.ndarray:
        """Add Gaussian noise."""
        noise = np.random.normal(0, self.noise_std, img.shape)
        return np.clip(img + noise, img.min(), img.max())
    
    def augment_batch(self, X_batch: np.ndarray, y_batch: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Augment a batch of images.
        
        Args:
            X_batch: Batch of images (batch_size, 784) or (batch_size, 28, 28)
            y_batch: Optional labels (unchanged)
        
        Returns:
            Augmented batch and labels
        """
        batch_size = X_batch.shape[0]
        augmented = []
        
        for i in range(batch_size):
            img = X_batch[i]
            aug_img = self.augment_image(img)
            
            # Flatten if needed
            if aug_img.ndim == 2:
                aug_img = aug_img.flatten()
            
            augmented.append(aug_img)
        
        return np.array(augmented, dtype=X_batch.dtype), y_batch


def create_augmentation_pipeline(phase: str = 'full') -> DataAugmentation:
    """
    Create augmentation pipeline for different training phases.
    
    Args:
        phase: 'full', 'reduced', 'minimal', or 'none'
    
    Returns:
        Configured DataAugmentation instance
    """
    if phase == 'full':
        return DataAugmentation(
            mirror_prob=0.5,
            rotation_range=15.0,
            scale_range=(0.9, 1.1),
            translation_range=2,
            noise_std=0.05,
            contrast_range=(0.8, 1.2),
            brightness_range=(0.9, 1.1),
            elastic_alpha=1.0,
            elastic_sigma=5.0
        )
    elif phase == 'reduced':
        return DataAugmentation(
            mirror_prob=0.3,
            rotation_range=10.0,
            scale_range=(0.95, 1.05),
            translation_range=1,
            noise_std=0.03,
            contrast_range=(0.9, 1.1),
            brightness_range=(0.95, 1.05),
            elastic_alpha=0.5,
            elastic_sigma=5.0
        )
    elif phase == 'minimal':
        return DataAugmentation(
            mirror_prob=0.2,
            rotation_range=5.0,
            scale_range=(0.98, 1.02),
            translation_range=1,
            noise_std=0.02,
            contrast_range=(0.95, 1.05),
            brightness_range=(0.98, 1.02),
            elastic_alpha=0.3,
            elastic_sigma=5.0
        )
    else:  # none
        return DataAugmentation(
            mirror_prob=0.0,
            rotation_range=0.0,
            scale_range=(1.0, 1.0),
            translation_range=0,
            noise_std=0.0,
            contrast_range=(1.0, 1.0),
            brightness_range=(1.0, 1.0),
            elastic_alpha=0.0,
            elastic_sigma=5.0
        )

