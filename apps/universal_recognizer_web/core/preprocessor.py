"""
Advanced preprocessing framework for universal character recognition.

State-of-the-art preprocessing pipeline that automatically transforms user drawings
to match EMNIST dataset format for maximum accuracy.
"""

import numpy as np
from PIL import Image
import io
import base64
from typing import Tuple, Dict, Optional

try:
    from scipy import ndimage
    from scipy.ndimage import binary_erosion, binary_dilation, label
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    # Fallback implementations
    def binary_erosion(arr, kernel, iterations=1):
        return arr
    def binary_dilation(arr, kernel, iterations=1):
        return arr


class AdvancedPreprocessor:
    """
    Advanced preprocessing pipeline for user-drawn characters.
    
    Automatically transforms drawings to match EMNIST dataset characteristics:
    - Automatic centering
    - Optimal scaling
    - Stroke normalization
    - Noise reduction
    - Contrast enhancement
    - Bounding box extraction
    - Intensity normalization
    """
    
    def __init__(self):
        """Initialize preprocessor with EMNIST-specific parameters."""
        # EMNIST characteristics
        self.target_size = 28
        self.optimal_char_size_ratio = 0.4  # Characters should fill ~40% of image
        self.padding_ratio = 0.1  # 10% padding around character
        
    def preprocess(self, image_data, return_metrics: bool = False, is_test_image: bool = False, return_debug: bool = False) -> Tuple[np.ndarray, Optional[Dict], Optional[Dict]]:
        """
        MINIMAL preprocessing pipeline with 180-DEGREE ROTATION.
        Only does: convert → rotate 180° → resize → normalize → flatten
        
        Args:
            image_data: Base64 string, numpy array, or PIL Image
            return_metrics: Whether to return quality metrics for display
            is_test_image: If True, skip the flip for test images
            return_debug: Whether to return debug images for visualization
        
        Returns:
            Preprocessed image (1, 784) ready for model
            Optional quality metrics dict
            Optional debug images dict
        """
        # Step 1: Convert to numpy array (basic conversion only, no inversion)
        img_array = self._to_numpy_minimal(image_data)
        if img_array is None:
            return (None, None, None) if return_debug else (None, None)
        
        # Store original for debug
        debug_images = {}
        if return_debug:
            debug_images['original'] = self._image_to_base64(img_array)
        
        # Step 2: ROTATE 180 DEGREES for user-drawn images (fixes M/W confusion)
        if not is_test_image:
            img_array = np.rot90(img_array, 2)  # Rotate 180 degrees (upside down)
            if return_debug:
                debug_images['flipped_upside_down'] = self._image_to_base64(img_array)
        
        # Step 3: Resize to 28x28 (REQUIRED - model needs this size)
        if img_array.shape != (self.target_size, self.target_size):
            img_array = self._resize_to_target(img_array)
            if return_debug:
                debug_images['after_resize'] = self._image_to_base64(img_array)
        
        # Step 4: Normalize to [0, 1]
        if img_array.max() > 1.0:
            img_array = img_array / 255.0
        
        # Step 5: EMNIST normalization (REQUIRED - model expects this)
        img_final = self._emnist_normalize(img_array)
        
        if return_debug:
            debug_images['final'] = self._image_to_base64(self._denormalize_for_display(img_final))
        
        # Calculate quality metrics if requested
        metrics = None
        if return_metrics:
            metrics = self._calculate_quality_metrics(
                img_array, 
                img_final, 
                {'bbox_width': 28, 'bbox_height': 28, 'center_offset': 0.0}
            )
        
        # Flatten for neural network
        result = img_final.flatten().reshape(1, -1)
        if return_debug:
            return result, metrics, debug_images
        return result, metrics
    
    def _to_numpy_minimal(self, image_data) -> Optional[np.ndarray]:
        """Convert to numpy with MINIMAL processing - no inversion, no checks."""
        try:
            if isinstance(image_data, np.ndarray):
                img = image_data.copy()
            elif isinstance(image_data, list):
                img = np.array(image_data, dtype=np.float32)
            elif isinstance(image_data, str):
                # Base64 string
                if image_data.startswith('data:image'):
                    image_data = image_data.split(',')[1]
                image_bytes = base64.b64decode(image_data)
                image = Image.open(io.BytesIO(image_bytes))
                image = image.convert('L')
                img = np.array(image, dtype=np.float32)
            elif isinstance(image_data, Image.Image):
                image = image_data.convert('L')
                img = np.array(image, dtype=np.float32)
            else:
                return None
            
            # Handle 1D arrays
            if img.ndim == 1:
                if img.size == 784:
                    img = img.reshape(28, 28)
                else:
                    return None
            
            # Ensure 2D array
            if img.ndim != 2:
                return None
            
            # Normalize to [0, 1] if needed
            if img.max() > 1.0:
                img = img / 255.0
            
            # NO INVERSION - just return as-is
            
            return img
        except Exception as e:
            print(f"Error converting to numpy: {e}")
            return None
    
    def _to_numpy(self, image_data) -> Optional[np.ndarray]:
        """Convert various input types to numpy array."""
        try:
            if isinstance(image_data, np.ndarray):
                img = image_data.copy()
            elif isinstance(image_data, list):
                img = np.array(image_data, dtype=np.float32)
            elif isinstance(image_data, str):
                # Base64 string
                if image_data.startswith('data:image'):
                    image_data = image_data.split(',')[1]
                image_bytes = base64.b64decode(image_data)
                image = Image.open(io.BytesIO(image_bytes))
                image = image.convert('L')
                img = np.array(image, dtype=np.float32)
            elif isinstance(image_data, Image.Image):
                image = image_data.convert('L')
                img = np.array(image, dtype=np.float32)
            else:
                return None
            
            # Handle 1D arrays from test mode (784 elements = 28x28 flattened)
            if img.ndim == 1:
                if img.size == 784:
                    # Reshape flattened array to 28x28
                    img = img.reshape(28, 28)
                else:
                    print(f"Warning: Unexpected 1D array size: {img.size}")
                    return None
            
            # Ensure 2D array
            if img.ndim != 2:
                print(f"Warning: Unexpected array dimensions: {img.ndim}")
                return None
            
            # Normalize to [0, 1] if needed
            if img.max() > 1.0:
                img = img / 255.0
            
            # Invert if background is white (EMNIST is black on white)
            if np.mean(img) > 0.5:
                img = 1.0 - img
            
            return img
        except Exception as e:
            print(f"Error converting to numpy: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _fix_emnist_orientation(self, img: np.ndarray) -> np.ndarray:
        """
        Fix EMNIST image orientation to match training data.
        
        For user-drawn images, we only need to rotate 180 degrees (upside down).
        This is because:
        - User draws M, model sees it upside down as W without this fix
        - User draws W, model sees it upside down as M without this fix
        
        Args:
            img: 2D numpy array (H, W)
        
        Returns:
            Image rotated 180 degrees (upside down)
        """
        # Rotate 180 degrees (k=2 means 2*90 = 180 degrees)
        rotated = np.rot90(img, 2)
        
        return rotated
    
    def _extract_and_center(self, img: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Extract tight bounding box and center the character.
        
        Returns:
            Centered image, metrics dict
        """
        # Threshold to get binary image
        threshold = 0.1
        binary = img > threshold
        
        # Find bounding box
        rows = np.any(binary, axis=1)
        cols = np.any(binary, axis=0)
        
        if not np.any(rows) or not np.any(cols):
            # Empty image - return centered empty image
            centered = np.zeros((self.target_size, self.target_size), dtype=np.float32)
            return centered, {'bbox_width': 0, 'bbox_height': 0, 'center_offset': 0.0}
        
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        
        # Extract bounding box with padding
        h, w = img.shape
        bbox_h = rmax - rmin + 1
        bbox_w = cmax - cmin + 1
        
        # Add padding
        pad_h = int(bbox_h * self.padding_ratio)
        pad_w = int(bbox_w * self.padding_ratio)
        
        rmin = max(0, rmin - pad_h)
        rmax = min(h - 1, rmax + pad_h)
        cmin = max(0, cmin - pad_w)
        cmax = min(w - 1, cmax + pad_w)
        
        # Extract region
        region = img[rmin:rmax+1, cmin:cmax+1]
        
        # Calculate center offset (for metrics)
        center_y = (rmin + rmax) / 2.0
        center_x = (cmin + cmax) / 2.0
        img_center_y = h / 2.0
        img_center_x = w / 2.0
        center_offset = np.sqrt((center_x - img_center_x)**2 + (center_y - img_center_y)**2)
        
        # Create centered image
        new_h, new_w = region.shape
        centered = np.zeros((self.target_size, self.target_size), dtype=np.float32)
        
        # If region is larger than target, scale it down first
        if new_h > self.target_size or new_w > self.target_size:
            scale = min(self.target_size / new_h, self.target_size / new_w) * 0.9  # 90% to add padding
            new_h = int(new_h * scale)
            new_w = int(new_w * scale)
            # Resize region using PIL for quality
            from PIL import Image
            pil_region = Image.fromarray((region * 255).astype(np.uint8))
            pil_region = pil_region.resize((new_w, new_h), Image.Resampling.LANCZOS)
            region = np.array(pil_region, dtype=np.float32) / 255.0
        
        # Calculate position to center the region
        start_y = (self.target_size - new_h) // 2
        start_x = (self.target_size - new_w) // 2
        
        # Ensure indices are within bounds
        start_y = max(0, start_y)
        start_x = max(0, start_x)
        end_y = min(start_y + new_h, self.target_size)
        end_x = min(start_x + new_w, self.target_size)
        
        # Calculate actual region size to place
        region_h = end_y - start_y
        region_w = end_x - start_x
        
        # Ensure region dimensions match
        if region_h > 0 and region_w > 0:
            # Crop region if needed to match available space
            region_to_place = region[:min(region_h, region.shape[0]), :min(region_w, region.shape[1])]
            # Ensure exact match
            if region_to_place.shape[0] != region_h or region_to_place.shape[1] != region_w:
                # Resize to exact dimensions if needed
                from PIL import Image
                pil_region = Image.fromarray((region_to_place * 255).astype(np.uint8))
                pil_region = pil_region.resize((region_w, region_h), Image.Resampling.LANCZOS)
                region_to_place = np.array(pil_region, dtype=np.float32) / 255.0
            
            centered[start_y:end_y, start_x:end_x] = region_to_place
        
        metrics = {
            'bbox_width': bbox_w,
            'bbox_height': bbox_h,
            'center_offset': float(center_offset),
            'original_size': (h, w)
        }
        
        return centered, metrics
    
    def _optimal_scale(self, img: np.ndarray) -> np.ndarray:
        """
        Scale character to optimal size matching EMNIST distribution.
        
        EMNIST characters typically fill ~40% of the image.
        """
        # Calculate current character size
        threshold = 0.1
        binary = img > threshold
        char_pixels = np.sum(binary)
        total_pixels = img.size
        current_ratio = char_pixels / total_pixels
        
        # Calculate scale factor to reach optimal ratio
        if current_ratio > 0:
            scale_factor = np.sqrt(self.optimal_char_size_ratio / current_ratio)
            # Clamp scale factor to reasonable range
            scale_factor = np.clip(scale_factor, 0.5, 2.0)
        else:
            scale_factor = 1.0
        
        # Apply scaling using interpolation
        if scale_factor != 1.0:
            from scipy.ndimage import zoom
            img = zoom(img, scale_factor, order=1)
            
            # Crop or pad to maintain size
            h, w = img.shape
            if h > self.target_size or w > self.target_size:
                # Crop to center
                start_y = (h - self.target_size) // 2
                start_x = (w - self.target_size) // 2
                img = img[start_y:start_y+self.target_size, start_x:start_x+self.target_size]
            elif h < self.target_size or w < self.target_size:
                # Pad to center
                new_img = np.zeros((self.target_size, self.target_size), dtype=np.float32)
                start_y = (self.target_size - h) // 2
                start_x = (self.target_size - w) // 2
                new_img[start_y:start_y+h, start_x:start_x+w] = img
                img = new_img
        
        return img
    
    def _normalize_strokes(self, img: np.ndarray) -> np.ndarray:
        """
        Normalize stroke thickness to match EMNIST characteristics.
        
        Uses morphological operations to standardize stroke width.
        """
        if not SCIPY_AVAILABLE:
            # Skip morphological operations if scipy not available
            return img
        
        # Threshold
        threshold = 0.1
        binary = (img > threshold).astype(np.uint8)
        
        # Thin strokes if too thick, thicken if too thin
        # Use opening to remove thin artifacts, then closing to fill gaps
        kernel_size = 1
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        # Opening (erosion then dilation) - removes thin lines
        opened = binary_erosion(binary, kernel, iterations=1)
        opened = binary_dilation(opened, kernel, iterations=1)
        
        # Closing (dilation then erosion) - fills gaps
        closed = binary_dilation(opened, kernel, iterations=1)
        closed = binary_erosion(closed, kernel, iterations=1)
        
        # Convert back to float and blend with original
        normalized = closed.astype(np.float32)
        
        # Blend with original to preserve intensity gradients
        img_normalized = img * 0.7 + normalized * 0.3
        
        return img_normalized
    
    def _reduce_noise(self, img: np.ndarray) -> np.ndarray:
        """
        Reduce noise using Gaussian blur and thresholding.
        """
        if SCIPY_AVAILABLE:
            # Light Gaussian blur to smooth
            sigma = 0.5
            img_blurred = ndimage.gaussian_filter(img, sigma=sigma)
        else:
            # Simple box blur fallback
            kernel_size = 3
            kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
            # Simple convolution
            h, w = img.shape
            img_blurred = np.zeros_like(img)
            pad = kernel_size // 2
            img_padded = np.pad(img, pad, mode='edge')
            for i in range(h):
                for j in range(w):
                    img_blurred[i, j] = np.sum(img_padded[i:i+kernel_size, j:j+kernel_size] * kernel)
        
        # Adaptive thresholding to clean up
        threshold = 0.15
        img_clean = np.where(img_blurred > threshold, img_blurred, 0.0)
        
        # Normalize back to [0, 1]
        if img_clean.max() > 0:
            img_clean = img_clean / img_clean.max()
        
        return img_clean
    
    def _enhance_contrast(self, img: np.ndarray) -> np.ndarray:
        """
        Enhance contrast to match EMNIST characteristics.
        
        EMNIST has high contrast between strokes and background.
        """
        # Apply contrast enhancement
        contrast_factor = 1.3
        img_contrast = np.clip(img * contrast_factor, 0, 1)
        
        # Apply gamma correction for better contrast
        gamma = 0.8
        img_contrast = np.power(img_contrast, gamma)
        
        return img_contrast
    
    def _resize_to_target(self, img: np.ndarray) -> np.ndarray:
        """Resize to target size (28x28) using high-quality interpolation."""
        if img.shape == (self.target_size, self.target_size):
            return img
        
        # Use PIL for high-quality resizing
        pil_img = Image.fromarray((img * 255).astype(np.uint8))
        pil_img = pil_img.resize((self.target_size, self.target_size), Image.Resampling.LANCZOS)
        resized = np.array(pil_img, dtype=np.float32) / 255.0
        
        return resized
    
    def _emnist_normalize(self, img: np.ndarray) -> np.ndarray:
        """
        Apply EMNIST-specific normalization matching training preprocessing EXACTLY.
        
        Training preprocessing (from data_loader.py):
        1. Normalize to [0, 1] (divide by 255)
        2. Flatten to (784,)
        3. Per-pixel normalization: (X - mean) / std where mean/std are per pixel (axis=0)
        4. Apply tanh to scale to [-1, 1]
        
        For single image, we match this by:
        - Normalize to [0, 1]
        - Flatten
        - Use global mean/std (since it's a single image, not a batch)
        - Apply tanh
        - Reshape back
        """
        # Ensure [0, 1] range
        if img.max() > 1.0:
            img = img / 255.0
        
        # Flatten for normalization (matching training which processes flattened data)
        img_flat = img.flatten()
        
        # For single image, use global mean/std (training uses per-pixel across batch)
        # But for single image, global mean/std is equivalent
        mean = np.mean(img_flat, keepdims=True)
        std = np.std(img_flat, keepdims=True) + 1e-8
        img_normalized = (img_flat - mean) / std
        
        # Scale to [-1, 1] using tanh (matching training exactly)
        img_normalized = np.tanh(img_normalized)
        
        # Reshape back to 28x28
        return img_normalized.reshape(28, 28)
    
    def _calculate_quality_metrics(self, original: np.ndarray, processed: np.ndarray, 
                                   bbox_metrics: Dict) -> Dict:
        """
        Calculate quality metrics for display (not used in prediction).
        
        Returns metrics about centering, size, clarity, etc.
        """
        # Clarity (contrast)
        clarity_score = float(np.std(processed) * 100)
        
        # Size (character coverage)
        threshold = 0.1
        char_pixels = np.sum(processed > threshold)
        total_pixels = processed.size
        size_ratio = char_pixels / total_pixels
        size_score = float(min(size_ratio * 200, 100))
        
        # Centering (from bbox metrics)
        center_offset = bbox_metrics.get('center_offset', 0.0)
        centering_score = float(max(100 - (center_offset / 14.0 * 100), 0))
        
        # Stroke quality (edge strength)
        if SCIPY_AVAILABLE:
            try:
                edges = ndimage.sobel(processed)
                edge_strength = np.mean(np.abs(edges))
                stroke_score = float(min(edge_strength * 100, 100))
            except:
                stroke_score = 50.0
        else:
            stroke_score = 50.0
        
        # Overall score
        overall_score = float((clarity_score + size_score + centering_score + stroke_score) / 4)
        
        return {
            'overall_score': overall_score,
            'clarity_score': clarity_score,
            'size_score': size_score,
            'centering_score': centering_score,
            'stroke_score': stroke_score,
            'metrics': {
                'contrast': float(np.std(processed)),
                'size_ratio': float(size_ratio),
                'center_offset': float(center_offset),
                'edge_strength': float(edge_strength) if 'edge_strength' in locals() else 0.0
            }
        }
    
    def _image_to_base64(self, img: np.ndarray) -> str:
        """Convert numpy image to base64 for display."""
        try:
            # Ensure [0, 1] range for display
            if img.min() < 0:
                # Denormalize if needed
                img_display = (img + 1) / 2.0
            else:
                img_display = img.copy()
            
            img_display = np.clip(img_display, 0, 1)
            img_uint8 = (img_display * 255).astype(np.uint8)
            
            from PIL import Image
            pil_img = Image.fromarray(img_uint8, mode='L')
            buffer = io.BytesIO()
            pil_img.save(buffer, format='PNG')
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            return f"data:image/png;base64,{img_base64}"
        except Exception as e:
            print(f"Error converting image to base64: {e}")
            return ""
    
    def _denormalize_for_display(self, img: np.ndarray) -> np.ndarray:
        """Convert normalized [-1, 1] image back to [0, 1] for display."""
        # Reverse tanh: approximate by (img + 1) / 2
        # This is approximate but good enough for visualization
        img_display = (img + 1) / 2.0
        return np.clip(img_display, 0, 1)


def preprocess_for_prediction(image_data, is_test_image: bool = False, return_debug: bool = False) -> np.ndarray:
    """
    Convenience function for prediction pipeline.
    
    Args:
        image_data: Base64 string, numpy array, or PIL Image
        is_test_image: If True, skip EMNIST orientation fix (test images already fixed)
        return_debug: If True, return debug images as well
    
    Returns:
        Preprocessed image (1, 784) ready for model, or (image, debug_images) if return_debug
    """
    preprocessor = AdvancedPreprocessor()
    result = preprocessor.preprocess(image_data, return_metrics=False, is_test_image=is_test_image, return_debug=return_debug)
    if return_debug:
        return result[0], result[2]  # image, debug_images
    return result[0]


def preprocess_with_metrics(image_data, is_test_image: bool = False, return_debug: bool = False) -> Tuple[np.ndarray, Dict, Optional[Dict]]:
    """
    Preprocess and return quality metrics for display.
    
    Args:
        image_data: Base64 string, numpy array, or PIL Image
        is_test_image: If True, skip EMNIST orientation fix (test images already fixed)
        return_debug: If True, return debug images as well
    
    Returns:
        Preprocessed image (1, 784), quality metrics dict, optional debug images dict
    """
    preprocessor = AdvancedPreprocessor()
    result = preprocessor.preprocess(image_data, return_metrics=True, is_test_image=is_test_image, return_debug=return_debug)
    if return_debug:
        return result[0], result[1], result[2]  # image, metrics, debug_images
    return result[0], result[1]

