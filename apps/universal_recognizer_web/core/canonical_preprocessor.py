"""Canonical preprocessing implementation aligned between training and inference."""

from __future__ import annotations

import base64
import io
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image

from .preprocess_contract import PreprocessContractV2, load_contract


TRANSFORMS = {
    "identity",
    "legacy_emnist_fix",
    "rot90",
    "rot180",
    "rot270",
    "flip_h",
    "flip_v",
    "flip_h_rot90",
    "flip_v_rot90",
}


def apply_transform(img: np.ndarray, transform_id: str) -> np.ndarray:
    if transform_id == "identity":
        return img
    if transform_id == "legacy_emnist_fix":
        return np.rot90(np.flip(img, axis=1), 3)
    if transform_id == "rot90":
        return np.rot90(img, 1)
    if transform_id == "rot180":
        return np.rot90(img, 2)
    if transform_id == "rot270":
        return np.rot90(img, 3)
    if transform_id == "flip_h":
        return np.flip(img, axis=1)
    if transform_id == "flip_v":
        return np.flip(img, axis=0)
    if transform_id == "flip_h_rot90":
        return np.rot90(np.flip(img, axis=1), 1)
    if transform_id == "flip_v_rot90":
        return np.rot90(np.flip(img, axis=0), 1)
    raise ValueError(f"Unsupported transform_id: {transform_id}")


class CanonicalPreprocessorV2:
    def __init__(self, contract: Optional[PreprocessContractV2] = None):
        self.contract = contract or load_contract()

    def _to_numpy(self, image_data: Any) -> Optional[np.ndarray]:
        try:
            if isinstance(image_data, np.ndarray):
                img = image_data.copy()
            elif isinstance(image_data, list):
                img = np.array(image_data, dtype=np.float32)
            elif isinstance(image_data, str):
                payload = image_data.split(",", 1)[1] if image_data.startswith("data:image") else image_data
                image_bytes = base64.b64decode(payload)
                image = Image.open(io.BytesIO(image_bytes)).convert("L")
                img = np.array(image, dtype=np.float32)
            elif isinstance(image_data, Image.Image):
                img = np.array(image_data.convert("L"), dtype=np.float32)
            else:
                return None

            if img.ndim == 1 and img.size == 784:
                img = img.reshape(28, 28)
            if img.ndim != 2:
                return None

            if img.max() > 1.0:
                img = img / 255.0
            return img.astype(np.float32)
        except Exception:
            return None

    def _resize(self, img: np.ndarray) -> np.ndarray:
        h, w = self.contract.target_size
        if img.shape == (h, w):
            return img
        pil = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8), mode="L")
        pil = pil.resize((w, h), Image.Resampling.BILINEAR)
        return np.asarray(pil, dtype=np.float32) / 255.0

    def _normalize(self, img_flat: np.ndarray) -> np.ndarray:
        mean, std, clip_min, clip_max, eps = self.contract.get_stats()

        if mean is None or std is None or mean.shape[0] != img_flat.shape[0] or std.shape[0] != img_flat.shape[0]:
            # Safe fallback until calibration stats are written.
            mean = np.full_like(img_flat, 0.5, dtype=np.float32)
            std = np.full_like(img_flat, 0.25, dtype=np.float32)

        z = (img_flat - mean) / (std + eps)
        z = np.clip(z, clip_min, clip_max)
        return np.tanh(z).astype(np.float32)

    def preprocess(
        self,
        image_data: Any,
        return_metrics: bool = False,
        return_debug: bool = False,
        skip_transform: bool = False,
        already_normalized: bool = False,
    ) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        img = self._to_numpy(image_data)
        if img is None:
            return None, None, None

        debug: Dict[str, Any] = {}
        if return_debug:
            debug["raw_shape"] = list(img.shape)
            debug["raw_min"] = float(img.min())
            debug["raw_max"] = float(img.max())

        # If test images already passed normalized tensors, trust them.
        if already_normalized and img.shape == (28, 28) and img.min() >= -1.1 and img.max() <= 1.1:
            processed = img.astype(np.float32).flatten().reshape(1, -1)
            metrics = self._quality_metrics(img, img)
            return processed, (metrics if return_metrics else None), (debug if return_debug else None)

        if not skip_transform:
            transform_id = self.contract.transform_id
            if transform_id not in TRANSFORMS:
                transform_id = "identity"
            img = apply_transform(img, transform_id)
            if return_debug:
                debug["transform_id"] = transform_id

        # polarity normalization: expect white foreground on black background
        if float(np.mean(img)) > self.contract.invert_threshold:
            img = 1.0 - img
            if return_debug:
                debug["inverted"] = True

        img = self._resize(img)
        flat = img.flatten().astype(np.float32)
        flat = self._normalize(flat)
        out = flat.reshape(1, -1)

        metrics = self._quality_metrics(img, flat.reshape(28, 28)) if return_metrics else None
        if return_debug:
            debug.update(
                {
                    "final_min": float(out.min()),
                    "final_max": float(out.max()),
                    "contract_version": self.contract.version,
                    "contract_checksum": self.contract.checksum,
                }
            )
        return out, metrics, (debug if return_debug else None)

    def _quality_metrics(self, img_input: np.ndarray, img_norm: np.ndarray) -> Dict[str, Any]:
        stroke = img_input > 0.1
        bbox = np.argwhere(stroke)
        if bbox.size == 0:
            size_ratio = 0.0
            center_offset = 1.0
        else:
            y0, x0 = bbox.min(axis=0)
            y1, x1 = bbox.max(axis=0)
            size_ratio = float(((y1 - y0 + 1) * (x1 - x0 + 1)) / (28 * 28))
            cy, cx = bbox.mean(axis=0)
            center_offset = float(np.sqrt((cy - 13.5) ** 2 + (cx - 13.5) ** 2) / 20.0)

        clarity = float(np.std(img_norm))
        overall = max(0.0, min(100.0, (clarity * 120 + (1 - center_offset) * 40 + size_ratio * 40)))
        return {
            "overall_score": overall,
            "clarity_score": float(min(100.0, clarity * 140)),
            "size_score": float(min(100.0, size_ratio * 250)),
            "centering_score": float(max(0.0, 100.0 - center_offset * 100.0)),
            "stroke_score": float(min(100.0, np.mean(stroke) * 300)),
            "metrics": {
                "size_ratio": size_ratio,
                "center_offset": center_offset,
            },
        }


def preprocess_for_prediction(image_data, is_test_image: bool = False, return_debug: bool = False):
    pp = CanonicalPreprocessorV2()
    out, _, debug = pp.preprocess(
        image_data,
        return_metrics=False,
        return_debug=return_debug,
        skip_transform=is_test_image,
        already_normalized=is_test_image,
    )
    if return_debug:
        return out, debug
    return out


def preprocess_with_metrics(image_data, is_test_image: bool = False, return_debug: bool = False):
    pp = CanonicalPreprocessorV2()
    out, metrics, debug = pp.preprocess(
        image_data,
        return_metrics=True,
        return_debug=return_debug,
        skip_transform=is_test_image,
        already_normalized=is_test_image,
    )
    if return_debug:
        return out, metrics, debug
    return out, metrics
