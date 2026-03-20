"""Canonical preprocessing for live handwritten digit recognition."""

from __future__ import annotations

import base64
import io
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw


TARGET_SIZE = (28, 28)
LINE_WIDTH = 18
INVERT_THRESHOLD = 0.65
FOREGROUND_THRESHOLD = 0.08
TARGET_GLYPH_BOX = 20
DENOISE_THRESHOLD = 0.08
MIN_NEIGHBORS = 2


class DigitCanonicalPreprocessor:
    """Crop, scale, center, and normalize live strokes to match the digit datasets."""

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
                img = img.reshape(TARGET_SIZE)
            if img.ndim != 2:
                return None

            if img.max() > 1.0:
                img = img / 255.0

            return img.astype(np.float32)
        except Exception:
            return None

    def _parse_payload(self, image_data: Any) -> Tuple[Optional[List[Dict[str, Any]]], Optional[np.ndarray], int, int]:
        if isinstance(image_data, dict):
            strokes = image_data.get("strokes")
            raster = image_data.get("raster") or image_data.get("image")
            canvas = image_data.get("canvas", {})
            if not isinstance(canvas, dict):
                canvas = {}

            width = int(canvas.get("width", 280))
            height = int(canvas.get("height", 280))
            raster_np = self._to_numpy(raster) if raster is not None else None

            return strokes if isinstance(strokes, list) else None, raster_np, width, height

        return None, self._to_numpy(image_data), 280, 280

    def _rasterize_strokes(self, strokes: List[Dict[str, Any]], canvas_w: int, canvas_h: int) -> np.ndarray:
        img = Image.new("L", (canvas_w, canvas_h), 0)
        draw = ImageDraw.Draw(img)

        for stroke in strokes:
            points = stroke.get("points", []) if isinstance(stroke, dict) else []
            xy: List[Tuple[float, float]] = []

            for point in points:
                if isinstance(point, dict):
                    x = point.get("x")
                    y = point.get("y")
                elif isinstance(point, (list, tuple)) and len(point) >= 2:
                    x, y = point[0], point[1]
                else:
                    continue

                if x is None or y is None:
                    continue

                x = float(x)
                y = float(y)

                if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
                    x *= canvas_w
                    y *= canvas_h

                xy.append((x, y))

            if len(xy) >= 2:
                draw.line(xy, fill=255, width=LINE_WIDTH, joint="curve")
            elif len(xy) == 1:
                x, y = xy[0]
                radius = LINE_WIDTH / 2.0
                draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=255)

        return np.asarray(img, dtype=np.float32) / 255.0

    def _denoise(self, img: np.ndarray) -> np.ndarray:
        foreground = (img > DENOISE_THRESHOLD).astype(np.uint8)
        padded = np.pad(foreground, 1, mode="constant")
        neighbors = (
            padded[:-2, :-2] + padded[:-2, 1:-1] + padded[:-2, 2:] +
            padded[1:-1, :-2] + padded[1:-1, 2:] +
            padded[2:, :-2] + padded[2:, 1:-1] + padded[2:, 2:]
        )
        keep = (foreground == 1) & (neighbors >= MIN_NEIGHBORS)
        out = img.copy()
        out[~keep] = 0.0
        return out

    def _translate_no_wrap(self, img: np.ndarray, ty: int, tx: int) -> np.ndarray:
        out = np.zeros_like(img)
        height, width = img.shape

        src_y0 = max(0, -ty)
        src_y1 = min(height, height - ty) if ty >= 0 else height
        dst_y0 = max(0, ty)
        dst_y1 = min(height, height + ty) if ty < 0 else height

        src_x0 = max(0, -tx)
        src_x1 = min(width, width - tx) if tx >= 0 else width
        dst_x0 = max(0, tx)
        dst_x1 = min(width, width + tx) if tx < 0 else width

        if src_y1 > src_y0 and src_x1 > src_x0 and dst_y1 > dst_y0 and dst_x1 > dst_x0:
            out[dst_y0:dst_y1, dst_x0:dst_x1] = img[src_y0:src_y1, src_x0:src_x1]

        return out

    def _crop_scale_center(self, img: np.ndarray) -> np.ndarray:
        target_h, target_w = TARGET_SIZE
        foreground = np.argwhere(img > FOREGROUND_THRESHOLD)
        canvas = np.zeros((target_h, target_w), dtype=np.float32)

        if foreground.size == 0:
            return canvas

        y0, x0 = foreground.min(axis=0)
        y1, x1 = foreground.max(axis=0)
        crop = img[y0:y1 + 1, x0:x1 + 1]

        crop_h, crop_w = crop.shape
        scale = min(TARGET_GLYPH_BOX / max(crop_h, 1), TARGET_GLYPH_BOX / max(crop_w, 1))
        resized_h = max(1, int(round(crop_h * scale)))
        resized_w = max(1, int(round(crop_w * scale)))

        pil = Image.fromarray((np.clip(crop, 0.0, 1.0) * 255).astype(np.uint8))
        resized = np.asarray(
            pil.resize((resized_w, resized_h), Image.Resampling.BILINEAR),
            dtype=np.float32,
        ) / 255.0

        start_y = (target_h - resized_h) // 2
        start_x = (target_w - resized_w) // 2
        canvas[start_y:start_y + resized_h, start_x:start_x + resized_w] = resized

        if np.sum(canvas) > 1e-6:
            yy, xx = np.indices(canvas.shape)
            mass = float(np.sum(canvas))
            center_y = float(np.sum(yy * canvas) / mass)
            center_x = float(np.sum(xx * canvas) / mass)
            shift_y = int(round((target_h - 1) / 2.0 - center_y))
            shift_x = int(round((target_w - 1) / 2.0 - center_x))
            canvas = self._translate_no_wrap(canvas, shift_y, shift_x)

        return canvas

    def preprocess_live(self, image_data: Any) -> Optional[np.ndarray]:
        strokes, raster_np, canvas_w, canvas_h = self._parse_payload(image_data)

        if strokes:
            stage = self._rasterize_strokes(strokes, canvas_w, canvas_h)
        elif raster_np is not None:
            stage = raster_np
        else:
            return None

        if float(np.mean(stage)) > INVERT_THRESHOLD:
            stage = 1.0 - stage

        denoised = self._denoise(stage)
        centered = self._crop_scale_center(denoised)
        return centered.reshape(1, -1).astype(np.float32)

    def preprocess_dataset(self, image_data: Any) -> Optional[np.ndarray]:
        arr = self._to_numpy(image_data)
        if arr is None:
            return None
        return arr.reshape(1, -1).astype(np.float32)


preprocessor = DigitCanonicalPreprocessor()


def preprocess_live_payload(image_data: Any) -> Optional[np.ndarray]:
    return preprocessor.preprocess_live(image_data)


def preprocess_dataset_image(image_data: Any) -> Optional[np.ndarray]:
    return preprocessor.preprocess_dataset(image_data)
