"""Canonical preprocessing implementation aligned between training and inference."""

from __future__ import annotations

import base64
import io
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

from .preprocess_contract import PreprocessContract, PreprocessContractV2, load_contract


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
    """Strict contract preprocessing for canvas-v1 payloads."""

    def __init__(self, contract: Optional[PreprocessContractV2] = None):
        self.contract: PreprocessContract = contract or load_contract()

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

    def _parse_payload(self, image_data: Any) -> Tuple[Optional[List[Dict[str, Any]]], Optional[np.ndarray], Dict[str, Any]]:
        meta: Dict[str, Any] = {}
        if isinstance(image_data, dict):
            strokes = image_data.get("strokes")
            raster = image_data.get("raster") or image_data.get("image")
            canvas = image_data.get("canvas", {})
            if not isinstance(canvas, dict):
                canvas = {}
            meta["canvas_width"] = int(canvas.get("width", self.contract.get_canvas_spec()[0]))
            meta["canvas_height"] = int(canvas.get("height", self.contract.get_canvas_spec()[1]))
            raster_np = self._to_numpy(raster) if raster is not None else None
            return strokes if isinstance(strokes, list) else None, raster_np, meta

        # Legacy direct raster path
        return None, self._to_numpy(image_data), meta

    def _rasterize_strokes(self, strokes: List[Dict[str, Any]], canvas_w: int, canvas_h: int) -> np.ndarray:
        img = Image.new("L", (canvas_w, canvas_h), 0)
        draw = ImageDraw.Draw(img)
        lw = max(1, int(round(self.contract.line_width)))

        for stroke in strokes:
            points = stroke.get("points", []) if isinstance(stroke, dict) else []
            xy: List[Tuple[float, float]] = []
            for p in points:
                if isinstance(p, dict):
                    x = p.get("x")
                    y = p.get("y")
                elif isinstance(p, (list, tuple)) and len(p) >= 2:
                    x, y = p[0], p[1]
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
                draw.line(xy, fill=255, width=lw, joint="curve")
            elif len(xy) == 1:
                x, y = xy[0]
                r = lw / 2.0
                draw.ellipse((x - r, y - r, x + r, y + r), fill=255)

        return np.asarray(img, dtype=np.float32) / 255.0

    def _denoise(self, img: np.ndarray) -> np.ndarray:
        enabled, thr, min_neighbors = self.contract.get_denoise()
        if not enabled:
            return img

        fg = (img > thr).astype(np.uint8)
        p = np.pad(fg, 1, mode="constant")
        neighbors = (
            p[:-2, :-2] + p[:-2, 1:-1] + p[:-2, 2:] +
            p[1:-1, :-2] + p[1:-1, 2:] +
            p[2:, :-2] + p[2:, 1:-1] + p[2:, 2:]
        )
        keep = (fg == 1) & (neighbors >= min_neighbors)
        out = img.copy()
        out[~keep] = 0.0
        return out

    def _crop_scale_center(self, img: np.ndarray) -> np.ndarray:
        h, w = self.contract.target_size
        thr = self.contract.bbox_threshold
        fg = np.argwhere(img > thr)
        canvas = np.zeros((h, w), dtype=np.float32)
        if fg.size == 0:
            return canvas

        y0, x0 = fg.min(axis=0)
        y1, x1 = fg.max(axis=0)
        crop = img[y0 : y1 + 1, x0 : x1 + 1]

        target_box = max(8, min(min(h, w), self.contract.target_glyph_box))
        ch, cw = crop.shape
        scale = min(target_box / max(ch, 1), target_box / max(cw, 1))
        nh = max(1, int(round(ch * scale)))
        nw = max(1, int(round(cw * scale)))

        pil = Image.fromarray((np.clip(crop, 0, 1) * 255).astype(np.uint8))
        resized = np.asarray(pil.resize((nw, nh), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0

        ys = (h - nh) // 2
        xs = (w - nw) // 2
        canvas[ys : ys + nh, xs : xs + nw] = resized

        if self.contract.center_of_mass_recenter and np.sum(canvas) > 1e-6:
            yy, xx = np.indices(canvas.shape)
            mass = np.sum(canvas)
            cy = float(np.sum(yy * canvas) / mass)
            cx = float(np.sum(xx * canvas) / mass)
            ty = int(round((h - 1) / 2.0 - cy))
            tx = int(round((w - 1) / 2.0 - cx))
            canvas = self._translate_no_wrap(canvas, ty, tx)

        return canvas

    def _translate_no_wrap(self, img: np.ndarray, ty: int, tx: int) -> np.ndarray:
        out = np.zeros_like(img)
        h, w = img.shape

        src_y0 = max(0, -ty)
        src_y1 = min(h, h - ty) if ty >= 0 else h
        dst_y0 = max(0, ty)
        dst_y1 = min(h, h + ty) if ty < 0 else h

        src_x0 = max(0, -tx)
        src_x1 = min(w, w - tx) if tx >= 0 else w
        dst_x0 = max(0, tx)
        dst_x1 = min(w, w + tx) if tx < 0 else w

        if src_y1 > src_y0 and src_x1 > src_x0 and dst_y1 > dst_y0 and dst_x1 > dst_x0:
            out[dst_y0:dst_y1, dst_x0:dst_x1] = img[src_y0:src_y1, src_x0:src_x1]
        return out

    def _normalize(self, img_flat: np.ndarray) -> np.ndarray:
        mean, std, clip_min, clip_max, eps = self.contract.get_stats()
        if mean is None or std is None or mean.shape[0] != img_flat.shape[0] or std.shape[0] != img_flat.shape[0]:
            mean = np.full_like(img_flat, 0.5, dtype=np.float32)
            std = np.full_like(img_flat, 0.25, dtype=np.float32)
        z = (img_flat - mean) / (std + eps)
        z = np.clip(z, clip_min, clip_max)
        return np.tanh(z).astype(np.float32)

    def _as_data_url(self, img: np.ndarray) -> str:
        pil = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8), mode="L")
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

    def preprocess(
        self,
        image_data: Any,
        return_metrics: bool = False,
        return_debug: bool = False,
        skip_transform: bool = False,
        already_normalized: bool = False,
        strict_mode: Optional[bool] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        strict = self.contract.strict_input if strict_mode is None else strict_mode
        strokes, raster_np, meta = self._parse_payload(image_data)
        debug: Dict[str, Any] = {}

        if already_normalized and isinstance(image_data, np.ndarray) and image_data.shape == (28, 28):
            arr = image_data.astype(np.float32)
            if arr.min() >= -1.1 and arr.max() <= 1.1:
                out = arr.flatten().reshape(1, -1)
                metrics = self._quality_metrics(np.clip((arr + 1.0) / 2.0, 0.0, 1.0), arr)
                if return_debug:
                    debug["contract_version"] = self.contract.version
                    debug["contract_checksum"] = self.contract.checksum
                return out, (metrics if return_metrics else None), (debug if return_debug else None)

        cw_default, ch_default, require_strokes = self.contract.get_canvas_spec()
        canvas_w = int(meta.get("canvas_width", cw_default))
        canvas_h = int(meta.get("canvas_height", ch_default))

        if strict and require_strokes and not strokes:
            return None, None, {"error": "strict_contract_v3_requires_strokes"} if return_debug else None

        if strokes:
            raw = self._rasterize_strokes(strokes, canvas_w, canvas_h)
            debug["source"] = "strokes"
        elif raster_np is not None and not strict:
            raw = raster_np
            debug["source"] = "raster"
        else:
            return None, None, {"error": "invalid_input_payload"} if return_debug else None

        if raw is None:
            return None, None, {"error": "failed_to_decode_input"} if return_debug else None

        if return_debug:
            debug["raw_shape"] = list(raw.shape)
            debug["raw_min"] = float(raw.min())
            debug["raw_max"] = float(raw.max())

        stage = raw
        if not skip_transform:
            t = self.contract.transform_id if self.contract.transform_id in TRANSFORMS else "identity"
            stage = apply_transform(stage, t)
            debug["transform_id"] = t

        # Expect white foreground on black background.
        if float(np.mean(stage)) > self.contract.invert_threshold:
            stage = 1.0 - stage
            debug["inverted"] = True

        denoised = self._denoise(stage)
        centered = self._crop_scale_center(denoised)

        flat = centered.reshape(-1).astype(np.float32)
        norm = self._normalize(flat)
        out = norm.reshape(1, -1)

        metrics = self._quality_metrics(centered, norm.reshape(28, 28)) if return_metrics else None
        if return_debug:
            debug.update(
                {
                    "contract_version": self.contract.version,
                    "contract_checksum": self.contract.checksum,
                    "final_min": float(out.min()),
                    "final_max": float(out.max()),
                    "stage_images": {
                        "raw": self._as_data_url(np.clip(raw, 0.0, 1.0)),
                        "denoised": self._as_data_url(np.clip(denoised, 0.0, 1.0)),
                        "centered": self._as_data_url(np.clip(centered, 0.0, 1.0)),
                    },
                    # Backward-compatible debug aliases for existing frontend panel.
                    "original": self._as_data_url(np.clip(raw, 0.0, 1.0)),
                    "after_resize": self._as_data_url(np.clip(centered, 0.0, 1.0)),
                    "final": self._as_data_url(np.clip((norm.reshape(28, 28) + 1.0) / 2.0, 0.0, 1.0)),
                    "stats": {
                        "original_min": float(raw.min()),
                        "original_max": float(raw.max()),
                        "original_mean": float(raw.mean()),
                        "final_min": float(out.min()),
                        "final_max": float(out.max()),
                        "final_mean": float(out.mean()),
                        "final_std": float(out.std()),
                    },
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
        strict_mode=(False if is_test_image else None),
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
        strict_mode=(False if is_test_image else None),
    )
    if return_debug:
        return out, metrics, debug
    return out, metrics
