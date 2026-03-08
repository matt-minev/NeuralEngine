"""Accuracy gate runner for Universal v3.

Usage:
  python -m apps.universal_recognizer_web.training.accuracy_gate --offline-min 94 --web-min 90
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Tuple

import numpy as np

_THIS = os.path.abspath(__file__)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_THIS))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from apps.universal_recognizer_web.core.model_manager import get_model_manager
from apps.universal_recognizer_web.core.predictor import predict_character
from apps.universal_recognizer_web.training.data_loader import load_emnist_data


CONFUSION_PAIRS = [
    ("O", "0"),
    ("I", "1"),
    ("l", "1"),
    ("s", "S"),
]


def index_to_character(index: int) -> str:
    if 0 <= index <= 9:
        return str(index)
    if 10 <= index <= 35:
        return chr(ord("A") + index - 10)
    if 36 <= index <= 61:
        return chr(ord("a") + index - 36)
    return "?"


def to_stroke_payload(img28: np.ndarray) -> Dict:
    up = np.kron(np.clip(img28, 0.0, 1.0), np.ones((10, 10), dtype=np.float32))
    ys, xs = np.where(up > 0.15)
    points = [{"x": float(x), "y": float(y), "t": int(i)} for i, (x, y) in enumerate(zip(xs, ys))]
    if not points:
        points = [{"x": 140.0, "y": 140.0, "t": 0}]
    return {
        "canvas": {"width": 280, "height": 280},
        "strokes": [{"points": points}],
    }


def _load_test_split():
    (_, _), (x_test, y_test) = load_emnist_data()
    y_true = np.argmax(y_test, axis=1)
    return x_test, y_true


def offline_accuracy(x_test: np.ndarray, y_true: np.ndarray, max_samples: int = 10000) -> Tuple[float, np.ndarray, np.ndarray]:
    mm = get_model_manager()
    model = mm.get_model()

    if max_samples > 0:
        x_test = x_test[:max_samples]
        y_true = y_true[:max_samples]

    preds = model.forward(x_test)
    y_pred = np.argmax(np.asarray(preds), axis=1)
    acc = float(np.mean(y_pred == y_true) * 100.0)
    return acc, y_true, y_pred


def web_synth_accuracy(x_test: np.ndarray, y_true: np.ndarray, max_samples: int = 2000) -> float:
    if max_samples > 0:
        x_test = x_test[:max_samples]
        y_true = y_true[:max_samples]

    correct = 0
    total = len(y_true)
    for i in range(total):
        arr = x_test[i].reshape(28, 28)
        img = np.clip((arr + 1.0) / 2.0, 0.0, 1.0)
        payload = to_stroke_payload(img)
        res = predict_character(payload, return_quality_metrics=False, is_test_image=False)
        if res is None:
            continue
        if int(res["predicted_index"]) == int(y_true[i]):
            correct += 1
    return float(correct / max(1, total) * 100.0)


def confusion_report(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    c_true = np.array([index_to_character(int(i)) for i in y_true])
    c_pred = np.array([index_to_character(int(i)) for i in y_pred])
    out: Dict[str, float] = {}
    for a, b in CONFUSION_PAIRS:
        mask = (c_true == a) | (c_true == b)
        if not np.any(mask):
            out[f"{a}/{b}"] = 0.0
            continue
        pair_err = np.mean(c_pred[mask] != c_true[mask]) * 100.0
        out[f"{a}/{b}"] = float(pair_err)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline-min", type=float, default=94.0)
    ap.add_argument("--web-min", type=float, default=90.0)
    ap.add_argument("--offline-samples", type=int, default=10000)
    ap.add_argument("--web-samples", type=int, default=2000)
    args = ap.parse_args()

    x_test, y_true_all = _load_test_split()
    off_acc, y_true, y_pred = offline_accuracy(x_test, y_true_all, args.offline_samples)
    web_acc = web_synth_accuracy(x_test, y_true_all, args.web_samples)
    conf = confusion_report(y_true, y_pred)

    print(f"offline_top1={off_acc:.2f}% (gate>={args.offline_min:.2f}%)")
    print(f"web_synth_top1={web_acc:.2f}% (gate>={args.web_min:.2f}%)")
    print("confusion_pair_error_rates=%")
    for k, v in conf.items():
        print(f"  {k}: {v:.2f}")

    ok = off_acc >= args.offline_min and web_acc >= args.web_min
    if not ok:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
