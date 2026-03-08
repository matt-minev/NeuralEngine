"""Calibrate preprocess contract v2 orientation using validation accuracy sweep."""

from __future__ import annotations

import os
import sys
import numpy as np

_THIS = os.path.abspath(__file__)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_THIS))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from apps.universal_recognizer_web.training.data_loader import (
    read_idx_images,
    read_idx_labels,
    preprocess_data,
)
from apps.universal_recognizer_web.core.preprocess_contract import load_contract, with_stats, save_contract
from apps.universal_recognizer_web.core.canonical_preprocessor import TRANSFORMS, apply_transform


def _default_data_dir():
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')


def evaluate_transform(model, images, labels_onehot, transform_id, train_mean, train_std, max_samples=5000):
    n = min(max_samples, images.shape[0])
    idx = np.random.RandomState(42).choice(images.shape[0], size=n, replace=False)
    batch = images[idx]
    y = np.argmax(labels_onehot[idx], axis=1)
    fixed = np.stack([apply_transform(img, transform_id) for img in batch], axis=0)
    flat = fixed.reshape(fixed.shape[0], -1).astype(np.float32)
    x = preprocess_data(flat, normalize=True, mean=train_mean, std=train_std)
    pred = np.argmax(model.forward(x), axis=1)
    return float(np.mean(pred == y) * 100.0)


def main():
    from apps.universal_recognizer_web.core.model_manager import get_model_manager
    mm = get_model_manager()
    model = mm.get_model()

    data_dir = _default_data_dir()
    train_images = read_idx_images(os.path.join(data_dir, 'emnist-byclass-train-images-idx3-ubyte.gz'))
    train_labels = read_idx_labels(os.path.join(data_dir, 'emnist-byclass-train-labels-idx1-ubyte.gz'))
    val_images = read_idx_images(os.path.join(data_dir, 'emnist-byclass-test-images-idx3-ubyte.gz'))
    val_labels = read_idx_labels(os.path.join(data_dir, 'emnist-byclass-test-labels-idx1-ubyte.gz'))

    train_flat = train_images.reshape(train_images.shape[0], -1).astype(np.float32) / 255.0
    mean = np.mean(train_flat, axis=0).astype(np.float32)
    std = np.std(train_flat, axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1e-6, std)

    y_val = np.zeros((val_labels.shape[0], 62), dtype=np.float32)
    y_val[np.arange(val_labels.shape[0]), val_labels] = 1.0

    scores = {}
    for t in sorted(TRANSFORMS):
        score = evaluate_transform(model, val_images, y_val, t, mean, std)
        scores[t] = score
        print(f"{t}: {score:.2f}%")

    best = max(scores.items(), key=lambda x: x[1])[0]
    print(f"Best transform: {best}")

    contract = load_contract()
    updated = with_stats(contract, mean, std)
    updated.data['transform_id'] = best
    updated.data['calibration_scores'] = scores
    save_contract(updated)
    print(f"Saved contract at {updated.source_path} checksum={updated.checksum}")


if __name__ == '__main__':
    main()
