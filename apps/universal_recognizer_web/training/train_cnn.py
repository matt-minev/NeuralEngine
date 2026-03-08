"""Train CNN model for Universal Recognizer with reasonable-time presets."""

from __future__ import annotations

import argparse
import os
import pickle
import random
import sys
import time
from typing import Tuple

import numpy as np

_THIS = os.path.abspath(__file__)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_THIS))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from apps.universal_recognizer_web.training.data_loader import create_data_splits
from apps.universal_recognizer_web.training.cnn_model import CNNConfig, UniversalCNN, AdamCNN, accuracy_from_probs
from apps.universal_recognizer_web.core.preprocess_contract import load_contract


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def sample_subset(x, y, limit: int):
    if limit <= 0 or limit >= x.shape[0]:
        return x, y
    idx = np.random.choice(x.shape[0], size=limit, replace=False)
    return x[idx], y[idx]


def iterate_minibatches(x, y, batch_size: int, xp):
    n = x.shape[0]
    idx = xp.random.permutation(n)
    for i in range(0, n, batch_size):
        b = idx[i:i + batch_size]
        yield x[b], y[b]


def evaluate(model: UniversalCNN, x, y) -> Tuple[float, float]:
    probs = model.predict(x, on_device=True)
    loss = -model.xp.sum(y * model.xp.log(model.xp.clip(probs, 1e-12, 1.0))) / max(1, y.shape[0])
    acc = accuracy_from_probs(probs, y)
    return float(loss), acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', type=str, default=os.getenv('NEURAL_ENGINE_DEVICE', 'auto'), choices=['auto', 'cpu', 'gpu'])
    ap.add_argument('--epochs', type=int, default=24)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--train-samples', type=int, default=180000)
    ap.add_argument('--val-samples', type=int, default=30000)
    ap.add_argument('--test-samples', type=int, default=20000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--data-dir', type=str, default=None)
    ap.add_argument('--save-path', type=str, default=None)
    args = ap.parse_args()

    set_seed(args.seed)

    (x_train, y_train), (x_val, y_val), (x_test, y_test) = create_data_splits(data_dir=args.data_dir)
    x_train, y_train = sample_subset(x_train, y_train, args.train_samples)
    x_val, y_val = sample_subset(x_val, y_val, args.val_samples)
    x_test, y_test = sample_subset(x_test, y_test, args.test_samples)

    model = UniversalCNN(CNNConfig(), device=args.device, dtype='float32')
    opt = AdamCNN(lr=args.lr)

    # Move all tensors once to backend for throughput.
    x_train = model.xp.asarray(x_train, dtype=model.dtype)
    y_train = model.xp.asarray(y_train, dtype=model.dtype)
    x_val = model.xp.asarray(x_val, dtype=model.dtype)
    y_val = model.xp.asarray(y_val, dtype=model.dtype)
    x_test = model.xp.asarray(x_test, dtype=model.dtype)
    y_test = model.xp.asarray(y_test, dtype=model.dtype)

    print('CNN Training Start')
    print(f'  Device: {model.device} ({model.backend_name}), using_gpu={model.using_gpu}')
    print(f'  Train samples: {x_train.shape[0]:,}, Val: {x_val.shape[0]:,}, Test: {x_test.shape[0]:,}')
    print(f'  Epochs: {args.epochs}, Batch size: {args.batch_size}, LR: {args.lr}')

    best_val_acc = -1.0
    best_state = None
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        model.training = True
        losses = []
        for xb, yb in iterate_minibatches(x_train, y_train, args.batch_size, model.xp):
            loss, _ = model.train_step(xb, yb, opt)
            losses.append(loss)

        model.training = False
        train_loss = float(np.mean(losses)) if losses else 0.0
        val_loss, val_acc = evaluate(model, x_val, y_val)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = [np.array(p, copy=True) for p in [
                model.w1, model.b1, model.w2, model.b2, model.w3, model.b3, model.w4, model.b4
            ]]

        print(f'Epoch {epoch:03d}/{args.epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.2f}%')

    if best_state is not None:
        model.w1, model.b1, model.w2, model.b2, model.w3, model.b3, model.w4, model.b4 = [model.xp.asarray(s, dtype=model.dtype) for s in best_state]

    test_loss, test_acc = evaluate(model, x_test, y_test)
    elapsed = (time.time() - started) / 60.0

    contract = load_contract()
    payload = {
        'model': model,
        'accuracy': test_acc,
        'avg_confidence': float(np.mean(np.max(np.asarray(model.predict(x_test[:4096], on_device=False)), axis=1)) * 100.0),
        'character_type_accuracies': {},
        'history': history,
        'training_time': elapsed * 60.0,
        'architecture': ['conv32', 'conv64', 'fc256', 'fc62'],
        'dataset': 'emnist_byclass',
        'classes': 62,
        'config': vars(args),
        'model_version': 'universal_v3_cnn',
        'contract_version': contract.version,
        'contract_checksum': contract.checksum,
        'calibration': {'temperature': 1.0},
        'engine_backend': 'cnn_numpy_cupy',
        'device': model.device,
        'backend_name': model.backend_name,
    }

    out = args.save_path
    if not out:
        out = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'universal_character_model.pkl')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'wb') as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    print('Training complete')
    print(f'  Best val acc: {best_val_acc:.2f}%')
    print(f'  Test acc: {test_acc:.2f}%')
    print(f'  Test loss: {test_loss:.4f}')
    print(f'  Saved: {out}')
    print(f'  Elapsed: {elapsed:.1f} min')


if __name__ == '__main__':
    main()
