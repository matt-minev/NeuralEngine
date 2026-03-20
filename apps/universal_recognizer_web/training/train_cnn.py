"""Train CNN model for Universal Recognizer with production-grade logging and
accuracy-boosting features: cosine LR annealing, label smoothing, and
on-the-fly data augmentation."""

from __future__ import annotations

import argparse
import datetime
import math
import os
import pickle
import random
import sys
import time
import traceback
from typing import Tuple

import numpy as np

_THIS = os.path.abspath(__file__)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_THIS))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from apps.universal_recognizer_web.training.data_loader import (
    create_data_splits, index_to_character,
)
from apps.universal_recognizer_web.training.cnn_model import (
    CNNConfig, UniversalCNN, AdamCNN, accuracy_from_probs,
)
from apps.universal_recognizer_web.training.data_augmentation import DataAugmentation
from apps.universal_recognizer_web.core.preprocess_contract import load_contract
from neural_backend import to_cpu


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
    # Use numpy for permutation (cupy.random may require curand)
    idx_np = np.random.permutation(n)
    idx = xp.asarray(idx_np) if xp is not np else idx_np
    for i in range(0, n, batch_size):
        b = idx[i:i + batch_size]
        yield x[b], y[b]


def evaluate(model: UniversalCNN, x, y, batch_size: int = 1024) -> Tuple[float, float]:
    """Evaluate in batches to avoid GPU OOM on large datasets."""
    n = x.shape[0]
    all_probs = []
    for i in range(0, n, batch_size):
        xb = model.xp.asarray(x[i:i + batch_size], dtype=model.dtype)
        pb = model.predict(xb, on_device=False)  # returns numpy
        all_probs.append(pb)
    probs = np.concatenate(all_probs, axis=0)
    y_np = np.asarray(y) if not isinstance(y, np.ndarray) else y
    loss = -np.sum(y_np * np.log(np.clip(probs, 1e-12, 1.0))) / max(1, n)
    pred = np.argmax(probs, axis=1)
    true = np.argmax(y_np, axis=1)
    acc = float(np.mean(pred == true) * 100.0)
    return float(loss), acc


def fmt_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m}m {s:02d}s"


def check_for_nan(tensor, name: str, xp) -> bool:
    if bool(xp.any(xp.isnan(tensor))):
        print(f"\n  *** NaN DETECTED in {name}! Training is diverging. ***")
        return True
    return False


def log_label_distribution(y_onehot, label: str):
    class_counts = np.asarray(y_onehot.sum(axis=0)).flatten()
    total = int(class_counts.sum())
    nonzero = int((class_counts > 0).sum())
    top5_idx = np.argsort(class_counts)[-5:][::-1]
    bot5_idx = np.argsort(class_counts)[:5]
    print(f"  {label} distribution: {nonzero}/62 classes present, {total:,} total samples")
    top5_str = ", ".join(f"'{index_to_character(i)}'={int(class_counts[i])}" for i in top5_idx)
    bot5_str = ", ".join(f"'{index_to_character(i)}'={int(class_counts[i])}" for i in bot5_idx)
    print(f"    Top-5: {top5_str}")
    print(f"    Bot-5: {bot5_str}")


def save_checkpoint(model, opt, epoch, best_val_acc, history, checkpoint_path):
    checkpoint = {
        'epoch': epoch,
        'best_val_acc': best_val_acc,
        'history': history,
        'weights': [np.array(to_cpu(p), copy=True) for p in [
            model.w1, model.b1, model.w2, model.b2, model.w3, model.b3, model.w4, model.b4
        ]],
        'optimizer_t': opt.t,
    }
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(checkpoint, f, protocol=pickle.HIGHEST_PROTOCOL)


# ---------------------------------------------------------------------------
# Learning rate schedule
# ---------------------------------------------------------------------------

def cosine_lr(epoch: int, total_epochs: int, warmup_epochs: int,
              lr_max: float, lr_min: float) -> float:
    """Cosine annealing with linear warmup."""
    if epoch <= warmup_epochs:
        # Linear warmup from lr_min to lr_max
        return lr_min + (lr_max - lr_min) * (epoch / max(1, warmup_epochs))
    # Cosine decay from lr_max to lr_min
    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Label smoothing
# ---------------------------------------------------------------------------

def smooth_labels(y_onehot, smoothing: float, num_classes: int, xp):
    """Apply label smoothing: shift probability mass from the true class to all classes."""
    if smoothing <= 0:
        return y_onehot
    return y_onehot * (1.0 - smoothing) + smoothing / num_classes


# ---------------------------------------------------------------------------
# Data augmentation (safe for character recognition — NO mirroring)
# ---------------------------------------------------------------------------

def create_safe_augmenter() -> DataAugmentation:
    """Create augmentation that is safe for character recognition.
    NO horizontal mirroring (would turn b→d, p→q, etc.)."""
    return DataAugmentation(
        mirror_prob=0.0,           # NEVER mirror for character recognition
        rotation_range=8.0,        # Conservative rotation (handwriting varies ~±8°)
        scale_range=(0.93, 1.07),  # Slight scale variation
        translation_range=2,       # ±2 pixel shift
        noise_std=0.03,            # Light noise
        contrast_range=(0.9, 1.1), # Slight contrast variation
        brightness_range=(0.95, 1.05),
        elastic_alpha=0.8,         # Subtle elastic deformation (mimics handwriting variation)
        elastic_sigma=5.0,
    )


def augment_batch_gpu(x_batch, augmenter: DataAugmentation, xp):
    """Augment a batch: move to CPU, augment, move back to device."""
    x_np = to_cpu(x_batch).astype(np.float32)
    x_aug, _ = augmenter.augment_batch(x_np)
    return xp.asarray(x_aug, dtype=x_batch.dtype) if xp is not np else x_aug


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Train UniversalCNN on EMNIST ByClass")
    ap.add_argument('--device', type=str, default=os.getenv('NEURAL_ENGINE_DEVICE', 'auto'), choices=['auto', 'cpu', 'gpu'])
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--lr-min', type=float, default=0.00005,
                    help='Minimum LR for cosine annealing')
    ap.add_argument('--warmup-epochs', type=int, default=3,
                    help='Number of linear warmup epochs')
    ap.add_argument('--label-smoothing', type=float, default=0.1,
                    help='Label smoothing factor (0=disabled)')
    ap.add_argument('--no-augment', action='store_true',
                    help='Disable data augmentation')
    ap.add_argument('--train-samples', type=int, default=0,
                    help='Max training samples (0=use all)')
    ap.add_argument('--val-samples', type=int, default=0,
                    help='Max validation samples (0=use all)')
    ap.add_argument('--test-samples', type=int, default=0,
                    help='Max test samples (0=use all)')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--data-dir', type=str, default=None)
    ap.add_argument('--save-path', type=str, default=None)
    ap.add_argument('--checkpoint-every', type=int, default=5,
                    help='Save checkpoint every N epochs (0 to disable)')
    args = ap.parse_args()

    # ---- Banner ----
    print("=" * 70)
    print("  NEURALENGINE CNN TRAINING")
    print(f"  Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    set_seed(args.seed)

    # ---- Contract check ----
    contract = load_contract()
    print(f"\n[1/5] CONTRACT CHECK")
    print(f"  Version    : {contract.version}")
    print(f"  Transform  : {contract.transform_id}")
    print(f"  Checksum   : {contract.checksum[:16]}...")
    if contract.transform_id != "flip_h_rot90":
        print(f"  *** WARNING: transform_id is '{contract.transform_id}', expected 'flip_h_rot90' ***")

    # ---- Data loading ----
    print(f"\n[2/5] LOADING DATA")
    load_start = time.time()
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = create_data_splits(data_dir=args.data_dir)
    if args.train_samples > 0:
        x_train, y_train = sample_subset(x_train, y_train, args.train_samples)
    if args.val_samples > 0:
        x_val, y_val = sample_subset(x_val, y_val, args.val_samples)
    if args.test_samples > 0:
        x_test, y_test = sample_subset(x_test, y_test, args.test_samples)
    load_time = time.time() - load_start
    print(f"  Data loaded in {fmt_time(load_time)}")

    # ---- Sanity checks ----
    print(f"\n[3/5] SANITY CHECKS")
    print(f"  Train: {x_train.shape} range [{x_train.min():.3f}, {x_train.max():.3f}]")
    print(f"  Val  : {x_val.shape} range [{x_val.min():.3f}, {x_val.max():.3f}]")
    print(f"  Test : {x_test.shape} range [{x_test.min():.3f}, {x_test.max():.3f}]")
    log_label_distribution(y_train, "Train")
    log_label_distribution(y_val, "Val")
    if np.any(np.isnan(x_train)):
        print("  *** FATAL: NaN in training data! Aborting. ***")
        sys.exit(1)
    print("  NaN check: PASSED")

    # ---- Model init ----
    print(f"\n[4/5] MODEL INITIALIZATION")
    model = UniversalCNN(CNNConfig(), device=args.device, dtype='float32')
    opt = AdamCNN(lr=args.lr)

    print(f"  Architecture: Conv{model.config.conv1_channels} -> Pool -> Conv{model.config.conv2_channels} -> Pool -> FC{model.config.fc_hidden} -> FC{model.config.num_classes}")
    print(f"  Parameters : {model.count_parameters():,}")
    print(f"  Device     : {model.device} ({model.backend_name}), GPU={model.using_gpu}")
    print(f"  Dtype      : {model.dtype}")
    print(f"  Dropout    : {model.config.dropout}")

    if not model.using_gpu and args.device == 'gpu':
        print("  *** WARNING: GPU requested but unavailable, training on CPU ***")

    # Move ONLY training data to GPU; val/test stay on CPU to save VRAM
    print("  Moving training data to device (val/test stay on CPU to save VRAM)...")
    x_train = model.xp.asarray(x_train, dtype=model.dtype)
    y_train = model.xp.asarray(y_train, dtype=model.dtype)
    # val and test remain as numpy arrays — evaluate() batches them to GPU as needed

    if model.using_gpu:
        try:
            import cupy as cp
            mem = cp.cuda.Device(0).mem_info
            free_mb = mem[0] / (1024 ** 2)
            total_mb = mem[1] / (1024 ** 2)
            print(f"  GPU Memory : {free_mb:.0f} MB free / {total_mb:.0f} MB total")
        except Exception:
            pass

    # Forward pass sanity check
    print("  Running forward pass sanity check...")
    try:
        sanity_x = model.xp.asarray(x_val[:8], dtype=model.dtype)
        test_pred = model.predict(sanity_x, on_device=True)
        pred_sum = float(model.xp.sum(test_pred[0]))
        print(f"  Forward pass OK: output shape={test_pred.shape}, prob_sum={pred_sum:.4f}")
        if abs(pred_sum - 1.0) > 0.01:
            print("  *** WARNING: Probability sum != 1.0! Softmax may be broken. ***")
    except Exception as e:
        print(f"  *** FATAL: Forward pass failed: {e} ***")
        traceback.print_exc()
        sys.exit(1)

    # ---- Label smoothing ----
    num_classes = 62
    if args.label_smoothing > 0:
        y_train = smooth_labels(y_train, args.label_smoothing, num_classes, model.xp)
        print(f"  Label smoothing: {args.label_smoothing} applied to training labels")

    # ---- Data augmentation ----
    augmenter = None
    if not args.no_augment:
        augmenter = create_safe_augmenter()
        print(f"  Data augmentation: ENABLED (rot=±8°, scale=0.93-1.07, shift=±2px, elastic, noise)")
    else:
        print(f"  Data augmentation: DISABLED")

    # ---- Checkpoint setup ----
    checkpoint_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'checkpoints')
    checkpoint_path = os.path.join(checkpoint_dir, 'training_checkpoint.pkl')

    # ---- Training loop ----
    n_batches = (x_train.shape[0] + args.batch_size - 1) // args.batch_size

    print(f"\n[5/5] TRAINING")
    print(f"  Epochs       : {args.epochs}")
    print(f"  Batch size   : {args.batch_size}")
    print(f"  Batches/ep   : {n_batches}")
    print(f"  LR schedule  : cosine annealing ({args.lr} -> {args.lr_min}), warmup={args.warmup_epochs}ep")
    print(f"  Smoothing    : {args.label_smoothing}")
    print(f"  Augmentation : {'ON' if augmenter else 'OFF'}")
    print("-" * 70)
    print(f"  {'Epoch':>7} | {'LR':>9} | {'Train Loss':>10} | {'Val Loss':>8} | {'Val Acc':>8} | {'Time':>8} | {'ETA':>10} | {'Note'}")
    print("-" * 70)

    best_val_acc = -1.0
    best_state = None
    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'lr': []}
    started = time.time()
    stale_epochs = 0

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # Update learning rate (cosine annealing with warmup)
        current_lr = cosine_lr(epoch, args.epochs, args.warmup_epochs, args.lr, args.lr_min)
        opt.lr = current_lr

        model.training = True
        losses = []

        for xb, yb in iterate_minibatches(x_train, y_train, args.batch_size, model.xp):
            # Apply data augmentation on-the-fly
            if augmenter is not None:
                xb = augment_batch_gpu(xb, augmenter, model.xp)

            loss, _ = model.train_step(xb, yb, opt)
            losses.append(loss)

            # NaN check periodically
            if len(losses) % 100 == 0 and (np.isnan(loss) or np.isinf(loss)):
                print(f"\n  *** FATAL: Loss is {loss} at epoch {epoch}, batch {len(losses)}! ***")
                print(f"  *** Training diverged. Try reducing --lr. ***")
                sys.exit(1)

        model.training = False
        train_loss = float(np.mean(losses)) if losses else 0.0
        val_loss, val_acc = evaluate(model, x_val, y_val)
        epoch_time = time.time() - epoch_start

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)

        # Track improvement
        note = ""
        if val_acc > best_val_acc:
            improvement = val_acc - best_val_acc if best_val_acc > 0 else 0
            best_val_acc = val_acc
            best_state = [np.array(to_cpu(p), copy=True) for p in [
                model.w1, model.b1, model.w2, model.b2, model.w3, model.b3, model.w4, model.b4
            ]]
            note = f"★ BEST" + (f" (+{improvement:.2f}%)" if improvement > 0 else "")
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= 10:
                note = f"⚠ no improvement for {stale_epochs} epochs"
            elif stale_epochs >= 5:
                note = f"stale x{stale_epochs}"

        # ETA calculation
        elapsed = time.time() - started
        avg_epoch_time = elapsed / epoch
        remaining = avg_epoch_time * (args.epochs - epoch)

        print(f"  {epoch:3d}/{args.epochs:3d}  | {current_lr:.7f} | {train_loss:10.4f} | {val_loss:8.4f} | {val_acc:7.2f}% | {fmt_time(epoch_time):>8} | {fmt_time(remaining):>10} | {note}")

        # Periodic checkpoint
        if args.checkpoint_every > 0 and epoch % args.checkpoint_every == 0:
            save_checkpoint(model, opt, epoch, best_val_acc, history, checkpoint_path)
            print(f"         ↳ Checkpoint saved (epoch {epoch})")

        # Early divergence detection
        if epoch >= 3 and train_loss > history['train_loss'][0] * 5:
            print(f"\n  *** WARNING: train_loss has grown 5x since epoch 1. Training may be diverging. ***")

    print("-" * 70)

    # ---- Restore best and evaluate ----
    if best_state is not None:
        model.w1, model.b1, model.w2, model.b2, model.w3, model.b3, model.w4, model.b4 = [
            model.xp.asarray(s, dtype=model.dtype) for s in best_state
        ]

    test_loss, test_acc = evaluate(model, x_test, y_test)
    elapsed = (time.time() - started) / 60.0

    # ---- Save model ----
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
        'classes': num_classes,
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

    # ---- Final summary ----
    print(f"\n{'=' * 70}")
    print(f"  TRAINING COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Best val accuracy : {best_val_acc:.2f}%")
    print(f"  Test accuracy     : {test_acc:.2f}%")
    print(f"  Test loss         : {test_loss:.4f}")
    print(f"  Total time        : {fmt_time(elapsed * 60)}")
    print(f"  Model saved to    : {out}")
    print(f"  Contract version  : {contract.version} (transform={contract.transform_id})")
    print(f"  Finished          : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
