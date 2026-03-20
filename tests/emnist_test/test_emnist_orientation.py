"""
EMNIST Orientation & Data Processing Verification.

Loads orientation-sensitive characters from EMNIST ByClass, applies ALL
candidate transforms side-by-side, generates a visual PNG grid so the
correct transform can be identified, then runs automated sanity checks.
"""

import os
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ------------------------------------------------------------------
# Path setup
# ------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, ROOT)

from apps.universal_recognizer_web.training.data_loader import (
    read_idx_images,
    read_idx_labels,
    index_to_character,
)
from apps.universal_recognizer_web.core.canonical_preprocessor import apply_transform

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
DATA_DIR = os.path.join(ROOT, "apps", "universal_recognizer_web", "data")
OUTPUT_PNG = os.path.join(SCRIPT_DIR, "emnist_orientation_grid.png")

# Characters whose orientation is easy to verify visually
ORIENTATION_CHARS = {
    22: "M",
    32: "W",
    10: "A",
    31: "V",
    6:  "6",
    9:  "9",
    39: "d",
    51: "p",
    37: "b",
    52: "q",
}

# All available transforms to compare
ALL_TRANSFORMS = [
    "raw",               # No transform (raw from EMNIST file)
    "identity",          # Same as raw (no-op)
    "legacy_emnist_fix", # np.rot90(np.flip(img, axis=1), 3)  — 180° rotation
    "rot90",             # np.rot90(img, 1)  — 90° CCW
    "rot180",            # np.rot90(img, 2)  — 180°
    "rot270",            # np.rot90(img, 3)  — 270° CCW (= 90° CW)
    "flip_h",            # np.flip(img, axis=1) — horizontal flip
    "flip_v",            # np.flip(img, axis=0) — vertical flip
    "flip_h_rot90",      # np.rot90(np.flip(img, axis=1), 1) — transpose!
    "flip_v_rot90",      # np.rot90(np.flip(img, axis=0), 1)
]


def load_samples():
    """Load one sample per target label from EMNIST test set."""
    print("Loading EMNIST ByClass test images...")
    images = read_idx_images(os.path.join(DATA_DIR, "emnist-byclass-test-images-idx3-ubyte.gz"))
    labels = read_idx_labels(os.path.join(DATA_DIR, "emnist-byclass-test-labels-idx1-ubyte.gz"))

    collected = {}
    for label in ORIENTATION_CHARS:
        idxs = np.where(labels == label)[0]
        if len(idxs) == 0:
            print(f"  WARNING: No samples for label {label} ({ORIENTATION_CHARS[label]})")
            continue
        # Pick 1 clear sample
        chosen = np.random.choice(idxs, size=1, replace=False)
        collected[label] = images[chosen[0]]
        print(f"  Label {label:2d} ('{ORIENTATION_CHARS[label]}'): {len(idxs):,} available")

    return collected


def generate_grid(collected):
    """Generate a PNG grid: rows=characters, columns=transforms."""
    cell = 56
    pad = 2
    label_w = 70
    header_h = 60

    num_rows = len(collected)
    num_cols = len(ALL_TRANSFORMS)

    grid_w = label_w + num_cols * (cell + pad) + pad
    grid_h = header_h + num_rows * (cell + pad) + pad

    canvas = Image.new("RGB", (grid_w, grid_h), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype("arial.ttf", 13)
        small_font = ImageFont.truetype("arial.ttf", 9)
    except Exception:
        font = ImageFont.load_default()
        small_font = font

    # Column headers (transform names, rotated text as horizontal)
    for t_idx, t_name in enumerate(ALL_TRANSFORMS):
        x = label_w + t_idx * (cell + pad) + pad
        # Draw name vertically by writing each line
        short_name = t_name.replace("legacy_emnist_fix", "legacy_fix").replace("flip_h_rot90", "flh_r90").replace("flip_v_rot90", "flv_r90")
        draw.text((x + 2, 5), short_name, fill=(180, 180, 180), font=small_font)

    # Draw each character row
    for row_idx, (label, img) in enumerate(sorted(collected.items())):
        char = ORIENTATION_CHARS[label]
        y = header_h + row_idx * (cell + pad) + pad

        # Row label
        draw.text((4, y + cell // 3), f"'{char}' ({label})", fill=(255, 200, 100), font=font)

        for t_idx, t_name in enumerate(ALL_TRANSFORMS):
            if t_name == "raw":
                show_img = img
            else:
                show_img = apply_transform(img, t_name)

            pil_img = Image.fromarray(show_img.astype(np.uint8))
            pil_img = pil_img.resize((cell, cell), Image.Resampling.NEAREST)

            x = label_w + t_idx * (cell + pad) + pad
            canvas.paste(pil_img.convert("RGB"), (x, y))

    canvas.save(OUTPUT_PNG)
    print(f"\nGrid saved to: {OUTPUT_PNG}")
    print(f"  Rows = characters, Columns = transforms")
    print(f"  Look for the column where ALL characters look correct\n")


def main():
    np.random.seed(42)
    print("=" * 60)
    print("EMNIST ORIENTATION — ALL TRANSFORMS COMPARISON")
    print("=" * 60)

    collected = load_samples()
    if not collected:
        print("ERROR: No samples loaded.")
        sys.exit(1)

    generate_grid(collected)

    print("Find the column where:")
    print("  M looks like M (not W)")
    print("  W looks like W (not M)")
    print("  6 looks like 6 (not 9)")
    print("  9 looks like 9 (not 6)")
    print("  b looks like b (not d/q)")
    print("  d looks like d (not b/p)")
    print("=" * 60)


if __name__ == "__main__":
    main()
