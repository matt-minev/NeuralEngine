# EMNIST ByClass Dataset Setup Guide

## Quick Start

The training pipeline requires the EMNIST ByClass dataset. Here's how to get it set up:

## Step 1: Download the Dataset

### Option A: Direct Download (Recommended)

1. **Visit the official EMNIST page:**
   - https://www.nist.gov/itl/iad/image-group/emnist-dataset
   - Or directly: https://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/

2. **Download the dataset:**
   - Look for "gzip.zip" or "emnist-byclass.zip"
   - This is a large file (~500MB compressed, ~2GB uncompressed)

3. **Extract the zip file:**
   ```bash
   unzip gzip.zip
   # or
   unzip emnist-byclass.zip
   ```

### Option B: Using wget/curl

```bash
# Create data directory
mkdir -p apps/universal_recognizer/data
cd apps/universal_recognizer/data

# Download (if direct link is available)
wget https://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip
unzip gzip.zip
```

## Step 2: Place Files in Correct Location

After extracting, you need these 5 files in the `apps/universal_recognizer_web/data/` directory:

```
apps/universal_recognizer_web/data/
├── emnist-byclass-train-images-idx3-ubyte.gz    (~450 MB)
├── emnist-byclass-train-labels-idx1-ubyte.gz    (~700 KB)
├── emnist-byclass-test-images-idx3-ubyte.gz     (~75 MB)
├── emnist-byclass-test-labels-idx1-ubyte.gz     (~120 KB)
└── emnist-byclass-mapping.txt                    (~1 KB)
```

**Copy these files** from the extracted folder to `apps/universal_recognizer_web/data/`

## Step 3: Verify Dataset

Run the verification script:

```bash
cd apps/universal_recognizer_web
python -m training.download_dataset --verify-only
```

This will check that all files are present and have the correct sizes.

## Step 4: Run Training

Once verified, you can start training:

```bash
cd apps/universal_recognizer_web
python -m training.train --config high_accuracy
```

## File Sizes (for verification)

- `emnist-byclass-train-images-idx3-ubyte.gz`: ~450 MB
- `emnist-byclass-test-images-idx3-ubyte.gz`: ~75 MB
- `emnist-byclass-train-labels-idx1-ubyte.gz`: ~700 KB
- `emnist-byclass-test-labels-idx1-ubyte.gz`: ~120 KB
- `emnist-byclass-mapping.txt`: ~1 KB

## Troubleshooting

### Files not found?
- Make sure files are in `apps/universal_recognizer_web/data/`
- Check file names match exactly (case-sensitive)
- Verify files are `.gz` format (compressed)

### Wrong file sizes?
- Re-download the dataset
- Check that files weren't corrupted during download
- Verify you downloaded "ByClass" split (not Digits, Letters, etc.)

### Alternative: Use existing data
If you already have EMNIST data elsewhere, you can:
1. Create a symlink: `ln -s /path/to/your/data apps/universal_recognizer/data`
2. Or specify data directory: `python -m training.train --data-dir /path/to/your/data`

## Dataset Information

- **Name**: EMNIST ByClass
- **Source**: NIST (National Institute of Standards and Technology)
- **Format**: Binary IDX format (gzipped)
- **Classes**: 62 (0-9, A-Z, a-z)
- **Training samples**: ~697,932
- **Test samples**: ~116,323
- **Image size**: 28x28 grayscale

## License

EMNIST is freely available for research and educational purposes. See NIST website for details.

