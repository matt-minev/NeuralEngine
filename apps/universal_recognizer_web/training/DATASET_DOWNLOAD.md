# EMNIST ByClass Dataset Download Guide

## Quick Summary

**Download Location:** https://www.nist.gov/itl/iad/image-group/emnist-dataset

**Files Needed:** 5 files (all in `.gz` format except mapping.txt)

**Place Files In:** `apps/universal_recognizer_web/data/`

## Step-by-Step Instructions

### 1. Download the Dataset

**Option A: Direct from NIST (Recommended)**
1. Visit: https://www.nist.gov/itl/iad/image-group/emnist-dataset
2. Click on "Download" or find the download section
3. Look for "gzip.zip" or "emnist-byclass.zip"
4. Download the file (~500MB compressed)

**Option B: Alternative Download**
- Some mirrors may have the dataset
- Search for "EMNIST ByClass gzip" if NIST link doesn't work

### 2. Extract the Files

```bash
# Extract the zip file
unzip gzip.zip
# or
unzip emnist-byclass.zip
```

After extraction, you'll find files like:
- `emnist-byclass-train-images-idx3-ubyte.gz`
- `emnist-byclass-train-labels-idx1-ubyte.gz`
- `emnist-byclass-test-images-idx3-ubyte.gz`
- `emnist-byclass-test-labels-idx1-ubyte.gz`
- `emnist-byclass-mapping.txt`

### 3. Copy Files to Correct Location

**Create the data directory:**
```bash
mkdir -p apps/universal_recognizer_web/data
```

**Copy the 5 required files:**
```bash
# From the extracted folder, copy to:
apps/universal_recognizer_web/data/
```

Required files:
- `emnist-byclass-train-images-idx3-ubyte.gz` (~450 MB, but may vary)
- `emnist-byclass-train-labels-idx1-ubyte.gz` (~700 KB)
- `emnist-byclass-test-images-idx3-ubyte.gz` (~75 MB)
- `emnist-byclass-test-labels-idx1-ubyte.gz` (~120 KB)
- `emnist-byclass-mapping.txt` (~1 KB)

### 4. Verify Setup

```bash
cd apps/universal_recognizer_web
python training/download_dataset.py --verify-only
```

You should see all files marked with ✓

### 5. Start Training

```bash
python training/train.py --config high_accuracy
```

## File Structure

After setup, your directory should look like:

```
apps/universal_recognizer_web/
├── data/
│   ├── emnist-byclass-train-images-idx3-ubyte.gz
│   ├── emnist-byclass-train-labels-idx1-ubyte.gz
│   ├── emnist-byclass-test-images-idx3-ubyte.gz
│   ├── emnist-byclass-test-labels-idx1-ubyte.gz
│   └── emnist-byclass-mapping.txt
├── models/
│   └── (trained model will be saved here)
└── training/
    └── (training scripts)
```

## Important Notes

- **Location**: Files MUST be in `apps/universal_recognizer_web/data/`
- **Format**: Files must be `.gz` (gzipped) format (except mapping.txt)
- **Naming**: File names must match exactly (case-sensitive)
- **Size**: File sizes may vary slightly between EMNIST versions

## Troubleshooting

**Files not found?**
- Double-check the path: `apps/universal_recognizer_web/data/`
- Verify file names match exactly
- Make sure files are `.gz` format

**File size warnings?**
- Different EMNIST versions may have different sizes
- As long as files exist and are not empty, they should work
- The training script will validate the files when loading

**Can't find download link?**
- NIST website: https://www.nist.gov/itl/iad/image-group/emnist-dataset
- Direct link: https://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/
- Look for files containing "byclass" or "gzip"

## Dataset Information

- **Name**: EMNIST ByClass
- **Source**: NIST (National Institute of Standards and Technology)
- **Format**: Binary IDX format (gzipped)
- **Classes**: 62 (0-9, A-Z, a-z)
- **Training samples**: ~697,932
- **Test samples**: ~116,323
- **Image size**: 28x28 grayscale
- **License**: Free for research and educational use

