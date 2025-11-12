# Quick Start: Download EMNIST Dataset

## TL;DR - Fastest Way

1. **Download from NIST:**
   ```bash
   # Create data directory
   mkdir -p apps/universal_recognizer/data
   cd apps/universal_recognizer/data
   
   # Download (you may need to do this manually via browser)
   # Visit: https://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip
   # Or use wget if direct link works:
   wget https://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip
   
   # Extract
   unzip gzip.zip
   
   # Copy ByClass files to data directory
   # (Files will be in the extracted folder)
   ```

2. **Verify:**
   ```bash
   cd ../../universal_recognizer_web
   python -m training.download_dataset --verify-only
   ```

3. **Train:**
   ```bash
   python -m training.train --config high_accuracy
   ```

## Detailed Instructions

### Step 1: Download EMNIST ByClass

**Option 1: Browser Download (Easiest)**
1. Go to: https://www.nist.gov/itl/iad/image-group/emnist-dataset
2. Click "Download" or find the "gzip.zip" file
3. Download the zip file (~500MB)

**Option 2: Command Line**
```bash
cd apps/universal_recognizer/data
wget https://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip
unzip gzip.zip
```

### Step 2: Extract and Organize Files

After extracting, you'll find files like:
- `emnist-byclass-train-images-idx3-ubyte.gz`
- `emnist-byclass-train-labels-idx1-ubyte.gz`
- `emnist-byclass-test-images-idx3-ubyte.gz`
- `emnist-byclass-test-labels-idx1-ubyte.gz`
- `emnist-byclass-mapping.txt`

**Copy these 5 files to:**
```
apps/universal_recognizer_web/data/
```

### Step 3: Verify Setup

```bash
cd apps/universal_recognizer_web
python -m training.download_dataset --verify-only
```

You should see all files marked with ✓

### Step 4: Start Training

```bash
python -m training.train --config high_accuracy
```

## File Structure After Setup

```
apps/
└── universal_recognizer_web/
    ├── data/
    │   ├── emnist-byclass-train-images-idx3-ubyte.gz    (~450 MB)
    │   ├── emnist-byclass-train-labels-idx1-ubyte.gz    (~700 KB)
    │   ├── emnist-byclass-test-images-idx3-ubyte.gz     (~75 MB)
    │   ├── emnist-byclass-test-labels-idx1-ubyte.gz     (~120 KB)
    │   └── emnist-byclass-mapping.txt                    (~1 KB)
    └── training/
        └── train.py
```

## Important Notes

- **Location**: Files go in `apps/universal_recognizer_web/data/`
- **Format**: Files must be `.gz` (gzipped) format
- **Naming**: File names must match exactly (case-sensitive)
- **Size**: Total dataset is ~525 MB compressed

## Troubleshooting

**"File not found" error?**
- Check file names match exactly
- Verify files are in `apps/universal_recognizer_web/data/`
- Make sure files are `.gz` format

**"File too small" warning?**
- Re-download the dataset
- Check for download corruption
- Verify you got "ByClass" split (not Digits/Letters)

**Can't find download link?**
- NIST website: https://www.nist.gov/itl/iad/image-group/emnist-dataset
- Direct link: https://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/
- Look for "gzip.zip" or "emnist-byclass" files

