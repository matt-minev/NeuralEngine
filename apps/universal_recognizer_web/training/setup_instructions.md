# Training Setup Instructions

## Prerequisites

Before running training, you need to:

1. **Activate your virtual environment** (if using one):
   ```bash
   # From NeuralEngine root
   source .venv/bin/activate  # On macOS/Linux
   # or
   .venv\Scripts\activate  # On Windows
   ```

2. **Install dependencies**:
   ```bash
   cd apps/universal_recognizer_web
   pip install -r requirements.txt
   ```

   Or install from root:
   ```bash
   # From NeuralEngine root
   pip install -r requirements.txt
   pip install -r apps/universal_recognizer_web/requirements.txt
   ```

## Running Training

Once dependencies are installed:

```bash
cd apps/universal_recognizer_web
python training/train.py --config high_accuracy
```

## Troubleshooting Import Errors

If you get `ModuleNotFoundError: No module named 'nn_core'`:

1. **Make sure you're in the right directory:**
   ```bash
   cd apps/universal_recognizer_web
   ```

2. **Check Python path:**
   ```bash
   python -c "import sys; print('\n'.join(sys.path))"
   ```
   Should include the NeuralEngine root directory.

3. **Try running from NeuralEngine root:**
   ```bash
   cd /Users/matt/Developer/NeuralEngine
   python -m apps.universal_recognizer_web.training.train --config high_accuracy
   ```

If you get `ModuleNotFoundError: No module named 'numpy'`:

1. **Activate virtual environment:**
   ```bash
   source .venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install numpy autograd scipy scikit-learn
   ```

