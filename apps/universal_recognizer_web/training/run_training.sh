#!/bin/bash
# Training script wrapper that ensures proper environment setup

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../../.." && pwd )"

echo "NeuralEngine Training Script"
echo "============================"
echo "Project root: $PROJECT_ROOT"
echo ""

# Check if virtual environment exists
if [ -d "$PROJECT_ROOT/.venv" ]; then
    echo "Activating virtual environment..."
    source "$PROJECT_ROOT/.venv/bin/activate"
fi

# Change to training directory
cd "$SCRIPT_DIR/.."

# Run training
python training/train.py "$@"

