# Universal Character Recognition - Training Plan

## Goal Description
The codebase has been successfully updated with the three major changes:
1. **GPU Acceleration**: Integrated via [neural_backend.py](file:///c:/Users/Matt/Desktop/CNN/NeuralEngine-main/neural_backend.py) (CuPy).
2. **C++ Backend**: Integrated via `neural_engine_native` (though when using the GPU, CuPy takes precedence for maximum performance).
3. **CNN Overhaul**: The model has been redesigned into a Convolutional Neural Network ([UniversalCNN](file:///c:/Users/Matt/Desktop/CNN/NeuralEngine-main/apps/universal_recognizer_web/training/cnn_model.py#87-306) in [cnn_model.py](file:///c:/Users/Matt/Desktop/CNN/NeuralEngine-main/apps/universal_recognizer_web/training/cnn_model.py)) featuring `Conv32 -> Pool -> Conv64 -> Pool -> FC256 -> FC62` layers. 

The standalone script [train_cnn.py](file:///c:/Users/Matt/Desktop/CNN/NeuralEngine-main/apps/universal_recognizer_web/training/train_cnn.py) correctly wires all these together. The training pipeline is **ready to roll**. 

The goal now is to establish a training plan that maximizes accuracy while strictly staying within a "few hours" limit on a single GPU.

## User Review Required
Please review the proposed hyperparameters below. EMNIST ByClass contains roughly 697,000 training samples. The default target in the script is only 180,000 samples, which is too low for the highest possible accuracy. Given GPU acceleration, we can scale this up significantly while still completing within 2-3 hours.

## Proposed Strategy (Training Run)

We will execute [train_cnn.py](file:///c:/Users/Matt/Desktop/CNN/NeuralEngine-main/apps/universal_recognizer_web/training/train_cnn.py) with the following optimized parameters:

* **Training Samples**: `600,000` (utilizing nearly the entire dataset to maximize generalizability across all 62 classes).
* **Validation Samples**: `60,000` (to ensure robust intermediate evaluation).
* **Epochs**: `50` (Using Adam with a CNN on 600k samples usually converges nicely by 40-50 epochs. Going beyond this often brings diminishing returns and risks overfitting the EMNIST set).
* **Batch Size**: `512` (Since the images are 28x28 grayscale and the CNN is relatively lightweight, a larger batch size maximizes GPU throughput without exceeding VRAM).
* **Learning Rate**: `0.001` (Standard Adam learning rate; since this is a relatively simple architecture, complex scheduling is less critical than having enough raw data).
* **Device**: `gpu` (Forces the CuPy backend).

### Execution Command:
```bash
python apps/universal_recognizer_web/training/train_cnn.py --device gpu --train-samples 600000 --val-samples 60000 --test-samples 30000 --epochs 50 --batch-size 512 --lr 0.001
```

## Verification Plan
### Automated Verification
- Track the output of the script to ensure `val_acc` climbs steadily and `train_loss` decreases.
- The script automatically evaluates the `test_acc` on the holdout test set at the end of training.
- The script saves the model to `models/universal_character_model.pkl`.

### Manual Verification
- After the model is saved, we will launch the [apps/universal_recognizer_web/app.py](file:///c:/Users/Matt/Desktop/NECNN/NeuralEngine/apps/universal_recognizer_web/app.py) web app to manually draw characters and verify the real-world recognition capabilities.
