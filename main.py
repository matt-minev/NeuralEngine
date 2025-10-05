"""
Neural Network Engine - Demo and overview.

Shows capabilities and architecture of the neural network engine.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional


def print_welcome_message():
    """Print welcome and overview of the engine."""
    print("=" * 60)
    print("NEURAL NETWORK ENGINE")
    print("=" * 60)
    print("\nWelcome to the Neural Network Engine!")
    print("This engine solves function approximation problems using gradient descent.")
    print("\nCore concepts:")
    print("  - Input: x (feature vector)")
    print("  - Output: y (predictions)")
    print("  - Parameters: theta (weights and biases)")
    print("  - Goal: Minimize Loss(theta) = ||y_true - f(x, theta)||^2")
    print("\nOptimization via gradient descent:")
    print("  theta_new = theta_old - alpha * gradient")
    print("=" * 60)


def demonstrate_basic_concepts():
    """Show simple function approximation examples."""
    print("\nBASIC FUNCTION APPROXIMATION")
    print("-" * 50)
    
    # linear function example
    print("\n1. Linear Function:")
    print("  Target: y = 2x + 1")
    print("  Network learns: f(x) = w*x + b where w~=2, b~=1")
    
    x_sample = np.array([1, 2, 3, 4, 5])
    y_target = 2 * x_sample + 1
    
    print(f"  Sample input: {x_sample}")
    print(f"  Target output: {y_target}")
    
    # non-linear example
    print("\n2. Non-linear Function:")
    print("  Target: y = sin(x)")
    print("  Requires multi-layer network with activations")
    
    x_nonlinear = np.linspace(0, 2*np.pi, 10)
    y_sine = np.sin(x_nonlinear)
    
    print(f"  Sample input: {x_nonlinear[:5]}...")
    print(f"  Target output: {y_sine[:5]}...")


def explain_automatic_differentiation():
    """Explain why autodiff is useful."""
    print("\nAUTOMATIC DIFFERENTIATION")
    print("-" * 50)
    
    print("\nManual differentiation (tedious):")
    print("  For loss L = (y_true - (w*x + b))^2")
    print("  Need to calculate:")
    print("    dL/dw = 2(y_true - (w*x + b)) * (-x)")
    print("    dL/db = 2(y_true - (w*x + b)) * (-1)")
    print("  Gets complex with more layers!")
    
    print("\nAutomatic differentiation (easy):")
    print("  1. Define forward pass: loss = mse(y_true, network(x))")
    print("  2. Autograd computes all gradients automatically")
    print("  3. Use gradients: theta = theta - alpha*grad")
    print("  Works for any architecture!")


def show_engine_architecture():
    """Display module structure of the engine."""
    print("\nENGINE ARCHITECTURE")
    print("-" * 50)
    
    print("\nCore modules:")
    print("  - nn_core.py     : layers and forward propagation")
    print("  - autodiff.py    : gradient computation and optimzation")
    print("  - data_utils.py  : data loading and preprocessing")
    print("  - utils.py       : activation functions and helpers")
    
    print("\nProcessing flow:")
    print("  1. Load data -> normalize")
    print("  2. Forward pass -> predictions")
    print("  3. Compute loss -> calculate gradients")
    print("  4. Update parameters")
    print("  5. Repeat until convergence")
    
    print("\nExample applications:")
    print("  - number_predictor/  : regression demo")
    print("  - digit_recognizer/  : image classification with GUI")


def preview_upcoming_features():
    """Overview of planned features."""
    print("\nUPCOMING FEATURES")
    print("-" * 50)
    
    print("\nNeural Network Core:")
    print("  - Configurable architecture")
    print("  - Multiple activation functions (ReLU, sigmoid, tanh, etc)")
    print("  - Flexible loss functions")
    
    print("\nAutomatic Differentiation:")
    print("  - Gradient computation for any network")
    print("  - Optimizers: SGD, Adam, RMSprop")
    print("  - Learning rate scheduling")
    print("  - Momentum and regularization")
    
    print("\nData Processing:")
    print("  - CSV/JSON loading")
    print("  - Auto normalization")
    print("  - Train/val/test splits")
    print("  - Data augmentation")
    
    print("\nInteractive Apps:")
    print("  - Real-time predictions")
    print("  - Draw-and-predict interface")
    print("  - Training visualization")


def main():
    """Main demo orchestrator."""
    print_welcome_message()
    demonstrate_basic_concepts()
    explain_automatic_differentiation()
    show_engine_architecture()
    preview_upcoming_features()


if __name__ == "__main__":
    main()
