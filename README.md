# 🧠 Neural Network Engine

**A modular, educational neural network library built from scratch in Python with automatic differentiation**

[![Tests](https://img.shields.io/badge/tests-32%20passed-brightgreen)](tests/)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-educational-orange)](README.md)

## 🎯 Overview

The Neural Network Engine is a powerful yet beginner-friendly library designed to solve **function approximation problems** using neural networks. Built with educational clarity and mathematical rigor, it demonstrates how modern deep learning works under the hood.

### 🔬 Core Concept

Neural networks solve the fundamental machine learning problem:

```
Given: f(x) = Neural Network with parameters θ
Goal: Find θ* that minimizes Loss(f(x, θ), y_true)
Method: Gradient descent using automatic differentiation
```

### ✨ Key Features

- **🔥 Automatic Differentiation**: Uses `autograd` for effortless gradient computation.
- **🏗️ Modular Architecture**: Clean, extensible design for experimentation.
- **📚 Educational Focus**: Extensively commented code explaining every concept.
- **⚡ High Performance**: Optimized for speed with high sample throughput.
- **🎨 Visualization Tools**: Includes tools for plotting network architecture and training progress.
- **📊 Comprehensive Testing**: A robust test suite ensures reliability and correctness across the core engine and applications.
- **🔧 Application Suite**: Comes with multiple pre-built applications demonstrating various capabilities.

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/matt-minev/NeuralEngine.git
cd NeuralEngine

# Install dependencies
pip install -r requirements.txt

# Run core tests to verify installation
pytest tests/test_nn.py -v
```

### Your First Neural Network

```python
from nn_core import NeuralNetwork, mean_squared_error
from autodiff import TrainingEngine, Adam
from data_utils import create_sample_data

# Create sample data: y = 2x₁ + 3x₂ + 1
X, y = create_sample_data(1000)

# Build neural network: 2 inputs -> 8 hidden -> 4 hidden -> 1 output
network = NeuralNetwork([2, 8, 4, 1], ['relu', 'relu', 'linear'])

# Set up training with Adam optimizer
trainer = TrainingEngine(network, Adam(learning_rate=0.001), mean_squared_error)

# Train the network
history = trainer.train(X, y, epochs=100, verbose=True)

# Make predictions
predictions = network.predict([[5, 2], [6, 4]])
print(f"Predictions: {predictions}")
```

## 📖 Architecture Overview

The project is organized into a core library and a suite of standalone applications.

```
NeuralEngine/
│
├── main.py                  # Entry point with educational demonstrations
├── nn_core.py               # Core neural network implementation
├── autodiff.py              # Automatic differentiation and optimizers
├── data_utils.py            # Data loading, preprocessing, and utilities
├── utils.py                 # Activation functions and helper tools
├── cleanup.py               # Utility scripts for cleanup
│
├── apps/                    # Standalone applications
│   ├── quadratic_equation/  # Desktop GUI for solving quadratic equations
│   ├── quadratic_web/       # Web app for the quadratic equation solver
│   ├── digit_recognizer/    # Base application for digit recognition
│   ├── digit_recognizer_extended/ # Extended version with enhanced models
│   ├── digit_recognizer_web/      # Web interface for digit recognition
│   └── universal_recognizer/    # Advanced character recognizer
│
├── tests/
│   └── test_nn.py           # Comprehensive test suite for the core engine
│
├── requirements.txt         # Project dependencies
├── README.md                # This file
├── LICENSE                  # Project license
└── CHANGELOG.md             # Changelog
```

## 🔧 Core Components

### Neural Network (`nn_core.py`)

- **Layer Class**: Individual computational units that form the network.
- **NeuralNetwork Class**: A multi-layer function approximator that chains layers together.

```python
# A network with 7 inputs, two hidden layers of 8 and 7 neurons, and 5 outputs
network = NeuralNetwork([7, 8, 7, 5], ['relu', 'relu', 'linear'])
predictions = network.forward(X)
```

### Automatic Differentiation (`autodiff.py`)

- **Optimizers**: Advanced gradient-based learning algorithms like `SGD` and `Adam`.
- **Training Engine**: A complete pipeline for training models, handling epochs, batching, and validation.

```python
trainer = TrainingEngine(network, Adam(learning_rate=0.001), mean_squared_error)
history = trainer.train(X, y, epochs=100, validation_data=(X_val, y_val))
```

### Data & General Utilities (`data_utils.py`, `utils.py`)

- **Data Processing**: Tools for loading, normalizing, scaling, and splitting datasets.
- **Activation Functions**: A collection of standard activations (`relu`, `sigmoid`, `tanh`, etc.).
- **Visualization**: Helpers to plot network architecture and training metrics.

```python
NetworkVisualizer.plot_network_architecture([2, 8, 4, 1])
NetworkVisualizer.plot_training_metrics(history)
```

## 🎮 Applications

The `apps/` directory contains several examples that use the `NeuralEngine` core.

### Quadratic Equation Solver

- **Description**: A desktop GUI (`quadratic_equation`) and a web app (`quadratic_web`) that train a neural network to find the roots of quadratic equations.
- **To Run (Desktop)**: `cd apps/quadratic_equation/ && python main.py`
- **To Run (Web)**: `cd apps/quadratic_web/ && python app.py`

### Digit Recognizer Suite

- **Description**: A collection of applications for recognizing handwritten digits. It includes a basic version, an extended version with more robust models, and a web interface.
- **To Run (Basic)**: `cd apps/digit_recognizer/ && python digit_recognizer.py`
- **To Run (Web)**: `cd apps/digit_recognizer_web/ && python app.py`

### Universal Character Recognizer

- **Description**: A more advanced application that recognizes a broader set of characters beyond digits, showcasing the engine's versatility.
- **To Run**: `cd apps/universal_recognizer/ && python universal_recognizer.py`

## 🧪 Testing

The project includes a comprehensive test suite covering the core engine and individual applications.

**Run Core Engine Tests**:

```bash
# Run all core tests with verbose output
pytest tests/test_nn.py -v
```

**Run Application-Specific Tests**:

```bash
# Example: Run tests for the digit recognizer
pytest apps/digit_recognizer/comprehensive_test.py

# Example: Run tests for the universal recognizer
pytest apps/universal_recognizer/comprehensive_universal_test.py

# Example: Run tests for the quadratic web app
pytest apps/quadratic_web/tests/test_web_app.py
```

**Test Coverage Includes**:

- ✅ Core engine: Layers, network, optimizers, loss functions
- ✅ Utilities: Data processing and helper functions
- ✅ Integration Workflows: End-to-end training and prediction
- ✅ Application Logic: Tests for each standalone application

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines.

### Development Setup

```bash
# Fork the repository and clone it
git clone https://github.com/matt-minev/NeuralEngine.git
cd NeuralEngine

# Install development dependencies
pip install -r requirements.txt
pip install pytest pytest-cov

# Run tests to ensure everything works
pytest
```

## 📋 Requirements

- **Python**: 3.8+
- **NumPy**: ≥1.20.0
- **Autograd**: ≥1.3.0
- **Pandas**: ≥1.3.0
- **Matplotlib**: ≥3.5.0
- **Scikit-learn**: ≥1.0.0
- **Pytest**: ≥6.2.0

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Autograd Team**: For providing excellent automatic differentiation.
- **NumPy Community**: For foundational numerical computing tools.
- **Open Source Community**: For the tools and libraries that made this possible.

## 📞 Contact

- **Developer**: Matt
- **Project**: Neural Network Engine
- **Status**: Active development and maintenance

**Ready to explore the fascinating world of neural networks? Start with the Quick Start guide above and dive into the mathematical beauty of machine learning! 🧠✨**
