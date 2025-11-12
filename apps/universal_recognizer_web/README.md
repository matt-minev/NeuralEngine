# Universal Character Recognition Web App

A Flask web application for universal alphanumeric character recognition (0-9, A-Z, a-z) with accessibility features for first graders and dyslexic users.

## Features

- **Universal Recognition**: Recognizes all 62 alphanumeric characters (0-9, A-Z, a-z)
- **Mirror Detection**: Automatically detects and suggests corrections for mirrored characters
- **Writing Quality Analysis**: Assesses clarity, size, centering, and stroke quality
- **Accessibility Support**: Provides helpful suggestions and resources for users with dyslexia or learning difficulties
- **Real-time Feedback**: Instant predictions with confidence scores

## Requirements

- Python 3.8+
- Trained universal character model at `apps/universal_recognizer_web/models/universal_character_model.pkl`

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure the trained model exists:
```bash
# The model should be at:
# apps/universal_recognizer_web/models/universal_character_model.pkl
```

## Running the Application

```bash
python app.py
```

Or:

```bash
python run.py
```

The application will be available at `http://localhost:8000`

## Usage

1. Draw any alphanumeric character (0-9, A-Z, a-z) on the canvas
2. The app will automatically predict the character
3. View predictions organized by character type (Digits, Uppercase, Lowercase)
4. Check the accessibility panel for:
   - Mirror detection results
   - Writing quality metrics
   - Helpful suggestions
   - Educational resources

## API Endpoints

- `GET /` - Main web interface
- `POST /predict` - Standard character prediction
- `POST /predict/accessibility` - Prediction with full accessibility analysis
- `GET /api/model/info` - Model information
- `GET /health` - Health check
- `GET /api/characters` - List of supported characters

## Accessibility Features

### Mirror Detection
Automatically detects if a character appears mirrored and suggests the correct orientation.

### Writing Quality Assessment
Analyzes:
- Overall quality score
- Clarity (stroke contrast)
- Size (character proportion)
- Centering (position on canvas)
- Stroke quality

### Suggestions
Provides context-aware suggestions based on detected issues:
- Low confidence warnings
- Mirror correction suggestions
- Dyslexia confusion pattern detection (b/d, p/q, etc.)
- Writing technique improvements

### Resources
Links to educational resources for:
- Understanding letter reversals
- Handwriting practice
- Improving writing skills

## Model

The app uses a trained NeuralEngine model with the following architecture:
- Input: 784 neurons (28x28 pixels)
- Hidden layers: 512 → 256 → 128
- Output: 62 neurons (62 classes)
- Activation: ReLU + Softmax

## Development

The application structure:
```
apps/universal_recognizer_web/
├── app.py                 # Flask application
├── run.py                 # Entry point
├── core/
│   ├── model_manager.py   # Model loading
│   ├── predictor.py       # Prediction logic
│   └── accessibility.py   # Accessibility features
├── static/
│   ├── css/
│   │   └── style.css      # Styles
│   └── js/
│       ├── main.js         # Main JavaScript
│       └── accessibility.js # Accessibility JS
└── templates/
    └── index.html         # Main template
```

## License

Part of the NeuralEngine project.

