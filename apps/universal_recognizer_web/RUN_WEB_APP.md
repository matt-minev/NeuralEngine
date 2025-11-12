# How to Run the Web App

## Quick Start (3 Steps)

### 1. Activate Virtual Environment
```bash
# From NeuralEngine root
source .venv/bin/activate
```

### 2. Navigate to Web App Directory
```bash
cd apps/universal_recognizer_web
```

### 3. Run the App
```bash
python app.py
```

That's it! The app will start and you'll see:
```
Starting Universal Character Recognition Web Application
============================================================
Model: Universal Character Recognizer
  Accuracy: 86.61%
  Classes: 62 (0-9, A-Z, a-z)
  Architecture: [784, 512, 256, 128, 62]

Starting web server...
Main app: http://localhost:8000
Model ready for universal character recognition!
```

### 4. Open in Browser
Open your browser and go to:
**http://localhost:8000**

## What You'll See

1. **Drawing Canvas**: Draw any character (0-9, A-Z, a-z)
2. **Prediction Display**: Shows the predicted character with confidence
3. **Tabs**: 
   - Top Predictions (all characters)
   - Digits (0-9)
   - Uppercase (A-Z)
   - Lowercase (a-z)
4. **Accessibility Panel**: Mirror detection, writing quality, suggestions

## Testing the Model

Try drawing:
- **Digits**: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
- **Uppercase**: A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z
- **Lowercase**: a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z

## Troubleshooting

### "Model could not be loaded"
Make sure the model file exists:
```bash
ls -lh apps/universal_recognizer_web/models/universal_character_model.pkl
```

If it doesn't exist, copy it from the root:
```bash
cp models/universal_character_model.pkl apps/universal_recognizer_web/models/
```

### "Module not found"
Install dependencies:
```bash
pip install -r requirements.txt
```

### Port 8000 in use
Change the port in `app.py`:
```python
app.run(debug=True, host='0.0.0.0', port=8001)
```

## Stopping the Server

Press `Ctrl+C` in the terminal to stop the server.

Enjoy your universal character recognizer! 🎉

