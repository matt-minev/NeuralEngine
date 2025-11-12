# Quick Start: Running the Web App

## Step 1: Verify Model Exists

The trained model should be at:
```
apps/universal_recognizer_web/models/universal_character_model.pkl
```

Check if it exists:
```bash
ls -lh apps/universal_recognizer_web/models/universal_character_model.pkl
```

You should see a file around 2-3 MB.

## Step 2: Install Dependencies (if not already done)

```bash
cd apps/universal_recognizer_web
pip install -r requirements.txt
```

## Step 3: Run the Web App

From the `apps/universal_recognizer_web` directory:

```bash
python app.py
```

Or:

```bash
python run.py
```

## Step 4: Open in Browser

Once the server starts, you'll see:
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

Open your browser and go to:
**http://localhost:8000**

## Step 5: Test It!

1. **Draw a character** on the canvas (any digit 0-9, letter A-Z, or a-z)
2. **See the prediction** appear automatically
3. **Check the tabs** for:
   - **Top Predictions**: All character types ranked by confidence
   - **Digits**: Predictions for 0-9
   - **Uppercase**: Predictions for A-Z
   - **Lowercase**: Predictions for a-z
4. **View Accessibility Panel** for:
   - Mirror detection
   - Writing quality analysis
   - Suggestions and resources

## Troubleshooting

### "Model could not be loaded"
- Check the model file exists: `ls apps/universal_recognizer_web/models/universal_character_model.pkl`
- Verify the path in `core/model_manager.py` matches your setup

### "Module not found" errors
- Make sure you're in the virtual environment: `source ../../.venv/bin/activate`
- Install dependencies: `pip install -r requirements.txt`

### Port 8000 already in use
- Change the port in `app.py` or `run.py`:
  ```python
  app.run(debug=True, host='0.0.0.0', port=8001)
  ```

## Features to Try

- **Draw digits**: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
- **Draw uppercase**: A, B, C, D, E, etc.
- **Draw lowercase**: a, b, c, d, e, etc.
- **Try mirroring**: Draw a backwards "b" or "d" to see mirror detection
- **Test accessibility**: Draw messy or unclear characters to see quality analysis

Enjoy your universal character recognizer! 🎉

