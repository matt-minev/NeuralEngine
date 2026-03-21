#!/usr/bin/env python3
"""Neural Engine landing page for launching the web applications."""

from flask import Flask, render_template, send_from_directory
from pathlib import Path

app = Flask(__name__)

APP_CARDS = [
    {
        "name": "Digit Recognizer",
        "slug": "digit",
        "icon": "🔢",
        "description": (
            "Draw handwritten digits, stream live confidence across the output layer, "
            "and jump into the showcase for dataset and activation deep-dives."
        ),
        "primary_url": "http://localhost:8001",
        "primary_label": "Launch App",
        "secondary_url": "http://localhost:8001/showcase",
        "secondary_label": "Open Showcase",
        "port": 8001,
        "status": "Live inference workspace",
    },
    {
        "name": "Universal Character Recognizer",
        "slug": "universal",
        "icon": "✍️",
        "description": (
            "Recognize digits, uppercase, and lowercase characters with mirror detection, "
            "writing diagnostics, and accessibility-aware tooling."
        ),
        "primary_url": "http://localhost:8003",
        "primary_label": "Launch App",
        "secondary_url": None,
        "secondary_label": None,
        "port": 8003,
        "status": "Extended character intelligence",
    },
    {
        "name": "Quadratic Neural Network",
        "slug": "quadratic",
        "icon": "📊",
        "description": (
            "Generate datasets, train neural models, and predict quadratic roots with "
            "an interactive research surface built for experimentation."
        ),
        "primary_url": "http://localhost:8002",
        "primary_label": "Launch App",
        "secondary_url": None,
        "secondary_label": None,
        "port": 8002,
        "status": "Equation modeling lab",
    },
]

# Serve shared assets from root
@app.route('/assets/<path:filename>')
def serve_assets(filename):
    """Serve shared assets from the root assets directory"""
    assets_dir = Path(__file__).parent.parent.parent / 'assets'
    return send_from_directory(assets_dir, filename)

@app.route('/')
def index():
    """Render the landing page."""
    return render_template("index.html", app_cards=APP_CARDS)

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🧠 NEURAL ENGINE - LANDING PAGE")
    print("="*70)
    print(f"🌐 Server running at: http://localhost:8000")
    print("="*70 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=8000)

