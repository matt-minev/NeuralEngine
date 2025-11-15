#!/usr/bin/env python3
"""
Neural Engine Landing Page
Beautiful entry point for all Neural Engine web applications
"""

from flask import Flask, render_template, send_from_directory
from pathlib import Path
import os

app = Flask(__name__)

# Serve shared assets from root
@app.route('/assets/<path:filename>')
def serve_assets(filename):
    """Serve shared assets from the root assets directory"""
    assets_dir = Path(__file__).parent.parent.parent / 'assets'
    return send_from_directory(assets_dir, filename)

@app.route('/')
def index():
    """Render the landing page"""
    return render_template('index.html')

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🧠 NEURAL ENGINE - LANDING PAGE")
    print("="*70)
    print(f"🌐 Server running at: http://localhost:8000")
    print("="*70 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=8000)

