// NeuralEngine Universal Character Recognizer - Main JavaScript
class UniversalCharacterRecognizer {
  constructor() {
    this.canvas = document.getElementById("drawingCanvas");
    this.ctx = this.canvas.getContext("2d");
    this.isDrawing = false;
    this.brushSize = 15;
    this.currentTab = 'all';
    this.lastPrediction = null;

    this.setupCanvas();
    this.setupEventListeners();
    this.setupTabs();
    console.log("🧠 Universal Character Recognizer initialized");
  }

  setupCanvas() {
    this.ctx.lineCap = "round";
    this.ctx.lineJoin = "round";
    this.ctx.fillStyle = "#000";
    this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
  }

  setupEventListeners() {
    // Mouse events
    this.canvas.addEventListener("mousedown", (e) => this.startDrawing(e));
    this.canvas.addEventListener("mousemove", (e) => this.draw(e));
    this.canvas.addEventListener("mouseup", () => this.stopDrawing());
    this.canvas.addEventListener("mouseout", () => this.stopDrawing());

    // Touch events
    this.canvas.addEventListener("touchstart", (e) => {
      e.preventDefault();
      const touch = e.touches[0];
      const rect = this.canvas.getBoundingClientRect();
      const mouseEvent = new MouseEvent("mousedown", {
        clientX: touch.clientX,
        clientY: touch.clientY,
      });
      this.canvas.dispatchEvent(mouseEvent);
    });

    this.canvas.addEventListener("touchmove", (e) => {
      e.preventDefault();
      const touch = e.touches[0];
      const mouseEvent = new MouseEvent("mousemove", {
        clientX: touch.clientX,
        clientY: touch.clientY,
      });
      this.canvas.dispatchEvent(mouseEvent);
    });

    this.canvas.addEventListener("touchend", (e) => {
      e.preventDefault();
      const mouseEvent = new MouseEvent("mouseup", {});
      this.canvas.dispatchEvent(mouseEvent);
    });

    // Brush size
    const brushSizeSlider = document.getElementById("brushSize");
    const brushValue = document.getElementById("brushValue");
    brushSizeSlider.addEventListener("input", (e) => {
      this.brushSize = parseInt(e.target.value);
      brushValue.textContent = this.brushSize;
    });

    // Clear button
    document.getElementById("clearBtn").addEventListener("click", () => {
      this.clearCanvas();
    });

    // Predict button
    document.getElementById("predictBtn").addEventListener("click", () => {
      this.predict();
    });

    // Auto-predict on drawing end (debounced)
    let predictionTimeout;
    this.canvas.addEventListener("mouseup", () => {
      clearTimeout(predictionTimeout);
      predictionTimeout = setTimeout(() => {
        if (this.hasDrawing()) {
          this.predict();
        }
      }, 500);
    });
  }

  setupTabs() {
    const tabButtons = document.querySelectorAll(".tab-btn");
    tabButtons.forEach(btn => {
      btn.addEventListener("click", () => {
        tabButtons.forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        this.currentTab = btn.dataset.tab;
        this.updatePredictionsDisplay();
      });
    });
  }

  startDrawing(e) {
    this.isDrawing = true;
    const rect = this.canvas.getBoundingClientRect();
    this.lastX = e.clientX - rect.left;
    this.lastY = e.clientY - rect.top;
    this.hideOverlay();
  }

  draw(e) {
    if (!this.isDrawing) return;

    const rect = this.canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    this.ctx.strokeStyle = "#fff";
    this.ctx.lineWidth = this.brushSize;
    this.ctx.beginPath();
    this.ctx.moveTo(this.lastX, this.lastY);
    this.ctx.lineTo(x, y);
    this.ctx.stroke();

    this.lastX = x;
    this.lastY = y;
  }

  stopDrawing() {
    this.isDrawing = false;
  }

  clearCanvas() {
    this.ctx.fillStyle = "#000";
    this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
    this.showOverlay();
    this.lastPrediction = null;
    this.updatePredictionsDisplay();
    if (window.updateAccessibilityDisplay) {
      window.updateAccessibilityDisplay(null);
    }
  }

  hasDrawing() {
    const imageData = this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height);
    const data = imageData.data;
    for (let i = 0; i < data.length; i += 4) {
      if (data[i] > 0 || data[i + 1] > 0 || data[i + 2] > 0) {
        return true;
      }
    }
    return false;
  }

  getCanvasImageData() {
    return this.canvas.toDataURL("image/png");
  }

  hideOverlay() {
    const overlay = document.getElementById("canvasOverlay");
    if (overlay) overlay.classList.remove("show");
  }

  showOverlay() {
    const overlay = document.getElementById("canvasOverlay");
    if (overlay) overlay.classList.add("show");
  }

  async predict() {
    if (!this.hasDrawing()) {
      return;
    }

    const imageData = this.getCanvasImageData();
    const startTime = performance.now();

    try {
      // Use accessibility endpoint for full analysis
      const response = await fetch("/predict/accessibility", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ image: imageData }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      this.lastPrediction = result;
      this.displayPrediction(result);
      
      if (window.updateAccessibilityDisplay) {
        window.updateAccessibilityDisplay(result);
      }
    } catch (error) {
      console.error("Prediction error:", error);
      alert("Failed to get prediction. Please try again.");
    }
  }

  displayPrediction(result) {
    const prediction = result.prediction;
    const predictionTime = result.prediction_time || 0;

    // Update main prediction display
    document.getElementById("predictedCharacter").textContent = prediction.predicted_character;
    document.getElementById("characterType").textContent = prediction.character_type;
    document.getElementById("confidence").textContent = `Confidence: ${prediction.confidence.toFixed(1)}%`;
    document.getElementById("predictionTime").textContent = `${predictionTime.toFixed(0)}ms`;

    // Update predictions list
    this.updatePredictionsDisplay();
  }

  updatePredictionsDisplay() {
    const container = document.getElementById("topPredictions");
    if (!this.lastPrediction || !this.lastPrediction.prediction) {
      container.innerHTML = '<p class="no-prediction">Draw a character to see predictions</p>';
      return;
    }

    const topPredictions = this.lastPrediction.prediction.top_predictions || [];
    const filtered = this.filterByTab(topPredictions);

    if (filtered.length === 0) {
      container.innerHTML = '<p class="no-prediction">No predictions in this category</p>';
      return;
    }

    container.innerHTML = filtered.map((pred, index) => `
      <div class="prediction-item">
        <div class="prediction-char">${pred.character}</div>
        <div class="prediction-type">${pred.type}</div>
        <div class="prediction-bar-container">
          <div class="prediction-bar-fill" style="width: ${pred.confidence}%"></div>
        </div>
        <div class="prediction-percentage">${pred.confidence.toFixed(1)}%</div>
      </div>
    `).join('');
  }

  filterByTab(predictions) {
    if (this.currentTab === 'all') return predictions;
    
    const typeMap = {
      'digits': 'Digit',
      'uppercase': 'Uppercase',
      'lowercase': 'Lowercase'
    };
    
    const targetType = typeMap[this.currentTab];
    return predictions.filter(p => p.type === targetType);
  }
}

// Initialize app when DOM is ready
document.addEventListener("DOMContentLoaded", () => {
  window.recognizer = new UniversalCharacterRecognizer();
});

