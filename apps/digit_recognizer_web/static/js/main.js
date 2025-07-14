// NeuralEngine Web App - Interactive JavaScript
class DigitRecognizer {
  constructor() {
    this.canvas = document.getElementById("drawingCanvas");
    this.ctx = this.canvas.getContext("2d");
    this.isDrawing = false;
    this.brushSize = 15;

    this.setupCanvas();
    this.setupEventListeners();
    this.setupPredictionDisplay();
    this.lastPredictionTime = 0;

    console.log("🧠 NeuralEngine Web App initialized");
  }

  setupCanvas() {
    // Set up canvas properties
    this.ctx.lineCap = "round";
    this.ctx.lineJoin = "round";
    this.ctx.fillStyle = "#000";
    this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

    // High DPI support
    const rect = this.canvas.getBoundingClientRect();
    const scaleX = this.canvas.width / rect.width;
    const scaleY = this.canvas.height / rect.height;
    this.scaleX = scaleX;
    this.scaleY = scaleY;
  }

  setupEventListeners() {
    // Mouse events
    this.canvas.addEventListener("mousedown", (e) => this.startDrawing(e));
    this.canvas.addEventListener("mousemove", (e) => this.draw(e));
    this.canvas.addEventListener("mouseup", () => this.stopDrawing());
    this.canvas.addEventListener("mouseout", () => this.stopDrawing());

    // Touch events for mobile
    this.canvas.addEventListener("touchstart", (e) => {
      e.preventDefault();
      const touch = e.touches[0];
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

    // Brush size control
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
  }

  setupPredictionDisplay() {
    // Create confidence bars for digits 0-9
    const confidenceBars = document.getElementById("confidenceBars");

    for (let digit = 0; digit < 10; digit++) {
      const barElement = document.createElement("div");
      barElement.className = "confidence-bar";
      barElement.innerHTML = `
                <span class="digit-label">${digit}</span>
                <div class="bar-container">
                    <div class="bar-fill" id="bar-${digit}"></div>
                </div>
                <span class="percentage" id="percent-${digit}">0%</span>
            `;
      confidenceBars.appendChild(barElement);
    }
  }

  getMousePos(e) {
    const rect = this.canvas.getBoundingClientRect();
    return {
      x: (e.clientX - rect.left) * this.scaleX,
      y: (e.clientY - rect.top) * this.scaleY,
    };
  }

  startDrawing(e) {
    this.isDrawing = true;
    const pos = this.getMousePos(e);
    this.ctx.beginPath();
    this.ctx.arc(pos.x, pos.y, this.brushSize / 2, 0, 2 * Math.PI);
    this.ctx.fillStyle = "#fff";
    this.ctx.fill();
    this.lastX = pos.x;
    this.lastY = pos.y;

    // Hide instructions overlay
    document.querySelector(".canvas-overlay").classList.remove("show");
  }

  draw(e) {
    if (!this.isDrawing) return;

    const pos = this.getMousePos(e);

    this.ctx.globalCompositeOperation = "source-over";
    this.ctx.strokeStyle = "#fff";
    this.ctx.lineWidth = this.brushSize;

    this.ctx.beginPath();
    this.ctx.moveTo(this.lastX, this.lastY);
    this.ctx.lineTo(pos.x, pos.y);
    this.ctx.stroke();

    this.lastX = pos.x;
    this.lastY = pos.y;

    // Debounced prediction
    this.debouncedPredict();
  }

  stopDrawing() {
    if (!this.isDrawing) return;
    this.isDrawing = false;

    // Final prediction
    this.predictDigit();
  }

  debouncedPredict() {
    clearTimeout(this.predictionTimeout);
    this.predictionTimeout = setTimeout(() => {
      this.predictDigit();
    }, 300);
  }

  async predictDigit() {
    try {
      // Convert canvas to image data
      const imageData = this.canvas.toDataURL("image/png");

      // Show loading state
      document.getElementById("predictedDigit").className =
        "predicted-digit loading";

      // Send prediction request
      const response = await fetch("/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ image: imageData }),
      });

      if (response.ok) {
        const result = await response.json();
        this.updatePredictionDisplay(result);
      } else {
        console.error("Prediction failed:", response.statusText);
      }
    } catch (error) {
      console.error("Prediction error:", error);
    }
  }

  updatePredictionDisplay(result) {
    const { predicted_digit, confidence, predictions, prediction_time } =
      result;

    // Update top prediction
    const predictedDigitEl = document.getElementById("predictedDigit");
    const confidenceEl = document.getElementById("confidence");
    const predictionTimeEl = document.getElementById("predictionTime");

    predictedDigitEl.textContent = predicted_digit;
    predictedDigitEl.className = "predicted-digit fade-in";

    // Color based on confidence
    if (confidence > 80) {
      predictedDigitEl.style.color = "#4caf50";
    } else if (confidence > 60) {
      predictedDigitEl.style.color = "#ff9800";
    } else {
      predictedDigitEl.style.color = "#f44336";
    }

    confidenceEl.textContent = `Confidence: ${confidence.toFixed(1)}%`;
    predictionTimeEl.textContent = `${prediction_time.toFixed(1)}ms`;

    // Update confidence bars
    predictions.forEach((prob, digit) => {
      const barFill = document.getElementById(`bar-${digit}`);
      const percentage = document.getElementById(`percent-${digit}`);

      const confidence = prob * 100;
      barFill.style.width = `${confidence}%`;
      percentage.textContent = `${confidence.toFixed(1)}%`;

      // Highlight top prediction
      if (digit === predicted_digit) {
        barFill.classList.add("top-prediction");
      } else {
        barFill.classList.remove("top-prediction");
      }
    });

    // Add fade-in animation
    document.querySelector(".confidence-bars").classList.add("fade-in");
    setTimeout(() => {
      document.querySelector(".confidence-bars").classList.remove("fade-in");
    }, 500);
  }

  clearCanvas() {
    this.ctx.fillStyle = "#000";
    this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

    // Reset prediction display
    document.getElementById("predictedDigit").textContent = "?";
    document.getElementById("predictedDigit").style.color = "#ff6b6b";
    document.getElementById("confidence").textContent = "Confidence: --%";
    document.getElementById("predictionTime").textContent = "--ms";

    // Reset confidence bars
    for (let digit = 0; digit < 10; digit++) {
      document.getElementById(`bar-${digit}`).style.width = "0%";
      document.getElementById(`percent-${digit}`).textContent = "0%";
      document
        .getElementById(`bar-${digit}`)
        .classList.remove("top-prediction");
    }

    // Show instructions overlay
    document.querySelector(".canvas-overlay").classList.add("show");

    console.log("Canvas cleared");
  }
}

// Initialize the app when DOM is loaded
document.addEventListener("DOMContentLoaded", () => {
  new DigitRecognizer();
});
