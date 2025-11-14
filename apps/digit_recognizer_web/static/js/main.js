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
    this.setupModelSelector();
    this.setupKeyboardShortcuts();
    this.setupTutorial();
    this.setupPredictionHistory();
    this.lastPredictionTime = 0;
    this.sequenceTracker = []; // Track digit sequence for easter egg
    this.targetSequence = [3, 1, 4]; // Pi digits sequence
    this.predictionHistory = [];
    this.historyMaxSize = 10;
    this.predictionCache = new Map(); // Cache for identical drawings

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
    
    // Ensure cursor stays as crosshair
    this.canvas.style.cursor = "crosshair";
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

    // Ensure cursor stays as crosshair while drawing
    this.canvas.style.cursor = "crosshair";

    // Debounced prediction
    this.debouncedPredict();
  }

  stopDrawing() {
    if (!this.isDrawing) return;
    this.isDrawing = false;

    // Ensure cursor stays as crosshair
    this.canvas.style.cursor = "crosshair";

    // Final prediction
    this.predictDigit();
  }

  debouncedPredict() {
    clearTimeout(this.predictionTimeout);
    this.predictionTimeout = setTimeout(() => {
      this.predictDigit();
    }, 300);
  }


  updatePredictionDisplay(result) {
    const { predicted_digit, confidence, predictions, prediction_time } =
      result;

    // Update top prediction
    const predictedDigitEl = document.getElementById("predictedDigit");
    const confidenceEl = document.getElementById("confidence");
    const predictionTimeEl = document.getElementById("predictionTime");

    // Add reveal animation
    predictedDigitEl.textContent = predicted_digit;
    predictedDigitEl.classList.add("reveal");
    
    // Remove reveal class after animation
    setTimeout(() => {
      predictedDigitEl.classList.remove("reveal");
    }, 400);

    // Color based on confidence - Modern color palette
    if (confidence > 80) {
      predictedDigitEl.style.color = "#66bb6a";
      predictedDigitEl.classList.add("prediction-success");
      // Only create particles for very high confidence (>95%)
      if (confidence > 95) {
        this.createParticles(predictedDigitEl);
      }
    } else if (confidence > 60) {
      predictedDigitEl.style.color = "#ffb74d";
      predictedDigitEl.classList.remove("prediction-success");
    } else {
      predictedDigitEl.style.color = "#ef5350";
      predictedDigitEl.classList.remove("prediction-success");
    }

    confidenceEl.textContent = `Confidence: ${confidence.toFixed(1)}%`;
    predictionTimeEl.textContent = `${prediction_time.toFixed(1)}ms`;

    // Add to prediction history
    this.addToHistory(predicted_digit, confidence);

    // Easter egg: Check for Pi sequence (3-1-4)
    this.checkPiSequence(predicted_digit);

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
    // Clear canvas immediately without animation to prevent flickering
    this.ctx.fillStyle = "#000";
    this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
    
    // Ensure cursor stays as crosshair
    this.canvas.style.cursor = "crosshair";

    // Reset prediction display
    const predictedDigitEl = document.getElementById("predictedDigit");
    predictedDigitEl.textContent = "?";
    predictedDigitEl.style.color = "#4fc3f7";
    predictedDigitEl.classList.remove("reveal", "prediction-success");
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

  checkPiSequence(digit) {
    console.log(`🔍 Checking digit: ${digit} (type: ${typeof digit})`);

    // Only add the digit if it's different from the last one in our sequence
    // This prevents duplicates from multiple prediction calls
    if (
      this.sequenceTracker.length === 0 ||
      this.sequenceTracker[this.sequenceTracker.length - 1] !== digit
    ) {
      this.sequenceTracker.push(digit);
      console.log(`✅ Added digit ${digit} to sequence`);
    } else {
      console.log(`⏭️ Skipping duplicate digit ${digit}`);
    }

    console.log(`📝 Current sequence: [${this.sequenceTracker.join(", ")}]`);

    // Keep only the last 3 digits
    if (this.sequenceTracker.length > 3) {
      this.sequenceTracker.shift();
    }

    console.log(`📝 After trimming: [${this.sequenceTracker.join(", ")}]`);

    // Check if we have the Pi sequence (3-1-4)
    if (
      this.sequenceTracker.length === 3 &&
      this.sequenceTracker[0] === 3 &&
      this.sequenceTracker[1] === 1 &&
      this.sequenceTracker[2] === 4
    ) {
      console.log("🎉 Pi sequence detected! Triggering easter egg...");
      this.triggerPiAnimation();
      this.sequenceTracker = []; // Reset sequence after triggering
    } else {
      console.log(
        `❌ Not Pi sequence. Need [3, 1, 4], got [${this.sequenceTracker.join(
          ", "
        )}]`
      );
    }
  }

  triggerPiAnimation() {
    // Show the Pi symbol
    const piSymbol = document.getElementById("piSymbol");
    piSymbol.classList.add("show");

    // Generate confetti
    this.generateConfetti();

    // Hide the Pi symbol after animation
    setTimeout(() => {
      piSymbol.classList.remove("show");
    }, 4000);
  }

  generateConfetti() {
    const confettiContainer = document.getElementById("confettiContainer");

    // Create multiple confetti pieces
    for (let i = 0; i < 50; i++) {
      const confetti = document.createElement("div");
      confetti.className = "confetti";
      confetti.style.left = Math.random() * 100 + "%";
      confetti.style.backgroundColor = this.getRandomColor();
      confetti.style.animationDelay = Math.random() * 2 + "s";
      confetti.style.animationDuration = Math.random() * 3 + 2 + "s";

      confettiContainer.appendChild(confetti);

      // Remove confetti after animation
      setTimeout(() => {
        if (confetti.parentNode) {
          confetti.parentNode.removeChild(confetti);
        }
      }, 5000);
    }
  }

  getRandomColor() {
    const colors = [
      "#ff6b6b",
      "#4ecdc4",
      "#45b7d1",
      "#f9ca24",
      "#f0932b",
      "#eb4d4b",
      "#6c5ce7",
      "#a29bfe",
    ];
    return colors[Math.floor(Math.random() * colors.length)];
  }

  setupModelSelector() {
    const modelSelect = document.getElementById("modelSelect");
    const modelSelector = document.querySelector(".model-selector");

    modelSelect.addEventListener("change", async (e) => {
      const selectedModel = e.target.value;
      console.log(`🔄 Switching to model: ${selectedModel}`);

      // Show loading state
      modelSelector.classList.add("loading");

      try {
        // Send model switch request
        const response = await fetch("/switch_model", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ model_name: selectedModel }),
        });

        if (response.ok) {
          const result = await response.json();
          console.log("✅ Model switched successfully:", result);

          // Update model info display
          this.updateModelInfo(result.model_info);

          // Clear canvas and reset predictions
          this.clearCanvas();

          // Reset sequence tracker
          this.sequenceTracker = [];
        } else {
          console.error("❌ Model switch failed:", response.statusText);
          alert("Failed to switch model. Please try again.");
        }
      } catch (error) {
        console.error("❌ Model switch error:", error);
        alert("Error switching model. Please try again.");
      } finally {
        // Remove loading state
        modelSelector.classList.remove("loading");
      }
    });
  }

  updateModelInfo(modelInfo) {
    // Update the model info display
    const elements = {
      architecture: document.querySelector(".stat .value"),
      parameters: document.querySelectorAll(".stat .value")[1],
      accuracy: document.querySelectorAll(".stat .value")[2],
    };

    if (elements.architecture) {
      elements.architecture.textContent = modelInfo.architecture.join(" → ");
    }
    if (elements.parameters) {
      elements.parameters.textContent = modelInfo.parameters.toLocaleString();
    }
    if (elements.accuracy) {
      elements.accuracy.textContent = `${modelInfo.accuracy.toFixed(2)}%`;
    }
  }

  setupKeyboardShortcuts() {
    document.addEventListener("keydown", (e) => {
      // C key to clear canvas
      if (e.key === "c" || e.key === "C") {
        if (!e.target.matches("input, textarea, select")) {
          e.preventDefault();
          this.clearCanvas();
        }
      }
      
      // Number keys 0-9 to quick-select (show hint)
      if (e.key >= "0" && e.key <= "9" && !e.target.matches("input, textarea, select")) {
        const digit = parseInt(e.key);
        this.showQuickSelectHint(digit);
      }
      
      // Escape to close tutorial
      if (e.key === "Escape") {
        this.hideTutorial();
      }
    });
  }

  showQuickSelectHint(digit) {
    const predictedDigitEl = document.getElementById("predictedDigit");
    const originalText = predictedDigitEl.textContent;
    predictedDigitEl.textContent = digit;
    predictedDigitEl.style.color = "#66bb6a";
    predictedDigitEl.classList.add("prediction-success");
    
    setTimeout(() => {
      predictedDigitEl.textContent = originalText;
      predictedDigitEl.classList.remove("prediction-success");
      predictedDigitEl.style.color = "#4fc3f7";
    }, 400);
  }

  setupTutorial() {
    // Check if user has seen tutorial
    const hasSeenTutorial = localStorage.getItem("digit_recognizer_tutorial_seen");
    
    if (!hasSeenTutorial) {
      // Create tutorial overlay
      const tutorialOverlay = document.createElement("div");
      tutorialOverlay.className = "tutorial-overlay show";
      tutorialOverlay.innerHTML = `
        <div class="tutorial-content">
          <h2>Welcome to NeuralEngine! 🧠</h2>
          <p>Draw digits (0-9) on the canvas to see predictions</p>
          <ul style="text-align: left; margin: 20px 0; list-style: none; padding: 0;">
            <li>✏️ Draw on the canvas</li>
            <li>⌨️ Press <strong>C</strong> to clear</li>
            <li>⌨️ Press <strong>0-9</strong> for quick hints</li>
            <li>🎯 View confidence levels below</li>
          </ul>
          <button class="btn btn-secondary" onclick="digitRecognizer.hideTutorial()">
            Got it!
          </button>
        </div>
      `;
      document.body.appendChild(tutorialOverlay);
      this.tutorialOverlay = tutorialOverlay;
    }
  }

  hideTutorial() {
    if (this.tutorialOverlay) {
      this.tutorialOverlay.classList.remove("show");
      localStorage.setItem("digit_recognizer_tutorial_seen", "true");
      setTimeout(() => {
        if (this.tutorialOverlay.parentNode) {
          this.tutorialOverlay.parentNode.removeChild(this.tutorialOverlay);
        }
      }, 300);
    }
  }

  setupPredictionHistory() {
    // Create history container
    const historyContainer = document.createElement("div");
    historyContainer.className = "prediction-history";
    historyContainer.innerHTML = `
      <div style="font-weight: 600; margin-bottom: 12px; color: white; font-size: 0.95rem; padding-bottom: 8px; border-bottom: 1px solid rgba(255, 255, 255, 0.1);">Recent Predictions</div>
      <div id="historyItems"></div>
    `;
    document.body.appendChild(historyContainer);
    this.historyContainer = historyContainer;
  }

  addToHistory(digit, confidence) {
    // Check if this is a duplicate of the most recent prediction
    // Only add if it's different from the last one (different digit or significantly different confidence)
    const lastPrediction = this.predictionHistory[0];
    const isDuplicate = lastPrediction && 
      lastPrediction.digit === digit && 
      Math.abs(lastPrediction.confidence - confidence) < 0.1; // Same digit and very similar confidence
    
    if (isDuplicate) {
      // Don't add duplicate - just update the timestamp
      return;
    }
    
    this.predictionHistory.unshift({ digit, confidence, timestamp: Date.now() });
    
    // Keep only last N items
    if (this.predictionHistory.length > this.historyMaxSize) {
      this.predictionHistory.pop();
    }
    
    this.updateHistoryDisplay();
  }

  updateHistoryDisplay() {
    const historyItems = document.getElementById("historyItems");
    if (!historyItems) return;
    
    historyItems.innerHTML = this.predictionHistory.map((item, index) => {
      const confidenceColor = item.confidence > 80 ? "#66bb6a" : item.confidence > 60 ? "#ffb74d" : "#ef5350";
      return `
      <div class="history-item" style="animation-delay: ${index * 0.05}s;">
        <span style="font-size: 1.3rem; font-weight: 700; color: ${confidenceColor};">${item.digit}</span>
        <span style="opacity: 0.8; font-size: 0.85rem;">${item.confidence.toFixed(1)}%</span>
      </div>
    `;
    }).join("");
    
    // Show history if it has items
    if (this.predictionHistory.length > 0) {
      this.historyContainer.classList.add("show");
    } else {
      this.historyContainer.classList.remove("show");
    }
  }

  createParticles(element) {
    const rect = element.getBoundingClientRect();
    const centerX = rect.left + rect.width / 2;
    const centerY = rect.top + rect.height / 2;
    
    // Create particles container if it doesn't exist
    let particlesContainer = document.querySelector(".particles-container");
    if (!particlesContainer) {
      particlesContainer = document.createElement("div");
      particlesContainer.className = "particles-container";
      document.body.appendChild(particlesContainer);
    }
    
    // Create 20 particles
    for (let i = 0; i < 20; i++) {
      const particle = document.createElement("div");
      particle.className = "particle";
      particle.style.left = centerX + "px";
      particle.style.top = centerY + "px";
      particle.style.backgroundColor = this.getRandomColor();
      particle.style.animationDelay = (Math.random() * 0.5) + "s";
      particle.style.transform = `translate(${(Math.random() - 0.5) * 100}px, ${(Math.random() - 0.5) * 100}px)`;
      
      particlesContainer.appendChild(particle);
      
      setTimeout(() => {
        if (particle.parentNode) {
          particle.parentNode.removeChild(particle);
        }
      }, 3000);
    }
  }

  async predictDigit() {
    try {
      // Convert canvas to image data
      const imageData = this.canvas.toDataURL("image/png");
      
      // Check cache
      const cacheKey = imageData.substring(0, 100); // Use first 100 chars as key
      if (this.predictionCache.has(cacheKey)) {
        const cached = this.predictionCache.get(cacheKey);
        this.updatePredictionDisplay(cached);
        return;
      }

      // Show loading state
      const predictedDigitEl = document.getElementById("predictedDigit");
      predictedDigitEl.className = "predicted-digit loading";

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
        
        // Cache result
        this.predictionCache.set(cacheKey, result);
        if (this.predictionCache.size > 50) {
          // Limit cache size
          const firstKey = this.predictionCache.keys().next().value;
          this.predictionCache.delete(firstKey);
        }
        
        this.updatePredictionDisplay(result);
      } else {
        console.error("Prediction failed:", response.statusText);
      }
    } catch (error) {
      console.error("Prediction error:", error);
    }
  }
}

// Initialize the app when DOM is loaded
let digitRecognizer;
document.addEventListener("DOMContentLoaded", () => {
  digitRecognizer = new DigitRecognizer();
});
