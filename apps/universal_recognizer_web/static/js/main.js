// NeuralEngine Universal Character Recognizer - Main JavaScript
class UniversalCharacterRecognizer {
  constructor() {
    this.canvas = document.getElementById("drawingCanvas");
    this.ctx = this.canvas.getContext("2d");
    this.isDrawing = false;
    // Fixed brush size - adaptive based on canvas size
    this.brushSize = Math.max(12, Math.min(20, this.canvas.width / 20));
    this.currentTab = "all";
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
    tabButtons.forEach((btn) => {
      btn.addEventListener("click", () => {
        tabButtons.forEach((b) => b.classList.remove("active"));
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
    const imageData = this.ctx.getImageData(
      0,
      0,
      this.canvas.width,
      this.canvas.height
    );
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
      // Check if debug mode is enabled
      const debugEnabled =
        document.getElementById("debugPanelContent")?.style.display !== "none";

      // Use accessibility endpoint for full analysis
      const url = debugEnabled
        ? "/predict?debug=true"
        : "/predict/accessibility";
      const response = await fetch(url, {
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

      // Display debug images if available
      if (result.debug_images) {
        this.displayDebugImages(result.debug_images, result.debug_images.stats);
      }

      if (window.updateAccessibilityDisplay) {
        window.updateAccessibilityDisplay(result);
      }
    } catch (error) {
      console.error("Prediction error:", error);
      alert("Failed to get prediction. Please try again.");
    }
  }

  displayPrediction(result) {
    // Handle both /predict and /predict/accessibility response formats
    const prediction = result.prediction || result;
    const predictionTime = result.prediction_time || 0;

    // Update main prediction display
    document.getElementById("predictedCharacter").textContent =
      prediction.predicted_character;
    document.getElementById("characterType").textContent =
      prediction.character_type;
    document.getElementById(
      "confidence"
    ).textContent = `Confidence: ${prediction.confidence.toFixed(1)}%`;
    document.getElementById(
      "predictionTime"
    ).textContent = `${predictionTime.toFixed(0)}ms`;

    // Update predictions list
    this.updatePredictionsDisplay();
  }

  updatePredictionsDisplay() {
    const container = document.getElementById("topPredictions");
    if (!this.lastPrediction) {
      container.innerHTML =
        '<p class="no-prediction">Draw a character to see predictions</p>';
      return;
    }

    // Handle both /predict and /predict/accessibility response formats
    const prediction = this.lastPrediction.prediction || this.lastPrediction;
    if (!prediction) {
      container.innerHTML =
        '<p class="no-prediction">Draw a character to see predictions</p>';
      return;
    }

    const topPredictions = prediction.top_predictions || [];
    const filtered = this.filterByTab(topPredictions);

    if (filtered.length === 0) {
      container.innerHTML =
        '<p class="no-prediction">No predictions in this category</p>';
      return;
    }

    container.innerHTML = filtered
      .map(
        (pred, index) => `
      <div class="prediction-item">
        <div class="prediction-char">${pred.character}</div>
        <div class="prediction-type">${pred.type}</div>
        <div class="prediction-bar-container">
          <div class="prediction-bar-fill" style="width: ${
            pred.confidence
          }%"></div>
        </div>
        <div class="prediction-percentage">${pred.confidence.toFixed(1)}%</div>
      </div>
    `
      )
      .join("");
  }

  filterByTab(predictions) {
    if (this.currentTab === "all") return predictions;

    const typeMap = {
      digits: "Digit",
      uppercase: "Uppercase",
      lowercase: "Lowercase",
    };

    const targetType = typeMap[this.currentTab];
    return predictions.filter((p) => p.type === targetType);
  }

  displayDebugImages(debugImages, stats) {
    const container = document.getElementById("debugImagesContainer");
    if (!container) return;

    const steps = [
      { key: "original", label: "Original" },
      { key: "flipped_upside_down", label: "Flipped Upside Down" },
      { key: "after_resize", label: "After Resize" },
      { key: "final", label: "Final (to Model)" },
    ];

    let html = '<div class="debug-images-grid">';
    steps.forEach((step) => {
      if (debugImages[step.key]) {
        html += `
          <div class="debug-image-item">
            <div class="debug-image-label">${step.label}</div>
            <img src="${debugImages[step.key]}" alt="${
          step.label
        }" class="debug-image" />
          </div>
        `;
      }
    });
    html += "</div>";
    container.innerHTML = html;

    // Display statistics
    const statsContainer = document.getElementById("debugStatsContainer");
    if (statsContainer && stats) {
      statsContainer.innerHTML = `
        <div class="debug-stats-grid">
          <div class="debug-stat-item">
            <span class="debug-stat-label">Original Range:</span>
            <span class="debug-stat-value">[${stats.original_min.toFixed(
              3
            )}, ${stats.original_max.toFixed(3)}]</span>
          </div>
          <div class="debug-stat-item">
            <span class="debug-stat-label">Original Mean:</span>
            <span class="debug-stat-value">${stats.original_mean.toFixed(
              3
            )}</span>
          </div>
          <div class="debug-stat-item">
            <span class="debug-stat-label">Final Range:</span>
            <span class="debug-stat-value">[${stats.final_min.toFixed(
              3
            )}, ${stats.final_max.toFixed(3)}]</span>
          </div>
          <div class="debug-stat-item">
            <span class="debug-stat-label">Final Mean:</span>
            <span class="debug-stat-value">${stats.final_mean.toFixed(3)}</span>
          </div>
          <div class="debug-stat-item">
            <span class="debug-stat-label">Final Std:</span>
            <span class="debug-stat-value">${stats.final_std.toFixed(3)}</span>
          </div>
        </div>
      `;
    }
  }
}

// Test Mode Handler
class TestModeHandler {
  constructor() {
    this.currentView = "single";
    this.currentSamples = [];
    this.setupEventListeners();
  }

  setupEventListeners() {
    // View toggle
    document.getElementById("singleViewBtn").addEventListener("click", (e) => {
      e.preventDefault();
      e.stopPropagation();
      this.switchView("single");
    });
    document.getElementById("gridViewBtn").addEventListener("click", (e) => {
      e.preventDefault();
      e.stopPropagation();
      this.switchView("grid");
    });

    // Load samples button
    document
      .getElementById("loadTestSamples")
      .addEventListener("click", (e) => {
        e.preventDefault();
        e.stopPropagation();
        this.loadSamples();
      });

    // Character filter change
    const characterFilter = document.getElementById("characterFilter");
    if (characterFilter) {
      characterFilter.addEventListener("change", (e) => {
        e.preventDefault();
        e.stopPropagation();
        // Optionally reload samples when filter changes
        if (this.currentSamples.length > 0) {
          this.loadSamples();
        }
      });
    }
  }

  switchView(view) {
    this.currentView = view;

    // Update buttons
    document.querySelectorAll(".view-btn").forEach((btn) => {
      btn.classList.remove("active");
    });
    document.getElementById(`${view}ViewBtn`).classList.add("active");

    // Update views
    document.getElementById("singleTestView").style.display =
      view === "single" ? "block" : "none";
    document.getElementById("gridTestView").style.display =
      view === "grid" ? "block" : "none";

    // If grid view and we have samples, display them
    if (view === "grid" && this.currentSamples.length > 0) {
      this.displayGrid();
    } else if (view === "single" && this.currentSamples.length > 0) {
      this.displaySingle(this.currentSamples[0]);
    }
  }

  async loadSamples() {
    // Store current scroll position to preserve it
    const scrollPosition =
      window.pageYOffset || document.documentElement.scrollTop;

    // Prevent any default behaviors
    const view = this.currentView;
    const count = view === "single" ? 1 : 9; // 1 for single, 9 for grid
    const character = document.getElementById("characterFilter").value;

    // Disable button during loading
    const loadBtn = document.getElementById("loadTestSamples");
    const originalBtnText = loadBtn.innerHTML;
    loadBtn.disabled = true;
    loadBtn.innerHTML = "⏳ Loading...";

    try {
      const url = `/api/test/random?count=${count}${
        character ? `&character=${character}` : ""
      }`;
      const response = await fetch(url, {
        method: "GET",
        headers: {
          "Content-Type": "application/json",
        },
        cache: "no-cache",
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      this.currentSamples = data.samples;

      if (view === "single") {
        await this.displaySingle(this.currentSamples[0]);
      } else {
        await this.displayGrid();
      }

      // Restore scroll position after content updates
      requestAnimationFrame(() => {
        window.scrollTo(0, scrollPosition);
      });
    } catch (error) {
      console.error("Error loading test samples:", error);
      // Use a non-blocking notification instead of alert
      this.showNotification(
        "Failed to load test samples. Please try again.",
        "error"
      );
      // Restore scroll position even on error
      requestAnimationFrame(() => {
        window.scrollTo(0, scrollPosition);
      });
    } finally {
      // Re-enable button
      loadBtn.disabled = false;
      loadBtn.innerHTML = originalBtnText;
    }
  }

  showNotification(message, type = "info") {
    // Create a temporary notification element
    const notification = document.createElement("div");
    notification.className = `test-notification test-notification-${type}`;
    notification.textContent = message;
    notification.style.cssText = `
      position: fixed;
      top: 20px;
      right: 20px;
      padding: 12px 24px;
      background: ${type === "error" ? "#ff3b30" : "#007aff"};
      color: white;
      border-radius: 8px;
      box-shadow: 0 4px 12px rgba(0,0,0,0.3);
      z-index: 10000;
      animation: slideIn 0.3s ease;
    `;

    document.body.appendChild(notification);

    // Remove after 3 seconds
    setTimeout(() => {
      notification.style.animation = "slideOut 0.3s ease";
      setTimeout(() => {
        if (notification.parentNode) {
          notification.parentNode.removeChild(notification);
        }
      }, 300);
    }, 3000);
  }

  // Helper function to rotate image 180 degrees
  rotateImage180(imageDataUrl) {
    return new Promise((resolve) => {
      const img = new Image();
      img.onload = () => {
        const canvas = document.createElement("canvas");
        const ctx = canvas.getContext("2d");
        canvas.width = img.width;
        canvas.height = img.height;

        // Translate to center, rotate 180 degrees, translate back
        ctx.translate(canvas.width / 2, canvas.height / 2);
        ctx.rotate(Math.PI); // 180 degrees
        ctx.translate(-canvas.width / 2, -canvas.height / 2);

        // Draw the image
        ctx.drawImage(img, 0, 0);

        // Convert back to data URL
        resolve(canvas.toDataURL("image/png"));
      };
      img.src = imageDataUrl;
    });
  }

  async displaySingle(sample) {
    // Store scroll position to preserve it
    const scrollPosition =
      window.pageYOffset || document.documentElement.scrollTop;

    // Hide placeholder and show content
    const placeholder = document.getElementById("singleTestPlaceholder");
    const content = document.getElementById("singleTestSample");
    if (placeholder) placeholder.style.display = "none";
    if (content) content.style.display = "grid";

    // Rotate image 180 degrees (test images are upside down)
    const rotatedImage = await this.rotateImage180(sample.image_data);
    document.getElementById("testImageSingle").src = rotatedImage;

    // Display ground truth
    document.getElementById("groundTruthSingle").textContent =
      sample.ground_truth;

    // Get prediction
    try {
      const response = await fetch("/api/test/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ image_array: sample.image_array }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      const prediction = data.prediction;

      document.getElementById("predictionSingle").textContent =
        prediction.character;
      document.getElementById(
        "confidenceSingle"
      ).textContent = `${prediction.confidence.toFixed(1)}%`;

      // Status badge
      const isCorrect = prediction.character === sample.ground_truth;
      const statusBadge = document.getElementById("testStatusSingle");
      statusBadge.innerHTML = `<span class="status-badge ${
        isCorrect ? "correct" : "incorrect"
      }">${isCorrect ? "✓ Correct" : "✗ Incorrect"}</span>`;

      // Restore scroll position after DOM updates
      requestAnimationFrame(() => {
        window.scrollTo(0, scrollPosition);
      });
    } catch (error) {
      console.error("Error getting prediction:", error);
      document.getElementById("predictionSingle").textContent = "Error";
      // Restore scroll position even on error
      requestAnimationFrame(() => {
        window.scrollTo(0, scrollPosition);
      });
    }
  }

  async displayGrid() {
    // Store scroll position to preserve it
    const scrollPosition =
      window.pageYOffset || document.documentElement.scrollTop;

    const grid = document.getElementById("testGrid");
    const placeholder = document.getElementById("gridTestPlaceholder");
    if (placeholder) placeholder.style.display = "none";
    grid.innerHTML =
      '<div class="test-grid-loading">Loading predictions...</div>';

    // Get predictions for all samples
    const predictions = await Promise.all(
      this.currentSamples.map(async (sample) => {
        try {
          const response = await fetch("/api/test/predict", {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify({ image_array: sample.image_array }),
          });

          if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
          }

          const data = await response.json();
          return {
            sample,
            prediction: data.prediction,
          };
        } catch (error) {
          console.error("Error getting prediction:", error);
          return {
            sample,
            prediction: null,
          };
        }
      })
    );

    // Rotate all images 180 degrees (test images are upside down)
    const rotatedImages = await Promise.all(
      predictions.map(({ sample }) => this.rotateImage180(sample.image_data))
    );

    // Display grid
    grid.innerHTML = predictions
      .map(({ sample, prediction }, index) => {
        if (!prediction) {
          return '<div class="test-grid-item error">Error</div>';
        }

        const isCorrect = prediction.character === sample.ground_truth;

        return `
        <div class="test-grid-item ${isCorrect ? "correct" : "incorrect"}">
          <img src="${rotatedImages[index]}" alt="Test sample" />
          <div class="test-grid-info">
            <div class="test-grid-label">Truth: <strong>${
              sample.ground_truth
            }</strong></div>
            <div class="test-grid-label">Pred: <strong>${
              prediction.character
            }</strong></div>
            <div class="test-grid-label">Conf: ${prediction.confidence.toFixed(
              1
            )}%</div>
            <div class="test-grid-status">${isCorrect ? "✓" : "✗"}</div>
          </div>
        </div>
      `;
      })
      .join("");

    // Restore scroll position after DOM updates
    requestAnimationFrame(() => {
      window.scrollTo(0, scrollPosition);
    });
  }
}

// Theme Toggle Handler
class ThemeHandler {
  constructor() {
    this.currentTheme = localStorage.getItem("theme") || "dark";
    this.init();
  }

  init() {
    document.documentElement.setAttribute("data-theme", this.currentTheme);
    this.updateIcon();

    document.getElementById("themeToggle").addEventListener("click", () => {
      this.toggle();
    });
  }

  toggle() {
    this.currentTheme = this.currentTheme === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", this.currentTheme);
    localStorage.setItem("theme", this.currentTheme);
    this.updateIcon();
  }

  updateIcon() {
    const icon = document.getElementById("themeIcon");
    icon.textContent = this.currentTheme === "dark" ? "☀️" : "🌙";
  }
}

// Mode Toggle Handler
class ModeHandler {
  constructor() {
    this.currentMode = "draw";
    this.init();
  }

  init() {
    document.getElementById("drawModeBtn").addEventListener("click", () => {
      this.switchMode("draw");
    });

    document.getElementById("testModeBtn").addEventListener("click", () => {
      this.switchMode("test");
    });
  }

  switchMode(mode) {
    this.currentMode = mode;

    // Update buttons
    document.querySelectorAll(".mode-btn").forEach((btn) => {
      btn.classList.remove("active");
    });
    document.getElementById(`${mode}ModeBtn`).classList.add("active");

    // Update content
    document.querySelectorAll(".mode-content").forEach((content) => {
      content.classList.remove("active");
    });
    document.getElementById(`${mode}ModeContent`).classList.add("active");
  }
}

// Advanced Metrics Toggle
function setupAdvancedMetricsToggle() {
  const toggleBtn = document.getElementById("toggleAdvancedMetrics");
  const content = document.getElementById("advancedMetricsContent");
  const icon = toggleBtn.querySelector(".toggle-icon");

  toggleBtn.addEventListener("click", () => {
    const isHidden = content.style.display === "none";
    content.style.display = isHidden ? "block" : "none";
    icon.textContent = isHidden ? "▲" : "▼";
  });
}

// Debug Panel Toggle
function setupDebugPanelToggle() {
  const toggleBtn = document.getElementById("toggleDebugPanel");
  const content = document.getElementById("debugPanelContent");
  const icon = toggleBtn.querySelector(".toggle-icon");

  toggleBtn.addEventListener("click", () => {
    const isHidden = content.style.display === "none";
    content.style.display = isHidden ? "block" : "none";
    icon.textContent = isHidden ? "▲" : "▼";
  });
}

// Initialize app when DOM is ready
document.addEventListener("DOMContentLoaded", () => {
  window.recognizer = new UniversalCharacterRecognizer();
  window.testMode = new TestModeHandler();
  window.themeHandler = new ThemeHandler();
  window.modeHandler = new ModeHandler();
  setupAdvancedMetricsToggle();
  setupDebugPanelToggle();
});
