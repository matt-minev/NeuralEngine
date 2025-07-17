/**
 * NeuralEngine Dataset Showcase - Enhanced Interactive JavaScript
 * Apple-inspired neural network visualization with data source toggle
 */

class DatasetShowcase {
  constructor() {
    this.canvas = document.getElementById("sampleCanvas");
    this.ctx = this.canvas.getContext("2d");
    this.networkSvg = document.getElementById("networkSvg");
    this.currentSample = null;
    this.currentPrediction = null;
    this.isAnimating = false;
    this.animationStep = 0;
    this.sampleCount = 0;
    this.modelAccuracy = 0;
    this.layerNodes = [];
    this.connections = [];
    this.animationId = null;
    this.useSyntheticData = false; // Toggle state

    // Network architecture (matching your real model)
    this.networkArchitecture = {
      inputSize: 784,
      hiddenLayers: [512, 256, 128],
      outputSize: 10,
    };

    this.initializeShowcase();
    this.setupEventListeners();
    this.loadFirstSample();
    this.setupNetworkVisualization();

    console.log("🎨 Enhanced Dataset Showcase initialized");
  }

  initializeShowcase() {
    // Set up canvas for high DPI displays
    const rect = this.canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width = rect.width * dpr;
    this.canvas.height = rect.height * dpr;
    this.ctx.scale(dpr, dpr);

    // Initialize prediction display
    this.updatePredictionDisplay("-", 0, "-");

    // Load model info
    this.loadModelInfo();

    // Set up toggle initial state
    this.updateToggleState();
  }

  setupEventListeners() {
    // Next sample button
    document.getElementById("nextSampleBtn").addEventListener("click", () => {
      this.loadNextSample();
    });

    // Image click handler - NEW
    this.canvas.addEventListener("click", () => {
      this.loadNextSample();
    });

    // Data source toggle - NEW
    document
      .getElementById("dataSourceToggle")
      .addEventListener("change", (e) => {
        this.useSyntheticData = e.target.checked;
        this.updateToggleState();
        this.loadNextSample(); // Load new sample with new data source
      });

    // Animation controls
    document
      .getElementById("playAnimationBtn")
      .addEventListener("click", () => {
        this.playAnimation();
      });

    document.getElementById("stepThroughBtn").addEventListener("click", () => {
      this.stepThroughAnimation();
    });

    // Back to drawing mode
    document.getElementById("backToDrawBtn").addEventListener("click", () => {
      window.location.href = "/";
    });

    // Keyboard shortcuts
    document.addEventListener("keydown", (e) => {
      if (e.key === " ") {
        e.preventDefault();
        this.loadNextSample();
      } else if (e.key === "Enter") {
        e.preventDefault();
        this.playAnimation();
      } else if (e.key === "ArrowRight") {
        e.preventDefault();
        this.stepThroughAnimation();
      } else if (e.key === "t" || e.key === "T") {
        e.preventDefault();
        // Toggle data source with T key
        const toggle = document.getElementById("dataSourceToggle");
        toggle.checked = !toggle.checked;
        toggle.dispatchEvent(new Event("change"));
      }
    });

    // Resize handler
    window.addEventListener("resize", () => {
      this.debounce(() => {
        this.setupNetworkVisualization();
      }, 250);
    });
  }

  updateToggleState() {
    const toggle = document.getElementById("dataSourceToggle");
    const toggleTexts = document.querySelectorAll(".toggle-text");

    if (this.useSyntheticData) {
      toggleTexts[0].style.color = "var(--text-secondary)";
      toggleTexts[1].style.color = "var(--text-primary)";
    } else {
      toggleTexts[0].style.color = "var(--text-primary)";
      toggleTexts[1].style.color = "var(--text-secondary)";
    }

    this.announceChange(
      `Switched to ${this.useSyntheticData ? "synthetic" : "real"} data`
    );
  }

  async loadModelInfo() {
    try {
      const response = await fetch("/model_info");
      const modelInfo = await response.json();
      this.modelAccuracy = modelInfo.accuracy || 0;
      this.updateModelAccuracy();
    } catch (error) {
      console.error("Failed to load model info:", error);
    }
  }

  updateModelAccuracy() {
    const accuracyElement = document.getElementById("modelAccuracy");
    accuracyElement.textContent = `${this.modelAccuracy.toFixed(1)}%`;
  }

  async loadFirstSample() {
    await this.loadSampleFromDataset();
  }

  async loadNextSample() {
    this.showLoading();
    await this.loadSampleFromDataset();
    this.hideLoading();
    this.resetAnimation();
  }

  async loadSampleFromDataset() {
    try {
      if (this.useSyntheticData) {
        // Use synthetic data generation
        const sampleData = await this.generateSampleData();
        this.currentSample = sampleData;
        this.sampleCount++;

        // Update sample counter
        document.getElementById("sampleCounter").textContent = this.sampleCount;

        // Display the sample
        this.displaySample(sampleData);

        // Make synthetic prediction
        await this.makeSyntheticPrediction(sampleData);

        // Announce to screen readers
        this.announceChange(
          `Loaded synthetic sample ${this.sampleCount}, predicted digit ${this.currentPrediction.predicted_digit}`
        );
      } else {
        // Call the actual backend API for real data
        const response = await fetch("/api/dataset/sample");
        const data = await response.json();

        if (data.error) {
          throw new Error(data.error);
        }

        this.currentSample = data.sample;
        this.currentPrediction = data.prediction;
        this.sampleCount++;

        // Update sample counter
        document.getElementById("sampleCounter").textContent = this.sampleCount;

        // Display the real sample
        this.displaySampleFromAPI(data.sample);

        // Update prediction display with real results
        this.updatePredictionDisplay(
          data.prediction.predicted_digit,
          data.prediction.confidence,
          data.sample.actual_label
        );

        // Announce to screen readers
        this.announceChange(
          `Loaded real sample ${this.sampleCount}, predicted digit ${
            data.prediction.predicted_digit
          }, confidence ${data.prediction.confidence.toFixed(1)}%`
        );
      }
    } catch (error) {
      console.error("Failed to load sample:", error);
      this.showError("Failed to load sample: " + error.message);
    }
  }

  displaySampleFromAPI(sampleData) {
    const img = new Image();
    img.onload = () => {
      // Clear canvas
      this.ctx.fillStyle = "#FFFFFF";
      this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

      // Draw the sample image scaled up
      const scale =
        Math.min(this.canvas.width / 280, this.canvas.height / 280) * 0.8;
      const x = (this.canvas.width - 280 * scale) / 2;
      const y = (this.canvas.height - 280 * scale) / 2;

      this.ctx.imageSmoothingEnabled = false;
      this.ctx.drawImage(img, x, y, 280 * scale, 280 * scale);

      // Add subtle border
      this.ctx.strokeStyle = "#E0E0E0";
      this.ctx.lineWidth = 2;
      this.ctx.strokeRect(x, y, 280 * scale, 280 * scale);
    };
    img.src = sampleData.image_data;
  }

  async generateSampleData() {
    // Generate synthetic dataset sample
    const digit = Math.floor(Math.random() * 10);
    const imageData = this.generateDigitImage(digit);

    return {
      image: imageData,
      label: digit,
      index: this.sampleCount,
    };
  }

  generateDigitImage(digit) {
    // Generate a more realistic representation of a digit
    const canvas = document.createElement("canvas");
    canvas.width = 28;
    canvas.height = 28;
    const ctx = canvas.getContext("2d");

    // Fill with black background
    ctx.fillStyle = "#000000";
    ctx.fillRect(0, 0, 28, 28);

    // Draw white digit with variations
    ctx.fillStyle = "#FFFFFF";
    ctx.font = "20px Arial";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";

    // Add some noise/variation for realism
    const offsetX = (Math.random() - 0.5) * 6;
    const offsetY = (Math.random() - 0.5) * 6;
    const rotation = (Math.random() - 0.5) * 0.4;

    ctx.save();
    ctx.translate(14, 14);
    ctx.rotate(rotation);
    ctx.fillText(digit.toString(), offsetX, offsetY);
    ctx.restore();

    // Add some noise
    const imageData = ctx.getImageData(0, 0, 28, 28);
    const data = imageData.data;

    for (let i = 0; i < data.length; i += 4) {
      if (Math.random() < 0.05) {
        // 5% chance of noise
        const noise = Math.random() * 100;
        data[i] = data[i + 1] = data[i + 2] = noise;
      }
    }

    ctx.putImageData(imageData, 0, 0);

    return canvas.toDataURL();
  }

  displaySample(sampleData) {
    const img = new Image();
    img.onload = () => {
      // Clear canvas
      this.ctx.fillStyle = "#FFFFFF";
      this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

      // Draw the sample image scaled up
      const scale =
        Math.min(this.canvas.width / 28, this.canvas.height / 28) * 0.8;
      const x = (this.canvas.width - 28 * scale) / 2;
      const y = (this.canvas.height - 28 * scale) / 2;

      this.ctx.imageSmoothingEnabled = false;
      this.ctx.drawImage(img, x, y, 28 * scale, 28 * scale);

      // Add subtle border
      this.ctx.strokeStyle = "#E0E0E0";
      this.ctx.lineWidth = 2;
      this.ctx.strokeRect(x, y, 28 * scale, 28 * scale);
    };
    img.src = sampleData.image;
  }

  async makeSyntheticPrediction(sampleData) {
    try {
      const response = await this.simulatePrediction(sampleData);
      this.currentPrediction = response;

      // Update prediction display
      this.updatePredictionDisplay(
        response.predicted_digit,
        response.confidence,
        sampleData.label
      );
    } catch (error) {
      console.error("Prediction failed:", error);
      this.showError("Prediction failed");
    }
  }

  async simulatePrediction(sampleData) {
    // Simulate more realistic high-confidence predictions
    const predictions = Array.from({ length: 10 }, () => Math.random() * 0.1);

    // Add some realistic variation
    const isCorrect = Math.random() > 0.05; // 95% accuracy simulation
    const predictedDigit = isCorrect
      ? sampleData.label
      : (sampleData.label + 1) % 10;

    // Set high confidence for the predicted digit
    predictions[predictedDigit] = 0.8 + Math.random() * 0.15; // 80-95% confidence

    // Normalize predictions
    const sum = predictions.reduce((a, b) => a + b, 0);
    const normalizedPredictions = predictions.map((p) => p / sum);

    return {
      predicted_digit: predictedDigit,
      confidence: normalizedPredictions[predictedDigit] * 100,
      predictions: normalizedPredictions,
    };
  }

  updatePredictionDisplay(digit, confidence, actualLabel) {
    const digitElement = document.getElementById("predictedDigit");
    const confidenceElement = document.getElementById("confidenceValue");
    const actualElement = document.getElementById("actualValue");

    // Animate digit change with enhanced effect
    digitElement.style.transform = "scale(0.8)";
    digitElement.style.opacity = "0.5";

    setTimeout(() => {
      digitElement.textContent = digit;
      digitElement.style.transform = "scale(1)";
      digitElement.style.opacity = "1";
    }, 150);

    // Update confidence with animation
    confidenceElement.textContent = `${confidence.toFixed(1)}%`;

    // Update actual label
    actualElement.textContent = actualLabel;

    // Add correct/incorrect styling with enhanced colors
    if (digit.toString() === actualLabel.toString()) {
      digitElement.style.color = "#34C759"; // Success color
      confidenceElement.style.color = "#34C759";
    } else {
      digitElement.style.color = "#FF3B30"; // Error color
      confidenceElement.style.color = "#FF3B30";
    }
  }

  setupNetworkVisualization() {
    this.layerNodes = [];
    this.connections = [];

    // Clear existing SVG content
    this.networkSvg.innerHTML = "";

    const svgRect = this.networkSvg.getBoundingClientRect();
    const width = svgRect.width || 1000;
    const height = svgRect.height || 600;

    // Calculate layer positions
    const layers = [
      this.networkArchitecture.inputSize,
      ...this.networkArchitecture.hiddenLayers,
      this.networkArchitecture.outputSize,
    ];

    const layerSpacing = width / (layers.length + 1);

    // Create layers
    layers.forEach((nodeCount, layerIndex) => {
      const layerNodes = [];
      const x = layerSpacing * (layerIndex + 1);

      // Limit visible nodes for large layers
      const visibleNodes = Math.min(nodeCount, 12);
      const nodeStep = nodeCount > 12 ? Math.floor(nodeCount / 12) : 1;

      for (let i = 0; i < visibleNodes; i++) {
        const actualNodeIndex = i * nodeStep;
        const y = 50 + (i * (height - 100)) / (visibleNodes - 1);

        const node = this.createNetworkNode(x, y, layerIndex, actualNodeIndex);
        layerNodes.push(node);
        this.networkSvg.appendChild(node);
      }

      this.layerNodes.push(layerNodes);

      // Create connections to next layer
      if (layerIndex < layers.length - 1) {
        this.createLayerConnections(layerIndex);
      }
    });

    // Add layer labels
    this.addLayerLabels();

    // Show initial layer info
    this.showLayerInfo(0);
  }

  createNetworkNode(x, y, layerIndex, nodeIndex) {
    const node = document.createElementNS(
      "http://www.w3.org/2000/svg",
      "circle"
    );
    node.setAttribute("cx", x);
    node.setAttribute("cy", y);
    node.setAttribute("r", 8);
    node.setAttribute("class", "network-node");

    if (layerIndex === 0) {
      node.classList.add("input");
    } else if (layerIndex === this.layerNodes.length) {
      node.classList.add("output");
    }

    // Add hover effect
    node.addEventListener("mouseenter", () => {
      this.showNodeInfo(layerIndex, nodeIndex);
    });

    return node;
  }

  createLayerConnections(layerIndex) {
    const currentLayer = this.layerNodes[layerIndex];
    const nextLayer = this.layerNodes[layerIndex + 1];

    if (!currentLayer || !nextLayer) return;

    // Create subset of connections to avoid clutter
    const maxConnections = 30;
    const connectionStep = Math.max(
      1,
      Math.floor((currentLayer.length * nextLayer.length) / maxConnections)
    );

    currentLayer.forEach((startNode, startIndex) => {
      nextLayer.forEach((endNode, endIndex) => {
        if ((startIndex * nextLayer.length + endIndex) % connectionStep === 0) {
          const connection = this.createConnection(startNode, endNode);
          this.connections.push(connection);
          this.networkSvg.insertBefore(connection, this.networkSvg.firstChild);
        }
      });
    });
  }

  createConnection(startNode, endNode) {
    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line.setAttribute("x1", startNode.getAttribute("cx"));
    line.setAttribute("y1", startNode.getAttribute("cy"));
    line.setAttribute("x2", endNode.getAttribute("cx"));
    line.setAttribute("y2", endNode.getAttribute("cy"));
    line.setAttribute("class", "network-connection");

    return line;
  }

  addLayerLabels() {
    const labels = ["Input", "Hidden 1", "Hidden 2", "Hidden 3", "Output"];
    const layers = [
      this.networkArchitecture.inputSize,
      ...this.networkArchitecture.hiddenLayers,
      this.networkArchitecture.outputSize,
    ];
    const layerSpacing = 1000 / (layers.length + 1);

    this.layerNodes.forEach((layer, index) => {
      if (index < labels.length) {
        const text = document.createElementNS(
          "http://www.w3.org/2000/svg",
          "text"
        );
        text.setAttribute("x", layerSpacing * (index + 1));
        text.setAttribute("y", 30);
        text.setAttribute("class", "network-layer-label");
        text.textContent = labels[index];
        this.networkSvg.appendChild(text);
      }
    });
  }

  showLayerInfo(layerIndex) {
    const layerInfo = document.getElementById("layerInfo");
    const layerName = document.getElementById("layerName");
    const layerDetails = document.getElementById("layerDetails");

    const layerNames = [
      "Input Layer",
      "Hidden Layer 1",
      "Hidden Layer 2",
      "Hidden Layer 3",
      "Output Layer",
    ];
    const layerSizes = [
      this.networkArchitecture.inputSize,
      ...this.networkArchitecture.hiddenLayers,
      this.networkArchitecture.outputSize,
    ];

    layerName.textContent = layerNames[layerIndex] || `Layer ${layerIndex + 1}`;
    layerDetails.textContent = `${layerSizes[layerIndex]} neurons`;

    layerInfo.classList.add("show");
  }

  showNodeInfo(layerIndex, nodeIndex) {
    this.showLayerInfo(layerIndex);
  }

  playAnimation() {
    if (this.isAnimating) return;

    this.isAnimating = true;
    this.animationStep = 0;

    const playBtn = document.getElementById("playAnimationBtn");
    playBtn.disabled = true;
    playBtn.innerHTML = '<span class="btn-icon">⏸️</span>Animating...';

    this.showProgressIndicator();
    this.animateNetworkFlow();
  }

  stepThroughAnimation() {
    if (this.isAnimating) return;

    this.animationStep++;
    this.animateStep(this.animationStep);

    if (this.animationStep >= this.layerNodes.length) {
      this.animationStep = 0;
    }
  }

  animateNetworkFlow() {
    const totalSteps = this.layerNodes.length;
    const stepDuration = 1000; // 1 second per step

    const animate = (step) => {
      if (step >= totalSteps) {
        this.completeAnimation();
        return;
      }

      this.animateStep(step);
      this.updateProgress((step + 1) / totalSteps);

      setTimeout(() => animate(step + 1), stepDuration);
    };

    animate(0);
  }

  animateStep(step) {
    // Reset all nodes and connections
    this.resetNetworkVisualization();

    // Activate current layer
    if (this.layerNodes[step]) {
      this.layerNodes[step].forEach((node) => {
        node.classList.add("active");
      });

      // Activate connections from previous layer
      if (step > 0) {
        this.activateConnections(step - 1, step);
      }
    }

    // Show layer info
    this.showLayerInfo(step);

    // Update progress text
    const layerNames = [
      "Input Processing",
      "Hidden Layer 1",
      "Hidden Layer 2",
      "Hidden Layer 3",
      "Output Generation",
    ];
    this.updateProgressText(layerNames[step] || `Layer ${step + 1}`);
  }

  activateConnections(fromLayer, toLayer) {
    // Enhanced connection activation with staggered timing
    const delay = Math.random() * 300;
    setTimeout(() => {
      this.connections.forEach((connection, index) => {
        if (Math.random() < 0.4) {
          // 40% chance for more visible flow
          setTimeout(() => {
            connection.classList.add("active");
          }, index * 10); // Staggered activation
        }
      });
    }, delay);
  }

  resetNetworkVisualization() {
    // Remove all active states
    this.layerNodes.forEach((layer) => {
      layer.forEach((node) => {
        node.classList.remove("active");
      });
    });

    this.connections.forEach((connection) => {
      connection.classList.remove("active");
    });
  }

  completeAnimation() {
    this.isAnimating = false;
    this.hideProgressIndicator();

    const playBtn = document.getElementById("playAnimationBtn");
    playBtn.disabled = false;
    playBtn.innerHTML = '<span class="btn-icon">▶️</span>Play Animation';

    // Reset visualization
    setTimeout(() => {
      this.resetNetworkVisualization();
    }, 1000);
  }

  resetAnimation() {
    this.animationStep = 0;
    this.isAnimating = false;
    this.resetNetworkVisualization();
    this.hideProgressIndicator();
  }

  showProgressIndicator() {
    const indicator = document.getElementById("progressIndicator");
    indicator.classList.add("show");
  }

  hideProgressIndicator() {
    const indicator = document.getElementById("progressIndicator");
    indicator.classList.remove("show");
  }

  updateProgress(progress) {
    const progressBar = document.getElementById("progressBar");
    progressBar.style.setProperty("--progress", `${progress * 100}%`);
  }

  updateProgressText(text) {
    const progressText = document.getElementById("progressText");
    progressText.textContent = text;
  }

  showLoading() {
    const overlay = document.getElementById("sampleOverlay");
    overlay.classList.add("loading");
  }

  hideLoading() {
    const overlay = document.getElementById("sampleOverlay");
    overlay.classList.remove("loading");
  }

  showError(message) {
    console.error(message);
    this.announceChange(`Error: ${message}`);
  }

  announceChange(message) {
    const announcements = document.getElementById("announcements");
    announcements.textContent = message;
    setTimeout(() => {
      announcements.textContent = "";
    }, 3000);
  }

  debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
      const later = () => {
        clearTimeout(timeout);
        func(...args);
      };
      clearTimeout(timeout);
      timeout = setTimeout(later, wait);
    };
  }
}

// Initialize the showcase when DOM is loaded
document.addEventListener("DOMContentLoaded", () => {
  new DatasetShowcase();
});

// Add CSS for progress bar animation
const style = document.createElement("style");
style.textContent = `
    .progress-bar::before {
        width: var(--progress, 0%);
    }
`;
document.head.appendChild(style);
