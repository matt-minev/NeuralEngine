/**
 * NeuralEngine Dataset Showcase - Enhanced Interactive JavaScript
 * Apple-inspired neural network visualization with data source toggle and advanced graph
 */

class DatasetShowcase {
  constructor() {
    this.canvas = document.getElementById("sampleCanvas");
    this.ctx = this.canvas.getContext("2d");
    this.currentSample = null;
    this.currentPrediction = null;
    this.isAnimating = false;
    this.animationStep = 0;
    this.sampleCount = 0;
    this.modelAccuracy = 0;
    this.animationId = null;
    this.useSyntheticData = false;
    this.currentModel = "enhanced_digit_model.pkl";
    this.isModelSwitching = false;
    this.firstLoadComplete = false;
    this.advancedGraph = null;

    // Network architecture (matching your real model)
    this.networkArchitecture = {
      inputSize: 784,
      hiddenLayers: [512, 256, 128],
      outputSize: 10,
    };

    this.initializeShowcase();
    this.setupEventListeners();
    this.setupAdvancedNetworkVisualization();
    this.loadFirstSample();

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

    // Setup model selector
    this.setupModelSelector();
  }

  setupEventListeners() {
    // Next sample button
    document.getElementById("nextSampleBtn").addEventListener("click", () => {
      this.loadNextSample();
    });

    // Image click handler
    this.canvas.addEventListener("click", () => {
      this.loadNextSample();
    });

    // Data source toggle
    document
      .getElementById("dataSourceToggle")
      .addEventListener("change", (e) => {
        this.useSyntheticData = e.target.checked;
        this.updateToggleState();
        this.loadNextSample();
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
        this.playAdvancedAnimation();
      } else if (e.key === "ArrowRight") {
        e.preventDefault();
        this.advancedGraph.reset();
      } else if (e.key === "t" || e.key === "T") {
        e.preventDefault();
        const toggle = document.getElementById("dataSourceToggle");
        toggle.checked = !toggle.checked;
        toggle.dispatchEvent(new Event("change"));
      }
    });

    // Resize handler
    window.addEventListener("resize", () => {
      this.debounce(() => {
        if (this.advancedGraph) {
          this.advancedGraph.updateArchitecture(this.networkArchitecture);
        }
      }, 250);
    });
  }

  setupModelSelector() {
    const modelSelector = document.getElementById("modelSelector");

    // Set initial value
    modelSelector.value = this.currentModel;

    // Add change event listener
    modelSelector.addEventListener("change", async (e) => {
      const newModel = e.target.value;
      if (newModel !== this.currentModel && !this.isModelSwitching) {
        await this.switchModel(newModel);
      }
    });

    // Add visual feedback on focus
    modelSelector.addEventListener("focus", () => {
      modelSelector.parentElement.classList.add("focused");
    });

    modelSelector.addEventListener("blur", () => {
      modelSelector.parentElement.classList.remove("focused");
    });
  }

  async switchModel(newModelName) {
    if (this.isModelSwitching) return;

    this.isModelSwitching = true;
    this.showModelSwitchToast(
      `Switching to ${this.getModelDisplayName(newModelName)}...`
    );

    try {
      const response = await fetch("/switch_model", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ model_name: newModelName }),
      });

      const result = await response.json();

      if (result.success) {
        this.currentModel = newModelName;
        this.modelAccuracy = result.model_info.accuracy;
        this.updateModelAccuracy();

        // Update network architecture if it changed
        if (result.model_info.architecture) {
          this.networkArchitecture = {
            inputSize: result.model_info.architecture[0],
            hiddenLayers: result.model_info.architecture.slice(1, -1),
            outputSize:
              result.model_info.architecture[
                result.model_info.architecture.length - 1
              ],
          };

          // Update the advanced graph with new architecture
          if (this.advancedGraph) {
            this.advancedGraph.updateArchitecture(this.networkArchitecture);
          }
        }

        this.showModelSwitchToast(
          `✅ Switched to ${this.getModelDisplayName(newModelName)}`,
          "success"
        );
        this.announceChange(
          `Model switched to ${this.getModelDisplayName(newModelName)}`
        );

        // Load a new sample with the new model
        setTimeout(() => {
          this.loadNextSample();
        }, 500);
      } else {
        throw new Error(result.error || "Failed to switch model");
      }
    } catch (error) {
      console.error("Model switch failed:", error);
      this.showModelSwitchToast(
        `❌ Failed to switch model: ${error.message}`,
        "error"
      );

      // Revert selector to previous model
      document.getElementById("modelSelector").value = this.currentModel;
    } finally {
      this.isModelSwitching = false;
      setTimeout(() => {
        this.hideModelSwitchToast();
      }, 2000);
    }
  }

  getModelDisplayName(modelName) {
    const displayNames = {
      "enhanced_digit_model.pkl": "Enhanced Model",
      "basic_digit_model.pkl": "Basic Model",
      "advanced_digit_model.pkl": "Advanced Model",
    };
    return displayNames[modelName] || modelName.replace(".pkl", "");
  }

  showModelSwitchToast(message, type = "info") {
    const toast = document.getElementById("modelSwitchToast");
    const messageElement = document.getElementById("toastMessage");

    messageElement.textContent = message;
    toast.className = `model-switch-toast ${type}`;
    toast.classList.add("show");
  }

  hideModelSwitchToast() {
    const toast = document.getElementById("modelSwitchToast");
    toast.classList.remove("show");
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
    this.showLoading();
    try {
      await this.loadSampleFromDataset();
      this.firstLoadComplete = true;
    } catch (error) {
      console.error("Failed to load first sample:", error);
      this.showError("Failed to load first sample");
    } finally {
      this.hideLoading();
    }
  }

  async loadNextSample() {
    if (!this.firstLoadComplete) return;

    this.showLoading();
    try {
      await this.loadSampleFromDataset();
    } catch (error) {
      console.error("Failed to load sample:", error);
      this.showError("Failed to load sample");
    } finally {
      this.hideLoading();
    }
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

  // Advanced Network Visualization Setup
  setupAdvancedNetworkVisualization() {
    // Create the advanced network graph
    this.advancedGraph = new AdvancedNeuralGraph(
      "advancedNetworkContainer",
      this.networkArchitecture
    );

    // Setup controls
    document
      .getElementById("playAnimationBtn")
      .addEventListener("click", () => {
        this.playAdvancedAnimation();
      });

    document.getElementById("resetNetworkBtn").addEventListener("click", () => {
      this.advancedGraph.reset();
    });

    document
      .getElementById("animationSpeed")
      .addEventListener("change", (e) => {
        this.advancedGraph.setAnimationSpeed(parseInt(e.target.value));
      });
  }

  async playAdvancedAnimation() {
    if (!this.currentSample) {
      this.showError("No image loaded for visualization");
      return;
    }

    const playBtn = document.getElementById("playAnimationBtn");
    playBtn.disabled = true;
    playBtn.innerHTML = '<span class="btn-icon">⏸️</span>Processing...';

    try {
      // Get real activation data from the current image
      const activationData = await this.getRealActivationData();

      if (!activationData) {
        throw new Error("Failed to get activation data");
      }

      // Add contextual information overlay
      this.showProcessingContext();

      // Run the animation with real data
      await this.advancedGraph.animateForwardPass(activationData);

      // Show final result explanation
      this.showResultExplanation();
    } catch (error) {
      console.error("Animation error:", error);
      this.showError("Failed to animate: " + error.message);
    } finally {
      playBtn.disabled = false;
      playBtn.innerHTML = '<span class="btn-icon">▶️</span>Play Animation';
    }
  }

  showProcessingContext() {
    const contextPanel = d3
      .select("body")
      .append("div")
      .attr("id", "processingContext")
      .style("position", "fixed")
      .style("top", "50%")
      .style("left", "20px")
      .style("transform", "translateY(-50%)")
      .style("background", "rgba(0, 122, 255, 0.9)")
      .style("color", "white")
      .style("padding", "20px")
      .style("border-radius", "12px")
      .style("max-width", "250px")
      .style("font-family", "Inter, sans-serif")
      .style("z-index", "1500");

    contextPanel.html(`
        <h3 style="margin-top: 0;">🧠 Processing "${
          this.currentPrediction?.predicted_digit || "?"
        }"</h3>
        <p style="margin-bottom: 15px;">Watch how the network analyzes this digit:</p>
        <div id="processingSteps">
            <div class="step active">1. Reading pixels...</div>
            <div class="step">2. Detecting edges...</div>
            <div class="step">3. Finding patterns...</div>
            <div class="step">4. Classifying digit...</div>
        </div>
    `);

    // Remove after animation
    setTimeout(() => {
      contextPanel.remove();
    }, 8000);
  }

  async generateActivationData() {
    // Generate realistic activation values for each layer
    const layers = [
      this.networkArchitecture.inputSize,
      ...this.networkArchitecture.hiddenLayers,
      this.networkArchitecture.outputSize,
    ];

    const activationData = [];

    for (let i = 0; i < layers.length; i++) {
      const layerSize = Math.min(layers[i], 15); // Match visible neurons
      const activations = [];

      for (let j = 0; j < layerSize; j++) {
        if (i === 0) {
          // Input layer: simulate pixel intensities
          activations.push(Math.random() * 0.8 + 0.1);
        } else if (i === layers.length - 1) {
          // Output layer: simulate softmax probabilities
          const value = Math.random();
          activations.push(j === 0 ? Math.max(value, 0.7) : value * 0.3);
        } else {
          // Hidden layers: simulate ReLU activations
          activations.push(Math.max(0, Math.random() * 2 - 0.5));
        }
      }

      activationData.push(activations);
    }

    return activationData;
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

  // Get real neural network activations for the current image
  async getRealActivationData() {
    if (!this.currentSample) return null;

    try {
      // Send current image to backend for detailed activation analysis
      const response = await fetch("/api/neural/activations", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          image_data: this.currentSample.image_data || this.currentSample.image,
          model_name: this.currentModel,
        }),
      });

      const data = await response.json();

      if (data.error) {
        console.warn("Failed to get real activations, using simulation");
        return this.generateRealisticActivationData();
      }

      return data.layer_activations;
    } catch (error) {
      console.error("Error fetching real activations:", error);
      return this.generateRealisticActivationData();
    }
  }

  // Enhanced realistic activation generation based on current image
  generateRealisticActivationData() {
    const layers = [
      this.networkArchitecture.inputSize,
      ...this.networkArchitecture.hiddenLayers,
      this.networkArchitecture.outputSize,
    ];

    const activationData = [];

    // Simulate realistic activations that make sense for digit recognition
    for (let i = 0; i < layers.length; i++) {
      const layerSize = Math.min(layers[i], 15);
      const activations = [];

      for (let j = 0; j < layerSize; j++) {
        if (i === 0) {
          // Input layer: simulate pixel intensities from actual image
          const intensity = this.getPixelIntensityForNeuron(j);
          activations.push(intensity);
        } else if (i === layers.length - 1) {
          // Output layer: simulate softmax with correct digit having highest activation
          const predictedDigit = this.currentPrediction?.predicted_digit || 0;
          if (j === predictedDigit) {
            activations.push(0.8 + Math.random() * 0.15); // High activation for predicted digit
          } else {
            activations.push(Math.random() * 0.3); // Lower for others
          }
        } else {
          // Hidden layers: simulate feature detection activations
          const featureStrength = this.simulateFeatureDetection(i, j);
          activations.push(Math.max(0, featureStrength)); // ReLU-like activation
        }
      }

      activationData.push(activations);
    }

    return activationData;
  }

  showNeuronInfoPanel(neuron, connectedNeurons) {
    // Remove existing panel
    d3.select("#neuronInfoPanel").remove();

    // Create info panel
    const panel = d3
      .select("body")
      .append("div")
      .attr("id", "neuronInfoPanel")
      .style("position", "fixed")
      .style("top", "20px")
      .style("right", "20px")
      .style("background", "rgba(0, 0, 0, 0.85)")
      .style("color", "white")
      .style("padding", "15px")
      .style("border-radius", "12px")
      .style("max-width", "300px")
      .style("font-family", "Inter, sans-serif")
      .style("font-size", "14px")
      .style("z-index", "2000")
      .style("box-shadow", "0 8px 32px rgba(0, 0, 0, 0.3)")
      .style("backdrop-filter", "blur(10px)");

    // Add content based on neuron layer
    const layerInfo = this.getLayerExplanation(neuron.layer);
    const activationLevel = this.getActivationLevel(neuron.activation);

    panel.html(`
        <div style="border-bottom: 1px solid rgba(255,255,255,0.2); padding-bottom: 10px; margin-bottom: 10px;">
            <strong style="color: #4facfe;">Neuron ${neuron.id}</strong>
        </div>
        
        <div style="margin-bottom: 8px;">
            <strong>Layer:</strong> ${layerInfo.name}<br>
            <small style="color: #aaa;">${layerInfo.description}</small>
        </div>
        
        <div style="margin-bottom: 8px;">
            <strong>Activation:</strong> 
            <span style="color: ${activationLevel.color}; font-weight: bold;">
                ${neuron.activation.toFixed(3)}
            </span>
            <small style="color: #aaa;"> (${
              activationLevel.description
            })</small>
        </div>
        
        <div style="margin-bottom: 8px;">
            <strong>Connected to:</strong> ${connectedNeurons.length} neurons
        </div>
        
        <div style="background: rgba(255,255,255,0.1); padding: 8px; border-radius: 6px; margin-top: 10px;">
            <small><strong>💡 What this means:</strong><br>
            ${this.getNeuronExplanation(neuron)}</small>
        </div>
    `);

    // Auto-hide after 10 seconds
    setTimeout(() => {
      d3.select("#neuronInfoPanel")
        .transition()
        .duration(500)
        .style("opacity", 0)
        .remove();
    }, 10000);
  }

  getLayerExplanation(layerIndex) {
    const explanations = {
      0: {
        name: "Input Layer",
        description:
          "Receives raw pixel data from the image (28×28 = 784 pixels)",
      },
      1: {
        name: "Hidden Layer 1",
        description: "Detects basic features like edges and simple shapes",
      },
      2: {
        name: "Hidden Layer 2",
        description: "Combines basic features into more complex patterns",
      },
      3: {
        name: "Hidden Layer 3",
        description: "Recognizes digit-specific features and patterns",
      },
      4: {
        name: "Output Layer",
        description:
          "Final classification - each neuron represents a digit (0-9)",
      },
    };

    return (
      explanations[layerIndex] || {
        name: `Layer ${layerIndex + 1}`,
        description: "Processing layer",
      }
    );
  }

  getActivationLevel(activation) {
    if (activation > 0.7) {
      return { color: "#34C759", description: "High - Strong response" };
    } else if (activation > 0.3) {
      return { color: "#FF9500", description: "Medium - Moderate response" };
    } else {
      return { color: "#8E8E93", description: "Low - Weak response" };
    }
  }

  getNeuronExplanation(neuron) {
    const layer = neuron.layer;
    const activation = neuron.activation;

    if (layer === 0) {
      return `This neuron represents a pixel in the input image. Activation of ${activation.toFixed(
        3
      )} indicates ${activation > 0.5 ? "bright" : "dark"} pixel intensity.`;
    } else if (layer === this.networkArchitecture.hiddenLayers.length + 1) {
      return `This output neuron represents digit ${neuron.index}. Higher activation means the network is more confident this digit is present.`;
    } else {
      return `This hidden neuron detects specific features in the image. Activation of ${activation.toFixed(
        3
      )} shows ${activation > 0.5 ? "strong" : "weak"} feature detection.`;
    }
  }
}

// Initialize the showcase when DOM is loaded
document.addEventListener("DOMContentLoaded", () => {
  new DatasetShowcase();
});
