/**
 * NeuralEngine dataset showcase - enhanced interactive javascript
 * neural network visualization with data source toggle
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

    // network architecture matching the real model
    this.networkArchitecture = {
      inputSize: 784,
      hiddenLayers: [512, 256, 128],
      outputSize: 10,
    };

    this.initializeShowcase();
    this.setupEventListeners();
    this.setupAdvancedNetworkVisualization();
    this.loadFirstSample();

    // make instance available globally for debugging
    window.datasetShowcase = this;

    console.log("Dataset showcase initialized");
  }

  initializeShowcase() {
    // setup canvas for high DPI displays
    const rect = this.canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width = rect.width * dpr;
    this.canvas.height = rect.height * dpr;
    this.ctx.scale(dpr, dpr);

    // initialize prediction display
    this.updatePredictionDisplay("-", 0, "-");

    // load model info from backend
    this.loadModelInfo();

    // set up toggle initial state
    this.updateToggleState();

    // setup model selector dropdown
    this.setupModelSelector();
  }

  setupEventListeners() {
    // next sample button click
    document.getElementById("nextSampleBtn").addEventListener("click", () => {
      this.loadNextSample();
    });

    // image click handler - also loads next sample
    this.canvas.addEventListener("click", () => {
      this.loadNextSample();
    });

    // data source toggle switch
    document
      .getElementById("dataSourceToggle")
      .addEventListener("change", (e) => {
        this.useSyntheticData = e.target.checked;
        this.updateToggleState();
        this.loadNextSample();
      });

    // back to drawing mode button
    document.getElementById("backToDrawBtn").addEventListener("click", () => {
      window.location.href = "/";
    });

    // keyboard shortcuts for convinience
    document.addEventListener("keydown", (e) => {
      if (e.key === " ") {
        e.preventDefault();
        this.loadNextSample();
      } else if (e.key === "Enter") {
        e.preventDefault();
        this.playAdvancedAnimation();
      } else if (e.key === "ArrowRight") {
        e.preventDefault();
        if (this.advancedGraph) {
          this.advancedGraph.reset();
        }
      } else if (e.key === "t" || e.key === "T") {
        e.preventDefault();
        const toggle = document.getElementById("dataSourceToggle");
        toggle.checked = !toggle.checked;
        toggle.dispatchEvent(new Event("change"));
      }
    });

    // resize handler with debounce
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

    // set initial value to current model
    modelSelector.value = this.currentModel;

    // add change event listener
    modelSelector.addEventListener("change", async (e) => {
      const newModel = e.target.value;
      if (newModel !== this.currentModel && !this.isModelSwitching) {
        await this.switchModel(newModel);
      }
    });

    // add visual feedback on focus
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

    // stop any running animations
    if (this.advancedGraph) {
      this.advancedGraph.reset();
    }

    try {
      const response = await fetch("/switch_model", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model_name: newModelName }),
      });

      const result = await response.json();

      if (result.success) {
        this.currentModel = newModelName;
        this.modelAccuracy = result.model_info.accuracy;
        this.updateModelAccuracy();

        // update network architecture visualization
        if (result.model_info.architecture) {
          this.networkArchitecture = {
            inputSize: result.model_info.architecture[0],
            hiddenLayers: result.model_info.architecture.slice(1, -1),
            outputSize:
              result.model_info.architecture[
                result.model_info.architecture.length - 1
              ],
          };

          // update the graph with new architecture
          if (this.advancedGraph) {
            this.advancedGraph.updateArchitecture(this.networkArchitecture);
          }
        }

        this.showModelSwitchToast(
          `Switched to ${this.getModelDisplayName(newModelName)}`,
          "success"
        );

        // load new sample after brief delay
        setTimeout(() => {
          this.loadNextSample();
        }, 1000);
      } else {
        throw new Error(result.error || "Failed to switch model");
      }
    } catch (error) {
      console.error("Model switch failed:", error);
      this.showModelSwitchToast(
        `Failed to switch model: ${error.message}`,
        "error"
      );
      // revert dropdown to previous model
      document.getElementById("modelSelector").value = this.currentModel;
    } finally {
      this.isModelSwitching = false;
      setTimeout(() => {
        this.hideModelSwitchToast();
      }, 3000);
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

    // update text colors based on selected mode
    if (this.useSyntheticData) {
      toggleTexts[0].style.color = "var(--text-secondary)";
      toggleTexts[1].style.color = "var(--text-primary)";
    } else {
      toggleTexts[0].style.color = "var(--text-primary)";
      toggleTexts[1].style.color = "var(--text-secondary)";
    }

    // announce change for accesibility
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
        // use synthetic data generation
        const sampleData = await this.generateSampleData();
        this.currentSample = sampleData;
        this.sampleCount++;

        // update sample counter
        document.getElementById("sampleCounter").textContent = this.sampleCount;

        // display the sample
        this.displaySample(sampleData);

        // make synthetic prediction
        await this.makeSyntheticPrediction(sampleData);

        // announce to screen readers
        this.announceChange(
          `Loaded synthetic sample ${this.sampleCount}, predicted digit ${this.currentPrediction.predicted_digit}`
        );
      } else {
        // call the backend API for real data
        const response = await fetch("/api/dataset/sample");
        const data = await response.json();

        if (data.error) {
          throw new Error(data.error);
        }

        this.currentSample = data.sample;
        this.currentPrediction = data.prediction;
        this.sampleCount++;

        // update sample counter
        document.getElementById("sampleCounter").textContent = this.sampleCount;

        // display the real sample
        this.displaySampleFromAPI(data.sample);

        // update prediction display with real results
        this.updatePredictionDisplay(
          data.prediction.predicted_digit,
          data.prediction.confidence,
          data.sample.actual_label
        );

        // announce to screen readers
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
      // clear canvas with white background
      this.ctx.fillStyle = "#FFFFFF";
      this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

      // draw the sample image scaled up
      const scale =
        Math.min(this.canvas.width / 280, this.canvas.height / 280) * 0.8;
      const x = (this.canvas.width - 280 * scale) / 2;
      const y = (this.canvas.height - 280 * scale) / 2;

      // disable smoothing for pixelated look
      this.ctx.imageSmoothingEnabled = false;
      this.ctx.drawImage(img, x, y, 280 * scale, 280 * scale);

      // add subtle border around image
      this.ctx.strokeStyle = "#E0E0E0";
      this.ctx.lineWidth = 2;
      this.ctx.strokeRect(x, y, 280 * scale, 280 * scale);
    };
    img.src = sampleData.image_data;
  }

  async generateSampleData() {
    // generate synthetic dataset sample
    const digit = Math.floor(Math.random() * 10);
    const imageData = this.generateDigitImage(digit);

    return {
      image: imageData,
      label: digit,
      index: this.sampleCount,
    };
  }

  generateDigitImage(digit) {
    // generate a more realistic representaton of a digit
    const canvas = document.createElement("canvas");
    canvas.width = 28;
    canvas.height = 28;
    const ctx = canvas.getContext("2d");

    // fill with black background
    ctx.fillStyle = "#000000";
    ctx.fillRect(0, 0, 28, 28);

    // draw white digit with variations
    ctx.fillStyle = "#FFFFFF";
    ctx.font = "20px Arial";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";

    // add some noise/variation for realism
    const offsetX = (Math.random() - 0.5) * 6;
    const offsetY = (Math.random() - 0.5) * 6;
    const rotation = (Math.random() - 0.5) * 0.4;

    ctx.save();
    ctx.translate(14, 14);
    ctx.rotate(rotation);
    ctx.fillText(digit.toString(), offsetX, offsetY);
    ctx.restore();

    // add some noise to make it look more natural
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
      // clear canvas
      this.ctx.fillStyle = "#FFFFFF";
      this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

      // draw the sample image scaled up
      const scale =
        Math.min(this.canvas.width / 28, this.canvas.height / 28) * 0.8;
      const x = (this.canvas.width - 28 * scale) / 2;
      const y = (this.canvas.height - 28 * scale) / 2;

      this.ctx.imageSmoothingEnabled = false;
      this.ctx.drawImage(img, x, y, 28 * scale, 28 * scale);

      // add subtle border
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

      // update prediction display
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
    // simulate more realistic high-confidence predictions
    const predictions = Array.from({ length: 10 }, () => Math.random() * 0.1);

    // add some realistic variation
    const isCorrect = Math.random() > 0.05; // 95% accuracy simulaton
    const predictedDigit = isCorrect
      ? sampleData.label
      : (sampleData.label + 1) % 10;

    // set high confidence for the predicted digit
    predictions[predictedDigit] = 0.8 + Math.random() * 0.15; // 80-95% confidence

    // normalize predictions to sum to 1
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

    // animate digit change with scale effect
    digitElement.style.transform = "scale(0.8)";
    digitElement.style.opacity = "0.5";

    setTimeout(() => {
      digitElement.textContent = digit;
      digitElement.style.transform = "scale(1)";
      digitElement.style.opacity = "1";
    }, 150);

    // update confidence with animation
    confidenceElement.textContent = `${confidence.toFixed(1)}%`;

    // update actual label
    actualElement.textContent = actualLabel;

    // add correct/incorrect styling
    if (digit.toString() === actualLabel.toString()) {
      digitElement.style.color = "#34C759"; // green for correct
      confidenceElement.style.color = "#34C759";
    } else {
      digitElement.style.color = "#FF3B30"; // red for incorrect
      confidenceElement.style.color = "#FF3B30";
    }
  }

  // advanced network visualization setup
  setupAdvancedNetworkVisualization() {
    // check if required dependencies are available
    if (typeof AdvancedNeuralGraph === "undefined") {
      console.error("AdvancedNeuralGraph class not found");
      return;
    }

    try {
      // create the advanced network graph
      this.advancedGraph = new AdvancedNeuralGraph(
        "advancedNetworkContainer",
        this.networkArchitecture
      );

      // setup control buttons
      document
        .getElementById("playAnimationBtn")
        .addEventListener("click", () => {
          this.playAdvancedAnimation();
        });

      document
        .getElementById("resetNetworkBtn")
        .addEventListener("click", () => {
          if (this.advancedGraph) {
            this.advancedGraph.reset();
          }
        });

      // setup speed control
      document
        .getElementById("animationSpeed")
        .addEventListener("change", (e) => {
          if (this.advancedGraph) {
            this.advancedGraph.setAnimationSpeed(parseInt(e.target.value));
          }
        });

      console.log("Advanced neural graph initialized");
    } catch (error) {
      console.error("Failed to initialize advanced neural graph:", error);
    }
  }

  async playAdvancedAnimation() {
    if (!this.currentSample) {
      this.showError("No image loaded for visualizaton");
      return;
    }

    if (!this.advancedGraph) {
      this.showError("Neural network visualization not initialized");
      return;
    }

    const playBtn = document.getElementById("playAnimationBtn");
    playBtn.disabled = true;
    playBtn.innerHTML = '<span class="btn-icon">⏸️</span>Processing...';

    try {
      // get real activation data from the current image
      const activationData = await this.getRealActivationData();

      if (!activationData) {
        throw new Error("Failed to get activation data");
      }

      // show processing context with real data
      this.showProcessingContext();

      // run the animation with real data
      await this.advancedGraph.animateForwardPass(activationData);

      // show final result explanation
      this.showResultExplanation();
    } catch (error) {
      console.error("Animation error:", error);
      this.showError("Failed to animate: " + error.message);
    } finally {
      playBtn.disabled = false;
      playBtn.innerHTML = '<span class="btn-icon">▶️</span>Play Animation';
    }
  }

  async getRealActivationData() {
    if (!this.currentSample) return null;

    try {
      // send current image to backend for detailed activation analysis
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
        console.warn("Failed to get real activations, using simulaton");
        return this.generateRealisticActivationData();
      }

      return data.layer_activations;
    } catch (error) {
      console.error("Error fetching real activations:", error);
      return this.generateRealisticActivationData();
    }
  }

  generateRealisticActivationData() {
    const layers = [
      this.networkArchitecture.inputSize,
      ...this.networkArchitecture.hiddenLayers,
      this.networkArchitecture.outputSize,
    ];

    const activationData = [];
    const predictedDigit = this.currentPrediction?.predicted_digit || 0;

    for (let i = 0; i < layers.length; i++) {
      const layerSize = Math.min(layers[i], 12);
      const activations = [];

      for (let j = 0; j < layerSize; j++) {
        if (i === 0) {
          // input layer: simulate pixel intensities
          activations.push(Math.random() * 0.8 + 0.1);
        } else if (i === layers.length - 1) {
          // output layer with correct digit having highest activation
          if (j === predictedDigit) {
            activations.push(0.85 + Math.random() * 0.1); // 85-95% for predicted digit
          } else {
            activations.push(Math.random() * 0.35 + 0.05); // 5-40% for others
          }
        } else {
          // hidden layers: simulate ReLU activations
          activations.push(Math.max(0, Math.random() * 1.2 - 0.3));
        }
      }

      activationData.push(activations);
    }

    return activationData;
  }

  showProcessingContext() {
    if (!this.advancedGraph) return;

    const digit = this.currentPrediction?.predicted_digit || "?";
    const confidence = this.currentPrediction?.confidence || 0;

    this.advancedGraph.showProcessingContext(digit, confidence);
  }

  showResultExplanation() {
    // remove any existing explanation
    const existing = document.getElementById("resultExplanation");
    if (existing) existing.remove();

    const explanation = document.createElement("div");
    explanation.id = "resultExplanation";
    explanation.style.cssText = `
            position: fixed;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(52, 199, 89, 0.9);
            color: white;
            padding: 15px 25px;
            border-radius: 12px;
            font-family: Inter, sans-serif;
            font-size: 14px;
            z-index: 1000;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.2);
            backdrop-filter: blur(10px);
        `;

    const digit = this.currentPrediction?.predicted_digit || "?";
    const confidence = this.currentPrediction?.confidence || 0;
    const actual =
      this.currentSample?.label || this.currentSample?.actual_label || "?";

    explanation.innerHTML = `
            <strong>Result:</strong> Predicted digit ${digit} with ${confidence.toFixed(
      1
    )}% confidence
            ${
              digit.toString() === actual.toString()
                ? " Correct!"
                : ` (Actual: ${actual})`
            }
        `;

    document.body.appendChild(explanation);

    // auto-hide after 5 seconds
    setTimeout(() => {
      if (explanation.parentNode) {
        explanation.remove();
      }
    }, 5000);
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

    // show error toast
    const errorToast = document.createElement("div");
    errorToast.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: rgba(255, 59, 48, 0.9);
            color: white;
            padding: 12px 20px;
            border-radius: 8px;
            font-family: Inter, sans-serif;
            font-size: 14px;
            z-index: 10000;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        `;
    errorToast.textContent = message;
    document.body.appendChild(errorToast);

    setTimeout(() => {
      if (errorToast.parentNode) {
        errorToast.remove();
      }
    }, 3000);
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

// initialize the showcase when DOM is loaded
document.addEventListener("DOMContentLoaded", () => {
  new DatasetShowcase();
});
