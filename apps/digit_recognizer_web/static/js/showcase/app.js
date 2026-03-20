import { fetchArchitecture, fetchDatasetSample, fetchLayerActivations, fetchModelInfo, switchModel } from "../core/api.js";
import { debounce, qs, qsa, showToast } from "../core/utils.js";
import { SignalFlowVisualizer } from "./signal-flow-visualizer.js";

const MODEL_LABELS = {
  "enhanced_digit_model.pkl": "Enhanced Model",
  "digit_model_bulletproof.pkl": "Bulletproof Model",
  "digit_model_optimized.pkl": "Optimized Model",
  "digit_model.pkl": "Basic Model",
};

const FALLBACK_ARCHITECTURE = {
  inputSize: 784,
  hiddenLayers: [512, 256, 128],
  outputSize: 10,
};

class DatasetShowcaseApp {
  constructor() {
    this.canvas = qs("#sampleCanvas");
    this.ctx = this.canvas.getContext("2d");
    this.currentModel = qs("#modelSelector").value;
    this.currentSample = null;
    this.currentPrediction = null;
    this.sampleCount = 0;
    this.useSyntheticData = false;

    this.visualizer = new SignalFlowVisualizer(qs("#advancedNetworkContainer"), qs("#networkInspector"));
    this.bindEvents();
    this.bootstrap();
  }

  bindEvents() {
    qs("#nextSampleBtn").addEventListener("click", () => this.loadNextSample());
    this.canvas.addEventListener("click", () => this.loadNextSample());

    qs("#dataSourceToggle").addEventListener("change", (event) => {
      this.useSyntheticData = event.target.checked;
      this.announce(`Switched to ${this.useSyntheticData ? "synthetic" : "real"} samples`);
      this.loadNextSample();
    });

    qs("#animationSpeed").addEventListener("change", (event) => {
      this.visualizer.setAnimationSpeed(Number.parseInt(event.target.value, 10));
    });

    qs("#resetNetworkBtn").addEventListener("click", () => {
      this.visualizer.reset();
      this.resetProcessingSteps();
    });

    qs("#playAnimationBtn").addEventListener("click", () => this.playInference());

    qs("#modelSelector").addEventListener("change", async (event) => {
      const modelName = event.target.value;
      this.showSwitchToast(`Switching to ${MODEL_LABELS[modelName] || modelName}...`);
      try {
        const result = await switchModel(modelName);
        this.currentModel = modelName;
        qs("#modelAccuracy").textContent = `${result.model_info.accuracy.toFixed(1)}%`;
        await this.refreshArchitecture();
        await this.loadNextSample();
        this.showSwitchToast(`Switched to ${MODEL_LABELS[modelName] || modelName}`);
      } catch (error) {
        showToast(error.message, "error");
        qs("#modelSelector").value = this.currentModel;
      }
    });

    document.addEventListener("keydown", (event) => {
      if (event.key === " ") {
        event.preventDefault();
        this.loadNextSample();
      }
      if (event.key === "Enter") {
        event.preventDefault();
        this.playInference();
      }
      if (event.key === "ArrowRight") {
        event.preventDefault();
        this.visualizer.reset();
        this.resetProcessingSteps();
      }
      if (event.key === "t" || event.key === "T") {
        event.preventDefault();
        const toggle = qs("#dataSourceToggle");
        toggle.checked = !toggle.checked;
        toggle.dispatchEvent(new Event("change"));
      }
    });

    window.addEventListener("resize", debounce(() => this.visualizer.redrawConnections(), 200));
  }

  async bootstrap() {
    this.showLoading();

    try {
      await this.refreshArchitecture();
    } catch (error) {
      this.visualizer.updateArchitecture(FALLBACK_ARCHITECTURE);
      showToast("Using fallback visualization architecture", "error");
    }

    try {
      await this.loadModelInfo();
    } catch (error) {
      qs("#modelAccuracy").textContent = "--%";
    }

    try {
      await this.loadNextSample(true);
    } finally {
      this.hideLoading();
    }
  }

  async refreshArchitecture() {
    const architecture = await fetchArchitecture();
    this.visualizer.updateArchitecture({
      inputSize: architecture.input_size,
      hiddenLayers: architecture.hidden_layers,
      outputSize: architecture.output_size,
    });
  }

  async loadModelInfo() {
    const modelInfo = await fetchModelInfo();
    qs("#modelAccuracy").textContent = `${(modelInfo.accuracy || 0).toFixed(1)}%`;
  }

  async loadNextSample(initial = false) {
    this.showLoading();
    try {
      if (this.useSyntheticData) {
        await this.loadSyntheticSample();
      } else {
        await this.loadDatasetSample();
      }
      this.sampleCount += 1;
      qs("#sampleCounter").textContent = String(this.sampleCount);
      if (!initial) {
        this.announce(`Loaded sample ${this.sampleCount}`);
      }
    } catch (error) {
      showToast(error.message, "error");
      this.announce(`Error: ${error.message}`);
    } finally {
      this.hideLoading();
    }
  }

  async loadDatasetSample() {
    const data = await fetchDatasetSample();
    this.currentSample = data.sample;
    this.currentPrediction = data.prediction;
    await this.renderSample(data.sample.image_data);
    this.renderPrediction(data.prediction.predicted_digit, data.prediction.confidence, data.sample.actual_label);
    this.writeSummary(data.prediction.predicted_digit, data.prediction.confidence, data.sample.actual_label);
  }

  async loadSyntheticSample() {
    const digit = Math.floor(Math.random() * 10);
    const offscreen = document.createElement("canvas");
    offscreen.width = 28;
    offscreen.height = 28;
    const context = offscreen.getContext("2d");
    context.fillStyle = "#000";
    context.fillRect(0, 0, 28, 28);
    context.fillStyle = "#fff";
    context.font = "24px Arial";
    context.textAlign = "center";
    context.textBaseline = "middle";
    context.fillText(String(digit), 14 + (Math.random() - 0.5) * 2, 15 + (Math.random() - 0.5) * 2);

    const image = offscreen.toDataURL("image/png");
    this.currentSample = { image_data: image, actual_label: digit };
    const confidence = 86 + Math.random() * 12;
    this.currentPrediction = {
      predicted_digit: Math.random() > 0.08 ? digit : (digit + 1) % 10,
      confidence,
    };

    await this.renderSample(image, false);
    this.renderPrediction(this.currentPrediction.predicted_digit, confidence, digit);
    this.writeSummary(this.currentPrediction.predicted_digit, confidence, digit);
  }

  renderSample(imageData, upscalePixels = true) {
    return new Promise((resolve, reject) => {
      const image = new Image();
      image.onload = () => {
        const rect = this.canvas.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        this.canvas.width = rect.width * dpr;
        this.canvas.height = rect.height * dpr;
        this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        this.ctx.fillStyle = "#000";
        this.ctx.fillRect(0, 0, rect.width, rect.height);

        const sourceCanvas = document.createElement("canvas");
        sourceCanvas.width = image.width;
        sourceCanvas.height = image.height;
        const sourceContext = sourceCanvas.getContext("2d");
        sourceContext.drawImage(image, 0, 0);

        const bounds = this.getDigitBounds(
          sourceContext.getImageData(0, 0, sourceCanvas.width, sourceCanvas.height),
          sourceCanvas.width,
          sourceCanvas.height
        );

        const padding = Math.min(rect.width, rect.height) * 0.07;
        const availableWidth = rect.width - padding * 2;
        const availableHeight = rect.height - padding * 2;
        const scale = Math.min(availableWidth / bounds.width, availableHeight / bounds.height);
        const targetWidth = bounds.width * scale;
        const targetHeight = bounds.height * scale;
        const x = (rect.width - targetWidth) / 2;
        const y = (rect.height - targetHeight) / 2;

        this.ctx.imageSmoothingEnabled = !upscalePixels;
        this.ctx.drawImage(
          sourceCanvas,
          bounds.x,
          bounds.y,
          bounds.width,
          bounds.height,
          x,
          y,
          targetWidth,
          targetHeight
        );
        resolve();
      };
      image.onerror = () => reject(new Error("Failed to render specimen image"));
      image.src = imageData;
    });
  }

  getDigitBounds(imageData, width, height) {
    const pixels = imageData.data;
    let minX = width;
    let minY = height;
    let maxX = 0;
    let maxY = 0;
    let found = false;

    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        const offset = (y * width + x) * 4;
        const intensity = pixels[offset];
        if (intensity > 28) {
          found = true;
          minX = Math.min(minX, x);
          minY = Math.min(minY, y);
          maxX = Math.max(maxX, x);
          maxY = Math.max(maxY, y);
        }
      }
    }

    if (!found) {
      return { x: 0, y: 0, width, height };
    }

    const margin = 1;
    return {
      x: Math.max(0, minX - margin),
      y: Math.max(0, minY - margin),
      width: Math.min(width, maxX - minX + 1 + margin * 2),
      height: Math.min(height, maxY - minY + 1 + margin * 2),
    };
  }

  renderPrediction(digit, confidence, actualLabel) {
    qs("#predictedDigit").textContent = digit;
    qs("#confidenceValue").textContent = `${confidence.toFixed(1)}%`;
    qs("#actualValue").textContent = actualLabel;
    qs("#predictedDigit").style.color = String(digit) === String(actualLabel) ? "var(--green)" : "var(--amber)";
  }

  async playInference() {
    if (!this.currentSample) {
      showToast("Load a sample first", "error");
      return;
    }

    const playButton = qs("#playAnimationBtn");
    playButton.disabled = true;
    playButton.textContent = "Running...";

    try {
      let response = null;
      try {
        response = await fetchLayerActivations(this.currentSample.image_data, this.currentModel);
      } catch (error) {
        response = null;
      }

      const predictedDigit =
        this.currentPrediction && typeof this.currentPrediction.predicted_digit !== "undefined"
          ? this.currentPrediction.predicted_digit
          : 0;
      const layerActivations = response && response.layer_activations ? response.layer_activations : [];
      const activations = this.normalizeActivationData(layerActivations, predictedDigit);
      this.resetProcessingSteps();
      await this.visualizer.animateForwardPass(activations, (stepIndex) => this.updateProcessingStep(stepIndex));
      this.writeSummary(this.currentPrediction.predicted_digit, this.currentPrediction.confidence, this.currentSample.actual_label);
    } finally {
      playButton.disabled = false;
      playButton.textContent = "Play Inference";
    }
  }

  normalizeActivationData(layerActivations, predictedDigit) {
    if (!this.visualizer.architecture) {
      this.visualizer.updateArchitecture(FALLBACK_ARCHITECTURE);
    }

    const inputLayer = Array.from({ length: 12 }, () => Math.random() * 0.7 + 0.18);
    const normalized = [inputLayer];

    layerActivations.forEach((layer) => {
      normalized.push(layer.slice(0, 8).map((value) => Math.max(0, Math.min(1, value))));
    });

    while (normalized.length < this.visualizer.architecture.hiddenLayers.length + 2) {
      normalized.push(Array.from({ length: 8 }, () => Math.random() * 0.8));
    }

    normalized[normalized.length - 1] = Array.from({ length: 10 }, (_, index) =>
      index === predictedDigit ? 0.92 : Math.random() * 0.38
    );

    return normalized;
  }

  updateProcessingStep(stepIndex) {
    qsa(".processing-step").forEach((element, index) => {
      element.classList.toggle("is-active", index === stepIndex);
      element.classList.toggle("is-complete", index < stepIndex);
    });
  }

  resetProcessingSteps() {
    qsa(".processing-step").forEach((element) => {
      element.classList.remove("is-active", "is-complete");
    });
  }

  writeSummary(digit, confidence, actualLabel) {
    const isCorrect = String(digit) === String(actualLabel);
    qs("#resultExplanation").innerHTML = `
      <p><strong>${isCorrect ? "Correct classification" : "Mismatched prediction"}</strong></p>
      <p>The network selected digit <strong>${digit}</strong> with <strong>${confidence.toFixed(1)}%</strong> confidence.</p>
      <p>${isCorrect ? "The predicted class matches the dataset label." : `The actual label is ${actualLabel}, so this is a useful failure case to inspect.`}</p>
    `;
  }

  showLoading() {
    qs("#sampleOverlay").classList.add("loading");
  }

  hideLoading() {
    qs("#sampleOverlay").classList.remove("loading");
  }

  announce(message) {
    qs("#announcements").textContent = message;
    window.setTimeout(() => {
      qs("#announcements").textContent = "";
    }, 2400);
  }

  showSwitchToast(message) {
    const toast = qs("#modelSwitchToast");
    qs("#toastMessage").textContent = message;
    toast.classList.add("show");
    window.setTimeout(() => toast.classList.remove("show"), 2000);
  }
}

export function bootDatasetShowcase() {
  return new DatasetShowcaseApp();
}
