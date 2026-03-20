import { predictDigit, switchModel } from "../core/api.js";
import { debounce, showToast, qs } from "../core/utils.js";
import { CanvasDrawer } from "./canvas-drawer.js";
import { PredictionPanel } from "./prediction-panel.js";
import { HistoryPanel } from "./history-panel.js";
import { TutorialOverlay } from "./tutorial.js";
import { createParticles, triggerPiCelebration } from "./effects.js";

const MODEL_LABELS = {
  "enhanced_digit_model.pkl": "Enhanced Model",
  "digit_model_bulletproof.pkl": "Bulletproof Model",
  "digit_model_optimized.pkl": "Optimized Model",
  "digit_model.pkl": "Basic Model",
};

class DigitRecognizerApp {
  constructor() {
    this.predictionCache = new Map();
    this.sequenceTracker = [];
    this.targetSequence = [3, 1, 4];

    this.canvasDrawer = new CanvasDrawer({
      canvas: qs("#drawingCanvas"),
      overlay: qs(".canvas-overlay"),
      onStroke: debounce(() => this.predict(), 280),
      onCommit: () => this.predict(),
    });

    this.predictionPanel = new PredictionPanel({
      digitEl: qs("#predictedDigit"),
      confidenceEl: qs("#confidence"),
      timeEl: qs("#predictionTime"),
      barsRoot: qs("#confidenceBars"),
      stateEl: qs("#predictionState"),
    });

    this.historyPanel = new HistoryPanel(qs("#historyItems"));
    this.tutorialOverlay = new TutorialOverlay();

    this.bindControls();
    this.tutorialOverlay.maybeShow();
  }

  bindControls() {
    qs("#clearBtn").addEventListener("click", () => this.clear());

    qs("#modelSelect").addEventListener("change", async (event) => {
      const modelName = event.target.value;

      try {
        const result = await switchModel(modelName);
        this.updateModelInfo(result.model_info, modelName);
        this.clear();
        showToast(`Switched to ${MODEL_LABELS[modelName] || modelName}`, "success");
      } catch (error) {
        showToast(error.message, "error");
      }
    });

    document.addEventListener("keydown", (event) => {
      if (event.target.matches("input, select, textarea")) {
        return;
      }

      if (event.key === "c" || event.key === "C") {
        event.preventDefault();
        this.clear();
      }

      if (event.key >= "0" && event.key <= "9") {
        this.predictionPanel.flashHint(event.key);
      }

      if (event.key === "Escape") {
        this.tutorialOverlay.close();
      }
    });
  }

  clear() {
    this.canvasDrawer.clear();
    this.predictionCache.clear();
    this.predictionPanel.reset();
  }

  async predict() {
    if (!this.canvasDrawer.hasInk()) {
      this.predictionPanel.reset();
      return;
    }

    const image = this.canvasDrawer.getPredictionPayload();
    const cacheKey = JSON.stringify(image.strokes);

    if (this.predictionCache.has(cacheKey)) {
      this.applyPrediction(this.predictionCache.get(cacheKey));
      return;
    }

    this.predictionPanel.setLoading();

    try {
      const result = await predictDigit(image);
      this.predictionCache.set(cacheKey, result);
      if (this.predictionCache.size > 50) {
        const oldestKey = this.predictionCache.keys().next().value;
        this.predictionCache.delete(oldestKey);
      }

      this.applyPrediction(result);
    } catch (error) {
      showToast(error.message, "error");
    }
  }

  applyPrediction(result) {
    this.predictionPanel.update(result);
    this.historyPanel.add(result.predicted_digit, result.confidence);
    this.trackSequence(result.predicted_digit);

    if (result.confidence > 95) {
      createParticles(qs("#predictedDigit"));
    }
  }

  trackSequence(digit) {
    if (this.sequenceTracker.length === 0 || this.sequenceTracker[this.sequenceTracker.length - 1] !== digit) {
      this.sequenceTracker.push(digit);
    }

    if (this.sequenceTracker.length > this.targetSequence.length) {
      this.sequenceTracker.shift();
    }

    const match = this.targetSequence.every((value, index) => this.sequenceTracker[index] === value);
    if (match) {
      triggerPiCelebration();
      this.sequenceTracker = [];
    }
  }

  updateModelInfo(modelInfo, modelName) {
    qs("#architectureValue").textContent = modelInfo.architecture.join(" → ");
    qs("#parametersValue").textContent = modelInfo.parameters.toLocaleString();
    qs("#accuracyValue").textContent = `${modelInfo.accuracy.toFixed(2)}%`;
    qs("#currentModelLabel").textContent = MODEL_LABELS[modelName] || modelName;
  }
}

export function bootDigitRecognizerApp() {
  return new DigitRecognizerApp();
}
