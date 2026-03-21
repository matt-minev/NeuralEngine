import { predictCharacter } from "../core/api.js";
import { qs } from "../core/utils.js";
import { CanvasDrawer } from "./canvas-drawer.js";
import { PredictionPanel } from "./prediction-panel.js";
import { renderDebugImages } from "./debug-panel.js";

export class UniversalDrawApp {
  constructor({ accessibilityPanel }) {
    this.accessibilityPanel = accessibilityPanel;

    this.canvasDrawer = new CanvasDrawer({
      canvas: qs("#drawingCanvas"),
      overlay: qs("#canvasOverlay"),
    });

    this.predictionPanel = new PredictionPanel();

    this.bindControls();
    this.setupAutoPredict();
  }

  bindControls() {
    qs("#clearBtn").addEventListener("click", () => {
      this.clear();
    });

    qs("#predictBtn").addEventListener("click", () => {
      this.predict();
    });
  }

  setupAutoPredict() {
    let predictionTimeout;
    this.canvasDrawer.canvas.addEventListener("mouseup", () => {
      clearTimeout(predictionTimeout);
      predictionTimeout = setTimeout(() => {
        if (this.canvasDrawer.hasDrawing()) {
          this.predict();
        }
      }, 500);
    });
  }

  clear() {
    this.canvasDrawer.clear();
    this.predictionPanel.reset();
    this.accessibilityPanel.update(null);
  }

  async predict() {
    if (!this.canvasDrawer.hasDrawing()) {
      return;
    }

    try {
      const debugEnabled = qs("#debugPanelContent")?.style.display !== "none";
      const result = await predictCharacter(this.canvasDrawer.getPayload(), debugEnabled);

      this.predictionPanel.update(result);

      if (result.debug_images) {
        renderDebugImages(result.debug_images, result.debug_images.stats);
      }

      this.accessibilityPanel.update(result);
    } catch (error) {
      console.error("Prediction error:", error);
      alert("Failed to get prediction. Please try again.");
    }
  }
}
