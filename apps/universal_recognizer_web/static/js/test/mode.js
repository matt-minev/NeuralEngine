import { getRandomTestSamples, predictTestSample } from "../core/api.js";
import { createNotification, preserveScroll, qs, qsa } from "../core/utils.js";

export class TestModeHandler {
  constructor() {
    this.currentView = "single";
    this.currentSamples = [];
  }

  init() {
    qs("#singleViewBtn").addEventListener("click", (event) => {
      event.preventDefault();
      event.stopPropagation();
      this.switchView("single");
    });

    qs("#gridViewBtn").addEventListener("click", (event) => {
      event.preventDefault();
      event.stopPropagation();
      this.switchView("grid");
    });

    qs("#loadTestSamples").addEventListener("click", (event) => {
      event.preventDefault();
      event.stopPropagation();
      this.loadSamples();
    });

    const characterFilter = qs("#characterFilter");
    if (characterFilter) {
      characterFilter.addEventListener("change", (event) => {
        event.preventDefault();
        event.stopPropagation();
        if (this.currentSamples.length > 0) {
          this.loadSamples();
        }
      });
    }
  }

  switchView(view) {
    this.currentView = view;

    qsa(".view-btn").forEach((btn) => btn.classList.remove("active"));
    qs(`#${view}ViewBtn`).classList.add("active");

    qs("#singleTestView").style.display = view === "single" ? "block" : "none";
    qs("#gridTestView").style.display = view === "grid" ? "block" : "none";

    if (view === "grid" && this.currentSamples.length > 0) {
      this.displayGrid();
    } else if (view === "single" && this.currentSamples.length > 0) {
      this.displaySingle(this.currentSamples[0]);
    }
  }

  async loadSamples() {
    const view = this.currentView;
    const count = view === "single" ? 1 : 9;
    const character = qs("#characterFilter").value;

    const loadButton = qs("#loadTestSamples");
    const originalButtonText = loadButton.innerHTML;
    loadButton.disabled = true;
    loadButton.innerHTML = "⏳ Loading...";

    try {
      const data = await preserveScroll(() => getRandomTestSamples(count, character));
      this.currentSamples = data.samples;

      if (view === "single") {
        await this.displaySingle(this.currentSamples[0]);
      } else {
        await this.displayGrid();
      }
    } catch (error) {
      console.error("Error loading test samples:", error);
      this.showNotification("Failed to load test samples. Please try again.", "error");
    } finally {
      loadButton.disabled = false;
      loadButton.innerHTML = originalButtonText;
    }
  }

  showNotification(message, type = "info") {
    const stack = qs("#toastStack") || document.body;
    const notification = createNotification(message, type);
    stack.appendChild(notification);

    setTimeout(() => {
      notification.style.animation = "slideOut 220ms var(--ease-smooth)";
      setTimeout(() => {
        if (notification.parentNode) {
          notification.parentNode.removeChild(notification);
        }
      }, 220);
    }, 3000);
  }

  rotateImage180(imageDataUrl) {
    return new Promise((resolve) => {
      const img = new Image();
      img.onload = () => {
        const canvas = document.createElement("canvas");
        const ctx = canvas.getContext("2d");
        canvas.width = img.width;
        canvas.height = img.height;

        ctx.translate(canvas.width / 2, canvas.height / 2);
        ctx.rotate(Math.PI);
        ctx.translate(-canvas.width / 2, -canvas.height / 2);
        ctx.drawImage(img, 0, 0);

        resolve(canvas.toDataURL("image/png"));
      };
      img.src = imageDataUrl;
    });
  }

  async displaySingle(sample) {
    await preserveScroll(async () => {
      const placeholder = qs("#singleTestPlaceholder");
      const content = qs("#singleTestSample");
      if (placeholder) placeholder.style.display = "none";
      if (content) content.style.display = "grid";

      const rotatedImage = await this.rotateImage180(sample.image_data);
      qs("#testImageSingle").src = rotatedImage;

      qs("#groundTruthSingle").textContent = sample.ground_truth;

      try {
        const data = await predictTestSample(sample.image_array);
        const prediction = data.prediction;

        qs("#predictionSingle").textContent = prediction.character;
        qs("#confidenceSingle").textContent = `${prediction.confidence.toFixed(1)}%`;

        const isCorrect = prediction.character === sample.ground_truth;
        qs("#testStatusSingle").innerHTML = `<span class="status-badge ${isCorrect ? "correct" : "incorrect"}">${isCorrect ? "✓ Correct" : "✗ Incorrect"}</span>`;
      } catch (error) {
        console.error("Error getting prediction:", error);
        qs("#predictionSingle").textContent = "Error";
      }
    });
  }

  async displayGrid() {
    await preserveScroll(async () => {
      const grid = qs("#testGrid");
      const placeholder = qs("#gridTestPlaceholder");

      if (placeholder) {
        placeholder.style.display = "none";
      }

      grid.innerHTML = '<div class="test-grid-loading">Loading predictions...</div>';

      const predictions = await Promise.all(
        this.currentSamples.map(async (sample) => {
          try {
            const data = await predictTestSample(sample.image_array);
            return { sample, prediction: data.prediction };
          } catch (error) {
            console.error("Error getting prediction:", error);
            return { sample, prediction: null };
          }
        })
      );

      const rotatedImages = await Promise.all(
        predictions.map(({ sample }) => this.rotateImage180(sample.image_data))
      );

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
                <div class="test-grid-label">Truth: <strong>${sample.ground_truth}</strong></div>
                <div class="test-grid-label">Pred: <strong>${prediction.character}</strong></div>
                <div class="test-grid-label">Conf: ${prediction.confidence.toFixed(1)}%</div>
                <div class="test-grid-status">${isCorrect ? "✓" : "✗"}</div>
              </div>
            </div>
          `;
        })
        .join("");
    });
  }
}
