import { getRandomTestSamples, predictTestSample } from "../core/api.js";
import { createNotification, preserveScroll, qs, qsa } from "../core/utils.js";

// Allow list of confusable character pairs — mistakes between these
// are expected and shown in yellow instead of red.
const CONFUSABLE_PAIRS = new Set([
  // Numbers vs Letters
  "O|0", "0|O",
  "I|l", "l|I",
  "I|1", "1|I",
  "l|1", "1|l",
  "q|9", "9|q",
  "g|9", "9|g",
  "c|C", "C|c",
  "k|K", "K|k",
  "m|M", "M|m",
  "o|O", "O|o",
  "p|P", "P|p",
  "s|S", "S|s",
  "u|U", "U|u",
  "v|V", "V|v",
  "w|W", "W|w",
  "x|X", "X|x",
  "y|Y", "Y|y",
  "z|Z", "Z|z",
  "q|g", "g|q",
]);

function classifyResult(predicted, groundTruth) {
  if (predicted === groundTruth) return "correct";
  if (CONFUSABLE_PAIRS.has(`${predicted}|${groundTruth}`)) return "confusable";
  return "incorrect";
}

function statusLabel(cls) {
  if (cls === "correct") return "✓ Correct";
  if (cls === "confusable") return "~ Confusable";
  return "✗ Incorrect";
}

function statusIcon(cls) {
  if (cls === "correct") return "✓";
  if (cls === "confusable") return "~";
  return "✗";
}

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

    // Show loading overlay on existing content instead of wiping it
    this._showLoadingOverlay();

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
      this._hideLoadingOverlay();
      loadButton.disabled = false;
      loadButton.innerHTML = originalButtonText;
    }
  }

  _showLoadingOverlay() {
    const containers = [qs("#singleTestContent"), qs("#testGrid")];
    containers.forEach((container) => {
      if (!container) return;
      if (container.querySelector(".test-loading-overlay")) return;
      const overlay = document.createElement("div");
      overlay.className = "test-loading-overlay";
      overlay.innerHTML = '<div class="test-loading-spinner"></div>';
      container.style.position = "relative";
      container.appendChild(overlay);
    });
  }

  _hideLoadingOverlay() {
    qsa(".test-loading-overlay").forEach((el) => {
      el.classList.add("fade-out");
      setTimeout(() => el.remove(), 200);
    });
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

  async displaySingle(sample) {
    await preserveScroll(async () => {
      const placeholder = qs("#singleTestPlaceholder");
      const content = qs("#singleTestSample");
      if (placeholder) placeholder.style.display = "none";
      if (content) content.style.display = "grid";

      qs("#testImageSingle").src = sample.image_data;
      qs("#groundTruthSingle").textContent = sample.ground_truth;

      try {
        const data = await predictTestSample(sample.image_array);
        const prediction = data.prediction;

        qs("#predictionSingle").textContent = prediction.character;
        qs("#confidenceSingle").textContent = `${prediction.confidence.toFixed(1)}%`;

        const cls = classifyResult(prediction.character, sample.ground_truth);
        qs("#testStatusSingle").innerHTML = `<span class="status-badge ${cls}">${statusLabel(cls)}</span>`;
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

      grid.innerHTML = predictions
        .map(({ sample, prediction }) => {
          if (!prediction) {
            return '<div class="test-grid-item error">Error</div>';
          }

          const cls = classifyResult(prediction.character, sample.ground_truth);

          return `
            <div class="test-grid-item ${cls}">
              <img src="${sample.image_data}" alt="Test sample" />
              <div class="test-grid-info">
                <div class="test-grid-label">Truth: <strong>${sample.ground_truth}</strong></div>
                <div class="test-grid-label">Pred: <strong>${prediction.character}</strong></div>
                <div class="test-grid-label">Conf: ${prediction.confidence.toFixed(1)}%</div>
                <div class="test-grid-status">${statusIcon(cls)}</div>
              </div>
            </div>
          `;
        })
        .join("");
    });
  }
}
