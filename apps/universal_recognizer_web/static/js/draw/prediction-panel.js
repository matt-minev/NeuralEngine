import { qsa, qs } from "../core/utils.js";

export class PredictionPanel {
  constructor() {
    this.currentTab = "all";
    this.lastPrediction = null;

    this.predictedCharacter = qs("#predictedCharacter");
    this.characterType = qs("#characterType");
    this.confidence = qs("#confidence");
    this.predictionTime = qs("#predictionTime");
    this.topPredictions = qs("#topPredictions");

    this.bindTabs();
  }

  bindTabs() {
    const tabButtons = qsa(".tab-btn");
    tabButtons.forEach((button) => {
      button.addEventListener("click", () => {
        tabButtons.forEach((item) => item.classList.remove("active"));
        button.classList.add("active");
        this.currentTab = button.dataset.tab;
        this.updatePredictionsDisplay();
      });
    });
  }

  reset() {
    this.lastPrediction = null;
    if (this.predictedCharacter) this.predictedCharacter.textContent = "?";
    if (this.characterType) this.characterType.textContent = "--";
    if (this.confidence) this.confidence.textContent = "Confidence: --%";
    if (this.predictionTime) this.predictionTime.textContent = "--ms";
    this.updatePredictionsDisplay();
  }

  update(result) {
    this.lastPrediction = result;

    const prediction = result.prediction || result;
    const predictionLatency = result.prediction_time || 0;
    const confidence = prediction.calibrated_confidence !== undefined
      ? prediction.calibrated_confidence
      : prediction.confidence;

    this.predictedCharacter.textContent = prediction.predicted_character;
    this.characterType.textContent = prediction.character_type;
    this.confidence.textContent = `Confidence: ${confidence.toFixed(1)}%`;
    this.predictionTime.textContent = `${predictionLatency.toFixed(0)}ms`;

    this.updatePredictionsDisplay();
  }

  updatePredictionsDisplay() {
    if (!this.topPredictions) {
      return;
    }

    if (!this.lastPrediction) {
      this.topPredictions.innerHTML = '<p class="no-prediction">Draw a character to see predictions</p>';
      return;
    }

    const prediction = this.lastPrediction.prediction || this.lastPrediction;
    if (!prediction) {
      this.topPredictions.innerHTML = '<p class="no-prediction">Draw a character to see predictions</p>';
      return;
    }

    const filtered = this.filterByTab(prediction.top_predictions || []);
    if (filtered.length === 0) {
      this.topPredictions.innerHTML = '<p class="no-prediction">No predictions in this category</p>';
      return;
    }

    this.topPredictions.innerHTML = filtered.map((item) => `
      <div class="prediction-item">
        <div class="prediction-char">${item.character}</div>
        <div class="prediction-type">${item.type}</div>
        <div class="prediction-bar-container">
          <div class="prediction-bar-fill" style="width: ${item.confidence}%"></div>
        </div>
        <div class="prediction-percentage">${item.confidence.toFixed(1)}%</div>
      </div>
    `).join("");
  }

  filterByTab(predictions) {
    if (this.currentTab === "all") {
      return predictions;
    }

    const typeMap = {
      digits: "Digit",
      uppercase: "Uppercase",
      lowercase: "Lowercase",
    };

    return predictions.filter((item) => item.type === typeMap[this.currentTab]);
  }
}
