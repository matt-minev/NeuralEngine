export class PredictionPanel {
  constructor({ digitEl, confidenceEl, timeEl, barsRoot, stateEl }) {
    this.digitEl = digitEl;
    this.confidenceEl = confidenceEl;
    this.timeEl = timeEl;
    this.barsRoot = barsRoot;
    this.stateEl = stateEl;
    this.initBars();
  }

  initBars() {
    this.barsRoot.innerHTML = "";
    for (let digit = 0; digit < 10; digit += 1) {
      const row = document.createElement("div");
      row.className = "confidence-bar";
      row.innerHTML = `
        <span class="digit-label">${digit}</span>
        <div class="bar-container"><div class="bar-fill" id="bar-${digit}"></div></div>
        <span class="percentage" id="percent-${digit}">0%</span>
      `;
      this.barsRoot.appendChild(row);
    }
  }

  setLoading() {
    this.digitEl.classList.add("loading");
    this.stateEl.textContent = "Predicting";
  }

  reset() {
    this.digitEl.textContent = "?";
    this.digitEl.dataset.state = "";
    this.digitEl.classList.remove("reveal", "loading");
    this.confidenceEl.textContent = "Confidence: --%";
    this.timeEl.textContent = "--ms";
    this.stateEl.textContent = "Idle";

    for (let digit = 0; digit < 10; digit += 1) {
      document.getElementById(`bar-${digit}`).style.width = "0%";
      document.getElementById(`bar-${digit}`).classList.remove("top-prediction");
      document.getElementById(`percent-${digit}`).textContent = "0%";
    }
  }

  update(result) {
    const { predicted_digit, confidence, predictions, prediction_time } = result;
    this.digitEl.classList.remove("loading");
    this.digitEl.textContent = predicted_digit;
    this.digitEl.classList.add("reveal");

    if (confidence > 80) {
      this.digitEl.dataset.state = "success";
      this.stateEl.textContent = "High confidence";
    } else if (confidence > 60) {
      this.digitEl.dataset.state = "warning";
      this.stateEl.textContent = "Moderate confidence";
    } else {
      this.digitEl.dataset.state = "error";
      this.stateEl.textContent = "Low confidence";
    }

    window.setTimeout(() => this.digitEl.classList.remove("reveal"), 420);

    this.confidenceEl.textContent = `Confidence: ${confidence.toFixed(1)}%`;
    this.timeEl.textContent = `${prediction_time.toFixed(1)}ms`;

    predictions.forEach((probability, digit) => {
      const value = probability * 100;
      const fill = document.getElementById(`bar-${digit}`);
      const percent = document.getElementById(`percent-${digit}`);
      fill.style.width = `${value}%`;
      fill.classList.toggle("top-prediction", digit === predicted_digit);
      percent.textContent = `${value.toFixed(1)}%`;
    });
  }

  flashHint(digit) {
    const previous = this.digitEl.textContent;
    const previousState = this.digitEl.dataset.state;
    this.digitEl.textContent = digit;
    this.digitEl.dataset.state = "success";
    window.setTimeout(() => {
      this.digitEl.textContent = previous;
      this.digitEl.dataset.state = previousState;
    }, 360);
  }
}
