/**
 * Quadratic Neural Network Web Application
 * Prediction Interface Handler
 *
 * Author: Matt
 * Location: Varna, Bulgaria
 * Date: July 2025
 *
 * Specialized JavaScript for prediction interface interactions
 */

// Prediction interface management
const PredictionManager = {
  // Current prediction state
  currentScenario: null,
  lastPrediction: null,
  predictionHistory: [],

  // UI elements
  scenarioSelect: null,
  inputContainer: null,
  resultsContainer: null,
  predictButton: null,

  // Initialize prediction interface
  init() {
    this.scenarioSelect = document.getElementById("prediction-scenario");
    this.inputContainer = document.getElementById("prediction-inputs");
    this.resultsContainer = document.getElementById("prediction-results");
    this.predictButton = document.getElementById("predict-btn");

    // Set up event listeners
    this.setupEventListeners();

    // Load scenarios
    this.loadScenarios();

    console.log("🎯 Prediction interface initialized");
  },

  // Setup event listeners
  setupEventListeners() {
    if (this.scenarioSelect) {
      this.scenarioSelect.addEventListener("change", (e) => {
        this.handleScenarioChange(e.target.value);
      });
    }

    // Keyboard shortcuts
    document.addEventListener("keydown", (e) => {
      if (e.ctrlKey && e.key === "Enter") {
        e.preventDefault();
        this.makePrediction();
      }
    });

    // Auto-prediction on input change (debounced)
    this.debouncedPredict = this.debounce(this.autoPredict.bind(this), 1000);
  },

  // Load available scenarios
  async loadScenarios() {
    try {
      const response = await fetch("/api/scenarios");
      const scenarios = await response.json();

      this.populateScenarioSelect(scenarios);

      // Set default scenario
      const firstScenario = Object.keys(scenarios)[0];
      if (firstScenario) {
        this.handleScenarioChange(firstScenario);
      }
    } catch (error) {
      console.error("Failed to load scenarios:", error);
      this.showError("Failed to load prediction scenarios");
    }
  },

  // Populate scenario select dropdown
  populateScenarioSelect(scenarios) {
    if (!this.scenarioSelect) return;

    this.scenarioSelect.innerHTML = "";

    Object.entries(scenarios).forEach(([key, scenario]) => {
      const option = document.createElement("option");
      option.value = key;
      option.textContent = `${scenario.name} - ${scenario.description}`;
      this.scenarioSelect.appendChild(option);
    });
  },

  // Handle scenario change
  handleScenarioChange(scenarioKey) {
    if (!AppState.scenarios[scenarioKey]) {
      console.error("Invalid scenario:", scenarioKey);
      return;
    }

    this.currentScenario = AppState.scenarios[scenarioKey];
    this.createInputFields();
    this.clearResults();

    // Update UI based on scenario
    this.updateScenarioInfo();
  },

  // Create input fields for current scenario
  createInputFields() {
    if (!this.inputContainer || !this.currentScenario) return;

    this.inputContainer.innerHTML = "";

    this.currentScenario.input_features.forEach((feature, index) => {
      const inputGroup = this.createInputGroup(feature, index);
      this.inputContainer.appendChild(inputGroup);
    });

    // Add animation
    this.inputContainer.classList.add("fade-in");
  },

  // Create individual input group
  createInputGroup(feature, index) {
    const group = document.createElement("div");
    group.className = "prediction-input-group";

    const label = document.createElement("label");
    label.className = "prediction-input-label";
    label.textContent = this.getFeatureLabel(feature);
    label.setAttribute("for", `input-${feature}`);

    const input = document.createElement("input");
    input.type = "number";
    input.step = "any";
    input.id = `input-${feature}`;
    input.className = "prediction-input";
    input.placeholder = this.getFeaturePlaceholder(feature);

    // Add input validation
    input.addEventListener("input", (e) => {
      this.validateInput(e.target, feature);
      this.debouncedPredict();
    });

    // Add tooltip
    const tooltip = this.createTooltip(feature);

    group.appendChild(label);
    group.appendChild(input);
    group.appendChild(tooltip);

    return group;
  },

  // Get feature label with icons
  getFeatureLabel(feature) {
    const labels = {
      a: "📊 Coefficient a",
      b: "📈 Coefficient b",
      c: "📉 Coefficient c",
      x1: "🎯 Root x₁",
      x2: "🎯 Root x₂",
    };

    return labels[feature] || feature;
  },

  // Get feature placeholder
  getFeaturePlaceholder(feature) {
    const placeholders = {
      a: "Enter coefficient a (≠ 0)",
      b: "Enter coefficient b",
      c: "Enter coefficient c",
      x1: "Enter first root",
      x2: "Enter second root",
    };

    return placeholders[feature] || `Enter ${feature} value`;
  },

  // Create tooltip for feature
  createTooltip(feature) {
    const tooltip = document.createElement("div");
    tooltip.className = "prediction-tooltip";

    const tooltipTexts = {
      a: "The coefficient of x² term. Cannot be zero for quadratic equations.",
      b: "The coefficient of x term. Can be any real number.",
      c: "The constant term. Can be any real number.",
      x1: "The first root of the quadratic equation.",
      x2: "The second root of the quadratic equation.",
    };

    tooltip.innerHTML = `
            <div class="tooltip-icon">ℹ️</div>
            <div class="tooltip-text">${
              tooltipTexts[feature] || "Parameter description"
            }</div>
        `;

    return tooltip;
  },

  // Validate input
  validateInput(input, feature) {
    const value = parseFloat(input.value);
    let isValid = true;
    let message = "";

    if (isNaN(value)) {
      isValid = false;
      message = "Please enter a valid number";
    } else if (feature === "a" && Math.abs(value) < 1e-10) {
      isValid = false;
      message = 'Coefficient "a" cannot be zero';
    } else if (Math.abs(value) > 1000) {
      isValid = false;
      message = "Value too large (max: 1000)";
    }

    // Update UI
    if (isValid) {
      input.classList.remove("error");
      input.classList.add("valid");
    } else {
      input.classList.add("error");
      input.classList.remove("valid");
    }

    // Show/hide error message
    this.showInputError(input, message);

    return isValid;
  },

  // Show input error
  showInputError(input, message) {
    const group = input.closest(".prediction-input-group");
    let errorDiv = group.querySelector(".input-error");

    if (message) {
      if (!errorDiv) {
        errorDiv = document.createElement("div");
        errorDiv.className = "input-error";
        group.appendChild(errorDiv);
      }
      errorDiv.textContent = message;
      errorDiv.style.display = "block";
    } else if (errorDiv) {
      errorDiv.style.display = "none";
    }
  },

  // Update scenario information display
  updateScenarioInfo() {
    if (!this.currentScenario) return;

    // Update scenario description
    const infoElement = document.querySelector(".prediction-scenario-info");
    if (infoElement) {
      infoElement.innerHTML = `
                <div class="scenario-info-card">
                    <h3 style="color: ${this.currentScenario.color}">${
        this.currentScenario.name
      }</h3>
                    <p>${this.currentScenario.description}</p>
                    <div class="scenario-details">
                        <span><strong>Input:</strong> ${this.currentScenario.input_features.join(
                          ", "
                        )}</span>
                        <span><strong>Output:</strong> ${this.currentScenario.target_features.join(
                          ", "
                        )}</span>
                    </div>
                </div>
            `;
    }
  },

  // Make prediction
  async makePrediction() {
    if (!this.currentScenario) {
      this.showError("Please select a prediction scenario");
      return;
    }

    // Validate all inputs
    const inputs = this.getInputValues();
    if (!inputs) {
      this.showError("Please enter valid values for all inputs");
      return;
    }

    // Show loading state
    this.showLoading(true);

    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          scenario: this.scenarioSelect.value,
          inputs: inputs,
        }),
      });

      const result = await response.json();

      if (result.success) {
        this.displayPredictionResults(result);
        this.storePredictionHistory(inputs, result);
      } else {
        this.showError(result.error || "Prediction failed");
      }
    } catch (error) {
      console.error("Prediction error:", error);
      this.showError("Failed to make prediction. Please try again.");
    } finally {
      this.showLoading(false);
    }
  },

  // Get input values
  getInputValues() {
    if (!this.currentScenario) return null;

    const inputs = [];
    let allValid = true;

    this.currentScenario.input_features.forEach((feature) => {
      const input = document.getElementById(`input-${feature}`);
      if (input) {
        const value = parseFloat(input.value);
        if (isNaN(value) || !this.validateInput(input, feature)) {
          allValid = false;
          return;
        }
        inputs.push(value);
      } else {
        allValid = false;
      }
    });

    return allValid ? inputs : null;
  },

  // Display prediction results
  displayPredictionResults(result) {
    if (!this.resultsContainer) return;

    this.lastPrediction = result;

    // Create results HTML
    const resultsHTML = this.createResultsHTML(result);

    // Display with animation
    this.resultsContainer.innerHTML = resultsHTML;
    this.resultsContainer.classList.add("fade-in");

    // Add interactive elements
    this.setupResultInteractions();
  },

  // Create results HTML
  createResultsHTML(result) {
    let html = `
            <div class="prediction-results-header">
                <h3>🎯 Prediction Results</h3>
                <div class="scenario-badge" style="background-color: ${this.currentScenario.color}20; color: ${this.currentScenario.color};">
                    ${result.scenario}
                </div>
            </div>
            
            <div class="prediction-results-grid">
        `;

    // Add prediction results
    result.predictions.forEach((prediction, index) => {
      const feature = result.target_features[index];
      const confidence = result.confidences ? result.confidences[index] : null;

      html += this.createPredictionCard(feature, prediction, confidence, index);
    });

    html += "</div>";

    // Add actual solutions comparison if available
    if (result.actual_solutions) {
      html += this.createComparisonSection(result);
    }

    // Add equation verification if applicable
    if (this.currentScenario.name === "Coefficients → Roots") {
      html += this.createEquationVerification(result);
    }

    return html;
  },

  // Create prediction card
  createPredictionCard(feature, prediction, confidence, index) {
    const confidencePercent = confidence
      ? (confidence * 100).toFixed(1)
      : "N/A";
    const confidenceLevel = confidence
      ? this.getConfidenceLevel(confidence)
      : "";
    const confidenceClass =
      confidence > 0.8 ? "high" : confidence > 0.6 ? "medium" : "low";

    return `
            <div class="prediction-card ${confidenceClass}">
                <div class="prediction-card-header">
                    <span class="feature-name">${feature}</span>
                    <span class="confidence-indicator ${confidenceClass}">
                        ${confidenceLevel}
                    </span>
                </div>
                <div class="prediction-value">
                    ${this.formatNumber(prediction, 6)}
                </div>
                <div class="prediction-confidence">
                    Confidence: ${confidencePercent}%
                </div>
                <div class="prediction-actions">
                    <button class="btn-small" onclick="PredictionManager.copyValue('${prediction}')">
                        📋 Copy
                    </button>
                </div>
            </div>
        `;
  },

  // Create comparison section
  createComparisonSection(result) {
    let html = `
            <div class="comparison-section">
                <h4>📊 Comparison with Actual Solutions</h4>
                <div class="comparison-grid">
        `;

    result.predictions.forEach((prediction, index) => {
      const feature = result.target_features[index];
      const actual = result.actual_solutions[index];
      const error = Math.abs(prediction - actual);
      const errorPercent = Math.abs(error / (actual + 1e-8)) * 100;

      const accuracyClass =
        error < 0.01
          ? "excellent"
          : error < 0.1
          ? "good"
          : error < 1.0
          ? "moderate"
          : "poor";

      html += `
                <div class="comparison-card ${accuracyClass}">
                    <div class="comparison-feature">${feature}</div>
                    <div class="comparison-values">
                        <div class="comparison-row">
                            <span>Predicted:</span>
                            <span class="value">${this.formatNumber(
                              prediction,
                              6
                            )}</span>
                        </div>
                        <div class="comparison-row">
                            <span>Actual:</span>
                            <span class="value">${this.formatNumber(
                              actual,
                              6
                            )}</span>
                        </div>
                        <div class="comparison-row error-row">
                            <span>Error:</span>
                            <span class="error-value">${this.formatNumber(
                              error,
                              6
                            )} (${errorPercent.toFixed(2)}%)</span>
                        </div>
                    </div>
                    <div class="accuracy-badge ${accuracyClass}">
                        ${this.getAccuracyLabel(error)}
                    </div>
                </div>
            `;
    });

    html += "</div></div>";
    return html;
  },

  // Create equation verification
  createEquationVerification(result) {
    const inputs = this.getInputValues();
    if (!inputs || inputs.length !== 3) return "";

    const [a, b, c] = inputs;
    const [x1, x2] = result.predictions;

    // Calculate equation errors
    const error1 = Math.abs(a * x1 * x1 + b * x1 + c);
    const error2 = Math.abs(a * x2 * x2 + b * x2 + c);

    const maxError = Math.max(error1, error2);
    const verificationClass =
      maxError < 0.001
        ? "perfect"
        : maxError < 0.01
        ? "good"
        : maxError < 1.0
        ? "moderate"
        : "poor";

    return `
            <div class="equation-verification">
                <h4>🔍 Equation Verification</h4>
                <div class="equation-display">
                    ${this.formatNumber(a, 3)}x² + ${this.formatNumber(
      b,
      3
    )}x + ${this.formatNumber(c, 3)} = 0
                </div>
                <div class="verification-results ${verificationClass}">
                    <div class="verification-test">
                        <span>x₁ = ${this.formatNumber(x1, 6)}</span>
                        <span>Error: ${this.formatNumber(error1, 8)}</span>
                    </div>
                    <div class="verification-test">
                        <span>x₂ = ${this.formatNumber(x2, 6)}</span>
                        <span>Error: ${this.formatNumber(error2, 8)}</span>
                    </div>
                    <div class="verification-status">
                        ${this.getVerificationStatus(maxError)}
                    </div>
                </div>
            </div>
        `;
  },

  // Setup result interactions
  setupResultInteractions() {
    // Add event listeners for interactive elements
    const copyButtons = this.resultsContainer.querySelectorAll(".btn-small");
    copyButtons.forEach((button) => {
      button.addEventListener("click", (e) => {
        e.stopPropagation();
      });
    });

    // Add hover effects
    const predictionCards =
      this.resultsContainer.querySelectorAll(".prediction-card");
    predictionCards.forEach((card) => {
      card.addEventListener("mouseenter", () => {
        card.style.transform = "translateY(-2px)";
      });

      card.addEventListener("mouseleave", () => {
        card.style.transform = "translateY(0)";
      });
    });
  },

  // Store prediction in history
  storePredictionHistory(inputs, result) {
    const historyEntry = {
      timestamp: new Date().toISOString(),
      scenario: this.currentScenario.name,
      inputs: inputs.slice(),
      predictions: result.predictions.slice(),
      confidences: result.confidences ? result.confidences.slice() : null,
    };

    this.predictionHistory.push(historyEntry);

    // Keep only last 50 predictions
    if (this.predictionHistory.length > 50) {
      this.predictionHistory = this.predictionHistory.slice(-50);
    }

    // Update history display if visible
    this.updateHistoryDisplay();
  },

  // Auto-prediction (for real-time updates)
  autoPredict() {
    if (!this.isAutoPreviewEnabled()) return;

    const inputs = this.getInputValues();
    if (
      inputs &&
      inputs.length === this.currentScenario.input_features.length
    ) {
      this.makePrediction();
    }
  },

  // Check if auto-preview is enabled
  isAutoPreviewEnabled() {
    return false; // Disabled by default for performance
  },

  // Show loading state
  showLoading(show) {
    if (this.predictButton) {
      if (show) {
        this.predictButton.innerHTML =
          '<div class="loading-spinner"></div> Predicting...';
        this.predictButton.disabled = true;
      } else {
        this.predictButton.innerHTML = "🔮 Make Prediction";
        this.predictButton.disabled = false;
      }
    }
  },

  // Show error message
  showError(message) {
    Utils.showNotification(message, "error");
  },

  // Clear results
  clearResults() {
    if (this.resultsContainer) {
      this.resultsContainer.innerHTML = `
                <div class="no-results">
                    <div class="no-results-icon">🎯</div>
                    <div class="no-results-text">Make a prediction to see results here</div>
                </div>
            `;
    }
  },

  // Copy value to clipboard
  copyValue(value) {
    navigator.clipboard
      .writeText(value)
      .then(() => {
        Utils.showNotification("Value copied to clipboard", "success");
      })
      .catch(() => {
        Utils.showNotification("Failed to copy value", "error");
      });
  },

  // Utility functions
  formatNumber(num, decimals = 6) {
    return Utils.formatNumber(num, decimals);
  },

  getConfidenceLevel(confidence) {
    if (confidence > 0.8) return "🟢 High";
    if (confidence > 0.6) return "🟡 Medium";
    return "🔴 Low";
  },

  getAccuracyLabel(error) {
    if (error < 0.01) return "🎉 Excellent";
    if (error < 0.1) return "👍 Good";
    if (error < 1.0) return "⚠️ Moderate";
    return "❌ Poor";
  },

  getVerificationStatus(error) {
    if (error < 0.001) return "✅ Perfect - Solutions satisfy the equation";
    if (error < 0.01) return "👍 Good - Small verification error";
    if (error < 1.0) return "⚠️ Moderate - Noticeable verification error";
    return "❌ Poor - Solutions do not satisfy the equation";
  },

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
  },

  // Update history display
  updateHistoryDisplay() {
    const historyContainer = document.getElementById("prediction-history");
    if (!historyContainer) return;

    if (this.predictionHistory.length === 0) {
      historyContainer.innerHTML = "<p>No prediction history yet</p>";
      return;
    }

    const historyHTML = this.predictionHistory
      .slice(-10)
      .reverse()
      .map((entry) => {
        const date = new Date(entry.timestamp);
        return `
                <div class="history-entry">
                    <div class="history-header">
                        <span class="history-scenario">${entry.scenario}</span>
                        <span class="history-time">${date.toLocaleTimeString()}</span>
                    </div>
                    <div class="history-details">
                        <span>Inputs: ${entry.inputs
                          .map((x) => this.formatNumber(x, 3))
                          .join(", ")}</span>
                        <span>Predictions: ${entry.predictions
                          .map((x) => this.formatNumber(x, 3))
                          .join(", ")}</span>
                    </div>
                </div>
            `;
      })
      .join("");

    historyContainer.innerHTML = historyHTML;
  },

  // Export predictions
  exportPredictions() {
    if (this.predictionHistory.length === 0) {
      Utils.showNotification("No predictions to export", "warning");
      return;
    }

    const csvData = this.generateCSV();
    const blob = new Blob([csvData], { type: "text/csv" });
    const url = URL.createObjectURL(blob);

    const link = document.createElement("a");
    link.href = url;
    link.download = `predictions_${new Date().toISOString().split("T")[0]}.csv`;
    link.click();

    URL.revokeObjectURL(url);
    Utils.showNotification("Predictions exported successfully", "success");
  },

  // Generate CSV data
  generateCSV() {
    const headers = [
      "Timestamp",
      "Scenario",
      "Inputs",
      "Predictions",
      "Confidences",
    ];
    let csv = headers.join(",") + "\n";

    this.predictionHistory.forEach((entry) => {
      const row = [
        entry.timestamp,
        entry.scenario,
        entry.inputs.join(";"),
        entry.predictions.join(";"),
        entry.confidences ? entry.confidences.join(";") : "N/A",
      ];
      csv += row.join(",") + "\n";
    });

    return csv;
  },
};

// Prediction examples for testing
const PredictionExamples = {
  examples: {
    coeff_to_roots: [
      { inputs: [1, -3, 2], description: "Simple quadratic: x² - 3x + 2 = 0" },
      { inputs: [1, 0, -4], description: "No linear term: x² - 4 = 0" },
      { inputs: [2, -4, 2], description: "Perfect square: 2x² - 4x + 2 = 0" },
    ],
    partial_coeff_to_missing: [
      { inputs: [1, -3, 1], description: "Given a=1, b=-3, x1=1" },
      { inputs: [2, -2, 2], description: "Given a=2, b=-2, x1=2" },
    ],
    roots_to_coeff: [
      { inputs: [1, 2], description: "Roots: x1=1, x2=2" },
      { inputs: [-1, 3], description: "Roots: x1=-1, x2=3" },
    ],
  },

  // Load example into inputs
  loadExample(scenarioKey, exampleIndex) {
    const examples = this.examples[scenarioKey];
    if (!examples || !examples[exampleIndex]) return;

    const example = examples[exampleIndex];
    const scenario = AppState.scenarios[scenarioKey];

    if (!scenario) return;

    // Fill input fields
    example.inputs.forEach((value, index) => {
      const feature = scenario.input_features[index];
      const input = document.getElementById(`input-${feature}`);
      if (input) {
        input.value = value;
        // Trigger validation
        input.dispatchEvent(new Event("input"));
      }
    });

    Utils.showNotification(`Loaded example: ${example.description}`, "info");
  },

  // Get examples for current scenario
  getExamples(scenarioKey) {
    return this.examples[scenarioKey] || [];
  },
};

// Initialize prediction manager when DOM is loaded
document.addEventListener("DOMContentLoaded", () => {
  // Initialize prediction manager
  PredictionManager.init();

  // Make available globally
  window.PredictionManager = PredictionManager;
  window.PredictionExamples = PredictionExamples;
});

// Global functions for HTML onclick events
window.makePrediction = () => PredictionManager.makePrediction();
window.clearPredictionResults = () => PredictionManager.clearResults();
window.exportPredictions = () => PredictionManager.exportPredictions();
window.loadExample = (scenarioKey, exampleIndex) =>
  PredictionExamples.loadExample(scenarioKey, exampleIndex);
