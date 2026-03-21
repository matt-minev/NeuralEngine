// Global functions for HTML onclick events
function getRandomButtonMarkup(labelText = "RANDOM", iconMarkup = "🎲") {
  return `
    <span class="btn-attention-icon" aria-hidden="true">${iconMarkup}</span>
    <span class="random-btn-label">${labelText}</span>
  `;
}

function uploadDataset() {
  const fileInput = document.getElementById("file-input");
  fileInput.onchange = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    if (!file.name.endsWith(".csv")) {
      Utils.showNotification("Please select a CSV file", "error");
      return;
    }

    try {
      Utils.showNotification("Uploading dataset...", "info");
      const result = await ApiClient.uploadFile(file);

      if (result.success) {
        Utils.showNotification(
          `Successfully loaded ${result.message}`,
          "success"
        );
        AppState.dataLoaded = true;
        DataSection.refresh();
      } else {
        Utils.showNotification(result.error || "Upload failed", "error");
      }
    } catch (error) {
      Utils.showNotification("Upload failed: " + error.message, "error");
    }
  };
  fileInput.click();
}

async function startTraining() {
  if (!AppState.dataLoaded) {
    try {
      const dataInfo = await ApiClient.request(API.dataInfo);

      if (!dataInfo.loaded) {
        Utils.showNotification("Please load a dataset first", "warning");
        return;
      }

      updateDataStatusDisplay(dataInfo);
    } catch (error) {
      Utils.showNotification("Please load a dataset first", "warning");
      return;
    }
  }

  const selectedScenarios = Array.from(
    document.querySelectorAll(
      '#scenarios-selection input[type="checkbox"]:checked'
    )
  ).map((cb) => cb.value);

  if (selectedScenarios.length === 0) {
    Utils.showNotification("Please select at least one scenario", "warning");
    return;
  }

  const epochs =
    parseInt(document.getElementById("epochs-input").value) || 1000;
  const learningRate =
    parseFloat(document.getElementById("learning-rate-input").value) || 0.001;

  try {
    const response = await ApiClient.request(API.startTraining, {
      method: "POST",
      body: JSON.stringify({
        scenarios: selectedScenarios,
        epochs: epochs,
        learning_rate: learningRate,
      }),
    });

    if (response.success) {
      AppState.isTraining = true;
      document.getElementById("start-training-btn").innerHTML =
        '<i class="loading-spinner"></i> Training...';
      document.getElementById("start-training-btn").disabled = true;
      document.getElementById("stop-training-btn").style.display =
        "inline-block";

      Utils.showNotification(
        `Training started: ${selectedScenarios.length} scenarios, ${epochs} epochs, learning rate: ${learningRate}`,
        "success"
      );
    }
  } catch (error) {
    Utils.showNotification(
      "Failed to start training: " + error.message,
      "error"
    );
  }
}

async function stopTraining() {
  if (!AppState.isTraining) {
    Utils.showNotification(
      "No training session is currently active",
      "warning"
    );
    return;
  }

  try {
    const response = await ApiClient.request(API.stopTraining, {
      method: "POST",
    });

    if (response.success) {
      AppState.isTraining = false;
      document.getElementById("start-training-btn").innerHTML =
        '<i class="fas fa-play"></i> Start Training';
      document.getElementById("start-training-btn").disabled = false;
      document.getElementById("stop-training-btn").style.display = "none";
      Utils.showNotification("Training stopped successfully", "success");
    } else {
      Utils.showNotification(
        response.error || "Failed to stop training",
        "error"
      );
    }
  } catch (error) {
    Utils.showNotification(
      "Failed to stop training: " + error.message,
      "error"
    );
  }
}

async function makePrediction() {
  const scenario = document.getElementById("prediction-scenario").value;
  const scenarioData = AppState.scenarios[scenario];

  if (!scenarioData) {
    Utils.showNotification("Please select a scenario", "warning");
    return;
  }

  // Get input values
  const inputs = [];
  for (const feature of scenarioData.input_features) {
    const input = document.getElementById(`input-${feature}`);
    if (!input || input.value === "") {
      Utils.showNotification(`Please enter a value for ${feature}`, "warning");
      return;
    }
    inputs.push(parseFloat(input.value));
  }

  try {
    const response = await ApiClient.request(API.predict, {
      method: "POST",
      body: JSON.stringify({
        scenario: scenario,
        inputs: inputs,
      }),
    });

    if (response.success) {
      displayPredictionResults(response, inputs);
    } else {
      Utils.showNotification(response.error || "Prediction failed", "error");
    }
  } catch (error) {
    // Check if error message indicates model not trained/loaded
    const errorMsg = error.message.toLowerCase();
    if (
      errorMsg.includes("not trained") ||
      errorMsg.includes("not loaded") ||
      errorMsg.includes("400")
    ) {
      const scenarioName = scenarioData ? scenarioData.name : scenario;
      Utils.showNotification(
        `Model not loaded for "${scenarioName}". Please train a model or load a saved model first.`,
        "error"
      );
    } else {
      Utils.showNotification("Prediction failed: " + error.message, "error");
    }
  }
}

async function randomTest() {
  const scenario = document.getElementById("prediction-scenario").value;
  const scenarioData = AppState.scenarios[scenario];

  if (!scenarioData) {
    Utils.showNotification("Please select a scenario first", "warning");
    return;
  }

  if (!AppState.dataLoaded) {
    Utils.showNotification("Please load a dataset first", "warning");
    return;
  }

  try {
    // Show loading state with cool animation
    const randomBtn = document.getElementById("random-test-btn");
    const originalHTML = getRandomButtonMarkup();
    randomBtn.innerHTML = getRandomButtonMarkup(
      "ROLLING",
      '<i class="fas fa-spinner fa-spin"></i>'
    );
    randomBtn.disabled = true;
    randomBtn.style.transform = "scale(0.95)";

    // Fetch random data
    const response = await ApiClient.request(API.randomData);

    if (!response.success) {
      Utils.showNotification(
        response.error || "Failed to get random data",
        "error"
      );
      return;
    }

    const randomData = response.data;

    // Validate that we have all required features
    const missingFeatures = scenarioData.input_features.filter(
      (feature) => randomData[feature] === undefined
    );

    if (missingFeatures.length > 0) {
      Utils.showNotification(
        `Random data missing required features: ${missingFeatures.join(", ")}`,
        "error"
      );
      randomBtn.innerHTML = originalHTML;
      randomBtn.disabled = false;
      randomBtn.style.transform = "scale(1)";
      return;
    }

    // Create animated population of fields
    const populateFieldsSequentially = async () => {
      for (let i = 0; i < scenarioData.input_features.length; i++) {
        const feature = scenarioData.input_features[i];
        const input = document.getElementById(`input-${feature}`);

        if (input && randomData[feature] !== undefined) {
          // Clear field first
          input.value = "";

          // Add loading animation
          input.style.background =
            "linear-gradient(90deg, var(--primary-color)20 0%, var(--primary-color)10 50%, var(--primary-color)20 100%)";
          input.style.backgroundSize = "200% 100%";
          input.style.animation = "shimmer 0.5s ease-in-out";

          // Wait a bit for effect
          await new Promise((resolve) => setTimeout(resolve, 200));

          // Populate with value
          input.value = randomData[feature];

          // Success animation
          input.style.background = "var(--success-color)20";
          input.style.border = "2px solid var(--success-color)";
          input.style.animation = "none";

          // Reset after delay
          setTimeout(() => {
            input.style.background = "";
            input.style.border = "";
          }, 1000);
        }
      }
    };

    // Add shimmer keyframes if not already present
    if (!document.querySelector("#shimmer-styles")) {
      const style = document.createElement("style");
      style.id = "shimmer-styles";
      style.textContent = `
        @keyframes shimmer {
          0% { background-position: -200% 0; }
          100% { background-position: 200% 0; }
        }
      `;
      document.head.appendChild(style);
    }

    await populateFieldsSequentially();

    // Show fun notification
    Utils.showNotification(
      `🎲 Random test data loaded! Values: ${scenarioData.input_features
        .map((f) => `${f}=${randomData[f]?.toFixed(3) || "N/A"}`)
        .join(", ")}`,
      "success"
    );

    // Reset button with success state
    randomBtn.innerHTML = getRandomButtonMarkup(
      "LOADED",
      '<i class="fas fa-check"></i>'
    );
    randomBtn.style.transform = "scale(1)";

    // Auto-submit after showing the populated values
    setTimeout(() => {
      randomBtn.innerHTML = getRandomButtonMarkup(
        "PREDICTING",
        '<i class="fas fa-brain"></i>'
      );
      makePrediction();
    }, 2000);

    // Reset button to original state
    setTimeout(() => {
      randomBtn.innerHTML = originalHTML;
      randomBtn.disabled = false;
    }, 5000);
  } catch (error) {
    Utils.showNotification("Random test failed: " + error.message, "error");

    // Reset button on error
    const randomBtn = document.getElementById("random-test-btn");
    randomBtn.innerHTML = getRandomButtonMarkup();
    randomBtn.style.transform = "scale(1)";
    randomBtn.disabled = false;
  }
}

// Display prediction results - router
function displayPredictionResults(response, inputs) {
  const resultsContainer = document.getElementById("prediction-results");
  const details = response.details;

  if (!details || details.display_type === "error") {
    resultsContainer.innerHTML = `
      <div class="error-analysis-enhanced">
        <h4><i class="fas fa-exclamation-triangle"></i> Prediction Analysis Failed</h4>
        <p>Could not generate the detailed prediction analysis.</p>
        <p><strong>Reason:</strong> ${
          details.message || "An unknown server error occurred."
        }</p>
      </div>`;
    return;
  }

  let html = "";
  switch (details.scenario_key) {
    case "coeff_to_roots":
    case "partial_coeff_to_missing": // Formerly 'partial_coeff'
    case "roots_to_coeff":
    case "single_missing":
      html = renderComparisonResults(details, response.confidences);
      break;
    case "verify_equation":
      html = renderVerificationResults(details, response.confidences);
      break;
    default:
      html = `<div class="error-analysis-enhanced"><h4>Unsupported scenario: ${details.scenario_key}</h4></div>`;
  }

  resultsContainer.innerHTML = html;

  // Trigger animations
  setTimeout(() => {
    resultsContainer
      .querySelectorAll(".slide-up, .scale-in")
      .forEach((el, index) => {
        el.style.animationDelay = `${index * 0.08}s`;
      });
  }, 50);
}

/**
 * Reusable utility to get a quality level object based on error magnitude.
 * This version has the corrected messages.
 * @param {number} error - The error value.
 * @param {boolean} isVerification - If true, low error is 'Excellent'.
 * @returns {object} - An object with level, color, message, and icon.
 */
function getQualityLevel(error, isVerification = false) {
  const excellent = {
    level: "excellent",
    color: "var(--success-color)",
    message: "Excellent!",
    icon: "🎯",
  };
  const good = {
    level: "good",
    color: "var(--primary-color)",
    message: "Good!",
    icon: "👍",
  };
  const fair = {
    level: "fair",
    color: "var(--warning-color)",
    message: "Fair",
    icon: "🤔",
  };
  const poor = {
    level: "poor",
    color: "var(--error-color)",
    message: "Needs Improvement!",
    icon: "😅",
  };

  if (isVerification) {
    if (error < 0.01) return { ...excellent, message: "Highly Consistent" };
    if (error < 0.5) return { ...good, message: "Largely Consistent" };
    if (error < 2.0) return { ...fair, message: "Minor Inconsistency" };
    return { ...poor, message: "Significant Inconsistency", icon: "⚠️" };
  }

  if (error < 0.1) return excellent;
  if (error < 0.5) return good;
  if (error < 1.0) return fair;
  return poor;
}

/**
 * Renders the enhanced comparison view, with the final fixes for the
 * quality badge's text and background styling.
 * @param {object} details - The structured details object from the backend.
 * @param {Array<number>} confidences - The array of confidence values.
 * @returns {string} - The complete HTML string for the results section.
 */
function renderComparisonResults(details, confidences) {
  const {
    scenario_info,
    equation_parts,
    predicted_values,
    actual_values,
    error_metrics,
    analysis,
  } = details;
  avgError = error_metrics["Average Error"] ?? 0;
  const overallQuality = getQualityLevel(avgError);

  const eq = (p) =>
    equation_parts[p] ?? predicted_values[p] ?? actual_values[p] ?? "?";
  const equation = Utils.formatQuadraticEquation(eq("a"), eq("b"), eq("c"));

  const predictedRows = Object.entries(predicted_values)
    .map(([key, value]) => {
      const error = error_metrics[`${key} Error`] ?? 0;
      const quality = getQualityLevel(error);
      return `
        <div class="solution-value">
          <span class="solution-label">${key} =</span>
          <span class="solution-number nn-prediction" style="color: ${
            quality.color
          }; text-shadow: 0 0 8px ${quality.color}30;">
            ${Utils.formatNumber(value, 6)}
          </span>
        </div>`;
    })
    .join("");

  let actualRows = "";
  if (
    analysis.actual_solution_type === "complex" ||
    analysis.actual_solution_type === "invalid"
  ) {
    actualRows = `<div class="solution-message-box">${analysis.actual_solution_message}</div>`;
  } else if (Object.keys(actual_values).length > 0) {
    actualRows = Object.entries(actual_values)
      .map(
        ([key, value]) => `
      <div class="solution-value">
        <span class="solution-label">${key} =</span>
        <span class="solution-number actual-solution">${Utils.formatNumber(
          value,
          6
        )}</span>
      </div>`
      )
      .join("");
  } else {
    actualRows = `<div class="solution-message-box">Ground truth could not be calculated.</div>`;
  }

  const errorCards = Object.entries(error_metrics)
    .map(([name, error]) => {
      if (name.includes("Average")) return "";
      const quality = getQualityLevel(error);
      return `
      <div class="error-metric-card" style="border-color: ${
        quality.color
      }; background: linear-gradient(135deg, ${
        quality.color
      }08, var(--surface-color));">
        <div class="metric-icon">${
          name.includes("x") || name.includes("₂") ? "📊" : "📈"
        }</div>
        <div class="metric-label">${name}</div>
        <div class="metric-value" style="color: ${
          quality.color
        };">${Utils.formatNumber(error, 6)}</div>
        <div class="metric-status" style="background: ${
          quality.color
        }20; color: ${quality.color};">${quality.message}</div>
      </div>`;
    })
    .join("");

  const avgConfidence =
    confidences && confidences.length > 0
      ? confidences.reduce((a, b) => a + b, 0) / confidences.length
      : 0;
  const confidenceLevel = Utils.getConfidenceLevel(avgConfidence);

  const detailedResultsRows = Object.entries(predicted_values)
    .map(([key, prediction], index) => {
      const confidence =
        confidences && confidences.length > index ? confidences[index] : 0;
      const confidenceLevelText = Utils.getConfidenceLevel(confidence);
      const errorValue = error_metrics[`${key} Error`] ?? null;
      const quality = getQualityLevel(errorValue);

      return `
      <div class="detailed-result-row" style="display: flex; justify-content: space-between; align-items: center; padding: 16px; background: var(--background-color); border-radius: var(--radius-medium); border: 1px solid var(--border-color); transition: all 0.3s ease;" onmouseover="this.style.transform='translateY(-2px)'; this.style.boxShadow='var(--shadow-light)'" onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='none'">
        <div>
          <strong style="color: ${quality.color};">${key}:</strong> 
          <span style="font-family: 'JetBrains Mono', monospace; color: ${
            quality.color
          }; font-weight: 600;">
            ${Utils.formatNumber(prediction, 6)}
          </span>
        </div>
        <div style="text-align: right;">
          <div>Confidence: <span style="font-weight: 600;">${Utils.formatPercentage(
            confidence * 100,
            1
          )}</span></div>
          <div style="font-size: 14px; margin-top: 4px;">${confidenceLevelText}</div>
          ${
            errorValue !== null
              ? `<div style="font-size: 12px; color: ${
                  quality.color
                }; margin-top: 2px;">Error: ${Utils.formatNumber(
                  errorValue,
                  4
                )}</div>`
              : ""
          }
        </div>
      </div>
    `;
    })
    .join("");

  const detailedResultsSection =
    Object.keys(predicted_values).length > 0
      ? `
    <div class="original-results-grid slide-up" style="display: grid; gap: 16px; margin-top: 24px; padding: 20px; background: var(--surface-color); border-radius: var(--radius-medium); border: 1px solid var(--border-color);">
      <h4 style="margin: 0 0 16px 0; display: flex; align-items: center; gap: 8px;">
        <i class="fas fa-list"></i>
        Detailed Results
      </h4>
      ${detailedResultsRows}
    </div>
  `
      : "";

  return `
    <div class="prediction-results-container fade-in">
      <div class="equation-display-section slide-up">
        <h3 class="section-subtitle"><i class="fas fa-function"></i> Quadratic Equation</h3>
        <div class="equation-display animated-equation">${equation}</div>
      </div>

      <div class="solution-comparison-section slide-up">
        <h4 class="comparison-title"><i class="fas fa-balance-scale"></i> Solution Comparison</h4>
        <div class="solution-comparison-grid">
          <div class="solution-column neural-prediction">
            <div class="solution-header neural-network"><i class="fas fa-brain"></i><span>Neural Network</span></div>
            <div class="solution-values">${predictedRows}</div>
            <div class="prediction-confidence">
              <span class="confidence-label">Avg. Confidence:</span>
              <span class="confidence-value">${confidenceLevel}</span>
            </div>
          </div>
          <div class="solution-column actual-solution">
            <div class="solution-header actual-solution"><i class="fas fa-check-circle"></i><span>Actual Solution</span></div>
            <div class="solution-values">${actualRows}</div>
            <div class="solution-message">Mathematical ground truth</div>
          </div>
        </div>
      </div>

      <!-- FINAL FIX: Quality Badge with class-based background and corrected text -->
      <div class="quality-badge-container scale-in">
        <div class="quality-badge quality-${overallQuality.level}">
          <div class="quality-icon">${overallQuality.icon}</div>
          <div class="quality-message">${overallQuality.message}</div>
          <div class="quality-metric">Avg Error: ${Utils.formatNumber(
            avgError,
            4
          )}</div>
        </div>
      </div>

      <div class="error-analysis-enhanced slide-up">
        <h4 class="error-title"><i class="fas fa-chart-line"></i> Error Analysis</h4>
        <div class="error-metrics-grid">${errorCards}</div>
      </div>

      <div class="performance-insights slide-up">
        <h4 class="insights-title"><i class="fas fa-lightbulb"></i> Performance Insights</h4>
        <div class="insights-content">
          <div class="insight-item"><strong>Prediction Type:</strong> ${
            scenario_info.name
          }</div>
          <div class="insight-item"><strong>Confidence Level:</strong> ${confidenceLevel}</div>
          <div class="insight-item">
            <strong>Overall Assessment:</strong> 
            <span style="color: ${overallQuality.color}; font-weight: 600;">${
    overallQuality.level.charAt(0).toUpperCase() + overallQuality.level.slice(1)
  }</span>
          </div>
        </div>
      </div>
      
      ${detailedResultsSection}
    </div>`;
}

/**
 * Renders the results for the 'Equation Verification' scenario with corrected
 * root display and enhanced styling.
 * @param {object} details - The structured details object from the backend.
 * @param {Array<number>} confidences - The array of confidence values.
 * @returns {string} - The complete HTML string for the results section.
 */
function renderVerificationResults(details, confidences) {
  const {
    equation_parts,
    predicted_values,
    actual_values,
    error_metrics,
    labels,
  } = details;
  const equation = Utils.formatQuadraticEquation(
    equation_parts.a,
    equation_parts.b,
    equation_parts.c
  );
  const actualError = actual_values["Actual Error"];
  const errorQuality = getQualityLevel(actualError, true); // Use verification mode

  return `
    <div class="prediction-results-container fade-in">
      <div class="equation-display-section slide-up">
        <h3 class="section-subtitle"><i class="fas fa-check-double"></i> Equation Under Test</h3>
        <div class="equation-display animated-equation">${equation}</div>
        
        <!-- FIX: Corrected and restyled root display -->
        <div class="equation-display-roots" style="margin-top: 16px;">
            <div class="solution-value">
                <span class="solution-label">Provided Root x₁ =</span>
                <span class="solution-number actual-solution">${Utils.formatNumber(
                  equation_parts["x₁"],
                  4
                )}</span>
            </div>
            <div class="solution-value">
                <span class="solution-label">Provided Root x₂ =</span>
                <span class="solution-number actual-solution">${Utils.formatNumber(
                  equation_parts["x₂"],
                  4
                )}</span>
            </div>
        </div>
      </div>

      <div class="quality-badge-container scale-in">
        <div class="quality-badge quality-${
          errorQuality.level
        }" style="background-color: ${errorQuality.color}20; color: ${
    errorQuality.color
  }; border: 1px solid ${errorQuality.color};">
          <div class="quality-icon">${errorQuality.icon}</div>
          <div class="quality-message">${errorQuality.message}</div>
          <div class="quality-metric">Actual Error: ${Utils.formatNumber(
            actualError,
            4
          )}</div>
        </div>
      </div>

      <div class="error-analysis-enhanced slide-up">
        <h4 class="error-title"><i class="fas fa-tasks"></i> Verification Analysis</h4>
        <div class="error-metrics-grid">
            <div class="error-metric-card">
              <div class="metric-label">${labels.predicted}</div>
              <div class="metric-value">${Utils.formatNumber(
                predicted_values["Predicted Error"],
                6
              )}</div>
            </div>
            <div class="error-metric-card">
              <div class="metric-label">${labels.actual}</div>
              <div class="metric-value">${Utils.formatNumber(
                actual_values["Actual Error"],
                6
              )}</div>
            </div>
            <div class="error-metric-card">
              <div class="metric-label">Prediction Deviation</div>
              <div class="metric-value">${Utils.formatNumber(
                error_metrics["Prediction Deviation"],
                6
              )}</div>
            </div>
        </div>
      </div>
    </div>`;
}

/**
 * Reusable utility to get a quality level based on error magnitude.
 * @param {number} error - The error value.
 * @param {boolean} isVerification - If true, low error is 'Excellent'.
 * @returns {object} - An object with level, color, message, and icon.
 */
function getQualityLevel(error, isVerification = false) {
  const excellent = {
    level: "excellent",
    color: "var(--success-color)",
    message: "Excellent Match",
    icon: "🎯",
  };
  const good = {
    level: "good",
    color: "var(--primary-color)",
    message: "Good Match",
    icon: "👍",
  };
  const fair = {
    level: "fair",
    color: "var(--warning-color)",
    message: "Fair Match",
    icon: "🤔",
  };
  const poor = {
    level: "poor",
    color: "var(--error-color)",
    message: "Needs Improvement!",
    icon: "😅",
  };

  if (isVerification) {
    if (error < 0.01) return { ...excellent, message: "Highly Consistent" };
    if (error < 0.5) return { ...good, message: "Largely Consistent" };
    if (error < 2.0) return { ...fair, message: "Minor Inconsistency" };
    return { ...poor, message: "Significant Inconsistency", icon: "⚠️" };
  }

  if (error < 0.1) return excellent;
  if (error < 0.5) return good;
  if (error < 1.0) return fair;
  return poor;
}

async function generateAnalysis() {
  try {
    const results = await ApiClient.request(API.results);
    if (Object.keys(results).length === 0) {
      Utils.showNotification(
        "No trained models available for analysis",
        "warning"
      );
      return;
    }

    await AnalysisSection.generateCharts();
    Utils.showNotification("Analysis generated successfully", "success");
  } catch (error) {
    Utils.showNotification(
      "Failed to generate analysis: " + error.message,
      "error"
    );
  }
}

async function generateComparison() {
  try {
    const results = await ApiClient.request(API.results);
    if (Object.keys(results).length < 2) {
      Utils.showNotification(
        "Need at least 2 trained models for comparison",
        "warning"
      );
      return;
    }

    await ComparisonSection.generateComparison();
    Utils.showNotification("Comparison generated successfully", "success");
  } catch (error) {
    Utils.showNotification(
      "Failed to generate comparison: " + error.message,
      "error"
    );
  }
}

