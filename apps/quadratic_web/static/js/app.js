/**
 * Quadratic Neural Network Web Application
 * Frontend JavaScript Application
 *
 * Author: Matt
 * Location: Varna, Bulgaria
 * Date: July 2025
 *
 * Beautiful Apple-like web interface for quadratic neural network analysis
 */

// Application state management
const AppState = {
  currentSection: "dashboard",
  isTraining: false,
  dataLoaded: false,
  scenarios: {},
  results: {},
  trainingInterval: null,
  charts: {
    metrics: null,
    accuracy: null,
    comparison: null,
  },
};

// API endpoints
const API = {
  health: "/api/health",
  scenarios: "/api/scenarios",
  uploadData: "/api/data/upload",
  dataInfo: "/api/data/info",
  startTraining: "/api/training/start",
  trainingStatus: "/api/training/status",
  stopTraining: "/api/training/stop",
  predict: "/api/predict",
  results: "/api/results",
  performanceAnalysis: "/api/analysis/performance",
};

// Utility functions
const Utils = {
  formatNumber: (num, decimals = 6) => {
    if (typeof num !== "number" || isNaN(num)) return "0.000000";
    if (Math.abs(num) < 1e-10) return "0.000000";
    return num.toFixed(decimals);
  },

  formatPercentage: (num, decimals = 1) => {
    return `${Utils.formatNumber(num, decimals)}%`;
  },

  getConfidenceLevel: (confidence) => {
    if (confidence > 0.8) return "🟢 High";
    if (confidence > 0.6) return "🟡 Medium";
    return "🔴 Low";
  },

  showNotification: (message, type = "info") => {
    const notification = document.createElement("div");
    notification.className = `notification notification-${type}`;
    notification.innerHTML = `
            <div class="notification-content">
                <i class="fas fa-${
                  type === "success"
                    ? "check-circle"
                    : type === "error"
                    ? "exclamation-circle"
                    : type === "warning"
                    ? "exclamation-triangle"
                    : "info-circle"
                }"></i>
                <span>${message}</span>
            </div>
        `;

    // Add styles if not already present
    if (!document.querySelector("#notification-styles")) {
      const styles = document.createElement("style");
      styles.id = "notification-styles";
      styles.textContent = `
                .notification {
                    position: fixed;
                    top: 20px;
                    right: 20px;
                    background: var(--surface-color);
                    border-radius: var(--radius-medium);
                    padding: 16px;
                    box-shadow: var(--shadow-heavy);
                    border: 1px solid var(--border-color);
                    z-index: 1000;
                    max-width: 400px;
                    animation: slideIn 0.3s ease;
                }
                .notification-content {
                    display: flex;
                    align-items: center;
                    gap: 12px;
                }
                .notification-success { border-left: 4px solid var(--success-color); }
                .notification-error { border-left: 4px solid var(--error-color); }
                .notification-warning { border-left: 4px solid var(--warning-color); }
                .notification-info { border-left: 4px solid var(--primary-color); }
                @keyframes slideIn {
                    from { transform: translateX(100%); opacity: 0; }
                    to { transform: translateX(0); opacity: 1; }
                }
            `;
      document.head.appendChild(styles);
    }

    document.body.appendChild(notification);

    // Auto remove after 5 seconds
    setTimeout(() => {
      notification.style.animation = "slideIn 0.3s ease reverse";
      setTimeout(() => {
        if (notification.parentNode) {
          notification.parentNode.removeChild(notification);
        }
      }, 300);
    }, 5000);
  },

  debounce: (func, wait) => {
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
};

// API helper functions
const ApiClient = {
  async request(url, options = {}) {
    try {
      const response = await fetch(url, {
        headers: {
          "Content-Type": "application/json",
          ...options.headers,
        },
        ...options,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error("API request failed:", error);
      throw error;
    }
  },

  async uploadFile(file) {
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(API.uploadData, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error("File upload failed:", error);
      throw error;
    }
  },
};

// Navigation management
const Navigation = {
  init() {
    // Set up navigation event listeners
    document.querySelectorAll(".nav-link").forEach((link) => {
      link.addEventListener("click", (e) => {
        e.preventDefault();
        const section = link.dataset.section;
        if (section) {
          this.showSection(section);
        }
      });
    });

    // Initialize with dashboard
    this.showSection("dashboard");
  },

  showSection(sectionId) {
    // Hide all sections
    document.querySelectorAll(".content-section").forEach((section) => {
      section.classList.remove("active");
    });

    // Show selected section
    const section = document.getElementById(sectionId);
    if (section) {
      section.classList.add("active");
      AppState.currentSection = sectionId;
    }

    // Update navigation active state
    document.querySelectorAll(".nav-link").forEach((link) => {
      link.classList.remove("active");
    });

    const activeLink = document.querySelector(`[data-section="${sectionId}"]`);
    if (activeLink) {
      activeLink.classList.add("active");
    }

    // Section-specific initialization
    switch (sectionId) {
      case "data":
        DataSection.refresh();
        break;
      case "training":
        TrainingSection.refresh();
        break;
      case "prediction":
        PredictionSection.refresh();
        break;
      case "analysis":
        AnalysisSection.refresh();
        break;
      case "comparison":
        ComparisonSection.refresh();
        break;
    }
  },
};

// Data section management
const DataSection = {
  init() {
    this.refresh();
  },

  async refresh() {
    try {
      const dataInfo = await ApiClient.request(API.dataInfo);
      this.updateDataInfo(dataInfo);
    } catch (error) {
      console.error("Failed to load data info:", error);
    }
  },

  updateDataInfo(dataInfo) {
    const infoContainer = document.getElementById("dataset-info");
    const tableContainer = document.getElementById("data-table");

    if (!dataInfo.loaded) {
      infoContainer.innerHTML = `
                <div style="color: var(--text-secondary); text-align: center; padding: 40px;">
                    <i class="fas fa-database" style="font-size: 48px; margin-bottom: 16px; opacity: 0.5;"></i>
                    <p>No dataset loaded. Please upload a CSV file to begin.</p>
                </div>
            `;
      tableContainer.style.display = "none";
      AppState.dataLoaded = false;
      return;
    }

    // Update info display
    infoContainer.innerHTML = `
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px;">
                <div class="info-card">
                    <h4><i class="fas fa-chart-bar"></i> Dataset Overview</h4>
                    <p><strong>Total Equations:</strong> ${dataInfo.total_equations.toLocaleString()}</p>
                    <p><strong>Features:</strong> a, b, c, x1, x2</p>
                    <p><strong>Format:</strong> Quadratic equation dataset</p>
                </div>
                <div class="info-card">
                    <h4><i class="fas fa-calculator"></i> Statistics</h4>
                    <p><strong>Coefficient 'a':</strong> ${Utils.formatNumber(
                      dataInfo.stats.columns.a.mean,
                      3
                    )} ± ${Utils.formatNumber(
      dataInfo.stats.columns.a.std,
      3
    )}</p>
                    <p><strong>Coefficient 'b':</strong> ${Utils.formatNumber(
                      dataInfo.stats.columns.b.mean,
                      3
                    )} ± ${Utils.formatNumber(
      dataInfo.stats.columns.b.std,
      3
    )}</p>
                    <p><strong>Coefficient 'c':</strong> ${Utils.formatNumber(
                      dataInfo.stats.columns.c.mean,
                      3
                    )} ± ${Utils.formatNumber(
      dataInfo.stats.columns.c.std,
      3
    )}</p>
                </div>
                <div class="info-card">
                    <h4><i class="fas fa-check-circle"></i> Quality Metrics</h4>
                    <p><strong>Integer Solutions (x1):</strong> ${Utils.formatNumber(
                      dataInfo.stats.quality.x1_whole_pct,
                      1
                    )}%</p>
                    <p><strong>Integer Solutions (x2):</strong> ${Utils.formatNumber(
                      dataInfo.stats.quality.x2_whole_pct,
                      1
                    )}%</p>
                    <p><strong>Data Quality:</strong> <span style="color: var(--success-color);">✓ Verified</span></p>
                </div>
            </div>
        `;

    // Update data table
    this.updateDataTable(dataInfo.sample_data);
    AppState.dataLoaded = true;
  },

  updateDataTable(sampleData) {
    const tableBody = document.getElementById("data-table-body");
    const table = document.getElementById("data-table");

    if (!sampleData || sampleData.length === 0) {
      table.style.display = "none";
      return;
    }

    // Clear existing data
    tableBody.innerHTML = "";

    // Add sample data rows
    sampleData.forEach((row, index) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
                <td style="padding: 8px; border-bottom: 1px solid var(--border-color);">${Utils.formatNumber(
                  row[0],
                  3
                )}</td>
                <td style="padding: 8px; border-bottom: 1px solid var(--border-color);">${Utils.formatNumber(
                  row[1],
                  3
                )}</td>
                <td style="padding: 8px; border-bottom: 1px solid var(--border-color);">${Utils.formatNumber(
                  row[2],
                  3
                )}</td>
                <td style="padding: 8px; border-bottom: 1px solid var(--border-color);">${Utils.formatNumber(
                  row[3],
                  3
                )}</td>
                <td style="padding: 8px; border-bottom: 1px solid var(--border-color);">${Utils.formatNumber(
                  row[4],
                  3
                )}</td>
            `;
      if (index % 2 === 0) {
        tr.style.backgroundColor = "var(--background-color)";
      }
      tableBody.appendChild(tr);
    });

    table.style.display = "table";
  },
};

// Training section management
const TrainingSection = {
  init() {
    this.loadScenarios();
    this.setupEventListeners();
  },

  async loadScenarios() {
    try {
      const scenarios = await ApiClient.request(API.scenarios);
      AppState.scenarios = scenarios;
      this.updateScenariosSelection(scenarios);
    } catch (error) {
      console.error("Failed to load scenarios:", error);
      Utils.showNotification("Failed to load training scenarios", "error");
    }
  },

  updateScenariosSelection(scenarios) {
    const container = document.getElementById("scenarios-selection");
    container.innerHTML = "";

    Object.entries(scenarios).forEach(([key, scenario]) => {
      const scenarioCard = document.createElement("div");
      scenarioCard.className = "scenario-card";
      scenarioCard.innerHTML = `
                <label style="display: flex; align-items: center; gap: 12px; padding: 16px; border: 1px solid var(--border-color); border-radius: var(--radius-medium); cursor: pointer; transition: all 0.3s ease;">
                    <input type="checkbox" value="${key}" checked style="width: 16px; height: 16px;">
                    <div style="flex: 1;">
                        <div style="font-weight: 600; color: var(--text-primary); margin-bottom: 4px;">${
                          scenario.name
                        }</div>
                        <div style="font-size: 14px; color: var(--text-secondary);">${
                          scenario.description
                        }</div>
                        <div style="font-size: 12px; color: var(--text-secondary); margin-top: 8px;">
                            <strong>Input:</strong> ${scenario.input_features.join(
                              ", "
                            )}<br>
                            <strong>Output:</strong> ${scenario.target_features.join(
                              ", "
                            )}
                        </div>
                    </div>
                    <div style="width: 12px; height: 12px; border-radius: 50%; background: ${
                      scenario.color
                    };"></div>
                </label>
            `;

      // Add hover effects
      const label = scenarioCard.querySelector("label");
      label.addEventListener("mouseenter", () => {
        label.style.borderColor = scenario.color;
        label.style.boxShadow = `0 4px 12px ${scenario.color}20`;
      });
      label.addEventListener("mouseleave", () => {
        label.style.borderColor = "var(--border-color)";
        label.style.boxShadow = "none";
      });

      container.appendChild(scenarioCard);
    });
  },

  setupEventListeners() {
    // Training progress monitoring
    this.startProgressMonitoring();
  },

  startProgressMonitoring() {
    if (AppState.trainingInterval) {
      clearInterval(AppState.trainingInterval);
    }

    AppState.trainingInterval = setInterval(async () => {
      if (AppState.isTraining) {
        await this.updateTrainingStatus();
      }
    }, 1000);
  },

  async updateTrainingStatus() {
    try {
      const status = await ApiClient.request(API.trainingStatus);
      this.updateProgressDisplay(status);
    } catch (error) {
      console.error("Failed to get training status:", error);
    }
  },

  updateProgressDisplay(status) {
    const progressFill = document.getElementById("training-progress-fill");
    const progressText = document.getElementById("training-progress-text");
    const logsContainer = document.getElementById("training-logs");

    // Update progress bar
    progressFill.style.width = `${status.progress}%`;

    // Update progress text
    if (status.is_training) {
      progressText.textContent = status.current_scenario
        ? `Training: ${status.current_scenario} (${status.progress.toFixed(
            1
          )}%)`
        : `Training in progress... (${status.progress.toFixed(1)}%)`;
    } else {
      progressText.textContent =
        status.progress === 100 ? "Training completed!" : "Ready to train";
    }

    // Update logs
    if (status.logs && status.logs.length > 0) {
      logsContainer.innerHTML = "";
      status.logs.forEach((log) => {
        const logEntry = document.createElement("div");
        logEntry.style.marginBottom = "4px";
        logEntry.innerHTML = `<span style="color: var(--text-secondary);">[${log.timestamp}]</span> ${log.message}`;
        logsContainer.appendChild(logEntry);
      });
      logsContainer.scrollTop = logsContainer.scrollHeight;
    }
  },

  async refresh() {
    if (!AppState.dataLoaded) {
      Utils.showNotification("Please load a dataset first", "warning");
      return;
    }
    await this.loadScenarios();
  },
};

// Prediction section management
const PredictionSection = {
  init() {
    this.loadScenarios();
    this.setupEventListeners();
  },

  async loadScenarios() {
    try {
      const scenarios = await ApiClient.request(API.scenarios);
      AppState.scenarios = scenarios;
      this.updateScenarioSelect(scenarios);
    } catch (error) {
      console.error("Failed to load scenarios:", error);
    }
  },

  updateScenarioSelect(scenarios) {
    const select = document.getElementById("prediction-scenario");
    select.innerHTML = "";

    Object.entries(scenarios).forEach(([key, scenario]) => {
      const option = document.createElement("option");
      option.value = key;
      option.textContent = `${scenario.name} - ${scenario.description}`;
      select.appendChild(option);
    });

    // Update input fields for first scenario
    if (Object.keys(scenarios).length > 0) {
      this.updateInputFields(Object.keys(scenarios)[0]);
    }
  },

  setupEventListeners() {
    const scenarioSelect = document.getElementById("prediction-scenario");
    scenarioSelect.addEventListener("change", (e) => {
      this.updateInputFields(e.target.value);
    });
  },

  updateInputFields(scenarioKey) {
    const scenario = AppState.scenarios[scenarioKey];
    if (!scenario) return;

    const inputsContainer = document.getElementById("prediction-inputs");
    inputsContainer.innerHTML = "";

    scenario.input_features.forEach((feature) => {
      const inputGroup = document.createElement("div");
      inputGroup.className = "form-group";
      inputGroup.innerHTML = `
                <label class="form-label">${feature}</label>
                <input type="number" class="form-input" id="input-${feature}" 
                       step="any" placeholder="Enter ${feature} value">
            `;
      inputsContainer.appendChild(inputGroup);
    });
  },

  async refresh() {
    await this.loadScenarios();
  },
};

// Analysis section management
const AnalysisSection = {
  init() {
    // Analysis section initialization
  },

  async refresh() {
    // Refresh analysis data
  },

  async generateCharts() {
    try {
      const analysisData = await ApiClient.request(API.performanceAnalysis);
      this.createMetricsChart(analysisData);
      this.createAccuracyChart(analysisData);
    } catch (error) {
      console.error("Failed to generate analysis:", error);
      Utils.showNotification("Failed to generate analysis charts", "error");
    }
  },

  createMetricsChart(data) {
    const ctx = document.getElementById("metrics-chart").getContext("2d");

    if (AppState.charts.metrics) {
      AppState.charts.metrics.destroy();
    }

    AppState.charts.metrics = new Chart(ctx, {
      type: "radar",
      data: {
        labels: ["R² Score", "MSE (inv)", "MAE (inv)", "Accuracy"],
        datasets: data.scenarios.map((scenario, index) => ({
          label: data.scenario_names[index],
          data: [
            data.metrics.r2_scores[index],
            1 -
              data.metrics.mse_values[index] /
                Math.max(...data.metrics.mse_values),
            1 -
              data.metrics.mae_values[index] /
                Math.max(...data.metrics.mae_values),
            data.metrics.accuracy_values[index] / 100,
          ],
          backgroundColor: data.colors[index] + "20",
          borderColor: data.colors[index],
          pointBackgroundColor: data.colors[index],
          pointBorderColor: "#fff",
          pointHoverBackgroundColor: "#fff",
          pointHoverBorderColor: data.colors[index],
        })),
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          r: {
            beginAtZero: true,
            max: 1,
            grid: {
              color: "var(--border-color)",
            },
            pointLabels: {
              color: "var(--text-primary)",
            },
          },
        },
        plugins: {
          legend: {
            labels: {
              color: "var(--text-primary)",
            },
          },
        },
      },
    });
  },

  createAccuracyChart(data) {
    const ctx = document.getElementById("accuracy-chart").getContext("2d");

    if (AppState.charts.accuracy) {
      AppState.charts.accuracy.destroy();
    }

    AppState.charts.accuracy = new Chart(ctx, {
      type: "bar",
      data: {
        labels: data.scenario_names,
        datasets: [
          {
            label: "Accuracy (%)",
            data: data.metrics.accuracy_values,
            backgroundColor: data.colors.map((color) => color + "80"),
            borderColor: data.colors,
            borderWidth: 2,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          y: {
            beginAtZero: true,
            max: 100,
            grid: {
              color: "var(--border-color)",
            },
            ticks: {
              color: "var(--text-primary)",
            },
          },
          x: {
            grid: {
              color: "var(--border-color)",
            },
            ticks: {
              color: "var(--text-primary)",
            },
          },
        },
        plugins: {
          legend: {
            labels: {
              color: "var(--text-primary)",
            },
          },
        },
      },
    });
  },
};

// Comparison section management
const ComparisonSection = {
  init() {
    // Comparison section initialization
  },

  async refresh() {
    // Refresh comparison data
  },

  async generateComparison() {
    try {
      const results = await ApiClient.request(API.results);
      this.createComparisonChart(results);
      this.generateComparisonReport(results);
    } catch (error) {
      console.error("Failed to generate comparison:", error);
      Utils.showNotification("Failed to generate comparison", "error");
    }
  },

  createComparisonChart(results) {
    const ctx = document.getElementById("comparison-chart").getContext("2d");

    if (AppState.charts.comparison) {
      AppState.charts.comparison.destroy();
    }

    const scenarios = Object.keys(results);
    const r2Scores = scenarios.map((s) => results[s].metrics.r2);
    const colors = scenarios.map((s) => results[s].scenario_info.color);

    AppState.charts.comparison = new Chart(ctx, {
      type: "doughnut",
      data: {
        labels: scenarios.map((s) => results[s].scenario_info.name),
        datasets: [
          {
            data: r2Scores,
            backgroundColor: colors.map((c) => c + "80"),
            borderColor: colors,
            borderWidth: 2,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: "bottom",
            labels: {
              color: "var(--text-primary)",
            },
          },
        },
      },
    });
  },

  generateComparisonReport(results) {
    const container = document.getElementById("model-rankings");

    // Sort scenarios by R² score
    const sortedScenarios = Object.entries(results).sort(
      ([, a], [, b]) => b.metrics.r2 - a.metrics.r2
    );

    let html = '<div style="display: grid; gap: 16px;">';

    sortedScenarios.forEach(([key, result], index) => {
      const medal =
        index === 0 ? "🥇" : index === 1 ? "🥈" : index === 2 ? "🥉" : "📊";
      const performance =
        result.metrics.r2 > 0.9
          ? "Excellent"
          : result.metrics.r2 > 0.7
          ? "Good"
          : result.metrics.r2 > 0.5
          ? "Fair"
          : "Poor";

      html += `
                <div style="display: flex; align-items: center; gap: 16px; padding: 16px; background: var(--background-color); border-radius: var(--radius-medium); border: 1px solid var(--border-color);">
                    <div style="font-size: 24px;">${medal}</div>
                    <div style="flex: 1;">
                        <h4 style="margin: 0 0 8px 0;">${
                          result.scenario_info.name
                        }</h4>
                        <p style="margin: 0; color: var(--text-secondary); font-size: 14px;">${
                          result.scenario_info.description
                        }</p>
                        <div style="margin-top: 8px; display: flex; gap: 16px; font-size: 14px;">
                            <span><strong>R²:</strong> ${Utils.formatNumber(
                              result.metrics.r2,
                              4
                            )}</span>
                            <span><strong>MSE:</strong> ${Utils.formatNumber(
                              result.metrics.mse,
                              6
                            )}</span>
                            <span><strong>Accuracy:</strong> ${Utils.formatNumber(
                              result.metrics.accuracy_10pct,
                              1
                            )}%</span>
                        </div>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-weight: 600; color: var(--text-primary);">${performance}</div>
                        <div style="width: 12px; height: 12px; border-radius: 50%; background: ${
                          result.scenario_info.color
                        }; margin: 8px auto 0;"></div>
                    </div>
                </div>
            `;
    });

    html += "</div>";
    container.innerHTML = html;
  },
};

// Global functions for HTML onclick events
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
    Utils.showNotification("Please load a dataset first", "warning");
    return;
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

  try {
    const response = await ApiClient.request(API.startTraining, {
      method: "POST",
      body: JSON.stringify({
        scenarios: selectedScenarios,
        epochs: epochs,
      }),
    });

    if (response.success) {
      AppState.isTraining = true;
      document.getElementById("start-training-btn").innerHTML =
        '<i class="loading-spinner"></i> Training...';
      document.getElementById("start-training-btn").disabled = true;
      Utils.showNotification("Training started successfully", "success");
    }
  } catch (error) {
    Utils.showNotification(
      "Failed to start training: " + error.message,
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
      displayPredictionResults(response);
    } else {
      Utils.showNotification(response.error || "Prediction failed", "error");
    }
  } catch (error) {
    Utils.showNotification("Prediction failed: " + error.message, "error");
  }
}

function displayPredictionResults(response) {
  const resultsContainer = document.getElementById("prediction-results");

  let html = `
        <div style="padding: 20px;">
            <h3 style="margin-bottom: 20px;">
                <i class="fas fa-brain"></i> Prediction Results
            </h3>
            <div style="display: grid; gap: 16px;">
    `;

  response.target_features.forEach((feature, index) => {
    const prediction = response.predictions[index];
    const confidence = response.confidences[index];
    const confidenceLevel = Utils.getConfidenceLevel(confidence);

    html += `
            <div style="display: flex; justify-content: space-between; align-items: center; padding: 16px; background: var(--background-color); border-radius: var(--radius-medium); border: 1px solid var(--border-color);">
                <div>
                    <strong>${feature}:</strong> ${Utils.formatNumber(
      prediction,
      6
    )}
                </div>
                <div style="text-align: right;">
                    <div>Confidence: ${Utils.formatPercentage(
                      confidence * 100,
                      1
                    )}</div>
                    <div style="font-size: 14px; margin-top: 4px;">${confidenceLevel}</div>
                </div>
            </div>
        `;
  });

  // Add actual solutions comparison if available
  if (response.actual_solutions) {
    html += `
            <div style="margin-top: 20px; padding: 16px; background: var(--success-color)20; border-radius: var(--radius-medium); border: 1px solid var(--success-color);">
                <h4 style="margin-bottom: 12px;">
                    <i class="fas fa-check-circle"></i> Actual Solutions Comparison
                </h4>
        `;

    response.actual_solutions.forEach((actual, index) => {
      const predicted = response.predictions[index];
      const error = Math.abs(predicted - actual);
      const errorPercent = Math.abs(error / (actual + 1e-8)) * 100;

      html += `
                <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                    <span>${
                      response.target_features[index]
                    }: Predicted ${Utils.formatNumber(
        predicted,
        6
      )} vs Actual ${Utils.formatNumber(actual, 6)}</span>
                    <span style="font-weight: 600; color: ${
                      error < 0.01
                        ? "var(--success-color)"
                        : error < 0.1
                        ? "var(--warning-color)"
                        : "var(--error-color)"
                    };">
                        Error: ${Utils.formatNumber(
                          error,
                          6
                        )} (${Utils.formatNumber(errorPercent, 2)}%)
                    </span>
                </div>
            `;
    });

    html += "</div>";
  }

  html += `
            </div>
        </div>
    `;

  resultsContainer.innerHTML = html;
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

// Application initialization
document.addEventListener("DOMContentLoaded", () => {
  console.log("🚀 Quadratic Neural Network Web Application");
  console.log("Initializing application...");

  // Initialize all sections
  Navigation.init();
  DataSection.init();
  TrainingSection.init();
  PredictionSection.init();
  AnalysisSection.init();
  ComparisonSection.init();

  // Check API health
  ApiClient.request(API.health)
    .then((response) => {
      console.log("✅ API connection established");
      document.getElementById("connection-status").textContent = "Connected";
    })
    .catch((error) => {
      console.error("❌ API connection failed:", error);
      document.getElementById("connection-status").textContent = "Disconnected";
      document.querySelector(".status-dot").style.backgroundColor =
        "var(--error-color)";
    });

  console.log("🎉 Application initialized successfully!");
});

// Handle training status updates
setInterval(async () => {
  if (AppState.isTraining) {
    try {
      const status = await ApiClient.request(API.trainingStatus);
      if (!status.is_training && AppState.isTraining) {
        // Training finished
        AppState.isTraining = false;
        document.getElementById("start-training-btn").innerHTML =
          '<i class="fas fa-play"></i> Start Training';
        document.getElementById("start-training-btn").disabled = false;
        Utils.showNotification("Training completed!", "success");
      }
    } catch (error) {
      console.error("Failed to check training status:", error);
    }
  }
}, 2000);
