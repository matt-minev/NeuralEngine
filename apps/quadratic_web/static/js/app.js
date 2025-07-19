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
  autoLoadDataset: null,
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
  randomData: "/api/data/random",
  clearData: "/api/data/clear",
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

  // NEW: Format quadratic equation
  formatQuadraticEquation: (a, b, c) => {
    let equation = "";

    // Handle coefficient 'a'
    if (a === 1) {
      equation += "x²";
    } else if (a === -1) {
      equation += "-x²";
    } else {
      equation += `${Utils.formatNumber(a, 3)}x²`;
    }

    // Handle coefficient 'b'
    if (b > 0) {
      if (b === 1) {
        equation += " + x";
      } else {
        equation += ` + ${Utils.formatNumber(b, 3)}x`;
      }
    } else if (b < 0) {
      if (b === -1) {
        equation += " - x";
      } else {
        equation += ` - ${Utils.formatNumber(Math.abs(b), 3)}x`;
      }
    }

    // Handle coefficient 'c'
    if (c > 0) {
      equation += ` + ${Utils.formatNumber(c, 3)}`;
    } else if (c < 0) {
      equation += ` - ${Utils.formatNumber(Math.abs(c), 3)}`;
    }

    equation += " = 0";
    return equation;
  },

  // NEW: Calculate actual quadratic solutions
  calculateActualSolutions: (a, b, c) => {
    if (Math.abs(a) < 1e-10) {
      if (Math.abs(b) < 1e-10) {
        return { type: "invalid", message: "Not a valid equation" };
      } else {
        const root = -c / b;
        return {
          type: "linear",
          roots: [root],
          message: "Linear equation (not quadratic)",
        };
      }
    }

    const discriminant = b * b - 4 * a * c;

    if (discriminant < 0) {
      return { type: "complex", message: "Complex roots (no real solutions)" };
    } else if (discriminant === 0) {
      const root = -b / (2 * a);
      return {
        type: "repeated",
        roots: [root],
        message: "One repeated real root",
      };
    } else {
      const sqrtDiscriminant = Math.sqrt(discriminant);
      const root1 = (-b + sqrtDiscriminant) / (2 * a);
      const root2 = (-b - sqrtDiscriminant) / (2 * a);
      return {
        type: "distinct",
        roots: [root1, root2],
        message: "Two distinct real roots",
      };
    }
  },

  // NEW: Calculate solution error
  calculateSolutionError: (predicted, actual) => {
    if (!actual || actual.type === "complex" || actual.type === "invalid") {
      return null;
    }

    const actualRoots = actual.roots;
    if (actualRoots.length === 1) {
      // Single root case
      const error1 = Math.abs(predicted[0] - actualRoots[0]);
      const error2 = Math.abs(predicted[1] - actualRoots[0]);
      return {
        x1_error: error1,
        x2_error: error2,
        avg_error: (error1 + error2) / 2,
        type: "single_root",
      };
    } else {
      // Two roots case - match closest pairs
      const error1 =
        Math.abs(predicted[0] - actualRoots[0]) +
        Math.abs(predicted[1] - actualRoots[1]);
      const error2 =
        Math.abs(predicted[0] - actualRoots[1]) +
        Math.abs(predicted[1] - actualRoots[0]);

      if (error1 <= error2) {
        return {
          x1_error: Math.abs(predicted[0] - actualRoots[0]),
          x2_error: Math.abs(predicted[1] - actualRoots[1]),
          avg_error:
            (Math.abs(predicted[0] - actualRoots[0]) +
              Math.abs(predicted[1] - actualRoots[1])) /
            2,
          type: "two_roots",
        };
      } else {
        return {
          x1_error: Math.abs(predicted[0] - actualRoots[1]),
          x2_error: Math.abs(predicted[1] - actualRoots[0]),
          avg_error:
            (Math.abs(predicted[0] - actualRoots[1]) +
              Math.abs(predicted[1] - actualRoots[0])) /
            2,
          type: "two_roots",
        };
      }
    }
  },
};

// Auto-load dataset from URL parameter
async function checkAndLoadDataset() {
  const urlParams = new URLSearchParams(window.location.search);
  const loadDataset = urlParams.get("load_dataset");

  if (loadDataset) {
    AppState.autoLoadDataset = loadDataset;

    try {
      Utils.showNotification("🔄 Loading generated dataset...", "info");

      const response = await fetch(`/api/data/load/${loadDataset}`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || "Failed to load dataset");
      }

      const result = await response.json();

      // Update app state
      AppState.dataLoaded = true;

      // Update the data display using existing logic
      updateDataStatusDisplay(result);

      Utils.showNotification(
        `✅ Dataset loaded successfully! ${result.total_equations.toLocaleString()} equations ready for training.`,
        "success"
      );

      // Clean URL
      const newUrl = window.location.pathname;
      window.history.replaceState({}, document.title, newUrl);
    } catch (error) {
      Utils.showNotification(
        `❌ Failed to load dataset: ${error.message}`,
        "error"
      );
    }
  }
}

// Update data status display with loaded dataset info
function updateDataStatusDisplay(dataInfo) {
  // Use the existing DataSection to update the display
  DataSection.updateDataInfo(dataInfo);
}

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
        // If it's an external link, allow default browser navigation
        if (link.classList.contains("external-link")) {
          return; // Let the browser handle the navigation
        }

        // For tab navigation, prevent default and handle internally
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

    // Check if this is an auto-loaded dataset
    const autoLoadBadge = dataInfo.auto_loaded
      ? '<div class="auto-load-badge">🎯 Auto-loaded from Dataset Generator</div>'
      : "";

    // Update info display with clear button
    infoContainer.innerHTML = `
        ${autoLoadBadge}
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
                )} ± ${Utils.formatNumber(dataInfo.stats.columns.a.std, 3)}</p>
                <p><strong>Coefficient 'b':</strong> ${Utils.formatNumber(
                  dataInfo.stats.columns.b.mean,
                  3
                )} ± ${Utils.formatNumber(dataInfo.stats.columns.b.std, 3)}</p>
                <p><strong>Coefficient 'c':</strong> ${Utils.formatNumber(
                  dataInfo.stats.columns.c.mean,
                  3
                )} ± ${Utils.formatNumber(dataInfo.stats.columns.c.std, 3)}</p>
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
        <div class="data-actions" style="margin-top: 16px; padding-top: 16px; border-top: 1px solid var(--border-color); display: flex; justify-content: flex-end;">
            <button id="clear-dataset-btn" class="btn btn-danger btn-small">
                <i class="fas fa-trash"></i>
                Clear Dataset
            </button>
        </div>
    `;

    // Add click handler for clear button
    const clearBtn = document.getElementById("clear-dataset-btn");
    if (clearBtn) {
      clearBtn.addEventListener("click", this.clearDataset.bind(this));
    }

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
  async clearDataset() {
    try {
      const confirmed = confirm(
        "Are you sure you want to clear the current dataset? This will also stop any ongoing training and clear all results."
      );

      if (!confirmed) return;

      Utils.showNotification("🗑️ Clearing dataset...", "info");

      const response = await ApiClient.request(API.clearData, {
        method: "POST",
      });

      if (response.success) {
        // Update app state
        AppState.dataLoaded = false;
        AppState.isTraining = false;
        AppState.results = {};

        // Refresh data section to show "no data" state
        await this.refresh();

        Utils.showNotification("✅ Dataset cleared successfully!", "success");
      } else {
        throw new Error(response.error || "Failed to clear dataset");
      }
    } catch (error) {
      console.error("Clear dataset error:", error);
      Utils.showNotification(
        `❌ Failed to clear dataset: ${error.message}`,
        "error"
      );
    }
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
      document.getElementById("stop-training-btn").style.display =
        "inline-block";
    } else {
      progressText.textContent =
        status.progress === 100 ? "Training completed!" : "Ready to train";
      document.getElementById("stop-training-btn").style.display = "none";
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

    // Show/hide random test button based on data availability
    const randomBtn = document.getElementById("random-test-btn");
    if (randomBtn) {
      randomBtn.style.display = AppState.dataLoaded ? "inline-flex" : "none";
    }
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
    Utils.showNotification("Prediction failed: " + error.message, "error");
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
    const originalHTML = randomBtn.innerHTML;
    randomBtn.innerHTML =
      '<i class="fas fa-spinner fa-spin"></i> Rolling the Dice...';
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
    randomBtn.innerHTML = '<i class="fas fa-check"></i> Data Loaded!';
    randomBtn.style.background = "var(--success-color)";
    randomBtn.style.transform = "scale(1)";

    // Auto-submit after showing the populated values
    setTimeout(() => {
      randomBtn.innerHTML = '<i class="fas fa-brain"></i> Predicting...';
      randomBtn.style.background = "var(--primary-color)";
      makePrediction();
    }, 2000);

    // Reset button to original state
    setTimeout(() => {
      randomBtn.innerHTML = originalHTML;
      randomBtn.style.background =
        "linear-gradient(135deg, #667eea 0%, #764ba2 100%)";
      randomBtn.style.boxShadow = "0 4px 12px rgba(102, 126, 234, 0.3)";
      randomBtn.disabled = false;
    }, 5000);
  } catch (error) {
    Utils.showNotification("Random test failed: " + error.message, "error");

    // Reset button on error
    const randomBtn = document.getElementById("random-test-btn");
    randomBtn.innerHTML = '<i class="fas fa-dice"></i> Random Test';
    randomBtn.style.background = "";
    randomBtn.style.transform = "scale(1)";
    randomBtn.disabled = false;
  }
}

// ENHANCED: Display prediction results with new features
function displayPredictionResults(response, inputs) {
  const resultsContainer = document.getElementById("prediction-results");

  // Calculate actual solutions if this is a coefficient-to-roots prediction
  const actualSolutions = Utils.calculateActualSolutions(
    inputs[0],
    inputs[1],
    inputs[2]
  );
  const solutionError = Utils.calculateSolutionError(
    response.predictions,
    actualSolutions
  );

  // Calculate quality metrics for color coding
  const x1Error = solutionError ? Math.abs(solutionError.x1_error) : 0;
  const x2Error = solutionError ? Math.abs(solutionError.x2_error) : 0;
  const avgError = solutionError ? solutionError.avg_error : 0;

  // Determine quality levels
  const getQualityLevel = (error) => {
    if (error < 0.1)
      return {
        level: "excellent",
        color: "var(--success-color)",
        message: "Excellent prediction! 🎯",
      };
    if (error < 0.5)
      return {
        level: "good",
        color: "var(--primary-color)",
        message: "Good prediction! 👍",
      };
    if (error < 1.0)
      return {
        level: "fair",
        color: "var(--warning-color)",
        message: "Fair prediction 🤔",
      };
    return {
      level: "poor",
      color: "var(--error-color)",
      message: "Needs improvement 😅",
    };
  };

  const x1Quality = getQualityLevel(x1Error);
  const x2Quality = getQualityLevel(x2Error);
  const overallQuality = getQualityLevel(avgError);

  let html = `
    <div class="prediction-results-container fade-in">
      <!-- Enhanced Equation Display -->
      <div class="equation-display-section slide-up">
        <h3 class="section-subtitle">
          <i class="fas fa-function"></i>
          Quadratic Equation
        </h3>
        <div class="equation-display animated-equation">
          ${Utils.formatQuadraticEquation(inputs[0], inputs[1], inputs[2])}
        </div>
      </div>

      <!-- Overall Quality Badge -->
      <div class="quality-badge-container scale-in">
        <div class="quality-badge quality-${overallQuality.level}">
          <div class="quality-icon">
            ${
              overallQuality.level === "excellent"
                ? "🎯"
                : overallQuality.level === "good"
                ? "👍"
                : overallQuality.level === "fair"
                ? "🤔"
                : "😅"
            }
          </div>
          <div class="quality-message">${overallQuality.message}</div>
          <div class="quality-metric">Avg Error: ${Utils.formatNumber(
            avgError,
            4
          )}</div>
        </div>
      </div>

      <!-- Enhanced Solution Comparison -->
      <div class="solution-comparison-section slide-up">
        <h4 class="comparison-title">
          <i class="fas fa-balance-scale"></i>
          Solution Comparison
        </h4>
        
        <div class="solution-comparison-grid">
          <!-- Neural Network Prediction -->
          <div class="solution-column neural-prediction">
            <div class="solution-header neural-network">
              <i class="fas fa-brain"></i>
              <span>Neural Network</span>
            </div>
            <div class="solution-values">
              <div class="solution-value">
                <span class="solution-label">x₁ =</span>
                <span class="solution-number nn-prediction" style="color: ${
                  x1Quality.color
                }; text-shadow: 0 0 8px ${x1Quality.color}30;">
                  ${Utils.formatNumber(response.predictions[0], 6)}
                </span>
              </div>
              <div class="solution-value">
                <span class="solution-label">x₂ =</span>
                <span class="solution-number nn-prediction" style="color: ${
                  x2Quality.color
                }; text-shadow: 0 0 8px ${x2Quality.color}30;">
                  ${Utils.formatNumber(response.predictions[1], 6)}
                </span>
              </div>
            </div>
            <div class="prediction-confidence">
              <span class="confidence-label">Confidence:</span>
              <span class="confidence-value">${Utils.getConfidenceLevel(
                response.confidences[0]
              )}</span>
            </div>
          </div>

          <!-- Actual Solution -->
          <div class="solution-column actual-solution">
            <div class="solution-header actual-solution">
              <i class="fas fa-check-circle"></i>
              <span>Actual Solution</span>
            </div>
            <div class="solution-values">
              ${
                actualSolutions.type === "distinct"
                  ? `
                <div class="solution-value">
                  <span class="solution-label">x₁ =</span>
                  <span class="solution-number actual-solution">
                    ${Utils.formatNumber(actualSolutions.roots[0], 6)}
                  </span>
                </div>
                <div class="solution-value">
                  <span class="solution-label">x₂ =</span>
                  <span class="solution-number actual-solution">
                    ${Utils.formatNumber(actualSolutions.roots[1], 6)}
                  </span>
                </div>
              `
                  : actualSolutions.type === "repeated"
                  ? `
                <div class="solution-value">
                  <span class="solution-label">x₁ = x₂ =</span>
                  <span class="solution-number actual-solution">
                    ${Utils.formatNumber(actualSolutions.roots[0], 6)}
                  </span>
                </div>
              `
                  : `
                <div style="color: var(--text-secondary); font-style: italic; text-align: center; padding: 16px;">
                  ${actualSolutions.message}
                </div>
              `
              }
            </div>
            <div class="solution-message">
              Mathematical ground truth
            </div>
          </div>
        </div>
      </div>

      ${
        solutionError
          ? `
      <!-- Enhanced Error Analysis -->
      <div class="error-analysis-enhanced slide-up">
        <h4 class="error-title">
          <i class="fas fa-chart-line"></i>
          Error Analysis
        </h4>
        
        <div class="error-metrics-grid">
          <div class="error-metric-card" style="border-color: ${
            x1Quality.color
          }; background: linear-gradient(135deg, ${
              x1Quality.color
            }05, var(--surface-color));">
            <div class="metric-icon">📊</div>
            <div class="metric-label">x₁ Error</div>
            <div class="metric-value" style="color: ${
              x1Quality.color
            };">${Utils.formatNumber(x1Error, 6)}</div>
            <div class="metric-status" style="background: ${
              x1Quality.color
            }20; color: ${x1Quality.color};">${x1Quality.message}</div>
          </div>
          
          <div class="error-metric-card" style="border-color: ${
            x2Quality.color
          }; background: linear-gradient(135deg, ${
              x2Quality.color
            }05, var(--surface-color));">
            <div class="metric-icon">📈</div>
            <div class="metric-label">x₂ Error</div>
            <div class="metric-value" style="color: ${
              x2Quality.color
            };">${Utils.formatNumber(x2Error, 6)}</div>
            <div class="metric-status" style="background: ${
              x2Quality.color
            }20; color: ${x2Quality.color};">${x2Quality.message}</div>
          </div>
          
          <div class="error-metric-card" style="border-color: ${
            overallQuality.color
          }; background: linear-gradient(135deg, ${
              overallQuality.color
            }05, var(--surface-color));">
            <div class="metric-icon">🎯</div>
            <div class="metric-label">Average Error</div>
            <div class="metric-value" style="color: ${
              overallQuality.color
            };">${Utils.formatNumber(avgError, 6)}</div>
            <div class="metric-status" style="background: ${
              overallQuality.color
            }20; color: ${overallQuality.color};">${
              overallQuality.message
            }</div>
          </div>
        </div>
      </div>
      `
          : ""
      }

      <!-- Performance Insights -->
      <div class="performance-insights slide-up">
        <h4 class="insights-title">
          <i class="fas fa-lightbulb"></i>
          Performance Insights
        </h4>
        <div class="insights-content">
          <div class="insight-item">
            <strong>Prediction Type:</strong> Coefficient to Root Prediction
          </div>
          <div class="insight-item">
            <strong>Confidence Level:</strong> ${Utils.getConfidenceLevel(
              response.confidences[0]
            )}
          </div>
          <div class="insight-item">
            <strong>Overall Assessment:</strong> 
            <span style="color: ${overallQuality.color}; font-weight: 600;">
              ${
                overallQuality.level.charAt(0).toUpperCase() +
                overallQuality.level.slice(1)
              }
            </span>
          </div>
        </div>
      </div>

      <!-- Original Results Grid (preserved for compatibility) -->
      <div class="original-results-grid slide-up" style="display: grid; gap: 16px; margin-top: 20px; padding: 20px; background: var(--surface-color); border-radius: var(--radius-medium); border: 1px solid var(--border-color);">
        <h4 style="margin: 0 0 16px 0; display: flex; align-items: center; gap: 8px;">
          <i class="fas fa-list"></i>
          Detailed Results
        </h4>
  `;

  response.target_features.forEach((feature, index) => {
    const prediction = response.predictions[index];
    const confidence = response.confidences[index];
    const confidenceLevel = Utils.getConfidenceLevel(confidence);
    const errorValue = index === 0 ? x1Error : x2Error;
    const qualityColor = index === 0 ? x1Quality.color : x2Quality.color;

    html += `
      <div style="display: flex; justify-content: space-between; align-items: center; padding: 16px; background: var(--background-color); border-radius: var(--radius-medium); border: 1px solid var(--border-color); transition: all 0.3s ease;" onmouseover="this.style.transform='translateY(-2px)'; this.style.boxShadow='var(--shadow-light)'" onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='none'">
        <div>
          <strong style="color: ${qualityColor};">${feature}:</strong> 
          <span style="font-family: 'JetBrains Mono', monospace; color: ${qualityColor}; font-weight: 600;">${Utils.formatNumber(
      prediction,
      6
    )}</span>
        </div>
        <div style="text-align: right;">
          <div>Confidence: <span style="font-weight: 600;">${Utils.formatPercentage(
            confidence * 100,
            1
          )}</span></div>
          <div style="font-size: 14px; margin-top: 4px;">${confidenceLevel}</div>
          ${
            solutionError
              ? `<div style="font-size: 12px; color: ${qualityColor}; margin-top: 2px;">Error: ${Utils.formatNumber(
                  errorValue,
                  4
                )}</div>`
              : ""
          }
        </div>
      </div>
    `;
  });

  html += `
      </div>
    </div>
  `;

  resultsContainer.innerHTML = html;

  // Trigger animations with delays
  setTimeout(() => {
    const elementsToAnimate = resultsContainer.querySelectorAll(".slide-up");
    elementsToAnimate.forEach((el, index) => {
      el.style.animationDelay = `${index * 0.1}s`;
    });
  }, 100);
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
document.addEventListener("DOMContentLoaded", async () => {
  console.log("🚀 Quadratic Neural Network Web Application");
  console.log("Initializing application...");

  // Initialize all sections
  Navigation.init();
  DataSection.init();
  TrainingSection.init();
  PredictionSection.init();
  AnalysisSection.init();
  ComparisonSection.init();

  // Check for auto-load dataset from URL parameter
  await checkAndLoadDataset();

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
        document.getElementById("stop-training-btn").style.display = "none";
        Utils.showNotification("Training completed!", "success");
      }
    } catch (error) {
      console.error("Failed to check training status:", error);
    }
  }
}, 2000);
