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
  savedModels: [],
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
  modelsList: "/api/models/list",
  modelsSave: "/api/models/save",
  modelsLoad: "/api/models/load",
  modelsDelete: "/api/models/delete",
  modelsInfo: "/api/models/info",
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
      case "model-management":
        ModelSection.refresh();
        ModelSection.refreshAppState();
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
    // Only show dataset warning when explicitly navigating to training section
    // Don't show it when refreshing due to model loading
    const isTrainingTabActive = AppState.currentSection === "training";

    if (!AppState.dataLoaded && isTrainingTabActive) {
      Utils.showNotification("Please load a dataset first", "warning");
      return;
    }

    await this.loadScenarios();
  },
};

// Model Manager functionality
const ModelSection = {
  init() {
    this.setupEventListeners();
  },

  // Replace the existing ModelSection.refresh() method with this enhanced version:
  async refresh() {
    console.log("🔄 Refreshing Model Management tab...");

    // Always load saved models list
    await this.loadSavedModelsList();

    // Fetch current results from backend to update AppState.results
    await this.refreshAppState();

    // Update save section based on current state
    this.updateSaveSection();
  },

  // Add this new method to ModelSection:
  async refreshAppState() {
    try {
      // Fetch current results from backend
      const results = await ApiClient.request(API.results);
      AppState.results = results;
      console.log(
        "✅ AppState.results updated:",
        Object.keys(AppState.results)
      );

      // Also check data loaded state
      const dataInfo = await ApiClient.request(API.dataInfo);
      AppState.dataLoaded = dataInfo.loaded;
    } catch (error) {
      console.error("Failed to refresh app state:", error);
    }
  },

  setupEventListeners() {
    const saveBtn = document.getElementById("saveModelBtn");
    const loadBtn = document.getElementById("loadModelBtn");
    const deleteBtn = document.getElementById("deleteModelBtn");
    const refreshBtn = document.getElementById("refreshModelsBtn");
    const modelsSelect = document.getElementById("savedModelsSelect");
    const selectAllBtn = document.getElementById("selectAllModelsBtn");
    const deselectAllBtn = document.getElementById("deselectAllModelsBtn");

    if (saveBtn) saveBtn.addEventListener("click", this.saveModel.bind(this));
    if (loadBtn) loadBtn.addEventListener("click", this.loadModel.bind(this));
    if (deleteBtn)
      deleteBtn.addEventListener("click", this.deleteModel.bind(this));
    if (refreshBtn)
      refreshBtn.addEventListener("click", this.loadSavedModelsList.bind(this));
    if (modelsSelect)
      modelsSelect.addEventListener("change", this.onModelSelect.bind(this));
    if (selectAllBtn) {
      selectAllBtn.addEventListener("click", () => {
        document
          .querySelectorAll("#modelsGrid input[type='checkbox']")
          .forEach((cb) => {
            cb.checked = true;
            cb.closest(".model-checkbox-card").classList.add("selected");
          });
        this.updateSelectionStatus();
      });
    }

    if (deselectAllBtn) {
      deselectAllBtn.addEventListener("click", () => {
        document
          .querySelectorAll("#modelsGrid input[type='checkbox']")
          .forEach((cb) => {
            cb.checked = false;
            cb.closest(".model-checkbox-card").classList.remove("selected");
          });
        this.updateSelectionStatus();
      });
    }
  },

  async loadSavedModelsList() {
    try {
      const response = await fetch(API.modelsList);
      const data = await response.json();

      if (data.success) {
        AppState.savedModels = data.models;
        // **KEY FIX: Call the new grid update method**
        this.updateModelsGrid();
        console.log(`✅ Loaded ${data.models.length} saved models`);
      }
    } catch (error) {
      console.error("Failed to load saved models:", error);
      Utils.showNotification("Failed to load saved models", "error");
    }
  },

  updateModelsDropdown() {
    const select = document.getElementById("savedModelsSelect");
    if (!select) return;

    select.innerHTML = '<option value="">Select a saved model...</option>';

    if (AppState.savedModels) {
      AppState.savedModels.forEach((model) => {
        const option = document.createElement("option");
        option.value = model.model_id;
        option.textContent = `${model.model_name} (${
          model.scenario_name
        }) - ${new Date(model.created_date).toLocaleDateString()}`;
        select.appendChild(option);
      });
    }
  },

  updateSaveSection() {
    const section = document.getElementById("modelSaveSection");
    const select = document.getElementById("scenarioSelect");
    const checkbox = document.getElementById("saveAllModelsCheckbox");

    if (!section || !select) return;

    // Check if any models are trained
    const trainedScenarios = Object.keys(AppState.results || {});

    if (trainedScenarios.length > 0) {
      section.style.display = "block";

      // Update scenario dropdown
      select.innerHTML = '<option value="">Choose scenario to save...</option>';
      trainedScenarios.forEach((key) => {
        const scenario = AppState.scenarios[key];
        if (scenario) {
          const option = document.createElement("option");
          option.value = key;
          option.textContent = scenario.name;
          select.appendChild(option);
        }
      });

      // Setup checkbox event listener
      if (checkbox && !checkbox.hasEventListener) {
        checkbox.addEventListener("change", this.onSaveAllToggle.bind(this));
        checkbox.hasEventListener = true;
      }

      // Show save all option only if multiple models are trained
      const saveAllContainer = checkbox?.closest(".save-mode-selection");
      if (saveAllContainer) {
        saveAllContainer.style.display =
          trainedScenarios.length > 1 ? "block" : "none";
      }
    } else {
      section.style.display = "none";
    }
  },

  async saveModel() {
    const modelName = document.getElementById("modelNameInput").value.trim();
    const isBoxChecked = document.getElementById(
      "saveAllModelsCheckbox"
    ).checked;

    if (!modelName) {
      const label = isBoxChecked ? "model prefix" : "model name";
      Utils.showNotification(`Please enter a ${label}`, "error");
      return;
    }

    try {
      let response;

      if (isBoxChecked) {
        // **BATCH SAVE MODE**
        Utils.showNotification("🚀 Saving all trained models...", "info");

        response = await fetch(API.modelsSave + "-batch", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            model_prefix: modelName,
          }),
        });

        const data = await response.json();

        if (data.success) {
          const successMsg = `✅ ${data.message}${
            data.warning ? ` (${data.warning})` : ""
          }`;
          Utils.showNotification(successMsg, "success");

          // Clear input and reset to single mode
          document.getElementById("modelNameInput").value = "";
          document.getElementById("saveAllModelsCheckbox").checked = false;
          this.onSaveAllToggle(); // Reset UI to single mode
        } else {
          Utils.showNotification(data.error, "error");
        }
      } else {
        // **SINGLE MODEL SAVE MODE (Legacy)**
        const scenarioKey = document.getElementById("scenarioSelect").value;

        if (!scenarioKey) {
          Utils.showNotification("Please select a scenario", "error");
          return;
        }

        Utils.showNotification("💾 Saving model...", "info");

        response = await fetch(API.modelsSave, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            model_name: modelName,
            scenario_key: scenarioKey,
          }),
        });

        const data = await response.json();

        if (data.success) {
          Utils.showNotification(data.message, "success");
          document.getElementById("modelNameInput").value = "";
          document.getElementById("scenarioSelect").value = "";
        } else {
          Utils.showNotification(data.error, "error");
        }
      }

      // **CRITICAL FIX: Refresh load section after any save**
      await this.loadSavedModelsList();
    } catch (error) {
      Utils.showNotification("Failed to save model(s)", "error");
      console.error("Save model error:", error);
    }
  },

  updateModelsGrid() {
    const modelsGrid = document.getElementById("modelsGrid");
    const batchControls = document.getElementById("batchSelectionControls");
    const modelsCountBadge = document.getElementById("modelsCountBadge");

    if (!modelsGrid) return;

    const models = AppState.savedModels || [];

    if (models.length === 0) {
      modelsGrid.innerHTML = `
      <div class="no-models-message">
        <i class="fas fa-folder-open"></i>
        <p>No saved models available</p>
      </div>
    `;
      batchControls.style.display = "none";
      modelsCountBadge.style.display = "none";
      return;
    }

    // Show batch controls and count
    batchControls.style.display = "flex";
    modelsCountBadge.style.display = "inline-flex";
    modelsCountBadge.textContent = models.length;

    // Generate model cards
    modelsGrid.innerHTML = "";

    models.forEach((model) => {
      const modelCard = document.createElement("div");
      modelCard.className = "model-checkbox-card";

      const createdDate = new Date(model.created_date).toLocaleDateString();
      const displayName = model.display_name || model.model_name;
      const isBatchModel = model.is_batch_model || false;

      modelCard.innerHTML = `
      <input type="checkbox" id="model-${model.model_id}" value="${
        model.model_id
      }">
      <div class="model-checkbox-checkmark"></div>
      <div class="model-checkbox-content">
        <div class="model-checkbox-title">${displayName}</div>
        <div class="model-checkbox-meta">
          <div class="model-checkbox-scenario">
            <span>${model.scenario_name}</span>
            ${
              isBatchModel
                ? '<div class="model-checkbox-badge">BATCH</div>'
                : ""
            }
          </div>
          <div class="model-checkbox-date">${createdDate}</div>
        </div>
        ${
          model.performance_metrics
            ? `
        <div class="model-checkbox-stats">
          <div class="model-stat">
            <div class="model-stat-label">R² Score</div>
            <div class="model-stat-value">${(
              model.performance_metrics.r2 * 100
            ).toFixed(1)}%</div>
            <div class="model-stat-progress">
              <div class="model-stat-progress-fill" style="width: ${(
                model.performance_metrics.r2 * 100
              ).toFixed(1)}%"></div>
            </div>
          </div>
          <div class="model-stat">
            <div class="model-stat-label">Accuracy</div>
            <div class="model-stat-value">${
              model.performance_metrics.accuracy_10pct?.toFixed(1) || 0
            }%</div>
            <div class="model-stat-progress">
              <div class="model-stat-progress-fill" style="width: ${
                model.performance_metrics.accuracy_10pct?.toFixed(1) || 0
              }%"></div>
            </div>
          </div>
        </div>
        `
            : ""
        }

      </div>
    `;

      // Add click handler for entire card
      modelCard.addEventListener("click", (e) => {
        if (e.target.type !== "checkbox") {
          const checkbox = modelCard.querySelector("input[type='checkbox']");
          checkbox.checked = !checkbox.checked;
          checkbox.dispatchEvent(new Event("change"));
        }
      });

      // Add change handler for checkbox
      const checkbox = modelCard.querySelector("input[type='checkbox']");
      checkbox.addEventListener("change", () => {
        modelCard.classList.toggle("selected", checkbox.checked);
        this.updateSelectionStatus();
      });

      modelsGrid.appendChild(modelCard);
    });
  },

  updateSelectionStatus() {
    const checkboxes = document.querySelectorAll(
      "#modelsGrid input[type='checkbox']"
    );
    const selected = document.querySelectorAll(
      "#modelsGrid input[type='checkbox']:checked"
    );
    const statusElement = document.getElementById("selectionStatus");
    const loadButton = document.getElementById("loadModelBtn");
    const loadButtonText = document.getElementById("loadButtonText");

    const selectedCount = selected.length;
    const totalCount = checkboxes.length;

    if (statusElement) {
      if (selectedCount === 0) {
        statusElement.textContent = "No models selected";
      } else if (selectedCount === 1) {
        statusElement.textContent = "1 model selected";
      } else {
        statusElement.textContent = `${selectedCount} models selected`;
      }
    }

    if (loadButton && loadButtonText) {
      loadButton.disabled = selectedCount === 0;
      if (selectedCount === 0) {
        loadButtonText.textContent = "Load Models";
      } else if (selectedCount === 1) {
        loadButtonText.textContent = "Load Model";
      } else {
        loadButtonText.textContent = `Load ${selectedCount} Models`;
      }
    }

    // Display info for selected models
    this.displaySelectedModelsInfo();
  },

  async loadModel() {
    // Get selected model IDs from checkboxes (NEW APPROACH)
    const selectedCheckboxes = document.querySelectorAll(
      "#modelsGrid input[type='checkbox']:checked"
    );
    const modelIds = Array.from(selectedCheckboxes).map((cb) => cb.value);

    if (modelIds.length === 0) {
      Utils.showNotification(
        "Please select at least one model to load",
        "error"
      );
      return;
    }

    // --- VALIDATION FOR DUPLICATE MODEL TYPES ---
    const selectedScenarios = new Set();
    for (const modelId of modelIds) {
      // Find the full model object from the application state[1]
      const model = AppState.savedModels.find((m) => m.model_id === modelId);
      if (model) {
        // Check if a model for this scenario has already been selected[1]
        if (selectedScenarios.has(model.scenario_key)) {
          Utils.showNotification(
            `Duplicate model type: You can only load one model for the "${model.scenario_name}" scenario at a time.`,
            "error"
          );
          return; // Stop the loading process
        }
        selectedScenarios.add(model.scenario_key);
      }
    }
    // --- END OF VALIDATION ---

    try {
      const loadingMessage =
        modelIds.length === 1
          ? "💾 Loading model..."
          : `💾 Loading ${modelIds.length} models...`;

      Utils.showNotification(loadingMessage, "info");

      const response = await fetch(API.modelsLoad, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model_ids: modelIds, // Send array for batch loading[2]
        }),
      });

      const data = await response.json();

      if (data.success) {
        // Success notification
        const successMessage =
          modelIds.length === 1
            ? `✅ Model "${data.loaded_models[0].model_name}" loaded successfully!`
            : `✅ ${data.loaded_count}/${data.total_count} models loaded successfully!`;

        Utils.showNotification(successMessage, "success");

        // Show warnings if some models failed
        if (data.warning) {
          Utils.showNotification(`⚠️ ${data.warning}`, "warning");
        }

        // Update frontend state to reflect loaded models
        await this.updateAppStateAfterLoad(data);

        // Clear selections after successful load
        this.clearModelSelections();
      } else {
        Utils.showNotification(data.error, "error");
      }
    } catch (error) {
      Utils.showNotification("Failed to load model(s)", "error");
      console.error("Load model error:", error);
    }
  },

  // Method to update app state after loading a model
  async updateAppStateAfterLoad(data) {
    try {
      // 1. Update AppState.results directly from loaded models
      data.loaded_models.forEach((model) => {
        // The backend should have already updated the results
        console.log(
          `✅ Model loaded: ${model.model_name} (${model.scenario_key})`
        );
      });

      // 2. Fetch updated results from backend to ensure sync
      try {
        const results = await ApiClient.request(API.results);
        AppState.results = results;
        console.log(
          "✅ Results updated after model load:",
          Object.keys(AppState.results)
        );
      } catch (error) {
        console.warn("Could not fetch updated results:", error);
      }

      // 3. Update save section to show newly available trained models
      this.updateSaveSection();

      // 4. Refresh all dependent sections
      this.refreshDependentSections();

      // 5. Display loaded model info for single model loads
      if (data.loaded_count === 1 && data.model_info) {
        this.displayModelInfo(data.model_info);
      }

      console.log("✅ App state fully updated after model load");
    } catch (error) {
      console.error("Failed to update app state after model load:", error);
      Utils.showNotification(
        "Models loaded but some features may not be updated. Please refresh the page.",
        "warning"
      );
    }
  },
  // Method to refresh all sections that depend on trained models
  refreshDependentSections() {
    // Refresh prediction section to show random button
    if (typeof PredictionSection !== "undefined" && PredictionSection.refresh) {
      PredictionSection.refresh();
    }

    // Enable analysis generation
    const generateAnalysisBtn = document.getElementById(
      "generate-analysis-btn"
    );
    if (generateAnalysisBtn) {
      generateAnalysisBtn.disabled = false;
      generateAnalysisBtn.style.opacity = "1";
    }

    // Enable comparison generation
    const generateComparisonBtn = document.getElementById(
      "generate-comparison-btn"
    );
    if (generateComparisonBtn) {
      generateComparisonBtn.disabled = false;
      generateComparisonBtn.style.opacity = "1";
    }

    // Update training section if needed
    if (typeof TrainingSection !== "undefined" && TrainingSection.refresh) {
      TrainingSection.refresh();
    }

    // Update data section display
    if (typeof DataSection !== "undefined" && DataSection.refresh) {
      DataSection.refresh();
    }

    console.log("✅ All dependent sections refreshed");
  },

  clearModelSelections() {
    // Clear all checkbox selections
    document
      .querySelectorAll("#modelsGrid input[type='checkbox']")
      .forEach((cb) => {
        cb.checked = false;
        cb.closest(".model-checkbox-card").classList.remove("selected");
      });
    this.updateSelectionStatus();
  },

  async updateAppStateAfterLoad(data) {
    try {
      // 1. Fetch updated results from backend
      const results = await ApiClient.request(API.results);
      AppState.results = results;
      console.log("✅ Results updated after model load:", AppState.results);

      // 2. Check and update data loaded state
      const dataInfo = await ApiClient.request(API.dataInfo);
      AppState.dataLoaded = dataInfo.loaded;

      // 3. Update save section to show newly available trained models
      this.updateSaveSection();

      // 4. Refresh all dependent sections
      this.refreshDependentSections();

      // 5. Display loaded model info for single model loads
      if (data.loaded_count === 1 && data.model_info) {
        this.displayModelInfo(data.model_info);
      }

      console.log("✅ App state fully updated after model load");
    } catch (error) {
      console.error("Failed to update app state after model load:", error);
      Utils.showNotification(
        "Models loaded but some features may not be updated. Please refresh the page.",
        "warning"
      );
    }
  },

  refreshDependentSections() {
    // Refresh prediction section to show random button
    if (typeof PredictionSection !== "undefined" && PredictionSection.refresh) {
      PredictionSection.refresh();
    }

    // Enable analysis generation
    const generateAnalysisBtn = document.getElementById(
      "generate-analysis-btn"
    );
    if (generateAnalysisBtn) {
      generateAnalysisBtn.disabled = false;
      generateAnalysisBtn.style.opacity = "1";
    }

    // Enable comparison generation
    const generateComparisonBtn = document.getElementById(
      "generate-comparison-btn"
    );
    if (generateComparisonBtn) {
      generateComparisonBtn.disabled = false;
      generateComparisonBtn.style.opacity = "1";
    }

    // Update training section if needed
    if (typeof TrainingSection !== "undefined" && TrainingSection.refresh) {
      TrainingSection.refresh();
    }

    console.log("✅ All dependent sections refreshed");
  },

  async deleteModel() {
    const selectedCheckboxes = document.querySelectorAll(
      "#modelsGrid input[type='checkbox']:checked"
    );
    const modelIds = Array.from(selectedCheckboxes).map((cb) => cb.value);

    if (modelIds.length === 0) {
      Utils.showNotification(
        "Please select at least one model to delete",
        "error"
      );
      return;
    }

    const confirmMessage =
      modelIds.length === 1
        ? "Are you sure you want to delete this model? This action cannot be undone."
        : `Are you sure you want to delete these ${modelIds.length} models? This action cannot be undone.`;

    if (!confirm(confirmMessage)) {
      return;
    }

    try {
      let deletedCount = 0;
      let failedCount = 0;

      for (const modelId of modelIds) {
        try {
          const response = await fetch(API.modelsDelete, {
            method: "DELETE",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ model_id: modelId }),
          });

          const data = await response.json();
          if (data.success) {
            deletedCount++;
          } else {
            failedCount++;
          }
        } catch {
          failedCount++;
        }
      }

      if (deletedCount > 0) {
        const message =
          modelIds.length === 1
            ? "Model deleted successfully"
            : `${deletedCount}/${modelIds.length} models deleted successfully`;

        Utils.showNotification(message, "success");

        // Refresh the models list
        await this.loadSavedModelsList();

        // Hide model info display
        document.getElementById("modelInfoDisplay").style.display = "none";
      }

      if (failedCount > 0) {
        Utils.showNotification(
          `${failedCount} models failed to delete`,
          "warning"
        );
      }
    } catch (error) {
      Utils.showNotification("Failed to delete model(s)", "error");
      console.error("Delete model error:", error);
    }
  },

  onModelSelect() {
    const modelId = document.getElementById("savedModelsSelect").value;
    const model = AppState.savedModels?.find((m) => m.model_id === modelId);

    if (model) {
      this.displayModelInfo(model);
    } else {
      document.getElementById("modelInfoDisplay").style.display = "none";
    }
  },

  displayModelInfo(model) {
    const display = document.getElementById("modelInfoDisplay");
    const content = document.getElementById("modelInfoContent");

    if (!display || !content) return;

    const createdDate = new Date(model.created_date).toLocaleString();
    const r2Score = model.performance_metrics?.r2 || 0;

    content.innerHTML = `
      <div class="model-meta-item">
        <strong>Name:</strong><br><span>${model.model_name}</span>
      </div>
      <div class="model-meta-item">
        <strong>Scenario:</strong><br><span>${model.scenario_name}</span>
      </div>
      <div class="model-meta-item">
        <strong>Dataset Size:</strong><br><span>${model.dataset_size.toLocaleString()} equations</span>
      </div>
      <div class="model-meta-item">
        <strong>Created:</strong><br><span>${createdDate}</span>
      </div>
      <div class="model-meta-item">
        <strong>R² Score:</strong><br><span>${Utils.formatNumber(
          r2Score,
          4
        )}</span>
      </div>
      <div class="model-meta-item">
        <strong>Training Time:</strong><br><span>${
          model.performance_metrics?.training_time?.toFixed(2) || "N/A"
        }s</span>
      </div>
    `;

    display.style.display = "block";
  },
  onSaveAllToggle() {
    const checkbox = document.getElementById("saveAllModelsCheckbox");
    const isChecked = checkbox.checked;

    // Update UI elements
    const modelNameLabel = document.getElementById("modelNameLabel");
    const modelNameInput = document.getElementById("modelNameInput");
    const modelNameHelp = document.getElementById("modelNameHelp");
    const scenarioGroup = document.getElementById("singleModelScenarioGroup");
    const batchPreview = document.getElementById("batchSavePreview");
    const saveButtonText = document.getElementById("saveButtonText");

    if (isChecked) {
      // Switch to batch mode
      modelNameLabel.textContent = "Model Prefix";
      modelNameInput.placeholder =
        "Enter prefix for all models (e.g., 'experiment1')...";
      modelNameHelp.style.display = "block";
      scenarioGroup.style.display = "none";
      batchPreview.style.display = "block";
      saveButtonText.textContent = "Save All Models";

      // Update batch preview
      this.updateBatchPreview();
    } else {
      // Switch to single mode
      modelNameLabel.textContent = "Model Name";
      modelNameInput.placeholder = "Enter model name...";
      modelNameHelp.style.display = "none";
      scenarioGroup.style.display = "block";
      batchPreview.style.display = "none";
      saveButtonText.textContent = "Save Model";
    }
  },
  updateBatchPreview() {
    const previewList = document.getElementById("batchPreviewList");
    const prefix =
      document.getElementById("modelNameInput").value.trim() || "model";

    if (!previewList) return;

    previewList.innerHTML = "";

    const trainedScenarios = Object.keys(AppState.results || {});
    trainedScenarios.forEach((key) => {
      const scenario = AppState.scenarios[key];
      if (scenario) {
        const previewItem = document.createElement("div");
        previewItem.className = "batch-preview-item";
        previewItem.innerHTML = `
        <div class="batch-preview-icon" style="background: ${scenario.color};"></div>
        <div style="flex: 1;">
          <div class="batch-preview-name">${prefix}_${key}</div>
          <div class="batch-preview-scenario">${scenario.name}</div>
        </div>
      `;
        previewList.appendChild(previewItem);
      }
    });

    // Update preview on input change
    const modelNameInput = document.getElementById("modelNameInput");
    if (modelNameInput && !modelNameInput.hasPreviewListener) {
      modelNameInput.addEventListener(
        "input",
        Utils.debounce(() => {
          if (document.getElementById("saveAllModelsCheckbox").checked) {
            this.updateBatchPreview();
          }
        }, 300)
      );
      modelNameInput.hasPreviewListener = true;
    }
  },
  displaySelectedModelsInfo() {
    const selectedCheckboxes = document.querySelectorAll(
      "#modelsGrid input[type='checkbox']:checked"
    );
    const selectedModelIds = Array.from(selectedCheckboxes).map(
      (cb) => cb.value
    );
    const display = document.getElementById("modelInfoDisplay");
    const content = document.getElementById("modelInfoContent");

    if (!display || !content) return;

    // Hide if no models selected
    if (selectedModelIds.length === 0) {
      display.style.display = "none";
      return;
    }

    // Get selected models data
    const selectedModels =
      AppState.savedModels?.filter((model) =>
        selectedModelIds.includes(model.model_id)
      ) || [];

    if (selectedModels.length === 0) {
      display.style.display = "none";
      return;
    }

    // Update header based on selection count
    const headerTitle = display.querySelector(".card-title");
    if (headerTitle) {
      const countText =
        selectedModels.length === 1
          ? "Model Information"
          : `${selectedModels.length} Models Selected`;
      headerTitle.innerHTML = `<i class="fas fa-info-circle"></i> ${countText}`;
    }

    // Generate info cards for selected models
    content.innerHTML = selectedModels
      .map((model) => this.generateModelInfoCard(model))
      .join("");

    // Show the display
    display.style.display = "block";
  },

  generateModelInfoCard(model) {
    const createdDate = new Date(model.created_date).toLocaleDateString(
      "en-US",
      {
        year: "numeric",
        month: "short",
        day: "numeric",
      }
    );
    const createdTime = new Date(model.created_date).toLocaleTimeString(
      "en-US",
      {
        hour: "2-digit",
        minute: "2-digit",
      }
    );

    // Performance metrics with proper scaling
    const r2Score = (model.performance_metrics?.r2 || 0) * 100;
    const accuracy = model.performance_metrics?.accuracy_10pct || 0;
    const mse = model.performance_metrics?.mse || 0;
    const mae = model.performance_metrics?.mae || 0;
    const trainingTime = model.performance_metrics?.training_time || 0;

    // Additional model details
    const modelSize = model.model_size_bytes
      ? (model.model_size_bytes / 1024).toFixed(1) + " KB"
      : "N/A";
    const datasetSize = (model.dataset_size || 0).toLocaleString();
    const version = model.version || "1.0";
    const description = model.description || "No description available";

    // Performance quality assessment
    const getPerformanceQuality = (r2, acc) => {
      const avgPerf = (r2 + acc) / 2;
      if (avgPerf >= 85)
        return {
          level: "excellent",
          color: "#34C759",
          bgColor: "rgba(52, 199, 89, 0.1)",
          icon: "🏆",
          label: "Excellent",
        };
      if (avgPerf >= 70)
        return {
          level: "good",
          color: "#007AFF",
          bgColor: "rgba(0, 122, 255, 0.1)",
          icon: "👍",
          label: "Very Good",
        };
      if (avgPerf >= 50)
        return {
          level: "fair",
          color: "#FF9500",
          bgColor: "rgba(255, 149, 0, 0.1)",
          icon: "⚡",
          label: "Good",
        };
      return {
        level: "poor",
        color: "#FF3B30",
        bgColor: "rgba(255, 59, 48, 0.1)",
        icon: "🔧",
        label: "Needs Work",
      };
    };

    const quality = getPerformanceQuality(r2Score, accuracy);
    const isBatchModel = model.is_batch_model || false;

    return `
    <div class="enhanced-model-info-card" data-model-id="${model.model_id}">
      <!-- Card Header with Model Name and Quality Badge -->
      <div class="model-info-header">
        <div class="model-title-section">
          <div class="model-name-container">
            <h3 class="model-name">${model.model_name}</h3>
            ${
              isBatchModel
                ? `
              <div class="batch-indicator">
                <span class="batch-box">📦</span>
                <span class="batch-text">BATCH</span>
              </div>
            `
                : ""
            }
          </div>
          <div class="model-scenario">${model.scenario_name}</div>
        </div>
        
        <div class="quality-indicator">
          <div class="quality-badge quality-${quality.level}">
            <div class="quality-icon">${quality.icon}</div>
            <div class="quality-label">${quality.label}</div>
          </div>
        </div>
      </div>

      <!-- Main Metrics Grid -->
      <div class="metrics-showcase-grid">
        <!-- R² Score Card -->
        <div class="metric-card r2-card">
          <div class="metric-header">
            <div class="metric-icon">📊</div>
            <div class="metric-label">R² Score</div>
          </div>
          <div class="metric-value-container">
            <div class="metric-value">${r2Score.toFixed(1)}%</div>
            <div class="metric-progress">
              <div class="metric-progress-fill r2-fill" style="width: ${r2Score}%"></div>
            </div>
          </div>
          <div class="metric-description">Variance Explained</div>
        </div>

        <!-- Accuracy Card -->
        <div class="metric-card accuracy-card">
          <div class="metric-header">
            <div class="metric-icon">🎯</div>
            <div class="metric-label">Accuracy</div>
          </div>
          <div class="metric-value-container">
            <div class="metric-value">${accuracy.toFixed(1)}%</div>
            <div class="metric-progress">
              <div class="metric-progress-fill accuracy-fill" style="width: ${accuracy}%"></div>
            </div>
          </div>
          <div class="metric-description">10% Tolerance</div>
        </div>

        <!-- Dataset Size Card -->
        <div class="metric-card dataset-card">
          <div class="metric-header">
            <div class="metric-icon">🗃️</div>
            <div class="metric-label">Dataset Size</div>
          </div>
          <div class="metric-value-container">
            <div class="metric-value large-number">${datasetSize}</div>
          </div>
          <div class="metric-description">Training Equations</div>
        </div>

        <!-- Model Size Card -->
        <div class="metric-card size-card">
          <div class="metric-header">
            <div class="metric-icon">💾</div>
            <div class="metric-label">Model Size</div>
          </div>
          <div class="metric-value-container">
            <div class="metric-value">${modelSize}</div>
          </div>
          <div class="metric-description">Storage Required</div>
        </div>
      </div>

      <!-- Technical Details Section -->
      <div class="technical-details-section">
        <div class="section-title">
          <div class="section-icon">⚙️</div>
          <span>Technical Details</span>
        </div>
        
        <div class="details-grid">
          <!-- Error Metrics -->
          <div class="detail-group error-metrics">
            <div class="detail-group-title">Error Metrics</div>
            <div class="detail-items">
              <div class="detail-item">
                <span class="detail-label">MSE</span>
                <span class="detail-value">${mse.toExponential(2)}</span>
              </div>
              <div class="detail-item">
                <span class="detail-label">MAE</span>
                <span class="detail-value">${mae.toExponential(2)}</span>
              </div>
            </div>
          </div>

          <!-- Model Info -->
          <div class="detail-group model-info">
            <div class="detail-group-title">Model Information</div>
            <div class="detail-items">
              <div class="detail-item">
                <span class="detail-label">Model ID</span>
                <span class="detail-value model-id">${model.model_id}</span>
              </div>
              <div class="detail-item">
                <span class="detail-label">Version</span>
                <span class="detail-value">${version}</span>
              </div>
              <div class="detail-item">
                <span class="detail-label">Training Time</span>
                <span class="detail-value">${trainingTime.toFixed(2)}s</span>
              </div>
            </div>
          </div>

          <!-- Creation Info -->
          <div class="detail-group creation-info">
            <div class="detail-group-title">Created</div>
            <div class="detail-items">
              <div class="detail-item">
                <span class="detail-label">📅 Date</span>
                <span class="detail-value">${createdDate}</span>
              </div>
              <div class="detail-item">
                <span class="detail-label">🕒 Time</span>
                <span class="detail-value">${createdTime}</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Performance Summary Footer -->
      <div class="performance-summary">
        <div class="summary-title">Performance Summary</div>
        <div class="summary-content">
          This model achieves <strong>${quality.label.toLowerCase()}</strong> performance with 
          <strong>${r2Score.toFixed(1)}%</strong> variance explanation and 
          <strong>${accuracy.toFixed(1)}%</strong> prediction accuracy.
        </div>
      </div>
    </div>
  `;
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
      this.createComparisonChart(analysisData);

      // Initialize chart controls after charts are created
      this.initChartControls();
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

    // Enhanced color palette for better visibility
    const enhancedColors = [
      "#007aff", // Bright blue
      "#34c759", // Bright green
      "#ff9500", // Bright orange
      "#ff3b30", // Bright red
      "#af52de", // Bright purple
      "#5ac8fa", // Bright cyan
    ];

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
          backgroundColor: enhancedColors[index % enhancedColors.length] + "30", // 30% opacity
          borderColor: enhancedColors[index % enhancedColors.length],
          pointBackgroundColor: enhancedColors[index % enhancedColors.length],
          pointBorderColor: "#ffffff",
          pointBorderWidth: 3,
          pointRadius: 6,
          pointHoverBackgroundColor: "#ffffff",
          pointHoverBorderColor: enhancedColors[index % enhancedColors.length],
          pointHoverRadius: 8,
          borderWidth: 3,
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
              color: "#d1d5db", // Darker grid lines
              lineWidth: 2,
            },
            angleLines: {
              color: "#d1d5db",
              lineWidth: 2,
            },
            pointLabels: {
              color: "#1f2937", // Dark text
              font: {
                size: 14,
                weight: "600",
              },
            },
            ticks: {
              color: "#6b7280",
              font: { size: 12 },
              stepSize: 0.2,
              showLabelBackdrop: true,
              backdropColor: "rgba(255, 255, 255, 0.8)",
              backdropPadding: 4,
            },
          },
        },
        plugins: {
          legend: {
            position: "bottom",
            labels: {
              color: "#1f2937",
              usePointStyle: true,
              font: {
                size: 14,
                weight: "600",
              },
              padding: 20,
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
              color: "var(--chart-border)",
            },
            ticks: {
              color: "var(--chart-text-secondary)",
            },
          },
          x: {
            grid: {
              color: "var(--chart-border)",
            },
            ticks: {
              color: "var(--chart-text-secondary)",
            },
          },
        },
        plugins: {
          legend: {
            position: "top",
            labels: {
              color: "var(--chart-text-primary)",
              usePointStyle: true,
              font: { size: 14, weight: "500" },
            },
          },
        },
      },
    });
  },
  createComparisonChart(data) {
    const ctx = document.getElementById("comparison-chart");
    if (!ctx) {
      console.error("Comparison chart canvas not found");
      return;
    }

    const chartCtx = ctx.getContext("2d");

    if (AppState.charts.comparison) {
      AppState.charts.comparison.destroy();
    }

    // Create a multi-metric comparison chart
    AppState.charts.comparison = new Chart(chartCtx, {
      type: "line",
      data: {
        labels: data.scenario_names,
        datasets: [
          {
            label: "R² Score",
            data: data.metrics.r2_scores,
            borderColor: "#007aff",
            backgroundColor: "rgba(0, 122, 255, 0.1)",
            tension: 0.4,
            fill: false,
            pointBackgroundColor: "#007aff",
            pointBorderColor: "#ffffff",
            pointBorderWidth: 3,
            pointRadius: 6,
            pointHoverRadius: 8,
          },
          {
            label: "Accuracy (%)",
            data: data.metrics.accuracy_values,
            borderColor: "#34c759",
            backgroundColor: "rgba(52, 199, 89, 0.1)",
            tension: 0.4,
            fill: false,
            pointBackgroundColor: "#34c759",
            pointBorderColor: "#ffffff",
            pointBorderWidth: 3,
            pointRadius: 6,
            pointHoverRadius: 8,
          },
          {
            label: "MSE (inv) × 100",
            data: data.metrics.mse_values.map(
              (mse) => (1 - mse / Math.max(...data.metrics.mse_values)) * 100
            ),
            borderColor: "#ff9500",
            backgroundColor: "rgba(255, 149, 0, 0.1)",
            tension: 0.4,
            fill: false,
            pointBackgroundColor: "#ff9500",
            pointBorderColor: "#ffffff",
            pointBorderWidth: 3,
            pointRadius: 6,
            pointHoverRadius: 8,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: "top",
            labels: {
              usePointStyle: true,
              font: { size: 14, weight: "600" },
              color: "#1f2937",
              padding: 20,
            },
          },
          tooltip: {
            backgroundColor: "rgba(0, 0, 0, 0.8)",
            titleColor: "#ffffff",
            bodyColor: "#ffffff",
            borderColor: "#007aff",
            borderWidth: 1,
            cornerRadius: 8,
            callbacks: {
              afterLabel: function (context) {
                if (context.datasetIndex === 0) return "Higher is better";
                if (context.datasetIndex === 1) return "Percentage accuracy";
                if (context.datasetIndex === 2) return "Inverted & scaled MSE";
                return "";
              },
            },
          },
        },
        scales: {
          y: {
            beginAtZero: true,
            max: 100,
            grid: {
              color: "#d1d5db",
              lineWidth: 1,
            },
            ticks: {
              color: "#6b7280",
              font: { size: 12, weight: "500" },
            },
          },
          x: {
            grid: {
              color: "#d1d5db",
              lineWidth: 1,
            },
            ticks: {
              color: "#6b7280",
              font: { size: 12, weight: "500" },
              maxRotation: 45,
            },
          },
        },
        interaction: {
          intersect: false,
          mode: "index",
        },
      },
    });
  },
  createCorrelationChart(data) {
    const ctx = document.getElementById("comparison-chart");
    if (!ctx) return;

    const chartCtx = ctx.getContext("2d");

    if (AppState.charts.comparison) {
      AppState.charts.comparison.destroy();
    }

    // Create scatter plot showing R² vs Accuracy correlation
    const scatterData = data.scenario_names.map((name, index) => ({
      x: data.metrics.r2_scores[index] * 100, // Convert to percentage
      y: data.metrics.accuracy_values[index],
      label: name,
    }));

    AppState.charts.comparison = new Chart(chartCtx, {
      type: "scatter",
      data: {
        datasets: [
          {
            label: "R² vs Accuracy Correlation",
            data: scatterData,
            backgroundColor: data.colors.map((color) => color + "80"),
            borderColor: data.colors,
            borderWidth: 3,
            pointRadius: 8,
            pointHoverRadius: 12,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            display: false,
          },
          tooltip: {
            backgroundColor: "rgba(0, 0, 0, 0.8)",
            callbacks: {
              title: function (context) {
                return context[0].raw.label;
              },
              label: function (context) {
                return [
                  `R² Score: ${(context.parsed.x / 100).toFixed(3)}`,
                  `Accuracy: ${context.parsed.y.toFixed(1)}%`,
                ];
              },
            },
          },
        },
        scales: {
          x: {
            title: {
              display: true,
              text: "R² Score (%)",
              color: "#1f2937",
              font: { size: 14, weight: "600" },
            },
            grid: { color: "#d1d5db" },
            ticks: { color: "#6b7280" },
          },
          y: {
            title: {
              display: true,
              text: "Accuracy (%)",
              color: "#1f2937",
              font: { size: 14, weight: "600" },
            },
            grid: { color: "#d1d5db" },
            ticks: { color: "#6b7280" },
          },
        },
      },
    });
  },
  initChartControls() {
    const chartControlBtns = document.querySelectorAll(".chart-control-btn");

    chartControlBtns.forEach((btn) => {
      btn.addEventListener("click", async (e) => {
        e.preventDefault();

        // Remove active class from all buttons in the same container
        const container = btn.closest(".chart-controls");
        container
          .querySelectorAll(".chart-control-btn")
          .forEach((b) => b.classList.remove("active"));

        // Add active class to clicked button
        btn.classList.add("active");

        // Get chart type and view
        const chartType = btn.dataset.chart;

        // Handle different chart views
        if (chartType === "metrics") {
          await this.showRadarView();
        } else if (chartType === "detailed") {
          await this.showDetailedView();
        } else if (chartType === "trends") {
          await this.showTrendsView();
        } else if (chartType === "correlation") {
          await this.showCorrelationView();
        }
      });
    });
  },

  async showTrendsView() {
    try {
      const analysisData = await ApiClient.request(API.performanceAnalysis);
      this.createComparisonChart(analysisData);
    } catch (error) {
      console.error("Failed to show trends view:", error);
    }
  },

  async showCorrelationView() {
    try {
      const analysisData = await ApiClient.request(API.performanceAnalysis);
      this.createCorrelationChart(analysisData);
    } catch (error) {
      console.error("Failed to show correlation view:", error);
    }
  },
  async showRadarView() {
    try {
      const analysisData = await ApiClient.request(API.performanceAnalysis);
      this.createMetricsChart(analysisData); // Recreate radar chart
    } catch (error) {
      console.error("Failed to show radar view:", error);
    }
  },

  async showDetailedView() {
    try {
      const analysisData = await ApiClient.request(API.performanceAnalysis);
      this.createDetailedMetricsChart(analysisData);
    } catch (error) {
      console.error("Failed to create detailed view:", error);
    }
  },

  createDetailedMetricsChart(data) {
    const ctx = document.getElementById("metrics-chart").getContext("2d");

    if (AppState.charts.metrics) {
      AppState.charts.metrics.destroy();
    }

    AppState.charts.metrics = new Chart(ctx, {
      type: "bar",
      data: {
        labels: data.scenario_names,
        datasets: [
          {
            label: "R² Score",
            data: data.metrics.r2_scores,
            backgroundColor: "#007aff",
            borderColor: "#005bb5",
            borderWidth: 2,
          },
          {
            label: "Accuracy (%)",
            data: data.metrics.accuracy_values,
            backgroundColor: "#34c759",
            borderColor: "#248a3d",
            borderWidth: 2,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: "top",
            labels: {
              usePointStyle: true,
              font: { size: 14, weight: "600" },
              color: "#1f2937", // Dark text instead of variable
            },
          },
        },
        scales: {
          y: {
            beginAtZero: true,
            grid: {
              color: "#d1d5db",
              lineWidth: 1,
            },
            ticks: {
              color: "#6b7280",
              font: { size: 12, weight: "500" },
            },
          },
          x: {
            grid: {
              color: "#d1d5db",
              lineWidth: 1,
            },
            ticks: {
              color: "#6b7280",
              font: { size: 12, weight: "500" },
            },
          },
        },
      },
    });
  },
};

// Enhanced Comparison section management
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
      if (Object.keys(results).length < 2) {
        Utils.showNotification(
          "Need at least 2 trained models for comparison",
          "warning"
        );
        return;
      }

      this.createComparisonChart(results);
      this.generateComparisonReport(results);
      this.generateInsights(results);
    } catch (error) {
      console.error("Failed to generate comparison:", error);
      Utils.showNotification("Failed to generate comparison", "error");
    }
  },

  createComparisonChart(results) {
    const ctx = document
      .getElementById("comparison-performance-chart")
      .getContext("2d");

    if (AppState.charts.comparison) {
      AppState.charts.comparison.destroy();
    }

    const scenarios = Object.keys(results);

    // Prepare data for grouped bar chart with multiple metrics
    const metrics = ["r2", "accuracy_10pct", "mae_inv", "mse_inv"];
    const metricLabels = [
      "R² Score",
      "Accuracy (10%)",
      "MAE (Inverted)",
      "MSE (Inverted)",
    ];
    const colors = ["#007aff", "#34c759", "#ff9500", "#ff3b30"];

    const datasets = metrics.map((metric, index) => ({
      label: metricLabels[index],
      data: scenarios.map((scenario) => {
        const result = results[scenario];
        // Process each metric appropriately
        if (metric === "mae_inv") {
          return result.metrics.mae ? (1 / (1 + result.metrics.mae)) * 100 : 0;
        }
        if (metric === "mse_inv") {
          return result.metrics.mse ? (1 / (1 + result.metrics.mse)) * 100 : 0;
        }
        if (metric === "accuracy_10pct") {
          return result.metrics.accuracy_10pct || 0;
        }
        if (metric === "r2") {
          return (result.metrics.r2 || 0) * 100; // Convert to percentage
        }
        return 0;
      }),
      backgroundColor: colors[index] + "90", // Add transparency
      borderColor: colors[index],
      borderWidth: 2,
      borderRadius: 8,
      borderSkipped: false,
    }));

    const labels = scenarios.map(
      (scenario) => results[scenario].scenario_info.name
    );

    AppState.charts.comparison = new Chart(ctx, {
      type: "bar",
      data: {
        labels: labels,
        datasets: datasets,
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          title: {
            display: true,
            text: "🏆 Neural Network Model Performance Comparison",
            font: {
              size: 18,
              weight: "bold",
            },
            color: "var(--text-primary)",
            padding: 20,
          },
          legend: {
            display: true,
            position: "top",
            labels: {
              usePointStyle: true,
              pointStyle: "circle",
              padding: 20,
              color: "var(--text-primary)",
              font: {
                size: 12,
              },
            },
          },
          tooltip: {
            mode: "index",
            intersect: false,
            backgroundColor: "rgba(255, 255, 255, 0.95)",
            titleColor: "var(--text-primary)",
            bodyColor: "var(--text-secondary)",
            borderColor: "var(--border-color)",
            borderWidth: 1,
            cornerRadius: 12,
            displayColors: true,
            callbacks: {
              label: function (context) {
                const label = context.dataset.label;
                const value = context.parsed.y;

                if (label === "R² Score") {
                  return `${label}: ${value.toFixed(1)}%`;
                } else if (label === "Accuracy (10%)") {
                  return `${label}: ${value.toFixed(1)}%`;
                } else {
                  return `${label}: ${value.toFixed(1)}`;
                }
              },
            },
          },
        },
        scales: {
          x: {
            display: true,
            title: {
              display: true,
              text: "Neural Network Scenarios",
              font: {
                size: 14,
                weight: "bold",
              },
              color: "var(--text-primary)",
            },
            grid: {
              display: false,
            },
            ticks: {
              color: "var(--text-secondary)",
              font: {
                size: 11,
              },
            },
          },
          y: {
            display: true,
            beginAtZero: true,
            max: 100,
            title: {
              display: true,
              text: "Performance Score (%)",
              font: {
                size: 14,
                weight: "bold",
              },
              color: "var(--text-primary)",
            },
            grid: {
              color: "var(--border-color)",
              lineWidth: 1,
            },
            ticks: {
              color: "var(--text-secondary)",
              font: {
                size: 11,
              },
              callback: function (value) {
                return value + "%";
              },
            },
          },
        },
        interaction: {
          mode: "index",
          intersect: false,
        },
        animation: {
          duration: 1500,
          easing: "easeInOutCubic",
        },
      },
    });
  },

  generateComparisonReport(results) {
    const container = document.getElementById("model-rankings");

    // Sort scenarios by composite score (weighted average of all metrics)
    const sortedScenarios = Object.entries(results).sort(([, a], [, b]) => {
      const scoreA = this.calculateCompositeScore(a.metrics);
      const scoreB = this.calculateCompositeScore(b.metrics);
      return scoreB - scoreA;
    });

    let html = `
      <div class="comparison-header">
        <h3>🏅 Model Performance Rankings</h3>
        <p class="section-description">
          Models ranked by composite performance score across all evaluation metrics
        </p>
      </div>
      <div class="model-rankings-grid">
    `;

    sortedScenarios.forEach(([key, result], index) => {
      const medals = ["🥇", "🥈", "🥉", "🏅"];
      const medal = medals[index] || "📊";
      const compositeScore = this.calculateCompositeScore(result.metrics);
      const performance = this.getPerformanceRating(compositeScore);
      const performanceColor = this.getPerformanceColor(compositeScore);

      html += `
        <div class="model-card" style="border-left-color: ${
          result.scenario_info.color
        }">
          <div class="model-header">
            <div class="model-rank">
              <span class="medal">${medal}</span>
              <span class="rank-number">#${index + 1}</span>
            </div>
            <div class="model-info">
              <h4 class="model-name">${result.scenario_info.name}</h4>
              <p class="model-description">${
                result.scenario_info.description
              }</p>
            </div>
            <div class="model-score">
              <div class="composite-score" style="color: ${performanceColor}">
                ${compositeScore.toFixed(1)}%
              </div>
              <div class="performance-label">${performance}</div>
            </div>
          </div>
          
          <div class="metrics-grid">
            <div class="metric-item">
              <span class="metric-label">R² Score</span>
              <span class="metric-value">${Utils.formatNumber(
                result.metrics.r2 * 100,
                1
              )}%</span>
              <div class="metric-bar">
                <div class="metric-fill" style="width: ${
                  result.metrics.r2 * 100
                }%; background: #007aff"></div>
              </div>
            </div>
            
            <div class="metric-item">
              <span class="metric-label">Accuracy</span>
              <span class="metric-value">${Utils.formatNumber(
                result.metrics.accuracy_10pct,
                1
              )}%</span>
              <div class="metric-bar">
                <div class="metric-fill" style="width: ${
                  result.metrics.accuracy_10pct
                }%; background: #34c759"></div>
              </div>
            </div>
            
            <div class="metric-item">
              <span class="metric-label">MSE</span>
              <span class="metric-value">${Utils.formatNumber(
                result.metrics.mse,
                6
              )}</span>
              <div class="metric-bar">
                <div class="metric-fill" style="width: ${
                  (1 / (1 + result.metrics.mse)) * 100
                }%; background: #ff3b30"></div>
              </div>
            </div>
            
            <div class="metric-item">
              <span class="metric-label">MAE</span>
              <span class="metric-value">${Utils.formatNumber(
                result.metrics.mae,
                6
              )}</span>
              <div class="metric-bar">
                <div class="metric-fill" style="width: ${
                  (1 / (1 + result.metrics.mae)) * 100
                }%; background: #ff9500"></div>
              </div>
            </div>
          </div>
          
          <div class="model-actions">
            <div class="model-indicator" style="background: ${
              result.scenario_info.color
            }"></div>
          </div>
        </div>
      `;
    });

    html += `</div>`;
    container.innerHTML = html;
  },

  generateInsights(results) {
    const scenarios = Object.keys(results);

    // Find best performing model
    let bestModel = "";
    let bestScore = -1;

    scenarios.forEach((scenario) => {
      const score = this.calculateCompositeScore(results[scenario].metrics);
      if (score > bestScore) {
        bestScore = score;
        bestModel = results[scenario].scenario_info.name;
      }
    });

    // Calculate averages
    const avgR2 =
      (scenarios.reduce((sum, s) => sum + results[s].metrics.r2, 0) /
        scenarios.length) *
      100;
    const avgAccuracy =
      scenarios.reduce((sum, s) => sum + results[s].metrics.accuracy_10pct, 0) /
      scenarios.length;

    // Count production-ready models (accuracy > 85%)
    const productionReady = scenarios.filter(
      (s) => results[s].metrics.accuracy_10pct > 85
    ).length;

    const insightsHtml = `
      <div class="insights-section">
        <h3>🧠 Performance Insights</h3>
        
        <div class="insight-cards">
          <div class="insight-card best-performer">
            <div class="insight-icon">🏆</div>
            <div class="insight-content">
              <h4>Top Performer</h4>
              <p><strong>${bestModel}</strong> achieves the highest composite score of <strong>${bestScore.toFixed(
      1
    )}%</strong></p>
              <div class="insight-recommendation">Recommended for production deployment</div>
            </div>
          </div>
          
          <div class="insight-card performance-overview">
            <div class="insight-icon">📊</div>
            <div class="insight-content">
              <h4>Overall Performance</h4>
              <p>Average R² Score: <strong>${avgR2.toFixed(1)}%</strong></p>
              <p>Average Accuracy: <strong>${avgAccuracy.toFixed(
                1
              )}%</strong></p>
              <div class="performance-summary">
                ${productionReady}/${
      scenarios.length
    } models are production-ready (85%+ accuracy)
              </div>
            </div>
          </div>
          
          <div class="insight-card recommendations">
            <div class="insight-icon">💡</div>
            <div class="insight-content">
              <h4>Key Recommendations</h4>
              <ul class="recommendation-list">
                <li>Models with R² > 80% show excellent variance explanation</li>
                <li>Accuracy above 85% indicates production readiness</li>
                <li>Lower error metrics (MSE, MAE) suggest better precision</li>
                <li>Consider ensemble methods for critical applications</li>
              </ul>
            </div>
          </div>
        </div>
        
        <div class="metric-explanations">
          <h4>📖 Understanding the Metrics</h4>
          <div class="explanation-grid">
            <div class="explanation-card">
              <div class="metric-badge" style="background: #007aff">R²</div>
              <div class="explanation-content">
                <h5>R² Score (Coefficient of Determination)</h5>
                <p>Measures how well the model explains variance in target data</p>
                <div class="scale-indicators">
                  <span class="scale excellent">90%+ Excellent</span>
                  <span class="scale good">80-90% Very Good</span>
                  <span class="scale fair">60-80% Acceptable</span>
                  <span class="scale poor">&lt;60% Poor</span>
                </div>
              </div>
            </div>
            
            <div class="explanation-card">
              <div class="metric-badge" style="background: #34c759">ACC</div>
              <div class="explanation-content">
                <h5>Accuracy (10% Tolerance)</h5>
                <p>Percentage of predictions within 10% of actual values</p>
                <div class="scale-indicators">
                  <span class="scale excellent">85%+ Production</span>
                  <span class="scale good">70-85% Good</span>
                  <span class="scale fair">50-70% Fair</span>
                  <span class="scale poor">&lt;50% Poor</span>
                </div>
              </div>
            </div>
            
            <div class="explanation-card">
              <div class="metric-badge" style="background: #ff9500">MAE</div>
              <div class="explanation-content">
                <h5>Mean Absolute Error</h5>
                <p>Average absolute difference between predictions and actual values</p>
                <div class="error-note">Lower values indicate better performance</div>
              </div>
            </div>
            
            <div class="explanation-card">
              <div class="metric-badge" style="background: #ff3b30">MSE</div>
              <div class="explanation-content">
                <h5>Mean Squared Error</h5>
                <p>Average squared difference - more sensitive to outliers</p>
                <div class="error-note">Penalizes larger errors more heavily than MAE</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    `;

    const insightsContainer =
      document.querySelector(".comparison-insights") ||
      document.getElementById("comparison-insights");
    if (insightsContainer) {
      insightsContainer.innerHTML = insightsHtml;
    }
  },

  calculateCompositeScore(metrics) {
    // Weighted composite score calculation
    const r2Weight = 0.4;
    const accuracyWeight = 0.3;
    const mseWeight = 0.15;
    const maeWeight = 0.15;

    const r2Score = (metrics.r2 || 0) * 100;
    const accuracyScore = metrics.accuracy_10pct || 0;
    const mseScore = (1 / (1 + metrics.mse)) * 100; // Inverted
    const maeScore = (1 / (1 + metrics.mae)) * 100; // Inverted

    return (
      r2Score * r2Weight +
      accuracyScore * accuracyWeight +
      mseScore * mseWeight +
      maeScore * maeWeight
    );
  },

  getPerformanceRating(score) {
    if (score >= 85) return "Excellent";
    if (score >= 70) return "Very Good";
    if (score >= 55) return "Good";
    if (score >= 40) return "Fair";
    return "Needs Improvement";
  },

  getPerformanceColor(score) {
    if (score >= 85) return "#34c759";
    if (score >= 70) return "#007aff";
    if (score >= 55) return "#ff9500";
    if (score >= 40) return "#ff9500";
    return "#ff3b30";
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
  const avgError = error_metrics["Average Error"] ?? 0;
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

// Application initialization
document.addEventListener("DOMContentLoaded", async () => {
  console.log("🚀 Quadratic Neural Network Web Application");
  console.log("Initializing application...");

  // Initialize all sections
  Navigation.init();
  DataSection.init();
  TrainingSection.init();
  ModelSection.init();
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

        // Fetch results and update save section
        try {
          const results = await ApiClient.request(API.results);
          AppState.results = results;
          console.log("✅ Results loaded:", AppState.results);

          // Update save section to show trained models
          ModelSection.updateSaveSection();
        } catch (error) {
          console.error("Failed to load results after training:", error);
        }
      }
    } catch (error) {
      console.error("Failed to check training status:", error);
    }
  }
}, 2000);
