/**
 * Quadratic Neural Network Web Application
 * Training Interface Management
 *
 * Author: Matt
 * Location: Varna, Bulgaria
 * Date: July 2025
 *
 * Enhanced training interface with real-time progress tracking and control
 */

// Training interface management
const TrainingManager = {
  // Training state
  isTraining: false,
  trainingProgress: 0,
  currentScenario: null,
  trainingLogs: [],
  selectedScenarios: [],

  // UI elements
  startButton: null,
  stopButton: null,
  progressBar: null,
  progressText: null,
  logsContainer: null,
  scenarioCheckboxes: {},
  epochsInput: null,

  // Training configuration
  defaultConfig: {
    epochs: 1000,
    learningRate: 0.001,
    batchSize: 32,
    validationSplit: 0.15,
  },

  // Initialize training manager
  init() {
    this.initializeElements();
    this.setupEventListeners();
    this.loadScenarios();
    this.startProgressMonitoring();

    console.log("🧠 Training interface initialized");
  },

  // Initialize UI elements
  initializeElements() {
    this.startButton = document.getElementById("start-training-btn");
    this.stopButton = document.getElementById("stop-training-btn");
    this.progressBar = document.getElementById("training-progress-fill");
    this.progressText = document.getElementById("training-progress-text");
    this.logsContainer = document.getElementById("training-logs");
    this.epochsInput = document.getElementById("epochs-input");

    // Create stop button if it doesn't exist
    if (!this.stopButton && this.startButton) {
      this.stopButton = document.createElement("button");
      this.stopButton.id = "stop-training-btn";
      this.stopButton.className = "btn btn-error";
      this.stopButton.innerHTML = '<i class="fas fa-stop"></i> Stop Training';
      this.stopButton.style.display = "none";
      this.startButton.parentNode.insertBefore(
        this.stopButton,
        this.startButton.nextSibling
      );
    }
  },

  // Setup event listeners
  setupEventListeners() {
    if (this.startButton) {
      this.startButton.addEventListener("click", () => this.startTraining());
    }

    if (this.stopButton) {
      this.stopButton.addEventListener("click", () => this.stopTraining());
    }

    // Scenario selection changes
    document.addEventListener("change", (e) => {
      if (
        e.target.type === "checkbox" &&
        e.target.closest(".scenario-selector")
      ) {
        this.updateSelectedScenarios();
      }
    });

    // Epochs input validation
    if (this.epochsInput) {
      this.epochsInput.addEventListener("input", (e) => {
        this.validateEpochs(e.target.value);
      });
    }

    // Keyboard shortcuts
    document.addEventListener("keydown", (e) => {
      if (e.ctrlKey && e.shiftKey && e.key === "T") {
        e.preventDefault();
        this.toggleTraining();
      }
    });
  },

  // Load available scenarios
  async loadScenarios() {
    try {
      const response = await fetch("/api/scenarios");
      const scenarios = await response.json();

      AppState.scenarios = scenarios;
      this.createScenarioSelector(scenarios);
      this.updateSelectedScenarios();
    } catch (error) {
      console.error("Failed to load scenarios:", error);
      Utils.showNotification("Failed to load training scenarios", "error");
    }
  },

  // Create scenario selector interface
  createScenarioSelector(scenarios) {
    const container = document.getElementById("scenarios-selection");
    if (!container) return;

    container.innerHTML = "";

    Object.entries(scenarios).forEach(([key, scenario]) => {
      const scenarioCard = this.createScenarioCard(key, scenario);
      container.appendChild(scenarioCard);
    });
  },

  // Create individual scenario card
  createScenarioCard(key, scenario) {
    const card = document.createElement("div");
    card.className = "scenario-selector";
    card.innerHTML = `
            <div class="scenario-card-content">
                <input type="checkbox" 
                       id="scenario-${key}" 
                       value="${key}" 
                       checked 
                       class="scenario-checkbox">
                <label for="scenario-${key}" class="scenario-label">
                    <div class="scenario-header">
                        <span class="scenario-name">${scenario.name}</span>
                        <span class="scenario-color-indicator" style="background-color: ${
                          scenario.color
                        }"></span>
                    </div>
                    <div class="scenario-description">${
                      scenario.description
                    }</div>
                    <div class="scenario-details">
                        <div class="scenario-io">
                            <strong>Input:</strong> ${scenario.input_features.join(
                              ", "
                            )}
                        </div>
                        <div class="scenario-io">
                            <strong>Output:</strong> ${scenario.target_features.join(
                              ", "
                            )}
                        </div>
                        <div class="scenario-architecture">
                            Architecture: ${scenario.network_architecture.join(
                              " → "
                            )}
                        </div>
                    </div>
                </label>
            </div>
        `;

    // Add interactive effects
    const checkbox = card.querySelector(".scenario-checkbox");
    const label = card.querySelector(".scenario-label");

    checkbox.addEventListener("change", (e) => {
      if (e.target.checked) {
        card.classList.add("selected");
        label.style.borderColor = scenario.color;
      } else {
        card.classList.remove("selected");
        label.style.borderColor = "var(--border-color)";
      }
    });

    // Hover effects
    label.addEventListener("mouseenter", () => {
      label.style.borderColor = scenario.color;
      label.style.boxShadow = `0 4px 12px ${scenario.color}30`;
    });

    label.addEventListener("mouseleave", () => {
      if (!checkbox.checked) {
        label.style.borderColor = "var(--border-color)";
        label.style.boxShadow = "none";
      }
    });

    this.scenarioCheckboxes[key] = checkbox;
    return card;
  },

  // Update selected scenarios
  updateSelectedScenarios() {
    this.selectedScenarios = [];

    Object.entries(this.scenarioCheckboxes).forEach(([key, checkbox]) => {
      if (checkbox.checked) {
        this.selectedScenarios.push(key);
      }
    });

    this.updateTrainingEstimate();
  },

  // Update training time estimate
  updateTrainingEstimate() {
    const epochs = parseInt(
      this.epochsInput?.value || this.defaultConfig.epochs
    );
    const scenarioCount = this.selectedScenarios.length;

    if (scenarioCount === 0) {
      this.updateEstimateDisplay("No scenarios selected");
      return;
    }

    // Rough estimation: 1-3 seconds per 100 epochs per scenario
    const estimatedTimePerScenario = (epochs / 100) * 2; // 2 seconds per 100 epochs
    const totalEstimatedTime = estimatedTimePerScenario * scenarioCount;

    this.updateEstimateDisplay(this.formatDuration(totalEstimatedTime));
  },

  // Update estimate display
  updateEstimateDisplay(estimate) {
    const estimateElement = document.getElementById("training-estimate");
    if (estimateElement) {
      estimateElement.textContent = `Estimated time: ${estimate}`;
    }
  },

  // Format duration in human-readable format
  formatDuration(seconds) {
    if (seconds < 60) {
      return `${seconds.toFixed(0)}s`;
    } else if (seconds < 3600) {
      const minutes = Math.floor(seconds / 60);
      const remainingSeconds = seconds % 60;
      return `${minutes}m ${remainingSeconds.toFixed(0)}s`;
    } else {
      const hours = Math.floor(seconds / 3600);
      const minutes = Math.floor((seconds % 3600) / 60);
      return `${hours}h ${minutes}m`;
    }
  },

  // Validate epochs input
  validateEpochs(value) {
    const epochs = parseInt(value);
    const input = this.epochsInput;

    if (isNaN(epochs) || epochs < 1) {
      input.classList.add("error");
      this.showInputError(input, "Epochs must be a positive number");
      return false;
    } else if (epochs > 10000) {
      input.classList.add("warning");
      this.showInputError(input, "High epoch count may take very long");
      return true;
    } else {
      input.classList.remove("error", "warning");
      this.hideInputError(input);
      this.updateTrainingEstimate();
      return true;
    }
  },

  // Show input error
  showInputError(input, message) {
    let errorDiv = input.parentNode.querySelector(".input-error");
    if (!errorDiv) {
      errorDiv = document.createElement("div");
      errorDiv.className = "input-error";
      input.parentNode.appendChild(errorDiv);
    }
    errorDiv.textContent = message;
    errorDiv.style.display = "block";
  },

  // Hide input error
  hideInputError(input) {
    const errorDiv = input.parentNode.querySelector(".input-error");
    if (errorDiv) {
      errorDiv.style.display = "none";
    }
  },

  // Start training
  async startTraining() {
    if (!AppState.dataLoaded) {
      Utils.showNotification("Please load a dataset first", "warning");
      return;
    }

    if (this.selectedScenarios.length === 0) {
      Utils.showNotification("Please select at least one scenario", "warning");
      return;
    }

    if (
      !this.validateEpochs(this.epochsInput?.value || this.defaultConfig.epochs)
    ) {
      return;
    }

    const epochs = parseInt(
      this.epochsInput?.value || this.defaultConfig.epochs
    );

    try {
      this.setTrainingState(true);
      this.clearLogs();
      this.addLog("🚀 Initiating training session...", "info");
      this.addLog(
        `📊 Selected scenarios: ${this.selectedScenarios.length}`,
        "info"
      );
      this.addLog(`⚙️ Epochs: ${epochs}`, "info");

      const response = await fetch("/api/training/start", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          scenarios: this.selectedScenarios,
          epochs: epochs,
        }),
      });

      const result = await response.json();

      if (result.success) {
        this.addLog("✅ Training started successfully", "success");
        Utils.showNotification("Training started successfully", "success");
      } else {
        this.setTrainingState(false);
        this.addLog(`❌ Failed to start training: ${result.error}`, "error");
        Utils.showNotification(
          result.error || "Failed to start training",
          "error"
        );
      }
    } catch (error) {
      this.setTrainingState(false);
      this.addLog(`❌ Training request failed: ${error.message}`, "error");
      Utils.showNotification(
        "Failed to start training: " + error.message,
        "error"
      );
    }
  },

  // Stop training
  async stopTraining() {
    try {
      const response = await fetch("/api/training/stop", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
      });

      const result = await response.json();

      if (result.success) {
        this.addLog("⏹️ Training stop requested", "warning");
        Utils.showNotification("Training stop requested", "warning");
      } else {
        Utils.showNotification("Failed to stop training", "error");
      }
    } catch (error) {
      Utils.showNotification(
        "Failed to stop training: " + error.message,
        "error"
      );
    }
  },

  // Toggle training state
  toggleTraining() {
    if (this.isTraining) {
      this.stopTraining();
    } else {
      this.startTraining();
    }
  },

  // Set training state
  setTrainingState(isTraining) {
    this.isTraining = isTraining;

    if (this.startButton) {
      this.startButton.style.display = isTraining ? "none" : "inline-flex";
    }

    if (this.stopButton) {
      this.stopButton.style.display = isTraining ? "inline-flex" : "none";
    }

    // Disable scenario selection during training
    Object.values(this.scenarioCheckboxes).forEach((checkbox) => {
      checkbox.disabled = isTraining;
    });

    if (this.epochsInput) {
      this.epochsInput.disabled = isTraining;
    }
  },

  // Start progress monitoring
  startProgressMonitoring() {
    setInterval(async () => {
      if (this.isTraining) {
        await this.updateTrainingProgress();
      }
    }, 1000);
  },

  // Update training progress
  async updateTrainingProgress() {
    try {
      const response = await fetch("/api/training/status");
      const status = await response.json();

      this.updateProgressDisplay(status);
      this.updateTrainingLogs(status);

      // Check if training finished
      if (!status.is_training && this.isTraining) {
        this.handleTrainingComplete();
      }
    } catch (error) {
      console.error("Failed to get training status:", error);
    }
  },

  // Update progress display
  updateProgressDisplay(status) {
    if (this.progressBar) {
      this.progressBar.style.width = `${status.progress || 0}%`;
    }

    if (this.progressText) {
      if (status.is_training) {
        const progressText = status.current_scenario
          ? `Training: ${status.current_scenario} (${(
              status.progress || 0
            ).toFixed(1)}%)`
          : `Training in progress... (${(status.progress || 0).toFixed(1)}%)`;
        this.progressText.textContent = progressText;
      } else {
        this.progressText.textContent =
          status.progress === 100 ? "Training completed!" : "Ready to train";
      }
    }
  },

  // Update training logs
  updateTrainingLogs(status) {
    if (status.logs && status.logs.length > 0) {
      // Only add new logs
      const newLogs = status.logs.slice(this.trainingLogs.length);
      newLogs.forEach((log) => {
        this.addLog(log.message, "info", log.timestamp);
      });
      this.trainingLogs = status.logs;
    }
  },

  // Handle training completion
  handleTrainingComplete() {
    this.setTrainingState(false);
    this.addLog("🎉 Training session completed!", "success");
    Utils.showNotification("Training completed successfully!", "success");

    // Refresh other sections
    if (typeof PredictionManager !== "undefined") {
      PredictionManager.refresh();
    }

    // Create completion animation
    this.createCompletionAnimation();
  },

  // Create completion animation
  createCompletionAnimation() {
    if (this.progressBar) {
      this.progressBar.style.background =
        "linear-gradient(90deg, var(--success-color), var(--primary-color))";
      this.progressBar.style.animation = "pulse 2s ease-in-out 3";
    }

    // Reset after animation
    setTimeout(() => {
      if (this.progressBar) {
        this.progressBar.style.background =
          "linear-gradient(90deg, var(--primary-color), var(--secondary-color))";
        this.progressBar.style.animation = "none";
      }
    }, 6000);
  },

  // Add log entry
  addLog(message, type = "info", timestamp = null) {
    if (!this.logsContainer) return;

    const logEntry = document.createElement("div");
    logEntry.className = `log-entry log-${type}`;

    const time = timestamp || new Date().toLocaleTimeString();
    logEntry.innerHTML = `
            <span class="log-timestamp">[${time}]</span>
            <span class="log-message">${message}</span>
        `;

    this.logsContainer.appendChild(logEntry);
    this.logsContainer.scrollTop = this.logsContainer.scrollHeight;

    // Add animation
    logEntry.style.opacity = "0";
    logEntry.style.transform = "translateY(10px)";
    setTimeout(() => {
      logEntry.style.opacity = "1";
      logEntry.style.transform = "translateY(0)";
      logEntry.style.transition = "all 0.3s ease";
    }, 10);

    // Keep only last 100 entries
    const logs = this.logsContainer.querySelectorAll(".log-entry");
    if (logs.length > 100) {
      logs[0].remove();
    }
  },

  // Clear logs
  clearLogs() {
    if (this.logsContainer) {
      this.logsContainer.innerHTML = "";
    }
    this.trainingLogs = [];
  },

  // Export training logs
  exportLogs() {
    const logs = this.trainingLogs
      .map((log) => `[${log.timestamp}] ${log.message}`)
      .join("\n");
    const blob = new Blob([logs], { type: "text/plain" });
    const url = URL.createObjectURL(blob);

    const link = document.createElement("a");
    link.href = url;
    link.download = `training_logs_${
      new Date().toISOString().split("T")[0]
    }.txt`;
    link.click();

    URL.revokeObjectURL(url);
    Utils.showNotification("Training logs exported successfully", "success");
  },

  // Save training configuration
  saveConfig() {
    const config = {
      selectedScenarios: this.selectedScenarios,
      epochs: parseInt(this.epochsInput?.value || this.defaultConfig.epochs),
      learningRate: this.defaultConfig.learningRate,
      timestamp: new Date().toISOString(),
    };

    localStorage.setItem("training_config", JSON.stringify(config));
    Utils.showNotification("Configuration saved", "success");
  },

  // Load training configuration
  loadConfig() {
    try {
      const config = JSON.parse(
        localStorage.getItem("training_config") || "{}"
      );

      if (config.selectedScenarios) {
        // Update scenario selections
        Object.entries(this.scenarioCheckboxes).forEach(([key, checkbox]) => {
          checkbox.checked = config.selectedScenarios.includes(key);
        });
        this.updateSelectedScenarios();
      }

      if (config.epochs && this.epochsInput) {
        this.epochsInput.value = config.epochs;
      }

      Utils.showNotification("Configuration loaded", "success");
    } catch (error) {
      Utils.showNotification("Failed to load configuration", "error");
    }
  },

  // Get training summary
  getTrainingSummary() {
    if (!AppState.results || Object.keys(AppState.results).length === 0) {
      return null;
    }

    const results = AppState.results;
    const scenarios = Object.keys(results);

    const summary = {
      totalScenarios: scenarios.length,
      bestR2: Math.max(...scenarios.map((s) => results[s].r2)),
      averageR2:
        scenarios.reduce((sum, s) => sum + results[s].r2, 0) / scenarios.length,
      bestScenario: scenarios.reduce((best, current) =>
        results[current].r2 > results[best].r2 ? current : best
      ),
      worstScenario: scenarios.reduce((worst, current) =>
        results[current].r2 < results[worst].r2 ? current : worst
      ),
    };

    return summary;
  },

  // Display training summary
  displayTrainingSummary() {
    const summary = this.getTrainingSummary();
    if (!summary) {
      Utils.showNotification("No training results available", "warning");
      return;
    }

    const summaryHtml = `
            <div class="training-summary">
                <h3>🎯 Training Summary</h3>
                <div class="summary-stats">
                    <div class="stat-item">
                        <span class="stat-label">Scenarios Trained:</span>
                        <span class="stat-value">${
                          summary.totalScenarios
                        }</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Best R² Score:</span>
                        <span class="stat-value">${summary.bestR2.toFixed(
                          4
                        )}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Average R² Score:</span>
                        <span class="stat-value">${summary.averageR2.toFixed(
                          4
                        )}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Best Scenario:</span>
                        <span class="stat-value">${
                          AppState.scenarios[summary.bestScenario]?.name ||
                          summary.bestScenario
                        }</span>
                    </div>
                </div>
            </div>
        `;

    // Display in modal or dedicated area
    this.showSummaryModal(summaryHtml);
  },

  // Show summary modal
  showSummaryModal(content) {
    const modal = document.createElement("div");
    modal.className = "training-summary-modal";
    modal.innerHTML = `
            <div class="modal-backdrop"></div>
            <div class="modal-content">
                ${content}
                <button class="btn btn-primary" onclick="this.closest('.training-summary-modal').remove()">
                    Close
                </button>
            </div>
        `;

    document.body.appendChild(modal);

    // Auto-remove after 10 seconds
    setTimeout(() => {
      if (modal.parentNode) {
        modal.remove();
      }
    }, 10000);
  },

  // Reset training interface
  reset() {
    this.setTrainingState(false);
    this.clearLogs();

    if (this.progressBar) {
      this.progressBar.style.width = "0%";
    }

    if (this.progressText) {
      this.progressText.textContent = "Ready to train";
    }

    // Reset scenario selections
    Object.values(this.scenarioCheckboxes).forEach((checkbox) => {
      checkbox.checked = true;
    });

    this.updateSelectedScenarios();

    Utils.showNotification("Training interface reset", "info");
  },
};

// Training configuration presets
const TrainingPresets = {
  presets: {
    quick: {
      name: "Quick Training",
      description: "Fast training for testing",
      epochs: 100,
      scenarios: ["coeff_to_roots", "single_missing"],
    },
    standard: {
      name: "Standard Training",
      description: "Balanced training for most use cases",
      epochs: 1000,
      scenarios: [
        "coeff_to_roots",
        "partial_coeff_to_missing",
        "roots_to_coeff",
      ],
    },
    comprehensive: {
      name: "Comprehensive Training",
      description: "Complete training of all scenarios",
      epochs: 2000,
      scenarios: [
        "coeff_to_roots",
        "partial_coeff_to_missing",
        "roots_to_coeff",
        "single_missing",
        "verify_equation",
      ],
    },
    production: {
      name: "Production Training",
      description: "High-quality training for production use",
      epochs: 5000,
      scenarios: [
        "coeff_to_roots",
        "partial_coeff_to_missing",
        "roots_to_coeff",
      ],
    },
  },

  // Apply preset
  applyPreset(presetName) {
    const preset = this.presets[presetName];
    if (!preset) {
      Utils.showNotification("Invalid preset selected", "error");
      return;
    }

    // Set epochs
    if (TrainingManager.epochsInput) {
      TrainingManager.epochsInput.value = preset.epochs;
    }

    // Update scenario selections
    Object.entries(TrainingManager.scenarioCheckboxes).forEach(
      ([key, checkbox]) => {
        checkbox.checked = preset.scenarios.includes(key);
      }
    );

    TrainingManager.updateSelectedScenarios();

    Utils.showNotification(`Applied preset: ${preset.name}`, "success");
  },

  // Get preset options HTML
  getPresetOptionsHtml() {
    return Object.entries(this.presets)
      .map(
        ([key, preset]) => `
            <div class="preset-option" onclick="TrainingPresets.applyPreset('${key}')">
                <h4>${preset.name}</h4>
                <p>${preset.description}</p>
                <div class="preset-details">
                    <span>Epochs: ${preset.epochs}</span>
                    <span>Scenarios: ${preset.scenarios.length}</span>
                </div>
            </div>
        `
      )
      .join("");
  },
};

// Initialize training manager when DOM is loaded
document.addEventListener("DOMContentLoaded", () => {
  TrainingManager.init();

  // Make available globally
  window.TrainingManager = TrainingManager;
  window.TrainingPresets = TrainingPresets;
});

// Global functions for HTML onclick events
window.startTraining = () => TrainingManager.startTraining();
window.stopTraining = () => TrainingManager.stopTraining();
window.exportTrainingLogs = () => TrainingManager.exportLogs();
window.saveTrainingConfig = () => TrainingManager.saveConfig();
window.loadTrainingConfig = () => TrainingManager.loadConfig();
window.showTrainingSummary = () => TrainingManager.displayTrainingSummary();
window.resetTraining = () => TrainingManager.reset();
