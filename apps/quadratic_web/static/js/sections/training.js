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
