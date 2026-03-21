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

    document.querySelector(".status-indicator");
  },

  async loadSavedModelsList() {
    try {
      const response = await fetch(API.modelsList);

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();

      if (data.error) {
        console.error("API error loading models:", data.error);
        Utils.showNotification(`Failed to load models: ${data.error}`, "error");
        AppState.savedModels = [];
        this.updateModelsGrid();
        return;
      }

      if (data.success !== false) {
        // Handle both success: true and missing success field (backward compatibility)
        AppState.savedModels = data.models || [];
        // **KEY FIX: Call the new grid update method**
        this.updateModelsGrid();
        console.log(`✅ Loaded ${AppState.savedModels.length} saved models`);
      } else {
        console.error("API returned success: false");
        AppState.savedModels = [];
        this.updateModelsGrid();
      }
    } catch (error) {
      console.error("Failed to load saved models:", error);
      Utils.showNotification("Failed to load saved models", "error");
      AppState.savedModels = [];
      this.updateModelsGrid();
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
