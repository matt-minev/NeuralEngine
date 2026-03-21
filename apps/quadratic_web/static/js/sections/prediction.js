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
