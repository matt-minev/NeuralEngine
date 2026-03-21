Object.assign(DatasetGenerator, {
  currentConfig: {
    equation_type: "school_grade",
    num_equations: 1000,
    coefficient_range: { min: -5, max: 5 }, // Match Fast preset default
    root_type: "integers",
    allow_complex: false,
    pattern_distribution: "auto",
    validation_split: 0.2,
    test_split: 0.1,
    balanced_patterns: true,
    remove_duplicates: false,
    use_augmentation: true,
    ensemble_size: 1,
    epochs: 1000,
    use_multi_phase: true,
  },

  generatedDataset: null,

  init() {
    this.setupEventListeners();
    this.setupPresets();
    this.updateRangeDisplay();
    this.updateRangePresets();

    // Sync currentConfig with actual form values on page load
    const numEqInput = document.getElementById('num-equations');
    if (numEqInput) {
      this.currentConfig.num_equations = parseInt(numEqInput.value) || 1000;
    }

    const coeffMinInput = document.getElementById('coeff-min');
    const coeffMaxInput = document.getElementById('coeff-max');
    if (coeffMinInput && coeffMaxInput) {
      this.currentConfig.coefficient_range.min = parseInt(coeffMinInput.value) || -5;
      this.currentConfig.coefficient_range.max = parseInt(coeffMaxInput.value) || 5;
    }

    // Initialize accuracy meter after a short delay to ensure AccuracyPredictor is loaded
    setTimeout(() => {
      console.log('Initializing accuracy meter with config:', this.currentConfig);
      this.updateAccuracyMeter();
      this.updatePossibleEquationsCount();
    }, 200);
  },

  updateAccuracyPrediction() {
    // Use the unified updateAccuracyMeter method
    this.updateAccuracyMeter();
    this.updateRecommendedEpochs();
  },

  setupPresets() {
    // Dataset size preset buttons
    document.querySelectorAll('.preset-btn').forEach(btn => {
      btn.addEventListener('click', (e) => {
        const value = parseInt(e.target.dataset.value);
        document.getElementById('num-equations').value = value;
        this.currentConfig.num_equations = value;
        // Update active state
        document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
        e.target.classList.add('active');
        this.updateAccuracyPrediction();
      });
    });

    // Coefficient range preset buttons (using data-min/data-max format)
    document.querySelectorAll('.range-preset-btn').forEach(btn => {
      btn.addEventListener('click', (e) => {
        document.querySelectorAll('.range-preset-btn').forEach(b => b.classList.remove('active'));
        e.target.classList.add('active');

        const min = parseInt(e.target.dataset.min);
        const max = parseInt(e.target.dataset.max);

        document.getElementById('coeff-min').value = min;
        document.getElementById('coeff-max').value = max;
        this.currentConfig.coefficient_range = { min, max };
        this.updateRangeDisplay();
        this.updateAccuracyPrediction();
      });
    });

    // Quick preset cards
    document.querySelectorAll('.preset-card').forEach(card => {
      card.addEventListener('click', (e) => {
        this.applyPreset(e.currentTarget.dataset.preset);
      });
    });
  },

  applyPreset(presetName) {
    const presets = {
      'fast': {
        equation_type: 'school_grade',
        num_equations: 1000,
        coefficient_range: { min: -5, max: 5 },
        epochs: 1000,
        ensemble_size: 1,
        use_augmentation: true
      },
      'balanced': {
        equation_type: 'integer_solutions',
        num_equations: 10000,
        coefficient_range: { min: -10, max: 10 },
        epochs: 1500,
        ensemble_size: 1,
        use_augmentation: true
      },
      'high-accuracy': {
        equation_type: 'fractional_solutions',
        num_equations: 50000,
        coefficient_range: { min: -15, max: 15 },
        epochs: 2000,
        ensemble_size: 3,
        use_augmentation: true
      },
      'elite': {
        equation_type: 'random',
        num_equations: 100000,
        coefficient_range: { min: -15, max: 15 },
        epochs: 2500,
        ensemble_size: 5,
        use_augmentation: true,
        use_multi_phase: true
      }
    };

    const preset = presets[presetName];
    if (!preset) return;

    console.log('Applying preset:', presetName, preset);

    // Apply preset values to config
    Object.assign(this.currentConfig, preset);

    // Update all UI elements
    const numEqInput = document.getElementById('num-equations');
    const coeffMinInput = document.getElementById('coeff-min');
    const coeffMaxInput = document.getElementById('coeff-max');
    const epochsInput = document.getElementById('recommended-epochs');
    const ensembleInput = document.getElementById('ensemble-size');
    const augmentationInput = document.getElementById('use-augmentation');

    if (numEqInput) numEqInput.value = preset.num_equations;
    if (coeffMinInput) coeffMinInput.value = preset.coefficient_range.min;
    if (coeffMaxInput) coeffMaxInput.value = preset.coefficient_range.max;
    if (epochsInput) epochsInput.value = preset.epochs;
    if (ensembleInput) ensembleInput.value = preset.ensemble_size;
    if (augmentationInput) augmentationInput.checked = preset.use_augmentation;

    // Update range preset button
    document.querySelectorAll('.range-preset-btn').forEach(btn => {
      btn.classList.remove('active');
      if (parseInt(btn.dataset.min) === preset.coefficient_range.min &&
        parseInt(btn.dataset.max) === preset.coefficient_range.max) {
        btn.classList.add('active');
      }
    });

    // Update preset card highlighting
    document.querySelectorAll('.preset-card').forEach(card => {
      card.classList.remove('active');
    });
    const clickedCard = document.querySelector(`.preset-card[data-preset="${presetName}"]`);
    if (clickedCard) {
      clickedCard.classList.add('active');
    }

    // Update equation amount preset buttons
    document.querySelectorAll('.preset-btn').forEach(btn => {
      btn.classList.remove('active');
      if (parseInt(btn.dataset.value) === preset.num_equations) {
        btn.classList.add('active');
      }
    });

    // Update displays
    this.updateRangeDisplay();
    this.updateRangePresets();

    // Update equation type selector
    if (preset.equation_type) {
      const selector = document.querySelector(`.type-selector[data-type="${preset.equation_type}"]`);
      if (selector) {
        // This will internally call this.updateAccuracyPrediction()
        this.selectEquationType(selector);
      } else {
        this.updatePossibleEquationsCount();
        this.updateAccuracyPrediction();
      }
    } else {
      this.updatePossibleEquationsCount();
      this.updateAccuracyPrediction();
    }

    this.showNotification(`Applied "${presetName}" preset! 🎯`, 'success');
  },

  updateAccuracyMeterWithPreset(hardcodedAccuracy, trainingTimeSeconds) {
    // Use hard-coded accuracy instead of calculating
    const accuracyPercent = Math.round(hardcodedAccuracy * 100);

    // Get accuracy level based on hard-coded value
    let level;
    if (hardcodedAccuracy >= 0.96) {
      level = { name: 'Elite', color: '#8B5CF6', icon: '👑' };
    } else if (hardcodedAccuracy >= 0.90) {
      level = { name: 'Excellent', color: '#10B981', icon: '✅' };
    } else if (hardcodedAccuracy >= 0.82) {
      level = { name: 'Good', color: '#3B82F6', icon: '⭐' };
    } else if (hardcodedAccuracy >= 0.75) {
      level = { name: 'Acceptable', color: '#F59E0B', icon: '⚠️' };
    } else {
      level = { name: 'Insufficient', color: '#EF4444', icon: '❌' };
    }

    // Update progress circle
    const progressCircle = document.getElementById("accuracy-progress");
    const accuracyValue = document.getElementById("accuracy-value");
    if (!progressCircle || !accuracyValue) {
      console.warn('Accuracy meter elements not found, retrying...');
      setTimeout(() => {
        this.updateAccuracyMeterWithPreset(hardcodedAccuracy, trainingTimeSeconds);
      }, 100);
      return;
    }

    const radius = 90; // Compact version radius (180px circle)
    const circumference = 2 * Math.PI * radius;
    const offset = circumference - (accuracyPercent / 100) * circumference;

    // Update progress circle
    progressCircle.style.strokeDasharray = `${circumference} ${circumference}`;
    progressCircle.style.strokeDashoffset = offset;
    progressCircle.style.stroke = level.color;
    progressCircle.style.transition = 'stroke-dashoffset 0.8s ease-in-out, stroke 0.3s ease';

    // Update accuracy value
    accuracyValue.textContent = `${accuracyPercent}%`;

    // Update accuracy level
    const accuracyLevel = document.getElementById("accuracy-level");
    if (accuracyLevel) {
      accuracyLevel.textContent = `${level.icon} ${level.name}`;
      accuracyLevel.style.color = level.color;
    }

    // Update confidence
    const accuracyConfidence = document.getElementById("accuracy-confidence");
    if (accuracyConfidence) {
      accuracyConfidence.textContent = `±3%`;
    }

    // Update breakdown (estimate based on accuracy)
    const r2Score = document.getElementById("r2-score");
    if (r2Score) {
      r2Score.textContent = (hardcodedAccuracy + 0.02).toFixed(3);
    }

    const maeValue = document.getElementById("mae-value");
    if (maeValue) {
      const mae = Math.max(0.001, 0.1 - (hardcodedAccuracy - 0.7) * 0.3);
      maeValue.textContent = mae.toFixed(2);
    }

    // Update training time
    const trainingTime = document.getElementById("training-time");
    if (trainingTime) {
      trainingTime.textContent = `~${trainingTimeSeconds}s`;
    }
  },

  setupEventListeners() {
    // Equation type selection
    document.querySelectorAll(".type-selector").forEach((selector) => {
      selector.addEventListener("click", (e) => {
        this.selectEquationType(e.currentTarget);
      });
    });

    // Parameter inputs with accuracy updates
    document.getElementById("num-equations").addEventListener("input", (e) => {
      this.currentConfig.num_equations = parseInt(e.target.value) || 1000;
      // Update preset button active state
      document.querySelectorAll('.preset-btn').forEach(btn => {
        if (parseInt(btn.dataset.value) === this.currentConfig.num_equations) {
          btn.classList.add('active');
        } else {
          btn.classList.remove('active');
        }
      });
      this.updateAccuracyPrediction();
    });

    document.getElementById("root-type").addEventListener("change", (e) => {
      this.currentConfig.root_type = e.target.value;
      this.updateAccuracyPrediction();
      this.updatePossibleEquationsCount();
    });

    document.getElementById("coeff-min").addEventListener("input", (e) => {
      this.currentConfig.coefficient_range.min = parseInt(e.target.value);
      this.updateRangeDisplay();
      this.updateRangePresets();
      this.updateAccuracyPrediction();
      this.updatePossibleEquationsCount();
    });

    document.getElementById("coeff-max").addEventListener("input", (e) => {
      this.currentConfig.coefficient_range.max = parseInt(e.target.value);
      this.updateRangeDisplay();
      this.updateRangePresets();
      this.updateAccuracyPrediction();
      this.updatePossibleEquationsCount();
    });

    document.getElementById("allow-complex").addEventListener("change", (e) => {
      this.currentConfig.allow_complex = e.target.checked;
    });

    // New advanced options
    const validationSplit = document.getElementById("validation-split");
    if (validationSplit) {
      validationSplit.addEventListener("input", (e) => {
        this.currentConfig.validation_split = parseFloat(e.target.value);
      });
    }

    const testSplit = document.getElementById("test-split");
    if (testSplit) {
      testSplit.addEventListener("input", (e) => {
        this.currentConfig.test_split = parseFloat(e.target.value);
      });
    }

    const balancedDist = document.getElementById("balanced-distribution");
    if (balancedDist) {
      balancedDist.addEventListener("change", (e) => {
        this.currentConfig.balanced_patterns = e.target.checked;
        this.updateAccuracyPrediction();
      });
    }

    const removeDupes = document.getElementById("remove-duplicates");
    if (removeDupes) {
      removeDupes.addEventListener("change", (e) => {
        this.currentConfig.remove_duplicates = e.target.checked;
      });
    }

    const useAug = document.getElementById("use-augmentation");
    if (useAug) {
      useAug.addEventListener("change", (e) => {
        this.currentConfig.use_augmentation = e.target.checked;
        this.updateAccuracyPrediction();
      });
    }

    const ensembleSize = document.getElementById("ensemble-size");
    if (ensembleSize) {
      ensembleSize.addEventListener("input", (e) => {
        this.currentConfig.ensemble_size = parseInt(e.target.value);
        this.updateAccuracyPrediction();
      });
    }

    const recommendedEpochs = document.getElementById("recommended-epochs");
    if (recommendedEpochs) {
      recommendedEpochs.addEventListener("input", (e) => {
        this.currentConfig.epochs = parseInt(e.target.value);
        this.updateAccuracyPrediction();
      });
    }

    // Advanced options toggle
    document.querySelector(".toggle-advanced").addEventListener("click", () => {
      DatasetGenerator.toggleAdvancedOptions();
    });

    // Generate button
    document.getElementById("generate-btn").addEventListener("click", () => {
      this.generateDataset();
    });

    // Download button
    document.getElementById("download-btn").addEventListener("click", () => {
      this.downloadDataset();
    });

    // Load dataset button
    document
      .getElementById("load-dataset-btn")
      .addEventListener("click", () => {
        this.loadDatasetIntoApp();
      });

    // Infinite mode checkbox
    document
      .getElementById("infinite-mode")
      ?.addEventListener("change", (e) => {
        this.toggleInfiniteMode(e.target.checked);
      });

    // Stop generation button
    document
      .getElementById("stop-generation")
      ?.addEventListener("click", () => {
        this.stopInfiniteGeneration();
      });
  },

  selectEquationType(selector) {
    // Remove active class from all selectors
    document
      .querySelectorAll(".type-selector")
      .forEach((s) => s.classList.remove("active"));

    // Add active class to selected
    selector.classList.add("active");

    // Update config
    this.currentConfig.equation_type = selector.dataset.type;

    // Show/hide root type selector based on selection
    const rootTypeGroup = document
      .getElementById("root-type")
      .closest(".parameter-group");
    if (this.currentConfig.equation_type === "school_grade") {
      rootTypeGroup.style.display = "block";
    } else {
      rootTypeGroup.style.display = "none";
    }

    // Update accuracy prediction
    this.updateAccuracyPrediction();
    this.updatePossibleEquationsCount();
  },

  updateRangeDisplay() {
    const min = this.currentConfig.coefficient_range.min;
    const max = this.currentConfig.coefficient_range.max;
    document.getElementById("range-display").textContent = `${min} to ${max}`;
  },

  updateRangePresets() {
    const min = this.currentConfig.coefficient_range.min;
    const max = this.currentConfig.coefficient_range.max;

    document.querySelectorAll(".range-preset-btn").forEach((btn) => {
      if (parseInt(btn.dataset.min) === min && parseInt(btn.dataset.max) === max) {
        btn.classList.add("active");
      } else {
        btn.classList.remove("active");
      }
    });
  },

  updateAccuracyMeter() {
    if (typeof AccuracyPredictor === 'undefined') {
      console.warn('AccuracyPredictor not loaded yet, retrying...');
      // Retry after a short delay
      setTimeout(() => this.updateAccuracyMeter(), 200);
      return;
    }

    // Calculate accuracy using real predictor
    // Get current training config if available
    const config = {
      ...this.currentConfig,
      epochs: this.currentConfig.epochs || this.calculateRecommendedEpochs(),
    };

    let prediction;
    try {
      prediction = AccuracyPredictor.predictAccuracy(config);
      this.currentPrediction = prediction;
      console.log('Accuracy Prediction:', {
        config: config,
        accuracy: prediction.accuracyPercent + '%',
        level: prediction.level.name
      });
    } catch (error) {
      console.error('Error predicting accuracy:', error);
      return;
    }

    // Force update by ensuring elements exist
    const progressCircle = document.getElementById("accuracy-progress");
    const accuracyValue = document.getElementById("accuracy-value");
    if (!progressCircle || !accuracyValue) {
      console.warn('Accuracy meter elements not found, retrying...');
      setTimeout(() => this.updateAccuracyMeter(), 100);
      return;
    }

    // Update accuracy circle (compact version uses radius 90 for 180px circle)
    const accuracyPercent = prediction.accuracyPercent;
    const radius = 90; // Compact version radius (180px circle)
    const circumference = 2 * Math.PI * radius;
    const offset = circumference - (accuracyPercent / 100) * circumference;

    // Update progress circle
    progressCircle.style.strokeDasharray = `${circumference} ${circumference}`;
    progressCircle.style.strokeDashoffset = offset;
    progressCircle.style.stroke = prediction.level.color;
    progressCircle.style.transition = 'stroke-dashoffset 0.8s ease-in-out, stroke 0.3s ease';

    // Update accuracy value (reuse variable from above)
    accuracyValue.textContent = `${accuracyPercent}%`;

    // Update accuracy level
    const accuracyLevel = document.getElementById("accuracy-level");
    if (accuracyLevel) {
      accuracyLevel.textContent = `${prediction.level.icon} ${prediction.level.name}`;
      accuracyLevel.style.color = prediction.level.color;
    }

    // Update confidence
    const accuracyConfidence = document.getElementById("accuracy-confidence");
    if (accuracyConfidence) {
      const uncertainty = Math.round(prediction.confidenceInterval.uncertainty * 100);
      accuracyConfidence.textContent = `±${uncertainty}%`;
    }

    // Update breakdown
    const r2Score = document.getElementById("r2-score");
    if (r2Score) {
      r2Score.textContent = prediction.r2Score.toFixed(3);
    }

    const maeValue = document.getElementById("mae-value");
    if (maeValue) {
      maeValue.textContent = prediction.mae.toFixed(2);
    }

    const trainingTime = document.getElementById("training-time");
    if (trainingTime) {
      const timeEstimate = AccuracyPredictor.estimateTrainingTime(config);
      let timeText = '';
      if (timeEstimate.unit === 'hours') {
        timeText = `~${timeEstimate.value}h`;
      } else if (timeEstimate.unit === 'minutes') {
        timeText = `~${timeEstimate.value}m`;
      } else if (timeEstimate.unit === 'seconds') {
        timeText = `~${timeEstimate.value}s`;
      } else {
        timeText = `~${timeEstimate.value} ${timeEstimate.unit}`;
      }
      trainingTime.textContent = timeText;
    }

    // Update possible equations count
    this.updatePossibleEquationsCount();
  },

  /**
   * Calculate and display the number of possible school-grade equations
   * for the current coefficient range
   */
  updatePossibleEquationsCount() {
    const countEl = document.getElementById("equations-count");
    if (!countEl) return;

    if (this.currentConfig.equation_type !== 'school_grade') {
      countEl.textContent = 'N/A';
      return;
    }

    // Calculate possible equations based on coefficient range
    // For school-grade equations with integer roots and perfect square discriminants
    const min = this.currentConfig.coefficient_range.min;
    const max = this.currentConfig.coefficient_range.max;

    // Realistic calculation:
    // - Range size: number of possible integer values
    // - For each root pair (r1, r2), we can generate equations with different a values
    // - With pattern-based generation (standard factoring, diff squares, perfect squares, etc.),
    //   we can generate many unique equations
    // - Account for: a values (1-5 typically), root combinations, pattern variations

    const rangeSize = max - min + 1;

    // More realistic estimate:
    // - Base: rangeSize^2 possible root pairs
    // - Multiply by a values (typically 1-5, but can be more)
    // - Multiply by pattern variations (standard, diff squares, perfect squares, etc.)
    // - Factor in that not all combinations are valid (perfect square discriminant constraint)
    // - But with our pattern-based approach, we can generate many valid equations

    // For school grade equations, we use patterns that ensure validity
    // So we can generate roughly: rootPairs * aValues * patterns * validityFactor
    const rootPairs = rangeSize * rangeSize; // All possible root pairs
    const aValues = 5; // Typically 1-5
    const patterns = 4; // Standard, diff squares, perfect squares, larger coeffs
    const validityFactor = 0.6; // ~60% of combinations are valid with patterns

    let estimate = Math.floor(rootPairs * aValues * patterns * validityFactor);

    // Ensure minimum reasonable estimate
    if (estimate < 1000) {
      estimate = Math.floor(rootPairs * 10); // At least 10 equations per root pair
    }

    // Cap at reasonable maximum (we can generate more with infinite mode)
    if (estimate > 1000000) {
      estimate = 1000000;
    }

    // Format with commas
    const formatted = estimate.toLocaleString();
    countEl.textContent = formatted;
  },

  calculateRecommendedEpochs() {
    const datasetSize = this.currentConfig.num_equations || 1000;

    // Logarithmic scaling: more data = more epochs needed
    if (datasetSize >= 100000) {
      return 2500;
    } else if (datasetSize >= 50000) {
      return 2000;
    } else if (datasetSize >= 10000) {
      return 1500;
    } else if (datasetSize >= 5000) {
      return 1200;
    } else {
      return 1000;
    }
  },

  updateRecommendedEpochs() {
    const epochs = this.calculateRecommendedEpochs();
    this.currentConfig.epochs = epochs;

    const epochsInput = document.getElementById("recommended-epochs");
    if (epochsInput) {
      epochsInput.value = epochs;

      const hint = document.getElementById("epochs-hint");
      if (hint) {
        if (epochs >= 2000) {
          hint.textContent = "Elite training recommended";
        } else if (epochs >= 1500) {
          hint.textContent = "High quality training";
        } else {
          hint.textContent = "Standard training";
        }
      }
    }
  },

  updateRecommendations() {
    if (typeof AccuracyPredictor === 'undefined' || !this.currentPrediction) {
      return;
    }

    const recommendations = AccuracyPredictor.getRecommendations(
      this.currentConfig,
      this.currentPrediction.accuracy
    );

    const recommendationsList = document.getElementById("recommendations-list");
    if (!recommendationsList) return;

    if (recommendations.length === 0) {
      recommendationsList.innerHTML = `
        <div class="recommendation-item" style="border-left-color: var(--success-color); background: rgba(52, 199, 89, 0.05);">
          <div class="recommendation-message">✅ Your configuration looks optimal!</div>
        </div>
      `;
      return;
    }

    recommendationsList.innerHTML = recommendations.map(rec => `
      <div class="recommendation-item priority-${rec.priority}">
        <div class="recommendation-message">${rec.message}</div>
        <div class="recommendation-impact">Impact: ${rec.impact}</div>
        <div class="recommendation-action">
          <button class="recommendation-btn" data-action="${rec.type}">
            Apply
          </button>
        </div>
      </div>
    `).join('');

    // Add event listeners to apply buttons
    recommendationsList.querySelectorAll(".recommendation-btn").forEach((btn) => {
      btn.addEventListener("click", (e) => {
        const actionType = e.target.dataset.action;
        const rec = recommendations.find(r => r.type === actionType);
        if (rec && rec.action) {
          const updates = rec.action();
          this.applyConfigUpdates(updates);
        }
      });
    });
  },

  applyConfigUpdates(updates) {
    Object.keys(updates).forEach(key => {
      this.currentConfig[key] = updates[key];

      // Update UI elements
      if (key === 'num_equations') {
        document.getElementById("num-equations").value = updates[key];
        this.updateRecommendedEpochs();
      } else if (key === 'coefficient_range') {
        document.getElementById("coeff-min").value = updates[key].min;
        document.getElementById("coeff-max").value = updates[key].max;
        this.updateRangeDisplay();
        this.updateRangePresets();
      } else if (key === 'equation_type') {
        // Trigger equation type selection
        const selector = document.querySelector(`[data-type="${updates[key]}"]`);
        if (selector) {
          this.selectEquationType(selector);
        }
      } else if (key === 'use_augmentation') {
        const checkbox = document.getElementById("use-augmentation");
        if (checkbox) checkbox.checked = updates[key];
      } else if (key === 'ensemble_size') {
        const input = document.getElementById("ensemble-size");
        if (input) input.value = updates[key];
      }
    });

    this.updateAccuracyMeter();
  },

  toggleAdvancedOptions() {
    const advancedOptions = document.querySelector(".advanced-options");
    const toggleText = document.querySelector(".toggle-text");
    const toggleIcon = document.querySelector(".toggle-icon");

    if (advancedOptions.style.display === "none") {
      advancedOptions.style.display = "block";
      toggleText.textContent = "Hide";
      toggleIcon.textContent = "▲";
    } else {
      advancedOptions.style.display = "none";
      toggleText.textContent = "Show";
      toggleIcon.textContent = "▼";
    }
  },

  infiniteMode: {
    active: false,
    intervalId: null,
    generatedCount: 0,
    currentRange: { min: -2, max: 2 },
  },

  toggleInfiniteMode(enabled) {
    const numEquationsGroup = document
      .getElementById("num-equations")
      ?.closest(".parameter-group");
    const infiniteStats = document.getElementById("infinite-stats");

    if (numEquationsGroup) {
      numEquationsGroup.style.display = enabled ? "none" : "block";
    }

    if (infiniteStats) {
      infiniteStats.style.display = enabled ? "block" : "none";
    }
  },

  async generateDataset() {
    const isInfiniteMode = document.getElementById("infinite-mode")?.checked;

    if (isInfiniteMode) {
      await this.startInfiniteGeneration();
    } else {
      await this.generateFiniteDataset();
    }
  },

  async startInfiniteGeneration() {
    try {
      // Start infinite generation
      const response = await fetch("/api/generate-dataset-infinite/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          equation_type: this.currentConfig.equation_type,
          root_type: this.currentConfig.root_type,
          allow_complex: this.currentConfig.allow_complex,
        }),
      });

      const result = await response.json();
      if (!result.success) {
        throw new Error(result.error);
      }

      // Update UI with proper button replacement
      this.infiniteMode.active = true;
      this.infiniteMode.generatedCount = 0;

      const generateBtn = document.getElementById("generate-btn");
      const stopBtn = document.getElementById("stop-generation");

      // Smooth transition out for generate button
      generateBtn.style.transition = "all 0.2s ease-out";
      generateBtn.style.opacity = "0";
      generateBtn.style.transform = "translateY(-10px)";

      // After transition, hide generate and show stop button
      setTimeout(() => {
        generateBtn.style.display = "none";

        // Show and animate in the stop button
        stopBtn.style.display = "flex";
        stopBtn.style.opacity = "0";
        stopBtn.style.transform = "translateY(10px)";
        stopBtn.style.transition = "all 0.2s ease-out";

        // Trigger animation
        setTimeout(() => {
          stopBtn.style.opacity = "1";
          stopBtn.style.transform = "translateY(0)";
        }, 10);
      }, 200);

      // Show infinite stats
      document.getElementById("infinite-stats").style.display = "flex";

      // Start batch generation loop
      this.infiniteGenerationLoop();

      this.showNotification("Infinite generation started! 🚀", "success");
    } catch (error) {
      this.showNotification(
        `Failed to start infinite generation: ${error.message}`,
        "error"
      );
    }
  },

  async generateFiniteDataset() {
    try {
      // Show progress
      this.showProgress();

      // Validate inputs
      if (
        this.currentConfig.num_equations < 100 ||
        this.currentConfig.num_equations > 10000000
      ) {
        throw new Error("Number of equations must be between 100 and 100,000");
      }

      if (
        this.currentConfig.coefficient_range.min >=
        this.currentConfig.coefficient_range.max
      ) {
        throw new Error("Minimum coefficient must be less than maximum");
      }

      // Generate dataset using the original API endpoint
      const response = await fetch("/api/generate-dataset", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(this.currentConfig),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || "Generation failed");
      }

      const result = await response.json();
      this.generatedDataset = result;

      // Hide progress and show results
      this.hideProgress();
      this.showResults(result);
      this.showNotification("Dataset generated successfully! 🎉", "success");
    } catch (error) {
      this.hideProgress();
      this.showNotification(`Generation failed: ${error.message}`, "error");
    }
  },

  async infiniteGenerationLoop() {
    if (!this.infiniteMode.active) return;

    try {
      const response = await fetch("/api/generate-dataset-infinite/batch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      });

      const result = await response.json();
      if (result.success) {
        // Update display
        this.infiniteMode.generatedCount = result.generated_count;
        this.infiniteMode.currentRange = result.current_range;

        document.getElementById("equations-generated").textContent =
          result.generated_count.toLocaleString();

        // Update range display if it exists
        const rangeDisplay = document.getElementById("infinite-range-display");
        if (rangeDisplay) {
          rangeDisplay.textContent = `${result.current_range.min} to ${result.current_range.max}`;
        }

        // Update preview table - FIX: Proper data format handling
        if (result.preview && result.preview.length > 0) {
          // Convert backend data format to frontend expected format
          const formattedPreview = result.preview.map((equation) => {
            // Backend returns: [a, b, c, x1, x2]
            // Frontend expects: {equation, a, b, c, x1, x2}
            const [a, b, c, x1, x2] = equation;
            return {
              equation: this.formatEquation(a, b, c),
              a: a,
              b: b,
              c: c,
              x1: x1,
              x2: x2,
            };
          });

          this.populatePreviewTable(formattedPreview);
        }
      }
    } catch (error) {
      console.error("Batch generation error:", error);
    }

    // Schedule next batch
    if (this.infiniteMode.active) {
      this.infiniteMode.intervalId = setTimeout(() => {
        this.infiniteGenerationLoop();
      }, 200); // 200ms between batches
    }
  },

  async stopInfiniteGeneration() {
    try {
      this.infiniteMode.active = false;

      if (this.infiniteMode.intervalId) {
        clearTimeout(this.infiniteMode.intervalId);
      }

      const response = await fetch("/api/generate-dataset-infinite/stop", {
        method: "POST",
      });

      const result = await response.json();
      if (result.success) {
        // Restore UI with smooth button transition
        const generateBtn = document.getElementById("generate-btn");
        const stopBtn = document.getElementById("stop-generation");

        // Smooth transition out for stop button
        stopBtn.style.transition = "all 0.2s ease-out";
        stopBtn.style.opacity = "0";
        stopBtn.style.transform = "translateY(-10px)";

        // After transition, hide stop and show generate button
        setTimeout(() => {
          stopBtn.style.display = "none";

          // Show and animate in the generate button
          generateBtn.style.display = "flex";
          generateBtn.style.opacity = "0";
          generateBtn.style.transform = "translateY(10px)";
          generateBtn.style.transition = "all 0.2s ease-out";

          // Trigger animation
          setTimeout(() => {
            generateBtn.style.opacity = "1";
            generateBtn.style.transform = "translateY(0)";
          }, 10);
        }, 200);

        this.showNotification(
          `Generation stopped! Final count: ${result.final_count.toLocaleString()} equations.`,
          "success"
        );

        // Show download if dataset was saved
        if (result.filename) {
          this.downloadFilename = result.filename;
          document.getElementById("generation-results").style.display = "block";
          this.populateStatistics(result.stats);

          // Scroll to results after a brief delay
          setTimeout(() => {
            document.getElementById("generation-results").scrollIntoView({
              behavior: "smooth",
            });
          }, 300);
        }
      }
    } catch (error) {
      this.showNotification(
        `Failed to stop generation: ${error.message}`,
        "error"
      );
    }
  },
  showProgress() {
    document.getElementById("generation-results").style.display = "none";
    document.getElementById("generation-progress").style.display = "block";

    // Animate progress bar
    const progressFill = document.getElementById("progress-fill");
    const progressText = document.getElementById("progress-text");

    let progress = 0;
    const interval = setInterval(() => {
      progress += Math.random() * 10;
      if (progress > 90) {
        progressText.textContent = "Finalizing dataset...";
        clearInterval(interval);
      } else {
        progressFill.style.width = `${progress}%`;
        progressText.textContent = `Generating equations... ${Math.round(
          progress
        )}%`;
      }
    }, 100);
  },

  hideProgress() {
    document.getElementById("generation-progress").style.display = "none";
  },

  showResults(result) {
    const resultsSection = document.getElementById("generation-results");
    resultsSection.style.display = "block";

    // Populate statistics
    this.populateStatistics(result.stats);

    // Populate preview table
    this.populatePreviewTable(result.preview);

    // Store filename for download
    this.downloadFilename = result.filename;

    // Scroll to results
    resultsSection.scrollIntoView({ behavior: "smooth" });
  },

  populateStatistics(stats) {
    const statsContainer = document.getElementById("dataset-stats");

    const totalEquations = stats.total_equations.toLocaleString();
    const integerRootsPercent = (
      ((stats.quality_metrics.integer_roots_x1 +
        stats.quality_metrics.integer_roots_x2) /
        (2 * stats.total_equations)) *
      100
    ).toFixed(1);
    const integerCoeffsPercent = (
      (stats.quality_metrics.integer_coefficients / stats.total_equations) *
      100
    ).toFixed(1);

    statsContainer.innerHTML = `
            <div class="stat-item">
                <div class="stat-icon">📊</div>
                <div class="stat-content">
                    <div class="stat-value">${totalEquations}</div>
                    <div class="stat-label">Total Equations</div>
                </div>
            </div>
            
            <div class="stat-item">
                <div class="stat-icon">🎯</div>
                <div class="stat-content">
                    <div class="stat-value">${integerRootsPercent}%</div>
                    <div class="stat-label">Integer Roots</div>
                </div>
            </div>
            
            <div class="stat-item">
                <div class="stat-icon">🔢</div>
                <div class="stat-content">
                    <div class="stat-value">${integerCoeffsPercent}%</div>
                    <div class="stat-label">Integer Coefficients</div>
                </div>
            </div>
            
            <div class="stat-item">
                <div class="stat-icon">📏</div>
                <div class="stat-content">
                    <div class="stat-value">${this.formatNumber(
      stats.coefficients.a.mean,
      2
    )}</div>
                    <div class="stat-label">Avg Coefficient 'a'</div>
                </div>
            </div>
            
            <div class="stat-item">
                <div class="stat-icon">📐</div>
                <div class="stat-content">
                    <div class="stat-value">${this.formatNumber(
      stats.roots.x1.mean,
      2
    )}</div>
                    <div class="stat-label">Avg Root x₁</div>
                </div>
            </div>
            
            <div class="stat-item">
                <div class="stat-icon">✨</div>
                <div class="stat-content">
                    <div class="stat-value">${this.currentConfig.equation_type === "school_grade"
        ? "Perfect"
        : "Good"
      }</div>
                    <div class="stat-label">Quality Rating</div>
                </div>
            </div>
        `;
  },

  populatePreviewTable(preview) {
    const tableBody = document.getElementById("preview-table-body");

    tableBody.innerHTML = preview
      .map((equation) => {
        const [a, b, c, x1, x2] = equation;
        const equationStr = this.formatEquation(a, b, c);

        return `
                <tr>
                    <td>${this.formatNumber(a, 3)}</td>
                    <td>${this.formatNumber(b, 3)}</td>
                    <td>${this.formatNumber(c, 3)}</td>
                    <td>${this.formatNumber(x1, 3)}</td>
                    <td>${this.formatNumber(x2, 3)}</td>
                    <td class="equation-cell">${equationStr}</td>
                </tr>
            `;
      })
      .join("");
  },

  formatEquation(a, b, c) {
    let equation = "";

    // Format coefficient a
    if (a === 1) {
      equation += "x²";
    } else if (a === -1) {
      equation += "-x²";
    } else {
      equation += `${this.formatNumber(a, 0)}x²`;
    }

    // Format coefficient b
    if (b > 0) {
      equation += b === 1 ? " + x" : ` + ${this.formatNumber(b, 0)}x`;
    } else if (b < 0) {
      equation +=
        b === -1 ? " - x" : ` - ${this.formatNumber(Math.abs(b), 0)}x`;
    }

    // Format coefficient c
    if (c > 0) {
      equation += ` + ${this.formatNumber(c, 0)}`;
    } else if (c < 0) {
      equation += ` - ${this.formatNumber(Math.abs(c), 0)}`;
    }

    equation += " = 0";
    return equation;
  },

  formatNumber(num, decimals = 6) {
    if (typeof num !== "number" || isNaN(num)) return "0";
    if (Math.abs(num) < 1e-10) return "0";

    // Check if it's effectively an integer
    if (Math.abs(num - Math.round(num)) < 1e-10) {
      return Math.round(num).toString();
    }

    return parseFloat(num.toFixed(decimals)).toString();
  },

  async downloadDataset() {
    if (!this.downloadFilename) {
      this.showNotification("No dataset to download", "error");
      return;
    }

    try {
      const response = await fetch(
        `/api/download-dataset/${this.downloadFilename}`
      );
      if (!response.ok) throw new Error("Download failed");

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = this.downloadFilename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      window.URL.revokeObjectURL(url);

      this.showNotification("Dataset downloaded successfully! 📥", "success");
    } catch (error) {
      this.showNotification(`Download failed: ${error.message}`, "error");
    }
  },

  async loadDatasetIntoApp() {
    if (!this.downloadFilename) {
      this.showNotification("No dataset to load", "error");
      return;
    }

    try {
      const response = await fetch(
        `/api/data/load/${encodeURIComponent(this.downloadFilename)}`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
        }
      );

      if (!response.ok) {
        const error = await response.json().catch(() => ({}));
        throw new Error(error.error || "Failed to load dataset");
      }

      // Redirect to main app with dataset parameter so the UI can sync to the loaded dataset
      window.location.href = `/?load_dataset=${encodeURIComponent(this.downloadFilename)}`;
    } catch (error) {
      this.showNotification(
        `Failed to load dataset: ${error.message}`,
        "error"
      );
    }
  },

  showNotification(message, type = "info") {
    const notification = document.createElement("div");
    notification.className = `notification notification-${type}`;
    notification.innerHTML = `
            <div class="notification-content">
                <span class="notification-message">${message}</span>
                <button class="notification-close">×</button>
            </div>
        `;

    document.body.appendChild(notification);

    // Auto-remove after 5 seconds
    setTimeout(() => {
      if (notification.parentNode) {
        notification.parentNode.removeChild(notification);
      }
    }, 5000);

    // Close button
    notification
      .querySelector(".notification-close")
      .addEventListener("click", () => {
        if (notification.parentNode) {
          notification.parentNode.removeChild(notification);
        }
      });
  },
});

