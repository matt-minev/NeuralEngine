/**
 * Dataset Generator JavaScript
 * Beautiful Apple-like interface for generating quadratic equation datasets
 */

const DatasetGenerator = {
  currentConfig: {
    equation_type: "school_grade",
    num_equations: 1000,
    coefficient_range: { min: -10, max: 10 },
    root_type: "mixed",
    allow_complex: false,
  },

  generatedDataset: null,

  init() {
    this.setupEventListeners();
    this.updateRangeDisplay();
  },

  setupEventListeners() {
    // Equation type selection
    document.querySelectorAll(".type-selector").forEach((selector) => {
      selector.addEventListener("click", (e) => {
        this.selectEquationType(e.currentTarget);
      });
    });

    // Parameter inputs
    document.getElementById("num-equations").addEventListener("input", (e) => {
      this.currentConfig.num_equations = parseInt(e.target.value);
    });

    document.getElementById("root-type").addEventListener("change", (e) => {
      this.currentConfig.root_type = e.target.value;
    });

    document.getElementById("coeff-min").addEventListener("input", (e) => {
      this.currentConfig.coefficient_range.min = parseInt(e.target.value);
      this.updateRangeDisplay();
    });

    document.getElementById("coeff-max").addEventListener("input", (e) => {
      this.currentConfig.coefficient_range.max = parseInt(e.target.value);
      this.updateRangeDisplay();
    });

    document.getElementById("allow-complex").addEventListener("change", (e) => {
      this.currentConfig.allow_complex = e.target.checked;
    });

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
  },

  updateRangeDisplay() {
    const min = this.currentConfig.coefficient_range.min;
    const max = this.currentConfig.coefficient_range.max;
    document.getElementById("range-display").textContent = `${min} to ${max}`;
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
        this.currentConfig.num_equations > 100000
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
                    <div class="stat-value">${
                      this.currentConfig.equation_type === "school_grade"
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
      // Redirect to main app with dataset parameter
      window.location.href = `/?load_dataset=${this.downloadFilename}`;
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
};

// Initialize when DOM is ready
document.addEventListener("DOMContentLoaded", () => {
  DatasetGenerator.init();
});
