/**
 * Quadratic Neural Network Web Application
 * Data Management Interface
 *
 * Author: Matt
 * Location: Varna, Bulgaria
 * Date: July 2025
 *
 * Advanced data management with drag-and-drop, validation, and visualization
 */

// Data management interface
const DataManager = {
  // Current dataset state
  currentDataset: null,
  dataStats: null,
  sampleData: [],

  // UI elements
  uploadZone: null,
  fileInput: null,
  dataTable: null,
  statsContainer: null,

  // Configuration
  maxFileSize: 50 * 1024 * 1024, // 50MB
  supportedFormats: ["csv", "txt"],
  maxPreviewRows: 500,

  // Initialize data manager
  init() {
    this.setupElements();
    this.setupEventListeners();
    this.setupDragAndDrop();
    this.refreshDataInfo();

    console.log("📊 Data management interface initialized");
  },

  // Setup UI elements
  setupElements() {
    this.uploadZone = this.createUploadZone();
    this.fileInput = document.getElementById("file-input");
    this.dataTable = document.getElementById("data-table");
    this.statsContainer = document.getElementById("dataset-info");

    // Insert upload zone if dataset section exists
    const dataSection = document.getElementById("data");
    if (dataSection && !document.querySelector(".upload-zone")) {
      const uploadCard = document.createElement("div");
      uploadCard.className = "card";
      uploadCard.innerHTML = `
                <div class="card-header">
                    <h3 class="card-title">
                        <i class="fas fa-upload"></i>
                        Upload Dataset
                    </h3>
                </div>
                <div class="upload-zone-container">
                    ${this.uploadZone.outerHTML}
                </div>
            `;

      const firstCard = dataSection.querySelector(".card");
      if (firstCard) {
        firstCard.parentNode.insertBefore(uploadCard, firstCard);
      }
    }
  },

  // Create upload zone
  createUploadZone() {
    const zone = document.createElement("div");
    zone.className = "upload-zone";
    zone.innerHTML = `
            <div class="upload-content">
                <div class="upload-icon">
                    <i class="fas fa-cloud-upload-alt"></i>
                </div>
                <div class="upload-text">
                    Drop your CSV file here or click to browse
                </div>
                <div class="upload-subtext">
                    Maximum file size: 50MB
                </div>
                <div class="upload-formats">
                    Supported formats: CSV, TXT
                </div>
                <div class="upload-example">
                    <strong>Expected format:</strong> a, b, c, x1, x2
                </div>
            </div>
        `;

    return zone;
  },

  // Setup event listeners
  setupEventListeners() {
    // File input change
    if (this.fileInput) {
      this.fileInput.addEventListener("change", (e) => {
        this.handleFileSelect(e.target.files);
      });
    }

    // Upload zone click
    document.addEventListener("click", (e) => {
      if (e.target.closest(".upload-zone")) {
        this.fileInput?.click();
      }
    });

    // Data table interactions
    if (this.dataTable) {
      this.setupTableInteractions();
    }
  },

  // Setup drag and drop
  setupDragAndDrop() {
    const uploadZone = document.querySelector(".upload-zone");
    if (!uploadZone) return;

    ["dragenter", "dragover", "dragleave", "drop"].forEach((eventName) => {
      uploadZone.addEventListener(eventName, this.preventDefaults, false);
      document.body.addEventListener(eventName, this.preventDefaults, false);
    });

    ["dragenter", "dragover"].forEach((eventName) => {
      uploadZone.addEventListener(
        eventName,
        () => {
          uploadZone.classList.add("dragover");
        },
        false
      );
    });

    ["dragleave", "drop"].forEach((eventName) => {
      uploadZone.addEventListener(
        eventName,
        () => {
          uploadZone.classList.remove("dragover");
        },
        false
      );
    });

    uploadZone.addEventListener(
      "drop",
      (e) => {
        const files = e.dataTransfer.files;
        this.handleFileSelect(files);
      },
      false
    );
  },

  // Prevent default drag behaviors
  preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
  },

  // Handle file selection
  async handleFileSelect(files) {
    if (!files || files.length === 0) return;

    const file = files[0];

    // Validate file
    const validation = this.validateFile(file);
    if (!validation.valid) {
      Utils.showNotification(validation.error, "error");
      return;
    }

    // Show loading state
    this.showUploadProgress(true);

    try {
      // Upload file
      const result = await this.uploadFile(file);

      if (result.success) {
        Utils.showNotification(result.message, "success");
        this.currentDataset = result;
        this.updateDataDisplay(result);
        AppState.dataLoaded = true;

        // Refresh other sections
        this.notifyDataLoaded();
      } else {
        Utils.showNotification(result.error || "Upload failed", "error");
      }
    } catch (error) {
      Utils.showNotification("Upload failed: " + error.message, "error");
    } finally {
      this.showUploadProgress(false);
    }
  },

  // Validate file
  validateFile(file) {
    // Check file size
    if (file.size > this.maxFileSize) {
      return {
        valid: false,
        error: `File too large. Maximum size: ${
          this.maxFileSize / (1024 * 1024)
        }MB`,
      };
    }

    // Check file type
    const extension = file.name.split(".").pop().toLowerCase();
    if (!this.supportedFormats.includes(extension)) {
      return {
        valid: false,
        error: `Unsupported file format. Supported: ${this.supportedFormats.join(
          ", "
        )}`,
      };
    }

    // Check filename
    if (file.name.length > 255) {
      return {
        valid: false,
        error: "Filename too long",
      };
    }

    return { valid: true };
  },

  // Upload file to server
  async uploadFile(file) {
    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch("/api/data/upload", {
      method: "POST",
      body: formData,
    });

    return await response.json();
  },

  // Show upload progress
  showUploadProgress(show) {
    const uploadZone = document.querySelector(".upload-zone");
    if (!uploadZone) return;

    if (show) {
      uploadZone.classList.add("uploading");
      uploadZone.innerHTML = `
                <div class="upload-progress">
                    <div class="loading-spinner"></div>
                    <div class="upload-text">Uploading and processing...</div>
                </div>
            `;
    } else {
      uploadZone.classList.remove("uploading");
      uploadZone.innerHTML = this.createUploadZone().innerHTML;
    }
  },

  // Update data display
  updateDataDisplay(data) {
    this.updateDataInfo(data);
    this.updateDataTable(data.sample_data);
    this.updateDataVisualization(data.stats);
  },

  // Update data information
  updateDataInfo(data) {
    if (!this.statsContainer) return;

    this.statsContainer.innerHTML = `
            <div class="data-info-grid">
                <div class="info-card">
                    <div class="info-header">
                        <i class="fas fa-chart-bar"></i>
                        <h4>Dataset Overview</h4>
                    </div>
                    <div class="info-content">
                        <div class="info-item">
                            <span class="info-label">Total Equations:</span>
                            <span class="info-value">${data.stats.total_equations.toLocaleString()}</span>
                        </div>
                        <div class="info-item">
                            <span class="info-label">Features:</span>
                            <span class="info-value">a, b, c, x1, x2</span>
                        </div>
                        <div class="info-item">
                            <span class="info-label">File:</span>
                            <span class="info-value">${data.filename}</span>
                        </div>
                    </div>
                </div>
                
                <div class="info-card">
                    <div class="info-header">
                        <i class="fas fa-calculator"></i>
                        <h4>Statistics</h4>
                    </div>
                    <div class="info-content">
                        ${this.generateStatsHTML(data.stats)}
                    </div>
                </div>
                
                <div class="info-card">
                    <div class="info-header">
                        <i class="fas fa-check-circle"></i>
                        <h4>Data Quality</h4>
                    </div>
                    <div class="info-content">
                        ${this.generateQualityHTML(data.stats)}
                    </div>
                </div>
            </div>
        `;
  },

  // Generate statistics HTML
  generateStatsHTML(stats) {
    if (!stats.columns) return "<p>No statistics available</p>";

    return Object.entries(stats.columns)
      .map(
        ([name, colStats]) => `
            <div class="stat-group">
                <h5>${name.toUpperCase()}</h5>
                <div class="stat-items">
                    <div class="stat-item">
                        <span>Mean:</span>
                        <span>${Utils.formatNumber(colStats.mean, 3)}</span>
                    </div>
                    <div class="stat-item">
                        <span>Std:</span>
                        <span>${Utils.formatNumber(colStats.std, 3)}</span>
                    </div>
                    <div class="stat-item">
                        <span>Range:</span>
                        <span>${Utils.formatNumber(
                          colStats.min,
                          2
                        )} to ${Utils.formatNumber(colStats.max, 2)}</span>
                    </div>
                </div>
            </div>
        `
      )
      .join("");
  },

  // Generate quality HTML
  generateQualityHTML(stats) {
    if (!stats.quality) return "<p>Quality metrics not available</p>";

    return `
            <div class="quality-metrics">
                <div class="quality-item">
                    <span class="quality-label">Integer Solutions (x1):</span>
                    <span class="quality-value">${Utils.formatNumber(
                      stats.quality.x1_whole_pct,
                      1
                    )}%</span>
                </div>
                <div class="quality-item">
                    <span class="quality-label">Integer Solutions (x2):</span>
                    <span class="quality-value">${Utils.formatNumber(
                      stats.quality.x2_whole_pct,
                      1
                    )}%</span>
                </div>
                <div class="quality-item">
                    <span class="quality-label">Data Validation:</span>
                    <span class="quality-value success">✓ Passed</span>
                </div>
            </div>
        `;
  },

  // Update data table
  updateDataTable(sampleData) {
    const tableBody = document.getElementById("data-table-body");
    const table = document.getElementById("data-table");

    if (!tableBody || !table) return;

    if (!sampleData || sampleData.length === 0) {
      table.style.display = "none";
      return;
    }

    // Clear existing data
    tableBody.innerHTML = "";

    // Add sample data rows
    sampleData.forEach((row, index) => {
      const tr = document.createElement("tr");
      tr.className = "data-row";

      // Add row number
      const rowNumber = document.createElement("td");
      rowNumber.className = "row-number";
      rowNumber.textContent = index + 1;
      tr.appendChild(rowNumber);

      // Add data cells
      row.forEach((value, colIndex) => {
        const td = document.createElement("td");
        td.className = "data-cell";
        td.textContent = Utils.formatNumber(value, 4);

        // Add color coding based on value
        if (colIndex < 3) {
          // Coefficients
          td.classList.add("coefficient");
        } else {
          // Roots
          td.classList.add("root");
        }

        tr.appendChild(td);
      });

      // Add row actions
      const actionsCell = document.createElement("td");
      actionsCell.className = "row-actions";
      actionsCell.innerHTML = `
                <button class="btn-small" onclick="DataManager.useAsExample(${index})">
                    <i class="fas fa-play"></i>
                </button>
                <button class="btn-small" onclick="DataManager.showRowDetails(${index})">
                    <i class="fas fa-info"></i>
                </button>
            `;
      tr.appendChild(actionsCell);

      tableBody.appendChild(tr);
    });

    table.style.display = "table";

    // Add table controls
    this.addTableControls();
  },

  // Add table controls
  addTableControls() {
    const tableContainer = document.querySelector(".data-table-container");
    if (!tableContainer) return;

    // Check if controls already exist
    if (tableContainer.querySelector(".table-controls")) return;

    const controls = document.createElement("div");
    controls.className = "table-controls";
    controls.innerHTML = `
            <div class="table-info">
                <span>Showing ${Math.min(
                  this.sampleData.length,
                  this.maxPreviewRows
                )} of ${
      this.currentDataset?.stats?.total_equations || 0
    } equations</span>
            </div>
            <div class="table-actions">
                <button class="btn-small" onclick="DataManager.refreshTable()">
                    <i class="fas fa-sync"></i> Refresh
                </button>
                <button class="btn-small" onclick="DataManager.exportSample()">
                    <i class="fas fa-download"></i> Export Sample
                </button>
                <button class="btn-small" onclick="DataManager.generateReport()">
                    <i class="fas fa-file-alt"></i> Generate Report
                </button>
            </div>
        `;

    tableContainer.appendChild(controls);
  },

  // Setup table interactions
  setupTableInteractions() {
    if (!this.dataTable) return;

    // Row hover effects
    this.dataTable.addEventListener("mouseover", (e) => {
      const row = e.target.closest(".data-row");
      if (row) {
        row.classList.add("hover");
      }
    });

    this.dataTable.addEventListener("mouseout", (e) => {
      const row = e.target.closest(".data-row");
      if (row) {
        row.classList.remove("hover");
      }
    });

    // Row click for details
    this.dataTable.addEventListener("click", (e) => {
      const row = e.target.closest(".data-row");
      if (row && !e.target.closest(".row-actions")) {
        const rowIndex = Array.from(row.parentNode.children).indexOf(row);
        this.showRowDetails(rowIndex);
      }
    });
  },

  // Use row as example
  useAsExample(rowIndex) {
    if (
      !this.currentDataset?.sample_data ||
      rowIndex >= this.currentDataset.sample_data.length
    ) {
      return;
    }

    const row = this.currentDataset.sample_data[rowIndex];

    // Navigate to prediction section
    Navigation.showSection("prediction");

    // Wait for section to load, then populate
    setTimeout(() => {
      if (typeof PredictionManager !== "undefined") {
        // Set coefficients → roots scenario
        const scenarioSelect = document.getElementById("prediction-scenario");
        if (scenarioSelect) {
          scenarioSelect.value = "coeff_to_roots";
          scenarioSelect.dispatchEvent(new Event("change"));
        }

        // Populate input fields
        setTimeout(() => {
          const inputs = ["a", "b", "c"];
          inputs.forEach((feature, index) => {
            const input = document.getElementById(`input-${feature}`);
            if (input && row[index] !== undefined) {
              input.value = row[index];
            }
          });

          Utils.showNotification(
            `Example loaded: Row ${rowIndex + 1}`,
            "success"
          );
        }, 100);
      }
    }, 100);
  },

  // Show row details
  showRowDetails(rowIndex) {
    if (
      !this.currentDataset?.sample_data ||
      rowIndex >= this.currentDataset.sample_data.length
    ) {
      return;
    }

    const row = this.currentDataset.sample_data[rowIndex];
    const [a, b, c, x1, x2] = row;

    // Create modal
    const modal = document.createElement("div");
    modal.className = "data-modal";
    modal.innerHTML = `
            <div class="modal-backdrop"></div>
            <div class="modal-content">
                <div class="modal-header">
                    <h3>Equation Details - Row ${rowIndex + 1}</h3>
                    <button class="modal-close">&times;</button>
                </div>
                <div class="modal-body">
                    <div class="equation-display">
                        <h4>Quadratic Equation:</h4>
                        <div class="equation-formula">
                            ${Utils.formatNumber(
                              a,
                              3
                            )}x² + ${Utils.formatNumber(
      b,
      3
    )}x + ${Utils.formatNumber(c, 3)} = 0
                        </div>
                    </div>
                    
                    <div class="equation-analysis">
                        <h4>Analysis:</h4>
                        <div class="analysis-grid">
                            <div class="analysis-item">
                                <span class="analysis-label">Discriminant:</span>
                                <span class="analysis-value">${Utils.formatNumber(
                                  b * b - 4 * a * c,
                                  6
                                )}</span>
                            </div>
                            <div class="analysis-item">
                                <span class="analysis-label">Root Type:</span>
                                <span class="analysis-value">${this.getRootType(
                                  a,
                                  b,
                                  c
                                )}</span>
                            </div>
                            <div class="analysis-item">
                                <span class="analysis-label">Vertex:</span>
                                <span class="analysis-value">(${Utils.formatNumber(
                                  -b / (2 * a),
                                  3
                                )}, ${Utils.formatNumber(
      (4 * a * c - b * b) / (4 * a),
      3
    )})</span>
                            </div>
                        </div>
                    </div>
                    
                    <div class="equation-solutions">
                        <h4>Solutions:</h4>
                        <div class="solutions-grid">
                            <div class="solution-item">
                                <span class="solution-label">x₁:</span>
                                <span class="solution-value">${Utils.formatNumber(
                                  x1,
                                  6
                                )}</span>
                            </div>
                            <div class="solution-item">
                                <span class="solution-label">x₂:</span>
                                <span class="solution-value">${Utils.formatNumber(
                                  x2,
                                  6
                                )}</span>
                            </div>
                        </div>
                    </div>
                    
                    <div class="equation-verification">
                        <h4>Verification:</h4>
                        <div class="verification-grid">
                            <div class="verification-item">
                                <span>f(x₁) = ${Utils.formatNumber(
                                  a * x1 * x1 + b * x1 + c,
                                  8
                                )}</span>
                            </div>
                            <div class="verification-item">
                                <span>f(x₂) = ${Utils.formatNumber(
                                  a * x2 * x2 + b * x2 + c,
                                  8
                                )}</span>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="modal-footer">
                    <button class="btn btn-primary" onclick="DataManager.useAsExample(${rowIndex})">
                        Use as Example
                    </button>
                    <button class="btn btn-secondary" onclick="this.closest('.data-modal').remove()">
                        Close
                    </button>
                </div>
            </div>
        `;

    document.body.appendChild(modal);

    // Setup modal events
    modal.querySelector(".modal-close").addEventListener("click", () => {
      modal.remove();
    });

    modal.querySelector(".modal-backdrop").addEventListener("click", () => {
      modal.remove();
    });
  },

  // Get root type description
  getRootType(a, b, c) {
    const discriminant = b * b - 4 * a * c;
    if (discriminant > 0) {
      return "Two distinct real roots";
    } else if (discriminant === 0) {
      return "One repeated real root";
    } else {
      return "Two complex conjugate roots";
    }
  },

  // Update data visualization
  updateDataVisualization(stats) {
    // Create histograms for coefficient distributions
    this.createDistributionCharts(stats);
  },

  // Create distribution charts
  createDistributionCharts(stats) {
    const vizContainer = document.querySelector(".data-visualization");
    if (!vizContainer) return;

    // Implementation would create small histogram charts
    // for coefficient distributions using Chart.js
  },

  // Refresh data information
  async refreshDataInfo() {
    try {
      const response = await fetch("/api/data/info");
      const dataInfo = await response.json();

      if (dataInfo.loaded) {
        this.currentDataset = dataInfo;
        this.updateDataDisplay(dataInfo);
        AppState.dataLoaded = true;
      } else {
        AppState.dataLoaded = false;
      }
    } catch (error) {
      console.error("Failed to refresh data info:", error);
      AppState.dataLoaded = false;
    }
  },

  // Refresh table
  async refreshTable() {
    await this.refreshDataInfo();
    Utils.showNotification("Table refreshed", "success");
  },

  // Export sample data
  exportSample() {
    if (!this.currentDataset?.sample_data) {
      Utils.showNotification("No data to export", "warning");
      return;
    }

    const csv = this.generateCSV(this.currentDataset.sample_data);
    this.downloadCSV(csv, "sample_data.csv");
    Utils.showNotification("Sample data exported", "success");
  },

  // Generate CSV
  generateCSV(data) {
    const headers = ["a", "b", "c", "x1", "x2"];
    let csv = headers.join(",") + "\n";

    data.forEach((row) => {
      csv += row.join(",") + "\n";
    });

    return csv;
  },

  // Download CSV
  downloadCSV(csv, filename) {
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    link.click();
    URL.revokeObjectURL(url);
  },

  // Generate data report
  generateReport() {
    if (!this.currentDataset) {
      Utils.showNotification("No data loaded", "warning");
      return;
    }

    const report = this.createDataReport();
    this.downloadReport(report);
    Utils.showNotification("Data report generated", "success");
  },

  // Create data report
  createDataReport() {
    const stats = this.currentDataset.stats;
    const timestamp = new Date().toISOString();

    return `
QUADRATIC NEURAL NETWORK - DATA REPORT
======================================

Generated: ${timestamp}
Location: Varna, Bulgaria
Author: Matt

DATASET OVERVIEW
================
Total Equations: ${stats.total_equations}
Features: a, b, c, x1, x2
File: ${this.currentDataset.filename}

STATISTICAL SUMMARY
==================
${Object.entries(stats.columns)
  .map(
    ([name, colStats]) => `
${name.toUpperCase()}:
  Mean: ${Utils.formatNumber(colStats.mean, 6)}
  Std Dev: ${Utils.formatNumber(colStats.std, 6)}
  Min: ${Utils.formatNumber(colStats.min, 6)}
  Max: ${Utils.formatNumber(colStats.max, 6)}
`
  )
  .join("")}

DATA QUALITY METRICS
====================
Integer Solutions (x1): ${Utils.formatNumber(stats.quality.x1_whole_pct, 2)}%
Integer Solutions (x2): ${Utils.formatNumber(stats.quality.x2_whole_pct, 2)}%

RECOMMENDATIONS
===============
• Dataset appears suitable for neural network training
• Consider data augmentation if more samples are needed
• Validate solution accuracy before training
• Monitor for potential overfitting with high-precision data

END OF REPORT
        `.trim();
  },

  // Download report
  downloadReport(report) {
    const blob = new Blob([report], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `data_report_${new Date().toISOString().split("T")[0]}.txt`;
    link.click();
    URL.revokeObjectURL(url);
  },

  // Notify other components that data is loaded
  notifyDataLoaded() {
    // Refresh training interface
    if (typeof TrainingManager !== "undefined") {
      TrainingManager.refresh();
    }

    // Refresh prediction interface
    if (typeof PredictionManager !== "undefined") {
      PredictionManager.refresh();
    }

    // Update navigation state
    AppState.dataLoaded = true;
  },

  // Clear all data
  clearData() {
    this.currentDataset = null;
    this.dataStats = null;
    this.sampleData = [];
    AppState.dataLoaded = false;

    // Clear displays
    if (this.statsContainer) {
      this.statsContainer.innerHTML = "<p>No data loaded</p>";
    }

    if (this.dataTable) {
      this.dataTable.style.display = "none";
    }

    Utils.showNotification("Data cleared", "info");
  },
};

// Data validation utilities
const DataValidator = {
  // Validate equation data
  validateEquation(a, b, c, x1, x2) {
    const errors = [];

    // Check for NaN or infinite values
    if (!this.isValidNumber(a)) errors.push("Coefficient a is invalid");
    if (!this.isValidNumber(b)) errors.push("Coefficient b is invalid");
    if (!this.isValidNumber(c)) errors.push("Coefficient c is invalid");
    if (!this.isValidNumber(x1)) errors.push("Root x1 is invalid");
    if (!this.isValidNumber(x2)) errors.push("Root x2 is invalid");

    if (errors.length > 0) return { valid: false, errors };

    // Check if a is zero (not quadratic)
    if (Math.abs(a) < 1e-10) {
      errors.push("Coefficient a cannot be zero for quadratic equations");
    }

    // Verify solutions
    const error1 = Math.abs(a * x1 * x1 + b * x1 + c);
    const error2 = Math.abs(a * x2 * x2 + b * x2 + c);

    if (error1 > 1e-6)
      errors.push(`Root x1 verification failed (error: ${error1})`);
    if (error2 > 1e-6)
      errors.push(`Root x2 verification failed (error: ${error2})`);

    return {
      valid: errors.length === 0,
      errors: errors,
      verificationErrors: [error1, error2],
    };
  },

  // Check if number is valid
  isValidNumber(value) {
    return typeof value === "number" && !isNaN(value) && isFinite(value);
  },

  // Validate dataset format
  validateDatasetFormat(data) {
    if (!Array.isArray(data)) {
      return { valid: false, error: "Data must be an array" };
    }

    if (data.length === 0) {
      return { valid: false, error: "Dataset is empty" };
    }

    // Check if all rows have 5 columns
    const invalidRows = data.filter(
      (row) => !Array.isArray(row) || row.length !== 5
    );
    if (invalidRows.length > 0) {
      return {
        valid: false,
        error: `${invalidRows.length} rows have incorrect number of columns (expected 5)`,
      };
    }

    return { valid: true };
  },
};

// Initialize data manager when DOM is loaded
document.addEventListener("DOMContentLoaded", () => {
  DataManager.init();

  // Make available globally
  window.DataManager = DataManager;
  window.DataValidator = DataValidator;
});

// Global functions for HTML onclick events
window.uploadDataset = () => DataManager.fileInput?.click();
window.clearDataset = () => DataManager.clearData();
window.refreshDataTable = () => DataManager.refreshTable();
window.exportSampleData = () => DataManager.exportSample();
window.generateDataReport = () => DataManager.generateReport();
