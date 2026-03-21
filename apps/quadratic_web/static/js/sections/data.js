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
