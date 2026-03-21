const AnalysisSection = {
  init() {
    // Analysis section initialization
    this.setPlaceholdersVisible(!this.hasCharts());
  },

  async refresh() {
    // Refresh analysis data
    this.setPlaceholdersVisible(!this.hasCharts());
  },

  async generateCharts() {
    try {
      const analysisData = await ApiClient.request(API.performanceAnalysis);
      this.createMetricsChart(analysisData);
      this.createAccuracyChart(analysisData);
      this.createComparisonChart(analysisData);
      this.setPlaceholdersVisible(false);

      // Initialize chart controls after charts are created
      this.initChartControls();
    } catch (error) {
      console.error("Failed to generate analysis:", error);
      this.setPlaceholdersVisible(true);
      Utils.showNotification("Failed to generate analysis charts", "error");
    }
  },
  setPlaceholdersVisible(visible) {
    ["metrics-chart-empty", "accuracy-chart-empty", "comparison-chart-empty"].forEach(
      (id) => {
        const el = document.getElementById(id);
        if (!el) return;
        el.classList.toggle("chart-empty-state--hidden", !visible);
      }
    );
    ["metrics-chart", "accuracy-chart", "comparison-chart"].forEach((id) => {
      const canvas = document.getElementById(id);
      if (!canvas) return;
      canvas.classList.toggle("chart-ready", !visible);
    });
  },
  hasCharts() {
    return Boolean(
      AppState.charts.metrics || AppState.charts.accuracy || AppState.charts.comparison
    );
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
