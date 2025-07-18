/**
 * Quadratic Neural Network Web Application
 * Chart Management & Visualization Library
 *
 * Author: Matt
 * Location: Varna, Bulgaria
 * Date: July 2025
 *
 * Advanced chart creation and management for neural network analysis
 */

// Chart configuration and management
const ChartManager = {
  // Chart instances storage
  charts: {},

  // Chart.js default configuration
  defaultConfig: {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        labels: {
          color: "var(--text-primary)",
          font: {
            family: "SF Pro Display, -apple-system, sans-serif",
            size: 12,
          },
        },
      },
      tooltip: {
        backgroundColor: "rgba(28, 28, 30, 0.95)",
        titleColor: "#FFFFFF",
        bodyColor: "#FFFFFF",
        borderColor: "var(--border-color)",
        borderWidth: 1,
        cornerRadius: 8,
        padding: 12,
        titleFont: {
          family: "SF Pro Display, -apple-system, sans-serif",
          size: 14,
          weight: "600",
        },
        bodyFont: {
          family: "SF Pro Display, -apple-system, sans-serif",
          size: 12,
        },
      },
    },
    scales: {
      x: {
        grid: {
          color: "var(--border-color)",
          lineWidth: 1,
        },
        ticks: {
          color: "var(--text-secondary)",
          font: {
            family: "SF Pro Display, -apple-system, sans-serif",
            size: 11,
          },
        },
      },
      y: {
        grid: {
          color: "var(--border-color)",
          lineWidth: 1,
        },
        ticks: {
          color: "var(--text-secondary)",
          font: {
            family: "SF Pro Display, -apple-system, sans-serif",
            size: 11,
          },
        },
      },
    },
  },

  // Color schemes
  colorSchemes: {
    primary: ["#007AFF", "#5856D6", "#34C759", "#FF9500", "#FF3B30", "#AF52DE"],
    pastel: ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", "#D63031"],
    gradient: [
      "#667eea",
      "#764ba2",
      "#f093fb",
      "#f5576c",
      "#4facfe",
      "#00f2fe",
    ],
  },

  // Initialize chart manager
  init() {
    // Set Chart.js defaults
    if (typeof Chart !== "undefined") {
      Chart.defaults.font.family = "SF Pro Display, -apple-system, sans-serif";
      Chart.defaults.color = "var(--text-primary)";
      Chart.defaults.borderColor = "var(--border-color)";
      Chart.defaults.backgroundColor = "var(--primary-color)";
    }
  },

  // Create performance metrics chart
  createMetricsChart(canvasId, data) {
    const ctx = document.getElementById(canvasId).getContext("2d");

    // Destroy existing chart
    if (this.charts[canvasId]) {
      this.charts[canvasId].destroy();
    }

    const config = {
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
          backgroundColor: this.hexToRgba(data.colors[index], 0.2),
          borderColor: data.colors[index],
          pointBackgroundColor: data.colors[index],
          pointBorderColor: "#fff",
          pointBorderWidth: 2,
          pointRadius: 4,
          pointHoverBackgroundColor: "#fff",
          pointHoverBorderColor: data.colors[index],
          pointHoverRadius: 6,
          borderWidth: 2,
        })),
      },
      options: {
        ...this.defaultConfig,
        scales: {
          r: {
            beginAtZero: true,
            max: 1,
            grid: {
              color: "var(--border-color)",
              lineWidth: 1,
            },
            angleLines: {
              color: "var(--border-color)",
              lineWidth: 1,
            },
            pointLabels: {
              color: "var(--text-primary)",
              font: {
                family: "SF Pro Display, -apple-system, sans-serif",
                size: 12,
                weight: "500",
              },
            },
            ticks: {
              color: "var(--text-secondary)",
              font: {
                family: "SF Pro Display, -apple-system, sans-serif",
                size: 10,
              },
              stepSize: 0.2,
            },
          },
        },
        plugins: {
          ...this.defaultConfig.plugins,
          title: {
            display: true,
            text: "Performance Metrics Comparison",
            color: "var(--text-primary)",
            font: {
              family: "SF Pro Display, -apple-system, sans-serif",
              size: 16,
              weight: "600",
            },
            padding: 20,
          },
        },
      },
    };

    this.charts[canvasId] = new Chart(ctx, config);
    return this.charts[canvasId];
  },

  // Create accuracy bar chart
  createAccuracyChart(canvasId, data) {
    const ctx = document.getElementById(canvasId).getContext("2d");

    if (this.charts[canvasId]) {
      this.charts[canvasId].destroy();
    }

    const config = {
      type: "bar",
      data: {
        labels: data.scenario_names,
        datasets: [
          {
            label: "Accuracy (%)",
            data: data.metrics.accuracy_values,
            backgroundColor: data.colors.map((color) =>
              this.hexToRgba(color, 0.8)
            ),
            borderColor: data.colors,
            borderWidth: 2,
            borderRadius: 8,
            borderSkipped: false,
          },
        ],
      },
      options: {
        ...this.defaultConfig,
        scales: {
          ...this.defaultConfig.scales,
          y: {
            ...this.defaultConfig.scales.y,
            beginAtZero: true,
            max: 100,
            ticks: {
              ...this.defaultConfig.scales.y.ticks,
              callback: function (value) {
                return value + "%";
              },
            },
          },
        },
        plugins: {
          ...this.defaultConfig.plugins,
          title: {
            display: true,
            text: "Accuracy Comparison",
            color: "var(--text-primary)",
            font: {
              family: "SF Pro Display, -apple-system, sans-serif",
              size: 16,
              weight: "600",
            },
            padding: 20,
          },
        },
      },
    };

    this.charts[canvasId] = new Chart(ctx, config);
    return this.charts[canvasId];
  },

  // Create comparison doughnut chart
  createComparisonChart(canvasId, results) {
    const ctx = document.getElementById(canvasId).getContext("2d");

    if (this.charts[canvasId]) {
      this.charts[canvasId].destroy();
    }

    const scenarios = Object.keys(results);
    const r2Scores = scenarios.map((s) => results[s].metrics.r2);
    const colors = scenarios.map((s) => results[s].scenario_info.color);
    const names = scenarios.map((s) => results[s].scenario_info.name);

    const config = {
      type: "doughnut",
      data: {
        labels: names,
        datasets: [
          {
            data: r2Scores,
            backgroundColor: colors.map((c) => this.hexToRgba(c, 0.8)),
            borderColor: colors,
            borderWidth: 2,
            hoverOffset: 4,
          },
        ],
      },
      options: {
        ...this.defaultConfig,
        cutout: "60%",
        plugins: {
          ...this.defaultConfig.plugins,
          title: {
            display: true,
            text: "R² Score Distribution",
            color: "var(--text-primary)",
            font: {
              family: "SF Pro Display, -apple-system, sans-serif",
              size: 16,
              weight: "600",
            },
            padding: 20,
          },
          legend: {
            ...this.defaultConfig.plugins.legend,
            position: "bottom",
            labels: {
              ...this.defaultConfig.plugins.legend.labels,
              padding: 20,
              usePointStyle: true,
              pointStyle: "circle",
            },
          },
        },
      },
    };

    this.charts[canvasId] = new Chart(ctx, config);
    return this.charts[canvasId];
  },

  // Create performance heatmap using Chart.js matrix
  createPerformanceHeatmap(canvasId, data) {
    const ctx = document.getElementById(canvasId).getContext("2d");

    if (this.charts[canvasId]) {
      this.charts[canvasId].destroy();
    }

    // Transform data for heatmap
    const heatmapData = [];
    const metrics = ["R²", "MSE", "MAE", "Accuracy"];

    data.scenarios.forEach((scenario, x) => {
      metrics.forEach((metric, y) => {
        let value;
        switch (metric) {
          case "R²":
            value = data.metrics.r2_scores[x];
            break;
          case "MSE":
            value =
              1 -
              data.metrics.mse_values[x] / Math.max(...data.metrics.mse_values);
            break;
          case "MAE":
            value =
              1 -
              data.metrics.mae_values[x] / Math.max(...data.metrics.mae_values);
            break;
          case "Accuracy":
            value = data.metrics.accuracy_values[x] / 100;
            break;
        }

        heatmapData.push({
          x: x,
          y: y,
          v: value,
          scenario: data.scenario_names[x],
          metric: metric,
        });
      });
    });

    const config = {
      type: "scatter",
      data: {
        datasets: [
          {
            label: "Performance",
            data: heatmapData.map((d) => ({
              x: d.x,
              y: d.y,
              performance: d.v,
              scenario: d.scenario,
              metric: d.metric,
            })),
            backgroundColor: (context) => {
              const value = context.parsed.performance;
              const alpha = Math.max(0.3, value);
              return this.hexToRgba("#34C759", alpha);
            },
            borderColor: "#34C759",
            pointRadius: 20,
            pointHoverRadius: 25,
          },
        ],
      },
      options: {
        ...this.defaultConfig,
        scales: {
          x: {
            type: "linear",
            position: "bottom",
            min: -0.5,
            max: data.scenarios.length - 0.5,
            ticks: {
              stepSize: 1,
              callback: function (value) {
                return data.scenario_names[value] || "";
              },
              color: "var(--text-secondary)",
            },
            grid: {
              display: false,
            },
          },
          y: {
            type: "linear",
            min: -0.5,
            max: metrics.length - 0.5,
            ticks: {
              stepSize: 1,
              callback: function (value) {
                return metrics[value] || "";
              },
              color: "var(--text-secondary)",
            },
            grid: {
              display: false,
            },
          },
        },
        plugins: {
          ...this.defaultConfig.plugins,
          title: {
            display: true,
            text: "Performance Heatmap",
            color: "var(--text-primary)",
            font: {
              family: "SF Pro Display, -apple-system, sans-serif",
              size: 16,
              weight: "600",
            },
            padding: 20,
          },
          legend: {
            display: false,
          },
          tooltip: {
            ...this.defaultConfig.plugins.tooltip,
            callbacks: {
              title: function (context) {
                const point = context[0];
                return `${point.raw.scenario} - ${point.raw.metric}`;
              },
              label: function (context) {
                const value = context.raw.performance;
                return `Performance: ${(value * 100).toFixed(1)}%`;
              },
            },
          },
        },
      },
    };

    this.charts[canvasId] = new Chart(ctx, config);
    return this.charts[canvasId];
  },

  // Create training progress chart
  createTrainingProgressChart(canvasId, trainingData) {
    const ctx = document.getElementById(canvasId).getContext("2d");

    if (this.charts[canvasId]) {
      this.charts[canvasId].destroy();
    }

    const config = {
      type: "line",
      data: {
        labels: trainingData.epochs,
        datasets: [
          {
            label: "Training Loss",
            data: trainingData.train_loss,
            borderColor: "#FF3B30",
            backgroundColor: this.hexToRgba("#FF3B30", 0.1),
            borderWidth: 2,
            fill: true,
            tension: 0.4,
            pointRadius: 0,
            pointHoverRadius: 4,
          },
          {
            label: "Validation Loss",
            data: trainingData.val_loss,
            borderColor: "#007AFF",
            backgroundColor: this.hexToRgba("#007AFF", 0.1),
            borderWidth: 2,
            fill: true,
            tension: 0.4,
            pointRadius: 0,
            pointHoverRadius: 4,
          },
        ],
      },
      options: {
        ...this.defaultConfig,
        scales: {
          ...this.defaultConfig.scales,
          y: {
            ...this.defaultConfig.scales.y,
            beginAtZero: true,
            type: "logarithmic",
          },
        },
        plugins: {
          ...this.defaultConfig.plugins,
          title: {
            display: true,
            text: "Training Progress",
            color: "var(--text-primary)",
            font: {
              family: "SF Pro Display, -apple-system, sans-serif",
              size: 16,
              weight: "600",
            },
            padding: 20,
          },
        },
        interaction: {
          intersect: false,
          mode: "index",
        },
      },
    };

    this.charts[canvasId] = new Chart(ctx, config);
    return this.charts[canvasId];
  },

  // Create error distribution chart
  createErrorDistributionChart(canvasId, data) {
    const ctx = document.getElementById(canvasId).getContext("2d");

    if (this.charts[canvasId]) {
      this.charts[canvasId].destroy();
    }

    const config = {
      type: "bar",
      data: {
        labels: data.scenario_names,
        datasets: [
          {
            label: "MSE",
            data: data.metrics.mse_values,
            backgroundColor: this.hexToRgba("#FF3B30", 0.7),
            borderColor: "#FF3B30",
            borderWidth: 2,
            borderRadius: 4,
            yAxisID: "y",
          },
          {
            label: "MAE",
            data: data.metrics.mae_values,
            backgroundColor: this.hexToRgba("#FF9500", 0.7),
            borderColor: "#FF9500",
            borderWidth: 2,
            borderRadius: 4,
            yAxisID: "y1",
          },
        ],
      },
      options: {
        ...this.defaultConfig,
        scales: {
          x: this.defaultConfig.scales.x,
          y: {
            ...this.defaultConfig.scales.y,
            type: "linear",
            display: true,
            position: "left",
            title: {
              display: true,
              text: "MSE",
              color: "var(--text-primary)",
            },
          },
          y1: {
            ...this.defaultConfig.scales.y,
            type: "linear",
            display: true,
            position: "right",
            title: {
              display: true,
              text: "MAE",
              color: "var(--text-primary)",
            },
            grid: {
              drawOnChartArea: false,
            },
          },
        },
        plugins: {
          ...this.defaultConfig.plugins,
          title: {
            display: true,
            text: "Error Distribution",
            color: "var(--text-primary)",
            font: {
              family: "SF Pro Display, -apple-system, sans-serif",
              size: 16,
              weight: "600",
            },
            padding: 20,
          },
        },
      },
    };

    this.charts[canvasId] = new Chart(ctx, config);
    return this.charts[canvasId];
  },

  // Create prediction confidence chart
  createConfidenceChart(canvasId, predictionData) {
    const ctx = document.getElementById(canvasId).getContext("2d");

    if (this.charts[canvasId]) {
      this.charts[canvasId].destroy();
    }

    const config = {
      type: "scatter",
      data: {
        datasets: [
          {
            label: "Predictions",
            data: predictionData.map((pred, i) => ({
              x: pred.actual,
              y: pred.predicted,
              confidence: pred.confidence,
            })),
            backgroundColor: (context) => {
              const confidence = context.raw.confidence;
              if (confidence > 0.8) return this.hexToRgba("#34C759", 0.8);
              if (confidence > 0.6) return this.hexToRgba("#FF9500", 0.8);
              return this.hexToRgba("#FF3B30", 0.8);
            },
            borderColor: (context) => {
              const confidence = context.raw.confidence;
              if (confidence > 0.8) return "#34C759";
              if (confidence > 0.6) return "#FF9500";
              return "#FF3B30";
            },
            pointRadius: 6,
            pointHoverRadius: 8,
          },
          {
            label: "Perfect Prediction",
            data: [
              {
                x: Math.min(...predictionData.map((p) => p.actual)),
                y: Math.min(...predictionData.map((p) => p.actual)),
              },
              {
                x: Math.max(...predictionData.map((p) => p.actual)),
                y: Math.max(...predictionData.map((p) => p.actual)),
              },
            ],
            type: "line",
            borderColor: "var(--text-secondary)",
            borderWidth: 2,
            borderDash: [5, 5],
            pointRadius: 0,
            pointHoverRadius: 0,
            fill: false,
          },
        ],
      },
      options: {
        ...this.defaultConfig,
        scales: {
          x: {
            ...this.defaultConfig.scales.x,
            title: {
              display: true,
              text: "Actual Values",
              color: "var(--text-primary)",
            },
          },
          y: {
            ...this.defaultConfig.scales.y,
            title: {
              display: true,
              text: "Predicted Values",
              color: "var(--text-primary)",
            },
          },
        },
        plugins: {
          ...this.defaultConfig.plugins,
          title: {
            display: true,
            text: "Prediction Accuracy vs Confidence",
            color: "var(--text-primary)",
            font: {
              family: "SF Pro Display, -apple-system, sans-serif",
              size: 16,
              weight: "600",
            },
            padding: 20,
          },
          tooltip: {
            ...this.defaultConfig.plugins.tooltip,
            callbacks: {
              title: function (context) {
                return "Prediction Details";
              },
              label: function (context) {
                const point = context.raw;
                return [
                  `Actual: ${point.x.toFixed(4)}`,
                  `Predicted: ${point.y.toFixed(4)}`,
                  `Confidence: ${(point.confidence * 100).toFixed(1)}%`,
                ];
              },
            },
          },
        },
      },
    };

    this.charts[canvasId] = new Chart(ctx, config);
    return this.charts[canvasId];
  },

  // Utility function to convert hex to rgba
  hexToRgba(hex, alpha) {
    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    if (result) {
      const r = parseInt(result[1], 16);
      const g = parseInt(result[2], 16);
      const b = parseInt(result[3], 16);
      return `rgba(${r}, ${g}, ${b}, ${alpha})`;
    }
    return hex;
  },

  // Destroy specific chart
  destroyChart(canvasId) {
    if (this.charts[canvasId]) {
      this.charts[canvasId].destroy();
      delete this.charts[canvasId];
    }
  },

  // Destroy all charts
  destroyAllCharts() {
    Object.keys(this.charts).forEach((canvasId) => {
      this.destroyChart(canvasId);
    });
  },

  // Update chart data
  updateChart(canvasId, newData) {
    if (this.charts[canvasId]) {
      this.charts[canvasId].data = newData;
      this.charts[canvasId].update("active");
    }
  },

  // Animate chart
  animateChart(canvasId, duration = 750) {
    if (this.charts[canvasId]) {
      this.charts[canvasId].update("active");
    }
  },

  // Export chart as image
  exportChart(canvasId, filename = "chart.png") {
    if (this.charts[canvasId]) {
      const url = this.charts[canvasId].toBase64Image();
      const link = document.createElement("a");
      link.download = filename;
      link.href = url;
      link.click();
    }
  },

  // Get chart data for export
  getChartData(canvasId) {
    if (this.charts[canvasId]) {
      return this.charts[canvasId].data;
    }
    return null;
  },

  // Resize chart
  resizeChart(canvasId) {
    if (this.charts[canvasId]) {
      this.charts[canvasId].resize();
    }
  },

  // Set theme for all charts
  setTheme(theme = "light") {
    const isDark = theme === "dark";

    Object.keys(this.charts).forEach((canvasId) => {
      const chart = this.charts[canvasId];
      if (chart) {
        // Update colors based on theme
        if (chart.options.scales) {
          Object.keys(chart.options.scales).forEach((scaleKey) => {
            const scale = chart.options.scales[scaleKey];
            if (scale.grid) {
              scale.grid.color = isDark ? "#38383A" : "#E5E5EA";
            }
            if (scale.ticks) {
              scale.ticks.color = isDark ? "#8E8E93" : "#6D6D80";
            }
          });
        }

        if (chart.options.plugins) {
          if (chart.options.plugins.legend) {
            chart.options.plugins.legend.labels.color = isDark
              ? "#FFFFFF"
              : "#1D1D1F";
          }
          if (chart.options.plugins.title) {
            chart.options.plugins.title.color = isDark ? "#FFFFFF" : "#1D1D1F";
          }
        }

        chart.update("none");
      }
    });
  },
};

// Chart animation utilities
const ChartAnimations = {
  // Fade in animation
  fadeIn: {
    onComplete: function (animation) {
      animation.chart.canvas.style.opacity = "1";
    },
    onProgress: function (animation) {
      animation.chart.canvas.style.opacity =
        animation.currentStep / animation.numSteps;
    },
  },

  // Slide up animation
  slideUp: {
    x: {
      type: "number",
      properties: ["x", "controlPointPreviousX", "controlPointNextX"],
      from: function (ctx) {
        return ctx.chart.chartArea.bottom;
      },
    },
    y: {
      type: "number",
      properties: ["y", "controlPointPreviousY", "controlPointNextY"],
      from: function (ctx) {
        return ctx.chart.chartArea.bottom;
      },
    },
  },

  // Scale animation
  scale: {
    x: {
      type: "number",
      properties: ["x", "controlPointPreviousX", "controlPointNextX"],
      from: function (ctx) {
        return (
          ctx.chart.chartArea.left +
          (ctx.chart.chartArea.right - ctx.chart.chartArea.left) / 2
        );
      },
    },
    y: {
      type: "number",
      properties: ["y", "controlPointPreviousY", "controlPointNextY"],
      from: function (ctx) {
        return (
          ctx.chart.chartArea.top +
          (ctx.chart.chartArea.bottom - ctx.chart.chartArea.top) / 2
        );
      },
    },
  },
};

// Chart export utilities
const ChartExporter = {
  // Export chart as PNG
  exportAsPNG(canvasId, filename = "chart.png", quality = 1.0) {
    const chart = ChartManager.charts[canvasId];
    if (!chart) return;

    const canvas = chart.canvas;
    const url = canvas.toDataURL("image/png", quality);
    this.downloadImage(url, filename);
  },

  // Export chart as JPEG
  exportAsJPEG(canvasId, filename = "chart.jpg", quality = 0.9) {
    const chart = ChartManager.charts[canvasId];
    if (!chart) return;

    const canvas = chart.canvas;
    const url = canvas.toDataURL("image/jpeg", quality);
    this.downloadImage(url, filename);
  },

  // Export chart data as JSON
  exportAsJSON(canvasId, filename = "chart-data.json") {
    const chart = ChartManager.charts[canvasId];
    if (!chart) return;

    const data = JSON.stringify(chart.data, null, 2);
    this.downloadText(data, filename, "application/json");
  },

  // Export chart data as CSV
  exportAsCSV(canvasId, filename = "chart-data.csv") {
    const chart = ChartManager.charts[canvasId];
    if (!chart) return;

    const data = chart.data;
    let csv = "";

    // Add headers
    csv += "Label," + data.datasets.map((d) => d.label).join(",") + "\n";

    // Add data rows
    data.labels.forEach((label, i) => {
      csv +=
        label +
        "," +
        data.datasets.map((d) => d.data[i] || "").join(",") +
        "\n";
    });

    this.downloadText(csv, filename, "text/csv");
  },

  // Download image helper
  downloadImage(url, filename) {
    const link = document.createElement("a");
    link.download = filename;
    link.href = url;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  },

  // Download text helper
  downloadText(text, filename, mimeType) {
    const blob = new Blob([text], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.download = filename;
    link.href = url;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  },
};

// Initialize chart manager when DOM is loaded
document.addEventListener("DOMContentLoaded", () => {
  ChartManager.init();
});

// Global chart functions for use in HTML
window.ChartManager = ChartManager;
window.ChartAnimations = ChartAnimations;
window.ChartExporter = ChartExporter;
