const ComparisonSection = {
  init() {
    // Comparison section initialization
    this.setPlaceholdersVisible(!this.hasComparisonRendered());
  },

  async refresh() {
    // Refresh comparison data
    this.setPlaceholdersVisible(!this.hasComparisonRendered());
  },

  getComparableResults(results) {
    const filtered = Object.fromEntries(
      Object.entries(results).filter(([, result]) => {
        return result?.scenario_info?.comparison_enabled !== false;
      })
    );

    return Object.keys(filtered).length > 0 ? filtered : results;
  },

  async generateComparison() {
    try {
      const results = this.getComparableResults(await ApiClient.request(API.results));
      if (Object.keys(results).length < 2) {
        Utils.showNotification(
          "Need at least 2 comparable models for comparison",
          "warning"
        );
        return;
      }

      this.createComparisonChart(results);
      this.generateComparisonReport(results);
      this.generateInsights(results);
      this.setPlaceholdersVisible(false);
    } catch (error) {
      console.error("Failed to generate comparison:", error);
      this.setPlaceholdersVisible(true);
      Utils.showNotification("Failed to generate comparison", "error");
    }
  },
  setPlaceholdersVisible(visible) {
    ["comparison-performance-empty", "comparison-insights-empty"].forEach(
      (id) => {
        const el = document.getElementById(id);
        if (!el) return;
        el.classList.toggle("chart-empty-state--hidden", !visible);
      }
    );
    const canvas = document.getElementById("comparison-performance-chart");
    if (canvas) {
      canvas.classList.toggle("chart-ready", !visible);
    }
  },
  hasComparisonRendered() {
    return Boolean(AppState.charts.comparisonOverview);
  },

  createComparisonChart(results) {
    results = this.getComparableResults(results);
    const ctx = document
      .getElementById("comparison-performance-chart")
      .getContext("2d");
    const chartTextPrimary = "#edf5ff";
    const chartTextSecondary = "#b6c9db";
    const chartGrid = "rgba(182, 201, 219, 0.18)";

    if (AppState.charts.comparisonOverview) {
      AppState.charts.comparisonOverview.destroy();
    }

    const scenarios = Object.keys(results);

    // Prepare data for grouped bar chart with multiple metrics
    const metrics = ["r2", "accuracy_10pct", "mae_inv", "mse_inv"];
    const metricLabels = [
      "R² Score",
      "Accuracy (10%)",
      "MAE (Inverted)",
      "MSE (Inverted)",
    ];
    const colors = ["#007aff", "#34c759", "#ff9500", "#ff3b30"];

    const datasets = metrics.map((metric, index) => ({
      label: metricLabels[index],
      data: scenarios.map((scenario) => {
        const result = results[scenario];
        // Process each metric appropriately
        if (metric === "mae_inv") {
          return result.metrics.mae ? (1 / (1 + result.metrics.mae)) * 100 : 0;
        }
        if (metric === "mse_inv") {
          return result.metrics.mse ? (1 / (1 + result.metrics.mse)) * 100 : 0;
        }
        if (metric === "accuracy_10pct") {
          return result.metrics.accuracy_10pct || 0;
        }
        if (metric === "r2") {
          return (result.metrics.r2 || 0) * 100; // Convert to percentage
        }
        return 0;
      }),
      backgroundColor: colors[index] + "90", // Add transparency
      borderColor: colors[index],
      borderWidth: 2,
      borderRadius: 8,
      borderSkipped: false,
    }));

    const labels = scenarios.map(
      (scenario) => results[scenario].scenario_info.name
    );

    AppState.charts.comparisonOverview = new Chart(ctx, {
      type: "bar",
      data: {
        labels: labels,
        datasets: datasets,
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          title: {
            display: true,
            text: "🏆 Neural Network Model Performance Comparison",
            font: {
              size: 18,
              weight: "bold",
            },
            color: chartTextPrimary,
            padding: 20,
          },
          legend: {
            display: true,
            position: "top",
            labels: {
              usePointStyle: true,
              pointStyle: "circle",
              padding: 20,
              color: chartTextPrimary,
              font: {
                size: 12,
              },
            },
          },
          tooltip: {
            mode: "index",
            intersect: false,
            backgroundColor: "rgba(11, 21, 34, 0.96)",
            titleColor: chartTextPrimary,
            bodyColor: chartTextSecondary,
            borderColor: "rgba(129, 174, 209, 0.25)",
            borderWidth: 1,
            cornerRadius: 12,
            displayColors: true,
            callbacks: {
              label: function (context) {
                const label = context.dataset.label;
                const value = context.parsed.y;

                if (label === "R² Score") {
                  return `${label}: ${value.toFixed(1)}%`;
                } else if (label === "Accuracy (10%)") {
                  return `${label}: ${value.toFixed(1)}%`;
                } else {
                  return `${label}: ${value.toFixed(1)}`;
                }
              },
            },
          },
        },
        scales: {
          x: {
            display: true,
            title: {
              display: true,
              text: "Neural Network Scenarios",
              font: {
                size: 14,
                weight: "bold",
              },
              color: chartTextPrimary,
            },
            grid: {
              display: false,
            },
            ticks: {
              color: chartTextSecondary,
              font: {
                size: 11,
              },
            },
          },
          y: {
            display: true,
            beginAtZero: true,
            max: 100,
            title: {
              display: true,
              text: "Performance Score (%)",
              font: {
                size: 14,
                weight: "bold",
              },
              color: chartTextPrimary,
            },
            grid: {
              color: chartGrid,
              lineWidth: 1,
            },
            ticks: {
              color: chartTextSecondary,
              font: {
                size: 11,
              },
              callback: function (value) {
                return value + "%";
              },
            },
          },
        },
        interaction: {
          mode: "index",
          intersect: false,
        },
        animation: {
          duration: 1500,
          easing: "easeInOutCubic",
        },
      },
    });
  },

  generateComparisonReport(results) {
    results = this.getComparableResults(results);
    const container = document.getElementById("model-rankings");

    // Sort scenarios by composite score (weighted average of all metrics)
    const sortedScenarios = Object.entries(results).sort(([, a], [, b]) => {
      const scoreA = this.calculateCompositeScore(a.metrics);
      const scoreB = this.calculateCompositeScore(b.metrics);
      return scoreB - scoreA;
    });

    let html = `
      <div class="comparison-header">
        <h3>🏅 Model Performance Rankings</h3>
        <p class="section-description">
          Models ranked by composite performance score across all evaluation metrics
        </p>
      </div>
      <div class="model-rankings-grid">
    `;

    sortedScenarios.forEach(([key, result], index) => {
      const medals = ["🥇", "🥈", "🥉", "🏅"];
      const medal = medals[index] || "📊";
      const compositeScore = this.calculateCompositeScore(result.metrics);
      const performance = this.getPerformanceRating(compositeScore);
      const performanceColor = this.getPerformanceColor(compositeScore);

      html += `
        <div class="model-card" style="border-left-color: ${
          result.scenario_info.color
        }">
          <div class="model-header">
            <div class="model-rank">
              <span class="medal">${medal}</span>
              <span class="rank-number">#${index + 1}</span>
            </div>
            <div class="model-info">
              <h4 class="model-name">${result.scenario_info.name}</h4>
              <p class="model-description">${
                result.scenario_info.description
              }</p>
            </div>
            <div class="model-score">
              <div class="composite-score" style="color: ${performanceColor}">
                ${compositeScore.toFixed(1)}%
              </div>
              <div class="performance-label">${performance}</div>
            </div>
          </div>
          
          <div class="metrics-grid">
            <div class="metric-item">
              <span class="metric-label">R² Score</span>
              <span class="metric-value">${Utils.formatNumber(
                result.metrics.r2 * 100,
                1
              )}%</span>
              <div class="metric-bar">
                <div class="metric-fill" style="width: ${
                  result.metrics.r2 * 100
                }%; background: #007aff"></div>
              </div>
            </div>
            
            <div class="metric-item">
              <span class="metric-label">Accuracy</span>
              <span class="metric-value">${Utils.formatNumber(
                result.metrics.accuracy_10pct,
                1
              )}%</span>
              <div class="metric-bar">
                <div class="metric-fill" style="width: ${
                  result.metrics.accuracy_10pct
                }%; background: #34c759"></div>
              </div>
            </div>
            
            <div class="metric-item">
              <span class="metric-label">MSE</span>
              <span class="metric-value">${Utils.formatNumber(
                result.metrics.mse,
                6
              )}</span>
              <div class="metric-bar">
                <div class="metric-fill" style="width: ${
                  (1 / (1 + result.metrics.mse)) * 100
                }%; background: #ff3b30"></div>
              </div>
            </div>
            
            <div class="metric-item">
              <span class="metric-label">MAE</span>
              <span class="metric-value">${Utils.formatNumber(
                result.metrics.mae,
                6
              )}</span>
              <div class="metric-bar">
                <div class="metric-fill" style="width: ${
                  (1 / (1 + result.metrics.mae)) * 100
                }%; background: #ff9500"></div>
              </div>
            </div>
          </div>
          
          <div class="model-actions">
            <div class="model-indicator" style="background: ${
              result.scenario_info.color
            }"></div>
          </div>
        </div>
      `;
    });

    html += `</div>`;
    container.innerHTML = html;
  },

  generateInsights(results) {
    results = this.getComparableResults(results);
    const scenarios = Object.keys(results);

    // Find best performing model
    let bestModel = "";
    let bestScore = -1;

    scenarios.forEach((scenario) => {
      const score = this.calculateCompositeScore(results[scenario].metrics);
      if (score > bestScore) {
        bestScore = score;
        bestModel = results[scenario].scenario_info.name;
      }
    });

    // Calculate averages
    const avgR2 =
      (scenarios.reduce((sum, s) => sum + results[s].metrics.r2, 0) /
        scenarios.length) *
      100;
    const avgAccuracy =
      scenarios.reduce((sum, s) => sum + results[s].metrics.accuracy_10pct, 0) /
      scenarios.length;

    // Count production-ready models (accuracy > 85%)
    const productionReady = scenarios.filter(
      (s) => results[s].metrics.accuracy_10pct > 85
    ).length;

    const insightsHtml = `
      <div class="insights-section">
        <div class="insight-cards">
          <div class="insight-card best-performer">
            <div class="insight-icon">🏆</div>
            <div class="insight-content">
              <h4>Top Performer</h4>
              <p><strong>${bestModel}</strong> achieves the highest composite score of <strong>${bestScore.toFixed(
      1
    )}%</strong></p>
              <div class="insight-recommendation">Recommended for production deployment</div>
            </div>
          </div>
          
          <div class="insight-card performance-overview">
            <div class="insight-icon">📊</div>
            <div class="insight-content">
              <h4>Overall Performance</h4>
              <p>Average R² Score: <strong>${avgR2.toFixed(1)}%</strong></p>
              <p>Average Accuracy: <strong>${avgAccuracy.toFixed(
                1
              )}%</strong></p>
              <div class="performance-summary">
                ${productionReady}/${
      scenarios.length
    } models are production-ready (85%+ accuracy)
              </div>
            </div>
          </div>
          
          <div class="insight-card recommendations">
            <div class="insight-icon">💡</div>
            <div class="insight-content">
              <h4>Key Recommendations</h4>
              <ul class="recommendation-list">
                <li>Models with R² > 80% show excellent variance explanation</li>
                <li>Accuracy above 85% indicates production readiness</li>
                <li>Lower error metrics (MSE, MAE) suggest better precision</li>
                <li>Consider ensemble methods for critical applications</li>
              </ul>
            </div>
          </div>
        </div>
        
        <div class="metric-explanations comparison-metric-explanations">
          <h4>📖 Understanding the Metrics</h4>
          <div class="explanation-grid">
            <div class="explanation-card">
              <div class="metric-badge" style="background: #007aff">R²</div>
              <div class="explanation-content">
                <h5>R² Score (Coefficient of Determination)</h5>
                <p>Measures how well the model explains variance in target data</p>
                <div class="scale-indicators">
                  <span class="scale excellent">90%+ Excellent</span>
                  <span class="scale good">80-90% Very Good</span>
                  <span class="scale fair">60-80% Acceptable</span>
                  <span class="scale poor">&lt;60% Poor</span>
                </div>
              </div>
            </div>
            
            <div class="explanation-card">
              <div class="metric-badge" style="background: #34c759">ACC</div>
              <div class="explanation-content">
                <h5>Accuracy (10% Tolerance)</h5>
                <p>Percentage of predictions within 10% of actual values</p>
                <div class="scale-indicators">
                  <span class="scale excellent">85%+ Production</span>
                  <span class="scale good">70-85% Good</span>
                  <span class="scale fair">50-70% Fair</span>
                  <span class="scale poor">&lt;50% Poor</span>
                </div>
              </div>
            </div>
            
            <div class="explanation-card">
              <div class="metric-badge" style="background: #ff9500">MAE</div>
              <div class="explanation-content">
                <h5>Mean Absolute Error</h5>
                <p>Average absolute difference between predictions and actual values</p>
                <div class="error-note">Lower values indicate better performance</div>
              </div>
            </div>
            
            <div class="explanation-card">
              <div class="metric-badge" style="background: #ff3b30">MSE</div>
              <div class="explanation-content">
                <h5>Mean Squared Error</h5>
                <p>Average squared difference - more sensitive to outliers</p>
                <div class="error-note">Penalizes larger errors more heavily than MAE</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    `;

    const insightsContainer =
      document.querySelector(".comparison-insights") ||
      document.getElementById("comparison-insights");
    if (insightsContainer) {
      insightsContainer.innerHTML = insightsHtml;
    }
  },

  calculateCompositeScore(metrics) {
    // Weighted composite score calculation
    const r2Weight = 0.4;
    const accuracyWeight = 0.3;
    const mseWeight = 0.15;
    const maeWeight = 0.15;

    const r2Score = (metrics.r2 || 0) * 100;
    const accuracyScore = metrics.accuracy_10pct || 0;
    const mseScore = (1 / (1 + metrics.mse)) * 100; // Inverted
    const maeScore = (1 / (1 + metrics.mae)) * 100; // Inverted

    return (
      r2Score * r2Weight +
      accuracyScore * accuracyWeight +
      mseScore * mseWeight +
      maeScore * maeWeight
    );
  },

  getPerformanceRating(score) {
    if (score >= 85) return "Excellent";
    if (score >= 70) return "Very Good";
    if (score >= 55) return "Good";
    if (score >= 40) return "Fair";
    return "Needs Improvement";
  },

  getPerformanceColor(score) {
    if (score >= 85) return "#34c759";
    if (score >= 70) return "#007aff";
    if (score >= 55) return "#ff9500";
    if (score >= 40) return "#ff9500";
    return "#ff3b30";
  },
};

