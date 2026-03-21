/**
 * Quadratic Neural Network Web Application
 * Frontend JavaScript Application
 *
 * Beautiful Apple-like web interface for quadratic neural network analysis
 */

// Application state management
const AppState = {
  currentSection: "dashboard",
  isTraining: false,
  dataLoaded: false,
  scenarios: {},
  results: {},
  savedModels: [],
  trainingInterval: null,
  autoLoadDataset: null,
  charts: {
    metrics: null,
    accuracy: null,
    comparison: null,
    comparisonOverview: null,
  },
};

// API endpoints
const API = {
  health: "/api/health",
  scenarios: "/api/scenarios",
  uploadData: "/api/data/upload",
  dataInfo: "/api/data/info",
  randomData: "/api/data/random",
  clearData: "/api/data/clear",
  startTraining: "/api/training/start",
  trainingStatus: "/api/training/status",
  stopTraining: "/api/training/stop",
  predict: "/api/predict",
  results: "/api/results",
  performanceAnalysis: "/api/analysis/performance",
  modelsList: "/api/models/list",
  modelsSave: "/api/models/save",
  modelsLoad: "/api/models/load",
  modelsDelete: "/api/models/delete",
  modelsInfo: "/api/models/info",
};

// Utility functions
const Utils = {
  formatNumber: (num, decimals = 6) => {
    if (typeof num !== "number" || isNaN(num)) return "0.000000";
    if (Math.abs(num) < 1e-10) return "0.000000";
    return num.toFixed(decimals);
  },

  formatPercentage: (num, decimals = 1) => {
    return `${Utils.formatNumber(num, decimals)}%`;
  },

  getConfidenceLevel: (confidence) => {
    if (confidence > 0.8) return "🟢 High";
    if (confidence > 0.6) return "🟡 Medium";
    return "🔴 Low";
  },

  showNotification: (message, type = "info") => {
    const notification = document.createElement("div");
    notification.className = `notification notification-${type}`;
    notification.innerHTML = `
            <div class="notification-content">
                <i class="fas fa-${
                  type === "success"
                    ? "check-circle"
                    : type === "error"
                    ? "exclamation-circle"
                    : type === "warning"
                    ? "exclamation-triangle"
                    : "info-circle"
                }"></i>
                <span>${message}</span>
            </div>
        `;

    // Add styles if not already present
    if (!document.querySelector("#notification-styles")) {
      const styles = document.createElement("style");
      styles.id = "notification-styles";
      styles.textContent = `
                .notification {
                    position: fixed;
                    top: 20px;
                    right: 20px;
                    background: var(--surface-color);
                    border-radius: var(--radius-medium);
                    padding: 16px;
                    box-shadow: var(--shadow-heavy);
                    border: 1px solid var(--border-color);
                    z-index: 1000;
                    max-width: 400px;
                    animation: slideIn 0.3s ease;
                }
                .notification-content {
                    display: flex;
                    align-items: center;
                    gap: 12px;
                }
                .notification-success { border-left: 4px solid var(--success-color); }
                .notification-error { border-left: 4px solid var(--error-color); }
                .notification-warning { border-left: 4px solid var(--warning-color); }
                .notification-info { border-left: 4px solid var(--primary-color); }
                @keyframes slideIn {
                    from { transform: translateX(100%); opacity: 0; }
                    to { transform: translateX(0); opacity: 1; }
                }
            `;
      document.head.appendChild(styles);
    }

    document.body.appendChild(notification);

    // Auto remove after 5 seconds
    setTimeout(() => {
      notification.style.animation = "slideIn 0.3s ease reverse";
      setTimeout(() => {
        if (notification.parentNode) {
          notification.parentNode.removeChild(notification);
        }
      }, 300);
    }, 5000);
  },

  debounce: (func, wait) => {
    let timeout;
    return function executedFunction(...args) {
      const later = () => {
        clearTimeout(timeout);
        func(...args);
      };
      clearTimeout(timeout);
      timeout = setTimeout(later, wait);
    };
  },

  // NEW: Format quadratic equation
  formatQuadraticEquation: (a, b, c) => {
    let equation = "";

    // Handle coefficient 'a'
    if (a === 1) {
      equation += "x²";
    } else if (a === -1) {
      equation += "-x²";
    } else {
      equation += `${Utils.formatNumber(a, 3)}x²`;
    }

    // Handle coefficient 'b'
    if (b > 0) {
      if (b === 1) {
        equation += " + x";
      } else {
        equation += ` + ${Utils.formatNumber(b, 3)}x`;
      }
    } else if (b < 0) {
      if (b === -1) {
        equation += " - x";
      } else {
        equation += ` - ${Utils.formatNumber(Math.abs(b), 3)}x`;
      }
    }

    // Handle coefficient 'c'
    if (c > 0) {
      equation += ` + ${Utils.formatNumber(c, 3)}`;
    } else if (c < 0) {
      equation += ` - ${Utils.formatNumber(Math.abs(c), 3)}`;
    }

    equation += " = 0";
    return equation;
  },

  // NEW: Calculate actual quadratic solutions
  calculateActualSolutions: (a, b, c) => {
    if (Math.abs(a) < 1e-10) {
      if (Math.abs(b) < 1e-10) {
        return { type: "invalid", message: "Not a valid equation" };
      } else {
        const root = -c / b;
        return {
          type: "linear",
          roots: [root],
          message: "Linear equation (not quadratic)",
        };
      }
    }

    const discriminant = b * b - 4 * a * c;

    if (discriminant < 0) {
      return { type: "complex", message: "Complex roots (no real solutions)" };
    } else if (discriminant === 0) {
      const root = -b / (2 * a);
      return {
        type: "repeated",
        roots: [root],
        message: "One repeated real root",
      };
    } else {
      const sqrtDiscriminant = Math.sqrt(discriminant);
      const root1 = (-b + sqrtDiscriminant) / (2 * a);
      const root2 = (-b - sqrtDiscriminant) / (2 * a);
      return {
        type: "distinct",
        roots: [root1, root2],
        message: "Two distinct real roots",
      };
    }
  },

  // NEW: Calculate solution error
  calculateSolutionError: (predicted, actual) => {
    if (!actual || actual.type === "complex" || actual.type === "invalid") {
      return null;
    }

    const actualRoots = actual.roots;
    if (actualRoots.length === 1) {
      // Single root case
      const error1 = Math.abs(predicted[0] - actualRoots[0]);
      const error2 = Math.abs(predicted[1] - actualRoots[0]);
      return {
        x1_error: error1,
        x2_error: error2,
        avg_error: (error1 + error2) / 2,
        type: "single_root",
      };
    } else {
      // Two roots case - match closest pairs
      const error1 =
        Math.abs(predicted[0] - actualRoots[0]) +
        Math.abs(predicted[1] - actualRoots[1]);
      const error2 =
        Math.abs(predicted[0] - actualRoots[1]) +
        Math.abs(predicted[1] - actualRoots[0]);

      if (error1 <= error2) {
        return {
          x1_error: Math.abs(predicted[0] - actualRoots[0]),
          x2_error: Math.abs(predicted[1] - actualRoots[1]),
          avg_error:
            (Math.abs(predicted[0] - actualRoots[0]) +
              Math.abs(predicted[1] - actualRoots[1])) /
            2,
          type: "two_roots",
        };
      } else {
        return {
          x1_error: Math.abs(predicted[0] - actualRoots[1]),
          x2_error: Math.abs(predicted[1] - actualRoots[0]),
          avg_error:
            (Math.abs(predicted[0] - actualRoots[1]) +
              Math.abs(predicted[1] - actualRoots[0])) /
            2,
          type: "two_roots",
        };
      }
    }
  },
};

// Auto-load dataset from URL parameter
async function checkAndLoadDataset() {
  const urlParams = new URLSearchParams(window.location.search);
  const loadDataset = urlParams.get("load_dataset");

  if (loadDataset) {
    AppState.autoLoadDataset = loadDataset;

    try {
      Utils.showNotification("🔄 Loading generated dataset...", "info");

      const response = await fetch(`/api/data/load/${loadDataset}`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || "Failed to load dataset");
      }

      const result = await response.json();

      // Update app state
      AppState.dataLoaded = true;

      // Update the data display using existing logic
      updateDataStatusDisplay(result);

      Utils.showNotification(
        `✅ Dataset loaded successfully! ${result.total_equations.toLocaleString()} equations ready for training.`,
        "success"
      );

      // Clean URL
      const newUrl = window.location.pathname;
      window.history.replaceState({}, document.title, newUrl);
    } catch (error) {
      Utils.showNotification(
        `❌ Failed to load dataset: ${error.message}`,
        "error"
      );
    }
  }
}

// Update data status display with loaded dataset info
function updateDataStatusDisplay(dataInfo) {
  // Use the existing DataSection to update the display
  DataSection.updateDataInfo(dataInfo);
}

// API helper functions
const ApiClient = {
  async request(url, options = {}) {
    try {
      const response = await fetch(url, {
        headers: {
          "Content-Type": "application/json",
          ...options.headers,
        },
        ...options,
      });

      if (!response.ok) {
        // Try to parse error message from response
        let errorMessage = `HTTP error! status: ${response.status}`;
        try {
          const errorData = await response.json();
          if (errorData.error) {
            errorMessage = errorData.error;
          }
        } catch (e) {
          // If JSON parsing fails, use default message
        }
        throw new Error(errorMessage);
      }

      return await response.json();
    } catch (error) {
      console.error("API request failed:", error);
      throw error;
    }
  },

  async uploadFile(file) {
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(API.uploadData, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error("File upload failed:", error);
      throw error;
    }
  },
};

