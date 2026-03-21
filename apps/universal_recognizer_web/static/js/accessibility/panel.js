export class AccessibilityPanel {
  constructor() {
    this.mirrorStatus = document.getElementById("mirrorStatus");
    this.qualityMetrics = document.getElementById("qualityMetrics");
    this.suggestionsList = document.getElementById("suggestionsList");
    this.resourcesList = document.getElementById("resourcesList");
  }

  update(result) {
    if (!result) {
      this.clear();
      return;
    }

    this.updateMirrorDetection(result.advisory?.mirror_candidate);
    this.updateWritingQuality(result.quality_metrics);
    this.updateSuggestions(result.accessibility);
    this.updateResources(result.accessibility);
  }

  clear() {
    this.mirrorStatus.innerHTML = "<p>Draw a character to analyze...</p>";
    this.qualityMetrics.innerHTML = "<p>Analyze your writing...</p>";
    this.suggestionsList.innerHTML = "<p>No suggestions yet</p>";
    this.updateResourcesInitial();
  }

  updateMirrorDetection(mirrorData) {
    if (!mirrorData || !mirrorData.detected) {
      this.mirrorStatus.innerHTML = `
        <div class="mirror-status mirror-not-detected">
          No strong mirror candidate detected. Orientation looks consistent.
        </div>
      `;
      return;
    }

    const mirrored = mirrorData.mirror_alt;
    this.mirrorStatus.innerHTML = `
      <div class="mirror-status mirror-detected">
        <strong>Mirror Candidate</strong>
        <p>Advisory only: no auto-correction applied.</p>
        <p>Alternative: "${mirrored.predicted_character}" (${mirrored.confidence.toFixed(1)}% confidence)</p>
      </div>
    `;
  }

  updateWritingQuality(qualityMetrics) {
    if (!qualityMetrics) {
      this.qualityMetrics.innerHTML = "<p>Quality analysis unavailable</p>";
      return;
    }

    const scores = [
      { label: "Overall", value: qualityMetrics.overall_score },
      { label: "Clarity", value: qualityMetrics.clarity_score },
      { label: "Size", value: qualityMetrics.size_score },
      { label: "Centering", value: qualityMetrics.centering_score },
      { label: "Stroke", value: qualityMetrics.stroke_score },
    ];

    this.qualityMetrics.innerHTML = `<div class="quality-metrics">${scores.map((score) => {
      let qualityClass = "quality-low";
      if (score.value >= 90) qualityClass = "quality-high";
      else if (score.value >= 75) qualityClass = "quality-medium";

      return `
        <div class="quality-score">
          <div class="quality-label">
            <span>${score.label}:</span>
            <span class="quality-value">${score.value.toFixed(0)}%</span>
          </div>
          <div class="quality-bar">
            <div class="quality-bar-fill ${qualityClass}" style="width: ${score.value}%"></div>
          </div>
        </div>
      `;
    }).join("")}</div>`;
  }

  updateSuggestions(accessibility) {
    if (!accessibility || !accessibility.suggestions || accessibility.suggestions.length === 0) {
      this.suggestionsList.innerHTML = `
        <div class="suggestions-list">
          <div class="suggestion-item">
            <div class="suggestion-message" style="color: var(--green);">✓ No issues detected - great writing!</div>
          </div>
        </div>
      `;
      return;
    }

    this.suggestionsList.innerHTML = `<div class="suggestions-list">${accessibility.suggestions.map((suggestion) => {
      const priorityClass = suggestion.priority === "high"
        ? "high-priority"
        : suggestion.priority === "medium"
          ? "medium-priority"
          : "";

      return `
        <div class="suggestion-item ${priorityClass}">
          <div class="suggestion-message">${suggestion.message}</div>
          ${suggestion.advice ? `<div class="suggestion-advice">${suggestion.advice}</div>` : ""}
        </div>
      `;
    }).join("")}</div>`;
  }

  updateResourcesInitial() {
    const defaultResources = [
      {
        title: "Neural Engine Documentation",
        description: "Learn more about how character recognition works and explore the Neural Engine framework.",
        url: "https://github.com/matt-minev/NeuralEngine",
      },
      {
        title: "Handwriting for Kids",
        description: "Interactive lessons and practice sheets to help children learn to write numbers and letters with simple step-by-step instructions.",
        url: "https://www.handwritingforkids.com/",
      },
      {
        title: "Improve Your Handwriting",
        description: "Tips and techniques for clearer character formation and better recognition accuracy.",
        url: "https://www.wikihow.com/Improve-Your-Handwriting",
      },
    ];

    this.resourcesList.innerHTML = defaultResources.map((resource) => `
      <div class="resource-item">
        <div class="resource-title">${resource.title}</div>
        <div class="resource-description">${resource.description}</div>
        <a href="${resource.url}" target="_blank" class="resource-link" rel="noopener noreferrer">Learn more →</a>
      </div>
    `).join("");
  }

  updateResources(accessibility) {
    if (!accessibility || !accessibility.resources || accessibility.resources.length === 0) {
      this.updateResourcesInitial();
      return;
    }

    this.resourcesList.innerHTML = accessibility.resources.map((resource) => {
      const url = resource.url === "#" ? "#" : resource.url;
      return `
        <div class="resource-item">
          <div class="resource-title">${resource.title}</div>
          <div class="resource-description">${resource.description}</div>
          ${url !== "#" ? `<a href="${url}" target="_blank" class="resource-link" rel="noopener noreferrer">Learn more →</a>` : ""}
        </div>
      `;
    }).join("");
  }
}
