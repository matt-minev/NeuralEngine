// NeuralEngine Universal Character Recognizer - Accessibility Features
window.updateAccessibilityDisplay = function(result) {
  if (!result) {
    clearAccessibilityDisplay();
    return;
  }

  updateMirrorDetection(result.mirror_detection);
  updateWritingQuality(result.quality_metrics);
  updateSuggestions(result.accessibility);
  updateResources(result.accessibility);
}

function clearAccessibilityDisplay() {
  document.getElementById("mirrorStatus").innerHTML = '<p>Draw a character to analyze...</p>';
  document.getElementById("qualityMetrics").innerHTML = '<p>Analyze your writing...</p>';
  document.getElementById("suggestionsList").innerHTML = '<p>No suggestions yet</p>';
  
  // Always show resources from the start
  updateResourcesInitial();
}

function updateMirrorDetection(mirrorData) {
  const container = document.getElementById("mirrorStatus");
  
  if (!mirrorData || !mirrorData.mirror_detected) {
    container.innerHTML = `
      <div class="mirror-status mirror-not-detected">
        No mirror writing detected. Character orientation looks correct!
      </div>
    `;
    return;
  }

  const original = mirrorData.original_prediction;
  const mirrored = mirrorData.mirrored_prediction;
  
  container.innerHTML = `
    <div class="mirror-status mirror-detected">
      <strong>Mirror Detected!</strong>
      <p>Your character appears to be mirrored.</p>
      <p>Original: "${original.predicted_character}" (${original.confidence.toFixed(1)}% confidence)</p>
      <p>Mirrored: "${mirrored.predicted_character}" (${mirrored.confidence.toFixed(1)}% confidence)</p>
      <p>Did you mean to write "${mirrored.predicted_character}"?</p>
    </div>
  `;
}

function updateWritingQuality(qualityMetrics) {
  const container = document.getElementById("qualityMetrics");
  
  if (!qualityMetrics) {
    container.innerHTML = '<p>Quality analysis unavailable</p>';
    return;
  }

  const scores = [
    { label: 'Overall', value: qualityMetrics.overall_score },
    { label: 'Clarity', value: qualityMetrics.clarity_score },
    { label: 'Size', value: qualityMetrics.size_score },
    { label: 'Centering', value: qualityMetrics.centering_score },
    { label: 'Stroke', value: qualityMetrics.stroke_score }
  ];

  container.innerHTML = '<div class="quality-metrics">' +
    scores.map(score => {
      let qualityClass = 'quality-low';
      if (score.value >= 90) qualityClass = 'quality-high';
      else if (score.value >= 75) qualityClass = 'quality-medium';
      
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
    }).join('') +
    '</div>';
}

function updateSuggestions(accessibility) {
  const container = document.getElementById("suggestionsList");
  
  if (!accessibility || !accessibility.suggestions || accessibility.suggestions.length === 0) {
    container.innerHTML = `
      <div class="suggestions-list">
        <div class="suggestion-item" style="border-color: rgba(34, 197, 94, 0.5); background: rgba(34, 197, 94, 0.05);">
          <div class="suggestion-message" style="color: #22c55e;">✓ No issues detected - great writing!</div>
        </div>
      </div>
    `;
    return;
  }

  container.innerHTML = '<div class="suggestions-list">' +
    accessibility.suggestions.map(suggestion => {
      const priorityClass = suggestion.priority === 'high' ? 'high-priority' : 
                            suggestion.priority === 'medium' ? 'medium-priority' : '';
      
      return `
        <div class="suggestion-item ${priorityClass}">
          <div class="suggestion-message">${suggestion.message}</div>
          ${suggestion.advice ? `<div class="suggestion-advice">${suggestion.advice}</div>` : ''}
        </div>
      `;
    }).join('') +
    '</div>';
}

function updateResourcesInitial() {
  const container = document.getElementById("resourcesList");
  
  // Default resources shown from the start
  const defaultResources = [
    {
      title: "Neural Engine Documentation",
      description: "Learn more about how character recognition works and explore the Neural Engine framework.",
      url: "https://github.com/matt-minev/NeuralEngine"
    },
    {
      title: "Improve Your Handwriting",
      description: "Tips and techniques for clearer character formation and better recognition accuracy.",
      url: "https://www.wikihow.com/Improve-Your-Handwriting"
    }
  ];
  
  container.innerHTML = defaultResources.map(resource => `
    <div class="resource-item">
      <div class="resource-title">${resource.title}</div>
      <div class="resource-description">${resource.description}</div>
      <a href="${resource.url}" target="_blank" class="resource-link">Learn more →</a>
    </div>
  `).join('');
}

function updateResources(accessibility) {
  const container = document.getElementById("resourcesList");
  
  if (!accessibility || !accessibility.resources || accessibility.resources.length === 0) {
    // If no custom resources, show defaults
    updateResourcesInitial();
    return;
  }

  container.innerHTML = accessibility.resources.map(resource => {
    const url = resource.url === '#' ? '#' : resource.url;
    return `
      <div class="resource-item">
        <div class="resource-title">${resource.title}</div>
        <div class="resource-description">${resource.description}</div>
        ${url !== '#' ? `<a href="${url}" target="_blank" class="resource-link">Learn more →</a>` : ''}
      </div>
    `;
  }).join('');
}

// Initialize accessibility display on page load
document.addEventListener("DOMContentLoaded", () => {
  clearAccessibilityDisplay();
});

