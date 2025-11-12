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
  document.getElementById("mirrorStatus").innerHTML = '<p>No mirror detected</p>';
  document.getElementById("qualityMetrics").innerHTML = '<p>Analyze your writing...</p>';
  document.getElementById("suggestionsList").innerHTML = '<p>No suggestions yet</p>';
  document.getElementById("resourcesList").innerHTML = '<p>Resources will appear here</p>';
}

function updateMirrorDetection(mirrorData) {
  const container = document.getElementById("mirrorStatus");
  
  if (!mirrorData || !mirrorData.mirror_detected) {
    container.innerHTML = `
      <div class="mirror-not-detected">
        <p>✓ No mirror detected - character orientation looks correct</p>
      </div>
    `;
    return;
  }

  const original = mirrorData.original_prediction;
  const mirrored = mirrorData.mirrored_prediction;
  
  container.innerHTML = `
    <div class="mirror-detected">
      <p><strong>⚠️ Mirror Detected!</strong></p>
      <p>Your character appears to be mirrored.</p>
      <p><strong>Original:</strong> "${original.predicted_character}" (${original.confidence.toFixed(1)}% confidence)</p>
      <p><strong>Mirrored:</strong> "${mirrored.predicted_character}" (${mirrored.confidence.toFixed(1)}% confidence)</p>
      <p><em>Did you mean to write "${mirrored.predicted_character}"?</em></p>
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

  container.innerHTML = scores.map(score => `
    <div class="quality-score">
      <div class="quality-label">${score.label}:</div>
      <div class="quality-bar">
        <div class="quality-bar-fill" style="width: ${score.value}%"></div>
      </div>
      <div class="quality-value">${score.value.toFixed(0)}%</div>
    </div>
  `).join('');
}

function updateSuggestions(accessibility) {
  const container = document.getElementById("suggestionsList");
  
  if (!accessibility || !accessibility.suggestions || accessibility.suggestions.length === 0) {
    container.innerHTML = '<p>✓ No issues detected - great writing!</p>';
    return;
  }

  container.innerHTML = accessibility.suggestions.map(suggestion => {
    const priorityClass = suggestion.priority === 'high' ? 'high-priority' : 
                          suggestion.priority === 'medium' ? 'medium-priority' : '';
    
    return `
      <div class="suggestion-item ${priorityClass}">
        <div class="suggestion-message">${suggestion.message}</div>
        <div class="suggestion-advice">💡 ${suggestion.advice}</div>
      </div>
    `;
  }).join('');
}

function updateResources(accessibility) {
  const container = document.getElementById("resourcesList");
  
  if (!accessibility || !accessibility.resources || accessibility.resources.length === 0) {
    container.innerHTML = '<p>No resources available</p>';
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

