import { qs } from "../core/utils.js";

export function renderDebugImages(debugImages, stats) {
  const container = qs("#debugImagesContainer");
  if (!container) {
    return;
  }

  const steps = [
    { key: "original", label: "Original" },
    { key: "flipped_upside_down", label: "Flipped Upside Down" },
    { key: "after_resize", label: "After Resize" },
    { key: "final", label: "Final (to Model)" },
  ];

  let html = '<div class="debug-images-grid">';
  steps.forEach((step) => {
    if (debugImages[step.key]) {
      html += `
        <div class="debug-image-item">
          <div class="debug-image-label">${step.label}</div>
          <img src="${debugImages[step.key]}" alt="${step.label}" class="debug-image" />
        </div>
      `;
    }
  });
  html += "</div>";
  container.innerHTML = html;

  const statsContainer = qs("#debugStatsContainer");
  if (!statsContainer || !stats) {
    return;
  }

  statsContainer.innerHTML = `
    <div class="debug-stats-grid">
      <div class="debug-stat-item">
        <span class="debug-stat-label">Original Range:</span>
        <span class="debug-stat-value">[${stats.original_min.toFixed(3)}, ${stats.original_max.toFixed(3)}]</span>
      </div>
      <div class="debug-stat-item">
        <span class="debug-stat-label">Original Mean:</span>
        <span class="debug-stat-value">${stats.original_mean.toFixed(3)}</span>
      </div>
      <div class="debug-stat-item">
        <span class="debug-stat-label">Final Range:</span>
        <span class="debug-stat-value">[${stats.final_min.toFixed(3)}, ${stats.final_max.toFixed(3)}]</span>
      </div>
      <div class="debug-stat-item">
        <span class="debug-stat-label">Final Mean:</span>
        <span class="debug-stat-value">${stats.final_mean.toFixed(3)}</span>
      </div>
      <div class="debug-stat-item">
        <span class="debug-stat-label">Final Std:</span>
        <span class="debug-stat-value">${stats.final_std.toFixed(3)}</span>
      </div>
    </div>
  `;
}
