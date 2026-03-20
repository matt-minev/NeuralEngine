async function requestJson(url, options = {}) {
  const response = await fetch(url, options);
  const payload = await response.json().catch(() => ({}));

  if (!response.ok || payload.error) {
    throw new Error(payload.error || `Request failed: ${response.status}`);
  }

  return payload;
}

export function predictDigit(image) {
  return requestJson("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image }),
  });
}

export function switchModel(modelName) {
  return requestJson("/switch_model", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model_name: modelName }),
  });
}

export function fetchModelInfo() {
  return requestJson("/model_info");
}

export function fetchDatasetSample() {
  return requestJson("/api/dataset/sample");
}

export function fetchArchitecture() {
  return requestJson("/api/model/architecture");
}

export function fetchLayerActivations(imageData, modelName) {
  return requestJson("/api/neural/activations", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image_data: imageData, model_name: modelName }),
  });
}
