async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }
  return response.json();
}

export function predictCharacter(payload, debugEnabled = false) {
  const url = debugEnabled ? "/predict?debug=true" : "/predict/accessibility";
  return fetchJson(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
}

export function getRandomTestSamples(count, character = "") {
  const url = `/api/test/random?count=${count}${character ? `&character=${character}` : ""}`;
  return fetchJson(url, {
    method: "GET",
    headers: {
      "Content-Type": "application/json",
    },
    cache: "no-cache",
  });
}

export function predictTestSample(imageArray) {
  return fetchJson("/api/test/predict", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ image_array: imageArray }),
  });
}
