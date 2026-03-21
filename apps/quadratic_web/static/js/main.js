// Application initialization
document.addEventListener("DOMContentLoaded", async () => {
  console.log("🚀 Quadratic Neural Network Web Application");
  console.log("Initializing application...");

  // Initialize all sections
  Navigation.init();
  DataSection.init();
  TrainingSection.init();
  ModelSection.init();
  PredictionSection.init();
  AnalysisSection.init();
  ComparisonSection.init();

  // Check for auto-load dataset from URL parameter
  await checkAndLoadDataset();

  // Check API health
  ApiClient.request(API.health)
    .then((response) => {
      console.log("✅ API connection established");
      document.getElementById("connection-status").textContent = "Connected";
    })
    .catch((error) => {
      console.error("❌ API connection failed:", error);
      document.getElementById("connection-status").textContent = "Disconnected";
      document.querySelector(".status-dot").style.backgroundColor =
        "var(--error-color)";
    });

  console.log("🎉 Application initialized successfully!");
});

// Handle training status updates
setInterval(async () => {
  if (AppState.isTraining) {
    try {
      const status = await ApiClient.request(API.trainingStatus);
      if (!status.is_training && AppState.isTraining) {
        // Training finished
        AppState.isTraining = false;
        document.getElementById("start-training-btn").innerHTML =
          '<i class="fas fa-play"></i> Start Training';
        document.getElementById("start-training-btn").disabled = false;
        document.getElementById("stop-training-btn").style.display = "none";
        Utils.showNotification("Training completed!", "success");

        // Fetch results and update save section
        try {
          const results = await ApiClient.request(API.results);
          AppState.results = results;
          console.log("✅ Results loaded:", AppState.results);

          // Update save section to show trained models
          ModelSection.updateSaveSection();
        } catch (error) {
          console.error("Failed to load results after training:", error);
        }
      }
    } catch (error) {
      console.error("Failed to check training status:", error);
    }
  }
}, 2000);
