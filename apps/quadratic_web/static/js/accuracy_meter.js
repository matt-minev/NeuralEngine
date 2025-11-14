/**
 * Accuracy Meter Component
 * Visual circular progress meter with accuracy breakdown
 */

const AccuracyMeter = {
  /**
   * Initialize the accuracy meter
   * @param {string} containerId - ID of container element
   */
  init(containerId) {
    this.container = document.getElementById(containerId);
    if (!this.container) {
      // Try to find existing accuracy meter elements
      this.progressCircle = document.getElementById('accuracy-progress');
      this.accuracyValue = document.getElementById('accuracy-value');
      this.accuracyLabel = document.getElementById('accuracy-level');
      this.accuracyRange = document.getElementById('accuracy-confidence');
      this.r2Score = document.getElementById('r2-score');
      this.maeValue = document.getElementById('mae-value');
      this.trainingTime = document.getElementById('training-time');
      return;
    }
    
    this.createMeter();
  },
  
  /**
   * Create the meter HTML structure (if container exists)
   */
  createMeter() {
    // Meter already exists in HTML, just get references
    this.progressCircle = document.getElementById('accuracy-progress');
    this.accuracyValue = document.getElementById('accuracy-value');
    this.accuracyLabel = document.getElementById('accuracy-level');
    this.accuracyRange = document.getElementById('accuracy-confidence');
    this.r2Score = document.getElementById('r2-score');
    this.maeValue = document.getElementById('mae-value');
    this.trainingTime = document.getElementById('training-time');
  },
  
  /**
   * Update the meter with new prediction data
   * @param {Object} prediction - Prediction object from AccuracyPredictor
   * @param {Object} timeEstimate - Time estimate object
   */
  update(prediction, timeEstimate) {
    if (!this.progressCircle || !this.accuracyValue) return;
    
    const accuracy = prediction.accuracy;
    const category = AccuracyPredictor.getAccuracyCategory(accuracy);
    
    // Calculate circumference and stroke-dashoffset (radius = 90 from HTML)
    const circumference = 2 * Math.PI * 90;
    const offset = circumference * (1 - accuracy);
    
    // Update progress circle
    this.progressCircle.style.strokeDasharray = `${circumference} ${circumference}`;
    this.progressCircle.style.strokeDashoffset = offset;
    this.progressCircle.style.stroke = category.color;
    this.progressCircle.style.transition = 'stroke-dashoffset 0.8s ease-in-out, stroke 0.3s ease';
    
    // Update text content
    this.accuracyValue.textContent = `${(accuracy * 100).toFixed(1)}%`;
    this.accuracyValue.style.color = category.color;
    if (this.accuracyLabel) {
      this.accuracyLabel.textContent = category.label;
    }
    if (this.accuracyRange) {
      this.accuracyRange.textContent = 
        `±${(prediction.confidenceInterval * 100).toFixed(1)}%`;
    }
    
    // Update breakdown
    if (this.r2Score) {
      this.r2Score.textContent = (prediction.r2Score * 100).toFixed(2) + '%';
    }
    if (this.maeValue) {
      this.maeValue.textContent = prediction.mae.toFixed(4);
    }
    if (this.trainingTime) {
      this.trainingTime.textContent = timeEstimate.formatted;
    }
  },
  
  /**
   * Show loading state
   */
  showLoading() {
    if (this.accuracyValue) {
      this.accuracyValue.textContent = '...';
    }
    if (this.accuracyLabel) {
      this.accuracyLabel.textContent = 'Calculating...';
    }
  },
  
  /**
   * Reset to default state
   */
  reset() {
    if (this.progressCircle) {
      const circumference = 2 * Math.PI * 90;
      this.progressCircle.style.strokeDashoffset = circumference.toString();
      this.progressCircle.style.stroke = '#d2d2d7';
    }
    if (this.accuracyValue) {
      this.accuracyValue.textContent = '85%';
      this.accuracyValue.style.color = '';
    }
    if (this.accuracyLabel) {
      this.accuracyLabel.textContent = 'Good';
    }
    if (this.accuracyRange) {
      this.accuracyRange.textContent = '±3%';
    }
  }
};

