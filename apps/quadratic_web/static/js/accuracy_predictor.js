/**
 * Accuracy Predictor Module
 * Predicts neural network accuracy based on dataset and training configuration
 */

const AccuracyPredictor = {
  /**
   * Predict accuracy based on configuration
   * @param {Object} config - Configuration object
   * @returns {Object} Prediction results with accuracy, metrics, and recommendations
   */
  predictAccuracy(config) {
    const numEq = config.num_equations || 1000;
    const rangeSize = config.coefficient_range.max - config.coefficient_range.min;
    
    // Base accuracy depends heavily on the equation type's inherent complexity
    let baseAccuracy = 0.80;
    if (config.equation_type === 'school_grade') {
      baseAccuracy = 0.95;
    } else if (config.equation_type === 'integer_solutions') {
      baseAccuracy = 0.85;
    } else if (config.equation_type === 'fractional_solutions') {
      baseAccuracy = 0.75;
    } else if (config.equation_type === 'random') {
      baseAccuracy = 0.60;
    }
    
    // Scaling based on dataset size: log10 curve
    // Math.log10(1000) - 3 = 0. Math.log10(10000) - 3 = 1.
    const eqFactor = Math.log10(Math.max(100, numEq)) - 3;
    baseAccuracy += eqFactor * 0.10; // +10% for each order of magnitude above 1k
    
    // Scaling based on range size
    // For range 10: (20 - 10) * 0.005 = +0.05
    // For range 40: (20 - 40) * 0.005 = -0.10
    const rangeFactor = (20 - rangeSize) * 0.005;
    baseAccuracy += rangeFactor;
    
    // Minor adjustments
    if (config.use_augmentation) baseAccuracy += 0.02;
    if (config.ensemble_size > 1) baseAccuracy += 0.01 * Math.min(config.ensemble_size, 3);
    
    const epochs = config.epochs || 1000;
    if (epochs >= 2000) baseAccuracy += 0.02;
    else if (epochs < 1000) baseAccuracy -= 0.02;

    // Clamp the accuracy to sensible bounds
    const accuracy = Math.min(0.999, Math.max(0.10, baseAccuracy));
    
    // Dynamic confidence interval based on dataset size
    const confidenceInterval = numEq < 5000 ? 0.05 : 0.02;
    
    // Other metrics estimates
    const r2Score = Math.min(0.999, accuracy + 0.01);
    const mae = Math.max(0.001, 0.2 - (accuracy - 0.5) * 0.4);
    
    const level = this.getAccuracyLevel(accuracy);
    
    return {
      accuracy: accuracy,
      accuracyPercent: Math.round(accuracy * 100),
      accuracyMin: Math.max(0.10, accuracy - confidenceInterval),
      accuracyMax: Math.min(0.999, accuracy + confidenceInterval),
      r2Score: r2Score,
      mae: mae,
      confidenceInterval: {
        lower: Math.max(0.10, accuracy - confidenceInterval),
        upper: Math.min(0.999, accuracy + confidenceInterval),
        uncertainty: confidenceInterval,
      },
      level: level,
    };
  },
  
  /**
   * Get accuracy level category
   */
  getAccuracyLevel(accuracy) {
    if (accuracy >= 0.95) return { name: 'Optimal', color: '#8B5CF6', icon: '🎯' };
    if (accuracy >= 0.85) return { name: 'Highly Reliable', color: '#10B981', icon: '✅' };
    if (accuracy >= 0.70) return { name: 'Moderate Variance', color: '#3B82F6', icon: '📊' };
    if (accuracy >= 0.50) return { name: 'High Variance', color: '#F59E0B', icon: '⚠️' };
    return { name: 'Experimental', color: '#EF4444', icon: '🧪' };
  },
  
  /**
   * Estimate training time based on configuration
   * @param {Object} config - Configuration object
   * @returns {Object} Time estimates in various units
   */
  estimateTrainingTime(config) {
    const datasetSize = config.num_equations || 1000;
    const epochs = config.epochs || this.recommendEpochs(datasetSize);
    const ensembleSize = config.ensemble_size || 1;
    
    // Realistic time estimation based on actual observed performance
    // 100k equations took ~160 seconds in practice
    // This suggests ~0.0000008 seconds per equation per epoch (with early stopping)
    // For quadratic equations, training is very fast due to simple input/output
    const baseTimePerEpoch = datasetSize * 0.0000008; // seconds per epoch
    let totalTime = baseTimePerEpoch * epochs * ensembleSize;
    
    // Account for early stopping (typically stops around 60-80% of max epochs)
    // and multi-phase overhead
    const earlyStoppingFactor = 0.7; // Average early stopping at 70% of epochs
    const overhead = config.use_multi_phase ? 1.15 : 1.0;
    totalTime = totalTime * earlyStoppingFactor * overhead;
    
    // Ensure minimum time (even small datasets take a few seconds)
    if (totalTime < 5) {
      totalTime = 5;
    }
    
    // Convert to appropriate units
    if (totalTime < 60) {
      return { value: Math.round(totalTime), unit: 'seconds', raw: totalTime };
    } else if (totalTime < 3600) {
      return { value: Math.round(totalTime / 60), unit: 'minutes', raw: totalTime };
    } else {
      const hours = totalTime / 3600;
      if (hours < 24) {
        return { value: Math.round(hours * 10) / 10, unit: 'hours', raw: totalTime };
      } else {
        return { value: Math.round(hours), unit: 'hours', raw: totalTime };
      }
    }
  },
  
  /**
   * Recommend number of epochs based on dataset size
   * @param {number} numEquations - Number of equations in dataset
   * @returns {number} Recommended epochs
   */
  recommendEpochs(numEquations) {
    if (numEquations < 2000) {
      return 1000;
    } else if (numEquations < 10000) {
      return 1500;
    } else if (numEquations < 50000) {
      return 2000;
    } else {
      return 2500;
    }
  },
  
  
  /**
   * Get recommendations for improving accuracy
   * @param {Object} config - Current configuration
   * @param {number} currentAccuracy - Current predicted accuracy
   * @returns {Array} Array of recommendation objects
   */
  getRecommendations(config, currentAccuracy) {
    const recommendations = [];
    
    // Dataset size recommendations
    if (config.num_equations < 10000 && currentAccuracy < 0.95) {
      recommendations.push({
        type: 'dataset_size',
        priority: 'high',
        message: `Increase dataset size to at least 10,000 equations for better accuracy`,
        impact: '+3-5%',
        action: () => ({ num_equations: 10000 })
      });
    }
    
    // Coefficient range recommendations
    const rangeSize = config.coefficient_range.max - config.coefficient_range.min;
    if (config.coefficient_range.max > 15 || config.coefficient_range.min < -15) {
      recommendations.push({
        type: 'coefficient_range',
        priority: 'medium',
        message: `Optimal coefficient range is -15 to 15 for school-grade equations`,
        impact: '+1-2%',
        action: () => ({ coefficient_range: { min: -15, max: 15 } })
      });
    }
    
    // Training configuration recommendations
    if (config.use_augmentation === false) {
      recommendations.push({
        type: 'augmentation',
        priority: 'high',
        message: 'Enable data augmentation to improve accuracy',
        impact: '+2%',
        action: () => ({ use_augmentation: true })
      });
    }
    
    if (currentAccuracy >= 0.95 && config.ensemble_size <= 1) {
      recommendations.push({
        type: 'ensemble',
        priority: 'medium',
        message: 'Consider ensemble training (3-5 models) for elite accuracy',
        impact: '+1-3%',
        action: () => ({ ensemble_size: 3 })
      });
    }
    
    // Epochs recommendations
    const recommendedEpochs = this.recommendEpochs(config.num_equations);
    if (config.epochs && config.epochs < recommendedEpochs) {
      recommendations.push({
        type: 'epochs',
        priority: 'medium',
        message: `Increase training epochs to ${recommendedEpochs} for optimal results`,
        impact: '+1%',
        action: () => ({ epochs: recommendedEpochs })
      });
    }
    
    return recommendations;
  }
};
