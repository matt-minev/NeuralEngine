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
    let baseAccuracy = 0.65; // Base accuracy for minimal dataset (reduced even more)
    
    // Dataset size factor (heavily penalize tiny datasets)
    const numEq = config.num_equations || 1000;
    if (numEq < 100) {
      baseAccuracy += 0.00; // Tiny dataset - RED territory (~65-68%)
    } else if (numEq < 250) {
      baseAccuracy += 0.03; // Very small dataset - RED (~68-71%)
    } else if (numEq < 500) {
      baseAccuracy += 0.06; // Small dataset - RED/ORANGE (~71-74%)
    } else if (numEq < 1000) {
      baseAccuracy += 0.10; // Small-medium - ORANGE (~75-78%)
    } else if (numEq < 5000) {
      baseAccuracy += 0.14; // Medium-small dataset (~79-82%)
    } else if (numEq < 10000) {
      baseAccuracy += 0.18; // Medium dataset (~83-86%)
    } else if (numEq < 25000) {
      baseAccuracy += 0.21; // Large dataset (~86-89%)
    } else if (numEq < 50000) {
      baseAccuracy += 0.24; // Very large dataset (~89-92%)
    } else if (numEq < 100000) {
      baseAccuracy += 0.27; // Huge dataset (~92-95%)
    } else {
      baseAccuracy += 0.30; // Elite dataset (~95-98%)
    }
    
    // Coefficient range factor (optimal: -15 to 15)
    const rangeSize = config.coefficient_range.max - config.coefficient_range.min;
    if (rangeSize <= 10) {
      baseAccuracy += 0.01; // Very small range, easy but less generalizable
    } else if (rangeSize <= 20) {
      baseAccuracy += 0.02; // Good range
    } else if (rangeSize <= 30) {
      baseAccuracy += 0.025; // Optimal range
    } else if (rangeSize <= 40) {
      baseAccuracy += 0.01; // Large range, harder
    } else {
      baseAccuracy -= 0.01; // Very large range, much harder
    }
    
    // Equation type factor (reduced impact)
    if (config.equation_type === 'school_grade') {
      baseAccuracy += 0.015; // School grade is easier to learn
    } else if (config.equation_type === 'integer_solutions') {
      baseAccuracy += 0.01;
    } else if (config.equation_type === 'random') {
      baseAccuracy -= 0.015; // Random is harder
    }
    
    // Root type factor (for school_grade)
    if (config.equation_type === 'school_grade') {
      if (config.root_type === 'integers') {
        baseAccuracy += 0.005; // Integers are slightly easier
      } else if (config.root_type === 'fractions') {
        baseAccuracy -= 0.005; // Fractions are slightly harder
      }
    }
    
    // Pattern distribution factor (reduced)
    if (config.pattern_distribution === 'auto' || config.balanced_patterns) {
      baseAccuracy += 0.005;
    }
    
    // Training config factors (reduced impact)
    if (config.use_augmentation !== false) {
      baseAccuracy += 0.01; // Reduced from 0.02
    }
    if (config.ensemble_size > 1) {
      baseAccuracy += 0.01 * Math.min(config.ensemble_size, 5); // Reduced from 0.015
    }
    
    // Epochs impact (more granular)
    const epochs = config.epochs || 1000;
    if (epochs < 1000) {
      baseAccuracy -= 0.01; // Too few epochs
    } else if (epochs >= 2500) {
      baseAccuracy += 0.015; // Excellent training
    } else if (epochs >= 2000) {
      baseAccuracy += 0.01; // Very good training
    } else if (epochs >= 1500) {
      baseAccuracy += 0.005; // Good training
    }
    
    if (config.use_multi_phase !== false) {
      baseAccuracy += 0.005; // Reduced from 0.01
    }
    
    // Clamp accuracy between 70% and 99%
    const accuracy = Math.min(0.99, Math.max(0.70, baseAccuracy));
    
    // Calculate confidence interval (±3% for most cases, ±5% for very small datasets)
    const confidenceInterval = config.num_equations < 5000 ? 0.05 : 0.03;
    
    // Estimate R² score (typically slightly higher than accuracy)
    const r2Score = Math.min(0.995, accuracy + 0.02);
    
    // Estimate MAE (lower is better, typically 0.01-0.1 for good models)
    const mae = Math.max(0.001, 0.1 - (accuracy - 0.7) * 0.3);
    
    const level = this.getAccuracyLevel(accuracy);
    
    return {
      accuracy: accuracy,
      accuracyPercent: Math.round(accuracy * 100),
      accuracyMin: Math.max(0.70, accuracy - confidenceInterval),
      accuracyMax: Math.min(0.99, accuracy + confidenceInterval),
      r2Score: r2Score,
      mae: mae,
      confidenceInterval: {
        lower: Math.max(0.70, accuracy - confidenceInterval),
        upper: Math.min(0.99, accuracy + confidenceInterval),
        uncertainty: confidenceInterval,
      },
      level: level,
    };
  },
  
  /**
   * Get accuracy level category
   */
  getAccuracyLevel(accuracy) {
    if (accuracy >= 0.96) return { name: 'Elite', color: '#8B5CF6', icon: '👑' };
    if (accuracy >= 0.90) return { name: 'Excellent', color: '#10B981', icon: '✅' };
    if (accuracy >= 0.82) return { name: 'Good', color: '#3B82F6', icon: '⭐' };
    if (accuracy >= 0.75) return { name: 'Acceptable', color: '#F59E0B', icon: '⚠️' };
    return { name: 'Insufficient', color: '#EF4444', icon: '❌' };
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
