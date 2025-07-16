"""
Helper utilities for the quadratic neural network application
"""

import numpy as np
from typing import Dict, List, Tuple, Any

def format_number(value: float, decimals: int = 6) -> str:
    """Format number for display"""
    if abs(value) < 1e-10:
        return "0.000000"
    return f"{value:.{decimals}f}"

def calculate_equation_error(a: float, b: float, c: float, x: float) -> float:
    """Calculate error for quadratic equation ax² + bx + c = 0"""
    return abs(a * x**2 + b * x + c)

def assess_performance(r2_score: float) -> str:
    """Assess model performance based on R² score"""
    if r2_score > 0.9:
        return "EXCELLENT"
    elif r2_score > 0.7:
        return "GOOD"
    elif r2_score > 0.5:
        return "FAIR"
    else:
        return "POOR"

def get_confidence_level(confidence: float) -> str:
    """Get confidence level description"""
    if confidence > 0.8:
        return "🟢 High"
    elif confidence > 0.6:
        return "🟡 Medium"
    else:
        return "🔴 Low"

def validate_quadratic_data(data: np.ndarray) -> Dict[str, Any]:
    """Validate quadratic equation data"""
    results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'stats': {}
    }
    
    if data.shape[1] != 5:
        results['valid'] = False
        results['errors'].append("Data must have exactly 5 columns: a, b, c, x1, x2")
        return results
    
    # Check for invalid values
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        results['valid'] = False
        results['errors'].append("Data contains NaN or infinite values")
    
    # Check for zero 'a' values
    zero_a_count = np.sum(data[:, 0] == 0)
    if zero_a_count > 0:
        results['warnings'].append(f"Found {zero_a_count} equations with a=0 (linear equations)")
    
    # Verify solutions
    verification_errors = []
    for i in range(min(100, len(data))):  # Check first 100 samples
        a, b, c, x1, x2 = data[i]
        error1 = calculate_equation_error(a, b, c, x1)
        error2 = calculate_equation_error(a, b, c, x2)
        
        if error1 > 1e-6 or error2 > 1e-6:
            verification_errors.append(i)
    
    if verification_errors:
        results['warnings'].append(f"Found {len(verification_errors)} equations with solution errors")
    
    # Calculate statistics
    results['stats'] = {
        'total_equations': len(data),
        'zero_a_count': zero_a_count,
        'verification_errors': len(verification_errors)
    }
    
    return results
