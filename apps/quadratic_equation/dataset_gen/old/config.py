"""
Configuration and utility classes for quadratic equation dataset generation.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass
class GenerationConfig:
    """Configuration parameters for dataset generation."""
    
    # Basic settings
    dataset_size: int = 1000
    dataset_name: str = "quadratic_dataset"
    infinite_mode: bool = False
    
    # Coefficient ranges
    a_min: float = -10.0
    a_max: float = 10.0
    b_min: float = -10.0
    b_max: float = 10.0
    c_min: float = -10.0
    c_max: float = 10.0
    
    # Generation preferences
    prioritize_whole_solutions: bool = True
    whole_solution_ratio: float = 0.6
    only_whole_solutions: bool = False
    allow_complex_solutions: bool = False
    force_real_solutions: bool = True
    systematic_generation: bool = False
    
    # NEW: Textbook mode options
    textbook_mode: bool = False
    textbook_max_coeff: int = 10
    textbook_prefer_perfect_discriminant: bool = True
    textbook_simple_solutions: bool = True
    textbook_integer_solutions_ratio: float = 0.7
    
    # Solution constraints
    solution_min: float = -100.0
    solution_max: float = 100.0
    
    # Advanced options
    avoid_zero_a: bool = True
    integer_coefficients_only: bool = False
    symmetric_solutions: bool = False


class QuadraticSolver:
    """Utility methods for solving and working with quadratic equations."""

    @staticmethod
    def solve_quadratic(a: float, b: float, c: float) -> Tuple[Optional[float], Optional[float]]:
        """
        Solve ax² + bx + c = 0
        
        Args:
            a, b, c: Coefficients of the quadratic equation
            
        Returns:
            Tuple of (x1, x2) solutions or (None, None) if no real solutions
        """
        if a == 0:
            # Linear equation: bx + c = 0
            if b == 0:
                return None, None
            return -c/b, -c/b
        
        discriminant = b**2 - 4*a*c
        
        if discriminant < 0:
            return None, None  # No real solutions
        
        sqrt_discriminant = np.sqrt(discriminant)
        x1 = (-b + sqrt_discriminant) / (2*a)
        x2 = (-b - sqrt_discriminant) / (2*a)
        
        return x1, x2

    @staticmethod
    def is_whole_number(x: float, tolerance: float = 1e-10) -> bool:
        """Check if a number is effectively a whole number."""
        return abs(x - round(x)) < tolerance

    @staticmethod
    def verify_solution(a: float, b: float, c: float, x: float, tolerance: float = 1e-10) -> bool:
        """Verify if x is a solution to ax² + bx + c = 0."""
        result = a * x**2 + b * x + c
        return abs(result) < tolerance
