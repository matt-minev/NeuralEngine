"""
Simple configuration for textbook quadratic equation generation.
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np


@dataclass
class TextbookConfig:
    """Configuration for textbook-style quadratic generation."""
    
    # Basic settings
    dataset_name: str = "textbook_quadratics"
    
    # Textbook mode settings (default)
    textbook_mode: bool = True
    max_coeff: int = 5  # Coefficients range from -max_coeff to +max_coeff
    force_perfect_discriminant: bool = True
    prefer_whole_solutions: bool = True
    
    # Advanced mode settings (when textbook_mode = False)
    advanced_max_coeff: int = 20
    allow_irrational_solutions: bool = True
    larger_solution_range: bool = True
    
    # Always enabled settings
    avoid_zero_a: bool = True  # Ensure it's actually quadratic
    allow_negative_a: bool = True  # Allow negative leading coefficients


class QuadraticSolver:
    """Simple quadratic equation solver and utilities."""
    
    @staticmethod
    def solve_quadratic(a: float, b: float, c: float) -> Tuple[Optional[float], Optional[float]]:
        """Solve ax² + bx + c = 0 and return (x1, x2)."""
        if abs(a) < 1e-10:  # Essentially zero
            if abs(b) < 1e-10:
                return None, None  # No solution
            return -c/b, -c/b  # Linear equation
        
        discriminant = b**2 - 4*a*c
        
        if discriminant < 0:
            return None, None  # No real solutions
        
        sqrt_d = np.sqrt(discriminant)
        x1 = (-b + sqrt_d) / (2*a)
        x2 = (-b - sqrt_d) / (2*a)
        
        return x1, x2
    
    @staticmethod
    def is_perfect_square(n: float, tolerance: float = 1e-10) -> bool:
        """Check if a number is a perfect square."""
        if n < 0:
            return False
        sqrt_n = np.sqrt(n)
        return abs(sqrt_n - round(sqrt_n)) < tolerance
    
    @staticmethod
    def is_whole_number(x: float, tolerance: float = 1e-10) -> bool:
        """Check if a number is effectively a whole number."""
        return abs(x - round(x)) < tolerance
    
    @staticmethod
    def verify_solution(a: float, b: float, c: float, x: float, tolerance: float = 1e-8) -> bool:
        """Verify if x is a solution to the quadratic equation."""
        result = a * x**2 + b * x + c
        return abs(result) < tolerance
    
    @staticmethod
    def format_equation(a: float, b: float, c: float) -> str:
        """Format equation as a readable string."""
        # Handle the 'a' coefficient
        if a == 1:
            eq = "x²"
        elif a == -1:
            eq = "-x²"
        else:
            eq = f"{a}x²"
        
        # Handle the 'b' coefficient
        if b > 0:
            if b == 1:
                eq += " + x"
            else:
                eq += f" + {b}x"
        elif b < 0:
            if b == -1:
                eq += " - x"
            else:
                eq += f" - {abs(b)}x"
        
        # Handle the 'c' coefficient
        if c > 0:
            eq += f" + {c}"
        elif c < 0:
            eq += f" - {abs(c)}"
        
        return eq + " = 0"


def get_textbook_ranges(config: TextbookConfig) -> dict:
    """Get coefficient ranges based on current mode."""
    if config.textbook_mode:
        return {
            'a_range': (-config.max_coeff, config.max_coeff),
            'b_range': (-config.max_coeff, config.max_coeff),
            'c_range': (-config.max_coeff, config.max_coeff),
            'solution_range': (-10, 10)  # Keep solutions reasonable
        }
    else:
        return {
            'a_range': (-config.advanced_max_coeff, config.advanced_max_coeff),
            'b_range': (-config.advanced_max_coeff, config.advanced_max_coeff),
            'c_range': (-config.advanced_max_coeff, config.advanced_max_coeff),
            'solution_range': (-50, 50)  # Larger range for advanced mode
        }
