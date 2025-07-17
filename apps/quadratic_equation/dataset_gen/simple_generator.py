"""
Simplified core generator for textbook quadratic equations.
Always generates infinitely until stopped - no modes or complexity.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Callable
import time
from threading import Event

from simple_config import TextbookConfig, QuadraticSolver, get_textbook_ranges


class TextbookQuadraticGenerator:
    """Simplified generator focused on realistic textbook equations."""
    
    def __init__(self, config: TextbookConfig):
        self.config = config
        self.data = []
        self.generation_stats = {
            'total_generated': 0,
            'perfect_discriminants': 0,
            'whole_solutions': 0,
            'textbook_friendly': 0
        }
        
        # Get coefficient ranges based on mode
        self.ranges = get_textbook_ranges(config)
        
    def reset_data(self):
        """Reset all data and statistics."""
        self.data = []
        self.generation_stats = {
            'total_generated': 0,
            'perfect_discriminants': 0,
            'whole_solutions': 0,
            'textbook_friendly': 0
        }
    
    def generate_textbook_equation(self) -> Tuple[float, float, float, float, float]:
        """Generate a single textbook-style quadratic equation."""
        
        if self.config.textbook_mode:
            # Default: Generate realistic textbook equations
            return self._generate_realistic_textbook_equation()
        else:
            # Advanced mode: More complex equations
            return self._generate_advanced_equation()
    
    def _generate_realistic_textbook_equation(self) -> Tuple[float, float, float, float, float]:
        """Generate equations like those found in 8th-12th grade textbooks."""
        
        max_attempts = 100
        
        for _ in range(max_attempts):
            # Method 1: Start with simple integer solutions (70% of the time)
            if np.random.random() < 0.7:
                result = self._generate_from_integer_solutions()
                if result:
                    return result
            
            # Method 2: Generate coefficients with perfect square discriminant (30% of the time)
            result = self._generate_perfect_square_discriminant()
            if result:
                return result
        
        # Fallback: guaranteed simple equation
        return self._generate_guaranteed_simple()
    
    def _generate_from_integer_solutions(self) -> Optional[Tuple[float, float, float, float, float]]:
        """Generate equation by starting with simple integer solutions."""
        
        # Choose small integer solutions (typical in textbooks)
        possible_solutions = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
        x1 = np.random.choice(possible_solutions)
        x2 = np.random.choice(possible_solutions)
        
        # Avoid duplicate solutions and zero solutions for more interesting equations
        while x1 == x2 or (x1 == 0 and x2 == 0):
            x2 = np.random.choice(possible_solutions)
        
        # Choose simple leading coefficient
        a_choices = [-3, -2, -1, 1, 2, 3]
        a = np.random.choice(a_choices)
        
        # Calculate coefficients from factored form: a(x - x1)(x - x2)
        # Expanding: a(x² - (x1+x2)x + x1*x2) = ax² - a(x1+x2)x + a*x1*x2
        b = -a * (x1 + x2)
        c = a * x1 * x2
        
        # Ensure coefficients are within textbook range
        if (abs(b) <= self.config.max_coeff and abs(c) <= self.config.max_coeff):
            return float(a), float(b), float(c), float(x1), float(x2)
        
        return None
    
    def _generate_perfect_square_discriminant(self) -> Optional[Tuple[float, float, float, float, float]]:
        """Generate equation with perfect square discriminant for clean solutions."""
        
        max_attempts = 50
        
        for _ in range(max_attempts):
            # Small integer coefficients
            a = np.random.randint(-self.config.max_coeff, self.config.max_coeff + 1)
            if a == 0:  # Ensure it's quadratic
                a = 1 if np.random.random() < 0.5 else -1
            
            b = np.random.randint(-self.config.max_coeff, self.config.max_coeff + 1)
            c = np.random.randint(-self.config.max_coeff, self.config.max_coeff + 1)
            
            # Calculate discriminant
            discriminant = b**2 - 4*a*c
            
            # Check if discriminant is non-negative and a perfect square
            if discriminant >= 0 and QuadraticSolver.is_perfect_square(discriminant):
                # Calculate solutions
                sqrt_d = int(round(np.sqrt(discriminant)))
                x1 = (-b + sqrt_d) / (2*a)
                x2 = (-b - sqrt_d) / (2*a)
                
                # Prefer simpler solutions for textbook friendliness
                if abs(x1) <= 10 and abs(x2) <= 10:
                    return float(a), float(b), float(c), float(x1), float(x2)
        
        return None
    
    def _generate_guaranteed_simple(self) -> Tuple[float, float, float, float, float]:
        """Generate a guaranteed simple equation as fallback."""
        # Simple equations like x² - 1 = 0, x² - 4 = 0, etc.
        
        simple_patterns = [
            # x² - n² = 0 (difference of squares)
            lambda: self._generate_difference_of_squares(),
            # (x - a)(x - b) with small a, b
            lambda: self._generate_simple_factored()
        ]
        
        pattern = np.random.choice(simple_patterns)
        return pattern()
    
    def _generate_difference_of_squares(self) -> Tuple[float, float, float, float, float]:
        """Generate x² - n² = 0 type equations."""
        n = np.random.randint(1, 4)  # n = 1, 2, or 3
        
        a = 1.0
        b = 0.0
        c = -(n**2)
        x1 = float(n)
        x2 = float(-n)
        
        return a, b, c, x1, x2
    
    def _generate_simple_factored(self) -> Tuple[float, float, float, float, float]:
        """Generate simple factored equations."""
        # (x - p)(x - q) = x² - (p+q)x + pq
        p = np.random.randint(-3, 4)
        q = np.random.randint(-3, 4)
        
        while p == q:  # Ensure different roots
            q = np.random.randint(-3, 4)
        
        a = 1.0
        b = -(p + q)
        c = p * q
        x1 = float(p)
        x2 = float(q)
        
        return a, b, c, x1, x2
    
    def _generate_advanced_equation(self) -> Tuple[float, float, float, float, float]:
        """Generate more complex equations for advanced mode."""
        
        max_attempts = 100
        
        for _ in range(max_attempts):
            # Larger coefficient range
            a = np.random.randint(-self.config.advanced_max_coeff, self.config.advanced_max_coeff + 1)
            if a == 0:
                a = 1 if np.random.random() < 0.5 else -1
            
            b = np.random.randint(-self.config.advanced_max_coeff, self.config.advanced_max_coeff + 1)
            c = np.random.randint(-self.config.advanced_max_coeff, self.config.advanced_max_coeff + 1)
            
            # Calculate solutions
            x1, x2 = QuadraticSolver.solve_quadratic(a, b, c)
            
            if x1 is not None and x2 is not None:
                # Allow irrational solutions in advanced mode
                if abs(x1) <= 50 and abs(x2) <= 50:  # Keep somewhat reasonable
                    return float(a), float(b), float(c), float(x1), float(x2)
        
        # Fallback to textbook equation
        return self._generate_realistic_textbook_equation()
    
    def generate_infinite(self, progress_callback: Optional[Callable] = None, 
                         stop_event: Optional[Event] = None):
        """
        Generate equations infinitely until stopped.
        No finite mode - always infinite generation.
        """
        
        count = 0
        
        while not (stop_event and stop_event.is_set()):
            try:
                # Generate single equation
                equation = self.generate_textbook_equation()
                a, b, c, x1, x2 = equation
                
                # Add to dataset
                self.data.append([a, b, c, x1, x2])
                
                # Update statistics
                self._update_statistics(a, b, c, x1, x2)
                count += 1
                
                # Progress callback every 10 equations
                if progress_callback and count % 10 == 0:
                    progress_callback(count, self.generation_stats)
                    
            except Exception as e:
                # Continue generating even if individual equation fails
                print(f"Equation generation error: {e}")
                continue
        
        return self.data
    
    def _update_statistics(self, a: float, b: float, c: float, x1: float, x2: float):
        """Update generation statistics."""
        self.generation_stats['total_generated'] += 1
        
        # Check for perfect discriminant
        discriminant = b**2 - 4*a*c
        if discriminant >= 0 and QuadraticSolver.is_perfect_square(discriminant):
            self.generation_stats['perfect_discriminants'] += 1
        
        # Check for whole number solutions
        if (QuadraticSolver.is_whole_number(x1) and QuadraticSolver.is_whole_number(x2)):
            self.generation_stats['whole_solutions'] += 1
        
        # Check if textbook friendly (small coefficients + reasonable solutions)
        if (abs(a) <= self.config.max_coeff and abs(b) <= self.config.max_coeff and 
            abs(c) <= self.config.max_coeff and abs(x1) <= 10 and abs(x2) <= 10):
            self.generation_stats['textbook_friendly'] += 1
    
    def get_latest_equation(self) -> Optional[dict]:
        """Get the most recently generated equation with metadata."""
        if not self.data:
            return None
        
        latest = self.data[-1]
        a, b, c, x1, x2 = latest
        
        return {
            'coefficients': {'a': a, 'b': b, 'c': c},
            'solutions': {'x1': x1, 'x2': x2},
            'equation_string': QuadraticSolver.format_equation(a, b, c),
            'discriminant': b**2 - 4*a*c,
            'is_perfect_square': QuadraticSolver.is_perfect_square(b**2 - 4*a*c),
            'whole_solutions': (QuadraticSolver.is_whole_number(x1) and 
                              QuadraticSolver.is_whole_number(x2))
        }
    
    def get_statistics(self) -> dict:
        """Get comprehensive generation statistics."""
        total = self.generation_stats['total_generated']
        
        if total == 0:
            return self.generation_stats
        
        return {
            **self.generation_stats,
            'percentages': {
                'perfect_discriminants': (self.generation_stats['perfect_discriminants'] / total) * 100,
                'whole_solutions': (self.generation_stats['whole_solutions'] / total) * 100,
                'textbook_friendly': (self.generation_stats['textbook_friendly'] / total) * 100
            }
        }
    
    def save_dataset(self, filepath: str):
        """Save generated dataset to CSV file."""
        if not self.data:
            raise ValueError("No data to save")
        
        df = pd.DataFrame(self.data, columns=['a', 'b', 'c', 'x1', 'x2'])
        df.to_csv(filepath, index=False)
        
        # Save metadata
        metadata_path = filepath.replace('.csv', '_metadata.txt')
        stats = self.get_statistics()
        
        with open(metadata_path, 'w') as f:
            f.write("QUADRATIC DATASET METADATA\n")
            f.write("=" * 30 + "\n")
            f.write(f"Total equations: {stats['total_generated']}\n")
            f.write(f"Perfect discriminants: {stats['perfect_discriminants']} ({stats['percentages']['perfect_discriminants']:.1f}%)\n")
            f.write(f"Whole solutions: {stats['whole_solutions']} ({stats['percentages']['whole_solutions']:.1f}%)\n")
            f.write(f"Textbook friendly: {stats['textbook_friendly']} ({stats['percentages']['textbook_friendly']:.1f}%)\n")
            f.write(f"Mode: {'Textbook' if self.config.textbook_mode else 'Advanced'}\n")
            f.write(f"Max coefficient: {self.config.max_coeff if self.config.textbook_mode else self.config.advanced_max_coeff}\n")
    
    def get_dataframe(self) -> pd.DataFrame:
        """Get dataset as pandas DataFrame."""
        return pd.DataFrame(self.data, columns=['a', 'b', 'c', 'x1', 'x2'])
    
    def add_manual_equation(self, a: float, b: float, c: float) -> bool:
        """Add a manually specified equation if it has real solutions."""
        x1, x2 = QuadraticSolver.solve_quadratic(a, b, c)
        
        if x1 is not None and x2 is not None:
            self.data.append([a, b, c, x1, x2])
            self._update_statistics(a, b, c, x1, x2)
            return True
        
        return False


# Convenience function for quick generation
def generate_textbook_equations(count: int = 100, textbook_mode: bool = True) -> pd.DataFrame:
    """
    Quick function to generate textbook equations.
    
    Args:
        count: Number of equations to generate
        textbook_mode: True for simple textbook equations, False for advanced
    
    Returns:
        DataFrame with columns [a, b, c, x1, x2]
    """
    config = TextbookConfig(textbook_mode=textbook_mode)
    generator = TextbookQuadraticGenerator(config)
    
    # Generate specified count (simulate finite generation)
    for _ in range(count):
        equation = generator.generate_textbook_equation()
        a, b, c, x1, x2 = equation
        generator.data.append([a, b, c, x1, x2])
        generator._update_statistics(a, b, c, x1, x2)
    
    return generator.get_dataframe()
