"""
Core dataset generation logic for quadratic equations.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Callable, Any
import time
from threading import Event

from config import GenerationConfig, QuadraticSolver


class DatasetGenerator:
    """Core dataset generation engine for quadratic equations."""
    
    def __init__(self, config: GenerationConfig):
        self.config = config
        self.data = []
        self.generation_stats = {
            'total_generated': 0,
            'whole_solutions': 0,
            'real_solutions': 0,
            'complex_solutions': 0,
            'rejected': 0
        }
        
        # Systematic generation state
        self.systematic_state = {
            'a_current': int(self.config.a_min) if self.config.integer_coefficients_only else self.config.a_min,
            'b_current': int(self.config.b_min) if self.config.integer_coefficients_only else self.config.b_min,
            'c_current': int(self.config.c_min) if self.config.integer_coefficients_only else self.config.c_min,
            'completed_cycle': False
        }
    
    def reset_stats(self):
        """Reset generation statistics."""
        self.generation_stats = {
            'total_generated': 0,
            'whole_solutions': 0,
            'real_solutions': 0,
            'complex_solutions': 0,
            'rejected': 0
        }
    
    def generate_coefficients(self) -> Tuple[float, float, float]:
        """Generate coefficients based on current configuration."""
        if self.config.systematic_generation:
            return self.generate_systematic_coefficients()
        
        if self.config.integer_coefficients_only:
            a = float(np.random.randint(int(self.config.a_min), int(self.config.a_max) + 1))
            b = float(np.random.randint(int(self.config.b_min), int(self.config.b_max) + 1))
            c = float(np.random.randint(int(self.config.c_min), int(self.config.c_max) + 1))
        else:
            a = np.random.uniform(self.config.a_min, self.config.a_max)
            b = np.random.uniform(self.config.b_min, self.config.b_max)
            c = np.random.uniform(self.config.c_min, self.config.c_max)
        
        if self.config.avoid_zero_a and a == 0:
            a = 1.0 if np.random.random() < 0.5 else -1.0
        
        return a, b, c
    
    def generate_systematic_coefficients(self) -> Tuple[float, float, float]:
        """Generate coefficients systematically for comprehensive coverage."""
        a = self.systematic_state['a_current']
        b = self.systematic_state['b_current']
        c = self.systematic_state['c_current']
        
        # Advance to next combination
        if self.config.integer_coefficients_only:
            # Integer systematic generation
            c += 1
            if c > int(self.config.c_max):
                c = int(self.config.c_min)
                b += 1
                if b > int(self.config.b_max):
                    b = int(self.config.b_min)
                    a += 1
                    if a > int(self.config.a_max):
                        a = int(self.config.a_min)
                        self.systematic_state['completed_cycle'] = True
        else:
            # Non-integer systematic generation with smaller steps
            step = 0.5
            c += step
            if c > self.config.c_max:
                c = self.config.c_min
                b += step
                if b > self.config.b_max:
                    b = self.config.b_min
                    a += step
                    if a > self.config.a_max:
                        a = self.config.a_min
                        self.systematic_state['completed_cycle'] = True
        
        # Update systematic state
        self.systematic_state['a_current'] = a
        self.systematic_state['b_current'] = b
        self.systematic_state['c_current'] = c
        
        # Handle zero 'a' coefficient
        if self.config.avoid_zero_a and a == 0:
            a = 1.0
        
        return float(a), float(b), float(c)
    
    def generate_whole_solution_equation(self) -> Tuple[float, float, float, float, float]:
        """Generate equation guaranteed to have at least one whole number solution."""
        # Generate whole number solutions first
        x1 = np.random.randint(int(self.config.solution_min), int(self.config.solution_max) + 1)
        
        if self.config.symmetric_solutions:
            x2 = -x1
        else:
            x2 = np.random.randint(int(self.config.solution_min), int(self.config.solution_max) + 1)
        
        # Generate coefficients from solutions using: a(x-x1)(x-x2) = ax² - a(x1+x2)x + ax1x2
        if self.config.integer_coefficients_only:
            a = float(np.random.randint(int(self.config.a_min), int(self.config.a_max) + 1))
        else:
            a = np.random.uniform(self.config.a_min, self.config.a_max)
        
        if self.config.avoid_zero_a and a == 0:
            a = 1.0
        
        b = -a * (x1 + x2)
        c = a * x1 * x2
        
        # Ensure integer coefficients if required
        if self.config.integer_coefficients_only:
            b = float(round(b))
            c = float(round(c))
        
        return a, b, c, x1, x2

    def generate_textbook_equation(self) -> Tuple[float, float, float, float, float]:
        """Generate textbook-style quadratic equation with reasonable coefficients."""
        
        if self.config.textbook_simple_solutions and np.random.random() < self.config.textbook_integer_solutions_ratio:
            return self.generate_simple_integer_solution_equation()
        else:
            return self.generate_perfect_discriminant_equation()

    def generate_simple_integer_solution_equation(self) -> Tuple[float, float, float, float, float]:
        """Generate equation by starting with simple integer solutions."""
        
        # Choose simple integer solutions
        x1 = np.random.randint(-5, 6)  # Solutions between -5 and 5
        x2 = np.random.randint(-5, 6)
        
        # Ensure solutions are different and not zero
        while x2 == x1 or x1 == 0:
            x2 = np.random.randint(-5, 6)
        
        # Choose a simple leading coefficient
        a = np.random.choice([1, 2, 3, -1, -2, -3])
        
        # Calculate b and c from the factored form: a(x - x1)(x - x2)
        # Expanding: a(x² - (x1+x2)x + x1*x2) = ax² - a(x1+x2)x + a*x1*x2
        b = -a * (x1 + x2)
        c = a * x1 * x2
        
        return float(a), float(b), float(c), float(x1), float(x2)

    def generate_perfect_discriminant_equation(self) -> Tuple[float, float, float, float, float]:
        """Generate equation with perfect square discriminant."""
        
        max_attempts = 50
        
        for _ in range(max_attempts):
            # Small, reasonable coefficients
            a = np.random.randint(1, self.config.textbook_max_coeff + 1)
            if np.random.random() < 0.3:  # 30% chance of negative 'a'
                a = -a
                
            b = np.random.randint(-self.config.textbook_max_coeff, self.config.textbook_max_coeff + 1)
            c = np.random.randint(-self.config.textbook_max_coeff, self.config.textbook_max_coeff + 1)
            
            # Calculate discriminant
            discriminant = b**2 - 4*a*c
            
            if discriminant >= 0:
                sqrt_discriminant = np.sqrt(discriminant)
                
                # Check if discriminant is a perfect square (for textbook friendliness)
                if self.config.textbook_prefer_perfect_discriminant:
                    if abs(sqrt_discriminant - round(sqrt_discriminant)) > 1e-10:
                        continue  # Skip if not a perfect square
                
                # Calculate solutions
                x1 = (-b + sqrt_discriminant) / (2*a)
                x2 = (-b - sqrt_discriminant) / (2*a)
                
                # Prefer simpler solutions
                if abs(x1) > 10 or abs(x2) > 10:
                    continue
                    
                return float(a), float(b), float(c), float(x1), float(x2)
        
        # Fallback to simple integer solution if perfect discriminant fails
        return self.generate_simple_integer_solution_equation()

    def generate_textbook_coefficients(self) -> Tuple[float, float, float]:
        """Generate small, textbook-appropriate coefficients."""
        
        a = np.random.randint(1, self.config.textbook_max_coeff + 1)
        if np.random.random() < 0.2:  # 20% chance of negative 'a'
            a = -a
            
        b = np.random.randint(-self.config.textbook_max_coeff, self.config.textbook_max_coeff + 1)
        c = np.random.randint(-self.config.textbook_max_coeff, self.config.textbook_max_coeff + 1)
        
        return float(a), float(b), float(c)

    def generate_single_equation(self) -> Optional[Tuple[float, float, float, float, float]]:
        """Generate a single quadratic equation with solutions."""
        max_attempts = 100
        
        for _ in range(max_attempts):
            # Check for textbook mode first
            if self.config.textbook_mode:
                try:
                    result = self.generate_textbook_equation()
                    if result:
                        a, b, c, x1, x2 = result
                        
                        # Validate the result
                        if self.config.only_whole_solutions:
                            if not (QuadraticSolver.is_whole_number(x1) or 
                                QuadraticSolver.is_whole_number(x2)):
                                self.generation_stats['rejected'] += 1
                                continue
                        
                        return result
                except Exception:
                    continue
            
            # Decide generation method for non-textbook mode
            if (self.config.prioritize_whole_solutions and 
                np.random.random() < self.config.whole_solution_ratio):
                try:
                    result = self.generate_whole_solution_equation()
                    if result:
                        a, b, c, x1, x2 = result
                        
                        # Check if we need to validate the result
                        if self.config.only_whole_solutions:
                            if not (QuadraticSolver.is_whole_number(x1) or 
                                QuadraticSolver.is_whole_number(x2)):
                                self.generation_stats['rejected'] += 1
                                continue
                        
                        return result
                except:
                    continue
            
            # Regular generation
            a, b, c = self.generate_coefficients()
            x1, x2 = QuadraticSolver.solve_quadratic(a, b, c)
            
            if x1 is None or x2 is None:
                if self.config.allow_complex_solutions:
                    # Handle complex solutions (placeholder)
                    continue
                else:
                    self.generation_stats['rejected'] += 1
                    continue
            
            # Check solution bounds
            if (x1 < self.config.solution_min or x1 > self.config.solution_max or
                x2 < self.config.solution_min or x2 > self.config.solution_max):
                self.generation_stats['rejected'] += 1
                continue
            
            # Check for whole number solutions if required
            if self.config.only_whole_solutions:
                if not (QuadraticSolver.is_whole_number(x1) or 
                    QuadraticSolver.is_whole_number(x2)):
                    self.generation_stats['rejected'] += 1
                    continue
            
            return a, b, c, x1, x2
        
        return None

    
    def generate_dataset(self, progress_callback: Optional[Callable] = None, 
                        stop_event: Optional[Event] = None) -> List[List[float]]:
        """
        Generate complete dataset with optional progress tracking.
        
        Args:
            progress_callback: Optional callback function for progress updates
            stop_event: Optional threading event to stop generation
            
        Returns:
            List of generated equations as [a, b, c, x1, x2] lists
        """
        self.data = []
        self.reset_stats()
        
        start_time = time.time()
        
        if self.config.infinite_mode:
            return self._generate_infinite_dataset(progress_callback, stop_event)
        else:
            return self._generate_finite_dataset(progress_callback, stop_event)
    
    def _generate_infinite_dataset(self, progress_callback: Optional[Callable], 
                              stop_event: Optional[Event]) -> List[List[float]]:
        """Generate dataset in infinite mode."""
        count = 0
        
        while not (stop_event and stop_event.is_set()):
            equation = self.generate_single_equation()
            if equation:
                a, b, c, x1, x2 = equation
                self.data.append([a, b, c, x1, x2])
                
                self._update_statistics(x1, x2)
                count += 1
                
                if progress_callback and count % 10 == 0:
                    progress_callback(count, float('inf'), self.generation_stats)
            
            # For textbook mode, don't check systematic generation completion
            # as it should run indefinitely until manually stopped
            if (self.config.systematic_generation and 
                not self.config.textbook_mode and  # NEW: Don't stop if textbook mode
                self.systematic_state['completed_cycle']):
                break
        
        return self.data

    
    def _generate_finite_dataset(self, progress_callback: Optional[Callable], 
                                stop_event: Optional[Event]) -> List[List[float]]:
        """Generate dataset in finite mode."""
        for i in range(self.config.dataset_size):
            if stop_event and stop_event.is_set():
                break
            
            equation = self.generate_single_equation()
            if equation:
                a, b, c, x1, x2 = equation
                self.data.append([a, b, c, x1, x2])
                
                self._update_statistics(x1, x2)
                
                if progress_callback:
                    progress_callback(i + 1, self.config.dataset_size, self.generation_stats)
        
        return self.data
    
    def _update_statistics(self, x1: float, x2: float):
        """Update generation statistics."""
        self.generation_stats['total_generated'] += 1
        
        if (QuadraticSolver.is_whole_number(x1) or 
            QuadraticSolver.is_whole_number(x2)):
            self.generation_stats['whole_solutions'] += 1
        
        self.generation_stats['real_solutions'] += 1
    
    def add_equation(self, a: float, b: float, c: float, x1: float, x2: float):
        """Add a manually created equation to the dataset."""
        # Verify the equation is correct
        if (QuadraticSolver.verify_solution(a, b, c, x1) and 
            QuadraticSolver.verify_solution(a, b, c, x2)):
            self.data.append([a, b, c, x1, x2])
            self._update_statistics(x1, x2)
            return True
        return False
    
    def save_dataset(self, filepath: str):
        """Save dataset to CSV file."""
        if not self.data:
            raise ValueError("No data to save")
        
        df = pd.DataFrame(self.data, columns=['a', 'b', 'c', 'x1', 'x2'])
        df.to_csv(filepath, index=False)
    
    def get_dataframe(self) -> pd.DataFrame:
        """Get dataset as pandas DataFrame."""
        return pd.DataFrame(self.data, columns=['a', 'b', 'c', 'x1', 'x2'])
    
    def get_statistics(self) -> dict:
        """Get current generation statistics."""
        return self.generation_stats.copy()
    
    def clear_data(self):
        """Clear all generated data."""
        self.data = []
        self.reset_stats()
    
    def get_data_summary(self) -> dict:
        """Get summary statistics of generated data."""
        if not self.data:
            return {}
        
        df = self.get_dataframe()
        
        summary = {
            'total_equations': len(self.data),
            'coefficient_stats': {
                'a': {'min': df['a'].min(), 'max': df['a'].max(), 'mean': df['a'].mean()},
                'b': {'min': df['b'].min(), 'max': df['b'].max(), 'mean': df['b'].mean()},
                'c': {'min': df['c'].min(), 'max': df['c'].max(), 'mean': df['c'].mean()}
            },
            'solution_stats': {
                'x1': {'min': df['x1'].min(), 'max': df['x1'].max(), 'mean': df['x1'].mean()},
                'x2': {'min': df['x2'].min(), 'max': df['x2'].max(), 'mean': df['x2'].mean()}
            },
            'generation_stats': self.generation_stats
        }
        
        return summary
