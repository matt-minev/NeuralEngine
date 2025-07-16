from dataclasses import dataclass
from typing import List

@dataclass
class PredictionScenario:
    """Configuration for different prediction scenarios"""
    name: str
    description: str
    input_features: List[str]
    target_features: List[str]
    input_indices: List[int]
    target_indices: List[int]
    network_architecture: List[int]
    activations: List[str]
    color: str

def get_default_scenarios():
    """Get default prediction scenarios"""
    return {
        'coeff_to_roots': PredictionScenario(
            name="Coefficients → Roots",
            description="Given a, b, c predict x1, x2",
            input_features=['a', 'b', 'c'],
            target_features=['x1', 'x2'],
            input_indices=[0, 1, 2],
            target_indices=[3, 4],
            network_architecture=[3, 16, 32, 16, 2],
            activations=['relu', 'relu', 'relu', 'linear'],
            color='#FF6B6B'
        ),
        'partial_coeff_to_missing': PredictionScenario(
            name="Partial Coefficients → Missing",
            description="Given a, b, x1 predict c, x2",
            input_features=['a', 'b', 'x1'],
            target_features=['c', 'x2'],
            input_indices=[0, 1, 3],
            target_indices=[2, 4],
            network_architecture=[3, 20, 24, 12, 2],
            activations=['relu', 'swish', 'relu', 'linear'],
            color='#4ECDC4'
        ),
        'roots_to_coeff': PredictionScenario(
            name="Roots → Coefficients",
            description="Given x1, x2 predict a, b, c",
            input_features=['x1', 'x2'],
            target_features=['a', 'b', 'c'],
            input_indices=[3, 4],
            target_indices=[0, 1, 2],
            network_architecture=[2, 20, 32, 20, 3],
            activations=['relu', 'swish', 'relu', 'linear'],
            color='#45B7D1'
        ),
        'single_missing': PredictionScenario(
            name="Single Missing Parameter",
            description="Given a, b, c, x1 predict x2",
            input_features=['a', 'b', 'c', 'x1'],
            target_features=['x2'],
            input_indices=[0, 1, 2, 3],
            target_indices=[4],
            network_architecture=[4, 24, 32, 16, 1],
            activations=['relu', 'swish', 'relu', 'linear'],
            color='#96CEB4'
        ),
        'verify_equation': PredictionScenario(
            name="Equation Verification",
            description="Given all parameters predict error",
            input_features=['a', 'b', 'c', 'x1', 'x2'],
            target_features=['error'],
            input_indices=[0, 1, 2, 3, 4],
            target_indices=[5],
            network_architecture=[5, 32, 24, 16, 1],
            activations=['relu', 'swish', 'relu', 'sigmoid'],
            color='#FFEAA7'
        )
    }
