"""
Training configuration for universal character recognition.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class TrainingConfig:
    """Configuration for training pipeline."""
    
    # Architecture
    layer_sizes: List[int] = None
    activations: List[str] = None
    
    # Training parameters
    epochs: int = 250
    batch_size: int = 64
    learning_rate: float = 0.001
    validation_split: float = 0.1
    
    # Learning rate schedule
    lr_schedule_type: str = 'cosine'  # 'cosine', 'step', 'exponential', 'constant'
    lr_schedule_params: dict = None
    
    # Early stopping
    early_stopping_patience: int = 15
    early_stopping_min_delta: float = 0.0001
    
    # Gradient clipping
    max_grad_norm: float = 5.0
    
    # Data augmentation phases
    augmentation_phases: dict = None
    
    # Checkpointing
    checkpoint_dir: str = 'models/checkpoints'
    checkpoint_freq: int = 10
    save_best: bool = True
    
    # Training phases
    phases: List[dict] = None
    
    def __post_init__(self):
        """Set defaults if not provided."""
        if self.layer_sizes is None:
            self.layer_sizes = [784, 512, 256, 128, 62]
        
        if self.activations is None:
            self.activations = ['relu', 'relu', 'relu', 'softmax']
        
        if self.lr_schedule_params is None:
            self.lr_schedule_params = {'T_max': 100, 'eta_min': 0.0}
        
        if self.augmentation_phases is None:
            self.augmentation_phases = {
                'full': {'mirror_prob': 0.5, 'rotation_range': 15.0},
                'reduced': {'mirror_prob': 0.3, 'rotation_range': 10.0},
                'minimal': {'mirror_prob': 0.2, 'rotation_range': 5.0},
                'none': {'mirror_prob': 0.0, 'rotation_range': 0.0}
            }
        
        if self.phases is None:
            self.phases = [
                {
                    'name': 'Phase 1: Fast Learning',
                    'epochs': 100,
                    'learning_rate': 0.001,
                    'augmentation': 'full'
                },
                {
                    'name': 'Phase 2: Fine-tuning',
                    'epochs': 50,
                    'learning_rate': 0.0005,
                    'augmentation': 'reduced'
                },
                {
                    'name': 'Phase 3: Optimization',
                    'epochs': 50,
                    'learning_rate': 0.0001,
                    'augmentation': 'minimal'
                },
                {
                    'name': 'Phase 4: Ultra-fine-tuning',
                    'epochs': 50,
                    'learning_rate': 0.00005,
                    'augmentation': 'none'
                }
            ]


def get_default_config() -> TrainingConfig:
    """Get default training configuration."""
    return TrainingConfig()


def get_high_accuracy_config() -> TrainingConfig:
    """Get configuration optimized for high accuracy."""
    config = TrainingConfig()
    config.epochs = 300
    config.batch_size = 128
    config.learning_rate = 0.001
    config.early_stopping_patience = 20
    config.max_grad_norm = 3.0
    
    config.phases = [
        {
            'name': 'Phase 1: Fast Learning',
            'epochs': 120,
            'learning_rate': 0.001,
            'augmentation': 'full'
        },
        {
            'name': 'Phase 2: Fine-tuning',
            'epochs': 60,
            'learning_rate': 0.0005,
            'augmentation': 'reduced'
        },
        {
            'name': 'Phase 3: Optimization',
            'epochs': 60,
            'learning_rate': 0.0001,
            'augmentation': 'minimal'
        },
        {
            'name': 'Phase 4: Ultra-fine-tuning',
            'epochs': 60,
            'learning_rate': 0.00005,
            'augmentation': 'none'
        }
    ]
    
    return config

