"""
Optimization Transforms Modules
==============================

Modular gradient transform implementation for learnable optimization.
Extracted from large transforms.py for better maintainability.

Classes:
- GradientTransform: Abstract base class for all transforms
- ScaleTransform: Learnable scaling transform
- BiasTransform: Learnable bias addition
- MomentumTransform: Learnable momentum-based transform
- AdaptiveTransform: Adaptive parameter transforms
- CompositeTransform: Composition of multiple transforms
- NoiseTransform: Gradient noise injection
- TemperatureTransform: Temperature-based gradient scaling
- TransformOptimizer: Optimizer for transform parameters
"""

# Base classes
from .base_transform import GradientTransform

# Individual transforms
from .scale_transform import ScaleTransform
from .bias_transform import BiasTransform
from .momentum_transform import MomentumTransform
from .adaptive_transform import AdaptiveTransform
from .composite_transform import CompositeTransform
from .noise_transform import NoiseTransform
from .temperature_transform import TemperatureTransform

# Optimizer
from .transform_optimizer import TransformOptimizer

__all__ = [
    'GradientTransform',
    'ScaleTransform',
    'BiasTransform', 
    'MomentumTransform',
    'AdaptiveTransform',
    'CompositeTransform',
    'NoiseTransform',
    'TemperatureTransform',
    'TransformOptimizer'
]