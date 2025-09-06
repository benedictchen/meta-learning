"""
Gradient Transform Utilities - Modular Implementation
====================================================

Advanced gradient transform classes for learnable optimization,
including scaling, bias, and composite transforms for meta-learning.

This file provides a clean interface to the modular transforms implementation.
All components have been extracted to transforms_modules/ for better maintainability.

Original 1152-line implementation preserved as transforms_original_1152_lines.py

Key Features:
- Learnable gradient scaling and bias transforms
- Momentum-based gradient accumulation
- Adaptive parameter transforms
- Composite transform chaining
- Gradient noise injection for regularization
- Temperature-based gradient scaling
- Professional optimization framework
- Comprehensive performance monitoring

Usage Examples:
    
    # Simple scaling transform
    from meta_learning.optimization.transforms import ScaleTransform
    scale_transform = ScaleTransform(parameter_shapes)
    
    # Composite transform with multiple components
    from meta_learning.optimization.transforms import CompositeTransform
    composite = CompositeTransform([scale_transform, bias_transform])
    
    # Transform optimization
    from meta_learning.optimization.transforms import TransformOptimizer
    optimizer = TransformOptimizer([composite], learning_rate=1e-3)

Scientific Background:
    Learnable gradient transforms enable meta-learning algorithms to adapt
    their optimization dynamics based on task characteristics. This modular
    implementation provides flexible building blocks for constructing
    sophisticated meta-optimizers.

Authors: Benedict Chen, 2024-2025
License: Custom Non-Commercial License with Donation Requirements
"""

# Import all components from modular implementation
from .transforms_modules import (
    # Base class
    GradientTransform,
    
    # Individual transforms
    ScaleTransform,
    BiasTransform, 
    MomentumTransform,
    AdaptiveTransform,
    CompositeTransform,
    NoiseTransform,
    TemperatureTransform,
    
    # Optimizer
    TransformOptimizer
)

# Backward compatibility aliases
GradTransform = GradientTransform  # Legacy alias

# Version and metadata
__version__ = "2.0.0"
__author__ = "Benedict Chen"
__status__ = "Production"
__implementation_type__ = "Modular"

# Export everything for backward compatibility
__all__ = [
    'GradientTransform',
    'GradTransform',  # Legacy alias
    'ScaleTransform',
    'BiasTransform',
    'MomentumTransform', 
    'AdaptiveTransform',
    'CompositeTransform',
    'NoiseTransform',
    'TemperatureTransform',
    'TransformOptimizer'
]

def get_transforms_info():
    """Get comprehensive information about transforms implementation."""
    return {
        'version': __version__,
        'implementation_type': __implementation_type__,
        'author': __author__,
        'status': __status__,
        'total_components': len(__all__),
        'modular_structure': {
            'base_transform': 'Abstract base class for all gradient transforms',
            'scale_transform': 'Learnable per-parameter or global scaling',
            'bias_transform': 'Learnable bias addition with momentum',
            'momentum_transform': 'Momentum-based gradient accumulation',
            'adaptive_transform': 'Adaptive normalization based on statistics',
            'composite_transform': 'Composition of multiple transforms',
            'noise_transform': 'Gradient noise injection for regularization',
            'temperature_transform': 'Temperature-based gradient scaling',
            'transform_optimizer': 'Optimizer for transform parameters'
        },
        'features': [
            'Learnable gradient scaling and bias transforms',
            'Momentum-based gradient accumulation', 
            'Adaptive parameter transforms',
            'Composite transform chaining',
            'Gradient noise injection for regularization',
            'Temperature-based gradient scaling',
            'Professional optimization framework',
            'Comprehensive performance monitoring'
        ],
        'backward_compatibility': True,
        'original_file_lines': 1152,
        'modular_files': 9,
        'maintainability_improvement': 'Significant'
    }

# Quick info function for debugging
def transforms_status():
    """Print transforms implementation status."""
    info = get_transforms_info()
    print(f"🚀 Transforms v{info['version']} ({info['implementation_type']} Implementation)")
    print(f"📦 {info['total_components']} components across {info['modular_files']} modules")
    print(f"📈 Maintainability: {info['maintainability_improvement']}")
    print(f"🔄 Backward Compatible: {info['backward_compatibility']}")
    print(f"⚡ Status: {info['status']}")

if __name__ == "__main__":
    # Demo the modular implementation
    transforms_status()
    print("\n🔧 Available Transforms:")
    for component in __all__:
        print(f"  - {component}")
    print(f"\n✅ Total: {len(__all__)} components available")
    print("📚 Use help(component_name) for detailed documentation")