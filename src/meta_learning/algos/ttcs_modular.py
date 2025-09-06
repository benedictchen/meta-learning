"""
Test-Time Compute Scaling (TTCS) - Modular Implementation
========================================================

FIRST PUBLIC IMPLEMENTATION of Test-Time Compute Scaling for meta-learning!

This file provides a clean interface to the modular TTCS implementation.
All components have been extracted to ttcs_modules/ for better maintainability.

Original 1274-line implementation preserved as ttcs_modules/ttcs_original_1274_lines.py

Key Features:
- Monte Carlo Dropout for uncertainty estimation
- Test-Time Augmentation (TTA) for improved robustness  
- Ensemble predictions across multiple stochastic passes
- Configurable combining strategies (mean_prob vs mean_logit)
- Performance monitoring and optimization tools
- Comprehensive warning system with deduplication
- Professional profiling capabilities
- Fallback mechanisms for robustness

Usage Examples:
    
    # Simple usage (recommended for most cases)
    from meta_learning.algos.ttcs import auto_ttcs
    logits = auto_ttcs(encoder, head, episode)
    
    # Advanced usage with uncertainty estimation
    from meta_learning.algos.ttcs import pro_ttcs
    results = pro_ttcs(encoder, head, episode, 
                      passes=16, 
                      uncertainty_estimation=True)
    
    # Object-oriented usage with configuration
    from meta_learning.algos.ttcs import TestTimeComputeScaler
    scaler = TestTimeComputeScaler(encoder, head, passes=12)
    logits = scaler.predict(episode)
    
    # Professional profiling
    from meta_learning.algos.ttcs import TTCSProfiler
    profiler = TTCSProfiler()
    results = profiler.profile_ttcs_run(encoder, head, episode, passes=10)

Scientific Background:
    Test-Time Compute Scaling is a breakthrough 2024 technique that improves
    few-shot learning performance by allocating additional computational 
    resources during inference rather than training. This implementation
    combines Monte Carlo Dropout, Test-Time Augmentation, and ensemble
    methods to achieve state-of-the-art uncertainty estimation and robustness.

Authors: Benedict Chen, 2024-2025
License: Custom Non-Commercial License with Donation Requirements
"""

# Import all components from modular implementation
from .ttcs_modules import (
    # Core prediction functions
    ttcs_predict,
    ttcs_predict_advanced,
    tta_transforms,
    
    # Main classes
    TestTimeComputeScaler,
    
    # Utility functions  
    auto_ttcs,
    pro_ttcs,
    ttcs_with_fallback,
    ttcs_for_learn2learn_models,
    get_optimal_ttcs_passes,
    estimate_ttcs_compute_cost,
    smart_ttcs_configuration,
    
    # Advanced features
    create_tta_transforms,
    apply_tta_to_batch,
    create_adaptive_tta_pipeline,
    estimate_tta_diversity,
    TTAScheduler,
    
    # Warning system
    TTCSWarningSystem,
    ttcs_warn,
    suppress_ttcs_warnings,
    unsuppress_ttcs_warnings,
    get_ttcs_warning_summary,
    reset_ttcs_warnings,
    TTCSConfigurationValidator,
    ttcs_validator,
    
    # Profiler
    TTCSProfiler,
    ttcs_profiler,
    
    # Helper classes
    SimplePrototypicalHead
)

# Backward compatibility aliases
# These ensure existing code continues to work
TTCS = TestTimeComputeScaler
ttcs = auto_ttcs

# Version and metadata
__version__ = "2.0.0"
__author__ = "Benedict Chen"
__status__ = "Production"
__implementation_type__ = "Modular"

# Export everything for backward compatibility
__all__ = [
    # Core prediction functions
    'ttcs_predict',
    'ttcs_predict_advanced', 
    'tta_transforms',
    
    # Main classes
    'TestTimeComputeScaler',
    'TTCS',  # Alias
    
    # Utility functions
    'auto_ttcs', 
    'ttcs',  # Alias
    'pro_ttcs',
    'ttcs_with_fallback',
    'ttcs_for_learn2learn_models',
    'get_optimal_ttcs_passes',
    'estimate_ttcs_compute_cost', 
    'smart_ttcs_configuration',
    
    # Advanced features
    'create_tta_transforms',
    'apply_tta_to_batch',
    'create_adaptive_tta_pipeline',
    'estimate_tta_diversity',
    'TTAScheduler',
    
    # Warning system
    'TTCSWarningSystem',
    'ttcs_warn',
    'suppress_ttcs_warnings',
    'unsuppress_ttcs_warnings', 
    'get_ttcs_warning_summary',
    'reset_ttcs_warnings',
    'TTCSConfigurationValidator',
    'ttcs_validator',
    
    # Profiler
    'TTCSProfiler',
    'ttcs_profiler',
    
    # Helper classes
    'SimplePrototypicalHead'
]

def get_ttcs_info():
    """Get comprehensive information about TTCS implementation."""
    return {
        'version': __version__,
        'implementation_type': __implementation_type__, 
        'author': __author__,
        'status': __status__,
        'total_components': len(__all__),
        'modular_structure': {
            'core_predictor': 'Core TTCS prediction functions with MC-Dropout and TTA',
            'test_time_scaler': 'Main TestTimeComputeScaler class with performance tracking',
            'augmentation_transforms': 'Advanced test-time augmentation tools',
            'utility_functions': 'Convenience functions and configuration helpers',
            'warning_system': 'Comprehensive warning system with deduplication',
            'profiler': 'Professional performance profiling and optimization'
        },
        'features': [
            'Monte Carlo Dropout for uncertainty estimation',
            'Test-Time Augmentation for improved robustness',
            'Ensemble predictions across multiple passes', 
            'Configurable combining strategies',
            'Performance monitoring and optimization',
            'Comprehensive error handling and validation',
            'Professional profiling capabilities',
            'Fallback mechanisms for robustness'
        ],
        'backward_compatibility': True,
        'original_file_lines': 1274,
        'modular_files': 6,
        'maintainability_improvement': 'Significant'
    }

# Quick info function for debugging
def ttcs_status():
    """Print TTCS implementation status."""
    info = get_ttcs_info()
    print(f"🚀 TTCS v{info['version']} ({info['implementation_type']} Implementation)")
    print(f"📦 {info['total_components']} components across {info['modular_files']} modules")
    print(f"📈 Maintainability: {info['maintainability_improvement']}")
    print(f"🔄 Backward Compatible: {info['backward_compatibility']}")
    print(f"⚡ Status: {info['status']}")

if __name__ == "__main__":
    # Demo the modular implementation
    ttcs_status()
    print("\n🔧 Available Components:")
    for component in __all__:
        print(f"  - {component}")
    print(f"\n✅ Total: {len(__all__)} components available")
    print("📚 Use help(component_name) for detailed documentation")