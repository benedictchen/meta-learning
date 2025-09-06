"""
TTCS (Test-Time Compute Scaling) Modules
========================================

Modular TTCS implementation broken down from the large ttcs.py file.
Each component is now in its own module for better maintainability.

FIRST PUBLIC IMPLEMENTATION of Test-Time Compute Scaling for meta-learning!

Features:
- MC-Dropout for uncertainty estimation
- Test-Time Augmentation (TTA) for images  
- Ensemble prediction across multiple stochastic passes
- Mean probability vs mean logit combining strategies
- Performance profiling and fallback mechanisms
"""

# Core prediction functions
from .core_predictor import (
    ttcs_predict,
    ttcs_predict_advanced,
    tta_transforms,
    _enable_dropout_for_inference,
    _disable_dropout_after_inference,
    _compute_uncertainty_metrics
)

# Main TestTimeComputeScaler class
from .test_time_scaler import (
    TestTimeComputeScaler
)

# Augmentation transforms
from .augmentation_transforms import (
    create_tta_transforms,
    apply_tta_to_batch,
    create_adaptive_tta_pipeline,
    estimate_tta_diversity,
    TTAScheduler
)

# Utility functions
from .utility_functions import (
    auto_ttcs,
    pro_ttcs,
    ttcs_with_fallback,
    ttcs_for_learn2learn_models,
    get_optimal_ttcs_passes,
    estimate_ttcs_compute_cost,
    smart_ttcs_configuration,
    SimplePrototypicalHead
)

# Warning system
from .warning_system import (
    TTCSWarningSystem,
    ttcs_warn,
    suppress_ttcs_warnings,
    unsuppress_ttcs_warnings,
    get_ttcs_warning_summary,
    reset_ttcs_warnings,
    TTCSConfigurationValidator,
    ttcs_validator
)

# Profiler
from .profiler import (
    TTCSProfiler,
    ttcs_profiler
)

__all__ = [
    # Core prediction functions
    'ttcs_predict',
    'ttcs_predict_advanced', 
    'tta_transforms',
    '_enable_dropout_for_inference',
    '_disable_dropout_after_inference',
    '_compute_uncertainty_metrics',
    
    # Main class
    'TestTimeComputeScaler',
    
    # Augmentation transforms
    'create_tta_transforms',
    'apply_tta_to_batch',
    'create_adaptive_tta_pipeline',
    'estimate_tta_diversity',
    'TTAScheduler',
    
    # Utility functions
    'auto_ttcs',
    'pro_ttcs',
    'ttcs_with_fallback',
    'ttcs_for_learn2learn_models',
    'get_optimal_ttcs_passes',
    'estimate_ttcs_compute_cost',
    'smart_ttcs_configuration',
    'SimplePrototypicalHead',
    
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
    'ttcs_profiler'
]