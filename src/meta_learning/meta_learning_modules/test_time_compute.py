"""
Test-Time Compute Scaling for Meta-Learning - Compatibility Layer
================================================================

This module provides backward compatibility by re-exporting all functionality
from the modular test_time_compute implementation.

Author: Benedict Chen (benedict@benedictchen.com)
License: Custom Non-Commercial License with Donation Requirements
"""

# Import everything from the modular interface
from .test_time_compute_modular import *

# Ensure all expected symbols are available
__all__ = [
    # Strategy enums
    'TestTimeComputeStrategy',
    'StateFallbackMethod',
    'StateForwardMethod', 
    'VerificationFallbackMethod',
    
    # Configuration
    'TestTimeComputeConfig',
    
    # Core implementation
    'TestTimeComputeScaler',
    
    # Factory functions
    'create_process_reward_config',
    'create_consistency_verification_config',
    'create_gradient_verification_config',
    'create_attention_reasoning_config',
    'create_feature_reasoning_config',
    'create_prototype_reasoning_config',
    'create_comprehensive_config',
    'create_fast_config'
]