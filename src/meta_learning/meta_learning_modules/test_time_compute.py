"""
🧪 Test Time Compute
=====================

🔬 Research Foundation:  
======================
Based on meta-learning and few-shot learning research:
- Finn, C., Abbeel, P. & Levine, S. (2017). "Model-Agnostic Meta-Learning for Fast Adaptation"
- Snell, J., Swersky, K. & Zemel, R. (2017). "Prototypical Networks for Few-shot Learning"
- Nichol, A., Achiam, J. & Schulman, J. (2018). "On First-Order Meta-Learning Algorithms"
🎯 ELI5 Summary:
This is like a quality control checker for our code! Just like how you might test 
if your bicycle brakes work before riding down a hill, this file tests if our algorithms 
work correctly before we use them for real research. It runs the code with known inputs 
and checks if we get the expected outputs.

🧪 Technical Details:
===================
Implementation details and technical specifications for this component.
Designed to work seamlessly within the research framework while
maintaining high performance and accuracy standards.

🧪 Testing Process Flow:
========================
Input Data → Algorithm → Expected Output
    ↓             ↓             ↓
[Test Cases] [Run Code]  [Check Results]
    ↓             ↓             ↓
   📊            ⚙️            ✅
    
Success: ✅ All tests pass
Failure: ❌ Fix and retest

"""
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