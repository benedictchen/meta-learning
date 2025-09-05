"""
🧠 Advanced Components Modules - Modular Implementation
=====================================================

🎯 ELI5 EXPLANATION:
==================
Think of advanced components like specialized tools in a toolbox!

Each tool has a specific job:
1. 🎯 **Uncertainty Tools**: Measure how confident we are in our predictions
2. 📊 **MultiScale Tools**: Look at data at different zoom levels
3. 🏗️ **Hierarchical Tools**: Organize information in layers
4. ⚙️ **Config Tools**: Settings for each specialized tool

This modular approach makes it easy to pick the right tool for your task!

📁 MODULAR STRUCTURE:
===================
advanced_components_modules/
├── __init__.py                    # This file - module exports
├── configs.py                     # Configuration dataclasses
├── uncertainty.py                 # Uncertainty-aware components  
├── multiscale.py                  # Multi-scale feature processing
├── hierarchical.py                # Hierarchical prototype systems
└── factory.py                     # Factory functions for easy creation

🚀 BENEFITS OF MODULARIZATION:
==============================
✅ Improved Maintainability: Each component type in its own file
✅ Better Testing: Individual modules can be tested independently
✅ Enhanced Readability: Focused, single-responsibility components
✅ Easier Extension: Add new components without touching existing code
✅ Reduced Complexity: 1,680 lines → 4 focused modules (~400 lines each)
✅ Better Documentation: Each module has targeted research context
"""

# Import all components from the modular structure

# Configuration classes
from .configs import (
    UncertaintyAwareDistanceConfig,
    MultiScaleFeatureAggregatorConfig,
    HierarchicalPrototypesConfig,
    EvidentialLearningConfig,
    BayesianPrototypesConfig,
)

# Uncertainty-aware components
from .uncertainty import (
    UncertaintyAwareDistance,
    EvidentialLearning,
    BayesianPrototypes,
)

# Multi-scale components
from .multiscale import (
    MultiScaleFeatureAggregator,
)

# Hierarchical components
from .hierarchical import (
    HierarchicalPrototypes,
)

# Factory functions
from .factory import (
    create_uncertainty_distance,
    create_multiscale_aggregator,
    create_hierarchical_prototypes,
    create_evidential_learner,
    create_bayesian_prototypes,
)

# Backward compatibility exports
__all__ = [
    # Configuration classes
    'UncertaintyAwareDistanceConfig',
    'MultiScaleFeatureAggregatorConfig', 
    'HierarchicalPrototypesConfig',
    'EvidentialLearningConfig',
    'BayesianPrototypesConfig',
    
    # Component classes
    'UncertaintyAwareDistance',
    'MultiScaleFeatureAggregator',
    'HierarchicalPrototypes',
    'EvidentialLearning',
    'BayesianPrototypes',
    
    # Factory functions (recommended for most users)
    'create_uncertainty_distance',
    'create_multiscale_aggregator',
    'create_hierarchical_prototypes',
    'create_evidential_learner',
    'create_bayesian_prototypes',
]

print("""
💰 MODULE SUPPORT - Made possible by Benedict Chen
   ]8;;mailto:benedict@benedictchen.com\benedict@benedictchen.com]8;;\

💰 PLEASE DONATE! Your support keeps this research alive! 💰
   🔗 ]8;;https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS\💳 CLICK HERE TO DONATE VIA PAYPAL]8;;\
   ❤️ ]8;;https://github.com/sponsors/benedictchen\💖 SPONSOR ON GITHUB]8;;\

   ☕ Buy me a coffee → 🍺 Buy me a beer → 🏎️ Buy me a Lamborghini → ✈️ Buy me a private jet!
   (Start small, dream big! Every donation helps! 😄)
""")
