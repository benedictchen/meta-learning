"""
🧠 Advanced Components - Modular Implementation  
==============================================

🎯 ELI5 EXPLANATION:
==================
Think of advanced components like specialized tools in a high-tech toolbox!

Each tool has been expertly crafted and tested by top researchers:
1. 🎯 **Uncertainty Tools**: Measure how confident the AI is in its predictions
2. 📊 **MultiScale Tools**: Look at data at different zoom levels simultaneously  
3. 🏗️ **Hierarchical Tools**: Organize information in smart tree-like structures
4. 🔬 **Advanced Learning**: Use cutting-edge probabilistic AI methods

This modular approach makes it easy to pick exactly the right tool for your task,
mix and match different approaches, and experiment with state-of-the-art AI!

🔬 RESEARCH FOUNDATION:
======================
Cutting-edge few-shot learning research implemented with scientific accuracy:

**Uncertainty Estimation:**
- Gal & Ghahramani (2016): "Dropout as a Bayesian Approximation" - Monte Carlo Dropout  
- Lakshminarayanan et al. (2017): "Simple and Scalable Predictive Uncertainty" - Deep Ensembles
- Sensoy et al. (2018): "Evidential Deep Learning to Quantify Classification" - Evidential Learning

**Multi-Scale Processing:**
- Lin et al. (2017): "Feature Pyramid Networks for Object Detection" - Feature Pyramids
- Yu & Koltun (2016): "Multi-Scale Context Aggregation by Dilated Convolutions" - Dilated Conv
- Wang et al. (2018): "Non-local Neural Networks" - Attention Multi-Scale

**Hierarchical Organization:**  
- Li et al. (2019): "Learning to Compose and Reason with Language Tree Structures" - Tree Hierarchies
- Tokmakov et al. (2019): "Learning Compositional Representations" - Compositional Prototypes
- Hinton et al. (2018): "Matrix Capsules with EM Routing" - Capsule Networks

**Advanced Probabilistic AI:**
- Edwards & Storkey (2016): "Towards a Neural Statistician" - Bayesian Prototypes
- Garnelo et al. (2018): "Conditional Neural Processes" - Neural Processes

📁 MODULAR STRUCTURE:
===================
This file now serves as the main interface to the modularized components:

advanced_components_modules/
├── __init__.py                    # Module exports
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
✅ Reduced Complexity: 1,680 lines → 5 focused modules (~300-400 lines each)
✅ Better Documentation: Each module has targeted research context

🎯 RECOMMENDED USAGE:
====================
```python
# Import modular components (RECOMMENDED)
from advanced_components_modules import (
    create_uncertainty_distance,
    create_multiscale_aggregator, 
    create_hierarchical_prototypes
)

# Quick setup with factory functions
uncertainty_dist = create_uncertainty_distance("monte_carlo_dropout")
multiscale = create_multiscale_aggregator("feature_pyramid")
hierarchical = create_hierarchical_prototypes("tree_structured")

# Advanced usage with full configuration
from advanced_components_modules.configs import UncertaintyAwareDistanceConfig

config = UncertaintyAwareDistanceConfig(
    uncertainty_method="evidential_deep_learning",
    evidential_num_classes=5,
    evidential_lambda_reg=0.02
)
uncertainty_dist = UncertaintyAwareDistance(config)
```

💰 SUPPORT THIS RESEARCH:
=========================
🙏 If this modular implementation helps your research:
💳 PayPal: https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS
💖 GitHub Sponsors: https://github.com/sponsors/benedictchen

Author: Benedict Chen (benedict@benedictchen.com)
Research-accurate implementations of cutting-edge few-shot learning methods
"""

# Import all components from the modular structure
from .advanced_components_modules import (
    # Configuration classes
    UncertaintyAwareDistanceConfig,
    MultiScaleFeatureAggregatorConfig, 
    HierarchicalPrototypesConfig,
    EvidentialLearningConfig,
    BayesianPrototypesConfig,
    
    # Component classes
    UncertaintyAwareDistance,
    MultiScaleFeatureAggregator,
    HierarchicalPrototypes,
    EvidentialLearning,
    BayesianPrototypes,
    
    # Factory functions (RECOMMENDED for most users)
    create_uncertainty_distance,
    create_multiscale_aggregator,
    create_hierarchical_prototypes,
    create_evidential_learner,
    create_bayesian_prototypes,
)

# Import additional utility classes from modules
from .advanced_components_modules.multiscale import PrototypeRefiner
from .advanced_components_modules.hierarchical import (
    UncertaintyEstimator, 
    ScaledDotProductAttention,
    AdditiveAttention,
    BilinearAttention, 
    GraphRelationModule,
    StandardRelationModule,
    TaskAdaptivePrototypes
)

# Backward compatibility exports
__all__ = [
    # Configuration classes
    'UncertaintyAwareDistanceConfig',
    'MultiScaleFeatureAggregatorConfig', 
    'HierarchicalPrototypesConfig',
    'EvidentialLearningConfig',
    'BayesianPrototypesConfig',
    
    # Component classes (for advanced users)
    'UncertaintyAwareDistance',
    'MultiScaleFeatureAggregator',
    'HierarchicalPrototypes',
    'EvidentialLearning',
    'BayesianPrototypes',
    
    # Utility classes (backward compatibility)
    'PrototypeRefiner',
    'UncertaintyEstimator',
    'ScaledDotProductAttention',
    'AdditiveAttention',
    'BilinearAttention',
    'GraphRelationModule', 
    'StandardRelationModule',
    'TaskAdaptivePrototypes',
    
    # Factory functions (recommended for most users)
    'create_uncertainty_distance',
    'create_multiscale_aggregator',
    'create_hierarchical_prototypes',
    'create_evidential_learner',
    'create_bayesian_prototypes',
]

def print_modular_info():
    """Print information about the modular structure."""
    print("🧠 Advanced Components - Modular Implementation")
    print("=" * 47)
    print()
    print("📁 MODULAR STRUCTURE:")
    print("   📋 configs.py         - Configuration dataclasses for all components")
    print("   🎯 uncertainty.py     - Uncertainty estimation (MC Dropout, Ensembles, Evidential)")
    print("   📊 multiscale.py      - Multi-scale processing (FPN, Dilated Conv, Attention)")
    print("   🏗️ hierarchical.py    - Hierarchical prototypes (Tree, Compositional, Capsule)")
    print("   🏭 factory.py         - Factory functions for easy component creation")
    print()
    print("🚀 MODULARIZATION BENEFITS:")
    print("   • Reduced from 1,680 lines to 5 focused modules")
    print("   • Each module ~300-400 lines with single responsibility")
    print("   • Better maintainability and testing")
    print("   • Easier to extend with new methods")
    print("   • Improved documentation and research context")
    print()
    print("🎯 RECOMMENDED USAGE:")
    print("   from advanced_components_modules import create_uncertainty_distance")
    print("   estimator = create_uncertainty_distance('monte_carlo_dropout')")
    print("   result = estimator(query_features, prototypes)")


if __name__ == "__main__":
    print_modular_info()