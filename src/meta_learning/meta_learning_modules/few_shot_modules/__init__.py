"""
📋   Init  
============

🔬 Research Foundation:  
======================
Based on meta-learning and few-shot learning research:
- Finn, C., Abbeel, P. & Levine, S. (2017). "Model-Agnostic Meta-Learning for Fast Adaptation"
- Snell, J., Swersky, K. & Zemel, R. (2017). "Prototypical Networks for Few-shot Learning"
- Nichol, A., Achiam, J. & Schulman, J. (2018). "On First-Order Meta-Learning Algorithms"
🎯 ELI5 Summary:
This file is an important component in our AI research system! Like different organs 
in your body that work together to keep you healthy, this file has a specific job that 
helps the overall algorithm work correctly and efficiently.

🧪 Technical Details:
===================
Implementation details and technical specifications for this component.
Designed to work seamlessly within the research framework while
maintaining high performance and accuracy standards.

📋 Component Integration:
========================
    ┌──────────┐
    │   This   │
    │Component │ ←→ Other Components
    └──────────┘
         ↑↓
    System Integration

"""
"""
Few-Shot Learning Modules Package
=================================

Modular implementation of advanced few-shot learning algorithms.

Modules:
- configurations.py: Configuration dataclasses for all algorithms
- core_networks.py: Main neural network architectures  
- advanced_components.py: Multi-scale features, attention, uncertainty
- utilities.py: Factory functions, evaluation utilities
"""

from .configurations import (
    FewShotConfig,
    PrototypicalConfig,
    MatchingConfig,
    RelationConfig
)

from .core_networks import (
    PrototypicalNetworks,
    SimplePrototypicalNetworks,
    MatchingNetworks,
    RelationNetworks
)

from .advanced_components import (
    MultiScaleFeatureAggregator,
    PrototypeRefiner,
    UncertaintyEstimator,
    ScaledDotProductAttention,
    AdditiveAttention,
    BilinearAttention,
    GraphRelationModule,
    StandardRelationModule,
    UncertaintyAwareDistance,
    HierarchicalPrototypes,
    TaskAdaptivePrototypes
)

from .utilities import (
    create_prototypical_network,
    compare_with_learn2learn_protonet,
    evaluate_on_standard_benchmarks
)

__all__ = [
    # Configurations
    'FewShotConfig',
    'PrototypicalConfig', 
    'MatchingConfig',
    'RelationConfig',
    
    # Core Networks
    'PrototypicalNetworks',
    'SimplePrototypicalNetworks',
    'MatchingNetworks',
    'RelationNetworks',
    
    # Advanced Components
    'MultiScaleFeatureAggregator',
    'PrototypeRefiner',
    'UncertaintyEstimator',
    'ScaledDotProductAttention',
    'AdditiveAttention',
    'BilinearAttention',
    'GraphRelationModule',
    'StandardRelationModule',
    'UncertaintyAwareDistance',
    'HierarchicalPrototypes',
    'TaskAdaptivePrototypes',
    
    # Utilities
    'create_prototypical_network',
    'compare_with_learn2learn_protonet',
    'evaluate_on_standard_benchmarks'
]