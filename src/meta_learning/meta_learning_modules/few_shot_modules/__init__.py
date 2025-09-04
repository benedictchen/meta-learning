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