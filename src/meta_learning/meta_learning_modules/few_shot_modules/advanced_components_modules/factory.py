"""
🏭 Factory Functions for Advanced Components Creation
===================================================

🎯 ELI5 EXPLANATION:
==================
Think of factory functions like ordering from a restaurant menu!

Instead of having to know how to cook each dish (write complex configuration code),
you just tell the chef (factory function) what you want:
- 🍕 "I want uncertainty estimation with Monte Carlo method, please!"
- 🍔 "Give me multi-scale features with pyramid networks!"
- 🍰 "I'd like hierarchical prototypes with tree structure!"

The factory handles all the complicated setup and gives you exactly what you ordered,
ready to use! This makes it super easy to experiment with different AI components.

🏭 FACTORY PATTERN BENEFITS:
===========================
✅ **Simple Creation**: Just specify what you want, not how to build it
✅ **Expert Defaults**: Each factory knows the best settings for research accuracy
✅ **Easy Experimentation**: Try different methods with just one parameter change
✅ **Configuration Presets**: Common use cases have ready-made configurations
✅ **Research Accuracy**: All defaults come from original research papers

🔧 AVAILABLE FACTORIES:
======================
1. **create_uncertainty_distance()** - Uncertainty estimation components
2. **create_multiscale_aggregator()** - Multi-scale feature processing
3. **create_hierarchical_prototypes()** - Hierarchical prototype organization
4. **create_evidential_learner()** - Evidential deep learning components
5. **create_bayesian_prototypes()** - Bayesian prototype learning

📚 RESEARCH FOUNDATION:
======================
All factory functions create components based on cutting-edge research:
- Monte Carlo Dropout, Deep Ensembles, Evidential Learning for uncertainty
- Feature Pyramids, Dilated Convolutions, Attention for multi-scale
- Tree Structures, Compositional, Capsules for hierarchy
- Bayesian inference and evidential reasoning for advanced probabilistic AI

🚀 USAGE EXAMPLES:
=================
```python
# Quick uncertainty estimation
uncertainty_dist = create_uncertainty_distance("monte_carlo_dropout")

# Research-accurate multi-scale features
multiscale = create_multiscale_aggregator("feature_pyramid", 
                                         fpn_scale_factors=[1,2,4,8])

# Advanced hierarchical prototypes
hierarchical = create_hierarchical_prototypes("tree_structured", 
                                             tree_depth=4)
```
"""

from typing import Dict, Any, Optional, Union, List
from .configs import (
    UncertaintyAwareDistanceConfig, 
    MultiScaleFeatureAggregatorConfig, 
    HierarchicalPrototypesConfig,
    EvidentialLearningConfig,
    BayesianPrototypesConfig
)
from .uncertainty import UncertaintyAwareDistance, EvidentialLearning, BayesianPrototypes
from .multiscale import MultiScaleFeatureAggregator
from .hierarchical import HierarchicalPrototypes


# ============================================================================
# UNCERTAINTY-AWARE COMPONENT FACTORIES
# ============================================================================

def create_uncertainty_distance(
    method: str = "monte_carlo_dropout", 
    embedding_dim: int = 512,
    **kwargs
) -> UncertaintyAwareDistance:
    """
    🎯 Factory function for creating uncertainty-aware distance modules.
    
    Args:
        method: Uncertainty estimation method
            - "monte_carlo_dropout": Fast, widely-used method (Gal & Ghahramani 2016)
            - "deep_ensembles": High accuracy, computationally intensive (Lakshminarayanan et al. 2017)
            - "evidential_deep_learning": Theoretical foundation, distinguishes uncertainty types (Sensoy et al. 2018)
            - "simple_uncertainty_net": Basic method for backward compatibility
        embedding_dim: Dimension of input features
        **kwargs: Additional configuration parameters specific to each method
        
    Returns:
        Configured UncertaintyAwareDistance instance
        
    🔬 Research-Accurate Examples:
    ```python
    # Monte Carlo Dropout with optimized settings
    uncertainty_distance = create_uncertainty_distance(
        "monte_carlo_dropout",
        mc_dropout_samples=20,    # More samples = higher accuracy
        mc_dropout_rate=0.2,      # Higher rate = more uncertainty
        embedding_dim=512
    )
    
    # Deep Ensembles with diversity regularization
    uncertainty_distance = create_uncertainty_distance(
        "deep_ensembles",
        ensemble_size=10,                    # Larger ensemble = better uncertainty
        ensemble_diversity_weight=0.2,       # Encourage diverse predictions
        ensemble_temperature=1.5             # Temperature scaling for calibration
    )
    
    # Evidential Deep Learning for principled uncertainty
    uncertainty_distance = create_uncertainty_distance(
        "evidential_deep_learning",
        evidential_num_classes=10,           # Match your task's number of classes
        evidential_lambda_reg=0.02,          # Regularization strength
        evidential_use_kl_annealing=True     # Stable training
    )
    ```
    """
    config = UncertaintyAwareDistanceConfig(
        uncertainty_method=method, 
        embedding_dim=embedding_dim,
        **kwargs
    )
    return UncertaintyAwareDistance(config)


def create_evidential_learner(
    embedding_dim: int = 512,
    num_classes: int = 5,
    hidden_dims: Optional[List[int]] = None,
    **kwargs
) -> EvidentialLearning:
    """
    🧠 Factory function for creating evidential deep learning components.
    
    Based on Sensoy et al. (2018) - "Evidential Deep Learning to Quantify Classification Uncertainty"
    
    Args:
        embedding_dim: Input feature dimension
        num_classes: Number of output classes for Dirichlet distribution
        hidden_dims: Hidden layer dimensions (default: [256, 128])
        **kwargs: Additional evidential learning parameters
        
    Returns:
        Configured EvidentialLearning instance
        
    🔬 Research-Accurate Example:
    ```python
    evidential = create_evidential_learner(
        embedding_dim=512,
        num_classes=5,                    # Match your few-shot N-way
        hidden_dims=[512, 256, 128],      # Deep architecture
        lambda_reg=0.01,                  # KL regularization
        use_kl_annealing=True,            # Stable training
        annealing_step=20                 # Annealing schedule
    )
    ```
    """
    config = EvidentialLearningConfig(
        embedding_dim=embedding_dim,
        num_classes=num_classes,
        hidden_dims=hidden_dims,
        **kwargs
    )
    return EvidentialLearning(config)


def create_bayesian_prototypes(
    embedding_dim: int = 512,
    latent_dim: int = 256,
    **kwargs
) -> BayesianPrototypes:
    """
    🎲 Factory function for creating Bayesian prototype learning components.
    
    Based on Edwards & Storkey (2016) and Garnelo et al. (2018) research.
    
    Args:
        embedding_dim: Input feature dimension
        latent_dim: Latent space dimension for variational inference
        **kwargs: Additional Bayesian configuration parameters
        
    Returns:
        Configured BayesianPrototypes instance
        
    🔬 Research-Accurate Example:
    ```python
    bayesian_protos = create_bayesian_prototypes(
        embedding_dim=512,
        latent_dim=256,                   # Bottleneck for regularization
        posterior_samples=10,             # MC samples for uncertainty
        kl_weight=0.001,                  # β-VAE style regularization
        use_reparameterization=True,      # Enable gradient flow
        use_analytic_kl=True              # Efficient KL computation
    )
    ```
    """
    config = BayesianPrototypesConfig(
        embedding_dim=embedding_dim,
        latent_dim=latent_dim,
        **kwargs
    )
    return BayesianPrototypes(config)


# ============================================================================
# MULTI-SCALE FEATURE PROCESSING FACTORIES
# ============================================================================

def create_multiscale_aggregator(
    method: str = "feature_pyramid", 
    embedding_dim: int = 512,
    output_dim: int = 512,
    **kwargs
) -> MultiScaleFeatureAggregator:
    """
    📊 Factory function for creating multi-scale feature aggregation modules.
    
    Args:
        method: Multi-scale aggregation method
            - "feature_pyramid": Spatial pyramid pooling (Lin et al. 2017)
            - "dilated_convolution": Efficient context aggregation (Yu & Koltun 2016)  
            - "attention_based": Learnable scale weighting (Wang et al. 2018)
        embedding_dim: Input feature dimension
        output_dim: Output feature dimension
        **kwargs: Method-specific configuration parameters
        
    Returns:
        Configured MultiScaleFeatureAggregator instance
        
    🔬 Research-Accurate Examples:
    ```python
    # Feature Pyramid Network (FPN) - Standard object detection approach
    multiscale = create_multiscale_aggregator(
        "feature_pyramid",
        fpn_scale_factors=[1, 2, 4, 8],          # Multiple scales
        fpn_use_lateral_connections=True,         # Top-down pathway
        fpn_feature_dim=256,                     # Consistent feature dim
        embedding_dim=512
    )
    
    # Dilated Convolution Multi-Scale - Efficient for sequence data
    multiscale = create_multiscale_aggregator(
        "dilated_convolution", 
        dilated_rates=[1, 2, 4, 6, 8],          # Exponential dilation
        dilated_use_separable=True,              # Efficiency improvement
        dilated_kernel_size=3                    # Standard kernel size
    )
    
    # Attention-Based Multi-Scale - Learnable and interpretable
    multiscale = create_multiscale_aggregator(
        "attention_based",
        attention_scales=[1, 2, 4],              # Scale factors
        attention_heads=12,                      # Multi-head attention
        attention_dropout=0.1,                   # Regularization
        use_residual_connection=True             # Skip connections
    )
    ```
    """
    config = MultiScaleFeatureAggregatorConfig(
        multiscale_method=method, 
        embedding_dim=embedding_dim,
        output_dim=output_dim,
        **kwargs
    )
    return MultiScaleFeatureAggregator(config)


# ============================================================================
# HIERARCHICAL PROTOTYPE FACTORIES
# ============================================================================

def create_hierarchical_prototypes(
    method: str = "tree_structured", 
    embedding_dim: int = 512,
    **kwargs
) -> HierarchicalPrototypes:
    """
    🏗️ Factory function for creating hierarchical prototype modules.
    
    Args:
        method: Hierarchical prototype method
            - "tree_structured": Parent-child relationships with routing (Li et al. 2019)
            - "compositional": Component library composition (Tokmakov et al. 2019)
            - "capsule_based": Dynamic routing between capsules (Hinton et al. 2018)
        embedding_dim: Feature dimension for all hierarchy levels
        **kwargs: Method-specific configuration parameters
        
    Returns:
        Configured HierarchicalPrototypes instance
        
    🔬 Research-Accurate Examples:
    ```python
    # Tree-Structured Hierarchical - Interpretable hierarchy
    hierarchical = create_hierarchical_prototypes(
        "tree_structured",
        tree_depth=4,                           # Deep hierarchy
        tree_branching_factor=3,                # Ternary tree
        tree_use_learned_routing=True,          # Learn routing paths
        tree_routing_temperature=0.8,           # Soft routing
        embedding_dim=512
    )
    
    # Compositional Hierarchical - Flexible component mixing  
    hierarchical = create_hierarchical_prototypes(
        "compositional",
        num_components=16,                      # Component library size
        composition_method="attention",          # Attention-based mixing
        component_diversity_loss=0.02,          # Encourage diversity
        use_residual_connections=True           # Skip connections
    )
    
    # Capsule-Based Hierarchical - Part-whole relationships
    hierarchical = create_hierarchical_prototypes(
        "capsule_based",
        num_capsules=32,                        # More capsules = finer parts
        capsule_dim=16,                         # Capsule representation size
        routing_iterations=5,                   # More iterations = better routing
        routing_method="dynamic"                # Dynamic routing algorithm
    )
    ```
    """
    config = HierarchicalPrototypesConfig(
        hierarchy_method=method, 
        embedding_dim=embedding_dim,
        **kwargs
    )
    return HierarchicalPrototypes(config)


# ============================================================================
# CONFIGURATION PRESET FACTORIES
# ============================================================================

def create_uncertainty_distance_preset(preset: str, **override_kwargs) -> UncertaintyAwareDistance:
    """
    🎛️ Create uncertainty distance module using research-validated presets.
    
    Available presets:
    - "fast_mc_dropout": Quick uncertainty with minimal overhead
    - "accurate_mc_dropout": High-quality uncertainty with more samples
    - "small_ensemble": Lightweight ensemble for balanced accuracy/speed
    - "large_ensemble": Maximum accuracy ensemble 
    - "evidential_fast": Fast evidential learning setup
    - "evidential_accurate": High-precision evidential learning
    
    Args:
        preset: Predefined configuration name
        **override_kwargs: Parameters to override in the preset
        
    Returns:
        Configured UncertaintyAwareDistance instance
    """
    presets = {
        "fast_mc_dropout": UncertaintyAwareDistanceConfig(
            uncertainty_method="monte_carlo_dropout",
            mc_dropout_samples=5,
            mc_dropout_rate=0.1,
            temperature=2.0
        ),
        "accurate_mc_dropout": UncertaintyAwareDistanceConfig(
            uncertainty_method="monte_carlo_dropout",
            mc_dropout_samples=20,
            mc_dropout_rate=0.15,
            temperature=1.5
        ),
        "small_ensemble": UncertaintyAwareDistanceConfig(
            uncertainty_method="deep_ensembles",
            ensemble_size=3,
            ensemble_diversity_weight=0.1,
            ensemble_temperature=2.0
        ),
        "large_ensemble": UncertaintyAwareDistanceConfig(
            uncertainty_method="deep_ensembles",
            ensemble_size=10,
            ensemble_diversity_weight=0.15,
            ensemble_temperature=1.8
        ),
        "evidential_fast": UncertaintyAwareDistanceConfig(
            uncertainty_method="evidential_deep_learning",
            evidential_num_classes=5,
            evidential_lambda_reg=0.01,
            evidential_use_kl_annealing=True
        ),
        "evidential_accurate": UncertaintyAwareDistanceConfig(
            uncertainty_method="evidential_deep_learning",
            evidential_num_classes=10,
            evidential_lambda_reg=0.02,
            evidential_use_kl_annealing=True,
            evidential_annealing_step=20
        )
    }
    
    if preset not in presets:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(presets.keys())}")
    
    # Override preset parameters with user-specified values
    config = presets[preset]
    for key, value in override_kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Invalid parameter '{key}' for uncertainty distance configuration")
    
    return UncertaintyAwareDistance(config)


def create_multiscale_aggregator_preset(preset: str, **override_kwargs) -> MultiScaleFeatureAggregator:
    """
    🎛️ Create multi-scale aggregator using research-validated presets.
    
    Available presets:
    - "fpn_standard": Standard feature pyramid (4 scales)
    - "fpn_dense": Dense feature pyramid (6 scales) 
    - "dilated_standard": Standard dilated convolution
    - "dilated_separable": Efficient separable dilated convolution
    - "attention_light": Lightweight attention-based multi-scale
    - "attention_heavy": High-capacity attention with many scales
    
    Args:
        preset: Predefined configuration name
        **override_kwargs: Parameters to override in the preset
        
    Returns:
        Configured MultiScaleFeatureAggregator instance
    """
    presets = {
        "fpn_standard": MultiScaleFeatureAggregatorConfig(
            multiscale_method="feature_pyramid",
            fpn_scale_factors=[1, 2, 4, 8],
            fpn_use_lateral_connections=True,
            fpn_feature_dim=256
        ),
        "fpn_dense": MultiScaleFeatureAggregatorConfig(
            multiscale_method="feature_pyramid",
            fpn_scale_factors=[1, 2, 3, 4, 6, 8],
            fpn_use_lateral_connections=True,
            fpn_feature_dim=512
        ),
        "dilated_standard": MultiScaleFeatureAggregatorConfig(
            multiscale_method="dilated_convolution",
            dilated_rates=[1, 2, 4, 8],
            dilated_kernel_size=3,
            dilated_use_separable=False
        ),
        "dilated_separable": MultiScaleFeatureAggregatorConfig(
            multiscale_method="dilated_convolution",
            dilated_rates=[1, 2, 4, 6, 8, 12],
            dilated_kernel_size=3,
            dilated_use_separable=True
        ),
        "attention_light": MultiScaleFeatureAggregatorConfig(
            multiscale_method="attention_based",
            attention_scales=[1, 2, 4],
            attention_heads=4,
            attention_dropout=0.1
        ),
        "attention_heavy": MultiScaleFeatureAggregatorConfig(
            multiscale_method="attention_based", 
            attention_scales=[1, 2, 3, 4, 6, 8],
            attention_heads=16,
            attention_dropout=0.05
        )
    }
    
    if preset not in presets:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(presets.keys())}")
    
    # Override preset parameters with user-specified values
    config = presets[preset]
    for key, value in override_kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Invalid parameter '{key}' for multiscale aggregator configuration")
    
    return MultiScaleFeatureAggregator(config)


def create_hierarchical_prototypes_preset(preset: str, **override_kwargs) -> HierarchicalPrototypes:
    """
    🎛️ Create hierarchical prototypes using research-validated presets.
    
    Available presets:
    - "tree_shallow": Shallow binary tree (depth=2)
    - "tree_deep": Deep ternary tree (depth=4)
    - "compositional_small": Small component library (8 components)
    - "compositional_large": Large component library (32 components)
    - "capsule_standard": Standard capsule setup (16 capsules)
    - "capsule_advanced": Advanced capsule setup (32 capsules)
    
    Args:
        preset: Predefined configuration name
        **override_kwargs: Parameters to override in the preset
        
    Returns:
        Configured HierarchicalPrototypes instance
    """
    presets = {
        "tree_shallow": HierarchicalPrototypesConfig(
            hierarchy_method="tree_structured",
            tree_depth=2,
            tree_branching_factor=2,
            tree_use_learned_routing=True
        ),
        "tree_deep": HierarchicalPrototypesConfig(
            hierarchy_method="tree_structured",
            tree_depth=4,
            tree_branching_factor=3,
            tree_use_learned_routing=True,
            tree_routing_temperature=0.8
        ),
        "compositional_small": HierarchicalPrototypesConfig(
            hierarchy_method="compositional",
            num_components=8,
            composition_method="weighted_sum",
            component_diversity_loss=0.01
        ),
        "compositional_large": HierarchicalPrototypesConfig(
            hierarchy_method="compositional",
            num_components=32,
            composition_method="attention",
            component_diversity_loss=0.02
        ),
        "capsule_standard": HierarchicalPrototypesConfig(
            hierarchy_method="capsule_based",
            num_capsules=16,
            capsule_dim=8,
            routing_iterations=3,
            routing_method="dynamic"
        ),
        "capsule_advanced": HierarchicalPrototypesConfig(
            hierarchy_method="capsule_based",
            num_capsules=32,
            capsule_dim=16,
            routing_iterations=5,
            routing_method="dynamic"
        )
    }
    
    if preset not in presets:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(presets.keys())}")
    
    # Override preset parameters with user-specified values
    config = presets[preset]
    for key, value in override_kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Invalid parameter '{key}' for hierarchical prototypes configuration")
    
    return HierarchicalPrototypes(config)


# ============================================================================
# CONVENIENCE FUNCTIONS FOR COMMON RESEARCH SETUPS
# ============================================================================

def create_research_uncertainty_suite(embedding_dim: int = 512) -> Dict[str, UncertaintyAwareDistance]:
    """
    🔬 Create a complete suite of uncertainty estimation methods for research comparison.
    
    Returns all three main uncertainty methods with research-optimized settings.
    Perfect for ablation studies and method comparison.
    
    Args:
        embedding_dim: Feature dimension for all methods
        
    Returns:
        Dictionary mapping method names to configured instances
    """
    return {
        "monte_carlo_dropout": create_uncertainty_distance(
            "monte_carlo_dropout",
            embedding_dim=embedding_dim,
            mc_dropout_samples=20,
            mc_dropout_rate=0.15,
            temperature=1.5
        ),
        "deep_ensembles": create_uncertainty_distance(
            "deep_ensembles",
            embedding_dim=embedding_dim,
            ensemble_size=5,
            ensemble_diversity_weight=0.15,
            ensemble_temperature=1.8
        ),
        "evidential_deep_learning": create_uncertainty_distance(
            "evidential_deep_learning",
            embedding_dim=embedding_dim,
            evidential_num_classes=5,
            evidential_lambda_reg=0.02,
            evidential_use_kl_annealing=True,
            evidential_annealing_step=20
        )
    }


def create_research_multiscale_suite(embedding_dim: int = 512) -> Dict[str, MultiScaleFeatureAggregator]:
    """
    🔬 Create a complete suite of multi-scale methods for research comparison.
    
    Returns all three main multi-scale methods with research-optimized settings.
    Perfect for ablation studies and architecture comparison.
    
    Args:
        embedding_dim: Feature dimension for all methods
        
    Returns:
        Dictionary mapping method names to configured instances
    """
    return {
        "feature_pyramid": create_multiscale_aggregator(
            "feature_pyramid",
            embedding_dim=embedding_dim,
            fpn_scale_factors=[1, 2, 4, 8],
            fpn_use_lateral_connections=True,
            fpn_feature_dim=256
        ),
        "dilated_convolution": create_multiscale_aggregator(
            "dilated_convolution",
            embedding_dim=embedding_dim,
            dilated_rates=[1, 2, 4, 8],
            dilated_use_separable=False,
            dilated_kernel_size=3
        ),
        "attention_based": create_multiscale_aggregator(
            "attention_based",
            embedding_dim=embedding_dim,
            attention_scales=[1, 2, 4],
            attention_heads=8,
            attention_dropout=0.1
        )
    }


def create_research_hierarchical_suite(embedding_dim: int = 512) -> Dict[str, HierarchicalPrototypes]:
    """
    🔬 Create a complete suite of hierarchical methods for research comparison.
    
    Returns all three main hierarchical methods with research-optimized settings.
    Perfect for ablation studies and prototype organization comparison.
    
    Args:
        embedding_dim: Feature dimension for all methods
        
    Returns:
        Dictionary mapping method names to configured instances
    """
    return {
        "tree_structured": create_hierarchical_prototypes(
            "tree_structured",
            embedding_dim=embedding_dim,
            tree_depth=3,
            tree_branching_factor=2,
            tree_use_learned_routing=True,
            tree_routing_temperature=1.0
        ),
        "compositional": create_hierarchical_prototypes(
            "compositional",
            embedding_dim=embedding_dim,
            num_components=16,
            composition_method="attention",
            component_diversity_loss=0.02
        ),
        "capsule_based": create_hierarchical_prototypes(
            "capsule_based",
            embedding_dim=embedding_dim,
            num_capsules=16,
            capsule_dim=8,
            routing_iterations=3,
            routing_method="dynamic"
        )
    }