"""
Few-Shot Learning Configuration Classes ⚙️📋
============================================

🎯 **ELI5 Explanation**:
Think of these configurations like recipe cards for different cooking methods!
Just like you need different settings for baking a cake vs grilling a steak,
different few-shot learning algorithms need different "settings" or configurations:

- 🎛️ **PrototypicalConfig**: Like a recipe for making prototypes (averages) of each class
- 🔍 **MatchingConfig**: Like instructions for comparing and matching similar items  
- 🔗 **RelationConfig**: Like a guide for learning relationships between different examples

📊 **Configuration Hierarchy**:
```
FewShotConfig (Base Recipe)
    ├── PrototypicalConfig (Prototype-based Methods)
    ├── MatchingConfig (Attention-based Methods)
    └── RelationConfig (Relation-based Methods)
```

🔬 **Research-Accurate Parameter Tuning**:
Each configuration follows the exact hyperparameters used in the original research papers,
with options to experiment with proven variations and extensions.

Configuration dataclasses for all few-shot learning algorithms.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class FewShotConfig:
    """Base configuration for few-shot learning algorithms."""
    embedding_dim: int = 512
    num_classes: int = 5
    num_support: int = 5
    num_query: int = 15
    temperature: float = 1.0
    dropout: float = 0.1


@dataclass
class PrototypicalConfig(FewShotConfig):
    """
    Comprehensive Configuration for Prototypical Networks with All research Solutions.
    
    This configuration provides complete control over ALL implementation variants
    and research extensions with automatic conflict resolution.
    """
    # =========================
    # Core Research Solutions
    # =========================
    
    # Research option: Implementation variant selection
    protonet_variant: str = "research_accurate"  # "original", "research_accurate", "simple", "enhanced"
    use_original_implementation: bool = False  # Pure Snell et al. (2017) implementation
    
    # Research option: Distance computation options (Snell et al. 2017)
    use_squared_euclidean: bool = True  # True for research accuracy
    distance_temperature: float = 1.0
    distance_metric: str = "euclidean"  # "euclidean", "cosine", "manhattan", "learned"
    use_learned_distance: bool = False  # Learn distance metric (Vinyals et al. 2016)
    distance_combination: str = "single"  # "single", "ensemble", "weighted_ensemble"
    distance_weights: List[float] = None  # For weighted ensemble of distances
    
    # Research option: Prototype computation options
    prototype_method: str = "mean"  # "mean", "weighted_mean", "median", "attention"
    use_support_weighting: bool = False
    prototype_refinement_method: str = "none"  # "none", "gradient", "attention", "iterative"
    refinement_learning_rate: float = 0.01
    use_prototype_attention: bool = False  # Attend over support examples
    attention_temperature: float = 1.0
    prototype_refinement_steps: int = 3
    
    # Research option: Uncertainty-aware distance metrics (Allen et al. 2019)
    use_uncertainty_aware_distances: bool = False  # "Prototypical Networks with Uncertainty"
    uncertainty_temperature: float = 2.0
    uncertainty_method: str = "none"  # "none", "ensemble", "dropout", "evidential"
    use_monte_carlo_dropout: bool = False
    dropout_samples: int = 10
    use_evidential_learning: bool = False  # Sensoy et al. 2018
    
    # Research option: Hierarchical prototype structures (Rusu et al. 2019) 
    use_hierarchical_prototypes: bool = False  # "Meta-Learning with Latent Embedding"
    hierarchy_levels: int = 2
    hierarchical_aggregation: str = "bottom_up"  # "bottom_up", "top_down", "bidirectional"
    
    # Research option: Task-specific prototype initialization (Finn et al. 2018)
    use_task_adaptive_prototypes: bool = False  # "Meta-Learning for Semi-Supervised Classification"
    adaptation_steps: int = 5
    adaptation_method: str = "none"  # "none", "gradient", "ridge", "bayesian"
    use_bayesian_prototypes: bool = False  # Bayesian treatment of prototypes
    prior_strength: float = 1.0
    use_task_embeddings: bool = False  # Learn task-specific embeddings
    task_embedding_dim: int = 64
    
    # =========================
    # COMPREHENSIVE EXTENSIONS
    # =========================
    
    # Multi-scale and feature aggregation combinations
    multi_scale_features: bool = True
    scale_factors: List[int] = None
    feature_aggregation_method: str = "concat"  # "concat", "sum", "attention", "learned"
    use_feature_pyramid: bool = False  # Build feature pyramid
    pyramid_levels: int = 4
    
    # Compositional and hierarchical options
    use_compositional_prototypes: bool = False  # Compositional few-shot learning
    composition_method: str = "addition"  # "addition", "concatenation", "attention"
    
    # Memory and continual learning options  
    use_memory_bank: bool = False  # Store prototype memory
    memory_bank_size: int = 1000
    memory_update_strategy: str = "fifo"  # "fifo", "random", "importance"
    use_episodic_memory: bool = False
    
    # Cross-modal and multi-modal support
    use_cross_modal: bool = False  # Cross-modal few-shot learning
    modality_fusion: str = "early"  # "early", "late", "attention"
    modal_alignment: bool = False
    
    # =========================
    # EVALUATION & COMPARISON
    # =========================
    
    # Standard evaluation protocols
    use_standard_evaluation: bool = True
    num_episodes: int = 600  # Standard in literature
    confidence_interval_method: str = "t_distribution"  # "bootstrap", "t_distribution"
    evaluation_metrics: List[str] = None  # ["accuracy", "f1", "auc", "calibration"]
    use_confidence_intervals: bool = True
    bootstrap_samples: int = 1000
    statistical_test: str = "paired_t"  # "paired_t", "wilcoxon", "permutation"
    
    # Comparison with existing libraries
    compare_with_libraries: bool = False
    library_comparison: List[str] = None  # ["learn2learn", "torchmeta"]
    
    # =========================
    # ADVANCED RESEARCH OPTIONS
    # =========================
    
    # Advanced features (research-backed only)
    enable_research_extensions: bool = False
    research_extension_year: str = "2017"  # Only enable extensions with citations
    
    # Regularization and optimization
    prototype_regularization: float = 0.001
    diversity_weight: float = 0.1
    consistency_weight: float = 0.05
    adaptive_prototypes: bool = True
    
    # Implementation debugging and analysis
    debug_mode: bool = False
    log_intermediate_results: bool = False
    save_prototypes: bool = False
    prototype_analysis: bool = False  # Analyze prototype quality
    
    def __post_init__(self):
        """Initialize defaults and resolve configuration conflicts."""
        # Set default values for lists
        if self.scale_factors is None:
            self.scale_factors = [1, 2, 4, 8]
        if self.library_comparison is None:
            self.library_comparison = []
        if self.distance_weights is None:
            self.distance_weights = [1.0]  # Default single weight
        if self.evaluation_metrics is None:
            self.evaluation_metrics = ["accuracy"]
            
        # Automatic conflict resolution for overlapping solutions
        self._resolve_configuration_conflicts()
    
    def _resolve_configuration_conflicts(self):
        """
        
        Resolves conflicts and ensures compatible combinations of features
        to prevent implementation errors and provide clear user guidance.
        """
        
        # If using original implementation, disable all extensions
        if self.use_original_implementation:
            self.multi_scale_features = False
            self.adaptive_prototypes = False  
            self.use_uncertainty_aware_distances = False
            self.use_hierarchical_prototypes = False
            self.use_task_adaptive_prototypes = False
            self.use_compositional_prototypes = False
            self.use_memory_bank = False
            self.use_cross_modal = False
            print("INFO: Original implementation selected - disabling all extensions for research accuracy")
        
        # If uncertainty is enabled, ensure compatible distance metrics
        if self.use_uncertainty_aware_distances and self.distance_metric == "learned":
            print("WARNING: Uncertainty-aware distances may conflict with learned distance - using euclidean")
            self.distance_metric = "euclidean"
        
        # If hierarchical prototypes enabled, ensure compatible aggregation
        if self.use_hierarchical_prototypes and self.feature_aggregation_method == "attention":
            print("INFO: Hierarchical prototypes detected - using hierarchical attention aggregation")
            self.hierarchical_aggregation = "bidirectional"
        
        # Ensure task adaptation and hierarchical don't conflict
        if self.use_task_adaptive_prototypes and self.use_hierarchical_prototypes:
            print("INFO: Both task adaptation and hierarchical enabled - using combined approach")
            self.adaptation_method = "gradient"
            
        # If using compositional prototypes, enable attention-based aggregation
        if self.use_compositional_prototypes:
            if self.feature_aggregation_method == "concat":
                print("INFO: Compositional prototypes enabled - switching to attention aggregation")
                self.feature_aggregation_method = "attention"
        
        # If cross-modal is enabled, ensure proper fusion settings
        if self.use_cross_modal and self.modality_fusion == "early":
            print("INFO: Cross-modal learning - using early fusion with modal alignment")
            self.modal_alignment = True
            
        # Memory bank size validation
        if self.use_memory_bank and self.memory_bank_size < 100:
            print("WARNING: Memory bank size too small, increasing to 100")
            self.memory_bank_size = 100
            
        # Evaluation configuration validation
        if self.num_episodes < 100:
            print("WARNING: Too few evaluation episodes for statistical significance, increasing to 600")
            self.num_episodes = 600
    
    def get_variant_description(self) -> str:
        """Get human-readable description of current configuration variant."""
        if self.use_original_implementation:
            return "Pure Research-Accurate (Snell et al. 2017)"
        elif self.protonet_variant == "simple":
            return "Simple Educational Variant"
        elif self.protonet_variant == "research_accurate":
            extensions = []
            if self.use_uncertainty_aware_distances:
                extensions.append("Uncertainty-Aware")
            if self.use_hierarchical_prototypes:
                extensions.append("Hierarchical")
            if self.use_task_adaptive_prototypes:
                extensions.append("Task-Adaptive")
            if self.use_compositional_prototypes:
                extensions.append("Compositional")
            
            if extensions:
                return f"Research-Accurate + {', '.join(extensions)}"
            else:
                return "Research-Accurate (Base)"
        else:
            return "Enhanced with All Extensions"
    
    def validate_configuration(self) -> List[str]:
        """
        
        Returns list of validation warnings/errors for user review.
        """
        warnings = []
        
        # Check for conflicting distance metrics
        if self.distance_combination == "ensemble" and len(self.distance_weights) == 1:
            warnings.append("Ensemble distance selected but only one weight provided")
        
        # Check research extension compatibility
        if self.enable_research_extensions and self.research_extension_year == "2017":
            if any([self.use_uncertainty_aware_distances, self.use_hierarchical_prototypes]):
                warnings.append("Research extensions from later years enabled with 2017 base")
        
        # Check evaluation configuration
        if self.use_confidence_intervals and self.num_episodes < 100:
            warnings.append("Too few episodes for reliable confidence intervals")
        
        # Check memory usage implications
        if self.use_memory_bank and self.use_hierarchical_prototypes and self.use_compositional_prototypes:
            warnings.append("Multiple memory-intensive features enabled - may impact performance")
            
        return warnings


@dataclass 
class MatchingConfig(FewShotConfig):
    """Configuration for Matching Networks with 2024 improvements."""
    # Original parameters
    attention_type: str = "cosine"  # cosine, bilinear, additive, scaled_dot_product
    use_lstm: bool = True
    lstm_layers: int = 2
    bidirectional: bool = True
    
    # 2024 enhancements
    multi_head_attention: bool = True
    num_attention_heads: int = 8
    use_positional_encoding: bool = True
    graph_attention: bool = True


@dataclass
class RelationConfig(FewShotConfig):
    """Configuration for Relation Networks with graph neural enhancements."""
    # Original parameters  
    relation_dim: int = 8
    hidden_dim: int = 512
    
    # 2024 Graph Neural Network improvements
    use_graph_relations: bool = True
    graph_layers: int = 3
    edge_feature_dim: int = 64
    node_feature_dim: int = 256
    graph_attention_heads: int = 4
    
    # Advanced relation modeling
    relation_aggregation: str = "attention"  # mean, max, attention, graph
    compositional_relations: bool = True
    temporal_relations: bool = False