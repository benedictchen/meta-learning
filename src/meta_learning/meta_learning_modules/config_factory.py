#!/usr/bin/env python3
"""
⚙️ Config Factory
==================

🎯 ELI5 Summary:
Think of this like a control panel for our algorithm! Just like how your TV remote 
has different buttons for volume, channels, and brightness, this file has all the settings 
that control how our AI algorithm behaves. Researchers can adjust these settings to get 
the best results for their specific problem.

🧪 Technical Details:
===================
Implementation details and technical specifications for this component.
Designed to work seamlessly within the research framework while
maintaining high performance and accuracy standards.

"""
"""
Meta-Learning Configuration Factory 🏭⚙️
================================================================

🎯 **ELI5 Explanation**:
Think of this like a master configuration wizard that helps you set up complex AI experiments!
Just like ordering a custom car where you pick engine type, transmission, paint color, and features,
this factory helps you configure every aspect of your meta-learning system:

- 🎛️ **Pick Your Algorithm**: MAML, Prototypical Networks, Matching Networks
- ⚙️ **Choose Extensions**: Uncertainty estimation, hierarchical learning, adaptive components  
- 🔧 **Hardware Settings**: GPU acceleration, mixed precision, memory optimization
- 📊 **Dataset Options**: Omniglot, miniImageNet, custom datasets with curriculum learning
- 🎯 **All Combinations**: Mix and match features like building with LEGO blocks

📊 **Configuration Factory Visualization**:
```
User Requirements:           Configuration Factory:        Complete Setup:
┌─────────────────┐         ┌─────────────────────┐       ┌──────────────────┐
│ "I want MAML    │         │                     │       │ ✅ MAML Config   │
│  with uncertain-│   ──→   │ Smart Configuration │  ──→  │ ✅ Uncertainty   │
│  ty estimation  │         │ Factory             │       │ ✅ GPU Support   │
│  on GPU"        │         │                     │       │ ✅ Best Defaults │
└─────────────────┘         └─────────────────────┘       └──────────────────┘
```

🔧 **Factory Functions Available**:
- 🏗️ **create_maml_config()**: Set up MAML with all variants (Standard, First-Order, ANIL)
- 🎯 **create_prototypical_config()**: Configure Prototypical Networks with extensions
- 📊 **create_uncertainty_config()**: Add uncertainty estimation to any algorithm
- ⚡ **create_hardware_config()**: Optimize for your specific hardware setup
- 🎲 **create_curriculum_config()**: Set up progressive difficulty learning

🔬 **Research-Accurate Presets**:
All factory functions use proven parameter combinations from original research papers,
so you get working systems without having to tune hundreds of hyperparameters manually.

This module provides factory functions to create configurations for ALL
implemented research solutions across all modules in the meta-learning package.

Users can pick and choose which solutions to enable with overlapping
configurations handled intelligently.
"""

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass

# Import all configuration classes
from .test_time_compute import TestTimeComputeConfig
from .few_shot_learning import PrototypicalConfig, MatchingConfig, RelationConfig
from .continual_meta_learning import ContinualMetaConfig, OnlineMetaConfig
from .maml_variants import MAMLConfig
from .utils_modules import TaskConfiguration, EvaluationConfig

# Import new component configurations
from .few_shot_modules.uncertainty_components import UncertaintyConfig
from .few_shot_modules.hierarchical_components import HierarchicalConfig
from .few_shot_modules.adaptive_components import TaskAdaptiveConfig
from .few_shot_modules.utilities import DatasetLoadingConfig
from .utils_modules.statistical_evaluation import TaskDifficultyConfig


@dataclass  
class UnifiedMetaLearningConfig:
    """
    Unified configuration for all meta-learning components and algorithms.
    
    Provides centralized configuration management following standard ML
    research practices for few-shot learning, MAML, and continual learning.
    """
    # Test-Time Compute Configuration
    test_time_compute: Optional[TestTimeComputeConfig] = None
    
    # Few-Shot Learning Configurations
    prototypical: Optional[PrototypicalConfig] = None
    matching: Optional[MatchingConfig] = None
    relation: Optional[RelationConfig] = None
    
    # Continual Learning Configurations  
    continual_meta: Optional[ContinualMetaConfig] = None
    online_meta: Optional[OnlineMetaConfig] = None
    
    # MAML Configuration
    maml: Optional[MAMLConfig] = None
    
    # Utility Configurations
    task: Optional[TaskConfiguration] = None
    evaluation: Optional[EvaluationConfig] = None
    
    # Advanced Component Configurations
    uncertainty: Optional[UncertaintyConfig] = None
    hierarchical: Optional[HierarchicalConfig] = None
    task_adaptive: Optional[TaskAdaptiveConfig] = None
    dataset_loading: Optional[DatasetLoadingConfig] = None
    task_difficulty: Optional[TaskDifficultyConfig] = None
    
    # Global settings that affect multiple modules
    global_seed: int = 42
    device: str = "auto"  # "auto", "cpu", "cuda"
    verbose: bool = True


# =============================================================================
# Meta-Learning Configuration Factory Functions
# =============================================================================

def create_all_algorithms_enabled_config() -> UnifiedMetaLearningConfig:
    """
    Create configuration enabling all available meta-learning algorithms.
    
    Activates MAML, few-shot learning, test-time compute scaling,
    continual learning, and all other implemented methods.
    """
    config = UnifiedMetaLearningConfig()
    
    # Test-Time Compute: All solutions enabled
    config.test_time_compute = TestTimeComputeConfig()
    config.test_time_compute.compute_strategy = "hybrid"
    config.test_time_compute.use_process_reward = True
    config.test_time_compute.use_test_time_training = True
    config.test_time_compute.use_gradient_verification = True
    config.test_time_compute.use_chain_of_thought = True
    config.test_time_compute.cot_method = "attention_based"
    config.test_time_compute.use_optimal_allocation = True
    config.test_time_compute.use_adaptive_distribution = True
    
    # Prototypical Networks: All research extensions enabled
    config.prototypical = PrototypicalConfig()
    config.prototypical.use_uncertainty_aware_distances = True
    config.prototypical.use_hierarchical_prototypes = True
    config.prototypical.use_task_adaptive_prototypes = True
    config.prototypical.protonet_variant = "research_accurate"
    config.prototypical.multi_scale_features = True
    config.prototypical.adaptive_prototypes = True
    config.prototypical.uncertainty_estimation = True
    
    # Matching Networks: Advanced attention mechanisms
    config.matching = MatchingConfig()
    config.matching.attention_mechanism = "scaled_dot_product"
    config.matching.context_encoding = True
    config.matching.support_set_encoding = "transformer"
    config.matching.bidirectional_lstm = True
    
    # Relation Networks: Graph neural networks enabled
    config.relation = RelationConfig()
    config.relation.use_graph_neural_network = True
    config.relation.edge_features = True
    config.relation.self_attention = True
    
    # Continual Learning: All EWC and Fisher solutions
    config.continual_meta = ContinualMetaConfig()
    config.continual_meta.ewc_method = "full"  # Use full Fisher matrix
    config.continual_meta.fisher_estimation_method = "exact"
    config.continual_meta.fisher_accumulation_method = "ema"
    config.continual_meta.memory_consolidation_method = "ewc"
    config.continual_meta.use_task_specific_importance = True
    config.continual_meta.use_gradient_importance = True
    
    # Online Meta-Learning: Advanced replay and adaptation
    config.online_meta = OnlineMetaConfig()
    config.online_meta.experience_replay = True
    config.online_meta.prioritized_replay = True
    config.online_meta.importance_sampling = True
    config.online_meta.adaptive_lr = True
    
    # MAML: All functional forward solutions
    config.maml = MAMLConfig()
    config.maml.functional_forward_method = "higher_style"
    config.maml.maml_variant = "maml"
    config.maml.inner_lr = 0.01
    config.maml.inner_steps = 5
    config.maml.first_order = False
    
    # Task Configuration: All difficulty estimation methods
    config.task = TaskConfiguration()
    config.task.difficulty_estimation_method = "entropy"  # Can switch between methods
    
    # Evaluation: All confidence interval methods
    config.evaluation = EvaluationConfig()
    config.evaluation.confidence_interval_method = "bca_bootstrap"
    config.evaluation.num_episodes = 600
    
    # Advanced Component Configurations: All research solutions enabled
    config.uncertainty = UncertaintyConfig()
    config.uncertainty.method = "monte_carlo_dropout"
    config.uncertainty.num_samples = 20
    config.uncertainty.dropout_rate = 0.1
    
    config.hierarchical = HierarchicalConfig()
    config.hierarchical.method = "multi_level"
    config.hierarchical.num_levels = 3
    config.hierarchical.level_temperatures = [1.0, 2.0, 4.0]
    
    config.task_adaptive = TaskAdaptiveConfig()
    config.task_adaptive.method = "attention_based"
    config.task_adaptive.attention_heads = 8
    config.task_adaptive.hidden_dim = 512
    
    config.dataset_loading = DatasetLoadingConfig()
    config.dataset_loading.method = "torchmeta"
    config.dataset_loading.fallback_to_synthetic = False  # ZERO FAKE DATA POLICY
    config.dataset_loading.warn_on_fallback = True
    
    config.task_difficulty = TaskDifficultyConfig()
    config.task_difficulty.method = "intra_class_variance"
    config.task_difficulty.fallback_method = "entropy"
    config.task_difficulty.warn_on_fallback = True
    
    return config


def create_paper_exact_config() -> UnifiedMetaLearningConfig:
    """
    Create configuration using exact paper implementations over optimizations.
    
    Prioritizes faithful reproduction of original research methods over performance.
    Uses unmodified algorithms as published in source papers.
    """
    config = UnifiedMetaLearningConfig()
    
    # Test-Time Compute: Research-accurate methods
    config.test_time_compute = TestTimeComputeConfig()
    config.test_time_compute.compute_strategy = "snell2024"
    config.test_time_compute.use_process_reward = True
    config.test_time_compute.prm_scoring_method = "weighted"
    
    # Prototypical Networks: Pure original implementation
    config.prototypical = PrototypicalConfig()
    config.prototypical.use_original_implementation = True
    config.prototypical.use_squared_euclidean = True
    config.prototypical.prototype_method = "mean"
    
    # Continual Learning: Kirkpatrick et al. 2017 exact
    config.continual_meta = ContinualMetaConfig()
    config.continual_meta.ewc_method = "diagonal"
    config.continual_meta.fisher_estimation_method = "empirical"
    config.continual_meta.fisher_sampling_method = "true_posterior"
    
    # MAML: Original Finn et al. 2017 implementation
    config.maml = MAMLConfig()
    config.maml.maml_variant = "maml"
    config.maml.functional_forward_method = "basic"
    config.maml.first_order = False
    
    # Evaluation: Standard meta-learning protocols
    config.evaluation = EvaluationConfig()
    config.evaluation.confidence_interval_method = "t_distribution"
    config.evaluation.num_episodes = 600
    
    return config


def create_performance_optimized_config() -> UnifiedMetaLearningConfig:
    """
    Create configuration optimized for performance and speed.
    
    PERFORMANCE-FIRST: Balanced accuracy with computational efficiency.
    """
    config = UnifiedMetaLearningConfig()
    
    # Test-Time Compute: Fast configuration
    config.test_time_compute = TestTimeComputeConfig()
    config.test_time_compute.compute_strategy = "basic"
    config.test_time_compute.max_compute_budget = 100
    config.test_time_compute.min_compute_steps = 3
    config.test_time_compute.use_chain_of_thought = True
    config.test_time_compute.cot_method = "prototype_based"  # Fastest method
    
    # Prototypical Networks: Simple but effective
    config.prototypical = PrototypicalConfig()
    config.prototypical.protonet_variant = "simple"
    config.prototypical.multi_scale_features = False
    config.prototypical.adaptive_prototypes = False
    
    # Continual Learning: Diagonal EWC for speed
    config.continual_meta = ContinualMetaConfig()
    config.continual_meta.ewc_method = "diagonal"
    config.continual_meta.fisher_estimation_method = "empirical"
    config.continual_meta.fisher_accumulation_method = "sum"
    
    # MAML: First-order for speed
    config.maml = MAMLConfig()
    config.maml.maml_variant = "fomaml"  # First-order MAML
    config.maml.functional_forward_method = "compiled"  # PyTorch 2.0 optimization
    config.maml.inner_steps = 3
    
    # Evaluation: Fast CI computation
    config.evaluation = EvaluationConfig()
    config.evaluation.confidence_interval_method = "bootstrap"
    config.evaluation.num_episodes = 300  # Reduced for speed
    
    return config


def create_specific_solution_config(
    solutions: List[str]
) -> UnifiedMetaLearningConfig:
    """
    Create configuration for specific research solutions only.
    
    Args:
        solutions: List of solution identifiers to enable
        
    Available solutions:
    - "process_reward_model": Test-time compute process reward verification
    - "consistency_verification": Test-time training consistency checks
    - "gradient_verification": Gradient-based step verification
    - "attention_reasoning": Attention-based reasoning paths
    - "feature_reasoning": Feature-based reasoning decomposition
    - "prototype_reasoning": Prototype-distance reasoning steps
    - "uncertainty_distances": Uncertainty-aware distance metrics
    - "hierarchical_prototypes": Multi-level prototype structures
    - "task_adaptive_prototypes": Task-specific prototype initialization
    - "full_fisher": Full Fisher Information Matrix computation
    - "evcl": Elastic Variational Continual Learning
    - "kfac_fisher": Kronecker-factored Fisher approximation
    - "functional_forward": Advanced functional forward methods
    - "difficulty_estimation": Advanced difficulty estimation methods
    - "bootstrap_ci": Advanced confidence interval methods
    - "monte_carlo_dropout": Monte Carlo Dropout uncertainty estimation
    - "deep_ensemble": Deep Ensemble uncertainty estimation
    - "evidential": Evidential deep learning uncertainty
    - "bayesian": Bayesian neural network uncertainty
    - "multi_level_prototypes": Multi-level hierarchical prototypes
    - "tree_structured_prototypes": Tree-structured hierarchical prototypes
    - "attention_based_adaptation": Attention-based task adaptation
    - "meta_learning_adaptation": Meta-learning based adaptation
    - "torchmeta_loading": Torchmeta dataset loading
    - "custom_splits_loading": Custom dataset splits loading
    - "intra_class_difficulty": Intra-class variance difficulty estimation
    - "inter_class_difficulty": Inter-class separation difficulty estimation
    """
    config = UnifiedMetaLearningConfig()
    
    # Initialize basic configurations
    config.test_time_compute = TestTimeComputeConfig()
    config.prototypical = PrototypicalConfig()
    config.continual_meta = ContinualMetaConfig()
    config.maml = MAMLConfig()
    config.evaluation = EvaluationConfig()
    
    # Enable specific solutions based on user selection
    for solution in solutions:
        if solution == "process_reward_model":
            config.test_time_compute.use_process_reward = True
            config.test_time_compute.use_process_reward_model = True
            
        elif solution == "consistency_verification":
            config.test_time_compute.use_test_time_training = True
            config.test_time_compute.adaptation_weight = 0.6
            
        elif solution == "gradient_verification":
            config.test_time_compute.use_gradient_verification = True
            
        elif solution == "attention_reasoning":
            config.test_time_compute.use_chain_of_thought = True
            config.test_time_compute.cot_method = "attention_based"
            
        elif solution == "feature_reasoning":
            config.test_time_compute.use_chain_of_thought = True
            config.test_time_compute.cot_method = "feature_based"
            
        elif solution == "prototype_reasoning":
            config.test_time_compute.use_chain_of_thought = True
            config.test_time_compute.cot_method = "prototype_based"
            
        elif solution == "uncertainty_distances":
            config.prototypical.use_uncertainty_aware_distances = True
            
        elif solution == "hierarchical_prototypes":
            config.prototypical.use_hierarchical_prototypes = True
            
        elif solution == "task_adaptive_prototypes":
            config.prototypical.use_task_adaptive_prototypes = True
            
        elif solution == "full_fisher":
            config.continual_meta.ewc_method = "full"
            config.continual_meta.fisher_estimation_method = "exact"
            
        elif solution == "evcl":
            config.continual_meta.ewc_method = "evcl"
            
        elif solution == "kfac_fisher":
            config.continual_meta.fisher_estimation_method = "kfac"
            
        elif solution == "functional_forward":
            config.maml.functional_forward_method = "higher_style"
            
        elif solution == "difficulty_estimation":
            config.task = TaskConfiguration(difficulty_estimation_method="entropy")
            
        elif solution == "bootstrap_ci":
            config.evaluation.confidence_interval_method = "bca_bootstrap"
            
        # New component solutions
        elif solution == "monte_carlo_dropout":
            if config.uncertainty is None:
                config.uncertainty = UncertaintyConfig()
            config.uncertainty.method = "monte_carlo_dropout"
            
        elif solution == "deep_ensemble":
            if config.uncertainty is None:
                config.uncertainty = UncertaintyConfig()
            config.uncertainty.method = "deep_ensemble"
            
        elif solution == "evidential":
            if config.uncertainty is None:
                config.uncertainty = UncertaintyConfig()
            config.uncertainty.method = "evidential"
            
        elif solution == "bayesian":
            if config.uncertainty is None:
                config.uncertainty = UncertaintyConfig()
            config.uncertainty.method = "bayesian"
            
        elif solution == "multi_level_prototypes":
            if config.hierarchical is None:
                config.hierarchical = HierarchicalConfig()
            config.hierarchical.method = "multi_level"
            
        elif solution == "tree_structured_prototypes":
            if config.hierarchical is None:
                config.hierarchical = HierarchicalConfig()
            config.hierarchical.method = "tree_structured"
            
        elif solution == "attention_based_adaptation":
            if config.task_adaptive is None:
                config.task_adaptive = TaskAdaptiveConfig()
            config.task_adaptive.method = "attention_based"
            
        elif solution == "meta_learning_adaptation":
            if config.task_adaptive is None:
                config.task_adaptive = TaskAdaptiveConfig()
            config.task_adaptive.method = "meta_learning"
            
        elif solution == "torchmeta_loading":
            if config.dataset_loading is None:
                config.dataset_loading = DatasetLoadingConfig()
            config.dataset_loading.method = "torchmeta"
            
        elif solution == "custom_splits_loading":
            if config.dataset_loading is None:
                config.dataset_loading = DatasetLoadingConfig()
            config.dataset_loading.method = "custom"
            
        elif solution == "intra_class_difficulty":
            if config.task_difficulty is None:
                config.task_difficulty = TaskDifficultyConfig()
            config.task_difficulty.method = "intra_class_variance"
            
        elif solution == "inter_class_difficulty":
            if config.task_difficulty is None:
                config.task_difficulty = TaskDifficultyConfig()
            config.task_difficulty.method = "inter_class_separation"
            
        else:
            print(f"Warning: Unknown solution '{solution}'. Ignoring.")
    
    return config


def create_modular_config(
    test_time_compute: Optional[str] = None,
    few_shot_method: Optional[str] = None,
    continual_method: Optional[str] = None,
    maml_variant: Optional[str] = None,
    evaluation_method: Optional[str] = None
) -> UnifiedMetaLearningConfig:
    """
    Create modular configuration by choosing specific methods for each component.
    
    Args:
        test_time_compute: "basic", "snell2024", "akyurek2024", "openai_o1", "hybrid"
        few_shot_method: "prototypical", "matching", "relation"
        continual_method: "ewc", "mas", "packnet", "hat"
        maml_variant: "maml", "fomaml", "reptile", "anil", "boil"
        evaluation_method: "bootstrap", "t_distribution", "bca_bootstrap"
    """
    config = UnifiedMetaLearningConfig()
    
    # Configure test-time compute
    if test_time_compute:
        config.test_time_compute = TestTimeComputeConfig()
        config.test_time_compute.compute_strategy = test_time_compute
        
        if test_time_compute in ["snell2024", "hybrid"]:
            config.test_time_compute.use_process_reward = True
        if test_time_compute in ["akyurek2024", "hybrid"]:
            config.test_time_compute.use_test_time_training = True
        if test_time_compute in ["openai_o1", "hybrid"]:
            config.test_time_compute.use_chain_of_thought = True
    
    # Configure few-shot method
    if few_shot_method == "prototypical":
        config.prototypical = PrototypicalConfig()
        config.prototypical.protonet_variant = "research_accurate"
    elif few_shot_method == "matching":
        config.matching = MatchingConfig()
        config.matching.attention_mechanism = "scaled_dot_product"
    elif few_shot_method == "relation":
        config.relation = RelationConfig()
        config.relation.use_graph_neural_network = True
    
    # Configure continual learning
    if continual_method:
        config.continual_meta = ContinualMetaConfig()
        config.continual_meta.memory_consolidation_method = continual_method
        
        if continual_method == "ewc":
            config.continual_meta.ewc_method = "diagonal"
        elif continual_method == "mas":
            config.continual_meta.use_gradient_importance = True
    
    # Configure MAML variant
    if maml_variant:
        config.maml = MAMLConfig()
        config.maml.maml_variant = maml_variant
        
        if maml_variant == "fomaml":
            config.maml.first_order = True
        elif maml_variant in ["anil", "boil"]:
            config.maml.functional_forward_method = "l2l_style"
    
    # Configure evaluation
    if evaluation_method:
        config.evaluation = EvaluationConfig()
        config.evaluation.confidence_interval_method = evaluation_method
    
    return config


def simplified_analysis_component_config() -> UnifiedMetaLearningConfig:
    """
    Create configuration with ALL new component solutions enabled.
    
    COMPREHENSIVE COMPONENTS: Enables all UncertaintyAwareDistance, 
    HierarchicalPrototypes, TaskAdaptivePrototypes, DatasetLoading, 
    and TaskDifficulty solutions with optimal settings.
    """
    config = UnifiedMetaLearningConfig()
    
    # Enable all uncertainty estimation methods (default to best)
    config.uncertainty = UncertaintyConfig()
    config.uncertainty.method = "monte_carlo_dropout"  # Most practical
    config.uncertainty.num_samples = 10
    config.uncertainty.dropout_rate = 0.1
    
    # Enable all hierarchical prototype methods
    config.hierarchical = HierarchicalConfig()
    config.hierarchical.method = "multi_level"
    config.hierarchical.num_levels = 3
    config.hierarchical.level_temperatures = [1.0, 2.0, 4.0]
    
    # Enable all task-adaptive methods
    config.task_adaptive = TaskAdaptiveConfig()
    config.task_adaptive.method = "attention_based"
    config.task_adaptive.attention_heads = 8
    config.task_adaptive.hidden_dim = 512
    
    # Enable robust dataset loading
    config.dataset_loading = DatasetLoadingConfig()
    config.dataset_loading.method = "torchmeta"
    config.dataset_loading.fallback_to_synthetic = False  # ZERO FAKE DATA POLICY
    config.dataset_loading.warn_on_fallback = True
    
    # Enable comprehensive difficulty estimation
    config.task_difficulty = TaskDifficultyConfig()
    config.task_difficulty.method = "intra_class_variance"
    config.task_difficulty.fallback_method = "entropy"
    config.task_difficulty.warn_on_fallback = True
    
    # Add basic configurations for completeness
    config.prototypical = PrototypicalConfig()
    config.prototypical.use_uncertainty_aware_distances = True
    config.prototypical.use_hierarchical_prototypes = True
    config.prototypical.use_task_adaptive_prototypes = True
    
    config.evaluation = EvaluationConfig()
    config.evaluation.confidence_interval_method = "bootstrap"
    config.evaluation.num_episodes = 200
    
    return config


def create_educational_config() -> UnifiedMetaLearningConfig:
    """
    Create configuration optimized for educational use and understanding.
    
    EDUCATIONAL: Simplified but still research-accurate implementations.
    """
    config = UnifiedMetaLearningConfig()
    
    # Simple but working implementations
    config.test_time_compute = TestTimeComputeConfig()
    config.test_time_compute.compute_strategy = "basic"
    config.test_time_compute.max_compute_budget = 50
    
    config.prototypical = PrototypicalConfig()
    config.prototypical.protonet_variant = "simple"
    
    config.maml = MAMLConfig()
    config.maml.maml_variant = "maml"
    config.maml.inner_steps = 3
    
    config.evaluation = EvaluationConfig()
    config.evaluation.confidence_interval_method = "t_distribution"
    config.evaluation.num_episodes = 100
    
    return config


def get_available_solutions() -> Dict[str, List[str]]:
    """
    Get a dictionary of all available research solutions organized by module.
    
    Returns:
        Dictionary mapping module names to lists of available solutions
    """
    return {
        "test_time_compute": [
            "process_reward_model",
            "consistency_verification", 
            "gradient_verification",
            "attention_reasoning",
            "feature_reasoning",
            "prototype_reasoning"
        ],
        "few_shot_learning": [
            "uncertainty_distances",
            "hierarchical_prototypes", 
            "task_adaptive_prototypes",
            "research_accurate_original"
        ],
        "uncertainty_components": [
            "monte_carlo_dropout",
            "deep_ensemble",
            "evidential",
            "bayesian"
        ],
        "hierarchical_components": [
            "multi_level",
            "tree_structured", 
            "coarse_to_fine",
            "adaptive_hierarchy"
        ],
        "adaptive_components": [
            "attention_based",
            "meta_learning",
            "context_dependent",
            "transformer_based"
        ],
        "dataset_loading": [
            "torchmeta",
            "custom_splits",
            "huggingface", 
            "synthetic_fallback"
        ],
        "difficulty_estimation": [
            "intra_class_variance",
            "inter_class_separation",
            "mdl_complexity",
            "gradient_based",
            "entropy"
        ],
        "continual_meta_learning": [
            "diagonal_fisher",
            "full_fisher",
            "kfac_fisher",
            "evcl",
            "gradient_importance"
        ],
        "maml_variants": [
            "l2l_functional_forward",
            "higher_functional_forward",
            "manual_functional_forward",
            "compiled_functional_forward"
        ],
        "utils": [
            "silhouette_difficulty",
            "entropy_difficulty",
            "knn_difficulty",
            "t_distribution_ci",
            "meta_learning_ci",
            "bca_bootstrap_ci"
        ]
    }


def print_solution_summary():
    """Print summary of available meta-learning configurations."""
    solutions = get_available_solutions()
    
    # Removed print spam: "...
    print("=" * 70)
    print(f"Total: {sum(len(module_solutions) for module_solutions in solutions.values())} solutions across {len(solutions)} modules")
    
    for module, module_solutions in solutions.items():
        print(f"\n📦 {module.replace('_', ' ').title()}:")
        for i, solution in enumerate(module_solutions, 1):
            pass  # Implementation needed
    
    print(f"\n🏭 Factory Functions Available:")
    print("  • create_all_algorithms_enabled_config() - Enable all algorithms")
    print("  • create_paper_exact_config() - Exact paper implementations")
    print("  • create_performance_optimized_config() - Performance-first approach")
    print("  • create_specific_solution_config([solutions]) - Pick specific solutions")
    print("  • create_modular_config(...) - Mix and match by module")
    print("  • create_comprehensive_component_config() - component configurations")
    print("  • create_educational_config() - Simplified for learning")


# Configuration validation
def validate_config(config: UnifiedMetaLearningConfig) -> Dict[str, List[str]]:
    """
    Validate configuration for potential conflicts or issues.
    
    Returns:
        Dictionary with 'warnings' and 'errors' lists
    """
    issues = {"warnings": [], "errors": []}
    
    # Check for conflicting settings
    if config.test_time_compute and config.maml:
        if (config.test_time_compute.use_test_time_training and 
            config.maml.maml_variant in ["anil", "boil"]):
            issues["warnings"].append(
                "Test-time training with ANIL/BOIL may have conflicting adaptation strategies"
            )
    
    # Check for performance implications
    if config.continual_meta and config.continual_meta.fisher_estimation_method == "exact":
        issues["warnings"].append(
            "Exact Fisher Information computation is very expensive - consider 'empirical' for large models"
        )
    
    # Check for research accuracy
    if (config.prototypical and 
        config.prototypical.use_uncertainty_aware_distances and
        not config.prototypical.uncertainty_estimation):
        issues["warnings"].append(
            "Uncertainty-aware distances require uncertainty_estimation=True for best results"
        )
    
    return issues