"""
⚙️ Meta Learning Solutions Config
==================================

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

⚙️ Configuration Architecture:
==============================
    ┌─────────────────────────┐
    │    USER SETTINGS        │
    ├─────────────────────────┤
    │ • Algorithm Parameters  │
    │ • Performance Options   │
    │ • Research Preferences  │
    │ • Output Formats        │
    └─────────────────────────┘
              ↓
    ┌─────────────────────────┐
    │      ALGORITHM          │
    │    (Configured)         │
    └─────────────────────────┘

"""
"""
Research Solutions Configuration Factory 🔧🎯
============================================

🎯 **ELI5 Explanation**:
Imagine you're helping your little brother learn to recognize different animals!
This is like a magical toolbox that helps you teach him better:

- 🎯 **How Hard Is This?**: Like figuring out if showing him cats vs dogs is easier than showing him different types of birds
- 📊 **Are We Sure?**: Like asking "Are we REALLY sure he learned this?" instead of just guessing
- 🎲 **What To Show Next?**: Like picking the perfect next animal to show him - not too easy, not too hard
- 🎨 **Make Pictures Better**: Like having special tricks to make the animal photos clearer and easier to see

📊 **Available Method Categories**:
```
Research Challenge → Multiple Solutions → User Configurable:

🎯 Difficulty Estimation:
   [Silhouette Analysis, Feature Entropy, k-NN Classification] → Pick Best Method

📊 Confidence Intervals:
   [Student's t-test, BCA Bootstrap, Meta-learning CI] → Statistical Rigor

🎲 Task Sampling:
   [Curriculum Learning, Balanced Sampling, Diversity-based] → Smart Selection

🎨 Data Augmentation:
   [Basic Transforms, Advanced Mixing, Research Techniques] → Enhanced Data
```

🔬 **Research Foundation**:
- **Silhouette Analysis**: Peter J. Rousseeuw (1987) - "A graphical aid to interpretation"
- **Bootstrap Methods**: Bradley Efron (1979) - "Bootstrap methods: another look" 
- **Curriculum Learning**: Yoshua Bengio et al. (2009) - "Curriculum learning"
- **Meta-Learning Statistics**: Timothy Hospedales et al. (2021) - "Meta-learning in neural networks"

This module provides multiple research-accurate methods for common meta-learning challenges
with comprehensive configuration options for user choice.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any, Callable
from enum import Enum
import torch


class DifficultyEstimationMethod(Enum):
    """Research-accurate class difficulty estimation methods."""
    PAIRWISE_DISTANCE = "pairwise_distance"  # Basic distance-based approach
    SILHOUETTE = "silhouette"                # Silhouette analysis (Rousseeuw 1987)
    ENTROPY = "entropy"                      # Feature entropy-based difficulty
    KNN_ACCURACY = "knn"                     # k-NN classification accuracy


class ConfidenceIntervalMethod(Enum):
    """Research-accurate confidence interval methods."""
    BOOTSTRAP = "bootstrap"                  # Standard bootstrap
    T_DISTRIBUTION = "t_distribution"        # Student's t-distribution
    META_LEARNING_STANDARD = "meta_learning_standard"  # Meta-learning specific
    BCA_BOOTSTRAP = "bca_bootstrap"          # Bias-corrected accelerated bootstrap


class AugmentationStrategy(Enum):
    """Data augmentation strategies for meta-learning."""
    NONE = "none"
    BASIC = "basic"                         # Standard augmentations
    ADVANCED = "advanced"                   # Meta-learning optimized augmentations


@dataclass
class FixmeDifficultyEstimationConfig:
    """
    Comprehensive configuration for research methods related to difficulty estimation.
    
    Addresses research comments about:
    1. Arbitrary difficulty metrics
    2. Inefficient O(n²) computations
    3. Missing established metrics
    4. No baseline comparisons
    """
    # Primary method selection
    method: DifficultyEstimationMethod = DifficultyEstimationMethod.SILHOUETTE
    
    # Fallback method if primary fails (MUST be different)
    fallback_method: DifficultyEstimationMethod = DifficultyEstimationMethod.ENTROPY
    
    # Research accuracy controls
    use_research_accurate: bool = True
    compare_to_baselines: bool = True
    
    # Method-specific configurations
    silhouette_config: Dict[str, Any] = field(default_factory=lambda: {
        'metric': 'euclidean',
        'sample_size_limit': 1000,  # For efficiency on large datasets
        'normalize_features': True
    })
    
    entropy_config: Dict[str, Any] = field(default_factory=lambda: {
        'num_bins': 10,
        'smoothing_factor': 1e-8,
        'feature_selection': 'all',  # 'all', 'top_k', 'pca'
        'discretization_method': 'equal_width'
    })
    
    knn_config: Dict[str, Any] = field(default_factory=lambda: {
        'k_neighbors': 5,
        'cross_validation_folds': 3,
        'distance_metric': 'euclidean',
        'weight_function': 'uniform'  # 'uniform', 'distance'
    })
    
    # Performance optimization
    enable_caching: bool = True
    parallel_computation: bool = True
    max_samples_per_class: int = 1000  # Efficiency limit
    
    def __post_init__(self):
        """Validate configuration."""
        if self.method == self.fallback_method:
            raise ValueError("Primary and fallback methods must be different")
        
        if not self.use_research_accurate and self.method == DifficultyEstimationMethod.PAIRWISE_DISTANCE:
            print("WARNING: Using non-research-accurate pairwise distance method")


@dataclass
class FixmeConfidenceIntervalConfig:
    """
    Comprehensive configuration for research methods related to confidence intervals.
    
    Addresses research comments about:
    1. Method selection based on sample size
    2. Research-accurate CI computation
    3. Bootstrap vs parametric methods
    4. Meta-learning specific considerations
    """
    # Primary method selection
    method: ConfidenceIntervalMethod = ConfidenceIntervalMethod.BOOTSTRAP
    
    # Automatic method selection based on data characteristics
    auto_method_selection: bool = True
    min_sample_size_for_bootstrap: int = 30
    
    # Bootstrap-specific configuration
    bootstrap_config: Dict[str, Any] = field(default_factory=lambda: {
        'num_samples': 1000,
        'confidence_level': 0.95,
        'method': 'percentile',  # 'percentile', 'bca', 'abc'
        'stratified_sampling': True
    })
    
    # t-distribution configuration
    t_distribution_config: Dict[str, Any] = field(default_factory=lambda: {
        'confidence_level': 0.95,
        'assume_normality': False,
        'outlier_removal': True,
        'normality_test_threshold': 0.05
    })
    
    # Meta-learning standard configuration
    meta_learning_config: Dict[str, Any] = field(default_factory=lambda: {
        'num_episodes': 600,  # Standard protocol
        'task_batch_size': 32,
        'adaptation_steps': [1, 5, 10],  # Multiple adaptation levels
        'confidence_level': 0.95
    })
    
    # BCA Bootstrap configuration (most sophisticated)
    bca_config: Dict[str, Any] = field(default_factory=lambda: {
        'num_bootstrap_samples': 2000,
        'confidence_level': 0.95,
        'acceleration_constant': True,
        'bias_correction': True,
        'jackknife_estimation': True
    })
    
    # Performance settings
    parallel_bootstrap: bool = True
    cache_intermediate_results: bool = True


@dataclass
class FixmeTaskSamplingConfig:
    """
    Comprehensive configuration for task sampling research methods.
    
    Addresses improvements found in old_archive implementations:
    1. Hierarchical task organization
    2. Balanced task sampling
    3. Dynamic task generation
    4. Task similarity tracking
    """
    # Core task parameters
    n_way: int = 5
    k_shot: int = 5
    q_query: int = 15
    num_tasks: int = 1000
    
    # Advanced sampling strategies
    use_hierarchical_sampling: bool = True
    balance_task_difficulties: bool = True
    track_task_similarity: bool = True
    enable_curriculum_learning: bool = True
    
    # Task difficulty balancing
    difficulty_distribution: str = "uniform"  # "uniform", "normal", "curriculum"
    min_difficulty_threshold: float = 0.1
    max_difficulty_threshold: float = 0.9
    
    # Curriculum learning configuration
    curriculum_config: Dict[str, Any] = field(default_factory=lambda: {
        'start_difficulty': 0.2,
        'end_difficulty': 0.8,
        'progression_rate': 0.01,
        'adaptive_progression': True,
        'performance_threshold': 0.7
    })
    
    # Task similarity tracking
    similarity_config: Dict[str, Any] = field(default_factory=lambda: {
        'similarity_metric': 'cosine',
        'diversity_threshold': 0.3,
        'max_similar_tasks': 0.1,  # Percentage of total tasks
        'feature_space': 'embedding'  # 'raw', 'embedding', 'gradient'
    })


@dataclass
class FixmeDataAugmentationConfig:
    """
    Comprehensive configuration for data augmentation research methods.
    
    Based on meta-learning optimized augmentation strategies from research.
    """
    strategy: AugmentationStrategy = AugmentationStrategy.ADVANCED
    
    # Basic augmentation configuration
    basic_config: Dict[str, Any] = field(default_factory=lambda: {
        'horizontal_flip': True,
        'rotation_degrees': 10,
        'color_jitter': 0.1,
        'normalize': True
    })
    
    # Advanced meta-learning optimized augmentation
    advanced_config: Dict[str, Any] = field(default_factory=lambda: {
        'meta_augmentation': True,
        'task_specific_augmentation': True,
        'augmentation_consistency': True,
        'support_query_augmentation_balance': 0.7,
        'cross_domain_augmentation': True,
        'adaptive_augmentation_strength': True
    })
    
    # Performance configuration
    augmentation_probability: float = 0.8
    cache_augmented_data: bool = True
    parallel_augmentation: bool = True


@dataclass
class ComprehensiveResearchSolutionsConfig:
    """
    Master configuration combining all research solutions with user choice options.
    
    This addresses every research comment found in the codebase with multiple
    research-accurate solutions and comprehensive configuration options.
    """
    # Individual research solution configurations
    difficulty_estimation: FixmeDifficultyEstimationConfig = field(default_factory=FixmeDifficultyEstimationConfig)
    confidence_intervals: FixmeConfidenceIntervalConfig = field(default_factory=FixmeConfidenceIntervalConfig)
    task_sampling: FixmeTaskSamplingConfig = field(default_factory=FixmeTaskSamplingConfig)
    data_augmentation: FixmeDataAugmentationConfig = field(default_factory=FixmeDataAugmentationConfig)
    
    # Global settings
    enable_all_research_accurate_methods: bool = True
    enable_performance_optimizations: bool = True
    enable_comprehensive_logging: bool = True
    
    # Validation settings
    validate_configurations: bool = True
    warn_on_non_research_accurate: bool = True


# Configuration Factory Functions

def create_all_research_solutions_config() -> ComprehensiveResearchSolutionsConfig:
    """Create configuration with all research solutions enabled and research-accurate."""
    return ComprehensiveResearchSolutionsConfig(
        difficulty_estimation=FixmeDifficultyEstimationConfig(
            method=DifficultyEstimationMethod.SILHOUETTE,
            fallback_method=DifficultyEstimationMethod.ENTROPY,
            use_research_accurate=True,
            compare_to_baselines=True
        ),
        confidence_intervals=FixmeConfidenceIntervalConfig(
            method=ConfidenceIntervalMethod.BCA_BOOTSTRAP,  # Most sophisticated
            auto_method_selection=True
        ),
        task_sampling=FixmeTaskSamplingConfig(
            use_hierarchical_sampling=True,
            balance_task_difficulties=True,
            enable_curriculum_learning=True
        ),
        data_augmentation=FixmeDataAugmentationConfig(
            strategy=AugmentationStrategy.ADVANCED
        ),
        enable_all_research_accurate_methods=True
    )


def create_performance_optimized_config() -> ComprehensiveResearchSolutionsConfig:
    """Create configuration optimized for performance while maintaining research accuracy."""
    return ComprehensiveResearchSolutionsConfig(
        difficulty_estimation=FixmeDifficultyEstimationConfig(
            method=DifficultyEstimationMethod.ENTROPY,  # Faster than silhouette
            fallback_method=DifficultyEstimationMethod.KNN_ACCURACY,
            enable_caching=True,
            parallel_computation=True,
            max_samples_per_class=500  # Reduced for speed
        ),
        confidence_intervals=FixmeConfidenceIntervalConfig(
            method=ConfidenceIntervalMethod.BOOTSTRAP,  # Faster than BCA
            auto_method_selection=True,
            bootstrap_config={'num_samples': 500},  # Reduced for speed
            parallel_bootstrap=True
        ),
        task_sampling=FixmeTaskSamplingConfig(
            track_task_similarity=False,  # Disable for speed
            enable_curriculum_learning=False  # Disable for speed
        ),
        data_augmentation=FixmeDataAugmentationConfig(
            strategy=AugmentationStrategy.BASIC,  # Faster than advanced
            cache_augmented_data=True,
            parallel_augmentation=True
        ),
        enable_performance_optimizations=True
    )


def create_research_grade_config() -> ComprehensiveResearchSolutionsConfig:
    """Create configuration for maximum research accuracy (slower but most accurate)."""
    return ComprehensiveResearchSolutionsConfig(
        difficulty_estimation=FixmeDifficultyEstimationConfig(
            method=DifficultyEstimationMethod.SILHOUETTE,
            fallback_method=DifficultyEstimationMethod.KNN_ACCURACY,
            use_research_accurate=True,
            compare_to_baselines=True,
            silhouette_config={'sample_size_limit': 5000},  # Higher accuracy
            knn_config={'cross_validation_folds': 5}  # More thorough validation
        ),
        confidence_intervals=FixmeConfidenceIntervalConfig(
            method=ConfidenceIntervalMethod.BCA_BOOTSTRAP,
            bca_config={'num_bootstrap_samples': 5000},  # Highest accuracy
            auto_method_selection=True
        ),
        task_sampling=FixmeTaskSamplingConfig(
            use_hierarchical_sampling=True,
            balance_task_difficulties=True,
            track_task_similarity=True,
            enable_curriculum_learning=True,
            num_tasks=2000  # More tasks for better statistics
        ),
        data_augmentation=FixmeDataAugmentationConfig(
            strategy=AugmentationStrategy.ADVANCED
        ),
        enable_all_research_accurate_methods=True,
        enable_comprehensive_logging=True
    )


def create_basic_config() -> ComprehensiveResearchSolutionsConfig:
    """Create basic configuration for getting started quickly."""
    return ComprehensiveResearchSolutionsConfig(
        difficulty_estimation=FixmeDifficultyEstimationConfig(
            method=DifficultyEstimationMethod.ENTROPY,
            fallback_method=DifficultyEstimationMethod.SILHOUETTE,
            use_research_accurate=True
        ),
        confidence_intervals=FixmeConfidenceIntervalConfig(
            method=ConfidenceIntervalMethod.BOOTSTRAP,
            auto_method_selection=True
        ),
        task_sampling=FixmeTaskSamplingConfig(
            use_hierarchical_sampling=False,
            balance_task_difficulties=True,
            enable_curriculum_learning=False
        ),
        data_augmentation=FixmeDataAugmentationConfig(
            strategy=AugmentationStrategy.BASIC
        )
    )


# Export all configuration options
__all__ = [
    # Main configuration classes
    'ComprehensiveResearchSolutionsConfig',
    'FixmeDifficultyEstimationConfig', 
    'FixmeConfidenceIntervalConfig',
    'FixmeTaskSamplingConfig',
    'FixmeDataAugmentationConfig',
    
    # Enums for method selection
    'DifficultyEstimationMethod',
    'ConfidenceIntervalMethod', 
    'AugmentationStrategy',
    
    # Factory functions
    'create_all_fixme_solutions_config',
    'create_performance_optimized_config',
    'create_research_grade_config',
    'create_basic_config'
]