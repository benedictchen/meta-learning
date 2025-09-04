"""
⚙️ Configurations
==================

🔬 Research Foundation:  
======================
Based on meta-learning and few-shot learning research:
- Finn, C., Abbeel, P. & Levine, S. (2017). "Model-Agnostic Meta-Learning for Fast Adaptation"
- Snell, J., Swersky, K. & Zemel, R. (2017). "Prototypical Networks for Few-shot Learning"
- Nichol, A., Achiam, J. & Schulman, J. (2018). "On First-Order Meta-Learning Algorithms"
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
Configuration Classes for Meta-Learning Utilities
================================================

Author: Benedict Chen (benedict@benedictchen.com)

This module contains all configuration dataclasses and classes for meta-learning
utilities, providing type-safe configuration management with sensible defaults.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from enum import Enum


class DatasetMethod(Enum):
    """Real dataset loading methods - NO SYNTHETIC DATA BY DEFAULT."""
    TORCHMETA = "torchmeta"          # Research-accurate meta-learning datasets
    TORCHVISION = "torchvision"      # Standard computer vision datasets
    HUGGINGFACE = "huggingface"      # Hugging Face datasets integration
    # REMOVED: SYNTHETIC = "synthetic" - violates no fake data policy


@dataclass
class TaskConfiguration:
    """Configuration for meta-learning tasks."""
    n_way: int = 5
    k_shot: int = 5
    q_query: int = 15
    num_tasks: int = 1000
    task_type: str = "classification"
    augmentation_strategy: str = "basic"  # basic, advanced, none
    
    # Configuration options for difficulty estimation methods
    difficulty_estimation_method: str = "pairwise_distance"  # "pairwise_distance", "silhouette", "entropy", "knn"
    use_research_accurate_difficulty: bool = False  # Enable research-backed methods


@dataclass
class EvaluationConfig:
    """Configuration for meta-learning evaluation."""
    confidence_intervals: bool = True
    num_bootstrap_samples: int = 1000
    significance_level: float = 0.05
    track_adaptation_curve: bool = True
    compute_uncertainty: bool = True
    
    # Configuration options for confidence interval methods
    ci_method: str = "bootstrap"  # "bootstrap", "t_distribution", "meta_learning_standard", "bca_bootstrap"
    use_research_accurate_ci: bool = False  # Enable research-backed CI methods
    num_episodes: int = 600  # Standard meta-learning evaluation protocol
    
    # Additional configuration for advanced CI methods
    min_sample_size_for_bootstrap: int = 30  # Minimum sample size for bootstrap vs t-distribution
    auto_method_selection: bool = True  # Automatically select best CI method based on data


@dataclass
class DatasetConfig:
    """Configuration for meta-learning dataset creation."""
    # Dataset Method Selection
    dataset_method: DatasetMethod = DatasetMethod.TORCHMETA
    
    # Dataset Configuration
    dataset_name: str = "omniglot"
    torchmeta_root: str = "./data"
    meta_split: str = "train"
    torchmeta_download: bool = True
    
    # Image Processing
    image_size: int = 28
    normalize_mean: List[float] = None
    normalize_std: List[float] = None
    
    # Synthetic Data Settings (ONLY if explicitly enabled)
    synthetic_seed: int = 42
    add_noise: bool = True
    noise_scale: float = 0.1
    require_user_confirmation_for_synthetic: bool = True
    
    # Original configuration (backward compatibility)
    dataset_type: str = "episodic"
    augmentation_strategy: str = "minimal"
    shuffle: bool = True
    stratified: bool = True
    normalize: bool = True
    cache_episodes: bool = False
    
    def __post_init__(self):
        # Set default normalize values
        if self.normalize_mean is None:
            self.normalize_mean = [0.5]
        if self.normalize_std is None:
            self.normalize_std = [0.5]


class MetricsConfig:
    """Configuration for evaluation metrics computation."""
    
    def __init__(
        self,
        compute_accuracy: bool = True,
        compute_loss: bool = True,
        compute_adaptation_speed: bool = False,
        compute_uncertainty: bool = False,
        track_gradients: bool = False,
        save_predictions: bool = False,
        **kwargs
    ):
        self.compute_accuracy = compute_accuracy
        self.compute_loss = compute_loss
        self.compute_adaptation_speed = compute_adaptation_speed
        self.compute_uncertainty = compute_uncertainty
        self.track_gradients = track_gradients
        self.save_predictions = save_predictions
        for key, value in kwargs.items():
            setattr(self, key, value)


class StatsConfig:
    """Configuration for statistical analysis."""
    
    def __init__(
        self,
        confidence_level: float = 0.95,
        num_bootstrap_samples: int = 1000,
        significance_test: str = "t_test",
        multiple_comparison_correction: str = "bonferroni",
        effect_size_method: str = "cohen_d",
        **kwargs
    ):
        self.confidence_level = confidence_level
        self.num_bootstrap_samples = num_bootstrap_samples
        self.significance_test = significance_test
        self.multiple_comparison_correction = multiple_comparison_correction
        self.effect_size_method = effect_size_method
        for key, value in kwargs.items():
            setattr(self, key, value)


class CurriculumConfig:
    """Configuration for curriculum learning strategies."""
    
    def __init__(
        self,
        strategy: str = "difficulty_based",
        initial_difficulty: float = 0.3,
        difficulty_increment: float = 0.1,
        difficulty_threshold: float = 0.8,
        adaptation_patience: int = 5,
        **kwargs
    ):
        self.strategy = strategy
        self.initial_difficulty = initial_difficulty
        self.difficulty_increment = difficulty_increment
        self.difficulty_threshold = difficulty_threshold
        self.adaptation_patience = adaptation_patience
        for key, value in kwargs.items():
            setattr(self, key, value)


@dataclass
class DiversityConfig:
    """
    Configuration for task diversity tracking with multiple metric support.
    
    Supports 5 research-accurate diversity metrics:
    1. cosine_similarity: Task feature similarity (default)
    2. feature_variance: Feature variance-based diversity (Chen et al. 2020)
    3. silhouette_score: Class separation diversity (Rousseeuw 1987)
    4. information_theoretic: Shannon entropy-based diversity
    5. jensen_shannon_divergence: Distribution divergence diversity
    """
    
    # Core configuration
    diversity_metric: str = "cosine_similarity"  # Main diversity method
    track_class_distribution: bool = True
    track_feature_diversity: bool = True
    diversity_threshold: float = 0.7
    
    # Feature variance method options
    variance_scale: float = 2.0  # Scaling factor for variance normalization
    
    # Information-theoretic method options  
    histogram_bins: int = 50  # Number of bins for histogram computation
    
    # Silhouette score method options
    min_samples_silhouette: int = 2  # Minimum samples for silhouette computation
    
    # Jensen-Shannon divergence options
    js_smoothing: float = 1e-8  # Smoothing factor for probability distributions
    
    # Global options
    handle_empty_tasks: str = "warn"  # "warn", "error", "ignore"
    fallback_score: float = 0.5  # Fallback diversity score on errors
    enable_warnings: bool = True  # Enable warning messages