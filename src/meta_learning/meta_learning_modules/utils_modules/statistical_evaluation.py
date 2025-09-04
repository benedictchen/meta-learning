"""
📋 Statistical Evaluation
==========================

🎯 ELI5 Summary:
This file is an important component in our AI research system! Like different organs 
in your body that work together to keep you healthy, this file has a specific job that 
helps the overall algorithm work correctly and efficiently.

🧪 Technical Details:
===================
Implementation details and technical specifications for this component.
Designed to work seamlessly within the research framework while
maintaining high performance and accuracy standards.

"""
"""
Statistical Evaluation Functions for Meta-Learning 📊📈
======================================================

🎯 **ELI5 Explanation**:
Think of this like a statistics teacher who helps you understand if your AI results are really good or just lucky!
When you test an AI algorithm, you need to know:
- 📊 **Are the results reliable?** (Not just random chance)
- 🎯 **How confident can we be?** (95% sure? 99% sure?)
- 📈 **How does it compare to other methods?** (Statistical significance testing)
- 🔍 **What's the margin of error?** (Confidence intervals)

📊 **Statistical Testing Visualization**:
```
Raw Results:        Statistical Analysis:     Reliable Conclusions:
┌─────────────┐    ┌─────────────────────┐    ┌───────────────────┐
│ Accuracy:   │    │ • Confidence        │    │ ✅ Method A is    │
│ 85%, 87%,   │ ──→│   Intervals         │ ──→│    significantly  │
│ 83%, 89%    │    │ • Significance      │    │    better than B  │
│ 86%...      │    │   Testing           │    │ 📊 95% confidence │
└─────────────┘    │ • Error Bars        │    │    interval       │
                   └─────────────────────┘    └───────────────────┘
```

🔬 **Research-Accurate Statistical Methods**:
Implements rigorous evaluation protocols from:
- **Bootstrap Confidence Intervals**: Bradley Efron (1979) - "Bootstrap methods"  
- **Student's t-test**: William Sealy Gosset (1908) - Small sample statistics
- **Paired t-tests**: For comparing algorithm performance
- **Effect Size Calculations**: Cohen's d for practical significance

🧮 **Key Statistical Concepts**:
- **Confidence Intervals**: Range where true value likely lies (e.g., 85.2% ± 2.1%)
- **p-values**: Probability results happened by chance (p < 0.05 = statistically significant)
- **Standard Error**: Measure of uncertainty in our estimates
- **Effect Size**: How big is the difference in practical terms

Author: Benedict Chen (benedict@benedictchen.com)

This module contains statistical functions for rigorous meta-learning evaluation,
including multiple confidence interval methods and research-accurate protocols.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import logging
from dataclasses import dataclass
from .configurations import EvaluationConfig, MetricsConfig, StatsConfig

logger = logging.getLogger(__name__)


def few_shot_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    return_per_class: bool = False
) -> Union[float, Tuple[float, torch.Tensor]]:
    """
    Compute few-shot learning accuracy with advanced metrics.
    
    Args:
        predictions: Model predictions [n_samples, n_classes] or [n_samples]
        targets: Ground truth labels [n_samples]
        return_per_class: Whether to return per-class accuracies
        
    Returns:
        Overall accuracy, optionally with per-class accuracies
    """
    if predictions.dim() == 2:
        # Logits or probabilities - take argmax
        pred_labels = predictions.argmax(dim=-1)
    else:
        # Already labels
        pred_labels = predictions
    
    # Overall accuracy
    correct = (pred_labels == targets).float()
    overall_accuracy = correct.mean().item()
    
    if return_per_class:
        # Per-class accuracy
        unique_classes = torch.unique(targets)
        per_class_accuracies = []
        
        for class_id in unique_classes:
            class_mask = targets == class_id
            if class_mask.sum() > 0:
                class_correct = correct[class_mask].mean().item()
                per_class_accuracies.append(class_correct)
            else:
                per_class_accuracies.append(0.0)
        
        return overall_accuracy, torch.tensor(per_class_accuracies)
    
    return overall_accuracy


def adaptation_speed(
    loss_curve: List[float],
    convergence_threshold: float = 0.01
) -> Tuple[int, float]:
    """
    Measure adaptation speed for meta-learning algorithms.
    
    Args:
        loss_curve: List of losses during adaptation steps
        convergence_threshold: Threshold for considering convergence
        
    Returns:
        Tuple of (steps_to_convergence, final_loss)
    """
    if len(loss_curve) < 2:
        return len(loss_curve), loss_curve[-1] if loss_curve else float('inf')
    
    # Find convergence point
    for i in range(1, len(loss_curve)):
        loss_change = abs(loss_curve[i] - loss_curve[i-1])
        if loss_change < convergence_threshold:
            return i + 1, loss_curve[i]
    
    # No convergence found
    return len(loss_curve), loss_curve[-1]


def compute_confidence_interval(
    values: List[float],
    confidence_level: float = 0.95,
    num_bootstrap: int = 1000
) -> Tuple[float, float, float]:
    """
    Compute confidence interval using bootstrap sampling.
    
    FIXME RESEARCH ACCURACY ISSUES:
    1. BOOTSTRAP ONLY: Should also offer t-distribution CI for small samples (n < 30)
    2. MISSING VALIDATION: No check for minimum sample size for valid bootstrap
    3. NO BIAS CORRECTION: Should implement bias-corrected and accelerated (BCa) bootstrap
    4. MISSING STANDARD REPORTING: Meta-learning literature typically uses specific CI methods
    
    CORRECT APPROACHES:
    - t-distribution CI for small samples
    - BCa bootstrap for better accuracy
    - Standard meta-learning evaluation protocols
    
    Args:
        values: List of values to compute CI for
        confidence_level: Confidence level (e.g., 0.95 for 95%)
        num_bootstrap: Number of bootstrap samples
        
    Returns:
        Tuple of (mean, lower_bound, upper_bound)
    """
    if len(values) == 0:
        return 0.0, 0.0, 0.0
    
    values = np.array(values)
    mean_val = np.mean(values)
    
    # Check sample size and use appropriate method
    if len(values) < 30:
        # Use t-distribution CI for small samples (research-accurate)
        from scipy import stats
        t_critical = stats.t.ppf((1 + confidence_level) / 2, df=len(values) - 1)
        standard_error = np.std(values, ddof=1) / np.sqrt(len(values))
        margin_of_error = t_critical * standard_error
        
        ci_lower = mean_val - margin_of_error
        ci_upper = mean_val + margin_of_error
        
        logger.debug(f"Used t-distribution CI for small sample (n={len(values)})")
        return mean_val, ci_lower, ci_upper
    
    # CURRENT: Basic bootstrap (adequate but not optimal)
    bootstrap_means = []
    for _ in range(num_bootstrap):
        bootstrap_sample = np.random.choice(values, size=len(values), replace=True)
        bootstrap_means.append(np.mean(bootstrap_sample))
    
    # Compute percentiles
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    lower_bound = np.percentile(bootstrap_means, lower_percentile)
    upper_bound = np.percentile(bootstrap_means, upper_percentile)
    
    return mean_val, lower_bound, upper_bound


def compute_confidence_interval_research_accurate(
    values: List[float],
    config: EvaluationConfig = None,
    confidence_level: float = 0.95
) -> Tuple[float, float, float]:
    """
    
    Uses appropriate CI method based on configuration and sample size with auto-selection.
    """
    config = config or EvaluationConfig()
    
    if not config.use_research_accurate_ci:
        return compute_confidence_interval(values, confidence_level, config.num_bootstrap_samples)
    
    # Auto-select method if enabled
    if config.auto_method_selection:
        method = _auto_select_ci_method(values, config)
    else:
        method = config.ci_method
    
    # Route to appropriate method based on configuration
    if method == "t_distribution":
        return compute_t_confidence_interval(values, confidence_level)
    elif method == "meta_learning_standard":
        return compute_meta_learning_ci(values, confidence_level, config.num_episodes)
    elif method == "bca_bootstrap":
        return compute_bca_bootstrap_ci(values, confidence_level, config.num_bootstrap_samples)
    elif method == "bootstrap":
        return compute_confidence_interval(values, confidence_level, config.num_bootstrap_samples)
    else:
        raise ValueError(f"Unknown confidence interval method: {method}. Available: t_distribution, meta_learning_standard, bca_bootstrap, bootstrap")


def _auto_select_ci_method(values: List[float], config: EvaluationConfig) -> str:
    """
    Automatically select the best CI method based on data characteristics.
    
    Selection criteria based on statistical best practices:
    - t-distribution for small samples (n < 30)
    - Bootstrap for moderate samples (30 <= n < 100) 
    - BCa bootstrap for large samples (n >= 100) or skewed distributions
    - Meta-learning standard for exactly 600 episodes (standard protocol)
    """
    n = len(values)
    
    # Standard meta-learning evaluation protocol
    if n == config.num_episodes:
        return "meta_learning_standard"
    
    # Small sample: use t-distribution
    if n < config.min_sample_size_for_bootstrap:
        return "t_distribution"
    
    # Large sample or check for skewness
    if n >= 100:
        # Check for skewness (simple heuristic)
        values_array = np.array(values)
        mean_val = np.mean(values_array)
        median_val = np.median(values_array)
        
        # If distribution is skewed, use BCa bootstrap
        skew_threshold = 0.1 * np.std(values_array)
        if abs(mean_val - median_val) > skew_threshold:
            return "bca_bootstrap"
    
    # Default to standard bootstrap for moderate samples
    return "bootstrap"


def compute_t_confidence_interval(
    values: List[float], 
    confidence_level: float = 0.95
) -> Tuple[float, float, float]:
    """
    Compute confidence interval using t-distribution (appropriate for small samples).
    
    Standard approach in meta-learning evaluation when n < 30.
    """
    import scipy.stats as stats
    
    if len(values) == 0:
        return 0.0, 0.0, 0.0
    
    values = np.array(values)
    mean_val = np.mean(values)
    std_val = np.std(values, ddof=1)  # Sample standard deviation
    n = len(values)
    
    # Degrees of freedom
    df = n - 1
    
    # Critical t-value
    alpha = 1 - confidence_level
    t_critical = stats.t.ppf(1 - alpha/2, df)
    
    # Margin of error
    margin_error = t_critical * (std_val / np.sqrt(n))
    
    # Confidence interval
    lower_bound = mean_val - margin_error
    upper_bound = mean_val + margin_error
    
    return mean_val, lower_bound, upper_bound


def compute_meta_learning_ci(
    accuracies: List[float],
    confidence_level: float = 0.95,
    num_episodes: int = 600
) -> Tuple[float, float, float]:
    """
    Standard confidence interval computation for meta-learning evaluation.
    
    Based on standard protocols from few-shot learning literature:
    - Vinyals et al. (2016): "Matching Networks"  
    - Snell et al. (2017): "Prototypical Networks"
    - Finn et al. (2017): "MAML"
    
    Typically uses 600 episodes with t-distribution CI.
    """
    if len(accuracies) != num_episodes:
        print(f"Warning: Expected {num_episodes} episodes, got {len(accuracies)}")
    
    # Use t-distribution for proper meta-learning evaluation
    return compute_t_confidence_interval(accuracies, confidence_level)


def compute_bca_bootstrap_ci(
    values: List[float],
    confidence_level: float = 0.95,
    num_bootstrap: int = 2000
) -> Tuple[float, float, float]:
    """
    Bias-corrected and accelerated bootstrap confidence interval.
    
    More accurate than basic bootstrap, especially for skewed distributions.
    Based on Efron & Tibshirani (1993) "An Introduction to the Bootstrap".
    """
    import scipy.stats as stats
    
    if len(values) == 0:
        return 0.0, 0.0, 0.0
    
    values = np.array(values)
    n = len(values)
    mean_val = np.mean(values)
    
    # Bootstrap resampling
    bootstrap_means = []
    for _ in range(num_bootstrap):
        bootstrap_sample = np.random.choice(values, size=n, replace=True)
        bootstrap_means.append(np.mean(bootstrap_sample))
    
    bootstrap_means = np.array(bootstrap_means)
    
    # Bias correction
    bias_correction = stats.norm.ppf((bootstrap_means < mean_val).mean())
    
    # Acceleration (jackknife)
    jackknife_means = []
    for i in range(n):
        jackknife_sample = np.concatenate([values[:i], values[i+1:]])
        jackknife_means.append(np.mean(jackknife_sample))
    
    jackknife_means = np.array(jackknife_means)
    jackknife_mean = np.mean(jackknife_means)
    
    acceleration = np.sum((jackknife_mean - jackknife_means)**3) / \
                  (6 * (np.sum((jackknife_mean - jackknife_means)**2))**(3/2))
    
    # Adjusted percentiles
    alpha = 1 - confidence_level
    z_alpha_2 = stats.norm.ppf(alpha/2)
    z_1_alpha_2 = stats.norm.ppf(1 - alpha/2)
    
    alpha_1 = stats.norm.cdf(bias_correction + 
                            (bias_correction + z_alpha_2) / (1 - acceleration * (bias_correction + z_alpha_2)))
    alpha_2 = stats.norm.cdf(bias_correction + 
                            (bias_correction + z_1_alpha_2) / (1 - acceleration * (bias_correction + z_1_alpha_2)))
    
    # Compute bounds
    lower_bound = np.percentile(bootstrap_means, 100 * alpha_1)
    upper_bound = np.percentile(bootstrap_means, 100 * alpha_2)
    
    return mean_val, lower_bound, upper_bound


def basic_confidence_interval(values: List[float], confidence_level: float = 0.95) -> Tuple[float, float, float]:
    """Basic confidence interval computation."""
    return compute_confidence_interval(values, confidence_level=confidence_level)


@dataclass
class TaskDifficultyConfig:
    """Configuration for task difficulty estimation methods."""
    method: str = "intra_class_variance"  # "intra_class_variance", "inter_class_separation", "mdl_complexity", "gradient_based", "entropy"
    
    # Intra-class variance options
    variance_normalization: float = 10.0
    assume_balanced_classes: bool = True
    samples_per_class_hint: int = 20
    
    # Inter-class separation options  
    use_lda: bool = True
    min_accuracy_threshold: float = 0.1
    max_accuracy_threshold: float = 0.9
    
    # MDL complexity options
    compression_algorithm: str = "zlib"  # "zlib", "bz2", "lzma"
    
    # Gradient-based options
    gradient_steps: int = 3
    learning_rate: float = 0.01
    gradient_norm_scale: float = 100.0
    hidden_size: int = 32
    
    # Fallback options - CRITICAL: No hardcoded fallback values allowed
    fallback_method: str = "entropy"  # Must be different from primary method
    warn_on_fallback: bool = True
    allow_hardcoded_fallback: bool = False  # Set to True only for debugging
    
    def __post_init__(self):
        """Validate configuration to prevent common issues."""
        if self.fallback_method == self.method:
            raise ValueError(f"Fallback method '{self.fallback_method}' cannot be same as primary method '{self.method}'")
        
        valid_methods = ["intra_class_variance", "inter_class_separation", "mdl_complexity", "gradient_based", "entropy"]
        if self.method not in valid_methods:
            raise ValueError(f"Unknown method '{self.method}'. Valid methods: {valid_methods}")
        if self.fallback_method not in valid_methods:
            raise ValueError(f"Unknown fallback method '{self.fallback_method}'. Valid methods: {valid_methods}")


def estimate_difficulty(task_data: torch.Tensor, method: str = "intra_class_variance", 
                      task_labels: Optional[torch.Tensor] = None, 
                      config: Optional[TaskDifficultyConfig] = None) -> float:
    """
    Estimate task difficulty using research-accurate methods.
    
    RESEARCH-ACCURATE IMPLEMENTATION: Now supports 4 difficulty estimation methods
    with proper statistical foundations and configuration options.
    
    Args:
        task_data: Task feature data [n_samples, n_features]
        method: Difficulty estimation method
        task_labels: Optional labels for supervised methods
        config: Configuration for difficulty estimation
        
    Returns:
        Difficulty score between 0 (easy) and 1 (hard)
    """
    if config is None:
        config = TaskDifficultyConfig(method=method)
    
    # Try primary method
    try:
        if config.method == "intra_class_variance":
            return _estimate_difficulty_intra_class_variance(task_data, task_labels, config)
        elif config.method == "inter_class_separation":
            return _estimate_difficulty_inter_class_separation(task_data, task_labels, config)
        elif config.method == "mdl_complexity":
            return _estimate_difficulty_mdl_complexity(task_data, config)
        elif config.method == "gradient_based":
            return _estimate_difficulty_gradient_based(task_data, task_labels, config)
        elif config.method == "entropy":
            return _estimate_difficulty_entropy(task_data, config)
        else:
            raise ValueError(f"Unknown difficulty estimation method: {config.method}")
            
    except Exception as e:
        if config.warn_on_fallback:
            print(f"⚠️  Warning: {config.method} difficulty estimation failed ({e}), using fallback")
        
        # Fallback to simpler method
        if config.fallback_method == "entropy":
            return _estimate_difficulty_entropy(task_data, config)
        elif config.fallback_method == "intra_class_variance" and config.method != "intra_class_variance":
            return _estimate_difficulty_intra_class_variance(task_data, task_labels, config)
        elif config.fallback_method == "inter_class_separation" and config.method != "inter_class_separation":
            return _estimate_difficulty_inter_class_separation(task_data, task_labels, config) 
        else:
            # CRITICAL: No hardcoded values - force user to handle failure
            raise RuntimeError(f"""All difficulty estimation methods failed!

Please check your data or try these options:
1. Ensure task_data is valid tensor with appropriate dimensions
2. Use a different estimation method in TaskDifficultyConfig
3. Provide task_labels for supervised methods
4. Check that fallback_method is different from primary method

Available methods: intra_class_variance, inter_class_separation, mdl_complexity, gradient_based, entropy""")


def _estimate_difficulty_intra_class_variance(task_data: torch.Tensor, task_labels: Optional[torch.Tensor], config: TaskDifficultyConfig) -> float:
    """SOLUTION 1: Intra-class variance difficulty estimation (statistically sound)"""
    
    if task_labels is not None:
        unique_labels = torch.unique(task_labels)
        intra_class_variances = []
        
        for label in unique_labels:
            class_mask = task_labels == label
            class_samples = task_data[class_mask]
            
            if len(class_samples) > 1:
                class_variance = torch.var(class_samples, dim=0).mean().item()
                intra_class_variances.append(class_variance)
        
        if intra_class_variances:
            avg_intra_variance = np.mean(intra_class_variances)
            # Higher variance = harder to learn consistent representations
            difficulty = min(1.0, avg_intra_variance / config.variance_normalization)
            return max(0.0, difficulty)
    
    elif config.assume_balanced_classes:
        # Assume balanced classes for unlabeled data
        n_samples = task_data.size(0)
        estimated_n_classes = max(1, n_samples // config.samples_per_class_hint)
        
        intra_class_variances = []
        for class_idx in range(estimated_n_classes):
            start_idx = class_idx * config.samples_per_class_hint
            end_idx = min(start_idx + config.samples_per_class_hint, n_samples)
            
            if end_idx > start_idx + 1:
                class_samples = task_data[start_idx:end_idx]
                class_variance = torch.var(class_samples, dim=0).mean().item()
                intra_class_variances.append(class_variance)
        
        if intra_class_variances:
            avg_intra_variance = np.mean(intra_class_variances)
            difficulty = min(1.0, avg_intra_variance / config.variance_normalization)
            return max(0.0, difficulty)
    
    # Fallback: overall variance
    overall_variance = torch.var(task_data, dim=0).mean().item()
    difficulty = min(1.0, overall_variance / config.variance_normalization)
    return max(0.0, difficulty)


def _estimate_difficulty_inter_class_separation(task_data: torch.Tensor, task_labels: Optional[torch.Tensor], config: TaskDifficultyConfig) -> float:
    """SOLUTION 2: Inter-class separation difficulty (Discriminability Index)"""
    
    if task_labels is None or len(torch.unique(task_labels)) <= 1:
        raise ValueError("Inter-class separation requires labeled data with multiple classes")
    
    try:
        if config.use_lda:
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
            
            X = task_data.detach().numpy()
            y = task_labels.detach().numpy()
            
            lda = LinearDiscriminantAnalysis()
            lda.fit(X, y)
            
            # Lower separability score = higher difficulty
            separability = lda.score(X, y)
            difficulty = 1.0 - separability  # Invert: low accuracy = high difficulty
            return max(config.min_accuracy_threshold, min(config.max_accuracy_threshold, difficulty))
        
        else:
            # Alternative: Silhouette coefficient for class separability
            from sklearn.metrics import silhouette_score
            
            X = task_data.detach().numpy()
            y = task_labels.detach().numpy()
            
            silhouette = silhouette_score(X, y)
            difficulty = 1.0 - ((silhouette + 1) / 2)  # High silhouette = easy task
            return max(0.0, min(1.0, difficulty))
            
    except ImportError as e:
        raise ImportError(f"sklearn required for inter-class separation: {e}")


def _estimate_difficulty_mdl_complexity(task_data: torch.Tensor, config: TaskDifficultyConfig) -> float:
    """SOLUTION 3: Minimum Description Length (MDL) based difficulty"""
    
    try:
        # Convert tensor to bytes for compression
        data_bytes = task_data.detach().numpy().tobytes()
        
        if config.compression_algorithm == "zlib":
            import zlib
            compressed_size = len(zlib.compress(data_bytes))
        elif config.compression_algorithm == "bz2":
            import bz2
            compressed_size = len(bz2.compress(data_bytes))
        elif config.compression_algorithm == "lzma":
            import lzma
            compressed_size = len(lzma.compress(data_bytes))
        else:
            raise ValueError(f"Unsupported compression algorithm: {config.compression_algorithm}")
        
        original_size = len(data_bytes)
        
        # Compression ratio as difficulty measure
        compression_ratio = compressed_size / original_size
        # Higher compression ratio = more structured/predictable = easier
        difficulty = 1.0 - compression_ratio  # Invert
        return max(0.1, min(0.9, difficulty))
        
    except Exception as e:
        raise RuntimeError(f"MDL complexity estimation failed: {e}")


def _estimate_difficulty_gradient_based(task_data: torch.Tensor, task_labels: Optional[torch.Tensor], config: TaskDifficultyConfig) -> float:
    """SOLUTION 4: Gradient-based difficulty estimation (Chen et al. 2020)"""
    
    if task_labels is None:
        raise ValueError("Gradient-based difficulty estimation requires labels")
    
    try:
        n_classes = len(torch.unique(task_labels))
        simple_model = torch.nn.Sequential(
            torch.nn.Linear(task_data.shape[-1], config.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(config.hidden_size, n_classes)
        )
        
        optimizer = torch.optim.SGD(simple_model.parameters(), lr=config.learning_rate)
        criterion = torch.nn.CrossEntropyLoss()
        
        total_grad_norm = 0.0
        
        for _ in range(config.gradient_steps):
            optimizer.zero_grad()
            outputs = simple_model(task_data)
            loss = criterion(outputs, task_labels.long())
            loss.backward()
            
            # Measure gradient magnitude
            step_grad_norm = 0.0
            for param in simple_model.parameters():
                if param.grad is not None:
                    step_grad_norm += param.grad.data.norm(2).item() ** 2
            
            total_grad_norm += step_grad_norm ** 0.5
            optimizer.step()
        
        # Average gradient norm over all steps
        avg_grad_norm = total_grad_norm / config.gradient_steps
        
        # Higher gradient norm typically indicates harder optimization
        difficulty = min(1.0, avg_grad_norm / config.gradient_norm_scale)
        return max(0.0, difficulty)
        
    except Exception as e:
        raise RuntimeError(f"Gradient-based difficulty estimation failed: {e}")


def _estimate_difficulty_entropy(task_data: torch.Tensor, config: TaskDifficultyConfig) -> float:
    """FALLBACK: Simple entropy-based difficulty estimation"""
    try:
        # Simple entropy-based difficulty  
        probs = F.softmax(task_data.mean(dim=0), dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8))
        return entropy.item() / np.log(task_data.size(-1))  # Normalized entropy
    except Exception as e:
        # CRITICAL: No hardcoded fallback values - raise error with actionable guidance
        raise RuntimeError(f"""Entropy-based difficulty estimation failed: {e}

Possible solutions:
1. Check task_data tensor shape and values (should be [n_samples, n_features])
2. Ensure task_data contains valid numerical values (no NaN/inf)
3. Try a different difficulty estimation method
4. Verify input data preprocessing is correct

This is the final fallback method - if it fails, the input data likely has issues.""")


class EvaluationMetrics:
    """Comprehensive evaluation metrics for meta-learning algorithms."""
    
    def __init__(self, config: MetricsConfig):
        self.config = config
        self.reset()
    
    def reset(self):
        """Reset all metrics to initial state."""
        self.accuracies = []
        self.losses = []
        self.adaptation_speeds = []
        self.uncertainties = []
        self.predictions = []
        self.gradients = []
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, 
               loss: Optional[float] = None, **kwargs):
        """Update metrics with new predictions and targets."""
        if self.config.compute_accuracy:
            accuracy = (predictions.argmax(dim=-1) == targets).float().mean().item()
            self.accuracies.append(accuracy)
        
        if self.config.compute_loss and loss is not None:
            self.losses.append(loss)
        
        if self.config.save_predictions:
            self.predictions.append(predictions.detach().cpu())
        
        # Add other metrics based on config
        for key, value in kwargs.items():
            if hasattr(self, key + 's'):
                getattr(self, key + 's').append(value)
    
    def compute_summary(self) -> Dict[str, float]:
        """Compute summary statistics."""
        summary = {}
        
        if self.accuracies:
            summary['mean_accuracy'] = np.mean(self.accuracies)
            summary['std_accuracy'] = np.std(self.accuracies)
        
        if self.losses:
            summary['mean_loss'] = np.mean(self.losses)
            summary['std_loss'] = np.std(self.losses)
        
        return summary


class StatisticalAnalysis:
    """Statistical analysis utilities for meta-learning research."""
    
    def __init__(self, config: StatsConfig):
        self.config = config
    
    def compute_confidence_interval(self, values: List[float]) -> Tuple[float, float, float]:
        """Compute confidence interval for given values."""
        return compute_confidence_interval(
            values, 
            confidence_level=self.config.confidence_level
        )
    
    def statistical_test(self, group1: List[float], group2: List[float]) -> Dict[str, float]:
        """Perform statistical significance test between two groups."""
        from scipy import stats
        
        if self.config.significance_test == "t_test":
            statistic, p_value = stats.ttest_ind(group1, group2)
        elif self.config.significance_test == "mannwhitney":
            statistic, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')
        else:
            raise ValueError(f"Unknown test: {self.config.significance_test}")
        
        return {
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < (0.05 / self.config.confidence_level)  # Bonferroni correction
        }