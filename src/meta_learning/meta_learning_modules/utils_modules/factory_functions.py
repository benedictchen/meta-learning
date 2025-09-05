"""
📋 Factory Functions
=====================

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

"""
"""
Factory Functions and Helper Classes for Meta-Learning 🏭⚙️
==========================================================

🎯 **ELI5 Explanation**:
Think of factory functions like IKEA assembly instructions for AI algorithms!
Just like IKEA gives you step-by-step instructions to build furniture from parts,
factory functions give you easy ways to build complex AI systems from components:

- 🏗️ **create_basic_task_config()**: Like "How to build a basic learning task"
- 📊 **create_evaluation_metrics()**: Like "How to build a report card system"  
- 🎯 **create_dataset_loader()**: Like "How to organize your training examples"
- ⚙️ **create_meta_learner()**: Like "How to assemble the complete learning system"

📊 **Factory Pattern Visualization**:
```
Raw Components:        Factory Function:       Ready-to-Use System:
┌─────────────────┐    ┌─────────────────┐     ┌─────────────────┐
│ • Parameters    │    │                 │     │                 │
│ • Configurations│ ──→│ create_system() │ ──→ │ Working AI      │
│ • Dependencies  │    │                 │     │ Algorithm       │
└─────────────────┘    └─────────────────┘     └─────────────────┘
```

🔧 **Design Pattern Benefits**:
- **Consistent Setup**: Same way to build systems every time
- **Error Prevention**: Automatically handles complex configuration
- **Research Accuracy**: Uses proven parameter combinations from papers
- **Easy Experimentation**: Swap out components like LEGO blocks

Author: Benedict Chen (benedict@benedictchen.com)

This module contains factory functions and helper classes for creating
configured instances of meta-learning components.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import logging
from .configurations import (
    TaskConfiguration, 
    EvaluationConfig,
    DatasetConfig, 
    MetricsConfig,
    StatsConfig,
    CurriculumConfig,
    DiversityConfig
)
from .dataset_sampling import MetaLearningDataset
from .statistical_evaluation import (
    EvaluationMetrics, 
    compute_confidence_interval_research_accurate,
    _auto_select_ci_method
)

logger = logging.getLogger(__name__)


def create_basic_task_config(n_way: int = 5, k_shot: int = 5, q_query: int = 15) -> TaskConfiguration:
    """Create basic task configuration with standard settings."""
    return TaskConfiguration(
        n_way=n_way,
        k_shot=k_shot,
        q_query=q_query,
        num_tasks=1000,
        task_type="classification",
        augmentation_strategy="basic",
        difficulty_estimation_method="pairwise_distance",
        use_research_accurate_difficulty=False
    )


def create_research_accurate_task_config(
    n_way: int = 5, 
    k_shot: int = 5, 
    q_query: int = 15,
    difficulty_method: str = "silhouette"
) -> TaskConfiguration:
    """Create research-accurate task configuration with proper difficulty estimation."""
    return TaskConfiguration(
        n_way=n_way,
        k_shot=k_shot,
        q_query=q_query,
        num_tasks=1000,
        task_type="classification",
        augmentation_strategy="advanced",
        difficulty_estimation_method=difficulty_method,  # "silhouette", "entropy", "knn"
        use_research_accurate_difficulty=True
    )


def create_basic_evaluation_config() -> EvaluationConfig:
    """Create basic evaluation configuration with standard settings."""
    return EvaluationConfig(
        confidence_intervals=True,
        num_bootstrap_samples=1000,
        significance_level=0.05,
        track_adaptation_curve=True,
        compute_uncertainty=True,
        ci_method="bootstrap",
        use_research_accurate_ci=False,
        num_episodes=600,
        min_sample_size_for_bootstrap=30,
        auto_method_selection=False
    )


def create_research_accurate_evaluation_config(ci_method: str = "auto") -> EvaluationConfig:
    """Create research-accurate evaluation configuration with proper CI methods."""
    return EvaluationConfig(
        confidence_intervals=True,
        num_bootstrap_samples=2000,  # Higher for better accuracy
        significance_level=0.05,
        track_adaptation_curve=True,
        compute_uncertainty=True,
        ci_method=ci_method,  # "auto", "t_distribution", "meta_learning_standard", "bca_bootstrap"
        use_research_accurate_ci=True,
        num_episodes=600,  # Standard meta-learning protocol
        min_sample_size_for_bootstrap=30,
        auto_method_selection=(ci_method == "auto")
    )


def create_meta_learning_standard_evaluation_config() -> EvaluationConfig:
    """Create evaluation configuration following standard meta-learning protocols."""
    return EvaluationConfig(
        confidence_intervals=True,
        num_bootstrap_samples=600,  # Not used with t-distribution
        significance_level=0.05,
        track_adaptation_curve=True,
        compute_uncertainty=True,
        ci_method="meta_learning_standard",
        use_research_accurate_ci=True,
        num_episodes=600,
        min_sample_size_for_bootstrap=30,
        auto_method_selection=False
    )


def create_dataset(
    data: torch.Tensor, 
    labels: torch.Tensor, 
    task_config: TaskConfiguration, 
    dataset_config: Optional[DatasetConfig] = None
) -> MetaLearningDataset:
    """Factory function to create a meta-learning dataset."""
    if dataset_config is None:
        dataset_config = DatasetConfig()
    
    return MetaLearningDataset(data, labels, task_config)


def create_metrics_evaluator(config: Optional[MetricsConfig] = None) -> EvaluationMetrics:
    """Factory function to create an evaluation metrics instance."""
    if config is None:
        config = MetricsConfig()
    
    return EvaluationMetrics(config)


def create_curriculum_scheduler(config: Optional[CurriculumConfig] = None) -> 'CurriculumLearning':
    """Factory function to create a curriculum learning scheduler."""
    if config is None:
        config = CurriculumConfig()
    
    return CurriculumLearning(config)


def track_task_diversity(tasks: List[torch.Tensor], config: Optional[DiversityConfig] = None) -> Dict[str, float]:
    """Track diversity across multiple tasks."""
    if config is None:
        config = DiversityConfig()
    
    tracker = TaskDiversityTracker(config)
    
    for task in tasks:
        tracker.add_task(task.mean(dim=0))  # Use mean as task feature
    
    return tracker.compute_diversity()


def evaluate_meta_learning_algorithm(
    algorithm,
    dataset: MetaLearningDataset,
    config: EvaluationConfig = None,
    num_episodes: int = None
) -> Dict[str, Any]:
    """
    Comprehensive evaluation of meta-learning algorithm with configurable methods.
    
    Args:
        algorithm: Meta-learning algorithm to evaluate
        dataset: MetaLearningDataset for evaluation
        config: EvaluationConfig for evaluation settings
        num_episodes: Number of evaluation episodes (overrides config)
        
    Returns:
        Dictionary with evaluation results and statistics
    """
    config = config or create_research_accurate_evaluation_config()
    num_episodes = num_episodes or config.num_episodes
    
    accuracies = []
    adaptation_curves = []
    
    logger.info(f"Starting evaluation with {num_episodes} episodes")
    
    for episode in range(num_episodes):
        # Sample task
        task = dataset.sample_task(task_idx=episode)
        
        # Evaluate algorithm on task
        result = algorithm.evaluate_task(
            task['support']['data'],
            task['support']['labels'],
            task['query']['data'],
            task['query']['labels'],
            return_adaptation_curve=config.track_adaptation_curve
        )
        
        accuracies.append(result['accuracy'])
        
        if config.track_adaptation_curve and 'adaptation_curve' in result:
            adaptation_curves.append(result['adaptation_curve'])
    
    # Compute statistics using configured CI method
    mean_accuracy, ci_lower, ci_upper = compute_confidence_interval_research_accurate(
        accuracies, config
    )
    
    results = {
        'mean_accuracy': mean_accuracy,
        'std_accuracy': np.std(accuracies),
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'all_accuracies': accuracies,
        'num_episodes': num_episodes,
        'ci_method_used': config.ci_method if not config.auto_method_selection 
                         else _auto_select_ci_method(accuracies, config)
    }
    
    if config.track_adaptation_curve and adaptation_curves:
        results['adaptation_curves'] = adaptation_curves
        results['mean_adaptation_curve'] = np.mean(adaptation_curves, axis=0).tolist()
    
    logger.info(f"Evaluation complete: {mean_accuracy:.3f} ± {ci_upper - ci_lower:.3f}")
    return results


class CurriculumLearning:
    """Curriculum learning implementation for meta-learning."""
    
    def __init__(self, config: CurriculumConfig):
        self.config = config
        self.current_difficulty = config.initial_difficulty
        self.patience_counter = 0
    
    def update_difficulty(self, performance_metric: float) -> float:
        """Update curriculum difficulty based on performance."""
        if performance_metric >= self.config.difficulty_threshold:
            self.current_difficulty = min(
                1.0, 
                self.current_difficulty + self.config.difficulty_increment
            )
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            
            if self.patience_counter >= self.config.adaptation_patience:
                # Reduce difficulty if struggling
                self.current_difficulty = max(
                    0.1,
                    self.current_difficulty - self.config.difficulty_increment / 2
                )
                self.patience_counter = 0
        
        return self.current_difficulty
    
    def get_current_difficulty(self) -> float:
        """Get current curriculum difficulty level."""
        return self.current_difficulty


class TaskDiversityTracker:
    """Track diversity of meta-learning tasks."""
    
    def __init__(self, config: DiversityConfig):
        self.config = config
        self.task_features = []
        self.class_distributions = []
    
    def add_task(self, task_features: torch.Tensor, class_distribution: Optional[torch.Tensor] = None):
        """Add a new task for diversity tracking."""
        self.task_features.append(task_features.detach().cpu())
        
        if class_distribution is not None and self.config.track_class_distribution:
            self.class_distributions.append(class_distribution.detach().cpu())
    
    def compute_diversity(self) -> Dict[str, float]:
        """Compute task diversity metrics."""
        if not self.task_features:
            return {'diversity_score': 0.0}
        
        if self.config.diversity_metric == "cosine_similarity":
            # Handle different task sizes by computing mean features first
            if len(self.task_features) < 2:
                return {'diversity_score': 0.0}
            
            # Compute mean features for each task 
            task_means = []
            for task_features in self.task_features:
                task_mean = task_features.mean(dim=0)
                task_means.append(task_mean)
            
            # Stack the mean features
            features = torch.stack(task_means)
            
            # Compute pairwise cosine similarities
            normalized_features = F.normalize(features, dim=-1)
            if normalized_features.dim() == 1:
                normalized_features = normalized_features.unsqueeze(0)
            similarities = torch.mm(normalized_features, normalized_features.t())
            
            # Average off-diagonal similarities (diversity = 1 - similarity)
            mask = ~torch.eye(similarities.size(0), dtype=bool)
            if mask.sum() > 0:
                avg_similarity = similarities[mask].mean().item()
                diversity_score = 1.0 - avg_similarity
                # Clamp to valid range [0, 1]
                diversity_score = max(0.0, min(1.0, diversity_score))
            else:
                diversity_score = 0.0
        
        elif self.config.diversity_metric == "feature_variance":
            try:
                # SOLUTION 1: Task-agnostic diversity using feature variance (Chen et al. 2020)
                all_features = torch.cat(self.task_features, dim=0)
                if all_features.size(0) == 0:
                    raise ValueError("Empty feature tensor provided")
                
                feature_variance = torch.var(all_features, dim=0).mean().item()
                # Normalize to [0,1] with configurable scaling
                variance_scale = getattr(self.config, 'variance_scale', 2.0)
                diversity_score = min(1.0, feature_variance / variance_scale)
                
            except Exception as e:
                print(f"Warning: Feature variance diversity computation failed: {str(e)}")
                diversity_score = 0.5
        
        elif self.config.diversity_metric == "silhouette_score":
            try:
                # SOLUTION 2: Class separation diversity metric (Rousseeuw 1987)
                if len(self.task_features) == 0:
                    raise ValueError("Empty task features list")
                
                # Collect all features and labels from tasks
                all_task_features = []
                all_task_labels = []
                
                for i, features in enumerate(self.task_features):
                    all_task_features.append(features)
                    # Create synthetic labels for this task's data
                    task_labels = torch.full((features.size(0),), i, dtype=torch.long)
                    all_task_labels.append(task_labels)
                
                if len(all_task_features) > 1:
                    try:
                        # Import sklearn with proper error handling
                        from sklearn.metrics import silhouette_score
                    except ImportError:
                        print("Warning: sklearn not available for silhouette diversity metric. Using fallback.")
                        diversity_score = 0.5
                    else:
                        all_features_tensor = torch.cat(all_task_features, dim=0)
                        all_labels_tensor = torch.cat(all_task_labels, dim=0)
                        
                        if torch.unique(all_labels_tensor).size(0) > 1:
                            all_features_np = all_features_tensor.detach().cpu().numpy()
                            all_labels_np = all_labels_tensor.detach().cpu().numpy()
                            
                            # Ensure features are 2D for sklearn
                            if all_features_np.ndim == 1:
                                all_features_np = all_features_np.reshape(-1, 1)
                            
                            silhouette = silhouette_score(all_features_np, all_labels_np)
                            diversity_score = (silhouette + 1) / 2  # Convert from [-1,1] to [0,1]
                        else:
                            diversity_score = 0.0  # No diversity with single task
                else:
                    diversity_score = 0.0  # No diversity with single task
                    
            except Exception as e:
                print(f"Warning: Silhouette diversity computation failed: {str(e)}")
                diversity_score = 0.5
        
        elif self.config.diversity_metric == "information_theoretic":
            try:
                # SOLUTION 3: Information-theoretic diversity (Shannon entropy)
                if len(self.task_features) == 0:
                    raise ValueError("Empty task features list")
                
                try:
                    from scipy.spatial.distance import pdist
                    import numpy as np
                except ImportError:
                    print("Warning: scipy not available for information-theoretic diversity. Using fallback.")
                    diversity_score = 0.5
                else:
                    all_features = torch.cat(self.task_features, dim=0)
                    if all_features.size(0) < 2:
                        diversity_score = 0.0
                    else:
                        all_features_np = all_features.detach().cpu().numpy()
                        
                        # Ensure features are 2D for scipy
                        if all_features_np.ndim == 1:
                            all_features_np = all_features_np.reshape(-1, 1)
                        
                        # Compute pairwise distances
                        distances = pdist(all_features_np, metric='euclidean')
                        if len(distances) == 0 or distances.sum() == 0:
                            diversity_score = 0.0
                        else:
                            # Convert distances to probability distribution
                            probs = distances / distances.sum()
                            # Remove zero probabilities to avoid log(0)
                            probs = probs[probs > 0]
                            
                            # Compute Shannon entropy as diversity measure
                            entropy_val = -np.sum(probs * np.log(probs))
                            max_entropy = np.log(len(probs)) if len(probs) > 1 else 1.0
                            diversity_score = entropy_val / max_entropy  # Normalize to [0,1]
                            diversity_score = min(1.0, max(0.0, diversity_score))
                
            except Exception as e:
                print(f"Warning: Information-theoretic diversity computation failed: {str(e)}")
                diversity_score = 0.5
        
        elif self.config.diversity_metric == "jensen_shannon_divergence":
            try:
                # SOLUTION 4: Jensen-Shannon divergence between task feature distributions
                if len(self.task_features) < 2:
                    diversity_score = 0.0
                else:
                    try:
                        from scipy.spatial.distance import jensenshannon
                        import numpy as np
                    except ImportError:
                        print("Warning: scipy not available for Jensen-Shannon diversity. Using fallback.")
                        diversity_score = 0.5
                    else:
                        # Compute feature histograms for each task
                        task_histograms = []
                        n_bins = getattr(self.config, 'histogram_bins', 50)
                        
                        # Find global feature range
                        all_features = torch.cat(self.task_features, dim=0)
                        feature_min = all_features.min().item()
                        feature_max = all_features.max().item()
                        
                        for features in self.task_features:
                            # Flatten features and compute histogram
                            flat_features = features.flatten().detach().cpu().numpy()
                            hist, _ = np.histogram(flat_features, bins=n_bins, 
                                                 range=(feature_min, feature_max), density=True)
                            # Normalize to probability distribution
                            hist = hist / (hist.sum() + 1e-8)
                            task_histograms.append(hist)
                        
                        # Compute average pairwise Jensen-Shannon divergence
                        js_divergences = []
                        for i in range(len(task_histograms)):
                            for j in range(i + 1, len(task_histograms)):
                                js_div = jensenshannon(task_histograms[i], task_histograms[j])
                                js_divergences.append(js_div)
                        
                        if js_divergences:
                            diversity_score = np.mean(js_divergences)
                        else:
                            diversity_score = 0.0
                
            except Exception as e:
                print(f"Warning: Jensen-Shannon diversity computation failed: {str(e)}")
                diversity_score = 0.5
        
        else:
            print(f"Warning: Unknown diversity metric '{self.config.diversity_metric}'. "
                  f"Supported methods: cosine_similarity, feature_variance, silhouette_score, "
                  f"information_theoretic, jensen_shannon_divergence")
            diversity_score = 0.5
        
        return {'diversity_score': diversity_score}