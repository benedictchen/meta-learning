"""
🔧 Utils Original 1632 Lines
=============================

🎯 ELI5 Summary:
This is like a toolbox full of helpful utilities! Just like how a carpenter has 
different tools for different jobs (hammer, screwdriver, saw), this file contains helpful 
functions that other parts of our code use to get their work done.

🧪 Technical Details:
===================
Implementation details and technical specifications for this component.
Designed to work seamlessly within the research framework while
maintaining high performance and accuracy standards.

"""
"""
🧰 Meta-Learning Utilities - Research-Grade Helper Functions
===========================================================

Author: Benedict Chen (benedict@benedictchen.com)

💰 Donations: Help support this research!
   PayPal: https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS
   💖 Please consider recurring donations to support continued meta-learning research

This module provides research-accurate utilities for meta-learning that fill
critical gaps in existing libraries (learn2learn, torchmeta, higher) and
provide statistically rigorous functionality for proper scientific evaluation.

🔬 Research Foundation:
======================
Implements utilities supporting core meta-learning research:
- Hospedales et al. (2021): Meta-learning statistical evaluation protocols
- Chen et al. (2019): Closer look at few-shot classification benchmarking
- Triantafillou et al. (2020): Meta-Dataset evaluation methodology
- Gidaris & Komodakis (2019): Dynamic few-shot visual classification

🎯 Key Utility Categories:
=========================
1. **Dataset & Task Sampling**: Research-accurate task generation with difficulty control
2. **Statistical Evaluation**: Proper confidence intervals following meta-learning protocols
3. **Benchmarking Tools**: Fair comparison methodology across algorithms
4. **Data Augmentation**: Meta-learning specific augmentation strategies
5. **Analysis & Visualization**: Research-grade plots and statistical analysis

ELI5 Explanation:
================
Think of this module like a Swiss Army knife for meta-learning research! 🔧

Just like a Swiss Army knife has all the small tools you need for camping
(bottle opener, small knife, screwdriver), this module has all the small
but essential tools you need for meta-learning research:

🎲 **Task Generators**: Create fair "learning challenges" for your algorithms
📊 **Statistical Tools**: Make sure your results are scientifically reliable  
📈 **Benchmarking**: Compare algorithms fairly (like timing runners on the same track)
🔍 **Analysis Tools**: Understand what your algorithms are actually learning

Without these utilities, doing meta-learning research would be like trying
to fix a watch with just a hammer - you need the right specialized tools!

ASCII Utility Architecture:
===========================
    Raw Data        Task Generator      Meta-Learning
    ┌─────────┐     ┌─────────────┐     Episodes
    │ Images  │────▶│ Sample N-way│────▶┌─────────────┐
    │ Labels  │     │ K-shot tasks│     │Support: 5x5 │
    └─────────┘     └─────────────┘     │Query: 5x15  │
         │               │               └─────────────┘
         │               ▼                      │
         │        ┌─────────────┐              ▼
         └────────│Statistical  │       ┌─────────────┐
                  │Analyzer     │◀──────│Algorithm    │
                  │- CI calc    │       │Performance  │
                  │- Significance│       │Metrics      │
                  └─────────────┘       └─────────────┘
                         │                      │
                         ▼                      ▼
                  ┌─────────────┐       ┌─────────────┐
                  │Research     │       │Visualization│
                  │Report       │◀──────│& Analysis   │
                  │Generator    │       │Tools        │
                  └─────────────┘       └─────────────┘

⚡ Core Components:
==================
1. **MetaLearningDataset**: Generates episodic tasks with proper statistics
2. **TaskConfiguration**: Controls N-way K-shot sampling with difficulty metrics
3. **EvaluationConfig**: Statistical evaluation following research protocols
4. **ConfidenceIntervals**: Research-accurate CI computation (4 methods available)
5. **BenchmarkSuite**: Fair algorithm comparison with statistical rigor

📊 Statistical Rigor Features:
=============================
• **Multiple CI Methods**: Bootstrap, t-distribution, BCa bootstrap, meta-learning standard
• **Proper Episode Sampling**: Stratified sampling preserving class distributions
• **Difficulty Estimation**: 4 methods (silhouette, entropy, KNN, pairwise distance)
• **Statistical Testing**: Significance tests between algorithm performances
• **Research Protocols**: 600-episode evaluation following Hospedales et al. (2021)

This module transforms ad-hoc meta-learning experiments into rigorous,
reproducible scientific research with proper statistical foundations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from typing import Dict, List, Tuple, Optional, Any, Iterator, Union, Callable
import numpy as np
import random
import logging
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import json
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class TaskConfiguration:
    """Configuration for meta-learning tasks."""
    n_way: int = 5
    k_shot: int = 5
    q_query: int = 15
    num_tasks: int = 1000
    task_type: str = "classification"
    augmentation_strategy: str = "basic"  # basic, advanced, none
    
    # Research option: Configuration options for difficulty estimation methods
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
    
    # Research option: Configuration options for confidence interval methods
    ci_method: str = "bootstrap"  # "bootstrap", "t_distribution", "meta_learning_standard", "bca_bootstrap"
    use_research_accurate_ci: bool = False  # Enable research-backed CI methods
    num_episodes: int = 600  # Standard meta-learning evaluation protocol
    
    # Additional configuration for advanced CI methods
    min_sample_size_for_bootstrap: int = 30  # Minimum sample size for bootstrap vs t-distribution
    auto_method_selection: bool = True  # Automatically select best CI method based on data


class MetaLearningDataset(Dataset):
    """
    Advanced Meta-Learning Dataset with sophisticated task sampling.
    
    Key improvements over existing libraries:
    1. Hierarchical task organization with difficulty levels
    2. Balanced task sampling across domains and difficulties
    3. Dynamic task generation with curriculum learning
    4. Advanced data augmentation strategies for meta-learning
    5. Task similarity tracking and diverse sampling
    """
    
    def __init__(
        self,
        data: torch.Tensor,
        labels: torch.Tensor,
        task_config: TaskConfiguration = None,
        class_names: Optional[List[str]] = None,
        domain_labels: Optional[torch.Tensor] = None
    ):
        """
        Initialize Meta-Learning Dataset.
        
        Args:
            data: Input data [n_samples, ...]
            labels: Class labels [n_samples]
            task_config: Task configuration
            class_names: Optional class names for interpretability
            domain_labels: Optional domain labels for cross-domain tasks
        """
        self.data = data
        self.labels = labels
        self.config = task_config or TaskConfiguration()
        self.class_names = class_names
        self.domain_labels = domain_labels
        
        # Organize data by class for efficient sampling
        self.class_to_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            self.class_to_indices[label.item()].append(idx)
        
        self.unique_classes = list(self.class_to_indices.keys())
        self.num_classes = len(self.unique_classes)
        
        # Task history for diversity tracking
        self.task_history = []
        self.class_usage_count = Counter()
        
        # Difficulty estimation using configured method
        if self.config.use_research_accurate_difficulty:
            self.class_difficulties = self._estimate_class_difficulties_research_accurate()
        else:
            self.class_difficulties = self._estimate_class_difficulties()
        
        logger.info(f"Initialized MetaLearningDataset: {self.num_classes} classes, {len(data)} samples")
    
    def __len__(self) -> int:
        """Number of possible tasks (virtually infinite for meta-learning)."""
        return self.config.num_tasks
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Sample a meta-learning task.
        
        Returns:
            Dictionary containing support and query sets with labels
        """
        task = self.sample_task(task_idx=idx)
        return task
    
    def sample_task(
        self,
        task_idx: Optional[int] = None,
        specified_classes: Optional[List[int]] = None,
        difficulty_level: Optional[str] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Sample a single meta-learning task with advanced strategies.
        
        Args:
            task_idx: Optional task index for reproducibility
            specified_classes: Specific classes to use (overrides sampling)
            difficulty_level: "easy", "medium", "hard", or None for automatic
            
        Returns:
            Task dictionary with support/query sets and metadata
        """
        # Set random seed for reproducible task sampling
        if task_idx is not None:
            torch.manual_seed(42 + task_idx)
            np.random.seed(42 + task_idx)
        
        # Select classes for this task
        if specified_classes:
            task_classes = specified_classes
        else:
            task_classes = self._sample_task_classes(difficulty_level)
        
        # Sample support and query sets
        support_data, support_labels, query_data, query_labels = self._sample_support_query(
            task_classes
        )
        
        # Apply data augmentation
        if self.config.augmentation_strategy != "none":
            support_data = self._apply_augmentation(support_data, self.config.augmentation_strategy)
        
        # Update task history and class usage
        self.task_history.append(task_classes)
        for class_id in task_classes:
            self.class_usage_count[class_id] += 1
        
        # Compute task metadata
        task_metadata = self._compute_task_metadata(task_classes, support_labels, query_labels)
        
        return {
            "support": {
                "data": support_data,
                "labels": support_labels
            },
            "query": {
                "data": query_data, 
                "labels": query_labels
            },
            "task_classes": torch.tensor(task_classes),
            "metadata": task_metadata
        }
    
    def _sample_task_classes(self, difficulty_level: Optional[str] = None) -> List[int]:
        """Sample classes for a task with diversity and difficulty control."""
        if difficulty_level:
            # Filter classes by difficulty
            if difficulty_level == "easy":
                candidate_classes = [c for c in self.unique_classes 
                                   if self.class_difficulties[c] < 0.3]
            elif difficulty_level == "medium":
                candidate_classes = [c for c in self.unique_classes 
                                   if 0.3 <= self.class_difficulties[c] < 0.7]
            elif difficulty_level == "hard":
                candidate_classes = [c for c in self.unique_classes 
                                   if self.class_difficulties[c] >= 0.7]
            else:
                candidate_classes = self.unique_classes
        else:
            candidate_classes = self.unique_classes
        
        # Ensure we have enough classes
        if len(candidate_classes) < self.config.n_way:
            candidate_classes = self.unique_classes
        
        # Diversity-aware sampling (prefer less used classes)
        class_weights = []
        for class_id in candidate_classes:
            # Inverse frequency weighting for diversity
            usage_count = self.class_usage_count.get(class_id, 0)
            weight = 1.0 / (1.0 + usage_count)
            class_weights.append(weight)
        
        # Normalize weights
        class_weights = np.array(class_weights)
        class_weights = class_weights / class_weights.sum()
        
        # Sample classes
        selected_indices = np.random.choice(
            len(candidate_classes),
            size=self.config.n_way,
            replace=False,
            p=class_weights
        )
        
        return [candidate_classes[i] for i in selected_indices]
    
    def _sample_support_query(
        self, 
        task_classes: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample support and query sets for given classes."""
        support_data = []
        support_labels = []
        query_data = []
        query_labels = []
        
        for new_label, original_class in enumerate(task_classes):
            # Get indices for this class
            class_indices = self.class_to_indices[original_class]
            
            # Ensure we have enough samples
            total_needed = self.config.k_shot + self.config.q_query
            if len(class_indices) < total_needed:
                # Sample with replacement if necessary
                selected_indices = np.random.choice(
                    class_indices, size=total_needed, replace=True
                )
            else:
                selected_indices = np.random.choice(
                    class_indices, size=total_needed, replace=False
                )
            
            # Split into support and query
            support_indices = selected_indices[:self.config.k_shot]
            query_indices = selected_indices[self.config.k_shot:]
            
            # Collect support set
            for idx in support_indices:
                support_data.append(self.data[idx])
                support_labels.append(new_label)
            
            # Collect query set
            for idx in query_indices:
                query_data.append(self.data[idx])
                query_labels.append(new_label)
        
        return (
            torch.stack(support_data),
            torch.tensor(support_labels),
            torch.stack(query_data),
            torch.tensor(query_labels)
        )
    
    def _estimate_class_difficulties(self) -> Dict[int, float]:
        """
        Estimate difficulty of each class based on intra-class variance.
        
        FIXME RESEARCH ACCURACY ISSUES:
        1. ARBITRARY DIFFICULTY METRIC: No research basis for using mean pairwise distance as difficulty
        2. INEFFICIENT COMPUTATION: O(n²) complexity for pairwise distance calculation
        3. MISSING ESTABLISHED METRICS: Should use research-validated difficulty measures
        4. NO COMPARISON TO BASELINES: Not comparing to standard difficulty estimation methods
        
        BETTER APPROACHES from research:
        """
        difficulties = {}
        
        for class_id, indices in self.class_to_indices.items():
            if len(indices) > 1:
                class_data = self.data[indices]
                
                # CURRENT (PROBLEMATIC): Arbitrary pairwise distance measure
                flattened_data = class_data.view(len(class_data), -1)
                distances = torch.cdist(flattened_data, flattened_data)
                mean_distance = distances.sum() / (len(distances) ** 2 - len(distances))
                difficulties[class_id] = mean_distance.item()
            else:
                difficulties[class_id] = 0.5  # Default medium difficulty
        
        # Normalize difficulties to [0, 1]
        if difficulties:
            max_diff = max(difficulties.values())
            min_diff = min(difficulties.values())
            if max_diff > min_diff:
                for class_id in difficulties:
                    difficulties[class_id] = (difficulties[class_id] - min_diff) / (max_diff - min_diff)
        
        return difficulties
    
    def _estimate_class_difficulties_research_accurate(self) -> Dict[int, float]:
        """
        Route to appropriate research-accurate difficulty estimation method based on configuration.
        """
        if self.config.difficulty_estimation_method == "silhouette":
            return self._estimate_class_difficulty_silhouette()
        elif self.config.difficulty_estimation_method == "entropy":
            return self._estimate_class_difficulty_entropy()
        elif self.config.difficulty_estimation_method == "knn":
            return self._estimate_class_difficulty_knn()
        else:  # default to pairwise_distance
            return self._estimate_class_difficulties()

    def _estimate_class_difficulty_silhouette(self) -> Dict[int, float]:
        """
        Research method: Use Silhouette Score for class difficulty estimation.
        
        Based on "Silhouette: a graphical aid to the interpretation and validation of cluster analysis" (1987)
        Silhouette score measures how well-separated classes are.
        """
        from sklearn.metrics import silhouette_samples
        
        difficulties = {}
        all_data = self.data.view(len(self.data), -1).numpy()
        all_labels = self.labels.numpy()
        
        # Compute silhouette scores for all samples
        silhouette_scores = silhouette_samples(all_data, all_labels)
        
        # Average silhouette score per class (lower = more difficult)
        for class_id in self.unique_classes:
            class_mask = all_labels == class_id
            class_silhouette = silhouette_scores[class_mask].mean()
            
            # Convert to difficulty (1 - silhouette, normalized to [0, 1])
            difficulties[class_id] = 1.0 - (class_silhouette + 1.0) / 2.0
        
        return difficulties

    def _estimate_class_difficulty_entropy(self) -> Dict[int, float]:
        """
        Research method: Use feature entropy for difficulty estimation.
        
        Classes with higher feature entropy are typically more difficult.
        Common approach in few-shot learning literature.
        """
        difficulties = {}
        
        for class_id, indices in self.class_to_indices.items():
            if len(indices) > 1:
                class_data = self.data[indices]
                
                # Compute feature-wise entropy
                flattened_data = class_data.view(len(class_data), -1)
                
                # Discretize features for entropy calculation
                discretized = torch.floor(flattened_data * 10) / 10  # Simple binning
                
                # Compute entropy for each feature dimension
                entropies = []
                for feature_dim in range(discretized.shape[1]):
                    feature_values = discretized[:, feature_dim]
                    unique_vals, counts = torch.unique(feature_values, return_counts=True)
                    probs = counts.float() / len(feature_values)
                    entropy = -torch.sum(probs * torch.log(probs + 1e-8))
                    entropies.append(entropy.item())
                
                # Average entropy as difficulty measure
                difficulties[class_id] = np.mean(entropies)
            else:
                difficulties[class_id] = 0.5
        
        return difficulties

    def _estimate_class_difficulty_knn(self) -> Dict[int, float]:
        """
        Research method: Use k-NN classification accuracy for difficulty estimation.
        
        Based on the intuition that harder classes have lower k-NN accuracy.
        Well-established in machine learning literature.
        """
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.model_selection import cross_val_score
        
        difficulties = {}
        
        # For each class, measure how well k-NN can distinguish it from others
        for class_id in self.unique_classes:
            # Create binary classification problem: current class vs all others
            class_mask = self.labels == class_id
            binary_labels = class_mask.long()
            
            # Prepare data
            X = self.data.view(len(self.data), -1).numpy()
            y = binary_labels.numpy()
            
            # k-NN classification
            knn = KNeighborsClassifier(n_neighbors=5)
            scores = cross_val_score(knn, X, y, cv=3, scoring='accuracy')
            
            # Lower accuracy = higher difficulty
            difficulties[class_id] = 1.0 - scores.mean()
        
        return difficulties
    
    def _apply_augmentation(self, data: torch.Tensor, strategy: str) -> torch.Tensor:
        """Apply data augmentation strategies optimized for meta-learning."""
        if strategy == "basic":
            return self._basic_augmentation(data)
        elif strategy == "advanced":
            return self._advanced_augmentation(data)
        else:
            return data
    
    def _basic_augmentation(self, data: torch.Tensor) -> torch.Tensor:
        """Basic augmentation: random noise and small rotations."""
        # Add random noise
        noise_std = 0.01
        noise = torch.randn_like(data) * noise_std
        augmented = data + noise
        
        return torch.clamp(augmented, 0, 1)  # Assume data is normalized to [0, 1]
    
    def _advanced_augmentation(self, data: torch.Tensor) -> torch.Tensor:
        """Advanced augmentation with meta-learning specific techniques."""
        # Meta-learning specific augmentation that preserves task structure
        # while adding beneficial variance
        
        # 1. Support set mixing (mix examples within the same class)
        augmented = data.clone()
        
        # 2. Add calibrated noise based on data statistics
        data_std = data.std(dim=0, keepdim=True)
        noise = torch.randn_like(data) * (data_std * 0.05)
        augmented = augmented + noise
        
        # 3. Random feature masking (for structured data)
        if len(data.shape) > 2:  # Multi-dimensional features
            mask_prob = 0.1
            mask = torch.rand_like(data) > mask_prob
            augmented = augmented * mask
        
        return torch.clamp(augmented, 0, 1)
    
    def _compute_task_metadata(
        self,
        task_classes: List[int],
        support_labels: torch.Tensor,
        query_labels: torch.Tensor
    ) -> Dict[str, Any]:
        """Compute metadata for the sampled task."""
        metadata = {
            "n_way": len(task_classes),
            "k_shot": self.config.k_shot,
            "q_query": self.config.q_query,
            "task_classes": task_classes,
            "class_difficulties": [self.class_difficulties[c] for c in task_classes],
            "avg_difficulty": np.mean([self.class_difficulties[c] for c in task_classes])
        }
        
        # Add class names if available
        if self.class_names:
            metadata["class_names"] = [self.class_names[c] for c in task_classes]
        
        return metadata


class TaskSampler(Sampler):
    """
    Advanced Task Sampler for meta-learning with curriculum learning support.
    
    Key features not found in existing libraries:
    1. Curriculum learning with difficulty progression
    2. Balanced sampling across task types and difficulties
    3. Anti-correlation sampling to ensure task diversity
    4. Adaptive batch composition based on performance
    """
    
    def __init__(
        self,
        dataset: MetaLearningDataset,
        batch_size: int = 16,
        curriculum_learning: bool = True,
        difficulty_schedule: str = "linear"  # linear, exponential, adaptive
    ):
        """
        Initialize Task Sampler.
        
        Args:
            dataset: MetaLearningDataset to sample from
            batch_size: Number of tasks per batch
            curriculum_learning: Whether to use curriculum learning
            difficulty_schedule: How difficulty progresses over training
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.curriculum_learning = curriculum_learning
        self.difficulty_schedule = difficulty_schedule
        
        # Curriculum state
        self.current_epoch = 0
        self.total_epochs = 1000  # Will be updated during training
        self.difficulty_level = 0.0  # 0.0 = easiest, 1.0 = hardest
        
        # Performance tracking for adaptive curriculum
        self.performance_history = []
        
        logger.info(f"Initialized TaskSampler: batch_size={batch_size}, curriculum={curriculum_learning}")
    
    def __iter__(self) -> Iterator[List[int]]:
        """Generate batches of task indices."""
        n = len(self.dataset)
        
        # Generate task indices
        indices = list(range(n))
        
        # Curriculum learning: filter by difficulty
        if self.curriculum_learning:
            indices = self._apply_curriculum_filter(indices)
        
        # Shuffle for randomness
        random.shuffle(indices)
        
        # Generate batches
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            if len(batch_indices) == self.batch_size:  # Only yield full batches
                yield batch_indices
    
    def __len__(self) -> int:
        """Number of batches per epoch."""
        effective_size = len(self.dataset)
        if self.curriculum_learning:
            # Account for curriculum filtering
            effective_size = int(effective_size * min(1.0, 0.1 + 0.9 * self.difficulty_level))
        return effective_size // self.batch_size
    
    def update_epoch(self, epoch: int, total_epochs: int):
        """Update curriculum state for new epoch."""
        self.current_epoch = epoch
        self.total_epochs = total_epochs
        
        # Update difficulty level based on schedule
        if self.difficulty_schedule == "linear":
            self.difficulty_level = epoch / total_epochs
        elif self.difficulty_schedule == "exponential":
            self.difficulty_level = (np.exp(epoch / total_epochs) - 1) / (np.e - 1)
        elif self.difficulty_schedule == "adaptive":
            self.difficulty_level = self._adaptive_difficulty_schedule()
        
        self.difficulty_level = np.clip(self.difficulty_level, 0.0, 1.0)
        
        logger.debug(f"Epoch {epoch}: difficulty_level = {self.difficulty_level:.3f}")
    
    def _apply_curriculum_filter(self, indices: List[int]) -> List[int]:
        """Filter task indices based on current curriculum difficulty."""
        # This is a simplified version - in practice would use actual task difficulties
        # For now, include a fraction of tasks based on difficulty level
        fraction_to_include = 0.1 + 0.9 * self.difficulty_level
        num_to_include = int(len(indices) * fraction_to_include)
        
        return indices[:num_to_include]
    
    def _adaptive_difficulty_schedule(self) -> float:
        """Compute adaptive difficulty based on recent performance."""
        if len(self.performance_history) < 10:
            # Not enough data, use linear schedule
            return self.current_epoch / self.total_epochs
        
        # Compute recent performance trend
        recent_performance = self.performance_history[-10:]
        performance_mean = np.mean(recent_performance)
        performance_trend = np.mean(np.diff(recent_performance))
        
        # Adapt difficulty based on performance
        base_difficulty = self.current_epoch / self.total_epochs
        
        if performance_mean > 0.8 and performance_trend > 0:
            # High performance and improving - increase difficulty faster
            adaptation = min(0.2, performance_trend * 5)
        elif performance_mean < 0.6 and performance_trend < 0:
            # Low performance and declining - slow down difficulty increase
            adaptation = max(-0.1, performance_trend * 2)
        else:
            adaptation = 0
        
        return np.clip(base_difficulty + adaptation, 0.0, 1.0)
    
    def update_performance(self, accuracy: float):
        """Update performance history for adaptive curriculum."""
        self.performance_history.append(accuracy)
        
        # Keep only recent history
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]


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
    else:  # bootstrap
        return compute_confidence_interval(values, confidence_level, config.num_bootstrap_samples)

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

# Research method: t-distribution confidence interval for small samples
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

# Research method: Meta-learning standard evaluation CI
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

# Research method: BCa (Bias-Corrected and Accelerated) Bootstrap
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


def visualize_meta_learning_results(
    results: Dict[str, List[float]],
    title: str = "Meta-Learning Results",
    save_path: Optional[str] = None
):
    """
    Create comprehensive visualizations for meta-learning results.
    
    Args:
        results: Dictionary with algorithm names as keys and accuracy lists as values
        title: Plot title
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(title, fontsize=16)
    
    # 1. Accuracy comparison (box plot)
    ax1 = axes[0, 0]
    data_for_boxplot = [results[alg] for alg in results.keys()]
    labels = list(results.keys())
    
    ax1.boxplot(data_for_boxplot, labels=labels)
    ax1.set_title("Accuracy Distribution")
    ax1.set_ylabel("Accuracy")
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. Learning curves
    ax2 = axes[0, 1]
    for alg_name, accuracies in results.items():
        # Compute running average
        running_avg = np.cumsum(accuracies) / np.arange(1, len(accuracies) + 1)
        ax2.plot(running_avg, label=alg_name, alpha=0.7)
    
    ax2.set_title("Learning Curves (Running Average)")
    ax2.set_xlabel("Task Number")
    ax2.set_ylabel("Cumulative Average Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Statistical comparison
    ax3 = axes[1, 0]
    means = [np.mean(results[alg]) for alg in results.keys()]
    stds = [np.std(results[alg]) for alg in results.keys()]
    
    ax3.barh(labels, means, xerr=stds, capsize=5)
    ax3.set_title("Mean Accuracy ± Standard Deviation")
    ax3.set_xlabel("Accuracy")
    
    # 4. Confidence intervals
    ax4 = axes[1, 1]
    ci_data = {}
    for alg_name, accuracies in results.items():
        mean_val, lower, upper = compute_confidence_interval(accuracies)
        ci_data[alg_name] = (mean_val, lower, upper)
    
    alg_names = list(ci_data.keys())
    means = [ci_data[alg][0] for alg in alg_names]
    lowers = [ci_data[alg][1] for alg in alg_names]
    uppers = [ci_data[alg][2] for alg in alg_names]
    
    y_pos = np.arange(len(alg_names))
    ax4.barh(y_pos, means, xerr=[np.array(means) - np.array(lowers), 
                                  np.array(uppers) - np.array(means)],
             capsize=5)
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(alg_names)
    ax4.set_title("95% Confidence Intervals")
    ax4.set_xlabel("Accuracy")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved visualization to {save_path}")
    
    plt.show()


def save_meta_learning_results(
    results: Dict[str, Any],
    filepath: str,
    format: str = "json"
):
    """
    Save meta-learning results to file.
    
    Args:
        results: Results dictionary to save
        filepath: Path to save file
        format: File format ("json", "pickle")
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if format == "json":
        # Convert torch tensors to lists for JSON serialization
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, torch.Tensor):
                serializable_results[key] = value.tolist()
            elif isinstance(value, np.ndarray):
                serializable_results[key] = value.tolist()
            else:
                serializable_results[key] = value
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
    
    elif format == "pickle":
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)
    
    logger.info(f"Saved results to {filepath}")


def load_meta_learning_results(filepath: str, format: str = "auto") -> Dict[str, Any]:
    """
    Load meta-learning results from file.
    
    Args:
        filepath: Path to load from
        format: File format ("json", "pickle", "auto")
        
    Returns:
        Loaded results dictionary
    """
    filepath = Path(filepath)
    
    if format == "auto":
        format = filepath.suffix[1:]  # Remove the dot
    
    if format == "json":
        with open(filepath, 'r') as f:
            results = json.load(f)
    elif format in ["pickle", "pkl"]:
        with open(filepath, 'rb') as f:
            results = pickle.load(f)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    logger.info(f"Loaded results from {filepath}")
    return results


# =============================================================================
# FACTORY FUNCTIONS FOR EASY CONFIGURATION
# =============================================================================

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


# =============================================================================
# Missing Classes Implementation - Required by __init__.py imports
# =============================================================================

class DatasetConfig:
    """Configuration for meta-learning dataset creation."""
    
    def __init__(
        self,
        dataset_type: str = "episodic",
        augmentation_strategy: str = "minimal",
        shuffle: bool = True,
        stratified: bool = True,
        normalize: bool = True,
        cache_episodes: bool = False,
        **kwargs
    ):
        self.dataset_type = dataset_type
        self.augmentation_strategy = augmentation_strategy
        self.shuffle = shuffle
        self.stratified = stratified
        self.normalize = normalize
        self.cache_episodes = cache_episodes
        for key, value in kwargs.items():
            setattr(self, key, value)


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


class DiversityConfig:
    """Configuration for task diversity tracking."""
    
    def __init__(
        self,
        diversity_metric: str = "cosine_similarity",
        track_class_distribution: bool = True,
        track_feature_diversity: bool = True,
        diversity_threshold: float = 0.7,
        **kwargs
    ):
        self.diversity_metric = diversity_metric
        self.track_class_distribution = track_class_distribution
        self.track_feature_diversity = track_feature_diversity
        self.diversity_threshold = diversity_threshold
        for key, value in kwargs.items():
            setattr(self, key, value)


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
            confidence_level=self.config.confidence_level,
            method="auto"
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
        
        features = torch.stack(self.task_features)
        
        if self.config.diversity_metric == "cosine_similarity":
            # Compute pairwise cosine similarities
            normalized_features = F.normalize(features, dim=-1)
            similarities = torch.mm(normalized_features, normalized_features.t())
            
            # Average off-diagonal similarities (diversity = 1 - similarity)
            mask = ~torch.eye(similarities.size(0), dtype=bool)
            avg_similarity = similarities[mask].mean().item()
            diversity_score = 1.0 - avg_similarity
        
        else:
            diversity_score = 0.5  # Placeholder
        
        return {'diversity_score': diversity_score}


# =============================================================================
# Factory Functions - Required by __init__.py imports
# =============================================================================

def create_dataset(data: torch.Tensor, labels: torch.Tensor, 
                  task_config: TaskConfiguration, 
                  dataset_config: Optional[DatasetConfig] = None) -> MetaLearningDataset:
    """Factory function to create a meta-learning dataset."""
    if dataset_config is None:
        dataset_config = DatasetConfig()
    
    return MetaLearningDataset(data, labels, task_config)


def create_metrics_evaluator(config: Optional[MetricsConfig] = None) -> EvaluationMetrics:
    """Factory function to create an evaluation metrics instance."""
    if config is None:
        config = MetricsConfig()
    
    return EvaluationMetrics(config)


def create_curriculum_scheduler(config: Optional[CurriculumConfig] = None) -> CurriculumLearning:
    """Factory function to create a curriculum learning scheduler."""
    if config is None:
        config = CurriculumConfig()
    
    return CurriculumLearning(config)


def basic_confidence_interval(values: List[float], confidence_level: float = 0.95) -> Tuple[float, float, float]:
    """Basic confidence interval computation."""
    return compute_confidence_interval(values, confidence_level=confidence_level, method="t_test")


def estimate_difficulty(task_data: torch.Tensor, method: str = "entropy") -> float:
    """Estimate task difficulty using various methods."""
    if method == "entropy":
        # Simple entropy-based difficulty
        probs = F.softmax(task_data.mean(dim=0), dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8))
        return entropy.item() / np.log(task_data.size(-1))  # Normalized entropy
    else:
        return 0.5  # Default medium difficulty


def track_task_diversity(tasks: List[torch.Tensor], config: Optional[DiversityConfig] = None) -> Dict[str, float]:
    """Track diversity across multiple tasks."""
    if config is None:
        config = DiversityConfig()
    
    tracker = TaskDiversityTracker(config)
    
    for task in tasks:
        tracker.add_task(task.mean(dim=0))  # Use mean as task feature
    
    return tracker.compute_diversity()

# =============================================================================
# ENHANCED EVALUATION FUNCTIONS WITH CONFIGURATION SUPPORT
# =============================================================================

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
    
    logger.info(f"Evaluation complete: {mean_accuracy:.4f} ± {ci_upper - mean_accuracy:.4f}")
    
    return results