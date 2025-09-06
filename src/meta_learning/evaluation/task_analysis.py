"""
💰 SUPPORT THIS RESEARCH - PLEASE DONATE! 💰

🙏 If this library helps your research or project, please consider donating:
💳 PayPal: https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS
⭐ GitHub Sponsors: https://github.com/sponsors/benedictchen

👨‍💻 Author: Benedict Chen
💰 Donations: Help support this work! Buy me a coffee ☕, beer 🍺, lamborghini 🏎️, or private island 🏝️
💖 Please consider recurring donations to fully support continued research

Your support enables cutting-edge AI research for everyone! 🚀

Task Analysis and Difficulty Assessment
======================================

Advanced task analysis tools for meta-learning, including hardness metrics,
difficulty assessment, and curriculum learning utilities.
"""

# Task analysis and curriculum learning implementation complete
# Includes hardness metrics, difficulty assessment, task quality evaluation,
# similarity analysis, outlier detection, and curriculum generation utilities

from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Any, Union
import torch
import torch.nn.functional as F
import numpy as np
from ..core.episode import Episode
from ..core.math_utils import pairwise_sqeuclidean


def hardness_metric(episode: Episode, num_classes: int) -> float:
    """
    Compute task hardness metric for an episode.
    
    Task hardness is computed based on prototype distances and class separation:
    - Higher intra-class variance = harder task
    - Lower inter-class separation = harder task
    
    Args:
        episode: Episode containing support/query data and labels
        num_classes: Number of classes in the episode
        
    Returns:
        float: Hardness score between 0.0 (easy) and 1.0 (hard)
    """
    support_data = episode.support_x
    support_labels = episode.support_y
    
    if len(support_data) == 0 or num_classes <= 1:
        return 0.0  # Handle edge cases
    
    # Compute class prototypes (centroids)
    prototypes = {}
    class_variances = {}
    
    for class_id in range(num_classes):
        class_mask = (support_labels == class_id)
        if class_mask.sum() == 0:
            continue
            
        class_data = support_data[class_mask]
        prototype = class_data.mean(dim=0)
        prototypes[class_id] = prototype
        
        # Compute intra-class variance
        if len(class_data) > 1:
            class_var = ((class_data - prototype) ** 2).mean()
            class_variances[class_id] = class_var.item()
        else:
            class_variances[class_id] = 0.0
    
    if len(prototypes) < 2:
        return 0.0
    
    # Compute average intra-class variance
    avg_intra_variance = np.mean(list(class_variances.values()))
    
    # Compute inter-class distances (minimum distance between prototypes)
    prototype_tensors = list(prototypes.values())
    min_inter_distance = float('inf')
    
    for i in range(len(prototype_tensors)):
        for j in range(i + 1, len(prototype_tensors)):
            dist = torch.norm(prototype_tensors[i] - prototype_tensors[j]).item()
            min_inter_distance = min(min_inter_distance, dist)
    
    # Hardness = intra-class variance / inter-class separation
    # Higher values indicate harder tasks
    if min_inter_distance > 0:
        hardness = avg_intra_variance / min_inter_distance
    else:
        hardness = 1.0  # Maximum hardness if classes overlap
    
    # Normalize to [0, 1] range using sigmoid-like function
    normalized_hardness = 1.0 / (1.0 + np.exp(-hardness))
    
    return normalized_hardness


class TaskDifficultyAnalyzer:
    """
    Comprehensive task difficulty analysis for meta-learning.
    
    Provides multiple difficulty measures:
    - Hardness metric based on class separation
    - Intra-class variance analysis
    - Inter-class distance analysis
    - Feature complexity assessment
    - Curriculum ordering utilities
    """
    
    def __init__(self, include_confidence: bool = True, confidence_bootstrap_samples: int = 100):
        """Initialize analyzer with configuration options."""
        self.include_confidence = include_confidence
        self.bootstrap_samples = confidence_bootstrap_samples
        self.difficulty_cache = {}
    
    def analyze_episode(self, episode: Episode) -> Dict[str, float]:
        """
        Comprehensive episode analysis with multiple difficulty metrics.
        
        Args:
            episode: Episode to analyze
            
        Returns:
            Dictionary containing various difficulty metrics
        """
        support_data = episode.support_x
        support_labels = episode.support_y
        
        if len(support_data) == 0:
            return {"hardness": 0.0, "intra_variance": 0.0, "inter_distance": 0.0}
        
        num_classes = len(torch.unique(support_labels))
        
        # Basic hardness metric
        hardness = hardness_metric(episode, num_classes)
        
        # Compute detailed metrics
        metrics = {
            "hardness": hardness,
            "num_classes": num_classes,
            "support_size": len(support_data),
            "feature_dim": support_data.shape[-1]
        }
        
        # Intra-class variance analysis
        intra_variances = []
        inter_distances = []
        prototypes = {}
        
        for class_id in range(num_classes):
            class_mask = (support_labels == class_id)
            if class_mask.sum() == 0:
                continue
                
            class_data = support_data[class_mask]
            prototype = class_data.mean(dim=0)
            prototypes[class_id] = prototype
            
            if len(class_data) > 1:
                class_var = ((class_data - prototype) ** 2).mean().item()
                intra_variances.append(class_var)
        
        # Inter-class distance analysis
        prototype_list = list(prototypes.values())
        for i in range(len(prototype_list)):
            for j in range(i + 1, len(prototype_list)):
                dist = torch.norm(prototype_list[i] - prototype_list[j]).item()
                inter_distances.append(dist)
        
        # Aggregate metrics
        metrics.update({
            "intra_variance": np.mean(intra_variances) if intra_variances else 0.0,
            "intra_variance_std": np.std(intra_variances) if len(intra_variances) > 1 else 0.0,
            "inter_distance": np.mean(inter_distances) if inter_distances else 0.0,
            "inter_distance_std": np.std(inter_distances) if len(inter_distances) > 1 else 0.0,
            "min_inter_distance": min(inter_distances) if inter_distances else 0.0,
            "max_inter_distance": max(inter_distances) if inter_distances else 0.0
        })
        
        # Feature complexity (variance across all features)
        feature_variances = support_data.var(dim=0)
        metrics.update({
            "feature_complexity": feature_variances.mean().item(),
            "feature_complexity_std": feature_variances.std().item()
        })
        
        # Confidence intervals if requested
        if self.include_confidence:
            hardness_samples = []
            for _ in range(min(self.bootstrap_samples, 50)):  # Limited for performance
                # Bootstrap sample
                indices = torch.randint(0, len(support_data), (len(support_data),))
                bootstrap_data = support_data[indices]
                bootstrap_labels = support_labels[indices]
                
                bootstrap_episode = Episode(
                    bootstrap_data, bootstrap_labels,
                    episode.query_x, episode.query_y
                )
                
                bootstrap_hardness = hardness_metric(bootstrap_episode, num_classes)
                hardness_samples.append(bootstrap_hardness)
            
            hardness_samples = np.array(hardness_samples)
            metrics.update({
                "hardness_ci_lower": np.percentile(hardness_samples, 2.5),
                "hardness_ci_upper": np.percentile(hardness_samples, 97.5),
                "hardness_std": np.std(hardness_samples)
            })
        
        return metrics
    
    def analyze_batch(self, episodes: List[Episode]) -> Dict[str, Any]:
        """
        Batch analysis with aggregated statistics across multiple episodes.
        
        Args:
            episodes: List of episodes to analyze
            
        Returns:
            Dictionary with aggregated statistics
        """
        if not episodes:
            return {}
        
        episode_metrics = [self.analyze_episode(ep) for ep in episodes]
        
        # Aggregate metrics across episodes
        aggregated = {}
        
        # Get all metric keys
        metric_keys = set()
        for metrics in episode_metrics:
            metric_keys.update(metrics.keys())
        
        # Compute statistics for each metric
        for key in metric_keys:
            values = [m.get(key, 0.0) for m in episode_metrics if key in m]
            if values:
                aggregated[f"{key}_mean"] = np.mean(values)
                aggregated[f"{key}_std"] = np.std(values)
                aggregated[f"{key}_min"] = np.min(values)
                aggregated[f"{key}_max"] = np.max(values)
                aggregated[f"{key}_median"] = np.median(values)
        
        # Additional batch-level metrics
        aggregated.update({
            "num_episodes": len(episodes),
            "avg_support_size": np.mean([len(ep.support_x) for ep in episodes]),
            "avg_query_size": np.mean([len(ep.query_x) for ep in episodes]),
            "difficulty_distribution": {
                "easy": sum(1 for m in episode_metrics if m.get("hardness", 0) < 0.33),
                "medium": sum(1 for m in episode_metrics if 0.33 <= m.get("hardness", 0) < 0.67),
                "hard": sum(1 for m in episode_metrics if m.get("hardness", 0) >= 0.67)
            }
        })
        
        return aggregated
    
    def compute_curriculum_order(self, episodes: List[Episode], strategy: str = "easy_to_hard") -> List[int]:
        """
        Compute curriculum ordering based on episode difficulty.
        
        Args:
            episodes: List of episodes to order
            strategy: Ordering strategy ("easy_to_hard", "hard_to_easy", "diverse")
            
        Returns:
            List of episode indices in curriculum order
        """
        if not episodes:
            return []
        
        # Compute difficulty scores for all episodes
        difficulty_scores = []
        for i, episode in enumerate(episodes):
            metrics = self.analyze_episode(episode)
            difficulty_scores.append((i, metrics.get("hardness", 0.0)))
        
        if strategy == "easy_to_hard":
            # Sort by increasing difficulty
            sorted_episodes = sorted(difficulty_scores, key=lambda x: x[1])
        elif strategy == "hard_to_easy":
            # Sort by decreasing difficulty
            sorted_episodes = sorted(difficulty_scores, key=lambda x: x[1], reverse=True)
        elif strategy == "diverse":
            # Interleave easy and hard episodes for diversity
            sorted_by_difficulty = sorted(difficulty_scores, key=lambda x: x[1])
            n = len(sorted_by_difficulty)
            
            diverse_order = []
            for i in range(n // 2):
                # Add one easy and one hard episode alternately
                diverse_order.append(sorted_by_difficulty[i])
                if n - 1 - i != i:  # Avoid duplicates in odd-length lists
                    diverse_order.append(sorted_by_difficulty[n - 1 - i])
            
            # Add any remaining middle episodes
            if n % 2 == 1:
                diverse_order.append(sorted_by_difficulty[n // 2])
            
            sorted_episodes = diverse_order
        else:
            raise ValueError(f"Unknown curriculum strategy: {strategy}")
        
        return [idx for idx, _ in sorted_episodes]


def curriculum_learning_helper(
    episodes: List[Episode],
    difficulty_scores: List[float],
    strategy: str = "easy_to_hard",
    batch_size: Optional[int] = None
) -> List[List[int]]:
    """
    Generate curriculum learning batches based on task difficulty.
    
    Args:
        episodes: List of episodes to organize
        difficulty_scores: Difficulty scores corresponding to each episode
        strategy: Curriculum strategy ("easy_to_hard", "hard_to_easy", "diverse", "mixed")
        batch_size: Size of each curriculum batch (None = single batch)
        
    Returns:
        List of batches, where each batch is a list of episode indices
    """
    if not episodes or not difficulty_scores:
        return []
    
    if len(episodes) != len(difficulty_scores):
        raise ValueError("Number of episodes must match number of difficulty scores")
    
    # Create list of (index, difficulty) pairs
    indexed_episodes = list(enumerate(difficulty_scores))
    
    # Sort based on strategy
    if strategy == "easy_to_hard":
        sorted_episodes = sorted(indexed_episodes, key=lambda x: x[1])
    elif strategy == "hard_to_easy":
        sorted_episodes = sorted(indexed_episodes, key=lambda x: x[1], reverse=True)
    elif strategy == "diverse":
        # Interleave easy and hard
        sorted_by_difficulty = sorted(indexed_episodes, key=lambda x: x[1])
        n = len(sorted_by_difficulty)
        
        diverse_order = []
        for i in range(n // 2):
            diverse_order.append(sorted_by_difficulty[i])  # Easy
            if n - 1 - i != i:
                diverse_order.append(sorted_by_difficulty[n - 1 - i])  # Hard
        
        if n % 2 == 1:
            diverse_order.append(sorted_by_difficulty[n // 2])  # Middle
            
        sorted_episodes = diverse_order
    elif strategy == "mixed":
        # Random order (no curriculum)
        sorted_episodes = indexed_episodes.copy()
        np.random.shuffle(sorted_episodes)
    else:
        raise ValueError(f"Unknown curriculum strategy: {strategy}")
    
    # Extract indices in curriculum order
    curriculum_order = [idx for idx, _ in sorted_episodes]
    
    # Create batches if batch_size is specified
    if batch_size is None:
        return [curriculum_order]
    
    batches = []
    for i in range(0, len(curriculum_order), batch_size):
        batch = curriculum_order[i:i + batch_size]
        batches.append(batch)
    
    return batches


def task_similarity_matrix(episodes: List[Episode]) -> torch.Tensor:
    """
    Compute task similarity matrix for episode clustering.
    
    Similarity is based on:
    - Prototype distances between episodes
    - Feature distribution similarity
    - Class structure similarity
    
    Args:
        episodes: List of episodes to compare
        
    Returns:
        Symmetric similarity matrix of shape (num_episodes, num_episodes)
    """
    if not episodes:
        return torch.tensor([])
    
    n_episodes = len(episodes)
    similarity_matrix = torch.zeros(n_episodes, n_episodes)
    
    # Compute prototypes for each episode
    episode_prototypes = []
    episode_features = []
    
    for episode in episodes:
        support_data = episode.support_x
        support_labels = episode.support_y
        
        if len(support_data) == 0:
            episode_prototypes.append({})
            episode_features.append(torch.tensor([]))
            continue
        
        # Compute class prototypes
        prototypes = {}
        unique_labels = torch.unique(support_labels)
        
        for label in unique_labels:
            class_mask = (support_labels == label)
            if class_mask.sum() > 0:
                class_data = support_data[class_mask]
                prototype = class_data.mean(dim=0)
                prototypes[label.item()] = prototype
        
        episode_prototypes.append(prototypes)
        
        # Overall episode feature statistics
        feature_mean = support_data.mean(dim=0)
        feature_std = support_data.std(dim=0)
        episode_features.append(torch.cat([feature_mean, feature_std]))
    
    # Compute pairwise similarities
    for i in range(n_episodes):
        similarity_matrix[i, i] = 1.0  # Self-similarity
        
        for j in range(i + 1, n_episodes):
            similarity = 0.0
            
            # Feature distribution similarity
            if len(episode_features[i]) > 0 and len(episode_features[j]) > 0:
                if episode_features[i].shape == episode_features[j].shape:
                    feature_sim = torch.cosine_similarity(
                        episode_features[i].unsqueeze(0),
                        episode_features[j].unsqueeze(0)
                    ).item()
                    similarity += 0.5 * max(0, feature_sim)  # Only positive similarity
            
            # Prototype-based similarity
            prototypes_i = episode_prototypes[i]
            prototypes_j = episode_prototypes[j]
            
            if prototypes_i and prototypes_j:
                # Find common classes and compare prototypes
                common_classes = set(prototypes_i.keys()) & set(prototypes_j.keys())
                
                if common_classes:
                    prototype_similarities = []
                    for class_id in common_classes:
                        proto_sim = torch.cosine_similarity(
                            prototypes_i[class_id].unsqueeze(0),
                            prototypes_j[class_id].unsqueeze(0)
                        ).item()
                        prototype_similarities.append(max(0, proto_sim))
                    
                    prototype_similarity = np.mean(prototype_similarities)
                    similarity += 0.5 * prototype_similarity
                
                # Class structure similarity (number of classes)
                num_classes_i = len(prototypes_i)
                num_classes_j = len(prototypes_j)
                max_classes = max(num_classes_i, num_classes_j)
                
                if max_classes > 0:
                    class_structure_sim = min(num_classes_i, num_classes_j) / max_classes
                    similarity += 0.1 * class_structure_sim
            
            # Episode size similarity
            size_i = len(episodes[i].support_x)
            size_j = len(episodes[j].support_x)
            
            if size_i > 0 and size_j > 0:
                size_similarity = min(size_i, size_j) / max(size_i, size_j)
                similarity += 0.1 * size_similarity
            
            # Normalize similarity to [0, 1]
            similarity = min(1.0, max(0.0, similarity))
            
            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity  # Symmetric
    
    return similarity_matrix


# Additional utility functions for advanced task analysis

def detect_task_outliers(episodes: List[Episode], threshold: float = 2.0) -> List[int]:
    """
    Detect outlier tasks that are unusually difficult or easy.
    
    Args:
        episodes: List of episodes to analyze
        threshold: Z-score threshold for outlier detection
        
    Returns:
        List of indices of outlier episodes
    """
    if len(episodes) < 3:
        return []  # Need at least 3 episodes for meaningful outlier detection
    
    analyzer = TaskDifficultyAnalyzer()
    difficulty_scores = []
    
    for episode in episodes:
        metrics = analyzer.analyze_episode(episode)
        difficulty_scores.append(metrics.get("hardness", 0.0))
    
    # Compute z-scores
    mean_difficulty = np.mean(difficulty_scores)
    std_difficulty = np.std(difficulty_scores)
    
    if std_difficulty == 0:
        return []  # All episodes have same difficulty
    
    z_scores = [(score - mean_difficulty) / std_difficulty for score in difficulty_scores]
    
    # Identify outliers
    outliers = []
    for i, z_score in enumerate(z_scores):
        if abs(z_score) > threshold:
            outliers.append(i)
    
    return outliers


def assess_task_quality(episode: Episode) -> Dict[str, float]:
    """
    Assess the quality of a task episode for meta-learning.
    
    Args:
        episode: Episode to assess
        
    Returns:
        Dictionary with quality metrics
    """
    support_data = episode.support_x
    support_labels = episode.support_y
    query_data = episode.query_x
    query_labels = episode.query_y
    
    quality_metrics = {}
    
    # Basic validity checks
    quality_metrics["has_support_data"] = len(support_data) > 0
    quality_metrics["has_query_data"] = len(query_data) > 0
    quality_metrics["support_size"] = len(support_data)
    quality_metrics["query_size"] = len(query_data)
    
    if len(support_data) == 0:
        quality_metrics["overall_quality"] = 0.0
        return quality_metrics
    
    # Label distribution analysis
    unique_support_labels = torch.unique(support_labels)
    num_classes = len(unique_support_labels)
    
    quality_metrics["num_classes"] = num_classes
    quality_metrics["valid_num_classes"] = num_classes >= 2
    
    # Class balance in support set
    if num_classes > 1:
        label_counts = []
        for label in unique_support_labels:
            count = (support_labels == label).sum().item()
            label_counts.append(count)
        
        min_count = min(label_counts)
        max_count = max(label_counts)
        
        quality_metrics["min_samples_per_class"] = min_count
        quality_metrics["max_samples_per_class"] = max_count
        quality_metrics["class_balance"] = min_count / max_count if max_count > 0 else 0.0
        quality_metrics["well_balanced"] = quality_metrics["class_balance"] >= 0.5
    else:
        quality_metrics["class_balance"] = 0.0
        quality_metrics["well_balanced"] = False
    
    # Query set validation
    if len(query_data) > 0 and len(query_labels) > 0:
        unique_query_labels = torch.unique(query_labels)
        support_label_set = set(unique_support_labels.tolist())
        query_label_set = set(unique_query_labels.tolist())
        
        # Check if query labels are subset of support labels
        quality_metrics["query_labels_in_support"] = query_label_set.issubset(support_label_set)
        quality_metrics["query_coverage"] = len(query_label_set) / len(support_label_set) if len(support_label_set) > 0 else 0.0
    else:
        quality_metrics["query_labels_in_support"] = False
        quality_metrics["query_coverage"] = 0.0
    
    # Feature quality
    if support_data.shape[-1] > 0:
        # Check for NaN or infinite values
        quality_metrics["has_nan"] = torch.isnan(support_data).any().item()
        quality_metrics["has_inf"] = torch.isinf(support_data).any().item()
        quality_metrics["data_valid"] = not (quality_metrics["has_nan"] or quality_metrics["has_inf"])
        
        # Feature variance (zero variance features are problematic)
        feature_vars = support_data.var(dim=0)
        quality_metrics["zero_variance_features"] = (feature_vars == 0).sum().item()
        quality_metrics["avg_feature_variance"] = feature_vars.mean().item()
        quality_metrics["good_feature_variance"] = quality_metrics["avg_feature_variance"] > 1e-6
    else:
        quality_metrics["data_valid"] = False
        quality_metrics["good_feature_variance"] = False
    
    # Overall quality score
    quality_factors = [
        quality_metrics.get("has_support_data", False),
        quality_metrics.get("has_query_data", False),
        quality_metrics.get("valid_num_classes", False),
        quality_metrics.get("well_balanced", False),
        quality_metrics.get("query_labels_in_support", False),
        quality_metrics.get("data_valid", False),
        quality_metrics.get("good_feature_variance", False)
    ]
    
    quality_metrics["overall_quality"] = sum(quality_factors) / len(quality_factors)
    
    return quality_metrics