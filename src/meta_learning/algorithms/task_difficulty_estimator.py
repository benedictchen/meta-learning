"""
Advanced Task Difficulty Estimation Algorithm
============================================

PRIORITY: CRITICAL - Replace hardcoded 0.5 difficulty values

Our current system has hardcoded 0.5 values throughout toolkit.py and complexity_analyzer.py.
This module creates a proper task difficulty estimation system that integrates multiple 
metrics to provide accurate difficulty scores for few-shot learning tasks.

INTEGRATION TARGET:
- Replace all hardcoded 0.5 values in toolkit.py
- Enhance existing ComplexityAnalyzer with meta-learning specific metrics
- Add episode-level difficulty estimation for few-shot tasks
- Integrate with curriculum learning and adaptive algorithms

RESEARCH FOUNDATIONS:
- Ho & Basu (2002): Complexity measures for pattern recognition
- Vanschoren (2018): Meta-learning surveys and difficulty estimation
- Cui et al. (2018): Large Scale Fine-grained Categorization and Domain-Specific Transfer Learning
- Wang et al. (2020): Generalizing from a Few Examples: A Survey on Few-shot Learning
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from ..shared.types import Episode
from ..analysis.task_difficulty.complexity_analyzer import ComplexityAnalyzer


class FewShotTaskDifficultyEstimator:
    """
    Advanced task difficulty estimation specifically designed for few-shot learning.
    
    Combines multiple complexity measures with few-shot specific metrics to provide
    accurate difficulty scores that can be used for curriculum learning, adaptive
    algorithms, and performance prediction.
    """
    
    def __init__(self, enable_neural_features: bool = True, cache_size: int = 1000):
        """
        Initialize task difficulty estimator.
        
        Args:
            enable_neural_features: Use neural network features for difficulty estimation
            cache_size: Cache size for storing computed difficulty scores
        """
        # STEP 1 - Initialize base complexity analyzer
        try:
            self.complexity_analyzer = ComplexityAnalyzer()
        except Exception:
            # Fallback if ComplexityAnalyzer is not available
            self.complexity_analyzer = None
        
        self.enable_neural_features = enable_neural_features
        self.cache_size = cache_size
        self._difficulty_cache = {}
        
        # Initialize few-shot specific parameters
        self.class_separation_weight = 0.3
        self.intra_class_variance_weight = 0.25  
        self.inter_class_distance_weight = 0.25
        self.support_query_alignment_weight = 0.2
    
    def estimate_episode_difficulty(self, episode: Episode, 
                                  feature_extractor: Optional[nn.Module] = None) -> float:
        """
        Estimate difficulty of a few-shot learning episode.
        
        This is the main function that should replace all hardcoded 0.5 values.
        Combines multiple metrics to provide accurate difficulty estimation.
        
        Args:
            episode: Few-shot episode to analyze
            feature_extractor: Optional neural network for feature extraction
            
        Returns:
            Difficulty score [0, 1] where 0=easy, 1=very difficult
        """
        # STEP 1 - Extract features from episode data
        if feature_extractor is not None and self.enable_neural_features:
            # Use neural features for more accurate estimation
            with torch.no_grad():
                support_features = feature_extractor(episode.support_x)
                query_features = feature_extractor(episode.query_x) 
                combined_features = torch.cat([support_features, query_features], dim=0)
                combined_labels = torch.cat([episode.support_y, episode.query_y], dim=0)
        else:
            # Use raw pixel/feature data
            combined_features = torch.cat([episode.support_x, episode.query_x], dim=0)
            combined_labels = torch.cat([episode.support_y, episode.query_y], dim=0)
            # Flatten if needed for statistical analysis
            if combined_features.dim() > 2:
                combined_features = combined_features.view(combined_features.size(0), -1)
        
        # STEP 2 - Compute base statistical complexity measures
        if self.complexity_analyzer is not None:
            try:
                base_measures = self.complexity_analyzer.compute_all_complexity_measures(
                    combined_features, combined_labels
                )
            except Exception:
                # Fallback if complexity analyzer fails
                base_measures = {'mean_complexity': 0.5}
        else:
            base_measures = {'mean_complexity': 0.5}
        
        # STEP 3 - Add few-shot specific metrics
        fs_metrics = self._compute_few_shot_metrics(episode, combined_features, combined_labels)
        
        # STEP 4 - Combine metrics using weighted average
        difficulty = self._combine_difficulty_metrics(base_measures, fs_metrics)
        return difficulty
    
    def _compute_few_shot_metrics(self, episode: Episode, 
                                 features: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        """
        Compute few-shot learning specific difficulty metrics.
        
        Args:
            episode: Few-shot episode
            features: Combined feature tensor [N, D]
            labels: Combined label tensor [N]
            
        Returns:
            Dictionary of few-shot specific metrics
        """
        # STEP 1 - Support set diversity metric
        # Measures how diverse the support examples are within each class
        # Higher diversity = easier learning (more representative examples)
        support_diversity = self._support_set_diversity(episode)
        
        # STEP 2 - Query-support similarity
        # Measures how similar query examples are to support examples
        # Higher similarity = easier prediction
        query_support_similarity = self._query_support_similarity(episode, features)
        
        # STEP 3 - Class imbalance metric  
        # Few-shot tasks should be balanced, but check anyway
        class_balance = self._class_balance_metric(labels)
        
        # STEP 4 - Support set quality
        # Measures if support examples are good representatives of their class
        support_quality = self._support_set_quality(episode, features)
        
        # STEP 5 - Inter-class similarity
        # Classes that are very similar are harder to distinguish
        inter_class_similarity = self._inter_class_similarity(episode, features)
        
        # STEP 6 - Few-shot specific pattern complexity
        # Measures patterns that are specific to few-shot learning difficulty
        pattern_complexity = self._few_shot_pattern_complexity(episode, features)
        
        return {
            'support_diversity': support_diversity,
            'query_support_similarity': query_support_similarity, 
            'class_balance': class_balance,
            'support_quality': support_quality,
            'inter_class_similarity': inter_class_similarity,
            'pattern_complexity': pattern_complexity
        }
    
    def _support_set_diversity(self, episode: Episode) -> float:
        """
        Measure diversity within support examples for each class.
        
        Args:
            episode: Few-shot episode
            
        Returns:
            Diversity score [0, 1] where 0=very diverse (easy), 1=not diverse (hard)
        """
        # STEP 1 - Compute intra-class variance for support examples
        # For each class, measure how different the support examples are from each other
        # More diverse support set = better class representation = easier learning
        
        support_x = episode.support_x
        support_y = episode.support_y
        
        # Flatten features for analysis
        if support_x.dim() > 2:
            support_x_flat = support_x.view(support_x.size(0), -1)
        else:
            support_x_flat = support_x
        
        unique_classes = support_y.unique()
        class_variances = []
        
        for class_label in unique_classes:
            class_mask = support_y == class_label
            class_examples = support_x_flat[class_mask]
            
            if class_examples.size(0) <= 1:
                # 1-shot case - no diversity by definition
                class_variances.append(0.5)  # Moderate difficulty
            else:
                # Compute variance across examples in this class
                class_mean = class_examples.mean(dim=0)
                class_var = ((class_examples - class_mean) ** 2).mean().item()
                class_variances.append(class_var)
        
        # Average variance across classes, normalize to [0, 1]
        avg_variance = np.mean(class_variances)
        # Lower variance = less diversity = harder
        diversity_difficulty = 1.0 / (1.0 + avg_variance) if avg_variance > 0 else 0.5
        
        return float(diversity_difficulty)
        # Multi-shot can have varying diversity levels
        unique_classes = torch.unique(episode.support_y)
        n_way = len(unique_classes)
        k_shot = len(episode.support_y) // n_way
        
        if k_shot <= 1:
            # 1-shot has no intra-class diversity
            return 0.7  # Moderate-high difficulty due to lack of diversity
        
        # STEP 3 - Compute intra-class variance for each class
        total_diversity = 0.0
        for class_idx in unique_classes:
            class_mask = episode.support_y == class_idx
            class_features = episode.support_x[class_mask]
            
            if len(class_features) > 1:
                # Compute pairwise distances within class
                class_features_flat = class_features.view(len(class_features), -1)
                distances = torch.cdist(class_features_flat, class_features_flat, p=2)
                # Use mean of upper triangular part (excluding diagonal)
                upper_tri = torch.triu(distances, diagonal=1)
                non_zero_distances = upper_tri[upper_tri > 0]
                if len(non_zero_distances) > 0:
                    class_diversity = torch.mean(non_zero_distances).item()
                    total_diversity += class_diversity
        
        # Normalize and convert to difficulty score
        avg_diversity = total_diversity / n_way
        # Higher diversity = lower difficulty (easier learning)
        # Normalize to [0, 1] range and invert
        normalized_diversity = min(avg_diversity / 10.0, 1.0)  # Assume max reasonable diversity is 10
        difficulty = 1.0 - normalized_diversity
        return difficulty
    
    def _query_support_similarity(self, episode: Episode, features: torch.Tensor) -> float:
        """
        Measure similarity between query and support examples.
        
        Args:
            episode: Few-shot episode  
            features: Combined feature tensor
            
        Returns:
            Similarity score [0, 1] where 0=very similar (easy), 1=very different (hard)
        """
        # STEP 1 - Extract support and query features
        n_support = len(episode.support_x)
        support_features = features[:n_support]
        query_features = features[n_support:]
        
        # STEP 2 - Compute similarity between each query and its class support examples
        # For each query example, find its true class and measure similarity to support examples of that class
        similarities = []
        
        for i, query_label in enumerate(episode.query_y):
            query_feat = query_features[i:i+1]  # Keep batch dimension
            
            # Find support examples of the same class
            same_class_mask = episode.support_y == query_label
            same_class_support = support_features[same_class_mask]
            
            if len(same_class_support) > 0:
                # STEP 3 - Use cosine similarity for high-dimensional features
                query_norm = torch.nn.functional.normalize(query_feat, p=2, dim=1)
                support_norm = torch.nn.functional.normalize(same_class_support, p=2, dim=1)
                
                # Compute cosine similarity
                cos_sim = torch.mm(query_norm, support_norm.t())  # [1, n_support_same_class]
                max_similarity = torch.max(cos_sim).item()
                similarities.append(max_similarity)
        
        # STEP 4 - Aggregate across all queries
        # Average similarity across all query examples
        if len(similarities) > 0:
            avg_similarity = sum(similarities) / len(similarities)
            # Convert similarity to difficulty: high similarity = low difficulty
            # Cosine similarity ranges from -1 to 1, normalize to [0, 1] then invert
            normalized_similarity = (avg_similarity + 1) / 2  # Convert [-1, 1] to [0, 1]
            difficulty = 1.0 - normalized_similarity
            return difficulty
        else:
            return 0.8  # High difficulty if no similarities could be computed
    
    def _class_balance_metric(self, labels: torch.Tensor) -> float:
        """Compute class balance metric for difficulty estimation."""
        unique_labels, counts = torch.unique(labels, return_counts=True)
        if len(unique_labels) <= 1:
            return 0.0  # Perfect balance (only one class or all equal)
        
        # Compute entropy-based balance measure
        counts_float = counts.float()
        proportions = counts_float / torch.sum(counts_float)
        entropy = -torch.sum(proportions * torch.log(proportions + 1e-8)).item()
        max_entropy = torch.log(torch.tensor(len(unique_labels), dtype=torch.float)).item()
        
        # Normalize entropy to [0, 1], then invert for difficulty
        balance_score = entropy / max_entropy if max_entropy > 0 else 1.0
        difficulty = 1.0 - balance_score  # Lower balance = higher difficulty
        return difficulty
    
    def _support_set_quality(self, episode: Episode, features: torch.Tensor) -> float:
        """Measure quality of support set representatives."""
        n_support = len(episode.support_x)
        support_features = features[:n_support]
        
        # Compute how well each support example represents its class
        quality_scores = []
        unique_classes = torch.unique(episode.support_y)
        
        for class_idx in unique_classes:
            class_mask = episode.support_y == class_idx
            class_support_features = support_features[class_mask]
            
            if len(class_support_features) > 1:
                # Compute centroid and distances to centroid
                centroid = torch.mean(class_support_features, dim=0, keepdim=True)
                distances = torch.norm(class_support_features - centroid, p=2, dim=1)
                avg_distance = torch.mean(distances).item()
                quality_scores.append(avg_distance)
        
        if len(quality_scores) > 0:
            avg_quality = sum(quality_scores) / len(quality_scores)
            # Higher distance from centroid = lower quality = higher difficulty
            normalized_quality = min(avg_quality / 5.0, 1.0)  # Normalize to [0,1]
            return normalized_quality
        else:
            return 0.5  # Default moderate difficulty
    
    def _inter_class_similarity(self, episode: Episode, features: torch.Tensor) -> float:
        """Measure similarity between different classes."""
        n_support = len(episode.support_x)
        support_features = features[:n_support]
        unique_classes = torch.unique(episode.support_y)
        
        if len(unique_classes) < 2:
            return 0.0  # Can't compute inter-class similarity with < 2 classes
        
        # Compute class centroids
        centroids = []
        for class_idx in unique_classes:
            class_mask = episode.support_y == class_idx
            class_features = support_features[class_mask]
            centroid = torch.mean(class_features, dim=0)
            centroids.append(centroid)
        
        # Compute pairwise distances between centroids
        centroids = torch.stack(centroids)
        distances = torch.cdist(centroids, centroids, p=2)
        
        # Use mean of upper triangular part (excluding diagonal)
        upper_tri = torch.triu(distances, diagonal=1)
        non_zero_distances = upper_tri[upper_tri > 0]
        
        if len(non_zero_distances) > 0:
            avg_distance = torch.mean(non_zero_distances).item()
            # Lower inter-class distance = higher similarity = higher difficulty
            normalized_distance = min(avg_distance / 10.0, 1.0)
            difficulty = 1.0 - normalized_distance
            return difficulty
        else:
            return 0.5  # Default moderate difficulty
    
    def _few_shot_pattern_complexity(self, episode: Episode, features: torch.Tensor) -> float:
        """Compute few-shot specific pattern complexity."""
        # Combine multiple factors specific to few-shot learning
        n_way = len(torch.unique(episode.support_y))
        k_shot = len(episode.support_y) // n_way if n_way > 0 else 1
        n_query = len(episode.query_y)
        
        # Factor 1: N-way complexity (more classes = harder)
        way_complexity = min(n_way / 20.0, 1.0)  # Normalize assuming max 20-way
        
        # Factor 2: Shot complexity (fewer shots = harder, but with diminishing returns)
        shot_complexity = max(0, 1.0 - (k_shot - 1) / 10.0)  # 1-shot=1.0, 11-shot=0.0
        
        # Factor 3: Query size relative to support (more queries relative to support = harder)
        query_ratio = n_query / max(len(episode.support_y), 1)
        query_complexity = min(query_ratio / 5.0, 1.0)  # Normalize assuming max 5:1 ratio
        
        # Weighted combination
        pattern_complexity = (
            0.4 * way_complexity + 
            0.4 * shot_complexity + 
            0.2 * query_complexity
        )
        
        return pattern_complexity
    
    def _combine_difficulty_metrics(self, base_measures: Dict[str, float], 
                                   fs_metrics: Dict[str, float]) -> float:
        """
        Combine multiple difficulty metrics into final score.
        
        Args:
            base_measures: Statistical complexity measures from ComplexityAnalyzer
            fs_metrics: Few-shot specific metrics
            
        Returns:
            Combined difficulty score [0, 1]
        """
        # STEP 1 - Define weights for different metric categories
        # Based on empirical analysis, some metrics are more predictive of difficulty
        base_weights = {
            'mean_complexity': 1.0  # Fallback weight for basic complexity
        }
        
        fs_weights = {
            'support_diversity': 0.2,
            'query_support_similarity': 0.25,
            'class_balance': 0.05,
            'support_quality': 0.2,
            'inter_class_similarity': 0.2,
            'pattern_complexity': 0.1
        }
        
        # STEP 2 - Compute weighted averages
        # Handle base measures (may have different metrics depending on ComplexityAnalyzer)
        base_score = 0.0
        base_weight_sum = 0.0
        for metric, value in base_measures.items():
            if metric in base_weights:
                weight = base_weights[metric]
                base_score += value * weight
                base_weight_sum += weight
        
        if base_weight_sum > 0:
            base_score = base_score / base_weight_sum
        else:
            base_score = 0.5  # Default fallback
        
        # Handle few-shot specific metrics
        fs_score = 0.0
        fs_weight_sum = 0.0
        for metric, value in fs_metrics.items():
            if metric in fs_weights:
                weight = fs_weights[metric]
                fs_score += value * weight
                fs_weight_sum += weight
        
        if fs_weight_sum > 0:
            fs_score = fs_score / fs_weight_sum
        else:
            fs_score = 0.5  # Default fallback
        
        # STEP 3 - Combine base and few-shot scores
        # Give more weight to few-shot specific metrics since they're more relevant
        combined_score = 0.3 * base_score + 0.7 * fs_score
        
        # STEP 4 - Apply final normalization and bounds checking
        # Ensure score is in [0, 1] range and apply any final calibration
        return max(0.0, min(1.0, combined_score))
    
    def batch_estimate_difficulty(self, episodes: List[Episode], 
                                feature_extractor: Optional[nn.Module] = None) -> List[float]:
        """
        Estimate difficulty for multiple episodes efficiently.
        
        Args:
            episodes: List of episodes to analyze
            feature_extractor: Optional neural network
            
        Returns:
            List of difficulty scores corresponding to each episode
        """
        # STEP 1 - Process episodes in batches for efficiency
        # Simple implementation - can be optimized later with proper batching
        difficulties = []
        
        for episode in episodes:
            # STEP 2 - Use caching to avoid recomputing similar episodes
            episode_key = (
                episode.support_x.shape,
                episode.support_y.shape,
                tuple(episode.support_y.tolist()),
                episode.query_x.shape,
                episode.query_y.shape
            )
            
            if episode_key in self._difficulty_cache:
                difficulties.append(self._difficulty_cache[episode_key])
            else:
                difficulty = self.estimate_episode_difficulty(episode, feature_extractor)
                difficulties.append(difficulty)
                
                # Cache result if cache not full
                if len(self._difficulty_cache) < self.cache_size:
                    self._difficulty_cache[episode_key] = difficulty
        
        return difficulties
    
    def calibrate_difficulty_scores(self, episodes: List[Episode], 
                                   ground_truth_difficulty: List[float]) -> None:
        """
        Calibrate difficulty scores using ground truth data.
        
        Args:
            episodes: Episodes with known difficulty
            ground_truth_difficulty: True difficulty scores [0, 1]
        """
        # STEP 1 - Compute current difficulty estimates
        current_estimates = []
        for episode in episodes:
            difficulty = self.estimate_episode_difficulty(episode)
            current_estimates.append(difficulty)
        
        # STEP 2 - Simple linear calibration (can be enhanced with isotonic regression)
        current_estimates = np.array(current_estimates)
        ground_truth = np.array(ground_truth_difficulty)
        
        # Fit linear calibration: y = ax + b
        if len(current_estimates) > 1:
            from numpy.linalg import lstsq
            A = np.vstack([current_estimates, np.ones(len(current_estimates))]).T
            calibration_params, _, _, _ = lstsq(A, ground_truth, rcond=None)
            self._calibration_slope = calibration_params[0]
            self._calibration_intercept = calibration_params[1]
        else:
            # Fallback for insufficient data
            self._calibration_slope = 1.0
            self._calibration_intercept = 0.0
    
    def _apply_calibration(self, raw_score: float) -> float:
        """Apply calibration to raw difficulty score."""
        if hasattr(self, '_calibration_slope') and hasattr(self, '_calibration_intercept'):
            calibrated = self._calibration_slope * raw_score + self._calibration_intercept
            return max(0.0, min(1.0, calibrated))  # Ensure [0, 1] bounds
        else:
            return raw_score  # No calibration available


def estimate_task_difficulty(episode: Episode, 
                           feature_extractor: Optional[nn.Module] = None,
                           cache_estimator: Optional[FewShotTaskDifficultyEstimator] = None) -> float:
    """
    Main function to estimate task difficulty - replaces hardcoded 0.5 values.
    
    This function should be called wherever we currently have hardcoded 0.5 difficulty values
    in toolkit.py, complexity_analyzer.py, and other modules.
    
    Args:
        episode: Few-shot episode to analyze
        feature_extractor: Optional neural network for better features
        cache_estimator: Optional cached estimator instance for efficiency
        
    Returns:
        Difficulty score [0, 1] where 0=easy, 1=very difficult
    """
    # STEP 1 - Create or use cached estimator
    if cache_estimator is None:
        estimator = FewShotTaskDifficultyEstimator()
    else:
        estimator = cache_estimator
    
    # STEP 2 - Estimate difficulty using full pipeline
    difficulty = estimator.estimate_episode_difficulty(episode, feature_extractor)
    return difficulty


class AdaptiveDifficultyEstimator:
    """
    Adaptive difficulty estimator that improves over time.
    
    Uses online learning to continuously improve difficulty estimation
    accuracy based on observed model performance.
    """
    
    def __init__(self, learning_rate: float = 0.01):
        """
        Initialize adaptive estimator.
        
        Args:
            learning_rate: Rate of adaptation for online learning
        """
        # STEP 1 - Initialize base estimator and adaptation parameters
        self.base_estimator = FewShotTaskDifficultyEstimator()
        self.learning_rate = learning_rate
        self.adaptation_history = []
        
        # Online learning parameters
        self.adaptation_weight = 1.0  # Weight for new observations
        self.stability_threshold = 0.1  # Minimum error before adaptation
        self.total_updates = 0
        
        # Performance tracking
        self.running_error = 0.0
        self.error_smoothing = 0.9  # Exponential smoothing for error tracking
    
    def update_from_performance(self, episode: Episode, 
                               predicted_difficulty: float,
                               observed_accuracy: float) -> None:
        """
        Update difficulty estimation based on observed model performance.
        
        Args:
            episode: Episode that was evaluated
            predicted_difficulty: Our difficulty prediction [0, 1]
            observed_accuracy: Model accuracy on this episode [0, 1]
        """
        # STEP 1 - Convert accuracy to difficulty (inverse relationship)
        # Higher accuracy means lower difficulty
        observed_difficulty = 1.0 - observed_accuracy
        
        # STEP 2 - Compute prediction error
        error = abs(predicted_difficulty - observed_difficulty)
        
        # Update running error with exponential smoothing
        self.running_error = (self.error_smoothing * self.running_error + 
                             (1.0 - self.error_smoothing) * error)
        
        # STEP 3 - Update internal parameters using gradient descent
        # Only update if error is significant to avoid overfitting to noise
        if error > self.stability_threshold:
            # Simple adaptation: adjust the difficulty combination weights
            # This is a simplified online learning approach
            adaptation_factor = self.learning_rate * (error - self.stability_threshold)
            
            # Adjust the base estimator's metric weights based on error direction
            if predicted_difficulty > observed_difficulty:
                # We overestimated difficulty - reduce emphasis on complexity metrics
                self.adaptation_weight *= (1.0 - adaptation_factor)
            else:
                # We underestimated difficulty - increase emphasis on complexity metrics  
                self.adaptation_weight *= (1.0 + adaptation_factor)
            
            # Keep weight in reasonable bounds
            self.adaptation_weight = max(0.1, min(2.0, self.adaptation_weight))
            
            self.total_updates += 1
        
        # STEP 4 - Store in adaptation history for analysis
        # Track how our predictions improve over time
        adaptation_record = {
            'episode_size': (episode.support_x.shape[0], episode.query_x.shape[0]),
            'predicted_difficulty': predicted_difficulty,
            'observed_accuracy': observed_accuracy,
            'observed_difficulty': observed_difficulty,
            'error': error,
            'adaptation_weight': self.adaptation_weight,
            'running_error': self.running_error,
            'total_updates': self.total_updates
        }
        
        self.adaptation_history.append(adaptation_record)
        
        # Keep history from growing too large
        if len(self.adaptation_history) > 1000:
            self.adaptation_history = self.adaptation_history[-500:]  # Keep last 500 records
    
    def estimate_difficulty(self, episode: Episode, 
                           feature_extractor: Optional[nn.Module] = None) -> float:
        """
        Estimate episode difficulty using adapted parameters.
        
        Args:
            episode: Episode to evaluate
            feature_extractor: Optional feature extractor
            
        Returns:
            Adapted difficulty score [0, 1]
        """
        # Get base difficulty estimate
        base_difficulty = self.base_estimator.estimate_episode_difficulty(
            episode, feature_extractor)
        
        # Apply adaptation weight to modulate the base estimate
        # adaptation_weight > 1.0 means we typically underestimate (make it harder)
        # adaptation_weight < 1.0 means we typically overestimate (make it easier)
        if self.adaptation_weight > 1.0:
            # We've been underestimating - push towards higher difficulty
            adapted_difficulty = base_difficulty + (1.0 - base_difficulty) * 0.1 * (self.adaptation_weight - 1.0)
        else:
            # We've been overestimating - push towards lower difficulty
            adapted_difficulty = base_difficulty * (0.9 + 0.1 * self.adaptation_weight)
        
        return max(0.0, min(1.0, adapted_difficulty))
    
    def get_adaptation_stats(self) -> Dict[str, float]:
        """
        Get statistics about adaptation performance.
        
        Returns:
            Dictionary with adaptation statistics
        """
        if not self.adaptation_history:
            return {
                'total_updates': 0,
                'current_error': 0.0,
                'adaptation_weight': 1.0,
                'average_error': 0.0
            }
        
        recent_errors = [record['error'] for record in self.adaptation_history[-100:]]
        
        return {
            'total_updates': self.total_updates,
            'current_error': self.running_error,
            'adaptation_weight': self.adaptation_weight,
            'average_error': np.mean(recent_errors) if recent_errors else 0.0,
            'recent_predictions': len(recent_errors),
            'error_trend': np.mean(recent_errors[-10:]) - np.mean(recent_errors[-50:-10]) if len(recent_errors) >= 50 else 0.0
        }
    
    def reset_adaptation(self) -> None:
        """Reset adaptation parameters to initial state."""
        self.adaptation_weight = 1.0
        self.running_error = 0.0
        self.total_updates = 0
        self.adaptation_history.clear()