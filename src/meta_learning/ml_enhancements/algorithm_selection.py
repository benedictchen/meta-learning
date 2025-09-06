"""
💰 SUPPORT THIS RESEARCH - PLEASE DONATE! 💰

🙏 If this algorithm selection helps your research, please donate:
💳 PayPal: https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS
⭐ GitHub Sponsors: https://github.com/sponsors/benedictchen

Algorithm Selection Module for Meta-Learning - Enhanced with Registry Integration
===============================================================================

This module provides automatic algorithm selection capabilities based on
task characteristics and historical performance data, now enhanced with
the algorithm registry for modular algorithm management.

Classes:
    AlgorithmSelector: Selects the best algorithm for a given episode
                      based on task features and performance history.

Enhanced Features:
- Integration with AlgorithmRegistry for modular algorithm support
- Ridge regression and matching networks support
- Advanced heuristic strategies with fallback mechanisms
- Performance-based selection with learning capabilities

💰 Please donate if this accelerates your research!
"""

from typing import Dict, List, Any, Optional
import time
import torch
import numpy as np

from ..shared.types import Episode
from .algorithm_registry import algorithm_registry, TaskDifficulty


class AlgorithmSelector:
    """
    Enhanced automatic algorithm selection based on task characteristics.
    
    Analyzes episode features (support set size, number of classes, etc.)
    and historical performance to select the most appropriate algorithm
    for each meta-learning task, now with algorithm registry integration.
    
    Enhanced Features:
    - Integration with AlgorithmRegistry for modular algorithm support
    - Support for ridge regression, matching networks, and all registered algorithms
    - Advanced performance tracking with statistical analysis
    - Learning-based selection that improves over time
    
    Attributes:
        algorithm_performance (Dict): Historical performance data for each algorithm
        registry: Reference to the algorithm registry
        selection_strategy: Current selection strategy
    """
    
    def __init__(self, selection_strategy: str = "heuristic_enhanced"):
        """
        Initialize the enhanced algorithm selector.
        
        Args:
            selection_strategy: Strategy for algorithm selection
        """
        # Get all registered algorithms for performance tracking
        registered_algorithms = algorithm_registry.get_all_algorithms()
        
        self.algorithm_performance = {
            name: [] for name in registered_algorithms.keys()
        }
        
        # Legacy algorithm support (for backwards compatibility)
        legacy_algorithms = ['maml', 'test_time_compute', 'protonet']
        for alg in legacy_algorithms:
            if alg not in self.algorithm_performance:
                self.algorithm_performance[alg] = []
        
        self.registry = algorithm_registry
        self.selection_strategy = selection_strategy
        self.performance_weights = {}  # Learned weights for performance-based selection
        
    def select_algorithm(self, episode: Episode) -> str:
        """
        Select best algorithm based on task characteristics using enhanced selection.
        
        Now uses the algorithm registry for intelligent selection with support for:
        - Ridge regression for stable, closed-form solutions
        - Matching networks for attention-based matching
        - Enhanced TTCS for difficult few-shot scenarios
        - All other registered algorithms
        
        Args:
            episode: The meta-learning episode to analyze
            
        Returns:
            Name of the selected algorithm from the registry
        """
        try:
            # Use registry-based selection
            selected = self.registry.select_algorithm(episode, self.selection_strategy)
            return selected
            
        except Exception:
            # Fallback to enhanced heuristics
            return self._fallback_selection(episode)
    
    def _fallback_selection(self, episode: Episode) -> str:
        """Enhanced fallback selection with ridge regression priority."""
        n_support = len(episode.support_y)
        n_classes = len(torch.unique(episode.support_y))
        n_shots_per_class = n_support // n_classes if n_classes > 0 else 1
        
        # Enhanced selection logic with ridge regression
        if n_shots_per_class >= 5 and n_classes <= 20:
            # Ridge regression excels with sufficient data and reasonable class count
            return 'ridge_regression'
        elif n_support < 5:  # Very few-shot
            return 'ttcs' if 'ttcs' in self.algorithm_performance else 'test_time_compute'
        elif n_classes > 10:  # Many classes
            # Try matching networks first, fallback to protonet
            if 'matching_networks' in self.algorithm_performance:
                return 'matching_networks'
            else:
                return 'protonet'
        elif n_shots_per_class <= 3:  # Few-shot scenarios
            # Ridge regression is very stable for few-shot
            return 'ridge_regression'
        else:
            return 'maml'  # General purpose
    
    def update_performance(self, algorithm: str, episode: Episode, accuracy: float):
        """Update performance history for algorithm selection.
        
        Records algorithm performance on specific episodes for future
        selection decisions. Maintains sliding window of recent performance.
        
        Args:
            algorithm: Name of the algorithm that was used
            episode: The episode it was evaluated on
            accuracy: Achieved accuracy (0.0 to 1.0)
        """
        self.algorithm_performance[algorithm].append({
            'accuracy': accuracy,
            'n_support': len(episode.support_y),
            'n_classes': len(torch.unique(episode.support_y)),
            'timestamp': time.time()
        })
        
        # Keep recent history
        if len(self.algorithm_performance[algorithm]) > 100:
            self.algorithm_performance[algorithm] = self.algorithm_performance[algorithm][-50:]
    
    def get_algorithm_stats(self, algorithm: str) -> Dict[str, float]:
        """
        Get enhanced performance statistics for an algorithm.
        
        Args:
            algorithm: Algorithm name to get stats for
            
        Returns:
            Dictionary with comprehensive statistics including confidence intervals
        """
        if algorithm not in self.algorithm_performance:
            return {'mean_accuracy': 0.0, 'count': 0}
            
        performances = self.algorithm_performance[algorithm]
        if not performances:
            return {'mean_accuracy': 0.0, 'count': 0}
            
        accuracies = [p['accuracy'] for p in performances]
        
        stats = {
            'mean_accuracy': sum(accuracies) / len(accuracies),
            'count': len(performances),
            'min_accuracy': min(accuracies),
            'max_accuracy': max(accuracies)
        }
        
        if len(accuracies) > 1:
            std_accuracy = np.std(accuracies)
            stats.update({
                'std_accuracy': std_accuracy,
                'confidence_interval_95': 1.96 * std_accuracy / np.sqrt(len(accuracies)),
                'median_accuracy': np.median(accuracies)
            })
        
        # Recent performance trend (last 10 episodes)
        if len(accuracies) >= 10:
            recent_accuracies = accuracies[-10:]
            older_accuracies = accuracies[:-10]
            stats['recent_trend'] = np.mean(recent_accuracies) - np.mean(older_accuracies)
        
        return stats
    
    def get_best_algorithms(self, episode: Episode, top_k: int = 3) -> List[str]:
        """
        Get top-k best algorithms for a given episode based on registry and performance.
        
        Args:
            episode: Episode to analyze
            top_k: Number of algorithms to return
            
        Returns:
            List of algorithm names sorted by suitability
        """
        n_support = len(episode.support_y)
        n_classes = len(torch.unique(episode.support_y))
        n_shots_per_class = n_support // n_classes if n_classes > 0 else 1
        
        # Estimate task difficulty
        if n_shots_per_class >= 10:
            difficulty = TaskDifficulty.EASY
        elif n_shots_per_class >= 5:
            difficulty = TaskDifficulty.MEDIUM
        elif n_shots_per_class >= 3:
            difficulty = TaskDifficulty.HARD
        else:
            difficulty = TaskDifficulty.VERY_HARD
        
        # Get suitable algorithms from registry
        try:
            suitable_metadata = self.registry.get_suitable_algorithms(
                n_shot=n_shots_per_class,
                n_classes=n_classes,
                task_difficulty=difficulty,
                max_algorithms=top_k * 2  # Get more candidates
            )
            
            # Combine registry priority with performance history
            algorithm_scores = []
            
            for metadata in suitable_metadata:
                registry_score = metadata.selection_priority
                
                # Add performance-based score
                perf_stats = self.get_algorithm_stats(metadata.name)
                performance_score = perf_stats.get('mean_accuracy', 0.5)  # Default to 0.5
                confidence = min(1.0, perf_stats.get('count', 0) / 50)  # Confidence based on sample size
                
                # Combined score: registry priority * (1-confidence) + performance * confidence
                combined_score = registry_score * (1 - confidence) + performance_score * confidence
                
                algorithm_scores.append((metadata.name, combined_score))
            
            # Sort by score and return top-k
            algorithm_scores.sort(key=lambda x: x[1], reverse=True)
            return [name for name, score in algorithm_scores[:top_k]]
            
        except Exception:
            # Fallback to heuristic selection
            fallback = self._fallback_selection(episode)
            return [fallback]
    
    def update_selection_strategy(self, strategy: str):
        """Update the selection strategy."""
        available_strategies = ["heuristic_enhanced", "performance_based"]
        if strategy in available_strategies:
            self.selection_strategy = strategy
        else:
            raise ValueError(f"Strategy must be one of {available_strategies}")
    
    def get_algorithm_recommendations(self, episode: Episode) -> Dict[str, Any]:
        """
        Get comprehensive algorithm recommendations with explanations.
        
        Args:
            episode: Episode to analyze
            
        Returns:
            Dictionary with recommendations and reasoning
        """
        n_support = len(episode.support_y)
        n_classes = len(torch.unique(episode.support_y))
        n_shots_per_class = n_support // n_classes if n_classes > 0 else 1
        
        # Get top algorithms
        top_algorithms = self.get_best_algorithms(episode, top_k=3)
        
        # Get detailed stats for each
        algorithm_details = {}
        for alg in top_algorithms:
            stats = self.get_algorithm_stats(alg)
            metadata = self.registry.get_algorithm_metadata(alg)
            
            algorithm_details[alg] = {
                'performance_stats': stats,
                'metadata': {
                    'type': metadata.algorithm_type.value if metadata else 'unknown',
                    'description': metadata.description if metadata else 'No description',
                    'computational_complexity': metadata.computational_complexity if metadata else 'Unknown'
                },
                'suitability_score': self._calculate_suitability_score(alg, episode)
            }
        
        return {
            'primary_recommendation': top_algorithms[0] if top_algorithms else 'ridge_regression',
            'alternatives': top_algorithms[1:],
            'task_analysis': {
                'n_support': n_support,
                'n_classes': n_classes,
                'n_shots_per_class': n_shots_per_class,
                'estimated_difficulty': self._estimate_difficulty(episode).value
            },
            'algorithm_details': algorithm_details,
            'selection_reasoning': self._generate_selection_reasoning(episode, top_algorithms)
        }
    
    def _calculate_suitability_score(self, algorithm: str, episode: Episode) -> float:
        """Calculate a suitability score for an algorithm on an episode."""
        metadata = self.registry.get_algorithm_metadata(algorithm)
        if not metadata:
            return 0.5
        
        n_support = len(episode.support_y)
        n_classes = len(torch.unique(episode.support_y))
        n_shots_per_class = n_support // n_classes if n_classes > 0 else 1
        
        score = metadata.selection_priority
        
        # Adjust based on task characteristics
        if n_shots_per_class < metadata.min_shots_recommended:
            score *= 0.7
        elif n_shots_per_class > metadata.max_shots_recommended:
            score *= 0.8
        
        if n_classes > 10 and not metadata.good_for_many_classes:
            score *= 0.6
        
        if n_shots_per_class <= 3 and not metadata.good_for_few_shot:
            score *= 0.5
        
        return score
    
    def _estimate_difficulty(self, episode: Episode) -> TaskDifficulty:
        """Estimate the difficulty of an episode."""
        n_support = len(episode.support_y)
        n_classes = len(torch.unique(episode.support_y))
        n_shots_per_class = n_support // n_classes if n_classes > 0 else 1
        
        if n_shots_per_class >= 10:
            return TaskDifficulty.EASY
        elif n_shots_per_class >= 5:
            return TaskDifficulty.MEDIUM
        elif n_shots_per_class >= 3:
            return TaskDifficulty.HARD
        else:
            return TaskDifficulty.VERY_HARD
    
    def _generate_selection_reasoning(self, episode: Episode, top_algorithms: List[str]) -> str:
        """Generate human-readable reasoning for algorithm selection."""
        n_support = len(episode.support_y)
        n_classes = len(torch.unique(episode.support_y))
        n_shots_per_class = n_support // n_classes if n_classes > 0 else 1
        
        reasoning = f"For a {n_classes}-way {n_shots_per_class}-shot task: "
        
        if not top_algorithms:
            return reasoning + "No suitable algorithms found."
        
        primary = top_algorithms[0]
        metadata = self.registry.get_algorithm_metadata(primary)
        
        if primary == 'ridge_regression':
            reasoning += "Ridge regression recommended for its stability and closed-form solution."
        elif primary == 'maml':
            reasoning += "MAML recommended for its general-purpose gradient-based adaptation."
        elif primary == 'protonet':
            reasoning += "Prototypical networks recommended for metric-based similarity matching."
        elif primary == 'ttcs':
            reasoning += "Test-time compute scaling recommended for challenging few-shot scenarios."
        elif primary == 'matching_networks':
            reasoning += "Matching networks recommended for attention-based few-shot learning."
        else:
            reasoning += f"{primary} recommended based on task characteristics and performance history."
        
        if len(top_algorithms) > 1:
            reasoning += f" Alternatives include: {', '.join(top_algorithms[1:])}."
        
        return reasoning