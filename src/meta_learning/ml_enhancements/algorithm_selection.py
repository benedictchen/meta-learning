"""
Algorithm Selection Module for Meta-Learning.

This module provides automatic algorithm selection capabilities based on
task characteristics and historical performance data.

Classes:
    AlgorithmSelector: Selects the best algorithm for a given episode
                      based on task features and performance history.

The selector uses heuristic rules combined with performance tracking
to choose between available meta-learning algorithms.
"""

from typing import Dict, List, Any
import time
import torch

from ..core.episode import Episode


class AlgorithmSelector:
    """Automatic algorithm selection based on task characteristics.
    
    Analyzes episode features (support set size, number of classes, etc.)
    and historical performance to select the most appropriate algorithm
    for each meta-learning task.
    
    Attributes:
        algorithm_performance (Dict): Historical performance data for each algorithm
    """
    
    def __init__(self):
        """Initialize the algorithm selector."""
        self.algorithm_performance = {
            'maml': [],
            'test_time_compute': [],
            'protonet': []
        }
        
    def select_algorithm(self, episode: Episode) -> str:
        """Select best algorithm based on task characteristics.
        
        Uses heuristic rules based on episode characteristics:
        - Very few-shot (< 5 support): test_time_compute
        - Many classes (> 10): protonet  
        - General cases: maml
        
        Args:
            episode: The meta-learning episode to analyze
            
        Returns:
            Name of the selected algorithm ('maml', 'test_time_compute', or 'protonet')
        """
        # Extract task features
        n_support = len(episode.support_y)
        n_classes = len(torch.unique(episode.support_y))
        n_query = len(episode.query_y)
        
        # Simple heuristic-based selection (would be ML-based in practice)
        if n_support < 5:  # Very few-shot
            return 'test_time_compute'  # Better for extremely low-shot scenarios
        elif n_classes > 10:  # Many classes
            return 'protonet'  # Good for multi-class scenarios
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
        """Get performance statistics for an algorithm.
        
        Args:
            algorithm: Algorithm name to get stats for
            
        Returns:
            Dictionary with mean accuracy, count, and other statistics
        """
        if algorithm not in self.algorithm_performance:
            return {'mean_accuracy': 0.0, 'count': 0}
            
        performances = self.algorithm_performance[algorithm]
        if not performances:
            return {'mean_accuracy': 0.0, 'count': 0}
            
        accuracies = [p['accuracy'] for p in performances]
        return {
            'mean_accuracy': sum(accuracies) / len(accuracies),
            'count': len(performances),
            'min_accuracy': min(accuracies),
            'max_accuracy': max(accuracies)
        }