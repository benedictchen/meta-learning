"""
A/B Testing Framework for Meta-Learning Algorithms.

This module provides statistical A/B testing capabilities to compare
different meta-learning algorithms and hyperparameter configurations.

Classes:
    ABTestingFramework: Creates and manages A/B tests for algorithm comparison
                       with deterministic group assignment and statistical analysis.

The framework uses hash-based assignment to ensure reproducible test groups
and provides statistical analysis of results across different algorithms.
"""

from typing import Dict, List, Any
import hashlib
import numpy as np


class ABTestingFramework:
    """A/B testing framework for algorithm comparison.
    
    Provides capabilities to create controlled experiments comparing
    different meta-learning algorithms with proper statistical analysis.
    
    Attributes:
        test_groups (Dict): Configuration and results for each A/B test
        results_cache (Dict): Cached analysis results
    """
    
    def __init__(self):
        """Initialize the A/B testing framework."""
        self.test_groups = {}
        self.results_cache = {}
        
    def create_ab_test(self, test_name: str, algorithms: List[str], allocation_ratio: List[float] = None):
        """Create A/B test configuration.
        
        Sets up a new A/B test with specified algorithms and allocation ratios.
        
        Args:
            test_name: Unique name for this A/B test
            algorithms: List of algorithm names to compare
            allocation_ratio: Optional list of allocation ratios (defaults to equal split)
            
        Raises:
            ValueError: If algorithms and allocation_ratio have different lengths
        """
        if allocation_ratio is None:
            allocation_ratio = [1.0 / len(algorithms)] * len(algorithms)
        
        if len(algorithms) != len(allocation_ratio):
            raise ValueError("Algorithms and allocation ratios must have same length")
        
        self.test_groups[test_name] = {
            'algorithms': algorithms,
            'allocation_ratio': allocation_ratio,
            'results': {alg: [] for alg in algorithms}
        }
    
    def assign_algorithm(self, test_name: str, episode_id: str) -> str:
        """Assign episode to algorithm group.
        
        Uses deterministic hash-based assignment to ensure reproducible
        group assignments while maintaining proper allocation ratios.
        
        Args:
            test_name: Name of the A/B test
            episode_id: Unique identifier for the episode
            
        Returns:
            Algorithm name assigned to this episode
            
        Raises:
            ValueError: If test_name is not found
        """
        if test_name not in self.test_groups:
            raise ValueError(f"Test {test_name} not found")
        
        # Deterministic assignment based on episode_id hash
        hash_val = int(hashlib.md5(episode_id.encode()).hexdigest(), 16)
        rand_val = (hash_val % 1000) / 1000.0
        
        algorithms = self.test_groups[test_name]['algorithms']
        ratios = self.test_groups[test_name]['allocation_ratio']
        
        cumulative_ratio = 0
        for i, ratio in enumerate(ratios):
            cumulative_ratio += ratio
            if rand_val <= cumulative_ratio:
                return algorithms[i]
        
        return algorithms[-1]  # Fallback
    
    def record_result(self, test_name: str, algorithm: str, result: Dict[str, Any]):
        """Record A/B test result.
        
        Stores the result of running an algorithm on an episode for later analysis.
        
        Args:
            test_name: Name of the A/B test
            algorithm: Algorithm that was used
            result: Dictionary containing result metrics (should include 'accuracy')
        """
        if test_name in self.test_groups:
            self.test_groups[test_name]['results'][algorithm].append(result)
    
    def analyze_ab_test(self, test_name: str) -> Dict[str, Any]:
        """Analyze A/B test results.
        
        Computes statistical summaries for each algorithm in the test,
        including mean accuracy, standard deviation, and sample size.
        
        Args:
            test_name: Name of the A/B test to analyze
            
        Returns:
            Dictionary with statistical analysis for each algorithm:
            {
                'algorithm_name': {
                    'mean_accuracy': float,
                    'std_accuracy': float, 
                    'n_samples': int
                }
            }
        """
        if test_name not in self.test_groups:
            return {}
        
        results = {}
        test_data = self.test_groups[test_name]['results']
        
        for algorithm, alg_results in test_data.items():
            if alg_results:
                accuracies = [r.get('accuracy', 0.0) for r in alg_results]
                results[algorithm] = {
                    'mean_accuracy': np.mean(accuracies),
                    'std_accuracy': np.std(accuracies),
                    'n_samples': len(accuracies)
                }
        
        return results
    
    def get_test_summary(self, test_name: str) -> Dict[str, Any]:
        """Get summary information about an A/B test.
        
        Args:
            test_name: Name of the A/B test
            
        Returns:
            Summary including configuration and basic statistics
        """
        if test_name not in self.test_groups:
            return {}
            
        test_config = self.test_groups[test_name]
        summary = {
            'algorithms': test_config['algorithms'],
            'allocation_ratio': test_config['allocation_ratio'],
            'total_episodes': sum(len(results) for results in test_config['results'].values()),
            'results_per_algorithm': {
                alg: len(results) for alg, results in test_config['results'].items()
            }
        }
        
        return summary