"""
💰 SUPPORT THIS RESEARCH - PLEASE DONATE! 💰

🙏 If this A/B testing framework helps your research, please donate:
💳 PayPal: https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS
⭐ GitHub Sponsors: https://github.com/sponsors/benedictchen

A/B Testing Framework for Meta-Learning Algorithms - Enhanced with Registry Integration
=====================================================================================

This module provides statistical A/B testing capabilities to compare
different meta-learning algorithms and hyperparameter configurations,
now enhanced with algorithm registry integration and advanced statistical analysis.

Enhanced Features:
- Integration with AlgorithmRegistry for automatic algorithm discovery
- Support for ridge regression, matching networks, and all registered algorithms
- Advanced statistical analysis with confidence intervals and effect sizes
- Multi-metric comparison (accuracy, speed, memory usage)
- Stratified testing based on task characteristics

💰 Please donate if this accelerates your research!
"""

from typing import Dict, List, Any, Optional, Tuple
import hashlib
import numpy as np
from scipy import stats
import time

from .algorithm_registry import algorithm_registry


class ABTestingFramework:
    """
    Enhanced A/B testing framework for algorithm comparison.
    
    Provides capabilities to create controlled experiments comparing
    different meta-learning algorithms with advanced statistical analysis
    and algorithm registry integration.
    
    Enhanced Features:
    - Multi-metric comparison (accuracy, speed, memory)
    - Statistical significance testing with p-values
    - Effect size calculation (Cohen's d)
    - Stratified testing by task characteristics
    - Integration with algorithm registry for automatic setup
    
    Attributes:
        test_groups (Dict): Configuration and results for each A/B test
        results_cache (Dict): Cached analysis results
        registry: Reference to algorithm registry
    """
    
    def __init__(self, use_registry: bool = True):
        """
        Initialize the enhanced A/B testing framework.
        
        Args:
            use_registry: Whether to use algorithm registry for automatic setup
        """
        self.test_groups = {}
        self.results_cache = {}
        self.registry = algorithm_registry if use_registry else None
        
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
    
    def create_registry_based_test(
        self,
        test_name: str,
        algorithm_types: Optional[List[str]] = None,
        include_ridge_regression: bool = True,
        max_algorithms: int = 4
    ):
        """
        Create A/B test using algorithms from the registry.
        
        Args:
            test_name: Unique name for the test
            algorithm_types: Types of algorithms to include (e.g., ['gradient_based', 'metric_based'])
            include_ridge_regression: Whether to include ridge regression
            max_algorithms: Maximum number of algorithms to include
        """
        if not self.registry:
            raise ValueError("Registry not available for automatic test creation")
        
        all_algorithms = self.registry.get_all_algorithms()
        
        # Filter by type if specified
        if algorithm_types:
            from .algorithm_registry import AlgorithmType
            type_enum_map = {t.value: t for t in AlgorithmType}
            filtered_algorithms = {}
            
            for name, metadata in all_algorithms.items():
                if metadata.algorithm_type.value in algorithm_types:
                    filtered_algorithms[name] = metadata
        else:
            filtered_algorithms = all_algorithms
        
        # Ensure ridge regression is included if requested
        if include_ridge_regression and 'ridge_regression' not in filtered_algorithms:
            ridge_metadata = self.registry.get_algorithm_metadata('ridge_regression')
            if ridge_metadata:
                filtered_algorithms['ridge_regression'] = ridge_metadata
        
        # Sort by selection priority and take top algorithms
        sorted_algorithms = sorted(
            filtered_algorithms.items(),
            key=lambda x: x[1].selection_priority,
            reverse=True
        )
        
        selected_algorithms = [name for name, metadata in sorted_algorithms[:max_algorithms]]
        
        # Create the A/B test
        self.create_ab_test(test_name, selected_algorithms)
        
        # Store additional metadata
        self.test_groups[test_name]['registry_metadata'] = {
            name: filtered_algorithms[name] for name in selected_algorithms
        }
    
    def record_multi_metric_result(
        self,
        test_name: str,
        algorithm: str,
        accuracy: float,
        execution_time: float = None,
        memory_usage: float = None,
        additional_metrics: Dict[str, Any] = None
    ):
        """
        Record multi-metric A/B test result.
        
        Args:
            test_name: Name of the test
            algorithm: Algorithm name
            accuracy: Primary accuracy metric
            execution_time: Execution time in seconds
            memory_usage: Memory usage in MB
            additional_metrics: Additional custom metrics
        """
        result = {
            'accuracy': accuracy,
            'timestamp': time.time()
        }
        
        if execution_time is not None:
            result['execution_time'] = execution_time
        
        if memory_usage is not None:
            result['memory_usage'] = memory_usage
        
        if additional_metrics:
            result.update(additional_metrics)
        
        self.record_result(test_name, algorithm, result)
    
    def analyze_ab_test_advanced(self, test_name: str) -> Dict[str, Any]:
        """
        Advanced statistical analysis of A/B test results.
        
        Args:
            test_name: Name of the test to analyze
            
        Returns:
            Comprehensive statistical analysis including:
            - Basic statistics (mean, std, etc.)
            - Statistical significance tests (t-tests, ANOVA)
            - Effect sizes (Cohen's d)
            - Confidence intervals
            - Performance rankings
        """
        if test_name not in self.test_groups:
            return {}
        
        results = {}
        test_data = self.test_groups[test_name]['results']
        algorithms = list(test_data.keys())
        
        # Basic statistics for each algorithm
        basic_stats = {}
        accuracy_data = {}
        
        for algorithm, alg_results in test_data.items():
            if alg_results:
                accuracies = [r.get('accuracy', 0.0) for r in alg_results]
                execution_times = [r.get('execution_time') for r in alg_results if r.get('execution_time') is not None]
                memory_usage = [r.get('memory_usage') for r in alg_results if r.get('memory_usage') is not None]
                
                accuracy_data[algorithm] = accuracies
                
                stats_dict = {
                    'accuracy': {
                        'mean': np.mean(accuracies),
                        'std': np.std(accuracies),
                        'n_samples': len(accuracies),
                        'median': np.median(accuracies),
                        'ci_95': self._calculate_confidence_interval(accuracies)
                    }
                }
                
                if execution_times:
                    stats_dict['execution_time'] = {
                        'mean': np.mean(execution_times),
                        'std': np.std(execution_times),
                        'median': np.median(execution_times),
                        'ci_95': self._calculate_confidence_interval(execution_times)
                    }
                
                if memory_usage:
                    stats_dict['memory_usage'] = {
                        'mean': np.mean(memory_usage),
                        'std': np.std(memory_usage),
                        'median': np.median(memory_usage),
                        'ci_95': self._calculate_confidence_interval(memory_usage)
                    }
                
                basic_stats[algorithm] = stats_dict
        
        results['basic_statistics'] = basic_stats
        
        # Statistical significance testing
        if len(algorithms) >= 2:
            results['statistical_tests'] = self._perform_statistical_tests(accuracy_data)
        
        # Performance ranking
        results['performance_ranking'] = self._calculate_performance_ranking(basic_stats)
        
        # Effect sizes
        if len(algorithms) == 2:
            results['effect_size'] = self._calculate_effect_size(accuracy_data)
        
        # Registry metadata if available
        if 'registry_metadata' in self.test_groups[test_name]:
            results['algorithm_metadata'] = self.test_groups[test_name]['registry_metadata']
        
        return results
    
    def get_best_algorithm(self, test_name: str, metric: str = 'accuracy') -> Tuple[str, Dict[str, Any]]:
        """
        Get the best performing algorithm from an A/B test.
        
        Args:
            test_name: Name of the test
            metric: Metric to optimize for ('accuracy', 'execution_time', 'memory_usage')
            
        Returns:
            Tuple of (best_algorithm_name, performance_details)
        """
        analysis = self.analyze_ab_test_advanced(test_name)
        
        if 'basic_statistics' not in analysis:
            return None, {}
        
        best_algorithm = None
        best_score = float('-inf') if metric == 'accuracy' else float('inf')
        
        for algorithm, stats in analysis['basic_statistics'].items():
            if metric in stats:
                score = stats[metric]['mean']
                
                if metric == 'accuracy':
                    if score > best_score:
                        best_score = score
                        best_algorithm = algorithm
                else:  # Lower is better for time/memory
                    if score < best_score:
                        best_score = score
                        best_algorithm = algorithm
        
        performance_details = {
            'best_score': best_score,
            'metric': metric,
            'algorithm_stats': analysis['basic_statistics'].get(best_algorithm, {}),
            'statistical_significance': analysis.get('statistical_tests', {})
        }
        
        return best_algorithm, performance_details
    
    def compare_algorithms(
        self,
        test_name: str,
        algorithm1: str,
        algorithm2: str,
        metric: str = 'accuracy'
    ) -> Dict[str, Any]:
        """
        Compare two specific algorithms from an A/B test.
        
        Args:
            test_name: Name of the test
            algorithm1: First algorithm to compare
            algorithm2: Second algorithm to compare
            metric: Metric to compare on
            
        Returns:
            Detailed comparison including statistical significance
        """
        if test_name not in self.test_groups:
            return {}
        
        test_data = self.test_groups[test_name]['results']
        
        if algorithm1 not in test_data or algorithm2 not in test_data:
            return {'error': 'One or both algorithms not found in test results'}
        
        # Extract data for comparison
        data1 = [r.get(metric, 0.0) for r in test_data[algorithm1] if r.get(metric) is not None]
        data2 = [r.get(metric, 0.0) for r in test_data[algorithm2] if r.get(metric) is not None]
        
        if not data1 or not data2:
            return {'error': f'Insufficient data for {metric} comparison'}
        
        # Statistical comparison
        comparison = {
            'algorithm1': {
                'name': algorithm1,
                'mean': np.mean(data1),
                'std': np.std(data1),
                'n_samples': len(data1),
                'ci_95': self._calculate_confidence_interval(data1)
            },
            'algorithm2': {
                'name': algorithm2,
                'mean': np.mean(data2),
                'std': np.std(data2),
                'n_samples': len(data2),
                'ci_95': self._calculate_confidence_interval(data2)
            }
        }
        
        # T-test
        try:
            t_stat, p_value = stats.ttest_ind(data1, data2)
            comparison['t_test'] = {
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
        except:
            comparison['t_test'] = {'error': 'T-test failed'}
        
        # Effect size (Cohen's d)
        try:
            pooled_std = np.sqrt(((len(data1) - 1) * np.var(data1, ddof=1) + 
                                (len(data2) - 1) * np.var(data2, ddof=1)) / 
                               (len(data1) + len(data2) - 2))
            cohens_d = (np.mean(data1) - np.mean(data2)) / pooled_std
            
            comparison['effect_size'] = {
                'cohens_d': cohens_d,
                'magnitude': self._interpret_cohens_d(cohens_d)
            }
        except:
            comparison['effect_size'] = {'error': 'Effect size calculation failed'}
        
        # Winner determination
        if metric == 'accuracy':
            comparison['winner'] = algorithm1 if np.mean(data1) > np.mean(data2) else algorithm2
        else:  # Lower is better
            comparison['winner'] = algorithm1 if np.mean(data1) < np.mean(data2) else algorithm2
        
        comparison['performance_difference'] = abs(np.mean(data1) - np.mean(data2))
        
        return comparison
    
    def _calculate_confidence_interval(self, data: List[float], confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for data."""
        if len(data) < 2:
            return (0.0, 0.0)
        
        mean = np.mean(data)
        sem = stats.sem(data)
        h = sem * stats.t.ppf((1 + confidence) / 2, len(data) - 1)
        
        return (mean - h, mean + h)
    
    def _perform_statistical_tests(self, accuracy_data: Dict[str, List[float]]) -> Dict[str, Any]:
        """Perform statistical significance tests."""
        algorithms = list(accuracy_data.keys())
        results = {}
        
        # ANOVA if more than 2 groups
        if len(algorithms) > 2:
            try:
                data_arrays = [accuracy_data[alg] for alg in algorithms if accuracy_data[alg]]
                f_stat, p_value = stats.f_oneway(*data_arrays)
                
                results['anova'] = {
                    'f_statistic': f_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
            except:
                results['anova'] = {'error': 'ANOVA test failed'}
        
        # Pairwise t-tests
        pairwise_tests = {}
        for i, alg1 in enumerate(algorithms):
            for alg2 in algorithms[i+1:]:
                if accuracy_data[alg1] and accuracy_data[alg2]:
                    try:
                        t_stat, p_value = stats.ttest_ind(accuracy_data[alg1], accuracy_data[alg2])
                        pairwise_tests[f"{alg1}_vs_{alg2}"] = {
                            't_statistic': t_stat,
                            'p_value': p_value,
                            'significant': p_value < 0.05
                        }
                    except:
                        pairwise_tests[f"{alg1}_vs_{alg2}"] = {'error': 'T-test failed'}
        
        results['pairwise_t_tests'] = pairwise_tests
        
        return results
    
    def _calculate_performance_ranking(self, basic_stats: Dict[str, Dict]) -> List[Dict[str, Any]]:
        """Calculate performance ranking of algorithms."""
        ranking = []
        
        for algorithm, stats in basic_stats.items():
            if 'accuracy' in stats:
                ranking.append({
                    'algorithm': algorithm,
                    'mean_accuracy': stats['accuracy']['mean'],
                    'std_accuracy': stats['accuracy']['std'],
                    'n_samples': stats['accuracy']['n_samples'],
                    'confidence_interval': stats['accuracy']['ci_95']
                })
        
        # Sort by mean accuracy (descending)
        ranking.sort(key=lambda x: x['mean_accuracy'], reverse=True)
        
        # Add rank
        for i, entry in enumerate(ranking):
            entry['rank'] = i + 1
        
        return ranking
    
    def _calculate_effect_size(self, accuracy_data: Dict[str, List[float]]) -> Dict[str, Any]:
        """Calculate effect size for two-algorithm comparison."""
        algorithms = list(accuracy_data.keys())
        if len(algorithms) != 2:
            return {'error': 'Effect size calculation requires exactly 2 algorithms'}
        
        alg1, alg2 = algorithms
        data1 = accuracy_data[alg1]
        data2 = accuracy_data[alg2]
        
        if not data1 or not data2:
            return {'error': 'Insufficient data for effect size calculation'}
        
        try:
            pooled_std = np.sqrt(((len(data1) - 1) * np.var(data1, ddof=1) + 
                                (len(data2) - 1) * np.var(data2, ddof=1)) / 
                               (len(data1) + len(data2) - 2))
            
            cohens_d = (np.mean(data1) - np.mean(data2)) / pooled_std
            
            return {
                'cohens_d': cohens_d,
                'magnitude': self._interpret_cohens_d(cohens_d),
                'algorithm1': alg1,
                'algorithm2': alg2
            }
        except:
            return {'error': 'Effect size calculation failed'}
    
    def _interpret_cohens_d(self, cohens_d: float) -> str:
        """Interpret Cohen's d effect size."""
        abs_d = abs(cohens_d)
        
        if abs_d < 0.2:
            return 'negligible'
        elif abs_d < 0.5:
            return 'small'
        elif abs_d < 0.8:
            return 'medium'
        else:
            return 'large'