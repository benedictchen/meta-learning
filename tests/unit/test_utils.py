"""
Comprehensive unit tests for utils module.

Tests all research solutions, dataset utilities, evaluation metrics, 
statistical analysis tools, and configuration options following 2024/2025 pytest best practices.
"""

import pytest
import torch
# DEMO/TEST REPRODUCIBILITY: Set seed for consistent synthetic data
torch.manual_seed(42)  # Reproducible test/demo data
import torch.nn as nn
# DEMO/TEST REPRODUCIBILITY: Set seed for consistent synthetic data
torch.manual_seed(42)  # Reproducible test/demo data
import numpy as np
import tempfile
import json
import os
from unittest.mock import Mock, patch, MagicMock
from hypothesis import given, strategies as st, settings
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path

from meta_learning.meta_learning_modules.utils import (
    MetaLearningDataset,
    DatasetConfig,
    EvaluationMetrics,
    MetricsConfig,
    StatisticalAnalysis,
    StatsConfig,
    CurriculumLearning,
    CurriculumConfig,
    TaskDiversityTracker,
    DiversityConfig,
    create_dataset,
    create_metrics_evaluator,
    create_curriculum_scheduler,
    basic_confidence_interval,
    compute_confidence_interval,
    estimate_difficulty,
    track_task_diversity
)


class TestDatasetConfig:
    """Test dataset configuration."""
    
    def test_dataset_config_defaults(self):
        """Test default dataset configuration values."""
        config = DatasetConfig()
        assert config.n_way == 5
        assert config.k_shot == 1
        assert config.query_shots == 15
        assert config.num_tasks == 1000
        assert config.dataset_path == "./data"
        assert config.cache_episodes == True
        
    def test_dataset_config_validation(self):
        """Test dataset configuration validation."""
        # Valid configurations
        config = DatasetConfig(n_way=3, k_shot=5, query_shots=10)
        assert config.n_way == 3
        assert config.k_shot == 5
        assert config.query_shots == 10
        
    @given(
        n_way=st.integers(min_value=2, max_value=20),
        k_shot=st.integers(min_value=1, max_value=10),
        num_tasks=st.integers(min_value=100, max_value=10000)
    )
    def test_dataset_config_property_based(self, n_way, k_shot, num_tasks):
        """Property-based test for dataset configuration."""
        config = DatasetConfig(n_way=n_way, k_shot=k_shot, num_tasks=num_tasks)
        assert config.n_way == n_way
        assert config.k_shot == k_shot
        assert config.num_tasks == num_tasks


class TestMetricsConfig:
    """Test metrics configuration."""
    
    def test_metrics_config_defaults(self):
        """Test default metrics configuration."""
        config = MetricsConfig()
        assert config.confidence_level == 0.95
        assert config.confidence_method == "auto"
        assert config.use_bootstrap == True
        assert config.bootstrap_samples == 1000
        assert config.compute_std_error == True
        
    def test_metrics_config_confidence_methods(self):
        """Test different confidence interval methods."""
        methods = ["auto", "bootstrap", "t_distribution", "normal", "bca"]
        for method in methods:
            config = MetricsConfig(confidence_method=method)
            assert config.confidence_method == method
            
    @given(
        confidence_level=st.floats(min_value=0.8, max_value=0.99),
        bootstrap_samples=st.integers(min_value=500, max_value=5000)
    )
    def test_metrics_config_property_based(self, confidence_level, bootstrap_samples):
        """Property-based test for metrics configuration."""
        config = MetricsConfig(
            confidence_level=confidence_level,
            bootstrap_samples=bootstrap_samples
        )
        assert config.confidence_level == confidence_level
        assert config.bootstrap_samples == bootstrap_samples


class TestStatsConfig:
    """Test statistical analysis configuration."""
    
    def test_stats_config_defaults(self):
        """Test default statistical configuration."""
        config = StatsConfig()
        assert config.significance_level == 0.05
        assert config.multiple_comparisons == "bonferroni"
        assert config.effect_size_measure == "cohen_d"
        assert config.use_welch_ttest == True
        
    def test_stats_config_multiple_comparisons(self):
        """Test multiple comparison correction methods."""
        methods = ["bonferroni", "holm", "fdr_bh", "none"]
        for method in methods:
            config = StatsConfig(multiple_comparisons=method)
            assert config.multiple_comparisons == method


class TestCurriculumConfig:
    """Test curriculum learning configuration."""
    
    def test_curriculum_config_defaults(self):
        """Test default curriculum configuration."""
        config = CurriculumConfig()
        assert config.curriculum_strategy == "difficulty_based"
        assert config.initial_difficulty == 0.3
        assert config.max_difficulty == 1.0
        assert config.difficulty_increment == 0.1
        assert config.adaptation_window == 100
        
    def test_curriculum_config_strategies(self):
        """Test different curriculum strategies."""
        strategies = ["difficulty_based", "diversity_based", "random", "fixed"]
        for strategy in strategies:
            config = CurriculumConfig(curriculum_strategy=strategy)
            assert config.curriculum_strategy == strategy


class TestDiversityConfig:
    """Test task diversity configuration."""
    
    def test_diversity_config_defaults(self):
        """Test default diversity configuration."""
        config = DiversityConfig()
        assert config.diversity_metric == "jensen_shannon"
        assert config.track_feature_diversity == True
        assert config.track_label_diversity == True
        assert config.diversity_window == 50
        assert config.min_diversity_threshold == 0.1


class TestMetaLearningDataset:
    """Test meta-learning dataset implementation."""
    
    @pytest.fixture
    def temp_dataset_dir(self):
        """Create temporary directory for dataset testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
            
    @pytest.fixture
    def sample_dataset_config(self, temp_dataset_dir):
        """Create sample dataset configuration."""
        return DatasetConfig(
            n_way=3,
            k_shot=2,
            query_shots=10,
            num_tasks=50,
            dataset_path=temp_dataset_dir
        )
        
    @pytest.fixture
    def mock_data_files(self, temp_dataset_dir):
        """Create mock data files for testing."""
        # Create some dummy data files
        data_dir = Path(temp_dataset_dir)
        data_dir.mkdir(exist_ok=True)
        
        # Create mock class directories
        for i in range(10):
            class_dir = data_dir / f"class_{i}"
            class_dir.mkdir(exist_ok=True)
            
            # Create mock data files
            for j in range(20):
                data_file = class_dir / f"sample_{j}.json"
                with open(data_file, 'w') as f:
                    json.dump({
                        "features": np.random.randn(32).tolist(),
                        "label": i
                    }, f)
                    
        return data_dir
        
    def test_meta_learning_dataset_init(self, sample_dataset_config, mock_data_files):
        """Test meta-learning dataset initialization."""
        dataset = MetaLearningDataset(sample_dataset_config)
        
        assert dataset.config == sample_dataset_config
        assert dataset.n_way == 3
        assert dataset.k_shot == 2
        assert dataset.query_shots == 10
        
    def test_dataset_length(self, sample_dataset_config, mock_data_files):
        """Test dataset length property."""
        dataset = MetaLearningDataset(sample_dataset_config)
        
        assert len(dataset) == sample_dataset_config.num_tasks
        
    def test_dataset_getitem(self, sample_dataset_config, mock_data_files):
        """Test dataset item access.""" 
        dataset = MetaLearningDataset(sample_dataset_config)
        
        # Get a sample episode
        episode = dataset[0]
        
        # Check episode structure
        assert 'support_x' in episode
        assert 'support_y' in episode  
        assert 'query_x' in episode
        assert 'query_y' in episode
        
        # Check dimensions
        assert episode['support_x'].shape == (3, 2, dataset.feature_dim)  # n_way, k_shot, features
        assert episode['support_y'].shape == (6,)  # n_way * k_shot
        assert episode['query_x'].shape == (30, dataset.feature_dim)  # n_way * query_shots, features
        assert episode['query_y'].shape == (30,)  # n_way * query_shots
        
    def test_dataset_caching(self, sample_dataset_config, mock_data_files):
        """Test dataset episode caching."""
        sample_dataset_config.cache_episodes = True
        dataset = MetaLearningDataset(sample_dataset_config)
        
        # Access same episode multiple times
        episode1 = dataset[5]
        episode2 = dataset[5]
        
        # Should be identical due to caching
        assert torch.allclose(episode1['support_x'], episode2['support_x'])
        assert torch.equal(episode1['support_y'], episode2['support_y'])
        
    def test_dataset_no_caching(self, sample_dataset_config, mock_data_files):
        """Test dataset without caching (episodes should be different)."""
        sample_dataset_config.cache_episodes = False
        dataset = MetaLearningDataset(sample_dataset_config)
        
        # Access same index multiple times
        episode1 = dataset[5]
        episode2 = dataset[5]
        
        # Should be different due to random sampling
        # Note: This might occasionally fail due to randomness, but very unlikely
        different = not torch.allclose(episode1['support_x'], episode2['support_x'])
        assert different or not torch.equal(episode1['support_y'], episode2['support_y'])
        
    def test_dataset_different_episodes(self, sample_dataset_config, mock_data_files):
        """Test that different indices return different episodes."""
        dataset = MetaLearningDataset(sample_dataset_config)
        
        episode1 = dataset[0]
        episode2 = dataset[1] 
        
        # Episodes should be different
        different = (not torch.allclose(episode1['support_x'], episode2['support_x']) or
                    not torch.equal(episode1['support_y'], episode2['support_y']))
        assert different


class TestEvaluationMetrics:
    """Test evaluation metrics implementation."""
    
    @pytest.fixture
    def metrics_evaluator(self):
        """Create evaluation metrics with default config."""
        config = MetricsConfig()
        return EvaluationMetrics(config)
        
    @pytest.fixture
    def sample_predictions(self):
        """Create sample predictions and targets."""
        batch_size = 100
        n_classes = 5
        
        # Create logits and targets
        logits = torch.randn(batch_size, n_classes)
        targets = torch.randint(0, n_classes, (batch_size,))
        
        # Create some correct predictions for testing
        predictions = torch.argmax(logits, dim=1)
        # Make first 70 predictions correct
        predictions[:70] = targets[:70]
        
        return logits, predictions, targets
        
    def test_evaluation_metrics_init(self):
        """Test evaluation metrics initialization."""
        config = MetricsConfig()
        metrics = EvaluationMetrics(config)
        
        assert metrics.config == config
        assert metrics.confidence_level == 0.95
        
    def test_accuracy_computation(self, metrics_evaluator, sample_predictions):
        """Test accuracy computation."""
        _, predictions, targets = sample_predictions
        
        accuracy = metrics_evaluator.compute_accuracy(predictions, targets)
        
        assert 0 <= accuracy <= 1
        assert isinstance(accuracy, float)
        # Should be around 0.7 since we made first 70 correct
        assert 0.6 <= accuracy <= 0.8
        
    def test_confidence_interval_computation(self, metrics_evaluator, sample_predictions):
        """Test confidence interval computation."""
        _, predictions, targets = sample_predictions
        
        accuracies = []
        for _ in range(10):
            # Create bootstrap samples
            indices = torch.randint(0, len(targets), (len(targets),))
            boot_pred = predictions[indices]
            boot_target = targets[indices]
            acc = (boot_pred == boot_target).float().mean().item()
            accuracies.append(acc)
            
        ci_lower, ci_upper = metrics_evaluator.compute_confidence_interval(accuracies)
        
        assert ci_lower <= ci_upper
        assert 0 <= ci_lower <= 1
        assert 0 <= ci_upper <= 1
        
    def test_bootstrap_confidence_interval(self, metrics_evaluator, sample_predictions):
        """Test bootstrap confidence interval computation."""
        _, predictions, targets = sample_predictions
        
        ci_lower, ci_upper = metrics_evaluator.bootstrap_confidence_interval(predictions, targets)
        
        assert ci_lower <= ci_upper
        assert 0 <= ci_lower <= 1
        assert 0 <= ci_upper <= 1
        
    def test_standard_error_computation(self, metrics_evaluator, sample_predictions):
        """Test standard error computation."""
        _, predictions, targets = sample_predictions
        
        std_error = metrics_evaluator.compute_standard_error(predictions, targets)
        
        assert std_error >= 0
        assert isinstance(std_error, float)
        
    def test_evaluate_episode(self, metrics_evaluator, sample_predictions):
        """Test complete episode evaluation."""
        logits, predictions, targets = sample_predictions
        
        results = metrics_evaluator.evaluate_episode(logits, targets)
        
        assert 'accuracy' in results
        assert 'confidence_interval' in results
        assert 'standard_error' in results
        
        assert 0 <= results['accuracy'] <= 1
        assert len(results['confidence_interval']) == 2
        assert results['confidence_interval'][0] <= results['confidence_interval'][1]
        assert results['standard_error'] >= 0
        
    def test_different_confidence_methods(self, sample_predictions):
        """Test different confidence interval methods."""
        methods = ["auto", "bootstrap", "t_distribution", "normal"]
        logits, predictions, targets = sample_predictions
        
        for method in methods:
            config = MetricsConfig(confidence_method=method)
            evaluator = EvaluationMetrics(config)
            
            results = evaluator.evaluate_episode(logits, targets)
            
            assert 'confidence_interval' in results
            assert len(results['confidence_interval']) == 2
            assert results['confidence_interval'][0] <= results['confidence_interval'][1]


class TestStatisticalAnalysis:
    """Test statistical analysis implementation."""
    
    @pytest.fixture
    def stats_analyzer(self):
        """Create statistical analysis with default config."""
        config = StatsConfig()
        return StatisticalAnalysis(config)
        
    @pytest.fixture
    def sample_results(self):
        """Create sample experimental results."""
        np.random.seed(42)  # For reproducibility
        
        # Method A: higher mean accuracy
        method_a = np.random.normal(0.85, 0.05, 50)
        method_a = np.clip(method_a, 0, 1)  # Clip to [0, 1]
        
        # Method B: lower mean accuracy
        method_b = np.random.normal(0.80, 0.05, 50)
        method_b = np.clip(method_b, 0, 1)
        
        return method_a, method_b
        
    def test_statistical_analysis_init(self):
        """Test statistical analysis initialization."""
        config = StatsConfig()
        analyzer = StatisticalAnalysis(config)
        
        assert analyzer.config == config
        assert analyzer.significance_level == 0.05
        
    def test_t_test_computation(self, stats_analyzer, sample_results):
        """Test t-test computation."""
        method_a, method_b = sample_results
        
        t_stat, p_value = stats_analyzer.compute_t_test(method_a, method_b)
        
        assert isinstance(t_stat, float)
        assert isinstance(p_value, float)
        assert 0 <= p_value <= 1
        
        # Method A should have higher mean, so t_stat should be positive
        assert t_stat > 0
        
    def test_welch_t_test(self, stats_analyzer, sample_results):
        """Test Welch's t-test (unequal variances)."""
        method_a, method_b = sample_results
        
        # Modify config to use Welch's t-test
        stats_analyzer.config.use_welch_ttest = True
        
        t_stat, p_value = stats_analyzer.compute_t_test(method_a, method_b)
        
        assert isinstance(t_stat, float)
        assert isinstance(p_value, float)
        assert 0 <= p_value <= 1
        
    def test_effect_size_computation(self, stats_analyzer, sample_results):
        """Test effect size computation."""
        method_a, method_b = sample_results
        
        effect_size = stats_analyzer.compute_effect_size(method_a, method_b)
        
        assert isinstance(effect_size, float)
        # Should be positive since method_a has higher mean
        assert effect_size > 0
        
    def test_multiple_comparisons_correction(self, stats_analyzer):
        """Test multiple comparisons correction."""
        # Create multiple p-values
        p_values = [0.01, 0.03, 0.06, 0.08, 0.12]
        
        corrected_p_values = stats_analyzer.correct_multiple_comparisons(p_values)
        
        assert len(corrected_p_values) == len(p_values)
        # Bonferroni correction should make p-values larger (more conservative)
        for original, corrected in zip(p_values, corrected_p_values):
            assert corrected >= original
            
    def test_compare_multiple_methods(self, stats_analyzer, sample_results):
        """Test comparison of multiple methods."""
        method_a, method_b = sample_results
        
        # Create a third method
        np.random.seed(123)
        method_c = np.random.normal(0.75, 0.05, 50)
        method_c = np.clip(method_c, 0, 1)
        
        methods = {
            'Method A': method_a,
            'Method B': method_b, 
            'Method C': method_c
        }
        
        comparison_results = stats_analyzer.compare_methods(methods)
        
        assert 'pairwise_comparisons' in comparison_results
        assert 'effect_sizes' in comparison_results
        assert 'corrected_p_values' in comparison_results
        
        # Check that we have comparisons for all pairs
        n_methods = len(methods)
        expected_pairs = n_methods * (n_methods - 1) // 2
        assert len(comparison_results['pairwise_comparisons']) == expected_pairs


class TestCurriculumLearning:
    """Test curriculum learning implementation."""
    
    @pytest.fixture
    def curriculum_scheduler(self):
        """Create curriculum learning scheduler."""
        config = CurriculumConfig()
        return CurriculumLearning(config)
        
    @pytest.fixture
    def sample_performance_history(self):
        """Create sample performance history."""
        # Simulating learning curve: starts low, improves over time
        history = []
        for i in range(200):
            # Add some noise to make it realistic
            base_performance = min(0.9, 0.3 + 0.6 * (i / 200))
            noisy_performance = base_performance + np.random.normal(0, 0.05)
            history.append(max(0, min(1, noisy_performance)))
        return history
        
    def test_curriculum_learning_init(self):
        """Test curriculum learning initialization."""
        config = CurriculumConfig()
        scheduler = CurriculumLearning(config)
        
        assert scheduler.config == config
        assert scheduler.current_difficulty == config.initial_difficulty
        
    def test_difficulty_based_curriculum(self, curriculum_scheduler, sample_performance_history):
        """Test difficulty-based curriculum scheduling.""" 
        initial_difficulty = curriculum_scheduler.current_difficulty
        
        # Simulate training with good performance
        for i, performance in enumerate(sample_performance_history[:50]):
            difficulty = curriculum_scheduler.update_difficulty(performance, step=i)
            
        # Difficulty should increase with good performance
        assert curriculum_scheduler.current_difficulty > initial_difficulty
        
    def test_curriculum_adaptation_window(self, sample_performance_history):
        """Test curriculum adaptation window."""
        config = CurriculumConfig(adaptation_window=20)
        scheduler = CurriculumLearning(config)
        
        # Performance history should only consider last 20 steps
        for i, performance in enumerate(sample_performance_history[:50]):
            scheduler.update_difficulty(performance, step=i)
            
        # Check that performance history is limited
        assert len(scheduler.performance_history) <= config.adaptation_window
        
    def test_max_difficulty_limit(self, sample_performance_history):
        """Test maximum difficulty limit."""
        config = CurriculumConfig(max_difficulty=0.8)
        scheduler = CurriculumLearning(config)
        
        # Simulate very good performance to try to push difficulty high
        for i in range(100):
            scheduler.update_difficulty(0.95, step=i)  # Consistently high performance
            
        # Should not exceed max difficulty
        assert scheduler.current_difficulty <= config.max_difficulty
        
    def test_different_curriculum_strategies(self, sample_performance_history):
        """Test different curriculum strategies."""
        strategies = ["difficulty_based", "diversity_based", "random", "fixed"]
        
        for strategy in strategies:
            config = CurriculumConfig(curriculum_strategy=strategy)
            scheduler = CurriculumLearning(config)
            
            # Update difficulty for a few steps
            for i, performance in enumerate(sample_performance_history[:10]):
                difficulty = scheduler.update_difficulty(performance, step=i)
                assert 0 <= difficulty <= 1
                
    def test_curriculum_step_function(self, curriculum_scheduler, sample_performance_history):
        """Test curriculum step function."""
        step = 0
        
        for performance in sample_performance_history[:20]:
            difficulty = curriculum_scheduler.step(performance)
            step += 1
            
            assert 0 <= difficulty <= 1
            assert isinstance(difficulty, float)
            
    def test_get_current_difficulty(self, curriculum_scheduler):
        """Test getting current difficulty."""
        difficulty = curriculum_scheduler.get_current_difficulty()
        
        assert isinstance(difficulty, float)
        assert 0 <= difficulty <= 1
        assert difficulty == curriculum_scheduler.config.initial_difficulty


class TestTaskDiversityTracker:
    """Test task diversity tracking implementation."""
    
    @pytest.fixture
    def diversity_tracker(self):
        """Create task diversity tracker."""
        config = DiversityConfig()
        return TaskDiversityTracker(config)
        
    @pytest.fixture
    def sample_tasks(self):
        """Create sample tasks for diversity tracking."""
        tasks = []
        for i in range(20):
            task = {
                'features': torch.randn(10, 32),  # Random features
                'labels': torch.randint(0, 5, (10,)),  # Random labels
                'task_id': f"diversity_task_{i}",
                'difficulty': np.random.uniform(0.2, 0.9)
            }
            tasks.append(task)
        return tasks
        
    def test_task_diversity_tracker_init(self):
        """Test task diversity tracker initialization."""
        config = DiversityConfig()
        tracker = TaskDiversityTracker(config)
        
        assert tracker.config == config
        assert len(tracker.task_history) == 0
        
    def test_add_task_to_tracker(self, diversity_tracker, sample_tasks):
        """Test adding tasks to diversity tracker."""
        task = sample_tasks[0]
        
        diversity_tracker.add_task(task)
        
        assert len(diversity_tracker.task_history) == 1
        assert diversity_tracker.task_history[0] == task
        
    def test_compute_feature_diversity(self, diversity_tracker, sample_tasks):
        """Test feature diversity computation."""
        # Add several tasks
        for task in sample_tasks[:5]:
            diversity_tracker.add_task(task)
            
        diversity = diversity_tracker.compute_feature_diversity()
        
        assert isinstance(diversity, float)
        assert 0 <= diversity <= 1
        
    def test_compute_label_diversity(self, diversity_tracker, sample_tasks):
        """Test label diversity computation."""
        # Add several tasks  
        for task in sample_tasks[:5]:
            diversity_tracker.add_task(task)
            
        diversity = diversity_tracker.compute_label_diversity()
        
        assert isinstance(diversity, float)
        assert 0 <= diversity <= 1
        
    def test_diversity_window_management(self, sample_tasks):
        """Test diversity window size management."""
        config = DiversityConfig(diversity_window=3)
        tracker = TaskDiversityTracker(config)
        
        # Add more tasks than window size
        for task in sample_tasks[:10]:
            tracker.add_task(task)
            
        # Should only keep tasks within window
        assert len(tracker.task_history) <= config.diversity_window
        
    def test_get_current_diversity(self, diversity_tracker, sample_tasks):
        """Test getting current diversity metrics."""
        # Add several tasks
        for task in sample_tasks[:8]:
            diversity_tracker.add_task(task)
            
        diversity_metrics = diversity_tracker.get_current_diversity()
        
        assert 'feature_diversity' in diversity_metrics
        assert 'label_diversity' in diversity_metrics
        assert 'overall_diversity' in diversity_metrics
        
        for key, value in diversity_metrics.items():
            assert isinstance(value, float)
            assert 0 <= value <= 1
            
    def test_different_diversity_metrics(self, sample_tasks):
        """Test different diversity metrics."""
        metrics = ["jensen_shannon", "bhattacharyya", "kl_divergence", "euclidean"]
        
        for metric in metrics:
            config = DiversityConfig(diversity_metric=metric)
            tracker = TaskDiversityTracker(config)
            
            # Add tasks
            for task in sample_tasks[:5]:
                tracker.add_task(task)
                
            diversity = tracker.compute_feature_diversity()
            assert isinstance(diversity, float)
            assert diversity >= 0  # Some metrics may not be bounded by 1


class TestUtilityFunctions:
    """Test standalone utility functions."""
    
    def test_basic_confidence_interval_function(self):
        """Test basic confidence interval function."""
        # Create sample data
        values = [0.8, 0.82, 0.78, 0.85, 0.79, 0.83, 0.81, 0.84, 0.80, 0.86]
        
        ci_lower, ci_upper = basic_confidence_interval(values, confidence_level=0.95)
        
        assert ci_lower <= ci_upper
        assert isinstance(ci_lower, float)
        assert isinstance(ci_upper, float)
        
    def test_basic_ci_different_confidence_levels(self):
        """Test basic CI with different confidence levels."""
        values = [0.75, 0.80, 0.78, 0.82, 0.77, 0.85, 0.79, 0.83]
        
        # Higher confidence level should give wider interval
        ci_90_lower, ci_90_upper = basic_confidence_interval(values, confidence_level=0.90)
        ci_99_lower, ci_99_upper = basic_confidence_interval(values, confidence_level=0.99)
        
        # 99% CI should be wider than 90% CI
        assert (ci_99_upper - ci_99_lower) >= (ci_90_upper - ci_90_lower)
        
    def test_basic_ci_small_sample_size(self):
        """Test basic CI with small sample size (should use t-distribution)."""
        # Small sample (n < 30)
        small_sample = [0.78, 0.82, 0.80, 0.79, 0.85]
        
        ci_lower, ci_upper = basic_confidence_interval(small_sample, confidence_level=0.95)
        
        assert ci_lower <= ci_upper
        assert isinstance(ci_lower, float)
        assert isinstance(ci_upper, float)
        
    def test_compute_confidence_interval_function(self):
        """Test enhanced confidence interval function."""
        values = [0.85, 0.87, 0.83, 0.88, 0.84, 0.86, 0.89, 0.82, 0.87, 0.85]
        
        # Test different methods
        methods = ["auto", "bootstrap", "t_distribution", "normal"]
        
        for method in methods:
            ci_lower, ci_upper = compute_confidence_interval(
                values, 
                method=method,
                confidence_level=0.95
            )
            
            assert ci_lower <= ci_upper
            assert isinstance(ci_lower, float)
            assert isinstance(ci_upper, float)
            
    def test_estimate_difficulty_function(self):
        """Test task difficulty estimation function."""
        # Create mock task data
        task_data = {
            'features': torch.randn(50, 64),
            'labels': torch.randint(0, 5, (50,)),
            'n_way': 5,
            'k_shot': 3
        }
        
        difficulty = estimate_difficulty(task_data, method="feature_variance")
        
        assert isinstance(difficulty, float)
        assert 0 <= difficulty <= 1
        
    def test_estimate_difficulty_different_methods(self):
        """Test different difficulty estimation methods."""
        task_data = {
            'features': torch.randn(30, 32),
            'labels': torch.randint(0, 3, (30,)),
            'n_way': 3,
            'k_shot': 2
        }
        
        methods = ["feature_variance", "label_entropy", "inter_class_distance", "intra_class_variance"]
        
        for method in methods:
            difficulty = estimate_difficulty(task_data, method=method)
            assert isinstance(difficulty, float)
            assert 0 <= difficulty <= 1
            
    def test_track_task_diversity_function(self):
        """Test task diversity tracking function."""
        tasks = []
        for i in range(10):
            task = {
                'features': torch.randn(20, 16),
                'labels': torch.randint(0, 4, (20,)),
                'task_id': f"track_task_{i}"
            }
            tasks.append(task)
            
        diversity_score = track_task_diversity(tasks, method="jensen_shannon")
        
        assert isinstance(diversity_score, float)
        assert diversity_score >= 0


class TestFactoryFunctions:
    """Test factory functions for creating utilities."""
    
    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary data directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create some mock data structure
            data_dir = Path(temp_dir)
            for i in range(5):
                class_dir = data_dir / f"class_{i}"
                class_dir.mkdir()
                for j in range(10):
                    (class_dir / f"sample_{j}.txt").touch()
            yield temp_dir
            
    def test_create_dataset_factory(self, temp_data_dir):
        """Test dataset creation factory function."""
        config = DatasetConfig(dataset_path=temp_data_dir, num_tasks=20)
        
        dataset = create_dataset("meta_learning", config)
        
        assert isinstance(dataset, MetaLearningDataset)
        assert dataset.config == config
        assert len(dataset) == 20
        
    def test_create_dataset_invalid_type(self):
        """Test dataset creation with invalid type."""
        config = DatasetConfig()
        
        with pytest.raises(ValueError, match="Unknown dataset type"):
            create_dataset("invalid_dataset_type", config)
            
    def test_create_metrics_evaluator_factory(self):
        """Test metrics evaluator creation factory function."""
        config = MetricsConfig()
        
        evaluator = create_metrics_evaluator("standard", config)
        
        assert isinstance(evaluator, EvaluationMetrics)
        assert evaluator.config == config
        
    def test_create_metrics_evaluator_invalid_type(self):
        """Test metrics evaluator creation with invalid type."""
        config = MetricsConfig()
        
        with pytest.raises(ValueError, match="Unknown metrics evaluator type"):
            create_metrics_evaluator("invalid_evaluator_type", config)
            
    def test_create_curriculum_scheduler_factory(self):
        """Test curriculum scheduler creation factory function."""
        config = CurriculumConfig()
        
        scheduler = create_curriculum_scheduler("difficulty_based", config)
        
        assert isinstance(scheduler, CurriculumLearning)
        assert scheduler.config == config
        
    def test_create_curriculum_scheduler_invalid_type(self):
        """Test curriculum scheduler creation with invalid type."""
        config = CurriculumConfig()
        
        with pytest.raises(ValueError, match="Unknown curriculum scheduler type"):
            create_curriculum_scheduler("invalid_scheduler_type", config)


class TestFixmeSolutions:
    """Test all research solutions are properly implemented."""
    
    @pytest.mark.fixme_solution
    def test_advanced_dataset_utilities_solution(self):
        """Test advanced dataset utilities FIXME solution."""
        config = DatasetConfig(n_way=4, k_shot=3, query_shots=12, num_tasks=100)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mock data structure
            data_dir = Path(temp_dir)
            for i in range(8):
                class_dir = data_dir / f"class_{i}"
                class_dir.mkdir()
                for j in range(15):
                    data_file = class_dir / f"sample_{j}.json"
                    with open(data_file, 'w') as f:
                        json.dump({
                            "features": np.random.randn(64).tolist(),
                            "label": i
                        }, f)
            
            config.dataset_path = temp_dir
            dataset = MetaLearningDataset(config)
            
            # Should work with advanced utilities
            episode = dataset[0]
            assert episode['support_x'].shape == (4, 3, 64)
            assert episode['query_x'].shape == (48, 64)
            
    @pytest.mark.fixme_solution
    def test_research_accurate_confidence_intervals_solution(self):
        """Test research-accurate confidence intervals FIXME solution."""
        # Test auto-selection of appropriate method based on sample size
        
        # Small sample (n < 30) - should use t-distribution
        small_sample = [0.82, 0.85, 0.80, 0.87, 0.83]
        ci_lower, ci_upper = compute_confidence_interval(
            small_sample, 
            method="auto",
            confidence_level=0.95
        )
        assert ci_lower <= ci_upper
        
        # Large sample (n >= 30) - can use normal distribution  
        large_sample = np.random.normal(0.85, 0.05, 100).tolist()
        ci_lower, ci_upper = compute_confidence_interval(
            large_sample,
            method="auto", 
            confidence_level=0.95
        )
        assert ci_lower <= ci_upper
        
    @pytest.mark.fixme_solution
    def test_multiple_difficulty_estimation_methods_solution(self):
        """Test multiple difficulty estimation methods FIXME solution."""
        task_data = {
            'features': torch.randn(40, 32),
            'labels': torch.randint(0, 5, (40,)),
            'n_way': 5,
            'k_shot': 2
        }
        
        methods = ["feature_variance", "label_entropy", "inter_class_distance", "intra_class_variance"]
        
        for method in methods:
            difficulty = estimate_difficulty(task_data, method=method)
            assert isinstance(difficulty, float)
            assert 0 <= difficulty <= 1
            
    @pytest.mark.fixme_solution
    def test_comprehensive_statistical_analysis_solution(self):
        """Test comprehensive statistical analysis FIXME solution."""
        config = StatsConfig(
            significance_level=0.05,
            multiple_comparisons="bonferroni",
            effect_size_measure="cohen_d"
        )
        analyzer = StatisticalAnalysis(config)
        
        # Create method comparison data
        methods = {
            'Method A': np.random.normal(0.85, 0.05, 30),
            'Method B': np.random.normal(0.80, 0.05, 30),
            'Method C': np.random.normal(0.75, 0.05, 30)
        }
        
        results = analyzer.compare_methods(methods)
        
        assert 'pairwise_comparisons' in results
        assert 'effect_sizes' in results  
        assert 'corrected_p_values' in results
        
        # Should have 3 pairwise comparisons for 3 methods
        assert len(results['pairwise_comparisons']) == 3
        
    @pytest.mark.fixme_solution
    def test_curriculum_learning_strategies_solution(self):
        """Test curriculum learning strategies FIXME solution."""
        strategies = ["difficulty_based", "diversity_based", "random", "fixed"]
        
        for strategy in strategies:
            config = CurriculumConfig(curriculum_strategy=strategy)
            scheduler = CurriculumLearning(config)
            
            # Test updating difficulty
            for i in range(10):
                performance = 0.8 + np.random.normal(0, 0.1)
                performance = max(0, min(1, performance))
                
                difficulty = scheduler.update_difficulty(performance, step=i)
                assert 0 <= difficulty <= 1
                
    @pytest.mark.fixme_solution 
    def test_task_diversity_tracking_solution(self):
        """Test task diversity tracking FIXME solution."""
        config = DiversityConfig(
            diversity_metric="jensen_shannon",
            track_feature_diversity=True,
            track_label_diversity=True
        )
        tracker = TaskDiversityTracker(config)
        
        # Add diverse tasks
        tasks = []
        for i in range(15):
            task = {
                'features': torch.randn(25, 48),
                'labels': torch.randint(0, 6, (25,)),
                'task_id': f"diversity_test_{i}"
            }
            tracker.add_task(task)
            
        diversity_metrics = tracker.get_current_diversity()
        
        assert 'feature_diversity' in diversity_metrics
        assert 'label_diversity' in diversity_metrics
        assert 'overall_diversity' in diversity_metrics


class TestUtilsIntegration:
    """Integration tests for utils module components."""
    
    @pytest.fixture
    def complete_utils_setup(self):
        """Create complete utils setup for integration testing."""
        # Create temporary data
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            for i in range(6):
                class_dir = data_dir / f"class_{i}"
                class_dir.mkdir()
                for j in range(20):
                    data_file = class_dir / f"sample_{j}.json"
                    with open(data_file, 'w') as f:
                        json.dump({
                            "features": np.random.randn(32).tolist(),
                            "label": i
                        }, f)
            
            # Create configurations
            dataset_config = DatasetConfig(dataset_path=temp_dir, n_way=3, k_shot=2, num_tasks=50)
            metrics_config = MetricsConfig(confidence_level=0.95, confidence_method="auto")
            curriculum_config = CurriculumConfig(curriculum_strategy="difficulty_based")
            
            # Create components
            dataset = create_dataset("meta_learning", dataset_config)
            evaluator = create_metrics_evaluator("standard", metrics_config)
            scheduler = create_curriculum_scheduler("difficulty_based", curriculum_config)
            
            yield dataset, evaluator, scheduler
            
    def test_complete_meta_learning_pipeline(self, complete_utils_setup):
        """Test complete meta-learning pipeline integration."""
        dataset, evaluator, scheduler = complete_utils_setup
        
        # Simulate training loop
        for epoch in range(10):
            # Get episode from dataset
            episode = dataset[epoch % len(dataset)]
            
            # Simulate model predictions (random for testing)
            n_queries = episode['query_x'].shape[0]
            n_classes = len(torch.unique(episode['query_y']))
            logits = torch.randn(n_queries, n_classes)
            
            # Evaluate episode
            results = evaluator.evaluate_episode(logits, episode['query_y'])
            
            # Update curriculum based on performance
            difficulty = scheduler.update_difficulty(results['accuracy'], step=epoch)
            
            # Check that all components work together
            assert 0 <= results['accuracy'] <= 1
            assert 0 <= difficulty <= 1
            assert len(results['confidence_interval']) == 2
            
    def test_dataset_evaluator_integration(self, complete_utils_setup):
        """Test integration between dataset and evaluator."""
        dataset, evaluator, _ = complete_utils_setup
        
        # Process multiple episodes
        accuracies = []
        for i in range(5):
            episode = dataset[i]
            
            # Create mock predictions (some correct, some wrong)
            query_y = episode['query_y']
            predictions = query_y.clone()
            # Make some predictions wrong
            wrong_indices = torch.randperm(len(query_y))[:len(query_y)//3]
            predictions[wrong_indices] = (predictions[wrong_indices] + 1) % len(torch.unique(query_y))
            
            accuracy = evaluator.compute_accuracy(predictions, query_y)
            accuracies.append(accuracy)
            
        # Compute overall confidence interval
        ci_lower, ci_upper = evaluator.compute_confidence_interval(accuracies)
        
        assert len(accuracies) == 5
        assert ci_lower <= ci_upper
        assert all(0 <= acc <= 1 for acc in accuracies)


@pytest.mark.property
class TestPropertyBasedUtils:
    """Property-based tests using Hypothesis for utils module."""
    
    @given(
        n_way=st.integers(min_value=2, max_value=10),
        k_shot=st.integers(min_value=1, max_value=5),
        query_shots=st.integers(min_value=5, max_value=20),
        confidence_level=st.floats(min_value=0.8, max_value=0.99)
    )
    @settings(max_examples=10, deadline=15000)
    def test_confidence_interval_properties(self, n_way, k_shot, query_shots, confidence_level):
        """Test properties of confidence interval computation."""
        # Generate random accuracies
        n_experiments = 10
        accuracies = []
        for _ in range(n_experiments):
            # Random accuracy between 0 and 1
            acc = np.random.uniform(0.0, 1.0)
            accuracies.append(acc)
            
        ci_lower, ci_upper = basic_confidence_interval(accuracies, confidence_level=confidence_level)
        
        # Properties that should always hold
        assert ci_lower <= ci_upper  # Lower bound <= upper bound
        assert 0 <= ci_lower <= 1    # Bounds within [0, 1] 
        assert 0 <= ci_upper <= 1
        assert isinstance(ci_lower, float)
        assert isinstance(ci_upper, float)
        
    @given(
        initial_difficulty=st.floats(min_value=0.1, max_value=0.8),
        max_difficulty=st.floats(min_value=0.5, max_value=1.0),
        difficulty_increment=st.floats(min_value=0.01, max_value=0.2)
    )
    @settings(max_examples=8, deadline=10000)
    def test_curriculum_properties(self, initial_difficulty, max_difficulty, difficulty_increment):
        """Test properties of curriculum learning."""
        # Ensure max_difficulty >= initial_difficulty
        if max_difficulty < initial_difficulty:
            max_difficulty = initial_difficulty + 0.1
            
        config = CurriculumConfig(
            initial_difficulty=initial_difficulty,
            max_difficulty=max_difficulty,
            difficulty_increment=difficulty_increment
        )
        scheduler = CurriculumLearning(config)
        
        # Simulate training with good performance
        for i in range(20):
            performance = 0.85 + np.random.normal(0, 0.05)  # Good performance
            performance = max(0, min(1, performance))
            
            difficulty = scheduler.update_difficulty(performance, step=i)
            
            # Properties that should hold
            assert initial_difficulty <= difficulty <= max_difficulty
            assert isinstance(difficulty, float)
            assert difficulty >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])