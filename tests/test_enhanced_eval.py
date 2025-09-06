"""
Tests for enhanced evaluation system with multiple seeds and cross-validation.

Tests cover:
- evaluate_multiple_seeds functionality
- MetaLearningCrossValidator
- comprehensive_evaluate integration
- Statistical aggregation and confidence intervals
- Learn2learn Accuracy class compatibility
- TorchMeta evaluation harness
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock, patch

from meta_learning.core.episode import Episode
from meta_learning.eval import (
    evaluate_multiple_seeds, MetaLearningCrossValidator, comprehensive_evaluate,
    Accuracy, TorchMetaEvaluationHarness, MetaLearningMetrics,
    EvaluationVisualizer
)


class TestEvaluateMultipleSeeds:
    """Test multi-seed evaluation functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        # Create mock episodes
        self.episodes = []
        for i in range(10):
            episode = Episode(
                support_x=torch.randn(25, 64),  # 5-way, 5-shot
                support_y=torch.repeat_interleave(torch.arange(5), 5),
                query_x=torch.randn(15, 64),    # 5-way, 3-query
                query_y=torch.repeat_interleave(torch.arange(5), 3)
            )
            self.episodes.append(episode)
        
        # Mock run_logits function
        self.mock_run_logits = Mock(return_value=torch.randn(15, 5))
    
    def test_evaluate_multiple_seeds_basic(self):
        """Test basic multi-seed evaluation."""
        results = evaluate_multiple_seeds(
            self.mock_run_logits, 
            self.episodes[:3], 
            seeds=[42, 43, 44],
            n_seeds=3
        )
        
        assert 'mean_accuracy' in results
        assert 'std_accuracy' in results
        assert 'confidence_interval' in results
        assert 'seed_results' in results
        assert len(results['seed_results']) == 3
        assert 0.0 <= results['mean_accuracy'] <= 1.0
        assert results['std_accuracy'] >= 0.0
    
    def test_evaluate_multiple_seeds_default_seeds(self):
        """Test multi-seed evaluation with default random seeds."""
        results = evaluate_multiple_seeds(
            self.mock_run_logits,
            self.episodes[:2],
            n_seeds=5
        )
        
        assert len(results['seed_results']) == 5
        # Should have different results for different seeds
        accuracies = [r['accuracy'] for r in results['seed_results']]
        assert len(set(accuracies)) > 1  # Should have some variation
    
    def test_evaluate_multiple_seeds_confidence_interval(self):
        """Test confidence interval calculation."""
        results = evaluate_multiple_seeds(
            self.mock_run_logits,
            self.episodes[:5],
            seeds=[1, 2, 3, 4, 5],
            confidence_level=0.95
        )
        
        ci = results['confidence_interval']
        assert len(ci) == 2
        assert ci[0] <= results['mean_accuracy'] <= ci[1]
        assert ci[1] > ci[0]  # Upper bound should be greater than lower
    
    def test_evaluate_multiple_seeds_single_seed(self):
        """Test behavior with single seed (should still work)."""
        results = evaluate_multiple_seeds(
            self.mock_run_logits,
            self.episodes[:2],
            seeds=[42],
            n_seeds=1
        )
        
        assert results['std_accuracy'] == 0.0  # No variation with single seed
        assert len(results['seed_results']) == 1


class TestMetaLearningCrossValidator:
    """Test cross-validation for meta-learning."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.episodes = []
        for i in range(20):  # Enough episodes for cross-validation
            episode = Episode(
                support_x=torch.randn(25, 64),
                support_y=torch.repeat_interleave(torch.arange(5), 5),
                query_x=torch.randn(15, 64),
                query_y=torch.repeat_interleave(torch.arange(5), 3)
            )
            self.episodes.append(episode)
    
    def test_cross_validator_initialization(self):
        """Test cross-validator initialization."""
        cv = MetaLearningCrossValidator(n_splits=5, shuffle=True, random_state=42)
        assert cv.n_splits == 5
        assert cv.shuffle == True
        assert cv.random_state == 42
    
    def test_cross_validate_basic(self):
        """Test basic cross-validation functionality."""
        cv = MetaLearningCrossValidator(n_splits=3, random_state=42)
        mock_run_logits = Mock(return_value=torch.randn(15, 5))
        
        results = cv.cross_validate(mock_run_logits, self.episodes)
        
        assert 'mean_accuracy' in results
        assert 'std_accuracy' in results
        assert 'fold_results' in results
        assert len(results['fold_results']) == 3
        assert 0.0 <= results['mean_accuracy'] <= 1.0
    
    def test_cross_validate_stratified(self):
        """Test stratified cross-validation."""
        cv = MetaLearningCrossValidator(n_splits=4, stratify=True, random_state=42)
        mock_run_logits = Mock(return_value=torch.randn(15, 5))
        
        results = cv.cross_validate(mock_run_logits, self.episodes)
        
        # Should complete without errors and have balanced folds
        assert len(results['fold_results']) == 4
        fold_sizes = [len(fold['test_episodes']) for fold in results['fold_results']]
        assert max(fold_sizes) - min(fold_sizes) <= 1  # Balanced splits
    
    def test_cross_validate_insufficient_episodes(self):
        """Test behavior with insufficient episodes."""
        cv = MetaLearningCrossValidator(n_splits=10)  # More splits than episodes
        mock_run_logits = Mock(return_value=torch.randn(15, 5))
        
        # Should handle gracefully or raise informative error
        with pytest.raises(ValueError, match="Cannot have more splits"):
            cv.cross_validate(mock_run_logits, self.episodes[:5])


class TestComprehensiveEvaluate:
    """Test comprehensive evaluation combining all features."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.episodes = []
        for i in range(15):
            episode = Episode(
                support_x=torch.randn(25, 64),
                support_y=torch.repeat_interleave(torch.arange(5), 5),
                query_x=torch.randn(15, 64),
                query_y=torch.repeat_interleave(torch.arange(5), 3)
            )
            self.episodes.append(episode)
        
        self.mock_run_logits = Mock(return_value=torch.randn(15, 5))
    
    def test_comprehensive_evaluate_all_features(self):
        """Test comprehensive evaluation with all features enabled."""
        results = comprehensive_evaluate(
            self.mock_run_logits,
            self.episodes,
            use_multiple_seeds=True,
            use_cross_validation=True,
            n_seeds=3,
            n_folds=3,
            confidence_level=0.95
        )
        
        # Should include results from both multi-seed and cross-validation
        assert 'multi_seed_results' in results
        assert 'cross_validation_results' in results
        assert 'combined_statistics' in results
        
        # Multi-seed results
        ms_results = results['multi_seed_results']
        assert 'mean_accuracy' in ms_results
        assert len(ms_results['seed_results']) == 3
        
        # Cross-validation results
        cv_results = results['cross_validation_results']
        assert 'mean_accuracy' in cv_results
        assert len(cv_results['fold_results']) == 3
    
    def test_comprehensive_evaluate_seeds_only(self):
        """Test comprehensive evaluation with only multi-seed."""
        results = comprehensive_evaluate(
            self.mock_run_logits,
            self.episodes,
            use_multiple_seeds=True,
            use_cross_validation=False,
            n_seeds=4
        )
        
        assert 'multi_seed_results' in results
        assert 'cross_validation_results' not in results
        assert len(results['multi_seed_results']['seed_results']) == 4
    
    def test_comprehensive_evaluate_cv_only(self):
        """Test comprehensive evaluation with only cross-validation."""
        results = comprehensive_evaluate(
            self.mock_run_logits,
            self.episodes,
            use_multiple_seeds=False,
            use_cross_validation=True,
            n_folds=5
        )
        
        assert 'cross_validation_results' in results
        assert 'multi_seed_results' not in results
        assert len(results['cross_validation_results']['fold_results']) == 5
    
    def test_comprehensive_evaluate_neither(self):
        """Test comprehensive evaluation with both features disabled."""
        results = comprehensive_evaluate(
            self.mock_run_logits,
            self.episodes,
            use_multiple_seeds=False,
            use_cross_validation=False
        )
        
        # Should fall back to basic evaluation
        assert 'accuracy' in results
        assert 'multi_seed_results' not in results
        assert 'cross_validation_results' not in results


class TestLearn2LearnAccuracy:
    """Test learn2learn compatibility with Accuracy class."""
    
    def test_accuracy_initialization(self):
        """Test Accuracy class initialization."""
        accuracy = Accuracy()
        assert hasattr(accuracy, 'reset')
        assert hasattr(accuracy, 'update')
        assert hasattr(accuracy, 'compute')
    
    def test_accuracy_computation(self):
        """Test accuracy computation."""
        accuracy = Accuracy()
        
        # Perfect predictions
        predictions = torch.tensor([0, 1, 2, 3, 4])
        targets = torch.tensor([0, 1, 2, 3, 4])
        
        accuracy.update(predictions, targets)
        result = accuracy.compute()
        
        assert result == 1.0
    
    def test_accuracy_partial_correct(self):
        """Test accuracy with partially correct predictions."""
        accuracy = Accuracy()
        
        predictions = torch.tensor([0, 1, 2, 2, 4])  # One wrong
        targets = torch.tensor([0, 1, 2, 3, 4])
        
        accuracy.update(predictions, targets)
        result = accuracy.compute()
        
        assert result == 0.8  # 4/5 correct
    
    def test_accuracy_reset(self):
        """Test accuracy reset functionality."""
        accuracy = Accuracy()
        
        predictions = torch.tensor([0, 1])
        targets = torch.tensor([0, 1])
        
        accuracy.update(predictions, targets)
        assert accuracy.compute() == 1.0
        
        accuracy.reset()
        # After reset, should start fresh
        accuracy.update(torch.tensor([0]), torch.tensor([1]))  # Wrong
        assert accuracy.compute() == 0.0


class TestTorchMetaEvaluationHarness:
    """Test TorchMeta evaluation harness."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.episodes = []
        for i in range(5):
            episode = Episode(
                support_x=torch.randn(10, 32),
                support_y=torch.repeat_interleave(torch.arange(2), 5),
                query_x=torch.randn(6, 32),
                query_y=torch.repeat_interleave(torch.arange(2), 3)
            )
            self.episodes.append(episode)
    
    def test_evaluation_harness_initialization(self):
        """Test evaluation harness initialization."""
        harness = TorchMetaEvaluationHarness(
            confidence_level=0.95,
            num_bootstrap_samples=1000
        )
        
        assert harness.confidence_level == 0.95
        assert harness.num_bootstrap_samples == 1000
    
    def test_evaluation_harness_evaluate(self):
        """Test evaluation harness evaluation."""
        harness = TorchMetaEvaluationHarness()
        mock_run_logits = Mock(return_value=torch.randn(6, 2))
        
        results = harness.evaluate_on_episodes(self.episodes, mock_run_logits)
        
        assert 'mean_accuracy' in results
        assert 'confidence_interval' in results
        assert 'episode_accuracies' in results
        assert len(results['episode_accuracies']) == len(self.episodes)
    
    def test_evaluation_harness_bootstrap_ci(self):
        """Test bootstrap confidence interval calculation."""
        harness = TorchMetaEvaluationHarness(num_bootstrap_samples=100)
        
        # Mock some episode accuracies
        accuracies = [0.8, 0.7, 0.9, 0.85, 0.75]
        ci = harness._compute_bootstrap_ci(accuracies, confidence_level=0.95)
        
        assert len(ci) == 2
        assert ci[0] <= np.mean(accuracies) <= ci[1]
        assert ci[1] > ci[0]


class TestMetaLearningMetrics:
    """Test meta-learning specific metrics."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.predictions = torch.randn(30, 5)  # 30 predictions, 5 classes
        self.targets = torch.randint(0, 5, (30,))
        
        # Prototype-based predictions (support set features)
        self.support_features = torch.randn(25, 64)  # 5-way, 5-shot
        self.query_features = torch.randn(15, 64)
    
    def test_metrics_initialization(self):
        """Test metrics initialization."""
        metrics = MetaLearningMetrics()
        assert hasattr(metrics, 'compute_prototype_quality')
        assert hasattr(metrics, 'compute_calibration_metrics')
        assert hasattr(metrics, 'compute_statistical_significance')
    
    def test_prototype_quality_metrics(self):
        """Test prototype quality computation."""
        metrics = MetaLearningMetrics()
        
        results = metrics.compute_prototype_quality(
            self.support_features,
            torch.repeat_interleave(torch.arange(5), 5)
        )
        
        assert 'separability' in results
        assert 'compactness' in results
        assert 'silhouette_score' in results
        assert results['separability'] >= 0
        assert results['compactness'] >= 0
    
    def test_calibration_metrics(self):
        """Test calibration metrics computation."""
        metrics = MetaLearningMetrics()
        
        # Convert logits to probabilities
        probabilities = torch.softmax(self.predictions, dim=1)
        
        results = metrics.compute_calibration_metrics(probabilities, self.targets)
        
        assert 'expected_calibration_error' in results
        assert 'reliability_diagram' in results
        assert 0.0 <= results['expected_calibration_error'] <= 1.0
    
    def test_statistical_significance(self):
        """Test statistical significance testing."""
        metrics = MetaLearningMetrics()
        
        # Two sets of results for comparison
        results_a = [0.8, 0.7, 0.9, 0.85, 0.75, 0.82, 0.78]
        results_b = [0.6, 0.5, 0.7, 0.65, 0.55, 0.62, 0.58]
        
        stats = metrics.compute_statistical_significance(results_a, results_b)
        
        assert 'p_value' in stats
        assert 'effect_size' in stats
        assert 'significant' in stats
        assert 0.0 <= stats['p_value'] <= 1.0
        assert isinstance(stats['significant'], bool)


class TestEvaluationVisualizer:
    """Test evaluation visualization tools."""
    
    def test_visualizer_initialization(self):
        """Test visualizer initialization."""
        viz = EvaluationVisualizer()
        assert hasattr(viz, 'plot_learning_curves')
        assert hasattr(viz, 'plot_confidence_intervals')
        assert hasattr(viz, 'plot_performance_distribution')
    
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.show')
    def test_plot_learning_curves(self, mock_show, mock_figure):
        """Test learning curve plotting."""
        viz = EvaluationVisualizer()
        
        # Mock learning data
        episodes = list(range(1, 101))
        accuracies = [0.5 + 0.3 * np.exp(-i/20) for i in episodes]
        
        # Should not raise an exception
        viz.plot_learning_curves(episodes, accuracies, title="Test Learning Curve")
        
        assert mock_figure.called
    
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.show')
    def test_plot_confidence_intervals(self, mock_show, mock_figure):
        """Test confidence interval plotting."""
        viz = EvaluationVisualizer()
        
        # Mock CI data
        methods = ['Method A', 'Method B', 'Method C']
        means = [0.8, 0.7, 0.85]
        cis = [(0.75, 0.85), (0.65, 0.75), (0.8, 0.9)]
        
        # Should not raise an exception
        viz.plot_confidence_intervals(methods, means, cis, title="Method Comparison")
        
        assert mock_figure.called
    
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.show')
    def test_plot_performance_distribution(self, mock_show, mock_figure):
        """Test performance distribution plotting."""
        viz = EvaluationVisualizer()
        
        # Mock performance data
        performances = np.random.beta(8, 2, 100) * 0.4 + 0.6  # Realistic accuracy distribution
        
        # Should not raise an exception
        viz.plot_performance_distribution(performances, title="Accuracy Distribution")
        
        assert mock_figure.called


# Integration tests
class TestEvaluationIntegration:
    """Test integration between evaluation components."""
    
    def setup_method(self):
        """Setup integration test fixtures."""
        self.episodes = []
        for i in range(12):
            episode = Episode(
                support_x=torch.randn(20, 32),
                support_y=torch.repeat_interleave(torch.arange(4), 5),
                query_x=torch.randn(12, 32),
                query_y=torch.repeat_interleave(torch.arange(4), 3)
            )
            self.episodes.append(episode)
        
        # Simple mock classifier that returns reasonable predictions
        def mock_run_logits(episode):
            batch_size = episode.query_x.size(0)
            n_classes = len(torch.unique(episode.support_y))
            logits = torch.randn(batch_size, n_classes)
            # Add some bias to make predictions slightly better than random
            logits[:, 0] += 0.5
            return logits
        
        self.mock_run_logits = mock_run_logits
    
    def test_full_evaluation_pipeline(self):
        """Test complete evaluation pipeline integration."""
        # Test all components working together
        
        # 1. Multi-seed evaluation
        seed_results = evaluate_multiple_seeds(
            self.mock_run_logits,
            self.episodes[:8],
            n_seeds=3,
            seeds=[42, 43, 44]
        )
        
        assert 'mean_accuracy' in seed_results
        
        # 2. Cross-validation
        cv = MetaLearningCrossValidator(n_splits=3, random_state=42)
        cv_results = cv.cross_validate(self.mock_run_logits, self.episodes[:9])
        
        assert 'mean_accuracy' in cv_results
        
        # 3. Comprehensive evaluation
        comp_results = comprehensive_evaluate(
            self.mock_run_logits,
            self.episodes,
            use_multiple_seeds=True,
            use_cross_validation=True,
            n_seeds=2,
            n_folds=3
        )
        
        assert 'multi_seed_results' in comp_results
        assert 'cross_validation_results' in comp_results
        
        # 4. Professional harness
        harness = TorchMetaEvaluationHarness(confidence_level=0.90)
        harness_results = harness.evaluate_on_episodes(
            self.episodes[:6], 
            self.mock_run_logits
        )
        
        assert 'confidence_interval' in harness_results
        
        # All should produce reasonable accuracy values
        for result in [seed_results, cv_results, harness_results]:
            accuracy = result.get('mean_accuracy', result.get('accuracy', 0))
            assert 0.0 <= accuracy <= 1.0
    
    def test_learn2learn_compatibility(self):
        """Test compatibility with learn2learn evaluation patterns."""
        # Test that our Accuracy class works with learn2learn-style evaluation
        accuracy_metric = Accuracy()
        
        total_accuracy = 0
        num_episodes = 0
        
        for episode in self.episodes[:5]:
            predictions = self.mock_run_logits(episode)
            predicted_labels = predictions.argmax(dim=1)
            
            accuracy_metric.update(predicted_labels, episode.query_y)
            num_episodes += 1
        
        final_accuracy = accuracy_metric.compute()
        assert 0.0 <= final_accuracy <= 1.0
        
        # Reset should work
        accuracy_metric.reset()
        assert accuracy_metric.total == 0
        assert accuracy_metric.count == 0


if __name__ == "__main__":
    pytest.main([__file__])