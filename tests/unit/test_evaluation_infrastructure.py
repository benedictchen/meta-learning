"""
Test suite for evaluation infrastructure.

Tests the comprehensive evaluation system implemented in src/meta_learning/eval.py
including statistical testing, uncertainty quantification, calibration analysis,
cross-validation, multi-seed evaluation, and visualization tools.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

from meta_learning.core.episode import Episode
from meta_learning.eval import (
    evaluate, _get_t_critical, MetricType, EvaluationConfig, 
    Accuracy, UncertaintyEvaluator, StatisticalTestSuite, LearnabilityAnalyzer,
    MetaLearningMetrics, EvaluationVisualizer, MetaLearningEvaluator,
    TorchMetaEvaluationHarness, evaluate_multiple_seeds, MetaLearningCrossValidator,
    comprehensive_evaluate
)


class TestBasicEvaluate:
    """Test the basic evaluate function."""
    
    def test_evaluate_basic_functionality(self):
        """Test basic evaluation functionality."""
        # Create mock episodes
        episodes = []
        for _ in range(10):
            support_x = torch.randn(25, 64)  # 5-way 5-shot
            support_y = torch.repeat_interleave(torch.arange(5), 5)
            query_x = torch.randn(15, 64)
            query_y = torch.repeat_interleave(torch.arange(5), 3)
            episodes.append(Episode(support_x, support_y, query_x, query_y))
        
        # Mock run_logits function
        def mock_run_logits(episode):
            # Return reasonable logits that result in ~60% accuracy
            logits = torch.randn(episode.query_x.shape[0], 5)
            # Add some signal based on true labels for realistic accuracy
            for i, true_label in enumerate(episode.query_y):
                logits[i, true_label] += 1.0  # Boost correct class
            return logits
        
        # Run evaluation
        results = evaluate(mock_run_logits, episodes)
        
        # Check results structure
        assert isinstance(results, dict)
        assert 'episodes' in results
        assert 'mean_acc' in results
        assert 'ci95' in results
        assert 'std_err' in results
        assert 'elapsed_s' in results
        
        # Check values
        assert results['episodes'] == 10
        assert 0.0 <= results['mean_acc'] <= 1.0
        assert results['ci95'] >= 0.0
        assert results['std_err'] >= 0.0
        assert results['elapsed_s'] > 0.0
        
    def test_evaluate_with_perfect_accuracy(self):
        """Test evaluation with perfect predictions."""
        episodes = []
        for _ in range(5):
            support_x = torch.randn(10, 32)
            support_y = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
            query_x = torch.randn(5, 32)
            query_y = torch.tensor([0, 1, 2, 3, 4])
            episodes.append(Episode(support_x, support_y, query_x, query_y))
        
        def perfect_run_logits(episode):
            # Return perfect predictions
            logits = torch.zeros(episode.query_x.shape[0], 5)
            for i, true_label in enumerate(episode.query_y):
                logits[i, true_label] = 10.0  # Very high confidence for correct class
            return logits
        
        results = evaluate(perfect_run_logits, episodes)
        
        # Should achieve perfect accuracy
        assert abs(results['mean_acc'] - 1.0) < 1e-6
        
    def test_evaluate_with_random_predictions(self):
        """Test evaluation with random predictions."""
        episodes = []
        for _ in range(20):
            support_x = torch.randn(15, 16)
            support_y = torch.repeat_interleave(torch.arange(3), 5)
            query_x = torch.randn(9, 16)
            query_y = torch.repeat_interleave(torch.arange(3), 3)
            episodes.append(Episode(support_x, support_y, query_x, query_y))
        
        def random_run_logits(episode):
            return torch.randn(episode.query_x.shape[0], 3)
        
        results = evaluate(random_run_logits, episodes)
        
        # Random accuracy should be around 33% for 3-way classification
        assert 0.1 <= results['mean_acc'] <= 0.6  # Allow some variance
        
    def test_evaluate_with_output_directory(self, tmp_path):
        """Test evaluation with output directory saving."""
        episodes = []
        for _ in range(3):
            episode = Episode(
                torch.randn(6, 8), torch.tensor([0, 0, 1, 1, 2, 2]),
                torch.randn(3, 8), torch.tensor([0, 1, 2])
            )
            episodes.append(episode)
        
        def simple_run_logits(episode):
            return torch.randn(episode.query_x.shape[0], 3)
        
        outdir = str(tmp_path / "eval_results")
        results = evaluate(simple_run_logits, episodes, outdir=outdir, dump_preds=True)
        
        # Check that files were created
        import os
        assert os.path.exists(os.path.join(outdir, "metrics.json"))
        assert os.path.exists(os.path.join(outdir, "preds.jsonl"))


class TestTCritical:
    """Test t-critical value computation."""
    
    def test_get_t_critical_known_values(self):
        """Test t-critical values for known degrees of freedom."""
        # Test some known values from t-table
        assert abs(_get_t_critical(1) - 12.71) < 0.01
        assert abs(_get_t_critical(4) - 2.78) < 0.01
        assert abs(_get_t_critical(9) - 2.26) < 0.01
        
    def test_get_t_critical_large_df(self):
        """Test t-critical for large degrees of freedom (should use normal approx)."""
        assert abs(_get_t_critical(50) - 1.96) < 0.01
        assert abs(_get_t_critical(100) - 1.96) < 0.01
        
    def test_get_t_critical_interpolation(self):
        """Test interpolation for intermediate values."""
        # Should be between known values
        val_6 = _get_t_critical(6)
        val_8 = _get_t_critical(8)
        val_7 = _get_t_critical(7)
        
        assert val_8 < val_7 < val_6  # t-values decrease with increasing df


class TestEvaluationConfig:
    """Test evaluation configuration."""
    
    def test_evaluation_config_defaults(self):
        """Test default configuration values."""
        config = EvaluationConfig([MetricType.ACCURACY])
        
        assert config.metrics == [MetricType.ACCURACY]
        assert config.confidence_level == 0.95
        assert config.bootstrap_samples == 1000
        assert config.statistical_tests == ["t_test", "wilcoxon"]
        assert config.visualization == False
        assert config.save_predictions == False
        
    def test_evaluation_config_custom(self):
        """Test custom configuration."""
        config = EvaluationConfig(
            metrics=[MetricType.ACCURACY, MetricType.F1_SCORE],
            confidence_level=0.99,
            bootstrap_samples=5000,
            statistical_tests=["t_test"],
            visualization=True,
            save_predictions=True
        )
        
        assert len(config.metrics) == 2
        assert config.confidence_level == 0.99
        assert config.bootstrap_samples == 5000
        assert config.statistical_tests == ["t_test"]
        assert config.visualization == True
        assert config.save_predictions == True


class TestAccuracy:
    """Test the Accuracy class."""
    
    def test_accuracy_compute_basic(self):
        """Test basic accuracy computation."""
        acc = Accuracy(confidence_level=0.95)
        
        predictions = torch.tensor([0, 1, 2, 0, 1])
        targets = torch.tensor([0, 1, 1, 0, 1])  # 4/5 correct = 80%
        
        result = acc.compute(predictions, targets)
        
        assert abs(result['accuracy'] - 0.8) < 1e-6
        assert result['n_samples'] == 5
        assert result['std_error'] > 0.0
        assert result['confidence_interval'] > 0.0
        
    def test_accuracy_perfect_predictions(self):
        """Test accuracy with perfect predictions."""
        acc = Accuracy()
        
        predictions = torch.tensor([1, 2, 3, 4])
        targets = torch.tensor([1, 2, 3, 4])
        
        result = acc.compute(predictions, targets)
        
        assert result['accuracy'] == 1.0
        assert result['std_error'] == 0.0
        
    def test_accuracy_single_sample(self):
        """Test accuracy with single sample."""
        acc = Accuracy()
        
        predictions = torch.tensor([1])
        targets = torch.tensor([1])
        
        result = acc.compute(predictions, targets)
        
        assert result['accuracy'] == 1.0
        assert result['std_error'] == 0.0
        assert result['confidence_interval'] == 0.0
        
    def test_accuracy_bootstrap_ci(self):
        """Test bootstrap confidence intervals."""
        acc = Accuracy()
        
        # Create reproducible predictions
        torch.manual_seed(42)
        predictions = torch.randint(0, 3, (100,))
        targets = torch.randint(0, 3, (100,))
        
        lower, upper = acc.bootstrap_confidence_interval(
            predictions, targets, n_bootstrap=100
        )
        
        # CI should be reasonable
        assert 0.0 <= lower <= upper <= 1.0
        assert upper - lower > 0.0  # Should have non-zero width


class TestUncertaintyEvaluator:
    """Test uncertainty evaluation functionality."""
    
    def test_compute_entropy(self):
        """Test entropy computation."""
        evaluator = UncertaintyEvaluator()
        
        # High confidence predictions (low entropy)
        confident_logits = torch.tensor([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
        entropy_confident = evaluator.compute_entropy(confident_logits)
        
        # Uncertain predictions (high entropy)
        uncertain_logits = torch.tensor([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]])
        entropy_uncertain = evaluator.compute_entropy(uncertain_logits)
        
        # Uncertain predictions should have higher entropy
        assert torch.all(entropy_uncertain > entropy_confident)
        
    def test_compute_confidence(self):
        """Test confidence computation."""
        evaluator = UncertaintyEvaluator()
        
        logits = torch.tensor([[5.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
        confidence = evaluator.compute_confidence(logits)
        
        assert confidence.shape == (2,)
        assert torch.all(confidence >= 0.0) and torch.all(confidence <= 1.0)
        assert confidence[0] > confidence[1]  # First should be more confident
        
    def test_calibration_analysis(self):
        """Test calibration analysis."""
        evaluator = UncertaintyEvaluator()
        
        # Create test data
        logits = torch.randn(100, 5)
        targets = torch.randint(0, 5, (100,))
        
        result = evaluator.calibration_analysis(logits, targets)
        
        assert 'ece' in result
        assert 'bin_accuracies' in result
        assert 'bin_confidences' in result
        assert 'bin_counts' in result
        
        assert result['ece'] >= 0.0
        assert len(result['bin_accuracies']) == 10  # default n_bins
        assert len(result['bin_confidences']) == 10
        assert len(result['bin_counts']) == 10


class TestStatisticalTestSuite:
    """Test statistical testing functionality."""
    
    def test_paired_t_test_identical_scores(self):
        """Test t-test with identical scores."""
        stats = StatisticalTestSuite()
        
        scores1 = [0.8, 0.7, 0.9, 0.6, 0.8]
        scores2 = [0.8, 0.7, 0.9, 0.6, 0.8]
        
        result = stats.paired_t_test(scores1, scores2)
        
        assert result['t_statistic'] == 0.0
        assert result['p_value'] == 1.0
        assert result['significant'] == False
        assert result['mean_difference'] == 0.0
        
    def test_paired_t_test_different_scores(self):
        """Test t-test with different scores."""
        stats = StatisticalTestSuite()
        
        scores1 = [0.9, 0.8, 0.9, 0.8, 0.9]  # Higher scores
        scores2 = [0.6, 0.5, 0.7, 0.5, 0.6]  # Lower scores
        
        result = stats.paired_t_test(scores1, scores2)
        
        assert result['t_statistic'] > 0.0  # scores1 > scores2
        assert result['mean_difference'] > 0.0
        # Can't guarantee significance with small sample, but should have reasonable p-value
        
    def test_paired_t_test_invalid_input(self):
        """Test t-test with invalid input."""
        stats = StatisticalTestSuite()
        
        with pytest.raises(ValueError):
            stats.paired_t_test([0.8, 0.7], [0.6])  # Different lengths
            
    def test_paired_t_test_small_sample(self):
        """Test t-test with very small sample."""
        stats = StatisticalTestSuite()
        
        result = stats.paired_t_test([0.8], [0.6])
        
        assert result['t_statistic'] == 0.0
        assert result['p_value'] == 1.0
        assert result['significant'] == False
        
    def test_wilcoxon_signed_rank_test(self):
        """Test Wilcoxon signed-rank test."""
        stats = StatisticalTestSuite()
        
        scores1 = [0.8, 0.7, 0.9, 0.6, 0.8]
        scores2 = [0.6, 0.5, 0.7, 0.4, 0.6]
        
        result = stats.wilcoxon_signed_rank_test(scores1, scores2)
        
        assert 'w_statistic' in result
        assert 'p_value' in result
        assert 'significant' in result
        assert result['w_statistic'] >= 0.0
        assert 0.0 <= result['p_value'] <= 1.0


class TestLearnabilityAnalyzer:
    """Test task learnability and difficulty analysis."""
    
    def test_compute_task_difficulty(self):
        """Test task difficulty computation."""
        analyzer = LearnabilityAnalyzer()
        
        # Create a simple episode
        support_x = torch.randn(15, 32)
        support_y = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
        query_x = torch.randn(9, 32)
        query_y = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])
        
        episode = Episode(support_x, support_y, query_x, query_y)
        
        result = analyzer.compute_task_difficulty(episode)
        
        assert 'class_balance' in result
        assert 'avg_feature_distance' in result
        assert 'intra_class_variance' in result
        assert 'inter_class_separation' in result
        assert 'difficulty_score' in result
        
        assert 0.0 <= result['class_balance'] <= 1.0
        assert result['avg_feature_distance'] >= 0.0
        assert result['intra_class_variance'] >= 0.0
        assert result['inter_class_separation'] >= 0.0
        
    def test_analyze_few_shot_complexity(self):
        """Test complexity analysis across multiple episodes."""
        analyzer = LearnabilityAnalyzer()
        
        episodes = []
        for _ in range(5):
            support_x = torch.randn(10, 16)
            support_y = torch.repeat_interleave(torch.arange(2), 5)
            query_x = torch.randn(4, 16)
            query_y = torch.repeat_interleave(torch.arange(2), 2)
            episodes.append(Episode(support_x, support_y, query_x, query_y))
        
        result = analyzer.analyze_few_shot_complexity(episodes)
        
        assert 'mean_difficulty' in result
        assert 'std_difficulty' in result
        assert 'mean_class_balance' in result
        assert 'difficulty_distribution' in result
        
        assert len(result['difficulty_distribution']) == 5


class TestMetaLearningEvaluator:
    """Test the main meta-learning evaluator."""
    
    def test_evaluator_initialization(self):
        """Test evaluator initialization with different configs."""
        # Default config
        evaluator1 = MetaLearningEvaluator()
        assert evaluator1.config is not None
        
        # Custom config
        config = EvaluationConfig([MetricType.ACCURACY, MetricType.LOSS])
        evaluator2 = MetaLearningEvaluator(config)
        assert evaluator2.config == config
        
    def test_evaluate_episodes(self):
        """Test episode evaluation."""
        evaluator = MetaLearningEvaluator()
        
        episodes = []
        for _ in range(5):
            support_x = torch.randn(12, 24)
            support_y = torch.repeat_interleave(torch.arange(3), 4)
            query_x = torch.randn(6, 24)
            query_y = torch.repeat_interleave(torch.arange(3), 2)
            episodes.append(Episode(support_x, support_y, query_x, query_y))
        
        def mock_run_logits(episode):
            return torch.randn(episode.query_x.shape[0], 3)
        
        results = evaluator.evaluate_episodes(mock_run_logits, episodes)
        
        assert 'aggregate' in results
        assert 'episodes' in results
        
        aggregate = results['aggregate']
        assert 'elapsed_time' in aggregate
        assert 'n_episodes' in aggregate
        assert aggregate['n_episodes'] == 5


class TestEvaluationVisualizer:
    """Test evaluation visualization tools."""
    
    def test_visualizer_initialization(self):
        """Test visualizer initialization."""
        viz1 = EvaluationVisualizer()
        assert viz1.figures == {}
        
    def test_plot_accuracy_distribution(self, capsys):
        """Test accuracy distribution plotting (console output)."""
        viz = EvaluationVisualizer()
        
        accuracies = [0.8, 0.7, 0.9, 0.6, 0.8, 0.7, 0.9]
        
        # This should print to console (no matplotlib)
        viz.plot_accuracy_distribution(accuracies)
        
        captured = capsys.readouterr()
        assert "Mean=" in captured.out
        assert "Std=" in captured.out
        
    def test_plot_calibration_curve(self, capsys):
        """Test calibration curve plotting."""
        viz = EvaluationVisualizer()
        
        calibration_results = {"ece": 0.15}
        
        viz.plot_calibration_curve(calibration_results)
        
        captured = capsys.readouterr()
        assert "Expected Calibration Error" in captured.out
        assert "0.150" in captured.out


class TestMultiSeedEvaluation:
    """Test multi-seed evaluation functionality."""
    
    @patch('meta_learning.eval.seed_all')
    def test_evaluate_multiple_seeds(self, mock_seed_all):
        """Test multi-seed evaluation."""
        # Mock episodes
        episodes = []
        for _ in range(3):
            episode = Episode(
                torch.randn(6, 8), torch.tensor([0, 0, 1, 1, 2, 2]),
                torch.randn(3, 8), torch.tensor([0, 1, 2])
            )
            episodes.append(episode)
        
        def mock_run_logits(episode):
            return torch.randn(episode.query_x.shape[0], 3)
        
        seeds = [42, 123, 456]
        
        # Mock the evaluate function to return consistent results
        with patch('meta_learning.eval.evaluate') as mock_evaluate:
            mock_evaluate.return_value = {
                'mean_acc': 0.7, 'ci95': 0.05, 'std_err': 0.02,
                'episodes': 3, 'elapsed_s': 1.0
            }
            
            results = evaluate_multiple_seeds(
                mock_run_logits, episodes, seeds=seeds, n_seeds=3
            )
            
            assert 'seeds_used' in results
            assert 'n_seeds' in results
            assert 'per_seed_results' in results
            assert 'mean_acc' in results
            assert 'consensus_ci95' in results
            
            assert results['n_seeds'] == 3
            assert len(results['per_seed_results']) == 3


class TestCrossValidation:
    """Test cross-validation functionality."""
    
    def test_cross_validator_initialization(self):
        """Test cross-validator initialization."""
        cv = MetaLearningCrossValidator(n_folds=3, stratified=False)
        
        assert cv.n_folds == 3
        assert cv.stratified == False
        
    def test_split_episodes(self):
        """Test episode splitting."""
        cv = MetaLearningCrossValidator(n_folds=3, stratified=False, random_state=42)
        
        episodes = []
        for _ in range(9):  # Divisible by 3 folds
            episode = Episode(
                torch.randn(4, 6), torch.tensor([0, 0, 1, 1]),
                torch.randn(2, 6), torch.tensor([0, 1])
            )
            episodes.append(episode)
        
        splits = cv.split_episodes(episodes)
        
        assert len(splits) == 3
        
        # Check that each split has train and val episodes
        for train_eps, val_eps in splits:
            assert len(train_eps) == 6  # 2/3 for training
            assert len(val_eps) == 3   # 1/3 for validation
            
    def test_cross_validate(self):
        """Test cross-validation execution."""
        cv = MetaLearningCrossValidator(n_folds=2, random_state=42)
        
        episodes = []
        for _ in range(4):
            episode = Episode(
                torch.randn(6, 8), torch.tensor([0, 0, 1, 1, 2, 2]),
                torch.randn(3, 8), torch.tensor([0, 1, 2])
            )
            episodes.append(episode)
        
        def mock_run_logits(episode):
            return torch.randn(episode.query_x.shape[0], 3)
        
        # Mock the evaluate function
        with patch('meta_learning.eval.evaluate') as mock_evaluate:
            mock_evaluate.return_value = {
                'mean_acc': 0.6, 'ci95': 0.1, 'std_err': 0.05,
                'episodes': 2, 'elapsed_s': 0.5
            }
            
            results = cv.cross_validate(mock_run_logits, episodes)
            
            assert 'n_folds' in results
            assert 'cv_mean_acc' in results
            assert 'cv_std_acc' in results
            assert 'fold_results' in results
            
            assert results['n_folds'] == 2
            assert len(results['fold_results']) == 2


class TestComprehensiveEvaluate:
    """Test comprehensive evaluation functionality."""
    
    def test_comprehensive_evaluate_basic(self):
        """Test comprehensive evaluation with basic options."""
        # Create a simple mock model
        model = Mock()
        model.eval = Mock()
        
        episodes = []
        for _ in range(3):
            episode = Episode(
                torch.randn(6, 4), torch.tensor([0, 0, 1, 1, 2, 2]),
                torch.randn(3, 4), torch.tensor([0, 1, 2])
            )
            episodes.append(episode)
        
        # Mock the evaluator
        with patch('meta_learning.eval.MetaLearningEvaluator') as mock_evaluator_class:
            mock_evaluator = Mock()
            mock_evaluator.evaluate_episodes.return_value = {
                'aggregate': {'mean_accuracy': 0.7},
                'episodes': []
            }
            mock_evaluator_class.return_value = mock_evaluator
            
            results = comprehensive_evaluate(model, episodes)
            
            assert 'basic_evaluation' in results
            assert results['basic_evaluation']['aggregate']['mean_accuracy'] == 0.7
            
    def test_comprehensive_evaluate_with_seeds(self):
        """Test comprehensive evaluation with multi-seed option."""
        model = Mock()
        model.eval = Mock()
        
        episodes = [Mock() for _ in range(3)]
        seeds = [42, 123]
        
        with patch('meta_learning.eval.MetaLearningEvaluator') as mock_evaluator_class, \
             patch('meta_learning.eval.evaluate_multiple_seeds') as mock_multi_seed:
            
            # Setup mocks
            mock_evaluator = Mock()
            mock_evaluator.evaluate_episodes.return_value = {'aggregate': {'mean_accuracy': 0.7}, 'episodes': []}
            mock_evaluator_class.return_value = mock_evaluator
            
            mock_multi_seed.return_value = {'mean_accuracy': 0.75, 'n_seeds': 2}
            
            results = comprehensive_evaluate(model, episodes, seeds=seeds)
            
            assert 'basic_evaluation' in results
            assert 'multi_seed_evaluation' in results
            assert results['multi_seed_evaluation']['n_seeds'] == 2


class TestIntegration:
    """Integration tests for evaluation infrastructure."""
    
    def test_end_to_end_evaluation_pipeline(self):
        """Test complete evaluation pipeline."""
        # Create realistic episodes
        episodes = []
        for _ in range(10):
            support_x = torch.randn(15, 64)  # 5-way 3-shot
            support_y = torch.repeat_interleave(torch.arange(5), 3)
            query_x = torch.randn(25, 64)    # 5 queries per class
            query_y = torch.repeat_interleave(torch.arange(5), 5)
            episodes.append(Episode(support_x, support_y, query_x, query_y))
        
        # Create a simple but realistic model function
        def realistic_run_logits(episode):
            # Compute simple prototypical network predictions
            support_features = episode.support_x
            query_features = episode.query_x
            
            # Compute prototypes
            unique_labels = torch.unique(episode.support_y)
            prototypes = []
            for label in unique_labels:
                mask = episode.support_y == label
                prototype = support_features[mask].mean(dim=0)
                prototypes.append(prototype)
            prototypes = torch.stack(prototypes)
            
            # Compute distances and logits
            distances = torch.cdist(query_features, prototypes)
            logits = -distances  # Negative distance as logits
            
            return logits
        
        # Run basic evaluation
        basic_results = evaluate(realistic_run_logits, episodes)
        
        assert basic_results['mean_acc'] > 0.15  # Should be better than random
        assert basic_results['episodes'] == 10
        
        # Run with evaluator
        config = EvaluationConfig([MetricType.ACCURACY], visualization=False)
        evaluator = MetaLearningEvaluator(config)
        
        detailed_results = evaluator.evaluate_episodes(realistic_run_logits, episodes)
        
        assert 'aggregate' in detailed_results
        assert detailed_results['aggregate']['n_episodes'] == 10
        
    def test_statistical_significance_detection(self):
        """Test that statistical tests can detect significant differences."""
        stats = StatisticalTestSuite()
        
        # Create clearly different score distributions
        high_scores = [0.9, 0.85, 0.92, 0.88, 0.91, 0.87, 0.90]
        low_scores = [0.4, 0.35, 0.42, 0.38, 0.41, 0.37, 0.40]
        
        t_test_result = stats.paired_t_test(high_scores, low_scores)
        
        # Should detect significant difference
        assert t_test_result['significant'] == True
        assert t_test_result['mean_difference'] > 0.4
        assert t_test_result['p_value'] < 0.05
        
    def test_uncertainty_calibration_relationship(self):
        """Test relationship between uncertainty and calibration."""
        evaluator = UncertaintyEvaluator()
        
        # Create well-calibrated predictions
        torch.manual_seed(42)
        
        # High confidence, high accuracy predictions
        confident_logits = torch.tensor([
            [5.0, 0.0, 0.0],  # Very confident in class 0
            [0.0, 5.0, 0.0],  # Very confident in class 1
            [0.0, 0.0, 5.0],  # Very confident in class 2
        ])
        confident_targets = torch.tensor([0, 1, 2])  # All correct
        
        # Low confidence, mixed accuracy predictions  
        uncertain_logits = torch.tensor([
            [1.1, 1.0, 0.9],  # Slightly prefers class 0
            [0.9, 1.1, 1.0],  # Slightly prefers class 1
            [1.0, 0.9, 1.1],  # Slightly prefers class 2
        ])
        uncertain_targets = torch.tensor([0, 1, 2])  # All correct but uncertain
        
        # Compute entropy (uncertainty)
        confident_entropy = evaluator.compute_entropy(confident_logits)
        uncertain_entropy = evaluator.compute_entropy(uncertain_logits)
        
        # Uncertain predictions should have higher entropy
        assert torch.all(uncertain_entropy > confident_entropy)
        
        # Compute calibration
        confident_predictions = confident_logits.argmax(-1)
        uncertain_predictions = uncertain_logits.argmax(-1)
        
        confident_cal = evaluator.calibration_analysis(confident_logits, confident_targets)
        uncertain_cal = evaluator.calibration_analysis(uncertain_logits, uncertain_targets)
        
        # Both should have reasonable calibration (perfect predictions in this case)
        assert confident_cal['ece'] >= 0.0
        assert uncertain_cal['ece'] >= 0.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])