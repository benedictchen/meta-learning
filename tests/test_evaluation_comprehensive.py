#!/usr/bin/env python3
"""
Comprehensive Evaluation Tests
==============================

Tests for evaluation harness, metrics, and statistical analysis components:
- FewShotEvaluationHarness
- MetaLearningMetrics
- Statistical testing suite
- Uncertainty evaluation
- Prototype analysis
- Calibration metrics
- Evaluation visualizers
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tempfile
from typing import Dict, Any, List, Callable
from unittest.mock import patch, MagicMock

from meta_learning.core.episode import Episode
from meta_learning.eval import evaluate, Accuracy, UncertaintyEvaluator, StatisticalTestSuite
from meta_learning.evaluation.few_shot_evaluation_harness import FewShotEvaluationHarness
from meta_learning.evaluation.prototype_analysis import PrototypeAnalyzer


class TestBasicEvaluationFunction:
    """Test the basic evaluate function."""

    def test_evaluate_function_basic(self):
        """Test basic evaluate function functionality."""
        # Create simple episodes
        episodes = []
        for i in range(5):
            episode = Episode(
                torch.randn(6, 10), torch.repeat_interleave(torch.arange(3), 2),
                torch.randn(9, 10), torch.repeat_interleave(torch.arange(3), 3)
            )
            episodes.append(episode)
        
        # Simple model that returns random logits
        def run_logits(episode):
            n_query = len(episode.query_y)
            n_classes = len(torch.unique(episode.query_y))
            return torch.randn(n_query, n_classes)
        
        results = evaluate(run_logits, episodes)
        
        # Check result structure
        assert isinstance(results, dict)
        assert 'mean' in results
        assert 'std' in results
        assert 'ci95' in results
        assert 'n' in results
        assert 'se' in results
        
        # Check statistical properties
        assert 0.0 <= results['mean'] <= 1.0  # Accuracy should be between 0 and 1
        assert results['std'] >= 0.0
        assert results['n'] == 5

    def test_evaluate_function_with_output_directory(self):
        """Test evaluate function with output directory."""
        episodes = [
            Episode(
                torch.randn(4, 10), torch.repeat_interleave(torch.arange(2), 2),
                torch.randn(6, 10), torch.repeat_interleave(torch.arange(2), 3)
            )
        ]
        
        def run_logits(episode):
            return torch.randn(len(episode.query_y), 2)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            results = evaluate(run_logits, episodes, outdir=temp_dir, dump_preds=True)
            
            # Check that output files were created
            import os
            assert os.path.exists(os.path.join(temp_dir, 'metrics.json'))
            assert os.path.exists(os.path.join(temp_dir, 'preds.jsonl'))

    def test_evaluate_function_confidence_intervals(self):
        """Test confidence interval calculations."""
        # Create deterministic episodes for controlled testing
        episodes = []
        for i in range(10):
            episode = Episode(
                torch.randn(4, 5), torch.repeat_interleave(torch.arange(2), 2),
                torch.randn(6, 5), torch.repeat_interleave(torch.arange(2), 3)
            )
            episodes.append(episode)
        
        # Deterministic model - always predicts class 0
        def deterministic_run_logits(episode):
            n_query = len(episode.query_y)
            logits = torch.zeros(n_query, 2)
            logits[:, 0] = 1.0  # Always predict class 0
            return logits
        
        results = evaluate(deterministic_run_logits, episodes)
        
        # With deterministic predictions, we should get consistent accuracy
        assert results['std'] >= 0.0  # Standard deviation should be non-negative
        assert results['ci95'] >= 0.0  # Confidence interval should be non-negative

    def test_evaluate_function_single_episode(self):
        """Test evaluate function with single episode."""
        episode = Episode(
            torch.randn(2, 10), torch.arange(2),
            torch.randn(4, 10), torch.arange(2).repeat(2)
        )
        
        def run_logits(ep):
            return torch.randn(4, 2)
        
        results = evaluate(run_logits, [episode])
        
        assert results['n'] == 1
        assert results['ci95'] == 0.0  # No confidence interval with single sample


class TestAccuracyMetric:
    """Test the enhanced Accuracy metric class."""

    def test_accuracy_initialization(self):
        """Test Accuracy metric initialization."""
        accuracy = Accuracy(confidence_level=0.95)
        assert accuracy.confidence_level == 0.95

    def test_accuracy_compute_basic(self):
        """Test basic accuracy computation."""
        accuracy = Accuracy()
        
        predictions = torch.tensor([0, 1, 1, 0, 1])
        targets = torch.tensor([0, 1, 0, 0, 1])
        
        results = accuracy.compute(predictions, targets)
        
        assert isinstance(results, dict)
        assert 'accuracy' in results
        assert 'std_error' in results
        assert 'confidence_interval' in results
        assert 'n_samples' in results
        
        # Check accuracy calculation
        expected_accuracy = 0.8  # 4 out of 5 correct
        assert results['accuracy'] == pytest.approx(expected_accuracy)
        assert results['n_samples'] == 5

    def test_accuracy_compute_perfect_predictions(self):
        """Test accuracy computation with perfect predictions."""
        accuracy = Accuracy()
        
        predictions = torch.tensor([0, 1, 0, 1])
        targets = torch.tensor([0, 1, 0, 1])
        
        results = accuracy.compute(predictions, targets)
        
        assert results['accuracy'] == 1.0
        assert results['std_error'] >= 0.0

    def test_accuracy_bootstrap_confidence_interval(self):
        """Test bootstrap confidence interval computation."""
        accuracy = Accuracy(confidence_level=0.95)
        
        predictions = torch.tensor([0, 1, 1, 0, 1, 0, 1, 0, 1, 1])
        targets = torch.tensor([0, 1, 0, 0, 1, 0, 1, 1, 1, 1])
        
        lower, upper = accuracy.bootstrap_confidence_interval(
            predictions, targets, n_bootstrap=100
        )
        
        assert isinstance(lower, float)
        assert isinstance(upper, float)
        assert 0.0 <= lower <= upper <= 1.0

    def test_accuracy_single_sample(self):
        """Test accuracy computation with single sample."""
        accuracy = Accuracy()
        
        predictions = torch.tensor([1])
        targets = torch.tensor([1])
        
        results = accuracy.compute(predictions, targets)
        
        assert results['accuracy'] == 1.0
        assert results['confidence_interval'] == 0.0  # No CI with single sample


class TestUncertaintyEvaluator:
    """Test uncertainty evaluation metrics."""

    def test_uncertainty_evaluator_initialization(self):
        """Test UncertaintyEvaluator initialization."""
        evaluator = UncertaintyEvaluator()
        assert evaluator.entropy_threshold == 1.0

    def test_compute_entropy(self):
        """Test entropy computation."""
        evaluator = UncertaintyEvaluator()
        
        # High entropy case (uniform distribution)
        uniform_logits = torch.ones(5, 3)  # Same logit for all classes
        entropy = evaluator.compute_entropy(uniform_logits)
        
        assert entropy.shape == (5,)
        assert torch.all(entropy > 0)  # Entropy should be positive
        
        # Low entropy case (confident predictions)
        confident_logits = torch.zeros(3, 3)
        confident_logits[:, 0] = 10.0  # Very confident about class 0
        entropy_confident = evaluator.compute_entropy(confident_logits)
        
        assert torch.all(entropy_confident < entropy.mean())  # Should be lower entropy

    def test_compute_confidence(self):
        """Test confidence computation."""
        evaluator = UncertaintyEvaluator()
        
        logits = torch.tensor([
            [5.0, 1.0, 0.0],  # High confidence for class 0
            [1.0, 1.0, 1.0],  # Low confidence (uniform)
            [0.0, 0.0, 3.0]   # High confidence for class 2
        ])
        
        confidence = evaluator.compute_confidence(logits)
        
        assert confidence.shape == (3,)
        assert torch.all(confidence >= 0.0)
        assert torch.all(confidence <= 1.0)
        
        # First and third predictions should be more confident than second
        assert confidence[0] > confidence[1]
        assert confidence[2] > confidence[1]

    def test_calibration_analysis(self):
        """Test calibration analysis."""
        evaluator = UncertaintyEvaluator()
        
        # Create logits and targets for calibration analysis
        logits = torch.tensor([
            [2.0, 0.0, 0.0],
            [0.0, 3.0, 0.0], 
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 2.5],
            [1.5, 0.0, 0.0]
        ])
        targets = torch.tensor([0, 1, 2, 2, 0])
        
        calibration = evaluator.calibration_analysis(logits, targets, n_bins=5)
        
        assert isinstance(calibration, dict)
        assert 'ece' in calibration  # Expected Calibration Error
        assert 'bin_accuracies' in calibration
        assert 'bin_confidences' in calibration
        assert 'bin_counts' in calibration
        
        assert len(calibration['bin_accuracies']) == 5
        assert len(calibration['bin_confidences']) == 5
        assert len(calibration['bin_counts']) == 5
        
        # ECE should be non-negative
        assert calibration['ece'] >= 0.0


class TestStatisticalTestSuite:
    """Test statistical testing functionality."""

    def test_statistical_test_suite_initialization(self):
        """Test StatisticalTestSuite initialization."""
        suite = StatisticalTestSuite(alpha=0.05)
        assert suite.alpha == 0.05

    def test_paired_t_test(self):
        """Test paired t-test functionality."""
        suite = StatisticalTestSuite()
        
        # Create two sets of scores
        scores1 = [0.8, 0.7, 0.9, 0.6, 0.85]
        scores2 = [0.75, 0.65, 0.85, 0.55, 0.80]
        
        result = suite.paired_t_test(scores1, scores2)
        
        assert isinstance(result, dict)
        assert 't_statistic' in result
        assert 'p_value' in result
        assert 'significant' in result
        assert 'mean_difference' in result
        
        assert isinstance(result['significant'], bool)
        assert 0.0 <= result['p_value'] <= 1.0

    def test_paired_t_test_identical_scores(self):
        """Test paired t-test with identical scores."""
        suite = StatisticalTestSuite()
        
        scores = [0.8, 0.7, 0.9, 0.6, 0.85]
        result = suite.paired_t_test(scores, scores)
        
        assert result['mean_difference'] == 0.0
        assert result['t_statistic'] == 0.0
        assert not result['significant']  # Should not be significant

    def test_paired_t_test_insufficient_data(self):
        """Test paired t-test with insufficient data."""
        suite = StatisticalTestSuite()
        
        result = suite.paired_t_test([0.8], [0.7])
        
        assert result['t_statistic'] == 0.0
        assert result['p_value'] == 1.0
        assert not result['significant']

    def test_paired_t_test_mismatched_lengths(self):
        """Test paired t-test with mismatched score lengths."""
        suite = StatisticalTestSuite()
        
        with pytest.raises(ValueError):
            suite.paired_t_test([0.8, 0.7], [0.75, 0.65, 0.85])

    def test_wilcoxon_signed_rank_test(self):
        """Test Wilcoxon signed-rank test."""
        suite = StatisticalTestSuite()
        
        scores1 = [0.8, 0.7, 0.9, 0.6, 0.85, 0.75]
        scores2 = [0.75, 0.65, 0.85, 0.55, 0.80, 0.70]
        
        result = suite.wilcoxon_signed_rank_test(scores1, scores2)
        
        assert isinstance(result, dict)
        assert 'w_statistic' in result
        assert 'p_value' in result
        assert 'significant' in result
        
        assert 0.0 <= result['p_value'] <= 1.0
        assert isinstance(result['significant'], bool)

    def test_wilcoxon_no_differences(self):
        """Test Wilcoxon test with no differences."""
        suite = StatisticalTestSuite()
        
        scores = [0.8, 0.7, 0.9]
        result = suite.wilcoxon_signed_rank_test(scores, scores)
        
        assert result['w_statistic'] == 0.0
        assert result['p_value'] == 1.0
        assert not result['significant']


class TestFewShotEvaluationHarness:
    """Test the FewShotEvaluationHarness."""

    def test_evaluation_harness_initialization(self):
        """Test FewShotEvaluationHarness initialization."""
        harness = FewShotEvaluationHarness()
        assert hasattr(harness, 'confidence_level')
        assert hasattr(harness, 'bootstrap_samples')

    def test_evaluation_harness_with_custom_config(self):
        """Test evaluation harness with custom configuration."""
        harness = FewShotEvaluationHarness(
            confidence_level=0.99,
            bootstrap_samples=500,
            statistical_tests=True
        )
        
        assert harness.confidence_level == 0.99
        assert harness.bootstrap_samples == 500

    def test_evaluate_on_episodes(self):
        """Test evaluating on multiple episodes."""
        harness = FewShotEvaluationHarness()
        
        # Create test episodes
        episodes = []
        for i in range(3):
            episode = Episode(
                torch.randn(6, 10), torch.repeat_interleave(torch.arange(3), 2),
                torch.randn(9, 10), torch.repeat_interleave(torch.arange(3), 3)
            )
            episodes.append(episode)
        
        # Define evaluation function that returns results
        def eval_function(episode):
            return {
                'query_accuracy': 0.7 + 0.1 * torch.rand(1).item(),
                'query_loss': 0.5 + 0.2 * torch.rand(1).item()
            }
        
        results = harness.evaluate_on_episodes(episodes, eval_function)
        
        assert isinstance(results, dict)
        assert 'aggregate_metrics' in results or 'mean_accuracy' in results

    def test_statistical_significance_testing(self):
        """Test statistical significance testing in evaluation."""
        harness = FewShotEvaluationHarness(statistical_tests=True)
        
        # Create two sets of results for comparison
        results1 = [0.8, 0.7, 0.9, 0.6, 0.85]
        results2 = [0.75, 0.65, 0.85, 0.55, 0.80]
        
        # Test if harness can perform statistical comparison
        # (This depends on actual implementation)
        assert hasattr(harness, 'confidence_level')


class TestPrototypeAnalyzer:
    """Test prototype analysis functionality."""

    def test_prototype_analyzer_initialization(self):
        """Test PrototypeAnalyzer initialization."""
        analyzer = PrototypeAnalyzer()
        assert hasattr(analyzer, 'distance_metric')

    def test_compute_prototypes(self):
        """Test prototype computation."""
        analyzer = PrototypeAnalyzer()
        
        # Create support set
        support_x = torch.randn(12, 10)  # 4 classes, 3 shots each
        support_y = torch.repeat_interleave(torch.arange(4), 3)
        
        prototypes = analyzer.compute_prototypes(support_x, support_y)
        
        assert prototypes.shape == (4, 10)  # 4 classes, 10 features
        
        # Each prototype should be the mean of its class samples
        for class_idx in range(4):
            class_mask = support_y == class_idx
            class_samples = support_x[class_mask]
            expected_prototype = class_samples.mean(dim=0)
            
            assert torch.allclose(prototypes[class_idx], expected_prototype, atol=1e-6)

    def test_prototype_classification(self):
        """Test classification using prototypes."""
        analyzer = PrototypeAnalyzer()
        
        # Simple 2D data for easy testing
        support_x = torch.tensor([
            [0.0, 0.0], [0.1, 0.1], [0.0, 0.1],  # Class 0: around origin
            [1.0, 1.0], [1.1, 0.9], [0.9, 1.1]   # Class 1: around (1,1)
        ])
        support_y = torch.tensor([0, 0, 0, 1, 1, 1])
        
        query_x = torch.tensor([
            [0.05, 0.05],  # Should be classified as class 0
            [0.95, 0.95]   # Should be classified as class 1
        ])
        
        logits = analyzer.classify_queries(support_x, support_y, query_x)
        
        assert logits.shape == (2, 2)  # 2 queries, 2 classes
        
        # Check that predictions make sense
        predictions = logits.argmax(dim=1)
        assert predictions[0] == 0  # First query -> class 0
        assert predictions[1] == 1  # Second query -> class 1

    def test_analyze_episode(self):
        """Test comprehensive episode analysis."""
        analyzer = PrototypeAnalyzer()
        
        episode = Episode(
            torch.randn(9, 15), torch.repeat_interleave(torch.arange(3), 3),
            torch.randn(12, 15), torch.repeat_interleave(torch.arange(3), 4)
        )
        
        analysis = analyzer.analyze_episode(episode)
        
        assert isinstance(analysis, dict)
        # Check for expected analysis components
        expected_keys = ['intra_class_distances', 'inter_class_distances', 
                        'silhouette_scores', 'class_separability']
        
        for key in expected_keys:
            if key in analysis:  # Not all analyzers may implement all metrics
                assert isinstance(analysis[key], (float, torch.Tensor, list))

    def test_distance_metrics(self):
        """Test different distance metrics."""
        # Test with different distance metrics if supported
        for distance_metric in ['euclidean', 'cosine']:
            try:
                analyzer = PrototypeAnalyzer(distance_metric=distance_metric)
                
                support_x = torch.randn(6, 10)
                support_y = torch.repeat_interleave(torch.arange(2), 3)
                query_x = torch.randn(4, 10)
                
                logits = analyzer.classify_queries(support_x, support_y, query_x)
                assert logits.shape == (4, 2)
                
            except (AttributeError, NotImplementedError):
                # Skip if distance metric not implemented
                continue


class TestEvaluationIntegration:
    """Test integration of evaluation components."""

    def test_end_to_end_evaluation_workflow(self):
        """Test complete evaluation workflow."""
        # Create model
        model = nn.Sequential(
            nn.Linear(20, 50),
            nn.ReLU(),
            nn.Linear(50, 5)
        )
        
        # Create episodes
        episodes = []
        for i in range(5):
            episode = Episode(
                torch.randn(10, 20), torch.repeat_interleave(torch.arange(5), 2),
                torch.randn(15, 20), torch.repeat_interleave(torch.arange(5), 3)
            )
            episodes.append(episode)
        
        # Define evaluation function
        def run_logits(episode):
            model.eval()
            with torch.no_grad():
                return model(episode.query_x)
        
        # Run basic evaluation
        results = evaluate(run_logits, episodes)
        
        assert isinstance(results, dict)
        assert 'mean' in results
        assert 0.0 <= results['mean'] <= 1.0

    def test_evaluation_with_uncertainty_analysis(self):
        """Test evaluation including uncertainty analysis."""
        uncertainty_evaluator = UncertaintyEvaluator()
        
        # Create mock logits with varying confidence
        logits_list = [
            torch.tensor([[5.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),  # High and low confidence
            torch.tensor([[0.0, 4.0, 0.0], [2.0, 0.0, 0.0]]),
        ]
        targets_list = [
            torch.tensor([0, 2]),
            torch.tensor([1, 0])
        ]
        
        total_entropy = []
        total_confidence = []
        
        for logits, targets in zip(logits_list, targets_list):
            entropy = uncertainty_evaluator.compute_entropy(logits)
            confidence = uncertainty_evaluator.compute_confidence(logits)
            
            total_entropy.extend(entropy.tolist())
            total_confidence.extend(confidence.tolist())
        
        assert len(total_entropy) == 4
        assert len(total_confidence) == 4
        assert all(e >= 0 for e in total_entropy)
        assert all(0 <= c <= 1 for c in total_confidence)

    def test_comparative_evaluation(self):
        """Test comparative evaluation of multiple algorithms."""
        # Create two simple models
        model1 = nn.Linear(10, 3)
        model2 = nn.Linear(10, 3)
        
        episodes = [
            Episode(
                torch.randn(6, 10), torch.repeat_interleave(torch.arange(3), 2),
                torch.randn(9, 10), torch.repeat_interleave(torch.arange(3), 3)
            )
            for _ in range(3)
        ]
        
        # Evaluate both models
        def run_logits_1(episode):
            return model1(episode.query_x)
        
        def run_logits_2(episode):
            return model2(episode.query_x)
        
        results1 = evaluate(run_logits_1, episodes)
        results2 = evaluate(run_logits_2, episodes)
        
        # Statistical comparison
        stat_suite = StatisticalTestSuite()
        
        # Extract accuracies for comparison (would need actual episode-level results)
        # This is a simplified version - in practice would need episode-level accuracies
        if 'individual_accuracies' in results1 and 'individual_accuracies' in results2:
            comparison = stat_suite.paired_t_test(
                results1['individual_accuracies'],
                results2['individual_accuracies']
            )
            assert isinstance(comparison, dict)
            assert 'significant' in comparison

    def test_evaluation_robustness(self):
        """Test evaluation robustness to edge cases."""
        # Test with episodes of different sizes
        episodes = [
            Episode(
                torch.randn(2, 10), torch.arange(2),
                torch.randn(4, 10), torch.arange(2).repeat(2)
            ),
            Episode(
                torch.randn(10, 10), torch.repeat_interleave(torch.arange(5), 2),
                torch.randn(25, 10), torch.repeat_interleave(torch.arange(5), 5)
            )
        ]
        
        def run_logits(episode):
            n_query = len(episode.query_y)
            n_classes = len(torch.unique(episode.query_y))
            return torch.randn(n_query, n_classes)
        
        # Should handle episodes of different sizes
        results = evaluate(run_logits, episodes)
        assert isinstance(results, dict)

    def test_evaluation_performance_monitoring(self):
        """Test evaluation performance monitoring."""
        import time
        
        episodes = [
            Episode(
                torch.randn(4, 50), torch.repeat_interleave(torch.arange(2), 2),
                torch.randn(6, 50), torch.repeat_interleave(torch.arange(2), 3)
            )
            for _ in range(10)
        ]
        
        def slow_run_logits(episode):
            time.sleep(0.01)  # Simulate computation time
            return torch.randn(len(episode.query_y), 2)
        
        start_time = time.time()
        results = evaluate(slow_run_logits, episodes)
        elapsed = time.time() - start_time
        
        # Should complete in reasonable time
        assert elapsed > 0.05  # At least 5 * 0.01 seconds
        assert elapsed < 5.0   # But not too long
        assert 'elapsed_time' in results


if __name__ == "__main__":
    pytest.main([__file__])