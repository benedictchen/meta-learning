"""
Tests for enhanced evaluation and statistical analysis tools.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch

from meta_learning.eval import (
    MetaLearningEvaluator, LearnabilityAnalyzer, StatisticalTestSuite,
    UncertaintyEvaluator, Accuracy, TorchMetaEvaluationHarness,
    MetaLearningMetrics, EvaluationVisualizer, evaluate_multiple_seeds,
    MetaLearningCrossValidator, comprehensive_evaluate
)


class TestMetaLearningEvaluator:
    """Test comprehensive meta-learning evaluator."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.evaluator = MetaLearningEvaluator(n_bootstrap=100, confidence_level=0.95)
        
        # Create mock model
        self.model = Mock()
        self.model.return_value = torch.tensor([
            [2.0, 1.0, 0.5],  # Predicts class 0
            [0.5, 2.0, 1.0],  # Predicts class 1  
            [1.0, 0.5, 2.0],  # Predicts class 2
        ])
    
    def test_initialization(self):
        """Test evaluator initialization."""
        assert self.evaluator.n_bootstrap == 100
        assert self.evaluator.confidence_level == 0.95
        assert self.evaluator.random_state == 42
    
    def test_evaluate_model_episodic(self):
        """Test model evaluation with episodic protocol."""
        episodes = [
            {
                "support_x": torch.randn(6, 64),
                "support_y": torch.tensor([0, 0, 1, 1, 2, 2]),
                "query_x": torch.randn(3, 64),
                "query_y": torch.tensor([0, 1, 2])
            }
        ] * 10  # Multiple episodes
        
        results = self.evaluator.evaluate_model(self.model, episodes, protocol="episodic")
        
        assert "mean_accuracy" in results
        assert "std_accuracy" in results
        assert "confidence_interval" in results
        assert "per_episode_accuracies" in results
        assert "bootstrap_mean" in results
        assert "bootstrap_std" in results
        
        # Check confidence interval structure
        ci = results["confidence_interval"]
        assert len(ci) == 2
        assert ci[0] <= results["mean_accuracy"] <= ci[1]
    
    def test_bootstrap_confidence_interval(self):
        """Test bootstrap confidence interval calculation."""
        accuracies = [0.8, 0.85, 0.9, 0.75, 0.82, 0.88, 0.92, 0.78, 0.86, 0.84]
        
        ci_lower, ci_upper, boot_mean, boot_std = self.evaluator._bootstrap_confidence_interval(accuracies)
        
        assert ci_lower <= boot_mean <= ci_upper
        assert ci_lower >= 0.0 and ci_upper <= 1.0
        assert boot_std >= 0.0
        
        # Bootstrap mean should be close to original mean
        original_mean = np.mean(accuracies)
        assert abs(boot_mean - original_mean) < 0.05
    
    def test_cross_validation_evaluation(self):
        """Test cross-validation evaluation."""
        episodes = [
            {
                "support_x": torch.randn(6, 64),
                "support_y": torch.tensor([0, 0, 1, 1, 2, 2]),
                "query_x": torch.randn(3, 64),
                "query_y": torch.tensor([0, 1, 2])
            }
        ] * 20  # Enough episodes for CV
        
        results = self.evaluator.evaluate_model(self.model, episodes, protocol="cross_validation", cv_folds=5)
        
        assert "cv_accuracies" in results
        assert "cv_mean_accuracy" in results
        assert "cv_std_accuracy" in results
        assert len(results["cv_accuracies"]) == 5


class TestLearnabilityAnalyzer:
    """Test learnability analyzer."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.analyzer = LearnabilityAnalyzer()
    
    def test_analyze_task_difficulty(self):
        """Test task difficulty analysis."""
        # Create episode with clear class separation
        support_x = torch.tensor([
            [1.0, 0.0], [1.1, 0.1],  # Class 0 - clear cluster
            [0.0, 1.0], [0.1, 1.1],  # Class 1 - clear cluster
            [2.0, 2.0], [2.1, 2.1]   # Class 2 - clear cluster
        ], dtype=torch.float32)
        support_y = torch.tensor([0, 0, 1, 1, 2, 2])
        
        episode = {
            "support_x": support_x,
            "support_y": support_y,
            "query_x": torch.tensor([[1.05, 0.05], [0.05, 1.05], [2.05, 2.05]]),
            "query_y": torch.tensor([0, 1, 2])
        }
        
        difficulty_metrics = self.analyzer.analyze_task_difficulty([episode])
        
        assert "mean_intra_class_distance" in difficulty_metrics
        assert "mean_inter_class_distance" in difficulty_metrics
        assert "separability_ratio" in difficulty_metrics
        assert "class_imbalance" in difficulty_metrics
        assert "feature_dimensionality" in difficulty_metrics
        
        # With clear separation, separability ratio should be high
        assert difficulty_metrics["separability_ratio"] > 1.0
    
    def test_complexity_analysis(self):
        """Test complexity analysis."""
        episodes = []
        for i in range(5):
            episodes.append({
                "support_x": torch.randn(6, 32),
                "support_y": torch.tensor([0, 0, 1, 1, 2, 2]),
                "query_x": torch.randn(3, 32),
                "query_y": torch.tensor([0, 1, 2])
            })
        
        complexity_metrics = self.analyzer.analyze_complexity(episodes)
        
        assert "average_n_way" in complexity_metrics
        assert "average_k_shot" in complexity_metrics
        assert "feature_dimensionality" in complexity_metrics
        assert "episode_diversity" in complexity_metrics
        assert "complexity_score" in complexity_metrics
        
        assert complexity_metrics["average_n_way"] == 3
        assert complexity_metrics["average_k_shot"] == 2
        assert complexity_metrics["feature_dimensionality"] == 32
    
    def test_learning_curve_analysis(self):
        """Test learning curve analysis."""
        # Create mock performance history
        performance_history = [0.3, 0.5, 0.7, 0.8, 0.85, 0.87, 0.88, 0.89, 0.89, 0.9]
        
        curve_metrics = self.analyzer.analyze_learning_curve(performance_history)
        
        assert "initial_performance" in curve_metrics
        assert "final_performance" in curve_metrics
        assert "improvement_rate" in curve_metrics
        assert "convergence_episode" in curve_metrics
        assert "plateau_detected" in curve_metrics
        
        assert curve_metrics["initial_performance"] == 0.3
        assert curve_metrics["final_performance"] == 0.9
        assert curve_metrics["improvement_rate"] > 0


class TestStatisticalTestSuite:
    """Test statistical test suite."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.test_suite = StatisticalTestSuite(alpha=0.05)
    
    def test_paired_t_test(self):
        """Test paired t-test."""
        # Create clearly different performance arrays
        performance_a = np.array([0.7, 0.72, 0.68, 0.71, 0.69, 0.73, 0.7, 0.71])
        performance_b = np.array([0.8, 0.82, 0.78, 0.81, 0.79, 0.83, 0.8, 0.81])
        
        result = self.test_suite.paired_t_test(performance_a, performance_b)
        
        assert "statistic" in result
        assert "p_value" in result
        assert "significant" in result
        assert "effect_size" in result
        assert "confidence_interval" in result
        
        # Should be significant difference
        assert result["significant"] == True
        assert result["p_value"] < 0.05
    
    def test_mann_whitney_u_test(self):
        """Test Mann-Whitney U test."""
        performance_a = np.array([0.6, 0.62, 0.58, 0.61, 0.59])
        performance_b = np.array([0.8, 0.82, 0.78, 0.81, 0.79])
        
        result = self.test_suite.mann_whitney_u_test(performance_a, performance_b)
        
        assert "statistic" in result
        assert "p_value" in result
        assert "significant" in result
        
        # Should be significant difference
        assert result["significant"] == True
    
    def test_wilcoxon_signed_rank_test(self):
        """Test Wilcoxon signed-rank test."""
        performance_a = np.array([0.7, 0.72, 0.68, 0.71, 0.69])
        performance_b = np.array([0.75, 0.77, 0.73, 0.76, 0.74])
        
        result = self.test_suite.wilcoxon_signed_rank_test(performance_a, performance_b)
        
        assert "statistic" in result
        assert "p_value" in result
        assert "significant" in result
    
    def test_multiple_comparison_correction(self):
        """Test multiple comparison correction."""
        # Create multiple p-values
        p_values = [0.01, 0.03, 0.07, 0.12, 0.25]
        
        result = self.test_suite.multiple_comparison_correction(p_values, method="bonferroni")
        
        assert "corrected_p_values" in result
        assert "significant" in result
        assert "alpha_corrected" in result
        
        # Bonferroni correction should increase p-values
        assert all(corrected >= original for corrected, original 
                  in zip(result["corrected_p_values"], p_values))
    
    def test_effect_size_calculation(self):
        """Test effect size calculation."""
        group_a = np.array([0.7, 0.72, 0.68, 0.71, 0.69])
        group_b = np.array([0.8, 0.82, 0.78, 0.81, 0.79])
        
        cohen_d = self.test_suite._calculate_cohen_d(group_a, group_b)
        
        assert isinstance(cohen_d, float)
        # Should indicate large effect size
        assert abs(cohen_d) > 0.8


class TestUncertaintyEvaluator:
    """Test uncertainty evaluator."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.evaluator = UncertaintyEvaluator()
    
    def test_calibration_analysis(self):
        """Test calibration analysis."""
        # Create mock predictions with confidences
        predictions = np.array([0, 1, 0, 1, 0])
        confidences = np.array([0.9, 0.8, 0.7, 0.85, 0.95])
        ground_truth = np.array([0, 1, 1, 1, 0])  # Some correct, some wrong
        
        calibration_metrics = self.evaluator.analyze_calibration(predictions, confidences, ground_truth)
        
        assert "ece" in calibration_metrics  # Expected Calibration Error
        assert "mce" in calibration_metrics  # Maximum Calibration Error
        assert "reliability_diagram" in calibration_metrics
        assert "brier_score" in calibration_metrics
        
        # ECE and MCE should be between 0 and 1
        assert 0 <= calibration_metrics["ece"] <= 1
        assert 0 <= calibration_metrics["mce"] <= 1
    
    def test_uncertainty_quality_metrics(self):
        """Test uncertainty quality metrics."""
        uncertainties = np.array([0.1, 0.3, 0.2, 0.4, 0.15])
        accuracies = np.array([0.95, 0.7, 0.85, 0.6, 0.9])  # High uncertainty should correlate with low accuracy
        
        quality_metrics = self.evaluator.analyze_uncertainty_quality(uncertainties, accuracies)
        
        assert "correlation_coefficient" in quality_metrics
        assert "spearman_correlation" in quality_metrics
        assert "auc_roc" in quality_metrics
        assert "mutual_information" in quality_metrics
        
        # Correlation should be negative (high uncertainty, low accuracy)
        assert quality_metrics["correlation_coefficient"] < 0
    
    def test_prediction_intervals(self):
        """Test prediction interval analysis."""
        predictions = np.array([0.8, 0.6, 0.7, 0.9, 0.5])
        intervals = np.array([[0.7, 0.9], [0.5, 0.7], [0.6, 0.8], [0.8, 1.0], [0.4, 0.6]])
        ground_truth = np.array([0.85, 0.55, 0.75, 0.95, 0.45])
        
        interval_metrics = self.evaluator.analyze_prediction_intervals(predictions, intervals, ground_truth)
        
        assert "coverage_probability" in interval_metrics
        assert "average_interval_width" in interval_metrics
        assert "sharpness" in interval_metrics
        
        # Coverage should be between 0 and 1
        assert 0 <= interval_metrics["coverage_probability"] <= 1


class TestAccuracy:
    """Test accuracy calculation utility."""
    
    def test_accuracy_calculation(self):
        """Test basic accuracy calculation."""
        predictions = torch.tensor([0, 1, 2, 1, 0])
        targets = torch.tensor([0, 1, 2, 2, 0])
        
        acc = Accuracy()
        accuracy = acc(predictions, targets)
        
        expected_accuracy = 4/5  # 4 correct out of 5
        assert torch.isclose(accuracy, torch.tensor(expected_accuracy))
    
    def test_accuracy_with_different_shapes(self):
        """Test accuracy with logits input."""
        logits = torch.tensor([
            [2.0, 1.0, 0.5],  # Predicts class 0
            [0.5, 2.0, 1.0],  # Predicts class 1
            [1.0, 0.5, 2.0],  # Predicts class 2
        ])
        targets = torch.tensor([0, 1, 2])
        
        acc = Accuracy()
        accuracy = acc(logits, targets)
        
        expected_accuracy = 1.0  # All correct
        assert torch.isclose(accuracy, torch.tensor(expected_accuracy))


class TestTorchMetaEvaluationHarness:
    """Test TorchMeta evaluation harness."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.harness = TorchMetaEvaluationHarness()
    
    def test_create_task_from_episode(self):
        """Test creating task from episode."""
        episode = {
            "support_x": torch.randn(6, 64),
            "support_y": torch.tensor([0, 0, 1, 1, 2, 2]),
            "query_x": torch.randn(3, 64),
            "query_y": torch.tensor([0, 1, 2])
        }
        
        task = self.harness._create_task_from_episode(episode)
        
        assert hasattr(task, 'x')
        assert hasattr(task, 'y')
        assert task.x.shape[0] == 9  # 6 support + 3 query
        assert task.y.shape[0] == 9
    
    def test_evaluate_with_torchmeta_style(self):
        """Test evaluation with TorchMeta style."""
        episodes = [
            {
                "support_x": torch.randn(6, 64),
                "support_y": torch.tensor([0, 0, 1, 1, 2, 2]),
                "query_x": torch.randn(3, 64),
                "query_y": torch.tensor([0, 1, 2])
            }
        ] * 5
        
        # Mock model
        model = Mock()
        model.return_value = torch.tensor([
            [2.0, 1.0, 0.5],  # Predicts class 0
            [0.5, 2.0, 1.0],  # Predicts class 1
            [1.0, 0.5, 2.0],  # Predicts class 2
        ])
        
        results = self.harness.evaluate_model(model, episodes)
        
        assert "accuracy" in results
        assert "confidence_interval" in results
        assert "per_task_accuracies" in results


class TestMetaLearningMetrics:
    """Test meta-learning specific metrics."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.metrics = MetaLearningMetrics()
    
    def test_adaptation_speed(self):
        """Test adaptation speed calculation."""
        learning_curves = [
            [0.3, 0.5, 0.7, 0.8, 0.85],  # Fast adaptation
            [0.2, 0.25, 0.3, 0.35, 0.4],  # Slow adaptation
            [0.4, 0.6, 0.75, 0.85, 0.9]   # Fast adaptation
        ]
        
        adaptation_metrics = self.metrics.compute_adaptation_speed(learning_curves)
        
        assert "mean_adaptation_speed" in adaptation_metrics
        assert "std_adaptation_speed" in adaptation_metrics
        assert "per_task_speeds" in adaptation_metrics
        
        assert len(adaptation_metrics["per_task_speeds"]) == 3
        assert adaptation_metrics["mean_adaptation_speed"] > 0
    
    def test_generalization_gap(self):
        """Test generalization gap calculation."""
        train_accuracies = [0.9, 0.85, 0.92, 0.88, 0.91]
        test_accuracies = [0.8, 0.75, 0.82, 0.78, 0.81]
        
        gap_metrics = self.metrics.compute_generalization_gap(train_accuracies, test_accuracies)
        
        assert "mean_gap" in gap_metrics
        assert "std_gap" in gap_metrics
        assert "per_task_gaps" in gap_metrics
        
        # Gap should be positive (train > test)
        assert gap_metrics["mean_gap"] > 0
        assert len(gap_metrics["per_task_gaps"]) == 5
    
    def test_few_shot_metrics(self):
        """Test few-shot specific metrics."""
        episodes = []
        for i in range(10):
            episodes.append({
                "support_x": torch.randn(6, 64),
                "support_y": torch.tensor([0, 0, 1, 1, 2, 2]),
                "query_x": torch.randn(3, 64),
                "query_y": torch.tensor([0, 1, 2])
            })
        
        few_shot_metrics = self.metrics.compute_few_shot_metrics(episodes, [0.8] * 10)
        
        assert "average_n_way" in few_shot_metrics
        assert "average_k_shot" in few_shot_metrics
        assert "shots_per_accuracy" in few_shot_metrics
        assert "way_difficulty_analysis" in few_shot_metrics


class TestEvaluationVisualizer:
    """Test evaluation visualizer."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.visualizer = EvaluationVisualizer()
    
    def test_create_learning_curve_plot(self):
        """Test learning curve plot creation."""
        learning_curves = [
            [0.3, 0.5, 0.7, 0.8, 0.85],
            [0.2, 0.4, 0.6, 0.75, 0.82],
            [0.4, 0.6, 0.75, 0.85, 0.9]
        ]
        
        plot_data = self.visualizer.create_learning_curve_plot(learning_curves)
        
        assert "x_values" in plot_data
        assert "mean_curve" in plot_data
        assert "confidence_bands" in plot_data
        assert "individual_curves" in plot_data
        
        assert len(plot_data["x_values"]) == 5
        assert len(plot_data["mean_curve"]) == 5


class TestEvaluateMultipleSeeds:
    """Test multi-seed evaluation function."""
    
    def test_evaluate_multiple_seeds(self):
        """Test evaluation across multiple seeds."""
        # Mock model and episodes
        model = Mock()
        model.return_value = torch.tensor([
            [2.0, 1.0, 0.5],
            [0.5, 2.0, 1.0],
            [1.0, 0.5, 2.0],
        ])
        
        episodes = [
            {
                "support_x": torch.randn(6, 64),
                "support_y": torch.tensor([0, 0, 1, 1, 2, 2]),
                "query_x": torch.randn(3, 64),
                "query_y": torch.tensor([0, 1, 2])
            }
        ] * 5
        
        results = evaluate_multiple_seeds(
            model, episodes, seeds=[42, 123, 456], 
            evaluation_fn=lambda m, e: {"accuracy": 0.85}
        )
        
        assert "seed_results" in results
        assert "aggregated_results" in results
        assert len(results["seed_results"]) == 3
        
        aggregated = results["aggregated_results"]
        assert "mean_accuracy" in aggregated
        assert "std_accuracy" in aggregated
        assert "confidence_interval" in aggregated


class TestMetaLearningCrossValidator:
    """Test meta-learning cross validator."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.cv = MetaLearningCrossValidator(n_splits=3, random_state=42)
    
    def test_split_episodes(self):
        """Test episode splitting for cross-validation."""
        episodes = []
        for i in range(15):
            episodes.append({
                "support_x": torch.randn(6, 64),
                "support_y": torch.tensor([0, 0, 1, 1, 2, 2]),
                "query_x": torch.randn(3, 64),
                "query_y": torch.tensor([0, 1, 2]),
                "task_id": i % 5  # 5 different tasks
            })
        
        splits = list(self.cv.split_episodes(episodes))
        
        assert len(splits) == 3
        for train_episodes, val_episodes in splits:
            assert len(train_episodes) > 0
            assert len(val_episodes) > 0
            assert len(train_episodes) + len(val_episodes) == 15
    
    def test_cross_validate(self):
        """Test cross-validation evaluation."""
        # Mock model and episodes
        model = Mock()
        model.return_value = torch.tensor([
            [2.0, 1.0, 0.5],
            [0.5, 2.0, 1.0], 
            [1.0, 0.5, 2.0],
        ])
        
        episodes = []
        for i in range(12):
            episodes.append({
                "support_x": torch.randn(6, 64),
                "support_y": torch.tensor([0, 0, 1, 1, 2, 2]),
                "query_x": torch.randn(3, 64),
                "query_y": torch.tensor([0, 1, 2])
            })
        
        # Mock evaluation function
        def mock_eval_fn(model, episodes):
            return {"accuracy": 0.8 + np.random.normal(0, 0.05)}
        
        results = self.cv.cross_validate(model, episodes, mock_eval_fn)
        
        assert "fold_results" in results
        assert "mean_accuracy" in results
        assert "std_accuracy" in results
        assert len(results["fold_results"]) == 3


class TestComprehensiveEvaluate:
    """Test comprehensive evaluation function."""
    
    def test_comprehensive_evaluate(self):
        """Test comprehensive evaluation with all metrics."""
        # Mock model
        model = Mock()
        model.return_value = torch.tensor([
            [2.0, 1.0, 0.5],
            [0.5, 2.0, 1.0],
            [1.0, 0.5, 2.0],
        ])
        
        episodes = []
        for i in range(20):
            episodes.append({
                "support_x": torch.randn(6, 64),
                "support_y": torch.tensor([0, 0, 1, 1, 2, 2]),
                "query_x": torch.randn(3, 64),
                "query_y": torch.tensor([0, 1, 2])
            })
        
        results = comprehensive_evaluate(
            model, episodes,
            include_bootstrap=True,
            include_cross_validation=True,
            include_learnability=True,
            include_statistical_tests=False,  # Skip for simplicity
            n_bootstrap=50,
            cv_folds=3
        )
        
        assert "basic_metrics" in results
        assert "bootstrap_results" in results
        assert "cross_validation_results" in results
        assert "learnability_analysis" in results
        
        # Check basic metrics
        basic = results["basic_metrics"]
        assert "mean_accuracy" in basic
        assert "std_accuracy" in basic
        
        # Check bootstrap results
        bootstrap = results["bootstrap_results"]
        assert "confidence_interval" in bootstrap
        assert "bootstrap_mean" in bootstrap
        
        # Check CV results
        cv = results["cross_validation_results"]
        assert "cv_mean_accuracy" in cv
        assert "fold_results" in cv