"""
Comprehensive tests for Advanced evaluation metrics implementation.

Tests prototype quality assessment, task difficulty estimation, uncertainty quantification,
and research-grade evaluation metrics for meta-learning.
"""
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from unittest.mock import MagicMock, patch
from typing import Dict, List, Tuple, Any
import math
from collections import Counter

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from meta_learning.meta_learning_modules.advanced_evaluation_metrics import (
    PrototypeAnalyzer,
    TaskDifficultyEstimator,
    UncertaintyQuantifier,
    FewShotMetrics,
    ResearchMetrics,
    EvaluationSuite,
    MetaLearningBenchmark,
    compute_prototype_quality,
    estimate_task_difficulty,
    quantify_prediction_uncertainty
)


class TestPrototypeAnalyzer:
    """Test PrototypeAnalyzer functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.analyzer = PrototypeAnalyzer()
    
    def test_analyze_prototypes_basic(self):
        """Test basic prototype analysis."""
        # Well-separated prototypes
        support_features = torch.tensor([
            [1.0, 0.0], [1.1, 0.1], [1.0, 0.1],  # Class 0
            [0.0, 1.0], [0.1, 1.1], [0.1, 1.0],  # Class 1  
            [-1.0, 0.0], [-1.1, 0.1], [-1.0, 0.1]  # Class 2
        ])
        support_labels = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])
        
        # Compute prototypes
        prototypes = torch.zeros(3, 2)
        for k in range(3):
            mask = support_labels == k
            prototypes[k] = support_features[mask].mean(dim=0)
        
        analysis = self.analyzer.analyze_prototypes(support_features, support_labels, prototypes)
        
        assert 'intra_class_variance' in analysis
        assert 'inter_class_separation' in analysis
        assert 'silhouette_score' in analysis
        assert 'prototype_quality_score' in analysis
        
        # Well-separated classes should have good scores
        assert analysis['inter_class_separation'] > 0.5
        assert analysis['silhouette_score'] > 0.0
    
    def test_compute_intra_class_variance(self):
        """Test intra-class variance computation."""
        # Tight clusters
        support_features = torch.tensor([
            [1.0, 0.0], [1.01, 0.01],  # Class 0: very tight
            [0.0, 1.0], [0.5, 1.5]     # Class 1: more spread out
        ])
        support_labels = torch.tensor([0, 0, 1, 1])
        prototypes = torch.tensor([[1.005, 0.005], [0.25, 1.25]])
        
        variances = self.analyzer.compute_intra_class_variance(
            support_features, support_labels, prototypes
        )
        
        assert len(variances) == 2
        # Class 0 should have lower variance than Class 1
        assert variances[0] < variances[1]
    
    def test_compute_inter_class_separation(self):
        """Test inter-class separation computation."""
        # Well-separated prototypes
        prototypes = torch.tensor([
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0]
        ])
        
        separation = self.analyzer.compute_inter_class_separation(prototypes)
        
        # Should be positive for separated prototypes
        assert separation > 0
        
        # Test with overlapping prototypes
        overlapping_prototypes = torch.tensor([
            [0.0, 0.0],
            [0.1, 0.1],
            [0.05, 0.05]
        ])
        
        overlap_separation = self.analyzer.compute_inter_class_separation(overlapping_prototypes)
        
        # Should be lower for overlapping prototypes
        assert overlap_separation < separation
    
    def test_compute_silhouette_score(self):
        """Test silhouette score computation."""
        # Perfect clusters
        support_features = torch.tensor([
            [1.0, 1.0], [1.0, 1.0], [1.0, 1.0],  # Class 0: identical points
            [3.0, 3.0], [3.0, 3.0], [3.0, 3.0],  # Class 1: identical points, far from class 0
        ])
        support_labels = torch.tensor([0, 0, 0, 1, 1, 1])
        
        score = self.analyzer.compute_silhouette_score(support_features, support_labels)
        
        # Perfect separation should give high score
        assert score > 0.8
        
        # Test with overlapping clusters
        overlapping_features = torch.tensor([
            [0.0, 0.0], [0.1, 0.1], [0.0, 0.1],  # Class 0
            [0.05, 0.05], [0.15, 0.05], [0.1, 0.0]  # Class 1: heavily overlapping
        ])
        overlapping_labels = torch.tensor([0, 0, 0, 1, 1, 1])
        
        overlap_score = self.analyzer.compute_silhouette_score(overlapping_features, overlapping_labels)
        
        # Overlapping clusters should have lower score
        assert overlap_score < score
    
    def test_prototype_stability_analysis(self):
        """Test prototype stability under perturbation."""
        support_features = torch.randn(30, 64)  # 3 classes, 10 examples each
        support_labels = torch.arange(3).repeat_interleave(10)
        
        stability = self.analyzer.analyze_prototype_stability(
            support_features, support_labels, n_perturbations=50
        )
        
        assert 'mean_stability' in stability
        assert 'per_class_stability' in stability
        assert 'stability_variance' in stability
        
        # Stability should be between 0 and 1
        assert 0 <= stability['mean_stability'] <= 1
        assert len(stability['per_class_stability']) == 3
    
    def test_prototype_discriminability(self):
        """Test prototype discriminability assessment."""
        # Highly discriminable prototypes (orthogonal)
        prototypes = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0], 
            [0.0, 0.0, 1.0]
        ])
        
        discriminability = self.analyzer.compute_prototype_discriminability(prototypes)
        
        assert discriminability > 0.8  # Should be high for orthogonal vectors
        
        # Low discriminability (similar prototypes)
        similar_prototypes = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.8, 0.2, 0.0]
        ])
        
        low_discriminability = self.analyzer.compute_prototype_discriminability(similar_prototypes)
        
        assert low_discriminability < discriminability


class TestTaskDifficultyEstimator:
    """Test TaskDifficultyEstimator functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.estimator = TaskDifficultyEstimator()
    
    def test_estimate_task_difficulty_basic(self):
        """Test basic task difficulty estimation."""
        # Easy task: well-separated classes
        easy_features = torch.tensor([
            [1.0, 0.0], [1.1, 0.0], [1.0, 0.1],  # Class 0
            [0.0, 1.0], [0.0, 1.1], [0.1, 1.0],  # Class 1
            [-1.0, 0.0], [-1.1, 0.0], [-1.0, 0.1]  # Class 2
        ])
        easy_labels = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])
        
        easy_difficulty = self.estimator.estimate_task_difficulty(easy_features, easy_labels)
        
        # Hard task: overlapping classes
        hard_features = torch.randn(9, 2) * 0.1  # All classes overlap heavily
        hard_labels = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])
        
        hard_difficulty = self.estimator.estimate_task_difficulty(hard_features, hard_labels)
        
        # Easy task should have lower difficulty score
        assert easy_difficulty['overall_difficulty'] < hard_difficulty['overall_difficulty']
        
        # Check that all components are present
        for key in ['class_separability', 'feature_complexity', 'within_class_variance', 'overall_difficulty']:
            assert key in easy_difficulty
            assert key in hard_difficulty
    
    def test_compute_class_separability(self):
        """Test class separability computation."""
        # Highly separable classes
        separable_features = torch.tensor([
            [2.0, 0.0], [2.1, 0.0],  # Class 0
            [0.0, 2.0], [0.0, 2.1]   # Class 1
        ])
        separable_labels = torch.tensor([0, 0, 1, 1])
        
        high_sep = self.estimator.compute_class_separability(separable_features, separable_labels)
        
        # Overlapping classes
        overlapping_features = torch.tensor([
            [0.0, 0.0], [0.1, 0.1],  # Class 0
            [0.05, 0.05], [0.15, 0.15]  # Class 1
        ])
        overlapping_labels = torch.tensor([0, 0, 1, 1])
        
        low_sep = self.estimator.compute_class_separability(overlapping_features, overlapping_labels)
        
        # High separability should be greater
        assert high_sep > low_sep
        assert high_sep > 0.5
    
    def test_compute_feature_complexity(self):
        """Test feature complexity computation."""
        # Simple features (low dimensionality, structured)
        simple_features = torch.tensor([
            [1.0, 0.0], [1.1, 0.0], [0.9, 0.0],  # Nearly 1D
            [0.0, 1.0], [0.0, 1.1], [0.0, 0.9]
        ])
        
        simple_complexity = self.estimator.compute_feature_complexity(simple_features)
        
        # Complex features (high-dimensional, random)
        complex_features = torch.randn(6, 100)  # High-dimensional random
        
        complex_complexity = self.estimator.compute_feature_complexity(complex_features)
        
        # Complex features should have higher complexity
        assert complex_complexity > simple_complexity
    
    def test_estimate_required_shots(self):
        """Test estimation of required shots for task."""
        # Task requiring few shots (simple, separable)
        easy_features = torch.tensor([
            [5.0, 0.0], [5.1, 0.1],  # Class 0: well-separated
            [0.0, 5.0], [0.1, 5.1]   # Class 1: well-separated
        ])
        easy_labels = torch.tensor([0, 0, 1, 1])
        
        easy_shots = self.estimator.estimate_required_shots(easy_features, easy_labels)
        
        # Task requiring many shots (complex, overlapping)
        hard_features = torch.randn(20, 50)  # High-dim random
        hard_labels = torch.randint(0, 5, (20,))  # 5 classes, potential overlap
        
        hard_shots = self.estimator.estimate_required_shots(hard_features, hard_labels)
        
        # Hard task should require more shots
        assert hard_shots['estimated_k_shot'] >= easy_shots['estimated_k_shot']
        
        # Check output structure
        assert 'estimated_k_shot' in easy_shots
        assert 'confidence' in easy_shots
        assert 'reasoning' in easy_shots
    
    def test_difficulty_calibration(self):
        """Test that difficulty estimates are well-calibrated."""
        # Create tasks with known difficulty ordering
        tasks = []
        
        # Task 1: Very easy (distant, tight clusters)
        task1_features = torch.tensor([
            [10.0, 0.0], [10.1, 0.0], [9.9, 0.0],   # Class 0
            [0.0, 10.0], [0.0, 10.1], [0.0, 9.9]    # Class 1  
        ])
        task1_labels = torch.tensor([0, 0, 0, 1, 1, 1])
        tasks.append((task1_features, task1_labels, 'very_easy'))
        
        # Task 2: Medium (moderate separation)
        task2_features = torch.tensor([
            [2.0, 0.0], [2.2, 0.1], [1.8, -0.1],   # Class 0
            [0.0, 2.0], [0.1, 2.2], [-0.1, 1.8]    # Class 1
        ])
        task2_labels = torch.tensor([0, 0, 0, 1, 1, 1])
        tasks.append((task2_features, task2_labels, 'medium'))
        
        # Task 3: Hard (overlapping clusters)
        task3_features = torch.randn(6, 2) * 0.5
        task3_labels = torch.tensor([0, 0, 0, 1, 1, 1])
        tasks.append((task3_features, task3_labels, 'hard'))
        
        # Estimate difficulties
        difficulties = []
        for features, labels, _ in tasks:
            diff = self.estimator.estimate_task_difficulty(features, labels)
            difficulties.append(diff['overall_difficulty'])
        
        # Should be in increasing order of difficulty
        assert difficulties[0] < difficulties[1] < difficulties[2]


class TestUncertaintyQuantifier:
    """Test UncertaintyQuantifier functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.quantifier = UncertaintyQuantifier()
    
    def test_compute_prediction_entropy(self):
        """Test prediction entropy computation."""
        # High confidence predictions (low entropy)
        high_conf_logits = torch.tensor([
            [5.0, 0.0, 0.0],   # Very confident class 0
            [0.0, 5.0, 0.0],   # Very confident class 1
            [0.0, 0.0, 5.0]    # Very confident class 2
        ])
        
        high_conf_entropy = self.quantifier.compute_prediction_entropy(high_conf_logits)
        
        # Low confidence predictions (high entropy)
        low_conf_logits = torch.tensor([
            [1.0, 0.9, 0.8],   # Similar scores
            [0.7, 1.0, 0.9],   # Similar scores
            [0.8, 0.7, 1.0]    # Similar scores
        ])
        
        low_conf_entropy = self.quantifier.compute_prediction_entropy(low_conf_logits)
        
        # Low confidence should have higher entropy
        assert low_conf_entropy.mean() > high_conf_entropy.mean()
        assert high_conf_entropy.mean() < 0.5  # Should be low entropy
    
    def test_compute_prediction_variance(self):
        """Test prediction variance computation with multiple forward passes."""
        # Mock model that returns different predictions each time
        def stochastic_model(x, n_samples=10):
            batch_size, n_classes = x.shape[0], 3
            predictions = []
            
            for _ in range(n_samples):
                # Add noise to simulate stochastic predictions
                logits = torch.randn(batch_size, n_classes) + x.sum(dim=-1, keepdim=True)
                predictions.append(F.softmax(logits, dim=-1))
            
            return torch.stack(predictions, dim=0)  # [n_samples, batch_size, n_classes]
        
        test_input = torch.randn(5, 10)
        variance = self.quantifier.compute_prediction_variance(stochastic_model, test_input)
        
        assert variance.shape == (5,)  # One variance per example
        assert (variance >= 0).all()   # Variance should be non-negative
    
    def test_compute_epistemic_uncertainty(self):
        """Test epistemic uncertainty computation."""
        # Create mock ensemble of models
        class MockEnsemble:
            def __init__(self, n_models=5):
                self.n_models = n_models
            
            def __call__(self, x):
                predictions = []
                for i in range(self.n_models):
                    # Each model has slight bias
                    logits = torch.randn(x.shape[0], 3) + i * 0.1
                    predictions.append(F.softmax(logits, dim=-1))
                return torch.stack(predictions, dim=0)
        
        ensemble = MockEnsemble()
        test_input = torch.randn(8, 10)
        
        epistemic_uncertainty = self.quantifier.compute_epistemic_uncertainty(ensemble, test_input)
        
        assert epistemic_uncertainty.shape == (8,)
        assert (epistemic_uncertainty >= 0).all()
    
    def test_compute_aleatoric_uncertainty(self):
        """Test aleatoric uncertainty computation."""
        # Model that predicts both mean and variance
        class AleatroicModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.mean_head = nn.Linear(10, 3)
                self.var_head = nn.Linear(10, 3)
            
            def forward(self, x):
                mean = self.mean_head(x)
                log_var = self.var_head(x)  # Predict log variance for stability
                return mean, log_var.exp()
        
        model = AleatroicModel()
        test_input = torch.randn(6, 10)
        
        aleatoric_uncertainty = self.quantifier.compute_aleatoric_uncertainty(model, test_input)
        
        assert aleatoric_uncertainty.shape == (6,)
        assert (aleatoric_uncertainty >= 0).all()
    
    def test_uncertainty_calibration(self):
        """Test uncertainty calibration assessment."""
        # Generate predictions with known uncertainty
        n_samples = 100
        confidences = torch.rand(n_samples)
        
        # Generate accuracies correlated with confidence
        accuracies = (confidences > 0.5).float() + torch.randn(n_samples) * 0.1
        accuracies = torch.clamp(accuracies, 0, 1)
        
        calibration = self.quantifier.assess_uncertainty_calibration(confidences, accuracies)
        
        assert 'calibration_error' in calibration
        assert 'reliability_diagram' in calibration
        assert 'confidence_intervals' in calibration
        
        # Calibration error should be between 0 and 1
        assert 0 <= calibration['calibration_error'] <= 1


class TestFewShotMetrics:
    """Test FewShotMetrics comprehensive evaluation."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.metrics = FewShotMetrics()
    
    def test_compute_accuracy_metrics(self):
        """Test accuracy metric computation."""
        # Perfect predictions
        perfect_predictions = torch.tensor([0, 1, 2, 0, 1, 2])
        perfect_targets = torch.tensor([0, 1, 2, 0, 1, 2])
        
        perfect_metrics = self.metrics.compute_accuracy_metrics(perfect_predictions, perfect_targets)
        
        assert perfect_metrics['accuracy'] == 1.0
        assert perfect_metrics['per_class_accuracy'].mean() == 1.0
        
        # Random predictions
        random_predictions = torch.randint(0, 3, (60,))
        random_targets = torch.randint(0, 3, (60,))
        
        random_metrics = self.metrics.compute_accuracy_metrics(random_predictions, random_targets)
        
        # Random should be worse than perfect
        assert random_metrics['accuracy'] < perfect_metrics['accuracy']
        assert 'per_class_accuracy' in random_metrics
        assert 'confusion_matrix' in random_metrics
    
    def test_compute_few_shot_specific_metrics(self):
        """Test few-shot specific metrics."""
        # Simulate episode results
        episode_results = [
            {'accuracy': 0.8, 'n_way': 5, 'k_shot': 1},
            {'accuracy': 0.85, 'n_way': 5, 'k_shot': 1},
            {'accuracy': 0.75, 'n_way': 5, 'k_shot': 1},
            {'accuracy': 0.9, 'n_way': 5, 'k_shot': 1},
            {'accuracy': 0.7, 'n_way': 5, 'k_shot': 1}
        ]
        
        metrics = self.metrics.compute_few_shot_specific_metrics(episode_results)
        
        assert 'mean_accuracy' in metrics
        assert 'confidence_interval' in metrics
        assert 'episode_variance' in metrics
        assert 'statistical_significance' in metrics
        
        # Mean should be around 0.8
        assert abs(metrics['mean_accuracy'] - 0.8) < 0.05
    
    def test_compute_convergence_metrics(self):
        """Test convergence analysis metrics."""
        # Simulate learning curve
        accuracies = [0.2, 0.4, 0.6, 0.75, 0.8, 0.82, 0.83, 0.83, 0.83]
        
        convergence = self.metrics.compute_convergence_metrics(accuracies)
        
        assert 'convergence_episode' in convergence
        assert 'final_performance' in convergence
        assert 'learning_rate' in convergence
        assert 'stability_score' in convergence
        
        # Should detect convergence around episode 5-6
        assert 4 <= convergence['convergence_episode'] <= 7
    
    def test_compute_robustness_metrics(self):
        """Test robustness metrics computation."""
        # Test with perturbations
        clean_results = [0.8, 0.82, 0.78, 0.85, 0.79]
        noisy_results = [0.75, 0.77, 0.73, 0.8, 0.74]  # Slightly worse with noise
        
        robustness = self.metrics.compute_robustness_metrics(clean_results, noisy_results)
        
        assert 'robustness_score' in robustness
        assert 'performance_degradation' in robustness
        assert 'noise_sensitivity' in robustness
        
        # Should detect performance degradation
        assert robustness['performance_degradation'] > 0
        assert 0 <= robustness['robustness_score'] <= 1


class TestResearchMetrics:
    """Test ResearchMetrics for publication-grade evaluation."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.metrics = ResearchMetrics()
    
    def test_statistical_significance_testing(self):
        """Test statistical significance testing."""
        # Two methods with different performance
        method_a_results = [0.8, 0.82, 0.78, 0.85, 0.79, 0.81, 0.83, 0.77]
        method_b_results = [0.7, 0.72, 0.68, 0.75, 0.69, 0.71, 0.73, 0.67]  # Clearly worse
        
        significance = self.metrics.test_statistical_significance(method_a_results, method_b_results)
        
        assert 'p_value' in significance
        assert 't_statistic' in significance
        assert 'effect_size' in significance
        assert 'confidence_interval' in significance
        
        # Should detect significant difference
        assert significance['p_value'] < 0.05
        assert significance['effect_size'] > 0.5  # Large effect size
    
    def test_meta_analysis_metrics(self):
        """Test meta-analysis across datasets/tasks."""
        # Results across multiple datasets
        dataset_results = {
            'miniImageNet': [0.8, 0.82, 0.78, 0.85],
            'tieredImageNet': [0.75, 0.77, 0.73, 0.8],
            'CIFAR-FS': [0.85, 0.87, 0.83, 0.9],
            'Omniglot': [0.95, 0.97, 0.93, 0.98]
        }
        
        meta_analysis = self.metrics.compute_meta_analysis_metrics(dataset_results)
        
        assert 'overall_mean' in meta_analysis
        assert 'between_dataset_variance' in meta_analysis
        assert 'within_dataset_variance' in meta_analysis
        assert 'heterogeneity_index' in meta_analysis
        assert 'dataset_rankings' in meta_analysis
        
        # Omniglot should rank highest, tieredImageNet lowest
        rankings = meta_analysis['dataset_rankings']
        assert rankings.index('Omniglot') < rankings.index('tieredImageNet')
    
    def test_compute_publication_metrics(self):
        """Test computation of publication-ready metrics."""
        # Comprehensive results
        results = {
            'accuracies': [0.8, 0.82, 0.78, 0.85, 0.79, 0.81, 0.83, 0.77, 0.84, 0.8],
            'task_configs': [
                {'n_way': 5, 'k_shot': 1, 'dataset': 'miniImageNet'},
                {'n_way': 5, 'k_shot': 5, 'dataset': 'miniImageNet'},
                {'n_way': 20, 'k_shot': 1, 'dataset': 'Omniglot'}
            ],
            'baseline_comparisons': {
                'ProtoNet': [0.75, 0.77, 0.73],
                'MAML': [0.78, 0.8, 0.76]
            }
        }
        
        pub_metrics = self.metrics.compute_publication_metrics(results)
        
        # Should include all standard publication metrics
        required_fields = [
            'mean_accuracy', 'confidence_interval', 'standard_deviation',
            'baseline_improvements', 'statistical_significance', 'effect_sizes'
        ]
        
        for field in required_fields:
            assert field in pub_metrics
    
    def test_reproducibility_metrics(self):
        """Test reproducibility assessment metrics."""
        # Multiple runs with same setup
        run_results = [
            [0.8, 0.82, 0.78, 0.85],  # Run 1
            [0.81, 0.83, 0.79, 0.86], # Run 2  
            [0.79, 0.81, 0.77, 0.84], # Run 3
        ]
        
        repro_metrics = self.metrics.assess_reproducibility(run_results)
        
        assert 'inter_run_variance' in repro_metrics
        assert 'reproducibility_score' in repro_metrics
        assert 'run_consistency' in repro_metrics
        
        # Should have high reproducibility (similar results)
        assert repro_metrics['reproducibility_score'] > 0.8


class TestEvaluationSuite:
    """Test EvaluationSuite comprehensive evaluation framework."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.suite = EvaluationSuite()
        
        # Mock model for testing
        self.mock_model = MagicMock()
        self.mock_model.return_value = torch.randn(10, 5)  # 10 queries, 5 classes
    
    def test_run_comprehensive_evaluation(self):
        """Test comprehensive evaluation pipeline."""
        # Mock episode data
        episodes = []
        for i in range(20):
            episode = {
                'support_x': torch.randn(25, 64),
                'support_y': torch.arange(5).repeat(5),
                'query_x': torch.randn(75, 64),
                'query_y': torch.arange(5).repeat(15),
                'n_way': 5,
                'k_shot': 5
            }
            episodes.append(episode)
        
        results = self.suite.run_comprehensive_evaluation(self.mock_model, episodes)
        
        # Should include all evaluation components
        expected_sections = [
            'accuracy_metrics', 'prototype_analysis', 'task_difficulty',
            'uncertainty_metrics', 'few_shot_metrics', 'statistical_analysis'
        ]
        
        for section in expected_sections:
            assert section in results
    
    def test_benchmark_comparison(self):
        """Test benchmarking against baselines."""
        # Mock multiple models
        models = {
            'our_method': MagicMock(),
            'protonet': MagicMock(), 
            'maml': MagicMock()
        }
        
        for model in models.values():
            model.return_value = torch.randn(15, 5)
        
        # Mock episodes
        episodes = [
            {
                'support_x': torch.randn(25, 64),
                'support_y': torch.arange(5).repeat(5),
                'query_x': torch.randn(75, 64),
                'query_y': torch.arange(5).repeat(15)
            }
            for _ in range(10)
        ]
        
        comparison = self.suite.benchmark_comparison(models, episodes)
        
        assert 'model_rankings' in comparison
        assert 'pairwise_comparisons' in comparison
        assert 'statistical_significance' in comparison
        
        # Should rank all three models
        assert len(comparison['model_rankings']) == 3
    
    def test_ablation_study(self):
        """Test ablation study functionality."""
        # Mock model variants
        model_variants = {
            'full_model': MagicMock(),
            'no_attention': MagicMock(),
            'no_adaptation': MagicMock(),
            'baseline': MagicMock()
        }
        
        for model in model_variants.values():
            model.return_value = torch.randn(15, 5)
        
        episodes = [{'support_x': torch.randn(25, 64), 'support_y': torch.arange(5).repeat(5),
                    'query_x': torch.randn(75, 64), 'query_y': torch.arange(5).repeat(15)} 
                   for _ in range(15)]
        
        ablation = self.suite.run_ablation_study(model_variants, episodes)
        
        assert 'component_importance' in ablation
        assert 'performance_drops' in ablation
        assert 'component_rankings' in ablation


class TestMetaLearningBenchmark:
    """Test MetaLearningBenchmark standardized evaluation."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.benchmark = MetaLearningBenchmark()
    
    def test_standard_benchmark_protocols(self):
        """Test standard benchmarking protocols."""
        protocols = self.benchmark.get_standard_protocols()
        
        # Should include standard protocols
        expected_protocols = [
            '5way_1shot_miniImageNet',
            '5way_5shot_miniImageNet', 
            '20way_1shot_Omniglot'
        ]
        
        for protocol in expected_protocols:
            assert protocol in protocols
            
            config = protocols[protocol]
            assert 'n_way' in config
            assert 'k_shot' in config
            assert 'n_episodes' in config
            assert 'dataset' in config
    
    def test_evaluate_on_benchmark(self):
        """Test evaluation on standard benchmark."""
        mock_model = MagicMock()
        mock_model.return_value = torch.randn(75, 5)  # 5-way classification
        
        # Evaluate on miniImageNet 5-way 1-shot
        results = self.benchmark.evaluate_on_benchmark(
            mock_model, 
            protocol='5way_1shot_miniImageNet',
            n_episodes=100
        )
        
        assert 'protocol' in results
        assert 'mean_accuracy' in results
        assert 'confidence_interval' in results
        assert 'episode_results' in results
        
        # Should evaluate specified number of episodes
        assert len(results['episode_results']) == 100
    
    def test_generate_benchmark_report(self):
        """Test benchmark report generation."""
        # Mock results across multiple protocols
        results = {
            '5way_1shot_miniImageNet': {'mean_accuracy': 0.8, 'ci': [0.78, 0.82]},
            '5way_5shot_miniImageNet': {'mean_accuracy': 0.85, 'ci': [0.83, 0.87]},
            '20way_1shot_Omniglot': {'mean_accuracy': 0.95, 'ci': [0.94, 0.96]}
        }
        
        report = self.benchmark.generate_benchmark_report(results)
        
        assert 'summary_table' in report
        assert 'detailed_analysis' in report
        assert 'recommendations' in report
        
        # Should format results properly
        assert 'miniImageNet' in report['summary_table']
        assert 'Omniglot' in report['summary_table']


class TestUtilityFunctions:
    """Test utility functions for evaluation."""
    
    def test_compute_prototype_quality_function(self):
        """Test standalone prototype quality computation."""
        support_features = torch.randn(30, 64)
        support_labels = torch.arange(5).repeat(6)  # 5 classes, 6 examples each
        
        # Compute prototypes
        prototypes = torch.zeros(5, 64)
        for k in range(5):
            mask = support_labels == k
            prototypes[k] = support_features[mask].mean(dim=0)
        
        quality = compute_prototype_quality(support_features, support_labels, prototypes)
        
        assert 'quality_score' in quality
        assert 'intra_class_variance' in quality
        assert 'inter_class_separation' in quality
        
        # Quality score should be between 0 and 1
        assert 0 <= quality['quality_score'] <= 1
    
    def test_estimate_task_difficulty_function(self):
        """Test standalone task difficulty estimation."""
        # Easy task
        easy_features = torch.tensor([
            [1.0, 0.0], [1.1, 0.1],  # Class 0
            [0.0, 1.0], [0.1, 1.1]   # Class 1
        ])
        easy_labels = torch.tensor([0, 0, 1, 1])
        
        difficulty = estimate_task_difficulty(easy_features, easy_labels)
        
        assert 'difficulty_score' in difficulty
        assert 'estimated_performance' in difficulty
        assert 'complexity_factors' in difficulty
        
        # Should predict reasonable difficulty
        assert 0 <= difficulty['difficulty_score'] <= 1
    
    def test_quantify_prediction_uncertainty_function(self):
        """Test standalone uncertainty quantification."""
        # Mock predictions with varying confidence
        predictions = torch.tensor([
            [0.9, 0.05, 0.05],   # High confidence
            [0.4, 0.3, 0.3],     # Low confidence
            [0.8, 0.1, 0.1]      # Medium-high confidence
        ])
        
        uncertainty = quantify_prediction_uncertainty(predictions)
        
        assert 'entropy' in uncertainty
        assert 'max_probability' in uncertainty  
        assert 'uncertainty_score' in uncertainty
        
        # Should have 3 uncertainty scores
        assert len(uncertainty['uncertainty_score']) == 3
        
        # Low confidence prediction should have higher uncertainty
        assert uncertainty['uncertainty_score'][1] > uncertainty['uncertainty_score'][0]


class TestRealWorldScenarios:
    """Test evaluation on realistic meta-learning scenarios."""
    
    def test_cross_domain_evaluation(self):
        """Test evaluation across different domains."""
        suite = EvaluationSuite()
        
        # Simulate cross-domain results
        domain_results = {
            'vision': {'accuracies': [0.8, 0.82, 0.78, 0.85]},
            'text': {'accuracies': [0.7, 0.72, 0.68, 0.75]},
            'speech': {'accuracies': [0.75, 0.77, 0.73, 0.8]}
        }
        
        cross_domain = suite.evaluate_cross_domain_transfer(domain_results)
        
        assert 'domain_performance' in cross_domain
        assert 'transfer_scores' in cross_domain
        assert 'domain_difficulty_ranking' in cross_domain
    
    def test_few_shot_learning_progression(self):
        """Test evaluation of learning progression (1-shot to many-shot)."""
        metrics = FewShotMetrics()
        
        # Simulate progression from 1-shot to 10-shot
        shot_progression = {
            1: [0.6, 0.62, 0.58, 0.65],
            2: [0.7, 0.72, 0.68, 0.75],
            5: [0.8, 0.82, 0.78, 0.85],
            10: [0.85, 0.87, 0.83, 0.9]
        }
        
        progression = metrics.analyze_shot_progression(shot_progression)
        
        assert 'learning_curve' in progression
        assert 'diminishing_returns_point' in progression
        assert 'optimal_shot_count' in progression
        
        # Should show improvement with more shots
        curve = progression['learning_curve']
        assert curve[1] < curve[2] < curve[5] < curve[10]


if __name__ == "__main__":
    pytest.main([__file__])