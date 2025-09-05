"""
💰 SUPPORT THIS RESEARCH - PLEASE DONATE! 💰

🙏 If this library helps your research or project, please consider donating:
💳 PayPal: https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS
⭐ GitHub Sponsors: https://github.com/sponsors/benedictchen

👨‍💻 Author: Benedict Chen
💰 Donations: Help support this work! Buy me a coffee ☕, beer 🍺, lamborghini 🏎️, or private island 🏝️
💖 Please consider recurring donations to fully support continued research

Your support enables cutting-edge AI research for everyone! 🚀

Comprehensive Uncertainty Components Coverage Tests
=================================================

Complete test coverage for uncertainty estimation in few-shot learning.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List
from unittest.mock import patch, MagicMock

from meta_learning.few_shot_modules.uncertainty_components import (
    UncertaintyConfig, UncertaintyAwareDistance, MonteCarloDropout,
    DeepEnsemble, EvidentialLearning, create_uncertainty_aware_distance
)


class TestUncertaintyConfig:
    """Test UncertaintyConfig class."""
    
    def test_uncertainty_config_creation_default(self):
        """Test UncertaintyConfig creation with defaults."""
        config = UncertaintyConfig()
        
        assert config.method == "monte_carlo_dropout"
        assert config.n_samples == 10
        assert config.dropout_rate == 0.1
        assert config.ensemble_size == 5
        assert config.temperature == 1.0
        assert config.enable_aleatoric == True
        assert config.enable_epistemic == True
    
    def test_uncertainty_config_creation_custom(self):
        """Test UncertaintyConfig creation with custom values."""
        config = UncertaintyConfig(
            method="deep_ensemble",
            n_samples=20,
            dropout_rate=0.2,
            ensemble_size=10,
            temperature=2.0,
            enable_aleatoric=False,
            enable_epistemic=True
        )
        
        assert config.method == "deep_ensemble"
        assert config.n_samples == 20
        assert config.dropout_rate == 0.2
        assert config.ensemble_size == 10
        assert config.temperature == 2.0
        assert config.enable_aleatoric == False
        assert config.enable_epistemic == True


class TestMonteCarloDropout:
    """Test MonteCarloDropout class."""
    
    def test_monte_carlo_dropout_creation(self):
        """Test MonteCarloDropout creation."""
        config = UncertaintyConfig(dropout_rate=0.15, n_samples=5)
        mc_dropout = MonteCarloDropout(config)
        
        assert mc_dropout.config.dropout_rate == 0.15
        assert mc_dropout.config.n_samples == 5
        assert isinstance(mc_dropout.dropout, nn.Dropout)
    
    def test_forward_with_uncertainty(self):
        """Test forward pass with uncertainty estimation."""
        config = UncertaintyConfig(dropout_rate=0.1, n_samples=3)
        mc_dropout = MonteCarloDropout(config)
        
        # Mock distance function
        def mock_distance_fn(support_features, support_labels, query_features):
            batch_size = query_features.size(0)
            n_classes = len(torch.unique(support_labels))
            return torch.randn(batch_size, n_classes)
        
        support_features = torch.randn(15, 32)
        support_labels = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
        query_features = torch.randn(9, 32)
        
        result = mc_dropout.forward_with_uncertainty(
            mock_distance_fn, support_features, support_labels, query_features
        )
        
        assert isinstance(result, dict)
        assert "logits" in result
        assert "probabilities" in result
        assert "total_uncertainty" in result
        assert "epistemic_uncertainty" in result
        assert "aleatoric_uncertainty" in result
        
        assert result["logits"].shape == (9, 3)
        assert result["total_uncertainty"].shape == (9,)
    
    def test_enable_uncertainty_mode(self):
        """Test enabling uncertainty mode."""
        config = UncertaintyConfig()
        mc_dropout = MonteCarloDropout(config)
        
        # Create a simple model to test with
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.Dropout(0.1),
            nn.Linear(20, 5)
        )
        
        original_training = model.training
        mc_dropout.enable_uncertainty_mode(model)
        
        # Should set model to train mode for stochastic behavior
        assert model.training == True
        
        # Restore original mode
        model.train(original_training)
    
    def test_compute_uncertainty_statistics(self):
        """Test uncertainty statistics computation."""
        config = UncertaintyConfig(enable_aleatoric=True, enable_epistemic=True)
        mc_dropout = MonteCarloDropout(config)
        
        # Create sample predictions from multiple forward passes
        n_samples, n_query, n_classes = 5, 6, 3
        logit_samples = torch.randn(n_samples, n_query, n_classes)
        
        stats = mc_dropout._compute_uncertainty_statistics(logit_samples)
        
        assert isinstance(stats, dict)
        assert "mean_logits" in stats
        assert "mean_probabilities" in stats
        assert "total_uncertainty" in stats
        assert "epistemic_uncertainty" in stats
        assert "aleatoric_uncertainty" in stats
        
        assert stats["mean_logits"].shape == (n_query, n_classes)
        assert stats["total_uncertainty"].shape == (n_query,)


class TestDeepEnsemble:
    """Test DeepEnsemble class."""
    
    def test_deep_ensemble_creation(self):
        """Test DeepEnsemble creation."""
        config = UncertaintyConfig(ensemble_size=3, method="deep_ensemble")
        ensemble = DeepEnsemble(config)
        
        assert ensemble.config.ensemble_size == 3
        assert len(ensemble.models) == 0  # No models added yet
    
    def test_add_model_to_ensemble(self):
        """Test adding models to ensemble."""
        config = UncertaintyConfig(ensemble_size=3)
        ensemble = DeepEnsemble(config)
        
        model1 = nn.Linear(10, 5)
        model2 = nn.Linear(10, 5)
        
        ensemble.add_model(model1)
        ensemble.add_model(model2)
        
        assert len(ensemble.models) == 2
    
    def test_forward_with_uncertainty_ensemble(self):
        """Test forward pass with ensemble uncertainty."""
        config = UncertaintyConfig(ensemble_size=2)
        ensemble = DeepEnsemble(config)
        
        # Add models to ensemble
        model1 = nn.Linear(32, 64)
        model2 = nn.Linear(32, 64)
        ensemble.add_model(model1)
        ensemble.add_model(model2)
        
        # Mock distance function
        def mock_distance_fn(support_features, support_labels, query_features):
            return torch.randn(query_features.size(0), 3)
        
        support_features = torch.randn(15, 32)
        support_labels = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
        query_features = torch.randn(9, 32)
        
        result = ensemble.forward_with_uncertainty(
            mock_distance_fn, support_features, support_labels, query_features
        )
        
        assert isinstance(result, dict)
        assert "logits" in result
        assert "ensemble_predictions" in result
        assert "ensemble_uncertainty" in result
        assert result["logits"].shape == (9, 3)
    
    def test_ensemble_empty_models(self):
        """Test ensemble behavior with no models."""
        config = UncertaintyConfig()
        ensemble = DeepEnsemble(config)
        
        def mock_distance_fn(support_features, support_labels, query_features):
            return torch.randn(query_features.size(0), 3)
        
        support_features = torch.randn(6, 16)
        support_labels = torch.tensor([0, 0, 1, 1, 2, 2])
        query_features = torch.randn(3, 16)
        
        # Should handle empty ensemble gracefully
        result = ensemble.forward_with_uncertainty(
            mock_distance_fn, support_features, support_labels, query_features
        )
        
        assert isinstance(result, dict)
        # Should still return some form of result


class TestEvidentialLearning:
    """Test EvidentialLearning class."""
    
    def test_evidential_learning_creation(self):
        """Test EvidentialLearning creation."""
        config = UncertaintyConfig(method="evidential")
        evidential = EvidentialLearning(config)
        
        assert evidential.config.method == "evidential"
    
    def test_dirichlet_parameters(self):
        """Test Dirichlet parameter computation."""
        config = UncertaintyConfig()
        evidential = EvidentialLearning(config)
        
        logits = torch.randn(5, 3)
        alpha = evidential._compute_dirichlet_parameters(logits)
        
        assert alpha.shape == logits.shape
        assert torch.all(alpha > 0)  # Dirichlet parameters should be positive
    
    def test_evidential_uncertainty(self):
        """Test evidential uncertainty computation."""
        config = UncertaintyConfig()
        evidential = EvidentialLearning(config)
        
        alpha = torch.tensor([
            [2.0, 3.0, 1.0],
            [5.0, 1.0, 2.0],
            [1.0, 1.0, 1.0]
        ])
        
        uncertainty = evidential._compute_evidential_uncertainty(alpha)
        
        assert uncertainty.shape == (3,)
        assert torch.all(uncertainty >= 0)  # Uncertainty should be non-negative
    
    def test_forward_with_evidential_uncertainty(self):
        """Test forward pass with evidential uncertainty."""
        config = UncertaintyConfig()
        evidential = EvidentialLearning(config)
        
        def mock_distance_fn(support_features, support_labels, query_features):
            return torch.randn(query_features.size(0), 3)
        
        support_features = torch.randn(12, 24)
        support_labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
        query_features = torch.randn(6, 24)
        
        result = evidential.forward_with_evidential_uncertainty(
            mock_distance_fn, support_features, support_labels, query_features
        )
        
        assert isinstance(result, dict)
        assert "logits" in result
        assert "evidential_uncertainty" in result
        assert "dirichlet_alpha" in result
        assert result["logits"].shape == (6, 3)


class TestUncertaintyAwareDistance:
    """Test UncertaintyAwareDistance class."""
    
    def test_uncertainty_aware_distance_creation(self):
        """Test UncertaintyAwareDistance creation."""
        config = UncertaintyConfig(method="monte_carlo_dropout")
        distance = UncertaintyAwareDistance(config, distance_type="euclidean")
        
        assert distance.config.method == "monte_carlo_dropout"
        assert distance.distance_type == "euclidean"
        assert isinstance(distance.uncertainty_estimator, MonteCarloDropout)
    
    def test_uncertainty_aware_distance_cosine(self):
        """Test UncertaintyAwareDistance with cosine distance."""
        config = UncertaintyConfig(method="deep_ensemble")
        distance = UncertaintyAwareDistance(config, distance_type="cosine")
        
        assert distance.distance_type == "cosine"
        assert isinstance(distance.uncertainty_estimator, DeepEnsemble)
    
    def test_uncertainty_aware_distance_evidential(self):
        """Test UncertaintyAwareDistance with evidential learning."""
        config = UncertaintyConfig(method="evidential")
        distance = UncertaintyAwareDistance(config, distance_type="euclidean")
        
        assert isinstance(distance.uncertainty_estimator, EvidentialLearning)
    
    def test_base_distance_computation_euclidean(self):
        """Test base distance computation with Euclidean distance."""
        config = UncertaintyConfig()
        distance = UncertaintyAwareDistance(config, distance_type="euclidean")
        
        support_features = torch.randn(9, 16)
        support_labels = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])
        query_features = torch.randn(6, 16)
        
        logits = distance._compute_base_distance(
            support_features, support_labels, query_features
        )
        
        assert logits.shape == (6, 3)
    
    def test_base_distance_computation_cosine(self):
        """Test base distance computation with cosine distance."""
        config = UncertaintyConfig()
        distance = UncertaintyAwareDistance(config, distance_type="cosine")
        
        support_features = torch.randn(6, 20)
        support_labels = torch.tensor([0, 0, 1, 1, 2, 2])
        query_features = torch.randn(3, 20)
        
        logits = distance._compute_base_distance(
            support_features, support_labels, query_features
        )
        
        assert logits.shape == (3, 3)
    
    def test_forward_with_full_uncertainty(self):
        """Test full forward pass with uncertainty."""
        config = UncertaintyConfig(method="monte_carlo_dropout", n_samples=3)
        distance = UncertaintyAwareDistance(config, distance_type="euclidean")
        
        support_features = torch.randn(12, 32)
        support_labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
        query_features = torch.randn(9, 32)
        
        result = distance.forward(support_features, support_labels, query_features)
        
        assert isinstance(result, dict)
        assert "logits" in result
        assert "probabilities" in result
        assert "total_uncertainty" in result
        
        assert result["logits"].shape == (9, 3)
        assert result["total_uncertainty"].shape == (9,)
    
    def test_temperature_scaling(self):
        """Test temperature scaling in uncertainty computation."""
        config = UncertaintyConfig(temperature=2.0)
        distance = UncertaintyAwareDistance(config)
        
        logits = torch.tensor([[2.0, 1.0, 0.5], [1.5, 2.5, 1.0]])
        scaled_logits = distance._apply_temperature_scaling(logits)
        
        # Temperature scaling should divide logits by temperature
        expected_logits = logits / 2.0
        assert torch.allclose(scaled_logits, expected_logits)


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_create_uncertainty_aware_distance_default(self):
        """Test creating uncertainty-aware distance with defaults."""
        distance = create_uncertainty_aware_distance()
        
        assert isinstance(distance, UncertaintyAwareDistance)
        assert distance.config.method == "monte_carlo_dropout"
        assert distance.distance_type == "euclidean"
    
    def test_create_uncertainty_aware_distance_custom(self):
        """Test creating uncertainty-aware distance with custom config."""
        distance = create_uncertainty_aware_distance(
            method="deep_ensemble",
            distance_type="cosine",
            n_samples=15,
            dropout_rate=0.25,
            ensemble_size=7,
            temperature=1.5
        )
        
        assert distance.config.method == "deep_ensemble"
        assert distance.distance_type == "cosine"
        assert distance.config.n_samples == 15
        assert distance.config.dropout_rate == 0.25
        assert distance.config.ensemble_size == 7
        assert distance.config.temperature == 1.5


class TestUncertaintyMetrics:
    """Test uncertainty quantification metrics."""
    
    def test_entropy_calculation(self):
        """Test entropy calculation for uncertainty."""
        config = UncertaintyConfig()
        mc_dropout = MonteCarloDropout(config)
        
        # Test with uniform probabilities (maximum entropy)
        uniform_probs = torch.ones(3, 3) / 3.0
        entropy = mc_dropout._compute_entropy(uniform_probs)
        
        expected_entropy = -torch.sum(uniform_probs * torch.log(uniform_probs + 1e-8), dim=1)
        assert torch.allclose(entropy, expected_entropy, atol=1e-6)
    
    def test_mutual_information_calculation(self):
        """Test mutual information calculation."""
        config = UncertaintyConfig()
        mc_dropout = MonteCarloDropout(config)
        
        # Create sample predictions
        prob_samples = torch.rand(5, 4, 3)  # 5 samples, 4 queries, 3 classes
        prob_samples = F.softmax(prob_samples, dim=-1)
        
        mutual_info = mc_dropout._compute_mutual_information(prob_samples)
        
        assert mutual_info.shape == (4,)
        assert torch.all(mutual_info >= 0)  # Mutual information should be non-negative


class TestNumericalStability:
    """Test numerical stability of uncertainty computations."""
    
    def test_log_probability_stability(self):
        """Test numerical stability in log probability calculations."""
        config = UncertaintyConfig()
        mc_dropout = MonteCarloDropout(config)
        
        # Test with very small probabilities
        small_probs = torch.tensor([[1e-10, 0.5, 0.5], [0.33, 0.33, 0.34]])
        
        # Should handle small probabilities without NaN/Inf
        entropy = mc_dropout._compute_entropy(small_probs)
        
        assert torch.all(torch.isfinite(entropy))
    
    def test_extreme_logits_handling(self):
        """Test handling of extreme logit values."""
        config = UncertaintyConfig()
        evidential = EvidentialLearning(config)
        
        # Test with extreme logits
        extreme_logits = torch.tensor([
            [100.0, -100.0, 50.0],
            [-200.0, 150.0, -50.0]
        ])
        
        alpha = evidential._compute_dirichlet_parameters(extreme_logits)
        
        # Should produce valid Dirichlet parameters
        assert torch.all(alpha > 0)
        assert torch.all(torch.isfinite(alpha))
    
    def test_zero_variance_handling(self):
        """Test handling of zero variance in ensemble predictions."""
        config = UncertaintyConfig()
        ensemble = DeepEnsemble(config)
        
        # Identical predictions (zero variance)
        identical_preds = torch.ones(3, 2, 4) * 0.25  # Same predictions
        
        variance = ensemble._compute_prediction_variance(identical_preds)
        
        # Should handle zero variance gracefully
        assert torch.all(torch.isfinite(variance))
        assert torch.all(variance >= 0)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_single_class_predictions(self):
        """Test uncertainty with single class predictions."""
        config = UncertaintyConfig(n_samples=3)
        distance = UncertaintyAwareDistance(config)
        
        # Single class scenario
        support_features = torch.randn(5, 8)
        support_labels = torch.zeros(5)  # All same class
        query_features = torch.randn(2, 8)
        
        result = distance.forward(support_features, support_labels, query_features)
        
        assert result["logits"].shape == (2, 1)  # Only one class
        assert torch.all(torch.isfinite(result["total_uncertainty"]))
    
    def test_empty_support_set(self):
        """Test handling of empty support set."""
        config = UncertaintyConfig()
        distance = UncertaintyAwareDistance(config)
        
        # Empty support set
        support_features = torch.empty(0, 10)
        support_labels = torch.empty(0, dtype=torch.long)
        query_features = torch.randn(3, 10)
        
        # Should handle gracefully or raise appropriate error
        with pytest.raises((RuntimeError, ValueError)):
            result = distance.forward(support_features, support_labels, query_features)
    
    def test_mismatched_feature_dimensions(self):
        """Test handling of mismatched feature dimensions."""
        config = UncertaintyConfig()
        distance = UncertaintyAwareDistance(config)
        
        support_features = torch.randn(6, 16)
        support_labels = torch.tensor([0, 0, 1, 1, 2, 2])
        query_features = torch.randn(3, 8)  # Different dimension
        
        # Should raise appropriate error
        with pytest.raises((RuntimeError, ValueError)):
            result = distance.forward(support_features, support_labels, query_features)
    
    def test_large_ensemble_size(self):
        """Test behavior with large ensemble size."""
        config = UncertaintyConfig(ensemble_size=100)
        ensemble = DeepEnsemble(config)
        
        # Large ensemble should work but may be slow
        assert ensemble.config.ensemble_size == 100
        assert len(ensemble.models) == 0  # No models added yet
    
    def test_zero_dropout_rate(self):
        """Test Monte Carlo dropout with zero dropout rate."""
        config = UncertaintyConfig(dropout_rate=0.0, n_samples=5)
        mc_dropout = MonteCarloDropout(config)
        
        assert mc_dropout.dropout.p == 0.0
        # Should still work, but uncertainty estimates may be less meaningful
    
    def test_high_temperature_scaling(self):
        """Test very high temperature scaling."""
        config = UncertaintyConfig(temperature=100.0)
        distance = UncertaintyAwareDistance(config)
        
        logits = torch.randn(3, 4)
        scaled_logits = distance._apply_temperature_scaling(logits)
        
        # High temperature should make predictions more uniform
        scaled_probs = F.softmax(scaled_logits, dim=-1)
        
        # Check that probabilities are more uniform than original
        original_probs = F.softmax(logits, dim=-1)
        scaled_entropy = -(scaled_probs * torch.log(scaled_probs + 1e-8)).sum(dim=-1)
        original_entropy = -(original_probs * torch.log(original_probs + 1e-8)).sum(dim=-1)
        
        # Higher temperature should generally increase entropy (more uniform)
        assert torch.all(scaled_entropy >= original_entropy - 1e-6)  # Allow small numerical errors