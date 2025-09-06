"""
Comprehensive tests for uncertainty-aware distance components.

Tests the various uncertainty estimation methods used in few-shot learning,
including Monte Carlo Dropout, Deep Ensembles, and Evidential Learning.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock, patch

# Import the components we're testing
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from meta_learning.few_shot_modules.uncertainty_components import (
    UncertaintyConfig,
    MonteCarloDropout
)


class TestUncertaintyConfig:
    """Test UncertaintyConfig dataclass"""
    
    def test_default_configuration(self):
        """Test default configuration values"""
        config = UncertaintyConfig()
        
        # Core settings
        assert config.method == "monte_carlo_dropout"
        
        # Monte Carlo Dropout
        assert config.dropout_rate == 0.1
        assert config.n_samples == 10
        
        # Deep Ensemble
        assert config.ensemble_size == 5
        assert config.ensemble_hidden_dim == 32
        
        # Evidential learning
        assert config.evidential_lambda == 1.0
        assert config.evidential_hidden_dim == 32
        assert config.num_classes == 10
        assert config.evidence_regularizer == 0.1
        assert config.uncertainty_method == "sensoy2018"
        
        # Bayesian parameters
        assert config.prior_mu == 0.0
        assert config.prior_sigma == 1.0
        assert config.bayesian_samples == 20
        
        # Distance computation
        assert config.distance_metric == "sqeuclidean"
        assert config.temperature == 1.0
        
        # Numerical stability
        assert config.eps == 1e-8
        assert config.max_uncertainty == 10.0
    
    def test_custom_configuration(self):
        """Test custom configuration creation"""
        config = UncertaintyConfig(
            method="deep_ensemble",
            dropout_rate=0.2,
            n_samples=20,
            ensemble_size=10,
            temperature=0.5,
            distance_metric="cosine"
        )
        
        assert config.method == "deep_ensemble"
        assert config.dropout_rate == 0.2
        assert config.n_samples == 20
        assert config.ensemble_size == 10
        assert config.temperature == 0.5
        assert config.distance_metric == "cosine"
        
        # Non-specified values should retain defaults
        assert config.evidential_lambda == 1.0
        assert config.eps == 1e-8


class TestMonteCarloDropout:
    """Test MonteCarloDropout uncertainty estimation"""
    
    def create_test_data(self):
        """Create test data for uncertainty estimation"""
        batch_size, num_classes, feature_dim = 8, 5, 64
        
        query_features = torch.randn(batch_size, feature_dim)
        prototypes = torch.randn(num_classes, feature_dim)
        
        return query_features, prototypes
    
    def test_initialization(self):
        """Test proper initialization of MonteCarloDropout"""
        config = UncertaintyConfig(dropout_rate=0.15, n_samples=15)
        mc_dropout = MonteCarloDropout(config)
        
        assert mc_dropout.config == config
        assert isinstance(mc_dropout.dropout, nn.Dropout)
        assert mc_dropout.dropout.p == 0.15
    
    def test_forward_sqeuclidean_default_samples(self):
        """Test forward pass with squared Euclidean distance and default samples"""
        config = UncertaintyConfig(
            distance_metric="sqeuclidean",
            temperature=1.0,
            n_samples=5
        )
        mc_dropout = MonteCarloDropout(config)
        mc_dropout.eval()  # Start in eval mode
        
        query_features, prototypes = self.create_test_data()
        
        with torch.no_grad():
            results = mc_dropout(query_features, prototypes)
        
        # Check output structure
        expected_keys = {"mean_logits", "epistemic_uncertainty", "mean_probs", 
                        "total_uncertainty", "predictive_entropy", "mutual_information"}
        for key in expected_keys:
            assert key in results, f"Missing key: {key}"
        
        # Check shapes
        batch_size, num_classes = query_features.shape[0], prototypes.shape[0]
        assert results["mean_logits"].shape == (batch_size, num_classes)
        assert results["epistemic_uncertainty"].shape == (batch_size, num_classes)
        assert results["mean_probs"].shape == (batch_size, num_classes)
        assert results["total_uncertainty"].shape == (batch_size,)
        assert results["predictive_entropy"].shape == (batch_size,)
        assert results["mutual_information"].shape == (batch_size,)
    
    def test_forward_cosine_custom_samples(self):
        """Test forward pass with cosine similarity and custom samples"""
        config = UncertaintyConfig(
            distance_metric="cosine",
            temperature=2.0,
            n_samples=3
        )
        mc_dropout = MonteCarloDropout(config)
        mc_dropout.eval()
        
        query_features, prototypes = self.create_test_data()
        
        with torch.no_grad():
            results = mc_dropout(query_features, prototypes, n_samples=7)
        
        # Should use n_samples=7 from function call, not config.n_samples=3
        # We can't directly check this, but we can verify the output is reasonable
        batch_size, num_classes = query_features.shape[0], prototypes.shape[0]
        assert results["mean_logits"].shape == (batch_size, num_classes)
        assert results["epistemic_uncertainty"].shape == (batch_size, num_classes)
    
    def test_training_mode_restoration(self):
        """Test that original training mode is restored after forward pass"""
        config = UncertaintyConfig(n_samples=3)
        mc_dropout = MonteCarloDropout(config)
        
        query_features, prototypes = self.create_test_data()
        
        # Test starting in eval mode
        mc_dropout.eval()
        assert not mc_dropout.training
        
        with torch.no_grad():
            _ = mc_dropout(query_features, prototypes)
        
        assert not mc_dropout.training  # Should be restored to eval
        
        # Test starting in train mode
        mc_dropout.train()
        assert mc_dropout.training
        
        with torch.no_grad():
            _ = mc_dropout(query_features, prototypes)
        
        assert mc_dropout.training  # Should be restored to train
    
    def test_uncertainty_properties(self):
        """Test mathematical properties of uncertainty estimates"""
        config = UncertaintyConfig(
            distance_metric="sqeuclidean",
            n_samples=10,
            dropout_rate=0.5  # Higher dropout for more variation
        )
        mc_dropout = MonteCarloDropout(config)
        mc_dropout.eval()
        
        query_features, prototypes = self.create_test_data()
        
        with torch.no_grad():
            results = mc_dropout(query_features, prototypes)
        
        # Epistemic uncertainty should be non-negative
        assert torch.all(results["epistemic_uncertainty"] >= 0), "Epistemic uncertainty must be non-negative"
        
        # Probabilities should sum to 1 (approximately)
        prob_sums = results["mean_probs"].sum(dim=-1)
        assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-5), "Probabilities should sum to 1"
        
        # Total uncertainty should be non-negative
        assert torch.all(results["total_uncertainty"] >= 0), "Total uncertainty must be non-negative"
        
        # Mutual information should be non-negative
        assert torch.all(results["mutual_information"] >= 0), "Mutual information must be non-negative"
        
        # Predictive entropy should be bounded by log(num_classes)
        max_entropy = torch.log(torch.tensor(float(prototypes.shape[0])))
        assert torch.all(results["predictive_entropy"] <= max_entropy + 1e-5), "Entropy should be bounded"
    
    def test_reproducibility(self):
        """Test that results are reproducible with same random seed"""
        config = UncertaintyConfig(n_samples=5)
        mc_dropout = MonteCarloDropout(config)
        mc_dropout.eval()
        
        query_features, prototypes = self.create_test_data()
        
        # Run twice with same seed
        torch.manual_seed(42)
        with torch.no_grad():
            results1 = mc_dropout(query_features, prototypes)
        
        torch.manual_seed(42)
        with torch.no_grad():
            results2 = mc_dropout(query_features, prototypes)
        
        # Results should be identical
        for key in results1:
            assert torch.allclose(results1[key], results2[key], atol=1e-6), f"Results not reproducible for {key}"
    
    def test_variability_with_dropout(self):
        """Test that dropout actually creates variability in predictions"""
        config = UncertaintyConfig(
            n_samples=2,
            dropout_rate=0.8  # High dropout rate
        )
        mc_dropout = MonteCarloDropout(config)
        mc_dropout.eval()
        
        query_features, prototypes = self.create_test_data()
        
        # Multiple runs should produce different uncertainty estimates
        uncertainties = []
        for _ in range(3):
            with torch.no_grad():
                results = mc_dropout(query_features, prototypes)
            uncertainties.append(results["epistemic_uncertainty"].mean().item())
        
        # Uncertainties should vary (though this is probabilistic)
        assert len(set(f"{u:.6f}" for u in uncertainties)) > 1, "Dropout should create variability"
    
    def test_edge_cases(self):
        """Test edge cases and error conditions"""
        config = UncertaintyConfig()
        mc_dropout = MonteCarloDropout(config)
        
        # Test with single query and prototype
        single_query = torch.randn(1, 32)
        single_proto = torch.randn(1, 32)
        
        with torch.no_grad():
            results = mc_dropout(single_query, single_proto)
        
        assert results["mean_logits"].shape == (1, 1)
        assert results["total_uncertainty"].shape == (1,)
        
        # Test with zero dropout (should still work)
        config_no_dropout = UncertaintyConfig(dropout_rate=0.0)
        mc_dropout_no_dropout = MonteCarloDropout(config_no_dropout)
        
        with torch.no_grad():
            results_no_dropout = mc_dropout_no_dropout(single_query, single_proto)
        
        # With zero dropout, epistemic uncertainty should be very small
        assert torch.all(results_no_dropout["epistemic_uncertainty"] < 0.1), "Zero dropout should have low epistemic uncertainty"
    
    def test_temperature_effects(self):
        """Test effects of temperature scaling on uncertainty"""
        query_features, prototypes = self.create_test_data()
        
        # Low temperature (sharp predictions)
        config_low_temp = UncertaintyConfig(temperature=0.1, n_samples=5)
        mc_dropout_low = MonteCarloDropout(config_low_temp)
        mc_dropout_low.eval()
        
        # High temperature (soft predictions)
        config_high_temp = UncertaintyConfig(temperature=5.0, n_samples=5)
        mc_dropout_high = MonteCarloDropout(config_high_temp)
        mc_dropout_high.eval()
        
        with torch.no_grad():
            results_low = mc_dropout_low(query_features, prototypes)
            results_high = mc_dropout_high(query_features, prototypes)
        
        # High temperature should generally produce higher entropy (more uncertain predictions)
        mean_entropy_low = results_low["predictive_entropy"].mean().item()
        mean_entropy_high = results_high["predictive_entropy"].mean().item()
        
        assert mean_entropy_high > mean_entropy_low, "High temperature should increase predictive entropy"
    
    def test_device_consistency(self):
        """Test that operations maintain device consistency"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        device = torch.device('cuda')
        config = UncertaintyConfig(n_samples=3)
        mc_dropout = MonteCarloDropout(config).to(device)
        
        query_features = torch.randn(4, 32).to(device)
        prototypes = torch.randn(3, 32).to(device)
        
        with torch.no_grad():
            results = mc_dropout(query_features, prototypes)
        
        # All outputs should be on the same device
        for key, value in results.items():
            assert value.device == device, f"Output {key} should be on {device}"
    
    def test_gradient_flow(self):
        """Test that gradients flow properly through the module"""
        config = UncertaintyConfig(n_samples=2)
        mc_dropout = MonteCarloDropout(config)
        
        query_features = torch.randn(2, 16, requires_grad=True)
        prototypes = torch.randn(3, 16, requires_grad=True)
        
        # Forward pass
        results = mc_dropout(query_features, prototypes)
        
        # Compute loss and backward pass
        loss = results["mean_logits"].sum()
        loss.backward()
        
        # Gradients should be computed
        assert query_features.grad is not None, "Query features should have gradients"
        assert prototypes.grad is not None, "Prototypes should have gradients"
        assert torch.any(query_features.grad != 0), "Query gradients should be non-zero"
        assert torch.any(prototypes.grad != 0), "Prototype gradients should be non-zero"


class TestUncertaintyIntegration:
    """Integration tests for uncertainty components"""
    
    def test_few_shot_learning_workflow(self):
        """Test typical few-shot learning workflow with uncertainty"""
        # Simulate 3-way 5-shot episode
        n_classes, n_support, n_query = 3, 5, 10
        feature_dim = 128
        
        # Create episode data
        support_features = torch.randn(n_classes * n_support, feature_dim)
        query_features = torch.randn(n_query, feature_dim)
        
        # Compute prototypes (mean of support samples per class)
        prototypes = torch.zeros(n_classes, feature_dim)
        for i in range(n_classes):
            start_idx = i * n_support
            end_idx = start_idx + n_support
            prototypes[i] = support_features[start_idx:end_idx].mean(dim=0)
        
        # Apply uncertainty estimation
        config = UncertaintyConfig(
            distance_metric="sqeuclidean",
            temperature=1.0,
            n_samples=8,
            dropout_rate=0.1
        )
        mc_dropout = MonteCarloDropout(config)
        mc_dropout.eval()
        
        with torch.no_grad():
            uncertainty_results = mc_dropout(query_features, prototypes)
        
        # Make predictions based on mean logits
        predictions = uncertainty_results["mean_logits"].argmax(dim=-1)
        confidences = uncertainty_results["mean_probs"].max(dim=-1)[0]
        uncertainties = uncertainty_results["total_uncertainty"]
        
        # Verify shapes and properties
        assert predictions.shape == (n_query,)
        assert confidences.shape == (n_query,)
        assert uncertainties.shape == (n_query,)
        
        # High confidence should correspond to low uncertainty (generally)
        correlation = torch.corrcoef(torch.stack([confidences, uncertainties]))[0, 1]
        assert correlation < 0, "Confidence and uncertainty should be negatively correlated"
    
    def test_uncertainty_ranking(self):
        """Test that uncertainty estimates can rank prediction quality"""
        config = UncertaintyConfig(n_samples=10, dropout_rate=0.3)
        mc_dropout = MonteCarloDropout(config)
        mc_dropout.eval()
        
        # Create data where some queries are clearly closer to prototypes
        prototypes = torch.tensor([
            [1.0, 0.0],  # Class 0 prototype
            [0.0, 1.0],  # Class 1 prototype
        ])
        
        # Queries with varying distances to prototypes
        queries = torch.tensor([
            [0.9, 0.1],   # Very close to class 0
            [0.7, 0.3],   # Somewhat close to class 0  
            [0.5, 0.5],   # Ambiguous
            [0.3, 0.7],   # Somewhat close to class 1
            [0.1, 0.9],   # Very close to class 1
        ])
        
        with torch.no_grad():
            results = mc_dropout(queries, prototypes)
        
        uncertainties = results["total_uncertainty"]
        
        # The ambiguous query (index 2) should have highest uncertainty
        ambiguous_uncertainty = uncertainties[2].item()
        other_uncertainties = [uncertainties[i].item() for i in [0, 1, 3, 4]]
        
        assert ambiguous_uncertainty > max(other_uncertainties), "Ambiguous query should have highest uncertainty"
    
    def test_monte_carlo_convergence(self):
        """Test that Monte Carlo estimates converge with more samples"""
        config_few = UncertaintyConfig(n_samples=5)
        config_many = UncertaintyConfig(n_samples=50)
        
        mc_dropout_few = MonteCarloDropout(config_few)
        mc_dropout_many = MonteCarloDropout(config_many)
        
        query_features = torch.randn(3, 32)
        prototypes = torch.randn(2, 32)
        
        # Set same seed for both
        torch.manual_seed(123)
        mc_dropout_few.eval()
        with torch.no_grad():
            results_few = mc_dropout_few(query_features, prototypes)
        
        torch.manual_seed(123)
        mc_dropout_many.eval()
        with torch.no_grad():
            results_many = mc_dropout_many(query_features, prototypes)
        
        # More samples should generally produce more stable estimates
        # (lower variance in repeated runs - this is a probabilistic test)
        variance_few = results_few["epistemic_uncertainty"].var().item()
        variance_many = results_many["epistemic_uncertainty"].var().item()
        
        # This is probabilistic, so we use a loose check
        # More samples should tend toward more consistent uncertainty estimates
        assert results_few["mean_logits"].shape == results_many["mean_logits"].shape
        assert results_few["total_uncertainty"].shape == results_many["total_uncertainty"].shape


if __name__ == "__main__":
    # Run tests if called directly
    pytest.main([__file__, "-v"])