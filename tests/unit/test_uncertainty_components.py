"""
Tests for UncertaintyAwareDistance implementations
================================================

Tests all 4 uncertainty estimation methods with proper configurations.
"""

import pytest
import torch
# DEMO/TEST REPRODUCIBILITY: Set seed for consistent synthetic data
torch.manual_seed(42)  # Reproducible test/demo data
import torch.nn as nn
# DEMO/TEST REPRODUCIBILITY: Set seed for consistent synthetic data
torch.manual_seed(42)  # Reproducible test/demo data
from meta_learning.meta_learning_modules.few_shot_modules.uncertainty_components import (
    UncertaintyAwareDistance, UncertaintyConfig
)


class TestUncertaintyAwareDistance:
    """Test suite for uncertainty-aware distance metrics."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.embedding_dim = 64
        self.n_classes = 5
        self.n_samples = 25
        
        # Create test data
        self.query_embeddings = torch.randn(self.n_samples, self.embedding_dim)
        self.prototypes = torch.randn(self.n_classes, self.embedding_dim)
        
    def test_monte_carlo_dropout_method(self):
        """Test Monte Carlo Dropout uncertainty estimation."""
        config = UncertaintyConfig(
            method="monte_carlo_dropout",
            n_samples=10,
            dropout_rate=0.1
        )
        
        uncertainty_dist = UncertaintyAwareDistance(self.embedding_dim, config)
        
        distances, uncertainties = uncertainty_dist(self.query_embeddings, self.prototypes)
        
        # Check output shapes
        assert distances.shape == (self.n_samples, self.n_classes)
        assert uncertainties.shape == (self.n_samples, self.n_classes)
        
        # Check uncertainty values are non-negative
        assert torch.all(uncertainties >= 0)
        
        # Check distances are non-negative
        assert torch.all(distances >= 0)
        
    def test_deep_ensemble_method(self):
        """Test Deep Ensemble uncertainty estimation."""
        config = UncertaintyConfig(
            method="deep_ensemble",
            num_ensemble_members=5,
            ensemble_hidden_dim=32
        )
        
        uncertainty_dist = UncertaintyAwareDistance(self.embedding_dim, config)
        
        distances, uncertainties = uncertainty_dist(self.query_embeddings, self.prototypes)
        
        # Check output shapes
        assert distances.shape == (self.n_samples, self.n_classes)
        assert uncertainties.shape == (self.n_samples, self.n_classes)
        
        # Check uncertainty values are non-negative
        assert torch.all(uncertainties >= 0)
        
    def test_evidential_method(self):
        """Test Evidential Deep Learning uncertainty estimation."""
        config = UncertaintyConfig(
            method="evidential",
            evidential_hidden_dim=32,
            evidence_regularizer=0.1
        )
        
        uncertainty_dist = UncertaintyAwareDistance(self.embedding_dim, config)
        
        distances, uncertainties = uncertainty_dist(self.query_embeddings, self.prototypes)
        
        # Check output shapes
        assert distances.shape == (self.n_samples, self.n_classes)
        assert uncertainties.shape == (self.n_samples, self.n_classes)
        
        # Check uncertainty values are non-negative
        assert torch.all(uncertainties >= 0)
        
    def test_bayesian_method(self):
        """Test Bayesian Neural Network uncertainty estimation."""
        config = UncertaintyConfig(
            method="bayesian",
            bayesian_samples=10,
            prior_std=0.1
        )
        
        uncertainty_dist = UncertaintyAwareDistance(self.embedding_dim, config)
        
        distances, uncertainties = uncertainty_dist(self.query_embeddings, self.prototypes)
        
        # Check output shapes
        assert distances.shape == (self.n_samples, self.n_classes)
        assert uncertainties.shape == (self.n_samples, self.n_classes)
        
        # Check uncertainty values are non-negative
        assert torch.all(uncertainties >= 0)
        
    def test_invalid_method(self):
        """Test error handling for invalid methods."""
        config = UncertaintyConfig(method="invalid_method")
        
        with pytest.raises(ValueError, match="Unknown uncertainty method"):
            uncertainty_dist = UncertaintyAwareDistance(self.embedding_dim, config)
            uncertainty_dist(self.query_embeddings, self.prototypes)
            
    def test_gradient_flow(self):
        """Test that gradients flow through uncertainty computations."""
        config = UncertaintyConfig(method="monte_carlo_dropout", num_samples=5)
        
        uncertainty_dist = UncertaintyAwareDistance(self.embedding_dim, config)
        
        # Enable gradients
        query_embeddings = self.query_embeddings.requires_grad_(True)
        prototypes = self.prototypes.requires_grad_(True)
        
        distances, uncertainties = uncertainty_dist(query_embeddings, prototypes)
        
        # Compute a simple loss
        loss = distances.mean() + uncertainties.mean()
        loss.backward()
        
        # Check gradients exist
        assert query_embeddings.grad is not None
        assert prototypes.grad is not None
        
    def test_config_defaults(self):
        """Test default configuration values."""
        config = UncertaintyConfig()
        
        assert config.method == "monte_carlo_dropout"
        assert config.num_samples == 10
        assert config.dropout_rate == 0.1
        
    def test_different_input_sizes(self):
        """Test with different input tensor sizes."""
        config = UncertaintyConfig(method="monte_carlo_dropout", num_samples=3)
        
        uncertainty_dist = UncertaintyAwareDistance(self.embedding_dim, config)
        
        # Test with single query
        single_query = torch.randn(1, self.embedding_dim)
        distances, uncertainties = uncertainty_dist(single_query, self.prototypes)
        
        assert distances.shape == (1, self.n_classes)
        assert uncertainties.shape == (1, self.n_classes)
        
        # Test with many queries
        many_queries = torch.randn(100, self.embedding_dim)
        distances, uncertainties = uncertainty_dist(many_queries, self.prototypes)
        
        assert distances.shape == (100, self.n_classes)
        assert uncertainties.shape == (100, self.n_classes)


class TestUncertaintyConfig:
    """Test uncertainty configuration class."""
    
    def test_config_creation(self):
        """Test configuration object creation."""
        config = UncertaintyConfig(
            method="deep_ensemble",
            num_ensemble_members=7,
            ensemble_hidden_dim=64
        )
        
        assert config.method == "deep_ensemble"
        assert config.num_ensemble_members == 7
        assert config.ensemble_hidden_dim == 64
        
    def test_config_validation(self):
        """Test configuration validation."""
        # Valid configurations should not raise errors
        valid_methods = ["monte_carlo_dropout", "deep_ensemble", "evidential", "bayesian"]
        
        for method in valid_methods:
            config = UncertaintyConfig(method=method)
            assert config.method == method


class TestUncertaintyIntegration:
    """Integration tests for uncertainty components."""
    
    def test_uncertainty_with_prototypical_networks(self):
        """Test uncertainty integration with prototypical networks."""
        from meta_learning.meta_learning_modules.few_shot_modules.utilities import create_prototypical_network, create_backbone_network
        from meta_learning.meta_learning_modules.few_shot_modules.configurations import PrototypicalConfig
        
        # Create backbone and prototypical network
        backbone = create_backbone_network("simple", input_channels=1, embedding_dim=64)
        proto_config = PrototypicalConfig(use_uncertainty_aware_distances=True)
        proto_net = create_prototypical_network(backbone, "enhanced", proto_config)
        
        # Create synthetic episode data
        n_way, n_support, n_query = 5, 5, 15
        support_images = torch.randn(n_way * n_support, 1, 28, 28)
        support_labels = torch.repeat_interleave(torch.arange(n_way), n_support)
        query_images = torch.randn(n_way * n_query, 1, 28, 28)
        
        # Forward pass
        result = proto_net(support_images, support_labels, query_images)
        
        # Check result structure
        assert isinstance(result, dict)
        assert 'logits' in result
        assert 'uncertainties' in result or 'distances' in result
        
        logits = result['logits']
        assert logits.shape == (n_way * n_query, n_way)
        
    def test_uncertainty_computation_consistency(self):
        """Test consistency of uncertainty computations across methods."""
        embedding_dim = 32
        query_embeddings = torch.randn(10, embedding_dim)
        prototypes = torch.randn(3, embedding_dim)
        
        methods = ["monte_carlo_dropout", "deep_ensemble", "evidential", "bayesian"]
        results = {}
        
        for method in methods:
            config = UncertaintyConfig(method=method, num_samples=5)
            uncertainty_dist = UncertaintyAwareDistance(embedding_dim, config)
            
            distances, uncertainties = uncertainty_dist(query_embeddings, prototypes)
            
            results[method] = {
                'distances': distances,
                'uncertainties': uncertainties
            }
            
            # Check basic properties
            assert torch.all(distances >= 0), f"Negative distances in {method}"
            assert torch.all(uncertainties >= 0), f"Negative uncertainties in {method}"
            assert torch.all(torch.isfinite(distances)), f"Non-finite distances in {method}"
            assert torch.all(torch.isfinite(uncertainties)), f"Non-finite uncertainties in {method}"
        
        # All methods should produce different uncertainty estimates
        # (this is a statistical test, could occasionally fail)
        uncertainty_means = [results[method]['uncertainties'].mean() for method in methods]
        assert len(set([round(u.item(), 3) for u in uncertainty_means])) > 1, "All methods produced identical uncertainties"