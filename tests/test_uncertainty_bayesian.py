"""
Comprehensive tests for uncertainty/bayesian_meta_learning.py module.

Tests all implemented uncertainty estimation classes and functions.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from meta_learning.uncertainty.bayesian_meta_learning import (
    UncertaintyAwareDistance, MonteCarloDropout, DeepEnsemble,
    EvidentialLearning, UncertaintyConfig, create_uncertainty_aware_distance
)


class TestUncertaintyAwareDistance:
    """Test UncertaintyAwareDistance implementation."""
    
    def test_initialization(self):
        """Test proper initialization."""
        dist = UncertaintyAwareDistance()
        assert dist.base_distance == 'euclidean'
        assert dist.uncertainty_method == 'monte_carlo'
        assert dist.num_samples == 10
        assert dist.temperature == 1.0
    
    def test_euclidean_distance_computation(self):
        """Test Euclidean distance computation with uncertainty."""
        dist = UncertaintyAwareDistance(base_distance='euclidean')
        
        query_features = torch.randn(5, 64)  # 5 queries
        prototype_features = torch.randn(3, 64)  # 3 prototypes
        
        distances, uncertainties = dist.compute_distances_with_uncertainty(
            query_features, prototype_features
        )
        
        assert distances.shape == (5, 3)
        assert uncertainties.shape == (5, 3)
        assert not torch.isnan(distances).any()
        assert not torch.isnan(uncertainties).any()
    
    def test_cosine_distance_computation(self):
        """Test cosine distance computation with uncertainty."""
        dist = UncertaintyAwareDistance(base_distance='cosine')
        
        query_features = torch.randn(3, 32)
        prototype_features = torch.randn(5, 32)
        
        distances, uncertainties = dist.compute_distances_with_uncertainty(
            query_features, prototype_features
        )
        
        assert distances.shape == (3, 5)
        assert uncertainties.shape == (3, 5)
        # Cosine distances should be in [0, 2]
        assert torch.all(distances >= 0)
        assert torch.all(distances <= 2)
    
    def test_temperature_scaling(self):
        """Test temperature scaling effect."""
        dist_low = UncertaintyAwareDistance(temperature=0.5)
        dist_high = UncertaintyAwareDistance(temperature=2.0)
        
        query_features = torch.randn(2, 16)
        prototype_features = torch.randn(2, 16)
        
        distances_low, _ = dist_low.compute_distances_with_uncertainty(
            query_features, prototype_features
        )
        distances_high, _ = dist_high.compute_distances_with_uncertainty(
            query_features, prototype_features
        )
        
        # Lower temperature should give higher scaled distances
        assert torch.all(distances_low > distances_high)
    
    def test_monte_carlo_uncertainty(self):
        """Test Monte Carlo uncertainty estimation."""
        # Create simple model with dropout
        model = nn.Sequential(
            nn.Linear(16, 8),
            nn.Dropout(0.5),
            nn.Linear(8, 16)
        )
        
        dist = UncertaintyAwareDistance(
            uncertainty_method='monte_carlo',
            num_samples=5
        )
        
        query_features = torch.randn(2, 16)
        prototype_features = torch.randn(3, 16)
        
        distances, uncertainties = dist.compute_distances_with_uncertainty(
            query_features, prototype_features, model
        )
        
        assert distances.shape == (2, 3)
        assert uncertainties.shape == (2, 3)
        assert torch.all(uncertainties >= 0)  # Variance should be non-negative


class TestMonteCarloDropout:
    """Test MonteCarloDropout implementation."""
    
    def test_initialization(self):
        """Test MonteCarloDropout initialization."""
        model = nn.Sequential(nn.Linear(10, 5), nn.Dropout(0.2))
        mc_dropout = MonteCarloDropout(model, num_samples=15, dropout_rate=0.3)
        
        assert mc_dropout.num_samples == 15
        assert mc_dropout.dropout_rate == 0.3
        assert mc_dropout.model == model
    
    def test_forward_with_uncertainty(self):
        """Test forward pass with uncertainty estimation."""
        model = nn.Sequential(
            nn.Linear(8, 16),
            nn.Dropout(0.5),
            nn.Linear(16, 4)
        )
        mc_dropout = MonteCarloDropout(model, num_samples=10)
        
        x = torch.randn(3, 8)
        mean_output, uncertainty = mc_dropout.forward_with_uncertainty(x)
        
        assert mean_output.shape == (3, 4)
        assert uncertainty.shape == (3, 4)
        assert torch.all(uncertainty >= 0)  # Variance should be non-negative
    
    def test_training_mode_preservation(self):
        """Test that original training mode is preserved."""
        model = nn.Sequential(nn.Linear(5, 3), nn.Dropout(0.1))
        mc_dropout = MonteCarloDropout(model)
        
        # Set model to eval mode
        model.eval()
        original_mode = model.training
        
        x = torch.randn(2, 5)
        mc_dropout.forward_with_uncertainty(x)
        
        # Model should be back to original mode
        assert model.training == original_mode
    
    def test_dropout_rate_update(self):
        """Test that dropout rates are updated."""
        model = nn.Sequential(
            nn.Linear(4, 8),
            nn.Dropout(0.1),  # Original rate
            nn.Linear(8, 2)
        )
        mc_dropout = MonteCarloDropout(model, dropout_rate=0.7)
        
        # Check that dropout rate was updated
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                assert module.p == 0.7


class TestDeepEnsemble:
    """Test DeepEnsemble implementation."""
    
    def test_initialization(self):
        """Test DeepEnsemble initialization."""
        def model_factory():
            return nn.Sequential(nn.Linear(6, 3))
        
        ensemble = DeepEnsemble(model_factory, num_models=5)
        
        assert len(ensemble.models) == 5
        assert ensemble.num_models == 5
        assert ensemble.ensemble_method == 'average'
        
        # Check that all models are different instances
        for i in range(len(ensemble.models)):
            for j in range(i + 1, len(ensemble.models)):
                assert ensemble.models[i] is not ensemble.models[j]
    
    def test_forward_ensemble(self):
        """Test ensemble forward pass."""
        def model_factory():
            return nn.Sequential(nn.Linear(5, 3))
        
        ensemble = DeepEnsemble(model_factory, num_models=3)
        
        x = torch.randn(4, 5)
        mean_pred, epistemic_uncertainty = ensemble.forward_ensemble(x)
        
        assert mean_pred.shape == (4, 3)
        assert epistemic_uncertainty.shape == (4, 3)
        assert torch.all(epistemic_uncertainty >= 0)  # Variance should be non-negative
    
    def test_ensemble_methods(self):
        """Test different ensemble combination methods."""
        def model_factory():
            return nn.Sequential(nn.Linear(3, 2))
        
        ensemble_avg = DeepEnsemble(model_factory, num_models=2, ensemble_method='average')
        ensemble_weighted = DeepEnsemble(model_factory, num_models=2, ensemble_method='weighted')
        
        x = torch.randn(2, 3)
        
        mean_avg, _ = ensemble_avg.forward_ensemble(x)
        mean_weighted, _ = ensemble_weighted.forward_ensemble(x)
        
        assert mean_avg.shape == mean_weighted.shape == (2, 2)
        # Results should generally be different for different methods
        assert not torch.allclose(mean_avg, mean_weighted, atol=1e-6)


class TestEvidentialLearning:
    """Test EvidentialLearning implementation."""
    
    def test_initialization(self):
        """Test EvidentialLearning initialization."""
        evidential = EvidentialLearning(num_classes=5, evidence_activation='relu')
        
        assert evidential.num_classes == 5
        assert evidential.evidence_activation == 'relu'
        assert evidential.activation == F.relu
    
    def test_activation_functions(self):
        """Test different activation functions."""
        activations = ['relu', 'exp', 'softplus']
        expected_funcs = [F.relu, torch.exp, F.softplus]
        
        for activation, expected_func in zip(activations, expected_funcs):
            evidential = EvidentialLearning(num_classes=3, evidence_activation=activation)
            assert evidential.activation == expected_func
    
    def test_invalid_activation(self):
        """Test invalid activation function raises error."""
        with pytest.raises(ValueError, match="Unknown activation"):
            EvidentialLearning(num_classes=3, evidence_activation='invalid')
    
    def test_compute_dirichlet_parameters(self):
        """Test Dirichlet parameter computation."""
        evidential = EvidentialLearning(num_classes=4, evidence_activation='relu')
        
        logits = torch.randn(3, 4)
        alpha = evidential.compute_dirichlet_parameters(logits)
        
        assert alpha.shape == (3, 4)
        assert torch.all(alpha >= 1.0)  # Alpha should be >= 1 due to +1 offset
        assert not torch.isnan(alpha).any()
    
    def test_evidential_loss(self):
        """Test evidential loss computation."""
        evidential = EvidentialLearning(num_classes=3)
        
        alpha = torch.tensor([[2.0, 3.0, 1.5], [1.2, 4.0, 2.1]])  # [batch_size=2, num_classes=3]
        targets = torch.tensor([1, 0])  # Class indices
        
        loss = evidential.evidential_loss(alpha, targets, global_step=5, annealing_step=10)
        
        assert loss.numel() == 1  # Scalar loss
        assert loss.item() >= 0  # Loss should be non-negative
        assert not torch.isnan(loss)
    
    def test_compute_uncertainty(self):
        """Test uncertainty computation."""
        evidential = EvidentialLearning(num_classes=3)
        
        alpha = torch.tensor([[3.0, 2.0, 1.0], [5.0, 4.0, 3.0]])
        aleatoric, epistemic = evidential.compute_uncertainty(alpha)
        
        assert aleatoric.shape == (2,)
        assert epistemic.shape == (2,)
        assert torch.all(aleatoric >= 0)
        assert torch.all(epistemic >= 0)
        assert not torch.isnan(aleatoric).any()
        assert not torch.isnan(epistemic).any()
    
    def test_kl_divergence(self):
        """Test KL divergence computation."""
        evidential = EvidentialLearning(num_classes=2)
        
        alpha = torch.tensor([[2.0, 3.0], [4.0, 1.5]])
        kl = evidential._kl_divergence(alpha)
        
        assert kl.shape == (2, 1)
        assert not torch.isnan(kl).any()


class TestUncertaintyConfig:
    """Test UncertaintyConfig implementation."""
    
    def test_initialization(self):
        """Test UncertaintyConfig initialization."""
        config = UncertaintyConfig(
            method='monte_carlo',
            num_samples=20,
            temperature=0.8,
            calibration=True
        )
        
        assert config.method == 'monte_carlo'
        assert config.num_samples == 20
        assert config.temperature == 0.8
        assert config.calibration == True
    
    def test_validation_valid_config(self):
        """Test that valid configurations pass validation."""
        # Should not raise any errors
        UncertaintyConfig(method='ensemble', num_samples=5, temperature=1.5)
        UncertaintyConfig(method='evidential', num_samples=1, temperature=0.1)
    
    def test_validation_invalid_method(self):
        """Test validation of invalid method."""
        with pytest.raises(ValueError, match="Method must be one of"):
            UncertaintyConfig(method='invalid_method')
    
    def test_validation_invalid_num_samples(self):
        """Test validation of invalid num_samples."""
        with pytest.raises(ValueError, match="num_samples must be positive"):
            UncertaintyConfig(num_samples=0)
        
        with pytest.raises(ValueError, match="num_samples must be positive"):
            UncertaintyConfig(num_samples=-5)
    
    def test_validation_invalid_temperature(self):
        """Test validation of invalid temperature."""
        with pytest.raises(ValueError, match="temperature must be positive"):
            UncertaintyConfig(temperature=0)
        
        with pytest.raises(ValueError, match="temperature must be positive"):
            UncertaintyConfig(temperature=-0.5)
    
    def test_create_uncertainty_estimator(self):
        """Test uncertainty estimator creation."""
        model = nn.Linear(4, 2)
        
        # Test Monte Carlo creation
        config_mc = UncertaintyConfig(method='monte_carlo', num_samples=8)
        estimator_mc = config_mc.create_uncertainty_estimator(model)
        assert isinstance(estimator_mc, MonteCarloDropout)
        assert estimator_mc.num_samples == 8
        
        # Test Ensemble creation
        config_ens = UncertaintyConfig(method='ensemble', num_models=4)
        estimator_ens = config_ens.create_uncertainty_estimator(model)
        assert isinstance(estimator_ens, DeepEnsemble)
        assert estimator_ens.num_models == 4
        
        # Test Evidential creation
        config_ev = UncertaintyConfig(method='evidential', num_classes=3)
        estimator_ev = config_ev.create_uncertainty_estimator(model)
        assert isinstance(estimator_ev, EvidentialLearning)
        assert estimator_ev.num_classes == 3
    
    def test_create_invalid_estimator(self):
        """Test creating estimator with invalid method."""
        # Bypass normal validation to test the create method specifically
        config = UncertaintyConfig.__new__(UncertaintyConfig)
        config.method = 'invalid'
        
        model = nn.Linear(2, 1)
        with pytest.raises(ValueError, match="Unknown uncertainty method"):
            config.create_uncertainty_estimator(model)


class TestFactoryFunction:
    """Test create_uncertainty_aware_distance factory function."""
    
    def test_basic_creation(self):
        """Test basic factory function usage."""
        dist = create_uncertainty_aware_distance()
        
        assert isinstance(dist, UncertaintyAwareDistance)
        assert dist.base_distance == 'euclidean'
        assert dist.uncertainty_method == 'monte_carlo'
    
    def test_custom_parameters(self):
        """Test factory function with custom parameters."""
        dist = create_uncertainty_aware_distance(
            base_distance='cosine',
            uncertainty_method='ensemble',
            num_samples=15,
            temperature=0.5
        )
        
        assert isinstance(dist, UncertaintyAwareDistance)
        assert dist.base_distance == 'cosine'
        assert dist.uncertainty_method == 'ensemble'
        assert dist.num_samples == 15
        assert dist.temperature == 0.5


class TestIntegrationScenarios:
    """Test integration scenarios with real-world usage patterns."""
    
    def test_prototypical_network_uncertainty(self):
        """Test uncertainty in prototypical network scenario."""
        # Create uncertainty-aware distance
        dist = create_uncertainty_aware_distance(
            base_distance='euclidean',
            uncertainty_method='monte_carlo',
            num_samples=5
        )
        
        # Simulate few-shot scenario
        n_way, k_shot, n_query = 3, 2, 5
        feature_dim = 32
        
        # Create features (simulating extracted features from backbone)
        support_features = torch.randn(n_way * k_shot, feature_dim)
        query_features = torch.randn(n_way * n_query, feature_dim)
        
        # Compute prototypes
        prototypes = support_features.view(n_way, k_shot, feature_dim).mean(dim=1)
        
        # Compute distances with uncertainty
        distances, uncertainties = dist.compute_distances_with_uncertainty(
            query_features, prototypes
        )
        
        assert distances.shape == (n_way * n_query, n_way)
        assert uncertainties.shape == (n_way * n_query, n_way)
        
        # Convert to logits and probabilities
        logits = -distances  # Negative distance as logits
        probs = F.softmax(logits, dim=1)
        
        assert torch.allclose(probs.sum(dim=1), torch.ones(n_way * n_query))
        assert torch.all(uncertainties >= 0)
    
    def test_evidential_classification_pipeline(self):
        """Test complete evidential classification pipeline."""
        num_classes = 4
        batch_size = 3
        
        evidential = EvidentialLearning(num_classes=num_classes)
        
        # Simulate network outputs
        logits = torch.randn(batch_size, num_classes)
        targets = torch.randint(0, num_classes, (batch_size,))
        
        # Convert to Dirichlet parameters
        alpha = evidential.compute_dirichlet_parameters(logits)
        
        # Compute loss
        loss = evidential.evidential_loss(alpha, targets, global_step=10)
        
        # Compute uncertainties
        aleatoric, epistemic = evidential.compute_uncertainty(alpha)
        
        assert not torch.isnan(loss)
        assert loss.item() >= 0
        assert aleatoric.shape == (batch_size,)
        assert epistemic.shape == (batch_size,)
        assert torch.all(aleatoric >= 0)
        assert torch.all(epistemic >= 0)
    
    def test_ensemble_meta_learning(self):
        """Test ensemble in meta-learning context."""
        def create_simple_model():
            return nn.Sequential(
                nn.Linear(16, 32),
                nn.ReLU(),
                nn.Linear(32, 5)
            )
        
        ensemble = DeepEnsemble(create_simple_model, num_models=3)
        
        # Simulate episode data
        episode_data = torch.randn(8, 16)  # 8 samples, 16 features
        
        mean_pred, uncertainty = ensemble.forward_ensemble(episode_data)
        
        assert mean_pred.shape == (8, 5)
        assert uncertainty.shape == (8, 5)
        assert torch.all(uncertainty >= 0)
        
        # Test that ensemble gives different results than individual models
        individual_pred = ensemble.models[0](episode_data)
        assert not torch.allclose(mean_pred, individual_pred, atol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__])