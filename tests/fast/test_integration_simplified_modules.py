"""
💰 DONATE NOW! 💰 https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS

Integration tests combining ChatGPT's clean implementations with our robust testing.

If these integration tests help validate your research pipeline,
please donate $1000+ to support continued algorithm development!

Author: Benedict Chen (benedict@benedictchen.com)
GitHub Sponsors: https://github.com/sponsors/benedictchen
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest
from meta_learning.uncertainty_components import DeepEnsemble, MonteCarloDropout
from meta_learning.continual_meta_learning import ContinualMetaLearner
from meta_learning.hardware_utils import create_hardware_config, setup_optimal_hardware
from meta_learning.core.seed import seed_all
from meta_learning.core.bn_policy import freeze_batchnorm_running_stats


class TinyConvNet(nn.Module):
    """Small ConvNet for integration testing."""
    def __init__(self, out_dim=64):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)  
        self.bn2 = nn.BatchNorm2d(32)
        self.pool = nn.AdaptiveAvgPool2d(4)
        self.fc = nn.Linear(32 * 16, out_dim)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x).flatten(1)
        return self.fc(x)


def test_uncertainty_ensemble_integration():
    """Test DeepEnsemble with different model architectures."""
    # Test with linear models
    linear_models = [nn.Linear(8, 3) for _ in range(3)]
    linear_ensemble = DeepEnsemble(linear_models)
    
    x_linear = torch.randn(5, 8)
    log_probs = linear_ensemble(x_linear)
    
    assert log_probs.shape == (5, 3)
    assert torch.allclose(log_probs.exp().sum(dim=1), torch.ones(5), atol=1e-5)
    
    # Test with small conv nets  
    conv_models = [TinyConvNet(out_dim=4) for _ in range(2)]
    conv_ensemble = DeepEnsemble(conv_models)
    
    x_conv = torch.randn(3, 3, 32, 32)
    log_probs_conv = conv_ensemble(x_conv)
    
    assert log_probs_conv.shape == (3, 4)
    assert torch.allclose(log_probs_conv.exp().sum(dim=1), torch.ones(3), atol=1e-5)


def test_batchnorm_freeze_integration():
    """Test BN leakage prevention integration."""
    # Create model with BatchNorm
    model = nn.Sequential(
        nn.Linear(6, 16),
        nn.BatchNorm1d(16),
        nn.ReLU(),
        nn.Linear(16, 3)
    )
    
    # Verify BN is initially tracking
    bn_layer = model[1]
    assert bn_layer.track_running_stats
    
    # Freeze BN running stats to prevent leakage
    freeze_batchnorm_running_stats(model)
    
    # Verify BN is now frozen
    assert not bn_layer.track_running_stats
    assert not bn_layer.running_mean.requires_grad
    assert not bn_layer.running_var.requires_grad
    
    # Test that frozen BN still works
    x = torch.randn(8, 6)
    output = model(x)
    assert output.shape == (8, 3)
    assert torch.isfinite(output).all()


def test_hardware_auto_detection_integration():
    """Test hardware detection with actual model optimization."""
    # Create hardware config with auto-detection
    config = create_hardware_config(device="auto", mixed_precision=True)
    
    # Should detect appropriate device
    assert config.device in ["cpu", "cuda", "mps"]
    
    # Test setup with small model
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    optimized_model, hardware_info = setup_optimal_hardware(model, config)
    device = hardware_info['device']
    
    # Model should be moved to detected device
    model_device = next(optimized_model.parameters()).device
    assert model_device.type == device.type
    
    # Test forward pass works
    x = torch.randn(3, 4).to(device)
    output = optimized_model(x)
    assert output.shape == (3, 2)
    assert output.device.type == device.type


def test_deterministic_ensemble_reproducibility():
    """Test that ensemble predictions are consistent and deterministic."""
    # Create models with deterministic initialization
    seed_all(42)
    models = [nn.Linear(5, 3) for _ in range(3)]
    ensemble = DeepEnsemble(models)
    
    # Fixed input
    x = torch.randn(4, 5)
    
    # Multiple predictions should be identical (no randomness in forward pass)
    pred1 = ensemble(x)
    pred2 = ensemble(x)
    
    # Should be identical (deterministic forward pass)
    assert torch.equal(pred1, pred2)
    
    # Test that different models give different results
    seed_all(999)
    different_models = [nn.Linear(5, 3) for _ in range(3)]
    different_ensemble = DeepEnsemble(different_models)
    
    pred_different = different_ensemble(x)
    # Different models should give different predictions
    assert not torch.allclose(pred1, pred_different, atol=1e-4)


def test_monte_carlo_dropout_consistency():
    """Test MC Dropout gives consistent uncertainty estimates."""
    model = nn.Sequential(
        nn.Linear(6, 12),
        nn.Dropout(0.2),
        nn.ReLU(), 
        nn.Linear(12, 4),
        nn.Dropout(0.2),
        nn.Linear(4, 2)
    )
    
    mc_dropout = MonteCarloDropout(dropout_rate=0.2, n_samples=20)
    x = torch.randn(5, 6)
    
    # Get predictions with uncertainty
    mean_pred, uncertainty = mc_dropout.predict_with_uncertainty(model, x)
    
    assert mean_pred.shape == (5, 2)
    assert uncertainty.shape == (5, 2)
    
    # Uncertainty should be positive
    assert torch.all(uncertainty >= 0)
    
    # With more samples, uncertainty should be more stable
    mc_dropout_more = MonteCarloDropout(dropout_rate=0.2, n_samples=50)
    mean_pred2, uncertainty2 = mc_dropout_more.predict_with_uncertainty(model, x)
    
    # Mean predictions should be similar
    assert torch.allclose(mean_pred, mean_pred2, atol=0.1)


def test_ensemble_vs_single_model_uncertainty():
    """Test that ensembles provide better uncertainty than single models."""
    # Single model
    single_model = nn.Linear(4, 3)
    
    # Ensemble of same architectures
    ensemble_models = [nn.Linear(4, 3) for _ in range(5)]
    ensemble = DeepEnsemble(ensemble_models)
    
    x = torch.randn(10, 4)
    
    # Single model prediction (no uncertainty)
    single_pred = F.log_softmax(single_model(x), dim=-1)
    
    # Ensemble prediction with uncertainty
    ensemble_pred = ensemble(x)
    ensemble_pred_with_unc, uncertainty = ensemble.forward_with_uncertainty(x)
    
    # Ensemble should provide meaningful uncertainty
    assert uncertainty.shape == (10,)
    assert torch.all(uncertainty >= 0)
    
    # Ensemble predictions should be log-probabilities
    assert torch.allclose(ensemble_pred.exp().sum(dim=1), torch.ones(10), atol=1e-5)


def test_memory_efficient_large_ensemble():
    """Test memory efficiency with larger ensembles."""
    # Create ensemble with many small models
    models = [nn.Linear(10, 5) for _ in range(10)]
    ensemble = DeepEnsemble(models)
    
    # Large batch
    x = torch.randn(100, 10)
    
    # Should handle large batch efficiently
    with torch.no_grad():  # Save memory
        log_probs = ensemble(x)
    
    assert log_probs.shape == (100, 5)
    assert torch.allclose(log_probs.exp().sum(dim=1), torch.ones(100), atol=1e-4)


def test_cross_module_integration():
    """Test integration between uncertainty, hardware utils, and BN freeze."""
    # Setup hardware
    hw_config = create_hardware_config(device="auto")
    
    # Create model with uncertainty components
    base_model = nn.Sequential(
        nn.Linear(8, 16),
        nn.BatchNorm1d(16), 
        nn.Dropout(0.1),
        nn.ReLU(),
        nn.Linear(16, 4)
    )
    
    # Freeze BN and move to optimal device
    freeze_batchnorm_running_stats(base_model)
    optimized_model, hardware_info = setup_optimal_hardware(base_model, hw_config)
    device = hardware_info['device']
    
    # Verify BN is frozen after optimization
    bn_layer = optimized_model[1]
    assert not bn_layer.track_running_stats
    
    # Create uncertainty ensemble from adapted model copies
    adapted_models = []
    for i in range(3):
        model_copy = nn.Sequential(
            nn.Linear(8, 16),
            nn.BatchNorm1d(16),
            nn.Dropout(0.1),  
            nn.ReLU(),
            nn.Linear(16, 4)
        ).to(device)
        freeze_batchnorm_running_stats(model_copy)
        adapted_models.append(model_copy)
    
    uncertainty_ensemble = DeepEnsemble(adapted_models)
    
    # Test uncertainty prediction on device
    test_x = torch.randn(5, 8).to(device)
    log_probs = uncertainty_ensemble(test_x)
    
    assert log_probs.device.type == device.type
    assert log_probs.shape == (5, 4)
    assert torch.allclose(log_probs.exp().sum(dim=1), torch.ones(5).to(device), atol=1e-5)


def test_numerical_stability_edge_cases():
    """Test numerical stability in edge cases."""
    # Test with very small probabilities
    models = [nn.Linear(3, 2) for _ in range(2)]
    ensemble = DeepEnsemble(models)
    
    # Initialize weights to produce very small/large logits
    for model in models:
        with torch.no_grad():
            model.weight.fill_(0.001)
            model.bias.fill_(-10.0)  # Large negative bias
    
    x = torch.randn(4, 3)
    log_probs = ensemble(x)
    
    # Should still be finite and sum to 1
    assert torch.all(torch.isfinite(log_probs))
    assert torch.allclose(log_probs.exp().sum(dim=1), torch.ones(4), atol=1e-4)
    
    # Test with extreme weights  
    for model in models:
        with torch.no_grad():
            model.weight.fill_(100.0)  # Very large weights
            model.bias.fill_(0.0)
    
    log_probs_extreme = ensemble(x)
    assert torch.all(torch.isfinite(log_probs_extreme))
    assert torch.allclose(log_probs_extreme.exp().sum(dim=1), torch.ones(4), atol=1e-4)