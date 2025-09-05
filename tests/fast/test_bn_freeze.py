"""
💰 DONATE NOW! 💰 https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS

Fast test for BatchNorm freezing functionality.

If BN leakage prevention helps your research avoid invalid results,
please donate to support continued algorithm development!

Author: Benedict Chen (benedict@benedictchen.com)
GitHub Sponsors: https://github.com/sponsors/benedictchen
"""

import torch
import torch.nn as nn
from meta_learning.core.bn_policy import freeze_batchnorm_running_stats


def test_freeze_batchnorm_running_stats():
    """Test that BN running stats are properly frozen."""
    # Create a model with BatchNorm
    model = nn.Sequential(
        nn.Linear(10, 16),
        nn.BatchNorm1d(16),
        nn.ReLU(),
        nn.Linear(16, 5)
    )
    
    # Get the BatchNorm layer
    bn_layer = model[1]
    
    # Initialize some dummy running stats
    bn_layer.running_mean.fill_(1.0)
    bn_layer.running_var.fill_(2.0)
    
    # Store original values
    original_mean = bn_layer.running_mean.clone()
    original_var = bn_layer.running_var.clone()
    
    # Freeze BN running stats
    freeze_batchnorm_running_stats(model)
    
    # Check that tracking is disabled
    assert not bn_layer.track_running_stats, "BN tracking should be disabled"
    
    # Check that requires_grad is False
    assert not bn_layer.running_mean.requires_grad, "Running mean should not require grad"
    assert not bn_layer.running_var.requires_grad, "Running var should not require grad"
    
    # Run some data through the model in training mode
    model.train()
    x = torch.randn(32, 10)
    _ = model(x)
    
    # Check that running stats didn't change (with explicit tracking verification)
    assert not bn_layer.track_running_stats, "track_running_stats should be False"
    assert torch.allclose(bn_layer.running_mean, original_mean, atol=0, rtol=0), \
        f"Running mean changed despite freezing: {bn_layer.running_mean} vs {original_mean}"
    assert torch.allclose(bn_layer.running_var, original_var, atol=0, rtol=0), \
        f"Running var changed despite freezing: {bn_layer.running_var} vs {original_var}"


def test_multiple_bn_layers():
    """Test freezing with multiple BN layers."""
    model = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.Conv2d(16, 32, 3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(32, 10)
    )
    
    # Freeze all BN layers
    freeze_batchnorm_running_stats(model)
    
    # Check all BN layers are frozen
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            assert not module.track_running_stats, "All BN layers should have tracking disabled"
            if module.running_mean is not None:
                assert not module.running_mean.requires_grad, "Running mean should not require grad"
            if module.running_var is not None:
                assert not module.running_var.requires_grad, "Running var should not require grad"


def test_freeze_preserves_other_parameters():
    """Test that freezing BN doesn't affect other parameters."""
    model = nn.Sequential(
        nn.Linear(10, 16),
        nn.BatchNorm1d(16),
        nn.ReLU(),
        nn.Linear(16, 5)
    )
    
    # Check that other parameters still require grad
    linear1_weight_requires_grad = model[0].weight.requires_grad
    linear2_weight_requires_grad = model[3].weight.requires_grad
    
    freeze_batchnorm_running_stats(model)
    
    # Other parameters should still require grad
    assert model[0].weight.requires_grad == linear1_weight_requires_grad
    assert model[3].weight.requires_grad == linear2_weight_requires_grad
    
    # BN weight and bias should still require grad (only running stats frozen)
    bn_layer = model[1]
    assert bn_layer.weight.requires_grad, "BN weight should still require grad"
    assert bn_layer.bias.requires_grad, "BN bias should still require grad"