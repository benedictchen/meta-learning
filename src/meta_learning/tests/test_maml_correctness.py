"""
Meta-Gradient Correctness Tests
==============================

This module provides regression tests to validate that MAML implementation
produces mathematically correct meta-gradients. It compares autograd results
with finite differences to catch gradient flow issues.

Key Tests:
1. ✅ Meta-gradient finite difference validation
2. ✅ Second-order vs first-order gradient comparison  
3. ✅ Gradient connectivity validation
4. ✅ BatchNorm isolation verification
5. ✅ ANIL/BOIL parameter partitioning validation

These tests will catch regressions where code "works but has wrong math."
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest
import math
from typing import List, Tuple

from ..core.maml_correct import (
    StatelessMAML, inner_adapt, query_forward, 
    get_all_param_names, get_anil_adapt_names, get_boil_adapt_names,
    validate_param_connectivity, set_episodic_bn_mode
)


class TinyTestNet(nn.Module):
    """Minimal network for meta-gradient testing."""
    
    def __init__(self, d_in: int = 5, d_hidden: int = 4, d_out: int = 3):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(),
        )
        self.head = nn.Linear(d_hidden, d_out)
        
    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)


class TinyNetWithBN(nn.Module):
    """Network with BatchNorm for isolation testing."""
    
    def __init__(self, d_in: int = 5, d_hidden: int = 4, d_out: int = 3):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.BatchNorm1d(d_hidden),
            nn.ReLU(),
        )
        self.head = nn.Linear(d_hidden, d_out)
        
    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)


def create_test_episodes(n_episodes: int = 3, n_support: int = 8, n_query: int = 8) -> List[Tuple]:
    """Create synthetic episodes for testing."""
    torch.manual_seed(42)  # Deterministic for testing
    episodes = []
    
    for _ in range(n_episodes):
        # Create synthetic few-shot episode
        support_x = torch.randn(n_support, 5)
        support_y = torch.randint(0, 3, (n_support,))
        query_x = torch.randn(n_query, 5) 
        query_y = torch.randint(0, 3, (n_query,))
        episodes.append((support_x, support_y, query_x, query_y))
    
    return episodes


def finite_difference_meta_grad(
    model: nn.Module,
    episodes: List[Tuple],
    param_name: str,
    inner_steps: int = 2,
    inner_lr: float = 0.1,
    adapt_param_names: List[str] = None,
    eps: float = 1e-4,
) -> float:
    """
    Compute meta-gradient using finite differences for validation.
    
    Args:
        model: Model to test
        episodes: List of (support_x, support_y, query_x, query_y) tuples
        param_name: Parameter to compute gradient for
        inner_steps: Number of inner adaptation steps
        inner_lr: Inner learning rate
        adapt_param_names: Parameters to adapt (None = all)
        eps: Finite difference epsilon
        
    Returns:
        Finite difference approximation of meta-gradient
    """
    if adapt_param_names is None:
        adapt_param_names = get_all_param_names(model)
    
    loss_fn = nn.CrossEntropyLoss()
    
    def compute_meta_loss(param_offset: float = 0.0) -> float:
        """Compute meta-loss with parameter perturbation."""
        # Apply parameter perturbation
        param_dict = dict(model.named_parameters())
        original_value = param_dict[param_name].data.clone()
        
        with torch.no_grad():
            param_dict[param_name].data.add_(param_offset)
        
        try:
            total_loss = 0.0
            for support_x, support_y, query_x, query_y in episodes:
                # Inner loop adaptation
                adapted_params = inner_adapt(
                    model, support_x, support_y, loss_fn,
                    inner_steps, inner_lr, adapt_param_names, use_second_order=False
                )
                
                # Query loss
                query_logits = query_forward(model, query_x, adapted_params)
                task_loss = loss_fn(query_logits, query_y)
                total_loss += task_loss.item()
            
            return total_loss / len(episodes)
        
        finally:
            # Restore original parameter value
            with torch.no_grad():
                param_dict[param_name].data.copy_(original_value)
    
    # Central difference approximation
    loss_plus = compute_meta_loss(eps)
    loss_minus = compute_meta_loss(-eps)
    
    return (loss_plus - loss_minus) / (2 * eps)


def test_meta_gradient_finite_difference():
    """Test that autograd meta-gradients match finite differences."""
    # Create test setup
    model = TinyTestNet()
    episodes = create_test_episodes(n_episodes=2)  # Small for speed
    loss_fn = nn.CrossEntropyLoss()
    adapt_param_names = get_all_param_names(model)
    
    # Compute meta-gradient via autograd
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    
    total_meta_loss = 0.0
    for support_x, support_y, query_x, query_y in episodes:
        adapted_params = inner_adapt(
            model, support_x, support_y, loss_fn,
            inner_steps=2, inner_lr=0.1, adapt_param_names=adapt_param_names,
            use_second_order=True  # Enable second-order for meta-gradients
        )
        
        query_logits = query_forward(model, query_x, adapted_params)
        task_loss = loss_fn(query_logits, query_y)
        task_loss.backward()
        total_meta_loss += task_loss.item()
    
    # Get autograd gradient for a specific parameter
    test_param_name = "backbone.0.weight"
    autograd_grad = model.backbone[0].weight.grad.mean().item()
    
    # Compute finite difference gradient
    finite_diff_grad = finite_difference_meta_grad(
        model, episodes, test_param_name,
        inner_steps=2, inner_lr=0.1, adapt_param_names=adapt_param_names
    )
    
    # Validate gradients match within tolerance
    assert math.isfinite(autograd_grad), "Autograd gradient is not finite"
    assert math.isfinite(finite_diff_grad), "Finite difference gradient is not finite"
    
    # Relative tolerance for numerical comparison
    relative_error = abs(autograd_grad - finite_diff_grad) / (abs(finite_diff_grad) + 1e-6)
    assert relative_error < 0.15, (
        f"Meta-gradient mismatch: autograd={autograd_grad:.6f}, "
        f"finite_diff={finite_diff_grad:.6f}, relative_error={relative_error:.4f}"
    )


def test_second_order_vs_first_order():
    """Test that second-order produces different gradients than first-order."""
    model = TinyTestNet()
    episodes = create_test_episodes(n_episodes=1)
    loss_fn = nn.CrossEntropyLoss()
    adapt_param_names = get_all_param_names(model)
    
    support_x, support_y, query_x, query_y = episodes[0]
    
    # Test second-order MAML
    model_second = TinyTestNet()
    model_second.load_state_dict(model.state_dict())  # Same initialization
    
    optimizer_second = torch.optim.SGD(model_second.parameters(), lr=1e-3)
    optimizer_second.zero_grad(set_to_none=True)
    
    adapted_params_second = inner_adapt(
        model_second, support_x, support_y, loss_fn,
        inner_steps=2, inner_lr=0.1, adapt_param_names=adapt_param_names,
        use_second_order=True
    )
    
    query_logits_second = query_forward(model_second, query_x, adapted_params_second)
    loss_second = loss_fn(query_logits_second, query_y)
    loss_second.backward()
    
    # Test first-order MAML (FOMAML)
    model_first = TinyTestNet()
    model_first.load_state_dict(model.state_dict())  # Same initialization
    
    optimizer_first = torch.optim.SGD(model_first.parameters(), lr=1e-3)
    optimizer_first.zero_grad(set_to_none=True)
    
    adapted_params_first = inner_adapt(
        model_first, support_x, support_y, loss_fn,
        inner_steps=2, inner_lr=0.1, adapt_param_names=adapt_param_names,
        use_second_order=False
    )
    
    query_logits_first = query_forward(model_first, query_x, adapted_params_first)
    loss_first = loss_fn(query_logits_first, query_y)
    loss_first.backward()
    
    # Compare gradients - they should be different
    grad_second = model_second.backbone[0].weight.grad.mean().item()
    grad_first = model_first.backbone[0].weight.grad.mean().item()
    
    # Gradients should be different (second-order includes Hessian terms)
    relative_diff = abs(grad_second - grad_first) / (abs(grad_second) + 1e-6)
    assert relative_diff > 0.01, (
        f"Second-order and first-order gradients are too similar: "
        f"second={grad_second:.6f}, first={grad_first:.6f}, diff={relative_diff:.4f}"
    )


def test_gradient_connectivity():
    """Test that all adapted parameters have proper gradient connectivity."""
    model = TinyTestNet()
    episodes = create_test_episodes(n_episodes=1)
    loss_fn = nn.CrossEntropyLoss()
    adapt_param_names = get_all_param_names(model)
    
    support_x, support_y, query_x, query_y = episodes[0]
    
    # Perform adaptation
    adapted_params = inner_adapt(
        model, support_x, support_y, loss_fn,
        inner_steps=1, inner_lr=0.1, adapt_param_names=adapt_param_names,
        use_second_order=True
    )
    
    # Compute query loss
    query_logits = query_forward(model, query_x, adapted_params)
    query_loss = loss_fn(query_logits, query_y)
    
    # Validate connectivity (should not raise exception)
    validate_param_connectivity(model, adapted_params, query_loss)


def test_anil_boil_partitioning():
    """Test ANIL and BOIL parameter partitioning."""
    model = TinyTestNet()
    
    # Test ANIL partitioning (adapt head only)
    anil_names = get_anil_adapt_names(model, ['head'])
    expected_anil = ['head.weight', 'head.bias']
    assert set(anil_names) == set(expected_anil), f"ANIL names: {anil_names}"
    
    # Test BOIL partitioning (adapt backbone, freeze head)
    boil_names = get_boil_adapt_names(model, ['head'])
    expected_boil = ['backbone.0.weight', 'backbone.0.bias']
    assert set(boil_names) == set(expected_boil), f"BOIL names: {boil_names}"
    
    # Test that ANIL + BOIL = all parameters
    all_names = set(get_all_param_names(model))
    anil_set = set(anil_names)
    boil_set = set(boil_names)
    assert anil_set.union(boil_set) == all_names
    assert anil_set.intersection(boil_set) == set()  # No overlap


def test_batchnorm_isolation():
    """Test that BatchNorm is properly isolated per episode."""
    model = TinyNetWithBN()
    
    # Before episodic mode
    bn_layer = model.backbone[1]  # BatchNorm1d layer
    assert bn_layer.training, "BN should start in training mode"
    
    # Apply episodic BN policy
    set_episodic_bn_mode(model)
    
    # After episodic mode
    assert not bn_layer.training, "BN should be in eval mode (frozen stats)"
    
    # But affine parameters should still require gradients
    assert bn_layer.weight.requires_grad, "BN affine parameters should be trainable"
    assert bn_layer.bias.requires_grad, "BN affine parameters should be trainable"


def test_stateless_maml_class():
    """Test the StatelessMAML wrapper class."""
    model = TinyTestNet()
    episodes = create_test_episodes(n_episodes=1)
    loss_fn = nn.CrossEntropyLoss()
    
    # Create StatelessMAML instance
    maml = StatelessMAML(
        model=model,
        inner_lr=0.1,
        inner_steps=2,
        use_second_order=True
    )
    
    support_x, support_y, query_x, query_y = episodes[0]
    
    # Test forward pass
    query_loss, query_logits = maml.forward(support_x, support_y, query_x, query_y, loss_fn)
    
    # Validate outputs
    assert query_loss.requires_grad, "Query loss should require gradients"
    assert query_logits.shape == (query_x.size(0), 3), "Logits shape incorrect"
    
    # Test adapt_only method
    adapted_params = maml.adapt_only(support_x, support_y, loss_fn)
    assert len(adapted_params) == len(list(model.parameters())), "Missing adapted parameters"


def test_no_data_mutation():
    """Test that original model parameters are never mutated via .data."""
    model = TinyTestNet()
    episodes = create_test_episodes(n_episodes=1)
    loss_fn = nn.CrossEntropyLoss()
    
    # Store original parameter values
    original_params = {name: param.data.clone() for name, param in model.named_parameters()}
    
    support_x, support_y, query_x, query_y = episodes[0]
    
    # Perform inner adaptation
    adapted_params = inner_adapt(
        model, support_x, support_y, loss_fn,
        inner_steps=2, inner_lr=0.1, 
        adapt_param_names=get_all_param_names(model),
        use_second_order=True
    )
    
    # Verify original parameters unchanged
    for name, param in model.named_parameters():
        torch.testing.assert_close(
            param.data, original_params[name],
            msg=f"Parameter {name} was mutated during inner adaptation"
        )


if __name__ == "__main__":
    # Run tests manually for debugging
    test_meta_gradient_finite_difference()
    test_second_order_vs_first_order()
    test_gradient_connectivity()
    test_anil_boil_partitioning()
    test_batchnorm_isolation()
    test_stateless_maml_class()
    test_no_data_mutation()
    print("✅ All mathematical correctness tests passed!")