#!/usr/bin/env python3
"""Test script for core utilities functions"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the functions directly by reading and executing the file
import importlib.util

# Load utils module
utils_path = os.path.join(os.path.dirname(__file__), 'src', 'meta_learning', 'core', 'utils.py')
spec = importlib.util.spec_from_file_location("utils", utils_path)
utils = importlib.util.module_from_spec(spec)

# Mock the Episode import to avoid circular dependency for testing
class MockEpisode:
    def __init__(self, support_x, support_y, query_x, query_y):
        self.support_x = support_x
        self.support_y = support_y
        self.query_x = query_x
        self.query_y = query_y

# Add mock to utils namespace
utils.Episode = MockEpisode
spec.loader.exec_module(utils)

def test_core_utilities():
    print("Testing core utility functions...")
    
    # Test basic functionality
    model = nn.Sequential(nn.Linear(5, 3), nn.ReLU(), nn.Linear(3, 2))
    print('✓ Original model created')
    
    # Test clone_module
    cloned = utils.clone_module(model)
    print('✓ clone_module: Success')
    
    # Test that gradients flow through cloned model
    x = torch.randn(10, 5)
    y = torch.randn(10, 2)
    loss = nn.MSELoss()(cloned(x), y)
    grads = torch.autograd.grad(loss, cloned.parameters(), retain_graph=True)
    print(f'✓ clone_module gradients: {len(grads)} gradients computed')
    
    # Test update_module 
    updates = {name: -0.01 * grad for (name, param), grad in zip(model.named_parameters(), grads)}
    updated = utils.update_module(cloned, updates)
    print('✓ update_module: Success')
    
    # Test detach_module
    detached = utils.detach_module(model)
    print('✓ detach_module: Success')
    
    # Test parameter shapes are preserved
    orig_params = list(model.parameters())
    cloned_params = list(cloned.parameters())
    updated_params = list(updated.parameters())
    detached_params = list(detached.parameters())
    
    assert len(orig_params) == len(cloned_params) == len(updated_params) == len(detached_params)
    for i, (orig, clone, upd, det) in enumerate(zip(orig_params, cloned_params, updated_params, detached_params)):
        assert orig.shape == clone.shape == upd.shape == det.shape, f"Shape mismatch at parameter {i}"
    print('✓ Parameter shapes preserved')
    
    # Test gradient computation works correctly
    assert all(p.requires_grad for p in cloned_params), "Cloned parameters should require gradients"
    assert all(p.requires_grad for p in updated_params), "Updated parameters should require gradients"
    assert all(p.requires_grad for p in detached_params), "Detached parameters should require gradients (keep_requires_grad=True)"
    print('✓ Gradient requirements preserved')
    
    # Test safe wrappers
    safe_cloned = utils.safe_clone_module(model)
    safe_updated = utils.safe_update_module(model, updates)
    print('✓ Safe wrappers working')
    
    # Test validation
    valid = utils.validate_module_updates(model, updates)
    assert valid, "Updates should be valid"
    print('✓ Validation working')
    
    print("🎉 All core utility tests passed!")

if __name__ == "__main__":
    test_core_utilities()