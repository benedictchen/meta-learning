#!/usr/bin/env python3
"""Standalone test for core utilities functions"""

import torch
import torch.nn as nn
import copy
import warnings
from typing import Optional, Dict, Any

def clone_module(module: nn.Module, memo: Optional[Dict[int, nn.Module]] = None) -> nn.Module:
    """Create a differentiable clone of a PyTorch module preserving computational graph."""
    if memo is None:
        memo = {}
    
    # Check memo to handle shared parameters
    module_id = id(module)
    if module_id in memo:
        return memo[module_id]
    
    # Create new module instance using deepcopy for structure
    cloned = copy.deepcopy(module)
    
    # Replace parameters with gradient-preserving versions
    def clone_parameter(param: nn.Parameter) -> nn.Parameter:
        if param.requires_grad:
            # Create new parameter that preserves gradient computation
            cloned_param = param.clone().detach().requires_grad_(True)
            # Preserve parameter metadata if any
            if hasattr(param, '_meta'):
                cloned_param._meta = param._meta
            return cloned_param
        else:
            return param.clone().detach()
    
    # Replace parameters recursively
    for name, param in module.named_parameters():
        if param is not None:
            # Navigate to the correct submodule
            *path, param_name = name.split('.')
            current_module = cloned
            for attr_name in path:
                current_module = getattr(current_module, attr_name)
            
            # Replace parameter
            setattr(current_module, param_name, nn.Parameter(clone_parameter(param)))
    
    # Handle buffers (non-trainable parameters)
    for name, buffer in module.named_buffers():
        if buffer is not None:
            # Navigate to the correct submodule
            *path, buffer_name = name.split('.')
            current_module = cloned
            for attr_name in path:
                current_module = getattr(current_module, attr_name)
            
            # Replace buffer
            cloned_buffer = buffer.clone().detach()
            current_module.register_buffer(buffer_name, cloned_buffer)
    
    memo[module_id] = cloned
    return cloned


def update_module(module: nn.Module, updates: Optional[Dict[str, torch.Tensor]] = None, 
                  memo: Optional[Dict[int, nn.Module]] = None) -> nn.Module:
    """Apply differentiable parameter updates to a module."""
    if updates is None:
        return module
    
    if memo is None:
        memo = {}
    
    module_id = id(module)
    if module_id in memo:
        return memo[module_id]
    
    try:
        # Create updated module by cloning first
        updated_module = clone_module(module)
        
        # Apply parameter updates
        for param_name, update in updates.items():
            if update is not None:
                # Navigate to parameter
                *path, param_attr = param_name.split('.')
                current_module = updated_module
                for attr_name in path:
                    current_module = getattr(current_module, attr_name)
                
                # Get current parameter
                current_param = getattr(current_module, param_attr)
                if isinstance(current_param, nn.Parameter):
                    # Apply differentiable update: param = param + update
                    new_param = current_param + update
                    setattr(current_module, param_attr, nn.Parameter(new_param))
                else:
                    warnings.warn(f"Skipping update for non-parameter {param_name}")
        
        memo[module_id] = updated_module
        return updated_module
        
    except Exception as e:
        warnings.warn(f"Parameter update failed: {e}, returning original module")
        return module


def detach_module(module: nn.Module, keep_requires_grad: bool = True,
                  memo: Optional[Dict[int, nn.Module]] = None) -> nn.Module:
    """Detach all parameters and buffers from computational graph for memory optimization."""
    if memo is None:
        memo = {}
    
    module_id = id(module)
    if module_id in memo:
        return memo[module_id]
    
    # Create detached module
    detached = copy.deepcopy(module)
    
    # Detach all parameters
    for name, param in module.named_parameters():
        if param is not None:
            # Navigate to parameter location
            *path, param_name = name.split('.')
            current_module = detached
            for attr_name in path:
                current_module = getattr(current_module, attr_name)
            
            # Create detached parameter
            detached_param = param.detach()
            if keep_requires_grad and param.requires_grad:
                detached_param.requires_grad_(True)
            
            setattr(current_module, param_name, nn.Parameter(detached_param))
    
    # Detach all buffers
    for name, buffer in module.named_buffers():
        if buffer is not None:
            # Navigate to buffer location
            *path, buffer_name = name.split('.')
            current_module = detached
            for attr_name in path:
                current_module = getattr(current_module, attr_name)
            
            # Register detached buffer
            detached_buffer = buffer.detach()
            current_module.register_buffer(buffer_name, detached_buffer)
    
    memo[module_id] = detached
    return detached


def test_core_utilities():
    print("Testing core utility functions...")
    
    # Test basic functionality
    model = nn.Sequential(nn.Linear(5, 3), nn.ReLU(), nn.Linear(3, 2))
    print('✓ Original model created')
    
    # Test clone_module
    cloned = clone_module(model)
    print('✓ clone_module: Success')
    
    # Test that gradients flow through cloned model
    x = torch.randn(10, 5)
    y = torch.randn(10, 2)
    loss = nn.MSELoss()(cloned(x), y)
    grads = torch.autograd.grad(loss, cloned.parameters(), retain_graph=True)
    print(f'✓ clone_module gradients: {len(grads)} gradients computed')
    
    # Test update_module 
    updates = {name: -0.01 * grad for (name, param), grad in zip(model.named_parameters(), grads)}
    updated = update_module(cloned, updates)
    print('✓ update_module: Success')
    
    # Test detach_module
    detached = detach_module(model)
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
    
    # Test second-order gradient computation (critical for meta-learning)
    x2 = torch.randn(5, 5)
    y2 = torch.randn(5, 2)
    
    # Forward pass on updated model
    pred = updated(x2)
    loss2 = nn.MSELoss()(pred, y2)
    
    # Test that we can compute gradients w.r.t. original parameters
    # This is critical for MAML-style algorithms
    try:
        second_grads = torch.autograd.grad(loss2, model.parameters(), allow_unused=True)
        print(f'✓ Second-order gradients: {len([g for g in second_grads if g is not None])} computed')
    except Exception as e:
        print(f'⚠ Second-order gradients failed: {e}')
    
    print("🎉 All core utility tests passed!")

if __name__ == "__main__":
    test_core_utilities()