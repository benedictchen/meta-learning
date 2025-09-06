#!/usr/bin/env python3
"""Debug gradient flow in Meta-SGD."""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add the src directory to Python path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from meta_learning.algorithms.meta_sgd import MetaSGD

def debug_gradient_flow():
    """Debug gradient flow step by step."""
    print("🔍 Debugging gradient flow in Meta-SGD...")
    
    # Simple model
    model = nn.Linear(2, 1)
    meta_sgd = MetaSGD(model, lr=0.5, first_order=False)
    
    print(f"Learning rate requires grad: {meta_sgd.lrs[0].requires_grad}")
    print(f"Learning rate grad_fn: {meta_sgd.lrs[0].grad_fn}")
    
    # Simple data
    x = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
    y = torch.tensor([[0.5]], dtype=torch.float32)
    
    # Clone model for adaptation
    task_model = meta_sgd.clone()
    print(f"Cloned model LR requires grad: {task_model.lrs[0].requires_grad}")
    print(f"Cloned model LR grad_fn: {task_model.lrs[0].grad_fn}")
    
    # Forward pass
    pred = task_model(x)
    print(f"Prediction: {pred}")
    print(f"Prediction requires_grad: {pred.requires_grad}")
    print(f"Prediction grad_fn: {pred.grad_fn}")
    
    # Compute loss
    loss = F.mse_loss(pred, y)
    print(f"Loss: {loss.item():.4f}")
    print(f"Loss requires_grad: {loss.requires_grad}")
    print(f"Loss grad_fn: {loss.grad_fn}")
    
    # Before adaptation - check if LR has gradients
    if task_model.lrs[0].grad is not None:
        print(f"LR grad before adaptation: {task_model.lrs[0].grad}")
    else:
        print("LR grad before adaptation: None")
    
    # Perform adaptation
    print("\n--- Performing adaptation ---")
    task_model.adapt(loss)
    
    # After adaptation - check model parameters
    adapted_pred = task_model(x)
    print(f"Adapted prediction: {adapted_pred}")
    print(f"Adapted prediction requires_grad: {adapted_pred.requires_grad}")
    print(f"Adapted prediction grad_fn: {adapted_pred.grad_fn}")
    
    # Meta loss
    meta_loss = F.mse_loss(adapted_pred, y)
    print(f"\nMeta-loss: {meta_loss.item():.4f}")
    print(f"Meta-loss requires_grad: {meta_loss.requires_grad}")
    print(f"Meta-loss grad_fn: {meta_loss.grad_fn}")
    
    # Check if learning rates are in computational graph
    print("\n--- Checking computational graph ---")
    if meta_loss.grad_fn is not None:
        print("Meta-loss has grad_fn - computational graph exists")
        
        # Check if we can compute gradients w.r.t. learning rates
        try:
            lr_grad = torch.autograd.grad(meta_loss, task_model.lrs[0], retain_graph=True)
            print(f"LR gradient computed: {lr_grad[0]}")
            return True
        except RuntimeError as e:
            print(f"❌ Cannot compute LR gradient: {e}")
            return False
    else:
        print("❌ Meta-loss has no grad_fn - no computational graph")
        return False

def debug_lr_connection():
    """Debug if learning rates are connected to the forward pass."""
    print("\n🔍 Debugging LR connection to forward pass...")
    
    model = nn.Linear(1, 1)
    meta_sgd = MetaSGD(model, lr=1.0, first_order=False)
    
    # Manually check what happens in adaptation
    x = torch.tensor([[1.0]])
    y = torch.tensor([[0.0]])
    
    # Forward pass to get loss
    pred = meta_sgd(x)
    loss = F.mse_loss(pred, y)
    
    print(f"Initial loss: {loss.item():.4f}")
    
    # Get gradients manually  
    grads = torch.autograd.grad(loss, meta_sgd.module.parameters(), create_graph=True)
    print(f"Gradients computed: {[g.shape for g in grads]}")
    
    # Check if gradients have grad_fn (for second-order)
    for i, g in enumerate(grads):
        print(f"Gradient {i} grad_fn: {g.grad_fn}")
    
    # Manual update using learning rates
    updated_params = []
    for param, lr, grad in zip(meta_sgd.module.parameters(), meta_sgd.lrs, grads):
        updated_param = param - lr * grad  # This should preserve gradients!
        updated_params.append(updated_param)
        print(f"Updated param grad_fn: {updated_param.grad_fn}")
        print(f"LR grad_fn: {lr.grad_fn}")
    
    return True

if __name__ == "__main__":
    success1 = debug_gradient_flow()
    success2 = debug_lr_connection()
    
    if success1 and success2:
        print("\n✅ Gradient flow debugging complete")
    else:
        print("\n❌ Issues found in gradient flow")