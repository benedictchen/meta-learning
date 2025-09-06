#!/usr/bin/env python3
"""Simple test to verify Meta-SGD learning rate updates work."""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add the src directory to Python path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from meta_learning.algorithms.meta_sgd import MetaSGD

def test_simple_meta_sgd():
    """Simple test that should show learning rate updates."""
    print("🧪 Simple Meta-SGD learning rate update test...")
    
    # Very simple model
    model = nn.Linear(1, 1)
    meta_sgd = MetaSGD(model, lr=1.0)
    
    # Meta-optimizer that includes learning rates
    meta_optimizer = torch.optim.SGD(meta_sgd.parameters(), lr=0.1)
    
    print("Initial parameters:")
    for name, param in meta_sgd.named_parameters():
        print(f"  {name}: {param.data}")
    
    # Simple data - we want model to output 2.0 when input is 1.0
    x = torch.tensor([[1.0]])
    target = torch.tensor([[2.0]])
    
    # Meta-training loop
    for step in range(3):
        print(f"\n--- Step {step + 1} ---")
        
        # Clone model for adaptation
        task_model = meta_sgd.clone()
        
        # Inner loop: adapt to task
        prediction = task_model(x)
        loss = F.mse_loss(prediction, target)
        print(f"Loss before adaptation: {loss.item():.4f}")
        
        # Adapt
        task_model.adapt(loss)
        
        # Check adaptation
        adapted_prediction = task_model(x)
        adapted_loss = F.mse_loss(adapted_prediction, target)
        print(f"Loss after adaptation: {adapted_loss.item():.4f}")
        
        # Meta-loss (using adapted model)
        meta_loss = adapted_loss
        
        # Meta-update
        meta_optimizer.zero_grad()
        meta_loss.backward()
        
        # Check if learning rates have gradients
        print("Learning rate gradients:")
        for i, lr in enumerate(task_model.lrs):
            if lr.grad is not None:
                print(f"  LR {i}: {lr.grad.data}")
            else:
                print(f"  LR {i}: None")
        
        # Check original learning rates gradients
        print("Original learning rate gradients:")
        for i, lr in enumerate(meta_sgd.lrs):
            if lr.grad is not None:
                print(f"  Original LR {i}: {lr.grad.data}")
            else:
                print(f"  Original LR {i}: None")
        
        meta_optimizer.step()
        
        # Print updated parameters
        print("Updated parameters:")
        for name, param in meta_sgd.named_parameters():
            print(f"  {name}: {param.data}")
    
    return True

if __name__ == "__main__":
    test_simple_meta_sgd()