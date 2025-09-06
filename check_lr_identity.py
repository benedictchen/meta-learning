#!/usr/bin/env python3
"""Check if learning rates are properly shared."""

import sys
import os
import torch
import torch.nn as nn

# Add the src directory to Python path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from meta_learning.algorithms.meta_sgd import MetaSGD

def check_lr_identity():
    """Check if cloned model shares learning rate objects."""
    print("🔍 Checking learning rate identity...")
    
    model = nn.Linear(2, 1)
    meta_sgd = MetaSGD(model, lr=0.5)
    
    # Clone the model
    cloned = meta_sgd.clone()
    
    # Check if learning rates are the same objects
    for i, (orig_lr, cloned_lr) in enumerate(zip(meta_sgd.lrs, cloned.lrs)):
        print(f"LR {i}: Same object? {orig_lr is cloned_lr}")
        print(f"LR {i}: Values equal? {torch.allclose(orig_lr, cloned_lr)}")
        print(f"LR {i}: Same ID? Original: {id(orig_lr)}, Cloned: {id(cloned_lr)}")
    
    # Try modifying original learning rates
    with torch.no_grad():
        meta_sgd.lrs[0].fill_(0.1)
    
    print(f"After modifying original:")
    print(f"Original LR value: {meta_sgd.lrs[0].item()}")
    print(f"Cloned LR value: {cloned.lrs[0].item()}")
    
    # Check if they are the same parameter list
    print(f"ParameterList identity: {meta_sgd.lrs is cloned.lrs}")
    
    return meta_sgd.lrs is cloned.lrs

if __name__ == "__main__":
    same = check_lr_identity()
    if same:
        print("✅ Learning rates are properly shared")
    else:
        print("❌ Learning rates are NOT shared")