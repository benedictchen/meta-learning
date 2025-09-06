#!/usr/bin/env python3
"""Debug Meta-SGD learning rates."""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add the src directory to Python path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from meta_learning.algorithms.meta_sgd import MetaSGD

def debug_learning_rates():
    """Debug why learning rates aren't updating."""
    print("🔍 Debugging Meta-SGD learning rates...")
    
    # Create simple model
    model = nn.Linear(3, 2)
    meta_sgd = MetaSGD(model, lr=0.05)
    
    print(f"Number of learning rate parameters: {len(meta_sgd.lrs)}")
    print(f"Learning rate requires_grad: {[lr.requires_grad for lr in meta_sgd.lrs]}")
    
    # Check if learning rates are in meta_sgd.parameters()
    all_params = list(meta_sgd.parameters())
    print(f"Total MetaSGD parameters: {len(all_params)}")
    
    # Separate module params from lr params
    module_params = list(meta_sgd.module.parameters())
    lr_params = list(meta_sgd.lrs)
    
    print(f"Module parameters: {len(module_params)}")
    print(f"LR parameters: {len(lr_params)}")
    
    # Check if LR params are properly included
    lr_param_ids = {id(p) for p in lr_params}
    all_param_ids = {id(p) for p in all_params}
    
    lr_in_all = lr_param_ids.intersection(all_param_ids)
    print(f"LR params in all params: {len(lr_in_all)}/{len(lr_param_ids)}")
    
    if len(lr_in_all) != len(lr_param_ids):
        print("❌ Learning rate parameters are NOT included in meta_sgd.parameters()!")
        print("This is why they're not being updated by the meta-optimizer")
        return False
    else:
        print("✅ Learning rate parameters are properly included")
        return True

if __name__ == "__main__":
    debug_learning_rates()