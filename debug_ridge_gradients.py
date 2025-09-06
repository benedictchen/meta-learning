#!/usr/bin/env python3
"""Debug Ridge regression gradient flow."""

import sys
import os
import torch
import torch.nn.functional as F

# Add the src directory to Python path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from meta_learning.algorithms.metaoptnet import DifferentiableRidge

def debug_ridge_gradients():
    """Debug what's happening with Ridge gradients."""
    print("🔍 Debugging Ridge regression gradients...")
    
    # Create Ridge solver
    ridge = DifferentiableRidge(lam=1.0)
    
    # Create data with requires_grad
    support_embeddings = torch.randn(10, 8, requires_grad=True)
    support_labels = torch.randint(0, 3, (10,))
    query_embeddings = torch.randn(5, 8, requires_grad=True)
    
    print(f"Support embeddings requires_grad: {support_embeddings.requires_grad}")
    print(f"Query embeddings requires_grad: {query_embeddings.requires_grad}")
    
    # Forward pass
    predictions = ridge(support_embeddings, support_labels, query_embeddings)
    
    print(f"Predictions requires_grad: {predictions.requires_grad}")
    print(f"Predictions grad_fn: {predictions.grad_fn}")
    
    # Try to backpropagate
    if predictions.requires_grad:
        loss = predictions.sum()
        loss.backward()
        print(f"Support grad exists: {support_embeddings.grad is not None}")
        print(f"Query grad exists: {query_embeddings.grad is not None}")
    else:
        print("❌ Cannot backpropagate - no grad_fn")
    
    return predictions.requires_grad

if __name__ == "__main__":
    success = debug_ridge_gradients()
    if success:
        print("✅ Ridge gradients working")
    else:
        print("❌ Ridge gradients broken")