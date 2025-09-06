#!/usr/bin/env python3
"""Test script for Ridge Regression implementation"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Mock Episode for testing
class MockEpisode:
    def __init__(self, support_x, support_y, query_x, query_y):
        self.support_x = support_x
        self.support_y = support_y
        self.query_x = query_x
        self.query_y = query_y

def test_ridge_regression():
    print("Testing Ridge Regression implementation...")
    
    # Create synthetic few-shot learning data
    np.random.seed(42)
    torch.manual_seed(42)
    
    n_classes = 3
    n_support_per_class = 5
    n_query_per_class = 3
    feature_dim = 10
    
    # Generate data
    support_data = []
    support_labels = []
    query_data = []
    query_labels = []
    
    for class_id in range(n_classes):
        # Create class-specific mean and some noise
        class_mean = torch.randn(feature_dim) * 2
        
        # Support data
        class_support = class_mean.unsqueeze(0) + 0.5 * torch.randn(n_support_per_class, feature_dim)
        support_data.append(class_support)
        support_labels.extend([class_id] * n_support_per_class)
        
        # Query data
        class_query = class_mean.unsqueeze(0) + 0.5 * torch.randn(n_query_per_class, feature_dim)
        query_data.append(class_query)
        query_labels.extend([class_id] * n_query_per_class)
    
    support_x = torch.cat(support_data, dim=0)
    support_y = torch.tensor(support_labels)
    query_x = torch.cat(query_data, dim=0)
    query_y = torch.tensor(query_labels)
    
    # Test the functions directly
    from src.meta_learning.algorithms.ridge_regression import (
        ridge_regression, standard_solver, woodbury_solver, 
        safe_ridge_regression, ridge_cross_validation
    )
    
    print(f"✓ Support set: {support_x.shape}, Query set: {query_x.shape}")
    
    # Test basic ridge regression
    try:
        weights, bias = ridge_regression(support_x, support_y, reg_lambda=0.1, num_classes=n_classes)
        print(f"✓ Ridge regression: weights {weights.shape}, bias {bias.shape if bias is not None else None}")
    except Exception as e:
        print(f"✗ Ridge regression failed: {e}")
        return False
    
    # Test standard solver
    try:
        targets_onehot = F.one_hot(support_y, num_classes=n_classes).float()
        X_with_bias = torch.cat([support_x, torch.ones(support_x.shape[0], 1)], dim=1)
        weights_std, cond_num = standard_solver(X_with_bias, targets_onehot, 0.1)
        print(f"✓ Standard solver: weights {weights_std.shape}, condition {cond_num:.2e}")
    except Exception as e:
        print(f"✗ Standard solver failed: {e}")
        return False
    
    # Test Woodbury solver
    try:
        weights_wood, cond_num = woodbury_solver(X_with_bias, targets_onehot, 0.1)
        print(f"✓ Woodbury solver: weights {weights_wood.shape}, condition {cond_num:.2e}")
    except Exception as e:
        print(f"✗ Woodbury solver failed: {e}")
        return False
    
    # Test safe wrapper
    try:
        safe_weights, safe_bias = safe_ridge_regression(support_x, support_y, reg_lambda=0.1, num_classes=n_classes)
        print(f"✓ Safe ridge regression: weights {safe_weights.shape}")
    except Exception as e:
        print(f"✗ Safe ridge regression failed: {e}")
        return False
    
    # Test cross-validation
    try:
        targets_for_cv = F.one_hot(support_y, num_classes=n_classes).float()
        best_lambda = ridge_cross_validation(support_x, targets_for_cv, n_folds=3)
        print(f"✓ Cross-validation: optimal lambda = {best_lambda}")
    except Exception as e:
        print(f"✗ Cross-validation failed: {e}")
        return False
    
    print("🎉 All Ridge Regression function tests passed!")
    
    # Test the Ridge Regression class
    from src.meta_learning.algorithms.ridge_regression import RidgeRegression
    
    try:
        # Create episode
        episode = MockEpisode(support_x, support_y, query_x, query_y)
        
        # Test Ridge Regression model
        ridge_model = RidgeRegression(reg_lambda=0.1, bias=True, scale=True)
        
        # Forward pass
        query_logits = ridge_model(episode)
        print(f"✓ RidgeRegression forward: {query_logits.shape}")
        
        # Check predictions make sense
        predictions = torch.argmax(query_logits, dim=1)
        accuracy = (predictions == query_y).float().mean()
        print(f"✓ Ridge regression accuracy: {accuracy:.3f}")
        
        # Test different configurations
        ridge_no_bias = RidgeRegression(reg_lambda=0.01, bias=False, scale=False)
        logits_no_bias = ridge_no_bias(episode)
        print(f"✓ RidgeRegression (no bias): {logits_no_bias.shape}")
        
        # Test Woodbury vs standard
        ridge_woodbury = RidgeRegression(reg_lambda=0.1, use_woodbury=True)
        logits_woodbury = ridge_woodbury(episode)
        
        ridge_standard = RidgeRegression(reg_lambda=0.1, use_woodbury=False)
        logits_standard = ridge_standard(episode)
        
        # Should give similar results
        diff = torch.abs(logits_woodbury - logits_standard).mean()
        print(f"✓ Woodbury vs Standard difference: {diff:.6f} (should be small)")
        
        print("🎉 All Ridge Regression class tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ RidgeRegression class test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Add the source directory to path
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
    
    success = test_ridge_regression()
    if success:
        print("\n✅ All Ridge Regression tests completed successfully!")
    else:
        print("\n❌ Some Ridge Regression tests failed!")
        sys.exit(1)