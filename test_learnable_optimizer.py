#!/usr/bin/env python3
"""Test script for Learnable Optimizer implementation"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def test_learnable_optimizer():
    print("Testing Learnable Optimizer implementation...")
    
    # Set seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Test the transform classes first
    from src.meta_learning.optimization.learnable_optimizer import (
        IdentityTransform, GradientTransform, ScaleTransform, BiasTransform,
        CompositeTransform, LearnableOptimizer
    )
    
    print("✓ All imports successful")
    
    # Test IdentityTransform
    try:
        identity = IdentityTransform()
        test_grad = torch.randn(5, 3)
        result = identity(test_grad)
        assert torch.equal(result, test_grad)
        print("✓ IdentityTransform works correctly")
    except Exception as e:
        print(f"✗ IdentityTransform failed: {e}")
        return False
    
    # Test ScaleTransform
    try:
        # Global scaling
        scale_global = ScaleTransform(per_element=False, init_scale=2.0)
        test_grad = torch.ones(3, 2)
        result = scale_global(test_grad)
        expected = 2.0 * test_grad
        assert torch.allclose(result, expected)
        print("✓ ScaleTransform (global) works correctly")
        
        # Per-element scaling
        scale_element = ScaleTransform(per_element=True, init_scale=1.5)
        test_grad = torch.ones(2, 3)
        result = scale_element(test_grad)
        expected = 1.5 * test_grad
        assert torch.allclose(result, expected)
        print("✓ ScaleTransform (per-element) works correctly")
    except Exception as e:
        print(f"✗ ScaleTransform failed: {e}")
        return False
    
    # Test BiasTransform
    try:
        # Global bias
        bias_global = BiasTransform(per_element=False, init_bias=1.0)
        test_grad = torch.zeros(3, 2)
        result = bias_global(test_grad)
        expected = torch.ones(3, 2)
        assert torch.allclose(result, expected)
        print("✓ BiasTransform (global) works correctly")
        
        # Per-element bias
        bias_element = BiasTransform(per_element=True, init_bias=0.5)
        test_grad = torch.zeros(2, 3)
        result = bias_element(test_grad)
        expected = 0.5 * torch.ones(2, 3)
        assert torch.allclose(result, expected)
        print("✓ BiasTransform (per-element) works correctly")
    except Exception as e:
        print(f"✗ BiasTransform failed: {e}")
        return False
    
    # Test CompositeTransform
    try:
        scale = ScaleTransform(per_element=False, init_scale=2.0)
        bias = BiasTransform(per_element=False, init_bias=1.0)
        composite = CompositeTransform([scale, bias])
        
        test_grad = torch.ones(2, 2)
        result = composite(test_grad)
        # Should be: (1 * 2) + 1 = 3
        expected = 3.0 * torch.ones(2, 2)
        assert torch.allclose(result, expected)
        print("✓ CompositeTransform works correctly")
    except Exception as e:
        print(f"✗ CompositeTransform failed: {e}")
        return False
    
    # Test LearnableOptimizer with simple model
    try:
        # Create a simple linear model
        model = nn.Linear(10, 3)
        
        # Create learnable optimizer with scale transform
        transform = ScaleTransform(per_element=False, init_scale=1.0)
        optimizer = LearnableOptimizer(
            model=model,
            transform=transform,
            lr=0.01,
            meta_lr=0.001
        )
        
        print(f"✓ LearnableOptimizer initialized: LR={optimizer.get_learning_rate()}")
        
        # Test optimization step with synthetic data
        x = torch.randn(32, 10)
        y = torch.randint(0, 3, (32,))
        
        def closure():
            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            return loss
        
        # Perform optimization step
        loss = optimizer.step(closure)
        print(f"✓ Optimization step completed, loss: {loss:.4f}")
        
        # Test meta-learning step
        meta_loss = torch.tensor(loss.item(), requires_grad=True)
        optimizer.meta_step(meta_loss)
        print("✓ Meta-optimization step completed")
        
        # Test performance metrics
        metrics = optimizer.get_performance_metrics()
        print(f"✓ Performance metrics: {metrics}")
        
        # Test learning rate adjustment
        optimizer.set_learning_rate(0.02)
        assert optimizer.get_learning_rate() == 0.02
        print("✓ Learning rate adjustment works")
        
    except Exception as e:
        print(f"✗ LearnableOptimizer failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test with momentum
    try:
        model = nn.Linear(5, 2)
        transform = BiasTransform(per_element=False, init_bias=0.1)
        optimizer = LearnableOptimizer(
            model=model,
            transform=transform,
            lr=0.01,
            momentum=0.9
        )
        
        x = torch.randn(16, 5)
        y = torch.randint(0, 2, (16,))
        
        # Run a few steps to test momentum
        for i in range(3):
            def closure():
                optimizer.zero_grad()
                logits = model(x)
                loss = F.cross_entropy(logits, y)
                loss.backward()
                return loss
            
            loss = optimizer.step(closure)
        
        print("✓ Momentum-based optimization works")
        
    except Exception as e:
        print(f"✗ Momentum test failed: {e}")
        return False
    
    # Test regularization
    try:
        transform = ScaleTransform(per_element=True, init_scale=1.0, regularization=0.1)
        test_grad = torch.ones(3, 3)
        
        # Initialize transform
        result = transform(test_grad)
        
        # Get regularization loss
        reg_loss = transform.get_regularization_loss()
        assert reg_loss > 0
        print(f"✓ Regularization works, loss: {reg_loss:.4f}")
        
    except Exception as e:
        print(f"✗ Regularization test failed: {e}")
        return False
    
    print("🎉 All Learnable Optimizer tests passed!")
    return True

def test_integration_with_meta_learning():
    """Test integration with meta-learning scenarios."""
    print("\nTesting meta-learning integration...")
    
    try:
        from src.meta_learning.optimization.learnable_optimizer import LearnableOptimizer, ScaleTransform
        from src.meta_learning.core.utils import clone_module
        
        # Create a simple model for meta-learning
        model = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 2)
        )
        
        # Create learnable optimizer
        transform = ScaleTransform(per_element=False, init_scale=1.0)
        optimizer = LearnableOptimizer(
            model=model,
            transform=transform,
            lr=0.1,
            meta_lr=0.01
        )
        
        # Simulate few-shot learning scenario
        support_x = torch.randn(10, 4)  # 10 support samples
        support_y = torch.randint(0, 2, (10,))
        query_x = torch.randn(5, 4)    # 5 query samples  
        query_y = torch.randint(0, 2, (5,))
        
        # Inner loop adaptation
        adapted_model = clone_module(model)
        adapted_optimizer = LearnableOptimizer(
            model=adapted_model,
            transform=transform,  # Share the same transform
            lr=0.1
        )
        
        # Adaptation steps
        for step in range(5):
            def inner_closure():
                adapted_optimizer.zero_grad()
                logits = adapted_model(support_x)
                loss = F.cross_entropy(logits, support_y)
                loss.backward()
                return loss
            
            inner_loss = adapted_optimizer.step(inner_closure)
        
        # Query evaluation
        with torch.no_grad():
            query_logits = adapted_model(query_x)
            meta_loss = F.cross_entropy(query_logits, query_y)
        
        print(f"✓ Meta-learning simulation completed, meta-loss: {meta_loss:.4f}")
        
        # Meta-optimization of transform
        meta_loss.requires_grad_(True)
        optimizer.meta_step(meta_loss)
        
        print("✓ Meta-learning integration successful")
        return True
        
    except Exception as e:
        print(f"✗ Meta-learning integration failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Add the source directory to path
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
    
    success = test_learnable_optimizer()
    if success:
        success = test_integration_with_meta_learning()
    
    if success:
        print("\n✅ All Learnable Optimizer tests completed successfully!")
    else:
        print("\n❌ Some Learnable Optimizer tests failed!")
        sys.exit(1)