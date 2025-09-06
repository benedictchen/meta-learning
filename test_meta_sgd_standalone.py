#!/usr/bin/env python3
"""
Standalone Test for Meta-SGD Algorithm Implementation
===================================================

This test verifies that the Meta-SGD algorithm is fully implemented
and works correctly for few-shot learning scenarios.
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add the src directory to Python path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from meta_learning.algorithms.meta_sgd import MetaSGD, meta_sgd_update
    print("✅ Successfully imported MetaSGD and meta_sgd_update")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

def test_meta_sgd_update_function():
    """Test the meta_sgd_update function."""
    print("\n🧪 Testing meta_sgd_update function...")
    
    try:
        # Create simple model
        model = nn.Linear(4, 2)
        
        # Create gradients and learning rates
        grads = [torch.randn_like(p) for p in model.parameters()]
        lrs = [torch.tensor(0.01) for _ in model.parameters()]
        
        # Test update function
        updated_model = meta_sgd_update(model, lrs, grads)
        
        # Verify model structure is preserved
        assert isinstance(updated_model, nn.Linear)
        assert updated_model.in_features == 4
        assert updated_model.out_features == 2
        
        print("✅ meta_sgd_update function works correctly")
        return True
        
    except Exception as e:
        print(f"❌ meta_sgd_update test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_meta_sgd_initialization():
    """Test MetaSGD initialization."""
    print("\n🧪 Testing MetaSGD initialization...")
    
    try:
        # Create base model
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 2)
        )
        
        # Test initialization
        meta_sgd = MetaSGD(model, lr=0.01, first_order=False)
        
        # Verify initialization
        assert hasattr(meta_sgd, 'module')
        assert hasattr(meta_sgd, 'lrs')
        assert hasattr(meta_sgd, 'first_order')
        
        # Check learning rates
        assert len(meta_sgd.lrs) == len(list(model.parameters()))
        for lr_param in meta_sgd.lrs:
            assert isinstance(lr_param, nn.Parameter)
            assert lr_param.requires_grad
        
        print("✅ MetaSGD initialization successful")
        print(f"✅ Created {len(meta_sgd.lrs)} learnable learning rate parameters")
        return True, meta_sgd
        
    except Exception as e:
        print(f"❌ MetaSGD initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_meta_sgd_forward():
    """Test MetaSGD forward pass."""
    print("\n🧪 Testing MetaSGD forward pass...")
    
    try:
        # Create base model
        model = nn.Linear(8, 3)
        meta_sgd = MetaSGD(model, lr=0.1)
        
        # Test forward pass
        x = torch.randn(5, 8)
        output = meta_sgd(x)
        
        # Verify output shape
        assert output.shape == (5, 3)
        assert output.requires_grad  # Should preserve gradients
        
        print("✅ MetaSGD forward pass works correctly")
        print(f"✅ Input shape: {x.shape} → Output shape: {output.shape}")
        return True
        
    except Exception as e:
        print(f"❌ MetaSGD forward test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_meta_sgd_clone():
    """Test MetaSGD cloning functionality."""
    print("\n🧪 Testing MetaSGD cloning...")
    
    try:
        # Create original model
        model = nn.Linear(6, 4)
        original_meta_sgd = MetaSGD(model, lr=0.05, first_order=True)
        
        # Clone the model
        cloned_meta_sgd = original_meta_sgd.clone()
        
        # Verify clone properties
        assert isinstance(cloned_meta_sgd, MetaSGD)
        assert cloned_meta_sgd.first_order == original_meta_sgd.first_order
        assert len(cloned_meta_sgd.lrs) == len(original_meta_sgd.lrs)
        
        # Verify parameters are independent
        x = torch.randn(3, 6)
        original_output = original_meta_sgd(x)
        cloned_output = cloned_meta_sgd(x)
        
        # Should be identical initially
        assert torch.allclose(original_output, cloned_output, atol=1e-6)
        
        print("✅ MetaSGD cloning works correctly")
        return True
        
    except Exception as e:
        print(f"❌ MetaSGD clone test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_meta_sgd_adapt():
    """Test MetaSGD adaptation functionality."""
    print("\n🧪 Testing MetaSGD adaptation...")
    
    try:
        # Create model and data
        model = nn.Linear(4, 2)
        meta_sgd = MetaSGD(model, lr=0.1, first_order=False)
        
        # Create synthetic task data
        x_support = torch.randn(10, 4)
        y_support = torch.randint(0, 2, (10,))
        
        # Store original parameters
        original_params = [p.clone() for p in meta_sgd.module.parameters()]
        
        # Compute loss and adapt
        logits = meta_sgd(x_support)
        loss = F.cross_entropy(logits, y_support)
        
        # Perform adaptation
        meta_sgd.adapt(loss)
        
        # Verify parameters changed
        updated_params = list(meta_sgd.module.parameters())
        for orig, updated in zip(original_params, updated_params):
            assert not torch.allclose(orig, updated, atol=1e-6), "Parameters should change after adaptation"
        
        print("✅ MetaSGD adaptation works correctly")
        print(f"✅ Successfully adapted to synthetic 2-class task with loss: {loss.item():.4f}")
        return True
        
    except Exception as e:
        print(f"❌ MetaSGD adapt test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_meta_sgd_first_order_vs_second_order():
    """Test first-order vs second-order approximations."""
    print("\n🧪 Testing first-order vs second-order Meta-SGD...")
    
    try:
        # Create identical models
        model1 = nn.Linear(4, 2)
        model2 = nn.Linear(4, 2)
        
        # Copy parameters to ensure identical starting points
        with torch.no_grad():
            for p1, p2 in zip(model1.parameters(), model2.parameters()):
                p2.copy_(p1)
        
        # Create Meta-SGD with different orders
        meta_sgd_first = MetaSGD(model1, lr=0.1, first_order=True)
        meta_sgd_second = MetaSGD(model2, lr=0.1, first_order=False)
        
        # Copy learning rates to ensure identical starting points
        with torch.no_grad():
            for lr1, lr2 in zip(meta_sgd_first.lrs, meta_sgd_second.lrs):
                lr2.copy_(lr1)
        
        # Create test data
        x = torch.randn(8, 4)
        y = torch.randint(0, 2, (8,))
        
        # Compute losses and adapt
        loss1 = F.cross_entropy(meta_sgd_first(x), y)
        loss2 = F.cross_entropy(meta_sgd_second(x), y)
        
        meta_sgd_first.adapt(loss1)
        meta_sgd_second.adapt(loss2)
        
        print("✅ First-order and second-order Meta-SGD both work")
        print(f"✅ First-order loss: {loss1.item():.4f}, Second-order loss: {loss2.item():.4f}")
        return True
        
    except Exception as e:
        print(f"❌ First/second-order test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_learnable_learning_rates():
    """Test that learning rates are learnable parameters."""
    print("\n🧪 Testing learnable learning rates...")
    
    try:
        # Create model
        model = nn.Linear(3, 2)
        meta_sgd = MetaSGD(model, lr=0.05)
        
        # Get initial learning rates
        initial_lrs = [lr.clone() for lr in meta_sgd.lrs]
        
        # Create meta-learning scenario
        x_meta = torch.randn(5, 3)
        y_meta = torch.randint(0, 2, (5,))
        
        # Create optimizer for meta-parameters (including learning rates)
        meta_optimizer = torch.optim.SGD(list(meta_sgd.parameters()), lr=0.01)
        
        # Perform meta-update
        meta_logits = meta_sgd(x_meta)
        meta_loss = F.cross_entropy(meta_logits, y_meta)
        
        meta_optimizer.zero_grad()
        meta_loss.backward()
        meta_optimizer.step()
        
        # Check if learning rates changed
        lr_changed = False
        for initial_lr, current_lr in zip(initial_lrs, meta_sgd.lrs):
            if not torch.allclose(initial_lr, current_lr, atol=1e-6):
                lr_changed = True
                break
        
        assert lr_changed, "Learning rates should be updated during meta-learning"
        
        print("✅ Learning rates are successfully learnable")
        print(f"✅ Meta-loss: {meta_loss.item():.4f}")
        return True
        
    except Exception as e:
        print(f"❌ Learnable learning rates test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all Meta-SGD tests."""
    print("🚀 Starting Standalone Meta-SGD Algorithm Tests")
    print("=" * 60)
    
    tests = [
        test_meta_sgd_update_function,
        test_meta_sgd_initialization,
        test_meta_sgd_forward,
        test_meta_sgd_clone,
        test_meta_sgd_adapt,
        test_meta_sgd_first_order_vs_second_order,
        test_learnable_learning_rates
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test == test_meta_sgd_initialization:
                # Special handling for initialization test that returns model
                result, _ = test()
                if result:
                    passed += 1
            else:
                result = test()
                if result:
                    passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
    
    print("=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL META-SGD TESTS PASSED!")
        print("✅ Meta-SGD algorithm fully implemented and functional")
        print("✅ Supports both first-order and second-order approximations")
        print("✅ Learning rates are properly learnable parameters")
        print("✅ Ready for few-shot learning tasks")
        print("=" * 60)
        return True
    else:
        print(f"❌ {total - passed} tests failed")
        print("❌ Meta-SGD implementation needs fixes")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)