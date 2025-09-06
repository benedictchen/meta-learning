#!/usr/bin/env python3
"""
Full Meta-SGD Learning Scenario Test
===================================

This test demonstrates Meta-SGD in a proper few-shot learning scenario
where learning rates should be updated through meta-learning.
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add the src directory to Python path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from meta_learning.algorithms.meta_sgd import MetaSGD

def test_full_meta_sgd_scenario():
    """Test Meta-SGD in a full meta-learning scenario."""
    print("🧪 Testing full Meta-SGD meta-learning scenario...")
    
    try:
        # Create a simple model for few-shot classification
        model = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 2)
        )
        
        # Initialize Meta-SGD
        meta_sgd = MetaSGD(model, lr=0.1, first_order=False)
        
        # Store initial learning rates
        initial_lrs = [lr.detach().clone() for lr in meta_sgd.lrs]
        print(f"Initial learning rates: {[f'{lr.mean().item():.4f}' for lr in initial_lrs]}")
        
        # Meta-optimizer (optimizes both model params and learning rates)
        meta_optimizer = torch.optim.Adam(meta_sgd.parameters(), lr=0.001)
        
        # Simulate meta-training over multiple tasks
        meta_losses = []
        for task_idx in range(3):
            print(f"\n--- Task {task_idx + 1} ---")
            
            # Clone the model for this task
            task_model = meta_sgd.clone()
            
            # Generate synthetic support set for this task
            x_support = torch.randn(16, 4)  # 16 support examples
            y_support = torch.randint(0, 2, (16,))  # 2 classes
            
            # Inner loop: adapt to support set
            support_logits = task_model(x_support)
            support_loss = F.cross_entropy(support_logits, y_support)
            print(f"Support loss before adaptation: {support_loss.item():.4f}")
            
            # Adapt model to this task
            task_model.adapt(support_loss)
            
            # Check adaptation effect
            adapted_logits = task_model(x_support)
            adapted_loss = F.cross_entropy(adapted_logits, y_support)
            print(f"Support loss after adaptation: {adapted_loss.item():.4f}")
            
            # Generate query set for meta-loss
            x_query = torch.randn(8, 4)  # 8 query examples
            y_query = torch.randint(0, 2, (8,))  # Same classes
            
            # Compute meta-loss on query set
            query_logits = task_model(x_query)
            meta_loss = F.cross_entropy(query_logits, y_query)
            meta_losses.append(meta_loss.item())
            print(f"Query meta-loss: {meta_loss.item():.4f}")
            
            # Meta-update: optimize both model parameters and learning rates
            meta_optimizer.zero_grad()
            meta_loss.backward()
            meta_optimizer.step()
        
        # Check if learning rates have changed
        final_lrs = [lr.detach().clone() for lr in meta_sgd.lrs]
        print(f"\nFinal learning rates: {[f'{lr.mean().item():.4f}' for lr in final_lrs]}")
        
        # Compare learning rates
        lr_changes = []
        for initial, final in zip(initial_lrs, final_lrs):
            change = torch.abs(final - initial).max().item()
            lr_changes.append(change)
        
        max_lr_change = max(lr_changes)
        print(f"Maximum learning rate change: {max_lr_change:.6f}")
        
        # Meta-loss should generally decrease
        print(f"Meta-loss progression: {[f'{loss:.4f}' for loss in meta_losses]}")
        
        # Success criteria
        if max_lr_change > 1e-5:
            print("✅ Learning rates successfully updated through meta-learning!")
            print(f"✅ Meta-SGD is learning to learn with adaptive learning rates")
            return True
        else:
            print("❌ Learning rates did not change significantly")
            return False
            
    except Exception as e:
        print(f"❌ Full Meta-SGD scenario failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_manual_lr_gradient_flow():
    """Test that gradients flow through learning rates manually."""
    print("\n🧪 Testing manual gradient flow through learning rates...")
    
    try:
        # Simple model
        model = nn.Linear(2, 1)
        meta_sgd = MetaSGD(model, lr=0.5, first_order=False)
        
        # Create data where we know the optimal learning rate
        x = torch.tensor([[1.0, 0.0]], requires_grad=False)
        y = torch.tensor([1.0], requires_grad=False)
        
        # Store initial LR
        initial_lr = meta_sgd.lrs[0].detach().clone()
        print(f"Initial weight LR: {initial_lr.mean().item():.4f}")
        
        # Manual meta-learning step
        meta_optimizer = torch.optim.SGD(meta_sgd.lrs, lr=0.1)  # Only optimize LRs
        
        for step in range(3):
            # Clone for adaptation
            task_model = meta_sgd.clone()
            
            # Inner loop: adapt
            pred = task_model(x)
            loss = F.mse_loss(pred, y)
            task_model.adapt(loss)
            
            # Meta-loss after adaptation
            adapted_pred = task_model(x)
            meta_loss = F.mse_loss(adapted_pred, y)
            
            # Update learning rates only
            meta_optimizer.zero_grad()
            meta_loss.backward()
            meta_optimizer.step()
            
            current_lr = meta_sgd.lrs[0].detach().clone()
            print(f"Step {step+1}: LR = {current_lr.mean().item():.4f}, Meta-loss = {meta_loss.item():.4f}")
        
        # Check if learning rates changed
        final_lr = meta_sgd.lrs[0].detach().clone()
        lr_change = torch.abs(final_lr - initial_lr).max().item()
        
        if lr_change > 1e-4:
            print(f"✅ Manual gradient flow works! LR change: {lr_change:.6f}")
            return True
        else:
            print(f"❌ Manual gradient flow failed. LR change: {lr_change:.6f}")
            return False
            
    except Exception as e:
        print(f"❌ Manual gradient flow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run comprehensive Meta-SGD tests."""
    print("🚀 Comprehensive Meta-SGD Learning Rate Tests")
    print("=" * 60)
    
    tests = [
        test_manual_lr_gradient_flow,
        test_full_meta_sgd_scenario,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            result = test()
            if result:
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
    
    print("=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 META-SGD LEARNING RATES WORKING!")
        return True
    else:
        print(f"❌ {total - passed} tests failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)