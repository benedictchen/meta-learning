#!/usr/bin/env python3
"""
Standalone Test for ANIL Algorithm Implementation
===============================================

This test verifies that the ANIL (Almost No Inner Loop) algorithm is fully 
implemented and works correctly for few-shot learning scenarios.
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add the src directory to Python path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from meta_learning.algorithms.anil import ANIL, ANILWrapper, anil_update
    print("✅ Successfully imported ANIL, ANILWrapper, and anil_update")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

def test_anil_initialization():
    """Test ANIL initialization and head/backbone separation."""
    print("\n🧪 Testing ANIL initialization...")
    
    try:
        # Create test model with clear head structure
        model = nn.Sequential(
            nn.Linear(10, 20),  # Backbone
            nn.ReLU(),          # Backbone
            nn.Linear(20, 15),  # Backbone  
            nn.ReLU(),          # Backbone
            nn.Linear(15, 5)    # This should be detected as head (final linear)
        )
        
        # Initialize ANIL
        anil = ANIL(model, head_lr=0.01, first_order=False)
        
        # Verify initialization
        assert hasattr(anil, 'model')
        assert hasattr(anil, 'head_params')
        assert hasattr(anil, 'backbone_params')
        assert hasattr(anil, 'head_lr')
        
        # Check parameter separation
        total_params = len(list(model.parameters()))
        head_count = len(anil.head_params)
        backbone_count = len(anil.backbone_params)
        
        print(f"✅ Total parameters: {total_params}")
        print(f"✅ Head parameters: {head_count}")
        print(f"✅ Backbone parameters: {backbone_count}")
        print(f"✅ Parameter split: {head_count + backbone_count == total_params}")
        
        assert head_count > 0, "Should have at least some head parameters"
        assert head_count + backbone_count == total_params, "All parameters should be categorized"
        
        print("✅ ANIL initialization successful")
        return True, anil
        
    except Exception as e:
        print(f"❌ ANIL initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_anil_adaptation():
    """Test ANIL head-only adaptation."""
    print("\n🧪 Testing ANIL head-only adaptation...")
    
    try:
        # Create model with identifiable head
        model = nn.Sequential(
            nn.Linear(4, 8),   # Backbone
            nn.ReLU(),         # Backbone
            nn.Linear(8, 2)    # Head (final layer)
        )
        
        anil = ANIL(model, head_lr=0.1, first_order=False)
        
        # Store original parameters
        original_head_params = [p.clone() for p in anil.head_params]
        original_backbone_params = [p.clone() for p in anil.backbone_params]
        
        # Create synthetic task data
        x_support = torch.randn(10, 4)
        y_support = torch.randint(0, 2, (10,))
        
        # Compute loss and adapt
        logits = anil(x_support)
        loss = F.cross_entropy(logits, y_support)
        
        print(f"Loss before adaptation: {loss.item():.4f}")
        
        # Perform head-only adaptation
        anil.adapt(loss)
        
        # Check adaptation results
        adapted_logits = anil(x_support)
        adapted_loss = F.cross_entropy(adapted_logits, y_support)
        print(f"Loss after adaptation: {adapted_loss.item():.4f}")
        
        # Verify ONLY head parameters changed
        head_changed = False
        for orig, current in zip(original_head_params, anil.head_params):
            if not torch.allclose(orig, current, atol=1e-6):
                head_changed = True
                break
        
        backbone_changed = False
        for orig, current in zip(original_backbone_params, anil.backbone_params):
            if not torch.allclose(orig, current, atol=1e-6):
                backbone_changed = True
                break
        
        assert head_changed, "Head parameters should change during adaptation"
        assert not backbone_changed, "Backbone parameters should remain frozen"
        
        print("✅ ANIL head-only adaptation works correctly")
        print(f"✅ Head changed: {head_changed}, Backbone frozen: {not backbone_changed}")
        return True
        
    except Exception as e:
        print(f"❌ ANIL adaptation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run key ANIL tests."""
    print("🚀 Starting Standalone ANIL Algorithm Tests")
    print("=" * 60)
    
    tests = [
        test_anil_initialization,
        test_anil_adaptation,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test == test_anil_initialization:
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
        print("🎉 ANIL TESTS PASSED!")
        print("✅ ANIL algorithm implemented and functional")
        print("✅ Head-only adaptation working correctly")
        print("=" * 60)
        return True
    else:
        print(f"❌ {total - passed} tests failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)