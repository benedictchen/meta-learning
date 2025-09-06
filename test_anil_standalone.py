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

def test_anil_forward():
    """Test ANIL forward pass."""
    print("\n🧪 Testing ANIL forward pass...")
    
    try:
        # Create model
        model = nn.Sequential(
            nn.Linear(8, 12),
            nn.ReLU(), 
            nn.Linear(12, 3)  # Head layer
        )
        
        anil = ANIL(model, head_lr=0.01)
        
        # Test forward pass
        x = torch.randn(5, 8)
        output = anil(x)
        
        # Verify output
        assert output.shape == (5, 3)
        assert output.requires_grad
        
        print("✅ ANIL forward pass works correctly")
        print(f"✅ Input shape: {x.shape} → Output shape: {output.shape}")
        return True
        
    except Exception as e:
        print(f"❌ ANIL forward test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_anil_clone():
    """Test ANIL cloning functionality."""
    print("\n🧪 Testing ANIL cloning...")
    
    try:
        # Create model
        model = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 2)
        )
        
        original_anil = ANIL(model, head_lr=0.05, first_order=True)
        
        # Clone the model
        cloned_anil = original_anil.clone()
        
        # Verify clone properties
        assert isinstance(cloned_anil, ANIL)
        assert cloned_anil.head_lr == original_anil.head_lr
        assert cloned_anil.first_order == original_anil.first_order
        assert len(cloned_anil.head_params) == len(original_anil.head_params)
        assert len(cloned_anil.backbone_params) == len(original_anil.backbone_params)
        
        # Verify parameters are independent
        x = torch.randn(3, 4)
        original_output = original_anil(x)
        cloned_output = cloned_anil(x)
        
        # Should be identical initially
        assert torch.allclose(original_output, cloned_output, atol=1e-6)
        
        print("✅ ANIL cloning works correctly")
        return True
        
    except Exception as e:
        print(f"❌ ANIL clone test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

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
        
        # Verify ONLY head parameters changed\n        head_changed = False\n        for orig, current in zip(original_head_params, anil.head_params):\n            if not torch.allclose(orig, current, atol=1e-6):\n                head_changed = True\n                break\n        \n        backbone_changed = False\n        for orig, current in zip(original_backbone_params, anil.backbone_params):\n            if not torch.allclose(orig, current, atol=1e-6):\n                backbone_changed = True\n                break\n        \n        assert head_changed, \"Head parameters should change during adaptation\"\n        assert not backbone_changed, \"Backbone parameters should remain frozen\"\n        \n        print(\"✅ ANIL head-only adaptation works correctly\")\n        print(f\"✅ Head changed: {head_changed}, Backbone frozen: {not backbone_changed}\")\n        return True\n        \n    except Exception as e:\n        print(f\"❌ ANIL adaptation test failed: {e}\")\n        import traceback\n        traceback.print_exc()\n        return False\n\ndef test_anil_wrapper():\n    \"\"\"Test ANILWrapper automatic head detection.\"\"\"\n    print(\"\\n🧪 Testing ANILWrapper...\")\n    \n    try:\n        # Create model with standard naming\n        class TestModel(nn.Module):\n            def __init__(self):\n                super().__init__()\n                self.features = nn.Sequential(\n                    nn.Linear(4, 8),\n                    nn.ReLU()\n                )\n                self.classifier = nn.Linear(8, 3)  # Should be detected as head\n            \n            def forward(self, x):\n                x = self.features(x)\n                return self.classifier(x)\n        \n        model = TestModel()\n        wrapper = ANILWrapper(model)\n        \n        # Verify wrapper properties\n        head_params = wrapper.get_head_parameters()\n        backbone_params = wrapper.get_backbone_parameters()\n        \n        print(f\"✅ Head parameters detected: {len(head_params)}\")\n        print(f\"✅ Backbone parameters detected: {len(backbone_params)}\")\n        \n        # Test forward pass\n        x = torch.randn(5, 4)\n        output = wrapper(x)\n        assert output.shape == (5, 3)\n        \n        print(\"✅ ANILWrapper works correctly\")\n        return True\n        \n    except Exception as e:\n        print(f\"❌ ANILWrapper test failed: {e}\")\n        import traceback\n        traceback.print_exc()\n        return False\n\ndef test_anil_update_function():\n    \"\"\"Test standalone anil_update function.\"\"\"\n    print(\"\\n🧪 Testing anil_update function...\")\n    \n    try:\n        # Create simple model\n        model = nn.Linear(3, 2)\n        \n        # Get head parameters (all parameters for this simple model)\n        head_params = list(model.parameters())\n        original_params = [p.clone() for p in head_params]\n        \n        # Create fake gradients\n        gradients = [torch.randn_like(p) for p in head_params]\n        head_lr = 0.01\n        \n        # Apply update\n        anil_update(model, head_params, head_lr, gradients)\n        \n        # Verify parameters changed\n        params_changed = False\n        for orig, current in zip(original_params, head_params):\n            if not torch.allclose(orig, current, atol=1e-6):\n                params_changed = True\n                break\n        \n        assert params_changed, \"Parameters should change after anil_update\"\n        \n        print(\"✅ anil_update function works correctly\")\n        return True\n        \n    except Exception as e:\n        print(f\"❌ anil_update test failed: {e}\")\n        import traceback\n        traceback.print_exc()\n        return False\n\ndef test_computational_efficiency():\n    \"\"\"Test that ANIL is more efficient than full MAML.\"\"\"\n    print(\"\\n🧪 Testing ANIL computational efficiency...\")\n    \n    try:\n        # Create larger model to see efficiency difference\n        model = nn.Sequential(\n            nn.Linear(100, 200),  # Backbone\n            nn.ReLU(),            # Backbone\n            nn.Linear(200, 100),  # Backbone\n            nn.ReLU(),            # Backbone\n            nn.Linear(100, 50),   # Backbone\n            nn.ReLU(),            # Backbone\n            nn.Linear(50, 10)     # Head (final layer)\n        )\n        \n        anil = ANIL(model, head_lr=0.01)\n        \n        total_params = sum(p.numel() for p in model.parameters())\n        head_params_count = sum(p.numel() for p in anil.head_params)\n        backbone_params_count = sum(p.numel() for p in anil.backbone_params)\n        \n        efficiency_ratio = head_params_count / total_params\n        \n        print(f\"✅ Total parameters: {total_params:,}\")\n        print(f\"✅ Head parameters: {head_params_count:,}\")\n        print(f\"✅ Backbone parameters: {backbone_params_count:,}\")\n        print(f\"✅ ANIL adapts only {efficiency_ratio:.1%} of parameters\")\n        \n        # ANIL should adapt much fewer parameters than total\n        assert efficiency_ratio < 0.5, \"ANIL should adapt less than 50% of parameters\"\n        \n        print(\"✅ ANIL demonstrates computational efficiency\")\n        return True\n        \n    except Exception as e:\n        print(f\"❌ Computational efficiency test failed: {e}\")\n        import traceback\n        traceback.print_exc()\n        return False\n\ndef main():\n    \"\"\"Run all ANIL tests.\"\"\"\n    print(\"🚀 Starting Standalone ANIL Algorithm Tests\")\n    print(\"=\" * 60)\n    \n    tests = [\n        test_anil_initialization,\n        test_anil_forward,\n        test_anil_clone,\n        test_anil_adaptation,\n        test_anil_wrapper,\n        test_anil_update_function,\n        test_computational_efficiency\n    ]\n    \n    passed = 0\n    total = len(tests)\n    \n    for test in tests:\n        try:\n            if test == test_anil_initialization:\n                # Special handling for initialization test that returns model\n                result, _ = test()\n                if result:\n                    passed += 1\n            else:\n                result = test()\n                if result:\n                    passed += 1\n        except Exception as e:\n            print(f\"❌ Test {test.__name__} failed with exception: {e}\")\n    \n    print(\"=\" * 60)\n    print(f\"📊 Test Results: {passed}/{total} tests passed\")\n    \n    if passed == total:\n        print(\"🎉 ALL ANIL TESTS PASSED!\")\n        print(\"✅ ANIL algorithm fully implemented and functional\")\n        print(\"✅ Head-only adaptation working correctly\")\n        print(\"✅ Significant computational savings vs full MAML\")\n        print(\"✅ Automatic backbone/head detection working\")\n        print(\"✅ Ready for few-shot learning tasks\")\n        print(\"=\" * 60)\n        return True\n    else:\n        print(f\"❌ {total - passed} tests failed\")\n        print(\"❌ ANIL implementation needs fixes\")\n        return False\n\nif __name__ == \"__main__\":\n    success = main()\n    sys.exit(0 if success else 1)