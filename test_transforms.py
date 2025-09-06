#!/usr/bin/env python3
"""
Test Gradient Transforms Implementation
======================================

Quick test to verify gradient transform functionality.
"""

import sys
import os
import torch
import torch.nn as nn

# Add the src directory to Python path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from meta_learning.optimization.transforms import (
        ScaleTransform, BiasTransform, MomentumTransform, 
        AdaptiveTransform, CompositeTransform, NoiseTransform,
        TemperatureTransform
    )
    print("✅ Successfully imported all transform classes")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

def test_transform_functionality():
    """Test core transform functionality."""
    print("\n🧪 Testing Transform Functionality...")
    
    try:
        # Create sample parameter shapes and test data
        param_shapes = {'weight': torch.Size([10]), 'bias': torch.Size([10])}
        test_grad = torch.randn(10)
        test_param = torch.randn(10)
        
        # Test ScaleTransform
        scale_transform = ScaleTransform(parameter_shapes=param_shapes, init_scale=1.0)
        scaled_grad = scale_transform(test_grad, test_param)
        assert scaled_grad.shape == test_grad.shape, "Scale transform shape mismatch"
        print("✅ ScaleTransform working")
        
        # Test BiasTransform
        bias_transform = BiasTransform(parameter_shapes=param_shapes)
        biased_grad = bias_transform(test_grad, test_param)
        assert biased_grad.shape == test_grad.shape, "Bias transform shape mismatch"
        print("✅ BiasTransform working")
        
        # Test MomentumTransform
        momentum_transform = MomentumTransform(parameter_shapes=param_shapes, init_momentum=0.9)
        momentum_grad = momentum_transform(test_grad, test_param)
        assert momentum_grad.shape == test_grad.shape, "Momentum transform shape mismatch"
        print("✅ MomentumTransform working")
        
        # Test AdaptiveTransform
        adaptive_transform = AdaptiveTransform(parameter_shapes=param_shapes)
        adaptive_grad = adaptive_transform(test_grad, test_param)
        assert adaptive_grad.shape == test_grad.shape, "Adaptive transform shape mismatch"
        print("✅ AdaptiveTransform working")
        
        # Test CompositeTransform
        transforms = [
            ScaleTransform(parameter_shapes=param_shapes, init_scale=0.5),
            BiasTransform(parameter_shapes=param_shapes)
        ]
        composite_transform = CompositeTransform(transforms)
        composite_grad = composite_transform(test_grad, test_param)
        assert composite_grad.shape == test_grad.shape, "Composite transform shape mismatch"
        print("✅ CompositeTransform working")
        
        # Test NoiseTransform
        noise_transform = NoiseTransform(noise_scale=0.01)
        noisy_grad = noise_transform(test_grad, test_param)
        assert noisy_grad.shape == test_grad.shape, "Noise transform shape mismatch"
        print("✅ NoiseTransform working")
        
        # Test TemperatureTransform
        temp_transform = TemperatureTransform(init_temperature=1.0)
        temp_grad = temp_transform(test_grad, test_param)
        assert temp_grad.shape == test_grad.shape, "Temperature transform shape mismatch"
        print("✅ TemperatureTransform working")
        
        # Test gradient flow (requires_grad preservation)  
        test_grad_req = torch.randn(10, requires_grad=True)
        test_param_req = torch.randn(10, requires_grad=True)
        scaled_grad_req = scale_transform(test_grad_req, test_param_req)
        assert scaled_grad_req.requires_grad, "Gradient flow broken"
        print("✅ Gradient flow preserved")
        
        return True
        
    except Exception as e:
        print(f"❌ Transform test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_transform_functionality()
    if success:
        print("🎉 All gradient transforms are working!")
    else:
        print("❌ Some transforms need fixes")
    sys.exit(0 if success else 1)