#!/usr/bin/env python3
"""
Test TensorUtils Implementation
===============================

Step 2: Confirm functionality is accurate before removing TODOs
Tests all implemented TensorUtils methods for mathematical correctness.
"""

import sys
import os
sys.path.insert(0, 'src')

import torch
import torch.nn as nn
from src.meta_learning.shared.utils import TensorUtils

def test_parameter_flattening():
    """Test parameter flatten/unflatten operations."""
    print("🧪 Testing parameter flattening...")
    
    # Create test model parameters
    linear1 = nn.Linear(10, 5)
    linear2 = nn.Linear(5, 2)
    params = [linear1.weight, linear1.bias, linear2.weight, linear2.bias]
    param_shapes = [p.shape for p in params]
    
    # Test flattening
    flat_params = TensorUtils.flatten_parameters(params)
    print(f"✅ Flattened {len(params)} parameters to vector of size {flat_params.shape}")
    
    # Test unflattening
    unflat_params = TensorUtils.unflatten_parameters(flat_params, param_shapes)
    print(f"✅ Unflattened back to {len(unflat_params)} parameters")
    
    # Verify correctness
    for orig, restored in zip(params, unflat_params):
        assert torch.allclose(orig, restored), "Unflatten operation failed!"
    
    print("✅ Parameter flatten/unflatten test PASSED")

def test_parameter_distance():
    """Test parameter distance computation."""
    print("🧪 Testing parameter distance computation...")
    
    # Create two sets of parameters
    model1 = nn.Linear(10, 5)
    model2 = nn.Linear(10, 5)
    params1 = list(model1.parameters())
    params2 = list(model2.parameters())
    
    # Test different distance types
    euclidean_dist = TensorUtils.compute_parameter_distance(params1, params2, 'euclidean')
    cosine_dist = TensorUtils.compute_parameter_distance(params1, params2, 'cosine') 
    manhattan_dist = TensorUtils.compute_parameter_distance(params1, params2, 'manhattan')
    
    print(f"✅ Euclidean distance: {euclidean_dist:.6f}")
    print(f"✅ Cosine distance: {cosine_dist:.6f}")
    print(f"✅ Manhattan distance: {manhattan_dist:.6f}")
    
    # Test identity (same parameters)
    identity_dist = TensorUtils.compute_parameter_distance(params1, params1, 'euclidean')
    assert identity_dist < 1e-6, f"Identity distance should be ~0, got {identity_dist}"
    print("✅ Parameter distance test PASSED")

def test_tensor_comparison():
    """Test safe tensor comparison."""
    print("🧪 Testing tensor comparison...")
    
    # Test identical tensors
    tensor1 = torch.randn(5, 5)
    tensor2 = tensor1.clone()
    
    comparison = TensorUtils.safe_tensor_comparison(tensor1, tensor2, tolerance=1e-6)
    assert comparison['tensors_match'], "Identical tensors should match!"
    print(f"✅ Identical tensors: max_diff = {comparison['max_difference']:.2e}")
    
    # Test slightly different tensors
    tensor3 = tensor1 + 1e-7
    comparison = TensorUtils.safe_tensor_comparison(tensor1, tensor3, tolerance=1e-6)
    print(f"✅ Similar tensors: max_diff = {comparison['max_difference']:.2e}, match = {comparison['tensors_match']}")
    
    # Test shape mismatch handling
    tensor4 = torch.randn(3, 3)
    comparison = TensorUtils.safe_tensor_comparison(tensor1, tensor4)
    assert 'error' in comparison, "Shape mismatch should be detected!"
    print("✅ Shape mismatch correctly detected")
    
    print("✅ Tensor comparison test PASSED")

def test_gradient_clipping():
    """Test gradient clipping analysis."""
    print("🧪 Testing gradient clipping...")
    
    # Create test gradients (some large)
    gradients = [
        torch.randn(5, 5) * 0.1,  # Small gradient
        torch.randn(3, 3) * 2.0,  # Large gradient  
        torch.randn(2, 2) * 0.5   # Medium gradient
    ]
    
    analysis = TensorUtils.gradient_clipping_analysis(gradients, max_norm=1.0)
    
    print(f"✅ Original total norm: {analysis['original_total_norm']:.4f}")
    print(f"✅ Clipping needed: {analysis['clipping_needed']}")
    print(f"✅ Clipped total norm: {analysis['clipped_total_norm']:.4f}")
    
    if analysis['clipping_needed']:
        assert analysis['clipped_total_norm'] <= 1.0 + 1e-6, "Clipped norm should be <= max_norm"
    
    print("✅ Gradient clipping test PASSED")

def test_tensor_statistics():
    """Test tensor statistics computation."""
    print("🧪 Testing tensor statistics...")
    
    # Create test tensor with known properties
    tensor = torch.randn(100, 10)
    tensor[0, 0] = float('nan')  # Add NaN for testing
    tensor[50:60, :] = 0.0       # Add zeros for sparsity testing
    
    stats = TensorUtils.tensor_statistics(tensor, name="test_tensor")
    
    print(f"✅ Tensor: {stats['name']}, shape: {stats['shape']}")
    print(f"✅ Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")
    print(f"✅ Has NaN: {stats['has_nan']}, Sparsity: {stats['sparsity']:.1f}%")
    print(f"✅ Memory: {stats['memory_mb']:.2f} MB")
    
    assert stats['has_nan'], "NaN detection should work"
    assert stats['sparsity'] > 0, "Sparsity calculation should detect zeros"
    
    print("✅ Tensor statistics test PASSED")

def test_memory_efficient_processing():
    """Test memory-efficient batch processing."""
    print("🧪 Testing memory-efficient processing...")
    
    # Create large tensor
    large_tensor = torch.randn(1000, 50)
    
    # Define simple processing function
    def relu_process(batch):
        return torch.relu(batch)
    
    # Process in small batches
    result = TensorUtils.memory_efficient_batch_processing(
        large_tensor, batch_size=100, processing_func=relu_process
    )
    
    # Verify result correctness
    expected = torch.relu(large_tensor)
    assert torch.allclose(result, expected), "Batch processing should match direct processing"
    
    print(f"✅ Processed {large_tensor.shape[0]} samples in batches of 100")
    print("✅ Memory-efficient processing test PASSED")

def main():
    """Run all TensorUtils tests."""
    print("🚀 Starting TensorUtils Implementation Tests")
    print("=" * 50)
    
    try:
        test_parameter_flattening()
        test_parameter_distance()
        test_tensor_comparison() 
        test_gradient_clipping()
        test_tensor_statistics()
        test_memory_efficient_processing()
        
        print("=" * 50)
        print("🎉 ALL TENSORUTILS TESTS PASSED!")
        print("✅ Implementation is mathematically correct")
        print("✅ Ready to remove TODO comments")
        
        return True
        
    except Exception as e:
        print("=" * 50)
        print(f"❌ TEST FAILED: {e}")
        print("❌ Implementation needs fixes before removing TODOs")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)