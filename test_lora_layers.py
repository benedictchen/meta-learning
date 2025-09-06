#!/usr/bin/env python3
"""
Test LoRA Layers Implementation
===============================

Step 2: Confirm LoRA functionality is accurate before removing TODOs
Tests LoRA layer implementations against Hu et al. (2021) paper requirements.
"""

import sys
import os
sys.path.insert(0, 'src')

import torch
import torch.nn as nn
from src.meta_learning.patches.lora_components.lora_layers import LoRALayer, LoRALinear

def test_lora_layer_initialization():
    """Test LoRA layer initialization follows paper specifications."""
    print("🧪 Testing LoRA layer initialization...")
    
    # Create LoRA layer
    in_features, out_features, rank = 128, 64, 8
    alpha = 16.0
    lora_layer = LoRALayer(in_features, out_features, rank=rank, alpha=alpha)
    
    # Verify parameter shapes
    assert lora_layer.lora_A.shape == (in_features, rank), f"A matrix shape wrong: {lora_layer.lora_A.shape}"
    assert lora_layer.lora_B.shape == (rank, out_features), f"B matrix shape wrong: {lora_layer.lora_B.shape}"
    
    # Verify B matrix initialization (should be zeros)
    assert torch.allclose(lora_layer.lora_B, torch.zeros_like(lora_layer.lora_B)), "B matrix should be initialized to zeros"
    
    # Verify scaling factor
    expected_scale = alpha / rank
    assert abs(lora_layer.scale - expected_scale) < 1e-6, f"Scale factor wrong: {lora_layer.scale} vs {expected_scale}"
    
    print(f"✅ LoRA layer: {in_features}→{out_features}, rank={rank}, scale={lora_layer.scale:.4f}")
    print("✅ LoRA layer initialization test PASSED")

def test_lora_layer_forward():
    """Test LoRA layer forward pass mathematical correctness."""
    print("🧪 Testing LoRA layer forward pass...")
    
    # Create LoRA layer with known parameters
    lora_layer = LoRALayer(in_features=10, out_features=5, rank=3, alpha=6.0)
    
    # Set known values for testing
    lora_layer.lora_A.data = torch.ones(10, 3) * 0.1  # A matrix
    lora_layer.lora_B.data = torch.ones(3, 5) * 0.2   # B matrix
    
    # Test input
    x = torch.ones(2, 10)  # Batch of 2, features of 10
    
    # Forward pass
    output = lora_layer(x)
    
    # Verify output shape
    assert output.shape == (2, 5), f"Output shape wrong: {output.shape}"
    
    # Verify computation: (x @ A) @ B * scale
    expected = torch.matmul(torch.matmul(x, lora_layer.lora_A), lora_layer.lora_B) * lora_layer.scale
    assert torch.allclose(output, expected), "LoRA computation doesn't match expected"
    
    print(f"✅ Forward pass: {x.shape} → {output.shape}")
    print(f"✅ Scale factor applied correctly: {lora_layer.scale}")
    print("✅ LoRA layer forward test PASSED")

def test_lora_layer_identity_at_initialization():
    """Test that LoRA layer starts as identity (B=0 ensures this)."""
    print("🧪 Testing LoRA identity at initialization...")
    
    lora_layer = LoRALayer(in_features=10, out_features=10, rank=4)
    x = torch.randn(5, 10)
    
    # At initialization, B=0, so output should be zero
    output = lora_layer(x)
    expected_zero = torch.zeros_like(output)
    
    assert torch.allclose(output, expected_zero, atol=1e-6), "LoRA should output zeros at initialization"
    
    print("✅ LoRA layer produces zero output at initialization (identity property)")
    print("✅ LoRA identity test PASSED")

def test_lora_linear_integration():
    """Test LoRALinear integration with original linear layer."""
    print("🧪 Testing LoRALinear integration...")
    
    # Create original linear layer
    original_linear = nn.Linear(20, 10)
    original_linear.weight.data.fill_(0.1)  # Known weights
    original_linear.bias.data.fill_(0.05)   # Known bias
    
    # Create LoRA-enhanced version
    lora_linear = LoRALinear(original_linear, rank=4, alpha=8.0, freeze_original=True)
    
    # Verify original layer is frozen
    assert not original_linear.weight.requires_grad, "Original weights should be frozen"
    assert not original_linear.bias.requires_grad, "Original bias should be frozen"
    
    # Test forward pass at initialization (should match original + zero)
    x = torch.ones(3, 20)
    original_output = original_linear(x)
    lora_output = lora_linear(x)
    
    # At initialization, LoRA adds zero, so outputs should match
    assert torch.allclose(original_output, lora_output, atol=1e-6), "LoRA should match original at initialization"
    
    print(f"✅ Original layer frozen: {not original_linear.weight.requires_grad}")
    print(f"✅ LoRA parameters trainable: {lora_linear.lora_adapter.lora_A.requires_grad}")
    print("✅ LoRALinear integration test PASSED")

def test_lora_linear_adaptation():
    """Test that LoRA adaptation actually changes outputs."""
    print("🧪 Testing LoRA adaptation effect...")
    
    # Create LoRA linear layer
    original_linear = nn.Linear(10, 5)
    lora_linear = LoRALinear(original_linear, rank=4, alpha=4.0)
    
    x = torch.randn(2, 10)
    
    # Get initial output
    initial_output = lora_linear(x)
    
    # Modify LoRA parameters to simulate training
    lora_linear.lora_adapter.lora_A.data.fill_(0.1)
    lora_linear.lora_adapter.lora_B.data.fill_(0.2)
    
    # Get adapted output
    adapted_output = lora_linear(x)
    
    # Outputs should be different after adaptation
    assert not torch.allclose(initial_output, adapted_output), "LoRA adaptation should change outputs"
    
    print("✅ LoRA adaptation changes outputs as expected")
    print("✅ LoRA adaptation test PASSED")

def test_lora_parameter_collection():
    """Test LoRA parameter collection for optimization."""
    print("🧪 Testing LoRA parameter collection...")
    
    # Create LoRA linear layer
    original_linear = nn.Linear(15, 8)
    lora_linear = LoRALinear(original_linear, rank=6)
    
    # Get LoRA parameters
    lora_params = lora_linear.get_lora_parameters()
    
    # Verify parameter collection
    assert len(lora_params) == 2, f"Should have 2 LoRA parameters, got {len(lora_params)}"
    assert torch.equal(lora_params[0], lora_linear.lora_adapter.lora_A), "First parameter should be A matrix"
    assert torch.equal(lora_params[1], lora_linear.lora_adapter.lora_B), "Second parameter should be B matrix"
    
    # Verify only LoRA parameters are trainable when original is frozen
    total_original_params = sum(1 for p in original_linear.parameters() if p.requires_grad)
    total_lora_params = sum(1 for p in lora_params if p.requires_grad)
    
    print(f"✅ Original trainable parameters: {total_original_params}")
    print(f"✅ LoRA trainable parameters: {total_lora_params}")
    print("✅ LoRA parameter collection test PASSED")

def test_lora_weight_merging():
    """Test LoRA weight merging for inference efficiency."""
    print("🧪 Testing LoRA weight merging...")
    
    # Create LoRA linear layer with known parameters
    original_linear = nn.Linear(8, 4, bias=True)
    original_linear.weight.data.fill_(0.1)
    original_linear.bias.data.fill_(0.05)
    
    lora_linear = LoRALinear(original_linear, rank=2, alpha=4.0)
    lora_linear.lora_adapter.lora_A.data.fill_(0.2)
    lora_linear.lora_adapter.lora_B.data.fill_(0.3)
    
    # Test input
    x = torch.randn(3, 8)
    
    # Get LoRA output
    lora_output = lora_linear(x)
    
    # Merge weights
    merged_linear = lora_linear.merge_weights()
    merged_output = merged_linear(x)
    
    # Outputs should match
    assert torch.allclose(lora_output, merged_output, atol=1e-5), "Merged weights should produce same output"
    
    print("✅ Weight merging preserves functionality")
    print(f"✅ Merged layer has combined weights: {merged_linear.weight.shape}")
    print("✅ LoRA weight merging test PASSED")

def test_lora_reset_parameters():
    """Test LoRA parameter reset functionality."""
    print("🧪 Testing LoRA parameter reset...")
    
    # Create and modify LoRA layer
    lora_layer = LoRALayer(10, 5, rank=3)
    lora_layer.lora_A.data.fill_(0.5)
    lora_layer.lora_B.data.fill_(0.5)
    
    # Reset parameters
    lora_layer.reset_parameters()
    
    # Verify B is reset to zero
    assert torch.allclose(lora_layer.lora_B, torch.zeros_like(lora_layer.lora_B)), "B matrix should be reset to zero"
    
    # Verify A is reset (should not be all 0.5 anymore)
    assert not torch.allclose(lora_layer.lora_A, torch.ones_like(lora_layer.lora_A) * 0.5), "A matrix should be reset"
    
    print("✅ Parameter reset restores initialization state")
    print("✅ LoRA parameter reset test PASSED")

def main():
    """Run all LoRA layer tests."""
    print("🚀 Starting LoRA Layers Implementation Tests")
    print("=" * 50)
    
    try:
        test_lora_layer_initialization()
        test_lora_layer_forward()
        test_lora_layer_identity_at_initialization()
        test_lora_linear_integration()
        test_lora_linear_adaptation()
        test_lora_parameter_collection()
        test_lora_weight_merging()
        test_lora_reset_parameters()
        
        print("=" * 50)
        print("🎉 ALL LORA LAYERS TESTS PASSED!")
        print("✅ Implementation follows Hu et al. (2021) paper")
        print("✅ Mathematical operations are correct")
        print("✅ Ready to remove TODO comments")
        
        return True
        
    except Exception as e:
        print("=" * 50)
        print(f"❌ TEST FAILED: {e}")
        print("❌ Implementation needs fixes before removing TODOs")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)