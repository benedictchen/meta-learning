#!/usr/bin/env python3
"""Test script for advanced mathematical utilities in Phase 2.2"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def test_magic_box():
    print("Testing magic_box() function...")
    
    try:
        from src.meta_learning.core.math_utils import magic_box
        
        # Test basic functionality
        x = torch.tensor([1.0, 2.0, -1.0], requires_grad=True)
        result = magic_box(x)
        
        # Forward pass should give 1.0
        expected = torch.ones_like(x)
        assert torch.allclose(result, expected), f"Expected {expected}, got {result}"
        print("✓ Forward pass correct: magic_box(x) = 1.0")
        
        # Test gradient computation
        y = result.sum()
        y.backward()
        
        # Gradient should be 1.0 for each element
        expected_grad = torch.ones_like(x)
        assert torch.allclose(x.grad, expected_grad), f"Expected grad {expected_grad}, got {x.grad}"
        print("✓ Gradient correct: d/dx magic_box(x) = 1.0")
        
        # Test with different shapes
        x_2d = torch.randn(3, 4, requires_grad=True)
        result_2d = magic_box(x_2d)
        assert result_2d.shape == x_2d.shape
        assert torch.allclose(result_2d, torch.ones_like(x_2d))
        print("✓ Works with 2D tensors")
        
        # Test gradient flow in computation graph
        x_graph = torch.randn(5, requires_grad=True)
        intermediate = x_graph ** 2
        magic_result = magic_box(intermediate)
        final = (magic_result * 2).sum()
        final.backward()
        
        # Should have gradients
        assert x_graph.grad is not None
        assert not torch.allclose(x_graph.grad, torch.zeros_like(x_graph.grad))
        print("✓ Gradients flow correctly through computation graph")
        
        return True
        
    except Exception as e:
        print(f"✗ magic_box test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pairwise_cosine_similarity():
    print("\nTesting pairwise_cosine_similarity() function...")
    
    try:
        from src.meta_learning.core.math_utils import pairwise_cosine_similarity
        
        # Test basic functionality
        a = torch.randn(3, 4)  # 3 embeddings of dimension 4
        b = torch.randn(2, 4)  # 2 embeddings of dimension 4
        
        similarities = pairwise_cosine_similarity(a, b)
        assert similarities.shape == (3, 2), f"Expected shape (3, 2), got {similarities.shape}"
        
        # Values should be in [-1, 1] range for normalized embeddings
        assert torch.all(similarities >= -1.1) and torch.all(similarities <= 1.1)
        print("✓ Basic functionality correct")
        
        # Test temperature scaling
        similarities_cold = pairwise_cosine_similarity(a, b, temperature=0.1)
        similarities_hot = pairwise_cosine_similarity(a, b, temperature=10.0)
        
        # Cold temperature should produce more extreme values
        assert torch.abs(similarities_cold).mean() >= torch.abs(similarities_hot).mean()
        print("✓ Temperature scaling works")
        
        # Test without normalization
        similarities_no_norm = pairwise_cosine_similarity(a, b, normalize=False)
        assert similarities_no_norm.shape == (3, 2)
        print("✓ Works without normalization")
        
        # Test edge case: empty tensors
        empty_a = torch.empty(0, 4)
        empty_b = torch.empty(2, 4)
        empty_sim = pairwise_cosine_similarity(empty_a, empty_b)
        assert empty_sim.shape == (0, 2)
        print("✓ Handles empty tensors")
        
        # Test self-similarity
        self_sim = pairwise_cosine_similarity(a, a)
        assert self_sim.shape == (3, 3)
        # Diagonal should be close to 1 (self-similarity)
        diagonal = torch.diag(self_sim)
        assert torch.allclose(diagonal, torch.ones(3), atol=1e-5)
        print("✓ Self-similarity correct")
        
        return True
        
    except Exception as e:
        print(f"✗ pairwise_cosine_similarity test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_matching_loss():
    print("\nTesting matching_loss() function...")
    
    try:
        from src.meta_learning.core.math_utils import matching_loss
        
        # Create mock few-shot learning data
        # 2-way 3-shot support set
        support_embeddings = torch.randn(6, 8)  # 6 samples, 8-dimensional
        support_labels = torch.tensor([0, 0, 0, 1, 1, 1])  # 3 samples per class
        
        # 2-way 2-query query set  
        query_embeddings = torch.randn(4, 8)  # 4 query samples
        query_labels = torch.tensor([0, 0, 1, 1])  # 2 queries per class
        
        # Test cosine distance metric
        loss_cosine = matching_loss(
            support_embeddings, support_labels,
            query_embeddings, query_labels,
            distance_metric="cosine"
        )
        
        assert torch.is_tensor(loss_cosine)
        assert loss_cosine.dim() == 0  # Scalar loss
        assert loss_cosine.item() >= 0  # Loss should be non-negative
        print(f"✓ Cosine matching loss: {loss_cosine.item():.4f}")
        
        # Test euclidean distance metric
        loss_euclidean = matching_loss(
            support_embeddings, support_labels,
            query_embeddings, query_labels,
            distance_metric="euclidean"
        )
        
        assert torch.is_tensor(loss_euclidean)
        assert loss_euclidean.item() >= 0
        print(f"✓ Euclidean matching loss: {loss_euclidean.item():.4f}")
        
        # Test manhattan distance metric
        loss_manhattan = matching_loss(
            support_embeddings, support_labels,
            query_embeddings, query_labels,
            distance_metric="manhattan"
        )
        
        assert torch.is_tensor(loss_manhattan)
        assert loss_manhattan.item() >= 0
        print(f"✓ Manhattan matching loss: {loss_manhattan.item():.4f}")
        
        # Test temperature scaling
        loss_cold = matching_loss(
            support_embeddings, support_labels,
            query_embeddings, query_labels,
            temperature=0.1
        )
        
        loss_hot = matching_loss(
            support_embeddings, support_labels,
            query_embeddings, query_labels,
            temperature=10.0
        )
        
        # Different temperatures should give different losses
        assert not torch.allclose(loss_cold, loss_hot)
        print("✓ Temperature scaling affects loss")
        
        # Test reduction modes
        loss_none = matching_loss(
            support_embeddings, support_labels,
            query_embeddings, query_labels,
            reduction="none"
        )
        
        assert loss_none.shape == (4,)  # Per-query losses
        print("✓ Reduction='none' works")
        
        # Test edge case: empty tensors
        loss_empty = matching_loss(
            torch.empty(0, 8), torch.empty(0, dtype=torch.long),
            torch.empty(0, 8), torch.empty(0, dtype=torch.long)
        )
        assert loss_empty.item() == 0.0
        print("✓ Handles empty tensors")
        
        return True
        
    except Exception as e:
        print(f"✗ matching_loss test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_attention_matching_loss():
    print("\nTesting attention_matching_loss() function...")
    
    try:
        from src.meta_learning.core.math_utils import attention_matching_loss
        
        # Create mock data
        support_embeddings = torch.randn(6, 8)
        support_labels = torch.tensor([0, 0, 0, 1, 1, 1])
        query_embeddings = torch.randn(4, 8)
        query_labels = torch.tensor([0, 0, 1, 1])
        
        # Test basic functionality
        loss = attention_matching_loss(
            support_embeddings, support_labels,
            query_embeddings, query_labels
        )
        
        assert torch.is_tensor(loss)
        assert loss.dim() == 0
        assert loss.item() >= 0
        print(f"✓ Attention matching loss: {loss.item():.4f}")
        
        # Test with different attention dimension
        loss_diff_dim = attention_matching_loss(
            support_embeddings, support_labels,
            query_embeddings, query_labels,
            attention_dim=32
        )
        
        assert torch.is_tensor(loss_diff_dim)
        print("✓ Works with different attention dimensions")
        
        # Test temperature scaling
        loss_cold = attention_matching_loss(
            support_embeddings, support_labels,
            query_embeddings, query_labels,
            temperature=0.1
        )
        
        loss_hot = attention_matching_loss(
            support_embeddings, support_labels,
            query_embeddings, query_labels,
            temperature=10.0
        )
        
        # Should produce different results
        assert not torch.allclose(loss_cold, loss_hot)
        print("✓ Temperature scaling works")
        
        # Test edge case: empty tensors
        loss_empty = attention_matching_loss(
            torch.empty(0, 8), torch.empty(0, dtype=torch.long),
            torch.empty(0, 8), torch.empty(0, dtype=torch.long)
        )
        assert loss_empty.item() == 0.0
        print("✓ Handles empty tensors")
        
        return True
        
    except Exception as e:
        print(f"✗ attention_matching_loss test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_learnable_distance_metric():
    print("\nTesting LearnableDistanceMetric class...")
    
    try:
        from src.meta_learning.core.math_utils import LearnableDistanceMetric, learnable_distance_metric
        
        input_dim = 8
        embeddings_a = torch.randn(3, input_dim)
        embeddings_b = torch.randn(2, input_dim)
        
        # Test Mahalanobis metric
        mahal_metric = LearnableDistanceMetric(input_dim, metric_type="mahalanobis")
        distances_mahal = mahal_metric(embeddings_a, embeddings_b)
        
        assert distances_mahal.shape == (3, 2)
        assert torch.all(distances_mahal >= 0)  # Distances should be non-negative
        print("✓ Mahalanobis distance metric works")
        
        # Test neural metric
        neural_metric = LearnableDistanceMetric(input_dim, hidden_dim=16, metric_type="neural")
        distances_neural = neural_metric(embeddings_a, embeddings_b)
        
        assert distances_neural.shape == (3, 2)
        assert torch.all(distances_neural >= 0) and torch.all(distances_neural <= 1)  # Sigmoid output
        print("✓ Neural distance metric works")
        
        # Test bilinear metric
        bilinear_metric = LearnableDistanceMetric(input_dim, metric_type="bilinear")
        distances_bilinear = bilinear_metric(embeddings_a, embeddings_b)
        
        assert distances_bilinear.shape == (3, 2)
        print("✓ Bilinear distance metric works")
        
        # Test learning (gradients flow through metrics)
        loss = distances_mahal.sum()
        loss.backward()
        
        # Parameters should have gradients
        for param in mahal_metric.parameters():
            assert param.grad is not None
            assert not torch.allclose(param.grad, torch.zeros_like(param.grad))
        
        print("✓ Gradients flow through learnable metrics")
        
        # Test convenience function
        distances_func = learnable_distance_metric(embeddings_a, embeddings_b, neural_metric)
        assert distances_func.shape == (3, 2)
        print("✓ Convenience function works")
        
        # Test invalid metric type
        try:
            invalid_metric = LearnableDistanceMetric(input_dim, metric_type="invalid")
            assert False, "Should have raised ValueError"
        except ValueError:
            print("✓ Invalid metric type raises ValueError")
        
        return True
        
    except Exception as e:
        print(f"✗ LearnableDistanceMetric test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integration():
    print("\nTesting integration of all math utilities...")
    
    try:
        from src.meta_learning.core.math_utils import (
            magic_box, pairwise_cosine_similarity, matching_loss,
            LearnableDistanceMetric
        )
        
        # Create a mini few-shot learning scenario
        support_embeddings = torch.randn(9, 16, requires_grad=True)  # 3-way 3-shot
        support_labels = torch.tensor([0,0,0, 1,1,1, 2,2,2])
        query_embeddings = torch.randn(6, 16, requires_grad=True)    # 3-way 2-query  
        query_labels = torch.tensor([0,0, 1,1, 2,2])
        
        # Use magic_box in stochastic context (mock)
        stochastic_weights = magic_box(torch.randn(16, requires_grad=True))
        enhanced_support = support_embeddings * stochastic_weights.unsqueeze(0)
        
        # Compute similarities
        similarities = pairwise_cosine_similarity(enhanced_support[:3], enhanced_support[3:6])
        assert similarities.shape == (3, 3)
        
        # Use learnable distance metric
        distance_metric = LearnableDistanceMetric(16, metric_type="neural")
        learned_distances = distance_metric(enhanced_support, query_embeddings)
        assert learned_distances.shape == (9, 6)
        
        # Compute matching loss
        loss = matching_loss(
            enhanced_support, support_labels,
            query_embeddings, query_labels,
            distance_metric="cosine"
        )
        
        # Add learnable distance metric to loss to ensure gradients flow
        distance_component = learned_distances.mean() * 0.01  # Small weight
        total_loss = loss + distance_component
        
        # Backpropagate to test full gradient flow
        total_loss.backward()
        
        # Check that gradients flow through everything
        assert support_embeddings.grad is not None
        assert query_embeddings.grad is not None
        
        # Check distance metric gradients (at least some should be non-None)
        distance_grads = [p.grad is not None for p in distance_metric.parameters()]
        assert any(distance_grads), "At least some distance metric parameters should have gradients"
        
        print("✓ All utilities integrate correctly")
        print(f"✓ Final loss: {loss.item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
    
    print("Testing PHASE 2.2 Advanced Mathematical Utilities")
    print("=" * 50)
    
    success = True
    success &= test_magic_box()
    success &= test_pairwise_cosine_similarity()
    success &= test_matching_loss()
    success &= test_attention_matching_loss()
    success &= test_learnable_distance_metric()
    success &= test_integration()
    
    print("\n" + "=" * 50)
    if success:
        print("✅ All Phase 2.2 mathematical utilities tests completed successfully!")
    else:
        print("❌ Some Phase 2.2 mathematical utilities tests failed!")
        sys.exit(1)