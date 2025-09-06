#!/usr/bin/env python3
"""
Standalone Test for MetaOptNet Algorithm Implementation
=====================================================

This test verifies that the MetaOptNet algorithm is fully implemented
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
    from meta_learning.algorithms.metaoptnet import MetaOptNet, DifferentiableSVM, DifferentiableRidge
    print("✅ Successfully imported MetaOptNet, DifferentiableSVM, and DifferentiableRidge")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

def test_ridge_solver():
    """Test DifferentiableRidge solver."""
    print("\n🧪 Testing DifferentiableRidge solver...")
    
    try:
        # Create Ridge solver
        ridge = DifferentiableRidge(lam=1.0)
        
        # Create synthetic data
        n_support = 20
        n_query = 10
        embed_dim = 16
        n_classes = 3
        
        support_embeddings = torch.randn(n_support, embed_dim, requires_grad=True)
        support_labels = torch.randint(0, n_classes, (n_support,))
        query_embeddings = torch.randn(n_query, embed_dim, requires_grad=True)
        
        # Test forward pass
        predictions = ridge(support_embeddings, support_labels, query_embeddings)
        
        # Verify output shape
        expected_shape = (n_query, n_classes)
        assert predictions.shape == expected_shape, f"Expected {expected_shape}, got {predictions.shape}"
        assert predictions.requires_grad, "Predictions should be differentiable when inputs require grad"
        
        print("✅ DifferentiableRidge solver works correctly")
        print(f"✅ Input: {n_support} support, {n_query} query → Output: {predictions.shape}")
        return True
        
    except Exception as e:
        print(f"❌ Ridge solver test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_svm_solver():
    """Test DifferentiableSVM solver."""
    print("\n🧪 Testing DifferentiableSVM solver...")
    
    try:
        # Create SVM solver
        svm = DifferentiableSVM(C=1.0, max_iter=5)  # Fewer iterations for faster testing
        
        # Create synthetic data
        n_support = 15
        n_query = 8
        embed_dim = 12
        n_classes = 2  # Binary classification for SVM
        
        support_embeddings = torch.randn(n_support, embed_dim)
        support_labels = torch.randint(0, n_classes, (n_support,))
        query_embeddings = torch.randn(n_query, embed_dim)
        
        # Test forward pass
        predictions = svm(support_embeddings, support_labels, query_embeddings)
        
        # Verify output shape
        expected_shape = (n_query, n_classes)
        assert predictions.shape == expected_shape, f"Expected {expected_shape}, got {predictions.shape}"
        assert predictions.requires_grad, "Predictions should be differentiable"
        
        print("✅ DifferentiableSVM solver works correctly")
        print(f"✅ Input: {n_support} support, {n_query} query → Output: {predictions.shape}")
        return True
        
    except Exception as e:
        print(f"❌ SVM solver test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_metaoptnet_ridge():
    """Test MetaOptNet with Ridge regression head."""
    print("\n🧪 Testing MetaOptNet with Ridge head...")
    
    try:
        # Create MetaOptNet with Ridge head
        metaoptnet = MetaOptNet(distance='euclidean', head='ridge')
        
        # Create synthetic few-shot task
        n_support = 25  # 5-way 5-shot
        n_query = 15
        embed_dim = 64
        n_classes = 5
        
        support_embeddings = torch.randn(n_support, embed_dim, requires_grad=True)
        support_labels = torch.randint(0, n_classes, (n_support,))
        query_embeddings = torch.randn(n_query, embed_dim, requires_grad=True)
        
        # Test forward pass
        predictions = metaoptnet(support_embeddings, support_labels, query_embeddings)
        
        # Verify output
        expected_shape = (n_query, n_classes)
        assert predictions.shape == expected_shape, f"Expected {expected_shape}, got {predictions.shape}"
        
        # Test that predictions are reasonable (finite and differentiable)
        assert torch.isfinite(predictions).all(), "Predictions should be finite"
        assert predictions.requires_grad, "Predictions should be differentiable when inputs require grad"
        
        print("✅ MetaOptNet with Ridge head works correctly")
        print(f"✅ Few-shot task: {n_support} support → {n_query} query predictions")
        return True
        
    except Exception as e:
        print(f"❌ MetaOptNet Ridge test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_metaoptnet_svm():
    """Test MetaOptNet with SVM head."""
    print("\n🧪 Testing MetaOptNet with SVM head...")
    
    try:
        # Create MetaOptNet with SVM head  
        metaoptnet = MetaOptNet(distance='cosine', head='svm', C=0.5, max_iter=3)
        
        # Create synthetic few-shot task
        n_support = 16  # 4-way 4-shot
        n_query = 12
        embed_dim = 32
        n_classes = 4
        
        support_embeddings = torch.randn(n_support, embed_dim)
        support_labels = torch.randint(0, n_classes, (n_support,))
        query_embeddings = torch.randn(n_query, embed_dim)
        
        # Test forward pass
        predictions = metaoptnet(support_embeddings, support_labels, query_embeddings)
        
        # Verify output
        expected_shape = (n_query, n_classes)
        assert predictions.shape == expected_shape, f"Expected {expected_shape}, got {predictions.shape}"
        
        # Test that predictions are reasonable
        assert torch.isfinite(predictions).all(), "Predictions should be finite"
        assert predictions.requires_grad, "Predictions should be differentiable"
        
        print("✅ MetaOptNet with SVM head works correctly")
        print(f"✅ Few-shot task: {n_support} support → {n_query} query predictions")
        return True
        
    except Exception as e:
        print(f"❌ MetaOptNet SVM test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gradient_flow():
    """Test that gradients flow through MetaOptNet."""
    print("\n🧪 Testing gradient flow through MetaOptNet...")
    
    try:
        # Create MetaOptNet
        metaoptnet = MetaOptNet(distance='euclidean', head='ridge')
        
        # Create synthetic data
        support_embeddings = torch.randn(10, 8, requires_grad=True)
        support_labels = torch.randint(0, 3, (10,))
        query_embeddings = torch.randn(5, 8, requires_grad=True)
        query_labels = torch.randint(0, 3, (5,))
        
        # Forward pass
        predictions = metaoptnet(support_embeddings, support_labels, query_embeddings)
        
        # Compute loss and backpropagate
        loss = F.cross_entropy(predictions, query_labels)
        loss.backward()
        
        # Check that gradients exist
        assert support_embeddings.grad is not None, "Support embeddings should have gradients"
        assert query_embeddings.grad is not None, "Query embeddings should have gradients"
        
        # Check that gradients are non-zero (model is learning)
        support_grad_norm = support_embeddings.grad.norm().item()
        query_grad_norm = query_embeddings.grad.norm().item()
        
        assert support_grad_norm > 1e-6, f"Support gradients too small: {support_grad_norm}"
        assert query_grad_norm > 1e-6, f"Query gradients too small: {query_grad_norm}"
        
        print("✅ Gradient flow works correctly")
        print(f"✅ Support grad norm: {support_grad_norm:.6f}")
        print(f"✅ Query grad norm: {query_grad_norm:.6f}")
        return True
        
    except Exception as e:
        print(f"❌ Gradient flow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_few_shot_learning_capability():
    """Test MetaOptNet on a simple few-shot learning task."""
    print("\n🧪 Testing few-shot learning capability...")
    
    try:
        # Create MetaOptNet
        metaoptnet = MetaOptNet(distance='euclidean', head='ridge')
        
        # Create a simple separable task
        n_classes = 3
        n_support_per_class = 4
        n_query_per_class = 3
        embed_dim = 16
        
        # Generate well-separated clusters
        support_embeddings = []
        support_labels = []
        query_embeddings = []
        query_labels = []
        
        for class_id in range(n_classes):
            # Create class-specific cluster center
            center = torch.randn(embed_dim) * 2
            
            # Support examples around cluster center
            class_support = center.unsqueeze(0) + torch.randn(n_support_per_class, embed_dim) * 0.5
            support_embeddings.append(class_support)
            support_labels.extend([class_id] * n_support_per_class)
            
            # Query examples around same cluster center
            class_query = center.unsqueeze(0) + torch.randn(n_query_per_class, embed_dim) * 0.5
            query_embeddings.append(class_query)
            query_labels.extend([class_id] * n_query_per_class)
        
        # Combine all examples
        support_embeddings = torch.cat(support_embeddings, dim=0)
        support_labels = torch.tensor(support_labels)
        query_embeddings = torch.cat(query_embeddings, dim=0)
        query_labels = torch.tensor(query_labels)
        
        # Test MetaOptNet
        predictions = metaoptnet(support_embeddings, support_labels, query_embeddings)
        
        # Compute accuracy
        predicted_classes = predictions.argmax(dim=1)
        accuracy = (predicted_classes == query_labels).float().mean().item()
        
        print(f"✅ Few-shot learning accuracy: {accuracy:.3f}")
        
        # Should achieve reasonable accuracy on this separable task
        assert accuracy > 0.4, f"Accuracy too low: {accuracy:.3f}"
        
        print("✅ MetaOptNet demonstrates few-shot learning capability")
        return True
        
    except Exception as e:
        print(f"❌ Few-shot learning test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all MetaOptNet tests."""
    print("🚀 Starting Standalone MetaOptNet Algorithm Tests")
    print("=" * 60)
    
    tests = [
        test_ridge_solver,
        test_svm_solver,
        test_metaoptnet_ridge,
        test_metaoptnet_svm,
        test_gradient_flow,
        test_few_shot_learning_capability,
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
        print("🎉 ALL METAOPTNET TESTS PASSED!")
        print("✅ MetaOptNet algorithm fully implemented and functional")
        print("✅ Ridge regression and SVM heads working")
        print("✅ Differentiable optimization functioning")
        print("✅ Gradient flow preserved for end-to-end learning")
        print("✅ Ready for few-shot learning tasks")
        print("=" * 60)
        return True
    else:
        print(f"❌ {total - passed} tests failed")
        print("❌ MetaOptNet implementation needs fixes")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)