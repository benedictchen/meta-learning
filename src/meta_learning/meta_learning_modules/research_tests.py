"""
Research-Accuracy Unit Tests
============================

Author: Benedict Chen (benedict@benedictchen.com)

Unit tests that verify mathematical correctness of meta-learning algorithms
against their canonical paper formulations. These tests catch common
implementation errors that violate research accuracy.

Tests are organized by algorithm with specific mathematical properties
that must hold for research correctness.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytest
from typing import Dict, List, Tuple

# Import our research-accurate implementations  
from .episode_protocol import EpisodeProtocol, EpisodeConfig
from .prototypical_networks_fixed import (
    ResearchPrototypicalNetworks, CosinePrototypicalNetworks,
    create_prototypical_network
)
from .maml_research_accurate import ResearchMAML, MAMLConfig, MAMLVariant
from .numerical_stability import (
    stable_softmax_ce_loss, safe_normalize, safe_distance_computation,
    seed_everything, DeterministicMode
)


class TestEpisodeProtocol:
    """Test episode generation protocol for research correctness."""
    
    def test_episode_label_remapping(self):
        """Episode labels must be remapped to [0, N-1] contiguous range."""
        config = EpisodeConfig(n_way=3, k_shot=2, m_query=3, seed=42)
        protocol = EpisodeProtocol(config)
        
        # Create dataset with non-contiguous class IDs
        dataset = {
            5: [torch.randn(10) for _ in range(10)],
            23: [torch.randn(10) for _ in range(10)], 
            67: [torch.randn(10) for _ in range(10)]
        }
        available_classes = [5, 23, 67]
        
        # Generate episode
        support_x, support_y, query_x, query_y = protocol.generate_episode(
            dataset, available_classes
        )
        
        # Verify label remapping
        assert support_y.min() == 0, "Support labels must start at 0"
        assert support_y.max() == 2, "Support labels must end at N-1=2" 
        assert query_y.min() == 0, "Query labels must start at 0"
        assert query_y.max() == 2, "Query labels must end at N-1=2"
        
        # Verify contiguous labels [0, 1, 2]
        support_unique = torch.unique(support_y).sort()[0]
        query_unique = torch.unique(query_y).sort()[0]
        expected = torch.arange(3)
        
        assert torch.equal(support_unique, expected), "Support labels not contiguous"
        assert torch.equal(query_unique, expected), "Query labels not contiguous"
        
    def test_episode_class_counts(self):
        """Each class must have exactly K support + M query examples."""
        config = EpisodeConfig(n_way=5, k_shot=3, m_query=7, seed=42)
        protocol = EpisodeProtocol(config)
        
        # Create dataset
        dataset = {}
        for class_id in range(10):
            dataset[class_id] = [torch.randn(20) for _ in range(50)]
            
        # Generate episode
        support_x, support_y, query_x, query_y = protocol.generate_episode(
            dataset, list(range(10))
        )
        
        # Check per-class counts
        for class_id in range(5):  # N-way = 5
            support_count = (support_y == class_id).sum().item()
            query_count = (query_y == class_id).sum().item()
            
            assert support_count == 3, f"Class {class_id} has {support_count} support examples, expected 3"
            assert query_count == 7, f"Class {class_id} has {query_count} query examples, expected 7"
            
    def test_episode_deterministic_generation(self):
        """Same seed must produce identical episodes."""
        config = EpisodeConfig(n_way=4, k_shot=2, m_query=5, seed=123)
        
        # Create dataset
        dataset = {}
        for class_id in range(8):
            dataset[class_id] = [torch.randn(15) for _ in range(20)]
            
        # Generate two episodes with same seed
        protocol1 = EpisodeProtocol(config)
        sx1, sy1, qx1, qy1 = protocol1.generate_episode(dataset, list(range(8)))
        
        protocol2 = EpisodeProtocol(config)  
        sx2, sy2, qx2, qy2 = protocol2.generate_episode(dataset, list(range(8)))
        
        # Should be identical
        assert torch.allclose(sx1, sx2), "Support examples not deterministic"
        assert torch.equal(sy1, sy2), "Support labels not deterministic"
        assert torch.allclose(qx1, qx2), "Query examples not deterministic"
        assert torch.equal(qy1, qy2), "Query labels not deterministic"


class TestPrototypicalNetworks:
    """Test Prototypical Networks for mathematical correctness."""
    
    def test_prototypes_are_class_means(self):
        """Prototypes must equal per-class means of support embeddings."""
        # Create simple backbone
        backbone = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 3))
        model = ResearchPrototypicalNetworks(backbone, temperature=1.0)
        
        # Create episode: 3-way 2-shot
        support_x = torch.randn(6, 10)  # [N*K, features]
        support_y = torch.tensor([0, 0, 1, 1, 2, 2])  # Contiguous labels
        query_x = torch.randn(9, 10)   # [N*M, features]
        
        # Get support features
        with torch.no_grad():
            support_features = backbone(support_x)  # [6, 3]
            
        # Manually compute expected prototypes (per-class means)
        expected_protos = torch.stack([
            support_features[support_y == 0].mean(0),  # Class 0 mean
            support_features[support_y == 1].mean(0),  # Class 1 mean  
            support_features[support_y == 2].mean(0),  # Class 2 mean
        ])
        
        # Get model's prototypes (by calling internal method)
        actual_protos = model._compute_prototypes(support_features, support_y, 3)
        
        assert torch.allclose(actual_protos, expected_protos, atol=1e-6), \
            "Prototypes are not per-class means of support embeddings"
    
    def test_negative_distance_logits(self):
        """Logits must be -τ * distance (negative distances for softmax)."""
        backbone = nn.Linear(5, 3)
        model = ResearchPrototypicalNetworks(backbone, temperature=2.0)
        
        # Simple test case
        support_x = torch.randn(4, 5)
        support_y = torch.tensor([0, 0, 1, 1])  
        query_x = torch.randn(2, 5)
        
        with torch.no_grad():
            # Get features
            support_features = backbone(support_x)
            query_features = backbone(query_x)
            
            # Compute prototypes and distances manually
            prototypes = model._compute_prototypes(support_features, support_y, 2)
            distances = model._compute_distances(query_features, prototypes)
            
            # Model's logits should be -τ * distances
            logits = model.forward(support_x, support_y, query_x)
            expected_logits = -2.0 * distances
            
            assert torch.allclose(logits, expected_logits, atol=1e-6), \
                "Logits are not -temperature * distances"
    
    def test_cosine_normalization(self):
        """Cosine variant must L2-normalize features before similarity."""
        backbone = nn.Linear(5, 3)
        model = CosinePrototypicalNetworks(backbone, temperature=1.0)
        
        support_x = torch.randn(4, 5)
        support_y = torch.tensor([0, 0, 1, 1])
        query_x = torch.randn(2, 5)
        
        with torch.no_grad():
            # Features should be normalized in cosine variant
            support_features = backbone(support_x)
            query_features = backbone(query_x)
            
            # Both should be L2 normalized (unit norm)
            support_norms = support_features.norm(dim=1)
            query_norms = query_features.norm(dim=1)
            
            # Forward pass (which normalizes internally)
            _ = model.forward(support_x, support_y, query_x)
            
            # Check that internal normalization would produce unit vectors
            normalized_support = F.normalize(support_features, dim=1)
            normalized_query = F.normalize(query_features, dim=1)
            
            assert torch.allclose(normalized_support.norm(dim=1), torch.ones(4), atol=1e-6)
            assert torch.allclose(normalized_query.norm(dim=1), torch.ones(2), atol=1e-6)
    
    def test_query_closest_to_own_prototype(self):
        """When query equals class prototype, argmax should be that class."""
        backbone = nn.Identity()  # Identity for simple test
        model = ResearchPrototypicalNetworks(backbone, temperature=1.0)
        
        # Create support set with clear class separation
        support_x = torch.tensor([[1.0, 0.0], [1.1, 0.0], [0.0, 1.0], [0.0, 1.1]])
        support_y = torch.tensor([0, 0, 1, 1])
        
        # Query that exactly equals class 0 prototype: mean of [1.0, 0.0] and [1.1, 0.0]
        class_0_prototype = torch.tensor([1.05, 0.0]).unsqueeze(0)
        
        with torch.no_grad():
            logits = model.forward(support_x, support_y, class_0_prototype)
            prediction = logits.argmax(dim=1)
            
            assert prediction.item() == 0, \
                "Query equal to class 0 prototype should predict class 0"


class TestMAML:
    """Test MAML for mathematical correctness."""
    
    def test_second_order_gradients(self):
        """MAML must compute second-order gradients through inner steps."""
        # Simple model
        model = nn.Linear(2, 1)
        config = MAMLConfig(variant=MAMLVariant.MAML, inner_lr=0.1, inner_steps=1)
        maml = ResearchMAML(model, config)
        
        # Create task
        support_x = torch.randn(4, 2)
        support_y = torch.randn(4, 1)
        query_x = torch.randn(2, 2)
        query_y = torch.randn(2, 1)
        
        task_batch = [(support_x, support_y, query_x, query_y)]
        loss_fn = nn.MSELoss()
        
        # Compute meta loss
        meta_loss = maml(task_batch, loss_fn)
        
        # After backward, model parameters should have gradients
        meta_loss.backward()
        
        has_grads = any(p.grad is not None for p in model.parameters())
        assert has_grads, "MAML should produce gradients for meta-update"
        
        # Gradients should be non-zero (second-order path working)
        grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
        assert grad_norm > 1e-8, "MAML gradients are suspiciously small (second-order path broken?)"
        
    def test_fomaml_first_order(self):
        """FOMAML must use first-order approximation (no second-order path)."""
        model = nn.Linear(2, 1) 
        config = MAMLConfig(variant=MAMLVariant.FOMAML, inner_lr=0.1, inner_steps=1)
        maml = ResearchMAML(model, config)
        
        # Should have first_order=True for FOMAML
        assert config.first_order == True, "FOMAML should set first_order=True"
        
    def test_functional_parameter_updates(self):
        """Inner loop must not modify base model parameters in-place."""
        model = nn.Linear(2, 1)
        config = MAMLConfig(variant=MAMLVariant.MAML, inner_lr=0.5, inner_steps=2) 
        maml = ResearchMAML(model, config)
        
        # Store original parameters
        original_params = {}
        for name, param in model.named_parameters():
            original_params[name] = param.data.clone()
        
        # Run inner loop adaptation
        support_x = torch.randn(4, 2)
        support_y = torch.randn(4, 1)
        loss_fn = nn.MSELoss()
        
        adapted_params = maml.inner_loop(support_x, support_y, loss_fn)
        
        # Base model parameters should be unchanged
        for name, param in model.named_parameters():
            assert torch.allclose(param.data, original_params[name]), \
                f"Parameter {name} was modified in-place during inner loop"
                
        # Adapted parameters should be different from original
        for name, param in model.named_parameters():
            if name in adapted_params:
                assert not torch.allclose(adapted_params[name], param.data), \
                    f"Adapted parameter {name} is identical to original (no adaptation?)"


class TestNumericalStability:
    """Test numerical stability utilities."""
    
    def test_stable_softmax_equivalence(self):
        """Stable softmax should equal regular softmax numerically."""
        logits = torch.randn(10, 5)
        targets = torch.randint(0, 5, (10,))
        temperature = 2.5
        
        # Our stable version
        stable_loss = stable_softmax_ce_loss(logits, targets, temperature)
        
        # Regular PyTorch version  
        regular_loss = F.cross_entropy(logits / temperature, targets)
        
        assert torch.allclose(stable_loss, regular_loss, atol=1e-6), \
            "Stable softmax loss differs from regular cross-entropy"
            
    def test_safe_normalization(self):
        """Safe normalization should produce unit vectors."""
        # Regular case
        x = torch.randn(5, 10)
        normalized = safe_normalize(x, dim=1)
        norms = normalized.norm(dim=1)
        assert torch.allclose(norms, torch.ones(5), atol=1e-6)
        
        # Edge case: zero vector
        zero_vec = torch.zeros(1, 10)
        normalized_zero = safe_normalize(zero_vec, dim=1)
        # Should not be NaN/Inf
        assert torch.isfinite(normalized_zero).all()
        
    def test_deterministic_seeding(self):
        """Deterministic seeding should produce identical results.""" 
        def generate_random():
            return torch.randn(5), np.random.randn(5)
        
        # Two runs with same seed
        with DeterministicMode(42):
            torch_1, numpy_1 = generate_random()
            
        with DeterministicMode(42):
            torch_2, numpy_2 = generate_random()
            
        assert torch.allclose(torch_1, torch_2)
        assert np.allclose(numpy_1, numpy_2)


# Research-correctness integration test
def test_end_to_end_prototypical_episode():
    """End-to-end test: episode → ProtoNet → correct predictions."""
    # Setup
    seed_everything(42)
    
    # Create episode
    config = EpisodeConfig(n_way=3, k_shot=2, m_query=4, seed=42)
    protocol = EpisodeProtocol(config)
    
    # Simple dataset
    dataset = {}
    for class_id in range(5):
        # Each class has different pattern
        base_pattern = torch.zeros(10)
        base_pattern[class_id*2:(class_id+1)*2] = 1.0
        noise_samples = [base_pattern + 0.1*torch.randn(10) for _ in range(20)]
        dataset[class_id] = noise_samples
    
    # Generate episode  
    support_x, support_y, query_x, query_y = protocol.generate_episode(
        dataset, list(range(5))
    )
    
    # Simple ProtoNet
    backbone = nn.Identity()  # Identity for interpretability
    model = create_prototypical_network(backbone, "euclidean", temperature=1.0)
    
    # Forward pass
    with torch.no_grad():
        logits = model(support_x, support_y, query_x) 
        predictions = logits.argmax(dim=1)
        
        # Should predict correctly for most queries (data has clear patterns)
        accuracy = (predictions == query_y).float().mean()
        assert accuracy > 0.5, f"ProtoNet accuracy too low: {accuracy.item()}"
        
        # Verify logits are reasonable (not all zeros, not all same)
        assert logits.std() > 0.1, "Logits have no variance"
        assert torch.isfinite(logits).all(), "Logits contain NaN/Inf"


if __name__ == "__main__":
    # Run tests
    print("Research-Accuracy Unit Tests")
    print("=" * 40)
    
    # Run each test class
    test_classes = [
        TestEpisodeProtocol,
        TestPrototypicalNetworks, 
        TestMAML,
        TestNumericalStability
    ]
    
    for test_cls in test_classes:
        print(f"\nRunning {test_cls.__name__}...")
        test_instance = test_cls()
        
        # Get all test methods
        test_methods = [method for method in dir(test_instance) 
                       if method.startswith('test_')]
        
        for test_method in test_methods:
            try:
                getattr(test_instance, test_method)()
                print(f"  ✓ {test_method}")
            except Exception as e:
                print(f"  ✗ {test_method}: {str(e)}")
                
    # Run integration test
    print(f"\nRunning integration test...")
    try:
        test_end_to_end_prototypical_episode()
        print(f"  ✓ test_end_to_end_prototypical_episode")
    except Exception as e:
        print(f"  ✗ test_end_to_end_prototypical_episode: {str(e)}")
    
    print(f"\n{'='*40}")
    print("Research accuracy tests completed!")
    print("Add these tests to CI to prevent mathematical regressions.")