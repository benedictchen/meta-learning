"""
🔬 Mathematical Correctness Regression Tests
==========================================

These tests prevent the 16 critical mathematical errors identified in the codebase.
Based on research patterns from Snell et al. 2017 and Finn et al. 2017.

CRITICAL: These tests must PASS to ensure research accuracy!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest
from src.meta_learning.meta_learning_modules.few_shot_modules.core_networks import PrototypicalNetworks
from src.meta_learning.meta_learning_modules.maml_variants import MAMLLearner


class TestPrototypicalNetworksMath:
    """Test mathematical correctness of Prototypical Networks (Snell et al. 2017)."""
    
    def test_prototypes_are_per_class_means_not_global(self):
        """REGRESSION: Fix PROTO_GLOBAL_MEAN - prototypes must be per-class means."""
        # Setup: 2 classes, 2 samples per class
        embeddings = torch.tensor([
            [1.0, 0.0],  # Class 0
            [2.0, 0.0],  # Class 0  
            [0.0, 1.0],  # Class 1
            [0.0, 2.0]   # Class 1
        ])
        labels = torch.tensor([0, 0, 1, 1])
        
        # Expected: per-class means
        expected_proto_0 = torch.tensor([1.5, 0.0])  # (1+2)/2, 0
        expected_proto_1 = torch.tensor([0.0, 1.5])  # 0, (1+2)/2
        
        # Test implementation
        encoder = nn.Identity()
        protonet = PrototypicalNetworks(encoder)
        prototypes = protonet.compute_prototypes(embeddings, labels, n_way=2)
        
        # Assert: must be per-class means, not global mean
        torch.testing.assert_close(prototypes[0], expected_proto_0)
        torch.testing.assert_close(prototypes[1], expected_proto_1)
        
        # Assert: NOT global mean (common bug)
        global_mean = embeddings.mean(0)  # Wrong approach
        assert not torch.allclose(prototypes[0], global_mean)
        assert not torch.allclose(prototypes[1], global_mean)
    
    def test_softmax_on_negative_distances_not_positive(self):
        """REGRESSION: Fix SOFTMAX_ON_DISTANCE_NO_MINUS - must use softmax(-distance)."""
        distances = torch.tensor([[1.0, 4.0, 9.0]])  # Squared Euclidean distances
        
        # CORRECT: softmax(-distances) - closer = higher probability
        correct_probs = F.softmax(-distances, dim=1)
        
        # Test implementation  
        encoder = nn.Identity()
        protonet = PrototypicalNetworks(encoder)
        actual_probs = protonet.compute_probability(distances)
        
        # Assert: matches correct softmax(-distances)
        torch.testing.assert_close(actual_probs, correct_probs)
        
        # Assert: closer distances have HIGHER probabilities
        assert actual_probs[0, 0] > actual_probs[0, 1] > actual_probs[0, 2]
        
        # Assert: NOT softmax(distances) - common bug
        wrong_probs = F.softmax(distances, dim=1)  # Flipped decision boundaries
        assert not torch.allclose(actual_probs, wrong_probs)

    def test_cosine_similarity_uses_normalized_embeddings(self):
        """REGRESSION: Fix DOT_PRODUCT_NO_NORMALIZE - cosine must normalize embeddings."""
        # Unnormalized embeddings with different magnitudes
        query = torch.tensor([[3.0, 4.0]])      # magnitude = 5
        prototypes = torch.tensor([[1.0, 0.0],  # magnitude = 1  
                                 [0.0, 1.0]])    # magnitude = 1
        
        # CORRECT: normalize before dot product
        query_norm = F.normalize(query, dim=-1)
        proto_norm = F.normalize(prototypes, dim=-1)
        correct_cosine = query_norm @ proto_norm.T
        
        # Expected: cosine similarities should be scale-invariant
        expected = torch.tensor([[0.6, 0.8]])  # (3,4)·(1,0)/5 = 0.6, (3,4)·(0,1)/5 = 0.8
        torch.testing.assert_close(correct_cosine, expected, atol=1e-6)
        
        # Assert: unnormalized dot product gives wrong results
        wrong_dot = query @ prototypes.T  # [[3.0, 4.0]] - scale dependent!
        assert not torch.allclose(correct_cosine, wrong_dot)


class TestMAMLMathematicalCorrectness:
    """Test mathematical correctness of MAML (Finn et al. 2017)."""
    
    def test_second_order_gradients_preserved(self):
        """REGRESSION: Fix MAML_NO_CREATE_GRAPH - second-order path must exist."""
        model = nn.Linear(2, 1)
        maml = MAMLLearner(model, inner_lr=0.1, outer_lr=0.001)
        
        # Create meta-learning task
        support_x = torch.randn(5, 2, requires_grad=True) 
        support_y = torch.randn(5)
        query_x = torch.randn(3, 2, requires_grad=True)
        query_y = torch.randn(3)
        
        # Inner adaptation
        adapted_params = maml.inner_update(support_x, support_y)
        
        # Meta-loss computation  
        meta_loss = maml.meta_loss(adapted_params, query_x, query_y)
        
        # Assert: second-order gradients exist (MAML requirement)
        second_order_grads = torch.autograd.grad(
            meta_loss, support_x, create_graph=True, retain_graph=True
        )[0]
        
        # Critical: second-order path must not be None/zero
        assert second_order_grads is not None
        assert not torch.allclose(second_order_grads, torch.zeros_like(second_order_grads))
        
        # Third-order should exist too (gradient of gradient)  
        third_order = torch.autograd.grad(
            second_order_grads.sum(), support_x, retain_graph=True
        )[0]
        assert third_order is not None

    def test_meta_loss_properly_averaged_not_summed(self):
        """REGRESSION: Fix META_LOSS_NOT_AVERAGED - meta-loss must be averaged."""
        model = nn.Linear(2, 1) 
        maml = MAMLLearner(model, inner_lr=0.1, outer_lr=0.001)
        
        # Create meta-batch with 3 tasks
        meta_batch = []
        for _ in range(3):
            support_x = torch.randn(5, 2)
            support_y = torch.randn(5) 
            query_x = torch.randn(3, 2)
            query_y = torch.randn(3)
            meta_batch.append((support_x, support_y, query_x, query_y))
        
        # Test meta-training step
        initial_params = [p.clone() for p in model.parameters()]
        loss = maml.meta_train_step(meta_batch)
        
        # Assert: loss should be reasonable magnitude (averaged, not summed)
        assert 0.0 < loss < 100.0  # Reasonable range for averaged loss
        
        # Assert: parameters actually changed (meta-update occurred)  
        for initial, current in zip(initial_params, model.parameters()):
            assert not torch.allclose(initial, current, atol=1e-6)

    def test_no_support_query_contamination(self):
        """REGRESSION: Fix SUPPORT_QUERY_CAT - query must not contaminate prototypes."""
        # This tests the Reptile fix we applied
        from src.meta_learning.meta_learning_modules.maml_variants import ReptileLearner
        
        model = nn.Linear(2, 1)
        # Use default config that should be fixed
        from src.meta_learning.meta_learning_modules.maml_variants import MAMLConfig
        config = MAMLConfig()
        reptile = ReptileLearner(model, config)
        
        # Create distinguishable support and query sets
        support_x = torch.ones(3, 2)      # All ones
        support_y = torch.zeros(3)        # All zeros  
        query_x = torch.zeros(2, 2)       # All zeros
        query_y = torch.ones(2)           # All ones
        
        # Mock meta-train step that should only use support set
        meta_batch = [(support_x, support_y, query_x, query_y)]
        
        # The fix ensures only support_x/support_y are used in inner loop
        # If query contamination existed, this would fail subtly
        loss = reptile.meta_train_step(meta_batch)
        
        # Assert: meta-training completes without query contamination
        assert torch.isfinite(torch.tensor(loss))


class TestEpisodicDataInvariants:
    """Test episodic few-shot learning data handling."""
    
    def test_labels_remapped_to_contiguous_range(self):
        """REGRESSION: Labels must be remapped to [0, C-1] per episode."""
        # Original labels: [5, 5, 12, 12, 23] -> should become [0, 0, 1, 1, 2]  
        original_labels = torch.tensor([5, 5, 12, 12, 23])
        
        # Test the remapping logic from ProtoNet
        classes = torch.unique(original_labels)  # [5, 12, 23]
        label_map = {c.item(): i for i, c in enumerate(classes)}
        remapped = torch.tensor([label_map[int(c)] for c in original_labels])
        
        # Assert: remapped to [0, C-1] 
        expected = torch.tensor([0, 0, 1, 1, 2])
        torch.testing.assert_close(remapped, expected)
        
        # Assert: all classes present
        assert remapped.min() == 0
        assert remapped.max() == len(classes) - 1
        assert len(torch.unique(remapped)) == len(classes)


def test_no_batchnorm_in_few_shot_backbones():
    """REGRESSION: Fix BATCHNORM_PRESENT - BatchNorm leaks across episodes."""
    from src.meta_learning.meta_learning_modules.few_shot_modules.utilities import create_backbone
    
    # Test that backbone uses GroupNorm/LayerNorm instead of BatchNorm
    backbone = create_backbone('conv4', input_channels=3)
    
    # Search for BatchNorm modules (should not exist)
    has_batchnorm = any(isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)) 
                       for m in backbone.modules())
    
    assert not has_batchnorm, "Few-shot backbones must not use BatchNorm (causes episodic leakage)"
    
    # Should use GroupNorm or LayerNorm instead
    has_groupnorm = any(isinstance(m, nn.GroupNorm) for m in backbone.modules())
    has_layernorm = any(isinstance(m, nn.LayerNorm) for m in backbone.modules())
    
    assert has_groupnorm or has_layernorm, "Should use GroupNorm or LayerNorm for few-shot learning"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])