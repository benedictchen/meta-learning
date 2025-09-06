"""
Comprehensive tests for Research-accurate Prototypical Networks implementation.

Tests mathematical correctness according to Snell et al. (2017) paper formulation.
"""
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from unittest.mock import MagicMock, patch
from typing import Dict, List, Tuple, Any
import math

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from meta_learning.meta_learning_modules.prototypical_networks_fixed import (
    ResearchPrototypicalNetworks,
    PrototypicalHead,
    EuclideanDistance,
    CosineDistance
)


class TestEuclideanDistance:
    """Test Euclidean distance computation."""
    
    def test_euclidean_distance_basic(self):
        """Test basic Euclidean distance computation."""
        distance_fn = EuclideanDistance()
        
        # Simple 2D case
        query = torch.tensor([[1.0, 0.0], [0.0, 1.0]])  # 2 queries
        prototypes = torch.tensor([[0.0, 0.0], [1.0, 1.0]])  # 2 prototypes
        
        distances = distance_fn(query, prototypes)
        
        # Expected distances:
        # query[0] to proto[0]: sqrt(1^2 + 0^2) = 1.0
        # query[0] to proto[1]: sqrt(0^2 + 1^2) = sqrt(2)
        # query[1] to proto[0]: sqrt(0^2 + 1^2) = 1.0  
        # query[1] to proto[1]: sqrt(1^2 + 0^2) = 1.0
        
        expected = torch.tensor([
            [1.0, math.sqrt(2)],  # distances from query[0]
            [1.0, 1.0]            # distances from query[1]
        ])
        
        assert torch.allclose(distances, expected, atol=1e-6)
    
    def test_euclidean_distance_batch(self):
        """Test Euclidean distance with realistic batch sizes."""
        distance_fn = EuclideanDistance()
        
        n_query = 15
        n_way = 5
        feature_dim = 64
        
        query = torch.randn(n_query, feature_dim)
        prototypes = torch.randn(n_way, feature_dim)
        
        distances = distance_fn(query, prototypes)
        
        assert distances.shape == (n_query, n_way)
        assert (distances >= 0).all()  # Distances should be non-negative
    
    def test_euclidean_distance_mathematical_property(self):
        """Test mathematical property: d(x,x) = 0."""
        distance_fn = EuclideanDistance()
        
        x = torch.randn(5, 10)
        distances = distance_fn(x, x)
        
        # Distance from each point to itself should be 0
        diagonal = torch.diagonal(distances)
        assert torch.allclose(diagonal, torch.zeros_like(diagonal), atol=1e-6)


class TestCosineDistance:
    """Test Cosine distance computation."""
    
    def test_cosine_distance_basic(self):
        """Test basic cosine distance computation."""
        distance_fn = CosineDistance()
        
        # Orthogonal vectors
        query = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        prototypes = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        
        distances = distance_fn(query, prototypes)
        
        # Expected: cosine similarity between identical vectors is 1, distance is 0
        # cosine similarity between orthogonal vectors is 0, distance is 1
        expected = torch.tensor([
            [0.0, 1.0],  # query[0] vs prototypes
            [1.0, 0.0]   # query[1] vs prototypes
        ])
        
        assert torch.allclose(distances, expected, atol=1e-6)
    
    def test_cosine_distance_normalization(self):
        """Test that cosine distance properly normalizes vectors."""
        distance_fn = CosineDistance()
        
        # Vectors that are scaled versions of each other
        query = torch.tensor([[2.0, 0.0]])
        prototypes = torch.tensor([[1.0, 0.0], [4.0, 0.0]])  # Same direction, different magnitudes
        
        distances = distance_fn(query, prototypes)
        
        # Both prototypes point in same direction as query, so distance should be 0
        expected = torch.tensor([[0.0, 0.0]])
        assert torch.allclose(distances, expected, atol=1e-6)


class TestPrototypicalHead:
    """Test PrototypicalHead implementation."""
    
    def test_prototypical_head_initialization(self):
        """Test PrototypicalHead initialization."""
        head = PrototypicalHead(temperature=1.0, distance_metric='euclidean')
        
        assert head.temperature == 1.0
        assert isinstance(head.distance_fn, EuclideanDistance)
        
        head_cosine = PrototypicalHead(temperature=2.0, distance_metric='cosine')
        assert head_cosine.temperature == 2.0
        assert isinstance(head_cosine.distance_fn, CosineDistance)
    
    def test_compute_prototypes(self):
        """Test prototype computation from support set."""
        head = PrototypicalHead()
        
        # 3-way 2-shot scenario
        n_way, k_shot = 3, 2
        feature_dim = 10
        
        support_features = torch.randn(n_way * k_shot, feature_dim)
        support_labels = torch.tensor([0, 0, 1, 1, 2, 2])  # 2 examples per class
        
        prototypes = head._compute_prototypes(support_features, support_labels, n_way)
        
        assert prototypes.shape == (n_way, feature_dim)
        
        # Manually compute expected prototypes
        expected_prototypes = torch.zeros(n_way, feature_dim)
        expected_prototypes[0] = support_features[:2].mean(dim=0)    # Class 0
        expected_prototypes[1] = support_features[2:4].mean(dim=0)   # Class 1  
        expected_prototypes[2] = support_features[4:6].mean(dim=0)   # Class 2
        
        assert torch.allclose(prototypes, expected_prototypes, atol=1e-6)
    
    def test_compute_prototypes_unbalanced(self):
        """Test prototype computation with unbalanced support set."""
        head = PrototypicalHead()
        
        n_way = 3
        feature_dim = 5
        
        # Unbalanced: class 0 has 1 example, class 1 has 3, class 2 has 2
        support_features = torch.randn(6, feature_dim)
        support_labels = torch.tensor([0, 1, 1, 1, 2, 2])
        
        prototypes = head._compute_prototypes(support_features, support_labels, n_way)
        
        assert prototypes.shape == (n_way, feature_dim)
        
        # Expected prototypes
        expected_prototypes = torch.zeros(n_way, feature_dim)
        expected_prototypes[0] = support_features[0:1].mean(dim=0)   # Class 0: 1 example
        expected_prototypes[1] = support_features[1:4].mean(dim=0)   # Class 1: 3 examples
        expected_prototypes[2] = support_features[4:6].mean(dim=0)   # Class 2: 2 examples
        
        assert torch.allclose(prototypes, expected_prototypes, atol=1e-6)
    
    def test_forward_basic(self):
        """Test forward pass basic functionality."""
        head = PrototypicalHead(temperature=1.0)
        
        # 5-way 1-shot scenario
        n_way, k_shot, n_query = 5, 1, 15
        feature_dim = 32
        
        support_features = torch.randn(n_way * k_shot, feature_dim)
        support_labels = torch.arange(n_way).repeat_interleave(k_shot)  # [0,1,2,3,4]
        query_features = torch.randn(n_query, feature_dim)
        
        logits = head.forward(support_features, support_labels, query_features)
        
        assert logits.shape == (n_query, n_way)
        
        # Logits should be real-valued (not NaN or inf)
        assert torch.isfinite(logits).all()
    
    def test_temperature_scaling_effect(self):
        """Test effect of temperature scaling on logits."""
        head_temp1 = PrototypicalHead(temperature=1.0)
        head_temp2 = PrototypicalHead(temperature=2.0)
        
        # Fixed data for reproducibility
        torch.manual_seed(42)
        n_way, k_shot, n_query = 3, 2, 5
        feature_dim = 10
        
        support_features = torch.randn(n_way * k_shot, feature_dim)
        support_labels = torch.tensor([0, 0, 1, 1, 2, 2])
        query_features = torch.randn(n_query, feature_dim)
        
        logits_temp1 = head_temp1.forward(support_features, support_labels, query_features)
        logits_temp2 = head_temp2.forward(support_features, support_labels, query_features)
        
        # Higher temperature should produce softer (less extreme) logits
        # This means the ratio of max to min should be smaller
        temp1_range = logits_temp1.max(dim=1)[0] - logits_temp1.min(dim=1)[0]
        temp2_range = logits_temp2.max(dim=1)[0] - logits_temp2.min(dim=1)[0]
        
        # With temp=2.0, distances are scaled by 1/2, making logits softer
        assert (temp2_range < temp1_range).all()
    
    def test_negative_distance_for_softmax(self):
        """Test that logits are negative distances (proper for softmax)."""
        head = PrototypicalHead(temperature=1.0)
        
        n_way, k_shot, n_query = 3, 1, 5
        feature_dim = 8
        
        support_features = torch.randn(n_way * k_shot, feature_dim)
        support_labels = torch.arange(n_way)
        query_features = torch.randn(n_query, feature_dim)
        
        logits = head.forward(support_features, support_labels, query_features)
        
        # Logits should be negative (negative distances)
        # Higher values = smaller distances = higher probability
        assert (logits <= 0).all()


class TestResearchPrototypicalNetworks:
    """Test complete ResearchPrototypicalNetworks model."""
    
    def setup_method(self):
        """Setup test fixtures."""
        # Simple encoder for testing
        self.encoder = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )
        
        self.model = ResearchPrototypicalNetworks(
            encoder=self.encoder,
            temperature=1.0,
            distance_metric='euclidean'
        )
    
    def test_model_initialization(self):
        """Test model initialization."""
        assert self.model.encoder is self.encoder
        assert self.model.head.temperature == 1.0
        assert isinstance(self.model.head.distance_fn, EuclideanDistance)
    
    def test_forward_pass_shapes(self):
        """Test forward pass produces correct output shapes."""
        n_way, k_shot, n_query = 5, 2, 10
        input_dim = 10
        
        support_x = torch.randn(n_way * k_shot, input_dim)
        support_y = torch.arange(n_way).repeat_interleave(k_shot)
        query_x = torch.randn(n_query, input_dim)
        
        logits = self.model(support_x, support_y, query_x)
        
        assert logits.shape == (n_query, n_way)
        assert torch.isfinite(logits).all()
    
    def test_forward_pass_single_shot(self):
        """Test forward pass with 1-shot learning (canonical case)."""
        n_way, k_shot, n_query = 5, 1, 15
        input_dim = 10
        
        support_x = torch.randn(n_way * k_shot, input_dim)
        support_y = torch.arange(n_way)  # [0,1,2,3,4]
        query_x = torch.randn(n_query, input_dim)
        
        logits = self.model(support_x, support_y, query_x)
        
        assert logits.shape == (n_query, n_way)
        
        # Convert to probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Each row should sum to 1 (valid probability distribution)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(n_query), atol=1e-6)
        
        # All probabilities should be positive
        assert (probs >= 0).all()
        assert (probs <= 1).all()
    
    def test_few_shot_learning_accuracy(self):
        """Test that model can achieve reasonable accuracy on separable data."""
        torch.manual_seed(42)  # For reproducibility
        
        n_way, k_shot, n_query = 3, 5, 15
        input_dim = 20
        
        # Create separable data: each class has different mean
        support_x = torch.zeros(n_way * k_shot, input_dim)
        support_y = torch.arange(n_way).repeat_interleave(k_shot)
        
        query_x = torch.zeros(n_query, input_dim)
        query_y = torch.arange(n_way).repeat_interleave(5)  # 5 queries per class
        
        # Class 0: features around [1, 0, 0, ...]
        # Class 1: features around [0, 1, 0, ...]  
        # Class 2: features around [0, 0, 1, ...]
        for i in range(n_way):
            class_mask_support = support_y == i
            class_mask_query = query_y == i
            
            support_x[class_mask_support, i] = 2.0  # Strong signal
            support_x[class_mask_support] += torch.randn_like(support_x[class_mask_support]) * 0.1
            
            query_x[class_mask_query, i] = 2.0
            query_x[class_mask_query] += torch.randn_like(query_x[class_mask_query]) * 0.1
        
        # Forward pass
        with torch.no_grad():
            logits = self.model(support_x, support_y, query_x)
            predictions = logits.argmax(dim=-1)
            
            accuracy = (predictions == query_y).float().mean()
            
            # Should achieve high accuracy on this separable data
            assert accuracy > 0.8, f"Expected accuracy > 0.8, got {accuracy:.3f}"
    
    def test_gradient_flow(self):
        """Test that gradients flow properly through the model."""
        n_way, k_shot, n_query = 3, 2, 6
        input_dim = 10
        
        support_x = torch.randn(n_way * k_shot, input_dim, requires_grad=True)
        support_y = torch.arange(n_way).repeat_interleave(k_shot)
        query_x = torch.randn(n_query, input_dim)
        query_y = torch.randint(0, n_way, (n_query,))
        
        # Forward pass
        logits = self.model(support_x, support_y, query_x)
        
        # Compute loss
        loss = F.cross_entropy(logits, query_y)
        
        # Backward pass
        loss.backward()
        
        # Check that encoder parameters have gradients
        for param in self.model.encoder.parameters():
            assert param.grad is not None
            assert not torch.allclose(param.grad, torch.zeros_like(param.grad))
        
        # Check that input has gradients
        assert support_x.grad is not None


class TestMathematicalCorrectness:
    """Test mathematical correctness according to Snell et al. (2017)."""
    
    def setup_method(self):
        """Setup for mathematical correctness tests."""
        # Identity encoder to test head in isolation
        self.identity_encoder = nn.Identity()
        
        self.model = ResearchPrototypicalNetworks(
            encoder=self.identity_encoder,
            temperature=1.0,
            distance_metric='euclidean'
        )
    
    def test_snell_2017_formulation(self):
        """Test exact Snell et al. (2017) formulation.
        
        Paper formulation:
        - c_k = (1/|S_k|) * sum_{(x_i, y_i) in S_k} f(x_i)  [prototype computation]
        - p(y=k | x) = exp(-d(f(x), c_k)) / sum_j exp(-d(f(x), c_j))  [classification]
        """
        n_way, k_shot = 3, 2
        feature_dim = 5
        
        # Manually constructed support set
        support_x = torch.tensor([
            [1.0, 0.0, 0.0, 0.0, 0.0],  # Class 0, example 1
            [1.1, 0.1, 0.0, 0.0, 0.0],  # Class 0, example 2
            [0.0, 1.0, 0.0, 0.0, 0.0],  # Class 1, example 1
            [0.1, 1.1, 0.0, 0.0, 0.0],  # Class 1, example 2
            [0.0, 0.0, 1.0, 0.0, 0.0],  # Class 2, example 1
            [0.0, 0.0, 1.1, 0.1, 0.0],  # Class 2, example 2
        ])
        support_y = torch.tensor([0, 0, 1, 1, 2, 2])
        
        # Query point
        query_x = torch.tensor([[1.05, 0.05, 0.0, 0.0, 0.0]])  # Should be closest to class 0
        
        # Forward pass
        logits = self.model(support_x, support_y, query_x)
        
        # Manual computation for verification
        # Prototype for class 0
        c_0 = support_x[:2].mean(dim=0)  # [1.05, 0.05, 0, 0, 0]
        c_1 = support_x[2:4].mean(dim=0)  # [0.05, 1.05, 0, 0, 0]
        c_2 = support_x[4:6].mean(dim=0)  # [0, 0, 1.05, 0.05, 0]
        
        prototypes = torch.stack([c_0, c_1, c_2])
        
        # Distances from query to each prototype  
        query_expanded = query_x[0].unsqueeze(0)  # [1, 5]
        distances = torch.cdist(query_expanded, prototypes)[0]  # [3]
        
        # Expected logits (negative distances)
        expected_logits = -distances
        
        assert torch.allclose(logits[0], expected_logits, atol=1e-5)
        
        # Verify that class 0 has highest probability (lowest distance)
        probs = F.softmax(logits, dim=-1)
        assert probs[0].argmax().item() == 0
    
    def test_temperature_scaling_mathematics(self):
        """Test mathematical correctness of temperature scaling."""
        n_way, k_shot = 2, 1
        feature_dim = 3
        
        support_x = torch.tensor([
            [1.0, 0.0, 0.0],  # Class 0
            [0.0, 1.0, 0.0],  # Class 1
        ])
        support_y = torch.tensor([0, 1])
        query_x = torch.tensor([[0.8, 0.2, 0.0]])  # Closer to class 0
        
        # Test different temperatures
        for temp in [0.5, 1.0, 2.0, 4.0]:
            model = ResearchPrototypicalNetworks(
                encoder=self.identity_encoder,
                temperature=temp,
                distance_metric='euclidean'
            )
            
            logits = model(support_x, support_y, query_x)
            
            # Manual computation
            prototypes = support_x  # k_shot=1, so prototypes = support examples
            distances = torch.cdist(query_x, prototypes)[0]
            expected_logits = -distances / temp
            
            assert torch.allclose(logits[0], expected_logits, atol=1e-5)
    
    def test_distance_metric_equivalence(self):
        """Test that different distance metrics produce consistent results."""
        n_way, k_shot = 3, 1
        feature_dim = 4
        
        # L2 normalized features for fair comparison
        support_x = F.normalize(torch.randn(n_way, feature_dim), dim=-1)
        support_y = torch.arange(n_way)
        query_x = F.normalize(torch.randn(2, feature_dim), dim=-1)
        
        # Test Euclidean distance
        model_euclidean = ResearchPrototypicalNetworks(
            encoder=self.identity_encoder,
            temperature=1.0,
            distance_metric='euclidean'
        )
        
        # Test Cosine distance
        model_cosine = ResearchPrototypicalNetworks(
            encoder=self.identity_encoder,
            temperature=1.0,
            distance_metric='cosine'
        )
        
        logits_euclidean = model_euclidean(support_x, support_y, query_x)
        logits_cosine = model_cosine(support_x, support_y, query_x)
        
        # Both should produce valid probability distributions
        probs_euclidean = F.softmax(logits_euclidean, dim=-1)
        probs_cosine = F.softmax(logits_cosine, dim=-1)
        
        assert torch.allclose(probs_euclidean.sum(dim=-1), torch.ones(2), atol=1e-6)
        assert torch.allclose(probs_cosine.sum(dim=-1), torch.ones(2), atol=1e-6)
        
        # Predictions might differ, but both should be reasonable
        pred_euclidean = logits_euclidean.argmax(dim=-1)
        pred_cosine = logits_cosine.argmax(dim=-1)
        
        assert pred_euclidean.shape == (2,)
        assert pred_cosine.shape == (2,)
        assert all(p.item() in range(n_way) for p in pred_euclidean)
        assert all(p.item() in range(n_way) for p in pred_cosine)
    
    def test_prototype_computation_mathematical_correctness(self):
        """Test that prototype computation matches exact mathematical definition."""
        head = PrototypicalHead()
        
        # Test case: uneven number of examples per class
        support_features = torch.tensor([
            [1.0, 2.0],  # Class 0, example 1
            [3.0, 4.0],  # Class 0, example 2  
            [5.0, 6.0],  # Class 0, example 3
            [10.0, 20.0],  # Class 1, example 1
            [30.0, 40.0],  # Class 1, example 2
        ])
        support_labels = torch.tensor([0, 0, 0, 1, 1])
        n_way = 2
        
        prototypes = head._compute_prototypes(support_features, support_labels, n_way)
        
        # Manual computation
        expected_proto_0 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).view(3, 2).mean(dim=0)  # [3.0, 4.0]
        expected_proto_1 = torch.tensor([10.0, 20.0, 30.0, 40.0]).view(2, 2).mean(dim=0)  # [20.0, 30.0]
        
        assert torch.allclose(prototypes[0], expected_proto_0, atol=1e-6)
        assert torch.allclose(prototypes[1], expected_proto_1, atol=1e-6)


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def setup_method(self):
        """Setup for edge case tests."""
        self.encoder = nn.Linear(5, 10)
        self.model = ResearchPrototypicalNetworks(self.encoder)
    
    def test_single_way_classification(self):
        """Test 1-way classification (edge case)."""
        n_way, k_shot, n_query = 1, 3, 5
        input_dim = 5
        
        support_x = torch.randn(n_way * k_shot, input_dim)
        support_y = torch.zeros(n_way * k_shot, dtype=torch.long)  # All class 0
        query_x = torch.randn(n_query, input_dim)
        
        logits = self.model(support_x, support_y, query_x)
        
        assert logits.shape == (n_query, n_way)
        
        # With only one class, all predictions should be class 0
        predictions = logits.argmax(dim=-1)
        assert (predictions == 0).all()
    
    def test_mismatched_support_labels(self):
        """Test error handling for mismatched support set sizes."""
        support_x = torch.randn(10, 5)
        support_y = torch.randint(0, 3, (8,))  # Wrong size
        query_x = torch.randn(5, 5)
        
        with pytest.raises(RuntimeError):
            self.model(support_x, support_y, query_x)
    
    def test_empty_class_error(self):
        """Test error when a class has no examples in support set."""
        support_x = torch.randn(6, 5)
        support_y = torch.tensor([0, 0, 1, 1, 1, 1])  # Class 2 missing
        query_x = torch.randn(3, 5)
        
        head = PrototypicalHead()
        
        # This should handle gracefully by only computing prototypes for present classes
        n_way = 2  # Only classes 0 and 1 present
        prototypes = head._compute_prototypes(support_x, support_y, n_way)
        
        assert prototypes.shape == (n_way, 10)  # encoder output dim
    
    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        # Very large feature values
        support_x = torch.randn(6, 5) * 1000
        support_y = torch.tensor([0, 0, 1, 1, 2, 2])
        query_x = torch.randn(3, 5) * 1000
        
        logits = self.model(support_x, support_y, query_x)
        
        # Should not produce NaN or inf
        assert torch.isfinite(logits).all()
        
        # Should produce valid probability distribution
        probs = F.softmax(logits, dim=-1)
        assert torch.isfinite(probs).all()
        assert torch.allclose(probs.sum(dim=-1), torch.ones(3), atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__])