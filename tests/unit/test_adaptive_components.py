"""
Tests for TaskAdaptivePrototypes implementations
===============================================

Tests all 4 task-adaptive prototype methods with proper configurations.
"""

import pytest
import torch
# DEMO/TEST REPRODUCIBILITY: Set seed for consistent synthetic data
torch.manual_seed(42)  # Reproducible test/demo data
import torch.nn as nn
# DEMO/TEST REPRODUCIBILITY: Set seed for consistent synthetic data
torch.manual_seed(42)  # Reproducible test/demo data
from meta_learning.meta_learning_modules.few_shot_modules.adaptive_components import (
    TaskAdaptivePrototypes, TaskAdaptiveConfig
)


class TestTaskAdaptivePrototypes:
    """Test suite for task-adaptive prototype methods."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.embedding_dim = 64
        self.n_classes = 5
        self.n_support_per_class = 5
        self.n_query = 15
        
        # Create test data
        self.support_embeddings = torch.randn(self.n_classes * self.n_support_per_class, self.embedding_dim)
        self.support_labels = torch.repeat_interleave(torch.arange(self.n_classes), self.n_support_per_class)
        self.query_embeddings = torch.randn(self.n_query, self.embedding_dim)
        
    def test_attention_based_method(self):
        """Test attention-based task adaptation."""
        config = TaskAdaptiveConfig(
            method="attention_based",
            attention_heads=8,
            attention_temperature=1.0
        )
        
        adaptive = TaskAdaptivePrototypes(self.embedding_dim, config)
        
        adapted_prototypes = adaptive(
            self.support_embeddings, 
            self.support_labels, 
            self.query_embeddings
        )
        
        # Should return adapted prototypes for each query
        assert adapted_prototypes.shape == (self.n_query, self.n_classes, self.embedding_dim)
        
        # Check that prototypes are finite
        assert torch.all(torch.isfinite(adapted_prototypes))
        
    def test_meta_learning_method(self):
        """Test meta-learning based task adaptation."""
        config = TaskAdaptiveConfig(
            method="meta_learning",
            meta_lr=0.01,
            meta_steps=3
        )
        
        adaptive = TaskAdaptivePrototypes(self.embedding_dim, config)
        
        adapted_prototypes = adaptive(
            self.support_embeddings, 
            self.support_labels, 
            self.query_embeddings
        )
        
        # Should return adapted prototypes
        assert adapted_prototypes.shape == (self.n_query, self.n_classes, self.embedding_dim)
        assert torch.all(torch.isfinite(adapted_prototypes))
        
    def test_context_dependent_method(self):
        """Test context-dependent task adaptation."""
        config = TaskAdaptiveConfig(
            method="context_dependent",
            context_window_size=3,
            context_aggregation="mean"
        )
        
        adaptive = TaskAdaptivePrototypes(self.embedding_dim, config)
        
        adapted_prototypes = adaptive(
            self.support_embeddings, 
            self.support_labels, 
            self.query_embeddings
        )
        
        # Should return context-adapted prototypes
        assert adapted_prototypes.shape == (self.n_query, self.n_classes, self.embedding_dim)
        assert torch.all(torch.isfinite(adapted_prototypes))
        
    def test_transformer_based_method(self):
        """Test transformer-based task adaptation."""
        config = TaskAdaptiveConfig(
            method="transformer_based",
            transformer_layers=2,
            transformer_heads=4,
            hidden_dim=128
        )
        
        adaptive = TaskAdaptivePrototypes(self.embedding_dim, config)
        
        adapted_prototypes = adaptive(
            self.support_embeddings, 
            self.support_labels, 
            self.query_embeddings
        )
        
        # Should return transformer-adapted prototypes
        assert adapted_prototypes.shape == (self.n_query, self.n_classes, self.embedding_dim)
        assert torch.all(torch.isfinite(adapted_prototypes))
        
    def test_invalid_method(self):
        """Test error handling for invalid methods."""
        config = TaskAdaptiveConfig(method="invalid_method")
        
        adaptive = TaskAdaptivePrototypes(self.embedding_dim, config)
        
        with pytest.raises(ValueError, match="Unknown task-adaptive method"):
            adaptive(self.support_embeddings, self.support_labels, self.query_embeddings)
            
    def test_gradient_flow(self):
        """Test that gradients flow through adaptive computations."""
        config = TaskAdaptiveConfig(method="attention_based", attention_heads=4)
        
        adaptive = TaskAdaptivePrototypes(self.embedding_dim, config)
        
        # Enable gradients
        support_embeddings = self.support_embeddings.requires_grad_(True)
        query_embeddings = self.query_embeddings.requires_grad_(True)
        
        adapted_prototypes = adaptive(support_embeddings, self.support_labels, query_embeddings)
        
        # Compute a simple loss
        loss = adapted_prototypes.mean()
        loss.backward()
        
        # Check gradients exist
        assert support_embeddings.grad is not None
        assert query_embeddings.grad is not None
        
    def test_config_defaults(self):
        """Test default configuration values."""
        config = TaskAdaptiveConfig()
        
        assert config.method == "attention_based"
        assert config.attention_heads == 8
        assert config.hidden_dim == 512
        
    def test_single_query_handling(self):
        """Test handling of single query case."""
        config = TaskAdaptiveConfig(method="attention_based")
        
        adaptive = TaskAdaptivePrototypes(self.embedding_dim, config)
        
        # Single query
        single_query = torch.randn(1, self.embedding_dim)
        
        adapted_prototypes = adaptive(
            self.support_embeddings, 
            self.support_labels, 
            single_query
        )
        
        assert adapted_prototypes.shape == (1, self.n_classes, self.embedding_dim)
        
    def test_many_queries_handling(self):
        """Test handling of many queries."""
        config = TaskAdaptiveConfig(method="attention_based")
        
        adaptive = TaskAdaptivePrototypes(self.embedding_dim, config)
        
        # Many queries
        many_queries = torch.randn(100, self.embedding_dim)
        
        adapted_prototypes = adaptive(
            self.support_embeddings, 
            self.support_labels, 
            many_queries
        )
        
        assert adapted_prototypes.shape == (100, self.n_classes, self.embedding_dim)
        
    def test_adaptation_consistency(self):
        """Test that adaptation is consistent across runs."""
        config = TaskAdaptiveConfig(method="context_dependent", context_window_size=2)
        
        adaptive = TaskAdaptivePrototypes(self.embedding_dim, config)
        
        # Run twice with same inputs
        adapted1 = adaptive(self.support_embeddings, self.support_labels, self.query_embeddings)
        adapted2 = adaptive(self.support_embeddings, self.support_labels, self.query_embeddings)
        
        # Should be identical (deterministic)
        torch.testing.assert_close(adapted1, adapted2, atol=1e-6, rtol=1e-6)


class TestTaskAdaptiveConfig:
    """Test task-adaptive configuration class."""
    
    def test_config_creation(self):
        """Test configuration object creation."""
        config = TaskAdaptiveConfig(
            method="transformer_based",
            transformer_layers=4,
            transformer_heads=8,
            hidden_dim=256
        )
        
        assert config.method == "transformer_based"
        assert config.transformer_layers == 4
        assert config.transformer_heads == 8
        assert config.hidden_dim == 256
        
    def test_config_validation(self):
        """Test configuration validation."""
        # Valid configurations should not raise errors
        valid_methods = ["attention_based", "meta_learning", "context_dependent", "transformer_based"]
        
        for method in valid_methods:
            config = TaskAdaptiveConfig(method=method)
            assert config.method == method


class TestTaskAdaptiveIntegration:
    """Integration tests for task-adaptive components."""
    
    def test_adaptive_with_different_tasks(self):
        """Test adaptation works with different task characteristics."""
        config = TaskAdaptiveConfig(method="attention_based")
        adaptive = TaskAdaptivePrototypes(64, config)
        
        # Test with different numbers of classes
        for n_classes in [2, 5, 10]:
            support_embeddings = torch.randn(n_classes * 3, 64)
            support_labels = torch.repeat_interleave(torch.arange(n_classes), 3)
            query_embeddings = torch.randn(5, 64)
            
            adapted_prototypes = adaptive(support_embeddings, support_labels, query_embeddings)
            
            assert adapted_prototypes.shape == (5, n_classes, 64)
            assert torch.all(torch.isfinite(adapted_prototypes))
            
    def test_adaptive_prototype_diversity(self):
        """Test that adapted prototypes are different for different queries."""
        config = TaskAdaptiveConfig(method="attention_based", attention_heads=4)
        adaptive = TaskAdaptivePrototypes(32, config)
        
        support_embeddings = torch.randn(15, 32)
        support_labels = torch.repeat_interleave(torch.arange(3), 5)
        
        # Two very different queries
        query1 = torch.randn(1, 32)
        query2 = -query1  # Opposite query
        
        adapted1 = adaptive(support_embeddings, support_labels, query1)
        adapted2 = adaptive(support_embeddings, support_labels, query2)
        
        # Adapted prototypes should be different
        diff = torch.abs(adapted1 - adapted2).mean()
        assert diff > 1e-3, "Adapted prototypes are too similar for different queries"
        
    def test_adaptive_computational_efficiency(self):
        """Test computational efficiency of adaptive methods."""
        # Test with efficient method
        config = TaskAdaptiveConfig(method="context_dependent", context_window_size=2)
        adaptive = TaskAdaptivePrototypes(128, config)
        
        # Large task
        large_support = torch.randn(500, 128)  # 50 classes, 10 examples each
        large_labels = torch.repeat_interleave(torch.arange(50), 10)
        large_queries = torch.randn(100, 128)
        
        # Should complete without issues
        adapted_prototypes = adaptive(large_support, large_labels, large_queries)
        
        assert adapted_prototypes.shape == (100, 50, 128)
        assert torch.all(torch.isfinite(adapted_prototypes))
        
    def test_adaptive_methods_comparison(self):
        """Test that different adaptive methods produce different results."""
        methods = ["attention_based", "meta_learning", "context_dependent", "transformer_based"]
        
        support_embeddings = torch.randn(15, 64)
        support_labels = torch.repeat_interleave(torch.arange(3), 5)
        query_embeddings = torch.randn(5, 64)
        
        results = {}
        
        for method in methods:
            config = TaskAdaptiveConfig(method=method)
            adaptive = TaskAdaptivePrototypes(64, config)
            
            adapted = adaptive(support_embeddings, support_labels, query_embeddings)
            results[method] = adapted
            
        # All methods should produce different results
        method_pairs = [(m1, m2) for i, m1 in enumerate(methods) for m2 in methods[i+1:]]
        
        for m1, m2 in method_pairs:
            diff = torch.abs(results[m1] - results[m2]).mean()
            assert diff > 1e-4, f"Methods {m1} and {m2} produced too similar results"
            
    def test_adaptive_with_extreme_inputs(self):
        """Test adaptation with extreme input values."""
        config = TaskAdaptiveConfig(method="attention_based")
        adaptive = TaskAdaptivePrototypes(32, config)
        
        # Test with very large values
        large_support = torch.randn(9, 32) * 100
        large_labels = torch.repeat_interleave(torch.arange(3), 3)
        large_queries = torch.randn(3, 32) * 100
        
        adapted = adaptive(large_support, large_labels, large_queries)
        
        assert torch.all(torch.isfinite(adapted)), "Adaptation failed with large inputs"
        
        # Test with very small values
        small_support = torch.randn(9, 32) * 0.001
        small_queries = torch.randn(3, 32) * 0.001
        
        adapted = adaptive(small_support, large_labels, small_queries)
        
        assert torch.all(torch.isfinite(adapted)), "Adaptation failed with small inputs"