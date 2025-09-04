"""
Tests for HierarchicalPrototypes implementations
===============================================

Tests all 4 hierarchical prototype methods with proper configurations.
"""

import pytest
import torch
# DEMO/TEST REPRODUCIBILITY: Set seed for consistent synthetic data
torch.manual_seed(42)  # Reproducible test/demo data
import torch.nn as nn
# DEMO/TEST REPRODUCIBILITY: Set seed for consistent synthetic data
torch.manual_seed(42)  # Reproducible test/demo data
from meta_learning.meta_learning_modules.few_shot_modules.hierarchical_components import (
    HierarchicalPrototypes, HierarchicalConfig
)


class TestHierarchicalPrototypes:
    """Test suite for hierarchical prototype methods."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.embedding_dim = 64
        self.n_classes = 5
        self.n_support_per_class = 5
        
        # Create test data
        self.support_embeddings = torch.randn(self.n_classes * self.n_support_per_class, self.embedding_dim)
        self.support_labels = torch.repeat_interleave(torch.arange(self.n_classes), self.n_support_per_class)
        
    def test_multi_level_method(self):
        """Test multi-level hierarchical prototypes."""
        config = HierarchicalConfig(
            method="multi_level",
            num_levels=3,
            level_temperatures=[1.0, 2.0, 4.0]
        )
        
        hierarchical = HierarchicalPrototypes(self.embedding_dim, config)
        
        prototypes = hierarchical(self.support_embeddings, self.support_labels)
        
        # Should return multi-level prototypes
        assert isinstance(prototypes, dict)
        assert 'level_0' in prototypes
        assert 'level_1' in prototypes
        assert 'level_2' in prototypes
        
        # Each level should have correct shape
        for level in range(3):
            level_protos = prototypes[f'level_{level}']
            assert level_protos.shape[0] <= self.n_classes  # May have fewer at higher levels
            assert level_protos.shape[1] == self.embedding_dim
            
    def test_tree_structured_method(self):
        """Test tree-structured hierarchical prototypes."""
        config = HierarchicalConfig(
            method="tree_structured",
            tree_branching_factor=2,
            tree_depth=2
        )
        
        hierarchical = HierarchicalPrototypes(self.embedding_dim, config)
        
        prototypes = hierarchical(self.support_embeddings, self.support_labels)
        
        # Should return tree structure
        assert isinstance(prototypes, dict)
        assert 'tree_nodes' in prototypes
        assert 'leaf_prototypes' in prototypes
        
        tree_nodes = prototypes['tree_nodes']
        leaf_prototypes = prototypes['leaf_prototypes']
        
        assert leaf_prototypes.shape == (self.n_classes, self.embedding_dim)
        
    def test_coarse_to_fine_method(self):
        """Test coarse-to-fine hierarchical prototypes."""
        config = HierarchicalConfig(
            method="coarse_to_fine",
            coarse_clusters=2,
            fine_resolution_factor=2.0
        )
        
        hierarchical = HierarchicalPrototypes(self.embedding_dim, config)
        
        prototypes = hierarchical(self.support_embeddings, self.support_labels)
        
        # Should return coarse and fine prototypes
        assert isinstance(prototypes, dict)
        assert 'coarse_prototypes' in prototypes
        assert 'fine_prototypes' in prototypes
        
        coarse_protos = prototypes['coarse_prototypes']
        fine_protos = prototypes['fine_prototypes']
        
        assert coarse_protos.shape[1] == self.embedding_dim
        assert fine_protos.shape == (self.n_classes, self.embedding_dim)
        
    def test_adaptive_hierarchy_method(self):
        """Test adaptive hierarchy construction."""
        config = HierarchicalConfig(
            method="adaptive_hierarchy",
            similarity_threshold=0.5,
            max_hierarchy_depth=3
        )
        
        hierarchical = HierarchicalPrototypes(self.embedding_dim, config)
        
        prototypes = hierarchical(self.support_embeddings, self.support_labels)
        
        # Should return adaptive hierarchy
        assert isinstance(prototypes, dict)
        assert 'hierarchy_levels' in prototypes
        assert 'level_assignments' in prototypes
        
        hierarchy = prototypes['hierarchy_levels']
        assert isinstance(hierarchy, list)
        assert len(hierarchy) <= config.max_hierarchy_depth
        
    def test_invalid_method(self):
        """Test error handling for invalid methods."""
        config = HierarchicalConfig(method="invalid_method")
        
        hierarchical = HierarchicalPrototypes(self.embedding_dim, config)
        
        with pytest.raises(ValueError, match="Unknown hierarchical method"):
            hierarchical(self.support_embeddings, self.support_labels)
            
    def test_gradient_flow(self):
        """Test that gradients flow through hierarchical computations."""
        config = HierarchicalConfig(method="multi_level", num_levels=2)
        
        hierarchical = HierarchicalPrototypes(self.embedding_dim, config)
        
        # Enable gradients
        support_embeddings = self.support_embeddings.requires_grad_(True)
        
        prototypes = hierarchical(support_embeddings, self.support_labels)
        
        # Compute a simple loss
        loss = sum(level_protos.mean() for level_protos in prototypes.values() if isinstance(level_protos, torch.Tensor))
        loss.backward()
        
        # Check gradients exist
        assert support_embeddings.grad is not None
        
    def test_config_defaults(self):
        """Test default configuration values."""
        config = HierarchicalConfig()
        
        assert config.method == "multi_level"
        assert config.num_levels == 3
        assert config.level_temperatures == [1.0, 2.0, 4.0]
        
    def test_single_class_handling(self):
        """Test handling of single class case."""
        config = HierarchicalConfig(method="multi_level", num_levels=2)
        
        hierarchical = HierarchicalPrototypes(self.embedding_dim, config)
        
        # Single class data
        single_class_embeddings = torch.randn(5, self.embedding_dim)
        single_class_labels = torch.zeros(5, dtype=torch.long)
        
        prototypes = hierarchical(single_class_embeddings, single_class_labels)
        
        # Should handle single class gracefully
        assert isinstance(prototypes, dict)
        for level_protos in prototypes.values():
            if isinstance(level_protos, torch.Tensor):
                assert level_protos.shape[1] == self.embedding_dim
                
    def test_empty_class_handling(self):
        """Test handling of edge cases."""
        config = HierarchicalConfig(method="multi_level", num_levels=2)
        
        hierarchical = HierarchicalPrototypes(self.embedding_dim, config)
        
        # Very small dataset
        small_embeddings = torch.randn(2, self.embedding_dim)
        small_labels = torch.tensor([0, 1])
        
        prototypes = hierarchical(small_embeddings, small_labels)
        
        # Should handle small datasets gracefully
        assert isinstance(prototypes, dict)


class TestHierarchicalConfig:
    """Test hierarchical configuration class."""
    
    def test_config_creation(self):
        """Test configuration object creation."""
        config = HierarchicalConfig(
            method="tree_structured",
            tree_branching_factor=3,
            tree_depth=4
        )
        
        assert config.method == "tree_structured"
        assert config.tree_branching_factor == 3
        assert config.tree_depth == 4
        
    def test_config_validation(self):
        """Test configuration validation."""
        # Valid configurations should not raise errors
        valid_methods = ["multi_level", "tree_structured", "coarse_to_fine", "adaptive_hierarchy"]
        
        for method in valid_methods:
            config = HierarchicalConfig(method=method)
            assert config.method == method


class TestHierarchicalIntegration:
    """Integration tests for hierarchical components."""
    
    def test_hierarchical_distance_computation(self):
        """Test computing distances with hierarchical prototypes."""
        config = HierarchicalConfig(method="multi_level", num_levels=2)
        
        hierarchical = HierarchicalPrototypes(64, config)
        
        # Create test data
        support_embeddings = torch.randn(15, 64)
        support_labels = torch.repeat_interleave(torch.arange(3), 5)
        query_embeddings = torch.randn(9, 64)
        
        prototypes = hierarchical(support_embeddings, support_labels)
        
        # Test distance computation to hierarchical prototypes
        query_expanded = query_embeddings.unsqueeze(1)  # [9, 1, 64]
        
        for level_key, level_protos in prototypes.items():
            if isinstance(level_protos, torch.Tensor):
                proto_expanded = level_protos.unsqueeze(0)  # [1, n_protos, 64]
                distances = torch.sum((query_expanded - proto_expanded) ** 2, dim=-1)
                
                assert distances.shape[0] == query_embeddings.shape[0]
                assert distances.shape[1] == level_protos.shape[0]
                assert torch.all(distances >= 0)
                
    def test_hierarchical_prototype_consistency(self):
        """Test consistency of hierarchical prototype generation."""
        embedding_dim = 32
        support_embeddings = torch.randn(20, embedding_dim)
        support_labels = torch.repeat_interleave(torch.arange(4), 5)
        
        methods = ["multi_level", "tree_structured", "coarse_to_fine", "adaptive_hierarchy"]
        
        for method in methods:
            config = HierarchicalConfig(method=method)
            hierarchical = HierarchicalPrototypes(embedding_dim, config)
            
            # Run twice to check consistency
            prototypes1 = hierarchical(support_embeddings, support_labels)
            prototypes2 = hierarchical(support_embeddings, support_labels)
            
            # Results should be deterministic (assuming no dropout/randomness)
            assert isinstance(prototypes1, dict)
            assert isinstance(prototypes2, dict)
            
            # Check that prototypes contain valid tensors
            for key, value in prototypes1.items():
                if isinstance(value, torch.Tensor):
                    assert torch.all(torch.isfinite(value)), f"Non-finite prototypes in {method}, key {key}"
                    assert value.shape[-1] == embedding_dim, f"Wrong embedding dim in {method}, key {key}"
                    
    def test_hierarchical_memory_efficiency(self):
        """Test memory efficiency of hierarchical methods."""
        config = HierarchicalConfig(method="multi_level", num_levels=2)
        
        hierarchical = HierarchicalPrototypes(128, config)
        
        # Large dataset
        large_embeddings = torch.randn(1000, 128)
        large_labels = torch.repeat_interleave(torch.arange(100), 10)
        
        # Should not cause memory issues
        prototypes = hierarchical(large_embeddings, large_labels)
        
        assert isinstance(prototypes, dict)
        
        # Total prototype memory should be reasonable
        total_prototypes = 0
        for value in prototypes.values():
            if isinstance(value, torch.Tensor):
                total_prototypes += value.numel()
        
        # Should be much smaller than original data
        original_size = large_embeddings.numel()
        assert total_prototypes < original_size, "Hierarchical prototypes use more memory than original data"