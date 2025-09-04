"""
Meta-Learning Integration Tests
================================

Integration tests that verify meta-learning components work together
and can be configured via the configuration system.
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock
from meta_learning.meta_learning_modules.config_factory import (
    create_all_fixme_solutions_config,
    create_comprehensive_component_config,
    create_specific_solution_config
)


class TestFixmeIntegration:
    """Integration tests for all research solutions working together."""
    
    def setup_method(self):
        """Set up integration test fixtures."""
        self.n_way = 5
        self.n_support = 5
        self.n_query = 15
        self.embedding_dim = 64
        
        # Create realistic test data
        self.support_images = torch.randn(self.n_way * self.n_support, 1, 28, 28)
        self.support_labels = torch.repeat_interleave(torch.arange(self.n_way), self.n_support)
        self.query_images = torch.randn(self.n_query, 1, 28, 28)
        self.query_labels = torch.repeat_interleave(torch.arange(self.n_way), 3)
        
    def test_comprehensive_config_instantiation(self):
        """Test that comprehensive configuration can be instantiated."""
        config = create_all_fixme_solutions_config()
        
        # Verify all components are configured
        assert config.test_time_compute is not None
        assert config.prototypical is not None
        assert config.uncertainty is not None
        assert config.hierarchical is not None
        assert config.task_adaptive is not None
        assert config.dataset_loading is not None
        assert config.task_difficulty is not None
        
        # Verify configuration values are sensible
        assert 0.0 < config.uncertainty.dropout_rate < 1.0
        assert config.hierarchical.num_levels > 0
        assert config.task_adaptive.attention_heads > 0
        
    def test_uncertainty_components_integration(self):
        """Test uncertainty components work with the configuration system."""
        from meta_learning.meta_learning_modules.few_shot_modules.uncertainty_components import UncertaintyAwareDistance
        
        config = create_comprehensive_component_config()
        
        # Test each uncertainty method
        methods = ["monte_carlo_dropout", "deep_ensemble", "evidential", "bayesian"]
        
        for method in methods:
            config.uncertainty.method = method
            
            uncertainty_dist = UncertaintyAwareDistance(self.embedding_dim, config.uncertainty)
            
            query_embeddings = torch.randn(self.n_query, self.embedding_dim)
            prototypes = torch.randn(self.n_way, self.embedding_dim)
            
            distances, uncertainties = uncertainty_dist(query_embeddings, prototypes)
            
            assert distances.shape == (self.n_query, self.n_way)
            assert uncertainties.shape == (self.n_query, self.n_way)
            assert torch.all(uncertainties >= 0)
            
    def test_hierarchical_components_integration(self):
        """Test hierarchical components work with the configuration system."""
        from meta_learning.meta_learning_modules.few_shot_modules.hierarchical_components import HierarchicalPrototypes
        
        config = create_comprehensive_component_config()
        
        # Test each hierarchical method
        methods = ["multi_level", "tree_structured", "coarse_to_fine", "adaptive_hierarchy"]
        
        for method in methods:
            config.hierarchical.method = method
            
            hierarchical = HierarchicalPrototypes(self.embedding_dim, config.hierarchical)
            
            support_embeddings = torch.randn(self.n_way * self.n_support, self.embedding_dim)
            support_labels = torch.repeat_interleave(torch.arange(self.n_way), self.n_support)
            
            prototypes = hierarchical(support_embeddings, support_labels)
            
            assert isinstance(prototypes, dict)
            assert len(prototypes) > 0
            
            # Check that all prototype tensors have correct embedding dimension
            for key, value in prototypes.items():
                if isinstance(value, torch.Tensor):
                    assert value.shape[-1] == self.embedding_dim
                    
    def test_adaptive_components_integration(self):
        """Test adaptive components work with the configuration system."""
        from meta_learning.meta_learning_modules.few_shot_modules.adaptive_components import TaskAdaptivePrototypes
        
        config = create_comprehensive_component_config()
        
        # Test each adaptive method
        methods = ["attention_based", "meta_learning", "context_dependent", "transformer_based"]
        
        for method in methods:
            config.task_adaptive.method = method
            
            adaptive = TaskAdaptivePrototypes(self.embedding_dim, config.task_adaptive)
            
            support_embeddings = torch.randn(self.n_way * self.n_support, self.embedding_dim)
            support_labels = torch.repeat_interleave(torch.arange(self.n_way), self.n_support)
            query_embeddings = torch.randn(self.n_query, self.embedding_dim)
            
            adapted_prototypes = adaptive(support_embeddings, support_labels, query_embeddings)
            
            assert adapted_prototypes.shape == (self.n_query, self.n_way, self.embedding_dim)
            assert torch.all(torch.isfinite(adapted_prototypes))
            
    def test_dataset_loading_integration(self):
        """Test dataset loading with configuration system."""
        from meta_learning.meta_learning_modules.few_shot_modules.utilities import sample_episode
        
        config = create_comprehensive_component_config()
        
        # Test synthetic data loading (safe for tests)
        config.dataset_loading.method = "synthetic"
        config.dataset_loading.require_user_confirmation_for_synthetic = False  # Skip confirmation for tests
        
        support_x, support_y, query_x, query_y = sample_episode(
            "omniglot", self.n_way, self.n_support, self.n_query, config.dataset_loading
        )
        
        assert support_x.shape[0] == self.n_way * self.n_support
        assert query_x.shape[0] == self.n_way * self.n_query
        assert support_y.max() < self.n_way
        assert query_y.max() < self.n_way
        
    def test_difficulty_estimation_integration(self):
        """Test difficulty estimation with configuration system."""
        from meta_learning.meta_learning_modules.utils_modules.statistical_evaluation import estimate_difficulty
        
        config = create_comprehensive_component_config()
        
        # Test different difficulty methods
        methods = ["intra_class_variance", "mdl_complexity", "entropy"]  # Skip sklearn-dependent methods
        
        task_data = torch.randn(50, 32)
        task_labels = torch.repeat_interleave(torch.arange(5), 10)
        
        for method in methods:
            config.task_difficulty.method = method
            
            difficulty = estimate_difficulty(task_data, method, task_labels, config.task_difficulty)
            
            assert 0.0 <= difficulty <= 1.0
            assert isinstance(difficulty, float)
            
    def test_test_time_compute_integration(self):
        """Test test-time compute with configuration system."""
        from meta_learning.meta_learning_modules.test_time_compute import TestTimeComputeScaler
        
        config = create_comprehensive_component_config()
        
        # Create a simple base model
        class SimpleModel(nn.Module):
            def forward(self, support_x, support_y, query_x):
                return torch.randn(len(query_x), len(torch.unique(support_y)))
                
        base_model = SimpleModel()
        
        # Configure test-time compute
        if config.test_time_compute is None:
            from meta_learning.meta_learning_modules.test_time_compute import TestTimeComputeConfig
            config.test_time_compute = TestTimeComputeConfig(compute_strategy="basic", max_compute_budget=5)
        
        scaler = TestTimeComputeScaler(base_model, config.test_time_compute)
        
        support_embeddings = torch.randn(self.n_way * self.n_support, self.embedding_dim)
        query_embeddings = torch.randn(self.n_query, self.embedding_dim)
        
        predictions, compute_info = scaler(support_embeddings, self.support_labels, query_embeddings)
        
        assert predictions.shape == (self.n_query, self.n_way)
        assert isinstance(compute_info, dict)
        assert 'compute_used' in compute_info
        
    def test_end_to_end_few_shot_episode(self):
        """Test complete few-shot episode with all components."""
        from meta_learning.meta_learning_modules.few_shot_modules.utilities import create_backbone_network, create_prototypical_network
        from meta_learning.meta_learning_modules.few_shot_modules.configurations import PrototypicalConfig
        
        config = create_all_fixme_solutions_config()
        
        # Create backbone network
        backbone = create_backbone_network("simple", input_channels=1, embedding_dim=self.embedding_dim)
        
        # Create prototypical network with advanced features
        proto_config = PrototypicalConfig()
        proto_config.use_uncertainty_aware_distances = True
        proto_config.use_hierarchical_prototypes = True
        proto_config.use_task_adaptive_prototypes = True
        
        proto_net = create_prototypical_network(backbone, "enhanced", proto_config)
        
        # Forward pass
        result = proto_net(self.support_images, self.support_labels, self.query_images)
        
        # Check results
        if isinstance(result, dict):
            assert 'logits' in result
            logits = result['logits']
        else:
            logits = result
            
        assert logits.shape == (self.n_query, self.n_way)
        assert torch.all(torch.isfinite(logits))
        
    def test_specific_solutions_selection(self):
        """Test selecting specific solutions via configuration."""
        specific_solutions = [
            "monte_carlo_dropout",
            "multi_level_prototypes", 
            "attention_based_adaptation",
            "torchmeta_loading",
            "intra_class_difficulty"
        ]
        
        config = create_specific_solution_config(specific_solutions)
        
        # Should have only specified components configured
        assert config.uncertainty is not None
        assert config.uncertainty.method == "monte_carlo_dropout"
        
        assert config.hierarchical is not None
        assert config.hierarchical.method == "multi_level"
        
        assert config.task_adaptive is not None
        assert config.task_adaptive.method == "attention_based"
        
        assert config.dataset_loading is not None
        assert config.dataset_loading.method == "torchmeta"
        
        assert config.task_difficulty is not None
        assert config.task_difficulty.method == "intra_class_variance"
        
    def test_configuration_consistency(self):
        """Test that configurations are internally consistent."""
        config = create_all_fixme_solutions_config()
        
        # Uncertainty configuration should be consistent
        if config.uncertainty is not None:
            assert config.uncertainty.method in ["monte_carlo_dropout", "deep_ensemble", "evidential", "bayesian"]
            if config.uncertainty.method == "monte_carlo_dropout":
                assert 0.0 < config.uncertainty.dropout_rate < 1.0
                assert config.uncertainty.num_samples > 0
                
        # Hierarchical configuration should be consistent
        if config.hierarchical is not None:
            assert config.hierarchical.method in ["multi_level", "tree_structured", "coarse_to_fine", "adaptive_hierarchy"]
            if config.hierarchical.method == "multi_level":
                assert config.hierarchical.num_levels > 0
                assert len(config.hierarchical.level_temperatures) == config.hierarchical.num_levels
                
        # Task adaptive configuration should be consistent
        if config.task_adaptive is not None:
            assert config.task_adaptive.method in ["attention_based", "meta_learning", "context_dependent", "transformer_based"]
            if config.task_adaptive.method == "attention_based":
                assert config.task_adaptive.attention_heads > 0
                
    def test_error_handling_integration(self):
        """Test error handling across all components."""
        config = create_comprehensive_component_config()
        
        # Test with invalid configurations
        with pytest.raises(ValueError):
            from meta_learning.meta_learning_modules.few_shot_modules.uncertainty_components import UncertaintyAwareDistance, UncertaintyConfig
            
            invalid_config = UncertaintyConfig(method="invalid_method")
            uncertainty_dist = UncertaintyAwareDistance(self.embedding_dim, invalid_config)
            
            query_embeddings = torch.randn(self.n_query, self.embedding_dim)
            prototypes = torch.randn(self.n_way, self.embedding_dim)
            uncertainty_dist(query_embeddings, prototypes)
            
    def test_memory_efficiency_integration(self):
        """Test memory efficiency with all components enabled."""
        config = create_all_fixme_solutions_config()
        
        # Reduce complexity for memory test
        config.uncertainty.num_samples = 3
        config.hierarchical.num_levels = 2
        config.task_adaptive.attention_heads = 2
        
        # Test with larger data
        large_support = torch.randn(50, self.embedding_dim)  # 10 classes, 5 examples each
        large_labels = torch.repeat_interleave(torch.arange(10), 5)
        large_query = torch.randn(30, self.embedding_dim)
        
        # Should handle larger data without memory issues
        from meta_learning.meta_learning_modules.few_shot_modules.uncertainty_components import UncertaintyAwareDistance
        from meta_learning.meta_learning_modules.few_shot_modules.hierarchical_components import HierarchicalPrototypes
        from meta_learning.meta_learning_modules.few_shot_modules.adaptive_components import TaskAdaptivePrototypes
        
        uncertainty_dist = UncertaintyAwareDistance(self.embedding_dim, config.uncertainty)
        hierarchical = HierarchicalPrototypes(self.embedding_dim, config.hierarchical)
        adaptive = TaskAdaptivePrototypes(self.embedding_dim, config.task_adaptive)
        
        # Test uncertainty
        prototypes = torch.randn(10, self.embedding_dim)
        distances, uncertainties = uncertainty_dist(large_query, prototypes)
        assert distances.shape == (30, 10)
        
        # Test hierarchical
        hierarchical_protos = hierarchical(large_support, large_labels)
        assert isinstance(hierarchical_protos, dict)
        
        # Test adaptive
        adapted_protos = adaptive(large_support, large_labels, large_query)
        assert adapted_protos.shape == (30, 10, self.embedding_dim)
        
    def test_reproducibility_integration(self):
        """Test reproducibility with fixed seeds."""
        config = create_comprehensive_component_config()
        
        # Set seeds for reproducibility
        torch.manual_seed(42)
        
        from meta_learning.meta_learning_modules.few_shot_modules.uncertainty_components import UncertaintyAwareDistance
        
        uncertainty_dist = UncertaintyAwareDistance(self.embedding_dim, config.uncertainty)
        
        query_embeddings = torch.randn(self.n_query, self.embedding_dim)
        prototypes = torch.randn(self.n_way, self.embedding_dim)
        
        # Run twice with same seed
        torch.manual_seed(42)
        distances1, uncertainties1 = uncertainty_dist(query_embeddings, prototypes)
        
        torch.manual_seed(42)
        distances2, uncertainties2 = uncertainty_dist(query_embeddings, prototypes)
        
        # Results should be identical
        torch.testing.assert_close(distances1, distances2, atol=1e-6, rtol=1e-6)
        # Note: Uncertainties might have some randomness depending on method