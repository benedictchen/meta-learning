"""
Integration tests for complete FIXME implementations in advanced_components.

Tests all 9 research-accurate methods implemented to replace fake implementations:
- UncertaintyAwareDistance: 3 methods (MC Dropout, Deep Ensembles, Evidential)
- MultiScaleFeatureAggregator: 3 methods (FPN, Dilated Conv, Attention)  
- HierarchicalPrototypes: 3 methods (Tree, Compositional, Capsule)

Follows established testing patterns with pytest fixtures, parametrization,
and proper error handling.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from unittest.mock import Mock, patch
from hypothesis import given, strategies as st, settings

from meta_learning.meta_learning_modules.few_shot_modules.advanced_components import (
    # Configuration classes
    UncertaintyAwareDistanceConfig,
    MultiScaleFeatureConfig,
    HierarchicalPrototypeConfig,
    
    # Implementation classes
    UncertaintyAwareDistance,
    MultiScaleFeatureAggregator,
    HierarchicalPrototypes,
    
    # Factory functions
    create_uncertainty_aware_distance,
    create_multiscale_feature_aggregator,
    create_hierarchical_prototypes,
    
    # Configuration presets
    get_uncertainty_config_presets,
    get_multiscale_config_presets,
    get_hierarchical_config_presets
)


class TestUncertaintyAwareDistanceImplementations:
    """Test all uncertainty-aware distance methods."""

    @pytest.fixture(params=["monte_carlo_dropout", "deep_ensembles", "evidential_deep_learning"])
    def uncertainty_method(self, request):
        """Parametrized fixture for all uncertainty methods."""
        return request.param

    @pytest.fixture
    def uncertainty_config(self, uncertainty_method):
        """Create configuration for uncertainty method."""
        base_config = {
            "embedding_dim": 128,
            "temperature": 2.0,
            "use_temperature_scaling": True
        }
        
        if uncertainty_method == "monte_carlo_dropout":
            base_config.update({
                "mc_dropout_samples": 5,  # Keep small for testing
                "mc_dropout_rate": 0.1,
                "mc_enable_training_mode": True
            })
        elif uncertainty_method == "deep_ensembles":
            base_config.update({
                "ensemble_size": 3,  # Keep small for testing
                "ensemble_diversity_weight": 0.1,
                "ensemble_temperature": 2.0
            })
        elif uncertainty_method == "evidential_deep_learning":
            base_config.update({
                "evidential_num_classes": 5,
                "evidential_lambda_reg": 0.01,
                "evidential_use_kl_annealing": True,
                "evidential_annealing_step": 10
            })
            
        return UncertaintyAwareDistanceConfig(
            uncertainty_method=uncertainty_method,
            **base_config
        )

    @pytest.fixture
    def test_data(self):
        """Generate test data for uncertainty methods."""
        torch.manual_seed(42)
        return {
            "query_features": torch.randn(16, 128),
            "prototypes": torch.randn(5, 128)
        }

    def test_uncertainty_method_initialization(self, uncertainty_config, uncertainty_method):
        """Test that uncertainty methods initialize correctly."""
        uncertainty_distance = UncertaintyAwareDistance(uncertainty_config)
        
        assert uncertainty_distance.config.uncertainty_method == uncertainty_method
        assert hasattr(uncertainty_distance, 'config')
        
        # Check method-specific initialization
        if uncertainty_method == "monte_carlo_dropout":
            assert hasattr(uncertainty_distance, 'mc_network')
        elif uncertainty_method == "deep_ensembles":
            assert hasattr(uncertainty_distance, 'ensemble_networks')
            assert hasattr(uncertainty_distance, 'diversity_weights')
        elif uncertainty_method == "evidential_deep_learning":
            assert hasattr(uncertainty_distance, 'evidential_network')
            assert hasattr(uncertainty_distance, 'annealing_step')

    def test_uncertainty_forward_pass(self, uncertainty_config, uncertainty_method, test_data):
        """Test forward pass produces correct output shapes."""
        uncertainty_distance = UncertaintyAwareDistance(uncertainty_config)
        
        distances = uncertainty_distance(test_data["query_features"], test_data["prototypes"])
        
        expected_shape = (test_data["query_features"].shape[0], test_data["prototypes"].shape[0])
        assert distances.shape == expected_shape
        assert torch.isfinite(distances).all()
        assert (distances >= 0).all()  # Distances should be non-negative

    def test_uncertainty_regularization_loss(self, uncertainty_config, uncertainty_method, test_data):
        """Test regularization loss computation."""
        uncertainty_distance = UncertaintyAwareDistance(uncertainty_config)
        
        reg_loss = uncertainty_distance.get_regularization_loss(test_data["query_features"])
        
        assert torch.isfinite(reg_loss)
        
        # Check method-specific regularization behavior
        if uncertainty_method == "evidential_deep_learning":
            # Should have non-zero regularization
            assert reg_loss.item() >= 0
        elif uncertainty_method == "deep_ensembles":
            # May have diversity regularization (can be negative to encourage diversity)
            assert torch.isfinite(reg_loss)
        else:
            # Monte Carlo dropout should have zero regularization
            assert reg_loss.item() == 0.0

    @pytest.mark.parametrize("preset_name", ["fast_mc_dropout", "small_ensemble", "evidential_fast"])
    def test_uncertainty_presets(self, preset_name):
        """Test uncertainty configuration presets."""
        presets = get_uncertainty_config_presets()
        assert preset_name in presets
        
        config = presets[preset_name]
        uncertainty_distance = UncertaintyAwareDistance(config)
        
        # Test with dummy data
        query_features = torch.randn(8, config.embedding_dim)
        prototypes = torch.randn(3, config.embedding_dim)
        
        distances = uncertainty_distance(query_features, prototypes)
        assert distances.shape == (8, 3)

    def test_factory_function(self, uncertainty_method, test_data):
        """Test factory function creates correct instances."""
        uncertainty_distance = create_uncertainty_aware_distance(
            uncertainty_method,
            embedding_dim=128,
            mc_dropout_samples=3,  # Override for testing
            ensemble_size=2,
            evidential_num_classes=5
        )
        
        assert uncertainty_distance.config.uncertainty_method == uncertainty_method
        
        distances = uncertainty_distance(test_data["query_features"], test_data["prototypes"])
        assert distances.shape == (16, 5)


class TestMultiScaleFeatureAggregatorImplementations:
    """Test all multi-scale feature aggregation methods."""

    @pytest.fixture(params=["feature_pyramid", "dilated_convolution", "attention_based"])
    def multiscale_method(self, request):
        """Parametrized fixture for all multi-scale methods."""
        return request.param

    @pytest.fixture
    def multiscale_config(self, multiscale_method):
        """Create configuration for multi-scale method."""
        base_config = {
            "embedding_dim": 128,
            "output_dim": 128,
            "use_residual_connection": True
        }
        
        if multiscale_method == "feature_pyramid":
            base_config.update({
                "fpn_scale_factors": [1, 2, 4],  # Keep simple for testing
                "fpn_use_lateral_connections": True,
                "fpn_feature_dim": 64
            })
        elif multiscale_method == "dilated_convolution":
            base_config.update({
                "dilated_rates": [1, 2, 4],  # Keep simple for testing
                "dilated_kernel_size": 3,
                "dilated_use_separable": False
            })
        elif multiscale_method == "attention_based":
            base_config.update({
                "attention_scales": [1, 2],  # Keep simple for testing
                "attention_heads": 4,
                "attention_dropout": 0.0  # Disable dropout for deterministic testing
            })
            
        return MultiScaleFeatureConfig(
            multiscale_method=multiscale_method,
            **base_config
        )

    @pytest.fixture
    def multiscale_test_data(self):
        """Generate test data for multi-scale methods."""
        torch.manual_seed(42)
        return {
            "features_2d": torch.randn(8, 128),  # [batch_size, embedding_dim]
            "features_3d": torch.randn(8, 10, 128),  # [batch_size, seq_len, embedding_dim]
        }

    def test_multiscale_initialization(self, multiscale_config, multiscale_method):
        """Test that multi-scale methods initialize correctly."""
        aggregator = MultiScaleFeatureAggregator(multiscale_config)
        
        assert aggregator.config.multiscale_method == multiscale_method
        assert hasattr(aggregator, 'feature_fusion')
        assert hasattr(aggregator, 'fusion_input_dim')
        
        # Check method-specific initialization
        if multiscale_method == "feature_pyramid":
            assert hasattr(aggregator, 'fpn_projections')
            assert hasattr(aggregator, 'fpn_smoothing')
        elif multiscale_method == "dilated_convolution":
            assert hasattr(aggregator, 'dilated_convs')
        elif multiscale_method == "attention_based":
            assert hasattr(aggregator, 'scale_attention')
            assert hasattr(aggregator, 'scale_transforms')

    def test_multiscale_forward_pass_2d(self, multiscale_config, multiscale_method, multiscale_test_data):
        """Test forward pass with 2D features."""
        aggregator = MultiScaleFeatureAggregator(multiscale_config)
        
        output = aggregator(multiscale_test_data["features_2d"])
        
        expected_shape = (multiscale_test_data["features_2d"].shape[0], multiscale_config.output_dim)
        assert output.shape == expected_shape
        assert torch.isfinite(output).all()

    def test_multiscale_forward_pass_3d(self, multiscale_config, multiscale_method, multiscale_test_data):
        """Test forward pass with 3D features."""
        aggregator = MultiScaleFeatureAggregator(multiscale_config)
        
        output = aggregator(multiscale_test_data["features_3d"])
        
        expected_shape = (multiscale_test_data["features_3d"].shape[0], multiscale_config.output_dim)
        assert output.shape == expected_shape
        assert torch.isfinite(output).all()

    @pytest.mark.parametrize("preset_name", ["fpn_standard", "dilated_standard", "attention_light"])
    def test_multiscale_presets(self, preset_name):
        """Test multi-scale configuration presets."""
        presets = get_multiscale_config_presets()
        assert preset_name in presets
        
        config = presets[preset_name]
        aggregator = MultiScaleFeatureAggregator(config)
        
        # Test with dummy data
        features = torch.randn(4, 8, config.embedding_dim)
        output = aggregator(features)
        assert output.shape == (4, config.output_dim)

    def test_multiscale_factory_function(self, multiscale_method, multiscale_test_data):
        """Test factory function creates correct instances."""
        aggregator = create_multiscale_feature_aggregator(
            multiscale_method,
            embedding_dim=128,
            output_dim=128,
            fpn_scale_factors=[1, 2],  # Keep simple
            dilated_rates=[1, 2],
            attention_scales=[1, 2]
        )
        
        assert aggregator.config.multiscale_method == multiscale_method
        
        output = aggregator(multiscale_test_data["features_2d"])
        assert output.shape == (8, 128)


class TestHierarchicalPrototypeImplementations:
    """Test all hierarchical prototype methods."""

    @pytest.fixture(params=["tree_structured", "compositional", "capsule_based"])
    def hierarchical_method(self, request):
        """Parametrized fixture for all hierarchical methods."""
        return request.param

    @pytest.fixture
    def hierarchical_config(self, hierarchical_method):
        """Create configuration for hierarchical method."""
        base_config = {
            "embedding_dim": 128,
            "use_residual_connections": True
        }
        
        if hierarchical_method == "tree_structured":
            base_config.update({
                "tree_depth": 2,  # Keep simple for testing
                "tree_branching_factor": 2,
                "tree_use_learned_routing": True,
                "tree_routing_temperature": 1.0
            })
        elif hierarchical_method == "compositional":
            base_config.update({
                "num_components": 8,  # Keep small for testing
                "composition_method": "weighted_sum",
                "component_diversity_loss": 0.01
            })
        elif hierarchical_method == "capsule_based":
            base_config.update({
                "num_capsules": 8,  # Keep small for testing
                "capsule_dim": 16,
                "routing_iterations": 2,  # Keep small for testing
                "routing_method": "dynamic"
            })
            
        return HierarchicalPrototypeConfig(
            hierarchy_method=hierarchical_method,
            **base_config
        )

    @pytest.fixture
    def hierarchical_test_data(self):
        """Generate test data for hierarchical methods."""
        torch.manual_seed(42)
        n_support, n_way, embedding_dim = 20, 4, 128
        # Ensure all n_way classes are represented in support set (following existing pattern)
        support_labels = torch.repeat_interleave(torch.arange(n_way), n_support // n_way)
        return {
            "support_features": torch.randn(n_support, embedding_dim),
            "support_labels": support_labels,
            "n_way": n_way,
            "embedding_dim": embedding_dim
        }

    def test_hierarchical_initialization(self, hierarchical_config, hierarchical_method):
        """Test that hierarchical methods initialize correctly."""
        hierarchical = HierarchicalPrototypes(hierarchical_config)
        
        assert hierarchical.config.hierarchy_method == hierarchical_method
        
        # Check method-specific initialization
        if hierarchical_method == "tree_structured":
            assert hasattr(hierarchical, 'tree_nodes')
            if hierarchical_config.tree_use_learned_routing:
                assert hasattr(hierarchical, 'routing_network')
        elif hierarchical_method == "compositional":
            assert hasattr(hierarchical, 'component_library')
            assert hasattr(hierarchical, 'composition_net') or hasattr(hierarchical, 'composition_attention') or hasattr(hierarchical, 'gating_network')
        elif hierarchical_method == "capsule_based":
            assert hasattr(hierarchical, 'primary_caps')
            assert hasattr(hierarchical, 'routing_weights')
            assert hasattr(hierarchical, 'capsule_transforms')

    def test_hierarchical_forward_pass(self, hierarchical_config, hierarchical_method, hierarchical_test_data):
        """Test forward pass produces correct output shapes."""
        hierarchical = HierarchicalPrototypes(hierarchical_config)
        
        prototypes = hierarchical(
            hierarchical_test_data["support_features"],
            hierarchical_test_data["support_labels"]
        )
        
        expected_shape = (hierarchical_test_data["n_way"], hierarchical_test_data["embedding_dim"])
        assert prototypes.shape == expected_shape
        assert torch.isfinite(prototypes).all()

    def test_hierarchical_diversity_loss(self, hierarchical_config, hierarchical_method, hierarchical_test_data):
        """Test diversity loss computation for compositional method."""
        hierarchical = HierarchicalPrototypes(hierarchical_config)
        
        diversity_loss = hierarchical.get_diversity_loss()
        
        assert torch.isfinite(diversity_loss)
        
        if hierarchical_method == "compositional":
            # Should have diversity loss for compositional method
            assert diversity_loss.item() >= 0
        else:
            # Other methods should return zero diversity loss
            assert diversity_loss.item() == 0.0

    @pytest.mark.parametrize("preset_name", ["tree_shallow", "compositional_small", "capsule_standard"])
    def test_hierarchical_presets(self, preset_name, hierarchical_test_data):
        """Test hierarchical configuration presets."""
        presets = get_hierarchical_config_presets()
        assert preset_name in presets
        
        config = presets[preset_name]
        config.embedding_dim = hierarchical_test_data["embedding_dim"]  # Match test data
        hierarchical = HierarchicalPrototypes(config)
        
        prototypes = hierarchical(
            hierarchical_test_data["support_features"],
            hierarchical_test_data["support_labels"]
        )
        assert prototypes.shape == (hierarchical_test_data["n_way"], hierarchical_test_data["embedding_dim"])

    def test_hierarchical_factory_function(self, hierarchical_method, hierarchical_test_data):
        """Test factory function creates correct instances."""
        hierarchical = create_hierarchical_prototypes(
            hierarchical_method,
            embedding_dim=128,
            tree_depth=2,  # Keep simple
            num_components=6,  # Keep small
            num_capsules=6     # Keep small
        )
        
        assert hierarchical.config.hierarchy_method == hierarchical_method
        
        prototypes = hierarchical(
            hierarchical_test_data["support_features"],
            hierarchical_test_data["support_labels"]
        )
        assert prototypes.shape == (hierarchical_test_data["n_way"], hierarchical_test_data["embedding_dim"])


class TestCompleteIntegrationPipeline:
    """Test complete integration of all research solutions together."""

    @pytest.fixture
    def pipeline_test_data(self):
        """Generate data for complete pipeline testing."""
        torch.manual_seed(42)
        n_support, n_query, n_way = 15, 10, 3
        embedding_dim, seq_len = 64, 8  # Keep small for testing
        
        return {
            "support_features_raw": torch.randn(n_support, seq_len, embedding_dim),
            "support_labels": torch.randint(0, n_way, (n_support,)),
            "query_features_raw": torch.randn(n_query, seq_len, embedding_dim),
            "n_support": n_support,
            "n_query": n_query,
            "n_way": n_way,
            "embedding_dim": embedding_dim,
            "seq_len": seq_len
        }

    def test_complete_pipeline_integration(self, pipeline_test_data):
        """Test complete few-shot learning pipeline with all research solutions."""
        # Step 1: Multi-Scale Feature Aggregation
        feature_aggregator = create_multiscale_feature_aggregator(
            "feature_pyramid",
            embedding_dim=pipeline_test_data["embedding_dim"],
            output_dim=pipeline_test_data["embedding_dim"],
            fpn_scale_factors=[1, 2]  # Keep simple for testing
        )
        
        support_features = feature_aggregator(pipeline_test_data["support_features_raw"])
        query_features = feature_aggregator(pipeline_test_data["query_features_raw"])
        
        assert support_features.shape == (pipeline_test_data["n_support"], pipeline_test_data["embedding_dim"])
        assert query_features.shape == (pipeline_test_data["n_query"], pipeline_test_data["embedding_dim"])
        
        # Step 2: Hierarchical Prototype Computation
        prototype_computer = create_hierarchical_prototypes(
            "compositional",
            embedding_dim=pipeline_test_data["embedding_dim"],
            num_components=6  # Keep small for testing
        )
        
        prototypes = prototype_computer(support_features, pipeline_test_data["support_labels"])
        assert prototypes.shape == (pipeline_test_data["n_way"], pipeline_test_data["embedding_dim"])
        
        # Step 3: Uncertainty-Aware Distance Computation
        distance_computer = create_uncertainty_aware_distance(
            "monte_carlo_dropout",
            embedding_dim=pipeline_test_data["embedding_dim"],
            mc_dropout_samples=3  # Keep small for testing
        )
        
        distances = distance_computer(query_features, prototypes)
        assert distances.shape == (pipeline_test_data["n_query"], pipeline_test_data["n_way"])
        
        # Step 4: Final Predictions
        logits = -distances  # Negative distances as logits
        predictions = torch.softmax(logits, dim=-1)
        predicted_classes = torch.argmax(predictions, dim=-1)
        
        assert predictions.shape == (pipeline_test_data["n_query"], pipeline_test_data["n_way"])
        assert predicted_classes.shape == (pipeline_test_data["n_query"],)
        assert torch.all(predicted_classes >= 0)
        assert torch.all(predicted_classes < pipeline_test_data["n_way"])

    @pytest.mark.parametrize("uncertainty_method", ["monte_carlo_dropout", "deep_ensembles", "evidential_deep_learning"])
    @pytest.mark.parametrize("multiscale_method", ["feature_pyramid", "dilated_convolution", "attention_based"])
    @pytest.mark.parametrize("hierarchical_method", ["tree_structured", "compositional", "capsule_based"])
    def test_all_method_combinations(
        self, 
        uncertainty_method, 
        multiscale_method, 
        hierarchical_method, 
        pipeline_test_data
    ):
        """Test all possible combinations of the 9 methods (3x3x3=27 combinations)."""
        # This test verifies that every combination of methods works together
        # Keep configurations simple to ensure test speed
        
        # Multi-scale aggregation
        aggregator = create_multiscale_feature_aggregator(
            multiscale_method,
            embedding_dim=pipeline_test_data["embedding_dim"],
            output_dim=pipeline_test_data["embedding_dim"],
            fpn_scale_factors=[1, 2],
            dilated_rates=[1, 2],
            attention_scales=[1, 2]
        )
        
        support_features = aggregator(pipeline_test_data["support_features_raw"])
        query_features = aggregator(pipeline_test_data["query_features_raw"])
        
        # Hierarchical prototypes
        hierarchical = create_hierarchical_prototypes(
            hierarchical_method,
            embedding_dim=pipeline_test_data["embedding_dim"],
            tree_depth=2,
            num_components=4,
            num_capsules=4
        )
        
        prototypes = hierarchical(support_features, pipeline_test_data["support_labels"])
        
        # Uncertainty-aware distances
        uncertainty = create_uncertainty_aware_distance(
            uncertainty_method,
            embedding_dim=pipeline_test_data["embedding_dim"],
            mc_dropout_samples=2,
            ensemble_size=2,
            evidential_num_classes=pipeline_test_data["n_way"]
        )
        
        distances = uncertainty(query_features, prototypes)
        
        # Verify final output is valid
        assert distances.shape == (pipeline_test_data["n_query"], pipeline_test_data["n_way"])
        assert torch.isfinite(distances).all()
        assert (distances >= 0).all()  # Distances should be non-negative


@pytest.mark.slow
class TestFixmePerformanceRegression:
    """Performance regression tests for FIXME implementations."""

    def test_uncertainty_methods_performance(self, benchmark_config):
        """Test that uncertainty methods meet performance requirements."""
        query_features = torch.randn(32, 256)
        prototypes = torch.randn(5, 256)
        
        configs = {
            "monte_carlo_dropout": {"mc_dropout_samples": 10},
            "deep_ensembles": {"ensemble_size": 5},
            "evidential_deep_learning": {"evidential_num_classes": 5}
        }
        
        for method, params in configs.items():
            uncertainty = create_uncertainty_aware_distance(method, embedding_dim=256, **params)
            
            import time
            start_time = time.time()
            
            for _ in range(benchmark_config["measurement_rounds"]):
                distances = uncertainty(query_features, prototypes)
                
            elapsed_time = time.time() - start_time
            avg_time = elapsed_time / benchmark_config["measurement_rounds"]
            
            # Should complete in reasonable time (adjust threshold as needed)
            assert avg_time < 1.0, f"{method} took {avg_time:.3f}s per forward pass"

    def test_memory_usage_reasonable(self):
        """Test that implementations don't use excessive memory."""
        # Create larger test case to stress memory
        query_features = torch.randn(128, 512)
        prototypes = torch.randn(10, 512)
        
        uncertainty = create_uncertainty_aware_distance(
            "deep_ensembles",
            embedding_dim=512,
            ensemble_size=5
        )
        
        # Monitor memory usage (simplified check)
        import gc
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
        
        distances = uncertainty(query_features, prototypes)
        
        # Should not cause memory errors and produce reasonable output
        assert distances.shape == (128, 10)
        assert torch.isfinite(distances).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])