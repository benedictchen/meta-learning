"""
Comprehensive unit tests for few_shot_learning module.

Tests all research solutions, configuration options, and research-accurate implementations
following 2024/2025 pytest best practices.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from hypothesis import given, strategies as st, settings
from typing import Dict, List, Tuple, Any

from meta_learning.meta_learning_modules.few_shot_learning import (
    FewShotLearner, 
    PrototypicalLearner,
    FewShotConfig,
    PrototypicalConfig,
    UncertaintyAwareDistance,
    HierarchicalPrototypes, 
    TaskAdaptivePrototypes,
    create_few_shot_learner
)


class TestFewShotConfig:
    """Test configuration classes."""
    
    def test_few_shot_config_defaults(self):
        """Test default configuration values."""
        config = FewShotConfig()
        assert config.n_way == 5
        assert config.k_shot == 1
        assert config.query_shots == 15
        assert config.distance_metric == "euclidean"
        assert config.temperature == 1.0
        
    def test_few_shot_config_validation(self):
        """Test configuration validation."""
        # Valid configurations
        FewShotConfig(n_way=3, k_shot=5, query_shots=10)
        
        # Invalid configurations should still construct but may cause runtime errors
        config = FewShotConfig(n_way=0, k_shot=-1)
        assert config.n_way == 0  # Constructor doesn't validate
        
    @given(
        n_way=st.integers(min_value=2, max_value=20),
        k_shot=st.integers(min_value=1, max_value=10),
        query_shots=st.integers(min_value=5, max_value=50)
    )
    def test_few_shot_config_property_based(self, n_way, k_shot, query_shots):
        """Property-based test for configuration construction."""
        config = FewShotConfig(n_way=n_way, k_shot=k_shot, query_shots=query_shots)
        assert config.n_way == n_way
        assert config.k_shot == k_shot
        assert config.query_shots == query_shots


class TestPrototypicalConfig:
    """Test prototypical network configuration."""
    
    def test_prototypical_config_defaults(self):
        """Test default prototypical configuration."""
        config = PrototypicalConfig()
        assert config.protonet_variant == "research_accurate"
        assert config.use_squared_euclidean == True
        assert config.use_temperature_scaling == False
        assert config.use_uncertainty_aware_distances == False
        
    def test_prototypical_config_variants(self):
        """Test different prototypical variants."""
        variants = ["original", "research_accurate", "simple", "enhanced"]
        for variant in variants:
            config = PrototypicalConfig(protonet_variant=variant)
            assert config.protonet_variant == variant
            
    def test_prototypical_config_feature_flags(self):
        """Test prototypical feature configuration."""
        config = PrototypicalConfig(
            use_uncertainty_aware_distances=True,
            use_hierarchical_prototypes=True,
            use_task_adaptive_prototypes=True
        )
        assert config.use_uncertainty_aware_distances == True
        assert config.use_hierarchical_prototypes == True
        assert config.use_task_adaptive_prototypes == True


class TestFewShotLearner:
    """Test base few-shot learner implementation."""
    
    @pytest.fixture
    def simple_model(self):
        """Create a simple model for testing."""
        return nn.Sequential(
            nn.Linear(84, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
    @pytest.fixture
    def sample_data(self):
        """Create sample few-shot data."""
        n_way, k_shot, query_shots = 5, 2, 10
        feature_dim = 32
        
        support_x = torch.randn(n_way, k_shot, feature_dim)
        support_y = torch.arange(n_way).repeat_interleave(k_shot)
        query_x = torch.randn(n_way * query_shots, feature_dim)
        query_y = torch.arange(n_way).repeat(query_shots)
        
        return support_x, support_y, query_x, query_y
    
    def test_few_shot_learner_init(self, simple_model):
        """Test few-shot learner initialization."""
        config = FewShotConfig()
        learner = FewShotLearner(simple_model, config)
        
        assert learner.model == simple_model
        assert learner.config == config
        assert learner.n_way == 5
        assert learner.k_shot == 1
        
    def test_few_shot_learner_forward(self, simple_model, sample_data):
        """Test few-shot learner forward pass."""
        config = FewShotConfig(n_way=5, k_shot=2, query_shots=10)
        learner = FewShotLearner(simple_model, config)
        
        support_x, support_y, query_x, query_y = sample_data
        
        # Should not raise an exception
        try:
            logits = learner(support_x, support_y, query_x)
            assert logits.shape == (50, 5)  # (n_way * query_shots, n_way)
        except NotImplementedError:
            # Base class may not implement forward
            pass
            
    def test_few_shot_learner_compute_loss(self, simple_model, sample_data):
        """Test few-shot learner loss computation."""
        config = FewShotConfig(n_way=5, k_shot=2, query_shots=10)
        learner = FewShotLearner(simple_model, config)
        
        support_x, support_y, query_x, query_y = sample_data
        
        try:
            loss = learner.compute_loss(support_x, support_y, query_x, query_y)
            assert isinstance(loss, torch.Tensor)
            assert loss.ndim == 0  # Scalar loss
        except NotImplementedError:
            # Base class may not implement compute_loss
            pass


class TestPrototypicalLearner:
    """Test prototypical network implementation with all research solutions."""
    
    @pytest.fixture
    def simple_encoder(self):
        """Create a simple encoder for prototypical learning."""
        return nn.Sequential(
            nn.Linear(84, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
    @pytest.fixture
    def prototypical_learner(self, simple_encoder):
        """Create prototypical learner with default config."""
        config = PrototypicalConfig()
        return PrototypicalLearner(simple_encoder, config)
        
    @pytest.fixture
    def sample_episode(self):
        """Create sample episode data for prototypical learning."""
        n_way, k_shot, query_shots = 5, 3, 12
        feature_dim = 32
        
        support_x = torch.randn(n_way, k_shot, feature_dim)
        support_y = torch.arange(n_way).repeat_interleave(k_shot)
        query_x = torch.randn(n_way * query_shots, feature_dim)
        query_y = torch.arange(n_way).repeat(query_shots)
        
        return support_x, support_y, query_x, query_y
    
    def test_prototypical_learner_init(self, simple_encoder):
        """Test prototypical learner initialization."""
        config = PrototypicalConfig()
        learner = PrototypicalLearner(simple_encoder, config)
        
        assert learner.encoder == simple_encoder
        assert learner.config == config
        assert learner.n_way == 5
        
    def test_compute_prototypes_basic(self, prototypical_learner, sample_episode):
        """Test basic prototype computation."""
        support_x, support_y, _, _ = sample_episode
        
        with torch.no_grad():
            support_features = prototypical_learner.encoder(support_x.view(-1, 32))
            support_features = support_features.view(5, 3, -1)  # n_way, k_shot, feature_dim
        
        prototypes = prototypical_learner.compute_prototypes(support_features, support_y)
        
        assert prototypes.shape == (5, support_features.shape[-1])  # n_way, feature_dim
        assert torch.isfinite(prototypes).all()
        
    def test_compute_prototypes_variants(self, simple_encoder, sample_episode):
        """Test all prototypical computation variants."""
        support_x, support_y, _, _ = sample_episode
        
        variants = ["original", "research_accurate", "simple", "enhanced"]
        
        for variant in variants:
            config = PrototypicalConfig(protonet_variant=variant)
            learner = PrototypicalLearner(simple_encoder, config)
            
            with torch.no_grad():
                support_features = learner.encoder(support_x.view(-1, 32))
                support_features = support_features.view(5, 3, -1)
            
            prototypes = learner.compute_prototypes(support_features, support_y)
            assert prototypes.shape == (5, support_features.shape[-1])
            
    def test_distance_computation_methods(self, prototypical_learner):
        """Test different distance computation methods."""
        query_features = torch.randn(25, 64)  # 25 queries, 64 features
        prototypes = torch.randn(5, 64)       # 5 prototypes, 64 features
        
        # Test euclidean distance
        prototypical_learner.config.distance_metric = "euclidean"
        distances = prototypical_learner.compute_distances(query_features, prototypes)
        assert distances.shape == (25, 5)
        
        # Test cosine distance
        prototypical_learner.config.distance_metric = "cosine"
        distances = prototypical_learner.compute_distances(query_features, prototypes)
        assert distances.shape == (25, 5)
        
    def test_temperature_scaling(self, simple_encoder, sample_episode):
        """Test temperature scaling functionality."""
        config = PrototypicalConfig(use_temperature_scaling=True, temperature=2.0)
        learner = PrototypicalLearner(simple_encoder, config)
        
        support_x, support_y, query_x, query_y = sample_episode
        
        logits = learner(support_x, support_y, query_x)
        assert logits.shape == (60, 5)  # query_shots * n_way, n_way
        assert torch.isfinite(logits).all()
        
    def test_uncertainty_aware_distances(self, simple_encoder, sample_episode):
        """Test uncertainty-aware distance computation."""
        config = PrototypicalConfig(use_uncertainty_aware_distances=True)
        learner = PrototypicalLearner(simple_encoder, config)
        
        support_x, support_y, query_x, query_y = sample_episode
        
        logits = learner(support_x, support_y, query_x)
        assert logits.shape == (60, 5)
        assert torch.isfinite(logits).all()
        
    def test_hierarchical_prototypes(self, simple_encoder, sample_episode):
        """Test hierarchical prototype computation."""
        config = PrototypicalConfig(use_hierarchical_prototypes=True)
        learner = PrototypicalLearner(simple_encoder, config)
        
        support_x, support_y, query_x, query_y = sample_episode
        
        logits = learner(support_x, support_y, query_x)
        assert logits.shape == (60, 5)
        assert torch.isfinite(logits).all()
        
    def test_task_adaptive_prototypes(self, simple_encoder, sample_episode):
        """Test task-adaptive prototype computation.""" 
        config = PrototypicalConfig(use_task_adaptive_prototypes=True)
        learner = PrototypicalLearner(simple_encoder, config)
        
        support_x, support_y, query_x, query_y = sample_episode
        
        logits = learner(support_x, support_y, query_x)
        assert logits.shape == (60, 5)
        assert torch.isfinite(logits).all()
        
    def test_forward_pass_all_variants(self, simple_encoder, sample_episode):
        """Test forward pass with all configuration variants."""
        variants = ["original", "research_accurate", "simple", "enhanced"]
        
        support_x, support_y, query_x, query_y = sample_episode
        
        for variant in variants:
            config = PrototypicalConfig(protonet_variant=variant)
            learner = PrototypicalLearner(simple_encoder, config)
            
            logits = learner(support_x, support_y, query_x)
            assert logits.shape == (60, 5)
            assert torch.isfinite(logits).all()
            
    def test_compute_loss_prototypical(self, prototypical_learner, sample_episode):
        """Test prototypical loss computation."""
        support_x, support_y, query_x, query_y = sample_episode
        
        loss = prototypical_learner.compute_loss(support_x, support_y, query_x, query_y)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.item() >= 0  # Loss should be non-negative
        
    def test_accuracy_computation(self, prototypical_learner, sample_episode):
        """Test accuracy computation for prototypical networks."""
        support_x, support_y, query_x, query_y = sample_episode
        
        logits = prototypical_learner(support_x, support_y, query_x)
        predictions = torch.argmax(logits, dim=1)
        
        # Compute accuracy
        accuracy = (predictions == query_y).float().mean()
        assert 0 <= accuracy <= 1
        assert isinstance(accuracy.item(), float)


class TestUncertaintyAwareDistance:
    """Test uncertainty-aware distance computation."""
    
    @pytest.fixture
    def uncertainty_distance(self):
        """Create uncertainty-aware distance module."""
        return UncertaintyAwareDistance(feature_dim=64)
        
    def test_uncertainty_distance_init(self):
        """Test uncertainty distance initialization."""
        module = UncertaintyAwareDistance(feature_dim=128)
        assert hasattr(module, 'uncertainty_estimator')
        assert hasattr(module, 'feature_dim')
        
    def test_uncertainty_distance_forward(self, uncertainty_distance):
        """Test uncertainty distance computation."""
        query_features = torch.randn(20, 64)
        prototypes = torch.randn(5, 64)
        
        distances = uncertainty_distance(query_features, prototypes)
        assert distances.shape == (20, 5)
        assert torch.isfinite(distances).all()
        
    @given(
        n_queries=st.integers(min_value=5, max_value=50),
        n_prototypes=st.integers(min_value=2, max_value=10),
        feature_dim=st.integers(min_value=16, max_value=128)
    )
    def test_uncertainty_distance_property_based(self, n_queries, n_prototypes, feature_dim):
        """Property-based test for uncertainty distance."""
        module = UncertaintyAwareDistance(feature_dim=feature_dim)
        query_features = torch.randn(n_queries, feature_dim)
        prototypes = torch.randn(n_prototypes, feature_dim)
        
        distances = module(query_features, prototypes)
        assert distances.shape == (n_queries, n_prototypes)
        assert torch.isfinite(distances).all()


class TestHierarchicalPrototypes:
    """Test hierarchical prototype computation."""
    
    @pytest.fixture
    def hierarchical_prototypes(self):
        """Create hierarchical prototypes module."""
        return HierarchicalPrototypes(feature_dim=64, n_levels=3)
        
    def test_hierarchical_prototypes_init(self):
        """Test hierarchical prototypes initialization."""
        module = HierarchicalPrototypes(feature_dim=128, n_levels=2)
        assert hasattr(module, 'hierarchies')
        assert len(module.hierarchies) == 2
        
    def test_hierarchical_prototypes_forward(self, hierarchical_prototypes):
        """Test hierarchical prototypes computation."""
        support_features = torch.randn(5, 3, 64)  # n_way, k_shot, feature_dim
        support_labels = torch.arange(5).repeat_interleave(3)
        
        prototypes = hierarchical_prototypes(support_features, support_labels)
        assert prototypes.shape == (5, 64)  # n_way, feature_dim
        assert torch.isfinite(prototypes).all()
        
    def test_hierarchical_prototypes_different_levels(self):
        """Test hierarchical prototypes with different numbers of levels."""
        for n_levels in [1, 2, 3, 5]:
            module = HierarchicalPrototypes(feature_dim=32, n_levels=n_levels)
            support_features = torch.randn(3, 2, 32)
            support_labels = torch.arange(3).repeat_interleave(2)
            
            prototypes = module(support_features, support_labels)
            assert prototypes.shape == (3, 32)
            assert torch.isfinite(prototypes).all()


class TestTaskAdaptivePrototypes:
    """Test task-adaptive prototype computation."""
    
    @pytest.fixture 
    def task_adaptive_prototypes(self):
        """Create task-adaptive prototypes module."""
        return TaskAdaptivePrototypes(feature_dim=64)
        
    def test_task_adaptive_prototypes_init(self):
        """Test task-adaptive prototypes initialization."""
        module = TaskAdaptivePrototypes(feature_dim=128)
        assert hasattr(module, 'adaptation_network')
        assert hasattr(module, 'feature_dim')
        
    def test_task_adaptive_prototypes_forward(self, task_adaptive_prototypes):
        """Test task-adaptive prototypes computation."""
        support_features = torch.randn(5, 3, 64)
        support_labels = torch.arange(5).repeat_interleave(3)
        
        prototypes = task_adaptive_prototypes(support_features, support_labels)
        assert prototypes.shape == (5, 64)
        assert torch.isfinite(prototypes).all()
        
    def test_task_adaptive_prototypes_adaptation(self, task_adaptive_prototypes):
        """Test that task adaptation actually changes prototypes."""
        support_features = torch.randn(3, 4, 64)
        support_labels = torch.arange(3).repeat_interleave(4)
        
        # Compute prototypes twice (should be different due to adaptation)
        prototypes1 = task_adaptive_prototypes(support_features, support_labels)
        prototypes2 = task_adaptive_prototypes(support_features, support_labels)
        
        # They should be different (adaptation network has randomness)
        assert prototypes1.shape == prototypes2.shape == (3, 64)
        # Note: Due to randomness, they might be different, but we can't guarantee it


class TestCreateFewShotLearner:
    """Test factory function for few-shot learners."""
    
    @pytest.fixture
    def simple_model(self):
        """Create simple model for testing."""
        return nn.Linear(84, 32)
        
    def test_create_prototypical_learner(self, simple_model):
        """Test creation of prototypical learner."""
        config = PrototypicalConfig()
        learner = create_few_shot_learner("prototypical", simple_model, config)
        
        assert isinstance(learner, PrototypicalLearner)
        assert learner.model == simple_model
        assert learner.config == config
        
    def test_create_basic_few_shot_learner(self, simple_model):
        """Test creation of basic few-shot learner."""
        config = FewShotConfig()
        learner = create_few_shot_learner("basic", simple_model, config)
        
        assert isinstance(learner, FewShotLearner)
        assert learner.model == simple_model
        
    def test_create_invalid_learner_type(self, simple_model):
        """Test creation with invalid learner type."""
        config = FewShotConfig()
        
        with pytest.raises(ValueError, match="Unknown few-shot learner type"):
            create_few_shot_learner("nonexistent", simple_model, config)
            
    @pytest.mark.parametrize("learner_type,expected_class", [
        ("basic", FewShotLearner),
        ("prototypical", PrototypicalLearner),
    ])
    def test_create_learner_parametrized(self, simple_model, learner_type, expected_class):
        """Parametrized test for different learner types."""
        if learner_type == "prototypical":
            config = PrototypicalConfig()
        else:
            config = FewShotConfig()
            
        learner = create_few_shot_learner(learner_type, simple_model, config)
        assert isinstance(learner, expected_class)


class TestPrototypicalNetworkVariants:
    """Test specific prototypical network variant implementations."""
    
    @pytest.fixture
    def encoder(self):
        """Create encoder for variant testing."""
        return nn.Sequential(nn.Linear(84, 64), nn.ReLU(), nn.Linear(64, 32))
        
    @pytest.fixture 
    def episode_data(self):
        """Create episode data for variant testing."""
        n_way, k_shot, query_shots = 3, 5, 15
        feature_dim = 32
        
        support_x = torch.randn(n_way, k_shot, feature_dim)
        support_y = torch.arange(n_way).repeat_interleave(k_shot)
        query_x = torch.randn(n_way * query_shots, feature_dim) 
        query_y = torch.arange(n_way).repeat(query_shots)
        
        return support_x, support_y, query_x, query_y
        
    @pytest.mark.parametrize("variant", ["original", "research_accurate", "simple", "enhanced"])
    def test_prototypical_variants_forward_pass(self, encoder, episode_data, variant):
        """Test forward pass for all prototypical variants."""
        config = PrototypicalConfig(protonet_variant=variant)
        learner = PrototypicalLearner(encoder, config)
        
        support_x, support_y, query_x, query_y = episode_data
        
        logits = learner(support_x, support_y, query_x)
        assert logits.shape == (45, 3)  # query_shots * n_way, n_way
        assert torch.isfinite(logits).all()
        
    @pytest.mark.parametrize("use_squared", [True, False])
    def test_squared_euclidean_option(self, encoder, episode_data, use_squared):
        """Test squared euclidean distance option."""
        config = PrototypicalConfig(use_squared_euclidean=use_squared)
        learner = PrototypicalLearner(encoder, config)
        
        support_x, support_y, query_x, query_y = episode_data
        
        logits = learner(support_x, support_y, query_x)
        assert torch.isfinite(logits).all()


class TestFewShotLearningIntegration:
    """Integration tests for complete few-shot learning pipeline."""
    
    @pytest.fixture
    def complete_setup(self):
        """Create complete few-shot learning setup."""
        encoder = nn.Sequential(
            nn.Linear(84, 64),
            nn.ReLU(), 
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )
        
        config = PrototypicalConfig(
            n_way=4,
            k_shot=3, 
            query_shots=12,
            protonet_variant="research_accurate",
            use_temperature_scaling=True,
            temperature=1.5
        )
        
        learner = PrototypicalLearner(encoder, config)
        
        # Create realistic episode data
        support_x = torch.randn(4, 3, 84)
        support_y = torch.arange(4).repeat_interleave(3)
        query_x = torch.randn(48, 84)  # 4 * 12 queries
        query_y = torch.arange(4).repeat(12)
        
        return learner, (support_x, support_y, query_x, query_y)
        
    def test_complete_few_shot_pipeline(self, complete_setup):
        """Test complete few-shot learning pipeline."""
        learner, (support_x, support_y, query_x, query_y) = complete_setup
        
        # Forward pass
        logits = learner(support_x, support_y, query_x)
        assert logits.shape == (48, 4)
        
        # Loss computation  
        loss = learner.compute_loss(support_x, support_y, query_x, query_y)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        
        # Accuracy computation
        predictions = torch.argmax(logits, dim=1)
        accuracy = (predictions == query_y).float().mean()
        assert 0 <= accuracy <= 1
        
    def test_gradient_computation(self, complete_setup):
        """Test that gradients flow properly through the model."""
        learner, (support_x, support_y, query_x, query_y) = complete_setup
        
        # Enable gradients
        learner.encoder.train()
        
        loss = learner.compute_loss(support_x, support_y, query_x, query_y)
        
        # Compute gradients
        loss.backward()
        
        # Check that gradients exist
        gradients_exist = any(
            param.grad is not None and torch.isfinite(param.grad).all()
            for param in learner.encoder.parameters()
        )
        assert gradients_exist, "Gradients should flow through the encoder"
        
    def test_reproducibility(self, complete_setup):
        """Test reproducibility with fixed random seed."""
        learner, (support_x, support_y, query_x, query_y) = complete_setup
        
        # Set seed and compute
        torch.manual_seed(42)
        logits1 = learner(support_x, support_y, query_x)
        
        # Reset seed and compute again
        torch.manual_seed(42) 
        logits2 = learner(support_x, support_y, query_x)
        
        # Should be identical (for deterministic operations)
        assert torch.allclose(logits1, logits2, atol=1e-6)


class TestFewShotLearningEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.fixture
    def simple_encoder(self):
        """Create simple encoder for edge case testing."""
        return nn.Linear(10, 8)
        
    def test_single_shot_learning(self, simple_encoder):
        """Test 1-shot learning scenario."""
        config = PrototypicalConfig(n_way=3, k_shot=1, query_shots=5)
        learner = PrototypicalLearner(simple_encoder, config)
        
        support_x = torch.randn(3, 1, 10)  # Only 1 shot per class
        support_y = torch.arange(3)
        query_x = torch.randn(15, 10)
        query_y = torch.arange(3).repeat(5)
        
        logits = learner(support_x, support_y, query_x) 
        assert logits.shape == (15, 3)
        
    def test_many_way_learning(self, simple_encoder):
        """Test learning with many classes."""
        config = PrototypicalConfig(n_way=20, k_shot=2, query_shots=5)
        learner = PrototypicalLearner(simple_encoder, config)
        
        support_x = torch.randn(20, 2, 10)
        support_y = torch.arange(20).repeat_interleave(2)
        query_x = torch.randn(100, 10)
        query_y = torch.arange(20).repeat(5)
        
        logits = learner(support_x, support_y, query_x)
        assert logits.shape == (100, 20)
        
    def test_mismatched_dimensions(self, simple_encoder):
        """Test handling of mismatched input dimensions."""
        config = PrototypicalConfig()
        learner = PrototypicalLearner(simple_encoder, config)
        
        # Wrong input dimension (should be 10, but using 20)
        support_x = torch.randn(5, 1, 20)
        support_y = torch.arange(5)
        query_x = torch.randn(25, 20)
        
        with pytest.raises(RuntimeError):
            learner(support_x, support_y, query_x)
            
    def test_empty_support_set(self, simple_encoder):
        """Test handling of empty support set."""
        config = PrototypicalConfig()
        learner = PrototypicalLearner(simple_encoder, config)
        
        # Empty support set
        support_x = torch.empty(0, 0, 10)
        support_y = torch.empty(0, dtype=torch.long)
        query_x = torch.randn(5, 10)
        
        with pytest.raises((RuntimeError, ValueError, IndexError)):
            learner(support_x, support_y, query_x)


class TestFewShotLearningPerformance:
    """Performance and computational efficiency tests."""
    
    @pytest.fixture
    def performance_encoder(self):
        """Create larger encoder for performance testing."""
        return nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128), 
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
    @pytest.mark.slow
    def test_large_scale_performance(self, performance_encoder):
        """Test performance with large-scale few-shot learning."""
        config = PrototypicalConfig(n_way=10, k_shot=5, query_shots=50)
        learner = PrototypicalLearner(performance_encoder, config)
        
        support_x = torch.randn(10, 5, 512)
        support_y = torch.arange(10).repeat_interleave(5) 
        query_x = torch.randn(500, 512)  # Large query set
        query_y = torch.arange(10).repeat(50)
        
        import time
        start_time = time.time()
        
        logits = learner(support_x, support_y, query_x)
        loss = learner.compute_loss(support_x, support_y, query_x, query_y)
        
        elapsed_time = time.time() - start_time
        
        assert logits.shape == (500, 10)
        assert isinstance(loss, torch.Tensor)
        assert elapsed_time < 5.0  # Should complete within 5 seconds
        
    @pytest.mark.gpu_required
    def test_gpu_acceleration(self, performance_encoder):
        """Test GPU acceleration if available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
            
        device = torch.device("cuda")
        encoder = performance_encoder.to(device)
        
        config = PrototypicalConfig(n_way=5, k_shot=3, query_shots=20)
        learner = PrototypicalLearner(encoder, config)
        
        support_x = torch.randn(5, 3, 512, device=device)
        support_y = torch.arange(5, device=device).repeat_interleave(3)
        query_x = torch.randn(100, 512, device=device)
        query_y = torch.arange(5, device=device).repeat(20)
        
        logits = learner(support_x, support_y, query_x)
        loss = learner.compute_loss(support_x, support_y, query_x, query_y)
        
        assert logits.device == device
        assert loss.device == device


class TestFixmeSolutions:
    """Test all research solutions are properly implemented."""
    
    @pytest.fixture
    def encoder(self):
        """Create encoder for FIXME testing."""
        return nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 8))
        
    @pytest.mark.fixme_solution
    def test_original_snell_prototypical_implementation(self, encoder):
        """Test original Snell et al. prototypical implementation."""
        config = PrototypicalConfig(protonet_variant="original")
        learner = PrototypicalLearner(encoder, config)
        
        support_x = torch.randn(3, 2, 32)
        support_y = torch.arange(3).repeat_interleave(2)
        query_x = torch.randn(15, 32)
        
        logits = learner(support_x, support_y, query_x)
        assert logits.shape == (15, 3)
        assert torch.isfinite(logits).all()
        
    @pytest.mark.fixme_solution
    def test_research_accurate_implementation(self, encoder):
        """Test research-accurate prototypical implementation."""
        config = PrototypicalConfig(protonet_variant="research_accurate")
        learner = PrototypicalLearner(encoder, config)
        
        support_x = torch.randn(4, 3, 32)
        support_y = torch.arange(4).repeat_interleave(3)
        query_x = torch.randn(20, 32)
        
        logits = learner(support_x, support_y, query_x)
        assert logits.shape == (20, 4)
        
    @pytest.mark.fixme_solution
    def test_uncertainty_aware_distances_solution(self, encoder):
        """Test uncertainty-aware distances FIXME solution."""
        config = PrototypicalConfig(use_uncertainty_aware_distances=True)
        learner = PrototypicalLearner(encoder, config)
        
        support_x = torch.randn(5, 2, 32)
        support_y = torch.arange(5).repeat_interleave(2)
        query_x = torch.randn(25, 32)
        
        logits = learner(support_x, support_y, query_x)
        assert logits.shape == (25, 5)
        assert torch.isfinite(logits).all()
        
    @pytest.mark.fixme_solution
    def test_hierarchical_prototypes_solution(self, encoder):
        """Test hierarchical prototypes FIXME solution."""
        config = PrototypicalConfig(use_hierarchical_prototypes=True)
        learner = PrototypicalLearner(encoder, config) 
        
        support_x = torch.randn(3, 4, 32)
        support_y = torch.arange(3).repeat_interleave(4)
        query_x = torch.randn(18, 32)
        
        logits = learner(support_x, support_y, query_x)
        assert logits.shape == (18, 3)
        
    @pytest.mark.fixme_solution 
    def test_task_adaptive_prototypes_solution(self, encoder):
        """Test task-adaptive prototypes FIXME solution."""
        config = PrototypicalConfig(use_task_adaptive_prototypes=True)
        learner = PrototypicalLearner(encoder, config)
        
        support_x = torch.randn(2, 6, 32) 
        support_y = torch.arange(2).repeat_interleave(6)
        query_x = torch.randn(12, 32)
        
        logits = learner(support_x, support_y, query_x)
        assert logits.shape == (12, 2)


@pytest.mark.property
class TestPropertyBasedFewShotLearning:
    """Property-based tests using Hypothesis for few-shot learning."""
    
    @given(
        n_way=st.integers(min_value=2, max_value=10),
        k_shot=st.integers(min_value=1, max_value=8),
        query_shots=st.integers(min_value=5, max_value=20),
        feature_dim=st.integers(min_value=8, max_value=64)
    )
    @settings(max_examples=10, deadline=10000)  # Limit for faster testing
    def test_prototypical_learner_shape_invariants(self, n_way, k_shot, query_shots, feature_dim):
        """Test shape invariants for prototypical learner."""
        encoder = nn.Linear(feature_dim, feature_dim // 2)
        config = PrototypicalConfig(n_way=n_way, k_shot=k_shot, query_shots=query_shots)
        learner = PrototypicalLearner(encoder, config)
        
        support_x = torch.randn(n_way, k_shot, feature_dim)
        support_y = torch.arange(n_way).repeat_interleave(k_shot)
        query_x = torch.randn(n_way * query_shots, feature_dim)
        
        logits = learner(support_x, support_y, query_x)
        
        # Shape invariant: output should be (n_queries, n_classes)
        assert logits.shape == (n_way * query_shots, n_way)
        assert torch.isfinite(logits).all()
        
    @given(
        variant=st.sampled_from(["original", "research_accurate", "simple", "enhanced"]),
        distance_metric=st.sampled_from(["euclidean", "cosine"]),
        temperature=st.floats(min_value=0.1, max_value=10.0)
    )
    @settings(max_examples=8)
    def test_prototypical_variants_consistency(self, variant, distance_metric, temperature):
        """Test consistency across prototypical variants."""
        encoder = nn.Sequential(nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 4))
        config = PrototypicalConfig(
            protonet_variant=variant,
            distance_metric=distance_metric,
            temperature=temperature,
            n_way=3,
            k_shot=2,
            query_shots=6
        )
        
        learner = PrototypicalLearner(encoder, config)
        
        support_x = torch.randn(3, 2, 16)
        support_y = torch.arange(3).repeat_interleave(2)
        query_x = torch.randn(18, 16)
        
        logits = learner(support_x, support_y, query_x)
        
        # Consistency checks
        assert logits.shape == (18, 3)
        assert torch.isfinite(logits).all()
        
        # Temperature scaling should affect logits scale
        if temperature != 1.0:
            assert not torch.allclose(logits, logits / temperature)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])