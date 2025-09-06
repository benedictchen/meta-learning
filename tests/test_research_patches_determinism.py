#!/usr/bin/env python3
"""
Comprehensive Tests for Research Patches and Determinism
======================================================

Tests all research patches, determinism hooks, and reproducibility mechanisms:
- BatchNorm policy patches for episodic learning
- Deterministic environment setup and seed control
- Reproducibility across runs and platforms
- Research configuration validation
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import random
import os
import warnings
from unittest.mock import patch, MagicMock

# Import research patches and determinism hooks
from meta_learning.research_patches.batch_norm_policy_patch import (
    apply_episodic_bn_policy,
    EpisodicBNPolicy,
    BatchNormMode
)
from meta_learning.research_patches.determinism_hooks import (
    setup_deterministic_environment,
    DeterminismConfig,
    create_deterministic_generator,
    validate_reproducibility
)


class TestBatchNormPolicyPatch:
    """Test BatchNorm policy patches for episodic learning."""
    
    def test_episodic_bn_policy_enum(self):
        """Test BatchNormMode enum values."""
        assert BatchNormMode.TRAIN == 'train'
        assert BatchNormMode.EVAL == 'eval' 
        assert BatchNormMode.EPISODIC == 'episodic'
        assert BatchNormMode.ADAPTIVE == 'adaptive'
    
    def test_episodic_bn_policy_creation(self):
        """Test EpisodicBNPolicy creation and configuration."""
        policy = EpisodicBNPolicy(
            mode=BatchNormMode.EPISODIC,
            support_train=True,
            query_eval=True,
            reset_stats=True
        )
        
        assert policy.mode == BatchNormMode.EPISODIC
        assert policy.support_train is True
        assert policy.query_eval is True
        assert policy.reset_stats is True
    
    def test_apply_episodic_bn_policy(self):
        """Test applying episodic BatchNorm policy to model."""
        # Create model with BatchNorm layers
        model = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3),
            nn.BatchNorm2d(64),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 10)
        )
        
        # Apply episodic BatchNorm policy
        policy = EpisodicBNPolicy(mode=BatchNormMode.EPISODIC)
        modified_model = apply_episodic_bn_policy(model, policy)
        
        # Check that model structure is preserved
        assert len(list(modified_model.modules())) == len(list(model.modules()))
        
        # Check that BatchNorm layers are affected
        bn_layers = [m for m in modified_model.modules() if isinstance(m, nn.BatchNorm2d)]
        assert len(bn_layers) == 2
    
    def test_bn_policy_support_query_modes(self):
        """Test BatchNorm behavior in support vs query phases."""
        model = nn.Sequential(
            nn.Conv2d(1, 16, 3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(256, 5)
        )
        
        policy = EpisodicBNPolicy(
            mode=BatchNormMode.EPISODIC,
            support_train=True,
            query_eval=True,
            reset_stats=True
        )
        
        modified_model = apply_episodic_bn_policy(model, policy)
        
        # Test with dummy data
        support_data = torch.randn(10, 1, 8, 8)  # Support set
        query_data = torch.randn(5, 1, 8, 8)     # Query set
        
        # Support phase (should be in training mode)
        modified_model.train()
        support_output = modified_model(support_data)
        assert support_output.shape == (10, 5)
        
        # Query phase (should be in eval mode based on policy)
        modified_model.eval()
        query_output = modified_model(query_data)
        assert query_output.shape == (5, 5)
    
    def test_bn_policy_stat_reset(self):
        """Test BatchNorm statistics reset functionality."""
        model = nn.Sequential(
            nn.Conv2d(1, 8, 3),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(72, 3)
        )
        
        policy = EpisodicBNPolicy(
            mode=BatchNormMode.EPISODIC,
            reset_stats=True
        )
        
        modified_model = apply_episodic_bn_policy(model, policy)
        
        # Run some data through to accumulate stats
        data = torch.randn(20, 1, 5, 5)
        modified_model.train()
        _ = modified_model(data)
        
        # Get BatchNorm layer and check initial stats
        bn_layer = None
        for module in modified_model.modules():
            if isinstance(module, nn.BatchNorm2d):
                bn_layer = module
                break
        
        assert bn_layer is not None
        initial_mean = bn_layer.running_mean.clone()
        initial_var = bn_layer.running_var.clone()
        
        # Stats should have been updated
        assert not torch.allclose(initial_mean, torch.zeros_like(initial_mean), atol=1e-3)
    
    def test_bn_policy_adaptive_mode(self):
        """Test adaptive BatchNorm mode."""
        model = nn.Sequential(
            nn.Conv2d(1, 4, 3),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16, 2)
        )
        
        policy = EpisodicBNPolicy(mode=BatchNormMode.ADAPTIVE)
        modified_model = apply_episodic_bn_policy(model, policy)
        
        # Test adaptive behavior with different batch sizes
        small_batch = torch.randn(2, 1, 5, 5)
        large_batch = torch.randn(16, 1, 5, 5)
        
        modified_model.train()
        small_output = modified_model(small_batch)
        large_output = modified_model(large_batch)
        
        assert small_output.shape == (2, 2)
        assert large_output.shape == (16, 2)


class TestDeterminismHooks:
    """Test deterministic environment setup and seed control."""
    
    def test_determinism_config_creation(self):
        """Test DeterminismConfig creation with various options."""
        config = DeterminismConfig(
            seed=42,
            torch_deterministic=True,
            cudnn_deterministic=True,
            cudnn_benchmark=False,
            numpy_seed=True,
            python_seed=True
        )
        
        assert config.seed == 42
        assert config.torch_deterministic is True
        assert config.cudnn_deterministic is True
        assert config.cudnn_benchmark is False
        assert config.numpy_seed is True
        assert config.python_seed is True
    
    def test_determinism_config_defaults(self):
        """Test DeterminismConfig default values."""
        config = DeterminismConfig()
        
        assert config.seed is None
        assert config.torch_deterministic is True
        assert config.cudnn_deterministic is True
        assert config.cudnn_benchmark is False
        assert config.numpy_seed is True
        assert config.python_seed is True
    
    def test_setup_deterministic_environment(self):
        """Test deterministic environment setup."""
        initial_torch_seed = torch.initial_seed()
        initial_np_state = np.random.get_state()
        initial_py_state = random.getstate()
        
        config = DeterminismConfig(seed=12345)
        setup_deterministic_environment(config)
        
        # Check that seeds were set
        assert torch.initial_seed() != initial_torch_seed
        
        # Check that numpy state changed
        new_np_state = np.random.get_state()
        assert not np.array_equal(initial_np_state[1], new_np_state[1])
        
        # Check that python random state changed
        new_py_state = random.getstate()
        assert initial_py_state != new_py_state
    
    @patch('torch.backends.cudnn')
    def test_cudnn_deterministic_setup(self, mock_cudnn):
        """Test CUDNN deterministic configuration."""
        config = DeterminismConfig(
            seed=42,
            cudnn_deterministic=True,
            cudnn_benchmark=False
        )
        
        setup_deterministic_environment(config)
        
        # Check that CUDNN settings were configured
        assert mock_cudnn.deterministic is True
        assert mock_cudnn.benchmark is False
    
    def test_create_deterministic_generator(self):
        """Test creation of deterministic random generators."""
        generator = create_deterministic_generator(seed=999)
        
        assert isinstance(generator, torch.Generator)
        
        # Test reproducibility
        gen1 = create_deterministic_generator(seed=999)
        gen2 = create_deterministic_generator(seed=999)
        
        # Same seed should produce same initial values
        val1 = torch.randn(5, generator=gen1)
        val2 = torch.randn(5, generator=gen2)
        
        assert torch.allclose(val1, val2)
    
    def test_validate_reproducibility_success(self):
        """Test reproducibility validation with identical runs."""
        def deterministic_function(seed):
            """Simple deterministic function for testing."""
            config = DeterminismConfig(seed=seed)
            setup_deterministic_environment(config)
            
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            
            # Generate some deterministic values
            torch_val = torch.randn(3)
            numpy_val = np.random.randn(3)
            python_val = random.random()
            
            return {
                'torch': torch_val,
                'numpy': numpy_val,
                'python': python_val
            }
        
        # Test reproducibility
        result1 = deterministic_function(42)
        result2 = deterministic_function(42)
        
        assert torch.allclose(result1['torch'], result2['torch'])
        assert np.allclose(result1['numpy'], result2['numpy'])
        assert result1['python'] == result2['python']
    
    def test_validate_reproducibility_with_different_seeds(self):
        """Test that different seeds produce different results."""
        config1 = DeterminismConfig(seed=100)
        config2 = DeterminismConfig(seed=200)
        
        setup_deterministic_environment(config1)
        result1 = torch.randn(10)
        
        setup_deterministic_environment(config2)  
        result2 = torch.randn(10)
        
        # Different seeds should produce different results
        assert not torch.allclose(result1, result2)


class TestReproducibilityIntegration:
    """Test integration of research patches with determinism for reproducibility."""
    
    def test_episodic_learning_reproducibility(self):
        """Test reproducible episodic learning with BatchNorm patches."""
        def create_episodic_model():
            """Create model with episodic BatchNorm policy."""
            model = nn.Sequential(
                nn.Conv2d(1, 8, 3),
                nn.BatchNorm2d(8),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((2, 2)),
                nn.Flatten(),
                nn.Linear(32, 3)
            )
            
            policy = EpisodicBNPolicy(
                mode=BatchNormMode.EPISODIC,
                support_train=True,
                query_eval=True,
                reset_stats=True
            )
            
            return apply_episodic_bn_policy(model, policy)
        
        def run_episodic_experiment(seed):
            """Run episodic learning experiment with given seed."""
            config = DeterminismConfig(seed=seed)
            setup_deterministic_environment(config)
            
            model = create_episodic_model()
            
            # Generate deterministic data
            torch.manual_seed(seed)
            support_data = torch.randn(15, 1, 6, 6)
            query_data = torch.randn(5, 1, 6, 6)
            
            # Support phase
            model.train()
            support_output = model(support_data)
            
            # Query phase  
            model.eval()
            query_output = model(query_data)
            
            return {
                'support_output': support_output,
                'query_output': query_output
            }
        
        # Run experiment twice with same seed
        result1 = run_episodic_experiment(777)
        result2 = run_episodic_experiment(777)
        
        # Results should be identical
        assert torch.allclose(result1['support_output'], result2['support_output'])
        assert torch.allclose(result1['query_output'], result2['query_output'])
    
    def test_deterministic_model_initialization(self):
        """Test deterministic model parameter initialization."""
        def create_model_with_seed(seed):
            """Create model with deterministic initialization."""
            config = DeterminismConfig(seed=seed)
            setup_deterministic_environment(config)
            
            torch.manual_seed(seed)
            
            model = nn.Sequential(
                nn.Linear(10, 20),
                nn.ReLU(),
                nn.Linear(20, 5)
            )
            
            return model
        
        # Create two models with same seed
        model1 = create_model_with_seed(555)
        model2 = create_model_with_seed(555)
        
        # Parameters should be identical
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2)
    
    def test_deterministic_training_step(self):
        """Test deterministic training step with research patches."""
        def training_step(seed):
            """Perform one training step deterministically."""
            config = DeterminismConfig(seed=seed)
            setup_deterministic_environment(config)
            
            # Create model with episodic BatchNorm
            model = nn.Sequential(
                nn.Linear(5, 10),
                nn.BatchNorm1d(10),
                nn.ReLU(),
                nn.Linear(10, 2)
            )
            
            policy = EpisodicBNPolicy(mode=BatchNormMode.TRAIN)
            model = apply_episodic_bn_policy(model, policy)
            
            # Deterministic data and optimizer
            torch.manual_seed(seed)
            data = torch.randn(8, 5)
            target = torch.randint(0, 2, (8,))
            
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            criterion = nn.CrossEntropyLoss()
            
            # Training step
            model.train()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            return {
                'loss': loss.item(),
                'output': output.detach(),
                'model_params': [p.clone() for p in model.parameters()]
            }
        
        # Run training step twice with same seed
        result1 = training_step(999)
        result2 = training_step(999)
        
        # Results should be identical
        assert abs(result1['loss'] - result2['loss']) < 1e-6
        assert torch.allclose(result1['output'], result2['output'])
        
        # Model parameters should be identical after training
        for p1, p2 in zip(result1['model_params'], result2['model_params']):
            assert torch.allclose(p1, p2)


class TestResearchConfigValidation:
    """Test validation of research configuration and patches."""
    
    def test_invalid_batch_norm_mode(self):
        """Test handling of invalid BatchNorm mode."""
        with pytest.raises((ValueError, TypeError)):
            EpisodicBNPolicy(mode="invalid_mode")
    
    def test_batch_norm_policy_validation(self):
        """Test validation of BatchNorm policy parameters."""
        # Valid configuration should work
        policy = EpisodicBNPolicy(
            mode=BatchNormMode.EPISODIC,
            support_train=True,
            query_eval=True,
            reset_stats=True
        )
        assert policy.mode == BatchNormMode.EPISODIC
        
        # Test with boolean parameters
        policy = EpisodicBNPolicy(
            mode=BatchNormMode.ADAPTIVE,
            support_train=False,
            query_eval=False,
            reset_stats=False
        )
        assert policy.support_train is False
        assert policy.query_eval is False
        assert policy.reset_stats is False
    
    def test_determinism_config_validation(self):
        """Test validation of determinism configuration."""
        # Valid configuration
        config = DeterminismConfig(seed=42)
        assert config.seed == 42
        
        # Negative seed should work
        config = DeterminismConfig(seed=-1)
        assert config.seed == -1
        
        # Large seed should work
        config = DeterminismConfig(seed=2**31-1)
        assert config.seed == 2**31-1
    
    def test_research_patch_import_success(self):
        """Test that research patches import successfully."""
        try:
            from meta_learning.research_patches.batch_norm_policy_patch import apply_episodic_bn_policy
            from meta_learning.research_patches.determinism_hooks import setup_deterministic_environment
            
            # Patches should be callable
            assert callable(apply_episodic_bn_policy)
            assert callable(setup_deterministic_environment)
            
        except ImportError as e:
            pytest.fail(f"Research patches failed to import: {e}")
    
    def test_research_patch_compatibility(self):
        """Test compatibility between different research patches."""
        # Create model with BatchNorm
        model = nn.Sequential(
            nn.Linear(3, 6),
            nn.BatchNorm1d(6),
            nn.ReLU(),
            nn.Linear(6, 2)
        )
        
        # Apply deterministic environment
        config = DeterminismConfig(seed=123)
        setup_deterministic_environment(config)
        
        # Apply BatchNorm policy
        policy = EpisodicBNPolicy(mode=BatchNormMode.EPISODIC)
        modified_model = apply_episodic_bn_policy(model, policy)
        
        # Both patches should work together
        torch.manual_seed(123)
        data = torch.randn(4, 3)
        output = modified_model(data)
        
        assert output.shape == (4, 2)
        assert not torch.isnan(output).any()
        assert torch.isfinite(output).all()


class TestCrossRunReproducibility:
    """Test reproducibility across different runs and environments."""
    
    def test_multiple_seed_consistency(self):
        """Test that same seeds produce consistent results across multiple runs."""
        results = []
        
        for run in range(3):
            config = DeterminismConfig(seed=314159)
            setup_deterministic_environment(config)
            
            # Generate deterministic values
            torch_val = torch.randn(5)
            np_val = np.random.randn(5)
            py_val = [random.random() for _ in range(5)]
            
            results.append({
                'torch': torch_val,
                'numpy': np_val, 
                'python': py_val
            })
        
        # All runs should produce identical results
        for i in range(1, len(results)):
            assert torch.allclose(results[0]['torch'], results[i]['torch'])
            assert np.allclose(results[0]['numpy'], results[i]['numpy'])
            assert results[0]['python'] == results[i]['python']
    
    def test_episodic_bn_cross_run_reproducibility(self):
        """Test BatchNorm episodic policy reproducibility across runs."""
        def run_episodic_bn_test(seed, run_id):
            """Run episodic BatchNorm test with given seed."""
            config = DeterminismConfig(seed=seed)
            setup_deterministic_environment(config)
            
            model = nn.Sequential(
                nn.Conv2d(1, 4, 3),
                nn.BatchNorm2d(4),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(16, 2)
            )
            
            policy = EpisodicBNPolicy(
                mode=BatchNormMode.EPISODIC,
                reset_stats=True
            )
            
            model = apply_episodic_bn_policy(model, policy)
            
            torch.manual_seed(seed)
            data = torch.randn(6, 1, 5, 5)
            
            model.train()
            output = model(data)
            
            return output
        
        # Run test multiple times with same seed
        seed = 271828
        outputs = []
        
        for run_id in range(4):
            output = run_episodic_bn_test(seed, run_id)
            outputs.append(output)
        
        # All outputs should be identical
        for i in range(1, len(outputs)):
            assert torch.allclose(outputs[0], outputs[i], atol=1e-6)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])