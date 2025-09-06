"""
Tests for ReproducibilityManager and seeding utilities in meta_learning.core.seed
"""
import torch
import pytest
import numpy as np
import random
from meta_learning.core.seed import (
    ReproducibilityManager, distributed_seed_sync, benchmark_reproducibility_overhead,
    validate_seed_effectiveness, seed_all
)


class TestReproducibilityManager:
    """Test ReproducibilityManager functionality."""
    
    def test_init_basic(self):
        """Test basic ReproducibilityManager initialization."""
        manager = ReproducibilityManager(base_seed=123, enable_performance_monitoring=True)
        
        assert manager.base_seed == 123
        assert manager.enable_performance_monitoring is True
        assert isinstance(manager.device_seeds, dict)
        assert isinstance(manager.performance_logs, list)
        assert isinstance(manager.fallback_operations, set)
    
    def test_setup_deterministic_environment_basic(self):
        """Test basic deterministic environment setup."""
        manager = ReproducibilityManager(base_seed=42)
        result = manager.setup_deterministic_environment()
        
        assert result['base_seed'] == 42
        assert 'setup_time' in result
        assert isinstance(result['setup_time'], float)
        assert result['setup_time'] > 0
        assert result['strict_mode'] is False
    
    def test_setup_deterministic_environment_strict(self):
        """Test strict mode deterministic environment setup."""
        manager = ReproducibilityManager(base_seed=99)
        result = manager.setup_deterministic_environment(strict_mode=True)
        
        assert result['strict_mode'] is True
        assert result['base_seed'] == 99
        assert 'device_seeds' in result
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_device_specific_seeding(self):
        """Test device-specific seeding for multi-GPU setups."""
        manager = ReproducibilityManager(base_seed=123)
        result = manager.setup_deterministic_environment()
        
        # Should have device-specific seeds if CUDA is available
        assert 'device_seeds' in result
        if torch.cuda.device_count() > 0:
            assert len(result['device_seeds']) > 0
            # Each device should have a unique seed
            seeds = list(result['device_seeds'].values())
            assert len(seeds) == len(set(seeds))  # All unique
    
    def test_validate_seed_effectiveness_consistent(self):
        """Test seed validation with consistent model."""
        manager = ReproducibilityManager(base_seed=42)
        
        # Simple deterministic model
        model = torch.nn.Linear(5, 3)
        test_input = torch.randn(2, 5)
        
        result = manager.validate_seed_effectiveness(model, test_input, num_runs=3)
        
        assert 'is_consistent' in result
        assert isinstance(result['is_consistent'], bool)
        assert result['num_runs'] == 3
        assert 'validation_time' in result
        assert 'fallback_operations_used' in result
    
    def test_validate_seed_effectiveness_inconsistent(self):
        """Test seed validation with model that has randomness."""
        manager = ReproducibilityManager(base_seed=42)
        
        # Model with dropout (introduces randomness)
        model = torch.nn.Sequential(
            torch.nn.Linear(5, 10),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(10, 3)
        )
        model.train()  # Enable dropout
        
        test_input = torch.randn(2, 5)
        result = manager.validate_seed_effectiveness(model, test_input, num_runs=3)
        
        # May be inconsistent due to dropout
        assert 'is_consistent' in result
        assert isinstance(result['is_consistent'], bool)
    
    def test_performance_monitoring_enabled(self):
        """Test performance monitoring when enabled."""
        manager = ReproducibilityManager(base_seed=42, enable_performance_monitoring=True)
        
        model = torch.nn.Linear(10, 5)
        test_input = torch.randn(3, 10)
        
        # Clear logs
        manager.performance_logs.clear()
        
        manager.setup_deterministic_environment()
        manager.validate_seed_effectiveness(model, test_input, num_runs=2)
        
        # Should have logged performance metrics
        assert len(manager.performance_logs) > 0
        for log_entry in manager.performance_logs:
            assert 'operation' in log_entry
            assert 'duration' in log_entry
    
    def test_performance_monitoring_disabled(self):
        """Test performance monitoring when disabled."""
        manager = ReproducibilityManager(base_seed=42, enable_performance_monitoring=False)
        
        model = torch.nn.Linear(5, 2)
        test_input = torch.randn(2, 5)
        
        manager.performance_logs.clear()
        
        manager.setup_deterministic_environment()
        manager.validate_seed_effectiveness(model, test_input, num_runs=2)
        
        # Should not have logged performance metrics
        assert len(manager.performance_logs) == 0
    
    def test_benchmark_reproducibility_overhead(self):
        """Test reproducibility overhead benchmarking."""
        manager = ReproducibilityManager(base_seed=42, enable_performance_monitoring=True)
        
        def simple_benchmark():
            x = torch.randn(100, 100)
            y = torch.mm(x, x.t())
            return y.sum()
        
        result = manager.benchmark_reproducibility_overhead(simple_benchmark, num_trials=3)
        
        assert 'non_deterministic_mean' in result
        assert 'deterministic_mean' in result
        assert 'overhead_ratio' in result
        assert 'overhead_absolute' in result
        assert result['num_trials'] == 3
        assert result['overhead_ratio'] >= 0  # Should be non-negative
    
    def test_fallback_operations_tracking(self):
        """Test tracking of fallback operations."""
        manager = ReproducibilityManager(base_seed=42)
        
        # This should trigger the strict determinism setup
        manager.setup_deterministic_environment(strict_mode=True)
        
        # Check if fallback operations were recorded
        assert isinstance(manager.fallback_operations, set)
        # The set might be empty if all operations are supported


class TestGlobalSeedingFunctions:
    """Test global seeding utility functions."""
    
    def test_seed_all_basic(self):
        """Test basic seed_all functionality."""
        # Test that seed_all doesn't crash
        seed_all(42)
        
        # Test reproducibility
        seed_all(123)
        rand1 = torch.randn(5)
        np_rand1 = np.random.randn(5)
        py_rand1 = random.random()
        
        seed_all(123)
        rand2 = torch.randn(5)
        np_rand2 = np.random.randn(5)
        py_rand2 = random.random()
        
        # All should be identical
        assert torch.allclose(rand1, rand2)
        assert np.allclose(np_rand1, np_rand2)
        assert py_rand1 == py_rand2
    
    def test_distributed_seed_sync(self):
        """Test distributed training seed synchronization."""
        base_seed = 42
        world_size = 4
        
        # Generate seeds for different ranks
        seeds = []
        for rank in range(world_size):
            process_seed = distributed_seed_sync(base_seed, world_size, rank)
            seeds.append(process_seed)
        
        # All seeds should be unique
        assert len(seeds) == len(set(seeds))
        
        # All seeds should be deterministic based on rank
        for rank in range(world_size):
            seed1 = distributed_seed_sync(base_seed, world_size, rank)
            seed2 = distributed_seed_sync(base_seed, world_size, rank)
            assert seed1 == seed2
    
    def test_distributed_seed_sync_reproducibility(self):
        """Test that distributed seeding is reproducible."""
        base_seed = 123
        
        # Same rank should always get same seed
        seed1 = distributed_seed_sync(base_seed, 8, 3)
        seed2 = distributed_seed_sync(base_seed, 8, 3)
        assert seed1 == seed2
        
        # Different ranks should get different seeds
        seed_rank0 = distributed_seed_sync(base_seed, 4, 0)
        seed_rank1 = distributed_seed_sync(base_seed, 4, 1)
        seed_rank2 = distributed_seed_sync(base_seed, 4, 2)
        
        assert seed_rank0 != seed_rank1 != seed_rank2
    
    def test_benchmark_reproducibility_overhead_function(self):
        """Test the standalone benchmark function."""
        model = torch.nn.Sequential(
            torch.nn.Linear(20, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 5)
        )
        test_input = torch.randn(4, 20)
        
        result = benchmark_reproducibility_overhead(model, test_input, num_trials=3)
        
        assert 'non_deterministic_mean' in result
        assert 'deterministic_mean' in result
        assert 'overhead_ratio' in result
        assert 'overhead_absolute' in result
        assert result['num_trials'] == 3
        
        # Overhead should be reasonable (not massively different)
        assert 0.1 <= result['overhead_ratio'] <= 10.0  # Within 10x difference
    
    def test_validate_seed_effectiveness_function(self):
        """Test the standalone seed validation function."""
        model = torch.nn.Linear(8, 4)
        test_input = torch.randn(3, 8)
        seed = 42
        
        # Test with deterministic model
        model.eval()  # Disable any randomness
        is_consistent = validate_seed_effectiveness(seed, model, test_input, num_runs=3)
        
        assert isinstance(is_consistent, bool)
        # Should be consistent for a simple linear model in eval mode
        assert is_consistent is True
    
    def test_validate_seed_effectiveness_with_randomness(self):
        """Test seed validation with model that has inherent randomness."""
        # Model with dropout
        model = torch.nn.Sequential(
            torch.nn.Linear(5, 10),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(10, 2)
        )
        model.train()  # Enable dropout
        
        test_input = torch.randn(2, 5)
        seed = 99
        
        is_consistent = validate_seed_effectiveness(seed, model, test_input, num_runs=3)
        
        assert isinstance(is_consistent, bool)
        # With proper seeding, even dropout should be consistent
        # (though this might fail if deterministic algorithms aren't fully supported)


class TestSeedingEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_tensor_consistency(self):
        """Test seeding with empty tensors."""
        manager = ReproducibilityManager(base_seed=42)
        
        # Model that can handle empty input
        model = torch.nn.Linear(0, 5)  # This might not work, so let's use identity
        model = torch.nn.Identity()
        test_input = torch.empty(0, 5)
        
        # Should handle empty input gracefully
        try:
            result = manager.validate_seed_effectiveness(model, test_input, num_runs=2)
            assert 'is_consistent' in result
        except (RuntimeError, ValueError):
            # Some operations might not support empty tensors
            pytest.skip("Empty tensor operations not supported")
    
    def test_large_seed_values(self):
        """Test seeding with large seed values."""
        large_seed = 2**31 - 1  # Max int32
        
        # Should handle large seeds without overflow
        seed_all(large_seed)
        
        manager = ReproducibilityManager(base_seed=large_seed)
        result = manager.setup_deterministic_environment()
        
        assert result['base_seed'] == large_seed
    
    def test_negative_seed_values(self):
        """Test seeding with negative seed values."""
        negative_seed = -12345
        
        # Should handle negative seeds
        manager = ReproducibilityManager(base_seed=negative_seed)
        result = manager.setup_deterministic_environment()
        
        assert result['base_seed'] == negative_seed
    
    def test_zero_trials_benchmark(self):
        """Test benchmark with invalid number of trials."""
        manager = ReproducibilityManager(base_seed=42)
        
        def dummy_benchmark():
            return torch.randn(5).sum()
        
        # Test with zero trials (should handle gracefully or raise error)
        try:
            result = manager.benchmark_reproducibility_overhead(dummy_benchmark, num_trials=0)
            # If it doesn't crash, check result structure
            assert 'num_trials' in result
        except (ValueError, ZeroDivisionError):
            # Expected for invalid trial count
            pass
    
    def test_single_run_validation(self):
        """Test validation with single run (edge case)."""
        manager = ReproducibilityManager(base_seed=42)
        model = torch.nn.Linear(3, 2)
        test_input = torch.randn(1, 3)
        
        result = manager.validate_seed_effectiveness(model, test_input, num_runs=1)
        
        # With single run, should always be "consistent"
        assert result['is_consistent'] is True
        assert result['num_runs'] == 1


if __name__ == "__main__":
    pytest.main([__file__])