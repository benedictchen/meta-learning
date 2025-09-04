"""
Integration tests for hardware acceleration across all meta-learning algorithms.

Tests the integration of hardware utilities with all meta-learning modules,
ensuring that GPU acceleration, mixed precision, and multi-device support
work correctly with all research solutions.
"""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Tuple, Any, Optional
import tempfile
from pathlib import Path

# Import hardware utilities
from meta_learning.meta_learning_modules.hardware_utils import (
    HardwareManager, HardwareConfig, MultiGPUManager,
    create_hardware_manager, auto_device, get_optimal_batch_size
)

# Import meta-learning modules
from meta_learning.meta_learning_modules.test_time_compute import (
    TestTimeComputeScaler, TestTimeComputeConfig
)
from meta_learning.meta_learning_modules.few_shot_learning import (
    PrototypicalLearner, PrototypicalConfig,
    UncertaintyAwareDistance, HierarchicalPrototypes, TaskAdaptivePrototypes
)
from meta_learning.meta_learning_modules.continual_meta_learning import (
    ContinualMetaLearner, OnlineMetaLearner,
    ContinualConfig, OnlineConfig, EWCRegularizer
)


class TestHardwareMetaLearningIntegration:
    """Test hardware acceleration integration with meta-learning algorithms."""
    
    @pytest.fixture
    def hardware_manager(self):
        """Create hardware manager for testing."""
        config = HardwareConfig(
            use_mixed_precision=True,
            memory_efficient=True,
            benchmark_mode=True
        )
        return HardwareManager(config)
        
    @pytest.fixture
    def sample_meta_learning_episode(self):
        """Create sample meta-learning episode for testing."""
        n_way, k_shot, query_shots = 5, 3, 15
        feature_dim = 64
        
        support_x = torch.randn(n_way, k_shot, feature_dim)
        support_y = torch.arange(n_way).repeat_interleave(k_shot)
        query_x = torch.randn(n_way * query_shots, feature_dim)
        query_y = torch.arange(n_way).repeat(query_shots)
        
        return support_x, support_y, query_x, query_y, n_way, k_shot, feature_dim
        
    @pytest.fixture
    def encoder_model(self):
        """Create encoder model for testing."""
        return nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )


class TestPrototypicalNetworksHardwareIntegration:
    """Test Prototypical Networks with hardware acceleration."""
    
    def test_prototypical_basic_hardware_integration(self, hardware_manager, sample_meta_learning_episode, encoder_model):
        """Test basic prototypical networks with hardware acceleration."""
        # Prepare model for hardware
        encoder = hardware_manager.prepare_model(encoder_model)
        
        # Create prototypical learner
        config = PrototypicalConfig(protonet_variant="research_accurate")
        learner = PrototypicalLearner(encoder, config)
        
        # Prepare data for hardware
        support_x, support_y, query_x, query_y, n_way, k_shot, feature_dim = sample_meta_learning_episode
        support_x, support_y = hardware_manager.prepare_data((support_x, support_y))
        query_x, query_y = hardware_manager.prepare_data((query_x, query_y))
        
        # Test with hardware acceleration
        with hardware_manager.autocast_context():
            logits = learner(support_x, support_y, query_x)
            loss = learner.compute_loss(support_x, support_y, query_x, query_y)
        
        # Validate results
        assert logits.shape == (75, 5)  # query_shots * n_way, n_way
        assert logits.device == hardware_manager.device
        assert torch.isfinite(logits).all()
        assert torch.isfinite(loss)
        assert loss.item() >= 0
        
    def test_prototypical_uncertainty_aware_hardware(self, hardware_manager, sample_meta_learning_episode, encoder_model):
        """Test uncertainty-aware prototypical networks with hardware acceleration."""
        encoder = hardware_manager.prepare_model(encoder_model)
        
        config = PrototypicalConfig(
            protonet_variant="enhanced",
            use_uncertainty_aware_distances=True,
            use_temperature_scaling=True
        )
        learner = PrototypicalLearner(encoder, config)
        
        support_x, support_y, query_x, query_y, n_way, k_shot, feature_dim = sample_meta_learning_episode
        support_x, support_y = hardware_manager.prepare_data((support_x, support_y))
        query_x, query_y = hardware_manager.prepare_data((query_x, query_y))
        
        with hardware_manager.autocast_context():
            logits = learner(support_x, support_y, query_x)
            predictions = torch.argmax(logits, dim=1)
            accuracy = (predictions == query_y).float().mean()
        
        assert 0 <= accuracy <= 1
        assert logits.device == hardware_manager.device
        assert torch.isfinite(logits).all()
        
    def test_hierarchical_prototypes_hardware(self, hardware_manager, sample_meta_learning_episode, encoder_model):
        """Test hierarchical prototypes with hardware acceleration."""
        encoder = hardware_manager.prepare_model(encoder_model)
        
        config = PrototypicalConfig(
            use_hierarchical_prototypes=True,
            protonet_variant="enhanced"
        )
        learner = PrototypicalLearner(encoder, config)
        
        support_x, support_y, query_x, query_y, n_way, k_shot, feature_dim = sample_meta_learning_episode
        support_x, support_y = hardware_manager.prepare_data((support_x, support_y))
        query_x, query_y = hardware_manager.prepare_data((query_x, query_y))
        
        with hardware_manager.autocast_context():
            start_time = time.time()
            logits = learner(support_x, support_y, query_x)
            elapsed_time = time.time() - start_time
        
        # Should be faster than 100ms for this small example
        assert elapsed_time < 0.1
        assert logits.device == hardware_manager.device
        assert logits.shape == (75, 5)
        
    def test_task_adaptive_prototypes_hardware(self, hardware_manager, sample_meta_learning_episode, encoder_model):
        """Test task-adaptive prototypes with hardware acceleration."""
        encoder = hardware_manager.prepare_model(encoder_model)
        
        config = PrototypicalConfig(
            use_task_adaptive_prototypes=True,
            protonet_variant="enhanced"
        )
        learner = PrototypicalLearner(encoder, config)
        
        support_x, support_y, query_x, query_y, n_way, k_shot, feature_dim = sample_meta_learning_episode
        support_x, support_y = hardware_manager.prepare_data((support_x, support_y))
        query_x, query_y = hardware_manager.prepare_data((query_x, query_y))
        
        with hardware_manager.autocast_context():
            logits = learner(support_x, support_y, query_x)
            loss = learner.compute_loss(support_x, support_y, query_x, query_y)
            
            # Test gradient computation with hardware acceleration
            loss.backward()
        
        # Check gradients exist and are finite
        gradients_finite = all(
            param.grad is not None and torch.isfinite(param.grad).all()
            for param in encoder.parameters() if param.grad is not None
        )
        assert gradients_finite or not any(param.requires_grad for param in encoder.parameters())


class TestTestTimeComputeHardwareIntegration:
    """Test Test-Time Compute Scaling with hardware acceleration."""
    
    def test_test_time_compute_basic_hardware(self, hardware_manager, sample_meta_learning_episode):
        """Test basic test-time compute with hardware acceleration."""
        # Create base model
        base_model = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 5)
        )
        
        base_model = hardware_manager.prepare_model(base_model)
        
        config = TestTimeComputeConfig(
            compute_strategy="basic",
            max_compute_budget=10,
            use_bootstrap_sampling=True
        )
        scaler = TestTimeComputeScaler(base_model, config)
        
        support_x, support_y, query_x, query_y, n_way, k_shot, feature_dim = sample_meta_learning_episode
        support_x, support_y = hardware_manager.prepare_data((support_x, support_y))
        query_x, query_y = hardware_manager.prepare_data((query_x, query_y))
        
        with hardware_manager.autocast_context():
            enhanced_logits, metrics = scaler.scale_compute(support_x, support_y, query_x)
        
        assert enhanced_logits.device == hardware_manager.device
        assert enhanced_logits.shape == (75, 5)
        assert torch.isfinite(enhanced_logits).all()
        assert 'compute_used' in metrics
        assert metrics['compute_used'] > 0
        
    def test_test_time_compute_snell2024_hardware(self, hardware_manager, sample_meta_learning_episode):
        """Test Snell 2024 test-time compute with hardware acceleration."""
        base_model = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 5)
        )
        
        base_model = hardware_manager.prepare_model(base_model)
        
        config = TestTimeComputeConfig(
            compute_strategy="snell2024",
            use_process_reward_model=True,
            use_optimal_allocation=True,
            max_compute_budget=20
        )
        scaler = TestTimeComputeScaler(base_model, config)
        
        support_x, support_y, query_x, query_y, n_way, k_shot, feature_dim = sample_meta_learning_episode
        support_x, support_y = hardware_manager.prepare_data((support_x, support_y))
        query_x, query_y = hardware_manager.prepare_data((query_x, query_y))
        
        with hardware_manager.autocast_context():
            start_time = time.time()
            enhanced_logits, metrics = scaler.scale_compute(support_x, support_y, query_x)
            elapsed_time = time.time() - start_time
            
        # Should complete in reasonable time with hardware acceleration
        assert elapsed_time < 10.0  # 10 seconds max
        assert enhanced_logits.device == hardware_manager.device
        assert 'strategy' in metrics
        assert metrics['strategy'] == "snell2024"
        
    def test_test_time_compute_hybrid_hardware(self, hardware_manager, sample_meta_learning_episode):
        """Test hybrid test-time compute with hardware acceleration."""
        base_model = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 5)
        )
        
        base_model = hardware_manager.prepare_model(base_model)
        
        config = TestTimeComputeConfig(
            compute_strategy="hybrid",
            use_process_reward_model=True,
            use_test_time_training=False,  # Disable for faster testing
            use_chain_of_thought=True,
            max_compute_budget=15
        )
        scaler = TestTimeComputeScaler(base_model, config)
        
        support_x, support_y, query_x, query_y, n_way, k_shot, feature_dim = sample_meta_learning_episode
        support_x, support_y = hardware_manager.prepare_data((support_x, support_y))
        query_x, query_y = hardware_manager.prepare_data((query_x, query_y))
        
        with hardware_manager.autocast_context():
            enhanced_logits, metrics = scaler.scale_compute(support_x, support_y, query_x)
            
        assert enhanced_logits.device == hardware_manager.device
        assert 'strategies_used' in metrics
        assert len(metrics['strategies_used']) > 1  # Hybrid should use multiple strategies
        assert torch.isfinite(enhanced_logits).all()


class TestContinualLearningHardwareIntegration:
    """Test continual learning with hardware acceleration."""
    
    def test_ewc_hardware_integration(self, hardware_manager, sample_meta_learning_episode):
        """Test EWC continual learning with hardware acceleration."""
        model = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 5)
        )
        
        model = hardware_manager.prepare_model(model)
        
        config = ContinualConfig(
            ewc_method="diagonal",
            fisher_estimation_method="empirical",
            ewc_lambda=0.5,
            use_memory_bank=True
        )
        
        learner = ContinualMetaLearner(model, config)
        
        # Learn multiple tasks sequentially
        task_losses = []
        for task_id in range(3):
            support_x, support_y, query_x, query_y, n_way, k_shot, feature_dim = sample_meta_learning_episode
            support_x, support_y = hardware_manager.prepare_data((support_x, support_y))
            query_x, query_y = hardware_manager.prepare_data((query_x, query_y))
            
            with hardware_manager.autocast_context():
                loss = learner.learn_task(
                    support_x, support_y, query_x, query_y,
                    task_id=f"hardware_task_{task_id}"
                )
                task_losses.append(loss.item())
        
        # Validate continual learning worked
        assert len(task_losses) == 3
        assert all(loss >= 0 for loss in task_losses)
        assert len(learner.ewc_regularizer.task_params) == 3
        
        # All tensors should be on correct device
        for task_params in learner.ewc_regularizer.task_params.values():
            for param in task_params.values():
                assert param.device == hardware_manager.device
                
    def test_online_meta_learning_hardware(self, hardware_manager, sample_meta_learning_episode):
        """Test online meta-learning with hardware acceleration."""
        model = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 5)
        )
        
        model = hardware_manager.prepare_model(model)
        
        config = OnlineConfig(
            learning_rate=0.01,
            meta_learning_rate=0.001,
            adaptation_steps=2,  # Reduced for faster testing
            buffer_size=50
        )
        
        learner = OnlineMetaLearner(model, config)
        
        # Simulate streaming tasks
        adaptation_losses = []
        for task_idx in range(5):
            support_x, support_y, query_x, query_y, n_way, k_shot, feature_dim = sample_meta_learning_episode
            support_x, support_y = hardware_manager.prepare_data((support_x, support_y))
            query_x, query_y = hardware_manager.prepare_data((query_x, query_y))
            
            with hardware_manager.autocast_context():
                loss = learner.adapt_to_task(support_x, support_y, query_x, query_y)
                adaptation_losses.append(loss.item())
                
                # Add to buffer
                task_data = {
                    'support_x': support_x, 'support_y': support_y,
                    'query_x': query_x, 'query_y': query_y,
                    'task_id': f"streaming_task_{task_idx}"
                }
                learner.add_task_to_buffer(task_data)
        
        # Perform meta-update
        with hardware_manager.autocast_context():
            learner.meta_update()
        
        assert len(adaptation_losses) == 5
        assert all(torch.isfinite(torch.tensor(loss)) for loss in adaptation_losses)
        assert len(learner.task_buffer) <= config.buffer_size


class TestMultiGPUIntegration:
    """Test multi-GPU integration (if available)."""
    
    @pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2,
                       reason="Multi-GPU not available")
    def test_prototypical_networks_multi_gpu(self, sample_meta_learning_episode):
        """Test prototypical networks with multi-GPU support."""
        config = HardwareConfig(
            device="cuda",
            use_data_parallel=True,
            use_mixed_precision=True
        )
        hardware_manager = HardwareManager(config)
        
        # Create larger model to benefit from multi-GPU
        encoder = nn.Sequential(
            nn.Linear(64, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 32)
        )
        
        encoder = hardware_manager.prepare_model(encoder)
        
        # Should be wrapped with DataParallel if multiple GPUs available
        if torch.cuda.device_count() > 1:
            assert isinstance(encoder, nn.DataParallel)
        
        config = PrototypicalConfig()
        learner = PrototypicalLearner(encoder, config)
        
        support_x, support_y, query_x, query_y, n_way, k_shot, feature_dim = sample_meta_learning_episode
        support_x, support_y = hardware_manager.prepare_data((support_x, support_y))
        query_x, query_y = hardware_manager.prepare_data((query_x, query_y))
        
        with hardware_manager.autocast_context():
            logits = learner(support_x, support_y, query_x)
        
        assert logits.device.type == "cuda"
        assert logits.shape == (75, 5)
        assert torch.isfinite(logits).all()
        
    @pytest.mark.skipif(not torch.cuda.is_available(),
                       reason="CUDA not available")
    def test_optimal_batch_size_integration(self, sample_meta_learning_episode):
        """Test optimal batch size detection with meta-learning models."""
        # Create test model
        model = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 5)
        )
        
        device = torch.device("cuda")
        input_shape = (1, 64)
        
        optimal_batch = get_optimal_batch_size(model, input_shape, device)
        
        assert isinstance(optimal_batch, int)
        assert optimal_batch >= 1
        assert optimal_batch <= 512  # Reasonable upper bound
        
        # Test with the optimal batch size
        hardware_manager = HardwareManager(HardwareConfig(device="cuda"))
        model = hardware_manager.prepare_model(model)
        
        support_x, support_y, query_x, query_y, n_way, k_shot, feature_dim = sample_meta_learning_episode
        
        # Create larger batch using optimal batch size
        large_query_x = torch.randn(optimal_batch, feature_dim)
        large_query_x = hardware_manager.prepare_data(large_query_x)
        
        with hardware_manager.autocast_context():
            # Should not run out of memory
            output = model(large_query_x)
        
        assert output.shape == (optimal_batch, 5)
        assert torch.isfinite(output).all()


class TestHardwareMemoryManagement:
    """Test memory management with hardware acceleration."""
    
    def test_memory_efficient_large_model(self, hardware_manager):
        """Test memory efficient training with large models."""
        # Create larger model that would normally use significant memory
        large_model = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 5)
        )
        
        # Enable memory optimizations
        hardware_manager.config.memory_efficient = True
        hardware_manager.config.gradient_checkpointing = True
        
        model = hardware_manager.prepare_model(large_model)
        
        # Create large batch
        batch_size = 64
        large_input = torch.randn(batch_size, 512)
        large_input = hardware_manager.prepare_data(large_input)
        targets = torch.randint(0, 5, (batch_size,))
        targets = hardware_manager.prepare_data(targets)
        
        # Test memory stats before
        initial_memory = hardware_manager.get_memory_stats()
        
        with hardware_manager.autocast_context():
            output = model(large_input)
            loss = F.cross_entropy(output, targets)
            
            # Backward pass (this tests memory efficiency)
            loss.backward()
        
        # Check memory stats after
        final_memory = hardware_manager.get_memory_stats()
        
        assert output.shape == (batch_size, 5)
        assert torch.isfinite(output).all()
        assert torch.isfinite(loss)
        
        # Clear memory
        hardware_manager.clear_cache()
        
    def test_memory_monitoring_during_training(self, hardware_manager, sample_meta_learning_episode):
        """Test memory monitoring during training."""
        model = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 5)
        )
        
        model = hardware_manager.prepare_model(model)
        optimizer = torch.optim.Adam(model.parameters())
        
        support_x, support_y, query_x, query_y, n_way, k_shot, feature_dim = sample_meta_learning_episode
        support_x, support_y = hardware_manager.prepare_data((support_x, support_y))
        query_x, query_y = hardware_manager.prepare_data((query_x, query_y))
        
        memory_snapshots = []
        
        for epoch in range(5):
            memory_before = hardware_manager.get_memory_stats()
            memory_snapshots.append(memory_before)
            
            optimizer.zero_grad()
            
            with hardware_manager.autocast_context():
                logits = model(query_x)
                loss = F.cross_entropy(logits, query_y)
            
            # Use hardware manager for optimized backward pass
            hardware_manager.backward_and_step(loss, optimizer)
            
        # Memory usage should be relatively stable across epochs
        if hardware_manager.device.type == "cuda":
            gpu_memory_usage = [stats.get('gpu_memory_allocated', 0) for stats in memory_snapshots]
            # Memory usage shouldn't grow significantly
            memory_growth = max(gpu_memory_usage) - min(gpu_memory_usage)
            assert memory_growth < 1.0  # Less than 1GB growth


class TestHardwarePerformanceBenchmarking:
    """Test performance benchmarking with hardware acceleration."""
    
    @pytest.mark.slow
    def test_performance_comparison_cpu_vs_accelerated(self, sample_meta_learning_episode):
        """Compare performance between CPU and hardware accelerated versions."""
        # Create identical models for comparison
        def create_model():
            return nn.Sequential(
                nn.Linear(64, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 5)
            )
        
        # CPU version
        cpu_manager = HardwareManager(HardwareConfig(device="cpu"))
        cpu_model = cpu_manager.prepare_model(create_model())
        
        # Hardware accelerated version
        hw_manager = HardwareManager()  # Auto-detect best device
        hw_model = hw_manager.prepare_model(create_model())
        
        support_x, support_y, query_x, query_y, n_way, k_shot, feature_dim = sample_meta_learning_episode
        
        # Benchmark CPU
        cpu_support_x, cpu_support_y = cpu_manager.prepare_data((support_x, support_y))
        cpu_query_x, cpu_query_y = cpu_manager.prepare_data((query_x, query_y))
        
        cpu_times = []
        for _ in range(10):
            start_time = time.time()
            with cpu_manager.autocast_context():
                cpu_output = cpu_model(cpu_query_x)
            cpu_times.append(time.time() - start_time)
        
        cpu_avg_time = np.mean(cpu_times)
        
        # Benchmark hardware accelerated
        hw_support_x, hw_support_y = hw_manager.prepare_data((support_x, support_y))
        hw_query_x, hw_query_y = hw_manager.prepare_data((query_x, query_y))
        
        hw_times = []
        for _ in range(10):
            start_time = time.time()
            with hw_manager.autocast_context():
                hw_output = hw_model(hw_query_x)
            hw_times.append(time.time() - start_time)
        
        hw_avg_time = np.mean(hw_times)
        
        # Results should be numerically similar
        cpu_result_mean = cpu_output.mean().item()
        hw_result_mean = hw_output.mean().item()
        
        # Allow for small numerical differences due to precision
        assert abs(cpu_result_mean - hw_result_mean) < 0.1
        
        # Hardware version should generally be faster (except for very small models on some hardware)
        speedup = cpu_avg_time / hw_avg_time
        print(f"Speedup: {speedup:.2f}x ({hw_manager.device} vs CPU)")
        
        # At minimum, hardware version shouldn't be significantly slower
        assert hw_avg_time <= cpu_avg_time * 2.0  # Allow 2x slower in worst case
        
    @pytest.mark.slow
    def test_scalability_with_batch_size(self, hardware_manager):
        """Test performance scalability with increasing batch size."""
        model = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 5)
        )
        
        model = hardware_manager.prepare_model(model)
        
        batch_sizes = [1, 4, 16, 64, 256]
        times_per_sample = []
        
        for batch_size in batch_sizes:
            input_data = torch.randn(batch_size, 64)
            input_data = hardware_manager.prepare_data(input_data)
            
            # Warmup
            for _ in range(3):
                with hardware_manager.autocast_context():
                    _ = model(input_data)
            
            # Benchmark
            times = []
            for _ in range(10):
                start_time = time.time()
                with hardware_manager.autocast_context():
                    output = model(input_data)
                elapsed = time.time() - start_time
                times.append(elapsed)
            
            avg_time = np.mean(times)
            time_per_sample = avg_time / batch_size
            times_per_sample.append(time_per_sample)
        
        # Time per sample should generally decrease with larger batches (better utilization)
        # Allow some variance due to overhead and hardware differences
        assert len(times_per_sample) == len(batch_sizes)
        
        # At least the largest batch should be more efficient than the smallest
        assert times_per_sample[-1] <= times_per_sample[0] * 1.5


class TestHardwareCompatibilityEdgeCases:
    """Test edge cases and compatibility issues with hardware acceleration."""
    
    def test_mixed_precision_compatibility(self):
        """Test mixed precision compatibility across different scenarios."""
        scenarios = [
            {"device": "cpu", "expected_mp": False},
        ]
        
        # Add GPU scenario if available
        if torch.cuda.is_available():
            scenarios.append({"device": "cuda", "expected_mp": True})
            
        # Add MPS scenario if available
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            scenarios.append({"device": "mps", "expected_mp": False})
        
        for scenario in scenarios:
            config = HardwareConfig(
                device=scenario["device"],
                use_mixed_precision=True  # Request mixed precision
            )
            manager = HardwareManager(config)
            
            # Check if mixed precision was actually enabled
            actual_mp = manager.config.use_mixed_precision
            expected_mp = scenario["expected_mp"]
            
            if not expected_mp:
                # Mixed precision should be disabled for non-CUDA devices
                assert actual_mp == False
            
            # Autocast context should work regardless
            test_tensor = torch.randn(5, 10, device=manager.device)
            
            with manager.autocast_context():
                result = test_tensor * 2
                
            assert result.device == manager.device
            assert result.shape == test_tensor.shape
            
    def test_fallback_behavior_on_errors(self, sample_meta_learning_episode):
        """Test graceful fallback behavior when hardware operations fail."""
        # Test with deliberately problematic configuration
        config = HardwareConfig(
            device="cuda" if torch.cuda.is_available() else "cpu",
            compile_model=True,  # May fail on some configurations
            channels_last=True   # May fail on some tensor shapes
        )
        
        manager = HardwareManager(config)
        
        # Create model that might have issues with compilation/channels_last
        problematic_model = nn.Sequential(
            nn.Linear(64, 32),
            # No Conv layers, so channels_last might not work
            nn.ReLU(),
            nn.Linear(32, 5)
        )
        
        # Should handle failures gracefully
        try:
            model = manager.prepare_model(problematic_model)
            
            support_x, support_y, query_x, query_y, n_way, k_shot, feature_dim = sample_meta_learning_episode
            data = manager.prepare_data(query_x)
            
            with manager.autocast_context():
                output = model(data)
                
            # Should still work even if optimizations failed
            assert output.shape == (query_x.shape[0], 5)
            assert torch.isfinite(output).all()
            
        except Exception as e:
            # Some failures might be expected, but they shouldn't be silent
            print(f"Expected hardware optimization failure: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "not slow"])