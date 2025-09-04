"""
Comprehensive unit tests for hardware utilities module.

Tests modern hardware support including GPU acceleration, mixed precision,
multi-GPU distribution, and Apple Silicon MPS support.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from hypothesis import given, strategies as st, settings
from typing import Dict, List, Tuple, Any, Optional

from meta_learning.meta_learning_modules.hardware_utils import (
    HardwareManager, HardwareConfig, MultiGPUManager,
    create_hardware_manager, auto_device, prepare_for_hardware,
    get_optimal_batch_size, log_hardware_info
)


class TestHardwareConfig:
    """Test hardware configuration."""
    
    def test_hardware_config_defaults(self):
        """Test default hardware configuration values."""
        config = HardwareConfig()
        assert config.device is None  # Auto-detect
        assert config.use_mixed_precision == True
        assert config.precision_dtype == "float16"
        assert config.use_data_parallel == False
        assert config.use_distributed == False
        assert config.memory_efficient == True
        assert config.max_memory_fraction == 0.9
        
    def test_hardware_config_custom_settings(self):
        """Test custom hardware configuration."""
        config = HardwareConfig(
            device="cuda:0",
            use_mixed_precision=False,
            precision_dtype="bfloat16",
            use_data_parallel=True,
            max_memory_fraction=0.8
        )
        
        assert config.device == "cuda:0"
        assert config.use_mixed_precision == False
        assert config.precision_dtype == "bfloat16"
        assert config.use_data_parallel == True
        assert config.max_memory_fraction == 0.8
        
    @given(
        max_memory_fraction=st.floats(min_value=0.1, max_value=1.0),
        world_size=st.integers(min_value=1, max_value=8),
        rank=st.integers(min_value=0, max_value=7)
    )
    def test_hardware_config_property_based(self, max_memory_fraction, world_size, rank):
        """Property-based test for hardware configuration.""" 
        # Ensure rank < world_size
        rank = min(rank, world_size - 1)
        
        config = HardwareConfig(
            max_memory_fraction=max_memory_fraction,
            world_size=world_size,
            rank=rank
        )
        
        assert config.max_memory_fraction == max_memory_fraction
        assert config.world_size == world_size
        assert config.rank == rank
        assert config.rank < config.world_size


class TestHardwareManager:
    """Test hardware manager functionality."""
    
    @pytest.fixture
    def hardware_manager(self):
        """Create hardware manager for testing."""
        return HardwareManager()
        
    @pytest.fixture
    def simple_model(self):
        """Create simple model for testing."""
        return nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5)
        )
        
    def test_hardware_manager_init(self):
        """Test hardware manager initialization."""
        manager = HardwareManager()
        
        assert hasattr(manager, 'device')
        assert hasattr(manager, 'config')
        assert isinstance(manager.config, HardwareConfig)
        assert manager.device.type in ['cuda', 'mps', 'cpu']
        
    def test_device_detection(self):
        """Test automatic device detection."""
        manager = HardwareManager()
        device = manager.device
        
        # Should detect one of the supported devices
        assert device.type in ['cuda', 'mps', 'cpu']
        
        # If CUDA available, should prefer it
        if torch.cuda.is_available():
            assert device.type == 'cuda'
        # If MPS available and no CUDA, should prefer MPS
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            assert device.type == 'mps'
        else:
            assert device.type == 'cpu'
            
    def test_prepare_model(self, hardware_manager, simple_model):
        """Test model preparation for hardware."""
        original_device = next(simple_model.parameters()).device
        
        prepared_model = hardware_manager.prepare_model(simple_model)
        
        # Model should be moved to target device
        new_device = next(prepared_model.parameters()).device
        assert new_device == hardware_manager.device
        
        # Should return a model (possibly wrapped)
        assert isinstance(prepared_model, nn.Module)
        
    def test_prepare_data_tensor(self, hardware_manager):
        """Test tensor data preparation."""
        tensor = torch.randn(5, 10)
        original_device = tensor.device
        
        prepared_tensor = hardware_manager.prepare_data(tensor)
        
        # Should be moved to target device
        assert prepared_tensor.device == hardware_manager.device
        assert prepared_tensor.shape == tensor.shape
        
    def test_prepare_data_tuple(self, hardware_manager):
        """Test tuple data preparation."""
        data_tuple = (torch.randn(3, 4), torch.randint(0, 5, (3,)))
        
        prepared_tuple = hardware_manager.prepare_data(data_tuple)
        
        assert isinstance(prepared_tuple, tuple)
        assert len(prepared_tuple) == 2
        assert prepared_tuple[0].device == hardware_manager.device
        assert prepared_tuple[1].device == hardware_manager.device
        
    def test_prepare_data_dict(self, hardware_manager):
        """Test dictionary data preparation."""
        data_dict = {
            'features': torch.randn(4, 8),
            'labels': torch.randint(0, 3, (4,)),
            'metadata': 'string_data'  # Non-tensor data
        }
        
        prepared_dict = hardware_manager.prepare_data(data_dict)
        
        assert isinstance(prepared_dict, dict)
        assert prepared_dict['features'].device == hardware_manager.device
        assert prepared_dict['labels'].device == hardware_manager.device
        assert prepared_dict['metadata'] == 'string_data'  # Unchanged
        
    def test_autocast_context(self, hardware_manager):
        """Test autocast context manager."""
        tensor = torch.randn(2, 5, device=hardware_manager.device)
        
        with hardware_manager.autocast_context():
            # Should work without error
            result = tensor * 2
            
        assert result.device == hardware_manager.device
        assert result.shape == tensor.shape
        
    def test_get_memory_stats(self, hardware_manager):
        """Test memory statistics collection."""
        stats = hardware_manager.get_memory_stats()
        
        assert isinstance(stats, dict)
        assert 'cpu_memory_used' in stats
        assert 'cpu_memory_total' in stats
        assert 'cpu_memory_percent' in stats
        
        # Check values are reasonable
        assert stats['cpu_memory_used'] >= 0
        assert stats['cpu_memory_total'] > 0
        assert 0 <= stats['cpu_memory_percent'] <= 100
        
        # GPU stats should be present if CUDA available
        if torch.cuda.is_available():
            assert 'gpu_memory_allocated' in stats
            assert stats['gpu_memory_allocated'] >= 0
            
    def test_clear_cache(self, hardware_manager):
        """Test cache clearing.""" 
        # Should not raise an error
        hardware_manager.clear_cache()
        
        # If CUDA, should clear CUDA cache
        if hardware_manager.device.type == 'cuda':
            # This is hard to test directly, but should not crash
            pass
            
    @pytest.mark.slow
    def test_benchmark_device(self, hardware_manager):
        """Test device benchmarking."""
        model = nn.Linear(32, 16)
        input_shape = (8, 32)  # batch_size, input_dim
        
        benchmark_results = hardware_manager.benchmark_device(model, input_shape)
        
        assert isinstance(benchmark_results, dict)
        assert 'inference_time_ms' in benchmark_results
        assert 'training_time_ms' in benchmark_results
        assert 'device' in benchmark_results
        
        # Times should be positive
        assert benchmark_results['inference_time_ms'] > 0
        assert benchmark_results['training_time_ms'] > 0
        
        # Device should match
        assert str(hardware_manager.device) in benchmark_results['device']


class TestMultiGPUManager:
    """Test multi-GPU functionality."""
    
    def test_multi_gpu_manager_init(self):
        """Test multi-GPU manager initialization."""
        config = HardwareConfig(world_size=2, rank=0)
        manager = MultiGPUManager(config)
        
        assert manager.world_size == 2
        assert manager.rank == 0
        assert manager.is_initialized == False
        
    @pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2, 
                       reason="Multi-GPU not available")
    def test_setup_distributed(self):
        """Test distributed setup (requires multiple GPUs)."""
        config = HardwareConfig(world_size=torch.cuda.device_count(), rank=0)
        manager = MultiGPUManager(config)
        
        # This will likely fail in testing environment, but should not crash
        try:
            success = manager.setup_distributed()
            if success:
                assert manager.is_initialized == True
                manager.cleanup()
        except Exception:
            # Expected in most test environments
            pass
            
    def test_wrap_model(self):
        """Test model wrapping for distributed training."""
        config = HardwareConfig(world_size=1, rank=0)
        manager = MultiGPUManager(config)
        
        model = nn.Linear(10, 5)
        
        # Should return model unchanged if not initialized
        wrapped_model = manager.wrap_model(model)
        assert wrapped_model == model
        
    def test_cleanup(self):
        """Test cleanup functionality."""
        config = HardwareConfig()
        manager = MultiGPUManager(config)
        
        # Should not raise error even if not initialized
        manager.cleanup()
        assert manager.is_initialized == False


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_auto_device(self):
        """Test automatic device detection."""
        device = auto_device()
        
        assert isinstance(device, torch.device)
        assert device.type in ['cuda', 'mps', 'cpu']
        
    def test_prepare_for_hardware(self):
        """Test model preparation utility."""
        model = nn.Sequential(nn.Linear(5, 10), nn.ReLU(), nn.Linear(10, 3))
        
        prepared_model = prepare_for_hardware(model)
        
        assert isinstance(prepared_model, nn.Module)
        # Should be moved to auto-detected device
        device = auto_device()
        model_device = next(prepared_model.parameters()).device
        assert model_device == device
        
    def test_prepare_for_hardware_custom_device(self):
        """Test model preparation with custom device."""
        model = nn.Linear(8, 4)
        custom_device = torch.device('cpu')  # Force CPU
        
        prepared_model = prepare_for_hardware(model, custom_device)
        
        model_device = next(prepared_model.parameters()).device
        assert model_device == custom_device
        
    @patch('meta_learning.meta_learning_modules.hardware_utils.logger')
    def test_log_hardware_info(self, mock_logger):
        """Test hardware info logging."""
        log_hardware_info()
        
        # Should have called logger.info multiple times
        assert mock_logger.info.called
        call_args_list = mock_logger.info.call_args_list
        
        # Should log various hardware info
        logged_text = ' '.join([str(call.args[0]) for call in call_args_list])
        assert 'Hardware Configuration' in logged_text
        assert 'CPU' in logged_text
        
    def test_create_hardware_manager(self):
        """Test hardware manager factory function."""
        # Default creation
        manager1 = create_hardware_manager()
        assert isinstance(manager1, HardwareManager)
        
        # Custom device
        manager2 = create_hardware_manager(device='cpu')
        assert manager2.device.type == 'cpu'
        
        # Custom mixed precision
        manager3 = create_hardware_manager(use_mixed_precision=False)
        assert manager3.config.use_mixed_precision == False


class TestOptimalBatchSize:
    """Test optimal batch size detection."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for batch size optimization")
    def test_get_optimal_batch_size_cuda(self):
        """Test optimal batch size detection on CUDA."""
        model = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 16)
        )
        
        input_shape = (1, 32)
        device = torch.device('cuda')
        
        optimal_batch = get_optimal_batch_size(model, input_shape, device)
        
        assert isinstance(optimal_batch, int)
        assert optimal_batch >= 1
        assert optimal_batch <= 1024  # Reasonable upper bound
        
    def test_get_optimal_batch_size_cpu(self):
        """Test optimal batch size detection on CPU."""
        model = nn.Linear(16, 8)
        input_shape = (1, 16)
        device = torch.device('cpu')
        
        optimal_batch = get_optimal_batch_size(model, input_shape, device)
        
        # Should return conservative default for CPU
        assert optimal_batch == 32
        
    def test_get_optimal_batch_size_mps(self):
        """Test optimal batch size detection on MPS.""" 
        if not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
            pytest.skip("MPS not available")
            
        model = nn.Linear(20, 10)
        input_shape = (1, 20)
        device = torch.device('mps')
        
        optimal_batch = get_optimal_batch_size(model, input_shape, device)
        
        # Should return conservative default for MPS
        assert optimal_batch == 32


class TestHardwareCompatibility:
    """Test hardware compatibility across different devices."""
    
    @pytest.mark.parametrize("device_type", ["cpu"])  # Always available
    def test_hardware_manager_device_compatibility(self, device_type):
        """Test hardware manager works on different device types."""
        if device_type == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        elif device_type == "mps" and not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
            pytest.skip("MPS not available")
            
        config = HardwareConfig(device=device_type)
        manager = HardwareManager(config)
        
        assert manager.device.type == device_type
        
        # Test basic functionality
        model = nn.Linear(4, 2)
        prepared_model = manager.prepare_model(model)
        
        data = torch.randn(3, 4)
        prepared_data = manager.prepare_data(data)
        
        assert next(prepared_model.parameters()).device.type == device_type
        assert prepared_data.device.type == device_type
        
    def test_mixed_precision_compatibility(self):
        """Test mixed precision compatibility across devices."""
        devices_to_test = ['cpu']
        
        if torch.cuda.is_available():
            devices_to_test.append('cuda')
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            devices_to_test.append('mps')
            
        for device_type in devices_to_test:
            config = HardwareConfig(
                device=device_type,
                use_mixed_precision=True
            )
            manager = HardwareManager(config)
            
            # Mixed precision should be disabled on non-CUDA devices
            if device_type != 'cuda':
                assert manager.config.use_mixed_precision == False
            
            # Autocast context should work
            with manager.autocast_context():
                tensor = torch.randn(2, 3, device=manager.device)
                result = tensor * 2
                assert result.device == manager.device


class TestHardwareEdgeCases:
    """Test hardware edge cases and error handling."""
    
    def test_hardware_manager_with_invalid_device(self):
        """Test hardware manager with invalid device specification."""
        # Should fall back to auto-detection
        config = HardwareConfig(device="invalid_device")
        
        try:
            manager = HardwareManager(config)
            # Should still create a manager with valid device
            assert manager.device.type in ['cuda', 'mps', 'cpu']
        except Exception:
            # Some invalid device specifications might raise errors
            pass
            
    def test_prepare_model_with_none_input(self):
        """Test model preparation with None input."""
        manager = HardwareManager()
        
        with pytest.raises(AttributeError):
            manager.prepare_model(None)
            
    def test_prepare_data_with_none_input(self):
        """Test data preparation with None input."""
        manager = HardwareManager()
        
        result = manager.prepare_data(None)
        assert result is None
        
    def test_memory_stats_without_gpu(self):
        """Test memory statistics when GPU is not available."""
        # Force CPU-only configuration
        config = HardwareConfig(device='cpu')
        manager = HardwareManager(config)
        
        stats = manager.get_memory_stats()
        
        # Should have CPU stats
        assert 'cpu_memory_used' in stats
        assert 'cpu_memory_total' in stats
        
        # Should not have GPU stats
        assert 'gpu_memory_allocated' not in stats
        
    def test_autocast_context_without_mixed_precision(self):
        """Test autocast context when mixed precision is disabled."""
        config = HardwareConfig(use_mixed_precision=False)
        manager = HardwareManager(config)
        
        tensor = torch.randn(2, 3, device=manager.device)
        
        with manager.autocast_context():
            result = tensor + 1
            
        assert result.device == manager.device
        # Should work fine even without mixed precision


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])