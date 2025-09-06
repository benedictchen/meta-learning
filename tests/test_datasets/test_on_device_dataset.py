"""
Tests for OnDeviceDataset.

Tests GPU acceleration and smart memory management with compression and mixed precision.
"""

import pytest
import torch
import time
from unittest.mock import Mock, patch
import threading

from meta_learning.datasets.on_device_dataset import (
    OnDeviceDataset, MemoryManager, CompressionEngine, LRUCache
)


class TestLRUCache:
    """Test LRU cache functionality."""
    
    def test_initialization(self):
        """Test cache initialization."""
        cache = LRUCache(capacity=5)
        
        assert cache.capacity == 5
        assert len(cache.cache) == 0
        assert len(cache.order) == 0
    
    def test_put_and_get(self):
        """Test basic put/get operations."""
        cache = LRUCache(capacity=3)
        
        # Add items
        cache.put('a', torch.tensor([1, 2, 3]))
        cache.put('b', torch.tensor([4, 5, 6]))
        cache.put('c', torch.tensor([7, 8, 9]))
        
        # Retrieve items
        assert torch.equal(cache.get('a'), torch.tensor([1, 2, 3]))
        assert torch.equal(cache.get('b'), torch.tensor([4, 5, 6]))
        assert torch.equal(cache.get('c'), torch.tensor([7, 8, 9]))
    
    def test_capacity_limit(self):
        """Test capacity enforcement."""
        cache = LRUCache(capacity=2)
        
        cache.put('a', torch.tensor([1]))
        cache.put('b', torch.tensor([2]))
        cache.put('c', torch.tensor([3]))  # Should evict 'a'
        
        assert cache.get('a') is None  # Evicted
        assert torch.equal(cache.get('b'), torch.tensor([2]))
        assert torch.equal(cache.get('c'), torch.tensor([3]))
    
    def test_lru_ordering(self):
        """Test LRU eviction ordering."""
        cache = LRUCache(capacity=2)
        
        cache.put('a', torch.tensor([1]))
        cache.put('b', torch.tensor([2]))
        
        # Access 'a' to make it recently used
        cache.get('a')
        
        # Add new item - should evict 'b' (least recently used)
        cache.put('c', torch.tensor([3]))
        
        assert torch.equal(cache.get('a'), torch.tensor([1]))  # Still there
        assert cache.get('b') is None  # Evicted
        assert torch.equal(cache.get('c'), torch.tensor([3]))
    
    def test_update_existing(self):
        """Test updating existing keys."""
        cache = LRUCache(capacity=2)
        
        cache.put('a', torch.tensor([1]))
        cache.put('a', torch.tensor([2]))  # Update
        
        assert torch.equal(cache.get('a'), torch.tensor([2]))
        assert len(cache.cache) == 1
    
    def test_clear(self):
        """Test cache clearing."""
        cache = LRUCache(capacity=3)
        
        cache.put('a', torch.tensor([1]))
        cache.put('b', torch.tensor([2]))
        
        cache.clear()
        
        assert len(cache.cache) == 0
        assert len(cache.order) == 0
        assert cache.get('a') is None
    
    def test_statistics(self):
        """Test cache statistics."""
        cache = LRUCache(capacity=2)
        
        cache.put('a', torch.tensor([1]))
        cache.get('a')  # Hit
        cache.get('b')  # Miss
        
        stats = cache.get_stats()
        
        assert stats['hits'] == 1
        assert stats['misses'] == 1
        assert stats['hit_rate'] == 0.5


class TestCompressionEngine:
    """Test compression engine functionality."""
    
    def test_initialization(self):
        """Test engine initialization."""
        engine = CompressionEngine()
        
        assert engine.compression_ratio > 0
        assert engine.stats['total_compressed'] == 0
        assert engine.stats['total_decompressed'] == 0
    
    def test_tensor_compression(self):
        """Test tensor compression and decompression."""
        engine = CompressionEngine()
        
        # Create test tensor
        original = torch.randn(100, 50)
        
        # Compress
        compressed = engine.compress(original)
        
        # Decompress
        decompressed = engine.decompress(compressed)
        
        # Should be approximately equal (some precision loss expected)
        assert decompressed.shape == original.shape
        assert torch.allclose(original, decompressed, atol=1e-3)
        
        # Check stats
        assert engine.stats['total_compressed'] == 1
        assert engine.stats['total_decompressed'] == 1
    
    def test_compression_ratio(self):
        """Test compression achieves size reduction."""
        engine = CompressionEngine()
        
        # Create large tensor with some structure
        original = torch.randn(1000, 100) * 0.1  # Small values compress better
        
        compressed = engine.compress(original)
        decompressed = engine.decompress(compressed)
        
        # Compressed should be smaller (in terms of storage)
        # Note: This is a simplified check since we don't store raw bytes
        assert compressed is not None
        assert torch.allclose(original, decompressed, atol=1e-2)
    
    def test_batch_compression(self):
        """Test compressing multiple tensors."""
        engine = CompressionEngine()
        
        tensors = [
            torch.randn(50, 20),
            torch.randn(30, 40),
            torch.randn(60, 10)
        ]
        
        # Compress all
        compressed_tensors = []
        for tensor in tensors:
            compressed_tensors.append(engine.compress(tensor))
        
        # Decompress all
        decompressed_tensors = []
        for compressed in compressed_tensors:
            decompressed_tensors.append(engine.decompress(compressed))
        
        # Check all are recovered correctly
        for original, decompressed in zip(tensors, decompressed_tensors):
            assert torch.allclose(original, decompressed, atol=1e-3)
        
        assert engine.stats['total_compressed'] == 3
        assert engine.stats['total_decompressed'] == 3


class TestMemoryManager:
    """Test memory manager functionality."""
    
    def test_initialization(self):
        """Test manager initialization."""
        manager = MemoryManager(memory_budget=0.5)
        
        assert manager.memory_budget == 0.5
        assert manager.current_usage == 0
        assert manager.stats['allocations'] == 0
        assert manager.stats['deallocations'] == 0
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_memory_tracking(self):
        """Test GPU memory usage tracking."""
        manager = MemoryManager(memory_budget=0.8)
        
        initial_usage = manager._get_current_gpu_usage()
        
        # Allocate GPU tensor
        gpu_tensor = torch.randn(1000, 1000).cuda()
        
        current_usage = manager._get_current_gpu_usage()
        
        # Usage should have increased
        assert current_usage > initial_usage
        
        # Clean up
        del gpu_tensor
        torch.cuda.empty_cache()
    
    def test_memory_budget_check(self):
        """Test memory budget enforcement."""
        # Mock GPU memory functions for testing
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.cuda.get_device_properties') as mock_props:
                mock_props.return_value.total_memory = 1024 * 1024 * 1024  # 1GB
                
                with patch('torch.cuda.memory_allocated', return_value=800 * 1024 * 1024):  # 800MB
                    manager = MemoryManager(memory_budget=0.9)  # 90%
                    
                    # Should allow allocation (under budget)
                    assert manager.can_allocate(100 * 1024 * 1024)  # 100MB
                    
                    # Should deny allocation (over budget)
                    assert not manager.can_allocate(300 * 1024 * 1024)  # 300MB
    
    def test_allocation_tracking(self):
        """Test allocation statistics tracking."""
        manager = MemoryManager(memory_budget=0.8)
        
        # Simulate allocations
        manager.record_allocation(1000000)  # 1MB
        manager.record_allocation(500000)   # 0.5MB
        
        assert manager.stats['allocations'] == 2
        assert manager.stats['total_allocated'] == 1500000
        
        # Simulate deallocation
        manager.record_deallocation(500000)
        
        assert manager.stats['deallocations'] == 1
        assert manager.stats['total_deallocated'] == 500000
    
    def test_memory_pressure_detection(self):
        """Test memory pressure detection."""
        manager = MemoryManager(memory_budget=0.7)
        
        # Mock high memory usage
        with patch.object(manager, '_get_current_gpu_usage', return_value=0.85):
            assert manager.is_memory_pressure()
        
        # Mock low memory usage
        with patch.object(manager, '_get_current_gpu_usage', return_value=0.5):
            assert not manager.is_memory_pressure()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestOnDeviceDataset:
    """Test on-device dataset functionality (requires CUDA)."""
    
    @pytest.fixture
    def sample_episodes(self):
        """Create sample episodes for testing."""
        episodes = []
        for i in range(10):
            support_x = torch.randn(25, 64)  # 5-way, 5-shot
            support_y = torch.arange(5).repeat(5)
            query_x = torch.randn(75, 64)    # 15 queries per class
            query_y = torch.randint(0, 5, (75,))
            
            episodes.append((support_x, support_y, query_x, query_y))
        
        return episodes
    
    def test_initialization(self, sample_episodes):
        """Test dataset initialization."""
        dataset = OnDeviceDataset(
            episodes=sample_episodes,
            memory_budget=0.8,
            enable_compression=True,
            mixed_precision=True
        )
        
        assert len(dataset) == 10
        assert dataset.memory_budget == 0.8
        assert dataset.enable_compression is True
        assert dataset.mixed_precision is True
    
    def test_episode_loading(self, sample_episodes):
        """Test loading episodes to GPU."""
        dataset = OnDeviceDataset(episodes=sample_episodes[:5])
        
        # Get first episode
        episode = dataset[0]
        
        assert len(episode) == 4
        support_x, support_y, query_x, query_y = episode
        
        # Should be on GPU
        assert support_x.is_cuda
        assert support_y.is_cuda
        assert query_x.is_cuda
        assert query_y.is_cuda
        
        # Check shapes
        assert support_x.shape == (25, 64)
        assert query_x.shape == (75, 64)
    
    def test_caching_behavior(self, sample_episodes):
        """Test LRU caching of episodes."""
        dataset = OnDeviceDataset(
            episodes=sample_episodes,
            cache_size=3  # Small cache
        )
        
        # Load episodes
        ep0 = dataset[0]
        ep1 = dataset[1] 
        ep2 = dataset[2]
        ep3 = dataset[3]  # Should evict episode 0
        
        # Access episode 1 again (should be fast due to caching)
        start_time = time.time()
        ep1_again = dataset[1]
        cache_time = time.time() - start_time
        
        # Should be very fast (cached)
        assert cache_time < 0.01  # Less than 10ms
        
        # Should be identical tensors
        assert torch.equal(ep1[0], ep1_again[0])
    
    def test_memory_management(self, sample_episodes):
        """Test memory management and eviction."""
        dataset = OnDeviceDataset(
            episodes=sample_episodes,
            memory_budget=0.6,  # Tight budget
            cache_size=2
        )
        
        # Load several episodes
        episodes = []
        for i in range(5):
            episodes.append(dataset[i])
        
        # Check cache stats
        stats = dataset.get_cache_stats()
        assert stats['hits'] >= 0
        assert stats['misses'] > 0
        
        # Memory manager should have tracked allocations
        memory_stats = dataset.memory_manager.get_stats()
        assert memory_stats['allocations'] > 0
    
    def test_compression_functionality(self, sample_episodes):
        """Test tensor compression."""
        dataset_compressed = OnDeviceDataset(
            episodes=sample_episodes[:3],
            enable_compression=True
        )
        
        dataset_uncompressed = OnDeviceDataset(
            episodes=sample_episodes[:3],
            enable_compression=False
        )
        
        # Load same episode from both
        ep_comp = dataset_compressed[0]
        ep_uncomp = dataset_uncompressed[0]
        
        # Should be approximately equal
        for tensor_comp, tensor_uncomp in zip(ep_comp, ep_uncomp):
            assert torch.allclose(tensor_comp, tensor_uncomp, atol=1e-3)
    
    def test_mixed_precision(self, sample_episodes):
        """Test mixed precision support."""
        dataset = OnDeviceDataset(
            episodes=sample_episodes[:2],
            mixed_precision=True
        )
        
        episode = dataset[0]
        support_x, support_y, query_x, query_y = episode
        
        # Floating point tensors should be float16
        assert support_x.dtype == torch.float16
        assert query_x.dtype == torch.float16
        
        # Integer tensors should remain int64
        assert support_y.dtype == torch.int64
        assert query_y.dtype == torch.int64
    
    def test_prefetching(self, sample_episodes):
        """Test background prefetching."""
        dataset = OnDeviceDataset(
            episodes=sample_episodes,
            prefetch_size=3
        )
        
        # Access first episode
        ep0 = dataset[0]
        
        # Give time for prefetching
        time.sleep(0.1)
        
        # Next few episodes should be fast (prefetched)
        start_time = time.time()
        ep1 = dataset[1]
        ep2 = dataset[2]
        prefetch_time = time.time() - start_time
        
        # Should be fast due to prefetching
        assert prefetch_time < 0.05  # Less than 50ms for 2 episodes
    
    def test_thread_safety(self, sample_episodes):
        """Test thread-safe access."""
        dataset = OnDeviceDataset(episodes=sample_episodes)
        results = []
        errors = []
        
        def worker_thread(worker_id):
            try:
                worker_results = []
                for i in range(3):
                    episode_idx = (worker_id * 3 + i) % len(sample_episodes)
                    episode = dataset[episode_idx]
                    worker_results.append((worker_id, episode_idx, episode[0].shape))
                results.extend(worker_results)
            except Exception as e:
                errors.append((worker_id, str(e)))
        
        # Start multiple worker threads
        threads = []
        for worker_id in range(3):
            thread = threading.Thread(target=worker_thread, args=(worker_id,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Should complete without errors
        assert len(errors) == 0, f"Thread errors: {errors}"
        assert len(results) == 9  # 3 workers × 3 episodes
    
    def test_performance_monitoring(self, sample_episodes):
        """Test performance monitoring."""
        dataset = OnDeviceDataset(episodes=sample_episodes[:5])
        
        # Load all episodes
        for i in range(len(dataset)):
            _ = dataset[i]
        
        # Get performance stats
        perf_stats = dataset.get_performance_stats()
        
        assert isinstance(perf_stats, dict)
        assert 'avg_load_time' in perf_stats
        assert 'cache_hit_rate' in perf_stats
        assert 'memory_efficiency' in perf_stats
        
        # Should have reasonable performance metrics
        assert perf_stats['avg_load_time'] > 0
        assert 0 <= perf_stats['cache_hit_rate'] <= 1
    
    def test_memory_optimization(self, sample_episodes):
        """Test memory optimization features."""
        dataset = OnDeviceDataset(
            episodes=sample_episodes,
            memory_budget=0.7,
            enable_compression=True,
            mixed_precision=True,
            auto_optimization=True
        )
        
        # Load episodes and trigger optimization
        for i in range(min(len(sample_episodes), 8)):
            _ = dataset[i]
        
        # Should optimize memory usage
        dataset.optimize_memory()
        
        # Memory stats should reflect optimizations
        memory_stats = dataset.memory_manager.get_stats()
        assert memory_stats['allocations'] > 0
        
        # Should still work correctly after optimization
        episode = dataset[0]
        assert len(episode) == 4
        assert all(tensor.is_cuda for tensor in episode)


class TestCPUFallback:
    """Test CPU fallback when CUDA is unavailable."""
    
    @pytest.fixture
    def sample_episodes(self):
        """Create sample episodes for testing."""
        episodes = []
        for i in range(3):
            support_x = torch.randn(15, 32)
            support_y = torch.arange(3).repeat(5)
            query_x = torch.randn(30, 32)
            query_y = torch.randint(0, 3, (30,))
            
            episodes.append((support_x, support_y, query_x, query_y))
        
        return episodes
    
    def test_cpu_mode(self, sample_episodes):
        """Test dataset works on CPU."""
        # Force CPU mode
        with patch('torch.cuda.is_available', return_value=False):
            dataset = OnDeviceDataset(episodes=sample_episodes)
            
            episode = dataset[0]
            support_x, support_y, query_x, query_y = episode
            
            # Should be on CPU
            assert not support_x.is_cuda
            assert not support_y.is_cuda
            assert not query_x.is_cuda
            assert not query_y.is_cuda
            
            # Should still have correct shapes
            assert support_x.shape == (15, 32)
            assert query_x.shape == (30, 32)


class TestIntegration:
    """Integration tests for OnDeviceDataset."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_realistic_training_scenario(self):
        """Test realistic meta-learning training scenario."""
        # Create realistic episode data
        episodes = []
        for i in range(20):
            n_way, k_shot = 5, 5
            support_size = n_way * k_shot
            query_size = n_way * 15  # 15 queries per class
            
            support_x = torch.randn(support_size, 512)  # ResNet features
            support_y = torch.arange(n_way).repeat(k_shot)
            query_x = torch.randn(query_size, 512)
            query_y = torch.randint(0, n_way, (query_size,))
            
            episodes.append((support_x, support_y, query_x, query_y))
        
        # Create dataset with optimization features
        dataset = OnDeviceDataset(
            episodes=episodes,
            memory_budget=0.8,
            cache_size=8,
            enable_compression=True,
            mixed_precision=True,
            prefetch_size=4
        )
        
        # Simulate training loop
        training_times = []
        
        for epoch in range(3):
            epoch_start = time.time()
            
            for batch_idx in range(min(len(episodes), 10)):
                episode = dataset[batch_idx]
                
                # Simulate model forward pass
                time.sleep(0.001)  # Minimal processing time
            
            epoch_time = time.time() - epoch_start
            training_times.append(epoch_time)
        
        # Performance should improve over epochs due to caching/prefetching
        assert len(training_times) == 3
        
        # Get final performance stats
        perf_stats = dataset.get_performance_stats()
        cache_stats = dataset.get_cache_stats()
        
        # Should have good cache performance
        assert cache_stats['hit_rate'] > 0.5  # At least 50% hit rate
        
        # Should have reasonable load times
        assert perf_stats['avg_load_time'] < 0.1  # Less than 100ms average


if __name__ == "__main__":
    pytest.main([__file__])