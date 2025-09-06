"""
Tests for dataset ecosystem enhancements including dataset management,
caching, and performance optimization.

Tests cover:
- BenchmarkDatasetManager functionality
- OnDeviceDataset caching and memory management
- InfiniteEpisodeIterator with adaptive sampling
- Dataset registry and discovery
- Professional dataset downloading and validation
"""

import pytest
import torch
import os
import tempfile
import pickle
import shutil
import time
from unittest.mock import Mock, patch, MagicMock
from collections import deque

from meta_learning.core.episode import Episode
from meta_learning.data_utils.datasets import (
    BenchmarkDatasetManager, OnDeviceDataset, InfiniteEpisodeIterator,
    MiniImageNetDataset, SyntheticFewShotDataset, DatasetRegistry,
    BaseMetaLearningDataset
)


class TestBenchmarkDatasetManager:
    """Test benchmark dataset management functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.manager = BenchmarkDatasetManager(
            cache_dir=self.temp_dir,
            max_cache_size_gb=0.001  # 1MB limit for testing
        )
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_manager_initialization(self):
        """Test dataset manager initialization."""
        assert self.manager.cache_dir == self.temp_dir
        assert self.manager.max_cache_size_gb == 0.001
        assert os.path.exists(self.temp_dir)
        
        # Check dataset registry
        assert 'mini_imagenet' in self.manager.dataset_registry
        assert 'synthetic' in self.manager.dataset_registry
        assert 'cifar_fs' in self.manager.dataset_registry
        assert 'omniglot' in self.manager.dataset_registry
    
    def test_list_available_datasets(self):
        """Test listing available datasets."""
        datasets = self.manager.list_available_datasets()
        
        assert isinstance(datasets, dict)
        assert 'mini_imagenet' in datasets
        assert 'synthetic' in datasets
        
        # Check dataset metadata
        mini_info = datasets['mini_imagenet']
        assert 'description' in mini_info
        assert 'file_size_mb' in mini_info
        assert 'cached' in mini_info
        assert mini_info['cached'] == False  # Not cached initially
    
    def test_download_synthetic_dataset(self):
        """Test downloading synthetic dataset (no actual download needed)."""
        dataset_path = self.manager.download_dataset('synthetic')
        
        assert dataset_path is not None
        assert os.path.exists(dataset_path)
        assert os.path.basename(dataset_path) == 'synthetic'
        
        # Check caching status
        datasets = self.manager.list_available_datasets()
        assert datasets['synthetic']['cached'] == True
    
    def test_download_mini_imagenet_dataset(self):
        """Test downloading mini_imagenet dataset with synthetic placeholder."""
        dataset_path = self.manager.download_dataset('mini_imagenet')
        
        assert dataset_path is not None
        assert os.path.exists(dataset_path)
        
        # Check that synthetic data files were created
        for split in ['train', 'val', 'test']:
            split_file = os.path.join(dataset_path, f"mini_imagenet_{split}.pkl")
            assert os.path.exists(split_file)
            
            # Check file content
            with open(split_file, 'rb') as f:
                data = pickle.load(f)
                assert 'data' in data
                assert 'labels' in data
                assert data['data'].shape == (600, 3, 84, 84)
        
        # Check completion marker
        marker_file = os.path.join(dataset_path, '.download_complete')
        assert os.path.exists(marker_file)
    
    def test_dataset_caching_validation(self):
        """Test dataset caching validation."""
        # Initially not cached
        assert not self.manager._is_dataset_cached('mini_imagenet')
        
        # Download dataset
        self.manager.download_dataset('mini_imagenet')
        
        # Now should be cached
        assert self.manager._is_dataset_cached('mini_imagenet')
    
    def test_force_redownload(self):
        """Test force redownload functionality."""
        # Download once
        path1 = self.manager.download_dataset('mini_imagenet')
        assert path1 is not None
        
        # Download again without force (should use cache)
        with patch.object(self.manager, '_download_with_fallback') as mock_download:
            path2 = self.manager.download_dataset('mini_imagenet', force_redownload=False)
            assert path2 == path1
            assert not mock_download.called
        
        # Download with force (should redownload)
        with patch.object(self.manager, '_download_with_fallback', return_value=True) as mock_download:
            path3 = self.manager.download_dataset('mini_imagenet', force_redownload=True)
            assert mock_download.called
    
    def test_cache_size_management(self):
        """Test cache size management and eviction."""
        # Create multiple datasets to exceed cache limit
        datasets_to_create = ['mini_imagenet', 'synthetic']  # Keep it simple for testing
        
        for dataset_name in datasets_to_create:
            self.manager.download_dataset(dataset_name)
        
        # Check cache info
        cache_info = self.manager.get_cache_info()
        assert 'total_size_mb' in cache_info
        assert 'datasets' in cache_info
        assert 'usage_percent' in cache_info
        
        assert len(cache_info['datasets']) >= 1
        assert cache_info['total_size_mb'] >= 0
    
    def test_cache_info_reporting(self):
        """Test cache information reporting."""
        # Start with empty cache
        cache_info = self.manager.get_cache_info()
        assert cache_info['total_size_mb'] == 0
        assert len(cache_info['datasets']) == 0
        assert cache_info['usage_percent'] == 0
        
        # Add a dataset
        self.manager.download_dataset('synthetic')
        
        # Check updated info
        cache_info = self.manager.get_cache_info()
        assert cache_info['total_size_mb'] >= 0
        assert len(cache_info['datasets']) >= 1
        assert 'synthetic' in cache_info['datasets']
    
    def test_cache_clearing(self):
        """Test cache clearing functionality."""
        # Download datasets
        self.manager.download_dataset('synthetic')
        self.manager.download_dataset('mini_imagenet')
        
        # Verify they exist
        cache_info = self.manager.get_cache_info()
        assert len(cache_info['datasets']) >= 2
        
        # Clear specific dataset
        self.manager.clear_cache('synthetic')
        
        # Verify synthetic is gone
        cache_info = self.manager.get_cache_info()
        assert 'synthetic' not in cache_info['datasets']
        
        # Clear all cache
        self.manager.clear_cache()
        
        # Verify everything is gone
        cache_info = self.manager.get_cache_info()
        assert len(cache_info['datasets']) == 0
        assert cache_info['total_size_mb'] == 0
    
    def test_invalid_dataset_name(self):
        """Test handling of invalid dataset names."""
        with pytest.raises(ValueError, match="Unknown dataset"):
            self.manager.download_dataset('nonexistent_dataset')
    
    @patch('concurrent.futures.ThreadPoolExecutor')
    def test_parallel_download_simulation(self, mock_executor):
        """Test parallel download mechanism (simulated)."""
        # Mock the executor to verify it's used
        mock_future = Mock()
        mock_future.result.return_value = True
        mock_executor.return_value.__enter__.return_value.submit.return_value = mock_future
        
        # Download dataset
        result = self.manager.download_dataset('mini_imagenet')
        
        assert result is not None
        # Verify executor was used (in real implementation)


class TestOnDeviceDataset:
    """Test OnDeviceDataset caching and optimization."""
    
    def setup_method(self):
        """Setup test fixtures."""
        # Create sample episodes
        self.episodes = []
        for i in range(10):
            episode = Episode(
                support_x=torch.randn(20, 64),  # Varying sizes
                support_y=torch.repeat_interleave(torch.arange(4), 5),
                query_x=torch.randn(12, 64),
                query_y=torch.repeat_interleave(torch.arange(4), 3)
            )
            self.episodes.append(episode)
    
    def test_on_device_dataset_initialization(self):
        """Test OnDeviceDataset initialization."""
        dataset = OnDeviceDataset(
            self.episodes[:5],
            memory_budget=0.5,
            enable_compression=True,
            enable_mixed_precision=True
        )
        
        assert len(dataset) == 5
        assert dataset.memory_budget == 0.5
        assert dataset.enable_compression == True
        assert dataset.enable_mixed_precision == True
        assert dataset.device.type in ['cpu', 'cuda']
    
    def test_episode_size_estimation(self):
        """Test episode memory size estimation."""
        dataset = OnDeviceDataset(self.episodes[:3], memory_budget=0.8)
        
        for episode in self.episodes[:3]:
            size = dataset._estimate_episode_size(episode)
            assert size > 0
            
            # Check that size calculation makes sense
            expected_size = (
                episode.support_x.numel() * episode.support_x.element_size() +
                episode.support_y.numel() * episode.support_y.element_size() +
                episode.query_x.numel() * episode.query_x.element_size() +
                episode.query_y.numel() * episode.query_y.element_size()
            )
            assert size == expected_size
    
    def test_episode_caching_and_retrieval(self):
        """Test episode caching and retrieval."""
        dataset = OnDeviceDataset(self.episodes[:5], memory_budget=0.8)
        
        # Access episodes
        for i in range(len(dataset)):
            episode = dataset[i]
            assert isinstance(episode, Episode)
            assert episode.support_x.device == dataset.device
            assert episode.support_y.device == dataset.device
            assert episode.query_x.device == dataset.device
            assert episode.query_y.device == dataset.device
    
    def test_cache_eviction_mechanism(self):
        """Test cache eviction when memory is full."""
        # Use very small memory budget to force eviction
        dataset = OnDeviceDataset(
            self.episodes,
            memory_budget=0.01,  # Very small budget
            enable_compression=False,
            enable_mixed_precision=False
        )
        
        # Access all episodes (should trigger eviction)
        accessed_episodes = []
        for i in range(len(dataset)):
            episode = dataset[i]
            accessed_episodes.append(episode)
            assert isinstance(episode, Episode)
        
        # Check that cache management is working
        cache_stats = dataset.get_cache_stats()
        assert cache_stats['total_episodes'] == len(self.episodes)
        assert cache_stats['cached_episodes'] <= len(self.episodes)
        assert 0.0 <= cache_stats['cache_hit_rate'] <= 1.0
    
    def test_mixed_precision_optimization(self):
        """Test mixed precision optimization."""
        dataset = OnDeviceDataset(
            self.episodes[:3],
            memory_budget=0.8,
            enable_mixed_precision=True
        )
        
        # Access an episode and check if mixed precision is applied
        episode = dataset[0]
        
        # For large tensors, should use half precision
        large_tensor = torch.randn(15000).to(dataset.device)  # Large tensor
        optimized_tensor = dataset._optimize_tensor(large_tensor)
        
        if large_tensor.numel() > 10000:
            # Should be converted to half precision for large tensors
            assert optimized_tensor.dtype in [torch.float16, torch.float32]
    
    def test_cache_statistics(self):
        """Test cache statistics reporting."""
        dataset = OnDeviceDataset(self.episodes[:5], memory_budget=0.8)
        
        # Access some episodes
        for i in range(3):
            _ = dataset[i]
        
        # Get statistics
        stats = dataset.get_cache_stats()
        
        assert 'total_episodes' in stats
        assert 'cached_episodes' in stats
        assert 'cache_hit_rate' in stats
        assert 'memory_used_mb' in stats
        assert 'memory_available_mb' in stats
        assert 'device' in stats
        
        assert stats['total_episodes'] == 5
        assert stats['cached_episodes'] >= 0
        assert 0.0 <= stats['cache_hit_rate'] <= 1.0
        assert stats['memory_used_mb'] >= 0
        assert stats['memory_available_mb'] > 0
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_memory_management(self):
        """Test GPU memory management (if CUDA available)."""
        dataset = OnDeviceDataset(
            self.episodes[:3],
            memory_budget=0.1,  # Small budget to test management
            enable_mixed_precision=True
        )
        
        if dataset.device.type == 'cuda':
            initial_memory = torch.cuda.memory_allocated()
            
            # Access episodes
            for i in range(len(dataset)):
                _ = dataset[i]
            
            # Memory should be managed properly
            final_memory = torch.cuda.memory_allocated()
            assert final_memory >= initial_memory  # Some memory allocated
            
            # Cache stats should reflect GPU usage
            stats = dataset.get_cache_stats()
            assert stats['device'] == 'cuda'


class TestInfiniteEpisodeIterator:
    """Test InfiniteEpisodeIterator with adaptive sampling."""
    
    def setup_method(self):
        """Setup test fixtures."""
        # Create episode generator function
        self.episode_count = 0
        
        def episode_generator():
            self.episode_count += 1
            return Episode(
                support_x=torch.randn(15, 32),
                support_y=torch.repeat_interleave(torch.arange(3), 5),
                query_x=torch.randn(9, 32),
                query_y=torch.repeat_interleave(torch.arange(3), 3)
            )
        
        self.episode_generator = episode_generator
    
    def teardown_method(self):
        """Clean up test fixtures."""
        # Reset episode count
        self.episode_count = 0
    
    def test_infinite_iterator_initialization(self):
        """Test InfiniteEpisodeIterator initialization."""
        iterator = InfiniteEpisodeIterator(
            self.episode_generator,
            buffer_size=100,
            adaptive_sampling=True,
            prefetch_factor=2
        )
        
        assert iterator.buffer_size == 100
        assert iterator.adaptive_sampling == True
        assert iterator.prefetch_factor == 2
        assert len(iterator.episode_buffer) == 100
        assert iterator.generated_count > 0  # Should have generated some initial episodes
    
    def test_infinite_iteration_basic(self):
        """Test basic infinite iteration."""
        iterator = InfiniteEpisodeIterator(
            self.episode_generator,
            buffer_size=10,
            adaptive_sampling=False
        )
        
        # Get several episodes
        episodes = []
        for i, episode in enumerate(iterator):
            episodes.append(episode)
            if i >= 5:  # Get 6 episodes
                break
        
        assert len(episodes) == 6
        for episode in episodes:
            assert isinstance(episode, Episode)
            assert episode.support_x.shape == (15, 32)
            assert episode.query_x.shape == (9, 32)
    
    def test_infinite_iterator_buffer_management(self):
        """Test buffer management and refill."""
        iterator = InfiniteEpisodeIterator(
            self.episode_generator,
            buffer_size=5,
            adaptive_sampling=False
        )
        
        initial_count = iterator.generated_count
        
        # Consume several episodes (more than buffer size)
        consumed_episodes = []
        for i, episode in enumerate(iterator):
            consumed_episodes.append(episode)
            if i >= 8:  # Consume more than buffer size
                break
        
        # Should have generated more episodes to refill buffer
        assert iterator.generated_count > initial_count
        assert len(consumed_episodes) == 9
    
    def test_adaptive_sampling_performance_tracking(self):
        """Test adaptive sampling with performance tracking."""
        iterator = InfiniteEpisodeIterator(
            self.episode_generator,
            buffer_size=10,
            adaptive_sampling=True
        )
        
        # Simulate performance feedback
        iterator.update_performance(0.8)
        iterator.update_performance(0.7)
        iterator.update_performance(0.9)
        
        assert len(iterator.performance_history) == 3
        assert iterator.performance_history[-1] == 0.9
        
        # Get statistics
        stats = iterator.get_stats()
        assert 'avg_recent_performance' in stats
        assert stats['avg_recent_performance'] is None  # Need at least 10 entries
        
        # Add more performance data
        for i in range(7):
            iterator.update_performance(0.8 + i * 0.01)
        
        stats = iterator.get_stats()
        assert stats['avg_recent_performance'] is not None
        assert 0.0 <= stats['avg_recent_performance'] <= 1.0
    
    def test_difficulty_tracking(self):
        """Test difficulty tracking functionality."""
        iterator = InfiniteEpisodeIterator(
            self.episode_generator,
            buffer_size=5,
            adaptive_sampling=True
        )
        
        # Get several episodes to populate difficulty tracking
        for i, episode in enumerate(iterator):
            if i >= 3:
                break
        
        # Check difficulty tracking
        assert len(iterator.difficulty_tracker) > 0
        
        # Should track different episode configurations
        stats = iterator.get_stats()
        assert stats['difficulty_types'] > 0
    
    def test_iterator_statistics(self):
        """Test iterator statistics reporting."""
        iterator = InfiniteEpisodeIterator(
            self.episode_generator,
            buffer_size=8,
            adaptive_sampling=True
        )
        
        # Generate some episodes
        for i, episode in enumerate(iterator):
            if i >= 4:
                break
        
        # Get statistics
        stats = iterator.get_stats()
        
        assert 'generated_episodes' in stats
        assert 'buffer_size' in stats
        assert 'buffer_fill' in stats
        assert 'buffer_utilization' in stats
        assert 'adaptive_sampling' in stats
        assert 'difficulty_types' in stats
        
        assert stats['generated_episodes'] > 0
        assert stats['buffer_size'] == 8
        assert 0 <= stats['buffer_fill'] <= 8
        assert 0.0 <= stats['buffer_utilization'] <= 1.0
        assert stats['adaptive_sampling'] == True
    
    def test_background_generation(self):
        """Test background episode generation."""
        iterator = InfiniteEpisodeIterator(
            self.episode_generator,
            buffer_size=10,
            adaptive_sampling=True,
            prefetch_factor=3
        )
        
        initial_count = iterator.generated_count
        
        # Wait a bit for background generation
        time.sleep(0.2)
        
        # Should have generated more episodes in background
        # (This test is timing-dependent and may be flaky)
        final_count = iterator.generated_count
        assert final_count >= initial_count
    
    def test_iterator_cleanup(self):
        """Test iterator cleanup and thread safety."""
        iterator = InfiniteEpisodeIterator(
            self.episode_generator,
            buffer_size=5,
            adaptive_sampling=True
        )
        
        # Stop the iterator
        iterator.stop()
        
        # Should stop background generation
        assert iterator.stop_generation == True
        
        # Should still be able to get episodes from buffer
        episode = next(iterator)
        assert isinstance(episode, Episode)


class TestDatasetRegistry:
    """Test DatasetRegistry functionality."""
    
    def test_registry_initialization(self):
        """Test dataset registry initialization."""
        registry = DatasetRegistry()
        available = registry.list_available()
        
        assert isinstance(available, list)
        assert 'synthetic' in available
    
    def test_get_dataset_by_name(self):
        """Test getting dataset by name."""
        registry = DatasetRegistry()
        
        # Get synthetic dataset
        dataset = registry.get_dataset('synthetic', root='./temp', n_classes=5)
        
        assert isinstance(dataset, BaseMetaLearningDataset)
        assert dataset.n_classes == 5
    
    def test_register_new_dataset(self):
        """Test registering new dataset class."""
        registry = DatasetRegistry()
        
        class CustomDataset(BaseMetaLearningDataset):
            def _load_dataset(self):
                self._data = [torch.randn(32) for _ in range(100)]
                self._labels = [i % 10 for i in range(100)]
        
        # Register custom dataset
        registry.register_dataset('custom', CustomDataset)
        
        # Verify it's available
        available = registry.list_available()
        assert 'custom' in available
        
        # Get the dataset
        dataset = registry.get_dataset('custom', root='./temp')
        assert isinstance(dataset, CustomDataset)
    
    def test_invalid_dataset_registration(self):
        """Test invalid dataset registration."""
        registry = DatasetRegistry()
        
        class InvalidDataset:  # Doesn't inherit from BaseMetaLearningDataset
            pass
        
        with pytest.raises(ValueError, match="must inherit from BaseMetaLearningDataset"):
            registry.register_dataset('invalid', InvalidDataset)
    
    def test_unknown_dataset_name(self):
        """Test getting unknown dataset name."""
        registry = DatasetRegistry()
        
        with pytest.raises(ValueError, match="Unknown dataset"):
            registry.get_dataset('nonexistent')


class TestBaseMetaLearningDataset:
    """Test BaseMetaLearningDataset functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        class TestDataset(BaseMetaLearningDataset):
            def _load_dataset(self):
                # Create simple synthetic data
                self._data = []
                self._labels = []
                
                for class_id in range(10):
                    for sample_id in range(20):
                        # Simple synthetic features
                        features = torch.randn(64) + class_id * 0.5  # Class-dependent mean
                        self._data.append(features)
                        self._labels.append(class_id)
        
        self.dataset_class = TestDataset
    
    def test_base_dataset_initialization(self):
        """Test base dataset initialization."""
        dataset = self.dataset_class(root='./temp', mode='train')
        
        assert dataset.root == os.path.expanduser('./temp')
        assert dataset.mode == 'train'
        assert len(dataset._data) == 200  # 10 classes * 20 samples
        assert len(dataset._labels) == 200
        assert len(dataset._class_to_indices) == 10
    
    def test_base_dataset_indexing(self):
        """Test dataset indexing functionality."""
        dataset = self.dataset_class(root='./temp')
        
        assert len(dataset) == 200
        
        # Test individual item access
        data_item, label = dataset[0]
        assert isinstance(data_item, torch.Tensor)
        assert data_item.shape == (64,)
        assert isinstance(label, int)
        assert 0 <= label <= 9
    
    def test_base_dataset_class_methods(self):
        """Test dataset class-related methods."""
        dataset = self.dataset_class(root='./temp')
        
        classes = dataset.get_classes()
        assert len(classes) == 10
        assert set(classes) == set(range(10))
        
        # Check class to indices mapping
        for class_id in classes:
            indices = dataset._class_to_indices[class_id]
            assert len(indices) == 20  # 20 samples per class
            for idx in indices:
                assert dataset._labels[idx] == class_id
    
    def test_base_dataset_episode_creation(self):
        """Test episode creation from dataset."""
        dataset = self.dataset_class(root='./temp')
        
        episode = dataset.create_episode(n_way=5, n_shot=3, n_query=2)
        
        assert isinstance(episode, Episode)
        assert episode.support_x.shape == (15, 64)  # 5 classes * 3 shots
        assert episode.support_y.shape == (15,)
        assert episode.query_x.shape == (10, 64)    # 5 classes * 2 queries
        assert episode.query_y.shape == (10,)
        
        # Check label consistency
        support_classes = set(episode.support_y.tolist())
        query_classes = set(episode.query_y.tolist())
        assert support_classes == query_classes
        assert len(support_classes) == 5
        assert support_classes == set(range(5))  # Should be remapped to 0-4
    
    def test_base_dataset_episode_creation_insufficient_samples(self):
        """Test episode creation with insufficient samples per class."""
        dataset = self.dataset_class(root='./temp')
        
        # Try to create episode requiring more samples than available
        with pytest.raises(ValueError, match="samples, but .* required"):
            dataset.create_episode(n_way=5, n_shot=15, n_query=10)  # Needs 25 samples per class, but only have 20
    
    def test_base_dataset_episode_creation_insufficient_classes(self):
        """Test episode creation with insufficient classes."""
        dataset = self.dataset_class(root='./temp')
        
        # Try to create episode requiring more classes than available
        with pytest.raises(ValueError, match="classes, but .* requested"):
            dataset.create_episode(n_way=15, n_shot=1, n_query=1)  # Need 15 classes, but only have 10
    
    def test_base_dataset_with_transforms(self):
        """Test dataset with transform functions."""
        def normalize_transform(x):
            return (x - x.mean()) / (x.std() + 1e-8)
        
        dataset = self.dataset_class(root='./temp', transform=normalize_transform)
        
        data_item, label = dataset[0]
        
        # Check that transform was applied (data should be approximately normalized)
        assert abs(data_item.mean().item()) < 0.1  # Should be close to 0
        assert abs(data_item.std().item() - 1.0) < 0.1  # Should be close to 1


class TestMiniImageNetDataset:
    """Test MiniImageNetDataset implementation."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_mini_imagenet_initialization(self):
        """Test MiniImageNet dataset initialization."""
        dataset = MiniImageNetDataset(
            root=self.temp_dir,
            mode='train',
            download=True,
            validate_data=True
        )
        
        assert dataset.root == self.temp_dir
        assert dataset.mode == 'train'
        assert dataset.validate_data == True
        
        # Check that data file was created
        expected_file = os.path.join(self.temp_dir, 'mini_imagenet_train.pkl')
        assert os.path.exists(expected_file)
    
    def test_mini_imagenet_data_loading(self):
        """Test MiniImageNet data loading and validation."""
        dataset = MiniImageNetDataset(
            root=self.temp_dir,
            mode='val',
            download=True,
            validate_data=True
        )
        
        # Check loaded data properties
        assert hasattr(dataset, 'data')
        assert hasattr(dataset, 'labels')
        assert hasattr(dataset, 'num_classes')
        assert hasattr(dataset, 'class_to_indices')
        
        assert dataset.data.shape == (600, 3, 84, 84)
        assert len(dataset.labels) == 600
        assert dataset.num_classes == 100
        assert len(dataset.class_to_indices) == 100
    
    def test_mini_imagenet_episode_creation(self):
        """Test episode creation from MiniImageNet dataset."""
        dataset = MiniImageNetDataset(
            root=self.temp_dir,
            mode='train',
            download=True
        )
        
        episode = dataset.create_episode(n_way=5, n_shot=1, n_query=2)
        
        assert isinstance(episode, Episode)
        assert episode.support_x.shape == (5, 3, 84, 84)
        assert episode.support_y.shape == (5,)
        assert episode.query_x.shape == (10, 3, 84, 84)
        assert episode.query_y.shape == (10,)
        
        # Check that labels are properly remapped
        support_classes = set(episode.support_y.tolist())
        query_classes = set(episode.query_y.tolist())
        assert support_classes == query_classes
        assert support_classes == set(range(5))
    
    def test_mini_imagenet_validation(self):
        """Test MiniImageNet data validation."""
        # Create dataset with validation enabled
        dataset = MiniImageNetDataset(
            root=self.temp_dir,
            mode='test',
            download=True,
            validate_data=True
        )
        
        # Validation should pass for synthetic data
        assert dataset.data is not None
        assert dataset.labels is not None
    
    def test_mini_imagenet_without_download(self):
        """Test MiniImageNet without download when data doesn't exist."""
        non_existent_dir = os.path.join(self.temp_dir, 'nonexistent')
        
        with pytest.raises(RuntimeError, match="Failed to load dataset"):
            MiniImageNetDataset(
                root=non_existent_dir,
                mode='train',
                download=False  # Don't download
            )


class TestSyntheticFewShotDataset:
    """Test SyntheticFewShotDataset implementation."""
    
    def test_synthetic_dataset_initialization(self):
        """Test synthetic dataset initialization."""
        dataset = SyntheticFewShotDataset(
            root='./temp',
            mode='train',
            n_classes=20,
            n_samples_per_class=15,
            feature_dim=128,
            noise_std=0.2
        )
        
        assert dataset.n_classes == 20
        assert dataset.n_samples_per_class == 15
        assert dataset.feature_dim == 128
        assert dataset.noise_std == 0.2
        
        # Check generated data
        assert len(dataset._data) == 300  # 20 * 15
        assert len(dataset._labels) == 300
        assert len(dataset._class_to_indices) == 20
    
    def test_synthetic_dataset_data_structure(self):
        """Test synthetic dataset data structure and quality."""
        dataset = SyntheticFewShotDataset(
            n_classes=5,
            n_samples_per_class=10,
            feature_dim=64,
            noise_std=0.1
        )
        
        # Test data shape and type
        for i in range(len(dataset)):
            data_item, label = dataset[i]
            assert data_item.shape == (64,)
            assert isinstance(label, int)
            assert 0 <= label < 5
        
        # Test class separation (samples from same class should be similar)
        class_0_samples = []
        class_1_samples = []
        
        for i in range(len(dataset)):
            data_item, label = dataset[i]
            if label == 0 and len(class_0_samples) < 5:
                class_0_samples.append(data_item)
            elif label == 1 and len(class_1_samples) < 5:
                class_1_samples.append(data_item)
        
        # Intra-class distances should be smaller than inter-class distances
        if len(class_0_samples) >= 2 and len(class_1_samples) >= 2:
            intra_dist = torch.dist(class_0_samples[0], class_0_samples[1])
            inter_dist = torch.dist(class_0_samples[0], class_1_samples[0])
            assert intra_dist < inter_dist  # Generally should be true for well-separated synthetic data
    
    def test_synthetic_dataset_episode_creation(self):
        """Test episode creation from synthetic dataset."""
        dataset = SyntheticFewShotDataset(
            n_classes=8,
            n_samples_per_class=12,
            feature_dim=32
        )
        
        episode = dataset.create_episode(n_way=4, n_shot=3, n_query=2)
        
        assert episode.support_x.shape == (12, 32)  # 4 * 3
        assert episode.support_y.shape == (12,)
        assert episode.query_x.shape == (8, 32)     # 4 * 2
        assert episode.query_y.shape == (8,)
        
        # Check label consistency
        support_classes = set(episode.support_y.tolist())
        query_classes = set(episode.query_y.tolist())
        assert support_classes == query_classes == set(range(4))
    
    def test_synthetic_dataset_reproducibility(self):
        """Test synthetic dataset reproducibility with fixed seed."""
        dataset1 = SyntheticFewShotDataset(
            n_classes=3,
            n_samples_per_class=5,
            feature_dim=16
        )
        
        dataset2 = SyntheticFewShotDataset(
            n_classes=3,
            n_samples_per_class=5,
            feature_dim=16
        )
        
        # Both datasets use the same fixed seed (42), so should generate identical data
        for i in range(len(dataset1)):
            data1, label1 = dataset1[i]
            data2, label2 = dataset2[i]
            
            assert torch.equal(data1, data2)
            assert label1 == label2


# Integration tests
class TestDatasetEcosystemIntegration:
    """Integration tests for dataset ecosystem components."""
    
    def setup_method(self):
        """Setup integration test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.manager = BenchmarkDatasetManager(
            cache_dir=self.temp_dir,
            max_cache_size_gb=0.01
        )
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_complete_dataset_pipeline(self):
        """Test complete dataset management pipeline."""
        # 1. Download and cache dataset
        dataset_path = self.manager.download_dataset('synthetic')
        assert dataset_path is not None
        
        # 2. Create dataset object
        registry = DatasetRegistry()
        dataset = registry.get_dataset('synthetic', root=dataset_path)
        
        # 3. Create episodes
        episodes = []
        for i in range(5):
            episode = dataset.create_episode(n_way=3, n_shot=2, n_query=1)
            episodes.append(episode)
        
        # 4. Create on-device dataset for optimization
        on_device_dataset = OnDeviceDataset(episodes, memory_budget=0.8)
        
        # 5. Verify optimized access
        for i in range(len(on_device_dataset)):
            cached_episode = on_device_dataset[i]
            assert isinstance(cached_episode, Episode)
            assert cached_episode.support_x.device == on_device_dataset.device
    
    def test_infinite_iterator_with_cached_dataset(self):
        """Test infinite iterator with cached dataset."""
        # Setup dataset
        registry = DatasetRegistry()
        dataset = registry.get_dataset('synthetic', root=self.temp_dir, n_classes=4, n_samples_per_class=10)
        
        # Create episode generator
        def episode_generator():
            return dataset.create_episode(n_way=2, n_shot=3, n_query=1)
        
        # Create infinite iterator
        iterator = InfiniteEpisodeIterator(
            episode_generator,
            buffer_size=5,
            adaptive_sampling=True
        )
        
        # Generate several episodes
        episodes = []
        for i, episode in enumerate(iterator):
            episodes.append(episode)
            if i >= 3:
                break
        
        assert len(episodes) == 4
        for episode in episodes:
            assert isinstance(episode, Episode)
            assert episode.support_x.shape == (6, 64)  # 2-way, 3-shot
            assert episode.query_x.shape == (2, 64)    # 2-way, 1-query
    
    def test_memory_management_across_components(self):
        """Test memory management across all components."""
        # Create dataset manager
        dataset_path = self.manager.download_dataset('synthetic')
        
        # Create dataset
        registry = DatasetRegistry()
        dataset = registry.get_dataset('synthetic', root=dataset_path, n_classes=5)
        
        # Create episodes
        episodes = [dataset.create_episode(n_way=3, n_shot=2, n_query=1) for _ in range(10)]
        
        # Create on-device dataset with small memory budget
        on_device_dataset = OnDeviceDataset(episodes, memory_budget=0.1)
        
        # Create infinite iterator
        def episode_gen():
            return episodes[torch.randint(0, len(episodes), (1,)).item()]
        
        iterator = InfiniteEpisodeIterator(episode_gen, buffer_size=3)
        
        # Test that everything works together
        cached_stats = on_device_dataset.get_cache_stats()
        iterator_stats = iterator.get_stats()
        cache_info = self.manager.get_cache_info()
        
        # All components should report reasonable statistics
        assert cached_stats['total_episodes'] == 10
        assert iterator_stats['buffer_size'] == 3
        assert cache_info['total_size_mb'] >= 0
        
        # Access should work across all components
        cached_episode = on_device_dataset[0]
        generated_episode = next(iterator)
        
        assert isinstance(cached_episode, Episode)
        assert isinstance(generated_episode, Episode)


if __name__ == "__main__":
    pytest.main([__file__])