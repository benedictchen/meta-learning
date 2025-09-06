"""
Tests for dataset classes and data management utilities.
"""
import pytest
import torch
import tempfile
import os
import pickle
import shutil
from unittest.mock import Mock, patch, MagicMock

from meta_learning.data_utils.datasets import (
    MiniImageNetDataset, BenchmarkDatasetManager, OnDeviceDataset,
    InfiniteEpisodeIterator, BaseMetaLearningDataset, SyntheticFewShotDataset
)
from meta_learning.core.episode import Episode


class TestMiniImageNetDataset:
    """Test MiniImageNetDataset functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.dataset = MiniImageNetDataset(
            root=self.temp_dir,
            mode='train',
            download=True,
            validate_data=True
        )
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test dataset initialization."""
        assert self.dataset.root == self.temp_dir
        assert self.dataset.mode == 'train'
        assert self.dataset.validate_data is True
        assert hasattr(self.dataset, 'data')
        assert hasattr(self.dataset, 'labels')
        assert hasattr(self.dataset, 'num_classes')
    
    def test_data_download_and_validation(self):
        """Test automatic data download and validation."""
        # Data should be downloaded and loaded
        assert self.dataset.data.shape[0] > 0
        assert self.dataset.labels.shape[0] > 0
        assert self.dataset.data.shape[0] == self.dataset.labels.shape[0]
        
        # Should have expected shape for images
        assert len(self.dataset.data.shape) == 4  # [N, C, H, W]
        assert self.dataset.data.shape[1:] == (3, 84, 84)
    
    def test_data_file_exists(self):
        """Test data file existence check."""
        # Should exist after initialization
        assert self.dataset._data_exists() is True
        
        # Should not exist if file removed
        os.remove(self.dataset.data_path)
        assert self.dataset._data_exists() is False
    
    def test_episode_creation(self):
        """Test episode creation from dataset."""
        episode = self.dataset.create_episode(n_way=3, n_shot=2, n_query=5)
        
        assert isinstance(episode, Episode)
        assert episode.support_x.shape == (6, 3, 84, 84)  # 3 classes * 2 shots
        assert episode.support_y.shape == (6,)
        assert episode.query_x.shape == (15, 3, 84, 84)   # 3 classes * 5 queries
        assert episode.query_y.shape == (15,)
        
        # Check label remapping
        support_classes = torch.unique(episode.support_y)
        query_classes = torch.unique(episode.query_y)
        assert len(support_classes) == 3
        assert len(query_classes) == 3
        assert torch.equal(support_classes, torch.arange(3))
    
    def test_episode_creation_insufficient_classes(self):
        """Test error handling when requesting too many classes."""
        # Create dataset with limited classes
        small_data = {
            'data': torch.randn(4, 3, 84, 84),
            'labels': torch.tensor([0, 0, 1, 1])  # Only 2 classes
        }
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
            pickle.dump(small_data, f)
            temp_path = f.name
        
        try:
            dataset = MiniImageNetDataset(root=os.path.dirname(temp_path), download=False)
            dataset.data_path = temp_path
            dataset._load_data()
            
            with pytest.raises(ValueError, match="Dataset has 2 classes, need 5"):
                dataset.create_episode(n_way=5)
        finally:
            os.unlink(temp_path)
    
    def test_data_validation(self):
        """Test data quality validation."""
        # Test with valid data
        self.dataset.data = torch.randn(10, 3, 84, 84)
        self.dataset.labels = torch.arange(10)
        self.dataset._validate_loaded_data()  # Should not raise
        
        # Test with NaN data
        self.dataset.data = torch.tensor([[[[float('nan')]]]])
        self.dataset.labels = torch.tensor([0])
        
        with pytest.raises(ValueError, match="Dataset contains NaN values"):
            self.dataset._validate_loaded_data()
    
    def test_transforms_application(self):
        """Test transform application during episode creation."""
        def dummy_transform(x):
            return x * 2
        
        dataset = MiniImageNetDataset(
            root=self.temp_dir,
            transform=dummy_transform,
            download=False
        )
        dataset.data = torch.ones(20, 3, 84, 84)
        dataset.labels = torch.repeat_interleave(torch.arange(5), 4)
        dataset._load_data()
        
        episode = dataset.create_episode(n_way=2, n_shot=1, n_query=1)
        
        # Data should be transformed (multiplied by 2)
        assert torch.allclose(episode.support_x, torch.ones_like(episode.support_x) * 2)


class TestBenchmarkDatasetManager:
    """Test BenchmarkDatasetManager functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_cache = tempfile.mkdtemp()
        self.manager = BenchmarkDatasetManager(
            cache_dir=self.temp_cache,
            max_cache_size_gb=0.1  # Small for testing
        )
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_cache, ignore_errors=True)
    
    def test_initialization(self):
        """Test manager initialization."""
        assert os.path.exists(self.manager.cache_dir)
        assert self.manager.max_cache_size_gb == 0.1
        assert 'mini_imagenet' in self.manager.dataset_registry
        assert 'synthetic' in self.manager.dataset_registry
    
    def test_list_available_datasets(self):
        """Test listing available datasets."""
        datasets = self.manager.list_available_datasets()
        
        assert isinstance(datasets, dict)
        assert 'mini_imagenet' in datasets
        assert 'description' in datasets['mini_imagenet']
        assert 'file_size_mb' in datasets['mini_imagenet']
        assert 'cached' in datasets['mini_imagenet']
    
    def test_download_synthetic_dataset(self):
        """Test downloading synthetic dataset (no actual download)."""
        dataset_path = self.manager.download_dataset('synthetic')
        
        assert dataset_path is not None
        assert os.path.exists(dataset_path)
        assert self.manager._is_dataset_cached('synthetic')
    
    def test_download_with_fallback(self):
        """Test download with URL fallback."""
        # Mock the download process
        dataset_path = self.manager.download_dataset('mini_imagenet')
        
        assert dataset_path is not None
        assert os.path.exists(dataset_path)
        
        # Check marker file exists
        marker_path = os.path.join(dataset_path, '.download_complete')
        assert os.path.exists(marker_path)
    
    def test_force_redownload(self):
        """Test forced redownload."""
        # Download once
        self.manager.download_dataset('synthetic')
        original_time = os.path.getmtime(os.path.join(self.temp_cache, 'synthetic'))
        
        # Force redownload
        import time
        time.sleep(0.1)  # Ensure different timestamp
        self.manager.download_dataset('synthetic', force_redownload=True)
        
        new_time = os.path.getmtime(os.path.join(self.temp_cache, 'synthetic'))
        assert new_time > original_time
    
    def test_cache_info(self):
        """Test cache information retrieval."""
        self.manager.download_dataset('synthetic')
        
        cache_info = self.manager.get_cache_info()
        
        assert 'cache_dir' in cache_info
        assert 'total_size_mb' in cache_info
        assert 'datasets' in cache_info
        assert 'synthetic' in cache_info['datasets']
    
    def test_cache_size_management(self):
        """Test automatic cache size management."""
        # Set very small cache limit
        self.manager.max_cache_size_bytes = 1000  # 1KB
        
        # Download multiple datasets to exceed limit
        self.manager.download_dataset('synthetic')
        self.manager.download_dataset('mini_imagenet')
        
        # Cache management should be triggered
        self.manager._manage_cache_size()
        
        # At least one dataset should remain
        cached_datasets = os.listdir(self.manager.cache_dir)
        assert len(cached_datasets) >= 0
    
    def test_clear_cache(self):
        """Test cache clearing functionality."""
        # Download some datasets
        self.manager.download_dataset('synthetic')
        self.manager.download_dataset('mini_imagenet')
        
        # Clear specific dataset
        self.manager.clear_cache('synthetic')
        assert not self.manager._is_dataset_cached('synthetic')
        assert self.manager._is_dataset_cached('mini_imagenet')
        
        # Clear all cache
        self.manager.clear_cache()
        assert not self.manager._is_dataset_cached('mini_imagenet')
    
    def test_unknown_dataset(self):
        """Test error handling for unknown dataset."""
        with pytest.raises(ValueError, match="Unknown dataset"):
            self.manager.download_dataset('nonexistent_dataset')


class TestOnDeviceDataset:
    """Test OnDeviceDataset functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create sample episodes
        self.episodes = []
        for i in range(10):
            episode = Episode(
                support_x=torch.randn(5, 3, 32, 32),
                support_y=torch.arange(5),
                query_x=torch.randn(15, 3, 32, 32),
                query_y=torch.repeat_interleave(torch.arange(5), 3)
            )
            self.episodes.append(episode)
        
        self.dataset = OnDeviceDataset(
            episodes=self.episodes,
            memory_budget=0.5,
            enable_compression=False,  # Disable for simpler testing
            enable_mixed_precision=False
        )
    
    def test_initialization(self):
        """Test dataset initialization."""
        assert len(self.dataset) == 10
        assert self.dataset.memory_budget == 0.5
        assert isinstance(self.dataset.cached_episodes, dict)
        assert isinstance(self.dataset.access_counts, dict)
    
    def test_episode_access(self):
        """Test episode access and caching."""
        episode = self.dataset[0]
        
        assert isinstance(episode, Episode)
        assert episode.support_x.shape == (5, 3, 32, 32)
        assert episode.query_x.shape == (15, 3, 32, 32)
        
        # Should track access
        assert self.dataset.access_counts[0] >= 1
    
    def test_cache_statistics(self):
        """Test cache performance statistics."""
        # Access some episodes
        for i in range(5):
            _ = self.dataset[i]
        
        stats = self.dataset.get_cache_stats()
        
        assert 'total_episodes' in stats
        assert 'cached_episodes' in stats
        assert 'cache_hit_rate' in stats
        assert 'memory_used_mb' in stats
        assert stats['total_episodes'] == 10
    
    def test_memory_optimization(self):
        """Test memory optimization features."""
        # Test tensor optimization
        test_tensor = torch.randn(100, 100).float()
        optimized = self.dataset._optimize_tensor(test_tensor)
        
        assert optimized.device == self.dataset.device
        
        # Test episode size estimation
        episode = self.episodes[0]
        size = self.dataset._estimate_episode_size(episode)
        assert size > 0


class TestInfiniteEpisodeIterator:
    """Test InfiniteEpisodeIterator functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.episode_count = 0
        
        def episode_generator():
            self.episode_count += 1
            return Episode(
                support_x=torch.randn(5, 3, 32, 32),
                support_y=torch.arange(5),
                query_x=torch.randn(15, 3, 32, 32),
                query_y=torch.repeat_interleave(torch.arange(5), 3)
            )
        
        self.iterator = InfiniteEpisodeIterator(
            episode_generator=episode_generator,
            buffer_size=10,
            adaptive_sampling=True,
            prefetch_factor=2
        )
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if hasattr(self.iterator, 'stop'):
            self.iterator.stop()
    
    def test_initialization(self):
        """Test iterator initialization."""
        assert self.iterator.buffer_size == 10
        assert self.iterator.adaptive_sampling is True
        assert self.iterator.prefetch_factor == 2
        assert len(self.iterator.episode_buffer) == 10
    
    def test_episode_generation(self):
        """Test infinite episode generation."""
        episode1 = next(self.iterator)
        episode2 = next(self.iterator)
        
        assert isinstance(episode1, Episode)
        assert isinstance(episode2, Episode)
        assert self.episode_count >= 2
    
    def test_performance_tracking(self):
        """Test performance tracking for adaptive sampling."""
        # Generate some episodes
        for _ in range(5):
            next(self.iterator)
        
        # Update performance
        self.iterator.update_performance(0.8)
        self.iterator.update_performance(0.7)
        
        assert len(self.iterator.performance_history) == 2
    
    def test_statistics(self):
        """Test iterator statistics."""
        # Generate some episodes
        for _ in range(3):
            next(self.iterator)
        
        stats = self.iterator.get_stats()
        
        assert 'generated_episodes' in stats
        assert 'buffer_utilization' in stats
        assert 'adaptive_sampling' in stats
        assert stats['adaptive_sampling'] is True


class TestBaseMetaLearningDataset:
    """Test BaseMetaLearningDataset functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create a concrete implementation for testing
        class TestDataset(BaseMetaLearningDataset):
            def __init__(self):
                data = [(torch.randn(3, 32, 32), i % 5) for i in range(50)]
                super().__init__(data)
            
            def __getitem__(self, index):
                return self.data[index]
        
        self.dataset = TestDataset()
    
    def test_initialization(self):
        """Test dataset initialization."""
        assert len(self.dataset) == 50
        assert len(self.dataset.get_classes()) == 5
        assert hasattr(self.dataset, '_class_to_indices')
    
    def test_class_indexing(self):
        """Test class-to-indices mapping."""
        classes = self.dataset.get_classes()
        
        for class_id in classes:
            indices = self.dataset._class_to_indices[class_id]
            assert len(indices) > 0
            
            # Verify all indices contain the correct class
            for idx in indices:
                _, label = self.dataset[idx]
                assert label == class_id
    
    def test_episode_creation(self):
        """Test episode creation from base dataset."""
        episode = self.dataset.create_episode(n_way=3, n_shot=2, n_query=4)
        
        assert isinstance(episode, Episode)
        assert episode.support_x.shape == (6, 3, 32, 32)
        assert episode.query_x.shape == (12, 3, 32, 32)
        
        # Check proper label remapping
        support_labels = torch.unique(episode.support_y)
        query_labels = torch.unique(episode.query_y)
        assert len(support_labels) == 3
        assert len(query_labels) == 3
        assert torch.equal(support_labels, torch.arange(3))
    
    def test_insufficient_classes_error(self):
        """Test error when requesting more classes than available."""
        with pytest.raises(ValueError, match="Dataset has 5 classes"):
            self.dataset.create_episode(n_way=10)
    
    def test_insufficient_samples_handling(self):
        """Test handling of insufficient samples per class."""
        # Create dataset with very few samples per class
        class SmallDataset(BaseMetaLearningDataset):
            def __init__(self):
                # Only 1 sample per class
                data = [(torch.randn(3, 32, 32), i) for i in range(3)]
                super().__init__(data)
            
            def __getitem__(self, index):
                return self.data[index]
        
        small_dataset = SmallDataset()
        
        # Should handle with replacement
        episode = small_dataset.create_episode(n_way=2, n_shot=2, n_query=2)
        assert episode.support_x.shape == (4, 3, 32, 32)
        assert episode.query_x.shape == (4, 3, 32, 32)


class TestSyntheticFewShotDataset:
    """Test SyntheticFewShotDataset functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.dataset = SyntheticFewShotDataset(
            n_classes=10,
            dim=64,
            noise=0.1,
            image_mode=False
        )
    
    def test_initialization(self):
        """Test dataset initialization."""
        assert self.dataset.n_classes == 10
        assert self.dataset.dim == 64
        assert self.dataset.noise == 0.1
        assert self.dataset.image_mode is False
    
    def test_data_generation(self):
        """Test synthetic data generation."""
        # Should have generated data
        assert hasattr(self.dataset, 'data')
        assert hasattr(self.dataset, 'labels')
        
        # Check dimensions
        if self.dataset.image_mode:
            expected_shape = (self.dataset.n_classes * 20, 3, 32, 32)
        else:
            expected_shape = (self.dataset.n_classes * 20, self.dataset.dim)
        
        assert self.dataset.data.shape == expected_shape
        assert len(self.dataset.labels) == self.dataset.n_classes * 20
    
    def test_sample_support_query(self):
        """Test support-query sampling."""
        xs, ys, xq, yq = self.dataset.sample_support_query(
            n_way=3, n_shot=2, n_query=5
        )
        
        assert xs.shape == (6, self.dataset.dim)  # 3 classes * 2 shots
        assert ys.shape == (6,)
        assert xq.shape == (15, self.dataset.dim)  # 3 classes * 5 queries
        assert yq.shape == (15,)
        
        # Check label remapping
        assert torch.unique(ys).tolist() == [0, 1, 2]
        assert torch.unique(yq).tolist() == [0, 1, 2]
    
    def test_image_mode(self):
        """Test image mode data generation."""
        image_dataset = SyntheticFewShotDataset(
            n_classes=5,
            dim=64,
            image_mode=True
        )
        
        assert image_dataset.data.shape == (100, 3, 32, 32)
        
        # Test sampling in image mode
        xs, ys, xq, yq = image_dataset.sample_support_query(2, 1, 3)
        assert xs.shape == (2, 3, 32, 32)
        assert xq.shape == (6, 3, 32, 32)


if __name__ == "__main__":
    pytest.main([__file__])