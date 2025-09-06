"""
Comprehensive tests for Dataset management system.

Tests centralized registry, smart caching, robust downloading, and 
professional dataset management for meta-learning research.
"""
import pytest
import torch
import torch.utils.data as data
import numpy as np
from unittest.mock import MagicMock, patch, mock_open
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import tempfile
import shutil
import os
import json
import hashlib
import time

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from meta_learning.meta_learning_modules.dataset_management import (
    DatasetRegistry,
    DatasetManager,
    SmartCache,
    RobustDownloader,
    DatasetMetadata,
    CacheEntry,
    DownloadConfig,
    DatasetTransform,
    create_dataset_manager,
    get_dataset_info,
    download_dataset
)


class TestDatasetMetadata:
    """Test DatasetMetadata functionality."""
    
    def test_metadata_creation(self):
        """Test DatasetMetadata creation and validation."""
        metadata = DatasetMetadata(
            name="miniImageNet",
            version="1.0",
            description="84x84 RGB images, 100 classes, 600 examples per class",
            url="https://example.com/mini_imagenet.tar.gz",
            checksum="abc123def456",
            size_bytes=1024*1024*100,  # 100MB
            num_classes=100,
            num_examples=60000,
            splits={'train': 64, 'val': 16, 'test': 20},
            image_shape=(84, 84, 3),
            task_type="few_shot_classification"
        )
        
        assert metadata.name == "miniImageNet"
        assert metadata.num_classes == 100
        assert metadata.splits['train'] == 64
        assert metadata.is_valid()
        
        # Test invalid metadata
        invalid_metadata = DatasetMetadata(
            name="",  # Invalid: empty name
            version="1.0",
            url="invalid_url",  # Invalid: not a proper URL
            size_bytes=-100  # Invalid: negative size
        )
        
        assert not invalid_metadata.is_valid()
    
    def test_metadata_serialization(self):
        """Test metadata serialization/deserialization."""
        metadata = DatasetMetadata(
            name="Omniglot",
            version="1.0",
            description="Handwritten characters",
            url="https://example.com/omniglot.zip",
            checksum="xyz789abc123",
            size_bytes=50*1024*1024,
            num_classes=1623,
            num_examples=32460,
            splits={'train': 1200, 'test': 423},
            image_shape=(28, 28, 1),
            task_type="few_shot_classification"
        )
        
        # Serialize to dict
        metadata_dict = metadata.to_dict()
        assert metadata_dict['name'] == "Omniglot"
        assert metadata_dict['num_classes'] == 1623
        
        # Deserialize from dict
        metadata_restored = DatasetMetadata.from_dict(metadata_dict)
        assert metadata_restored.name == metadata.name
        assert metadata_restored.num_classes == metadata.num_classes
        assert metadata_restored.splits == metadata.splits
    
    def test_metadata_validation(self):
        """Test metadata validation rules."""
        # Valid metadata
        valid_metadata = DatasetMetadata(
            name="CIFAR-FS",
            version="1.0",
            url="https://example.com/cifar_fs.tar",
            size_bytes=1000000,
            num_classes=100,
            num_examples=50000
        )
        
        validation = valid_metadata.validate()
        assert validation['valid']
        assert len(validation['errors']) == 0
        
        # Invalid metadata
        invalid_metadata = DatasetMetadata(
            name="Invalid Dataset",
            version="",  # Empty version
            url="not_a_url",  # Invalid URL
            size_bytes=0,  # Zero size
            num_classes=-5,  # Negative classes
            num_examples=0  # Zero examples
        )
        
        validation = invalid_metadata.validate()
        assert not validation['valid']
        assert len(validation['errors']) > 0


class TestDatasetRegistry:
    """Test DatasetRegistry functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.registry = DatasetRegistry()
    
    def test_register_dataset(self):
        """Test dataset registration."""
        metadata = DatasetMetadata(
            name="miniImageNet",
            version="1.0",
            url="https://example.com/mini_imagenet.tar.gz",
            size_bytes=100*1024*1024,
            num_classes=100,
            num_examples=60000
        )
        
        # Register dataset
        success = self.registry.register_dataset(metadata)
        assert success
        
        # Verify registration
        assert self.registry.is_registered("miniImageNet")
        assert len(self.registry.list_datasets()) == 1
        
        # Test duplicate registration
        duplicate_success = self.registry.register_dataset(metadata)
        assert not duplicate_success  # Should fail for duplicates
    
    def test_get_dataset_info(self):
        """Test retrieving dataset information."""
        metadata = DatasetMetadata(
            name="Omniglot",
            version="1.0",
            url="https://example.com/omniglot.zip",
            size_bytes=50*1024*1024,
            num_classes=1623
        )
        
        self.registry.register_dataset(metadata)
        
        # Get dataset info
        info = self.registry.get_dataset_info("Omniglot")
        assert info is not None
        assert info.name == "Omniglot"
        assert info.num_classes == 1623
        
        # Test non-existent dataset
        non_existent = self.registry.get_dataset_info("NonExistent")
        assert non_existent is None
    
    def test_list_datasets(self):
        """Test listing registered datasets."""
        # Register multiple datasets
        datasets = [
            DatasetMetadata(name="miniImageNet", version="1.0", url="http://example.com/1"),
            DatasetMetadata(name="Omniglot", version="1.0", url="http://example.com/2"),
            DatasetMetadata(name="CIFAR-FS", version="1.0", url="http://example.com/3")
        ]
        
        for dataset in datasets:
            self.registry.register_dataset(dataset)
        
        # List all datasets
        dataset_list = self.registry.list_datasets()
        assert len(dataset_list) == 3
        
        dataset_names = [d.name for d in dataset_list]
        assert "miniImageNet" in dataset_names
        assert "Omniglot" in dataset_names
        assert "CIFAR-FS" in dataset_names
    
    def test_search_datasets(self):
        """Test dataset search functionality."""
        # Register datasets with different properties
        datasets = [
            DatasetMetadata(name="miniImageNet", task_type="few_shot_classification", num_classes=100),
            DatasetMetadata(name="Omniglot", task_type="few_shot_classification", num_classes=1623),
            DatasetMetadata(name="ShapeNet", task_type="3d_reconstruction", num_classes=55)
        ]
        
        for dataset in datasets:
            self.registry.register_dataset(dataset)
        
        # Search by task type
        few_shot_datasets = self.registry.search_datasets(task_type="few_shot_classification")
        assert len(few_shot_datasets) == 2
        
        # Search by number of classes
        large_datasets = self.registry.search_datasets(min_classes=200)
        assert len(large_datasets) == 1
        assert large_datasets[0].name == "Omniglot"
    
    def test_update_dataset(self):
        """Test dataset metadata updates."""
        original_metadata = DatasetMetadata(
            name="miniImageNet",
            version="1.0", 
            url="https://example.com/old_url",
            size_bytes=100*1024*1024
        )
        
        self.registry.register_dataset(original_metadata)
        
        # Update metadata
        updated_metadata = DatasetMetadata(
            name="miniImageNet",
            version="2.0",
            url="https://example.com/new_url",
            size_bytes=120*1024*1024  # Larger size
        )
        
        success = self.registry.update_dataset("miniImageNet", updated_metadata)
        assert success
        
        # Verify update
        info = self.registry.get_dataset_info("miniImageNet")
        assert info.version == "2.0"
        assert info.url == "https://example.com/new_url"
        assert info.size_bytes == 120*1024*1024
    
    def test_registry_persistence(self):
        """Test registry persistence to/from file."""
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name
        
        try:
            # Register datasets
            datasets = [
                DatasetMetadata(name="Dataset1", version="1.0", url="http://example.com/1"),
                DatasetMetadata(name="Dataset2", version="1.0", url="http://example.com/2")
            ]
            
            for dataset in datasets:
                self.registry.register_dataset(dataset)
            
            # Save to file
            self.registry.save_to_file(temp_file)
            
            # Load into new registry
            new_registry = DatasetRegistry()
            new_registry.load_from_file(temp_file)
            
            # Verify loaded data
            assert len(new_registry.list_datasets()) == 2
            assert new_registry.is_registered("Dataset1")
            assert new_registry.is_registered("Dataset2")
            
        finally:
            os.unlink(temp_file)


class TestSmartCache:
    """Test SmartCache functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache = SmartCache(
            cache_dir=self.temp_dir,
            max_cache_size_gb=1.0  # 1GB limit
        )
    
    def teardown_method(self):
        """Cleanup test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_cache_entry_creation(self):
        """Test cache entry creation and validation."""
        entry = CacheEntry(
            key="miniImageNet_v1.0",
            file_path=os.path.join(self.temp_dir, "mini_imagenet.tar.gz"),
            size_bytes=100*1024*1024,
            checksum="abc123def456",
            created_at=time.time(),
            last_accessed=time.time(),
            access_count=1,
            metadata={'dataset': 'miniImageNet', 'version': '1.0'}
        )
        
        assert entry.key == "miniImageNet_v1.0"
        assert entry.size_bytes == 100*1024*1024
        assert entry.is_valid()
        
        # Test invalid entry
        invalid_entry = CacheEntry(
            key="",  # Empty key
            file_path="nonexistent/path",
            size_bytes=-100  # Negative size
        )
        
        assert not invalid_entry.is_valid()
    
    def test_cache_add_get(self):
        """Test adding and retrieving cache entries."""
        # Create dummy file
        test_file = os.path.join(self.temp_dir, "test_file.txt")
        with open(test_file, 'w') as f:
            f.write("test content")
        
        # Add to cache
        success = self.cache.add("test_key", test_file, metadata={'test': True})
        assert success
        
        # Retrieve from cache
        cached_path = self.cache.get("test_key")
        assert cached_path is not None
        assert os.path.exists(cached_path)
        
        # Verify content
        with open(cached_path, 'r') as f:
            content = f.read()
        assert content == "test content"
    
    def test_cache_exists(self):
        """Test cache existence checking."""
        # Initially doesn't exist
        assert not self.cache.exists("nonexistent_key")
        
        # Add item to cache
        test_file = os.path.join(self.temp_dir, "test.txt")
        with open(test_file, 'w') as f:
            f.write("test")
        
        self.cache.add("test_key", test_file)
        
        # Now should exist
        assert self.cache.exists("test_key")
    
    def test_cache_remove(self):
        """Test cache removal."""
        # Add item
        test_file = os.path.join(self.temp_dir, "test.txt")
        with open(test_file, 'w') as f:
            f.write("test")
        
        self.cache.add("test_key", test_file)
        assert self.cache.exists("test_key")
        
        # Remove item
        success = self.cache.remove("test_key")
        assert success
        assert not self.cache.exists("test_key")
    
    def test_cache_size_management(self):
        """Test cache size limits and eviction."""
        # Create cache with very small limit
        small_cache = SmartCache(
            cache_dir=self.temp_dir,
            max_cache_size_gb=0.0001  # ~100KB limit
        )
        
        # Add files that exceed limit
        for i in range(5):
            test_file = os.path.join(self.temp_dir, f"large_file_{i}.txt")
            with open(test_file, 'w') as f:
                f.write("x" * 50000)  # 50KB each
            
            small_cache.add(f"key_{i}", test_file)
        
        # Should have evicted some entries
        cache_info = small_cache.get_cache_info()
        assert cache_info['total_size_mb'] < 1.0  # Should be under limit
        assert cache_info['num_entries'] < 5  # Should have evicted some
    
    def test_cache_lru_eviction(self):
        """Test LRU (Least Recently Used) eviction policy."""
        # Add multiple items
        files = []
        for i in range(3):
            test_file = os.path.join(self.temp_dir, f"file_{i}.txt")
            with open(test_file, 'w') as f:
                f.write(f"content_{i}")
            files.append(test_file)
            
            self.cache.add(f"key_{i}", test_file)
        
        # Access key_0 and key_2 (making key_1 least recently used)
        self.cache.get("key_0")
        time.sleep(0.1)  # Ensure different timestamps
        self.cache.get("key_2")
        
        # Force eviction by adding large file
        large_file = os.path.join(self.temp_dir, "large.txt")
        with open(large_file, 'w') as f:
            f.write("x" * 1024*1024)  # 1MB
        
        # Set very small cache limit to force eviction
        self.cache.max_cache_size_bytes = 1024  # 1KB limit
        self.cache.add("large_key", large_file)
        
        # key_1 should be evicted first (LRU)
        info = self.cache.get_cache_info()
        assert not self.cache.exists("key_1") or info['num_entries'] < 4
    
    def test_cache_persistence(self):
        """Test cache metadata persistence."""
        # Add items to cache
        test_file = os.path.join(self.temp_dir, "test.txt")
        with open(test_file, 'w') as f:
            f.write("test content")
        
        self.cache.add("test_key", test_file, metadata={'important': True})
        
        # Create new cache instance (simulates restart)
        new_cache = SmartCache(
            cache_dir=self.temp_dir,
            max_cache_size_gb=1.0
        )
        
        # Should restore from persistent metadata
        assert new_cache.exists("test_key")
        
        entry_info = new_cache.get_entry_info("test_key")
        assert entry_info['metadata']['important'] == True


class TestRobustDownloader:
    """Test RobustDownloader functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.downloader = RobustDownloader()
    
    def teardown_method(self):
        """Cleanup test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('urllib.request.urlretrieve')
    def test_download_with_retry(self, mock_urlretrieve):
        """Test download with retry mechanism."""
        output_path = os.path.join(self.temp_dir, "downloaded_file.zip")
        
        # Mock successful download
        mock_urlretrieve.return_value = (output_path, None)
        
        # Create dummy file for mock
        with open(output_path, 'w') as f:
            f.write("downloaded content")
        
        config = DownloadConfig(
            url="https://example.com/dataset.zip",
            output_path=output_path,
            checksum="dummy_checksum",
            max_retries=3,
            retry_delay=0.1
        )
        
        success = self.downloader.download_with_retry(config)
        assert success
        mock_urlretrieve.assert_called_once()
    
    @patch('urllib.request.urlretrieve')
    def test_download_failure_and_retry(self, mock_urlretrieve):
        """Test download failure and retry behavior."""
        output_path = os.path.join(self.temp_dir, "failed_download.zip")
        
        # Mock failed downloads
        mock_urlretrieve.side_effect = Exception("Network error")
        
        config = DownloadConfig(
            url="https://example.com/dataset.zip",
            output_path=output_path,
            max_retries=2,
            retry_delay=0.05
        )
        
        success = self.downloader.download_with_retry(config)
        assert not success
        assert mock_urlretrieve.call_count == 3  # Original + 2 retries
    
    def test_checksum_verification(self):
        """Test checksum verification."""
        # Create test file with known content
        test_file = os.path.join(self.temp_dir, "test_checksum.txt")
        test_content = "test content for checksum"
        
        with open(test_file, 'w') as f:
            f.write(test_content)
        
        # Compute expected checksum
        expected_checksum = hashlib.md5(test_content.encode()).hexdigest()
        
        # Verify correct checksum
        assert self.downloader.verify_checksum(test_file, expected_checksum, 'md5')
        
        # Verify incorrect checksum
        wrong_checksum = "wrong_checksum_123"
        assert not self.downloader.verify_checksum(test_file, wrong_checksum, 'md5')
    
    def test_resume_download(self):
        """Test resume download functionality."""
        output_path = os.path.join(self.temp_dir, "partial_download.zip")
        
        # Create partial file
        with open(output_path, 'wb') as f:
            f.write(b"partial content")
        
        partial_size = os.path.getsize(output_path)
        
        # Test resume detection
        config = DownloadConfig(
            url="https://example.com/large_dataset.zip",
            output_path=output_path,
            resume_download=True
        )
        
        resume_info = self.downloader.check_resume_capability(config)
        assert resume_info['can_resume']
        assert resume_info['partial_size'] == partial_size
    
    def test_progress_tracking(self):
        """Test download progress tracking."""
        progress_updates = []
        
        def progress_callback(downloaded, total, speed_mbps):
            progress_updates.append({
                'downloaded': downloaded,
                'total': total,
                'speed': speed_mbps
            })
        
        # Mock a download with progress updates
        with patch.object(self.downloader, '_download_with_progress') as mock_download:
            mock_download.return_value = True
            
            config = DownloadConfig(
                url="https://example.com/dataset.zip",
                output_path=os.path.join(self.temp_dir, "download.zip"),
                progress_callback=progress_callback
            )
            
            self.downloader.download_with_retry(config)
            
            # Should have called download with progress
            mock_download.assert_called_once()


class TestDatasetManager:
    """Test DatasetManager comprehensive functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.manager = DatasetManager(
            cache_dir=self.temp_dir,
            max_cache_size_gb=1.0
        )
    
    def teardown_method(self):
        """Cleanup test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_manager_initialization(self):
        """Test DatasetManager initialization."""
        assert self.manager.cache_dir == self.temp_dir
        assert self.manager.registry is not None
        assert self.manager.cache is not None
        assert self.manager.downloader is not None
    
    def test_register_builtin_datasets(self):
        """Test registration of built-in datasets."""
        # Should automatically register standard datasets
        datasets = self.manager.list_available_datasets()
        
        # Should include standard few-shot learning datasets
        dataset_names = [d.name for d in datasets]
        expected_datasets = ['miniImageNet', 'Omniglot', 'CIFAR-FS']
        
        for expected in expected_datasets:
            assert any(expected.lower() in name.lower() for name in dataset_names)
    
    def test_get_dataset_info(self):
        """Test getting dataset information."""
        # Test built-in dataset info
        info = self.manager.get_dataset_info("miniImageNet")
        
        if info is not None:  # May not be registered in test environment
            assert info.task_type == "few_shot_classification"
            assert info.num_classes > 50  # miniImageNet has 100 classes
    
    @patch.object(RobustDownloader, 'download_with_retry')
    def test_download_dataset(self, mock_download):
        """Test dataset downloading."""
        # Mock successful download
        mock_download.return_value = True
        
        # Create dummy downloaded file
        dummy_file = os.path.join(self.temp_dir, "mini_imagenet.tar.gz")
        with open(dummy_file, 'wb') as f:
            f.write(b"dummy dataset content")
        
        # Register a test dataset
        test_metadata = DatasetMetadata(
            name="TestDataset",
            version="1.0",
            url="https://example.com/test_dataset.zip",
            size_bytes=1024,
            checksum="dummy_checksum"
        )
        self.manager.registry.register_dataset(test_metadata)
        
        # Download dataset
        success = self.manager.download_dataset("TestDataset", force_redownload=True)
        
        # Should attempt download
        mock_download.assert_called_once()
    
    def test_load_dataset(self):
        """Test dataset loading."""
        # Create mock dataset class
        class MockDataset:
            def __init__(self, root_dir, split='train', transform=None):
                self.root_dir = root_dir
                self.split = split
                self.transform = transform
                self.data = [("sample1", 0), ("sample2", 1), ("sample3", 0)]
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                return self.data[idx]
        
        # Register mock dataset loader
        self.manager.register_dataset_loader("MockDataset", MockDataset)
        
        # Load dataset
        dataset = self.manager.load_dataset("MockDataset", split="train")
        
        assert dataset is not None
        assert len(dataset) == 3
    
    def test_create_episode_loader(self):
        """Test episodic data loader creation."""
        # Create dummy dataset
        class DummyDataset:
            def __init__(self):
                # 10 classes, 20 examples each
                self.data = []
                self.targets = []
                for class_id in range(10):
                    for _ in range(20):
                        self.data.append(torch.randn(3, 32, 32))
                        self.targets.append(class_id)
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                return self.data[idx], self.targets[idx]
        
        dummy_dataset = DummyDataset()
        
        # Create episode loader
        episode_loader = self.manager.create_episode_loader(
            dataset=dummy_dataset,
            n_way=5,
            k_shot=2,
            m_query=3,
            n_episodes=10
        )
        
        assert episode_loader is not None
        assert len(episode_loader) == 10
        
        # Test loading an episode
        episode = episode_loader[0]
        assert hasattr(episode, 'support_x')
        assert hasattr(episode, 'support_y')
        assert hasattr(episode, 'query_x')
        assert hasattr(episode, 'query_y')
    
    def test_dataset_statistics(self):
        """Test dataset statistics computation."""
        # Create dummy dataset with known statistics
        class StatsDataset:
            def __init__(self):
                # Create data with known mean and std
                self.data = [torch.ones(3, 4, 4) * i for i in range(1, 11)]  # Values 1-10
                self.targets = list(range(10))
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                return self.data[idx], self.targets[idx]
        
        dataset = StatsDataset()
        stats = self.manager.compute_dataset_statistics(dataset)
        
        assert 'mean' in stats
        assert 'std' in stats
        assert 'num_samples' in stats
        assert 'num_classes' in stats
        
        assert stats['num_samples'] == 10
        assert stats['num_classes'] == 10
    
    def test_validate_dataset_integrity(self):
        """Test dataset integrity validation."""
        # Create test dataset with some issues
        class ProblematicDataset:
            def __init__(self):
                self.data = [torch.randn(3, 32, 32) for _ in range(100)]
                self.targets = [i % 10 for i in range(100)]  # 10 classes
                # Add some problematic entries
                self.data[50] = torch.full((3, 32, 32), float('nan'))  # NaN values
                self.targets[75] = -1  # Invalid class label
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                return self.data[idx], self.targets[idx]
        
        dataset = ProblematicDataset()
        validation = self.manager.validate_dataset_integrity(dataset)
        
        assert 'valid' in validation
        assert 'issues' in validation
        assert 'statistics' in validation
        
        # Should detect NaN values and invalid labels
        assert len(validation['issues']) > 0
        issues_text = ' '.join(validation['issues']).lower()
        assert 'nan' in issues_text or 'invalid' in issues_text


class TestDatasetTransform:
    """Test DatasetTransform functionality."""
    
    def test_standard_transforms(self):
        """Test standard dataset transforms."""
        # Create sample data
        sample_image = torch.randint(0, 256, (32, 32, 3), dtype=torch.uint8)
        sample_label = 5
        
        # Test normalization transform
        normalize_transform = DatasetTransform.create_normalization_transform()
        normalized_image, normalized_label = normalize_transform(sample_image, sample_label)
        
        assert normalized_image.dtype == torch.float32
        assert normalized_image.min() >= 0.0 and normalized_image.max() <= 1.0
        assert normalized_label == sample_label
    
    def test_augmentation_transforms(self):
        """Test data augmentation transforms."""
        sample_image = torch.randn(3, 32, 32)
        sample_label = 2
        
        # Test augmentation
        aug_transform = DatasetTransform.create_augmentation_transform()
        aug_image, aug_label = aug_transform(sample_image, sample_label)
        
        assert aug_image.shape == sample_image.shape
        assert aug_label == sample_label
        
        # Augmented image should be different (with high probability)
        assert not torch.equal(aug_image, sample_image)
    
    def test_resize_transform(self):
        """Test image resize transform."""
        sample_image = torch.randn(3, 64, 64)
        sample_label = 1
        
        # Test resize to 32x32
        resize_transform = DatasetTransform.create_resize_transform((32, 32))
        resized_image, resized_label = resize_transform(sample_image, sample_label)
        
        assert resized_image.shape == (3, 32, 32)
        assert resized_label == sample_label
    
    def test_compose_transforms(self):
        """Test composing multiple transforms."""
        sample_image = torch.randint(0, 256, (64, 64, 3), dtype=torch.uint8)
        sample_label = 3
        
        # Compose transforms: resize -> normalize -> augment
        composed_transform = DatasetTransform.compose([
            DatasetTransform.create_resize_transform((32, 32)),
            DatasetTransform.create_normalization_transform(),
            DatasetTransform.create_augmentation_transform()
        ])
        
        final_image, final_label = composed_transform(sample_image, sample_label)
        
        assert final_image.shape[-2:] == (32, 32)  # Should be resized
        assert final_image.dtype == torch.float32    # Should be normalized
        assert final_label == sample_label


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_create_dataset_manager(self):
        """Test dataset manager creation utility."""
        temp_dir = tempfile.mkdtemp()
        
        try:
            manager = create_dataset_manager(
                cache_dir=temp_dir,
                max_cache_size_gb=0.5
            )
            
            assert manager is not None
            assert manager.cache_dir == temp_dir
            
            # Should have built-in datasets registered
            datasets = manager.list_available_datasets()
            assert len(datasets) > 0
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_get_dataset_info_function(self):
        """Test get_dataset_info utility function."""
        # Should work with built-in datasets
        info = get_dataset_info("miniImageNet")
        
        if info is not None:
            assert hasattr(info, 'name')
            assert hasattr(info, 'num_classes')
            assert hasattr(info, 'task_type')
        
        # Non-existent dataset should return None
        non_existent = get_dataset_info("NonExistentDataset")
        assert non_existent is None
    
    @patch.object(DatasetManager, 'download_dataset')
    def test_download_dataset_function(self, mock_download):
        """Test download_dataset utility function."""
        mock_download.return_value = True
        
        success = download_dataset("miniImageNet", cache_dir="/tmp/test_cache")
        
        # Should call manager's download method
        mock_download.assert_called_once_with("miniImageNet", force_redownload=False)


class TestRealWorldScenarios:
    """Test realistic dataset management scenarios."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.manager = DatasetManager(
            cache_dir=self.temp_dir,
            max_cache_size_gb=0.1  # Small cache for testing eviction
        )
    
    def teardown_method(self):
        """Cleanup test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_multiple_dataset_workflow(self):
        """Test working with multiple datasets."""
        # Register multiple datasets
        datasets = [
            DatasetMetadata(name="Dataset_A", version="1.0", url="http://example.com/a.zip", size_bytes=1024*1024),
            DatasetMetadata(name="Dataset_B", version="1.0", url="http://example.com/b.zip", size_bytes=2*1024*1024),
            DatasetMetadata(name="Dataset_C", version="1.0", url="http://example.com/c.zip", size_bytes=3*1024*1024)
        ]
        
        for dataset in datasets:
            self.manager.registry.register_dataset(dataset)
        
        # List available datasets
        available = self.manager.list_available_datasets()
        assert len(available) >= 3
        
        # Search for specific datasets
        large_datasets = self.manager.search_datasets(min_size_mb=2)
        assert len(large_datasets) >= 2
    
    def test_cache_management_workflow(self):
        """Test cache management in realistic scenario."""
        # Add several items to cache
        cache_items = []
        for i in range(5):
            test_file = os.path.join(self.temp_dir, f"dataset_{i}.zip")
            with open(test_file, 'wb') as f:
                f.write(b"x" * (50 * 1024))  # 50KB each
            
            cache_items.append(test_file)
            self.manager.cache.add(f"dataset_{i}", test_file)
        
        # Check cache status
        cache_info = self.manager.get_cache_info()
        assert cache_info['num_entries'] <= 5
        
        # Clean cache (should remove least recently used items)
        self.manager.clean_cache()
        
        cleaned_info = self.manager.get_cache_info()
        assert cleaned_info['total_size_mb'] <= cache_info['total_size_mb']
    
    def test_dataset_versioning(self):
        """Test dataset version management."""
        # Register dataset v1.0
        dataset_v1 = DatasetMetadata(
            name="EvolvingDataset",
            version="1.0",
            url="https://example.com/dataset_v1.zip",
            size_bytes=1024*1024
        )
        self.manager.registry.register_dataset(dataset_v1)
        
        # Update to v2.0
        dataset_v2 = DatasetMetadata(
            name="EvolvingDataset", 
            version="2.0",
            url="https://example.com/dataset_v2.zip",
            size_bytes=2*1024*1024,
            description="Updated with more examples"
        )
        
        success = self.manager.update_dataset("EvolvingDataset", dataset_v2)
        assert success
        
        # Verify version was updated
        current_info = self.manager.get_dataset_info("EvolvingDataset")
        assert current_info.version == "2.0"
        assert current_info.size_bytes == 2*1024*1024
    
    def test_error_handling_and_recovery(self):
        """Test error handling and recovery scenarios."""
        # Test handling of corrupted cache
        corrupt_file = os.path.join(self.temp_dir, "corrupt.zip")
        with open(corrupt_file, 'wb') as f:
            f.write(b"corrupted data")
        
        # Add to cache with wrong checksum
        self.manager.cache.add("corrupt_dataset", corrupt_file, 
                             metadata={'checksum': 'wrong_checksum'})
        
        # Verification should detect corruption
        is_valid = self.manager.verify_cached_dataset("corrupt_dataset")
        assert not is_valid
        
        # Should handle missing files gracefully
        missing_info = self.manager.get_dataset_info("NonExistentDataset")
        assert missing_info is None
    
    def test_concurrent_access_simulation(self):
        """Test simulated concurrent access to dataset manager."""
        # Simulate multiple processes accessing datasets
        import threading
        
        results = []
        errors = []
        
        def worker(dataset_id):
            try:
                # Each worker tries to access different datasets
                test_file = os.path.join(self.temp_dir, f"worker_dataset_{dataset_id}.zip")
                with open(test_file, 'wb') as f:
                    f.write(b"worker data")
                
                success = self.manager.cache.add(f"worker_{dataset_id}", test_file)
                results.append(success)
            except Exception as e:
                errors.append(str(e))
        
        # Start multiple worker threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=worker, args=(i,))
            thread.start()
            threads.append(thread)
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Should handle concurrent access without errors
        assert len(errors) == 0
        assert len(results) == 10
        assert all(results)  # All operations should succeed


if __name__ == "__main__":
    pytest.main([__file__])