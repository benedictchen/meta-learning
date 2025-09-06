"""
Test Suite for Dataset Management System
=======================================

Comprehensive tests for the dataset management functionality including:
- Dataset registry operations
- Smart caching system
- Robust downloading capabilities
- Dataset loading and validation
- Error handling and edge cases

Author: Test Suite Generator
"""

import pytest
import torch
import tempfile
import shutil
import json
import hashlib
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from meta_learning.meta_learning_modules.dataset_management import (
    DatasetInfo,
    DatasetRegistry,
    SmartCache,
    RobustDownloader,
    DatasetManager,
    DownloadProgress,
    get_dataset_manager,
    simple_progress_callback
)


class TestDatasetInfo:
    """Test DatasetInfo dataclass functionality."""
    
    def test_dataset_info_creation(self):
        """Test creating DatasetInfo with required fields."""
        info = DatasetInfo(
            name="test_dataset",
            description="Test dataset for unit tests",
            urls=["http://example.com/data.zip"],
            checksums={"md5": "abc123"},
            file_size=1000000,
            n_classes=10,
            n_samples=5000,
            image_size=(32, 32)
        )
        
        assert info.name == "test_dataset"
        assert info.description == "Test dataset for unit tests"
        assert info.urls == ["http://example.com/data.zip"]
        assert info.checksums == {"md5": "abc123"}
        assert info.file_size == 1000000
        assert info.n_classes == 10
        assert info.n_samples == 5000
        assert info.image_size == (32, 32)
        assert info.dependencies == []  # Default empty list
        assert info.transforms is None  # Default None
        assert info.metadata == {}  # Default empty dict
    
    def test_dataset_info_with_dependencies(self):
        """Test DatasetInfo with dependencies."""
        info = DatasetInfo(
            name="dependent_dataset",
            description="Dataset with dependencies",
            urls=["http://example.com/dep.zip"],
            checksums={"sha256": "xyz789"},
            file_size=2000000,
            n_classes=20,
            n_samples=10000,
            image_size=(64, 64),
            dependencies=["base_dataset", "utils_dataset"]
        )
        
        assert info.dependencies == ["base_dataset", "utils_dataset"]
    
    def test_dataset_info_with_metadata(self):
        """Test DatasetInfo with custom metadata."""
        metadata = {
            "license": "MIT",
            "year": 2024,
            "authors": ["Test Author"]
        }
        
        info = DatasetInfo(
            name="meta_dataset",
            description="Dataset with metadata",
            urls=["http://example.com/meta.tar.gz"],
            checksums={"md5": "meta123", "sha1": "meta456"},
            file_size=500000,
            n_classes=5,
            n_samples=2500,
            image_size=None,
            metadata=metadata
        )
        
        assert info.metadata == metadata
        assert info.image_size is None


class TestDatasetRegistry:
    """Test DatasetRegistry functionality."""
    
    def setup_method(self):
        """Set up test registry."""
        self.registry = DatasetRegistry()
        # Clear built-in datasets for clean testing
        self.registry.datasets.clear()
    
    def test_register_dataset(self):
        """Test registering a new dataset."""
        info = DatasetInfo(
            name="test_dataset",
            description="Test dataset",
            urls=["http://example.com/test.zip"],
            checksums={"md5": "test123"},
            file_size=1000,
            n_classes=5,
            n_samples=100,
            image_size=(28, 28)
        )
        
        self.registry.register_dataset(info)
        
        assert "test_dataset" in self.registry.datasets
        retrieved = self.registry.get_dataset_info("test_dataset")
        assert retrieved == info
    
    def test_register_duplicate_dataset(self):
        """Test registering duplicate dataset (should overwrite)."""
        info1 = DatasetInfo(
            name="duplicate",
            description="First version",
            urls=["http://example.com/v1.zip"],
            checksums={"md5": "v1_hash"},
            file_size=1000,
            n_classes=5,
            n_samples=100,
            image_size=(28, 28)
        )
        
        info2 = DatasetInfo(
            name="duplicate",
            description="Second version",
            urls=["http://example.com/v2.zip"],
            checksums={"md5": "v2_hash"},
            file_size=2000,
            n_classes=10,
            n_samples=200,
            image_size=(32, 32)
        )
        
        self.registry.register_dataset(info1)
        self.registry.register_dataset(info2)
        
        retrieved = self.registry.get_dataset_info("duplicate")
        assert retrieved == info2  # Should be the second version
        assert retrieved.description == "Second version"
    
    def test_get_nonexistent_dataset(self):
        """Test getting info for non-existent dataset."""
        result = self.registry.get_dataset_info("nonexistent")
        assert result is None
    
    def test_list_datasets(self):
        """Test listing all registered datasets."""
        datasets = ["dataset_a", "dataset_b", "dataset_c"]
        
        for name in datasets:
            info = DatasetInfo(
                name=name,
                description=f"Description for {name}",
                urls=[f"http://example.com/{name}.zip"],
                checksums={"md5": f"{name}_hash"},
                file_size=1000,
                n_classes=5,
                n_samples=100,
                image_size=(28, 28)
            )
            self.registry.register_dataset(info)
        
        listed = self.registry.list_datasets()
        assert set(listed) == set(datasets)
    
    def test_dependency_resolution_simple(self):
        """Test simple dependency resolution."""
        # Create datasets with linear dependency chain
        base_info = DatasetInfo(
            name="base",
            description="Base dataset",
            urls=["http://example.com/base.zip"],
            checksums={"md5": "base_hash"},
            file_size=1000,
            n_classes=5,
            n_samples=100,
            image_size=(28, 28)
        )
        
        dep_info = DatasetInfo(
            name="dependent",
            description="Dependent dataset",
            urls=["http://example.com/dep.zip"],
            checksums={"md5": "dep_hash"},
            file_size=2000,
            n_classes=10,
            n_samples=200,
            image_size=(32, 32),
            dependencies=["base"]
        )
        
        self.registry.register_dataset(base_info)
        self.registry.register_dataset(dep_info)
        
        resolved = self.registry.resolve_dependencies("dependent")
        assert resolved == ["base", "dependent"]
    
    def test_dependency_resolution_complex(self):
        """Test complex dependency resolution with multiple levels."""
        # Create complex dependency graph: final -> [mid1, mid2] -> base
        base_info = DatasetInfo(
            name="base",
            description="Base dataset",
            urls=["http://example.com/base.zip"],
            checksums={"md5": "base_hash"},
            file_size=1000,
            n_classes=5,
            n_samples=100,
            image_size=(28, 28)
        )
        
        mid1_info = DatasetInfo(
            name="mid1",
            description="Middle dataset 1",
            urls=["http://example.com/mid1.zip"],
            checksums={"md5": "mid1_hash"},
            file_size=1500,
            n_classes=7,
            n_samples=150,
            image_size=(28, 28),
            dependencies=["base"]
        )
        
        mid2_info = DatasetInfo(
            name="mid2",
            description="Middle dataset 2",
            urls=["http://example.com/mid2.zip"],
            checksums={"md5": "mid2_hash"},
            file_size=1800,
            n_classes=8,
            n_samples=180,
            image_size=(28, 28),
            dependencies=["base"]
        )
        
        final_info = DatasetInfo(
            name="final",
            description="Final dataset",
            urls=["http://example.com/final.zip"],
            checksums={"md5": "final_hash"},
            file_size=3000,
            n_classes=15,
            n_samples=300,
            image_size=(32, 32),
            dependencies=["mid1", "mid2"]
        )
        
        for info in [base_info, mid1_info, mid2_info, final_info]:
            self.registry.register_dataset(info)
        
        resolved = self.registry.resolve_dependencies("final")
        
        # Base should come first, then mid1 and mid2, then final
        assert "base" in resolved
        assert "mid1" in resolved
        assert "mid2" in resolved
        assert "final" in resolved
        assert resolved.index("base") < resolved.index("mid1")
        assert resolved.index("base") < resolved.index("mid2")
        assert resolved.index("final") == len(resolved) - 1
    
    def test_dependency_resolution_unknown_dependency(self):
        """Test dependency resolution with unknown dependency."""
        info = DatasetInfo(
            name="broken",
            description="Dataset with unknown dependency",
            urls=["http://example.com/broken.zip"],
            checksums={"md5": "broken_hash"},
            file_size=1000,
            n_classes=5,
            n_samples=100,
            image_size=(28, 28),
            dependencies=["unknown_dataset"]
        )
        
        self.registry.register_dataset(info)
        
        with pytest.raises(ValueError, match="Unknown dataset: unknown_dataset"):
            self.registry.resolve_dependencies("broken")
    
    def test_builtin_datasets_registration(self):
        """Test that built-in datasets are registered correctly."""
        # Create fresh registry to test built-in registration
        fresh_registry = DatasetRegistry()
        
        builtin_datasets = ["miniimagenet", "cifar_fs", "omniglot", "tieredimagenet"]
        
        for dataset in builtin_datasets:
            info = fresh_registry.get_dataset_info(dataset)
            assert info is not None
            assert info.name == dataset
            assert len(info.urls) > 0
            assert len(info.checksums) > 0
            assert info.file_size > 0
            assert info.n_classes > 0


class TestSmartCache:
    """Test SmartCache functionality."""
    
    def setup_method(self):
        """Set up temporary cache directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache = SmartCache(
            cache_dir=self.temp_dir,
            max_size_gb=0.001  # 1MB for testing
        )
    
    def teardown_method(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_cache_initialization(self):
        """Test cache initialization."""
        assert self.cache.cache_dir.exists()
        assert self.cache.max_size_bytes == int(0.001 * 1024**3)
        assert self.cache.eviction_policy == "lru_frequency"
        assert isinstance(self.cache.access_counts, dict)
        assert isinstance(self.cache.last_access_times, dict)
    
    def test_get_cache_path(self):
        """Test getting cache path for dataset file."""
        path = self.cache.get_cache_path("test_dataset", "data.zip")
        
        expected = Path(self.temp_dir) / "test_dataset" / "data.zip"
        assert path == expected
        assert path.parent.exists()  # Dataset directory should be created
    
    def test_cache_and_retrieve_file(self):
        """Test caching and retrieving file."""
        test_data = b"Hello, World! This is test data."
        dataset_name = "test_dataset"
        filename = "test_file.txt"
        
        # Cache the file
        self.cache.cache_file(dataset_name, filename, test_data)
        
        # Check if cached
        assert self.cache.is_cached(dataset_name, filename)
        
        # Retrieve the file
        retrieved_data = self.cache.get_cached_file(dataset_name, filename)
        assert retrieved_data == test_data
    
    def test_cache_with_checksum_validation(self):
        """Test caching with checksum validation."""
        test_data = b"Test data for checksum validation"
        checksum = hashlib.md5(test_data).hexdigest()
        
        dataset_name = "checksum_test"
        filename = "data.bin"
        
        # Cache the file
        self.cache.cache_file(dataset_name, filename, test_data)
        
        # Check with correct checksum
        assert self.cache.is_cached(dataset_name, filename, checksum)
        
        # Check with incorrect checksum
        assert not self.cache.is_cached(dataset_name, filename, "wrong_checksum")
    
    def test_cache_eviction_lru_frequency(self):
        """Test cache eviction with LRU + frequency policy."""
        # Create files that exceed cache limit
        large_data = b"x" * 1024  # 1KB files
        
        # Cache multiple files
        files = [
            ("dataset1", "file1.dat"),
            ("dataset1", "file2.dat"),
            ("dataset2", "file3.dat"),
            ("dataset2", "file4.dat")
        ]
        
        for dataset, filename in files:
            self.cache.cache_file(dataset, filename, large_data)
        
        # Access some files more frequently
        self.cache.is_cached("dataset1", "file1.dat")  # Access twice
        self.cache.is_cached("dataset1", "file1.dat")
        self.cache.is_cached("dataset2", "file3.dat")  # Access once
        
        # Add another large file to trigger eviction
        self.cache.cache_file("dataset3", "large_file.dat", large_data)
        
        # Check that some files were evicted
        total_size = self.cache.get_total_size()
        assert total_size <= self.cache.max_size_bytes
    
    def test_cache_statistics(self):
        """Test cache statistics reporting."""
        # Add some test files
        test_data = b"Test data" * 100  # ~900 bytes
        
        self.cache.cache_file("dataset1", "file1.dat", test_data)
        self.cache.cache_file("dataset1", "file2.dat", test_data)
        
        stats = self.cache.get_cache_stats()
        
        assert "total_size_gb" in stats
        assert "max_size_gb" in stats
        assert "utilization_percent" in stats
        assert "num_cached_files" in stats
        assert "eviction_policy" in stats
        assert "cache_dir" in stats
        
        assert stats["num_cached_files"] >= 2
        assert stats["eviction_policy"] == "lru_frequency"
        assert stats["max_size_gb"] == 0.001
    
    def test_metadata_persistence(self):
        """Test that cache metadata persists across instances."""
        test_data = b"Persistent test data"
        
        # Cache file with first instance
        self.cache.cache_file("persistent", "data.txt", test_data)
        self.cache.is_cached("persistent", "data.txt")  # Update access
        
        # Create new cache instance with same directory
        new_cache = SmartCache(
            cache_dir=self.temp_dir,
            max_size_gb=0.001
        )
        
        # Check if metadata was loaded
        key = "persistent/data.txt"
        assert key in new_cache.access_counts
        assert key in new_cache.last_access_times


class TestRobustDownloader:
    """Test RobustDownloader functionality."""
    
    def setup_method(self):
        """Set up downloader and temp directory."""
        self.downloader = RobustDownloader(max_workers=2, timeout=5)
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def teardown_method(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('requests.Session.get')
    def test_successful_download(self, mock_get):
        """Test successful file download."""
        # Mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {'content-length': '13'}
        mock_response.iter_content = Mock(return_value=[b'Hello, World!'])
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        output_path = self.temp_dir / "downloaded_file.txt"
        urls = ["http://example.com/file.txt"]
        
        success = self.downloader.download_file(urls, output_path)
        
        assert success
        assert output_path.exists()
        assert output_path.read_bytes() == b'Hello, World!'
    
    @patch('requests.Session.get')
    def test_download_with_resume(self, mock_get):
        """Test download with resume support."""
        # Create partial file
        partial_data = b"Hello"
        output_path = self.temp_dir / "partial_file.txt"
        output_path.write_bytes(partial_data)
        
        # Mock response for resume
        mock_response = Mock()
        mock_response.status_code = 206  # Partial Content
        mock_response.headers = {'content-length': '8'}
        mock_response.iter_content = Mock(return_value=[b', World!'])
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        urls = ["http://example.com/file.txt"]
        
        success = self.downloader.download_file(urls, output_path)
        
        assert success
        assert output_path.read_bytes() == b'Hello, World!'
        
        # Check that resume headers were sent
        mock_get.assert_called_with(
            urls[0],
            headers={'Range': 'bytes=5-'},
            stream=True,
            timeout=5
        )
    
    @patch('requests.Session.get')
    def test_download_multiple_sources(self, mock_get):
        """Test download from multiple sources (fallback)."""
        # First URL fails, second succeeds
        def side_effect(url, **kwargs):
            if "fail.com" in url:
                raise Exception("Connection failed")
            else:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.headers = {'content-length': '7'}
                mock_response.iter_content = Mock(return_value=[b'Success'])
                mock_response.raise_for_status = Mock()
                return mock_response
        
        mock_get.side_effect = side_effect
        
        output_path = self.temp_dir / "multi_source.txt"
        urls = ["http://fail.com/file.txt", "http://success.com/file.txt"]
        
        success = self.downloader.download_file(urls, output_path)
        
        assert success
        assert output_path.read_bytes() == b'Success'
        assert mock_get.call_count == 2  # Both URLs tried
    
    @patch('requests.Session.get')
    def test_download_with_checksum_validation(self, mock_get):
        """Test download with checksum validation."""
        test_data = b'Test data for checksum'
        expected_md5 = hashlib.md5(test_data).hexdigest()
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {'content-length': str(len(test_data))}
        mock_response.iter_content = Mock(return_value=[test_data])
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        output_path = self.temp_dir / "checksum_test.txt"
        urls = ["http://example.com/file.txt"]
        checksums = {"md5": expected_md5}
        
        success = self.downloader.download_file(urls, output_path, checksums=checksums)
        
        assert success
        assert output_path.read_bytes() == test_data
    
    @patch('requests.Session.get')
    def test_download_checksum_mismatch(self, mock_get):
        """Test download with checksum mismatch."""
        test_data = b'Test data'
        wrong_checksum = "wrong_checksum_value"
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {'content-length': str(len(test_data))}
        mock_response.iter_content = Mock(return_value=[test_data])
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        output_path = self.temp_dir / "checksum_fail.txt"
        urls = ["http://example.com/file.txt"]
        checksums = {"md5": wrong_checksum}
        
        success = self.downloader.download_file(urls, output_path, checksums=checksums)
        
        assert not success
        assert not output_path.exists()  # File should be removed after checksum failure
    
    def test_progress_callback(self):
        """Test progress callback functionality."""
        progress_updates = []
        
        def progress_callback(progress):
            progress_updates.append(progress)
        
        # This test would require mocking the entire download process
        # For now, just test that the callback structure works
        progress = DownloadProgress(
            total_bytes=1000,
            downloaded_bytes=500,
            speed_mbps=1.5,
            status="downloading"
        )
        
        progress_callback(progress)
        
        assert len(progress_updates) == 1
        assert progress_updates[0].total_bytes == 1000
        assert progress_updates[0].downloaded_bytes == 500


class TestDatasetManager:
    """Test DatasetManager integration."""
    
    def setup_method(self):
        """Set up manager with temporary directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.manager = DatasetManager(
            cache_dir=self.temp_dir,
            max_cache_size_gb=0.001
        )
    
    def teardown_method(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_manager_initialization(self):
        """Test DatasetManager initialization."""
        assert isinstance(self.manager.registry, DatasetRegistry)
        assert isinstance(self.manager.cache, SmartCache)
        assert isinstance(self.manager.downloader, RobustDownloader)
        assert isinstance(self.manager.progress_callbacks, list)
    
    def test_list_available_datasets(self):
        """Test listing available datasets."""
        datasets = self.manager.list_available_datasets()
        
        # Should include built-in datasets
        expected_datasets = ["miniimagenet", "cifar_fs", "omniglot", "tieredimagenet"]
        for dataset in expected_datasets:
            assert dataset in datasets
    
    def test_get_dataset_info(self):
        """Test getting dataset information."""
        info = self.manager.get_dataset_info("miniimagenet")
        
        assert info is not None
        assert info.name == "miniimagenet"
        assert "84x84 images from ImageNet" in info.description
        assert info.n_classes == 100
        assert info.image_size == (84, 84)
    
    def test_get_nonexistent_dataset_info(self):
        """Test getting info for non-existent dataset."""
        info = self.manager.get_dataset_info("nonexistent_dataset")
        assert info is None
    
    def test_progress_callback_management(self):
        """Test progress callback management."""
        callback1 = Mock()
        callback2 = Mock()
        
        self.manager.add_progress_callback(callback1)
        self.manager.add_progress_callback(callback2)
        
        assert len(self.manager.progress_callbacks) == 2
        assert callback1 in self.manager.progress_callbacks
        assert callback2 in self.manager.progress_callbacks
    
    def test_cache_statistics(self):
        """Test getting cache statistics."""
        stats = self.manager.get_cache_stats()
        
        assert "total_size_gb" in stats
        assert "max_size_gb" in stats
        assert "utilization_percent" in stats
        assert "num_cached_files" in stats
        assert "eviction_policy" in stats
        assert "cache_dir" in stats
    
    def test_clear_cache_all(self):
        """Test clearing all cache."""
        # Create some dummy cached files
        cache_dir = Path(self.temp_dir)
        test_dir = cache_dir / "test_dataset"
        test_dir.mkdir(parents=True, exist_ok=True)
        test_file = test_dir / "test_file.dat"
        test_file.write_bytes(b"test data")
        
        assert test_file.exists()
        
        # Clear all cache
        self.manager.clear_cache()
        
        # Check that files are gone but cache dir exists
        assert cache_dir.exists()
        assert not test_file.exists()
    
    def test_clear_cache_specific_dataset(self):
        """Test clearing cache for specific dataset."""
        # Create cached files for multiple datasets
        cache_dir = Path(self.temp_dir)
        
        for dataset in ["dataset1", "dataset2"]:
            test_dir = cache_dir / dataset
            test_dir.mkdir(parents=True, exist_ok=True)
            test_file = test_dir / "data.dat"
            test_file.write_bytes(b"test data")
        
        # Clear cache for one dataset only
        self.manager.clear_cache("dataset1")
        
        # Check that only dataset1 was cleared
        assert not (cache_dir / "dataset1").exists()
        assert (cache_dir / "dataset2" / "data.dat").exists()
    
    @patch.object(DatasetManager, '_download_dataset')
    @patch.object(DatasetManager, '_load_cached_dataset')
    def test_get_dataset_cached(self, mock_load, mock_download):
        """Test getting dataset that's already cached."""
        # Mock cached dataset
        mock_dataset = Mock()
        mock_load.return_value = mock_dataset
        
        # Mock cache check to return True
        with patch.object(self.manager.cache, 'is_cached', return_value=True):
            result = self.manager.get_dataset("miniimagenet")
        
        assert result == mock_dataset
        mock_load.assert_called_once()
        mock_download.assert_not_called()
    
    @patch.object(DatasetManager, '_download_dataset')
    def test_get_dataset_not_cached_no_download(self, mock_download):
        """Test getting dataset that's not cached with download=False."""
        # Mock cache check to return False
        with patch.object(self.manager.cache, 'is_cached', return_value=False):
            result = self.manager.get_dataset("miniimagenet", download=False)
        
        assert result is None
        mock_download.assert_not_called()
    
    def test_get_unknown_dataset(self):
        """Test getting unknown dataset raises error."""
        with pytest.raises(ValueError, match="Unknown dataset: unknown_dataset"):
            self.manager.get_dataset("unknown_dataset")


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_get_dataset_manager_singleton(self):
        """Test global dataset manager singleton."""
        manager1 = get_dataset_manager()
        manager2 = get_dataset_manager()
        
        assert manager1 is manager2  # Should be same instance
        assert isinstance(manager1, DatasetManager)
    
    def test_simple_progress_callback(self):
        """Test simple progress callback function."""
        progress = DownloadProgress(
            total_bytes=1000,
            downloaded_bytes=500,
            speed_mbps=2.5,
            status="downloading"
        )
        
        # Test that callback runs without error
        # (Output testing would require capturing stdout)
        try:
            simple_progress_callback("test_dataset", progress)
        except Exception as e:
            pytest.fail(f"Progress callback raised an exception: {e}")


class TestDownloadProgress:
    """Test DownloadProgress dataclass."""
    
    def test_download_progress_defaults(self):
        """Test DownloadProgress with default values."""
        progress = DownloadProgress()
        
        assert progress.total_bytes == 0
        assert progress.downloaded_bytes == 0
        assert progress.speed_mbps == 0.0
        assert progress.eta_seconds is None
        assert progress.status == "pending"
        assert isinstance(progress.start_time, float)
    
    def test_download_progress_with_values(self):
        """Test DownloadProgress with explicit values."""
        start_time = time.time()
        
        progress = DownloadProgress(
            total_bytes=2000,
            downloaded_bytes=1000,
            start_time=start_time,
            speed_mbps=5.0,
            eta_seconds=200.0,
            status="downloading"
        )
        
        assert progress.total_bytes == 2000
        assert progress.downloaded_bytes == 1000
        assert progress.start_time == start_time
        assert progress.speed_mbps == 5.0
        assert progress.eta_seconds == 200.0
        assert progress.status == "downloading"


# Integration test
class TestDatasetManagementIntegration:
    """Integration tests for the dataset management system."""
    
    def setup_method(self):
        """Set up integration test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.manager = DatasetManager(
            cache_dir=self.temp_dir,
            max_cache_size_gb=0.001
        )
    
    def teardown_method(self):
        """Clean up integration test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_full_workflow_mock_download(self):
        """Test full workflow with mocked download."""
        # Register a test dataset
        test_info = DatasetInfo(
            name="integration_test",
            description="Integration test dataset",
            urls=["http://example.com/test.zip"],
            checksums={"md5": "test_hash_123"},
            file_size=1000,
            n_classes=5,
            n_samples=100,
            image_size=(32, 32)
        )
        
        self.manager.registry.register_dataset(test_info)
        
        # Mock successful download
        with patch.object(self.manager.downloader, 'download_file', return_value=True):
            with patch.object(self.manager, '_load_cached_dataset') as mock_load:
                mock_dataset = Mock()
                mock_load.return_value = mock_dataset
                
                result = self.manager.get_dataset("integration_test")
                
                assert result == mock_dataset
                mock_load.assert_called_once()
    
    def test_dependency_resolution_integration(self):
        """Test dependency resolution in full workflow."""
        # Create datasets with dependencies
        base_info = DatasetInfo(
            name="base_dataset",
            description="Base dataset",
            urls=["http://example.com/base.zip"],
            checksums={"md5": "base_hash"},
            file_size=500,
            n_classes=3,
            n_samples=50,
            image_size=(16, 16)
        )
        
        derived_info = DatasetInfo(
            name="derived_dataset",
            description="Derived dataset",
            urls=["http://example.com/derived.zip"],
            checksums={"md5": "derived_hash"},
            file_size=1000,
            n_classes=6,
            n_samples=100,
            image_size=(32, 32),
            dependencies=["base_dataset"]
        )
        
        self.manager.registry.register_dataset(base_info)
        self.manager.registry.register_dataset(derived_info)
        
        # Mock download for both datasets
        with patch.object(self.manager.downloader, 'download_file', return_value=True):
            with patch.object(self.manager, '_load_cached_dataset') as mock_load:
                mock_dataset = Mock()
                mock_load.return_value = mock_dataset
                
                result = self.manager.get_dataset("derived_dataset")
                
                # Should load the derived dataset after resolving dependencies
                assert result == mock_dataset


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])