"""
Tests for BenchmarkDatasetManager.

Tests the professional dataset downloading, caching, and integrity verification system.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import json
import time

from meta_learning.datasets.benchmark_manager import (
    BenchmarkDatasetManager, DatasetRegistry
)


class TestDatasetRegistry:
    """Test dataset registry functionality."""
    
    def test_registry_contains_expected_datasets(self):
        """Test that registry contains standard benchmark datasets."""
        registry = DatasetRegistry()
        
        expected_datasets = ['mini_imagenet', 'cifar_fs', 'omniglot']
        for dataset in expected_datasets:
            assert dataset in registry.DATASETS
    
    def test_dataset_metadata_structure(self):
        """Test that dataset metadata has required fields."""
        registry = DatasetRegistry()
        
        for dataset_name, metadata in registry.DATASETS.items():
            assert 'urls' in metadata
            assert 'checksums' in metadata
            assert 'description' in metadata
            assert isinstance(metadata['urls'], list)
            assert isinstance(metadata['checksums'], dict)
            assert len(metadata['urls']) > 0


class TestBenchmarkDatasetManager:
    """Test benchmark dataset manager."""
    
    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def manager(self, temp_cache_dir):
        """Create manager with temporary cache."""
        return BenchmarkDatasetManager(cache_dir=temp_cache_dir, max_cache_size_gb=1.0)
    
    def test_initialization(self, temp_cache_dir):
        """Test manager initialization."""
        manager = BenchmarkDatasetManager(cache_dir=temp_cache_dir)
        
        assert manager.cache_dir == Path(temp_cache_dir)
        assert manager.cache_dir.exists()
        assert manager.registry is not None
        assert manager.metadata_file.name == "metadata.json"
    
    def test_list_available_datasets(self, manager):
        """Test listing available datasets."""
        datasets = manager.list_available_datasets()
        
        assert isinstance(datasets, dict)
        assert len(datasets) > 0
        assert 'mini_imagenet' in datasets
        assert 'cifar_fs' in datasets
        assert 'omniglot' in datasets
    
    def test_add_custom_dataset(self, manager):
        """Test adding custom dataset to registry."""
        custom_urls = ["http://example.com/data.zip"]
        custom_checksums = {"data.zip": "sha256:abcd1234"}
        
        manager.add_dataset(
            name="custom_dataset",
            urls=custom_urls,
            checksums=custom_checksums,
            description="Custom test dataset"
        )
        
        datasets = manager.list_available_datasets()
        assert 'custom_dataset' in datasets
        assert datasets['custom_dataset']['urls'] == custom_urls
        assert datasets['custom_dataset']['checksums'] == custom_checksums
    
    def test_checksum_verification(self, manager):
        """Test checksum verification functionality."""
        # Create test file
        test_file = manager.cache_dir / "test_file.txt"
        test_content = b"test content"
        test_file.write_bytes(test_content)
        
        # Calculate expected checksum (sha256 of "test content")
        import hashlib
        expected_checksum = "sha256:" + hashlib.sha256(test_content).hexdigest()
        
        # Test verification
        assert manager._verify_checksum(test_file, expected_checksum)
        assert not manager._verify_checksum(test_file, "sha256:wrong_hash")
    
    def test_cache_size_management(self, manager):
        """Test cache size management and eviction."""
        # Create mock cached datasets
        manager._cached_datasets = {
            "dataset1": {"last_accessed": time.time() - 100, "size": 500 * 1024**3},  # Old, large
            "dataset2": {"last_accessed": time.time() - 50, "size": 100 * 1024**3},   # Recent, small
        }
        
        # Create corresponding directories
        (manager.cache_dir / "dataset1").mkdir()
        (manager.cache_dir / "dataset2").mkdir()
        
        # Mock the directory size calculation to return large size for dataset1
        with patch.object(Path, 'rglob') as mock_rglob:
            mock_files = [Mock(stat=Mock(return_value=Mock(st_size=500 * 1024**3)))]
            mock_rglob.return_value = mock_files
            
            # This should trigger eviction of dataset1 (older)
            manager._manage_cache_size()
        
        # dataset1 should be removed from metadata (would be removed from disk in real scenario)
        assert "dataset1" not in manager._cached_datasets
    
    @patch('meta_learning.datasets.benchmark_manager.urlretrieve')
    def test_download_with_retry(self, mock_urlretrieve, manager):
        """Test download with retry functionality."""
        test_url = "http://example.com/data.zip"
        test_dest = manager.cache_dir / "data.zip"
        
        # Test successful download
        mock_urlretrieve.return_value = None
        result = manager._download_with_retry(test_url, test_dest, max_retries=2)
        assert result is True
        mock_urlretrieve.assert_called_with(test_url, test_dest)
        
        # Test failed download with retries
        mock_urlretrieve.side_effect = Exception("Network error")
        result = manager._download_with_retry(test_url, test_dest, max_retries=2)
        assert result is False
        assert mock_urlretrieve.call_count == 3  # 1 + 2 retries
    
    def test_cache_info(self, manager):
        """Test cache information retrieval."""
        # Create test cache structure
        test_dataset_dir = manager.cache_dir / "test_dataset"
        test_dataset_dir.mkdir()
        test_file = test_dataset_dir / "data.txt"
        test_file.write_text("test data")
        
        cache_info = manager.get_cache_info()
        
        assert isinstance(cache_info, dict)
        assert 'cache_dir' in cache_info
        assert 'total_size_mb' in cache_info
        assert 'max_size_gb' in cache_info
        assert 'utilization' in cache_info
        assert 'datasets' in cache_info
    
    def test_clear_cache(self, manager):
        """Test cache clearing functionality."""
        # Create test cache
        test_dataset_dir = manager.cache_dir / "test_dataset"
        test_dataset_dir.mkdir()
        test_file = test_dataset_dir / "data.txt"
        test_file.write_text("test data")
        
        manager._cached_datasets["test_dataset"] = {"cached_at": time.time()}
        
        # Clear specific dataset
        manager.clear_cache("test_dataset")
        assert not test_dataset_dir.exists()
        assert "test_dataset" not in manager._cached_datasets
        
        # Create again and clear all
        test_dataset_dir.mkdir()
        test_file.write_text("test data")
        manager._cached_datasets["test_dataset"] = {"cached_at": time.time()}
        
        manager.clear_cache()
        assert len(list(manager.cache_dir.iterdir())) <= 1  # Only metadata.json might remain
    
    def test_metadata_persistence(self, manager):
        """Test metadata save/load functionality."""
        # Add test metadata
        test_metadata = {
            "test_dataset": {
                "downloaded_at": time.time(),
                "last_accessed": time.time(),
                "verified": True
            }
        }
        manager._cached_datasets.update(test_metadata)
        manager._save_metadata()
        
        # Create new manager and check metadata loaded
        new_manager = BenchmarkDatasetManager(cache_dir=manager.cache_dir)
        assert "test_dataset" in new_manager._cached_datasets
        assert new_manager._cached_datasets["test_dataset"]["verified"] is True


class TestIntegration:
    """Integration tests for dataset management system."""
    
    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_end_to_end_dataset_management(self, temp_cache_dir):
        """Test complete dataset management workflow."""
        manager = BenchmarkDatasetManager(cache_dir=temp_cache_dir)
        
        # List datasets
        datasets = manager.list_available_datasets()
        assert len(datasets) > 0
        
        # Add custom dataset
        manager.add_dataset(
            name="test_dataset",
            urls=["http://example.com/test.zip"],
            description="Test dataset"
        )
        
        # Verify addition
        updated_datasets = manager.list_available_datasets()
        assert "test_dataset" in updated_datasets
        
        # Get cache info
        cache_info = manager.get_cache_info()
        assert isinstance(cache_info, dict)
        
        # Test cache management
        manager.clear_cache()
        final_cache_info = manager.get_cache_info()
        assert final_cache_info["total_size_mb"] == 0


if __name__ == "__main__":
    pytest.main([__file__])