"""
Tests for auto-download dataset integration functionality.
Tests the AutoDownloadDataset base class and enhanced dataset implementations.
"""

import pytest
import torch
import tempfile
import shutil
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import json

from meta_learning.data_utils.auto_download_datasets import (
    AutoDownloadDataset, 
    EnhancedMiniImageNet, 
    EnhancedOmniglot,
    DatasetRegistry,
    DatasetConfig
)
from meta_learning.shared.types import Episode


class TestDatasetConfig:
    """Test DatasetConfig dataclass."""
    
    def test_config_creation(self):
        config = DatasetConfig(
            name="test_dataset",
            urls=["http://example.com/data.zip"],
            checksums={"data.zip": "abc123"},
            checksum_type="md5",
            expected_files=["data.txt"],
            cache_subdir="test_cache"
        )
        
        assert config.name == "test_dataset"
        assert config.urls == ["http://example.com/data.zip"]
        assert config.checksums == {"data.zip": "abc123"}
        assert config.checksum_type == "md5"
        assert config.expected_files == ["data.txt"]
        assert config.cache_subdir == "test_cache"
    
    def test_config_defaults(self):
        config = DatasetConfig(
            name="minimal_dataset",
            urls=["http://example.com/data.zip"],
            checksums={"data.zip": "abc123"}
        )
        
        assert config.checksum_type == "md5"
        assert config.expected_files is None
        assert config.cache_subdir is None


class TestAutoDownloadDataset:
    """Test AutoDownloadDataset base class functionality."""
    
    @pytest.fixture
    def temp_dir(self):
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_config(self):
        return DatasetConfig(
            name="test_dataset",
            urls=["http://example.com/file1.zip", "http://mirror.com/file1.zip"],
            checksums={"file1.zip": "abc123"},
            checksum_type="md5"
        )
    
    def test_initialization(self, temp_dir, mock_config):
        class TestDataset(AutoDownloadDataset):
            def _get_dataset_config(self):
                return mock_config
            
            def _load_dataset(self):
                pass  # Skip loading for test
        
        dataset = TestDataset(root=temp_dir, download=False)  # Don't download
        
        assert dataset.root == Path(temp_dir)
        assert dataset.config.name == mock_config.name
        assert dataset.cache_dir == Path(temp_dir) / "test_dataset"
    
    @patch('meta_learning.data_utils.auto_download_datasets.download_file')
    def test_download_with_progress_success_first_mirror(self, mock_download, temp_dir, mock_config):
        mock_download.return_value = True
        
        dataset = AutoDownloadDataset(root=temp_dir, config=mock_config)
        progress_callback = Mock()
        
        result = dataset._download_with_progress(progress_callback)
        
        assert result is True
        mock_download.assert_called_once()
        progress_callback.assert_called()
    
    @patch('meta_learning.data_utils.auto_download_datasets.download_file')
    def test_download_with_progress_fallback_to_mirror(self, mock_download, temp_dir, mock_config):
        # First call fails, second succeeds
        mock_download.side_effect = [False, True]
        
        dataset = AutoDownloadDataset(root=temp_dir, config=mock_config)
        progress_callback = Mock()
        
        result = dataset._download_with_progress(progress_callback)
        
        assert result is True
        assert mock_download.call_count == 2
    
    @patch('meta_learning.data_utils.auto_download_datasets.download_file')
    def test_download_with_progress_all_mirrors_fail(self, mock_download, temp_dir, mock_config):
        mock_download.return_value = False
        
        dataset = AutoDownloadDataset(root=temp_dir, config=mock_config)
        progress_callback = Mock()
        
        result = dataset._download_with_progress(progress_callback)
        
        assert result is False
        assert mock_download.call_count == 2  # Both mirrors tried
    
    @patch('meta_learning.data_utils.auto_download_datasets.zipfile.ZipFile')
    def test_extract_archive_zip(self, mock_zipfile, temp_dir, mock_config):
        mock_zip_instance = Mock()
        mock_zipfile.return_value.__enter__.return_value = mock_zip_instance
        
        dataset = AutoDownloadDataset(root=temp_dir, config=mock_config)
        archive_path = Path(temp_dir) / "test.zip"
        archive_path.touch()  # Create empty file
        
        dataset._extract_archive(archive_path)
        
        mock_zipfile.assert_called_once_with(archive_path, 'r')
        mock_zip_instance.extractall.assert_called_once()
    
    @patch('meta_learning.data_utils.auto_download_datasets.tarfile.open')
    def test_extract_archive_tar(self, mock_tarfile, temp_dir, mock_config):
        mock_tar_instance = Mock()
        mock_tarfile.return_value.__enter__.return_value = mock_tar_instance
        
        dataset = AutoDownloadDataset(root=temp_dir, config=mock_config)
        archive_path = Path(temp_dir) / "test.tar.gz"
        archive_path.touch()  # Create empty file
        
        dataset._extract_archive(archive_path)
        
        mock_tarfile.assert_called_once_with(archive_path, 'r')
        mock_tar_instance.extractall.assert_called_once()
    
    def test_extract_archive_unsupported_format(self, temp_dir, mock_config):
        dataset = AutoDownloadDataset(root=temp_dir, config=mock_config)
        archive_path = Path(temp_dir) / "test.rar"
        archive_path.touch()  # Create empty file
        
        with pytest.raises(ValueError, match="Unsupported archive format"):
            dataset._extract_archive(archive_path)
    
    def test_is_downloaded_true_when_data_dir_exists(self, temp_dir, mock_config):
        dataset = AutoDownloadDataset(root=temp_dir, config=mock_config)
        dataset.data_dir.mkdir(parents=True, exist_ok=True)
        
        assert dataset.is_downloaded() is True
    
    def test_is_downloaded_false_when_data_dir_missing(self, temp_dir, mock_config):
        dataset = AutoDownloadDataset(root=temp_dir, config=mock_config)
        
        assert dataset.is_downloaded() is False
    
    @patch('meta_learning.data_utils.auto_download_datasets.AutoDownloadDataset._download_with_progress')
    @patch('meta_learning.data_utils.auto_download_datasets.AutoDownloadDataset._extract_archive')
    def test_download_and_extract_success(self, mock_extract, mock_download, temp_dir, mock_config):
        mock_download.return_value = True
        
        dataset = AutoDownloadDataset(root=temp_dir, config=mock_config)
        progress_callback = Mock()
        
        result = dataset.download_and_extract(progress_callback)
        
        assert result is True
        mock_download.assert_called_once_with(progress_callback)
        mock_extract.assert_called_once()
    
    @patch('meta_learning.data_utils.auto_download_datasets.AutoDownloadDataset._download_with_progress')
    def test_download_and_extract_download_failure(self, mock_download, temp_dir, mock_config):
        mock_download.return_value = False
        
        dataset = AutoDownloadDataset(root=temp_dir, config=mock_config)
        progress_callback = Mock()
        
        result = dataset.download_and_extract(progress_callback)
        
        assert result is False


class TestEnhancedMiniImageNet:
    """Test EnhancedMiniImageNet dataset implementation."""
    
    @pytest.fixture
    def temp_dir(self):
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_initialization_with_download_true(self, temp_dir):
        with patch.object(EnhancedMiniImageNet, 'download_and_extract') as mock_download:
            mock_download.return_value = True
            
            dataset = EnhancedMiniImageNet(root=temp_dir, split="train", download=True)
            
            assert dataset.split == "train"
            mock_download.assert_called_once()
    
    def test_initialization_with_download_false(self, temp_dir):
        with patch.object(EnhancedMiniImageNet, 'is_downloaded') as mock_is_downloaded:
            mock_is_downloaded.return_value = True
            
            dataset = EnhancedMiniImageNet(root=temp_dir, split="val", download=False)
            
            assert dataset.split == "val"
    
    def test_initialization_not_downloaded_error(self, temp_dir):
        with patch.object(EnhancedMiniImageNet, 'is_downloaded') as mock_is_downloaded:
            mock_is_downloaded.return_value = False
            
            with pytest.raises(RuntimeError, match="Dataset not found"):
                EnhancedMiniImageNet(root=temp_dir, split="test", download=False)
    
    def test_download_failure_error(self, temp_dir):
        with patch.object(EnhancedMiniImageNet, 'download_and_extract') as mock_download:
            mock_download.return_value = False
            
            with pytest.raises(RuntimeError, match="Failed to download"):
                EnhancedMiniImageNet(root=temp_dir, split="train", download=True)
    
    @patch('meta_learning.data_utils.auto_download_datasets.MiniImageNetDataset')
    def test_sample_support_query_delegation(self, mock_base_dataset, temp_dir):
        with patch.object(EnhancedMiniImageNet, 'is_downloaded') as mock_is_downloaded:
            mock_is_downloaded.return_value = True
            
            # Mock the base dataset
            mock_base_instance = Mock()
            mock_base_dataset.return_value = mock_base_instance
            mock_base_instance.sample_support_query.return_value = (
                torch.randn(15, 3, 84, 84),  # xs
                torch.randint(0, 5, (15,)),   # ys
                torch.randn(25, 3, 84, 84),  # xq
                torch.randint(0, 5, (25,))   # yq
            )
            
            dataset = EnhancedMiniImageNet(root=temp_dir, split="train", download=False)
            xs, ys, xq, yq = dataset.sample_support_query(5, 3, 5, seed=42)
            
            mock_base_instance.sample_support_query.assert_called_once_with(5, 3, 5, seed=42)
            assert xs.shape == (15, 3, 84, 84)
            assert ys.shape == (15,)
            assert xq.shape == (25, 3, 84, 84)
            assert yq.shape == (25,)


class TestEnhancedOmniglot:
    """Test EnhancedOmniglot dataset implementation."""
    
    @pytest.fixture
    def temp_dir(self):
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_initialization_with_download_true(self, temp_dir):
        with patch.object(EnhancedOmniglot, 'download_and_extract') as mock_download:
            with patch.object(EnhancedOmniglot, '_setup_omniglot_data') as mock_setup:
                mock_download.return_value = True
                
                dataset = EnhancedOmniglot(root=temp_dir, split="train", download=True)
                
                assert dataset.split == "train"
                mock_download.assert_called_once()
                mock_setup.assert_called_once()
    
    def test_initialization_not_downloaded_error(self, temp_dir):
        with patch.object(EnhancedOmniglot, 'is_downloaded') as mock_is_downloaded:
            mock_is_downloaded.return_value = False
            
            with pytest.raises(RuntimeError, match="Dataset not found"):
                EnhancedOmniglot(root=temp_dir, split="test", download=False)
    
    def test_setup_omniglot_data_creates_structure(self, temp_dir):
        with patch.object(EnhancedOmniglot, 'is_downloaded') as mock_is_downloaded:
            mock_is_downloaded.return_value = True
            
            # Create mock data directory structure
            data_dir = Path(temp_dir) / "omniglot"
            data_dir.mkdir(parents=True)
            
            # Create fake alphabet and character structure
            alphabet_dir = data_dir / "images_background" / "Alphabet_1"
            char_dir = alphabet_dir / "character01"
            char_dir.mkdir(parents=True)
            
            # Create fake image files
            for i in range(3):
                (char_dir / f"image_{i}.png").touch()
            
            dataset = EnhancedOmniglot(root=temp_dir, split="train", download=False)
            
            # Check that class mapping was created
            assert hasattr(dataset, 'class_to_files')
            assert len(dataset.class_to_files) > 0
    
    @patch('meta_learning.data_utils.auto_download_datasets.Image.open')
    @patch('meta_learning.data_utils.auto_download_datasets.transforms')
    def test_sample_support_query(self, mock_transforms, mock_image_open, temp_dir):
        with patch.object(EnhancedOmniglot, 'is_downloaded') as mock_is_downloaded:
            mock_is_downloaded.return_value = True
            
            # Mock transform
            mock_transform = Mock()
            mock_transform.return_value = torch.randn(1, 28, 28)
            mock_transforms.Compose.return_value = mock_transform
            
            # Mock PIL Image
            mock_image = Mock()
            mock_image_open.return_value = mock_image
            
            dataset = EnhancedOmniglot(root=temp_dir, split="train", download=False)
            
            # Set up mock class structure
            dataset.class_to_files = {
                0: [f"path_to_class_0_img_{i}.png" for i in range(10)],
                1: [f"path_to_class_1_img_{i}.png" for i in range(10)],
                2: [f"path_to_class_2_img_{i}.png" for i in range(10)],
                3: [f"path_to_class_3_img_{i}.png" for i in range(10)],
                4: [f"path_to_class_4_img_{i}.png" for i in range(10)]
            }
            
            xs, ys, xq, yq = dataset.sample_support_query(5, 1, 5, seed=42)
            
            # Check output shapes
            assert xs.shape == (5, 1, 28, 28)  # 5-way, 1-shot support
            assert ys.shape == (5,)
            assert xq.shape == (25, 1, 28, 28)  # 5-way, 5 queries per class
            assert yq.shape == (25,)
            
            # Check label range
            assert torch.all(ys >= 0) and torch.all(ys < 5)
            assert torch.all(yq >= 0) and torch.all(yq < 5)


class TestDatasetRegistry:
    """Test DatasetRegistry functionality."""
    
    def test_initialization(self):
        registry = DatasetRegistry()
        assert len(registry.datasets) == 0
    
    def test_register_dataset(self):
        registry = DatasetRegistry()
        mock_config = DatasetConfig(name="test", urls=["http://example.com"])
        mock_class = Mock()
        
        registry.register("test_dataset", mock_class, mock_config)
        
        assert "test_dataset" in registry.datasets
        assert registry.datasets["test_dataset"]["class"] == mock_class
        assert registry.datasets["test_dataset"]["config"] == mock_config
    
    def test_get_dataset_class(self):
        registry = DatasetRegistry()
        mock_config = DatasetConfig(name="test", urls=["http://example.com"])
        mock_class = Mock()
        
        registry.register("test_dataset", mock_class, mock_config)
        
        result_class = registry.get_dataset_class("test_dataset")
        assert result_class == mock_class
    
    def test_get_dataset_class_not_found(self):
        registry = DatasetRegistry()
        
        with pytest.raises(ValueError, match="Dataset 'unknown' not found"):
            registry.get_dataset_class("unknown")
    
    def test_get_dataset_config(self):
        registry = DatasetRegistry()
        mock_config = DatasetConfig(name="test", urls=["http://example.com"])
        mock_class = Mock()
        
        registry.register("test_dataset", mock_class, mock_config)
        
        result_config = registry.get_dataset_config("test_dataset")
        assert result_config == mock_config
    
    def test_list_available_datasets(self):
        registry = DatasetRegistry()
        mock_config1 = DatasetConfig(name="test1", urls=["http://example.com"])
        mock_config2 = DatasetConfig(name="test2", urls=["http://example.com"])
        
        registry.register("dataset1", Mock(), mock_config1)
        registry.register("dataset2", Mock(), mock_config2)
        
        available = registry.list_available()
        
        assert set(available) == {"dataset1", "dataset2"}
    
    def test_create_dataset_instance(self, tmpdir):
        registry = DatasetRegistry()
        
        # Use a real dataset class for this test
        mock_config = DatasetConfig(name="test", urls=["http://example.com"])
        
        # Create a simple mock dataset class
        class MockDataset:
            def __init__(self, root, **kwargs):
                self.root = root
                self.kwargs = kwargs
        
        registry.register("test_dataset", MockDataset, mock_config)
        
        instance = registry.create_dataset("test_dataset", root=str(tmpdir), split="train")
        
        assert isinstance(instance, MockDataset)
        assert instance.root == str(tmpdir)
        assert instance.kwargs["split"] == "train"


class TestIntegrationAutoDownloadDatasets:
    """Integration tests for auto-download dataset functionality."""
    
    @pytest.fixture
    def temp_dir(self):
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_enhanced_miniimagenet_fallback_to_existing_dataset(self, temp_dir):
        """Test that EnhancedMiniImageNet falls back to existing MiniImageNetDataset when data exists."""
        
        # Create the expected directory structure
        data_dir = Path(temp_dir) / "miniimagenet"
        splits_dir = data_dir / "splits"
        images_dir = data_dir / "images"
        
        splits_dir.mkdir(parents=True)
        images_dir.mkdir(parents=True)
        
        # Create a minimal train.csv file
        train_csv = splits_dir / "train.csv"
        with open(train_csv, 'w') as f:
            f.write("file,cls\n")
            f.write("img1.jpg,class1\n")
            f.write("img2.jpg,class1\n")
            f.write("img3.jpg,class2\n")
        
        # Create corresponding image files
        for img in ["img1.jpg", "img2.jpg", "img3.jpg"]:
            (images_dir / img).touch()
        
        with patch('meta_learning.data_utils.auto_download_datasets.MiniImageNetDataset') as mock_dataset_class:
            # Mock the MiniImageNetDataset to avoid actual image loading
            mock_instance = Mock()
            mock_dataset_class.return_value = mock_instance
            mock_instance.sample_support_query.return_value = (
                torch.randn(2, 3, 84, 84),  # xs
                torch.tensor([0, 1]),       # ys
                torch.randn(2, 3, 84, 84),  # xq  
                torch.tensor([0, 1])        # yq
            )
            
            # Test that EnhancedMiniImageNet can be created without download
            dataset = EnhancedMiniImageNet(root=temp_dir, split="train", download=False)
            
            # Test that it can generate episodes
            xs, ys, xq, yq = dataset.sample_support_query(2, 1, 1, seed=42)
            
            assert xs.shape[0] == 2  # 2-way support
            assert ys.shape[0] == 2
            assert xq.shape[0] == 2  # 2-way query
            assert yq.shape[0] == 2
    
    def test_dataset_registry_with_real_datasets(self):
        """Test DatasetRegistry with real dataset classes."""
        
        registry = DatasetRegistry()
        
        # Test registration
        assert len(registry.list_available()) == 0
        
        # Since registry might be pre-populated, check that we can add more
        initial_count = len(registry.list_available())
        
        # Register a test dataset
        test_config = DatasetConfig(name="test", urls=["http://example.com"])
        registry.register("test_dataset", EnhancedMiniImageNet, test_config)
        
        assert len(registry.list_available()) == initial_count + 1
        assert "test_dataset" in registry.list_available()
        
        # Test retrieval
        dataset_class = registry.get_dataset_class("test_dataset")
        assert dataset_class == EnhancedMiniImageNet
        
        config = registry.get_dataset_config("test_dataset")
        assert config == test_config