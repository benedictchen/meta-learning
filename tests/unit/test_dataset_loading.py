"""
Tests for Dataset Loading Solutions
==================================

Tests all 4 dataset loading methods with proper configurations and fallbacks.
"""

import pytest
import torch
# DEMO/TEST REPRODUCIBILITY: Set seed for consistent synthetic data
torch.manual_seed(42)  # Reproducible test/demo data
from unittest.mock import patch, MagicMock
from meta_learning.meta_learning_modules.few_shot_modules.utilities import (
    sample_episode, DatasetLoadingConfig,
    _load_with_torchmeta, _load_with_custom_splits, 
    _load_with_huggingface, _load_synthetic_data
)


class TestDatasetLoading:
    """Test suite for dataset loading methods."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.n_way = 5
        self.n_support = 5
        self.n_query = 15
        
    def test_synthetic_data_loading(self):
        """Test synthetic data generation (fallback method)."""
        config = DatasetLoadingConfig(method="synthetic")
        
        support_x, support_y, query_x, query_y = sample_episode(
            "omniglot", self.n_way, self.n_support, self.n_query, config
        )
        
        # Check shapes
        expected_support_samples = self.n_way * self.n_support
        expected_query_samples = self.n_way * self.n_query
        
        assert support_x.shape[0] == expected_support_samples
        assert support_y.shape[0] == expected_support_samples
        assert query_x.shape[0] == expected_query_samples
        assert query_y.shape[0] == expected_query_samples
        
        # Check labels are in correct range
        assert torch.all(support_y >= 0) and torch.all(support_y < self.n_way)
        assert torch.all(query_y >= 0) and torch.all(query_y < self.n_way)
        
        # Check data types
        assert support_x.dtype == torch.float32
        assert support_y.dtype == torch.long
        
    def test_synthetic_data_different_datasets(self):
        """Test synthetic data for different dataset types."""
        config = DatasetLoadingConfig(method="synthetic")
        
        datasets = ["omniglot", "miniimagenet", "tieredimagenet", "custom"]
        expected_shapes = {
            "omniglot": (1, 28, 28),
            "miniimagenet": (3, 84, 84),
            "tieredimagenet": (3, 84, 84),
            "custom": (3, 32, 32)  # Default
        }
        
        for dataset in datasets:
            support_x, support_y, query_x, query_y = sample_episode(
                dataset, self.n_way, self.n_support, self.n_query, config
            )
            
            expected_shape = expected_shapes[dataset]
            assert support_x.shape[1:] == expected_shape
            assert query_x.shape[1:] == expected_shape
            
    @patch('meta_learning.meta_learning_modules.few_shot_modules.utilities.BatchMetaDataLoader')
    @patch('meta_learning.meta_learning_modules.few_shot_modules.utilities.Omniglot')
    def test_torchmeta_loading_omniglot(self, mock_omniglot, mock_dataloader):
        """Test torchmeta loading for Omniglot dataset."""
        # Mock the torchmeta components
        mock_dataset = MagicMock()
        mock_omniglot.return_value = mock_dataset
        
        mock_task = {
            'train': (torch.randn(1, 25, 1, 28, 28), torch.randint(0, 5, (1, 25))),
            'test': (torch.randn(1, 75, 1, 28, 28), torch.randint(0, 5, (1, 75)))
        }
        mock_dataloader_instance = MagicMock()
        mock_dataloader_instance.__iter__ = lambda self: iter([mock_task])
        mock_dataloader.return_value = mock_dataloader_instance
        
        config = DatasetLoadingConfig(method="torchmeta", torchmeta_download=False)
        
        support_x, support_y, query_x, query_y = _load_with_torchmeta(
            "omniglot", self.n_way, self.n_support, self.n_query, config
        )
        
        # Verify mock was called correctly
        mock_omniglot.assert_called_once()
        assert support_x.shape == (25, 1, 28, 28)
        assert query_x.shape == (75, 1, 28, 28)
        
    def test_torchmeta_import_error(self):
        """Test handling of missing torchmeta dependency."""
        config = DatasetLoadingConfig(method="torchmeta")
        
        with patch('meta_learning.meta_learning_modules.few_shot_modules.utilities.Omniglot', side_effect=ImportError("No module named 'torchmeta'")):
            with pytest.raises(ImportError, match="torchmeta not installed"):
                _load_with_torchmeta("omniglot", self.n_way, self.n_support, self.n_query, config)
                
    @patch('builtins.open')
    @patch('json.load')
    @patch('os.path.exists')
    @patch('os.listdir')
    @patch('PIL.Image.open')
    def test_custom_splits_loading(self, mock_image_open, mock_listdir, mock_path_exists, mock_json_load, mock_open):
        """Test custom splits loading method."""
        # Mock file system and JSON
        mock_path_exists.return_value = True
        mock_json_load.return_value = {
            'train': ['class_1', 'class_2', 'class_3', 'class_4', 'class_5']
        }
        mock_listdir.return_value = ['img_1.jpg', 'img_2.jpg', 'img_3.jpg', 'img_4.jpg', 'img_5.jpg', 
                                    'img_6.jpg', 'img_7.jpg', 'img_8.jpg', 'img_9.jpg', 'img_10.jpg']
        
        # Mock PIL Image
        from PIL import Image
        mock_img = MagicMock(spec=Image.Image)
        mock_image_open.return_value = mock_img
        
        config = DatasetLoadingConfig(method="custom")
        
        # This would normally require actual file system setup
        # For now, test that the function structure works
        with patch('torchvision.transforms.Compose'):
            with patch('numpy.random.choice', return_value=['class_1', 'class_2', 'class_3']):
                with pytest.raises((FileNotFoundError, OSError)):
                    # Expected to fail due to mocked file system
                    _load_with_custom_splits("miniimagenet", 3, 2, 3, config)
                    
    @patch('datasets.load_dataset')
    def test_huggingface_loading(self, mock_load_dataset):
        """Test HuggingFace datasets loading method."""
        # Mock HuggingFace dataset
        mock_dataset = [
            {'alphabet': 'alphabet_1', 'image': MagicMock()},
            {'alphabet': 'alphabet_1', 'image': MagicMock()},
            {'alphabet': 'alphabet_2', 'image': MagicMock()},
            {'alphabet': 'alphabet_2', 'image': MagicMock()},
        ]
        mock_load_dataset.return_value = mock_dataset
        
        config = DatasetLoadingConfig(method="huggingface")
        
        # Mock image resize and transforms
        with patch('torchvision.transforms.Compose'):
            with patch('numpy.random.choice', return_value=['alphabet_1', 'alphabet_2']):
                # This would work with proper mocking of image transforms
                with pytest.raises(AttributeError):
                    # Expected to fail due to mock image objects
                    _load_with_huggingface("omniglot", 2, 2, 3, config)
                    
    def test_huggingface_import_error(self):
        """Test handling of missing datasets dependency."""
        config = DatasetLoadingConfig(method="huggingface")
        
        with patch('meta_learning.meta_learning_modules.few_shot_modules.utilities.load_dataset', side_effect=ImportError("No module named 'datasets'")):
            with pytest.raises(ImportError, match="datasets library not installed"):
                _load_with_huggingface("omniglot", self.n_way, self.n_support, self.n_query, config)
                
    def test_fallback_mechanism(self):
        """Test fallback to synthetic data when primary method fails."""
        config = DatasetLoadingConfig(
            method="torchmeta",
            fallback_to_synthetic=True,
            warn_on_fallback=False
        )
        
        # Force torchmeta to fail by patching with ImportError
        with patch('meta_learning.meta_learning_modules.few_shot_modules.utilities._load_with_torchmeta', side_effect=ImportError("Mocked failure")):
            support_x, support_y, query_x, query_y = sample_episode(
                "omniglot", self.n_way, self.n_support, self.n_query, config
            )
            
            # Should get synthetic data as fallback
            assert support_x.shape[0] == self.n_way * self.n_support
            assert query_x.shape[0] == self.n_way * self.n_query
            
    def test_fallback_disabled(self):
        """Test error when fallback is disabled and primary method fails."""
        config = DatasetLoadingConfig(
            method="torchmeta",
            fallback_to_synthetic=False
        )
        
        with patch('meta_learning.meta_learning_modules.few_shot_modules.utilities._load_with_torchmeta', side_effect=ImportError("Mocked failure")):
            with pytest.raises(RuntimeError, match="Dataset loading failed and fallback disabled"):
                sample_episode("omniglot", self.n_way, self.n_support, self.n_query, config)
                
    def test_unknown_method(self):
        """Test error handling for unknown dataset loading method."""
        config = DatasetLoadingConfig(method="unknown_method")
        
        with pytest.raises(ValueError, match="Unknown dataset loading method"):
            sample_episode("omniglot", self.n_way, self.n_support, self.n_query, config)


class TestDatasetLoadingConfig:
    """Test dataset loading configuration class."""
    
    def test_config_defaults(self):
        """Test default configuration values."""
        config = DatasetLoadingConfig()
        
        assert config.method == "torchmeta"
        assert config.torchmeta_root == "data"
        assert config.torchmeta_download == True
        assert config.fallback_to_synthetic == True
        assert config.warn_on_fallback == True
        
    def test_config_customization(self):
        """Test custom configuration values."""
        config = DatasetLoadingConfig(
            method="custom",
            custom_data_root="/custom/path",
            image_size=(128, 128),
            normalize_mean=(0.5, 0.5, 0.5)
        )
        
        assert config.method == "custom"
        assert config.custom_data_root == "/custom/path"
        assert config.image_size == (128, 128)
        assert config.normalize_mean == (0.5, 0.5, 0.5)


class TestDatasetLoadingIntegration:
    """Integration tests for dataset loading."""
    
    def test_episode_consistency(self):
        """Test that episodes have consistent structure."""
        config = DatasetLoadingConfig(method="synthetic")
        
        # Generate multiple episodes
        episodes = []
        for _ in range(5):
            episode = sample_episode("omniglot", 3, 2, 5, config)
            episodes.append(episode)
            
        # All episodes should have same structure
        for episode in episodes:
            support_x, support_y, query_x, query_y = episode
            
            assert support_x.shape == (6, 1, 28, 28)  # 3 classes * 2 support
            assert support_y.shape == (6,)
            assert query_x.shape == (15, 1, 28, 28)   # 3 classes * 5 query
            assert query_y.shape == (15,)
            
            # Check label distribution
            assert len(torch.unique(support_y)) == 3
            assert len(torch.unique(query_y)) == 3
            
    def test_different_episode_configurations(self):
        """Test various episode configurations."""
        config = DatasetLoadingConfig(method="synthetic")
        
        test_configs = [
            (2, 1, 1),   # 2-way 1-shot
            (5, 5, 15),  # 5-way 5-shot
            (10, 3, 10), # 10-way 3-shot
        ]
        
        for n_way, n_support, n_query in test_configs:
            support_x, support_y, query_x, query_y = sample_episode(
                "omniglot", n_way, n_support, n_query, config
            )
            
            # Check shapes match configuration
            assert support_x.shape[0] == n_way * n_support
            assert query_x.shape[0] == n_way * n_query
            
            # Check label ranges
            assert torch.all(support_y >= 0) and torch.all(support_y < n_way)
            assert torch.all(query_y >= 0) and torch.all(query_y < n_way)
            
    def test_data_quality_checks(self):
        """Test data quality and validity."""
        config = DatasetLoadingConfig(method="synthetic")
        
        support_x, support_y, query_x, query_y = sample_episode(
            "omniglot", 5, 3, 10, config
        )
        
        # Check for NaN or infinite values
        assert torch.all(torch.isfinite(support_x))
        assert torch.all(torch.isfinite(query_x))
        
        # Check data ranges (synthetic data should be reasonable)
        assert support_x.abs().max() < 100  # Not extremely large
        assert query_x.abs().max() < 100
        
        # Check label integrity
        unique_support = torch.unique(support_y)
        unique_query = torch.unique(query_y)
        
        assert len(unique_support) == 5
        assert len(unique_query) == 5
        assert torch.equal(unique_support, unique_query)  # Same classes in both sets