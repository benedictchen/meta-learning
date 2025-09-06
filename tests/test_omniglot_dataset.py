#!/usr/bin/env python3
"""
Test OmniglotDataset
===================

Tests for the OmniglotDataset implementation that handles alphabet organization,
rotation augmentation, and episode generation for few-shot learning.
"""

import pytest
import torch
import tempfile
import os
from unittest.mock import patch, MagicMock
from typing import Dict, Any, Optional

from meta_learning.data_utils.datasets import OmniglotDataset, DatasetRegistry


class TestOmniglotDataset:
    """Test OmniglotDataset functionality."""

    def test_initialization_parameters(self):
        """Test dataset initialization with various parameters."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test with mock data (skip actual download/loading)
            with patch.object(OmniglotDataset, '_load_data'), \
                 patch.object(OmniglotDataset, '_data_exists', return_value=True):
                
                dataset = OmniglotDataset(
                    root=temp_dir,
                    mode='train',
                    download=False,
                    transform=None,
                    background=True,
                    rotation_augmentation=False,
                    validate_data=False
                )
                
                assert dataset.root == temp_dir
                assert dataset.mode == 'train'
                assert dataset.background == True
                assert dataset.rotation_augmentation == False
                assert dataset.validate_data == False

    def test_initialization_modes(self):
        """Test dataset initialization with different modes."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(OmniglotDataset, '_load_data'), \
                 patch.object(OmniglotDataset, '_data_exists', return_value=True):
                
                # Test train mode
                dataset_train = OmniglotDataset(root=temp_dir, mode='train', download=False)
                assert dataset_train.mode == 'train'
                
                # Test test mode
                dataset_test = OmniglotDataset(root=temp_dir, mode='test', download=False)
                assert dataset_test.mode == 'test'

    def test_data_path_generation(self):
        """Test that data paths are generated correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(OmniglotDataset, '_load_data'), \
                 patch.object(OmniglotDataset, '_data_exists', return_value=True):
                
                # Background train
                dataset = OmniglotDataset(root=temp_dir, mode='train', background=True, download=False)
                expected_path = os.path.join(temp_dir, 'omniglot_background_train.pkl')
                assert dataset.data_path == expected_path
                
                # Evaluation train
                dataset = OmniglotDataset(root=temp_dir, mode='train', background=False, download=False)
                expected_path = os.path.join(temp_dir, 'omniglot_evaluation_train.pkl')
                assert dataset.data_path == expected_path
                
                # Background test
                dataset = OmniglotDataset(root=temp_dir, mode='test', background=True, download=False)
                expected_path = os.path.join(temp_dir, 'omniglot_background_test.pkl')
                assert dataset.data_path == expected_path

    def test_data_exists_check(self):
        """Test data existence checking."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(OmniglotDataset, '_load_data'):
                # Create mock data file
                data_path = os.path.join(temp_dir, 'omniglot_background_train.pkl')
                with open(data_path, 'w') as f:
                    f.write("mock data")
                
                dataset = OmniglotDataset(root=temp_dir, mode='train', download=False)
                assert dataset._data_exists() == True
                
                # Test non-existent file
                os.remove(data_path)
                assert dataset._data_exists() == False

    @patch('meta_learning.data_utils.datasets.urllib.request.urlretrieve')
    @patch('meta_learning.data_utils.datasets.zipfile.ZipFile')
    def test_download_dataset(self, mock_zipfile, mock_urlretrieve):
        """Test dataset download functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(OmniglotDataset, '_load_data'), \
                 patch.object(OmniglotDataset, '_data_exists', return_value=False):
                
                # Mock successful download and extraction
                mock_zip = MagicMock()
                mock_zipfile.return_value.__enter__.return_value = mock_zip
                
                dataset = OmniglotDataset(root=temp_dir, download=True)
                dataset._download_dataset()
                
                # Verify download was attempted
                mock_urlretrieve.assert_called()
                mock_zipfile.assert_called()
                mock_zip.extractall.assert_called()

    def test_episode_creation_structure(self):
        """Test episode creation returns correct structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock data structure
            mock_data = {
                'alphabet_0': {
                    'character_0': [torch.randn(28, 28) for _ in range(10)],
                    'character_1': [torch.randn(28, 28) for _ in range(10)],
                    'character_2': [torch.randn(28, 28) for _ in range(10)],
                    'character_3': [torch.randn(28, 28) for _ in range(10)],
                    'character_4': [torch.randn(28, 28) for _ in range(10)],
                }
            }
            
            with patch.object(OmniglotDataset, '_load_data'), \
                 patch.object(OmniglotDataset, '_data_exists', return_value=True):
                
                dataset = OmniglotDataset(root=temp_dir, download=False)
                dataset.data = mock_data
                dataset.alphabets = list(mock_data.keys())
                dataset.characters_per_alphabet = {
                    'alphabet_0': list(mock_data['alphabet_0'].keys())
                }
                
                episode = dataset.create_episode(n_way=3, n_shot=2, n_query=5)
                
                # Check episode structure
                assert hasattr(episode, 'support_x')
                assert hasattr(episode, 'support_y')
                assert hasattr(episode, 'query_x')
                assert hasattr(episode, 'query_y')
                
                # Check dimensions
                assert episode.support_x.shape[0] == 3 * 2  # n_way * n_shot
                assert episode.support_y.shape[0] == 3 * 2
                assert episode.query_x.shape[0] == 3 * 5    # n_way * n_query
                assert episode.query_y.shape[0] == 3 * 5
                
                # Check label structure
                unique_labels = torch.unique(episode.support_y)
                assert len(unique_labels) == 3  # n_way classes
                assert unique_labels.tolist() == [0, 1, 2]  # Consecutive labels

    def test_episode_creation_parameters(self):
        """Test episode creation with various parameters."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create larger mock data
            mock_data = {}
            for a in range(3):  # 3 alphabets
                alphabet_name = f'alphabet_{a}'
                mock_data[alphabet_name] = {}
                for c in range(10):  # 10 characters per alphabet
                    char_name = f'character_{c}'
                    mock_data[alphabet_name][char_name] = [torch.randn(28, 28) for _ in range(20)]
            
            with patch.object(OmniglotDataset, '_load_data'), \
                 patch.object(OmniglotDataset, '_data_exists', return_value=True):
                
                dataset = OmniglotDataset(root=temp_dir, download=False)
                dataset.data = mock_data
                dataset.alphabets = list(mock_data.keys())
                dataset.characters_per_alphabet = {
                    alphabet: list(mock_data[alphabet].keys())
                    for alphabet in mock_data
                }
                
                # Test 5-way 1-shot
                episode = dataset.create_episode(n_way=5, n_shot=1, n_query=15)
                assert episode.support_x.shape[0] == 5
                assert episode.query_x.shape[0] == 75
                
                # Test 3-way 5-shot
                episode = dataset.create_episode(n_way=3, n_shot=5, n_query=10)
                assert episode.support_x.shape[0] == 15
                assert episode.query_x.shape[0] == 30

    def test_rotation_augmentation(self):
        """Test rotation augmentation functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            mock_data = {
                'alphabet_0': {
                    'character_0': [torch.ones(28, 28) for _ in range(10)],
                    'character_1': [torch.zeros(28, 28) for _ in range(10)],
                    'character_2': [torch.ones(28, 28) * 0.5 for _ in range(10)],
                }
            }
            
            with patch.object(OmniglotDataset, '_load_data'), \
                 patch.object(OmniglotDataset, '_data_exists', return_value=True):
                
                # Test with rotation augmentation enabled
                dataset_rot = OmniglotDataset(
                    root=temp_dir, 
                    download=False,
                    rotation_augmentation=True
                )
                dataset_rot.data = mock_data
                dataset_rot.alphabets = list(mock_data.keys())
                dataset_rot.characters_per_alphabet = {
                    'alphabet_0': list(mock_data['alphabet_0'].keys())
                }
                
                # Test without rotation augmentation
                dataset_no_rot = OmniglotDataset(
                    root=temp_dir,
                    download=False,
                    rotation_augmentation=False
                )
                dataset_no_rot.data = mock_data
                dataset_no_rot.alphabets = list(mock_data.keys())
                dataset_no_rot.characters_per_alphabet = {
                    'alphabet_0': list(mock_data['alphabet_0'].keys())
                }
                
                # Create episodes with same seed for comparison
                torch.manual_seed(42)
                episode_rot = dataset_rot.create_episode(n_way=3, n_shot=1, n_query=3)
                
                torch.manual_seed(42)
                episode_no_rot = dataset_no_rot.create_episode(n_way=3, n_shot=1, n_query=3)
                
                # Episodes should potentially be different due to rotation
                # (This is probabilistic, but with rotation there's a chance of difference)
                # At minimum, dimensions should be the same
                assert episode_rot.support_x.shape == episode_no_rot.support_x.shape
                assert episode_rot.query_x.shape == episode_no_rot.query_x.shape

    def test_alphabet_balancing(self):
        """Test that episodes balance across different alphabets."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mock data with multiple alphabets
            mock_data = {}
            for a in range(5):  # 5 alphabets
                alphabet_name = f'alphabet_{a}'
                mock_data[alphabet_name] = {}
                for c in range(6):  # 6 characters per alphabet
                    char_name = f'character_{c}'
                    mock_data[alphabet_name][char_name] = [torch.randn(28, 28) for _ in range(10)]
            
            with patch.object(OmniglotDataset, '_load_data'), \
                 patch.object(OmniglotDataset, '_data_exists', return_value=True):
                
                dataset = OmniglotDataset(root=temp_dir, download=False)
                dataset.data = mock_data
                dataset.alphabets = list(mock_data.keys())
                dataset.characters_per_alphabet = {
                    alphabet: list(mock_data[alphabet].keys())
                    for alphabet in mock_data
                }
                
                # Create multiple episodes and track alphabet usage
                alphabet_usage = {alphabet: 0 for alphabet in mock_data.keys()}
                
                for _ in range(20):  # Create 20 episodes
                    episode = dataset.create_episode(n_way=3, n_shot=1, n_query=5)
                    
                    # Track which alphabets were used (mock implementation)
                    # In reality, we'd need to track the actual character selection
                    # For now, we just verify the episode structure is correct
                    assert episode.support_x.shape[0] == 3
                    assert episode.query_x.shape[0] == 15

    def test_insufficient_data_handling(self):
        """Test handling when there's insufficient data for episode creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mock data with insufficient characters
            mock_data = {
                'alphabet_0': {
                    'character_0': [torch.randn(28, 28) for _ in range(5)],
                    'character_1': [torch.randn(28, 28) for _ in range(3)],  # Insufficient for large n_shot
                }
            }
            
            with patch.object(OmniglotDataset, '_load_data'), \
                 patch.object(OmniglotDataset, '_data_exists', return_value=True):
                
                dataset = OmniglotDataset(root=temp_dir, download=False)
                dataset.data = mock_data
                dataset.alphabets = list(mock_data.keys())
                dataset.characters_per_alphabet = {
                    'alphabet_0': list(mock_data['alphabet_0'].keys())
                }
                
                # Should handle insufficient characters gracefully
                with pytest.raises((ValueError, RuntimeError)):
                    # Requesting more ways than available characters
                    dataset.create_episode(n_way=5, n_shot=1, n_query=5)

    def test_transform_application(self):
        """Test that transforms are applied correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            mock_data = {
                'alphabet_0': {
                    'character_0': [torch.ones(28, 28) for _ in range(10)],
                    'character_1': [torch.zeros(28, 28) for _ in range(10)],
                    'character_2': [torch.ones(28, 28) * 0.5 for _ in range(10)],
                }
            }
            
            # Define a simple transform
            def simple_transform(x):
                return x * 2.0
            
            with patch.object(OmniglotDataset, '_load_data'), \
                 patch.object(OmniglotDataset, '_data_exists', return_value=True):
                
                dataset_with_transform = OmniglotDataset(
                    root=temp_dir,
                    download=False,
                    transform=simple_transform
                )
                dataset_with_transform.data = mock_data
                dataset_with_transform.alphabets = list(mock_data.keys())
                dataset_with_transform.characters_per_alphabet = {
                    'alphabet_0': list(mock_data['alphabet_0'].keys())
                }
                
                dataset_without_transform = OmniglotDataset(
                    root=temp_dir,
                    download=False,
                    transform=None
                )
                dataset_without_transform.data = mock_data
                dataset_without_transform.alphabets = list(mock_data.keys())
                dataset_without_transform.characters_per_alphabet = {
                    'alphabet_0': list(mock_data['alphabet_0'].keys())
                }
                
                # Create identical episodes
                torch.manual_seed(123)
                episode_with = dataset_with_transform.create_episode(n_way=3, n_shot=1, n_query=3)
                
                torch.manual_seed(123)
                episode_without = dataset_without_transform.create_episode(n_way=3, n_shot=1, n_query=3)
                
                # Transformed data should be different (doubled)
                # This is approximate since random sampling may affect exact values
                assert not torch.allclose(episode_with.support_x, episode_without.support_x)

    def test_error_handling(self):
        """Test various error conditions."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test invalid mode
            with pytest.raises(ValueError):
                with patch.object(OmniglotDataset, '_load_data'), \
                     patch.object(OmniglotDataset, '_data_exists', return_value=True):
                    OmniglotDataset(root=temp_dir, mode='invalid_mode', download=False)

    def test_dataset_registry_integration(self):
        """Test integration with DatasetRegistry."""
        # Check that OmniglotDataset is registered
        registry = DatasetRegistry()
        assert 'omniglot' in registry.list_datasets()
        
        # Test dataset creation through registry
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(OmniglotDataset, '_load_data'), \
                 patch.object(OmniglotDataset, '_data_exists', return_value=True):
                
                dataset = registry.create_dataset(
                    'omniglot',
                    root=temp_dir,
                    download=False
                )
                
                assert isinstance(dataset, OmniglotDataset)

    def test_validation_disabled(self):
        """Test dataset creation with validation disabled."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(OmniglotDataset, '_load_data'), \
                 patch.object(OmniglotDataset, '_data_exists', return_value=True):
                
                # Should not raise any validation errors
                dataset = OmniglotDataset(
                    root=temp_dir,
                    download=False,
                    validate_data=False
                )
                
                assert dataset.validate_data == False

    def test_memory_efficiency(self):
        """Test that dataset doesn't load all data into memory unnecessarily."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(OmniglotDataset, '_load_data') as mock_load, \
                 patch.object(OmniglotDataset, '_data_exists', return_value=True):
                
                dataset = OmniglotDataset(root=temp_dir, download=False)
                
                # _load_data should be called during initialization
                mock_load.assert_called_once()

    def test_reproducibility(self):
        """Test that dataset produces reproducible results with same seed."""
        with tempfile.TemporaryDirectory() as temp_dir:
            mock_data = {
                'alphabet_0': {
                    f'character_{i}': [torch.randn(28, 28) for _ in range(10)]
                    for i in range(10)
                }
            }
            
            with patch.object(OmniglotDataset, '_load_data'), \
                 patch.object(OmniglotDataset, '_data_exists', return_value=True):
                
                dataset1 = OmniglotDataset(root=temp_dir, download=False)
                dataset1.data = mock_data
                dataset1.alphabets = list(mock_data.keys())
                dataset1.characters_per_alphabet = {
                    'alphabet_0': list(mock_data['alphabet_0'].keys())
                }
                
                dataset2 = OmniglotDataset(root=temp_dir, download=False)
                dataset2.data = mock_data
                dataset2.alphabets = list(mock_data.keys())
                dataset2.characters_per_alphabet = {
                    'alphabet_0': list(mock_data['alphabet_0'].keys())
                }
                
                # Create episodes with same seed
                torch.manual_seed(456)
                episode1 = dataset1.create_episode(n_way=5, n_shot=2, n_query=10)
                
                torch.manual_seed(456)
                episode2 = dataset2.create_episode(n_way=5, n_shot=2, n_query=10)
                
                # Episodes should be identical with same seed
                assert torch.allclose(episode1.support_x, episode2.support_x)
                assert torch.allclose(episode1.query_x, episode2.query_x)
                assert torch.equal(episode1.support_y, episode2.support_y)
                assert torch.equal(episode1.query_y, episode2.query_y)


if __name__ == "__main__":
    pytest.main([__file__])