"""
Tests for new Episode methods added to meta_learning.core.episode
"""
import torch
import pytest
from unittest.mock import MagicMock, patch
from meta_learning.core.episode import Episode, remap_labels


class TestEpisodeFromPartition:
    """Test Episode.from_partition method."""
    
    @patch('meta_learning.core.episode.partition_task')
    def test_from_partition_basic(self, mock_partition):
        """Test basic from_partition functionality."""
        # Mock partition_task return value
        support_data = torch.randn(6, 10)
        support_labels = torch.tensor([0, 0, 1, 1, 2, 2])
        query_data = torch.randn(9, 10)
        query_labels = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])
        
        mock_partition.return_value = ((support_data, support_labels), (query_data, query_labels))
        
        # Test data
        data = torch.randn(15, 10)
        labels = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
        
        # Call method
        episode = Episode.from_partition(data, labels, shots=2)
        
        # Verify partition_task was called correctly
        mock_partition.assert_called_once_with(data, labels, 2)
        
        # Verify Episode structure
        assert torch.equal(episode.support_x, support_data)
        assert torch.equal(episode.support_y, support_labels)
        assert torch.equal(episode.query_x, query_data)
        assert torch.equal(episode.query_y, query_labels)
    
    @patch('meta_learning.core.episode.partition_task')
    def test_from_partition_different_shots(self, mock_partition):
        """Test from_partition with different shot values."""
        # Mock return for 1-shot
        mock_partition.return_value = ((torch.randn(3, 5), torch.arange(3)), 
                                      (torch.randn(6, 5), torch.repeat_interleave(torch.arange(3), 2)))
        
        data = torch.randn(9, 5)
        labels = torch.repeat_interleave(torch.arange(3), 3)
        
        episode = Episode.from_partition(data, labels, shots=1)
        
        mock_partition.assert_called_once_with(data, labels, 1)
        assert episode.support_x.shape[0] == 3  # 3 classes × 1 shot
        assert episode.query_x.shape[0] == 6   # 3 classes × 2 query
    
    @patch('meta_learning.core.episode.partition_task')
    def test_from_partition_preserves_device(self, mock_partition):
        """Test that from_partition preserves tensor device."""
        device = torch.device('cpu')
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        
        # Mock return with device-specific tensors
        support_data = torch.randn(4, 8, device=device)
        support_labels = torch.tensor([0, 0, 1, 1], device=device)
        query_data = torch.randn(8, 8, device=device)
        query_labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], device=device)
        
        mock_partition.return_value = ((support_data, support_labels), (query_data, query_labels))
        
        data = torch.randn(12, 8, device=device)
        labels = torch.tensor([0] * 6 + [1] * 6, device=device)
        
        episode = Episode.from_partition(data, labels, shots=2)
        
        assert episode.support_x.device == device
        assert episode.support_y.device == device
        assert episode.query_x.device == device
        assert episode.query_y.device == device


class TestEpisodeExtendedFromRawData:
    """Test enhanced from_raw_data with comprehensive documentation and examples."""
    
    def test_from_raw_data_comprehensive_docstring_example(self):
        """Test the example from the comprehensive docstring."""
        # Create synthetic data: 100 examples, 10 classes as in docstring
        torch.manual_seed(42)  # For reproducible test
        data = torch.randn(100, 64)  # 100 examples, 64 features
        labels = torch.repeat_interleave(torch.arange(10), 10)  # 10 each
        
        # Create 5-way 1-shot episode (though this would sample from all 10 classes)
        episode = Episode.from_raw_data(data, labels, n_shot=1, n_query=15)
        
        # Verify the docstring claims
        assert episode.num_classes == 10  # Uses all available classes
        assert episode.support_x.shape == (10, 64)  # 10 classes × 1 shot
        assert episode.query_x.shape == (150, 64)   # 10 classes × 15 query
    
    def test_from_raw_data_reproducible_sampling_docstring(self):
        """Test reproducible sampling as documented in docstring."""
        data = torch.randn(40, 32)
        labels = torch.repeat_interleave(torch.arange(4), 10)  # 4 classes, 10 samples each
        
        # This will always produce the same episode split
        episode1 = Episode.from_raw_data(data, labels, n_shot=2, n_query=3, random_state=42)
        episode2 = Episode.from_raw_data(data, labels, n_shot=2, n_query=3, random_state=42)
        
        # Verify identical episodes
        assert torch.equal(episode1.support_x, episode2.support_x)
        assert torch.equal(episode1.support_y, episode2.support_y)
        assert torch.equal(episode1.query_x, episode2.query_x)
        assert torch.equal(episode1.query_y, episode2.query_y)
    
    def test_from_raw_data_error_handling_comprehensive(self):
        """Test comprehensive error handling as documented."""
        data = torch.randn(15, 8)
        labels = torch.tensor([0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3])  # Uneven class sizes
        
        # Class 0 has only 2 samples, should fail for n_shot=2, n_query=2
        with pytest.raises(ValueError, match="Class 0 has only 2 samples, need 4"):
            Episode.from_raw_data(data, labels, n_shot=2, n_query=2)
        
        # Should work for smaller requirements
        episode = Episode.from_raw_data(data, labels, n_shot=1, n_query=1)
        assert episode.num_classes == 4
    
    def test_from_raw_data_arbitrary_dimensions(self):
        """Test from_raw_data with arbitrary tensor dimensions."""
        # Test with image-like data (C, H, W)
        data = torch.randn(20, 3, 32, 32)  # 20 RGB images
        labels = torch.repeat_interleave(torch.arange(4), 5)  # 4 classes, 5 each
        
        episode = Episode.from_raw_data(data, labels, n_shot=2, n_query=2, random_state=123)
        
        assert episode.support_x.shape == (8, 3, 32, 32)   # 4 classes × 2 shots
        assert episode.query_x.shape == (8, 3, 32, 32)     # 4 classes × 2 queries
        assert episode.support_x.dim() == 4  # Preserves original dimensions
    
    def test_from_raw_data_single_class_edge_case(self):
        """Test from_raw_data with single class (edge case)."""
        data = torch.randn(10, 5)
        labels = torch.zeros(10, dtype=torch.long)  # All same class
        
        episode = Episode.from_raw_data(data, labels, n_shot=3, n_query=2)
        
        assert episode.num_classes == 1
        assert episode.support_x.shape == (3, 5)
        assert episode.query_x.shape == (2, 5)
        assert torch.all(episode.support_y == 0)
        assert torch.all(episode.query_y == 0)


class TestEpisodeExtendedFromDataset:
    """Test enhanced from_dataset with comprehensive documentation."""
    
    def test_from_dataset_docstring_example_mock(self):
        """Test the CIFAR-10 example from docstring (mocked)."""
        # Mock CIFAR-10 dataset structure
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=50)  # Smaller for test
        
        def mock_getitem(idx):
            # Return (image, label) where each class has 5 samples
            label = idx // 5  # Classes 0-9
            image = torch.randn(3, 32, 32)  # CIFAR-10 shape
            return image, label
        
        mock_dataset.__getitem__ = mock_getitem
        
        # Create 5-way 5-shot episode as in docstring
        episode = Episode.from_dataset(mock_dataset, n_way=5, n_shot=2, 
                                     n_query=2, random_state=42)
        
        # Verify docstring claims
        assert episode.num_classes == 5
        assert episode.support_x.shape == (10, 3, 32, 32)  # 5 classes × 2 shots
        assert episode.query_x.shape == (10, 3, 32, 32)    # 5 classes × 2 queries
    
    def test_from_dataset_class_sampling_verification(self):
        """Test that n_way classes are properly sampled."""
        # Create dataset with 8 classes
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=80)  # 10 samples per class
        
        def mock_getitem(idx):
            return torch.randn(16), idx // 10  # 8 classes, 10 samples each
        
        mock_dataset.__getitem__ = mock_getitem
        
        # Sample 3-way episode
        episode = Episode.from_dataset(mock_dataset, n_way=3, n_shot=2, 
                                     n_query=3, random_state=999)
        
        assert episode.num_classes == 3
        # Should have exactly 3 unique classes in support and query
        support_classes = set(episode.support_y.tolist())
        query_classes = set(episode.query_y.tolist())
        assert len(support_classes) == 3
        assert len(query_classes) == 3
        assert support_classes == query_classes
    
    def test_from_dataset_insufficient_classes_error(self):
        """Test error when dataset has insufficient classes."""
        # Dataset with only 2 classes
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=10)
        mock_dataset.__getitem__ = lambda idx: (torch.randn(4), idx % 2)
        
        # Request 5-way episode (impossible)
        with pytest.raises(ValueError, match="Dataset has only 2 classes, need 5"):
            Episode.from_dataset(mock_dataset, n_way=5, n_shot=1, n_query=1)
    
    def test_from_dataset_preserves_tensor_properties(self):
        """Test that from_dataset preserves tensor dtypes and properties."""
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=20)
        
        def mock_getitem(idx):
            # Return float32 data and int64 labels
            data = torch.randn(8, dtype=torch.float32)
            label = torch.tensor(idx // 5, dtype=torch.int64)  # 4 classes
            return data, label
        
        mock_dataset.__getitem__ = mock_getitem
        
        episode = Episode.from_dataset(mock_dataset, n_way=3, n_shot=1, n_query=2)
        
        assert episode.support_x.dtype == torch.float32
        assert episode.query_x.dtype == torch.float32
        assert episode.support_y.dtype == torch.int64
        assert episode.query_y.dtype == torch.int64


class TestEpisodeExtendedToDevice:
    """Test enhanced to_device with comprehensive documentation."""
    
    def test_to_device_docstring_example_cpu_to_cpu(self):
        """Test the CPU to CPU example from docstring."""
        # Create episode on CPU as in docstring
        support_x = torch.randn(5, 64)
        support_y = torch.arange(5)
        query_x = torch.randn(15, 64)
        query_y = torch.repeat_interleave(torch.arange(5), 3)
        episode = Episode(support_x, support_y, query_x, query_y)
        
        # Move to CPU (no-op but should work)
        cpu_episode = episode.to_device(torch.device('cpu'))
        
        assert cpu_episode.support_x.device.type == 'cpu'
        assert cpu_episode.support_y.device.type == 'cpu'
        assert cpu_episode.query_x.device.type == 'cpu'
        assert cpu_episode.query_y.device.type == 'cpu'
        
        # Data should be preserved
        assert torch.equal(episode.support_x, cpu_episode.support_x)
        assert torch.equal(episode.support_y, cpu_episode.support_y)
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_to_device_docstring_example_cpu_to_gpu(self):
        """Test the CPU to GPU example from docstring."""
        # Create episode on CPU
        support_x = torch.randn(5, 64)
        support_y = torch.arange(5)
        query_x = torch.randn(15, 64)
        query_y = torch.repeat_interleave(torch.arange(5), 3)
        episode = Episode(support_x, support_y, query_x, query_y)
        
        # Move to GPU with pinned memory as in docstring
        device = torch.device('cuda:0')
        gpu_episode = episode.to_device(device, pin_memory=True)
        
        assert gpu_episode.support_x.device.type == 'cuda'
        assert gpu_episode.support_y.device.type == 'cuda'
        assert gpu_episode.query_x.device.type == 'cuda'
        assert gpu_episode.query_y.device.type == 'cuda'
        
        # Data should be preserved (allowing for device transfer precision)
        assert torch.allclose(episode.support_x.cpu(), gpu_episode.support_x.cpu())
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available") 
    def test_to_device_round_trip_gpu_cpu(self):
        """Test GPU to CPU round trip as in docstring."""
        # Start on CPU
        episode = Episode(
            torch.randn(6, 32), torch.randint(0, 3, (6,)),
            torch.randn(9, 32), torch.randint(0, 3, (9,))
        )
        
        # CPU → GPU → CPU
        gpu_episode = episode.to_device(torch.device('cuda:0'), pin_memory=True)
        cpu_episode = gpu_episode.to_device(torch.device('cpu'))
        
        # Should end up back on CPU with same data
        assert cpu_episode.support_x.device.type == 'cpu'
        assert torch.allclose(episode.support_x, cpu_episode.support_x)
        assert torch.equal(episode.support_y, cpu_episode.support_y)
    
    def test_to_device_pinned_memory_mechanism(self):
        """Test pinned memory mechanism in detail."""
        episode = Episode(
            torch.randn(4, 16), torch.tensor([0, 0, 1, 1]),
            torch.randn(4, 16), torch.tensor([0, 0, 1, 1])
        )
        
        # Mock pin_memory to verify it's called
        with patch.object(torch.Tensor, 'pin_memory') as mock_pin:
            mock_pin.return_value.to = MagicMock(return_value=torch.randn(4, 16))
            
            if torch.cuda.is_available():
                device = torch.device('cuda:0')
                episode.to_device(device, pin_memory=True)
                # pin_memory should be called for CPU → CUDA transfers
                assert mock_pin.call_count >= 1
    
    def test_to_device_no_pinning_for_same_device(self):
        """Test that pinned memory isn't used for same-device transfers."""
        episode = Episode(
            torch.randn(2, 8), torch.tensor([0, 1]),
            torch.randn(2, 8), torch.tensor([0, 1])
        )
        
        # Move CPU → CPU with pin_memory=True (should be ignored)
        with patch.object(torch.Tensor, 'pin_memory') as mock_pin:
            new_episode = episode.to_device(torch.device('cpu'), pin_memory=True)
            # pin_memory should not be called for same device
            mock_pin.assert_not_called()


class TestExtendedRemapLabels:
    """Test enhanced remap_labels with comprehensive documentation."""
    
    def test_remap_labels_docstring_example(self):
        """Test the exact example from the comprehensive docstring."""
        # Labels with arbitrary values as in docstring
        y_support = torch.tensor([10, 25, 10, 37, 25])
        y_query = torch.tensor([10, 10, 25, 37, 37, 25])
        
        # Remap to contiguous [0, 1, 2] as claimed in docstring
        support_remapped, query_remapped = remap_labels(y_support, y_query)
        
        # Verify the exact output claimed in docstring
        expected_support = torch.tensor([0, 1, 0, 2, 1])  # From docstring
        expected_query = torch.tensor([0, 0, 1, 2, 2, 1])    # From docstring
        
        assert torch.equal(support_remapped, expected_support)
        assert torch.equal(query_remapped, expected_query)
        
        # Verify mapping consistency as mentioned in docstring
        unique_remapped = torch.unique(support_remapped)
        expected_unique = torch.tensor([0, 1, 2])
        assert torch.equal(unique_remapped, expected_unique)
    
    def test_remap_labels_comprehensive_error_handling(self):
        """Test error handling as documented in docstring."""
        y_support = torch.tensor([5, 10])
        y_query = torch.tensor([5, 10, 15])  # 15 not in support
        
        # Should raise KeyError as documented
        with pytest.raises(KeyError):
            remap_labels(y_support, y_query)
    
    def test_remap_labels_consistency_verification(self):
        """Test mapping consistency as emphasized in docstring."""
        y_support = torch.tensor([100, 200, 100, 300, 200, 300])
        y_query = torch.tensor([100, 200, 300, 100, 300, 200])
        
        support_remapped, query_remapped = remap_labels(y_support, y_query)
        
        # Verify contiguous remapping
        unique_support = torch.unique(support_remapped)
        expected_range = torch.arange(len(unique_support))
        assert torch.equal(torch.sort(unique_support).values, expected_range)
        
        # Verify query is subset of support classes
        unique_query = torch.unique(query_remapped)
        assert torch.all(torch.isin(unique_query, unique_support))
    
    def test_remap_labels_preserves_device_and_dtype(self):
        """Test that remap_labels preserves device and returns long dtype."""
        device = torch.device('cpu')
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        
        y_support = torch.tensor([7, 3, 7], device=device, dtype=torch.int32)
        y_query = torch.tensor([3, 7, 3], device=device, dtype=torch.int16)
        
        support_remapped, query_remapped = remap_labels(y_support, y_query)
        
        # Should preserve device
        assert support_remapped.device == device
        assert query_remapped.device == device
        
        # Should return long dtype as documented
        assert support_remapped.dtype == torch.int64
        assert query_remapped.dtype == torch.int64
    
    def test_remap_labels_single_class_edge_case(self):
        """Test remap_labels with single class (edge case)."""
        y_support = torch.tensor([42, 42, 42])
        y_query = torch.tensor([42, 42])
        
        support_remapped, query_remapped = remap_labels(y_support, y_query)
        
        # Single class should map to 0
        assert torch.all(support_remapped == 0)
        assert torch.all(query_remapped == 0)
        assert support_remapped.dtype == torch.int64
        assert query_remapped.dtype == torch.int64


class TestEpisodePropertiesDocumentation:
    """Test that Episode properties work as documented."""
    
    def test_episode_comprehensive_docstring_example(self):
        """Test the comprehensive example from Episode class docstring.""" 
        # Create a simple 5-way 1-shot episode as in docstring
        support_x = torch.randn(5, 64)  # 5 support examples, 64 features
        support_y = torch.arange(5)     # Labels [0, 1, 2, 3, 4]
        query_x = torch.randn(15, 64)   # 15 query examples
        query_y = torch.repeat_interleave(torch.arange(5), 3)  # [0,0,0,1,1,1,...]
        
        episode = Episode(support_x, support_y, query_x, query_y)
        episode.validate()  # Validates the episode structure
        
        # Verify docstring claims
        assert episode.num_classes == 5     # As claimed in docstring
        assert episode.num_samples == 20    # As claimed in docstring (5 + 15)
    
    def test_episode_attributes_documentation(self):
        """Test Episode attributes as documented in comprehensive docstring."""
        # Test the exact shapes mentioned in docstring
        n_support, n_query = 8, 12
        feature_dim = 128
        
        episode = Episode(
            support_x=torch.randn(n_support, feature_dim),     # [n_support, ...]
            support_y=torch.randint(0, 4, (n_support,)),       # [n_support] 
            query_x=torch.randn(n_query, feature_dim),         # [n_query, ...]
            query_y=torch.randint(0, 4, (n_query,))            # [n_query]
        )
        
        # Verify shapes match documentation
        assert episode.support_x.shape == (n_support, feature_dim)
        assert episode.support_y.shape == (n_support,)
        assert episode.query_x.shape == (n_query, feature_dim)
        assert episode.query_y.shape == (n_query,)
        
        # Verify label range as documented [0, n_classes-1]
        all_labels = torch.cat([episode.support_y, episode.query_y])
        assert torch.all(all_labels >= 0)
        assert torch.all(all_labels < episode.num_classes)
    
    def test_episode_note_about_contiguous_labels(self):
        """Test the note about contiguous labels in docstring."""
        # Non-contiguous labels should fail validation
        episode = Episode(
            support_x=torch.randn(4, 10),
            support_y=torch.tensor([0, 2, 0, 2]),  # Missing class 1
            query_x=torch.randn(2, 10),
            query_y=torch.tensor([0, 2])
        )
        
        # Should fail validation due to non-contiguous labels
        with pytest.raises(AssertionError, match="labels must be.*contiguous"):
            episode.validate()


if __name__ == "__main__":
    pytest.main([__file__])