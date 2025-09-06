"""
Tests for Episode functionality in meta_learning.core.episode
"""
import torch
import pytest
from unittest.mock import MagicMock
from meta_learning.core.episode import Episode


class TestEpisodeFactoryMethods:
    """Test Episode factory methods."""
    
    def test_from_raw_data_basic(self):
        """Test creating Episode from raw data."""
        # Create data with 3 classes, 4 samples per class
        data = torch.randn(12, 5)  # 12 samples, 5 features
        labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
        
        episode = Episode.from_raw_data(data, labels, n_shot=2, n_query=1, random_state=42)
        
        # Should have 3 classes * 2 shots = 6 support samples
        assert episode.support_x.shape[0] == 6
        # Should have 3 classes * 1 query = 3 query samples
        assert episode.query_x.shape[0] == 3
        # Features should be preserved
        assert episode.support_x.shape[1] == 5
        assert episode.query_x.shape[1] == 5
        # Labels should be remapped to 0, 1, 2
        assert set(episode.support_y.tolist()) == {0, 1, 2}
        assert set(episode.query_y.tolist()) == {0, 1, 2}
    
    def test_from_raw_data_insufficient_samples(self):
        """Test error handling when insufficient samples per class."""
        data = torch.randn(5, 3)
        labels = torch.tensor([0, 0, 1, 1, 1])  # Class 0 has only 2 samples
        
        with pytest.raises(ValueError, match="Class 0 has only 2 samples"):
            Episode.from_raw_data(data, labels, n_shot=2, n_query=2)
    
    def test_from_raw_data_reproducibility(self):
        """Test reproducibility with same random state."""
        data = torch.randn(20, 4)
        labels = torch.repeat_interleave(torch.arange(4), 5)
        
        episode1 = Episode.from_raw_data(data, labels, n_shot=2, n_query=1, random_state=123)
        episode2 = Episode.from_raw_data(data, labels, n_shot=2, n_query=1, random_state=123)
        
        # Should produce identical episodes
        assert torch.equal(episode1.support_x, episode2.support_x)
        assert torch.equal(episode1.support_y, episode2.support_y)
        assert torch.equal(episode1.query_x, episode2.query_x)
        assert torch.equal(episode1.query_y, episode2.query_y)
    
    def test_from_dataset_basic(self):
        """Test creating Episode from dataset."""
        # Mock dataset with 20 samples, 4 classes
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=20)
        
        def mock_getitem(idx):
            return torch.randn(3), idx // 5  # 5 samples per class
        
        mock_dataset.__getitem__ = mock_getitem
        
        episode = Episode.from_dataset(mock_dataset, n_way=3, n_shot=1, n_query=1, random_state=42)
        
        # Should sample 3 classes
        assert episode.num_classes == 3
        # Should have 3 support + 3 query samples
        assert episode.support_x.shape[0] == 3
        assert episode.query_x.shape[0] == 3
    
    def test_from_dataset_insufficient_classes(self):
        """Test error when dataset has insufficient classes."""
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=4)
        mock_dataset.__getitem__ = lambda idx: (torch.randn(2), 0)  # All same class
        
        with pytest.raises(ValueError, match="Dataset has only 1 classes, need 5"):
            Episode.from_dataset(mock_dataset, n_way=5, n_shot=1, n_query=1)


class TestEpisodeDeviceManagement:
    """Test Episode device management."""
    
    def test_to_device_cpu_to_cpu(self):
        """Test moving Episode from CPU to CPU."""
        episode = Episode(
            support_x=torch.randn(6, 4),
            support_y=torch.randint(0, 3, (6,)),
            query_x=torch.randn(3, 4),
            query_y=torch.randint(0, 3, (3,))
        )
        
        new_episode = episode.to_device(torch.device('cpu'))
        
        assert new_episode.support_x.device.type == 'cpu'
        assert new_episode.support_y.device.type == 'cpu'
        assert new_episode.query_x.device.type == 'cpu'
        assert new_episode.query_y.device.type == 'cpu'
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_to_device_cpu_to_cuda(self):
        """Test moving Episode from CPU to CUDA."""
        episode = Episode(
            support_x=torch.randn(6, 4),
            support_y=torch.randint(0, 3, (6,)),
            query_x=torch.randn(3, 4),
            query_y=torch.randint(0, 3, (3,))
        )
        
        cuda_device = torch.device('cuda:0')
        new_episode = episode.to_device(cuda_device, pin_memory=True)
        
        assert new_episode.support_x.device.type == 'cuda'
        assert new_episode.support_y.device.type == 'cuda'
        assert new_episode.query_x.device.type == 'cuda'
        assert new_episode.query_y.device.type == 'cuda'
    
    def test_to_device_preserves_data(self):
        """Test that to_device preserves data values."""
        episode = Episode(
            support_x=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            support_y=torch.tensor([0, 1]),
            query_x=torch.tensor([[5.0, 6.0]]),
            query_y=torch.tensor([0])
        )
        
        new_episode = episode.to_device(torch.device('cpu'))
        
        assert torch.equal(episode.support_x, new_episode.support_x)
        assert torch.equal(episode.support_y, new_episode.support_y)
        assert torch.equal(episode.query_x, new_episode.query_x)
        assert torch.equal(episode.query_y, new_episode.query_y)


class TestEpisodeCompatibility:
    """Test TorchMeta Task compatibility."""
    
    def test_num_classes_property(self):
        """Test num_classes property."""
        episode = Episode(
            support_x=torch.randn(6, 4),
            support_y=torch.tensor([0, 0, 1, 1, 2, 2]),
            query_x=torch.randn(3, 4),
            query_y=torch.tensor([0, 1, 2])
        )
        
        assert episode.num_classes == 3
    
    def test_num_samples_property(self):
        """Test num_samples property."""
        episode = Episode(
            support_x=torch.randn(6, 4),
            support_y=torch.randint(0, 3, (6,)),
            query_x=torch.randn(4, 4),
            query_y=torch.randint(0, 3, (4,))
        )
        
        assert episode.num_samples == 10  # 6 + 4
    
    def test_len_method(self):
        """Test __len__ method for TorchMeta compatibility."""
        episode = Episode(
            support_x=torch.randn(5, 3),
            support_y=torch.randint(0, 2, (5,)),
            query_x=torch.randn(3, 3),
            query_y=torch.randint(0, 2, (3,))
        )
        
        assert len(episode) == 8  # 5 + 3
    
    def test_getitem_method(self):
        """Test __getitem__ method for TorchMeta compatibility."""
        support_x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        support_y = torch.tensor([0, 1])
        query_x = torch.tensor([[5.0, 6.0]])
        query_y = torch.tensor([0])
        
        episode = Episode(
            support_x=support_x,
            support_y=support_y,
            query_x=query_x,
            query_y=query_y
        )
        
        # Test support set access
        data0, label0 = episode[0]
        assert torch.equal(data0, torch.tensor([1.0, 2.0]))
        assert label0 == 0
        
        data1, label1 = episode[1]
        assert torch.equal(data1, torch.tensor([3.0, 4.0]))
        assert label1 == 1
        
        # Test query set access
        data2, label2 = episode[2]
        assert torch.equal(data2, torch.tensor([5.0, 6.0]))
        assert label2 == 0
    
    def test_getitem_out_of_bounds(self):
        """Test __getitem__ with out of bounds index."""
        episode = Episode(
            support_x=torch.randn(2, 3),
            support_y=torch.tensor([0, 1]),
            query_x=torch.randn(1, 3),
            query_y=torch.tensor([0])
        )
        
        with pytest.raises(IndexError):
            _ = episode[3]  # Only indices 0, 1, 2 are valid


class TestEpisodeValidation:
    """Test Episode validation functionality."""
    
    def test_validate_successful(self):
        """Test successful validation."""
        episode = Episode(
            support_x=torch.randn(4, 5),
            support_y=torch.tensor([0, 0, 1, 1]),
            query_x=torch.randn(2, 5),
            query_y=torch.tensor([0, 1])
        )
        
        # Should not raise any exception
        episode.validate()
        episode.validate(expect_n_classes=2)
    
    def test_validate_shape_mismatch(self):
        """Test validation with shape mismatch."""
        with pytest.raises(AssertionError, match="support X/Y mismatch"):
            episode = Episode(
                support_x=torch.randn(4, 5),
                support_y=torch.tensor([0, 0, 1]),  # Wrong length
                query_x=torch.randn(2, 5),
                query_y=torch.tensor([0, 1])
            )
            episode.validate()
    
    def test_validate_wrong_dtype(self):
        """Test validation with wrong label dtype."""
        with pytest.raises(AssertionError, match="labels must be int64"):
            episode = Episode(
                support_x=torch.randn(2, 3),
                support_y=torch.tensor([0.0, 1.0]),  # Wrong dtype (float)
                query_x=torch.randn(1, 3),
                query_y=torch.tensor([0])
            )
            episode.validate()
    
    def test_validate_wrong_class_count(self):
        """Test validation with wrong expected class count."""
        episode = Episode(
            support_x=torch.randn(6, 4),
            support_y=torch.tensor([0, 0, 1, 1, 2, 2]),
            query_x=torch.randn(3, 4),
            query_y=torch.tensor([0, 1, 2])
        )
        
        with pytest.raises(AssertionError, match="expected 2 classes, got 3"):
            episode.validate(expect_n_classes=2)


class TestEpisodeIntegration:
    """Integration tests for Episode functionality."""
    
    def test_full_workflow(self):
        """Test complete Episode workflow."""
        # Create synthetic dataset
        torch.manual_seed(42)
        data = torch.randn(50, 8)  # 50 samples, 8 features
        labels = torch.repeat_interleave(torch.arange(5), 10)  # 5 classes, 10 samples each
        
        # Create episode from raw data
        episode = Episode.from_raw_data(data, labels, n_shot=3, n_query=2, random_state=123)
        
        # Validate episode
        episode.validate(expect_n_classes=5)
        
        # Test properties
        assert episode.num_classes == 5
        assert episode.num_samples == 25  # 5 * (3 + 2)
        assert len(episode) == 25
        
        # Test device movement (if applicable)
        cpu_episode = episode.to_device(torch.device('cpu'))
        assert cpu_episode.support_x.device.type == 'cpu'
        
        # Test indexing
        for i in range(len(episode)):
            data_i, label_i = episode[i]
            assert data_i.shape[0] == 8  # Feature dimension
            assert 0 <= label_i < 5  # Valid class range


if __name__ == "__main__":
    pytest.main([__file__])