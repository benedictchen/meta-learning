# TODO: EPISODE UTILITIES TESTING - Comprehensive test coverage
# TODO: Test create_episode_from_data functionality
# TODO: Test merge_episodes with various scenarios
# TODO: Test balance_episode with different class distributions
# TODO: Test augment_episode with custom augmentation functions
# TODO: Test split_episode with different ratios
# TODO: Test compute_episode_statistics accuracy

"""Tests for episode manipulation utilities."""

import pytest
import torch
from meta_learning.data_utils import (
    create_episode_from_data,
    merge_episodes,
    balance_episode,
    augment_episode,
    split_episode,
    compute_episode_statistics
)


class TestCreateEpisodeFromData:
    """Test create_episode_from_data function."""
    
    def test_basic_creation(self):
        """Test basic episode creation."""
        support_x = torch.randn(10, 32)
        support_y = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
        query_x = torch.randn(25, 32)
        query_y = torch.tensor([0]*5 + [1]*5 + [2]*5 + [3]*5 + [4]*5)
        
        episode = create_episode_from_data(support_x, support_y, query_x, query_y)
        
        assert episode.support_x.shape == (10, 32)
        assert episode.support_y.shape == (10,)
        assert episode.query_x.shape == (25, 32)
        assert episode.query_y.shape == (25,)
        assert episode.num_classes == 5
    
    def test_tensor_devices(self):
        """Test episode creation with different devices."""
        if torch.cuda.is_available():
            support_x = torch.randn(4, 16).cuda()
            support_y = torch.tensor([0, 0, 1, 1]).cuda()
            query_x = torch.randn(6, 16).cuda()
            query_y = torch.tensor([0, 0, 0, 1, 1, 1]).cuda()
            
            episode = create_episode_from_data(support_x, support_y, query_x, query_y)
            
            assert episode.support_x.device == support_x.device
            assert episode.support_y.device == support_y.device
    
    def test_empty_episode(self):
        """Test creation with minimal data."""
        support_x = torch.randn(2, 8)
        support_y = torch.tensor([0, 1])
        query_x = torch.randn(2, 8)
        query_y = torch.tensor([0, 1])
        
        episode = create_episode_from_data(support_x, support_y, query_x, query_y)
        
        assert episode.num_classes == 2
        assert len(episode.support_x) == 2
        assert len(episode.query_x) == 2


class TestMergeEpisodes:
    """Test merge_episodes function."""
    
    def setup_method(self):
        """Setup test episodes."""
        self.episode1 = create_episode_from_data(
            torch.randn(4, 16),
            torch.tensor([0, 0, 1, 1]),
            torch.randn(6, 16),
            torch.tensor([0, 0, 0, 1, 1, 1])
        )
        
        self.episode2 = create_episode_from_data(
            torch.randn(6, 16),
            torch.tensor([0, 0, 1, 1, 2, 2]),
            torch.randn(9, 16),
            torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])
        )
    
    def test_basic_merge(self):
        """Test basic episode merging."""
        merged = merge_episodes(self.episode1, self.episode2)
        
        # Check dimensions
        assert merged.support_x.shape == (10, 16)  # 4 + 6
        assert merged.query_x.shape == (15, 16)    # 6 + 9
        assert len(merged.support_y) == 10
        assert len(merged.query_y) == 15
    
    def test_single_episode_merge(self):
        """Test merging a single episode."""
        merged = merge_episodes(self.episode1)
        
        # Should be identical to original
        assert torch.equal(merged.support_x, self.episode1.support_x)
        assert torch.equal(merged.support_y, self.episode1.support_y)
        assert torch.equal(merged.query_x, self.episode1.query_x)
        assert torch.equal(merged.query_y, self.episode1.query_y)
    
    def test_empty_merge(self):
        """Test merging with no episodes."""
        with pytest.raises(ValueError, match="At least one episode must be provided"):
            merge_episodes()
    
    def test_multiple_episodes_merge(self):
        """Test merging multiple episodes."""
        episode3 = create_episode_from_data(
            torch.randn(2, 16),
            torch.tensor([0, 1]),
            torch.randn(4, 16),
            torch.tensor([0, 0, 1, 1])
        )
        
        merged = merge_episodes(self.episode1, self.episode2, episode3)
        
        # Check total dimensions
        assert merged.support_x.shape == (12, 16)  # 4 + 6 + 2
        assert merged.query_x.shape == (19, 16)    # 6 + 9 + 4


class TestBalanceEpisode:
    """Test balance_episode function."""
    
    def test_already_balanced_episode(self):
        """Test balancing an already balanced episode."""
        # Create balanced episode (2 shots per class)
        support_x = torch.randn(6, 16)
        support_y = torch.tensor([0, 0, 1, 1, 2, 2])
        query_x = torch.randn(9, 16)
        query_y = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])
        
        episode = create_episode_from_data(support_x, support_y, query_x, query_y)
        balanced = balance_episode(episode, target_shots_per_class=2)
        
        # Should maintain 2 shots per class
        for cls in [0, 1, 2]:
            assert (balanced.support_y == cls).sum() == 2
    
    def test_imbalanced_episode_subsampling(self):
        """Test balancing by subsampling majority classes."""
        # Create imbalanced episode (more shots for class 0)
        support_x = torch.randn(7, 16)
        support_y = torch.tensor([0, 0, 0, 0, 1, 2, 2])  # 4, 1, 2 shots
        query_x = torch.randn(6, 16)
        query_y = torch.tensor([0, 0, 1, 1, 2, 2])
        
        episode = create_episode_from_data(support_x, support_y, query_x, query_y)
        balanced = balance_episode(episode, target_shots_per_class=1)
        
        # Should have exactly 1 shot per class
        for cls in [0, 1, 2]:
            assert (balanced.support_y == cls).sum() == 1
    
    def test_imbalanced_episode_oversampling(self):
        """Test balancing by oversampling minority classes."""
        # Create episode with insufficient samples for some classes
        support_x = torch.randn(4, 16)
        support_y = torch.tensor([0, 0, 1, 2])  # 2, 1, 1 shots
        query_x = torch.randn(3, 16)
        query_y = torch.tensor([0, 1, 2])
        
        episode = create_episode_from_data(support_x, support_y, query_x, query_y)
        balanced = balance_episode(episode, target_shots_per_class=2)
        
        # Should have exactly 2 shots per class (with oversampling)
        for cls in [0, 1, 2]:
            assert (balanced.support_y == cls).sum() == 2
    
    def test_automatic_target_selection(self):
        """Test automatic selection of target shots per class."""
        support_x = torch.randn(8, 16)
        support_y = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2])  # 3, 2, 3 shots
        query_x = torch.randn(6, 16)
        query_y = torch.tensor([0, 0, 1, 1, 2, 2])
        
        episode = create_episode_from_data(support_x, support_y, query_x, query_y)
        balanced = balance_episode(episode)  # Should choose min = 2
        
        # Should balance to minimum class size (2)
        for cls in [0, 1, 2]:
            assert (balanced.support_y == cls).sum() == 2


class TestAugmentEpisode:
    """Test augment_episode function."""
    
    def test_default_augmentation(self):
        """Test episode augmentation with default noise."""
        support_x = torch.randn(4, 16)
        support_y = torch.tensor([0, 0, 1, 1])
        query_x = torch.randn(4, 16)
        query_y = torch.tensor([0, 0, 1, 1])
        
        episode = create_episode_from_data(support_x, support_y, query_x, query_y)
        augmented = augment_episode(episode)
        
        # Support set should be modified, query set unchanged
        assert not torch.equal(augmented.support_x, episode.support_x)
        assert torch.equal(augmented.query_x, episode.query_x)
        assert torch.equal(augmented.support_y, episode.support_y)
        assert torch.equal(augmented.query_y, episode.query_y)
    
    def test_custom_augmentation(self):
        """Test episode augmentation with custom function."""
        def double_augmentation(x):
            return x * 2
        
        support_x = torch.ones(2, 8)
        support_y = torch.tensor([0, 1])
        query_x = torch.ones(2, 8)
        query_y = torch.tensor([0, 1])
        
        episode = create_episode_from_data(support_x, support_y, query_x, query_y)
        augmented = augment_episode(episode, augmentation_fn=double_augmentation)
        
        # Support set should be doubled
        assert torch.allclose(augmented.support_x, episode.support_x * 2)
        # Query set should remain unchanged
        assert torch.equal(augmented.query_x, episode.query_x)
    
    def test_augmentation_preserves_shapes(self):
        """Test that augmentation preserves tensor shapes."""
        original_shape = (6, 32, 32, 3)  # Image-like data
        support_x = torch.randn(*original_shape)
        support_y = torch.tensor([0, 0, 1, 1, 2, 2])
        query_x = torch.randn(9, 32, 32, 3)
        query_y = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])
        
        episode = create_episode_from_data(support_x, support_y, query_x, query_y)
        augmented = augment_episode(episode)
        
        assert augmented.support_x.shape == original_shape
        assert augmented.query_x.shape == (9, 32, 32, 3)


class TestSplitEpisode:
    """Test split_episode function."""
    
    def test_default_split(self):
        """Test episode splitting with default ratio."""
        support_x = torch.randn(4, 16)
        support_y = torch.tensor([0, 0, 1, 1])
        query_x = torch.randn(10, 16)  # Even number for clean split
        query_y = torch.tensor([0]*5 + [1]*5)
        
        episode = create_episode_from_data(support_x, support_y, query_x, query_y)
        ep1, ep2 = split_episode(episode)
        
        # Check support sets are identical
        assert torch.equal(ep1.support_x, episode.support_x)
        assert torch.equal(ep2.support_x, episode.support_x)
        assert torch.equal(ep1.support_y, episode.support_y)
        assert torch.equal(ep2.support_y, episode.support_y)
        
        # Check query sets are split
        assert len(ep1.query_x) + len(ep2.query_x) == len(episode.query_x)
    
    def test_custom_split_ratio(self):
        """Test episode splitting with custom ratio."""
        support_x = torch.randn(2, 8)
        support_y = torch.tensor([0, 1])
        query_x = torch.randn(10, 8)
        query_y = torch.tensor([0]*5 + [1]*5)
        
        episode = create_episode_from_data(support_x, support_y, query_x, query_y)
        ep1, ep2 = split_episode(episode, query_ratio=0.3)
        
        # First episode should have ~30% of query samples
        expected_size1 = max(1, int(10 * 0.3))  # At least 1
        assert len(ep1.query_x) == expected_size1
        assert len(ep2.query_x) == 10 - expected_size1
    
    def test_minimal_query_split(self):
        """Test splitting with minimal query samples."""
        support_x = torch.randn(2, 4)
        support_y = torch.tensor([0, 1])
        query_x = torch.randn(2, 4)  # Minimal query set
        query_y = torch.tensor([0, 1])
        
        episode = create_episode_from_data(support_x, support_y, query_x, query_y)
        ep1, ep2 = split_episode(episode, query_ratio=0.5)
        
        # Should still create valid episodes
        assert len(ep1.query_x) >= 1
        assert len(ep2.query_x) >= 1
        assert len(ep1.query_x) + len(ep2.query_x) == 2


class TestComputeEpisodeStatistics:
    """Test compute_episode_statistics function."""
    
    def test_basic_statistics(self):
        """Test basic episode statistics computation."""
        support_x = torch.randn(6, 32, 32)
        support_y = torch.tensor([0, 0, 1, 1, 2, 2])
        query_x = torch.randn(9, 32, 32)
        query_y = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])
        
        episode = create_episode_from_data(support_x, support_y, query_x, query_y)
        stats = compute_episode_statistics(episode)
        
        # Check basic counts
        assert stats['n_support'] == 6
        assert stats['n_query'] == 9
        assert stats['n_support_classes'] == 3
        assert stats['n_query_classes'] == 3
        
        # Check class lists
        assert set(stats['support_classes']) == {0, 1, 2}
        assert set(stats['query_classes']) == {0, 1, 2}
        
        # Check shapes
        assert stats['data_shape'] == [32, 32]
        assert stats['support_shape'] == [6, 32, 32]
        assert stats['query_shape'] == [9, 32, 32]
    
    def test_mismatched_classes(self):
        """Test statistics with different support/query classes."""
        support_x = torch.randn(4, 16)
        support_y = torch.tensor([0, 0, 1, 1])
        query_x = torch.randn(6, 16)
        query_y = torch.tensor([0, 0, 1, 1, 1, 1])  # More class 1
        
        episode = create_episode_from_data(support_x, support_y, query_x, query_y)
        stats = compute_episode_statistics(episode)
        
        assert stats['n_support_classes'] == 2
        assert stats['n_query_classes'] == 2
        assert stats['support_classes'] == [0, 1]
        assert stats['query_classes'] == [0, 1]
    
    def test_single_class_episode(self):
        """Test statistics for single-class episode."""
        support_x = torch.randn(3, 8)
        support_y = torch.tensor([5, 5, 5])  # Only class 5
        query_x = torch.randn(3, 8)
        query_y = torch.tensor([5, 5, 5])
        
        episode = create_episode_from_data(support_x, support_y, query_x, query_y)
        stats = compute_episode_statistics(episode)
        
        assert stats['n_support_classes'] == 1
        assert stats['n_query_classes'] == 1
        assert stats['support_classes'] == [5]
        assert stats['query_classes'] == [5]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])