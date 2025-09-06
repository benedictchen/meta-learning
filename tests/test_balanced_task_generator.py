# TODO: BALANCED TASK GENERATOR TESTING - Comprehensive balancing tests  
# TODO: Test dataset analysis functionality
# TODO: Test class balancing strategies
# TODO: Test episode generation with different configurations
# TODO: Test imbalance ratio calculations
# TODO: Test similarity-based class filtering
# TODO: Test error handling for insufficient data

"""Tests for BalancedTaskGenerator class."""

import pytest
import torch
from unittest.mock import Mock, patch
from meta_learning.data_utils.iterators import BalancedTaskGenerator


class MockDataset:
    """Mock dataset for testing BalancedTaskGenerator."""
    
    def __init__(self, data_labels_pairs):
        """
        Args:
            data_labels_pairs: List of (data, label) tuples
        """
        self.data = data_labels_pairs
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def __iter__(self):
        return iter(self.data)


class TestBalancedTaskGenerator:
    """Test BalancedTaskGenerator class."""
    
    def test_init_basic(self):
        """Test basic initialization."""
        # Create balanced dataset (2 samples per class)
        dataset = MockDataset([
            (torch.randn(32), 0), (torch.randn(32), 0),
            (torch.randn(32), 1), (torch.randn(32), 1),
            (torch.randn(32), 2), (torch.randn(32), 2),
        ])
        
        generator = BalancedTaskGenerator(dataset, n_way=3, n_shot=1, n_query=1)
        
        assert generator.n_way == 3
        assert generator.n_shot == 1
        assert generator.n_query == 1
        assert generator.imbalance_ratio == 1.0  # Perfectly balanced
        assert len(generator.available_classes) == 3
        assert set(generator.available_classes) == {0, 1, 2}
    
    def test_init_imbalanced_dataset(self):
        """Test initialization with imbalanced dataset."""
        # Create imbalanced dataset (4, 2, 1 samples per class)
        dataset = MockDataset([
            (torch.randn(16), 0), (torch.randn(16), 0), 
            (torch.randn(16), 0), (torch.randn(16), 0),
            (torch.randn(16), 1), (torch.randn(16), 1),
            (torch.randn(16), 2),
        ])
        
        generator = BalancedTaskGenerator(dataset, n_way=2, n_shot=1, n_query=1)
        
        # Imbalance ratio should be max/min = 4/1 = 4.0
        assert generator.imbalance_ratio == 4.0
        assert generator.class_counts[0] == 4
        assert generator.class_counts[1] == 2
        assert generator.class_counts[2] == 1
    
    def test_init_with_custom_strategies(self):
        """Test initialization with custom balancing strategies."""
        dataset = MockDataset([
            (torch.randn(8), 0), (torch.randn(8), 1), (torch.randn(8), 2)
        ])
        
        custom_strategies = ['class', 'similarity', 'difficulty']
        generator = BalancedTaskGenerator(
            dataset, 
            n_way=2, 
            n_shot=1, 
            n_query=1,
            balance_strategies=custom_strategies
        )
        
        assert generator.balance_strategies == custom_strategies
        assert generator.similarity_threshold == 0.8  # default
    
    def test_analyze_dataset_functionality(self):
        """Test dataset analysis functionality."""
        dataset = MockDataset([
            (torch.randn(4), 0), (torch.randn(4), 0),
            (torch.randn(4), 1), 
            (torch.randn(4), 2), (torch.randn(4), 2), (torch.randn(4), 2),
        ])
        
        generator = BalancedTaskGenerator(dataset, n_way=3, n_shot=1, n_query=1)
        
        # Check class counts
        assert generator.class_counts[0] == 2
        assert generator.class_counts[1] == 1
        assert generator.class_counts[2] == 3
        
        # Check class features storage (should be limited)
        for class_id in generator.class_features:
            assert len(generator.class_features[class_id]) <= 5
        
        # Check imbalance ratio
        expected_ratio = 3 / 1  # max=3, min=1
        assert generator.imbalance_ratio == expected_ratio
    
    def test_generate_episode_basic(self):
        """Test basic episode generation."""
        # Create dataset with enough samples
        dataset = MockDataset([
            (torch.randn(16), 0), (torch.randn(16), 0), (torch.randn(16), 0),
            (torch.randn(16), 1), (torch.randn(16), 1), (torch.randn(16), 1),
            (torch.randn(16), 2), (torch.randn(16), 2), (torch.randn(16), 2),
        ])
        
        generator = BalancedTaskGenerator(dataset, n_way=3, n_shot=1, n_query=1)
        episode = generator.generate_episode(random_state=42)
        
        # Check episode structure
        assert episode.num_classes == 3
        assert len(episode.support_x) == 3  # 1 per class
        assert len(episode.query_x) == 3    # 1 per class
        assert episode.support_x.shape[1] == 16  # Feature dimension
        
        # Check label remapping (should be 0, 1, 2)
        support_labels = sorted(episode.support_y.tolist())
        query_labels = sorted(episode.query_y.tolist())
        assert support_labels == [0, 1, 2]
        assert query_labels == [0, 1, 2]
    
    def test_generate_episode_reproducible(self):
        """Test that episode generation is reproducible with random state."""
        dataset = MockDataset([
            (torch.randn(8), i % 3) for i in range(12)  # 4 samples per class
        ])
        
        generator = BalancedTaskGenerator(dataset, n_way=3, n_shot=1, n_query=2)
        
        # Generate same episode twice with same random state
        episode1 = generator.generate_episode(random_state=42)
        episode2 = generator.generate_episode(random_state=42)
        
        # Should be identical
        assert torch.equal(episode1.support_x, episode2.support_x)
        assert torch.equal(episode1.support_y, episode2.support_y)
        assert torch.equal(episode1.query_x, episode2.query_x)
        assert torch.equal(episode1.query_y, episode2.query_y)
    
    def test_generate_episode_insufficient_samples(self):
        """Test episode generation with insufficient samples per class."""
        # Some classes have fewer samples than needed
        dataset = MockDataset([
            (torch.randn(8), 0),  # Class 0: only 1 sample
            (torch.randn(8), 1), (torch.randn(8), 1),  # Class 1: 2 samples
            (torch.randn(8), 2), (torch.randn(8), 2), (torch.randn(8), 2),  # Class 2: 3 samples
        ])
        
        generator = BalancedTaskGenerator(dataset, n_way=3, n_shot=2, n_query=1)
        episode = generator.generate_episode(random_state=42)
        
        # Should still generate episode (with oversampling for insufficient classes)
        assert episode.num_classes == 3
        assert len(episode.support_x) == 6  # 2 per class
        assert len(episode.query_x) == 3   # 1 per class
    
    def test_select_balanced_classes_basic(self):
        """Test basic class selection."""
        dataset = MockDataset([
            (torch.randn(4), 0), (torch.randn(4), 0),
            (torch.randn(4), 1),
            (torch.randn(4), 2), (torch.randn(4), 2), (torch.randn(4), 2),
        ])
        
        generator = BalancedTaskGenerator(
            dataset, n_way=2, n_shot=1, n_query=1,
            balance_strategies=['class']
        )
        
        # Should prefer underrepresented classes
        selected_classes = generator._select_balanced_classes()
        assert len(selected_classes) == 2
        
        # Class 1 (count=1) should be preferred over class 2 (count=3)
        assert 1 in selected_classes  # Most underrepresented
    
    def test_select_balanced_classes_no_class_strategy(self):
        """Test class selection without class balancing strategy."""
        dataset = MockDataset([
            (torch.randn(4), i % 5) for i in range(15)  # 3 samples per class
        ])
        
        generator = BalancedTaskGenerator(
            dataset, n_way=3, n_shot=1, n_query=1,
            balance_strategies=[]  # No specific strategies
        )
        
        selected_classes = generator._select_balanced_classes()
        assert len(selected_classes) == 3
        assert all(cls in generator.available_classes for cls in selected_classes)
    
    def test_select_balanced_classes_insufficient_classes(self):
        """Test class selection when dataset has fewer classes than needed."""
        dataset = MockDataset([
            (torch.randn(4), 0), (torch.randn(4), 0),
            (torch.randn(4), 1), (torch.randn(4), 1),
        ])
        
        generator = BalancedTaskGenerator(dataset, n_way=5, n_shot=1, n_query=1)
        
        with pytest.raises(ValueError, match="Dataset has only 2 classes, need 5"):
            generator._select_balanced_classes()
    
    def test_filter_similar_classes(self):
        """Test similarity-based class filtering (simplified version)."""
        dataset = MockDataset([
            (torch.randn(4), i) for i in range(10)
        ])
        
        generator = BalancedTaskGenerator(dataset, n_way=3, n_shot=1, n_query=1)
        
        candidates = [0, 1, 2, 3, 4]
        selected_classes = [0, 2]
        
        # Current implementation returns all candidates (simplified)
        filtered = generator._filter_similar_classes(candidates, selected_classes)
        assert filtered == candidates
    
    def test_tensor_label_handling(self):
        """Test handling of tensor vs scalar labels."""
        # Mix of tensor and scalar labels
        dataset = MockDataset([
            (torch.randn(8), torch.tensor(0)), (torch.randn(8), 0),
            (torch.randn(8), torch.tensor(1)), (torch.randn(8), 1),
            (torch.randn(8), torch.tensor(2)), (torch.randn(8), 2),
        ])
        
        generator = BalancedTaskGenerator(dataset, n_way=3, n_shot=1, n_query=1)
        
        # Should handle both formats correctly
        assert len(generator.available_classes) == 3
        assert set(generator.available_classes) == {0, 1, 2}
        
        # Episode generation should work
        episode = generator.generate_episode(random_state=42)
        assert episode.num_classes == 3
    
    def test_empty_dataset(self):
        """Test behavior with empty dataset."""
        dataset = MockDataset([])
        
        # Empty dataset should be handled gracefully
        generator = BalancedTaskGenerator(dataset, n_way=1, n_shot=1, n_query=1)
        assert len(generator.class_counts) == 0
    
    def test_single_class_dataset(self):
        """Test behavior with single-class dataset."""
        dataset = MockDataset([
            (torch.randn(8), 0), (torch.randn(8), 0), (torch.randn(8), 0)
        ])
        
        generator = BalancedTaskGenerator(dataset, n_way=1, n_shot=1, n_query=1)
        
        assert len(generator.available_classes) == 1
        assert generator.imbalance_ratio == 1.0
        
        episode = generator.generate_episode()
        assert episode.num_classes == 1
        assert len(episode.support_x) == 1
        assert len(episode.query_x) == 1
    
    def test_large_n_way_n_shot(self):
        """Test with large n_way and n_shot requirements."""
        # Create dataset with many samples per class
        data_per_class = 10
        n_classes = 8
        dataset = MockDataset([
            (torch.randn(16), class_id)
            for class_id in range(n_classes)
            for _ in range(data_per_class)
        ])
        
        generator = BalancedTaskGenerator(dataset, n_way=5, n_shot=3, n_query=4)
        episode = generator.generate_episode(random_state=42)
        
        assert episode.num_classes == 5
        assert len(episode.support_x) == 15  # 5 classes * 3 shots
        assert len(episode.query_x) == 20   # 5 classes * 4 queries
    
    def test_class_counts_accuracy(self):
        """Test accuracy of class counting."""
        # Specific class distribution
        dataset = MockDataset([
            (torch.randn(4), 0),  # 1 sample
            (torch.randn(4), 1), (torch.randn(4), 1), (torch.randn(4), 1),  # 3 samples
            (torch.randn(4), 2), (torch.randn(4), 2),  # 2 samples
            (torch.randn(4), 5), (torch.randn(4), 5), (torch.randn(4), 5), 
            (torch.randn(4), 5), (torch.randn(4), 5),  # 5 samples
        ])
        
        generator = BalancedTaskGenerator(dataset, n_way=3, n_shot=1, n_query=1)
        
        assert generator.class_counts[0] == 1
        assert generator.class_counts[1] == 3
        assert generator.class_counts[2] == 2
        assert generator.class_counts[5] == 5
        
        # Imbalance ratio should be 5/1 = 5.0
        assert generator.imbalance_ratio == 5.0
    
    def test_feature_storage_limit(self):
        """Test that feature storage is limited to prevent memory issues."""
        # Create many samples for one class
        dataset = MockDataset([
            (torch.randn(32), 0) for _ in range(20)  # 20 samples, class 0
        ] + [
            (torch.randn(32), 1) for _ in range(5)   # 5 samples, class 1
        ])
        
        generator = BalancedTaskGenerator(dataset, n_way=2, n_shot=1, n_query=1)
        
        # Feature storage should be limited to 5 per class
        assert len(generator.class_features[0]) <= 5
        assert len(generator.class_features[1]) <= 5
        
        # But class counts should be accurate
        assert generator.class_counts[0] == 20
        assert generator.class_counts[1] == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])