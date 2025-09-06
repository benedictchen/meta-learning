"""
Comprehensive tests for task partitioning and utility functions.

Tests the core functions for partitioning classification tasks into support
and query sets, which is essential for few-shot learning episodes.
"""

import pytest
import torch
import numpy as np
from typing import Tuple

# Import the functions we're testing
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from meta_learning.core.task_utils import partition_task, remap_labels


class TestPartitionTask:
    """Test partition_task function"""
    
    def test_basic_partition_5way_1shot(self):
        """Test basic 5-way 1-shot partition"""
        # Create 5-way classification data with 2 samples per class (1 support, 1 query)
        data = torch.randn(10, 64)  # 10 samples, 64 features
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])  # 2 samples per class
        
        (support_x, support_y), (query_x, query_y) = partition_task(data, labels, shots=1)
        
        # Check shapes
        assert support_x.shape == (5, 64), f"Expected support shape (5, 64), got {support_x.shape}"
        assert support_y.shape == (5,), f"Expected support labels shape (5,), got {support_y.shape}"
        assert query_x.shape == (5, 64), f"Expected query shape (5, 64), got {query_x.shape}"
        assert query_y.shape == (5,), f"Expected query labels shape (5,), got {query_y.shape}"
        
        # Check that each class appears exactly once in support and query
        unique_support_labels = torch.unique(support_y)
        unique_query_labels = torch.unique(query_y)
        assert len(unique_support_labels) == 5, f"Expected 5 classes in support, got {len(unique_support_labels)}"
        assert len(unique_query_labels) == 5, f"Expected 5 classes in query, got {len(unique_query_labels)}"
        
        # Check that support and query have same class distribution
        assert torch.equal(torch.sort(support_y)[0], torch.sort(query_y)[0]), "Support and query should have same classes"
    
    def test_basic_partition_3way_5shot(self):
        """Test 3-way 5-shot partition"""
        # Create 3-way classification data with 8 samples per class (5 support, 3 query)
        data = torch.randn(24, 32)  # 24 samples, 32 features
        labels = torch.tensor([0]*8 + [1]*8 + [2]*8)  # 8 samples per class
        
        (support_x, support_y), (query_x, query_y) = partition_task(data, labels, shots=5)
        
        # Check shapes
        assert support_x.shape == (15, 32), f"Expected support shape (15, 32), got {support_x.shape}"
        assert support_y.shape == (15,), f"Expected support labels shape (15,), got {support_y.shape}"
        assert query_x.shape == (9, 32), f"Expected query shape (9, 32), got {query_x.shape}"
        assert query_y.shape == (9,), f"Expected query labels shape (9,), got {query_y.shape}"
        
        # Check class distribution in support (5 samples per class)
        for class_id in [0, 1, 2]:
            support_count = (support_y == class_id).sum().item()
            query_count = (query_y == class_id).sum().item()
            assert support_count == 5, f"Expected 5 support samples for class {class_id}, got {support_count}"
            assert query_count == 3, f"Expected 3 query samples for class {class_id}, got {query_count}"
    
    def test_data_integrity_preservation(self):
        """Test that original data is preserved correctly in partitions"""
        # Create specific data pattern for tracking
        data = torch.arange(12).float().reshape(6, 2)  # [[0,1], [2,3], [4,5], [6,7], [8,9], [10,11]]
        labels = torch.tensor([0, 1, 0, 1, 2, 2])  # 2 samples each for classes 0, 1, 2
        
        (support_x, support_y), (query_x, query_y) = partition_task(data, labels, shots=1)
        
        # Check that all original data appears somewhere
        all_partition_data = torch.cat([support_x, query_x], dim=0)
        all_partition_labels = torch.cat([support_y, query_y], dim=0)
        
        # Sort to compare
        original_sorted_idx = torch.argsort(data.sum(dim=1))
        partition_sorted_idx = torch.argsort(all_partition_data.sum(dim=1))
        
        # Data should be preserved (though reordered)
        assert len(all_partition_data) == len(data), "Data length should be preserved"
        assert torch.equal(torch.sort(data.sum(dim=1))[0], torch.sort(all_partition_data.sum(dim=1))[0]), "All data values should be preserved"
    
    def test_non_consecutive_labels(self):
        """Test partition with non-consecutive label values"""
        data = torch.randn(8, 10)
        labels = torch.tensor([10, 10, 25, 25, 99, 99, 3, 3])  # Non-consecutive labels
        
        (support_x, support_y), (query_x, query_y) = partition_task(data, labels, shots=1)
        
        # Check shapes
        assert support_x.shape == (4, 10), f"Expected support shape (4, 10), got {support_x.shape}"
        assert query_x.shape == (4, 10), f"Expected query shape (4, 10), got {query_x.shape}"
        
        # Check that original label values are preserved
        all_partition_labels = torch.cat([support_y, query_y])
        original_unique_labels = torch.unique(labels)
        partition_unique_labels = torch.unique(all_partition_labels)
        
        assert torch.equal(torch.sort(original_unique_labels)[0], torch.sort(partition_unique_labels)[0]), "Original label values should be preserved"
    
    def test_single_class_single_shot(self):
        """Test edge case with single class"""
        data = torch.randn(2, 5)
        labels = torch.tensor([7, 7])  # Same class
        
        (support_x, support_y), (query_x, query_y) = partition_task(data, labels, shots=1)
        
        assert support_x.shape == (1, 5), f"Expected support shape (1, 5), got {support_x.shape}"
        assert query_x.shape == (1, 5), f"Expected query shape (1, 5), got {query_x.shape}"
        assert support_y.item() == 7, f"Expected support label 7, got {support_y.item()}"
        assert query_y.item() == 7, f"Expected query label 7, got {query_y.item()}"
    
    def test_device_consistency(self):
        """Test that partition maintains device consistency"""
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        
        data = torch.randn(6, 8).to(device)
        labels = torch.tensor([0, 0, 1, 1, 2, 2]).to(device)
        
        (support_x, support_y), (query_x, query_y) = partition_task(data, labels, shots=1)
        
        assert support_x.device == device, f"Support data should be on {device}"
        assert support_y.device == device, f"Support labels should be on {device}"
        assert query_x.device == device, f"Query data should be on {device}"
        assert query_y.device == device, f"Query labels should be on {device}"
    
    def test_dtype_preservation(self):
        """Test that partition preserves data types"""
        data = torch.randn(4, 3, dtype=torch.float64)
        labels = torch.tensor([0, 0, 1, 1], dtype=torch.int32)
        
        (support_x, support_y), (query_x, query_y) = partition_task(data, labels, shots=1)
        
        assert support_x.dtype == torch.float64, f"Expected support dtype float64, got {support_x.dtype}"
        assert support_y.dtype == torch.int32, f"Expected support labels dtype int32, got {support_y.dtype}"
        assert query_x.dtype == torch.float64, f"Expected query dtype float64, got {query_x.dtype}"
        assert query_y.dtype == torch.int32, f"Expected query labels dtype int32, got {query_y.dtype}"
    
    def test_multidimensional_data(self):
        """Test partition with multidimensional data (e.g., images)"""
        # Simulate 4 grayscale images of size 8x8
        data = torch.randn(4, 1, 8, 8)  # NCHW format
        labels = torch.tensor([0, 0, 1, 1])
        
        (support_x, support_y), (query_x, query_y) = partition_task(data, labels, shots=1)
        
        assert support_x.shape == (2, 1, 8, 8), f"Expected support shape (2, 1, 8, 8), got {support_x.shape}"
        assert query_x.shape == (2, 1, 8, 8), f"Expected query shape (2, 1, 8, 8), got {query_x.shape}"
        assert support_y.shape == (2,), f"Expected support labels shape (2,), got {support_y.shape}"
        assert query_y.shape == (2,), f"Expected query labels shape (2,), got {query_y.shape}"
    
    def test_error_cases(self):
        """Test error cases and assertions"""
        data = torch.randn(6, 4)
        labels = torch.tensor([0, 0, 1, 1, 2, 2])
        
        # Test mismatched data and label lengths
        with pytest.raises(AssertionError):
            partition_task(torch.randn(5, 4), labels, shots=1)  # Data has 5 samples, labels have 6
        
        # Test insufficient samples per class
        with pytest.raises(AssertionError):
            partition_task(data, labels, shots=3)  # Only 2 samples per class, but asking for 3 shots
        
        # Test unequal samples per class (not supported)
        unequal_labels = torch.tensor([0, 0, 0, 1, 1, 2])  # 3 samples of class 0, 2 of class 1, 1 of class 2
        with pytest.raises(AssertionError):
            partition_task(torch.randn(6, 4), unequal_labels, shots=1)


class TestRemapLabels:
    """Test remap_labels function"""
    
    def test_basic_remapping(self):
        """Test basic label remapping to consecutive integers"""
        labels = torch.tensor([5, 3, 5, 3, 7])
        remapped = remap_labels(labels)
        
        # Should be remapped to [1, 0, 1, 0, 2] (sorted order: 3->0, 5->1, 7->2)
        expected = torch.tensor([1, 0, 1, 0, 2])
        assert torch.equal(remapped, expected), f"Expected {expected}, got {remapped}"
    
    def test_already_consecutive(self):
        """Test that consecutive labels remain unchanged"""
        labels = torch.tensor([0, 1, 2, 0, 1, 2])
        remapped = remap_labels(labels)
        
        expected = torch.tensor([0, 1, 2, 0, 1, 2])
        assert torch.equal(remapped, expected), f"Expected {expected}, got {remapped}"
    
    def test_single_class(self):
        """Test remapping with single class"""
        labels = torch.tensor([99, 99, 99])
        remapped = remap_labels(labels)
        
        expected = torch.tensor([0, 0, 0])
        assert torch.equal(remapped, expected), f"Expected {expected}, got {remapped}"
    
    def test_negative_labels(self):
        """Test remapping with negative labels"""
        labels = torch.tensor([-5, 10, -5, 0, 10])
        remapped = remap_labels(labels)
        
        # Sorted order: -5->0, 0->1, 10->2
        expected = torch.tensor([0, 2, 0, 1, 2])
        assert torch.equal(remapped, expected), f"Expected {expected}, got {remapped}"
    
    def test_large_label_values(self):
        """Test remapping with large label values"""
        labels = torch.tensor([1000000, 500000, 1000000])
        remapped = remap_labels(labels)
        
        expected = torch.tensor([1, 0, 1])  # 500000->0, 1000000->1
        assert torch.equal(remapped, expected), f"Expected {expected}, got {remapped}"
    
    def test_dtype_preservation(self):
        """Test that output dtype matches input dtype"""
        labels = torch.tensor([10, 20, 10], dtype=torch.int32)
        remapped = remap_labels(labels)
        
        assert remapped.dtype == torch.int32, f"Expected dtype int32, got {remapped.dtype}"
        assert torch.equal(remapped, torch.tensor([0, 1, 0], dtype=torch.int32))
    
    def test_device_consistency(self):
        """Test that remapping maintains device consistency"""
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        
        labels = torch.tensor([7, 3, 7, 1]).to(device)
        remapped = remap_labels(labels)
        
        assert remapped.device == device, f"Remapped labels should be on {device}"
        # Check values: sorted order 1->0, 3->1, 7->2
        expected = torch.tensor([2, 1, 2, 0]).to(device)
        assert torch.equal(remapped, expected), f"Expected {expected}, got {remapped}"
    
    def test_empty_tensor(self):
        """Test remapping empty tensor"""
        labels = torch.tensor([], dtype=torch.long)
        remapped = remap_labels(labels)
        
        assert remapped.shape == (0,), f"Expected empty tensor, got shape {remapped.shape}"
        assert remapped.dtype == torch.long, f"Expected dtype long, got {remapped.dtype}"
    
    def test_maintains_class_relationships(self):
        """Test that remapping maintains relative class relationships"""
        labels = torch.tensor([100, 50, 100, 200, 50, 200, 100])
        remapped = remap_labels(labels)
        
        # Original class groupings should be preserved
        class_100_mask = labels == 100
        class_50_mask = labels == 50  
        class_200_mask = labels == 200
        
        # All instances of original class should map to same new class
        remapped_100_values = remapped[class_100_mask].unique()
        remapped_50_values = remapped[class_50_mask].unique()
        remapped_200_values = remapped[class_200_mask].unique()
        
        assert len(remapped_100_values) == 1, "All instances of class 100 should map to same value"
        assert len(remapped_50_values) == 1, "All instances of class 50 should map to same value"
        assert len(remapped_200_values) == 1, "All instances of class 200 should map to same value"
        
        # Verify the mapping is correct (50->0, 100->1, 200->2 due to sorting)
        expected = torch.tensor([1, 0, 1, 2, 0, 2, 1])
        assert torch.equal(remapped, expected), f"Expected {expected}, got {remapped}"


class TestIntegrationTaskUtils:
    """Integration tests combining partition_task and remap_labels"""
    
    def test_partition_with_remapping(self):
        """Test typical workflow: partition then remap labels"""
        # Create data with non-consecutive labels
        data = torch.randn(8, 16)
        labels = torch.tensor([10, 10, 50, 50, 25, 25, 99, 99])
        
        # Partition first
        (support_x, support_y), (query_x, query_y) = partition_task(data, labels, shots=1)
        
        # Combine support and query for remapping
        all_labels = torch.cat([support_y, query_y])
        remapped_all = remap_labels(all_labels)
        
        # Split back
        n_support = len(support_y)
        remapped_support = remapped_all[:n_support]
        remapped_query = remapped_all[n_support:]
        
        # Verify remapping worked correctly
        assert len(remapped_support.unique()) == 4, "Should have 4 classes in support"
        assert len(remapped_query.unique()) == 4, "Should have 4 classes in query"
        assert torch.equal(torch.sort(remapped_support.unique())[0], torch.arange(4)), "Support should have classes 0-3"
        assert torch.equal(torch.sort(remapped_query.unique())[0], torch.arange(4)), "Query should have classes 0-3"
    
    def test_episode_creation_workflow(self):
        """Test complete episode creation workflow"""
        # Simulate creating a 5-way 2-shot episode
        n_classes = 5
        shots_per_class = 2
        query_per_class = 3
        feature_dim = 64
        
        # Create balanced dataset
        total_per_class = shots_per_class + query_per_class
        data = torch.randn(n_classes * total_per_class, feature_dim)
        labels = torch.repeat_interleave(torch.arange(n_classes), total_per_class)
        
        # Shuffle to simulate real dataset
        perm = torch.randperm(len(data))
        data = data[perm]
        labels = labels[perm]
        
        # Partition into support and query
        (support_x, support_y), (query_x, query_y) = partition_task(data, labels, shots=shots_per_class)
        
        # Verify episode structure
        assert support_x.shape == (n_classes * shots_per_class, feature_dim)
        assert query_x.shape == (n_classes * query_per_class, feature_dim)
        
        # Verify class distribution
        for class_id in range(n_classes):
            support_count = (support_y == class_id).sum().item()
            query_count = (query_y == class_id).sum().item()
            assert support_count == shots_per_class, f"Support should have {shots_per_class} samples for class {class_id}"
            assert query_count == query_per_class, f"Query should have {query_per_class} samples for class {class_id}"
    
    def test_memory_efficiency(self):
        """Test that functions work with larger tensors without memory issues"""
        # Create larger data to test memory efficiency
        n_samples = 1000
        feature_dim = 512
        n_classes = 10
        shots = 5
        
        data = torch.randn(n_samples, feature_dim)
        labels = torch.randint(0, n_classes, (n_samples,))
        
        # Ensure balanced classes for testing
        balanced_data = []
        balanced_labels = []
        samples_per_class = n_samples // n_classes
        
        for class_id in range(n_classes):
            class_data = torch.randn(samples_per_class, feature_dim)
            class_labels = torch.full((samples_per_class,), class_id)
            balanced_data.append(class_data)
            balanced_labels.append(class_labels)
        
        data = torch.cat(balanced_data, dim=0)
        labels = torch.cat(balanced_labels, dim=0)
        
        # Should work without memory issues
        (support_x, support_y), (query_x, query_y) = partition_task(data, labels, shots=shots)
        
        assert support_x.shape == (n_classes * shots, feature_dim)
        expected_query_samples = n_classes * (samples_per_class - shots)
        assert query_x.shape == (expected_query_samples, feature_dim)
    
    def test_reproducibility(self):
        """Test that results are deterministic given same input"""
        data = torch.randn(12, 8)
        labels = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
        
        # Run partition multiple times
        results1 = partition_task(data, labels, shots=2)
        results2 = partition_task(data, labels, shots=2)
        
        (support_x1, support_y1), (query_x1, query_y1) = results1
        (support_x2, support_y2), (query_x2, query_y2) = results2
        
        # Results should be identical
        assert torch.equal(support_x1, support_x2), "Support data should be deterministic"
        assert torch.equal(support_y1, support_y2), "Support labels should be deterministic"
        assert torch.equal(query_x1, query_x2), "Query data should be deterministic"
        assert torch.equal(query_y1, query_y2), "Query labels should be deterministic"


if __name__ == "__main__":
    # Run tests if called directly
    pytest.main([__file__, "-v"])