"""
Comprehensive tests for episode contract validation.

Tests all aspects of episode validation including parameter consistency,
tensor shapes, label ranges, class distributions, and API compliance.
"""

import torch
import pytest
from typing import List

from meta_learning.meta_learning_modules.episode_contract import (
    EpisodeContract, EpisodeValidationLevel, create_episode_contract,
    validate_episode_batch
)


class TestEpisodeContract:
    """Test suite for EpisodeContract validation."""
    
    @pytest.fixture
    def valid_episode_data(self):
        """Create valid episode data for testing."""
        n_way, k_shot, m_query = 5, 3, 2
        
        support_x = torch.randn(n_way * k_shot, 128)
        support_y = torch.repeat_interleave(torch.arange(n_way), k_shot)
        query_x = torch.randn(n_way * m_query, 128)
        query_y = torch.repeat_interleave(torch.arange(n_way), m_query)
        
        return n_way, k_shot, m_query, support_x, support_y, query_x, query_y
    
    def test_valid_episode_creation(self, valid_episode_data):
        """Test creation of valid episode contract."""
        n_way, k_shot, m_query, support_x, support_y, query_x, query_y = valid_episode_data
        
        episode = create_episode_contract(
            n_way, k_shot, m_query,
            support_x, support_y, query_x, query_y,
            episode_id="test_valid"
        )
        
        assert episode.n_way == n_way
        assert episode.k_shot == k_shot
        assert episode.m_query == m_query
        assert episode._validated == True
        assert len(episode._validation_errors) == 0
        assert episode.episode_id == "test_valid"
    
    def test_parameter_validation(self, valid_episode_data):
        """Test parameter validation logic."""
        n_way, k_shot, m_query, support_x, support_y, query_x, query_y = valid_episode_data
        
        # Test negative parameters
        with pytest.raises(ValueError, match="n_way must be positive"):
            create_episode_contract(-1, k_shot, m_query, support_x, support_y, query_x, query_y)
        
        with pytest.raises(ValueError, match="k_shot must be positive"):
            create_episode_contract(n_way, -1, m_query, support_x, support_y, query_x, query_y)
            
        with pytest.raises(ValueError, match="m_query must be positive"):
            create_episode_contract(n_way, k_shot, -1, support_x, support_y, query_x, query_y)
    
    def test_tensor_shape_validation(self, valid_episode_data):
        """Test tensor shape validation."""
        n_way, k_shot, m_query, support_x, support_y, query_x, query_y = valid_episode_data
        
        # Test wrong support batch size
        bad_support_x = torch.randn(10, 128)  # Should be n_way * k_shot = 15
        with pytest.raises(ValueError, match="support_x batch size"):
            create_episode_contract(n_way, k_shot, m_query, bad_support_x, support_y, query_x, query_y)
        
        # Test wrong query batch size
        bad_query_x = torch.randn(5, 128)  # Should be n_way * m_query = 10
        with pytest.raises(ValueError, match="query_x batch size"):
            create_episode_contract(n_way, k_shot, m_query, support_x, support_y, bad_query_x, query_y)
        
        # Test feature dimension mismatch
        bad_query_x = torch.randn(n_way * m_query, 64)  # Different feature dim
        with pytest.raises(ValueError, match="different feature shapes"):
            create_episode_contract(n_way, k_shot, m_query, support_x, support_y, bad_query_x, query_y)
    
    def test_label_consistency_validation(self, valid_episode_data):
        """Test label consistency validation."""
        n_way, k_shot, m_query, support_x, support_y, query_x, query_y = valid_episode_data
        
        # Test wrong label range
        bad_support_y = torch.tensor([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5])  # [1,5] not [0,4]
        with pytest.raises(ValueError, match="expected range"):
            create_episode_contract(n_way, k_shot, m_query, support_x, bad_support_y, query_x, query_y)
        
        # Test missing classes
        incomplete_support_y = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0, 1, 1, 1])  # Missing classes 3,4
        with pytest.raises(ValueError, match="expected range"):
            create_episode_contract(n_way, k_shot, m_query, support_x, incomplete_support_y, query_x, query_y)
        
        # Test support/query class mismatch
        mismatched_query_y = torch.repeat_interleave(torch.arange(3), m_query)  # Only 3 classes
        with pytest.raises(ValueError, match="different classes"):
            create_episode_contract(n_way, k_shot, m_query, support_x, support_y, query_x, mismatched_query_y)
    
    def test_class_distribution_validation(self, valid_episode_data):
        """Test balanced class distribution validation."""
        n_way, k_shot, m_query, support_x, support_y, query_x, query_y = valid_episode_data
        
        # Test unbalanced support set
        unbalanced_support_y = torch.tensor([0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4])  # 4 vs 3 examples
        with pytest.raises(ValueError, match="support set is unbalanced"):
            create_episode_contract(n_way, k_shot, m_query, support_x, unbalanced_support_y, query_x, query_y)
        
        # Test unbalanced query set
        unbalanced_query_y = torch.tensor([0, 0, 0, 1, 1, 2, 2, 3, 3, 4])  # 3, 2, 2, 2, 1 examples
        with pytest.raises(ValueError, match="query set is unbalanced"):
            create_episode_contract(n_way, k_shot, m_query, support_x, support_y, query_x, unbalanced_query_y)
    
    def test_tensor_properties_validation(self, valid_episode_data):
        """Test tensor properties validation."""
        n_way, k_shot, m_query, support_x, support_y, query_x, query_y = valid_episode_data
        
        # Test NaN values
        nan_support_x = support_x.clone()
        nan_support_x[0, 0] = float('nan')
        with pytest.raises(ValueError, match="support_x contains NaN"):
            create_episode_contract(n_way, k_shot, m_query, nan_support_x, support_y, query_x, query_y)
        
        # Test infinite values
        inf_query_x = query_x.clone()
        inf_query_x[0, 0] = float('inf')
        with pytest.raises(ValueError, match="query_x contains infinite"):
            create_episode_contract(n_way, k_shot, m_query, support_x, support_y, inf_query_x, query_y)
        
        # Test wrong label dtypes
        float_support_y = support_y.float()
        with pytest.raises(ValueError, match="support_y must be integer"):
            create_episode_contract(n_way, k_shot, m_query, support_x, float_support_y, query_x, query_y)
    
    def test_validation_levels(self, valid_episode_data):
        """Test different validation levels."""
        n_way, k_shot, m_query, support_x, support_y, query_x, query_y = valid_episode_data
        
        # Test lenient mode - should warn but not raise
        bad_support_x = torch.randn(10, 128)  # Wrong size
        
        episode = create_episode_contract(
            n_way, k_shot, m_query, bad_support_x, support_y, query_x, query_y,
            validation_level=EpisodeValidationLevel.LENIENT
        )
        
        # Should have validation errors but be created
        assert len(episode._validation_errors) > 0
        assert episode._validated == True
        
        # Test disabled validation
        episode_disabled = create_episode_contract(
            n_way, k_shot, m_query, bad_support_x, support_y, query_x, query_y,
            validation_level=EpisodeValidationLevel.DISABLED
        )
        
        assert len(episode_disabled._validation_errors) == 0
        assert episode_disabled._validated == True
    
    def test_from_raw_data(self):
        """Test creating episode from raw data with parameter inference."""
        # Create balanced data
        n_way, k_shot, m_query = 3, 4, 2
        
        support_x = torch.randn(n_way * k_shot, 64)
        support_y = torch.repeat_interleave(torch.arange(n_way), k_shot)
        query_x = torch.randn(n_way * m_query, 64)
        query_y = torch.repeat_interleave(torch.arange(n_way), m_query)
        
        episode = EpisodeContract.from_raw_data(
            support_x, support_y, query_x, query_y,
            episode_id="inferred"
        )
        
        assert episode.n_way == n_way
        assert episode.k_shot == k_shot
        assert episode.m_query == m_query
        assert episode.episode_id == "inferred"
    
    def test_prediction_validation(self, valid_episode_data):
        """Test prediction output validation."""
        n_way, k_shot, m_query, support_x, support_y, query_x, query_y = valid_episode_data
        
        episode = create_episode_contract(
            n_way, k_shot, m_query,
            support_x, support_y, query_x, query_y
        )
        
        # Test valid logits
        valid_logits = torch.randn(n_way * m_query, n_way)
        assert episode.validate_prediction_output(valid_logits) == True
        
        # Test valid class predictions
        valid_classes = torch.randint(0, n_way, (n_way * m_query,))
        assert episode.validate_prediction_output(valid_classes) == True
        
        # Test wrong batch size
        wrong_batch_logits = torch.randn(5, n_way)  # Should be 10
        with pytest.raises(ValueError, match="batch size"):
            episode.validate_prediction_output(wrong_batch_logits)
        
        # Test wrong class dimension
        wrong_class_logits = torch.randn(n_way * m_query, 3)  # Should be n_way=5
        with pytest.raises(ValueError, match="class dimension"):
            episode.validate_prediction_output(wrong_class_logits)
        
        # Test invalid class predictions
        invalid_classes = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])  # Class 5+ invalid
        with pytest.raises(ValueError, match="outside valid range"):
            episode.validate_prediction_output(invalid_classes)
    
    def test_device_handling(self, valid_episode_data):
        """Test device handling and device consistency."""
        n_way, k_shot, m_query, support_x, support_y, query_x, query_y = valid_episode_data
        
        episode = create_episode_contract(
            n_way, k_shot, m_query,
            support_x, support_y, query_x, query_y
        )
        
        # Test to_device
        if torch.cuda.is_available():
            cuda_episode = episode.to_device(torch.device('cuda'))
            assert cuda_episode.support_x.device.type == 'cuda'
            assert cuda_episode.query_x.device.type == 'cuda'
        
        # Test device mismatch detection
        if torch.cuda.is_available():
            cuda_support_x = support_x.cuda()
            with pytest.raises(ValueError, match="different devices"):
                create_episode_contract(
                    n_way, k_shot, m_query,
                    cuda_support_x, support_y, query_x, query_y  # query_x on CPU
                )
    
    def test_episode_summary(self, valid_episode_data):
        """Test episode summary generation."""
        n_way, k_shot, m_query, support_x, support_y, query_x, query_y = valid_episode_data
        
        episode = create_episode_contract(
            n_way, k_shot, m_query,
            support_x, support_y, query_x, query_y,
            episode_id="summary_test"
        )
        episode.original_classes = [10, 11, 12, 13, 14]
        episode.data_source = "test_dataset"
        
        summary = episode.get_episode_summary()
        
        assert summary['episode_id'] == "summary_test"
        assert summary['n_way'] == n_way
        assert summary['k_shot'] == k_shot
        assert summary['m_query'] == m_query
        assert summary['support_shape'] == list(support_x.shape)
        assert summary['query_shape'] == list(query_x.shape)
        assert summary['validated'] == True
        assert summary['validation_errors'] == 0
        assert summary['data_source'] == "test_dataset"
        assert summary['original_classes'] == [10, 11, 12, 13, 14]


class TestEpisodeBatchValidation:
    """Test episode batch validation functionality."""
    
    def test_empty_batch_validation(self):
        """Test validation of empty episode batch."""
        result = validate_episode_batch([])
        
        assert result['valid'] == True
        assert result['errors'] == []
    
    def test_consistent_batch_validation(self):
        """Test validation of consistent episode batch."""
        episodes = []
        
        for i in range(3):
            n_way, k_shot, m_query = 5, 2, 3
            support_x = torch.randn(n_way * k_shot, 128)
            support_y = torch.repeat_interleave(torch.arange(n_way), k_shot)
            query_x = torch.randn(n_way * m_query, 128)
            query_y = torch.repeat_interleave(torch.arange(n_way), m_query)
            
            episode = create_episode_contract(
                n_way, k_shot, m_query,
                support_x, support_y, query_x, query_y,
                episode_id=f"batch_{i}"
            )
            episodes.append(episode)
        
        result = validate_episode_batch(episodes)
        
        assert result['valid'] == True
        assert result['errors'] == []
        assert result['batch_size'] == 3
        assert result['parameters']['n_way'] == 5
    
    def test_inconsistent_batch_validation(self):
        """Test detection of inconsistent episode batches."""
        episodes = []
        
        # First episode: 5-way 2-shot
        n_way, k_shot, m_query = 5, 2, 3
        support_x = torch.randn(n_way * k_shot, 128)
        support_y = torch.repeat_interleave(torch.arange(n_way), k_shot)
        query_x = torch.randn(n_way * m_query, 128)
        query_y = torch.repeat_interleave(torch.arange(n_way), m_query)
        
        episode1 = create_episode_contract(
            n_way, k_shot, m_query,
            support_x, support_y, query_x, query_y,
            episode_id="batch_0"
        )
        episodes.append(episode1)
        
        # Second episode: 3-way 2-shot (different n_way)
        n_way2, k_shot2, m_query2 = 3, 2, 3
        support_x2 = torch.randn(n_way2 * k_shot2, 128)
        support_y2 = torch.repeat_interleave(torch.arange(n_way2), k_shot2)
        query_x2 = torch.randn(n_way2 * m_query2, 128)
        query_y2 = torch.repeat_interleave(torch.arange(n_way2), m_query2)
        
        episode2 = create_episode_contract(
            n_way2, k_shot2, m_query2,
            support_x2, support_y2, query_x2, query_y2,
            episode_id="batch_1"
        )
        episodes.append(episode2)
        
        result = validate_episode_batch(episodes)
        
        assert result['valid'] == False
        assert len(result['errors']) > 0
        assert any("n_way" in error for error in result['errors'])
    
    def test_device_inconsistency_detection(self):
        """Test detection of device inconsistencies in batch."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
            
        episodes = []
        
        # CPU episode
        n_way, k_shot, m_query = 5, 2, 3
        support_x = torch.randn(n_way * k_shot, 128)
        support_y = torch.repeat_interleave(torch.arange(n_way), k_shot)
        query_x = torch.randn(n_way * m_query, 128)
        query_y = torch.repeat_interleave(torch.arange(n_way), m_query)
        
        cpu_episode = create_episode_contract(
            n_way, k_shot, m_query,
            support_x, support_y, query_x, query_y,
            episode_id="cpu_episode"
        )
        episodes.append(cpu_episode)
        
        # CUDA episode
        cuda_episode = cpu_episode.to_device(torch.device('cuda'))
        cuda_episode.episode_id = "cuda_episode"
        episodes.append(cuda_episode)
        
        result = validate_episode_batch(episodes)
        
        assert result['valid'] == False
        assert any("different device" in error for error in result['errors'])


class TestIntegrationScenarios:
    """Integration tests for real-world usage scenarios."""
    
    def test_complete_prototypical_networks_workflow(self):
        """Test complete workflow for Prototypical Networks."""
        # Create episode as ProtoNet would
        n_way, k_shot, m_query = 5, 1, 15
        
        support_x = torch.randn(n_way * k_shot, 1600)  # 40x40 flattened images
        support_y = torch.arange(n_way)  # One example per class
        query_x = torch.randn(n_way * m_query, 1600)
        query_y = torch.repeat_interleave(torch.arange(n_way), m_query)
        
        episode = create_episode_contract(
            n_way, k_shot, m_query,
            support_x, support_y, query_x, query_y,
            episode_id="protonet_test"
        )
        
        # Simulate ProtoNet forward pass
        feature_dim = 64
        support_features = torch.randn(n_way * k_shot, feature_dim)
        query_features = torch.randn(n_way * m_query, feature_dim)
        
        # Compute prototypes (one per class)
        prototypes = support_features  # k_shot=1, so one prototype per class
        
        # Compute distances and logits
        distances = torch.cdist(query_features, prototypes)
        logits = -distances  # Negative distances for softmax
        
        # Validate predictions
        valid = episode.validate_prediction_output(logits)
        assert valid == True
        
        # Test with predicted classes
        predicted_classes = logits.argmax(dim=1)
        valid_classes = episode.validate_prediction_output(predicted_classes)
        assert valid_classes == True
    
    def test_complete_maml_workflow(self):
        """Test complete workflow for MAML."""
        # MAML typically uses smaller query sets
        n_way, k_shot, m_query = 5, 5, 5
        
        support_x = torch.randn(n_way * k_shot, 784)  # 28x28 MNIST-like
        support_y = torch.repeat_interleave(torch.arange(n_way), k_shot)
        query_x = torch.randn(n_way * m_query, 784)
        query_y = torch.repeat_interleave(torch.arange(n_way), m_query)
        
        episode = create_episode_contract(
            n_way, k_shot, m_query,
            support_x, support_y, query_x, query_y,
            episode_id="maml_test"
        )
        
        # Simulate MAML inner loop
        # (In practice, would adapt model parameters on support set)
        
        # Simulate outer loop evaluation on query set
        query_logits = torch.randn(n_way * m_query, n_way)
        
        # Validate query predictions
        valid = episode.validate_prediction_output(query_logits)
        assert valid == True
        
        # Test query loss computation (typical MAML pattern)
        query_loss = torch.nn.functional.cross_entropy(query_logits, episode.query_y)
        assert torch.isfinite(query_loss)
    
    def test_real_world_error_patterns(self):
        """Test detection of common real-world errors."""
        
        # Error 1: Forgetting to remap labels
        support_x = torch.randn(15, 128)
        support_y = torch.tensor([10, 10, 10, 25, 25, 25, 40, 40, 40, 55, 55, 55, 70, 70, 70])  # Original IDs
        query_x = torch.randn(10, 128)
        query_y = torch.tensor([10, 10, 25, 25, 40, 40, 55, 55, 70, 70])  # Original IDs
        
        with pytest.raises(ValueError, match="expected range"):
            create_episode_contract(5, 3, 2, support_x, support_y, query_x, query_y)
        
        # Error 2: Swapping support/query sizes
        n_way, k_shot, m_query = 5, 1, 5  # Should be 5 support, 25 query
        support_x = torch.randn(25, 128)  # Accidentally used query size
        support_y = torch.repeat_interleave(torch.arange(n_way), 5)  # Wrong count
        query_x = torch.randn(5, 128)  # Accidentally used support size
        query_y = torch.arange(5)  # Wrong count
        
        with pytest.raises(ValueError, match="support_x batch size"):
            create_episode_contract(n_way, k_shot, m_query, support_x, support_y, query_x, query_y)
        
        # Error 3: Feature dimension mismatch (common in multi-modal data)
        n_way, k_shot, m_query = 3, 2, 2
        support_x = torch.randn(6, 128)  # Text features
        support_y = torch.repeat_interleave(torch.arange(3), 2)
        query_x = torch.randn(6, 2048)  # Image features (wrong!)
        query_y = torch.repeat_interleave(torch.arange(3), 2)
        
        with pytest.raises(ValueError, match="different feature shapes"):
            create_episode_contract(n_way, k_shot, m_query, support_x, support_y, query_x, query_y)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])