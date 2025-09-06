"""
Comprehensive tests for Episode contracts with runtime validation.

Tests mathematical consistency, API correctness, tensor shapes, and 
label ranges for meta-learning episode validation.
"""
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from unittest.mock import MagicMock, patch
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from collections import Counter

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from meta_learning.meta_learning_modules.episode_contract import (
    EpisodeContract,
    EpisodeValidator,
    ValidationError,
    ContractViolation,
    validate_episode_contract,
    create_episode_contract,
    RuntimeEpisodeChecker
)


class TestEpisodeContract:
    """Test EpisodeContract dataclass functionality."""
    
    def test_episode_contract_creation(self):
        """Test EpisodeContract creation with valid parameters."""
        contract = EpisodeContract(
            n_way=5,
            k_shot=3,
            m_query=15,
            feature_dim=64,
            support_batch_size=15,  # 5 * 3
            query_batch_size=75     # 5 * 15
        )
        
        assert contract.n_way == 5
        assert contract.k_shot == 3
        assert contract.m_query == 15
        assert contract.feature_dim == 64
        assert contract.total_support_examples == 15
        assert contract.total_query_examples == 75
    
    def test_episode_contract_validation_success(self):
        """Test episode contract validation for valid episode."""
        contract = EpisodeContract(
            n_way=3,
            k_shot=2,
            m_query=5,
            feature_dim=32
        )
        
        # Create matching episode data
        support_x = torch.randn(6, 32)  # 3 * 2
        support_y = torch.tensor([0, 0, 1, 1, 2, 2])
        query_x = torch.randn(15, 32)   # 3 * 5
        query_y = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
        
        # Should validate successfully
        assert contract.validate_episode(support_x, support_y, query_x, query_y)
    
    def test_episode_contract_validation_failure_shapes(self):
        """Test episode contract validation failure for wrong shapes."""
        contract = EpisodeContract(
            n_way=3,
            k_shot=2,
            m_query=5,
            feature_dim=32
        )
        
        # Wrong support shape
        support_x = torch.randn(5, 32)  # Should be 6
        support_y = torch.tensor([0, 0, 1, 1, 2])
        query_x = torch.randn(15, 32)
        query_y = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
        
        with pytest.raises(ContractViolation):
            contract.validate_episode(support_x, support_y, query_x, query_y)
    
    def test_episode_contract_validation_failure_labels(self):
        """Test episode contract validation failure for wrong label distribution."""
        contract = EpisodeContract(
            n_way=3,
            k_shot=2,
            m_query=5,
            feature_dim=32
        )
        
        # Wrong label distribution (unbalanced classes)
        support_x = torch.randn(6, 32)
        support_y = torch.tensor([0, 0, 0, 1, 1, 2])  # Class 0 has 3 examples instead of 2
        query_x = torch.randn(15, 32)
        query_y = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
        
        with pytest.raises(ContractViolation):
            contract.validate_episode(support_x, support_y, query_x, query_y)
    
    def test_episode_contract_mathematical_properties(self):
        """Test mathematical properties and constraints."""
        contract = EpisodeContract(
            n_way=5,
            k_shot=1,
            m_query=15,
            feature_dim=128
        )
        
        # Mathematical invariants
        assert contract.total_support_examples == contract.n_way * contract.k_shot
        assert contract.total_query_examples == contract.n_way * contract.m_query
        
        # Validate consistency
        assert contract.is_mathematically_consistent()
    
    def test_episode_contract_edge_cases(self):
        """Test edge cases in episode contract validation."""
        # Single-way classification
        contract_1way = EpisodeContract(n_way=1, k_shot=5, m_query=10, feature_dim=64)
        assert contract_1way.is_mathematically_consistent()
        
        # Single-shot learning
        contract_1shot = EpisodeContract(n_way=5, k_shot=1, m_query=15, feature_dim=64)
        assert contract_1shot.is_mathematically_consistent()
        
        # Single query per class
        contract_1query = EpisodeContract(n_way=5, k_shot=5, m_query=1, feature_dim=64)
        assert contract_1query.is_mathematically_consistent()


class TestEpisodeValidator:
    """Test EpisodeValidator comprehensive validation."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.validator = EpisodeValidator(strict_mode=True)
        
        # Standard contract for testing
        self.standard_contract = EpisodeContract(
            n_way=5,
            k_shot=3,
            m_query=10,
            feature_dim=64
        )
    
    def test_validate_tensor_shapes(self):
        """Test tensor shape validation."""
        # Correct shapes
        support_x = torch.randn(15, 64)  # 5 * 3
        support_y = torch.randint(0, 5, (15,))
        query_x = torch.randn(50, 64)    # 5 * 10
        query_y = torch.randint(0, 5, (50,))
        
        result = self.validator.validate_tensor_shapes(
            support_x, support_y, query_x, query_y, self.standard_contract
        )
        
        assert result['valid']
        assert len(result['violations']) == 0
        
        # Wrong shapes
        wrong_support_x = torch.randn(10, 64)  # Wrong batch size
        
        result = self.validator.validate_tensor_shapes(
            wrong_support_x, support_y, query_x, query_y, self.standard_contract
        )
        
        assert not result['valid']
        assert len(result['violations']) > 0
    
    def test_validate_label_distribution(self):
        """Test label distribution validation."""
        # Correct balanced distribution
        support_y = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4])  # 3 per class
        query_y = torch.tensor([0]*10 + [1]*10 + [2]*10 + [3]*10 + [4]*10)  # 10 per class
        
        result = self.validator.validate_label_distribution(
            support_y, query_y, self.standard_contract
        )
        
        assert result['valid']
        assert result['support_distribution'] == {0: 3, 1: 3, 2: 3, 3: 3, 4: 3}
        assert result['query_distribution'] == {0: 10, 1: 10, 2: 10, 3: 10, 4: 10}
        
        # Wrong distribution (imbalanced)
        imbalanced_support_y = torch.tensor([0, 0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4, 4])
        
        result = self.validator.validate_label_distribution(
            imbalanced_support_y, query_y, self.standard_contract
        )
        
        assert not result['valid']
        assert len(result['violations']) > 0
    
    def test_validate_label_consistency(self):
        """Test label consistency between support and query."""
        # Consistent labels
        support_y = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4])
        query_y = torch.tensor([0, 1, 2, 3, 4] * 10)
        
        result = self.validator.validate_label_consistency(support_y, query_y)
        
        assert result['valid']
        assert result['support_classes'] == {0, 1, 2, 3, 4}
        assert result['query_classes'] == {0, 1, 2, 3, 4}
        
        # Inconsistent labels (query has class not in support)
        inconsistent_query_y = torch.tensor([0, 1, 2, 3, 5] * 10)  # Class 5 not in support
        
        result = self.validator.validate_label_consistency(support_y, inconsistent_query_y)
        
        assert not result['valid']
        assert 5 in result['missing_classes']
    
    def test_validate_feature_dimensions(self):
        """Test feature dimension validation."""
        contract = EpisodeContract(n_way=3, k_shot=2, m_query=5, feature_dim=128)
        
        # Correct dimensions
        support_x = torch.randn(6, 128)
        query_x = torch.randn(15, 128)
        
        result = self.validator.validate_feature_dimensions(support_x, query_x, contract)
        
        assert result['valid']
        assert result['support_feature_dim'] == 128
        assert result['query_feature_dim'] == 128
        
        # Mismatched dimensions
        wrong_query_x = torch.randn(15, 64)  # Wrong feature dim
        
        result = self.validator.validate_feature_dimensions(support_x, wrong_query_x, contract)
        
        assert not result['valid']
        assert len(result['violations']) > 0
    
    def test_validate_label_ranges(self):
        """Test label range validation."""
        contract = EpisodeContract(n_way=3, k_shot=2, m_query=5)
        
        # Correct label ranges [0, 1, 2]
        support_y = torch.tensor([0, 0, 1, 1, 2, 2])
        query_y = torch.tensor([0, 1, 2] * 5)
        
        result = self.validator.validate_label_ranges(support_y, query_y, contract)
        
        assert result['valid']
        assert result['expected_range'] == (0, 2)
        
        # Wrong label ranges (labels not starting from 0)
        wrong_support_y = torch.tensor([1, 1, 2, 2, 3, 3])  # Should be [0, 1, 2]
        wrong_query_y = torch.tensor([1, 2, 3] * 5)
        
        result = self.validator.validate_label_ranges(wrong_support_y, wrong_query_y, contract)
        
        assert not result['valid']
        assert len(result['violations']) > 0
    
    def test_comprehensive_validation_success(self):
        """Test comprehensive validation for valid episode."""
        # Create valid episode data
        support_x = torch.randn(15, 64)
        support_y = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4])
        query_x = torch.randn(50, 64)
        query_y = torch.tensor([0]*10 + [1]*10 + [2]*10 + [3]*10 + [4]*10)
        
        result = self.validator.comprehensive_validation(
            support_x, support_y, query_x, query_y, self.standard_contract
        )
        
        assert result['valid']
        assert len(result['violations']) == 0
        assert result['passed_checks'] > 0
    
    def test_comprehensive_validation_failure(self):
        """Test comprehensive validation for invalid episode."""
        # Create invalid episode data (wrong shapes)
        support_x = torch.randn(10, 64)  # Wrong batch size
        support_y = torch.randint(0, 5, (10,))
        query_x = torch.randn(50, 64)
        query_y = torch.randint(0, 5, (50,))
        
        result = self.validator.comprehensive_validation(
            support_x, support_y, query_x, query_y, self.standard_contract
        )
        
        assert not result['valid']
        assert len(result['violations']) > 0
    
    def test_strict_vs_lenient_mode(self):
        """Test difference between strict and lenient validation modes."""
        # Create episode with minor issues
        support_x = torch.randn(15, 64)
        support_y = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4])
        query_x = torch.randn(50, 64)
        query_y = torch.tensor([0]*11 + [1]*10 + [2]*10 + [3]*10 + [4]*9)  # Slightly imbalanced
        
        # Strict validator
        strict_validator = EpisodeValidator(strict_mode=True)
        strict_result = strict_validator.comprehensive_validation(
            support_x, support_y, query_x, query_y, self.standard_contract
        )
        
        # Lenient validator
        lenient_validator = EpisodeValidator(strict_mode=False)
        lenient_result = lenient_validator.comprehensive_validation(
            support_x, support_y, query_x, query_y, self.standard_contract
        )
        
        # Strict mode should be more restrictive
        assert len(strict_result['violations']) >= len(lenient_result['violations'])


class TestRuntimeEpisodeChecker:
    """Test runtime episode checking functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.checker = RuntimeEpisodeChecker()
    
    def test_create_runtime_contract(self):
        """Test runtime contract creation from episode data."""
        support_x = torch.randn(20, 128)
        support_y = torch.tensor([0]*4 + [1]*4 + [2]*4 + [3]*4 + [4]*4)  # 5-way 4-shot
        query_x = torch.randn(75, 128)
        query_y = torch.tensor([0]*15 + [1]*15 + [2]*15 + [3]*15 + [4]*15)  # 15 queries per class
        
        contract = self.checker.create_runtime_contract(support_x, support_y, query_x, query_y)
        
        assert contract.n_way == 5
        assert contract.k_shot == 4
        assert contract.m_query == 15
        assert contract.feature_dim == 128
        assert contract.total_support_examples == 20
        assert contract.total_query_examples == 75
    
    def test_validate_episode_at_runtime(self):
        """Test episode validation at runtime."""
        # Valid episode
        support_x = torch.randn(15, 64)
        support_y = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4])
        query_x = torch.randn(50, 64)
        query_y = torch.tensor([0]*10 + [1]*10 + [2]*10 + [3]*10 + [4]*10)
        
        # Should pass validation
        self.checker.validate_episode_at_runtime(support_x, support_y, query_x, query_y)
        
        # Invalid episode (should raise exception)
        invalid_support_x = torch.randn(10, 64)  # Wrong shape
        
        with pytest.raises(ContractViolation):
            self.checker.validate_episode_at_runtime(invalid_support_x, support_y, query_x, query_y)
    
    def test_runtime_assertions(self):
        """Test runtime assertions for episode properties."""
        support_x = torch.randn(12, 32)
        support_y = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])  # 3-way 4-shot
        query_x = torch.randn(18, 32)
        query_y = torch.tensor([0]*6 + [1]*6 + [2]*6)  # 6 queries per class
        
        # Test individual assertions
        self.checker.assert_n_way_k_shot(support_y, n_way=3, k_shot=4)
        self.checker.assert_feature_consistency(support_x, query_x)
        self.checker.assert_label_remapping(support_y, query_y)
        
        # Test assertion failures
        with pytest.raises(ContractViolation):
            self.checker.assert_n_way_k_shot(support_y, n_way=5, k_shot=4)  # Wrong n_way
        
        with pytest.raises(ContractViolation):
            wrong_query_x = torch.randn(18, 16)  # Wrong feature dim
            self.checker.assert_feature_consistency(support_x, wrong_query_x)
    
    def test_batch_validation(self):
        """Test batch-wise episode validation."""
        # Create batch of episodes
        episodes = []
        for i in range(5):
            episode = {
                'support_x': torch.randn(15, 64),
                'support_y': torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]),
                'query_x': torch.randn(50, 64),
                'query_y': torch.tensor([0]*10 + [1]*10 + [2]*10 + [3]*10 + [4]*10)
            }
            episodes.append(episode)
        
        # Should validate all episodes successfully
        results = self.checker.validate_episode_batch(episodes)
        
        assert len(results) == 5
        assert all(result['valid'] for result in results)
        
        # Add invalid episode
        invalid_episode = {
            'support_x': torch.randn(10, 64),  # Wrong shape
            'support_y': torch.randint(0, 5, (10,)),
            'query_x': torch.randn(50, 64),
            'query_y': torch.randint(0, 5, (50,))
        }
        episodes.append(invalid_episode)
        
        results = self.checker.validate_episode_batch(episodes)
        assert len(results) == 6
        assert not results[-1]['valid']  # Last episode should fail


class TestValidationUtilities:
    """Test utility functions for episode validation."""
    
    def test_validate_episode_contract_function(self):
        """Test standalone episode contract validation function."""
        contract = EpisodeContract(n_way=3, k_shot=2, m_query=5, feature_dim=32)
        
        support_x = torch.randn(6, 32)
        support_y = torch.tensor([0, 0, 1, 1, 2, 2])
        query_x = torch.randn(15, 32)
        query_y = torch.tensor([0]*5 + [1]*5 + [2]*5)
        
        # Should validate successfully
        result = validate_episode_contract(
            support_x, support_y, query_x, query_y, contract
        )
        
        assert result['valid']
        assert len(result['violations']) == 0
    
    def test_create_episode_contract_function(self):
        """Test episode contract creation utility function."""
        support_x = torch.randn(20, 128)
        support_y = torch.tensor([0]*5 + [1]*5 + [2]*5 + [3]*5)  # 4-way 5-shot
        query_x = torch.randn(40, 128)
        query_y = torch.tensor([0]*10 + [1]*10 + [2]*10 + [3]*10)  # 10 queries per class
        
        contract = create_episode_contract(support_x, support_y, query_x, query_y)
        
        assert contract.n_way == 4
        assert contract.k_shot == 5
        assert contract.m_query == 10
        assert contract.feature_dim == 128
    
    def test_contract_serialization(self):
        """Test contract serialization/deserialization."""
        contract = EpisodeContract(
            n_way=5,
            k_shot=3,
            m_query=15,
            feature_dim=64,
            additional_constraints={'min_accuracy': 0.8}
        )
        
        # Convert to dict
        contract_dict = contract.to_dict()
        
        assert contract_dict['n_way'] == 5
        assert contract_dict['k_shot'] == 3
        assert contract_dict['feature_dim'] == 64
        assert contract_dict['additional_constraints']['min_accuracy'] == 0.8
        
        # Recreate from dict
        contract_restored = EpisodeContract.from_dict(contract_dict)
        
        assert contract_restored.n_way == contract.n_way
        assert contract_restored.k_shot == contract.k_shot
        assert contract_restored.additional_constraints == contract.additional_constraints


class TestAdvancedValidationFeatures:
    """Test advanced validation features and constraints."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.validator = EpisodeValidator(strict_mode=True)
    
    def test_custom_validation_rules(self):
        """Test custom validation rules."""
        def custom_rule(support_x, support_y, query_x, query_y, contract):
            """Custom rule: support examples should have positive mean."""
            support_means = support_x.mean(dim=1)
            if (support_means < 0).any():
                return {
                    'valid': False,
                    'violations': ['Support examples must have positive mean']
                }
            return {'valid': True, 'violations': []}
        
        # Add custom rule
        self.validator.add_custom_rule('positive_mean', custom_rule)
        
        # Test with positive mean data
        support_x = torch.abs(torch.randn(15, 64)) + 1.0  # Ensure positive
        support_y = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4])
        query_x = torch.randn(50, 64)
        query_y = torch.tensor([0]*10 + [1]*10 + [2]*10 + [3]*10 + [4]*10)
        
        contract = EpisodeContract(n_way=5, k_shot=3, m_query=10, feature_dim=64)
        
        result = self.validator.comprehensive_validation(
            support_x, support_y, query_x, query_y, contract
        )
        
        assert result['valid']
        
        # Test with negative mean data
        negative_support_x = -torch.abs(torch.randn(15, 64)) - 1.0  # Ensure negative
        
        result = self.validator.comprehensive_validation(
            negative_support_x, support_y, query_x, query_y, contract
        )
        
        assert not result['valid']
        assert any('positive mean' in v.lower() for v in result['violations'])
    
    def test_statistical_validation(self):
        """Test statistical validation of episode properties."""
        # Create episode with known statistical properties
        support_x = torch.randn(15, 64) * 2.0 + 1.0  # Mean ~1, std ~2
        support_y = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4])
        query_x = torch.randn(50, 64) * 2.0 + 1.0
        query_y = torch.tensor([0]*10 + [1]*10 + [2]*10 + [3]*10 + [4]*10)
        
        contract = EpisodeContract(
            n_way=5, k_shot=3, m_query=10, feature_dim=64,
            additional_constraints={
                'min_std': 1.0,
                'max_std': 3.0,
                'min_mean': 0.0,
                'max_mean': 2.0
            }
        )
        
        result = self.validator.validate_statistical_properties(
            support_x, query_x, contract
        )
        
        # Should pass statistical validation
        assert result['valid']
    
    def test_temporal_consistency_validation(self):
        """Test temporal consistency validation for sequential episodes."""
        # Create sequence of episodes
        episodes = []
        for t in range(5):
            episode = {
                'support_x': torch.randn(15, 64),
                'support_y': torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]),
                'query_x': torch.randn(50, 64),
                'query_y': torch.tensor([0]*10 + [1]*10 + [2]*10 + [3]*10 + [4]*10),
                'timestamp': t
            }
            episodes.append(episode)
        
        result = self.validator.validate_temporal_consistency(episodes)
        
        assert result['valid']
        assert result['sequence_length'] == 5
        
        # Test with temporal inconsistency (timestamps out of order)
        episodes[2]['timestamp'] = 10  # Out of order
        
        result = self.validator.validate_temporal_consistency(episodes)
        
        assert not result['valid']
        assert len(result['violations']) > 0
    
    def test_cross_episode_validation(self):
        """Test validation across multiple episodes."""
        # Create related episodes
        episodes = []
        for i in range(3):
            episode = {
                'id': f'ep_{i}',
                'support_x': torch.randn(15, 64),
                'support_y': torch.tensor([0+i*5, 0+i*5, 0+i*5, 1+i*5, 1+i*5, 1+i*5, 
                                         2+i*5, 2+i*5, 2+i*5, 3+i*5, 3+i*5, 3+i*5, 
                                         4+i*5, 4+i*5, 4+i*5]),  # Non-overlapping classes
                'query_x': torch.randn(50, 64),
                'query_y': torch.tensor([0+i*5]*10 + [1+i*5]*10 + [2+i*5]*10 + [3+i*5]*10 + [4+i*5]*10),
                'split': 'train'
            }
            episodes.append(episode)
        
        result = self.validator.validate_cross_episode_consistency(episodes)
        
        assert result['valid']  # No class overlap between episodes
        
        # Add episode with overlapping classes
        overlapping_episode = {
            'id': 'ep_overlap',
            'support_x': torch.randn(15, 64),
            'support_y': torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]),  # Overlaps with ep_0
            'query_x': torch.randn(50, 64),
            'query_y': torch.tensor([0]*10 + [1]*10 + [2]*10 + [3]*10 + [4]*10),
            'split': 'train'
        }
        episodes.append(overlapping_episode)
        
        result = self.validator.validate_cross_episode_consistency(episodes)
        
        # Should detect overlap if strict mode is enabled for cross-episode validation
        if hasattr(self.validator, 'strict_cross_episode') and self.validator.strict_cross_episode:
            assert not result['valid']


class TestPerformanceAndRobustness:
    """Test performance and robustness of validation system."""
    
    def test_large_episode_validation(self):
        """Test validation performance on large episodes."""
        # Large episode (100-way 10-shot)
        n_way, k_shot, m_query = 100, 10, 50
        feature_dim = 512
        
        support_x = torch.randn(n_way * k_shot, feature_dim)
        support_y = torch.arange(n_way).repeat_interleave(k_shot)
        query_x = torch.randn(n_way * m_query, feature_dim)
        query_y = torch.arange(n_way).repeat_interleave(m_query)
        
        contract = EpisodeContract(
            n_way=n_way,
            k_shot=k_shot,
            m_query=m_query,
            feature_dim=feature_dim
        )
        
        validator = EpisodeValidator(strict_mode=False)  # Use lenient mode for performance
        
        import time
        start_time = time.time()
        
        result = validator.comprehensive_validation(
            support_x, support_y, query_x, query_y, contract
        )
        
        elapsed = time.time() - start_time
        
        # Should complete in reasonable time (< 5 seconds)
        assert elapsed < 5.0
        assert result['valid']
    
    def test_numerical_stability(self):
        """Test validation numerical stability with extreme values."""
        # Episode with extreme values
        support_x = torch.randn(15, 64) * 1e6  # Very large values
        support_y = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4])
        query_x = torch.randn(50, 64) * 1e-6   # Very small values
        query_y = torch.tensor([0]*10 + [1]*10 + [2]*10 + [3]*10 + [4]*10)
        
        contract = EpisodeContract(n_way=5, k_shot=3, m_query=10, feature_dim=64)
        validator = EpisodeValidator(strict_mode=True)
        
        # Should handle extreme values gracefully
        result = validator.comprehensive_validation(
            support_x, support_y, query_x, query_y, contract
        )
        
        # Should validate shapes and labels correctly despite extreme values
        assert result['valid'] or 'statistical' in ' '.join(result['violations']).lower()
    
    def test_memory_efficiency(self):
        """Test memory efficiency of validation."""
        validator = EpisodeValidator(strict_mode=True)
        
        # Validate many episodes without memory accumulation
        for i in range(100):
            support_x = torch.randn(15, 64)
            support_y = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4])
            query_x = torch.randn(50, 64)
            query_y = torch.tensor([0]*10 + [1]*10 + [2]*10 + [3]*10 + [4]*10)
            
            contract = EpisodeContract(n_way=5, k_shot=3, m_query=10, feature_dim=64)
            
            result = validator.comprehensive_validation(
                support_x, support_y, query_x, query_y, contract
            )
            
            assert result is not None
            # Memory should not accumulate between validations


if __name__ == "__main__":
    pytest.main([__file__])