"""
Comprehensive tests for Leakage detection and prevention system.

Tests train/test split validation, normalization statistics monitoring, 
and episode isolation verification for meta-learning scenarios.
"""
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from unittest.mock import MagicMock, patch
from typing import Dict, List, Tuple, Any, Set
import warnings

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from meta_learning.meta_learning_modules.leakage_guard import (
    LeakageDetector,
    NormalizationLeakageMonitor,
    EpisodeIsolationValidator,
    DataLeakageGuard,
    CrossEpisodeContaminationDetector,
    validate_data_split,
    detect_normalization_leakage
)


class TestLeakageDetector:
    """Test main LeakageDetector functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        # Define train/test split
        self.train_classes = set(range(64))  # Classes 0-63 for training
        self.test_classes = set(range(64, 100))  # Classes 64-99 for testing
        
        self.detector = LeakageDetector(
            train_classes=self.train_classes,
            test_classes=self.test_classes
        )
    
    def test_detector_initialization(self):
        """Test LeakageDetector initialization."""
        assert self.detector.train_classes == self.train_classes
        assert self.detector.test_classes == self.test_classes
        assert len(self.detector.train_classes & self.detector.test_classes) == 0  # No overlap
    
    def test_validate_episode_classes_valid(self):
        """Test episode class validation for valid episodes."""
        # Episode with only training classes
        train_episode_classes = {5, 12, 23, 35, 47}
        result = self.detector.validate_episode_classes(train_episode_classes, split='train')
        
        assert result['valid']
        assert len(result['violations']) == 0
        
        # Episode with only test classes
        test_episode_classes = {67, 72, 85, 91, 98}
        result = self.detector.validate_episode_classes(test_episode_classes, split='test')
        
        assert result['valid']
        assert len(result['violations']) == 0
    
    def test_validate_episode_classes_invalid(self):
        """Test episode class validation for invalid episodes."""
        # Episode with test classes during training
        contaminated_classes = {5, 12, 67}  # 67 is test class
        result = self.detector.validate_episode_classes(contaminated_classes, split='train')
        
        assert not result['valid']
        assert len(result['violations']) > 0
        assert 67 in result['violations']
        
        # Episode with train classes during testing
        contaminated_test = {23, 72, 85}  # 23 is train class
        result = self.detector.validate_episode_classes(contaminated_test, split='test')
        
        assert not result['valid']
        assert len(result['violations']) > 0
        assert 23 in result['violations']
    
    def test_detect_cross_split_contamination(self):
        """Test detection of cross-split contamination."""
        # Dataset with mixed train/test examples
        data_classes = {5, 12, 67, 72, 23}  # Mix of train and test
        
        contamination = self.detector.detect_cross_split_contamination(data_classes)
        
        assert contamination['has_contamination']
        assert contamination['train_in_data']
        assert contamination['test_in_data']
        
        train_contaminants = contamination['contaminated_classes']['train_classes_in_data']
        test_contaminants = contamination['contaminated_classes']['test_classes_in_data']
        
        assert 5 in train_contaminants
        assert 12 in train_contaminants
        assert 23 in train_contaminants
        assert 67 in test_contaminants
        assert 72 in test_contaminants
    
    def test_no_contamination_detection(self):
        """Test that clean data shows no contamination."""
        # Only training classes
        clean_train_classes = {5, 12, 23, 35}
        contamination = self.detector.detect_cross_split_contamination(clean_train_classes)
        
        assert not contamination['has_contamination']
        assert contamination['train_in_data']
        assert not contamination['test_in_data']
        
        # Only test classes
        clean_test_classes = {67, 72, 85, 91}
        contamination = self.detector.detect_cross_split_contamination(clean_test_classes)
        
        assert not contamination['has_contamination']
        assert not contamination['train_in_data']
        assert contamination['test_in_data']


class TestNormalizationLeakageMonitor:
    """Test normalization statistics leakage monitoring."""
    
    def setup_method(self):
        """Setup test fixtures."""
        train_classes = set(range(50))
        test_classes = set(range(50, 100))
        
        self.monitor = NormalizationLeakageMonitor(
            train_classes=train_classes,
            test_classes=test_classes
        )
    
    def test_capture_normalization_stats(self):
        """Test capturing normalization statistics."""
        # Sample data with known statistics
        data = torch.randn(100, 10)
        data_classes = set(range(25))  # Training classes
        
        stats = self.monitor.capture_normalization_stats(data, data_classes)
        
        assert 'mean' in stats
        assert 'std' in stats
        assert 'min' in stats
        assert 'max' in stats
        assert stats['n_samples'] == 100
        assert stats['data_classes'] == data_classes
        
        # Check that statistics are reasonable
        assert torch.allclose(stats['mean'], data.mean(dim=0), atol=1e-6)
        assert torch.allclose(stats['std'], data.std(dim=0), atol=1e-6)
    
    def test_validate_normalization_stats_clean(self):
        """Test validation of clean normalization statistics."""
        # Training data only
        train_data = torch.randn(100, 5)
        train_classes = {5, 12, 23}
        train_stats = self.monitor.capture_normalization_stats(train_data, train_classes)
        
        result = self.monitor.validate_normalization_stats(train_stats, train_classes, "training")
        
        assert result['valid']
        assert len(result['warnings']) == 0
        assert len(result['errors']) == 0
    
    def test_validate_normalization_stats_contaminated(self):
        """Test validation of contaminated normalization statistics."""
        # Data computed on train+test classes (leakage)
        mixed_data = torch.randn(100, 5)
        mixed_classes = {5, 12, 67, 72}  # Mix of train and test
        mixed_stats = self.monitor.capture_normalization_stats(mixed_data, mixed_classes)
        
        # Validate as if it's training normalization (should detect test contamination)
        result = self.monitor.validate_normalization_stats(mixed_stats, mixed_classes, "training")
        
        assert not result['valid']
        assert len(result['errors']) > 0
        
        error_text = ' '.join(result['errors']).lower()
        assert 'test' in error_text and 'contamination' in error_text
    
    def test_detect_statistics_drift(self):
        """Test detection of statistics drift between episodes."""
        # Episode 1 statistics
        episode1_data = torch.randn(50, 8)
        episode1_classes = {5, 12, 23}
        stats1 = self.monitor.capture_normalization_stats(episode1_data, episode1_classes)
        
        # Episode 2 with different distribution (drift)
        episode2_data = torch.randn(50, 8) + 3.0  # Shifted distribution
        episode2_classes = {5, 12, 23}  # Same classes
        stats2 = self.monitor.capture_normalization_stats(episode2_data, episode2_classes)
        
        drift_report = self.monitor.detect_statistics_drift([stats1, stats2])
        
        assert drift_report['has_drift']
        assert drift_report['mean_drift'] > 2.0  # Should detect the +3.0 shift
        assert drift_report['max_feature_drift'] > 2.0
    
    def test_no_statistics_drift(self):
        """Test that similar distributions show no drift."""
        # Two similar episodes
        base_data = torch.randn(50, 8)
        classes = {5, 12, 23}
        
        stats1 = self.monitor.capture_normalization_stats(base_data + torch.randn_like(base_data) * 0.1, classes)
        stats2 = self.monitor.capture_normalization_stats(base_data + torch.randn_like(base_data) * 0.1, classes)
        
        drift_report = self.monitor.detect_statistics_drift([stats1, stats2])
        
        assert not drift_report['has_drift']
        assert drift_report['mean_drift'] < 0.5
    
    def test_batch_normalization_leakage_detection(self):
        """Test specific BatchNorm leakage detection."""
        # Simulate BatchNorm running statistics computed on mixed data
        mixed_classes = {5, 12, 67, 72}  # Train + test classes
        
        # Create fake BatchNorm statistics
        bn_stats = {
            'running_mean': torch.randn(64),
            'running_var': torch.ones(64),
            'num_batches_tracked': torch.tensor(100),
            'data_classes': mixed_classes
        }
        
        # Should detect contamination
        result = self.monitor.validate_batch_norm_stats(bn_stats, "training")
        
        assert not result['valid']
        assert len(result['errors']) > 0


class TestEpisodeIsolationValidator:
    """Test episode isolation validation."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.validator = EpisodeIsolationValidator()
    
    def test_validate_episode_isolation_success(self):
        """Test successful episode isolation validation."""
        # Create two properly isolated episodes
        episode1 = {
            'support_classes': {0, 1, 2, 3, 4},
            'query_classes': {0, 1, 2, 3, 4},
            'support_indices': set(range(25)),
            'query_indices': set(range(25, 50))
        }
        
        episode2 = {
            'support_classes': {5, 6, 7, 8, 9},
            'query_classes': {5, 6, 7, 8, 9},
            'support_indices': set(range(100, 125)),
            'query_indices': set(range(125, 150))
        }
        
        result = self.validator.validate_episode_isolation([episode1, episode2])
        
        assert result['isolated']
        assert len(result['violations']) == 0
        assert len(result['warnings']) == 0
    
    def test_validate_episode_isolation_failure(self):
        """Test episode isolation validation failure."""
        # Create episodes with overlapping data
        episode1 = {
            'support_classes': {0, 1, 2, 3, 4},
            'query_classes': {0, 1, 2, 3, 4},
            'support_indices': set(range(25)),
            'query_indices': set(range(25, 50))
        }
        
        episode2 = {
            'support_classes': {0, 1, 5, 6, 7},  # Classes 0,1 overlap with episode1
            'query_classes': {0, 1, 5, 6, 7},
            'support_indices': set(range(20, 45)),  # Indices overlap with episode1
            'query_indices': set(range(45, 70))
        }
        
        result = self.validator.validate_episode_isolation([episode1, episode2])
        
        assert not result['isolated']
        assert len(result['violations']) > 0
        
        # Should detect both class and data overlap
        violations_text = ' '.join(result['violations']).lower()
        assert 'class' in violations_text or 'overlap' in violations_text
    
    def test_validate_support_query_isolation(self):
        """Test support-query isolation within episodes."""
        # Episode with support-query data overlap (invalid)
        bad_episode = {
            'support_classes': {0, 1, 2},
            'query_classes': {0, 1, 2},
            'support_indices': set(range(15)),
            'query_indices': set(range(10, 25))  # Overlap with support indices 10-14
        }
        
        result = self.validator.validate_support_query_isolation(bad_episode)
        
        assert not result['isolated']
        assert len(result['violations']) > 0
        
        # Should detect support-query data overlap
        violations_text = ' '.join(result['violations']).lower()
        assert 'support' in violations_text and 'query' in violations_text
        
        # Good episode (no overlap)
        good_episode = {
            'support_classes': {0, 1, 2},
            'query_classes': {0, 1, 2},
            'support_indices': set(range(15)),
            'query_indices': set(range(15, 30))  # No overlap
        }
        
        result = self.validator.validate_support_query_isolation(good_episode)
        
        assert result['isolated']
        assert len(result['violations']) == 0


class TestDataLeakageGuard:
    """Test comprehensive data leakage guard."""
    
    def setup_method(self):
        """Setup test fixtures."""
        train_classes = set(range(64))
        test_classes = set(range(64, 100))
        
        self.guard = DataLeakageGuard(
            train_classes=train_classes,
            test_classes=test_classes,
            strict_mode=True
        )
    
    def test_comprehensive_leakage_check_clean(self):
        """Test comprehensive leakage check on clean data."""
        # Clean training episode
        episode_data = {
            'support_x': torch.randn(25, 10),
            'support_y': torch.tensor([5, 5, 5, 5, 5, 12, 12, 12, 12, 12, 
                                     23, 23, 23, 23, 23, 35, 35, 35, 35, 35,
                                     47, 47, 47, 47, 47]),  # All training classes
            'query_x': torch.randn(75, 10),
            'query_y': torch.tensor([5]*15 + [12]*15 + [23]*15 + [35]*15 + [47]*15),
            'split': 'train'
        }
        
        result = self.guard.comprehensive_leakage_check(episode_data)
        
        assert result['safe']
        assert len(result['violations']) == 0
        assert len(result['warnings']) == 0
    
    def test_comprehensive_leakage_check_contaminated(self):
        """Test comprehensive leakage check on contaminated data."""
        # Contaminated training episode (contains test class)
        episode_data = {
            'support_x': torch.randn(25, 10),
            'support_y': torch.tensor([5, 5, 5, 5, 5, 12, 12, 12, 12, 12,
                                     23, 23, 23, 23, 23, 35, 35, 35, 35, 35,
                                     67, 67, 67, 67, 67]),  # Class 67 is test class!
            'query_x': torch.randn(75, 10),
            'query_y': torch.tensor([5]*15 + [12]*15 + [23]*15 + [35]*15 + [67]*15),
            'split': 'train'
        }
        
        result = self.guard.comprehensive_leakage_check(episode_data)
        
        assert not result['safe']
        assert len(result['violations']) > 0
        
        violations_text = ' '.join(result['violations']).lower()
        assert 'test' in violations_text and '67' in violations_text
    
    def test_normalization_leakage_detection(self):
        """Test detection of normalization-based leakage."""
        # Episode normalized using mixed train/test statistics
        episode_data = {
            'support_x': torch.randn(25, 10),
            'support_y': torch.tensor([5]*5 + [12]*5 + [23]*5 + [35]*5 + [47]*5),
            'query_x': torch.randn(75, 10),
            'query_y': torch.tensor([5]*15 + [12]*15 + [23]*15 + [35]*15 + [47]*15),
            'split': 'train',
            'normalization_stats': {
                'data_classes': {5, 12, 23, 67, 72},  # Mixed train/test classes
                'mean': torch.zeros(10),
                'std': torch.ones(10)
            }
        }
        
        result = self.guard.comprehensive_leakage_check(episode_data)
        
        assert not result['safe']
        assert len(result['violations']) > 0
        
        violations_text = ' '.join(result['violations']).lower()
        assert 'normalization' in violations_text or 'statistics' in violations_text
    
    def test_batch_wise_leakage_detection(self):
        """Test detection of batch-wise leakage patterns."""
        # Create batches with suspicious patterns
        batches = [
            {'classes': {5, 12, 23}, 'split': 'train'},
            {'classes': {67, 72, 85}, 'split': 'train'},  # Test classes in training!
            {'classes': {35, 47, 55}, 'split': 'train'}
        ]
        
        result = self.guard.detect_batch_wise_leakage(batches)
        
        assert result['has_leakage']
        assert len(result['problematic_batches']) > 0
        
        # Should identify batch 1 as problematic
        problem_batch_ids = [b['batch_id'] for b in result['problematic_batches']]
        assert 1 in problem_batch_ids
    
    def test_temporal_leakage_detection(self):
        """Test detection of temporal leakage (future information)."""
        # Episodes with temporal ordering issues
        episodes = [
            {
                'timestamp': 100,
                'classes': {5, 12, 23},
                'split': 'train',
                'data_collection_time': 95  # Data collected after episode time (suspicious)
            },
            {
                'timestamp': 200,
                'classes': {35, 47, 55},
                'split': 'train',
                'data_collection_time': 190  # Normal
            }
        ]
        
        result = self.guard.detect_temporal_leakage(episodes)
        
        assert result['has_temporal_issues']
        assert len(result['temporal_violations']) > 0


class TestCrossEpisodeContaminationDetector:
    """Test cross-episode contamination detection."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.detector = CrossEpisodeContaminationDetector()
    
    def test_detect_data_reuse_across_episodes(self):
        """Test detection of data reuse across episodes."""
        # Episodes with overlapping data indices
        episodes = [
            {
                'id': 'ep1',
                'data_indices': set(range(50)),
                'classes': {0, 1, 2, 3, 4}
            },
            {
                'id': 'ep2', 
                'data_indices': set(range(25, 75)),  # Overlaps with ep1 (25-49)
                'classes': {5, 6, 7, 8, 9}
            },
            {
                'id': 'ep3',
                'data_indices': set(range(100, 150)),  # No overlap
                'classes': {10, 11, 12, 13, 14}
            }
        ]
        
        result = self.detector.detect_data_reuse_across_episodes(episodes)
        
        assert result['has_reuse']
        assert len(result['reuse_pairs']) > 0
        
        # Should detect ep1-ep2 overlap
        reuse_pairs = [(pair['episode1'], pair['episode2']) for pair in result['reuse_pairs']]
        assert ('ep1', 'ep2') in reuse_pairs or ('ep2', 'ep1') in reuse_pairs
    
    def test_detect_class_contamination_patterns(self):
        """Test detection of class contamination patterns."""
        # Episodes with suspicious class patterns
        training_episodes = [
            {'classes': {0, 1, 2, 3, 4}, 'split': 'train'},
            {'classes': {5, 6, 7, 8, 9}, 'split': 'train'},
            {'classes': {67, 68, 69}, 'split': 'train'},  # Test classes in training
        ]
        
        test_episodes = [
            {'classes': {67, 68, 69, 70, 71}, 'split': 'test'},
            {'classes': {5, 72, 73, 74, 75}, 'split': 'test'},  # Train class in testing
        ]
        
        all_episodes = training_episodes + test_episodes
        
        result = self.detector.detect_class_contamination_patterns(all_episodes)
        
        assert result['has_contamination']
        assert len(result['contaminated_classes']) > 0
        
        # Should detect classes that appear in both splits
        contaminated = result['contaminated_classes']
        assert any(cls in contaminated for cls in [5, 67, 68, 69])
    
    def test_detect_model_contamination(self):
        """Test detection of model-based contamination."""
        # Simulate model that has seen test data during training
        model_training_classes = {0, 1, 2, 3, 4, 67, 68}  # Mix of train and test
        episode_test_classes = {67, 68, 69, 70, 71}  # Test episode
        
        result = self.detector.detect_model_contamination(
            model_training_classes, 
            episode_test_classes
        )
        
        assert result['contaminated']
        assert len(result['overlapping_classes']) > 0
        
        # Should detect overlap
        overlapping = result['overlapping_classes']
        assert 67 in overlapping
        assert 68 in overlapping


class TestUtilityFunctions:
    """Test utility functions for leakage detection."""
    
    def test_validate_data_split_clean(self):
        """Test data split validation for clean splits."""
        train_classes = set(range(50))
        test_classes = set(range(50, 100))
        
        # Clean episode data
        episode_classes = {5, 12, 23, 35, 47}
        
        result = validate_data_split(episode_classes, train_classes, test_classes, 'train')
        
        assert result['valid']
        assert len(result['violations']) == 0
    
    def test_validate_data_split_contaminated(self):
        """Test data split validation for contaminated splits."""
        train_classes = set(range(50))
        test_classes = set(range(50, 100))
        
        # Contaminated episode data
        episode_classes = {5, 12, 67}  # 67 is test class
        
        result = validate_data_split(episode_classes, train_classes, test_classes, 'train')
        
        assert not result['valid']
        assert len(result['violations']) > 0
        assert 67 in result['violations']
    
    def test_detect_normalization_leakage_function(self):
        """Test normalization leakage detection utility function."""
        # Statistics computed on mixed data
        stats = {
            'data_classes': {5, 12, 67, 72},  # Mix of train and test
            'mean': torch.zeros(10),
            'std': torch.ones(10)
        }
        
        train_classes = set(range(50))
        test_classes = set(range(50, 100))
        
        result = detect_normalization_leakage(stats, train_classes, test_classes, 'train')
        
        assert result['has_leakage']
        assert len(result['leaked_classes']) > 0
        
        # Should detect test classes in training normalization
        leaked = result['leaked_classes']
        assert 67 in leaked
        assert 72 in leaked


class TestRealisticLeakageScenarios:
    """Test realistic leakage scenarios in meta-learning."""
    
    def setup_method(self):
        """Setup realistic meta-learning scenario."""
        # Realistic train/test split (miniImageNet style)
        self.train_classes = set(range(64))      # 64 training classes
        self.val_classes = set(range(64, 80))    # 16 validation classes
        self.test_classes = set(range(80, 100))  # 20 test classes
        
        self.guard = DataLeakageGuard(
            train_classes=self.train_classes,
            test_classes=self.test_classes | self.val_classes,
            strict_mode=True
        )
    
    def test_meta_learning_episode_validation(self):
        """Test validation of meta-learning episodes."""
        # Valid training episode
        train_episode = self._create_episode(
            classes=[5, 12, 23, 35, 47],  # Training classes
            n_support=5,
            n_query=15,
            split='train'
        )
        
        result = self.guard.comprehensive_leakage_check(train_episode)
        assert result['safe']
        
        # Invalid training episode (contains validation class)
        invalid_train_episode = self._create_episode(
            classes=[5, 12, 23, 35, 67],  # 67 is validation class
            n_support=5,
            n_query=15,
            split='train'
        )
        
        result = self.guard.comprehensive_leakage_check(invalid_train_episode)
        assert not result['safe']
    
    def test_data_augmentation_leakage(self):
        """Test leakage through data augmentation statistics."""
        # Data augmentation computed on full dataset (including test)
        augmentation_stats = {
            'data_classes': set(range(100)),  # All classes used for normalization
            'mean': torch.tensor([0.485, 0.456, 0.406]),
            'std': torch.tensor([0.229, 0.224, 0.225])
        }
        
        # Training episode using these statistics
        episode = self._create_episode(
            classes=[5, 12, 23, 35, 47],
            n_support=5,
            n_query=15,
            split='train'
        )
        episode['normalization_stats'] = augmentation_stats
        
        result = self.guard.comprehensive_leakage_check(episode)
        assert not result['safe']  # Should detect test data in normalization
    
    def test_pretrained_model_contamination(self):
        """Test contamination from pretrained models."""
        # Pretrained model that has seen some test classes
        pretrained_classes = set(range(70)) | {85, 87, 91}  # Includes some test classes
        
        # Test episode
        test_episode_classes = {85, 87, 91, 93, 95}
        
        detector = CrossEpisodeContaminationDetector()
        result = detector.detect_model_contamination(
            pretrained_classes,
            test_episode_classes
        )
        
        assert result['contaminated']
        assert len(result['overlapping_classes']) == 3  # 85, 87, 91
    
    def test_batch_normalization_running_stats_leakage(self):
        """Test leakage through BatchNorm running statistics."""
        # Simulate BatchNorm that computed running stats on mixed data
        model = nn.Sequential(
            nn.Conv2d(3, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # BatchNorm stats computed on train+test data
        bn_layer = model[1]
        bn_layer.running_mean = torch.randn(64)
        bn_layer.running_var = torch.ones(64)
        bn_layer.num_batches_tracked = torch.tensor(1000)
        
        # Add metadata about which classes were used
        bn_stats = {
            'running_mean': bn_layer.running_mean,
            'running_var': bn_layer.running_var,
            'data_classes': self.train_classes | {85, 87, 91}  # Some test classes
        }
        
        monitor = NormalizationLeakageMonitor(self.train_classes, self.test_classes)
        result = monitor.validate_batch_norm_stats(bn_stats, "training")
        
        assert not result['valid']
    
    def _create_episode(self, classes: List[int], n_support: int, n_query: int, split: str) -> Dict:
        """Helper to create episode data."""
        n_way = len(classes)
        
        # Create support set
        support_y = torch.tensor(classes).repeat_interleave(n_support)
        support_x = torch.randn(n_way * n_support, 84, 84, 3)  # Image-like data
        
        # Create query set  
        query_y = torch.tensor(classes).repeat_interleave(n_query)
        query_x = torch.randn(n_way * n_query, 84, 84, 3)
        
        return {
            'support_x': support_x,
            'support_y': support_y,
            'query_x': query_x,
            'query_y': query_y,
            'split': split
        }


class TestPerformanceAndScalability:
    """Test performance and scalability of leakage detection."""
    
    def test_large_scale_validation(self):
        """Test leakage detection on large-scale data."""
        # Large number of classes
        train_classes = set(range(10000))
        test_classes = set(range(10000, 12000))
        
        detector = LeakageDetector(train_classes, test_classes)
        
        # Large episode
        episode_classes = set(range(100))  # 100 training classes
        
        # Should still be fast
        import time
        start_time = time.time()
        
        result = detector.validate_episode_classes(episode_classes, 'train')
        
        elapsed = time.time() - start_time
        
        # Should complete in reasonable time (< 1 second)
        assert elapsed < 1.0
        assert result['valid']
    
    def test_memory_efficiency(self):
        """Test memory efficiency of leakage detection."""
        train_classes = set(range(1000))
        test_classes = set(range(1000, 2000))
        
        guard = DataLeakageGuard(train_classes, test_classes)
        
        # Process many episodes
        for i in range(100):
            episode_data = {
                'support_x': torch.randn(25, 10),
                'support_y': torch.randint(0, 1000, (25,)),
                'query_x': torch.randn(75, 10),
                'query_y': torch.randint(0, 1000, (75,)),
                'split': 'train'
            }
            
            result = guard.comprehensive_leakage_check(episode_data)
            
            # Should not accumulate memory between episodes
            assert result is not None


if __name__ == "__main__":
    pytest.main([__file__])