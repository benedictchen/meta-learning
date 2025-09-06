#!/usr/bin/env python3
"""Test suite for enhanced learnability analyzer with hardness metrics integration"""

import torch
import torch.nn as nn
import numpy as np
import pytest
from meta_learning.core.episode import Episode
from meta_learning.evaluation.enhanced_learnability import EnhancedLearnabilityAnalyzer


class TestEnhancedLearnabilityAnalyzer:
    """Test enhanced learnability analyzer functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = EnhancedLearnabilityAnalyzer()
        
    def create_test_episode(self, n_classes=3, n_shots=5, feature_dim=64, difficulty='medium'):
        """Create test episode with controlled difficulty."""
        np.random.seed(42)
        torch.manual_seed(42)
        
        support_x_list = []
        support_y_list = []
        
        for class_idx in range(n_classes):
            if difficulty == 'easy':
                # Well-separated classes
                center = torch.randn(feature_dim) * 3 + class_idx * 5
                noise_scale = 0.1
            elif difficulty == 'hard':
                # Overlapping classes
                center = torch.randn(feature_dim) * 0.5
                noise_scale = 2.0
            else:  # medium
                center = torch.randn(feature_dim) * 2 + class_idx * 2
                noise_scale = 0.5
            
            class_samples = center.unsqueeze(0) + torch.randn(n_shots, feature_dim) * noise_scale
            support_x_list.append(class_samples)
            support_y_list.extend([class_idx] * n_shots)
        
        support_x = torch.cat(support_x_list, dim=0)
        support_y = torch.tensor(support_y_list)
        
        # Create query set
        query_x = torch.randn(n_classes * 2, feature_dim)
        query_y = torch.tensor([i for i in range(n_classes)] * 2)
        
        return Episode(support_x, support_y, query_x, query_y)
    
    def test_enhanced_task_difficulty_computation(self):
        """Test enhanced task difficulty computation."""
        episode = self.create_test_episode()
        
        metrics = self.analyzer.compute_enhanced_task_difficulty(episode)
        
        # Verify all expected metrics are present
        expected_metrics = [
            'class_balance', 'avg_feature_distance', 'intra_class_variance',
            'inter_class_separation', 'difficulty_score', 'hardness_score',
            'composite_difficulty', 'separability_ratio', 'task_complexity_profile'
        ]
        
        for metric in expected_metrics:
            assert metric in metrics, f"Missing metric: {metric}"
            assert isinstance(metrics[metric], (float, dict)), f"Invalid type for {metric}"
        
        # Verify reasonable value ranges
        assert 0.0 <= metrics['hardness_score'] <= 1.0, "Hardness score out of range"
        assert 0.0 <= metrics['composite_difficulty'] <= 1.0, "Composite difficulty out of range"
        assert 0.0 <= metrics['separability_ratio'] <= 1.0, "Separability ratio out of range"
        
        # Verify complexity profile structure
        profile = metrics['task_complexity_profile']
        expected_profile_keys = [
            'dimensionality_complexity', 'support_complexity', 'class_count_complexity',
            'shot_count_complexity', 'distribution_gap'
        ]
        for key in expected_profile_keys:
            assert key in profile, f"Missing profile key: {key}"
            assert 0.0 <= profile[key] <= 1.0, f"Profile value out of range: {key}"
    
    def test_hardness_integration(self):
        """Test that hardness metric is properly integrated."""
        easy_episode = self.create_test_episode(difficulty='easy')
        hard_episode = self.create_test_episode(difficulty='hard')
        
        easy_metrics = self.analyzer.compute_enhanced_task_difficulty(easy_episode)
        hard_metrics = self.analyzer.compute_enhanced_task_difficulty(hard_episode)
        
        # Hard episode should have higher hardness and difficulty scores
        assert hard_metrics['hardness_score'] > easy_metrics['hardness_score'], \
            "Hard episode should have higher hardness score"
        assert hard_metrics['composite_difficulty'] > easy_metrics['composite_difficulty'], \
            "Hard episode should have higher composite difficulty"
    
    def test_separability_ratio_computation(self):
        """Test separability ratio computation."""
        # Create episode with known separability
        well_separated = self.create_test_episode(difficulty='easy')
        poorly_separated = self.create_test_episode(difficulty='hard')
        
        well_sep_metrics = self.analyzer.compute_enhanced_task_difficulty(well_separated)
        poor_sep_metrics = self.analyzer.compute_enhanced_task_difficulty(poorly_separated)
        
        # Well-separated episode should have higher separability ratio
        assert well_sep_metrics['separability_ratio'] > poor_sep_metrics['separability_ratio'], \
            "Well-separated episode should have higher separability ratio"
    
    def test_hardness_distribution_analysis(self):
        """Test hardness distribution analysis across episodes."""
        episodes = []
        for difficulty in ['easy', 'medium', 'hard']:
            for _ in range(5):
                episodes.append(self.create_test_episode(difficulty=difficulty))
        
        analysis = self.analyzer.analyze_hardness_distribution(episodes)
        
        # Verify structure
        assert 'hardness_statistics' in analysis
        assert 'difficulty_statistics' in analysis
        assert 'separability_statistics' in analysis
        assert 'episode_count' in analysis
        
        # Verify statistics
        hardness_stats = analysis['hardness_statistics']
        assert 0.0 <= hardness_stats['mean'] <= 1.0
        assert hardness_stats['std'] >= 0.0
        assert hardness_stats['min'] <= hardness_stats['mean'] <= hardness_stats['max']
        
        # Verify percentiles
        percentiles = hardness_stats['percentiles']
        assert percentiles['25'] <= percentiles['50'] <= percentiles['75']
        
        # Verify correlation computations don't fail
        difficulty_stats = analysis['difficulty_statistics']
        assert 'correlation_with_hardness' in difficulty_stats
        assert not np.isnan(difficulty_stats['correlation_with_hardness'])
    
    def test_curriculum_ordering(self):
        """Test curriculum learning order generation."""
        episodes = []
        expected_difficulties = []
        
        # Create episodes with known difficulty ordering
        for difficulty in ['easy', 'easy', 'medium', 'medium', 'hard', 'hard']:
            episodes.append(self.create_test_episode(difficulty=difficulty))
            expected_difficulties.append(difficulty)
        
        # Test gradual strategy (easy to hard)
        gradual_order = self.analyzer.generate_curriculum_ordering(episodes, strategy='gradual')
        assert len(gradual_order) == len(episodes)
        assert all(idx in gradual_order for idx in range(len(episodes)))
        
        # Verify ordering is roughly correct (first episodes should be easier)
        first_half = gradual_order[:len(episodes)//2]
        second_half = gradual_order[len(episodes)//2:]
        
        # Compute average difficulty for each half
        first_difficulties = [self.analyzer.compute_enhanced_task_difficulty(episodes[i])['composite_difficulty'] 
                             for i in first_half]
        second_difficulties = [self.analyzer.compute_enhanced_task_difficulty(episodes[i])['composite_difficulty'] 
                              for i in second_half]
        
        assert np.mean(first_difficulties) < np.mean(second_difficulties), \
            "Gradual curriculum should have easier episodes first"
        
        # Test hard_first strategy
        hard_first_order = self.analyzer.generate_curriculum_ordering(episodes, strategy='hard_first')
        assert len(hard_first_order) == len(episodes)
        
        # Test mixed strategy
        mixed_order = self.analyzer.generate_curriculum_ordering(episodes, strategy='mixed')
        assert len(mixed_order) == len(episodes)
        assert mixed_order != gradual_order  # Should be different from gradual
    
    def test_task_difficulty_comparison(self):
        """Test comparison between episode sets."""
        easy_episodes = [self.create_test_episode(difficulty='easy') for _ in range(10)]
        hard_episodes = [self.create_test_episode(difficulty='hard') for _ in range(10)]
        
        comparison = self.analyzer.compare_task_difficulties(easy_episodes, hard_episodes)
        
        # Verify structure
        assert 'set1_analysis' in comparison
        assert 'set2_analysis' in comparison
        assert 'comparison' in comparison
        
        comp_stats = comparison['comparison']
        assert 'mean_difference' in comp_stats
        assert 't_statistic' in comp_stats
        assert 'effect_size' in comp_stats
        assert 'harder_set' in comp_stats
        
        # Hard episodes should be identified as harder
        assert comp_stats['harder_set'] == 2, "Hard episodes should be identified as set 2 (harder)"
        assert comp_stats['mean_difference'] < 0, "Easy episodes (set 1) should have lower mean difficulty"
    
    def test_difficulty_recommendations(self):
        """Test task difficulty recommendations."""
        easy_episode = self.create_test_episode(difficulty='easy')
        hard_episode = self.create_test_episode(difficulty='hard')
        
        easy_rec = self.analyzer.get_task_difficulty_recommendations(easy_episode)
        hard_rec = self.analyzer.get_task_difficulty_recommendations(hard_episode)
        
        # Verify structure
        for rec in [easy_rec, hard_rec]:
            assert 'metrics' in rec
            assert 'recommendations' in rec
            
            recs = rec['recommendations']
            assert 'algorithm_preference' in recs
            assert 'suggested_algorithms' in recs
            assert 'reasoning' in recs
        
        # Easy episode should prefer simpler algorithms
        assert 'simple' in easy_rec['recommendations']['algorithm_preference'] or \
               'ridge_regression' in easy_rec['recommendations']['suggested_algorithms'], \
               "Easy episodes should prefer simpler algorithms"
        
        # Hard episode should prefer complex algorithms or have high attention requirement
        hard_recs = hard_rec['recommendations']
        assert 'complex' in hard_recs.get('algorithm_preference', '') or \
               'maml' in hard_recs.get('suggested_algorithms', []) or \
               hard_recs.get('attention_required', '') == 'high', \
               "Hard episodes should prefer complex algorithms or require high attention"
    
    def test_complexity_profile_computation(self):
        """Test complexity profile computation."""
        # Test different episode characteristics
        small_episode = self.create_test_episode(n_classes=2, n_shots=3, feature_dim=32)
        large_episode = self.create_test_episode(n_classes=10, n_shots=10, feature_dim=256)
        
        small_metrics = self.analyzer.compute_enhanced_task_difficulty(small_episode)
        large_metrics = self.analyzer.compute_enhanced_task_difficulty(large_episode)
        
        small_profile = small_metrics['task_complexity_profile']
        large_profile = large_metrics['task_complexity_profile']
        
        # Large episode should have higher complexity in some dimensions
        assert large_profile['dimensionality_complexity'] >= small_profile['dimensionality_complexity'], \
            "Higher dimensional episode should have higher dimensionality complexity"
        assert large_profile['class_count_complexity'] > small_profile['class_count_complexity'], \
            "More classes should result in higher class count complexity"
    
    def test_composite_difficulty_weighting(self):
        """Test that composite difficulty properly weights different factors."""
        episode = self.create_test_episode()
        
        # Test internal composite calculation
        base_metrics = self.analyzer.compute_task_difficulty(episode)
        num_classes = len(torch.unique(episode.support_y))
        hardness_score = 0.8  # High hardness
        
        composite = self.analyzer._compute_composite_difficulty(base_metrics, hardness_score)
        
        assert 0.0 <= composite <= 1.0, "Composite difficulty should be normalized"
        assert composite > 0.3, "High hardness should result in higher composite difficulty (0.4 * 0.8 = 0.32 minimum)"


if __name__ == "__main__":
    # Run basic functionality test
    analyzer = EnhancedLearnabilityAnalyzer()
    
    # Test episode creation
    episode = Episode(
        support_x=torch.randn(15, 64),
        support_y=torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]),
        query_x=torch.randn(6, 64),
        query_y=torch.tensor([0, 0, 1, 1, 2, 2])
    )
    
    print("Testing Enhanced LearnabilityAnalyzer...")
    
    # Test enhanced metrics computation
    try:
        metrics = analyzer.compute_enhanced_task_difficulty(episode)
        print(f"✅ Enhanced metrics computed: {list(metrics.keys())}")
        print(f"   Hardness score: {metrics['hardness_score']:.3f}")
        print(f"   Composite difficulty: {metrics['composite_difficulty']:.3f}")
        print(f"   Separability ratio: {metrics['separability_ratio']:.3f}")
    except Exception as e:
        print(f"❌ Enhanced metrics computation failed: {e}")
    
    # Test recommendations
    try:
        recommendations = analyzer.get_task_difficulty_recommendations(episode)
        print(f"✅ Recommendations generated")
        print(f"   Algorithm preference: {recommendations['recommendations']['algorithm_preference']}")
        print(f"   Suggested algorithms: {recommendations['recommendations']['suggested_algorithms']}")
    except Exception as e:
        print(f"❌ Recommendations failed: {e}")
    
    print("Basic functionality test completed!")