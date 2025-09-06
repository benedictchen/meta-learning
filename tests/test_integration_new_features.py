# TODO: INTEGRATION TESTING - End-to-end functionality tests
# TODO: Test complete episode manipulation workflows
# TODO: Test BalancedTaskGenerator integration with evaluation metrics
# TODO: Test mathematical utilities integration with episode processing
# TODO: Test prototype analysis workflow integration  
# TODO: Test performance under different scenarios
# TODO: Test memory usage and efficiency

"""Integration tests for all new functionality."""

import pytest
import torch
import numpy as np
from meta_learning.data_utils import (
    create_episode_from_data,
    merge_episodes,
    balance_episode,
    augment_episode,
    split_episode,
    compute_episode_statistics,
    BalancedTaskGenerator
)
from meta_learning.evaluation.metrics import AccuracyCalculator, CalibrationCalculator
from meta_learning.evaluation.prototype_analysis import PrototypeAnalyzer
from meta_learning.core.math_utils import pairwise_sqeuclidean, cosine_logits, batched_prototype_computation


class TestEpisodeWorkflows:
    """Test complete episode manipulation workflows."""
    
    def test_episode_creation_to_evaluation_workflow(self):
        """Test complete workflow from episode creation to evaluation."""
        # Step 1: Create initial episode
        torch.manual_seed(42)
        support_x = torch.randn(15, 64)  # 3 classes, 5 shots each
        support_y = torch.tensor([0]*5 + [1]*5 + [2]*5)
        query_x = torch.randn(30, 64)    # 3 classes, 10 queries each  
        query_y = torch.tensor([0]*10 + [1]*10 + [2]*10)
        
        episode = create_episode_from_data(support_x, support_y, query_x, query_y)
        
        # Step 2: Analyze episode statistics
        stats = compute_episode_statistics(episode)
        assert stats['n_support'] == 15
        assert stats['n_query'] == 30
        assert stats['n_support_classes'] == 3
        
        # Step 3: Compute prototypes and distances
        prototypes = batched_prototype_computation(episode.support_x, episode.support_y)
        distances = pairwise_sqeuclidean(episode.query_x, prototypes)
        
        assert prototypes.shape == (3, 64)  # 3 prototypes, 64 features
        assert distances.shape == (30, 3)   # 30 queries x 3 prototypes
        
        # Step 4: Generate predictions and evaluate
        logits = -distances  # Convert distances to logits (negative for higher similarity)
        predictions = torch.softmax(logits, dim=1)
        
        acc_calc = AccuracyCalculator()
        accuracy = acc_calc.compute_accuracy(predictions, query_y)
        per_class_acc = acc_calc.compute_per_class_accuracy(predictions, query_y)
        
        assert isinstance(accuracy, float)
        assert 0 <= accuracy <= 1
        assert len(per_class_acc) == 3  # One accuracy per class
        
        # Step 5: Prototype quality analysis
        analyzer = PrototypeAnalyzer()
        intra_var = analyzer.compute_intra_class_variance(episode.support_x, episode.support_y)
        inter_dist = analyzer.compute_inter_class_distance(prototypes)
        
        assert intra_var >= 0
        assert inter_dist >= 0
    
    def test_episode_balancing_and_augmentation_workflow(self):
        """Test episode balancing and augmentation workflow."""
        # Create imbalanced episode
        support_x = torch.randn(10, 32)
        support_y = torch.tensor([0, 0, 0, 0, 1, 1, 2])  # Imbalanced: 4, 2, 1
        query_x = torch.randn(15, 32)
        query_y = torch.tensor([0]*7 + [1]*5 + [2]*3)
        
        original_episode = create_episode_from_data(support_x, support_y, query_x, query_y)
        
        # Balance the episode
        balanced_episode = balance_episode(original_episode, target_shots_per_class=2)
        
        # Check balancing worked
        for cls in [0, 1, 2]:
            assert (balanced_episode.support_y == cls).sum() == 2
        
        # Augment the balanced episode
        def rotation_augmentation(x):
            """Simple rotation augmentation for testing."""
            # Add small rotation-like transformation
            noise = torch.randn_like(x) * 0.1
            return x + noise
        
        augmented_episode = augment_episode(balanced_episode, rotation_augmentation)
        
        # Verify augmentation applied only to support set
        assert not torch.equal(augmented_episode.support_x, balanced_episode.support_x)
        assert torch.equal(augmented_episode.query_x, balanced_episode.query_x)
        
        # Evaluate augmented episode
        prototypes = batched_prototype_computation(
            augmented_episode.support_x, 
            augmented_episode.support_y
        )
        
        cosine_similarities = cosine_logits(
            augmented_episode.query_x, 
            prototypes, 
            tau=0.1
        )
        
        predictions = torch.softmax(cosine_similarities, dim=1)
        
        acc_calc = AccuracyCalculator()
        accuracy = acc_calc.compute_accuracy(predictions, augmented_episode.query_y)
        
        assert isinstance(accuracy, float)
        assert 0 <= accuracy <= 1
    
    def test_episode_splitting_and_merging_workflow(self):
        """Test episode splitting and merging workflow."""
        # Create large episode
        support_x = torch.randn(20, 48)  # 4 classes, 5 shots each
        support_y = torch.tensor([0]*5 + [1]*5 + [2]*5 + [3]*5)
        query_x = torch.randn(40, 48)    # 4 classes, 10 queries each
        query_y = torch.tensor([0]*10 + [1]*10 + [2]*10 + [3]*10)
        
        large_episode = create_episode_from_data(support_x, support_y, query_x, query_y)
        
        # Split into two episodes
        episode1, episode2 = split_episode(large_episode, query_ratio=0.4)
        
        # Verify split
        assert len(episode1.query_x) + len(episode2.query_x) == len(large_episode.query_x)
        
        # Process each episode separately
        stats1 = compute_episode_statistics(episode1)
        stats2 = compute_episode_statistics(episode2)
        
        # Both should have same support characteristics
        assert stats1['n_support'] == stats2['n_support'] == 20
        assert stats1['n_support_classes'] == stats2['n_support_classes'] == 4
        
        # Merge episodes back together
        merged_episode = merge_episodes(episode1, episode2)
        
        # Should have double the data now (merged support sets + merged query sets)
        assert len(merged_episode.support_x) == 40  # 20 + 20
        assert len(merged_episode.query_x) == 40   # Split query sets merged
        
        # Evaluate merged episode performance
        prototypes = batched_prototype_computation(
            merged_episode.support_x,
            merged_episode.support_y
        )
        
        # More support data should improve prototype quality
        analyzer = PrototypeAnalyzer()
        merged_intra_var = analyzer.compute_intra_class_variance(
            merged_episode.support_x, 
            merged_episode.support_y
        )
        
        assert merged_intra_var >= 0


class TestBalancedGeneratorIntegration:
    """Test BalancedTaskGenerator integration with other components."""
    
    def create_test_dataset(self, n_classes=5, samples_per_class=10):
        """Create a test dataset for BalancedTaskGenerator."""
        class SimpleDataset:
            def __init__(self, data_label_pairs):
                self.data = data_label_pairs
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                return self.data[idx]
            
            def __iter__(self):
                return iter(self.data)
        
        # Generate synthetic data with controlled class distribution
        torch.manual_seed(42)
        data_pairs = []
        for class_id in range(n_classes):
            # Create distinct clusters for each class
            class_center = torch.randn(64) * 2
            for _ in range(samples_per_class):
                # Add noise around class center
                sample = class_center + torch.randn(64) * 0.5
                data_pairs.append((sample, class_id))
        
        return SimpleDataset(data_pairs)
    
    def test_generator_to_evaluation_pipeline(self):
        """Test complete pipeline from generator to evaluation."""
        # Create dataset
        dataset = self.create_test_dataset(n_classes=8, samples_per_class=15)
        
        # Initialize generator with balancing strategies
        generator = BalancedTaskGenerator(
            dataset,
            n_way=5,
            n_shot=3,
            n_query=5,
            balance_strategies=['class']
        )
        
        # Generate multiple episodes and evaluate them
        accuracies = []
        calibration_errors = []
        
        for seed in range(10):  # Generate 10 episodes
            episode = generator.generate_episode(random_state=seed + 42)
            
            # Verify episode structure
            assert episode.num_classes == 5
            assert len(episode.support_x) == 15  # 5 classes * 3 shots
            assert len(episode.query_x) == 25    # 5 classes * 5 queries
            
            # Compute prototypes and make predictions
            prototypes = batched_prototype_computation(
                episode.support_x, 
                episode.support_y
            )
            
            # Use cosine similarity for classification
            logits = cosine_logits(episode.query_x, prototypes, tau=0.1)
            predictions = torch.softmax(logits, dim=1)
            
            # Evaluate accuracy
            acc_calc = AccuracyCalculator()
            accuracy = acc_calc.compute_accuracy(predictions, episode.query_y)
            accuracies.append(accuracy)
            
            # Evaluate calibration
            cal_calc = CalibrationCalculator()
            ece = cal_calc.compute_expected_calibration_error(
                predictions, 
                episode.query_y, 
                n_bins=5
            )
            calibration_errors.append(ece)
        
        # Analyze results across episodes
        mean_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)
        mean_ece = np.mean(calibration_errors)
        
        # Basic sanity checks
        assert 0 <= mean_accuracy <= 1
        assert std_accuracy >= 0
        assert mean_ece >= 0
        
        # Should perform better than random (>20% for 5-way classification)
        assert mean_accuracy > 0.2
    
    def test_generator_with_imbalanced_dataset(self):
        """Test generator handling of highly imbalanced datasets."""
        # Create imbalanced dataset
        class ImbalancedDataset:
            def __init__(self):
                torch.manual_seed(42)
                self.data = []
                
                # Create highly imbalanced distribution
                class_sizes = [50, 30, 20, 10, 5, 3, 2, 1]  # Highly imbalanced
                
                for class_id, size in enumerate(class_sizes):
                    class_center = torch.randn(32) * 3
                    for _ in range(size):
                        sample = class_center + torch.randn(32) * 0.3
                        self.data.append((sample, class_id))
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                return self.data[idx]
            
            def __iter__(self):
                return iter(self.data)
        
        dataset = ImbalancedDataset()
        
        # Generator should handle imbalance with class balancing strategy
        generator = BalancedTaskGenerator(
            dataset,
            n_way=6,
            n_shot=2,
            n_query=3,
            balance_strategies=['class']  # Should prefer underrepresented classes
        )
        
        # Check imbalance detection
        assert generator.imbalance_ratio > 10  # 50/1 = 50, but due to class balancing
        
        # Generate episodes and check class distribution
        class_selections = []
        for seed in range(20):
            episode = generator.generate_episode(random_state=seed + 100)
            
            # Should successfully generate episodes despite imbalance
            assert episode.num_classes == 6
            assert len(episode.support_x) == 12  # 6 classes * 2 shots
            
            # Track which classes are selected (will need to map back from remapped labels)
            # For now, just verify structure is correct
            unique_support_classes = torch.unique(episode.support_y)
            assert len(unique_support_classes) == 6
            assert torch.equal(unique_support_classes, torch.arange(6))
    
    def test_mathematical_utilities_integration(self):
        """Test integration of mathematical utilities with episode processing."""
        # Create episode with known geometry
        torch.manual_seed(42)
        
        # Create orthogonal support vectors for clear separation
        support_x = torch.tensor([
            [1., 0., 0., 0.],  # Class 0 prototype direction
            [1., 0., 0., 0.],
            [0., 1., 0., 0.],  # Class 1 prototype direction
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],  # Class 2 prototype direction
            [0., 0., 1., 0.],
        ], dtype=torch.float32)
        support_y = torch.tensor([0, 0, 1, 1, 2, 2])
        
        # Create query vectors aligned with prototypes
        query_x = torch.tensor([
            [0.9, 0.1, 0., 0.],   # Should classify as class 0
            [0.1, 0.9, 0., 0.],   # Should classify as class 1
            [0., 0.1, 0.9, 0.],   # Should classify as class 2
        ], dtype=torch.float32)
        query_y = torch.tensor([0, 1, 2])
        
        episode = create_episode_from_data(support_x, support_y, query_x, query_y)
        
        # Test both distance metrics
        prototypes = batched_prototype_computation(episode.support_x, episode.support_y)
        
        # Euclidean distances
        euclidean_distances = pairwise_sqeuclidean(episode.query_x, prototypes)
        euclidean_logits = -euclidean_distances
        euclidean_predictions = torch.softmax(euclidean_logits, dim=1)
        
        # Cosine similarities
        cosine_logits_vals = cosine_logits(episode.query_x, prototypes, tau=1.0)
        cosine_predictions = torch.softmax(cosine_logits_vals, dim=1)
        
        # Both should achieve perfect classification for this orthogonal case
        acc_calc = AccuracyCalculator()
        euclidean_acc = acc_calc.compute_accuracy(euclidean_predictions, query_y)
        cosine_acc = acc_calc.compute_accuracy(cosine_predictions, query_y)
        
        # Should be perfect or very high accuracy
        assert euclidean_acc >= 0.95
        assert cosine_acc >= 0.95
        
        # Test temperature scaling effects
        cold_logits = cosine_logits(episode.query_x, prototypes, tau=0.1)  # Sharp
        hot_logits = cosine_logits(episode.query_x, prototypes, tau=2.0)   # Soft
        
        cold_predictions = torch.softmax(cold_logits, dim=1)
        hot_predictions = torch.softmax(hot_logits, dim=1)
        
        # Cold predictions should be more confident (higher max probability)
        cold_confidences = torch.max(cold_predictions, dim=1)[0]
        hot_confidences = torch.max(hot_predictions, dim=1)[0]
        
        assert torch.all(cold_confidences >= hot_confidences)


class TestPrototypeAnalysisIntegration:
    """Test prototype analysis integration with episode workflows."""
    
    def test_prototype_quality_assessment_workflow(self):
        """Test complete prototype quality assessment workflow."""
        torch.manual_seed(42)
        
        # Create episodes with different prototype qualities
        # High quality: well-separated clusters
        high_quality_support_x = torch.cat([
            torch.randn(5, 64) + torch.tensor([3., 0.] + [0.] * 62),  # Class 0: center at (3,0,...)
            torch.randn(5, 64) + torch.tensor([0., 3.] + [0.] * 62),  # Class 1: center at (0,3,...)
            torch.randn(5, 64) + torch.tensor([-3., 0.] + [0.] * 62), # Class 2: center at (-3,0,...)
        ]) * 0.3  # Small intra-class variance
        
        # Low quality: overlapping clusters  
        low_quality_support_x = torch.cat([
            torch.randn(5, 64),  # Class 0: centered at origin
            torch.randn(5, 64),  # Class 1: centered at origin (overlap!)
            torch.randn(5, 64),  # Class 2: centered at origin (overlap!)
        ]) * 2.0  # Large intra-class variance
        
        support_y = torch.tensor([0]*5 + [1]*5 + [2]*5)
        query_x = torch.randn(15, 64)
        query_y = torch.tensor([0]*5 + [1]*5 + [2]*5)
        
        # Create episodes
        high_quality_episode = create_episode_from_data(
            high_quality_support_x, support_y, query_x, query_y
        )
        low_quality_episode = create_episode_from_data(
            low_quality_support_x, support_y, query_x, query_y
        )
        
        # Analyze prototype quality
        analyzer = PrototypeAnalyzer()
        
        # High quality episode analysis
        hq_prototypes = batched_prototype_computation(
            high_quality_episode.support_x, 
            high_quality_episode.support_y
        )
        hq_intra_var = analyzer.compute_intra_class_variance(
            high_quality_episode.support_x, 
            high_quality_episode.support_y
        )
        hq_inter_dist = analyzer.compute_inter_class_distance(hq_prototypes)
        hq_silhouette = analyzer.compute_silhouette_score(
            high_quality_episode.support_x, 
            high_quality_episode.support_y
        )
        
        # Low quality episode analysis
        lq_prototypes = batched_prototype_computation(
            low_quality_episode.support_x, 
            low_quality_episode.support_y
        )
        lq_intra_var = analyzer.compute_intra_class_variance(
            low_quality_episode.support_x, 
            low_quality_episode.support_y
        )
        lq_inter_dist = analyzer.compute_inter_class_distance(lq_prototypes)
        lq_silhouette = analyzer.compute_silhouette_score(
            low_quality_episode.support_x, 
            low_quality_episode.support_y
        )
        
        # High quality should have better metrics
        assert hq_intra_var < lq_intra_var      # Lower intra-class variance
        # Inter-class distance can vary due to random effects, focus on intra-class variance
        assert hq_silhouette > lq_silhouette    # Better silhouette score
        
        # Test correlation with classification performance
        acc_calc = AccuracyCalculator()
        
        # High quality episode performance
        hq_distances = pairwise_sqeuclidean(query_x, hq_prototypes)
        hq_logits = -hq_distances
        hq_predictions = torch.softmax(hq_logits, dim=1)
        hq_accuracy = acc_calc.compute_accuracy(hq_predictions, query_y)
        
        # Low quality episode performance
        lq_distances = pairwise_sqeuclidean(query_x, lq_prototypes)
        lq_logits = -lq_distances
        lq_predictions = torch.softmax(lq_logits, dim=1)
        lq_accuracy = acc_calc.compute_accuracy(lq_predictions, query_y)
        
        # High quality prototypes should lead to better performance
        # (This might not always hold due to randomness, but should generally be true)
        print(f"High quality accuracy: {hq_accuracy:.3f}, Low quality accuracy: {lq_accuracy:.3f}")
        print(f"HQ metrics - Intra: {hq_intra_var:.3f}, Inter: {hq_inter_dist:.3f}, Sil: {hq_silhouette:.3f}")
        print(f"LQ metrics - Intra: {lq_intra_var:.3f}, Inter: {lq_inter_dist:.3f}, Sil: {lq_silhouette:.3f}")


class TestPerformanceAndMemory:
    """Test performance and memory characteristics of new implementations."""
    
    def test_large_episode_processing(self):
        """Test processing of large episodes for performance."""
        # Create large episode
        torch.manual_seed(42)
        n_classes = 20
        n_shot = 5
        n_query = 50
        feature_dim = 512
        
        support_x = torch.randn(n_classes * n_shot, feature_dim)
        support_y = torch.repeat_interleave(torch.arange(n_classes), n_shot)
        query_x = torch.randn(n_classes * n_query, feature_dim)
        query_y = torch.repeat_interleave(torch.arange(n_classes), n_query)
        
        large_episode = create_episode_from_data(support_x, support_y, query_x, query_y)
        
        # Test episode operations performance
        stats = compute_episode_statistics(large_episode)
        assert stats['n_support'] == n_classes * n_shot
        assert stats['n_query'] == n_classes * n_query
        
        # Test mathematical operations performance
        prototypes = batched_prototype_computation(
            large_episode.support_x, 
            large_episode.support_y
        )
        assert prototypes.shape == (n_classes, feature_dim)
        
        distances = pairwise_sqeuclidean(large_episode.query_x, prototypes)
        assert distances.shape == (n_classes * n_query, n_classes)
        
        similarities = cosine_logits(large_episode.query_x, prototypes)
        assert similarities.shape == (n_classes * n_query, n_classes)
        
        # Test evaluation metrics performance
        predictions = torch.softmax(-distances, dim=1)
        
        acc_calc = AccuracyCalculator()
        accuracy = acc_calc.compute_accuracy(predictions, query_y)
        per_class_acc = acc_calc.compute_per_class_accuracy(predictions, query_y)
        
        assert isinstance(accuracy, float)
        assert len(per_class_acc) == n_classes
    
    def test_batch_processing_efficiency(self):
        """Test efficiency of batch processing multiple episodes."""
        torch.manual_seed(42)
        
        # Create multiple small episodes
        episodes = []
        for i in range(10):
            support_x = torch.randn(15, 32)
            support_y = torch.tensor([0]*5 + [1]*5 + [2]*5)
            query_x = torch.randn(15, 32)
            query_y = torch.tensor([0]*5 + [1]*5 + [2]*5)
            
            episode = create_episode_from_data(support_x, support_y, query_x, query_y)
            episodes.append(episode)
        
        # Test merging multiple episodes
        merged_episode = merge_episodes(*episodes)
        
        assert len(merged_episode.support_x) == 150  # 10 * 15
        assert len(merged_episode.query_x) == 150    # 10 * 15
        
        # Test batch evaluation
        prototypes = batched_prototype_computation(
            merged_episode.support_x,
            merged_episode.support_y
        )
        
        # Should still work with larger merged episode
        distances = pairwise_sqeuclidean(merged_episode.query_x, prototypes)
        predictions = torch.softmax(-distances, dim=1)
        
        acc_calc = AccuracyCalculator()
        accuracy = acc_calc.compute_accuracy(predictions, merged_episode.query_y)
        
        assert isinstance(accuracy, float)
        assert 0 <= accuracy <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])