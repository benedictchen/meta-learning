"""
Comprehensive Test Suite for 100% Coverage
==========================================

This test file is designed to achieve 100% code coverage by systematically testing
all implemented functionality in the meta-learning package.

Author: Benedict Chen (benedict@benedictchen.com)
"""

import pytest
import torch
# DEMO/TEST REPRODUCIBILITY: Set seed for consistent synthetic data
torch.manual_seed(42)  # Reproducible test/demo data
import torch.nn as nn
# DEMO/TEST REPRODUCIBILITY: Set seed for consistent synthetic data
torch.manual_seed(42)  # Reproducible test/demo data
import numpy as np
from typing import Dict, List, Tuple, Optional
from unittest.mock import Mock, patch

# Core imports - all verified to work
from meta_learning.meta_learning_modules import (
    # Test-time compute
    TestTimeComputeScaler, TestTimeComputeConfig,
    
    # MAML variants
    MAMLLearner, MAMLConfig, FirstOrderMAML, ReptileLearner,
    ANILLearner, BOILLearner, MAMLenLLM, MAMLenLLMConfig,
    
    # Few-shot learning
    PrototypicalNetworks, PrototypicalConfig,
    MatchingNetworks, MatchingConfig,
    RelationNetworks, RelationConfig,
    
    # Continual learning
    OnlineMetaLearner, ContinualMetaConfig,
    
    # Utilities
    MetaLearningDataset, TaskConfiguration, EvaluationConfig,
    DatasetConfig, MetricsConfig, StatsConfig, CurriculumConfig, DiversityConfig,
    EvaluationMetrics, StatisticalAnalysis, CurriculumLearning, TaskDiversityTracker,
    
    # Hardware utilities
    HardwareManager, HardwareConfig, MultiGPUManager,
)

class TestComprehensiveCoverage:
    """Comprehensive tests for 100% coverage."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")  # Always use CPU for tests
        self.n_way = 5
        self.k_shot = 5
        self.q_query = 15
        self.input_dim = 84 * 84 * 3  # Standard image size
        self.hidden_dim = 64
        self.embedding_dim = 64
        
        # Create sample data
        self.support_data = torch.randn(self.n_way * self.k_shot, 3, 84, 84)
        self.support_labels = torch.repeat_interleave(torch.arange(self.n_way), self.k_shot)
        self.query_data = torch.randn(self.n_way * self.q_query, 3, 84, 84)
        self.query_labels = torch.repeat_interleave(torch.arange(self.n_way), self.q_query)
        
        # Create larger dataset for meta-learning
        self.dataset_size = 1000
        self.n_classes = 20
        self.full_data = torch.randn(self.dataset_size, 3, 84, 84)
        self.full_labels = torch.randint(0, self.n_classes, (self.dataset_size,))

    # ==========================================================================
    # Test-Time Compute Scaling Tests
    # ==========================================================================
    
    def test_test_time_compute_config_comprehensive(self):
        """Test all TestTimeComputeConfig configurations."""
        # Test all strategies
        strategies = ["basic", "snell2024", "akyurek2024", "openai_o1", "hybrid"]
        
        for strategy in strategies:
            config = TestTimeComputeConfig(
                compute_strategy=strategy,
                max_compute_budget=50,
                use_process_reward_model=True,
                use_optimal_allocation=True,
                confidence_threshold=0.8
            )
            
            assert config.compute_strategy == strategy
            assert config.max_compute_budget == 50
            assert config.use_process_reward_model is True
            assert config.confidence_threshold == 0.8
    
    def test_test_time_compute_scaler_all_strategies(self):
        """Test TestTimeComputeScaler with all strategies."""
        
        # Create a simple base model
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 16, 3)
                self.pool = nn.AdaptiveAvgPool2d(1)
                self.fc = nn.Linear(16, 5)
            
            def forward(self, support_set, support_labels, query_set):
                # Process support set to learn prototypes
                support_features = self.pool(self.conv(support_set)).flatten(1)
                query_features = self.pool(self.conv(query_set)).flatten(1)
                
                # Simple prototype computation
                prototypes = []
                for class_id in range(5):  # 5-way
                    class_mask = support_labels == class_id
                    if class_mask.any():
                        prototype = support_features[class_mask].mean(dim=0)
                    else:
                        prototype = torch.zeros(16)
                    prototypes.append(prototype)
                
                prototypes = torch.stack(prototypes)
                
                # Classify by distance to prototypes
                distances = torch.cdist(query_features, prototypes)
                return -distances  # Negative distance as logits
        
        base_model = SimpleModel()
        
        # Test each strategy
        strategies = ["basic", "snell2024", "openai_o1"]
        
        for strategy in strategies:
            config = TestTimeComputeConfig(
                compute_strategy=strategy,
                max_compute_budget=5,  # Small for fast testing
                confidence_threshold=0.5
            )
            
            scaler = TestTimeComputeScaler(base_model, config)
            
            # Test forward pass
            with torch.no_grad():
                output = scaler.scale_compute(
                    self.support_data, self.support_labels, self.query_data
                )
                
                assert output.shape == (75, 5)  # 75 queries, 5 classes
                assert torch.isfinite(output).all()

    # ==========================================================================
    # MAML Variants Tests  
    # ==========================================================================
    
    def test_all_maml_variants_comprehensive(self):
        """Test all MAML variant implementations."""
        
        # Simple model for MAML
        class SimpleMAMLModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Sequential(
                    nn.Linear(84*84*3, 128),
                    nn.ReLU(),
                    nn.Linear(128, 5)
                )
            
            def forward(self, x):
                return self.fc(x.view(x.size(0), -1))
        
        model = SimpleMAMLModel()
        config = MAMLConfig(inner_lr=0.01, meta_lr=0.001, inner_steps=3)
        
        # Test each variant
        variants = [
            ("MAMLLearner", MAMLLearner),
            ("FirstOrderMAML", FirstOrderMAML),
            ("ReptileLearner", ReptileLearner),
            ("ANILLearner", ANILLearner),
            ("BOILLearner", BOILLearner),
        ]
        
        for variant_name, variant_class in variants:
            learner = variant_class(model, config)
            
            # Create a task batch
            task_batch = []
            for _ in range(2):  # Small batch for testing
                task_batch.append((
                    self.support_data[:10],    # Smaller for speed
                    self.support_labels[:10],
                    self.query_data[:10],
                    self.query_labels[:10]
                ))
            
            # Test meta-training step
            results = learner.meta_train_step(task_batch, return_metrics=True)
            
            assert isinstance(results, dict)
            assert 'meta_loss' in results
            assert torch.isfinite(torch.tensor(results['meta_loss']))
    
    def test_maml_en_llm_comprehensive(self):
        """Test MAML-en-LLM implementation."""
        config = MAMLenLLMConfig(
            context_optimization_method='ems',
            synthetic_data_generation=True,
            cross_domain_adaptation=True
        )
        
        # Simple transformer-like model
        class SimpleTransformer(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Linear(100, 64)  # Token embeddings
                self.output = nn.Linear(64, 10)      # Output layer
            
            def forward(self, x):
                return self.output(self.embedding(x))
        
        model = SimpleTransformer()
        maml_llm = MAMLenLLM(model, config)
        
        # Test synthetic data generation
        synthetic_data = maml_llm.generate_synthetic_data(
            task_type="text_classification",
            num_examples=10,
            difficulty="medium"
        )
        
        assert isinstance(synthetic_data, dict)
        assert 'data' in synthetic_data
        assert 'labels' in synthetic_data

    # ==========================================================================
    # Few-Shot Learning Tests
    # ==========================================================================
    
    def test_prototypical_networks_comprehensive(self):
        """Test PrototypicalNetworks with all configurations."""
        
        # Test different configurations
        configs = [
            PrototypicalConfig(
                embedding_dim=64,
                use_squared_euclidean=True,
                use_uncertainty_aware_distances=False
            ),
            PrototypicalConfig(
                embedding_dim=128, 
                use_squared_euclidean=False,
                use_uncertainty_aware_distances=True,
                use_hierarchical_prototypes=True
            )
        ]
        
        for config in configs:
            model = PrototypicalNetworks(config)
            
            # Test forward pass
            results = model.forward(
                self.support_data, self.support_labels, self.query_data[:15],
                return_uncertainty=True
            )
            
            # Check output format
            if isinstance(results, dict):
                assert 'logits' in results
                logits = results['logits']
            else:
                logits = results
            
            assert logits.shape == (15, 5)  # 15 queries, 5 classes
            assert torch.isfinite(logits).all()
    
    def test_matching_networks_comprehensive(self):
        """Test MatchingNetworks implementation."""
        config = MatchingConfig(
            embedding_dim=64,
            attention_method='cosine',
            use_full_context_embeddings=True
        )
        
        model = MatchingNetworks(config)
        
        # Test forward pass
        logits = model.forward(
            self.support_data, self.support_labels, self.query_data[:10]
        )
        
        assert logits.shape == (10, 5)  # 10 queries, 5 classes
        assert torch.isfinite(logits).all()
    
    def test_relation_networks_comprehensive(self):
        """Test RelationNetworks implementation."""
        config = RelationConfig(
            embedding_dim=64,
            relation_dim=8,
            use_relation_attention=True
        )
        
        model = RelationNetworks(config)
        
        # Test forward pass
        logits = model.forward(
            self.support_data, self.support_labels, self.query_data[:10]
        )
        
        assert logits.shape == (10, 5)  # 10 queries, 5 classes
        assert torch.isfinite(logits).all()

    # ==========================================================================
    # Continual Learning Tests
    # ==========================================================================
    
    def test_online_meta_learner_comprehensive(self):
        """Test OnlineMetaLearner implementation."""
        config = ContinualMetaConfig(
            memory_size=100,
            consolidation_strength=0.4,
            memory_consolidation_method='ewc'
        )
        
        learner = OnlineMetaLearner(config)
        
        # Test learning a task
        task_data = {
            'support_x': self.support_data,
            'support_y': self.support_labels,
            'query_x': self.query_data[:15],
            'query_y': self.query_labels[:15],
            'task_name': 'test_task_1'
        }
        
        results = learner.learn_task(task_data)
        
        assert isinstance(results, dict)
        assert 'task_id' in results
        
        # Test evaluation
        eval_results = learner.evaluate_all_tasks()
        
        assert isinstance(eval_results, dict)

    # ==========================================================================
    # Utilities Tests
    # ==========================================================================
    
    def test_meta_learning_dataset_comprehensive(self):
        """Test MetaLearningDataset implementation."""
        task_config = TaskConfiguration(
            n_way=self.n_way,
            k_shot=self.k_shot,
            q_query=self.q_query,
            num_episodes=100
        )
        
        dataset = MetaLearningDataset(
            self.full_data, self.full_labels, task_config
        )
        
        assert len(dataset) == 100
        
        # Test getting an item
        episode = dataset[0]
        
        assert isinstance(episode, dict)
        assert 'support' in episode
        assert 'query' in episode
        assert episode['support']['data'].shape[0] == self.n_way * self.k_shot
        assert episode['query']['data'].shape[0] == self.n_way * self.q_query
    
    def test_evaluation_metrics_comprehensive(self):
        """Test EvaluationMetrics implementation."""
        config = MetricsConfig(
            compute_accuracy=True,
            compute_loss=True,
            compute_uncertainty=True
        )
        
        metrics = EvaluationMetrics(config)
        
        # Test updating metrics
        predictions = torch.randn(20, 5)
        targets = torch.randint(0, 5, (20,))
        loss = 1.5
        
        metrics.update(predictions, targets, loss=loss)
        
        # Test computing summary
        summary = metrics.compute_summary()
        
        assert isinstance(summary, dict)
        assert 'mean_accuracy' in summary
        assert 'mean_loss' in summary
    
    def test_statistical_analysis_comprehensive(self):
        """Test StatisticalAnalysis implementation."""
        config = StatsConfig(
            confidence_level=0.95,
            significance_test='t_test'
        )
        
        analyzer = StatisticalAnalysis(config)
        
        # Test confidence interval computation
        values = [0.8, 0.75, 0.82, 0.78, 0.85, 0.79, 0.81, 0.77]
        mean_val, ci_lower, ci_upper = analyzer.compute_confidence_interval(values)
        
        assert 0.0 <= mean_val <= 1.0
        assert ci_lower <= mean_val <= ci_upper
        
        # Test statistical test
        group1 = [0.8, 0.75, 0.82, 0.78, 0.85]
        group2 = [0.7, 0.68, 0.72, 0.69, 0.73]
        
        test_results = analyzer.statistical_test(group1, group2)
        
        assert isinstance(test_results, dict)
        assert 'statistic' in test_results
        assert 'p_value' in test_results
        assert 'significant' in test_results
    
    def test_curriculum_learning_comprehensive(self):
        """Test CurriculumLearning implementation."""
        config = CurriculumConfig(
            strategy='difficulty_based',
            initial_difficulty=0.3,
            difficulty_increment=0.1,
            difficulty_threshold=0.8
        )
        
        curriculum = CurriculumLearning(config)
        
        # Test difficulty updates
        initial_difficulty = curriculum.get_current_difficulty()
        assert initial_difficulty == 0.3
        
        # Good performance should increase difficulty
        new_difficulty = curriculum.update_difficulty(0.9)
        assert new_difficulty > initial_difficulty
        
        # Poor performance should eventually decrease difficulty
        for _ in range(10):  # Multiple poor performances
            new_difficulty = curriculum.update_difficulty(0.3)
        
        assert new_difficulty < 0.3  # Should have decreased
    
    def test_task_diversity_tracker_comprehensive(self):
        """Test TaskDiversityTracker implementation."""
        config = DiversityConfig(
            diversity_metric='cosine_similarity',
            track_class_distribution=True,
            track_feature_diversity=True
        )
        
        tracker = TaskDiversityTracker(config)
        
        # Add some tasks
        task_features = [
            torch.randn(64),  # Task 1 features
            torch.randn(64),  # Task 2 features
            torch.randn(64),  # Task 3 features
        ]
        
        for features in task_features:
            tracker.add_task(features)
        
        # Compute diversity
        diversity_scores = tracker.compute_diversity()
        
        assert isinstance(diversity_scores, dict)
        assert 'diversity_score' in diversity_scores
        assert 0.0 <= diversity_scores['diversity_score'] <= 1.0

    # ==========================================================================
    # Hardware Utilities Tests
    # ==========================================================================
    
    def test_hardware_manager_comprehensive(self):
        """Test HardwareManager implementation."""
        config = HardwareConfig(
            device='cpu',  # Always test with CPU
            mixed_precision=False,
            compile_model=False
        )
        
        manager = HardwareManager(config)
        
        # Test device detection
        assert manager.device.type == 'cpu'
        
        # Test model preparation
        model = nn.Linear(10, 5)
        prepared_model = manager.prepare_model(model)
        
        assert isinstance(prepared_model, nn.Module)
        
        # Test data preparation
        data = torch.randn(32, 10)
        prepared_data = manager.prepare_data(data)
        
        assert prepared_data.device.type == 'cpu'
    
    def test_multi_gpu_manager_comprehensive(self):
        """Test MultiGPUManager implementation."""
        config = HardwareConfig(device='cpu')  # CPU-only testing
        
        # MultiGPU manager should handle CPU gracefully
        manager = MultiGPUManager(config)
        
        # Test model wrapping (should be no-op on CPU)
        model = nn.Linear(10, 5)
        wrapped_model = manager.wrap_model(model)
        
        assert isinstance(wrapped_model, nn.Module)

    # ==========================================================================
    # Configuration Tests
    # ==========================================================================
    
    def test_all_config_classes_comprehensive(self):
        """Test all configuration classes."""
        
        # Test all config classes with various parameters
        configs = [
            (DatasetConfig, {'dataset_type': 'episodic', 'shuffle': True}),
            (MetricsConfig, {'compute_accuracy': True, 'compute_loss': True}),
            (StatsConfig, {'confidence_level': 0.95, 'num_bootstrap_samples': 1000}),
            (CurriculumConfig, {'strategy': 'difficulty_based', 'initial_difficulty': 0.2}),
            (DiversityConfig, {'diversity_metric': 'cosine_similarity', 'diversity_threshold': 0.7}),
            (TaskConfiguration, {'n_way': 5, 'k_shot': 5, 'q_query': 15}),
            (EvaluationConfig, {'confidence_intervals': True, 'num_bootstrap_samples': 100}),
            (TestTimeComputeConfig, {'compute_strategy': 'basic', 'max_compute_budget': 10}),
            (MAMLConfig, {'inner_lr': 0.01, 'meta_lr': 0.001, 'inner_steps': 5}),
            (MAMLenLLMConfig, {'context_optimization_method': 'ems', 'synthetic_data_generation': True}),
            (PrototypicalConfig, {'embedding_dim': 64, 'use_squared_euclidean': True}),
            (MatchingConfig, {'embedding_dim': 64, 'attention_method': 'cosine'}),
            (RelationConfig, {'embedding_dim': 64, 'relation_dim': 8}),
            (ContinualMetaConfig, {'memory_size': 100, 'consolidation_strength': 0.4}),
            (HardwareConfig, {'device': 'cpu', 'mixed_precision': False}),
        ]
        
        for config_class, params in configs:
            config = config_class(**params)
            
            # Verify parameters were set
            for key, value in params.items():
                assert hasattr(config, key)
                assert getattr(config, key) == value

    # ==========================================================================
    # Integration Tests
    # ==========================================================================
    
    def test_complete_meta_learning_pipeline(self):
        """Test complete end-to-end meta-learning pipeline."""
        
        # 1. Create dataset
        task_config = TaskConfiguration(n_way=3, k_shot=3, q_query=5)  # Smaller for speed
        dataset = MetaLearningDataset(self.full_data[:100], self.full_labels[:100], task_config)
        
        # 2. Create model
        proto_config = PrototypicalConfig(embedding_dim=32)  # Smaller for speed
        model = PrototypicalNetworks(proto_config)
        
        # 3. Create MAML learner
        maml_config = MAMLConfig(inner_lr=0.1, meta_lr=0.01, inner_steps=2)
        
        class MAMLCompatibleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(84*84*3, 3)
            
            def forward(self, x):
                return self.fc(x.view(x.size(0), -1))
        
        maml_model = MAMLCompatibleModel()
        maml_learner = MAMLLearner(maml_model, maml_config)
        
        # 4. Create test-time compute scaler
        ttc_config = TestTimeComputeConfig(compute_strategy='basic', max_compute_budget=3)
        ttc_scaler = TestTimeComputeScaler(model, ttc_config)
        
        # 5. Test the pipeline
        episode = dataset[0]
        
        # Test prototypical networks
        proto_logits = model.forward(
            episode['support']['data'], episode['support']['labels'],
            episode['query']['data']
        )
        
        if isinstance(proto_logits, dict):
            proto_logits = proto_logits['logits']
        
        assert proto_logits.shape == (15, 3)  # 5 queries per class, 3 classes
        
        # Test test-time compute
        with torch.no_grad():
            ttc_logits = ttc_scaler.scale_compute(
                episode['support']['data'], episode['support']['labels'],
                episode['query']['data']
            )
        
        assert ttc_logits.shape == (15, 3)
        
        # Test MAML (simplified)
        task_batch = [(
            episode['support']['data'], episode['support']['labels'],
            episode['query']['data'], episode['query']['labels']
        )]
        
        maml_results = maml_learner.meta_train_step(task_batch)
        
        assert isinstance(maml_results, dict)
        assert 'meta_loss' in maml_results

    # ==========================================================================
    # Property-Based and Edge Case Tests
    # ==========================================================================
    
    @pytest.mark.parametrize("n_way,k_shot,q_query", [
        (2, 1, 1),    # Minimal case
        (5, 5, 15),   # Standard case
        (10, 10, 20), # Larger case
    ])
    def test_meta_learning_dataset_property_based(self, n_way, k_shot, q_query):
        """Property-based testing for MetaLearningDataset."""
        task_config = TaskConfiguration(
            n_way=n_way, k_shot=k_shot, q_query=q_query, num_episodes=10
        )
        
        # Ensure we have enough classes and samples
        min_classes = n_way
        min_samples_per_class = k_shot + q_query
        
        if self.n_classes >= min_classes:
            dataset = MetaLearningDataset(self.full_data, self.full_labels, task_config)
            episode = dataset[0]
            
            # Property: Support set has correct size
            expected_support_size = n_way * k_shot
            assert episode['support']['data'].shape[0] == expected_support_size
            
            # Property: Query set has correct size  
            expected_query_size = n_way * q_query
            assert episode['query']['data'].shape[0] == expected_query_size
            
            # Property: Labels are in correct range
            assert episode['support']['labels'].max() < n_way
            assert episode['query']['labels'].max() < n_way
    
    def test_error_handling_comprehensive(self):
        """Test error handling across all components."""
        
        # Test invalid configurations
        with pytest.raises((ValueError, TypeError, AssertionError)):
            TaskConfiguration(n_way=0, k_shot=1, q_query=1)  # Invalid n_way
        
        with pytest.raises((ValueError, TypeError)):
            TestTimeComputeConfig(compute_strategy="invalid_strategy")
        
        with pytest.raises((ValueError, TypeError)):
            MAMLConfig(inner_lr=-1.0)  # Invalid learning rate
        
        # Test invalid data shapes
        invalid_support = torch.randn(5, 10)  # Wrong dimensions
        valid_labels = torch.randint(0, 5, (5,))
        valid_query = torch.randn(10, 3, 84, 84)
        
        proto_config = PrototypicalConfig()
        model = PrototypicalNetworks(proto_config)
        
        # This should handle the error gracefully or raise informative error
        try:
            model.forward(invalid_support, valid_labels, valid_query)
        except (RuntimeError, ValueError, AssertionError):
            pass  # Expected to fail
    
    def test_backward_compatibility_comprehensive(self):
        """Test all backward compatibility aliases."""
        
        # Test MAML aliases
        from meta_learning.meta_learning_modules import (
            MAML, FOMAML, Reptile, ANIL, BOIL,
            FewShotLearner, PrototypicalLearner,
            ContinualMetaLearner
        )
        
        # Verify aliases point to correct classes
        assert MAML == MAMLLearner
        assert FOMAML == FirstOrderMAML
        assert Reptile == ReptileLearner
        assert ANIL == ANILLearner
        assert BOIL == BOILLearner
        
        assert FewShotLearner == PrototypicalNetworks
        assert PrototypicalLearner == PrototypicalNetworks
        
        assert ContinualMetaLearner == OnlineMetaLearner


# =============================================================================
# Performance and Stress Tests
# =============================================================================

class TestPerformanceAndStress:
    """Performance and stress tests."""
    
    def test_large_scale_meta_learning(self):
        """Test meta-learning with larger datasets."""
        # Create larger synthetic dataset
        large_data = torch.randn(5000, 3, 28, 28)  # Smaller images for speed
        large_labels = torch.randint(0, 50, (5000,))  # 50 classes
        
        task_config = TaskConfiguration(n_way=10, k_shot=5, q_query=15, num_episodes=10)
        dataset = MetaLearningDataset(large_data, large_labels, task_config)
        
        assert len(dataset) == 10
        episode = dataset[0]
        
        assert episode['support']['data'].shape == (50, 3, 28, 28)  # 10*5
        assert episode['query']['data'].shape == (150, 3, 28, 28)   # 10*15
    
    def test_memory_efficiency(self):
        """Test memory efficiency of implementations."""
        # Test with different batch sizes to ensure memory doesn't explode
        batch_sizes = [1, 4, 8, 16]
        
        for batch_size in batch_sizes:
            data = torch.randn(batch_size * 25, 3, 84, 84)
            labels = torch.repeat_interleave(torch.arange(5), batch_size * 5)
            query_data = torch.randn(batch_size * 25, 3, 84, 84) 
            
            config = PrototypicalConfig(embedding_dim=32)  # Smaller for memory
            model = PrototypicalNetworks(config)
            
            with torch.no_grad():
                logits = model.forward(data, labels, query_data)
                
                if isinstance(logits, dict):
                    logits = logits['logits']
                
                expected_shape = (batch_size * 25, 5)
                assert logits.shape == expected_shape


# =============================================================================  
# Factory Function Tests
# =============================================================================

class TestFactoryFunctions:
    """Test all factory functions."""
    
    def test_create_functions_comprehensive(self):
        """Test all create_* factory functions."""
        from meta_learning.meta_learning_modules import (
            create_dataset, create_metrics_evaluator, create_curriculum_scheduler,
            create_hardware_manager, create_maml_learner, create_few_shot_learner,
            create_continual_learner
        )
        
        # Test create_dataset
        data = torch.randn(100, 3, 84, 84)
        labels = torch.randint(0, 10, (100,))
        task_config = TaskConfiguration(n_way=5, k_shot=5, q_query=15)
        
        dataset = create_dataset(data, labels, task_config)
        assert isinstance(dataset, MetaLearningDataset)
        
        # Test create_metrics_evaluator
        metrics = create_metrics_evaluator()
        assert isinstance(metrics, EvaluationMetrics)
        
        # Test create_curriculum_scheduler
        curriculum = create_curriculum_scheduler()
        assert isinstance(curriculum, CurriculumLearning)
        
        # Test create_hardware_manager
        hardware = create_hardware_manager()
        assert isinstance(hardware, HardwareManager)
        
        # Test create_few_shot_learner
        config = PrototypicalConfig()
        few_shot = create_few_shot_learner(config)
        assert isinstance(few_shot, PrototypicalNetworks)
        
        # Test create_continual_learner
        config = ContinualMetaConfig()
        continual = create_continual_learner(config)
        assert isinstance(continual, OnlineMetaLearner)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])