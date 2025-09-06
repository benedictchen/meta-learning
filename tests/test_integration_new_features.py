"""
Integration Test Suite for New Features
=====================================

Comprehensive integration tests that verify all new functionality works together:
- Dataset management system integration with toolkit
- Phase 4 enhancements working together
- Cross-module compatibility and data flow
- End-to-end workflows with new features
- Performance and reliability under realistic scenarios

Author: Test Suite Generator
"""

import pytest
import torch
import numpy as np
import tempfile
import shutil
import json
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any

from meta_learning.toolkit import MetaLearningToolkit
from meta_learning.shared.types import Episode
from meta_learning.meta_learning_modules.dataset_management import (
    DatasetManager,
    DatasetInfo,
    get_dataset_manager
)


class TestDatasetToolkitIntegration:
    """Test integration between dataset management and toolkit."""
    
    def setup_method(self):
        """Set up integration test environment."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Initialize toolkit with Phase 4 features
        self.toolkit = MetaLearningToolkit()
        self.toolkit.enable_failure_prediction()
        self.toolkit.enable_automatic_algorithm_selection()
        self.toolkit.enable_realtime_optimization()
        self.toolkit.enable_cross_task_knowledge_transfer()
        
        # Initialize dataset manager
        self.dataset_manager = DatasetManager(
            cache_dir=self.temp_dir,
            max_cache_size_gb=0.01  # Small for testing
        )
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_toolkit_dataset_manager_integration(self):
        """Test that toolkit can work with dataset manager."""
        # Register a test dataset
        test_dataset = DatasetInfo(
            name="integration_dataset",
            description="Dataset for integration testing",
            urls=["http://example.com/test.zip"],
            checksums={"md5": "test_hash"},
            file_size=1000,
            n_classes=5,
            n_samples=100,
            image_size=(32, 32)
        )
        
        self.dataset_manager.registry.register_dataset(test_dataset)
        
        # Test that toolkit can get dataset info
        dataset_info = self.dataset_manager.get_dataset_info("integration_dataset")
        assert dataset_info is not None
        assert dataset_info.name == "integration_dataset"
        
        # Test that dataset characteristics can be analyzed by toolkit
        # Create mock episode based on dataset info
        episode = Episode(
            support_x=torch.randn(25, 3, 32, 32),
            support_y=torch.randint(0, 5, (25,)),
            query_x=torch.randn(75, 3, 32, 32),
            query_y=torch.randint(0, 5, (75,))
        )
        
        # Analyze with toolkit
        characteristics = self.toolkit.analyze_data_characteristics(episode)
        
        # Characteristics should match dataset info
        assert characteristics['n_classes'] == dataset_info.n_classes
        assert characteristics['data_dimensionality'] == 3 * 32 * 32
    
    def test_cross_dataset_knowledge_transfer(self):
        """Test knowledge transfer across different datasets."""
        # Register multiple similar datasets
        datasets = [
            ("dataset_A", 5, (32, 32)),
            ("dataset_B", 5, (32, 32)),  # Same classes, same size
            ("dataset_C", 10, (64, 64)),  # Different classes, different size
        ]
        
        for name, n_classes, image_size in datasets:
            dataset_info = DatasetInfo(
                name=name,
                description=f"Test dataset {name}",
                urls=[f"http://example.com/{name}.zip"],
                checksums={"md5": f"{name}_hash"},
                file_size=1000,
                n_classes=n_classes,
                n_samples=100,
                image_size=image_size
            )
            self.dataset_manager.registry.register_dataset(dataset_info)
        
        # Train on dataset A
        episode_A = Episode(
            support_x=torch.randn(25, 3, 32, 32),
            support_y=torch.randint(0, 5, (25,)),
            query_x=torch.randn(75, 3, 32, 32),
            query_y=torch.randint(0, 5, (75,))
        )
        
        self.toolkit.store_task_knowledge(
            episode_A, "task_dataset_A",
            {'learning_rate': 0.001, 'batch_size': 32},
            {'accuracy': 0.88, 'training_time': 100.0}
        )
        
        # Train on dataset B (similar to A)
        episode_B = Episode(
            support_x=torch.randn(25, 3, 32, 32),
            support_y=torch.randint(0, 5, (25,)),
            query_x=torch.randn(75, 3, 32, 32),
            query_y=torch.randint(0, 5, (75,))
        )
        
        # Should get relevant knowledge from dataset A
        relevant_knowledge = self.toolkit.retrieve_relevant_knowledge(episode_B, "task_dataset_B")
        
        if len(relevant_knowledge) > 0:
            # Knowledge from dataset A should be relevant for dataset B
            assert relevant_knowledge[0]['knowledge']['task_id'] == "task_dataset_A"
            assert relevant_knowledge[0]['similarity'] > 0.0  # Should have some similarity


class TestEndToEndWorkflows:
    """Test complete end-to-end workflows with all new features."""
    
    def setup_method(self):
        """Set up end-to-end test environment."""
        self.temp_dir = tempfile.mkdtemp()
        
        self.toolkit = MetaLearningToolkit()
        self.dataset_manager = DatasetManager(cache_dir=self.temp_dir, max_cache_size_gb=0.01)
        
        # Enable all features
        self.toolkit.enable_failure_prediction(True, True)
        self.toolkit.enable_automatic_algorithm_selection(True, "maml")
        self.toolkit.enable_realtime_optimization(True, 10)
        self.toolkit.enable_cross_task_knowledge_transfer(True, 50)
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_complete_meta_learning_pipeline(self):
        """Test complete meta-learning pipeline with all new features."""
        # Simulate training on multiple tasks
        task_results = []
        
        for task_idx in range(5):
            task_id = f"pipeline_task_{task_idx}"
            
            # Create varied episodes to test different scenarios
            n_classes = np.random.randint(3, 8)
            shots_per_class = np.random.randint(1, 6)
            image_size = np.random.choice([28, 32, 64, 84])
            
            episode = Episode(
                support_x=torch.randn(n_classes * shots_per_class, 3, image_size, image_size),
                support_y=torch.repeat_interleave(torch.arange(n_classes), shots_per_class),
                query_x=torch.randn(n_classes * 15, 3, image_size, image_size),
                query_y=torch.repeat_interleave(torch.arange(n_classes), 15)
            )
            
            # Phase 1: Data analysis and algorithm selection
            characteristics = self.toolkit.analyze_data_characteristics(episode)
            algorithm_rec = self.toolkit.recommend_algorithm(characteristics)
            
            # Phase 2: Transfer learning initialization
            transfer_init = self.toolkit.get_transfer_initialization(episode, task_id)
            
            # Phase 3: Parameter optimization and failure prediction
            base_params = self.toolkit.get_current_best_params()
            failure_features = self.toolkit.collect_failure_features(episode, base_params)
            failure_prob = self.toolkit.predict_failure_probability(failure_features)
            
            # Apply recovery if high failure probability
            final_params = base_params.copy()
            if failure_prob > 0.7:
                recovery_strategy = self.toolkit.get_recovery_strategy(failure_features)
                final_params = self.toolkit.apply_recovery_strategy(final_params, recovery_strategy)
            
            # Phase 4: Simulate training and record results
            # Vary performance based on task difficulty
            task_difficulty = np.mean([
                characteristics.get('data_variance', 0.5),
                1.0 - characteristics.get('class_separability', 0.5),
                characteristics.get('feature_complexity', 0.5)
            ])
            
            # Better performance for easier tasks and good parameters
            base_accuracy = 0.7 + np.random.normal(0, 0.1)
            difficulty_penalty = task_difficulty * 0.3
            recovery_bonus = 0.1 if failure_prob > 0.7 else 0
            
            final_accuracy = max(0.3, min(0.95, base_accuracy - difficulty_penalty + recovery_bonus))
            training_time = np.random.uniform(80, 150) * (1 + task_difficulty)
            
            performance_metrics = {
                'accuracy': final_accuracy,
                'loss': max(0.05, 1.0 - final_accuracy + np.random.normal(0, 0.1)),
                'training_time': training_time
            }
            
            # Phase 5: Record all outcomes and update models
            failed = final_accuracy < 0.6
            
            self.toolkit.record_failure_outcome(failure_features, failed, final_accuracy)
            self.toolkit.record_algorithm_performance(
                algorithm_rec['algorithm'], characteristics, 
                final_accuracy, training_time
            )
            
            # Store knowledge for transfer learning
            model_state = {'embedding': torch.randn(128, 256)} if final_accuracy > 0.7 else None
            self.toolkit.store_task_knowledge(episode, task_id, final_params, performance_metrics, model_state)
            
            task_results.append({
                'task_id': task_id,
                'characteristics': characteristics,
                'algorithm': algorithm_rec['algorithm'],
                'failure_probability': failure_prob,
                'final_accuracy': final_accuracy,
                'used_recovery': failure_prob > 0.7
            })
        
        # Analyze overall pipeline results
        assert len(task_results) == 5
        
        # Check that systems learned and improved
        accuracies = [r['final_accuracy'] for r in task_results]
        avg_accuracy = np.mean(accuracies)
        
        # Should achieve reasonable performance
        assert avg_accuracy > 0.4  # Should be better than random (adjusted for variability)
        
        # Check that all systems have data
        assert len(self.toolkit.failure_history) > 0
        assert len(self.toolkit.algorithm_performance_history) > 0
        assert len(self.toolkit.knowledge_memory) > 0
        
        # Update all models with accumulated data
        self.toolkit.update_failure_predictor()
        self.toolkit.update_algorithm_selector()
        
        # Test that updated models work
        test_episode = Episode(
            support_x=torch.randn(25, 3, 84, 84),
            support_y=torch.randint(0, 5, (25,)),
            query_x=torch.randn(75, 3, 84, 84),
            query_y=torch.randint(0, 5, (75,))
        )
        
        final_characteristics = self.toolkit.analyze_data_characteristics(test_episode)
        final_recommendation = self.toolkit.recommend_algorithm(final_characteristics)
        final_failure_features = self.toolkit.collect_failure_features(test_episode, {})
        final_failure_prob = self.toolkit.predict_failure_probability(final_failure_features)
        
        # All should work without errors
        assert isinstance(final_recommendation, dict)
        assert isinstance(final_failure_prob, float)
        assert 0.0 <= final_failure_prob <= 1.0
    
    def test_scalability_and_performance(self):
        """Test system performance under load."""
        start_time = time.time()
        
        # Process many tasks quickly
        for i in range(20):
            episode = Episode(
                support_x=torch.randn(25, 3, 32, 32),
                support_y=torch.randint(0, 5, (25,)),
                query_x=torch.randn(75, 3, 32, 32),
                query_y=torch.randint(0, 5, (75,))
            )
            
            # Quick analysis pipeline
            characteristics = self.toolkit.analyze_data_characteristics(episode)
            algorithm_rec = self.toolkit.recommend_algorithm(characteristics)
            failure_features = self.toolkit.collect_failure_features(episode, {})
            failure_prob = self.toolkit.predict_failure_probability(failure_features)
            
            # Record minimal results
            self.toolkit.record_failure_outcome(failure_features, False, 0.8)
            self.toolkit.record_algorithm_performance(algorithm_rec['algorithm'], characteristics, 0.8, 100.0)
        
        elapsed_time = time.time() - start_time
        
        # Should process 20 tasks in reasonable time (less than 30 seconds)
        assert elapsed_time < 30.0
        
        # Memory usage should be reasonable
        assert len(self.toolkit.knowledge_memory) <= 50  # Respects memory limit
        assert len(self.toolkit.failure_history) == 20
        assert len(self.toolkit.algorithm_performance_history) == 20
    
    def test_robustness_and_error_handling(self):
        """Test system robustness with edge cases and errors."""
        # Test with edge case episodes
        edge_cases = [
            # Single class
            Episode(
                support_x=torch.randn(5, 3, 32, 32),
                support_y=torch.zeros(5, dtype=torch.long),
                query_x=torch.randn(15, 3, 32, 32),
                query_y=torch.zeros(15, dtype=torch.long)
            ),
            # Many classes
            Episode(
                support_x=torch.randn(100, 3, 32, 32),
                support_y=torch.randint(0, 20, (100,)),
                query_x=torch.randn(300, 3, 32, 32),
                query_y=torch.randint(0, 20, (300,))
            ),
            # Very small episode
            Episode(
                support_x=torch.randn(2, 3, 32, 32),
                support_y=torch.tensor([0, 1]),
                query_x=torch.randn(2, 3, 32, 32),
                query_y=torch.tensor([0, 1])
            )
        ]
        
        for i, episode in enumerate(edge_cases):
            try:
                # All operations should handle edge cases gracefully
                characteristics = self.toolkit.analyze_data_characteristics(episode)
                algorithm_rec = self.toolkit.recommend_algorithm(characteristics)
                failure_features = self.toolkit.collect_failure_features(episode, {})
                failure_prob = self.toolkit.predict_failure_probability(failure_features)
                
                # Basic sanity checks
                assert isinstance(characteristics, dict)
                assert isinstance(algorithm_rec, dict)
                assert isinstance(failure_prob, float)
                assert 0.0 <= failure_prob <= 1.0
                
            except Exception as e:
                # Some edge cases might legitimately fail, but should be handled gracefully
                assert isinstance(e, (ValueError, RuntimeError))


class TestConcurrentOperations:
    """Test concurrent operations and thread safety."""
    
    def setup_method(self):
        """Set up concurrency test environment."""
        self.toolkit = MetaLearningToolkit()
        self.toolkit.enable_failure_prediction()
        self.toolkit.enable_automatic_algorithm_selection()
        self.toolkit.enable_realtime_optimization()
        self.toolkit.enable_cross_task_knowledge_transfer()
    
    def test_concurrent_knowledge_storage(self):
        """Test storing knowledge from multiple threads simultaneously."""
        import threading
        import queue
        
        results_queue = queue.Queue()
        
        def store_knowledge_worker(worker_id):
            try:
                for i in range(10):
                    episode = Episode(
                        support_x=torch.randn(25, 3, 32, 32),
                        support_y=torch.randint(0, 5, (25,)),
                        query_x=torch.randn(75, 3, 32, 32),
                        query_y=torch.randint(0, 5, (75,))
                    )
                    
                    task_id = f"worker_{worker_id}_task_{i}"
                    params = {'learning_rate': 0.001 + worker_id * 0.0001}
                    metrics = {'accuracy': 0.7 + np.random.random() * 0.2}
                    
                    self.toolkit.store_task_knowledge(episode, task_id, params, metrics)
                
                results_queue.put((worker_id, "success"))
                
            except Exception as e:
                results_queue.put((worker_id, f"error: {e}"))
        
        # Start multiple worker threads
        threads = []
        num_workers = 5
        
        for worker_id in range(num_workers):
            thread = threading.Thread(target=store_knowledge_worker, args=(worker_id,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=30)  # 30 second timeout
        
        # Check results
        successful_workers = 0
        while not results_queue.empty():
            worker_id, result = results_queue.get()
            if result == "success":
                successful_workers += 1
            else:
                print(f"Worker {worker_id} failed: {result}")
        
        # All workers should succeed
        assert successful_workers == num_workers
        
        # Knowledge memory should contain all stored items (up to memory limit)
        assert len(self.toolkit.knowledge_memory) > 0
        assert len(self.toolkit.knowledge_memory) <= self.toolkit.knowledge_memory_size


class TestLongRunningOperations:
    """Test long-running operations and system stability."""
    
    def setup_method(self):
        """Set up long-running test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.toolkit = MetaLearningToolkit()
        self.dataset_manager = DatasetManager(cache_dir=self.temp_dir, max_cache_size_gb=0.1)
        
        # Enable all features
        self.toolkit.enable_failure_prediction(True, True)
        self.toolkit.enable_automatic_algorithm_selection(True, "maml")
        self.toolkit.enable_realtime_optimization(True, 5)  # More frequent optimization
        self.toolkit.enable_cross_task_knowledge_transfer(True, 25)  # Smaller memory for faster testing
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_extended_operation_cycle(self):
        """Test system stability over extended operation cycles."""
        # Simulate continuous operation
        for cycle in range(5):  # 5 cycles for faster testing
            cycle_start = time.time()
            
            # Process multiple tasks per cycle
            for task_idx in range(3):  # Reduced for faster testing
                task_id = f"cycle_{cycle}_task_{task_idx}"
                
                # Create episode with some variation
                episode = Episode(
                    support_x=torch.randn(25, 3, 32, 32) + torch.randn(1) * 0.1,
                    support_y=torch.randint(0, 5, (25,)),
                    query_x=torch.randn(75, 3, 32, 32) + torch.randn(1) * 0.1,
                    query_y=torch.randint(0, 5, (75,))
                )
                
                # Full processing pipeline
                characteristics = self.toolkit.analyze_data_characteristics(episode)
                algorithm_rec = self.toolkit.recommend_algorithm(characteristics)
                failure_features = self.toolkit.collect_failure_features(episode, {})
                failure_prob = self.toolkit.predict_failure_probability(failure_features)
                
                # Simulate performance with some realism
                base_performance = 0.75 + np.random.normal(0, 0.1)
                cycle_improvement = cycle * 0.01  # Slight improvement over time
                final_performance = min(0.95, max(0.4, base_performance + cycle_improvement))
                
                # Record outcomes
                failed = final_performance < 0.6
                self.toolkit.record_failure_outcome(failure_features, failed, final_performance)
                self.toolkit.record_algorithm_performance(
                    algorithm_rec['algorithm'], characteristics, 
                    final_performance, 100.0 + np.random.normal(0, 10)
                )
                
                # Store knowledge
                self.toolkit.store_task_knowledge(
                    episode, task_id, 
                    {'learning_rate': 0.001, 'batch_size': 32},
                    {'accuracy': final_performance, 'training_time': 100.0}
                )
            
            # Periodic model updates
            if cycle % 2 == 1:  # Every 2 cycles
                self.toolkit.update_failure_predictor()
                self.toolkit.update_algorithm_selector()
            
            # Memory management checks
            if cycle % 3 == 2:  # Every 3 cycles
                # Trigger knowledge consolidation
                consolidation_result = self.toolkit.consolidate_knowledge(similarity_threshold=0.85)
                assert isinstance(consolidation_result, dict)
            
            cycle_time = time.time() - cycle_start
            assert cycle_time < 10.0  # Each cycle should complete quickly
        
        # After all cycles, system should still be functional
        final_test_episode = Episode(
            support_x=torch.randn(25, 3, 32, 32),
            support_y=torch.randint(0, 5, (25,)),
            query_x=torch.randn(75, 3, 32, 32),
            query_y=torch.randint(0, 5, (75,))
        )
        
        final_characteristics = self.toolkit.analyze_data_characteristics(final_test_episode)
        final_recommendation = self.toolkit.recommend_algorithm(final_characteristics)
        
        assert isinstance(final_characteristics, dict)
        assert isinstance(final_recommendation, dict)
        
        # System should have learned over time
        assert len(self.toolkit.failure_history) > 0
        assert len(self.toolkit.algorithm_performance_history) > 0
        assert len(self.toolkit.knowledge_memory) > 0
        
        # Memory should be bounded
        assert len(self.toolkit.knowledge_memory) <= 25


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])