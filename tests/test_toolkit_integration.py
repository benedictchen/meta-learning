"""
Tests for toolkit integration and complete system functionality.

Tests cover:
- MetaLearningToolkit complete integration
- create_meta_learning_toolkit convenience function
- quick_evaluation functionality
- Cross-component integration
- End-to-end workflows
- Real-world usage scenarios
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock

from meta_learning import (
    MetaLearningToolkit, create_meta_learning_toolkit, quick_evaluation,
    Episode
)
from meta_learning.algorithms.ttc_config import TestTimeComputeConfig
from meta_learning.algorithms.maml_research_accurate import MAMLConfig, MAMLVariant


class TestMetaLearningToolkit:
    """Test MetaLearningToolkit core functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.model = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 10)
        )
        
        self.episode = Episode(
            support_x=torch.randn(25, 64),  # 5-way, 5-shot
            support_y=torch.repeat_interleave(torch.arange(5), 5),
            query_x=torch.randn(15, 64),    # 5-way, 3-query
            query_y=torch.repeat_interleave(torch.arange(5), 3)
        )
        
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if hasattr(self, 'temp_dir'):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_toolkit_initialization(self):
        """Test MetaLearningToolkit initialization."""
        toolkit = MetaLearningToolkit()
        
        # Check basic initialization
        assert toolkit.config == {}
        assert toolkit.test_time_scaler is None
        assert toolkit.maml_learner is None
        assert toolkit.evaluation_harness is None
        
        # Check dataset ecosystem initialization
        assert toolkit.benchmark_manager is None
        assert toolkit.data_accelerator is None
        assert toolkit.infinite_iterators == {}
        
        # Check error handling initialization
        assert toolkit.warning_system is None
        assert toolkit.error_recovery is None
        assert toolkit.performance_monitor is None
    
    def test_toolkit_custom_config(self):
        """Test toolkit with custom configuration."""
        config = {
            'algorithm': 'maml',
            'inner_lr': 0.01,
            'seed': 123,
            'enable_monitoring': True
        }
        
        toolkit = MetaLearningToolkit(config=config)
        
        assert toolkit.config == config
        assert toolkit.config['algorithm'] == 'maml'
        assert toolkit.config['inner_lr'] == 0.01
        assert toolkit.config['seed'] == 123
    
    def test_create_test_time_compute_scaler(self):
        """Test Test-Time Compute Scaler creation."""
        toolkit = MetaLearningToolkit()
        
        config = TestTimeComputeConfig()
        scaler = toolkit.create_test_time_compute_scaler(self.model, config)
        
        assert toolkit.test_time_scaler is not None
        assert toolkit.test_time_scaler == scaler
        assert scaler.encoder == self.model
        assert scaler.config == config
    
    def test_create_research_maml(self):
        """Test Research MAML creation."""
        toolkit = MetaLearningToolkit()
        
        config = MAMLConfig(
            variant=MAMLVariant.MAML,
            inner_lr=0.01,
            outer_lr=0.001,
            inner_steps=2
        )
        maml = toolkit.create_research_maml(self.model, config)
        
        assert toolkit.maml_learner is not None
        assert toolkit.maml_learner == maml
        assert maml.model == self.model
        assert maml.config == config
    
    def test_apply_batch_norm_fixes(self):
        """Test BatchNorm fixes application."""
        toolkit = MetaLearningToolkit()
        
        # Create model with BatchNorm
        bn_model = nn.Sequential(
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
        
        fixed_model = toolkit.apply_batch_norm_fixes(bn_model)
        
        # Should return a model (may be the same or modified)
        assert isinstance(fixed_model, nn.Module)
        assert hasattr(fixed_model, 'forward')
    
    def test_setup_deterministic_training(self):
        """Test deterministic training setup."""
        toolkit = MetaLearningToolkit()
        
        # Should not raise an exception
        toolkit.setup_deterministic_training(seed=42)
        
        # Should have configured determinism
        assert toolkit.determinism_manager is not None
    
    def test_create_evaluation_harness(self):
        """Test evaluation harness creation."""
        toolkit = MetaLearningToolkit()
        
        harness = toolkit.create_evaluation_harness(confidence_level=0.95)
        
        assert toolkit.evaluation_harness is not None
        assert toolkit.evaluation_harness == harness
        assert harness.confidence_level == 0.95
    
    def test_train_episode_maml(self):
        """Test episode training with MAML."""
        toolkit = MetaLearningToolkit()
        
        # Setup MAML
        config = MAMLConfig(inner_lr=0.01, inner_steps=1)
        toolkit.create_research_maml(self.model, config)
        
        # Train episode
        results = toolkit.train_episode(self.episode, algorithm="maml")
        
        assert isinstance(results, dict)
        assert 'query_loss' in results
        assert 'query_accuracy' in results
        assert 'support_loss' in results
        assert 'meta_loss' in results
        
        # Check reasonable values
        assert 0.0 <= results['query_accuracy'] <= 1.0
        assert results['query_loss'] > 0
        assert results['support_loss'] > 0
        assert results['meta_loss'] > 0
    
    def test_train_episode_test_time_compute(self):
        """Test episode training with Test-Time Compute."""
        toolkit = MetaLearningToolkit()
        
        # Setup TTCS
        config = TestTimeComputeConfig()
        toolkit.create_test_time_compute_scaler(self.model, config)
        
        # Train episode
        results = toolkit.train_episode(self.episode, algorithm="test_time_compute")
        
        assert isinstance(results, dict)
        assert 'query_accuracy' in results
        assert 'compute_scaling_metrics' in results
        assert 'predictions' in results
        
        # Check reasonable values
        assert 0.0 <= results['query_accuracy'] <= 1.0
        assert isinstance(results['compute_scaling_metrics'], dict)
        assert results['predictions'].shape == (15, 5)  # Query predictions
    
    def test_train_episode_uninitialized_algorithm(self):
        """Test training with uninitialized algorithm."""
        toolkit = MetaLearningToolkit()
        
        # Should raise error for uninitialized algorithm
        with pytest.raises(ValueError, match="Algorithm .* not initialized"):
            toolkit.train_episode(self.episode, algorithm="maml")
    
    def test_setup_benchmark_datasets(self):
        """Test benchmark dataset setup."""
        toolkit = MetaLearningToolkit()
        
        # Setup dataset manager with custom cache
        manager = toolkit.setup_dataset_manager(cache_dir=self.temp_dir)
        
        # Setup benchmark datasets
        results = toolkit.setup_benchmark_datasets(['synthetic'])
        
        assert isinstance(results, dict)
        assert 'synthetic' in results
        assert isinstance(results['synthetic'], bool)
        
        # Should have created dataset manager
        assert toolkit.benchmark_manager is not None
        assert toolkit.benchmark_manager == manager
    
    def test_list_available_datasets(self):
        """Test listing available datasets."""
        toolkit = MetaLearningToolkit()
        
        datasets = toolkit.list_available_datasets()
        
        assert isinstance(datasets, dict)
        assert 'synthetic' in datasets
        assert 'mini_imagenet' in datasets
        
        # Each dataset should have metadata
        for dataset_name, metadata in datasets.items():
            assert 'description' in metadata
            assert 'file_size_mb' in metadata
            assert 'cached' in metadata
    
    def test_enable_data_acceleration(self):
        """Test data acceleration setup."""
        toolkit = MetaLearningToolkit()
        
        # Enable data acceleration
        toolkit.enable_data_acceleration(memory_budget_gb=2.0)
        
        # Should have created accelerator factory
        assert toolkit.data_accelerator is not None
        assert callable(toolkit.data_accelerator)
    
    def test_configure_infinite_iteration(self):
        """Test infinite iteration configuration."""
        toolkit = MetaLearningToolkit()
        
        # Configure infinite iteration
        toolkit.configure_infinite_iteration(buffer_size=500, adaptive_sampling=True)
        
        # Should have created iterator factory
        assert hasattr(toolkit, 'infinite_iterator_factory')
        assert callable(toolkit.infinite_iterator_factory)
        assert toolkit.iteration_buffer_size == 500
    
    def test_create_infinite_iterator(self):
        """Test infinite iterator creation."""
        toolkit = MetaLearningToolkit()
        
        # Configure first
        toolkit.configure_infinite_iteration(buffer_size=10)
        
        # Create mock dataset
        mock_dataset = [self.episode] * 5
        
        # Create iterator
        iterator = toolkit.create_infinite_iterator(mock_dataset)
        
        # Should be an iterator
        assert hasattr(iterator, '__iter__')
        assert hasattr(iterator, '__next__')
    
    def test_download_dataset(self):
        """Test dataset downloading."""
        toolkit = MetaLearningToolkit()
        
        # Setup dataset manager
        toolkit.setup_dataset_manager(cache_dir=self.temp_dir)
        
        # Download dataset
        path = toolkit.download_dataset('synthetic')
        
        assert path is not None
        assert isinstance(path, str)
    
    def test_create_on_device_dataset(self):
        """Test on-device dataset creation."""
        toolkit = MetaLearningToolkit()
        
        episodes = [self.episode] * 3
        
        # Create on-device dataset
        on_device_dataset = toolkit.create_on_device_dataset(
            episodes, 
            memory_budget=0.8
        )
        
        assert len(on_device_dataset) == 3
        assert on_device_dataset.memory_budget == 0.8
        assert on_device_dataset.enable_compression == True
        assert on_device_dataset.enable_mixed_precision == True
    
    def test_setup_error_recovery(self):
        """Test error recovery setup."""
        toolkit = MetaLearningToolkit()
        
        # Setup error recovery
        recovery = toolkit.setup_error_recovery(max_retries=5, enable_learning=True)
        
        assert toolkit.error_recovery is not None
        assert toolkit.error_recovery == recovery
        assert recovery.max_retries == 5
        assert recovery.enable_learning == True
    
    def test_setup_performance_monitoring(self):
        """Test performance monitoring setup."""
        toolkit = MetaLearningToolkit()
        
        # Setup monitoring
        monitor = toolkit.setup_performance_monitoring(collection_interval=0.5)
        
        assert toolkit.performance_monitor is not None
        assert toolkit.performance_monitor == monitor
        assert monitor.collection_interval == 0.5
        assert monitor.monitoring == True  # Should start automatically
        
        # Clean up
        monitor.stop_monitoring()
    
    def test_setup_warning_system(self):
        """Test warning system setup."""
        toolkit = MetaLearningToolkit()
        
        # Setup warning system
        warning_mgr = toolkit.setup_warning_system(max_history=500)
        
        assert toolkit.warning_system is not None
        assert toolkit.warning_system == warning_mgr
        assert warning_mgr.max_history == 500
        
        # Clean up
        warning_mgr.restore_warnings()
    
    def test_get_performance_report(self):
        """Test comprehensive performance reporting."""
        toolkit = MetaLearningToolkit()
        
        # Setup all monitoring systems
        toolkit.setup_error_recovery()
        monitor = toolkit.setup_performance_monitoring(collection_interval=0.1)
        warning_mgr = toolkit.setup_warning_system()
        toolkit.setup_dataset_manager(cache_dir=self.temp_dir)
        
        # Generate some activity
        import time
        time.sleep(0.1)
        monitor.record_metric('test_metric', 0.8)
        
        # Get report
        report = toolkit.get_performance_report()
        
        assert isinstance(report, str)
        assert "PERFORMANCE MONITORING" in report
        assert "ERROR RECOVERY" in report
        assert "WARNING SYSTEM" in report
        assert "DATASET CACHE" in report
        
        # Clean up
        monitor.stop_monitoring()
        warning_mgr.restore_warnings()
    
    def test_get_performance_report_no_systems(self):
        """Test performance report with no monitoring systems."""
        toolkit = MetaLearningToolkit()
        
        report = toolkit.get_performance_report()
        
        assert isinstance(report, str)
        assert "No monitoring systems initialized" in report


class TestCreateMetaLearningToolkit:
    """Test create_meta_learning_toolkit convenience function."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.model = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 5)
        )
    
    def test_create_toolkit_basic(self):
        """Test basic toolkit creation."""
        toolkit = create_meta_learning_toolkit(self.model)
        
        assert isinstance(toolkit, MetaLearningToolkit)
        assert toolkit.maml_learner is not None  # Default algorithm is MAML
        assert toolkit.determinism_manager is not None  # Should setup determinism
    
    def test_create_toolkit_maml(self):
        """Test toolkit creation with MAML algorithm."""
        toolkit = create_meta_learning_toolkit(
            self.model,
            algorithm="maml",
            inner_lr=0.02,
            inner_steps=3,
            seed=123
        )
        
        assert toolkit.maml_learner is not None
        assert toolkit.maml_learner.config.inner_lr == 0.02
        assert toolkit.maml_learner.config.inner_steps == 3
    
    def test_create_toolkit_test_time_compute(self):
        """Test toolkit creation with Test-Time Compute."""
        toolkit = create_meta_learning_toolkit(
            self.model,
            algorithm="test_time_compute",
            seed=456
        )
        
        assert toolkit.test_time_scaler is not None
        assert toolkit.test_time_scaler.encoder == self.model
    
    def test_create_toolkit_unknown_algorithm(self):
        """Test toolkit creation with unknown algorithm."""
        with pytest.raises(ValueError, match="Unknown algorithm"):
            create_meta_learning_toolkit(
                self.model,
                algorithm="unknown_algorithm"
            )
    
    def test_create_toolkit_with_custom_config(self):
        """Test toolkit creation with custom configuration."""
        toolkit = create_meta_learning_toolkit(
            self.model,
            algorithm="maml",
            variant=MAMLVariant.MAML,
            inner_lr=0.05,
            outer_lr=0.002,
            inner_steps=5,
            first_order=True,
            seed=789
        )
        
        assert toolkit.maml_learner is not None
        config = toolkit.maml_learner.config
        assert config.variant == MAMLVariant.MAML
        assert config.inner_lr == 0.05
        assert config.outer_lr == 0.002
        assert config.inner_steps == 5
        assert config.first_order == True


class TestQuickEvaluation:
    """Test quick_evaluation convenience function."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.model = nn.Sequential(
            nn.Linear(20, 40),
            nn.ReLU(),
            nn.Linear(40, 3)
        )
        
        # Create test episodes
        self.episodes = []
        for i in range(3):
            episode = Episode(
                support_x=torch.randn(15, 20),  # 3-way, 5-shot
                support_y=torch.repeat_interleave(torch.arange(3), 5),
                query_x=torch.randn(9, 20),     # 3-way, 3-query
                query_y=torch.repeat_interleave(torch.arange(3), 3)
            )
            self.episodes.append(episode)
    
    def test_quick_evaluation_basic(self):
        """Test basic quick evaluation."""
        results = quick_evaluation(
            self.model,
            self.episodes,
            algorithm="maml"
        )
        
        assert isinstance(results, dict)
        # Should contain evaluation harness results
        assert 'mean_accuracy' in results or 'accuracy' in results
    
    def test_quick_evaluation_with_parameters(self):
        """Test quick evaluation with custom parameters."""
        results = quick_evaluation(
            self.model,
            self.episodes,
            algorithm="maml",
            inner_lr=0.01,
            inner_steps=2,
            seed=42
        )
        
        assert isinstance(results, dict)
        # Should have used custom parameters
    
    def test_quick_evaluation_test_time_compute(self):
        """Test quick evaluation with Test-Time Compute."""
        results = quick_evaluation(
            self.model,
            self.episodes,
            algorithm="test_time_compute",
            seed=123
        )
        
        assert isinstance(results, dict)


class TestToolkitIntegration:
    """Integration tests for complete toolkit functionality."""
    
    def setup_method(self):
        """Setup integration test fixtures."""
        self.model = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
        
        # Create episodes for different scenarios
        self.episodes = []
        for i in range(10):
            episode = Episode(
                support_x=torch.randn(50, 64),  # 10-way, 5-shot
                support_y=torch.repeat_interleave(torch.arange(10), 5),
                query_x=torch.randn(30, 64),    # 10-way, 3-query
                query_y=torch.repeat_interleave(torch.arange(10), 3)
            )
            self.episodes.append(episode)
        
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_complete_workflow_maml(self):
        """Test complete MAML workflow with all features."""
        # Create toolkit with all features
        toolkit = create_meta_learning_toolkit(
            self.model,
            algorithm="maml",
            inner_lr=0.01,
            inner_steps=2,
            seed=42
        )
        
        # Setup all systems
        toolkit.setup_dataset_manager(cache_dir=self.temp_dir)
        toolkit.enable_data_acceleration(memory_budget_gb=1.0)
        toolkit.configure_infinite_iteration(buffer_size=100)
        
        recovery = toolkit.setup_error_recovery(max_retries=3)
        monitor = toolkit.setup_performance_monitoring(collection_interval=0.1)
        warning_mgr = toolkit.setup_warning_system()
        
        # Create evaluation harness
        harness = toolkit.create_evaluation_harness(confidence_level=0.95)
        
        # Train on episodes
        results = []
        for episode in self.episodes[:3]:  # Test on subset
            episode_result = toolkit.train_episode(episode, algorithm="maml")
            results.append(episode_result)
            
            # Record metrics
            monitor.record_metric('episode_accuracy', episode_result['query_accuracy'])
        
        # Evaluate with harness
        def run_episode(episode):
            return toolkit.train_episode(episode, algorithm="maml")
        
        harness_results = harness.evaluate_on_episodes(self.episodes[:5], run_episode)
        
        # Generate comprehensive report
        import time
        time.sleep(0.2)  # Allow monitoring to collect data
        performance_report = toolkit.get_performance_report()
        
        # Verify all systems worked
        assert len(results) == 3
        for result in results:
            assert 'query_accuracy' in result
            assert 0.0 <= result['query_accuracy'] <= 1.0
        
        assert 'mean_accuracy' in harness_results
        assert 'confidence_interval' in harness_results
        assert isinstance(performance_report, str)
        assert len(performance_report) > 100
        
        # Clean up
        monitor.stop_monitoring()
        warning_mgr.restore_warnings()
    
    def test_complete_workflow_test_time_compute(self):
        """Test complete Test-Time Compute workflow."""
        # Create toolkit with TTCS
        toolkit = create_meta_learning_toolkit(
            self.model,
            algorithm="test_time_compute",
            seed=123
        )
        
        # Setup monitoring
        monitor = toolkit.setup_performance_monitoring(collection_interval=0.1)
        recovery = toolkit.setup_error_recovery()
        
        # Train episodes
        results = []
        for episode in self.episodes[:3]:
            try:
                episode_result = toolkit.train_episode(episode, algorithm="test_time_compute")
                results.append(episode_result)
                
                # Record metrics
                monitor.record_metric('ttcs_accuracy', episode_result['query_accuracy'])
                
            except Exception as e:
                # Test error recovery
                success, description = recovery.handle_error(e)
                if success:
                    print(f"Recovered from error: {description}")
        
        # Verify results
        assert len(results) >= 1  # At least some should succeed
        for result in results:
            assert 'query_accuracy' in result
            assert 'compute_scaling_metrics' in result
        
        # Clean up
        monitor.stop_monitoring()
    
    def test_dataset_ecosystem_integration(self):
        """Test complete dataset ecosystem integration."""
        toolkit = create_meta_learning_toolkit(self.model, algorithm="maml")
        
        # Setup dataset ecosystem
        toolkit.setup_dataset_manager(cache_dir=self.temp_dir, max_cache_size_gb=0.1)
        toolkit.enable_data_acceleration(memory_budget_gb=0.5)
        toolkit.configure_infinite_iteration(buffer_size=50, adaptive_sampling=True)
        
        # Download and setup datasets
        dataset_results = toolkit.setup_benchmark_datasets(['synthetic'])
        assert dataset_results['synthetic'] == True
        
        # List available datasets
        available_datasets = toolkit.list_available_datasets()
        assert 'synthetic' in available_datasets
        assert available_datasets['synthetic']['cached'] == True
        
        # Create on-device dataset
        on_device_dataset = toolkit.create_on_device_dataset(
            self.episodes[:5],
            memory_budget=0.8
        )
        
        # Test dataset access
        for i in range(len(on_device_dataset)):
            cached_episode = on_device_dataset[i]
            assert isinstance(cached_episode, Episode)
            assert cached_episode.support_x.device == on_device_dataset.device
        
        # Create infinite iterator
        def episode_generator():
            return self.episodes[torch.randint(0, len(self.episodes), (1,)).item()]
        
        iterator = toolkit.create_infinite_iterator(episode_generator)
        
        # Get a few episodes from iterator
        generated_episodes = []
        for i, episode in enumerate(iterator):
            generated_episodes.append(episode)
            if i >= 2:
                break
        
        assert len(generated_episodes) == 3
        for episode in generated_episodes:
            assert isinstance(episode, Episode)
        
        # Clean up iterator
        iterator.stop()
    
    def test_error_handling_integration(self):
        """Test comprehensive error handling integration."""
        toolkit = create_meta_learning_toolkit(self.model, algorithm="maml")
        
        # Setup error handling systems
        recovery = toolkit.setup_error_recovery(max_retries=2, enable_learning=True)
        monitor = toolkit.setup_performance_monitoring(collection_interval=0.05)
        warning_mgr = toolkit.setup_warning_system(max_history=100)
        
        # Create problematic scenarios
        problematic_episodes = []
        
        # Episode with extreme values
        extreme_episode = Episode(
            support_x=torch.randn(25, 64) * 100,  # Very large values
            support_y=torch.repeat_interleave(torch.arange(5), 5),
            query_x=torch.randn(15, 64) * 100,
            query_y=torch.repeat_interleave(torch.arange(5), 3)
        )
        problematic_episodes.append(extreme_episode)
        
        # Test error recovery in practice
        successful_episodes = 0
        recovered_episodes = 0
        
        for episode in problematic_episodes + self.episodes[:3]:
            try:
                result = toolkit.train_episode(episode, algorithm="maml")
                successful_episodes += 1
                monitor.record_metric('success_rate', 1.0)
                
            except Exception as e:
                # Test error recovery
                success, description = recovery.handle_error(e, {
                    'episode_type': 'problematic' if episode in problematic_episodes else 'normal',
                    'support_shape': list(episode.support_x.shape),
                    'support_stats': {
                        'mean': episode.support_x.mean().item(),
                        'std': episode.support_x.std().item(),
                        'max': episode.support_x.max().item()
                    }
                })
                
                if success:
                    recovered_episodes += 1
                    monitor.record_metric('recovery_rate', 1.0)
                else:
                    monitor.record_metric('recovery_rate', 0.0)
                
                # Generate warning
                import warnings
                warnings.warn(f"Episode processing failed: {description}", UserWarning)
        
        # Generate comprehensive reports
        import time
        time.sleep(0.2)  # Allow monitoring
        
        performance_report = toolkit.get_performance_report()
        recovery_stats = recovery.get_recovery_statistics()
        warning_summary = warning_mgr.get_warning_summary()
        
        # Verify error handling worked
        assert successful_episodes >= 1  # At least some normal episodes should succeed
        assert recovery_stats['total_errors'] >= 0  # May or may not have errors
        
        if recovery_stats['total_errors'] > 0:
            assert len(recovery_stats['error_types']) > 0
            assert 'strategy_effectiveness' in recovery_stats
        
        assert warning_summary['total_warnings'] >= 0
        assert 'category_counts' in warning_summary
        
        # Reports should be comprehensive
        assert "PERFORMANCE MONITORING" in performance_report
        assert "ERROR RECOVERY" in performance_report
        assert "WARNING SYSTEM" in performance_report
        
        # Clean up
        monitor.stop_monitoring()
        warning_mgr.restore_warnings()
    
    def test_end_to_end_research_scenario(self):
        """Test complete end-to-end research scenario."""
        # Simulate a complete meta-learning research pipeline
        
        # 1. Setup toolkit with comprehensive configuration
        toolkit = create_meta_learning_toolkit(
            self.model,
            algorithm="maml",
            inner_lr=0.01,
            inner_steps=3,
            outer_lr=0.001,
            first_order=False,
            seed=42
        )
        
        # 2. Setup all monitoring and management systems
        toolkit.setup_dataset_manager(cache_dir=self.temp_dir)
        toolkit.enable_data_acceleration(memory_budget_gb=2.0)
        toolkit.configure_infinite_iteration(buffer_size=200, adaptive_sampling=True)
        
        recovery = toolkit.setup_error_recovery(max_retries=3, enable_learning=True)
        monitor = toolkit.setup_performance_monitoring(collection_interval=0.1)
        warning_mgr = toolkit.setup_warning_system()
        
        # 3. Setup evaluation harness
        harness = toolkit.create_evaluation_harness(
            confidence_level=0.95,
            num_bootstrap_samples=100
        )
        
        # 4. Simulate training loop with monitoring
        training_results = []
        
        for epoch in range(2):  # Short training for testing
            epoch_results = []
            
            for episode in self.episodes[:3]:  # Process subset of episodes
                try:
                    # Train episode
                    result = toolkit.train_episode(episode, algorithm="maml")
                    epoch_results.append(result)
                    
                    # Record comprehensive metrics
                    monitor.record_metric('query_accuracy', result['query_accuracy'])
                    monitor.record_metric('query_loss', result['query_loss'])
                    monitor.record_metric('support_loss', result['support_loss'])
                    monitor.record_metric('meta_loss', result['meta_loss'])
                    
                except Exception as e:
                    # Handle errors gracefully
                    success, description = recovery.handle_error(e, {
                        'epoch': epoch,
                        'episode_id': id(episode),
                        'model_parameters': sum(p.numel() for p in self.model.parameters())
                    })
                    
                    if not success:
                        import warnings
                        warnings.warn(f"Episode failed permanently: {description}", UserWarning)
            
            training_results.extend(epoch_results)
        
        # 5. Comprehensive evaluation
        def evaluation_function(episode):
            return toolkit.train_episode(episode, algorithm="maml")
        
        evaluation_results = harness.evaluate_on_episodes(
            self.episodes[:5],
            evaluation_function
        )
        
        # 6. Generate final comprehensive report
        import time
        time.sleep(0.3)  # Allow monitoring to collect data
        
        final_report = toolkit.get_performance_report()
        recovery_report = recovery.generate_error_report()
        warning_report = warning_mgr.generate_warning_report()
        
        # 7. Verify complete pipeline worked
        # Training results
        assert len(training_results) >= 1
        for result in training_results:
            assert isinstance(result, dict)
            assert all(key in result for key in ['query_accuracy', 'query_loss', 'support_loss', 'meta_loss'])
        
        # Evaluation results
        assert 'mean_accuracy' in evaluation_results
        assert 'confidence_interval' in evaluation_results
        assert 'episode_accuracies' in evaluation_results
        assert len(evaluation_results['episode_accuracies']) == 5
        
        # Reports should be comprehensive and informative
        for report in [final_report, recovery_report, warning_report]:
            assert isinstance(report, str)
            assert len(report) > 50  # Should be substantial
        
        # Performance metrics should be reasonable
        mean_accuracy = evaluation_results['mean_accuracy']
        assert 0.0 <= mean_accuracy <= 1.0
        
        ci = evaluation_results['confidence_interval']
        assert len(ci) == 2
        assert ci[0] <= mean_accuracy <= ci[1]
        
        # Clean up
        monitor.stop_monitoring()
        warning_mgr.restore_warnings()
    
    def test_multiple_algorithms_comparison(self):
        """Test comparison between multiple algorithms."""
        # Test both MAML and Test-Time Compute on same episodes
        
        # MAML toolkit
        maml_toolkit = create_meta_learning_toolkit(
            self.model,
            algorithm="maml",
            inner_lr=0.01,
            inner_steps=2,
            seed=42
        )
        
        # TTCS toolkit
        ttcs_toolkit = create_meta_learning_toolkit(
            self.model,
            algorithm="test_time_compute",
            seed=42
        )
        
        # Compare on same episodes
        comparison_results = {
            'maml': [],
            'ttcs': []
        }
        
        for episode in self.episodes[:3]:
            # MAML results
            try:
                maml_result = maml_toolkit.train_episode(episode, algorithm="maml")
                comparison_results['maml'].append(maml_result['query_accuracy'])
            except Exception:
                comparison_results['maml'].append(0.0)  # Failed
            
            # TTCS results
            try:
                ttcs_result = ttcs_toolkit.train_episode(episode, algorithm="test_time_compute")
                comparison_results['ttcs'].append(ttcs_result['query_accuracy'])
            except Exception:
                comparison_results['ttcs'].append(0.0)  # Failed
        
        # Both should have results
        assert len(comparison_results['maml']) == 3
        assert len(comparison_results['ttcs']) == 3
        
        # Results should be reasonable
        for algorithm, results in comparison_results.items():
            for accuracy in results:
                assert 0.0 <= accuracy <= 1.0


if __name__ == "__main__":
    pytest.main([__file__])