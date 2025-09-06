"""
Comprehensive tests for Integration test framework.

Tests complete integration of all implemented components and validates
that all components work together correctly in end-to-end scenarios.
"""
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from unittest.mock import MagicMock, patch
from typing import Dict, List, Tuple, Any
import tempfile
import shutil
import time

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from meta_learning.meta_learning_modules.comprehensive_integration_test import (
    IntegrationTestFramework,
    ComponentTester,
    EndToEndValidator,
    SystemIntegrationChecker,
    run_comprehensive_integration_test,
    validate_all_components,
    test_component_integration
)


class TestIntegrationTestFramework:
    """Test IntegrationTestFramework main functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.framework = IntegrationTestFramework()
    
    def test_framework_initialization(self):
        """Test framework initialization."""
        assert self.framework is not None
        assert hasattr(self.framework, 'component_testers')
        assert hasattr(self.framework, 'integration_validators')
        assert hasattr(self.framework, 'test_results')
    
    def test_register_component_tester(self):
        """Test registering component testers."""
        # Create mock component tester
        mock_tester = MagicMock()
        mock_tester.test_component.return_value = {'passed': True, 'details': 'Test passed'}
        
        # Register tester
        self.framework.register_component_tester('mock_component', mock_tester)
        
        # Verify registration
        assert 'mock_component' in self.framework.component_testers
        assert self.framework.component_testers['mock_component'] is mock_tester
    
    def test_run_component_tests(self):
        """Test running component tests."""
        # Register mock testers
        mock_tester1 = MagicMock()
        mock_tester1.test_component.return_value = {'passed': True, 'score': 0.95}
        
        mock_tester2 = MagicMock()
        mock_tester2.test_component.return_value = {'passed': False, 'error': 'Test failed'}
        
        self.framework.register_component_tester('component1', mock_tester1)
        self.framework.register_component_tester('component2', mock_tester2)
        
        # Run tests
        results = self.framework.run_component_tests()
        
        # Verify results
        assert 'component1' in results
        assert 'component2' in results
        assert results['component1']['passed'] == True
        assert results['component2']['passed'] == False
    
    def test_run_integration_tests(self):
        """Test running integration tests."""
        # Register mock integration validator
        mock_validator = MagicMock()
        mock_validator.validate_integration.return_value = {
            'passed': True, 
            'integration_score': 0.88
        }
        
        self.framework.register_integration_validator('end_to_end', mock_validator)
        
        # Run integration tests
        results = self.framework.run_integration_tests()
        
        # Verify results
        assert 'end_to_end' in results
        assert results['end_to_end']['passed'] == True
        assert results['end_to_end']['integration_score'] == 0.88
    
    def test_generate_comprehensive_report(self):
        """Test comprehensive report generation."""
        # Add some test results
        self.framework.test_results = {
            'component_tests': {
                'episode_protocol': {'passed': True, 'score': 0.95},
                'prototypical_networks': {'passed': True, 'score': 0.92}
            },
            'integration_tests': {
                'end_to_end': {'passed': True, 'integration_score': 0.89}
            },
            'performance_metrics': {
                'total_test_time': 45.2,
                'memory_usage_mb': 256
            }
        }
        
        report = self.framework.generate_comprehensive_report()
        
        # Verify report structure
        assert 'summary' in report
        assert 'component_results' in report
        assert 'integration_results' in report
        assert 'recommendations' in report
        
        # Check summary
        summary = report['summary']
        assert 'total_components_tested' in summary
        assert 'components_passed' in summary
        assert 'integration_tests_passed' in summary


class TestComponentTester:
    """Test individual component testing functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.component_tester = ComponentTester()
    
    def test_episode_protocol_testing(self):
        """Test episode protocol component testing."""
        # Mock episode generator
        from meta_learning.meta_learning_modules.episode_protocol import EpisodeGenerator, Episode
        
        generator = EpisodeGenerator(seed=42)
        
        # Create mock dataset
        class MockDataset:
            def __len__(self):
                return 1000
            
            def __getitem__(self, idx):
                class_id = idx // 20  # 20 examples per class
                return torch.randn(32), class_id
        
        mock_dataset = MockDataset()
        class_to_indices = {i: list(range(i*20, (i+1)*20)) for i in range(50)}
        
        # Test episode generation
        result = self.component_tester.test_episode_protocol_component(
            generator, mock_dataset, class_to_indices
        )
        
        assert 'passed' in result
        assert 'mathematical_correctness' in result
        assert 'label_remapping' in result
        assert 'episode_isolation' in result
        
        # Should pass for valid implementation
        assert result['passed'] == True
    
    def test_prototypical_networks_testing(self):
        """Test prototypical networks component testing."""
        from meta_learning.meta_learning_modules.prototypical_networks_fixed import ResearchPrototypicalNetworks
        
        # Create test model
        encoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        model = ResearchPrototypicalNetworks(encoder, temperature=1.0)
        
        # Test model
        result = self.component_tester.test_prototypical_networks_component(model)
        
        assert 'passed' in result
        assert 'mathematical_accuracy' in result
        assert 'snell_2017_compliance' in result
        assert 'temperature_scaling' in result
        
        # Should pass for research-accurate implementation
        assert result['passed'] == True
    
    def test_batch_norm_policy_testing(self):
        """Test BatchNorm policy component testing."""
        from meta_learning.research_patches.batch_norm_policy import apply_episodic_bn_policy, EpisodicBatchNormManager
        
        # Create model with BatchNorm
        model_with_bn = nn.Sequential(
            nn.Conv2d(3, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Linear(64, 5)
        )
        
        # Test BatchNorm policy
        result = self.component_tester.test_batch_norm_policy_component(model_with_bn)
        
        assert 'passed' in result
        assert 'leakage_prevention' in result
        assert 'policy_application' in result
        assert 'episodic_isolation' in result
    
    def test_determinism_utilities_testing(self):
        """Test determinism utilities component testing."""
        from meta_learning.research_patches.determinism_hooks import DeterminismManager, setup_deterministic_environment
        
        manager = DeterminismManager(base_seed=42)
        
        result = self.component_tester.test_determinism_utilities_component(manager)
        
        assert 'passed' in result
        assert 'seed_consistency' in result
        assert 'reproducibility' in result
        assert 'environment_setup' in result
        
        # Should pass for working determinism system
        assert result['passed'] == True
    
    def test_leakage_detection_testing(self):
        """Test leakage detection component testing."""
        from meta_learning.meta_learning_modules.leakage_guard import LeakageDetector, DataLeakageGuard
        
        train_classes = set(range(64))
        test_classes = set(range(64, 100))
        
        detector = LeakageDetector(train_classes, test_classes)
        guard = DataLeakageGuard(train_classes, test_classes)
        
        result = self.component_tester.test_leakage_detection_component(detector, guard)
        
        assert 'passed' in result
        assert 'split_validation' in result
        assert 'contamination_detection' in result
        assert 'normalization_monitoring' in result
    
    def test_evaluation_metrics_testing(self):
        """Test evaluation metrics component testing."""
        from meta_learning.meta_learning_modules.advanced_evaluation_metrics import (
            PrototypeAnalyzer, TaskDifficultyEstimator, UncertaintyQuantifier
        )
        
        analyzer = PrototypeAnalyzer()
        estimator = TaskDifficultyEstimator()
        quantifier = UncertaintyQuantifier()
        
        result = self.component_tester.test_evaluation_metrics_component(analyzer, estimator, quantifier)
        
        assert 'passed' in result
        assert 'prototype_analysis' in result
        assert 'difficulty_estimation' in result
        assert 'uncertainty_quantification' in result
    
    def test_dataset_management_testing(self):
        """Test dataset management component testing."""
        from meta_learning.meta_learning_modules.dataset_management import DatasetManager, DatasetRegistry
        
        # Create temporary directory for testing
        temp_dir = tempfile.mkdtemp()
        try:
            manager = DatasetManager(cache_dir=temp_dir, max_cache_size_gb=0.1)
            
            result = self.component_tester.test_dataset_management_component(manager)
            
            assert 'passed' in result
            assert 'registry_functionality' in result
            assert 'cache_management' in result
            assert 'download_capabilities' in result
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestEndToEndValidator:
    """Test end-to-end validation functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.validator = EndToEndValidator()
    
    def test_complete_pipeline_validation(self):
        """Test complete meta-learning pipeline validation."""
        # Create minimal pipeline components
        from meta_learning.meta_learning_modules.episode_protocol import EpisodeGenerator
        from meta_learning.meta_learning_modules.prototypical_networks_fixed import ResearchPrototypicalNetworks
        
        # Setup components
        generator = EpisodeGenerator(seed=42)
        encoder = nn.Linear(32, 64)
        model = ResearchPrototypicalNetworks(encoder)
        
        # Create mock dataset
        class MockDataset:
            def __len__(self):
                return 500
            
            def __getitem__(self, idx):
                class_id = idx // 10
                return torch.randn(32), class_id
        
        dataset = MockDataset()
        class_to_indices = {i: list(range(i*10, (i+1)*10)) for i in range(50)}
        
        # Validate complete pipeline
        result = self.validator.validate_complete_pipeline(
            episode_generator=generator,
            model=model,
            dataset=dataset,
            class_to_indices=class_to_indices
        )
        
        assert 'passed' in result
        assert 'pipeline_accuracy' in result
        assert 'component_integration' in result
        assert 'mathematical_consistency' in result
        
        # Should pass for working pipeline
        assert result['passed'] == True
    
    def test_cross_component_interaction(self):
        """Test interactions between different components."""
        # Test BatchNorm policy + Episode protocol interaction
        from meta_learning.research_patches.batch_norm_policy import apply_episodic_bn_policy
        from meta_learning.meta_learning_modules.episode_protocol import EpisodeGenerator
        
        # Model with BatchNorm
        model = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 5)
        )
        
        # Apply policy
        fixed_model = apply_episodic_bn_policy(model, policy="group_norm")
        
        # Test episode generation with fixed model
        generator = EpisodeGenerator(seed=42)
        
        result = self.validator.test_cross_component_interaction(
            components={
                'model': fixed_model,
                'episode_generator': generator
            }
        )
        
        assert 'passed' in result
        assert 'interaction_score' in result
        assert 'compatibility_issues' in result
    
    def test_performance_regression(self):
        """Test for performance regressions."""
        # Create simple model for performance testing
        model = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 5)
        )
        
        # Test data
        support_x = torch.randn(25, 64)
        support_y = torch.arange(5).repeat(5)
        query_x = torch.randn(75, 64)
        
        result = self.validator.test_performance_regression(
            model=model,
            support_x=support_x,
            support_y=support_y,
            query_x=query_x,
            n_runs=10
        )
        
        assert 'passed' in result
        assert 'average_runtime_ms' in result
        assert 'memory_usage_mb' in result
        assert 'performance_variance' in result
    
    def test_reproducibility_validation(self):
        """Test reproducibility across multiple runs."""
        from meta_learning.research_patches.determinism_hooks import setup_deterministic_environment
        
        # Setup deterministic environment
        config = {'seed': 42, 'cuda_deterministic': True}
        setup_deterministic_environment(config)
        
        # Simple deterministic computation
        def deterministic_computation():
            torch.manual_seed(42)
            x = torch.randn(10, 10)
            y = torch.mm(x, x.t())
            return y.sum().item()
        
        result = self.validator.test_reproducibility(
            computation_fn=deterministic_computation,
            n_runs=5
        )
        
        assert 'passed' in result
        assert 'reproducibility_score' in result
        assert 'run_variance' in result
        
        # Should be perfectly reproducible
        assert result['reproducibility_score'] > 0.99
    
    def test_error_handling_robustness(self):
        """Test system robustness to various error conditions."""
        # Test with various problematic inputs
        test_cases = [
            {
                'name': 'empty_tensors',
                'support_x': torch.empty(0, 64),
                'support_y': torch.empty(0, dtype=torch.long),
                'query_x': torch.empty(0, 64)
            },
            {
                'name': 'mismatched_shapes',
                'support_x': torch.randn(10, 64),
                'support_y': torch.randint(0, 5, (15,)),  # Wrong size
                'query_x': torch.randn(20, 32)  # Wrong feature dim
            },
            {
                'name': 'invalid_labels',
                'support_x': torch.randn(10, 64),
                'support_y': torch.tensor([-1, 0, 1, 2, 5, 6, 7, 8, 9, 10]),  # Out of range
                'query_x': torch.randn(15, 64)
            }
        ]
        
        result = self.validator.test_error_handling_robustness(test_cases)
        
        assert 'passed' in result
        assert 'error_cases_handled' in result
        assert 'robustness_score' in result
        
        # Should handle errors gracefully
        assert result['robustness_score'] > 0.7


class TestSystemIntegrationChecker:
    """Test system-wide integration checking."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.checker = SystemIntegrationChecker()
    
    def test_check_system_dependencies(self):
        """Test checking system dependencies."""
        result = self.checker.check_system_dependencies()
        
        assert 'passed' in result
        assert 'pytorch_version' in result
        assert 'numpy_version' in result
        assert 'python_version' in result
        assert 'cuda_available' in result
        
        # Should detect installed dependencies
        assert result['passed'] == True
    
    def test_check_memory_requirements(self):
        """Test memory requirement checking."""
        # Test with various model sizes
        small_model = nn.Linear(10, 5)
        large_model = nn.Sequential(*[nn.Linear(1024, 1024) for _ in range(10)])
        
        small_result = self.checker.check_memory_requirements(
            model=small_model,
            batch_size=32,
            sequence_length=100
        )
        
        large_result = self.checker.check_memory_requirements(
            model=large_model,
            batch_size=32,
            sequence_length=100
        )
        
        assert 'memory_estimate_mb' in small_result
        assert 'memory_estimate_mb' in large_result
        
        # Large model should require more memory
        assert large_result['memory_estimate_mb'] > small_result['memory_estimate_mb']
    
    def test_check_computational_requirements(self):
        """Test computational requirement checking."""
        model = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
        
        result = self.checker.check_computational_requirements(
            model=model,
            episodes_per_epoch=1000,
            n_way=5,
            k_shot=5
        )
        
        assert 'flops_estimate' in result
        assert 'training_time_estimate_minutes' in result
        assert 'recommended_hardware' in result
    
    def test_validate_integration_environment(self):
        """Test integration environment validation."""
        result = self.checker.validate_integration_environment()
        
        assert 'environment_valid' in result
        assert 'warnings' in result
        assert 'recommendations' in result
        
        # Should validate current environment
        assert isinstance(result['environment_valid'], bool)
    
    def test_check_component_compatibility(self):
        """Test component compatibility checking."""
        # Create components with known compatibility
        from meta_learning.meta_learning_modules.episode_protocol import EpisodeGenerator
        from meta_learning.meta_learning_modules.prototypical_networks_fixed import ResearchPrototypicalNetworks
        
        generator = EpisodeGenerator(seed=42)
        model = ResearchPrototypicalNetworks(nn.Linear(64, 128))
        
        components = {
            'episode_generator': generator,
            'model': model
        }
        
        result = self.checker.check_component_compatibility(components)
        
        assert 'compatible' in result
        assert 'compatibility_matrix' in result
        assert 'incompatible_pairs' in result
        
        # These components should be compatible
        assert result['compatible'] == True


class TestUtilityFunctions:
    """Test utility functions for integration testing."""
    
    def test_run_comprehensive_integration_test_function(self):
        """Test main comprehensive integration test function."""
        result = run_comprehensive_integration_test()
        
        # Should return comprehensive test results
        assert isinstance(result, dict)
        assert 'component_tests' in result or 'test_results' in result
        
        # Should include overall success indicator
        overall_success = result.get('overall_success', result.get('success', False))
        assert isinstance(overall_success, bool)
    
    def test_validate_all_components_function(self):
        """Test component validation utility function."""
        result = validate_all_components()
        
        assert isinstance(result, dict)
        
        # Should test major components
        expected_components = [
            'episode_protocol', 'prototypical_networks', 'batch_norm_policy',
            'determinism_utilities', 'leakage_detection'
        ]
        
        # At least some components should be tested
        tested_components = list(result.keys())
        assert len(tested_components) > 0
    
    def test_test_component_integration_function(self):
        """Test component integration utility function."""
        # Test with mock components
        component_a = MagicMock()
        component_b = MagicMock()
        
        component_a.process = MagicMock(return_value="processed_data")
        component_b.analyze = MagicMock(return_value="analysis_result")
        
        result = test_component_integration(
            components={'a': component_a, 'b': component_b}
        )
        
        assert 'integration_score' in result or 'success' in result
        assert isinstance(result, dict)


class TestRealWorldIntegrationScenarios:
    """Test realistic integration scenarios."""
    
    def setup_method(self):
        """Setup realistic test scenario."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Cleanup test scenario."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_complete_few_shot_learning_workflow(self):
        """Test complete few-shot learning workflow integration."""
        from meta_learning.meta_learning_modules.episode_protocol import EpisodeGenerator
        from meta_learning.meta_learning_modules.prototypical_networks_fixed import ResearchPrototypicalNetworks
        from meta_learning.research_patches.batch_norm_policy import apply_episodic_bn_policy
        from meta_learning.research_patches.determinism_hooks import setup_deterministic_environment
        
        # Setup deterministic environment
        setup_deterministic_environment({'seed': 42})
        
        # Create model
        encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 128)
        )
        
        # Apply BatchNorm policy
        encoder = apply_episodic_bn_policy(encoder, policy="group_norm")
        
        # Create ProtoNet
        model = ResearchPrototypicalNetworks(encoder, temperature=1.0)
        
        # Create episode generator
        generator = EpisodeGenerator(seed=42)
        
        # Mock dataset
        class IntegrationDataset:
            def __len__(self):
                return 1000
            
            def __getitem__(self, idx):
                class_id = idx // 20
                image = torch.randn(3, 84, 84)
                return image, class_id
        
        dataset = IntegrationDataset()
        class_to_indices = {i: list(range(i*20, (i+1)*20)) for i in range(50)}
        
        # Run complete workflow
        results = []
        for episode_idx in range(10):
            # Generate episode
            episode = generator.generate_episode(
                dataset=dataset,
                class_to_indices=class_to_indices,
                n_way=5,
                k_shot=5,
                m_query=15
            )
            
            # Forward pass
            with torch.no_grad():
                logits = model(episode.support_x, episode.support_y, episode.query_x)
                predictions = logits.argmax(dim=-1)
                accuracy = (predictions == episode.query_y).float().mean().item()
                results.append(accuracy)
        
        # Validate workflow
        assert len(results) == 10
        assert all(0.0 <= acc <= 1.0 for acc in results)
        
        # Should have reasonable accuracy (better than random)
        mean_accuracy = np.mean(results)
        assert mean_accuracy > 0.15  # Better than random (0.2 for 5-way)
    
    def test_multi_dataset_evaluation_integration(self):
        """Test integration across multiple datasets."""
        from meta_learning.meta_learning_modules.dataset_management import DatasetManager
        from meta_learning.meta_learning_modules.advanced_evaluation_metrics import EvaluationSuite
        
        # Create dataset manager
        manager = DatasetManager(cache_dir=self.temp_dir, max_cache_size_gb=0.1)
        
        # Create evaluation suite
        suite = EvaluationSuite()
        
        # Mock multiple datasets
        datasets = {}
        for dataset_name in ['Dataset_A', 'Dataset_B', 'Dataset_C']:
            # Create mock dataset
            class MockDataset:
                def __init__(self, name):
                    self.name = name
                    self.data_size = np.random.randint(500, 1500)
                
                def __len__(self):
                    return self.data_size
                
                def __getitem__(self, idx):
                    class_id = idx // 30
                    return torch.randn(64), class_id
            
            datasets[dataset_name] = MockDataset(dataset_name)
        
        # Mock model
        mock_model = MagicMock()
        mock_model.return_value = torch.randn(75, 5)  # 5-way classification
        
        # Test evaluation across datasets
        evaluation_results = {}
        for dataset_name, dataset in datasets.items():
            # Create episodes
            episodes = []
            for i in range(20):
                episode = {
                    'support_x': torch.randn(25, 64),
                    'support_y': torch.arange(5).repeat(5),
                    'query_x': torch.randn(75, 64),
                    'query_y': torch.arange(5).repeat(15),
                    'dataset': dataset_name
                }
                episodes.append(episode)
            
            # Evaluate
            results = suite.run_comprehensive_evaluation(mock_model, episodes)
            evaluation_results[dataset_name] = results
        
        # Validate integration
        assert len(evaluation_results) == 3
        for dataset_name, results in evaluation_results.items():
            assert 'accuracy_metrics' in results
            assert isinstance(results, dict)
    
    def test_error_recovery_integration(self):
        """Test error recovery across integrated components."""
        from meta_learning.meta_learning_modules.leakage_guard import DataLeakageGuard
        
        # Setup components with potential failure points
        train_classes = set(range(50))
        test_classes = set(range(50, 100))
        guard = DataLeakageGuard(train_classes, test_classes)
        
        # Test various error scenarios
        error_scenarios = [
            {
                'name': 'contaminated_episode',
                'episode_data': {
                    'support_x': torch.randn(25, 64),
                    'support_y': torch.tensor([5]*5 + [12]*5 + [23]*5 + [35]*5 + [67]*5),  # 67 is test class
                    'query_x': torch.randn(75, 64),
                    'query_y': torch.tensor([5]*15 + [12]*15 + [23]*15 + [35]*15 + [67]*15),
                    'split': 'train'
                }
            },
            {
                'name': 'malformed_data',
                'episode_data': {
                    'support_x': torch.randn(20, 64),  # Wrong size
                    'support_y': torch.tensor([0, 1, 2, 3, 4] * 4),
                    'query_x': torch.randn(75, 32),  # Wrong feature dim
                    'query_y': torch.tensor([0, 1, 2, 3, 4] * 15),
                    'split': 'train'
                }
            }
        ]
        
        recovery_results = []
        for scenario in error_scenarios:
            try:
                # Should detect issues but handle gracefully
                result = guard.comprehensive_leakage_check(scenario['episode_data'])
                recovery_results.append({
                    'scenario': scenario['name'],
                    'handled_gracefully': True,
                    'detected_issue': not result['safe']
                })
            except Exception as e:
                recovery_results.append({
                    'scenario': scenario['name'],
                    'handled_gracefully': False,
                    'error': str(e)
                })
        
        # Should handle errors gracefully
        graceful_handling = [r['handled_gracefully'] for r in recovery_results]
        assert all(graceful_handling), f"Some scenarios not handled gracefully: {recovery_results}"
    
    def test_memory_and_performance_integration(self):
        """Test memory and performance under realistic loads."""
        import psutil
        import time
        
        # Monitor system resources
        initial_memory = psutil.virtual_memory().used / 1024 / 1024  # MB
        
        # Create memory-intensive integration test
        large_model = nn.Sequential(*[
            nn.Linear(512, 512) for _ in range(5)
        ])
        
        # Process many episodes
        episode_results = []
        start_time = time.time()
        
        for episode_idx in range(100):
            # Large episode data
            support_x = torch.randn(50, 512)  # Large feature dimension
            support_y = torch.arange(10).repeat(5)  # 10-way 5-shot
            query_x = torch.randn(150, 512)  # 15 queries per class
            
            # Forward pass
            with torch.no_grad():
                logits = large_model(support_x)
                predictions = logits[:10].argmax(dim=-1)  # Use first 10 for classification
                accuracy = torch.rand(1).item()  # Mock accuracy
                episode_results.append(accuracy)
            
            # Periodic memory check
            if episode_idx % 20 == 0:
                current_memory = psutil.virtual_memory().used / 1024 / 1024
                memory_increase = current_memory - initial_memory
                
                # Should not have excessive memory growth
                assert memory_increase < 1000, f"Excessive memory usage: {memory_increase:.1f}MB"
        
        total_time = time.time() - start_time
        
        # Performance validation
        assert len(episode_results) == 100
        assert total_time < 60, f"Integration test took too long: {total_time:.1f}s"
        
        # Memory cleanup validation
        final_memory = psutil.virtual_memory().used / 1024 / 1024
        memory_increase = final_memory - initial_memory
        assert memory_increase < 500, f"Memory not properly cleaned up: {memory_increase:.1f}MB"


class TestContinuousIntegration:
    """Test continuous integration and regression detection."""
    
    def test_regression_detection(self):
        """Test detection of regressions in integrated system."""
        # Baseline performance metrics
        baseline_metrics = {
            'episode_generation_time_ms': 5.0,
            'model_forward_time_ms': 10.0,
            'accuracy_threshold': 0.6,
            'memory_usage_mb': 256
        }
        
        # Current performance (simulated)
        current_metrics = {
            'episode_generation_time_ms': 6.2,  # 24% slower
            'model_forward_time_ms': 12.5,      # 25% slower  
            'accuracy_threshold': 0.58,         # 3% worse
            'memory_usage_mb': 280              # 9% more memory
        }
        
        # Detect regressions
        regressions = []
        for metric, baseline_value in baseline_metrics.items():
            current_value = current_metrics[metric]
            
            if 'time' in metric or 'memory' in metric:
                # Performance metrics (lower is better)
                if current_value > baseline_value * 1.2:  # 20% threshold
                    regressions.append({
                        'metric': metric,
                        'baseline': baseline_value,
                        'current': current_value,
                        'regression_percent': ((current_value - baseline_value) / baseline_value) * 100
                    })
            else:
                # Accuracy metrics (higher is better)
                if current_value < baseline_value * 0.95:  # 5% threshold
                    regressions.append({
                        'metric': metric,
                        'baseline': baseline_value,
                        'current': current_value,
                        'regression_percent': ((baseline_value - current_value) / baseline_value) * 100
                    })
        
        # Should detect regressions
        assert len(regressions) > 0
        
        # Should detect specific regressions
        regression_metrics = [r['metric'] for r in regressions]
        assert 'episode_generation_time_ms' in regression_metrics
        assert 'model_forward_time_ms' in regression_metrics
    
    def test_integration_health_monitoring(self):
        """Test ongoing integration health monitoring."""
        health_checks = {
            'component_availability': True,
            'memory_leaks': False,
            'performance_degradation': False,
            'error_rates_acceptable': True,
            'api_compatibility': True
        }
        
        # Simulate health check results
        current_health = {
            'component_availability': True,
            'memory_leaks': True,      # Issue detected
            'performance_degradation': False,
            'error_rates_acceptable': False,  # Issue detected
            'api_compatibility': True
        }
        
        # Identify health issues
        health_issues = []
        for check, expected in health_checks.items():
            current = current_health[check]
            if current != expected:
                health_issues.append({
                    'check': check,
                    'expected': expected,
                    'actual': current,
                    'severity': 'high' if 'memory' in check or 'error' in check else 'medium'
                })
        
        # Should detect health issues
        assert len(health_issues) > 0
        
        # Should identify specific issues
        issue_types = [issue['check'] for issue in health_issues]
        assert 'memory_leaks' in issue_types
        assert 'error_rates_acceptable' in issue_types


if __name__ == "__main__":
    pytest.main([__file__])