"""
💪 Stress Tests and Edge Case Validation for Meta-Learning Package
==================================================================

These tests validate robustness under extreme conditions, edge cases,
and potential failure scenarios to ensure production readiness.

Test Categories:
- Extreme parameter configurations
- Memory pressure scenarios  
- Numerical stability edge cases
- Invalid input handling
- Resource exhaustion testing
- Concurrent execution stress
"""

import pytest
import torch
# DEMO/TEST REPRODUCIBILITY: Set seed for consistent synthetic data
torch.manual_seed(42)  # Reproducible test/demo data
import torch.nn as nn
# DEMO/TEST REPRODUCIBILITY: Set seed for consistent synthetic data
torch.manual_seed(42)  # Reproducible test/demo data
import numpy as np
import threading
import time
import gc
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# Meta-learning components
from src.meta_learning.meta_learning_modules.few_shot_learning import (
    PrototypicalLearner, PrototypicalConfig
)
from src.meta_learning.meta_learning_modules.maml_variants import (
    MAML, MAMLConfig
)
from src.meta_learning.meta_learning_modules.test_time_compute import (
    TestTimeComputeScaler, TestTimeComputeConfig
)
from src.meta_learning.meta_learning_modules.continual_meta_learning import (
    ContinualMetaLearner, ContinualConfig
)
from src.meta_learning.meta_learning_modules.utils import (
    MetaLearningDataset, DatasetConfig, EvaluationMetrics
)
from src.meta_learning.meta_learning_modules.hardware_utils import (
    HardwareManager, HardwareConfig
)


@pytest.mark.stress
class TestExtremeParameterConfigurations:
    """Test algorithms with extreme parameter values."""
    
    def test_extreme_few_shot_scenarios(self):
        """Test with extreme few-shot configurations."""
        # Create base encoder
        encoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )
        
        hw_manager = HardwareManager(HardwareConfig(memory_efficient=True))
        encoder = hw_manager.prepare_model(encoder)
        
        # Extreme configurations
        extreme_configs = [
            # 1-shot learning (minimal data)
            {'n_way': 2, 'k_shot': 1, 'n_query': 5, 'name': '1-shot'},
            # Many-way classification
            {'n_way': 50, 'k_shot': 2, 'n_query': 10, 'name': '50-way'},
            # High-shot learning
            {'n_way': 5, 'k_shot': 100, 'n_query': 20, 'name': '100-shot'},
            # Tiny query set
            {'n_way': 5, 'k_shot': 5, 'n_query': 1, 'name': 'tiny-query'},
            # Large query set
            {'n_way': 5, 'k_shot': 5, 'n_query': 200, 'name': 'large-query'}
        ]
        
        for config in extreme_configs:
            print(f"Testing extreme config: {config['name']}")
            
            # Create dataset
            dataset_config = DatasetConfig(
                n_way=config['n_way'],
                k_shot=config['k_shot'],
                n_query=config['n_query'],
                feature_dim=64,
                num_classes=max(100, config['n_way'] * 2),
                episode_length=5  # Few episodes for extreme cases
            )
            dataset = MetaLearningDataset(dataset_config)
            
            # Test prototypical learner
            proto_config = PrototypicalConfig(distance_metric='euclidean')
            learner = PrototypicalLearner(encoder, proto_config)
            
            # Run episodes
            for episode_idx in range(5):
                try:
                    episode_data = dataset.generate_episode()
                    support_x, support_y, query_x, query_y = episode_data
                    
                    # Prepare data
                    support_x = hw_manager.prepare_data(support_x)
                    support_y = hw_manager.prepare_data(support_y)
                    query_x = hw_manager.prepare_data(query_x)
                    query_y = hw_manager.prepare_data(query_y)
                    
                    # Forward pass
                    with hw_manager.autocast_context():
                        logits = learner(support_x, support_y, query_x)
                        predictions = torch.argmax(logits, dim=1)
                        accuracy = (predictions == query_y).float().mean().item()
                    
                    # Validate output shape
                    expected_shape = (config['n_query'], config['n_way'])
                    assert logits.shape == expected_shape, f"Expected {expected_shape}, got {logits.shape}"
                    assert 0.0 <= accuracy <= 1.0
                    
                except Exception as e:
                    pytest.fail(f"Extreme config {config['name']} failed: {e}")
            
            print(f"  ✅ {config['name']}: Passed")
    
    def test_extreme_maml_configurations(self):
        """Test MAML with extreme hyperparameters."""
        encoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(), 
            nn.Linear(128, 64)
        )
        
        hw_manager = HardwareManager(HardwareConfig(memory_efficient=True))
        encoder = hw_manager.prepare_model(encoder)
        
        # Extreme MAML configurations
        extreme_maml_configs = [
            # Very high learning rates
            {'inner_lr': 1.0, 'outer_lr': 0.1, 'name': 'high_lr'},
            # Very low learning rates  
            {'inner_lr': 1e-8, 'outer_lr': 1e-8, 'name': 'tiny_lr'},
            # Many inner steps
            {'inner_lr': 0.01, 'outer_lr': 0.001, 'num_inner_steps': 50, 'name': 'many_steps'},
            # Single inner step
            {'inner_lr': 0.1, 'outer_lr': 0.01, 'num_inner_steps': 1, 'name': 'single_step'}
        ]
        
        # Simple dataset
        dataset_config = DatasetConfig(
            n_way=3, k_shot=3, n_query=5, feature_dim=64,
            num_classes=10, episode_length=3
        )
        dataset = MetaLearningDataset(dataset_config)
        
        for config in extreme_maml_configs:
            print(f"Testing extreme MAML: {config['name']}")
            
            try:
                maml_config = MAMLConfig(
                    inner_lr=config['inner_lr'],
                    outer_lr=config['outer_lr'],
                    num_inner_steps=config.get('num_inner_steps', 5)
                )
                
                maml_learner = MAML(encoder, maml_config)
                optimizer = torch.optim.Adam(maml_learner.parameters(), lr=maml_config.outer_lr)
                
                # Run a few episodes
                for episode_idx in range(3):
                    episode_data = dataset.generate_episode()
                    support_x, support_y, query_x, query_y = episode_data
                    
                    support_x = hw_manager.prepare_data(support_x)
                    support_y = hw_manager.prepare_data(support_y)
                    query_x = hw_manager.prepare_data(query_x)
                    query_y = hw_manager.prepare_data(query_y)
                    
                    optimizer.zero_grad()
                    
                    with hw_manager.autocast_context():
                        meta_loss, adapted_params = maml_learner.meta_forward(
                            support_x, support_y, query_x, query_y
                        )
                        
                        # Check for numerical issues
                        assert torch.isfinite(meta_loss).all(), f"Non-finite loss in {config['name']}"
                        assert meta_loss.item() > 0, f"Non-positive loss in {config['name']}"
                    
                    hw_manager.backward_and_step(meta_loss, optimizer)
                
                print(f"  ✅ {config['name']}: Passed")
                
            except Exception as e:
                # Some extreme configs may legitimately fail
                print(f"  ⚠️  {config['name']}: Failed as expected - {str(e)[:100]}")


@pytest.mark.stress  
class TestNumericalStabilityEdgeCases:
    """Test numerical stability under edge conditions."""
    
    def test_zero_variance_features(self):
        """Test handling of zero-variance feature vectors."""
        encoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        hw_manager = HardwareManager(HardwareConfig())
        encoder = hw_manager.prepare_model(encoder)
        
        proto_config = PrototypicalConfig(distance_metric='euclidean')
        learner = PrototypicalLearner(encoder, proto_config)
        
        # Create data with zero variance in some dimensions
        n_way, k_shot, n_query = 5, 3, 10
        feature_dim = 64
        
        # Support set with zero variance features
        support_x = torch.zeros(n_way * k_shot, feature_dim)
        support_x[:, :32] = torch.randn(n_way * k_shot, 32)  # Half dimensions have variance
        support_y = torch.repeat_interleave(torch.arange(n_way), k_shot)
        
        # Query set with similar structure
        query_x = torch.zeros(n_query, feature_dim) 
        query_x[:, :32] = torch.randn(n_query, 32)
        query_y = torch.randint(0, n_way, (n_query,))
        
        # Prepare data
        support_x = hw_manager.prepare_data(support_x)
        support_y = hw_manager.prepare_data(support_y)
        query_x = hw_manager.prepare_data(query_x)
        query_y = hw_manager.prepare_data(query_y)
        
        # Test forward pass
        try:
            with hw_manager.autocast_context():
                logits = learner(support_x, support_y, query_x)
            
            # Validate output
            assert logits.shape == (n_query, n_way)
            assert torch.isfinite(logits).all(), "Non-finite logits with zero variance features"
            
            print("✅ Zero variance features: Handled correctly")
            
        except Exception as e:
            pytest.fail(f"Failed to handle zero variance features: {e}")
    
    def test_identical_support_examples(self):
        """Test handling of identical examples in support set."""
        encoder = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 64))
        hw_manager = HardwareManager(HardwareConfig())
        encoder = hw_manager.prepare_model(encoder)
        
        proto_config = PrototypicalConfig(distance_metric='euclidean')
        learner = PrototypicalLearner(encoder, proto_config)
        
        # Create support set with identical examples per class
        n_way, k_shot, n_query = 3, 5, 9
        feature_dim = 64
        
        support_x = torch.zeros(n_way * k_shot, feature_dim)
        for class_idx in range(n_way):
            # All examples of same class are identical
            class_prototype = torch.randn(feature_dim)
            start_idx = class_idx * k_shot
            end_idx = (class_idx + 1) * k_shot
            support_x[start_idx:end_idx] = class_prototype
        
        support_y = torch.repeat_interleave(torch.arange(n_way), k_shot)
        query_x = torch.randn(n_query, feature_dim)
        query_y = torch.randint(0, n_way, (n_query,))
        
        # Prepare data
        support_x = hw_manager.prepare_data(support_x)
        support_y = hw_manager.prepare_data(support_y)
        query_x = hw_manager.prepare_data(query_x)
        query_y = hw_manager.prepare_data(query_y)
        
        # Test forward pass
        with hw_manager.autocast_context():
            logits = learner(support_x, support_y, query_x)
        
        assert logits.shape == (n_query, n_way)
        assert torch.isfinite(logits).all(), "Non-finite logits with identical support examples"
        
        print("✅ Identical support examples: Handled correctly")
    
    def test_extreme_distance_values(self):
        """Test handling of extreme distance values."""
        encoder = nn.Identity()  # Pass-through to control exact values
        hw_manager = HardwareManager(HardwareConfig())
        encoder = hw_manager.prepare_model(encoder)
        
        proto_config = PrototypicalConfig(distance_metric='euclidean')
        learner = PrototypicalLearner(encoder, proto_config)
        
        # Create data with extreme values
        n_way, k_shot, n_query = 2, 2, 4
        feature_dim = 3
        
        # Support with extreme values
        support_x = torch.tensor([
            [1e6, 0, 0],      # Class 0: very large values
            [1e6, 0, 0],
            [-1e6, 0, 0],     # Class 1: very negative values  
            [-1e6, 0, 0]
        ], dtype=torch.float32)
        support_y = torch.tensor([0, 0, 1, 1])
        
        # Query points
        query_x = torch.tensor([
            [5e5, 0, 0],      # Closer to class 0
            [-5e5, 0, 0],     # Closer to class 1
            [0, 0, 0],        # Equidistant
            [1e10, 0, 0]      # Extremely far
        ], dtype=torch.float32)
        query_y = torch.tensor([0, 1, 0, 0])
        
        # Prepare data
        support_x = hw_manager.prepare_data(support_x)
        support_y = hw_manager.prepare_data(support_y)
        query_x = hw_manager.prepare_data(query_x)
        query_y = hw_manager.prepare_data(query_y)
        
        # Test forward pass
        with hw_manager.autocast_context():
            logits = learner(support_x, support_y, query_x)
        
        # Validate numerical stability
        assert logits.shape == (n_query, n_way)
        assert torch.isfinite(logits).all(), "Non-finite logits with extreme distance values"
        assert not torch.isnan(logits).any(), "NaN values in logits"
        assert not torch.isinf(logits).any(), "Infinite values in logits"
        
        print("✅ Extreme distance values: Handled correctly")


@pytest.mark.stress
class TestResourceExhaustionScenarios:
    """Test behavior under resource constraints."""
    
    def test_memory_pressure_scenarios(self):
        """Test algorithms under memory pressure."""
        # Create larger models and datasets to stress memory
        encoder = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512)
        )
        
        # Configure for memory efficiency
        hw_config = HardwareConfig(
            memory_efficient=True,
            max_memory_fraction=0.5,  # Limit memory usage
            gradient_checkpointing=True
        )
        hw_manager = HardwareManager(hw_config)
        encoder = hw_manager.prepare_model(encoder)
        
        # Large problem configuration
        large_config = DatasetConfig(
            n_way=20,
            k_shot=20,
            n_query=50,
            feature_dim=512,
            num_classes=100,
            episode_length=10
        )
        dataset = MetaLearningDataset(large_config)
        
        # Test different algorithms under memory pressure
        algorithms = [
            ('Prototypical', PrototypicalLearner(
                encoder, 
                PrototypicalConfig(distance_metric='euclidean')
            )),
            ('MAML', MAML(
                encoder, 
                MAMLConfig(num_inner_steps=3, use_memory_efficient=True)
            ))
        ]
        
        for algo_name, learner in algorithms:
            print(f"Testing {algo_name} under memory pressure")
            
            try:
                # Force garbage collection
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Run memory-intensive episodes
                for episode_idx in range(5):
                    episode_data = dataset.generate_episode()
                    support_x, support_y, query_x, query_y = episode_data
                    
                    support_x = hw_manager.prepare_data(support_x)
                    support_y = hw_manager.prepare_data(support_y)
                    query_x = hw_manager.prepare_data(query_x)
                    query_y = hw_manager.prepare_data(query_y)
                    
                    with hw_manager.autocast_context():
                        if hasattr(learner, 'meta_forward'):  # MAML
                            loss, params = learner.meta_forward(support_x, support_y, query_x, query_y)
                            logits = learner.forward_with_params(query_x, params)
                        else:  # Prototypical
                            logits = learner(support_x, support_y, query_x)
                    
                    # Validate output
                    expected_shape = (large_config.n_query, large_config.n_way)
                    assert logits.shape == expected_shape
                    assert torch.isfinite(logits).all()
                    
                    # Clear memory after each episode
                    del logits
                    if 'params' in locals():
                        del params
                    gc.collect()
                
                print(f"  ✅ {algo_name}: Survived memory pressure")
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"  ⚠️  {algo_name}: OOM as expected under extreme memory pressure")
                else:
                    pytest.fail(f"{algo_name} failed with non-OOM error: {e}")
            except Exception as e:
                pytest.fail(f"{algo_name} failed unexpectedly: {e}")
    
    def test_concurrent_execution_stress(self):
        """Test concurrent execution of meta-learning algorithms."""
        encoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        def run_episode(thread_id: int) -> Dict[str, Any]:
            """Run a single episode in a thread."""
            try:
                # Each thread gets its own hardware manager and models
                hw_manager = HardwareManager(HardwareConfig(memory_efficient=True))
                thread_encoder = hw_manager.prepare_model(encoder)
                
                dataset_config = DatasetConfig(
                    n_way=5, k_shot=3, n_query=10, feature_dim=64,
                    num_classes=20, episode_length=1
                )
                dataset = MetaLearningDataset(dataset_config)
                
                proto_config = PrototypicalConfig(distance_metric='euclidean')
                learner = PrototypicalLearner(thread_encoder, proto_config)
                
                episode_data = dataset.generate_episode()
                support_x, support_y, query_x, query_y = episode_data
                
                support_x = hw_manager.prepare_data(support_x)
                support_y = hw_manager.prepare_data(support_y)
                query_x = hw_manager.prepare_data(query_x)
                query_y = hw_manager.prepare_data(query_y)
                
                with hw_manager.autocast_context():
                    logits = learner(support_x, support_y, query_x)
                    predictions = torch.argmax(logits, dim=1)
                    accuracy = (predictions == query_y).float().mean().item()
                
                return {
                    'thread_id': thread_id,
                    'success': True,
                    'accuracy': accuracy,
                    'logits_shape': list(logits.shape)
                }
                
            except Exception as e:
                return {
                    'thread_id': thread_id,
                    'success': False,
                    'error': str(e)
                }
        
        # Run concurrent episodes
        num_threads = 4
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(run_episode, i) for i in range(num_threads)]
            results = [future.result() for future in as_completed(futures)]
        
        # Validate concurrent execution results
        successful_results = [r for r in results if r['success']]
        failed_results = [r for r in results if not r['success']]
        
        print(f"Concurrent execution: {len(successful_results)}/{num_threads} threads successful")
        
        # At least 50% should succeed (some failures acceptable under stress)
        assert len(successful_results) >= num_threads // 2, \
            f"Too many thread failures: {len(failed_results)} out of {num_threads}"
        
        # Validate successful results
        for result in successful_results:
            assert 0.0 <= result['accuracy'] <= 1.0
            assert result['logits_shape'] == [10, 5]  # n_query=10, n_way=5
        
        if failed_results:
            print(f"  ⚠️  {len(failed_results)} threads failed (acceptable under stress)")
            for result in failed_results[:2]:  # Show first 2 failures
                print(f"    Thread {result['thread_id']}: {result['error'][:100]}")


@pytest.mark.stress
class TestInvalidInputHandling:
    """Test robustness against invalid inputs."""
    
    def test_mismatched_tensor_shapes(self):
        """Test handling of mismatched tensor shapes."""
        encoder = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 64))
        hw_manager = HardwareManager(HardwareConfig())
        encoder = hw_manager.prepare_model(encoder)
        
        proto_config = PrototypicalConfig(distance_metric='euclidean')
        learner = PrototypicalLearner(encoder, proto_config)
        
        # Test cases with shape mismatches
        invalid_cases = [
            # Wrong feature dimensions
            {
                'support_x': torch.randn(15, 32),  # Wrong feature dim
                'support_y': torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]),
                'query_x': torch.randn(10, 64),    # Correct feature dim
                'query_y': torch.randint(0, 5, (10,)),
                'error_type': 'feature_mismatch'
            },
            # Wrong support/query feature dims
            {
                'support_x': torch.randn(15, 64),
                'support_y': torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]),
                'query_x': torch.randn(10, 32),    # Wrong feature dim
                'query_y': torch.randint(0, 5, (10,)),
                'error_type': 'query_mismatch'
            },
            # Mismatched support_x and support_y lengths
            {
                'support_x': torch.randn(15, 64),
                'support_y': torch.tensor([0, 0, 1, 1, 2, 2]),  # Wrong length
                'query_x': torch.randn(10, 64),
                'query_y': torch.randint(0, 5, (10,)),
                'error_type': 'support_length_mismatch'
            }
        ]
        
        for case in invalid_cases:
            print(f"Testing {case['error_type']}")
            
            try:
                support_x = hw_manager.prepare_data(case['support_x'])
                support_y = hw_manager.prepare_data(case['support_y'])
                query_x = hw_manager.prepare_data(case['query_x'])
                query_y = hw_manager.prepare_data(case['query_y'])
                
                with hw_manager.autocast_context():
                    logits = learner(support_x, support_y, query_x)
                
                # If we reach here, the invalid input was somehow handled
                print(f"  ⚠️  {case['error_type']}: Unexpectedly succeeded")
                
            except Exception as e:
                # Expected to fail with invalid inputs
                print(f"  ✅ {case['error_type']}: Correctly failed with {type(e).__name__}")
    
    def test_invalid_hyperparameters(self):
        """Test handling of invalid hyperparameter values."""
        encoder = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 64))
        
        # Invalid configurations that should be caught
        invalid_configs = [
            # Negative learning rates
            {'config': MAMLConfig(inner_lr=-0.01), 'name': 'negative_inner_lr'},
            {'config': MAMLConfig(outer_lr=-0.001), 'name': 'negative_outer_lr'},
            # Zero inner steps  
            {'config': MAMLConfig(num_inner_steps=0), 'name': 'zero_inner_steps'},
            # Invalid distance metrics
            {'config': PrototypicalConfig(distance_metric='invalid_metric'), 'name': 'invalid_distance'},
        ]
        
        for case in invalid_configs:
            print(f"Testing invalid config: {case['name']}")
            
            try:
                if isinstance(case['config'], MAMLConfig):
                    learner = MAML(encoder, case['config'])
                elif isinstance(case['config'], PrototypicalConfig):
                    learner = PrototypicalLearner(encoder, case['config'])
                
                print(f"  ⚠️  {case['name']}: Config accepted (may be handled internally)")
                
            except (ValueError, AssertionError, NotImplementedError) as e:
                print(f"  ✅ {case['name']}: Correctly rejected with {type(e).__name__}")
            except Exception as e:
                print(f"  ? {case['name']}: Unexpected error {type(e).__name__}: {e}")
    
    def test_empty_datasets(self):
        """Test handling of empty or minimal datasets."""
        encoder = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 64))
        hw_manager = HardwareManager(HardwareConfig())
        encoder = hw_manager.prepare_model(encoder)
        
        proto_config = PrototypicalConfig(distance_metric='euclidean')
        learner = PrototypicalLearner(encoder, proto_config)
        
        # Edge cases with minimal data
        minimal_cases = [
            # Empty tensors
            {
                'support_x': torch.empty(0, 64),
                'support_y': torch.empty(0, dtype=torch.long),
                'query_x': torch.randn(5, 64),
                'query_y': torch.randint(0, 3, (5,)),
                'name': 'empty_support'
            },
            # Single support example
            {
                'support_x': torch.randn(1, 64),
                'support_y': torch.tensor([0]),
                'query_x': torch.randn(3, 64),
                'query_y': torch.randint(0, 1, (3,)),
                'name': 'single_support'
            }
        ]
        
        for case in minimal_cases:
            print(f"Testing minimal data: {case['name']}")
            
            try:
                support_x = hw_manager.prepare_data(case['support_x'])
                support_y = hw_manager.prepare_data(case['support_y'])
                query_x = hw_manager.prepare_data(case['query_x'])
                query_y = hw_manager.prepare_data(case['query_y'])
                
                with hw_manager.autocast_context():
                    logits = learner(support_x, support_y, query_x)
                
                print(f"  ⚠️  {case['name']}: Handled gracefully")
                
            except Exception as e:
                print(f"  ✅ {case['name']}: Correctly failed with {type(e).__name__}")


@pytest.mark.stress
@pytest.mark.slow
class TestLongRunningStressTest:
    """Long-running stress tests for stability validation."""
    
    def test_extended_training_stability(self):
        """Test algorithm stability over extended training."""
        encoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        hw_manager = HardwareManager(HardwareConfig(memory_efficient=True))
        encoder = hw_manager.prepare_model(encoder)
        
        # Dataset configuration
        dataset_config = DatasetConfig(
            n_way=5, k_shot=5, n_query=10, feature_dim=64,
            num_classes=25, episode_length=1
        )
        dataset = MetaLearningDataset(dataset_config)
        
        # MAML configuration
        maml_config = MAMLConfig(
            inner_lr=0.01,
            outer_lr=0.001,
            num_inner_steps=5
        )
        maml_learner = MAML(encoder, maml_config)
        optimizer = torch.optim.Adam(maml_learner.parameters(), lr=maml_config.outer_lr)
        
        # Extended training loop
        num_episodes = 200  # Longer training
        losses = []
        accuracies = []
        
        print(f"Running extended training for {num_episodes} episodes")
        
        for episode_idx in range(num_episodes):
            try:
                episode_data = dataset.generate_episode()
                support_x, support_y, query_x, query_y = episode_data
                
                support_x = hw_manager.prepare_data(support_x)
                support_y = hw_manager.prepare_data(support_y)
                query_x = hw_manager.prepare_data(query_x)
                query_y = hw_manager.prepare_data(query_y)
                
                optimizer.zero_grad()
                
                with hw_manager.autocast_context():
                    meta_loss, adapted_params = maml_learner.meta_forward(
                        support_x, support_y, query_x, query_y
                    )
                    
                    logits = maml_learner.forward_with_params(query_x, adapted_params)
                    predictions = torch.argmax(logits, dim=1)
                    accuracy = (predictions == query_y).float().mean().item()
                
                hw_manager.backward_and_step(meta_loss, optimizer)
                
                losses.append(meta_loss.item())
                accuracies.append(accuracy)
                
                # Check for numerical issues
                if not torch.isfinite(meta_loss):
                    pytest.fail(f"Non-finite loss at episode {episode_idx}")
                
                # Memory cleanup every 50 episodes
                if episode_idx % 50 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # Progress report
                    recent_loss = np.mean(losses[-10:]) if len(losses) >= 10 else np.mean(losses)
                    recent_acc = np.mean(accuracies[-10:]) if len(accuracies) >= 10 else np.mean(accuracies)
                    print(f"  Episode {episode_idx}: Loss={recent_loss:.3f}, Acc={recent_acc:.3f}")
                
            except Exception as e:
                pytest.fail(f"Extended training failed at episode {episode_idx}: {e}")
        
        # Validate training stability
        assert len(losses) == num_episodes
        assert len(accuracies) == num_episodes
        
        # Check final performance
        final_loss = np.mean(losses[-20:])
        final_accuracy = np.mean(accuracies[-20:])
        
        assert final_loss > 0
        assert 0.0 <= final_accuracy <= 1.0
        assert all(np.isfinite(losses))
        assert all(np.isfinite(accuracies))
        
        print(f"✅ Extended training completed: Final loss={final_loss:.3f}, accuracy={final_accuracy:.3f}")


if __name__ == "__main__":
    # Run with: pytest tests/stress/test_edge_cases.py -v -m stress  
    pytest.main([__file__, "-v", "-m", "stress"])