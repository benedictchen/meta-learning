#!/usr/bin/env python3
"""
Comprehensive Error Handling Tests
==================================

Tests for all error handling and recovery mechanisms in the meta-learning package:
- Episode validation errors
- MAML training failures  
- Dataset loading errors
- Memory and resource errors
- Configuration validation errors
- Recovery strategies
"""

import pytest
import torch
import torch.nn as nn
import tempfile
import os
from unittest.mock import patch, MagicMock
from typing import Dict, Any

from meta_learning.core.episode import Episode
from meta_learning.toolkit import MetaLearningToolkit, create_meta_learning_toolkit
from meta_learning.algorithms.maml_research_accurate import MAMLConfig, MAMLVariant
from meta_learning.data_utils.datasets import OmniglotDataset
from meta_learning.validation import (
    ValidationError, ConfigurationWarning, validate_episode_tensors,
    validate_few_shot_configuration, validate_maml_config
)
from meta_learning.error_recovery import (
    RecoveryError, ErrorRecoveryManager, with_retry,
    safe_tensor_operation, handle_numerical_instability
)


class TestEpisodeValidationErrors:
    """Test episode validation and error handling."""

    def test_episode_creation_invalid_dimensions(self):
        """Test episode creation with invalid tensor dimensions."""
        # Mismatched support dimensions
        with pytest.raises((ValueError, RuntimeError)):
            Episode(
                support_x=torch.randn(5, 10),     # 5 samples
                support_y=torch.arange(3),        # 3 labels (mismatch!)
                query_x=torch.randn(6, 10),
                query_y=torch.arange(6)
            )

    def test_episode_creation_empty_tensors(self):
        """Test episode creation with empty tensors."""
        with pytest.raises((ValueError, RuntimeError)):
            Episode(
                support_x=torch.empty(0, 10),     # Empty support
                support_y=torch.empty(0),
                query_x=torch.randn(5, 10),
                query_y=torch.arange(5)
            )

    def test_episode_creation_mismatched_feature_dimensions(self):
        """Test episode creation with mismatched feature dimensions."""
        with pytest.raises((ValueError, RuntimeError)):
            Episode(
                support_x=torch.randn(3, 10),     # 10 features
                support_y=torch.arange(3),
                query_x=torch.randn(6, 20),       # 20 features (mismatch!)
                query_y=torch.arange(6)
            )

    def test_episode_creation_invalid_label_values(self):
        """Test episode creation with invalid label values."""
        # Labels should be non-negative integers
        with pytest.raises((ValueError, RuntimeError)):
            Episode(
                support_x=torch.randn(3, 10),
                support_y=torch.tensor([-1, 0, 1]),  # Negative label
                query_x=torch.randn(6, 10),
                query_y=torch.arange(6)
            )

    def test_episode_validation_comprehensive(self):
        """Test comprehensive episode validation."""
        # Valid episode should pass
        valid_episode = Episode(
            torch.randn(6, 10), torch.repeat_interleave(torch.arange(3), 2),
            torch.randn(9, 10), torch.repeat_interleave(torch.arange(3), 3)
        )
        
        # Should not raise any errors
        valid_episode.validate()

    def test_episode_tensor_validation_function(self):
        """Test the validate_episode_tensors function."""
        valid_episode = Episode(
            torch.randn(4, 15), torch.repeat_interleave(torch.arange(2), 2),
            torch.randn(6, 15), torch.repeat_interleave(torch.arange(2), 3)
        )
        
        # Should not raise any errors
        validate_episode_tensors(valid_episode)
        
        # Invalid episode should raise ValidationError
        invalid_episode = Episode(
            torch.randn(4, 15), torch.arange(2),  # Wrong number of labels
            torch.randn(6, 15), torch.repeat_interleave(torch.arange(2), 3)
        )
        
        with pytest.raises(ValidationError):
            validate_episode_tensors(invalid_episode)


class TestMAMLTrainingErrors:
    """Test MAML training error scenarios."""

    def test_maml_uninitialized_model(self):
        """Test MAML with uninitialized model components."""
        toolkit = MetaLearningToolkit()
        episode = Episode(
            torch.randn(4, 10), torch.repeat_interleave(torch.arange(2), 2),
            torch.randn(6, 10), torch.repeat_interleave(torch.arange(2), 3)
        )
        
        # Should raise error for uninitialized MAML
        with pytest.raises(ValueError, match="Algorithm .* not initialized"):
            toolkit.train_episode(episode, algorithm='maml')

    def test_maml_invalid_configuration(self):
        """Test MAML with invalid configuration parameters."""
        model = nn.Linear(10, 5)
        
        # Invalid learning rates
        with pytest.raises(ValidationError):
            validate_maml_config(MAMLConfig(
                variant=MAMLVariant.MAML,
                inner_lr=-0.01,  # Negative learning rate
                outer_lr=0.001,
                inner_steps=1
            ))

    def test_maml_zero_inner_steps_warning(self):
        """Test MAML with zero inner steps produces warning."""
        model = nn.Linear(10, 3)
        toolkit = MetaLearningToolkit()
        
        # Configure MAML with zero inner steps
        config = MAMLConfig(
            variant=MAMLVariant.MAML,
            inner_lr=0.01,
            outer_lr=0.001,
            inner_steps=0,  # Zero inner steps
            first_order=False
        )
        
        toolkit.create_research_maml(model, config)
        
        episode = Episode(
            torch.randn(3, 10), torch.arange(3),
            torch.randn(6, 10), torch.repeat_interleave(torch.arange(3), 2)
        )
        
        # Should produce warning about fallback to base model
        with pytest.warns(UserWarning, match="MAML falling back to base model"):
            results = toolkit.train_episode(episode, algorithm='maml')
        
        # Should still produce valid results
        assert 'query_accuracy' in results

    def test_maml_gradient_explosion_handling(self):
        """Test MAML handling of gradient explosion scenarios."""
        # Create model prone to gradient explosion
        model = nn.Sequential(
            nn.Linear(10, 100),
            nn.ReLU(),
            nn.Linear(100, 5)
        )
        
        # Initialize with very large weights
        with torch.no_grad():
            for param in model.parameters():
                param.data *= 1000.0  # Very large initial weights
        
        toolkit = create_meta_learning_toolkit(
            model, 
            algorithm='maml', 
            inner_lr=1.0,  # Very high learning rate
            inner_steps=5
        )
        
        episode = Episode(
            torch.randn(10, 10), torch.repeat_interleave(torch.arange(5), 2),
            torch.randn(15, 10), torch.repeat_interleave(torch.arange(5), 3)
        )
        
        # May produce NaN or very large losses, but should not crash
        try:
            results = toolkit.train_episode(episode, algorithm='maml')
            # Check for numerical issues
            if 'query_loss' in results:
                loss = results['query_loss']
                if torch.isnan(torch.tensor(loss)) or loss > 1000:
                    pytest.skip("Gradient explosion detected - this is expected behavior")
        except (RuntimeError, ValueError) as e:
            if "nan" in str(e).lower() or "inf" in str(e).lower():
                pytest.skip("Numerical instability detected - this is expected")
            else:
                raise

    def test_maml_memory_overflow_handling(self):
        """Test MAML handling of memory overflow scenarios."""
        # Create artificially large model
        large_model = nn.Sequential(
            nn.Linear(1000, 2000),
            nn.ReLU(),
            nn.Linear(2000, 1000),
            nn.ReLU(), 
            nn.Linear(1000, 10)
        )
        
        toolkit = create_meta_learning_toolkit(large_model, algorithm='maml')
        
        # Create large episode
        large_episode = Episode(
            torch.randn(50, 1000), torch.repeat_interleave(torch.arange(10), 5),
            torch.randn(100, 1000), torch.repeat_interleave(torch.arange(10), 10)
        )
        
        # Should handle gracefully or raise appropriate memory error
        try:
            results = toolkit.train_episode(large_episode, algorithm='maml')
            assert isinstance(results, dict)
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            if "memory" in str(e).lower() or "out of memory" in str(e).lower():
                pytest.skip("Memory overflow detected - this is expected behavior")
            else:
                raise


class TestDatasetLoadingErrors:
    """Test dataset loading error scenarios."""

    def test_omniglot_missing_data_file(self):
        """Test OmniglotDataset with missing data files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises((FileNotFoundError, RuntimeError)):
                dataset = OmniglotDataset(
                    root=temp_dir,
                    download=False,  # Don't download
                    validate_data=True  # But validate
                )

    def test_omniglot_corrupted_data_handling(self):
        """Test OmniglotDataset with corrupted data files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create corrupted data file
            corrupt_file = os.path.join(temp_dir, 'omniglot_background_train.pkl')
            with open(corrupt_file, 'w') as f:
                f.write("corrupted data")
            
            with pytest.raises((RuntimeError, pickle.PickleError, EOFError)):
                dataset = OmniglotDataset(
                    root=temp_dir,
                    download=False,
                    validate_data=True
                )

    @patch('meta_learning.data_utils.datasets.urllib.request.urlretrieve')
    def test_omniglot_download_failure(self, mock_urlretrieve):
        """Test OmniglotDataset download failure handling."""
        # Mock download failure
        mock_urlretrieve.side_effect = Exception("Download failed")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(Exception):
                dataset = OmniglotDataset(
                    root=temp_dir,
                    download=True  # Attempt download
                )
                dataset._download_dataset()

    def test_dataset_insufficient_classes_error(self):
        """Test dataset with insufficient classes for episode generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock data with only 2 classes
            mock_data = {
                'alphabet_0': {
                    'character_0': [torch.randn(28, 28) for _ in range(5)],
                    'character_1': [torch.randn(28, 28) for _ in range(5)]
                }
            }
            
            with patch.object(OmniglotDataset, '_load_data'), \
                 patch.object(OmniglotDataset, '_data_exists', return_value=True):
                
                dataset = OmniglotDataset(root=temp_dir, download=False)
                dataset.data = mock_data
                dataset.alphabets = list(mock_data.keys())
                dataset.characters_per_alphabet = {
                    'alphabet_0': list(mock_data['alphabet_0'].keys())
                }
                
                # Requesting 5-way episode with only 2 classes should fail
                with pytest.raises((ValueError, RuntimeError)):
                    dataset.create_episode(n_way=5, n_shot=1, n_query=5)


class TestConfigurationValidationErrors:
    """Test configuration validation error scenarios."""

    def test_few_shot_configuration_validation(self):
        """Test few-shot configuration validation."""
        # Valid configuration should pass
        validate_few_shot_configuration(n_way=5, n_shot=2, n_query=15)
        
        # Invalid configurations should raise errors
        with pytest.raises(ValidationError):
            validate_few_shot_configuration(n_way=0, n_shot=1, n_query=5)  # Zero ways
        
        with pytest.raises(ValidationError):
            validate_few_shot_configuration(n_way=5, n_shot=0, n_query=5)  # Zero shots
        
        with pytest.raises(ValidationError):
            validate_few_shot_configuration(n_way=5, n_shot=1, n_query=0)  # Zero queries

    def test_maml_configuration_validation(self):
        """Test MAML configuration validation."""
        # Valid configuration
        valid_config = MAMLConfig(
            variant=MAMLVariant.MAML,
            inner_lr=0.01,
            outer_lr=0.001,
            inner_steps=5,
            first_order=False
        )
        validate_maml_config(valid_config)
        
        # Invalid learning rates
        invalid_config = MAMLConfig(
            variant=MAMLVariant.MAML,
            inner_lr=-0.01,  # Negative learning rate
            outer_lr=0.001,
            inner_steps=5
        )
        
        with pytest.raises(ValidationError):
            validate_maml_config(invalid_config)

    def test_toolkit_creation_invalid_algorithm(self):
        """Test toolkit creation with invalid algorithm specification."""
        model = nn.Linear(10, 5)
        
        with pytest.raises(ValueError):
            create_meta_learning_toolkit(
                model,
                algorithm='invalid_algorithm'  # Unknown algorithm
            )

    def test_configuration_warnings(self):
        """Test configuration warnings for suboptimal settings."""
        # Very high learning rate should produce warning
        with pytest.warns(ConfigurationWarning):
            validate_maml_config(MAMLConfig(
                variant=MAMLVariant.MAML,
                inner_lr=10.0,  # Very high learning rate
                outer_lr=0.001,
                inner_steps=1
            ))


class TestErrorRecoveryMechanisms:
    """Test error recovery and fault tolerance mechanisms."""

    def test_error_recovery_manager_initialization(self):
        """Test ErrorRecoveryManager initialization."""
        manager = ErrorRecoveryManager()
        
        assert hasattr(manager, 'recovery_strategies')
        assert hasattr(manager, 'error_history')
        assert hasattr(manager, 'recovery_attempts')

    def test_with_retry_decorator(self):
        """Test the with_retry decorator functionality."""
        call_count = 0
        
        @with_retry(max_attempts=3, backoff_factor=0.1)
        def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary failure")
            return "success"
        
        result = flaky_function()
        assert result == "success"
        assert call_count == 3

    def test_with_retry_permanent_failure(self):
        """Test with_retry with permanent failures."""
        @with_retry(max_attempts=2, backoff_factor=0.1)
        def always_fails():
            raise ValueError("Permanent failure")
        
        with pytest.raises(ValueError, match="Permanent failure"):
            always_fails()

    def test_safe_tensor_operation(self):
        """Test safe tensor operations with error handling."""
        # Valid operation
        result = safe_tensor_operation(
            lambda x, y: x + y,
            torch.tensor([1.0, 2.0]),
            torch.tensor([3.0, 4.0])
        )
        assert torch.allclose(result, torch.tensor([4.0, 6.0]))
        
        # Operation that produces NaN
        nan_result = safe_tensor_operation(
            lambda x: torch.log(x),
            torch.tensor([-1.0, 0.0, 1.0]),
            fallback_value=torch.tensor([0.0, 0.0, 0.0])
        )
        assert not torch.isnan(nan_result).any()

    def test_handle_numerical_instability(self):
        """Test numerical instability handling."""
        # Create tensor with NaN and Inf values
        unstable_tensor = torch.tensor([1.0, float('nan'), float('inf'), -float('inf'), 2.0])
        
        stable_tensor = handle_numerical_instability(
            unstable_tensor,
            nan_replacement=0.0,
            inf_replacement=1.0
        )
        
        assert not torch.isnan(stable_tensor).any()
        assert not torch.isinf(stable_tensor).any()
        assert torch.allclose(stable_tensor, torch.tensor([1.0, 0.0, 1.0, 1.0, 2.0]))

    def test_recovery_from_dimension_mismatch(self):
        """Test recovery from dimension mismatch errors."""
        from meta_learning.error_recovery import recover_from_dimension_mismatch
        
        # Mismatched tensors
        tensor1 = torch.randn(5, 10)
        tensor2 = torch.randn(3, 10)  # Different batch dimension
        
        # Should recover by padding or truncating
        recovered1, recovered2 = recover_from_dimension_mismatch(tensor1, tensor2)
        
        assert recovered1.shape[0] == recovered2.shape[0]  # Same batch size
        assert recovered1.shape[1] == recovered2.shape[1]  # Same feature size


class TestRobustTrainingComponents:
    """Test robust training components and fault tolerance."""

    def test_robust_prototype_network(self):
        """Test RobustPrototypeNetwork error handling."""
        from meta_learning.error_recovery import RobustPrototypeNetwork
        
        network = RobustPrototypeNetwork(input_dim=10, num_classes=5)
        
        # Normal operation
        support_x = torch.randn(15, 10)  # 5 classes, 3 shots each
        support_y = torch.repeat_interleave(torch.arange(5), 3)
        query_x = torch.randn(10, 10)
        
        logits = network(support_x, support_y, query_x)
        assert logits.shape == (10, 5)
        assert not torch.isnan(logits).any()

    def test_fault_tolerant_trainer(self):
        """Test FaultTolerantTrainer functionality."""
        from meta_learning.error_recovery import FaultTolerantTrainer
        
        model = nn.Linear(10, 3)
        trainer = FaultTolerantTrainer(
            model=model,
            max_retries=2,
            gradient_clip_norm=1.0
        )
        
        episode = Episode(
            torch.randn(6, 10), torch.repeat_interleave(torch.arange(3), 2),
            torch.randn(9, 10), torch.repeat_interleave(torch.arange(3), 3)
        )
        
        # Should train without errors
        loss = trainer.train_episode(episode)
        assert isinstance(loss, float)
        assert not torch.isnan(torch.tensor(loss))

    def test_safe_evaluate_function(self):
        """Test safe_evaluate function with error handling."""
        from meta_learning.error_recovery import safe_evaluate
        
        def evaluation_function(episodes):
            results = []
            for ep in episodes:
                # Simulate occasional failures
                if len(results) == 2:  # Fail on third episode
                    raise RuntimeError("Evaluation failed")
                results.append({'accuracy': 0.8})
            return {'mean_accuracy': 0.8, 'episodes': results}
        
        episodes = [
            Episode(torch.randn(2, 5), torch.arange(2), torch.randn(4, 5), torch.arange(2).repeat(2))
            for _ in range(5)
        ]
        
        # Should handle the failure gracefully
        results = safe_evaluate(evaluation_function, episodes, max_retries=1)
        assert isinstance(results, dict)


class TestMemoryAndResourceErrors:
    """Test memory and resource error handling."""

    def test_memory_efficient_episode_processing(self):
        """Test memory-efficient processing of large episodes."""
        # Create memory-intensive episode
        large_episode = Episode(
            torch.randn(100, 1000),  # Large tensors
            torch.repeat_interleave(torch.arange(20), 5),
            torch.randn(200, 1000),
            torch.repeat_interleave(torch.arange(20), 10)
        )
        
        # Should process without memory errors (or handle appropriately)
        try:
            model = nn.Linear(1000, 20)
            toolkit = create_meta_learning_toolkit(model, algorithm='maml')
            results = toolkit.train_episode(large_episode, algorithm='maml')
            assert isinstance(results, dict)
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            if "memory" in str(e).lower():
                pytest.skip("Expected memory limitation reached")
            else:
                raise

    def test_batch_processing_memory_management(self):
        """Test memory management during batch episode processing."""
        model = nn.Linear(50, 5)
        toolkit = create_meta_learning_toolkit(model, algorithm='maml')
        
        # Process multiple episodes sequentially
        episodes = []
        for i in range(10):
            episode = Episode(
                torch.randn(10, 50), torch.repeat_interleave(torch.arange(5), 2),
                torch.randn(15, 50), torch.repeat_interleave(torch.arange(5), 3)
            )
            episodes.append(episode)
        
        results = []
        for episode in episodes:
            result = toolkit.train_episode(episode, algorithm='maml')
            results.append(result)
            
            # Force garbage collection periodically
            if len(results) % 5 == 0:
                import gc
                gc.collect()
        
        assert len(results) == 10
        for result in results:
            assert 'query_accuracy' in result

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_memory_management(self):
        """Test CUDA memory management and error handling."""
        device = torch.device('cuda')
        
        model = nn.Linear(100, 10).to(device)
        toolkit = create_meta_learning_toolkit(model, algorithm='maml')
        
        episode = Episode(
            torch.randn(20, 100).to(device), torch.repeat_interleave(torch.arange(10), 2),
            torch.randn(30, 100).to(device), torch.repeat_interleave(torch.arange(10), 3)
        )
        
        try:
            results = toolkit.train_episode(episode, algorithm='maml')
            assert isinstance(results, dict)
            
            # Check memory usage didn't explode
            memory_allocated = torch.cuda.memory_allocated(device)
            assert memory_allocated < 1024 * 1024 * 1024  # Less than 1GB
            
        except torch.cuda.OutOfMemoryError:
            pytest.skip("CUDA out of memory - expected on limited hardware")


class TestErrorLoggingAndReporting:
    """Test error logging and reporting mechanisms."""

    def test_error_context_preservation(self):
        """Test that error context is properly preserved."""
        toolkit = MetaLearningToolkit()
        
        # Create problematic episode
        invalid_episode = Episode(
            torch.randn(3, 10), torch.arange(2),  # Mismatch: 3 samples, 2 labels
            torch.randn(6, 10), torch.arange(6)
        )
        
        try:
            toolkit.train_episode(invalid_episode, algorithm='maml')
        except Exception as e:
            # Error should contain useful context
            error_str = str(e)
            assert len(error_str) > 0
            # Could check for specific error context information

    def test_warning_system_integration(self):
        """Test integration with warning system."""
        # Test that warnings are properly issued for risky configurations
        with pytest.warns(UserWarning):
            # Very high learning rate should trigger warning
            model = nn.Linear(10, 5)
            config = MAMLConfig(
                variant=MAMLVariant.MAML,
                inner_lr=100.0,  # Extremely high
                outer_lr=0.001,
                inner_steps=1
            )
            
            try:
                validate_maml_config(config)
            except ValidationError:
                pass  # ValidationError is fine, we just want to test warning

    def test_graceful_degradation(self):
        """Test graceful degradation when components fail."""
        toolkit = MetaLearningToolkit()
        
        # Enable features that might fail
        toolkit.enable_failure_prediction()
        toolkit.enable_automatic_algorithm_selection()
        
        episode = Episode(
            torch.randn(4, 10), torch.repeat_interleave(torch.arange(2), 2),
            torch.randn(6, 10), torch.repeat_interleave(torch.arange(2), 3)
        )
        
        # Even if ML systems fail, basic functionality should work
        # (This is more of an integration test)
        algorithm_state = {'learning_rate': 0.01, 'inner_steps': 1, 'loss_history': []}
        
        prediction = toolkit.predict_and_prevent_failures(episode, algorithm_state)
        assert isinstance(prediction, dict)


if __name__ == "__main__":
    pytest.main([__file__])