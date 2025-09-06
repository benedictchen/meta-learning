"""
Integration tests for meta-learning modules.

Tests interactions between validation, warnings, error recovery, and evaluation modules.
"""

import pytest
import torch
import warnings
from unittest.mock import Mock, patch

from meta_learning.validation import (
    ValidationError, ConfigurationWarning, validate_episode_tensors,
    validate_complete_config, ValidationContext
)
from meta_learning.warnings_system import (
    MetaLearningWarnings, get_warning_system, warn_if_suboptimal_config
)
from meta_learning.error_recovery import (
    ErrorRecoveryManager, RobustPrototypeNetwork, safe_evaluate,
    create_robust_episode, handle_numerical_instability
)
from meta_learning.evaluation.metrics import (
    EvaluationMetrics, Accuracy, CalibrationCalculator
)


class TestValidationWarningsIntegration:
    """Test integration between validation and warnings systems."""
    
    def test_validation_triggers_warnings(self):
        """Test that validation functions trigger appropriate warnings."""
        warning_system = MetaLearningWarnings(enabled=True)
        
        # Test configuration that should trigger multiple warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # This should trigger warnings through validation
            config = {
                "n_way": 25,  # High n_way warning
                "k_shot": 15,  # High k_shot warning
                "n_query": 3,  # Small query set warning
                "distance": "cosine",
                "tau": 0.05,  # Low temperature warning
            }
            validate_complete_config(config)
            
            # Should have captured multiple warnings
            assert len(w) >= 2
            warning_messages = [str(warning.message) for warning in w]
            assert any("high" in msg for msg in warning_messages)
    
    def test_validation_context_controls_warnings(self):
        """Test that ValidationContext properly controls warning emission."""
        config = {
            "n_way": 25,  # Would normally trigger warning
            "k_shot": 15,  # Would normally trigger warning
            "distance": "cosine",
            "tau": 0.05,  # Would normally trigger warning
        }
        
        # Test with warnings disabled
        with ValidationContext(warnings_enabled=False):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                validate_complete_config(config)
                # Should have no warnings when disabled
                config_warnings = [warning for warning in w if issubclass(warning.category, ConfigurationWarning)]
                assert len(config_warnings) == 0
        
        # Test with warnings enabled
        with ValidationContext(warnings_enabled=True):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                validate_complete_config(config)
                # Should have warnings when enabled
                config_warnings = [warning for warning in w if issubclass(warning.category, ConfigurationWarning)]
                assert len(config_warnings) > 0
    
    def test_warning_deduplication_across_systems(self):
        """Test that warnings are not duplicated across systems."""
        # Reset global warning system
        warning_system = get_warning_system()
        warning_system.reset()
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # Call the same configuration multiple times
            config = {"n_way": 25, "k_shot": 2}
            for _ in range(3):
                warn_if_suboptimal_config(**config)
            
            # Should only get one warning due to deduplication
            config_warnings = [warning for warning in w if issubclass(warning.category, ConfigurationWarning)]
            assert len(config_warnings) == 1


class TestValidationErrorRecoveryIntegration:
    """Test integration between validation and error recovery systems."""
    
    def test_validation_with_error_recovery(self):
        """Test validation within error recovery context."""
        recovery_manager = ErrorRecoveryManager(enable_logging=False)
        
        # Create episode with validation and recovery
        episode = create_robust_episode(
            n_way=5, k_shot=2, n_query=10, feature_dim=64
        )
        
        # Validate the recovered episode
        validate_episode_tensors(
            episode["support_x"], episode["support_y"],
            episode["query_x"], episode["query_y"]
        )
        
        # Should pass validation without errors
        assert episode["support_x"].shape[0] == 10  # 5 way * 2 shot
        assert episode["query_x"].shape[0] == 50   # 5 way * 10 query
    
    def test_validation_catches_recovery_failures(self):
        """Test validation catches issues in recovered data."""
        # Create a scenario where recovery produces invalid data
        def create_bad_episode(*args, **kwargs):
            return {
                "support_x": torch.randn(10, 64),
                "support_y": torch.tensor([0, 1, 2, 3, 4, 0, 1, 2, 3, 4]),  # Valid
                "query_x": torch.randn(15, 32),  # Wrong feature dim!
                "query_y": torch.tensor([0, 1, 2, 3, 4] * 3)
            }
        
        bad_episode = create_bad_episode()
        
        with pytest.raises(ValidationError, match="feature dimensions don't match"):
            validate_episode_tensors(
                bad_episode["support_x"], bad_episode["support_y"],
                bad_episode["query_x"], bad_episode["query_y"]
            )
    
    def test_robust_network_with_validation(self):
        """Test robust network with input validation."""
        mock_network = Mock()
        mock_network.return_value = torch.randn(6, 3)  # Valid output
        
        recovery_manager = ErrorRecoveryManager(enable_logging=False)
        robust_network = RobustPrototypeNetwork(mock_network, recovery_manager)
        
        # Test with valid inputs
        support_x = torch.randn(6, 64)
        support_y = torch.tensor([0, 0, 1, 1, 2, 2])
        query_x = torch.randn(6, 64)
        
        # Validate inputs before passing to robust network
        validate_episode_tensors(support_x, support_y, query_x, torch.tensor([0, 0, 1, 1, 2, 2]))
        
        result = robust_network.forward_with_recovery(support_x, support_y, query_x)
        assert result.shape == (6, 3)
    
    def test_numerical_instability_handling_with_validation(self):
        """Test numerical instability handling followed by validation."""
        # Create tensor with numerical issues
        bad_tensor = torch.tensor([[1.0, float('nan')], [float('inf'), 2.0]])
        
        # Handle instability
        cleaned_tensor = handle_numerical_instability(bad_tensor)
        
        # Validate the cleaned tensor doesn't have issues
        assert not torch.isnan(cleaned_tensor).any()
        assert not torch.isinf(cleaned_tensor).any()
        
        # Should be usable in episode validation
        support_x = cleaned_tensor.repeat(5, 1)
        support_y = torch.arange(5)
        query_x = cleaned_tensor.repeat(3, 1)
        query_y = torch.arange(3)
        
        # Should pass validation now
        validate_episode_tensors(support_x, support_y, query_x, query_y)


class TestWarningsErrorRecoveryIntegration:
    """Test integration between warnings and error recovery systems."""
    
    def test_error_recovery_triggers_warnings(self):
        """Test that error recovery operations trigger appropriate warnings."""
        recovery_manager = ErrorRecoveryManager(enable_logging=True)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # Create scenario that triggers numerical instability warnings
            bad_tensor = torch.tensor([[float('nan'), 1.0], [2.0, float('inf')]])
            cleaned_tensor = handle_numerical_instability(bad_tensor)
            
            # Should trigger ConfigurationWarning about numerical issues
            config_warnings = [warning for warning in w if issubclass(warning.category, ConfigurationWarning)]
            assert len(config_warnings) >= 1
            assert any("NaN" in str(w.message) or "Infinite" in str(w.message) for w in config_warnings)
    
    def test_recovery_manager_warning_integration(self):
        """Test recovery manager integrates with warning system."""
        recovery_manager = ErrorRecoveryManager(enable_logging=True)
        
        # Record recovery attempts
        recovery_manager.record_recovery("test_recovery", True)
        recovery_manager.record_recovery("test_recovery", False)
        recovery_manager.record_recovery("another_recovery", True)
        
        stats = recovery_manager.recovery_stats
        assert stats["total_recoveries"] == 3
        assert stats["successful_recoveries"] == 2
        assert stats["failed_recoveries"] == 1
        assert "test_recovery" in stats["recovery_types"]
        assert "another_recovery" in stats["recovery_types"]
    
    def test_safe_evaluate_with_warnings(self):
        """Test safe evaluation triggers warnings for problematic episodes."""
        mock_model = Mock()
        mock_model.side_effect = [
            torch.randn(5, 3),  # Success
            RuntimeError("Test error"),  # Failure - should trigger warning
            torch.randn(5, 3),  # Success after recovery
        ]
        
        episodes = [
            {
                "support_x": torch.randn(6, 64),
                "support_y": torch.tensor([0, 0, 1, 1, 2, 2]),
                "query_x": torch.randn(5, 64),
                "query_y": torch.tensor([0, 1, 2, 0, 1])
            }
        ] * 3
        
        recovery_manager = ErrorRecoveryManager(enable_logging=True)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            results = safe_evaluate(mock_model, episodes, recovery_manager)
            
            # Should have results despite some failures
            assert "accuracy" in results
            assert results["failed_episodes"] >= 1


class TestFullPipelineIntegration:
    """Test complete pipeline integration across all modules."""
    
    def test_end_to_end_pipeline(self):
        """Test complete end-to-end pipeline with all modules."""
        # Step 1: Configuration validation with warnings
        config = {
            "n_way": 5,
            "k_shot": 2, 
            "n_query": 15,
            "distance": "sqeuclidean",
            "tau": 1.0,
            "inner_lr": 0.01,
            "inner_steps": 3,
            "outer_lr": 0.001
        }
        
        # Validate configuration (should pass without warnings)
        validate_complete_config(config)
        
        # Step 2: Create robust episode with error recovery
        episode = create_robust_episode(
            n_way=config["n_way"],
            k_shot=config["k_shot"], 
            n_query=config["n_query"],
            feature_dim=64
        )
        
        # Step 3: Validate episode data
        validate_episode_tensors(
            episode["support_x"], episode["support_y"],
            episode["query_x"], episode["query_y"]
        )
        
        # Step 4: Create mock model and robust wrapper
        mock_model = Mock()
        mock_model.return_value = torch.randn(
            config["n_way"] * config["n_query"], 
            config["n_way"]
        )
        
        recovery_manager = ErrorRecoveryManager(enable_logging=False)
        robust_model = RobustPrototypeNetwork(mock_model, recovery_manager)
        
        # Step 5: Run inference with error recovery
        predictions = robust_model.forward_with_recovery(
            episode["support_x"], episode["support_y"], episode["query_x"]
        )
        
        # Step 6: Evaluate with metrics
        accuracy_calc = Accuracy()
        accuracy_result = accuracy_calc.compute(predictions, episode["query_y"])
        
        metrics = EvaluationMetrics(
            accuracy=accuracy_result.mean,
            accuracy_ci=accuracy_result.confidence_interval
        )
        
        # Verify end-to-end pipeline works
        assert isinstance(metrics.accuracy, float)
        assert 0.0 <= metrics.accuracy <= 1.0
        assert len(metrics.accuracy_ci) == 2
    
    def test_pipeline_with_failures_and_recovery(self):
        """Test pipeline gracefully handles failures with recovery."""
        config = {
            "n_way": 3,
            "k_shot": 1,
            "n_query": 5,
            "distance": "cosine", 
            "tau": 0.5
        }
        
        # Step 1: Create episode (might trigger fallback)
        with patch('meta_learning.error_recovery.torch.randn') as mock_randn:
            # Mock randn to occasionally return problematic values
            def problematic_randn(*args, **kwargs):
                tensor = torch.randn(*args, **kwargs)
                if args[0] > 10:  # For larger tensors, introduce problems
                    tensor[0, 0] = float('nan')
                return tensor
            
            mock_randn.side_effect = problematic_randn
            
            # Should handle problematic generation gracefully
            episode = create_robust_episode(
                n_way=config["n_way"],
                k_shot=config["k_shot"],
                n_query=config["n_query"], 
                feature_dim=32
            )
        
        # Validate recovered episode
        validate_episode_tensors(
            episode["support_x"], episode["support_y"],
            episode["query_x"], episode["query_y"]
        )
        
        # Should have valid episode despite initial problems
        assert episode["support_x"].shape[0] == config["n_way"] * config["k_shot"]
        assert episode["query_x"].shape[0] == config["n_way"] * config["n_query"]
    
    def test_pipeline_performance_monitoring(self):
        """Test pipeline with performance monitoring."""
        import time
        
        start_time = time.time()
        
        # Create multiple episodes and process them
        episodes = []
        for i in range(5):
            episode = create_robust_episode(
                n_way=3, k_shot=2, n_query=5, feature_dim=64
            )
            validate_episode_tensors(
                episode["support_x"], episode["support_y"],
                episode["query_x"], episode["query_y"]
            )
            episodes.append(episode)
        
        # Mock model for evaluation
        mock_model = Mock()
        mock_model.return_value = torch.randn(15, 3)  # 3 way * 5 query
        
        # Evaluate all episodes
        results = safe_evaluate(mock_model, episodes)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Verify performance is reasonable (should be fast for small episodes)
        assert processing_time < 5.0  # Less than 5 seconds
        assert results["accuracy"] >= 0.0
        assert results["total_episodes"] == 5


class TestErrorPropagationIntegration:
    """Test how errors propagate through integrated systems."""
    
    def test_validation_error_in_pipeline(self):
        """Test validation errors are properly caught in pipeline."""
        # Create invalid episode
        episode = {
            "support_x": torch.randn(6, 64),
            "support_y": torch.tensor([0, 1, 2, 0, 1, 2]), 
            "query_x": torch.randn(9, 32),  # Wrong feature dim
            "query_y": torch.tensor([0, 1, 2] * 3)
        }
        
        with pytest.raises(ValidationError):
            validate_episode_tensors(
                episode["support_x"], episode["support_y"],
                episode["query_x"], episode["query_y"]
            )
    
    def test_recovery_handles_validation_failures(self):
        """Test recovery system handles validation failures gracefully."""
        recovery_manager = ErrorRecoveryManager(enable_logging=False)
        
        # Create scenario where validation would fail but recovery handles it
        def mock_create_bad_episode(*args, **kwargs):
            # This would normally fail validation
            return {
                "support_x": torch.randn(6, 64),
                "support_y": torch.tensor([0, 1, 2, 0, 1, 2]),
                "query_x": torch.empty(0, 64),  # Empty query set
                "query_y": torch.empty(0, dtype=torch.long)
            }
        
        # The recovery system should detect and fix the empty query issue
        with patch('meta_learning.error_recovery.torch.randn') as mock_randn:
            mock_randn.return_value = torch.randn(15, 64)  # Valid fallback
            
            # Should create valid episode despite initial problems
            episode = create_robust_episode(n_way=3, k_shot=2, n_query=5, feature_dim=64)
            
            # Should pass validation after recovery
            validate_episode_tensors(
                episode["support_x"], episode["support_y"],
                episode["query_x"], episode["query_y"]
            )
    
    def test_warning_system_error_handling(self):
        """Test warning system handles errors in warning generation."""
        warning_system = MetaLearningWarnings(enabled=True)
        
        # Test with invalid input that might cause warning system errors
        try:
            # This should not crash even with problematic input
            warnings_list = warning_system.warn_if_suboptimal_few_shot(
                n_way=None,  # Invalid input
                k_shot=2
            )
            # If it doesn't crash, it should return empty list
            assert isinstance(warnings_list, list)
        except Exception as e:
            # Should handle errors gracefully
            assert "n_way" in str(e) or "NoneType" in str(e)