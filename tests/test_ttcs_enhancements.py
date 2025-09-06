"""
Tests for TTCS (Test-Time Compute Scaling) error handling and validation enhancements.

Tests cover:
- Input validation with descriptive errors
- Error handling and graceful degradation
- TTCSWarningSystem functionality
- Compatibility checks and model validation
- TTCSProfiler performance monitoring
- Fallback mechanisms and recovery
"""

import pytest
import torch
import torch.nn as nn
import warnings
from unittest.mock import Mock, patch, MagicMock

from meta_learning.algos.ttcs import (
    TTCSWarningSystem, ttcs_with_fallback, ttcs_for_learn2learn_models,
    TTCSProfiler, ValidationError
)
from meta_learning.core.episode import Episode


class TestTTCSInputValidation:
    """Test TTCS input validation enhancements."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.valid_encoder = nn.Linear(64, 128)
        self.valid_support_x = torch.randn(25, 64)
        self.valid_support_y = torch.repeat_interleave(torch.arange(5), 5)
        self.valid_query_x = torch.randn(15, 64)
        self.valid_task_context = {"n_classes": 5}
    
    def test_encoder_validation(self):
        """Test encoder parameter validation."""
        from meta_learning.algorithms.ttc_scaler import TestTimeComputeScaler
        from meta_learning.algorithms.ttc_config import TestTimeComputeConfig
        
        config = TestTimeComputeConfig()
        
        # Valid encoder should work
        scaler = TestTimeComputeScaler(self.valid_encoder, config)
        assert scaler.encoder == self.valid_encoder
        
        # Invalid encoder types should raise ValidationError
        with pytest.raises(ValidationError, match="encoder must be a torch.nn.Module"):
            TestTimeComputeScaler("not_a_module", config)
        
        with pytest.raises(ValidationError, match="encoder must be a torch.nn.Module"):
            TestTimeComputeScaler(None, config)
        
        with pytest.raises(ValidationError, match="encoder must be a torch.nn.Module"):
            TestTimeComputeScaler(123, config)
    
    def test_support_set_validation(self):
        """Test support set validation."""
        from meta_learning.algorithms.ttc_scaler import TestTimeComputeScaler
        from meta_learning.algorithms.ttc_config import TestTimeComputeConfig
        
        config = TestTimeComputeConfig()
        scaler = TestTimeComputeScaler(self.valid_encoder, config)
        
        # Test invalid support_x types
        with pytest.raises(ValidationError, match="support_set must be a torch.Tensor"):
            scaler.scale_compute("not_tensor", self.valid_support_y, self.valid_query_x, self.valid_task_context)
        
        with pytest.raises(ValidationError, match="support_set must be a torch.Tensor"):
            scaler.scale_compute(None, self.valid_support_y, self.valid_query_x, self.valid_task_context)
        
        # Test invalid support_y types
        with pytest.raises(ValidationError, match="support_labels must be a torch.Tensor"):
            scaler.scale_compute(self.valid_support_x, "not_tensor", self.valid_query_x, self.valid_task_context)
    
    def test_query_set_validation(self):
        """Test query set validation."""
        from meta_learning.algorithms.ttc_scaler import TestTimeComputeScaler
        from meta_learning.algorithms.ttc_config import TestTimeComputeConfig
        
        config = TestTimeComputeConfig()
        scaler = TestTimeComputeScaler(self.valid_encoder, config)
        
        # Test invalid query_x types
        with pytest.raises(ValidationError, match="query_set must be a torch.Tensor"):
            scaler.scale_compute(self.valid_support_x, self.valid_support_y, "not_tensor", self.valid_task_context)
        
        with pytest.raises(ValidationError, match="query_set must be a torch.Tensor"):
            scaler.scale_compute(self.valid_support_x, self.valid_support_y, None, self.valid_task_context)
    
    def test_task_context_validation(self):
        """Test task context validation."""
        from meta_learning.algorithms.ttc_scaler import TestTimeComputeScaler
        from meta_learning.algorithms.ttc_config import TestTimeComputeConfig
        
        config = TestTimeComputeConfig()
        scaler = TestTimeComputeScaler(self.valid_encoder, config)
        
        # Test invalid task_context types
        with pytest.raises(ValidationError, match="task_context must be a dictionary"):
            scaler.scale_compute(self.valid_support_x, self.valid_support_y, self.valid_query_x, "not_dict")
        
        with pytest.raises(ValidationError, match="task_context must be a dictionary"):
            scaler.scale_compute(self.valid_support_x, self.valid_support_y, self.valid_query_x, None)
        
        # Test missing required keys
        with pytest.raises(ValidationError, match="task_context must contain 'n_classes'"):
            scaler.scale_compute(self.valid_support_x, self.valid_support_y, self.valid_query_x, {})
    
    def test_dimension_mismatch_validation(self):
        """Test validation of tensor dimension mismatches."""
        from meta_learning.algorithms.ttc_scaler import TestTimeComputeScaler
        from meta_learning.algorithms.ttc_config import TestTimeComputeConfig
        
        config = TestTimeComputeConfig()
        scaler = TestTimeComputeScaler(self.valid_encoder, config)
        
        # Test feature dimension mismatch
        wrong_dim_query = torch.randn(15, 32)  # Wrong feature dimension
        with pytest.raises(ValidationError, match="Feature dimension mismatch"):
            scaler.scale_compute(self.valid_support_x, self.valid_support_y, wrong_dim_query, self.valid_task_context)
        
        # Test label count mismatch
        wrong_labels = torch.repeat_interleave(torch.arange(3), 5)  # Only 3 classes instead of 5
        wrong_context = {"n_classes": 3}
        with pytest.raises(ValidationError, match="Number of support labels.*does not match.*n_classes"):
            scaler.scale_compute(self.valid_support_x, wrong_labels, self.valid_query_x, wrong_context)


class TestTTCSWarningSystem:
    """Test TTCS warning system functionality."""
    
    def setup_method(self):
        """Setup warning system fixtures."""
        self.warning_system = TTCSWarningSystem(max_warnings=5, deduplication_window=60)
    
    def test_warning_system_initialization(self):
        """Test warning system initialization."""
        assert self.warning_system.max_warnings == 5
        assert self.warning_system.deduplication_window == 60
        assert len(self.warning_system.warning_history) == 0
        assert len(self.warning_system.warning_counts) == 0
    
    def test_warning_emission(self):
        """Test warning emission and tracking."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            self.warning_system.warn("Test warning message", "TestCategory")
            
            assert len(w) == 1
            assert "Test warning message" in str(w[0].message)
            assert len(self.warning_system.warning_history) == 1
            assert self.warning_system.warning_counts["TestCategory"] == 1
    
    def test_warning_deduplication(self):
        """Test warning deduplication within window."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # Emit same warning multiple times
            for i in range(3):
                self.warning_system.warn("Duplicate warning", "TestCategory")
            
            # Should only emit once due to deduplication
            assert len(w) == 1
            assert len(self.warning_system.warning_history) == 1
            assert self.warning_system.warning_counts["TestCategory"] == 1
    
    def test_warning_limit(self):
        """Test warning limit enforcement."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # Emit more warnings than the limit
            for i in range(7):  # Limit is 5
                self.warning_system.warn(f"Warning {i}", "TestCategory")
            
            # Should stop emitting after limit
            assert len(w) <= 6  # 5 + 1 limit exceeded warning
            assert any("Maximum warnings exceeded" in str(warning.message) for warning in w)
    
    def test_warning_categories(self):
        """Test warning categorization."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            self.warning_system.warn("Performance warning", "Performance")
            self.warning_system.warn("Compatibility warning", "Compatibility")
            self.warning_system.warn("Another performance warning", "Performance")
            
            assert self.warning_system.warning_counts["Performance"] == 2
            assert self.warning_system.warning_counts["Compatibility"] == 1
    
    def test_warning_statistics(self):
        """Test warning statistics collection."""
        self.warning_system.warn("Warning 1", "Category1")
        self.warning_system.warn("Warning 2", "Category2")
        self.warning_system.warn("Warning 3", "Category1")
        
        stats = self.warning_system.get_warning_statistics()
        
        assert stats["total_warnings"] == 3
        assert stats["categories"]["Category1"] == 2
        assert stats["categories"]["Category2"] == 1
        assert len(stats["recent_warnings"]) == 3


class TestTTCSProfiler:
    """Test TTCS profiler functionality."""
    
    def setup_method(self):
        """Setup profiler fixtures."""
        self.profiler = TTCSProfiler()
    
    def test_profiler_initialization(self):
        """Test profiler initialization."""
        assert hasattr(self.profiler, 'start_profiling')
        assert hasattr(self.profiler, 'stop_profiling')
        assert hasattr(self.profiler, 'get_profile_results')
        assert self.profiler.is_profiling == False
    
    def test_profiler_start_stop(self):
        """Test profiler start and stop functionality."""
        self.profiler.start_profiling()
        assert self.profiler.is_profiling == True
        
        self.profiler.stop_profiling()
        assert self.profiler.is_profiling == False
    
    def test_profiler_timing(self):
        """Test profiler timing measurements."""
        self.profiler.start_profiling()
        
        # Simulate some computation
        import time
        time.sleep(0.01)
        
        self.profiler.stop_profiling()
        
        results = self.profiler.get_profile_results()
        assert 'total_time' in results
        assert results['total_time'] > 0
        assert results['total_time'] < 1.0  # Should be very quick
    
    def test_profiler_memory_tracking(self):
        """Test profiler memory tracking."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for memory tracking test")
        
        self.profiler.start_profiling()
        
        # Allocate some GPU memory
        large_tensor = torch.randn(1000, 1000).cuda()
        
        self.profiler.stop_profiling()
        
        results = self.profiler.get_profile_results()
        assert 'memory_usage' in results
        assert results['memory_usage']['peak_memory'] > 0
        
        # Cleanup
        del large_tensor
        torch.cuda.empty_cache()
    
    def test_profiler_nested_calls(self):
        """Test profiler behavior with nested profiling calls."""
        self.profiler.start_profiling()
        self.profiler.start_profiling()  # Nested start
        
        import time
        time.sleep(0.005)
        
        self.profiler.stop_profiling()
        self.profiler.stop_profiling()  # Nested stop
        
        results = self.profiler.get_profile_results()
        assert 'total_time' in results
        assert results['total_time'] > 0


class TestTTCSFallbackMechanisms:
    """Test TTCS fallback mechanisms and error recovery."""
    
    def setup_method(self):
        """Setup fallback test fixtures."""
        self.encoder = nn.Linear(64, 128)
        self.support_x = torch.randn(20, 64)
        self.support_y = torch.repeat_interleave(torch.arange(4), 5)
        self.query_x = torch.randn(12, 64)
        self.task_context = {"n_classes": 4}
    
    @patch('meta_learning.algorithms.ttc_scaler.TestTimeComputeScaler.scale_compute')
    def test_ttcs_with_fallback_success(self, mock_scale_compute):
        """Test ttcs_with_fallback when TTCS succeeds."""
        # Mock successful TTCS operation
        expected_predictions = torch.randn(12, 4)
        expected_metrics = {"compute_scale_factor": 2.5}
        mock_scale_compute.return_value = (expected_predictions, expected_metrics)
        
        predictions, metrics = ttcs_with_fallback(
            self.encoder,
            self.support_x,
            self.support_y,
            self.query_x,
            self.task_context
        )
        
        assert torch.equal(predictions, expected_predictions)
        assert metrics == expected_metrics
        assert mock_scale_compute.called
    
    @patch('meta_learning.algorithms.ttc_scaler.TestTimeComputeScaler.scale_compute')
    def test_ttcs_with_fallback_error(self, mock_scale_compute):
        """Test ttcs_with_fallback when TTCS fails."""
        # Mock TTCS failure
        mock_scale_compute.side_effect = RuntimeError("TTCS failed")
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            predictions, metrics = ttcs_with_fallback(
                self.encoder,
                self.support_x,
                self.support_y,
                self.query_x,
                self.task_context
            )
            
            # Should fallback to simple nearest neighbor
            assert predictions is not None
            assert predictions.shape == (12, 4)
            assert 'fallback_used' in metrics
            assert metrics['fallback_used'] == True
            assert len(w) > 0  # Should emit warning
    
    @patch('meta_learning.algorithms.ttc_scaler.TestTimeComputeScaler.scale_compute')
    def test_ttcs_with_fallback_validation_error(self, mock_scale_compute):
        """Test ttcs_with_fallback with validation errors."""
        # Mock validation error
        mock_scale_compute.side_effect = ValidationError("Invalid input")
        
        with pytest.raises(ValidationError, match="Invalid input"):
            ttcs_with_fallback(
                self.encoder,
                self.support_x,
                self.support_y,
                self.query_x,
                self.task_context
            )
    
    def test_ttcs_for_learn2learn_models(self):
        """Test TTCS compatibility with learn2learn models."""
        # Create a mock learn2learn-style model
        class MockLearn2LearnModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = nn.Linear(64, 128)
                self.classifier = nn.Linear(128, 10)
            
            def forward(self, x):
                return self.classifier(self.features(x))
        
        model = MockLearn2LearnModel()
        
        # Test that we can extract encoder from learn2learn model
        predictions, metrics = ttcs_for_learn2learn_models(
            model,
            self.support_x,
            self.support_y,
            self.query_x,
            self.task_context
        )
        
        assert predictions is not None
        assert predictions.shape[0] == self.query_x.shape[0]
        assert 'model_type' in metrics
        assert metrics['model_type'] == 'learn2learn_compatible'


class TestTTCSCompatibilityChecks:
    """Test TTCS compatibility checks and model validation."""
    
    def test_has_dropout_layers_detection(self):
        """Test dropout layer detection."""
        from meta_learning.algos.ttcs import _has_dropout_layers
        
        # Model with dropout
        model_with_dropout = nn.Sequential(
            nn.Linear(64, 128),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )
        assert _has_dropout_layers(model_with_dropout) == True
        
        # Model without dropout
        model_without_dropout = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
        assert _has_dropout_layers(model_without_dropout) == False
        
        # Nested model with dropout
        nested_model = nn.Module()
        nested_model.encoder = nn.Sequential(nn.Linear(64, 128), nn.Dropout(0.3))
        nested_model.classifier = nn.Linear(128, 10)
        assert _has_dropout_layers(nested_model) == True
    
    def test_model_compatibility_warnings(self):
        """Test model compatibility warning generation."""
        from meta_learning.algorithms.ttc_scaler import TestTimeComputeScaler
        from meta_learning.algorithms.ttc_config import TestTimeComputeConfig
        
        # Model without dropout (should warn about MC-Dropout compatibility)
        model_no_dropout = nn.Linear(64, 10)
        config = TestTimeComputeConfig()
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            scaler = TestTimeComputeScaler(model_no_dropout, config)
            
            # Should emit compatibility warning
            warning_messages = [str(warning.message) for warning in w]
            assert any("MC-Dropout" in msg for msg in warning_messages)
    
    def test_device_compatibility_checks(self):
        """Test device compatibility validation."""
        from meta_learning.algorithms.ttc_scaler import TestTimeComputeScaler
        from meta_learning.algorithms.ttc_config import TestTimeComputeConfig
        
        encoder = nn.Linear(64, 128)
        config = TestTimeComputeConfig()
        scaler = TestTimeComputeScaler(encoder, config)
        
        # CPU tensors with CPU model (should work)
        support_x = torch.randn(20, 64)
        support_y = torch.repeat_interleave(torch.arange(4), 5)
        query_x = torch.randn(12, 64)
        task_context = {"n_classes": 4}
        
        predictions, metrics = scaler.scale_compute(support_x, support_y, query_x, task_context)
        assert predictions is not None
        
        # Test CUDA compatibility if available
        if torch.cuda.is_available():
            encoder_cuda = encoder.cuda()
            scaler_cuda = TestTimeComputeScaler(encoder_cuda, config)
            
            support_x_cuda = support_x.cuda()
            support_y_cuda = support_y.cuda()
            query_x_cuda = query_x.cuda()
            
            predictions_cuda, metrics_cuda = scaler_cuda.scale_compute(
                support_x_cuda, support_y_cuda, query_x_cuda, task_context
            )
            assert predictions_cuda.device.type == 'cuda'


class TestTTCSErrorHandlingIntegration:
    """Integration tests for TTCS error handling components."""
    
    def setup_method(self):
        """Setup integration test fixtures."""
        self.encoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64)
        )
        
        # Create episode
        self.episode = Episode(
            support_x=torch.randn(20, 64),
            support_y=torch.repeat_interleave(torch.arange(4), 5),
            query_x=torch.randn(12, 64),
            query_y=torch.repeat_interleave(torch.arange(4), 3)
        )
    
    def test_complete_ttcs_pipeline_with_monitoring(self):
        """Test complete TTCS pipeline with monitoring and error handling."""
        from meta_learning.algorithms.ttc_scaler import TestTimeComputeScaler
        from meta_learning.algorithms.ttc_config import TestTimeComputeConfig
        
        # Initialize components
        warning_system = TTCSWarningSystem()
        profiler = TTCSProfiler()
        config = TestTimeComputeConfig()
        scaler = TestTimeComputeScaler(self.encoder, config)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # Start profiling
            profiler.start_profiling()
            
            # Run TTCS with error handling
            predictions, metrics = ttcs_with_fallback(
                self.encoder,
                self.episode.support_x,
                self.episode.support_y,
                self.episode.query_x,
                {"n_classes": 4}
            )
            
            # Stop profiling
            profiler.stop_profiling()
            
            # Verify results
            assert predictions is not None
            assert predictions.shape == (12, 4)
            assert isinstance(metrics, dict)
            
            # Check profiling results
            profile_results = profiler.get_profile_results()
            assert 'total_time' in profile_results
            assert profile_results['total_time'] > 0
    
    def test_error_recovery_cascade(self):
        """Test error recovery cascade with multiple fallback levels."""
        # Create a model that will definitely fail
        class FailingModel(nn.Module):
            def forward(self, x):
                raise RuntimeError("Intentional failure for testing")
        
        failing_encoder = FailingModel()
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # Should fallback gracefully
            predictions, metrics = ttcs_with_fallback(
                failing_encoder,
                self.episode.support_x,
                self.episode.support_y,
                self.episode.query_x,
                {"n_classes": 4}
            )
            
            # Should still produce valid output via fallback
            assert predictions is not None
            assert predictions.shape == (12, 4)
            assert metrics['fallback_used'] == True
            
            # Should emit appropriate warnings
            warning_messages = [str(warning.message) for warning in w]
            assert any("TTCS failed" in msg for msg in warning_messages)
    
    def test_validation_with_comprehensive_error_messages(self):
        """Test validation with comprehensive error messages."""
        from meta_learning.algorithms.ttc_scaler import TestTimeComputeScaler
        from meta_learning.algorithms.ttc_config import TestTimeComputeConfig
        
        config = TestTimeComputeConfig()
        scaler = TestTimeComputeScaler(self.encoder, config)
        
        # Test comprehensive error message for dimension mismatch
        wrong_query = torch.randn(12, 32)  # Wrong feature dimension
        
        try:
            scaler.scale_compute(
                self.episode.support_x,
                self.episode.support_y,
                wrong_query,
                {"n_classes": 4}
            )
            assert False, "Should have raised ValidationError"
        except ValidationError as e:
            error_msg = str(e)
            assert "Feature dimension mismatch" in error_msg
            assert "support_set: 64" in error_msg
            assert "query_set: 32" in error_msg


if __name__ == "__main__":
    pytest.main([__file__])