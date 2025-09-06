"""
Tests for error handling and monitoring system including intelligent
error recovery, performance monitoring, and warning management.

Tests cover:
- IntelligentErrorRecovery with learning capabilities
- PerformanceMonitor with real-time metrics
- WarningManager with filtering and categorization
- Error recovery strategies and fallback mechanisms
- Professional monitoring and alerting systems
"""

import pytest
import torch
import torch.nn as nn
import time
import threading
import warnings
from unittest.mock import Mock, patch, MagicMock
from collections import defaultdict

from meta_learning.error_handling import (
    IntelligentErrorRecovery, PerformanceMonitor, WarningManager,
    ErrorType, WarningCategory, ErrorContext, WarningInfo,
    MemoryErrorRecovery, NumericalInstabilityRecovery, DimensionMismatchRecovery,
    with_error_recovery, monitor_performance
)


class TestErrorRecoveryStrategies:
    """Test individual error recovery strategies."""
    
    def test_memory_error_recovery(self):
        """Test memory error recovery strategy."""
        strategy = MemoryErrorRecovery()
        
        # Create memory error context
        error_context = ErrorContext(
            error_type=ErrorType.MEMORY_ERROR,
            error_message="CUDA out of memory. Tried to allocate 2.00 GiB",
            stack_trace="",
            timestamp=time.time(),
            system_state={}
        )
        
        # Should be able to handle memory errors
        assert strategy.can_handle(error_context) == True
        
        # Test recovery
        success, description = strategy.recover(error_context)
        assert isinstance(success, bool)
        assert isinstance(description, str)
        assert "cache" in description.lower() or "memory" in description.lower()
    
    def test_numerical_instability_recovery(self):
        """Test numerical instability recovery strategy."""
        strategy = NumericalInstabilityRecovery()
        
        # Create numerical error context
        error_context = ErrorContext(
            error_type=ErrorType.NUMERICAL_INSTABILITY,
            error_message="Found NaN values in forward pass",
            stack_trace="",
            timestamp=time.time(),
            system_state={}
        )
        
        # Should be able to handle numerical errors
        assert strategy.can_handle(error_context) == True
        
        # Test recovery
        success, description = strategy.recover(error_context)
        assert isinstance(success, bool)
        assert isinstance(description, str)
        assert "numerical" in description.lower() or "stability" in description.lower()
    
    def test_dimension_mismatch_recovery(self):
        """Test dimension mismatch recovery strategy."""
        strategy = DimensionMismatchRecovery()
        
        # Create dimension error context
        error_context = ErrorContext(
            error_type=ErrorType.DIMENSION_MISMATCH,
            error_message="Expected tensor of size [32, 64], got [32, 128]",
            stack_trace="",
            timestamp=time.time(),
            system_state={}
        )
        
        # Should be able to handle dimension errors
        assert strategy.can_handle(error_context) == True
        
        # Test recovery (returns analysis, not actual fix)
        success, description = strategy.recover(error_context)
        assert isinstance(success, bool)
        assert isinstance(description, str)
        assert "dimension" in description.lower() or "tensor" in description.lower()
    
    def test_strategy_specificity(self):
        """Test that strategies correctly identify their error types."""
        memory_strategy = MemoryErrorRecovery()
        numerical_strategy = NumericalInstabilityRecovery()
        dimension_strategy = DimensionMismatchRecovery()
        
        # Memory error
        memory_context = ErrorContext(
            error_type=ErrorType.MEMORY_ERROR,
            error_message="out of memory",
            stack_trace="", timestamp=time.time(), system_state={}
        )
        
        assert memory_strategy.can_handle(memory_context) == True
        assert numerical_strategy.can_handle(memory_context) == False
        assert dimension_strategy.can_handle(memory_context) == False
        
        # NaN error
        nan_context = ErrorContext(
            error_type=ErrorType.UNKNOWN_ERROR,
            error_message="Found inf values in computation",
            stack_trace="", timestamp=time.time(), system_state={}
        )
        
        assert memory_strategy.can_handle(nan_context) == False
        assert numerical_strategy.can_handle(nan_context) == True
        assert dimension_strategy.can_handle(nan_context) == False


class TestIntelligentErrorRecovery:
    """Test IntelligentErrorRecovery system."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.recovery = IntelligentErrorRecovery(max_retries=3, enable_learning=True)
    
    def test_error_recovery_initialization(self):
        """Test error recovery system initialization."""
        assert self.recovery.max_retries == 3
        assert self.recovery.enable_learning == True
        assert len(self.recovery.strategies) > 0
        assert len(self.recovery.error_history) == 0
        assert len(self.recovery.recovery_success_rates) == 0
    
    def test_error_classification(self):
        """Test error type classification."""
        # Memory error
        memory_error = RuntimeError("CUDA out of memory")
        error_type = self.recovery._classify_error(memory_error)
        assert error_type == ErrorType.MEMORY_ERROR
        
        # Numerical error
        nan_error = ValueError("Found NaN in forward pass")
        error_type = self.recovery._classify_error(nan_error)
        assert error_type == ErrorType.NUMERICAL_INSTABILITY
        
        # Dimension error
        size_error = RuntimeError("size mismatch: expected [32, 64], got [32, 128]")
        error_type = self.recovery._classify_error(size_error)
        assert error_type == ErrorType.DIMENSION_MISMATCH
        
        # Unknown error
        unknown_error = ValueError("Unknown error message")
        error_type = self.recovery._classify_error(unknown_error)
        assert error_type == ErrorType.UNKNOWN_ERROR
    
    def test_system_state_collection(self):
        """Test system state collection."""
        state = self.recovery._get_system_state()
        
        assert 'timestamp' in state
        assert 'torch_version' in state
        assert 'cuda_available' in state
        
        if torch.cuda.is_available():
            assert 'cuda_memory_allocated' in state
            assert 'cuda_memory_cached' in state
            assert 'cuda_device_count' in state
        
        assert isinstance(state['timestamp'], float)
        assert isinstance(state['cuda_available'], bool)
    
    def test_error_handling_success(self):
        """Test successful error handling."""
        # Create a recoverable error (memory error)
        memory_error = RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        
        success, description = self.recovery.handle_error(memory_error, {"context": "test"})
        
        # Should attempt recovery
        assert isinstance(success, bool)
        assert isinstance(description, str)
        
        # Should record in history
        assert len(self.recovery.error_history) == 1
        error_record = self.recovery.error_history[0]
        assert error_record.error_type == ErrorType.MEMORY_ERROR
        assert "out of memory" in error_record.error_message
        assert "context" in error_record.system_state
    
    def test_error_handling_with_learning(self):
        """Test error handling with learning enabled."""
        # Handle multiple similar errors
        for i in range(3):
            memory_error = RuntimeError(f"CUDA out of memory {i}")
            self.recovery.handle_error(memory_error)
        
        # Check learning data
        assert len(self.recovery.error_history) == 3
        assert len(self.recovery.recovery_success_rates) > 0
        assert 'memory_error' in self.recovery.recovery_success_rates
        
        # Check strategy effectiveness tracking
        assert len(self.recovery.strategy_effectiveness) > 0
        for strategy_name, stats in self.recovery.strategy_effectiveness.items():
            assert 'attempts' in stats
            assert 'successes' in stats
            assert stats['attempts'] >= 0
            assert stats['successes'] >= 0
    
    def test_error_recovery_statistics(self):
        """Test error recovery statistics collection."""
        # Generate some errors for statistics
        errors = [
            RuntimeError("CUDA out of memory"),
            ValueError("Found NaN values"),
            RuntimeError("size mismatch"),
        ]
        
        for error in errors:
            self.recovery.handle_error(error)
        
        # Get statistics
        stats = self.recovery.get_recovery_statistics()
        
        assert 'total_errors' in stats
        assert 'error_types' in stats
        assert 'strategy_effectiveness' in stats
        assert 'recent_success_rate' in stats
        
        assert stats['total_errors'] == 3
        assert len(stats['error_types']) > 0
        assert 0.0 <= stats['recent_success_rate'] <= 1.0
    
    def test_error_recovery_report(self):
        """Test error recovery report generation."""
        # Generate some errors
        memory_error = RuntimeError("CUDA out of memory")
        nan_error = ValueError("Found NaN")
        
        self.recovery.handle_error(memory_error)
        self.recovery.handle_error(nan_error)
        
        # Generate report
        report = self.recovery.generate_error_report()
        
        assert isinstance(report, str)
        assert "ERROR RECOVERY REPORT" in report
        assert "Total errors handled" in report
        assert "Recent success rate" in report
        assert "Strategy Effectiveness" in report
    
    def test_learning_pattern_detection(self):
        """Test learning from error patterns."""
        # Simulate repeated errors with specific keywords
        for i in range(5):
            error = RuntimeError("tensor dimension mismatch in layer")
            self.recovery.handle_error(error)
        
        # Check that keywords were learned
        assert len(self.recovery.learned_patterns) > 0
        
        # Common keywords should appear
        for keyword in ['tensor', 'dimension', 'mismatch']:
            if keyword in self.recovery.learned_patterns:
                pattern_stats = self.recovery.learned_patterns[keyword]
                assert pattern_stats['attempts'] > 0


class TestPerformanceMonitor:
    """Test PerformanceMonitor functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.monitor = PerformanceMonitor(collection_interval=0.1)  # Fast collection for testing
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if self.monitor.monitoring:
            self.monitor.stop_monitoring()
    
    def test_monitor_initialization(self):
        """Test performance monitor initialization."""
        assert self.monitor.collection_interval == 0.1
        assert self.monitor.monitoring == False
        assert len(self.monitor.metrics) == 0
        assert len(self.monitor.alerts) == 0
        assert len(self.monitor.thresholds) > 0
    
    def test_monitor_start_stop(self):
        """Test monitor start and stop functionality."""
        assert self.monitor.monitoring == False
        
        self.monitor.start_monitoring()
        assert self.monitor.monitoring == True
        assert self.monitor.monitor_thread is not None
        
        self.monitor.stop_monitoring()
        assert self.monitor.monitoring == False
    
    def test_custom_metric_recording(self):
        """Test custom metric recording."""
        # Record some metrics
        self.monitor.record_metric('test_metric', 0.5)
        self.monitor.record_metric('test_metric', 0.7)
        self.monitor.record_metric('another_metric', 1.2)
        
        # Check metrics were recorded
        assert 'test_metric' in self.monitor.metrics
        assert 'another_metric' in self.monitor.metrics
        assert len(self.monitor.metrics['test_metric']) == 2
        assert len(self.monitor.metrics['another_metric']) == 1
        
        # Check metric values
        timestamps, values = zip(*self.monitor.metrics['test_metric'])
        assert list(values) == [0.5, 0.7]
    
    def test_threshold_alerting(self):
        """Test threshold-based alerting."""
        # Set a low threshold for testing
        self.monitor.thresholds['test_metric'] = 0.8
        
        # Record metrics below and above threshold
        self.monitor.record_metric('test_metric', 0.5)  # Below threshold
        assert len(self.monitor.alerts) == 0
        
        self.monitor.record_metric('test_metric', 1.0)  # Above threshold
        assert len(self.monitor.alerts) == 1
        
        timestamp, alert_msg = self.monitor.alerts[0]
        assert 'test_metric' in alert_msg
        assert '1.0' in alert_msg
        assert 'exceeds threshold 0.8' in alert_msg
    
    def test_background_monitoring(self):
        """Test background monitoring functionality."""
        self.monitor.start_monitoring()
        
        # Wait for some monitoring cycles
        time.sleep(0.3)  # Should allow a few collection cycles
        
        # Should have collected some system metrics
        if torch.cuda.is_available():
            assert 'gpu_memory_usage_percent' in self.monitor.metrics
        
        # Stop monitoring
        self.monitor.stop_monitoring()
        
        # Should have some data
        total_metrics = sum(len(metric_data) for metric_data in self.monitor.metrics.values())
        assert total_metrics > 0
    
    def test_performance_summary(self):
        """Test performance summary generation."""
        # Record some test data
        self.monitor.record_metric('accuracy', 0.85)
        self.monitor.record_metric('accuracy', 0.90)
        self.monitor.record_metric('loss', 0.5)
        
        # Generate alert
        self.monitor.thresholds['accuracy'] = 0.8
        self.monitor.record_metric('accuracy', 0.95)  # Above threshold
        
        summary = self.monitor.get_performance_summary()
        
        assert 'monitoring_duration_seconds' in summary
        assert 'total_alerts' in summary
        assert 'recent_alerts' in summary
        assert 'metrics_summary' in summary
        
        # Check metrics summary
        metrics_summary = summary['metrics_summary']
        assert 'accuracy' in metrics_summary
        assert 'loss' in metrics_summary
        
        accuracy_stats = metrics_summary['accuracy']
        assert 'current' in accuracy_stats
        assert 'average' in accuracy_stats
        assert 'min' in accuracy_stats
        assert 'max' in accuracy_stats
        assert 'data_points' in accuracy_stats
        
        assert accuracy_stats['current'] == 0.95
        assert accuracy_stats['min'] == 0.85
        assert accuracy_stats['max'] == 0.95
        assert accuracy_stats['data_points'] == 3
    
    def test_performance_report(self):
        """Test performance report generation."""
        # Add some test data
        self.monitor.record_metric('test_metric', 1.0)
        self.monitor.record_metric('test_metric', 2.0)
        
        # Generate alert
        self.monitor.thresholds['test_metric'] = 0.5
        self.monitor.record_metric('test_metric', 1.5)
        
        report = self.monitor.get_performance_report()
        
        assert isinstance(report, str)
        assert "PERFORMANCE MONITORING REPORT" in report
        assert "Monitoring duration" in report
        assert "Total alerts" in report
        assert "Metrics Summary" in report
        assert "test_metric" in report


class TestWarningManager:
    """Test WarningManager functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.warning_manager = WarningManager(max_history=100)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        self.warning_manager.restore_warnings()
    
    def test_warning_manager_initialization(self):
        """Test warning manager initialization."""
        assert self.warning_manager.max_history == 100
        assert len(self.warning_manager.warnings) == 0
        assert len(self.warning_manager.warning_counts) == 0
        assert len(self.warning_manager.suppressed_warnings) == 0
    
    def test_warning_capture(self):
        """Test warning capture functionality."""
        # Generate a warning
        warnings.warn("Test warning message", UserWarning)
        
        # Should be captured
        assert len(self.warning_manager.warnings) == 1
        
        warning_info = self.warning_manager.warnings[0]
        assert isinstance(warning_info, WarningInfo)
        assert "Test warning message" in warning_info.message
        assert warning_info.count == 1
    
    def test_warning_categorization(self):
        """Test automatic warning categorization."""
        # Generate warnings with different categories
        warnings.warn("Performance is slow", UserWarning)
        warnings.warn("This feature is deprecated", DeprecationWarning)
        warnings.warn("Memory allocation failed", UserWarning)
        warnings.warn("NaN detected in computation", RuntimeWarning)
        
        # Check categorization
        categories = [w.category for w in self.warning_manager.warnings]
        
        # Should have categorized warnings
        assert WarningCategory.PERFORMANCE in categories
        assert WarningCategory.DEPRECATION in categories
        assert WarningCategory.MEMORY in categories
        assert WarningCategory.NUMERICAL in categories
    
    def test_warning_deduplication(self):
        """Test warning deduplication."""
        # Generate identical warnings
        for i in range(3):
            warnings.warn("Duplicate warning", UserWarning)
        
        # Should only store one warning but increase count
        assert len(self.warning_manager.warnings) == 1
        warning_info = self.warning_manager.warnings[0]
        assert warning_info.count == 3
    
    def test_warning_suppression(self):
        """Test warning suppression functionality."""
        # Generate a warning first
        warnings.warn("Test warning to suppress", UserWarning)
        assert len(self.warning_manager.warnings) == 1
        
        # Suppress warnings containing "suppress"
        self.warning_manager.suppress_warning(pattern="suppress")
        
        # Generate same warning again
        warnings.warn("Test warning to suppress", UserWarning)
        
        # Should still be only one warning (suppressed the duplicate)
        assert len(self.warning_manager.warnings) == 1
    
    def test_warning_summary(self):
        """Test warning summary generation."""
        # Generate various warnings
        warnings.warn("Performance warning", UserWarning)
        warnings.warn("Deprecated feature", DeprecationWarning)
        warnings.warn("Performance warning", UserWarning)  # Duplicate
        warnings.warn("Memory issue", UserWarning)
        
        summary = self.warning_manager.get_warning_summary()
        
        assert 'total_warnings' in summary
        assert 'recent_warnings' in summary
        assert 'category_counts' in summary
        assert 'suppressed_count' in summary
        assert 'top_warnings' in summary
        
        assert summary['total_warnings'] >= 3
        assert len(summary['category_counts']) > 0
        assert len(summary['top_warnings']) > 0
        
        # Check top warnings format
        top_warning = summary['top_warnings'][0]
        assert len(top_warning) == 2  # (message, count)
        assert isinstance(top_warning[1], int)
    
    def test_warning_report(self):
        """Test warning report generation."""
        # Generate test warnings
        warnings.warn("Test performance warning", UserWarning)
        warnings.warn("Test deprecated feature", DeprecationWarning)
        
        report = self.warning_manager.generate_warning_report()
        
        assert isinstance(report, str)
        assert "WARNING MANAGEMENT REPORT" in report
        assert "Total warnings captured" in report
        assert "Warning Categories" in report
        assert "Top Warning Messages" in report
    
    def test_warning_clearing(self):
        """Test warning history clearing."""
        # Generate warnings
        warnings.warn("Warning 1", UserWarning)
        warnings.warn("Warning 2", UserWarning)
        
        assert len(self.warning_manager.warnings) == 2
        
        # Clear warnings
        self.warning_manager.clear_warnings()
        
        assert len(self.warning_manager.warnings) == 0
        assert len(self.warning_manager.warning_counts) == 0
    
    def test_max_history_limit(self):
        """Test warning history size limit."""
        # Create manager with small limit
        small_manager = WarningManager(max_history=3)
        
        # Generate more warnings than the limit
        for i in range(5):
            warnings.warn(f"Warning {i}", UserWarning)
        
        # Should only keep the most recent warnings
        assert len(small_manager.warnings) <= 3
        
        # Clean up
        small_manager.restore_warnings()


class TestErrorHandlingDecorators:
    """Test error handling decorators."""
    
    def test_with_error_recovery_decorator(self):
        """Test with_error_recovery decorator."""
        @with_error_recovery(max_retries=2)
        def failing_function(fail_count=0):
            if hasattr(failing_function, 'call_count'):
                failing_function.call_count += 1
            else:
                failing_function.call_count = 1
            
            if failing_function.call_count <= fail_count:
                raise RuntimeError(f"Failure {failing_function.call_count}")
            
            return "Success"
        
        # Function that succeeds immediately
        result = failing_function(fail_count=0)
        assert result == "Success"
        
        # Reset call count
        failing_function.call_count = 0
        
        # Function that fails once then succeeds
        result = failing_function(fail_count=1)
        assert result == "Success"
        assert failing_function.call_count == 2  # Should have retried
    
    def test_monitor_performance_decorator(self):
        """Test monitor_performance decorator."""
        monitor = PerformanceMonitor(collection_interval=1.0)
        
        @monitor_performance(monitor)
        def timed_function(duration=0.01):
            time.sleep(duration)
            return "completed"
        
        # Execute function
        result = timed_function(duration=0.02)
        assert result == "completed"
        
        # Check that execution time was recorded
        assert f'timed_function_execution_time' in monitor.metrics
        execution_times = monitor.metrics[f'timed_function_execution_time']
        assert len(execution_times) == 1
        
        timestamp, exec_time = execution_times[0]
        assert exec_time >= 0.015  # Should be at least the sleep time
    
    def test_monitor_performance_with_errors(self):
        """Test performance monitoring with function errors."""
        monitor = PerformanceMonitor(collection_interval=1.0)
        
        @monitor_performance(monitor)
        def error_function():
            time.sleep(0.01)
            raise ValueError("Test error")
        
        # Execute function that raises error
        with pytest.raises(ValueError, match="Test error"):
            error_function()
        
        # Check that error time was recorded
        assert 'error_function_error_time' in monitor.metrics
        error_times = monitor.metrics['error_function_error_time']
        assert len(error_times) == 1
        
        timestamp, exec_time = error_times[0]
        assert exec_time >= 0.005


class TestErrorHandlingIntegration:
    """Integration tests for error handling components."""
    
    def setup_method(self):
        """Setup integration test fixtures."""
        self.recovery = IntelligentErrorRecovery(max_retries=2, enable_learning=True)
        self.monitor = PerformanceMonitor(collection_interval=0.1)
        self.warning_manager = WarningManager(max_history=50)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if self.monitor.monitoring:
            self.monitor.stop_monitoring()
        self.warning_manager.restore_warnings()
    
    def test_complete_error_handling_pipeline(self):
        """Test complete error handling pipeline integration."""
        # Start monitoring
        self.monitor.start_monitoring()
        
        # Simulate various errors and recoveries
        errors = [
            RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"),
            ValueError("Found NaN values in computation"),
            RuntimeError("Expected tensor size [32, 64], got [32, 128]"),
        ]
        
        for error in errors:
            # Handle error
            success, description = self.recovery.handle_error(error)
            
            # Record performance metric
            recovery_success = 1.0 if success else 0.0
            self.monitor.record_metric('recovery_success_rate', recovery_success)
            
            # Generate warning
            warnings.warn(f"Error handled: {description}", UserWarning)
        
        # Stop monitoring
        time.sleep(0.2)  # Let monitoring collect some data
        self.monitor.stop_monitoring()
        
        # Verify all systems worked together
        # 1. Error recovery
        recovery_stats = self.recovery.get_recovery_statistics()
        assert recovery_stats['total_errors'] == 3
        assert len(recovery_stats['error_types']) > 0
        
        # 2. Performance monitoring
        perf_summary = self.monitor.get_performance_summary()
        assert 'recovery_success_rate' in perf_summary['metrics_summary']
        
        # 3. Warning management
        warning_summary = self.warning_manager.get_warning_summary()
        assert warning_summary['total_warnings'] >= 3
    
    def test_cascading_error_recovery(self):
        """Test cascading error recovery with multiple strategies."""
        # Create a complex error that might trigger multiple strategies
        class ComplexError(Exception):
            def __init__(self):
                super().__init__("CUDA out of memory during matrix multiply with NaN values")
        
        complex_error = ComplexError()
        
        # Handle the error
        success, description = self.recovery.handle_error(complex_error, {
            'operation': 'matrix_multiply',
            'tensor_shapes': [(32, 64), (64, 128)],
            'memory_usage_gb': 2.5
        })
        
        # Should attempt recovery
        assert isinstance(success, bool)
        assert isinstance(description, str)
        
        # Should record in error history with context
        assert len(self.recovery.error_history) == 1
        error_record = self.recovery.error_history[0]
        assert 'operation' in error_record.system_state
        assert 'memory_usage_gb' in error_record.system_state
    
    def test_performance_monitoring_during_recovery(self):
        """Test performance monitoring during error recovery operations."""
        # Start monitoring
        self.monitor.start_monitoring()
        
        # Create a recovery system with monitoring
        @monitor_performance(self.monitor)
        def recovery_operation():
            # Simulate some recovery work
            time.sleep(0.01)
            return "recovery_complete"
        
        # Execute recovery operations
        for i in range(3):
            result = recovery_operation()
            assert result == "recovery_complete"
        
        # Stop monitoring
        self.monitor.stop_monitoring()
        
        # Check that recovery operations were timed
        assert 'recovery_operation_execution_time' in self.monitor.metrics
        execution_times = self.monitor.metrics['recovery_operation_execution_time']
        assert len(execution_times) == 3
        
        # All execution times should be reasonable
        for timestamp, exec_time in execution_times:
            assert exec_time >= 0.005  # Should be at least the sleep time
            assert exec_time < 1.0     # Should be reasonable
    
    def test_warning_filtering_during_recovery(self):
        """Test warning filtering during error recovery."""
        # Set up warning suppression for common recovery warnings
        self.warning_manager.suppress_warning(pattern="fallback")
        
        # Generate various warnings during simulated recovery
        warnings.warn("Attempting recovery with fallback strategy", UserWarning)
        warnings.warn("Performance degradation detected", UserWarning)
        warnings.warn("Using fallback algorithm", UserWarning)
        warnings.warn("Memory optimization applied", UserWarning)
        
        # Check warning filtering
        summary = self.warning_manager.get_warning_summary()
        
        # Should have filtered out fallback warnings
        assert summary['suppressed_count'] > 0
        assert summary['total_warnings'] < 4  # Some warnings should be suppressed
        
        # Should still capture non-suppressed warnings
        assert summary['total_warnings'] > 0
    
    def test_learning_from_recovery_patterns(self):
        """Test learning from recovery patterns across sessions."""
        # Simulate repeated similar errors (like in a training loop)
        base_errors = [
            "CUDA out of memory during forward pass",
            "CUDA out of memory during backward pass",
            "CUDA out of memory during optimizer step",
        ]
        
        # Handle multiple instances of similar errors
        for i in range(3):
            for error_msg in base_errors:
                error = RuntimeError(f"{error_msg} (iteration {i})")
                success, description = self.recovery.handle_error(error, {
                    'iteration': i,
                    'batch_size': 32,
                    'model_size': 'large'
                })
        
        # Check learning results
        stats = self.recovery.get_recovery_statistics()
        assert stats['total_errors'] == 9
        
        # Should have learned patterns from error messages
        assert len(self.recovery.learned_patterns) > 0
        common_keywords = ['cuda', 'memory', 'forward', 'backward']
        
        for keyword in common_keywords:
            if keyword in self.recovery.learned_patterns:
                pattern_stats = self.recovery.learned_patterns[keyword]
                assert pattern_stats['attempts'] > 0
    
    def test_comprehensive_monitoring_report(self):
        """Test comprehensive monitoring report across all components."""
        # Start monitoring
        self.monitor.start_monitoring()
        
        # Simulate a complete ML training scenario with errors
        for epoch in range(2):
            # Simulate training metrics
            self.monitor.record_metric('train_accuracy', 0.7 + epoch * 0.1)
            self.monitor.record_metric('train_loss', 1.0 - epoch * 0.2)
            
            # Simulate some errors during training
            if epoch == 1:
                error = RuntimeError("Memory allocation failed")
                success, desc = self.recovery.handle_error(error)
                
                # Record recovery metrics
                self.monitor.record_metric('errors_recovered', 1.0 if success else 0.0)
                
                # Generate warnings
                warnings.warn("Training instability detected", UserWarning)
                warnings.warn("Switching to gradient checkpointing", UserWarning)
        
        # Stop monitoring and generate comprehensive report
        time.sleep(0.15)  # Allow some monitoring cycles
        self.monitor.stop_monitoring()
        
        # Generate reports from all systems
        performance_report = self.monitor.get_performance_report()
        recovery_report = self.recovery.generate_error_report()
        warning_report = self.warning_manager.generate_warning_report()
        
        # Verify comprehensive reporting
        assert "PERFORMANCE MONITORING REPORT" in performance_report
        assert "train_accuracy" in performance_report
        assert "train_loss" in performance_report
        
        assert "ERROR RECOVERY REPORT" in recovery_report
        assert "Total errors handled: 1" in recovery_report
        
        assert "WARNING MANAGEMENT REPORT" in warning_report
        assert "Total warnings captured" in warning_report
        
        # All reports should contain relevant information
        for report in [performance_report, recovery_report, warning_report]:
            assert len(report) > 50  # Should be substantial reports
            assert isinstance(report, str)


if __name__ == "__main__":
    pytest.main([__file__])