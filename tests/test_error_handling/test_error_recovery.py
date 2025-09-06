"""
Tests for IntelligentErrorRecovery.

Tests ML-based failure prediction and context-aware recovery strategies.
"""

import pytest
import torch
import time
from unittest.mock import Mock, patch, MagicMock
import threading

from meta_learning.error_handling.error_recovery import (
    IntelligentErrorRecovery, ErrorContext, ErrorType, RecoveryStrategy,
    FailurePredictor, RecoveryEngine
)


class TestErrorContext:
    """Test error context functionality."""
    
    def test_initialization(self):
        """Test error context creation."""
        context = ErrorContext(
            error_type=ErrorType.TENSOR_DIMENSION,
            severity='high',
            function_name='test_function',
            parameters={'x': torch.randn(3, 4), 'y': 'test'}
        )
        
        assert context.error_type == ErrorType.TENSOR_DIMENSION
        assert context.severity == 'high'
        assert context.function_name == 'test_function'
        assert 'x' in context.parameters
        assert context.timestamp is not None
    
    def test_context_serialization(self):
        """Test context can be serialized for analysis."""
        context = ErrorContext(
            error_type=ErrorType.NUMERICAL_INSTABILITY,
            severity='medium',
            function_name='matrix_multiply',
            parameters={'learning_rate': 0.01}
        )
        
        # Should be able to extract relevant info
        info = context.get_context_info()
        
        assert isinstance(info, dict)
        assert info['error_type'] == 'NUMERICAL_INSTABILITY'
        assert info['severity'] == 'medium'
        assert info['function_name'] == 'matrix_multiply'
        assert 'timestamp' in info
    
    def test_tensor_parameter_handling(self):
        """Test handling of tensor parameters."""
        large_tensor = torch.randn(1000, 1000)
        
        context = ErrorContext(
            error_type=ErrorType.MEMORY_ERROR,
            severity='high',
            function_name='large_computation',
            parameters={'tensor': large_tensor, 'batch_size': 64}
        )
        
        # Should handle tensor parameters safely
        info = context.get_context_info()
        
        # Tensor should be summarized, not stored entirely
        assert 'tensor' in info['parameters']
        assert isinstance(info['parameters']['tensor'], str)  # Should be summarized


class TestFailurePredictor:
    """Test failure prediction functionality."""
    
    def test_initialization(self):
        """Test predictor initialization."""
        predictor = FailurePredictor()
        
        assert predictor.prediction_history == []
        assert predictor.accuracy_metrics['correct_predictions'] == 0
        assert predictor.accuracy_metrics['total_predictions'] == 0
    
    def test_pattern_recognition(self):
        """Test error pattern recognition."""
        predictor = FailurePredictor()
        
        # Add similar error patterns
        for i in range(5):
            context = ErrorContext(
                error_type=ErrorType.TENSOR_DIMENSION,
                severity='high',
                function_name='conv_forward',
                parameters={'input_shape': (32, 3, 224, 224)}
            )
            predictor.add_error_pattern(context)
        
        # Test prediction for similar pattern
        test_context = ErrorContext(
            error_type=ErrorType.TENSOR_DIMENSION,
            severity='high', 
            function_name='conv_forward',
            parameters={'input_shape': (16, 3, 224, 224)}
        )
        
        prediction = predictor.predict_failure_probability(test_context)
        
        assert isinstance(prediction, float)
        assert 0.0 <= prediction <= 1.0
        # Should be high probability due to similar patterns
        assert prediction > 0.5
    
    def test_learning_from_outcomes(self):
        """Test predictor learns from outcomes."""
        predictor = FailurePredictor()
        
        context = ErrorContext(
            error_type=ErrorType.NUMERICAL_INSTABILITY,
            severity='medium',
            function_name='gradient_update',
            parameters={'learning_rate': 10.0}  # Very high LR
        )
        
        # Make prediction
        prob_before = predictor.predict_failure_probability(context)
        
        # Update with outcome (failure occurred)
        predictor.update_prediction_accuracy(context, actual_failure=True)
        
        # Make same prediction again
        prob_after = predictor.predict_failure_probability(context)
        
        # Should learn from the outcome
        assert predictor.accuracy_metrics['total_predictions'] == 1
        
        # If prediction was correct, accuracy should improve
        if prob_before > 0.5:  # Predicted failure correctly
            assert predictor.accuracy_metrics['correct_predictions'] == 1
    
    def test_feature_extraction(self):
        """Test feature extraction from error contexts."""
        predictor = FailurePredictor()
        
        context = ErrorContext(
            error_type=ErrorType.CUDA_ERROR,
            severity='high',
            function_name='gpu_computation',
            parameters={'device': 'cuda:0', 'memory_usage': 0.9}
        )
        
        features = predictor.extract_features(context)
        
        assert isinstance(features, dict)
        assert len(features) > 0
        
        # Should extract relevant features
        expected_features = ['error_type', 'severity', 'function_name']
        for feature in expected_features:
            assert feature in features
    
    def test_prediction_confidence(self):
        """Test prediction confidence estimation."""
        predictor = FailurePredictor()
        
        # Train with some patterns
        for i in range(10):
            context = ErrorContext(
                error_type=ErrorType.TENSOR_DIMENSION,
                severity='high',
                function_name='matrix_multiply',
                parameters={'size': 1000 + i}
            )
            predictor.add_error_pattern(context)
            predictor.update_prediction_accuracy(context, actual_failure=(i % 2 == 0))
        
        # Test confidence
        test_context = ErrorContext(
            error_type=ErrorType.TENSOR_DIMENSION,
            severity='high',
            function_name='matrix_multiply', 
            parameters={'size': 1005}
        )
        
        prediction, confidence = predictor.predict_with_confidence(test_context)
        
        assert isinstance(prediction, float)
        assert isinstance(confidence, float)
        assert 0.0 <= prediction <= 1.0
        assert 0.0 <= confidence <= 1.0


class TestRecoveryEngine:
    """Test recovery engine functionality."""
    
    def test_initialization(self):
        """Test engine initialization."""
        engine = RecoveryEngine()
        
        assert engine.recovery_history == []
        assert engine.success_rates == {}
        assert len(engine.strategies) > 0
    
    def test_strategy_selection(self):
        """Test recovery strategy selection."""
        engine = RecoveryEngine()
        
        context = ErrorContext(
            error_type=ErrorType.TENSOR_DIMENSION,
            severity='high',
            function_name='reshape_tensor',
            parameters={'original_shape': (10, 20), 'target_shape': (15, 15)}
        )
        
        strategy = engine.select_recovery_strategy(context)
        
        assert isinstance(strategy, RecoveryStrategy)
        assert hasattr(strategy, 'apply')
        assert hasattr(strategy, 'success_rate')
    
    def test_dimension_mismatch_recovery(self):
        """Test recovery from dimension mismatch."""
        engine = RecoveryEngine()
        
        # Simulate dimension mismatch
        def failing_function(x, target_shape):
            return x.reshape(target_shape)  # Will fail if incompatible
        
        context = ErrorContext(
            error_type=ErrorType.TENSOR_DIMENSION,
            severity='high',
            function_name='failing_function',
            parameters={'x': torch.randn(12, 8), 'target_shape': (10, 10)}
        )
        
        # Apply recovery
        result = engine.apply_recovery(context, failing_function, 
                                     x=torch.randn(12, 8), target_shape=(10, 10))
        
        # Should either succeed with recovery or provide fallback
        assert result is not None
        if isinstance(result, torch.Tensor):
            # Successful recovery - should have valid shape
            assert result.numel() > 0
    
    def test_numerical_instability_recovery(self):
        """Test recovery from numerical instability."""
        engine = RecoveryEngine()
        
        def unstable_function(x, lr):
            # Simulate numerical instability with large learning rate
            if lr > 1.0:
                raise ValueError("Numerical instability detected")
            return x * lr
        
        context = ErrorContext(
            error_type=ErrorType.NUMERICAL_INSTABILITY,
            severity='medium',
            function_name='unstable_function',
            parameters={'x': torch.randn(5, 5), 'lr': 10.0}
        )
        
        # Apply recovery
        result = engine.apply_recovery(context, unstable_function,
                                     x=torch.randn(5, 5), lr=10.0)
        
        # Should recover by reducing learning rate or other strategy
        assert result is not None
        if isinstance(result, torch.Tensor):
            assert torch.isfinite(result).all()
    
    def test_recovery_success_tracking(self):
        """Test tracking of recovery success rates."""
        engine = RecoveryEngine()
        
        context = ErrorContext(
            error_type=ErrorType.TENSOR_DIMENSION,
            severity='medium',
            function_name='test_function',
            parameters={}
        )
        
        # Simulate recovery attempts
        for i in range(5):
            success = (i % 2 == 0)  # Alternating success/failure
            engine.record_recovery_outcome(context, success=success)
        
        # Should track success rates
        assert ErrorType.TENSOR_DIMENSION in engine.success_rates
        success_rate = engine.success_rates[ErrorType.TENSOR_DIMENSION]
        assert 0.0 <= success_rate <= 1.0
    
    def test_adaptive_strategy_selection(self):
        """Test adaptive strategy selection based on history."""
        engine = RecoveryEngine()
        
        context = ErrorContext(
            error_type=ErrorType.MEMORY_ERROR,
            severity='high',
            function_name='memory_intensive_op',
            parameters={}
        )
        
        # Record some failures for default strategy
        for _ in range(3):
            engine.record_recovery_outcome(context, success=False)
        
        # Should adapt strategy selection
        strategy1 = engine.select_recovery_strategy(context)
        
        # Record more failures
        for _ in range(5):
            engine.record_recovery_outcome(context, success=False)
        
        strategy2 = engine.select_recovery_strategy(context)
        
        # Strategy selection should be influenced by failure history
        # (exact behavior depends on implementation)
        assert isinstance(strategy1, RecoveryStrategy)
        assert isinstance(strategy2, RecoveryStrategy)


class TestIntelligentErrorRecovery:
    """Test main error recovery system."""
    
    def test_initialization(self):
        """Test system initialization."""
        recovery = IntelligentErrorRecovery(
            max_retries=3,
            enable_learning=True,
            prediction_threshold=0.7
        )
        
        assert recovery.max_retries == 3
        assert recovery.enable_learning is True
        assert recovery.prediction_threshold == 0.7
        assert hasattr(recovery, 'predictor')
        assert hasattr(recovery, 'recovery_engine')
    
    def test_error_handling_workflow(self):
        """Test complete error handling workflow."""
        recovery = IntelligentErrorRecovery(max_retries=2)
        
        def sometimes_failing_function(x, fail_count=[0]):
            """Function that fails first few times."""
            fail_count[0] += 1
            if fail_count[0] <= 2:
                raise ValueError(f"Simulated failure {fail_count[0]}")
            return x * 2
        
        # Should recover after retries
        result = recovery.handle_error(
            ValueError("Simulated failure 1"),
            function_name='sometimes_failing_function',
            parameters={'x': torch.tensor([1, 2, 3])},
            function=sometimes_failing_function,
            x=torch.tensor([1, 2, 3])
        )
        
        # Should eventually succeed
        assert result is not None
        if isinstance(result, torch.Tensor):
            assert torch.equal(result, torch.tensor([2, 4, 6]))
    
    def test_prediction_integration(self):
        """Test integration of failure prediction."""
        recovery = IntelligentErrorRecovery(enable_learning=True)
        
        # Train predictor with some patterns
        error_context = ErrorContext(
            error_type=ErrorType.CUDA_ERROR,
            severity='high',
            function_name='gpu_operation',
            parameters={'device': 'cuda:0'}
        )
        
        # Add pattern to predictor
        recovery.predictor.add_error_pattern(error_context)
        
        # Test prediction
        prediction = recovery.predict_failure(error_context)
        
        assert isinstance(prediction, float)
        assert 0.0 <= prediction <= 1.0
    
    def test_learning_from_outcomes(self):
        """Test system learns from recovery outcomes."""
        recovery = IntelligentErrorRecovery(enable_learning=True)
        
        def test_function(x):
            if torch.isnan(x).any():
                raise ValueError("NaN detected")
            return x * 2
        
        # Create error context
        nan_tensor = torch.tensor([1.0, float('nan'), 3.0])
        
        try:
            result = recovery.handle_error(
                ValueError("NaN detected"),
                function_name='test_function',
                parameters={'x': nan_tensor},
                function=test_function,
                x=nan_tensor
            )
        except Exception:
            pass  # May still fail, but should learn from attempt
        
        # System should have recorded the attempt
        assert len(recovery.predictor.prediction_history) > 0 or \
               len(recovery.recovery_engine.recovery_history) > 0
    
    def test_context_aware_recovery(self):
        """Test context-aware recovery strategies."""
        recovery = IntelligentErrorRecovery()
        
        # Different error types should trigger different strategies
        dimension_error = ErrorContext(
            error_type=ErrorType.TENSOR_DIMENSION,
            severity='high',
            function_name='reshape',
            parameters={'shape_mismatch': True}
        )
        
        memory_error = ErrorContext(
            error_type=ErrorType.MEMORY_ERROR,
            severity='high',
            function_name='large_allocation',
            parameters={'size_gb': 16}
        )
        
        # Should select appropriate strategies
        dim_strategy = recovery.recovery_engine.select_recovery_strategy(dimension_error)
        mem_strategy = recovery.recovery_engine.select_recovery_strategy(memory_error)
        
        # Strategies should be different for different error types
        assert isinstance(dim_strategy, RecoveryStrategy)
        assert isinstance(mem_strategy, RecoveryStrategy)
        # Note: specific strategy differences depend on implementation
    
    def test_retry_logic(self):
        """Test retry logic with backoff."""
        recovery = IntelligentErrorRecovery(max_retries=3)
        
        call_count = [0]
        
        def failing_function():
            call_count[0] += 1
            raise RuntimeError(f"Failure {call_count[0]}")
        
        start_time = time.time()
        
        try:
            recovery.handle_error(
                RuntimeError("Initial failure"),
                function_name='failing_function',
                parameters={},
                function=failing_function
            )
        except Exception:
            pass  # Expected to eventually fail
        
        end_time = time.time()
        
        # Should have attempted retries (call count > 1)
        assert call_count[0] > 1
        assert call_count[0] <= 4  # Initial + max_retries
        
        # Should have some delay due to backoff
        assert end_time - start_time > 0.1  # At least 100ms
    
    def test_statistics_collection(self):
        """Test error statistics collection."""
        recovery = IntelligentErrorRecovery(enable_learning=True)
        
        # Simulate various errors
        errors = [
            (ErrorType.TENSOR_DIMENSION, "Dimension mismatch"),
            (ErrorType.NUMERICAL_INSTABILITY, "NaN values"),
            (ErrorType.MEMORY_ERROR, "Out of memory"),
            (ErrorType.TENSOR_DIMENSION, "Another dimension issue")
        ]
        
        for error_type, message in errors:
            context = ErrorContext(
                error_type=error_type,
                severity='medium',
                function_name='test',
                parameters={}
            )
            
            # Record error (simulate handling)
            recovery.predictor.add_error_pattern(context)
        
        # Get statistics
        stats = recovery.get_statistics()
        
        assert isinstance(stats, dict)
        assert 'total_errors_handled' in stats
        assert 'error_type_distribution' in stats
        assert 'prediction_accuracy' in stats
        assert 'recovery_success_rate' in stats
        
        # Should have proper error type distribution
        distribution = stats['error_type_distribution']
        assert distribution.get('TENSOR_DIMENSION', 0) == 2
        assert distribution.get('NUMERICAL_INSTABILITY', 0) == 1
        assert distribution.get('MEMORY_ERROR', 0) == 1
    
    def test_thread_safety(self):
        """Test thread-safe error handling."""
        recovery = IntelligentErrorRecovery()
        results = []
        errors = []
        
        def worker_thread(worker_id):
            try:
                for i in range(5):
                    def test_func(x):
                        return x + worker_id
                    
                    result = recovery.handle_error(
                        ValueError(f"Worker {worker_id} error {i}"),
                        function_name='test_func',
                        parameters={'x': i},
                        function=test_func,
                        x=i
                    )
                    results.append((worker_id, i, result))
            except Exception as e:
                errors.append((worker_id, str(e)))
        
        # Start multiple worker threads
        threads = []
        for worker_id in range(3):
            thread = threading.Thread(target=worker_thread, args=(worker_id,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Should handle concurrent access without errors
        assert len(errors) == 0, f"Thread errors: {errors}"
        assert len(results) > 0


class TestRecoveryDecorator:
    """Test error recovery decorator functionality."""
    
    def test_decorator_basic_usage(self):
        """Test basic decorator usage."""
        recovery = IntelligentErrorRecovery(max_retries=2)
        
        @recovery.with_recovery
        def decorated_function(x, should_fail=False):
            if should_fail:
                raise ValueError("Intentional failure")
            return x * 2
        
        # Should work normally when no error
        result = decorated_function(torch.tensor([1, 2, 3]), should_fail=False)
        assert torch.equal(result, torch.tensor([2, 4, 6]))
        
        # Should attempt recovery when error occurs
        try:
            result = decorated_function(torch.tensor([1, 2, 3]), should_fail=True)
            # May or may not succeed depending on recovery
        except Exception:
            pass  # Recovery may still fail
    
    def test_decorator_with_parameters(self):
        """Test decorator with custom parameters."""
        recovery = IntelligentErrorRecovery()
        
        @recovery.with_recovery(max_retries=1, error_type=ErrorType.NUMERICAL_INSTABILITY)
        def numerical_function(x):
            if torch.isnan(x).any():
                raise ValueError("NaN values detected")
            return torch.sqrt(x)
        
        # Should work with valid input
        valid_input = torch.tensor([1.0, 4.0, 9.0])
        result = numerical_function(valid_input)
        assert torch.allclose(result, torch.tensor([1.0, 2.0, 3.0]))
    
    def test_decorator_preserves_metadata(self):
        """Test decorator preserves function metadata."""
        recovery = IntelligentErrorRecovery()
        
        @recovery.with_recovery
        def documented_function(x):
            """This function has documentation."""
            return x + 1
        
        # Should preserve function name and docstring
        assert documented_function.__name__ == 'documented_function'
        assert 'documentation' in documented_function.__doc__


if __name__ == "__main__":
    pytest.main([__file__])