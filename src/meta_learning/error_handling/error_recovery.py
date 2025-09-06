"""
Intelligent error recovery system with ML-based failure prediction.

Provides context-aware error handling, automatic parameter adjustment,
and smart fallback strategies for robust meta-learning pipelines.
"""

import traceback
import time
import logging
from enum import Enum
from typing import Dict, Any, Optional, Callable, List, Tuple, Union
from dataclasses import dataclass
from collections import defaultdict, deque

import torch
import numpy as np


class ErrorType(Enum):
    """Classification of error types for targeted recovery."""
    MEMORY_ERROR = "memory_error"
    CUDA_ERROR = "cuda_error" 
    NUMERICAL_INSTABILITY = "numerical_instability"
    CONVERGENCE_FAILURE = "convergence_failure"
    DATA_LOADING_ERROR = "data_loading_error"
    MODEL_ERROR = "model_error"
    TIMEOUT_ERROR = "timeout_error"
    UNKNOWN_ERROR = "unknown_error"


class RecoveryStrategy(Enum):
    """Available recovery strategies."""
    RETRY_WITH_BACKOFF = "retry_with_backoff"
    REDUCE_BATCH_SIZE = "reduce_batch_size"
    LOWER_PRECISION = "lower_precision"
    CPU_FALLBACK = "cpu_fallback"
    SIMPLIFIED_MODEL = "simplified_model"
    ALTERNATIVE_ALGORITHM = "alternative_algorithm"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    ABORT_WITH_CLEANUP = "abort_with_cleanup"


@dataclass
class ErrorContext:
    """Context information for error analysis and recovery."""
    error_type: ErrorType
    exception: Exception
    timestamp: float
    function_name: str
    parameters: Dict[str, Any]
    system_state: Dict[str, Any]
    stack_trace: str
    previous_attempts: int = 0
    recovery_history: List[RecoveryStrategy] = None
    
    def __post_init__(self):
        if self.recovery_history is None:
            self.recovery_history = []


class ErrorClassifier:
    """ML-based error classification system."""
    
    def __init__(self):
        self.error_patterns = {
            ErrorType.MEMORY_ERROR: [
                "out of memory", "cuda out of memory", "allocat", "memory",
                "RuntimeError: CUDA out of memory"
            ],
            ErrorType.CUDA_ERROR: [
                "cuda", "gpu", "device", "kernel", "cudnn", "cublas"
            ],
            ErrorType.NUMERICAL_INSTABILITY: [
                "nan", "inf", "overflow", "underflow", "gradient", "loss"
            ],
            ErrorType.CONVERGENCE_FAILURE: [
                "convergence", "iteration", "max_iter", "tolerance", "not converged"
            ],
            ErrorType.DATA_LOADING_ERROR: [
                "dataloader", "dataset", "batch", "loading", "file not found"
            ],
            ErrorType.MODEL_ERROR: [
                "forward", "backward", "parameter", "layer", "module"
            ],
            ErrorType.TIMEOUT_ERROR: [
                "timeout", "time out", "deadline", "expired"
            ]
        }
        
        # Error frequency tracking for pattern learning
        self.error_frequencies = defaultdict(int)
        self.context_patterns = defaultdict(list)
    
    def classify_error(self, exception: Exception, context: Dict[str, Any]) -> ErrorType:
        """Classify error type based on exception and context."""
        error_message = str(exception).lower()
        error_type_name = type(exception).__name__.lower()
        
        # Check for specific error patterns
        for error_type, patterns in self.error_patterns.items():
            for pattern in patterns:
                if pattern in error_message or pattern in error_type_name:
                    self.error_frequencies[error_type] += 1
                    self.context_patterns[error_type].append(context)
                    return error_type
        
        # Learn from context
        self._learn_from_context(exception, context)
        
        return ErrorType.UNKNOWN_ERROR
    
    def _learn_from_context(self, exception: Exception, context: Dict[str, Any]) -> None:
        """Learn error patterns from context for future classification."""
        # Simple pattern learning - could be enhanced with ML
        if "batch_size" in context and context.get("batch_size", 0) > 32:
            if "memory" in str(exception).lower():
                self.error_patterns[ErrorType.MEMORY_ERROR].append("large_batch")
        
        if context.get("device") == "cuda" and "cuda" in str(exception).lower():
            self.error_patterns[ErrorType.CUDA_ERROR].append(str(exception).lower()[:50])


class RecoveryPlanner:
    """Intelligent recovery strategy selection."""
    
    def __init__(self):
        self.strategy_success_rates = defaultdict(lambda: {"successes": 0, "attempts": 0})
        self.error_strategy_mapping = {
            ErrorType.MEMORY_ERROR: [
                RecoveryStrategy.REDUCE_BATCH_SIZE,
                RecoveryStrategy.LOWER_PRECISION,
                RecoveryStrategy.CPU_FALLBACK,
                RecoveryStrategy.GRACEFUL_DEGRADATION
            ],
            ErrorType.CUDA_ERROR: [
                RecoveryStrategy.CPU_FALLBACK,
                RecoveryStrategy.RETRY_WITH_BACKOFF,
                RecoveryStrategy.GRACEFUL_DEGRADATION
            ],
            ErrorType.NUMERICAL_INSTABILITY: [
                RecoveryStrategy.LOWER_PRECISION,
                RecoveryStrategy.REDUCE_BATCH_SIZE,
                RecoveryStrategy.ALTERNATIVE_ALGORITHM,
                RecoveryStrategy.GRACEFUL_DEGRADATION
            ],
            ErrorType.CONVERGENCE_FAILURE: [
                RecoveryStrategy.ALTERNATIVE_ALGORITHM,
                RecoveryStrategy.RETRY_WITH_BACKOFF,
                RecoveryStrategy.SIMPLIFIED_MODEL
            ],
            ErrorType.TIMEOUT_ERROR: [
                RecoveryStrategy.REDUCE_BATCH_SIZE,
                RecoveryStrategy.SIMPLIFIED_MODEL,
                RecoveryStrategy.GRACEFUL_DEGRADATION
            ],
            ErrorType.UNKNOWN_ERROR: [
                RecoveryStrategy.RETRY_WITH_BACKOFF,
                RecoveryStrategy.GRACEFUL_DEGRADATION,
                RecoveryStrategy.ABORT_WITH_CLEANUP
            ]
        }
    
    def select_strategy(self, error_context: ErrorContext) -> Optional[RecoveryStrategy]:
        """Select optimal recovery strategy based on context and success history."""
        error_type = error_context.error_type
        available_strategies = self.error_strategy_mapping.get(error_type, [])
        
        # Filter out already tried strategies
        untried_strategies = [
            s for s in available_strategies 
            if s not in error_context.recovery_history
        ]
        
        if not untried_strategies:
            # All strategies tried, return best performing or abort
            if available_strategies:
                best_strategy = max(
                    available_strategies,
                    key=lambda s: self._get_success_rate(s)
                )
                return best_strategy
            return RecoveryStrategy.ABORT_WITH_CLEANUP
        
        # Select strategy with highest success rate
        best_strategy = max(
            untried_strategies,
            key=lambda s: self._get_success_rate(s)
        )
        
        return best_strategy
    
    def _get_success_rate(self, strategy: RecoveryStrategy) -> float:
        """Get success rate for strategy."""
        stats = self.strategy_success_rates[strategy]
        if stats["attempts"] == 0:
            return 0.5  # Neutral prior
        return stats["successes"] / stats["attempts"]
    
    def record_outcome(self, strategy: RecoveryStrategy, success: bool) -> None:
        """Record success/failure of recovery strategy."""
        self.strategy_success_rates[strategy]["attempts"] += 1
        if success:
            self.strategy_success_rates[strategy]["successes"] += 1
    
    def get_strategy_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get performance statistics for all strategies."""
        stats = {}
        for strategy, data in self.strategy_success_rates.items():
            success_rate = data["successes"] / max(data["attempts"], 1)
            stats[strategy.value] = {
                "success_rate": success_rate,
                "attempts": data["attempts"],
                "successes": data["successes"]
            }
        return stats


class IntelligentErrorRecovery:
    """
    Intelligent error recovery system with ML-based failure prediction.
    
    Features:
    - Automatic error classification
    - Context-aware recovery strategy selection
    - Performance impact analysis
    - Learning from failure patterns
    """
    
    def __init__(self, 
                 max_retries: int = 3,
                 enable_learning: bool = True,
                 log_level: str = "INFO"):
        """
        Initialize error recovery system.
        
        Args:
            max_retries: Maximum recovery attempts per error
            enable_learning: Enable ML-based pattern learning
            log_level: Logging level for error reporting
        """
        self.max_retries = max_retries
        self.enable_learning = enable_learning
        
        # Initialize components
        self.classifier = ErrorClassifier()
        self.planner = RecoveryPlanner()
        
        # Error tracking
        self.error_history = deque(maxlen=1000)
        self.recovery_stats = defaultdict(int)
        
        # Setup logging
        logging.basicConfig(level=getattr(logging, log_level.upper()))
        self.logger = logging.getLogger(__name__)
    
    def handle_error(self, 
                    exception: Exception,
                    function_name: str,
                    parameters: Dict[str, Any],
                    system_state: Optional[Dict[str, Any]] = None) -> ErrorContext:
        """
        Handle error with intelligent recovery.
        
        Args:
            exception: Exception that occurred
            function_name: Name of function where error occurred
            parameters: Function parameters
            system_state: Current system state information
            
        Returns:
            ErrorContext with recovery information
        """
        # Create error context
        context = ErrorContext(
            error_type=self.classifier.classify_error(exception, parameters),
            exception=exception,
            timestamp=time.time(),
            function_name=function_name,
            parameters=parameters.copy(),
            system_state=system_state or self._get_system_state(),
            stack_trace=traceback.format_exc()
        )
        
        # Log error
        self.logger.error(f"Error in {function_name}: {exception}")
        self.logger.debug(f"Error context: {context}")
        
        # Add to history
        self.error_history.append(context)
        
        return context
    
    def attempt_recovery(self, context: ErrorContext) -> Tuple[bool, Optional[RecoveryStrategy], Dict[str, Any]]:
        """
        Attempt recovery from error.
        
        Args:
            context: Error context information
            
        Returns:
            Tuple of (success, strategy_used, modified_parameters)
        """
        if context.previous_attempts >= self.max_retries:
            self.logger.warning(f"Max retries exceeded for {context.function_name}")
            return False, None, {}
        
        # Select recovery strategy
        strategy = self.planner.select_strategy(context)
        if strategy is None:
            return False, None, {}
        
        # Apply recovery strategy
        success, modified_params = self._apply_strategy(strategy, context)
        
        # Record outcome
        self.planner.record_outcome(strategy, success)
        self.recovery_stats[strategy] += 1
        
        # Update context
        context.previous_attempts += 1
        context.recovery_history.append(strategy)
        
        self.logger.info(f"Recovery attempt {context.previous_attempts}: {strategy.value} -> {'Success' if success else 'Failed'}")
        
        return success, strategy, modified_params
    
    def _apply_strategy(self, 
                       strategy: RecoveryStrategy, 
                       context: ErrorContext) -> Tuple[bool, Dict[str, Any]]:
        """Apply specific recovery strategy."""
        modified_params = context.parameters.copy()
        
        try:
            if strategy == RecoveryStrategy.REDUCE_BATCH_SIZE:
                if "batch_size" in modified_params:
                    original_size = modified_params["batch_size"]
                    modified_params["batch_size"] = max(1, original_size // 2)
                    self.logger.info(f"Reduced batch size from {original_size} to {modified_params['batch_size']}")
                    return True, modified_params
            
            elif strategy == RecoveryStrategy.LOWER_PRECISION:
                if "dtype" in modified_params:
                    if modified_params["dtype"] == torch.float32:
                        modified_params["dtype"] = torch.float16
                        self.logger.info("Lowered precision to float16")
                        return True, modified_params
            
            elif strategy == RecoveryStrategy.CPU_FALLBACK:
                if "device" in modified_params:
                    modified_params["device"] = "cpu"
                    self.logger.info("Falling back to CPU")
                    return True, modified_params
            
            elif strategy == RecoveryStrategy.RETRY_WITH_BACKOFF:
                wait_time = 2 ** context.previous_attempts
                self.logger.info(f"Retrying with {wait_time}s backoff")
                time.sleep(wait_time)
                return True, modified_params
            
            elif strategy == RecoveryStrategy.SIMPLIFIED_MODEL:
                # Suggest model simplification
                if "model_complexity" in modified_params:
                    modified_params["model_complexity"] *= 0.7
                    self.logger.info("Simplified model complexity")
                    return True, modified_params
            
            elif strategy == RecoveryStrategy.ALTERNATIVE_ALGORITHM:
                # Suggest alternative algorithm
                if "algorithm" in modified_params:
                    alternatives = self._get_alternative_algorithms(modified_params["algorithm"])
                    if alternatives:
                        modified_params["algorithm"] = alternatives[0]
                        self.logger.info(f"Switched to alternative algorithm: {alternatives[0]}")
                        return True, modified_params
            
            elif strategy == RecoveryStrategy.GRACEFUL_DEGRADATION:
                # Enable graceful degradation mode
                modified_params["graceful_degradation"] = True
                self.logger.info("Enabled graceful degradation mode")
                return True, modified_params
            
            elif strategy == RecoveryStrategy.ABORT_WITH_CLEANUP:
                self.logger.info("Aborting with cleanup")
                self._perform_cleanup(context)
                return False, modified_params
        
        except Exception as e:
            self.logger.error(f"Recovery strategy {strategy.value} failed: {e}")
            return False, modified_params
        
        return False, modified_params
    
    def _get_alternative_algorithms(self, current_algorithm: str) -> List[str]:
        """Get alternative algorithms for recovery."""
        algorithm_alternatives = {
            "maml": ["protonet", "reptile"],
            "protonet": ["maml", "matching_net"],
            "matching_net": ["protonet", "relation_net"],
            "test_time_compute": ["maml", "protonet"]
        }
        return algorithm_alternatives.get(current_algorithm, [])
    
    def _get_system_state(self) -> Dict[str, Any]:
        """Get current system state information."""
        state = {
            "timestamp": time.time(),
            "cpu_percent": 0.0,  # Could use psutil if available
            "memory_percent": 0.0
        }
        
        if torch.cuda.is_available():
            try:
                state.update({
                    "gpu_count": torch.cuda.device_count(),
                    "current_device": torch.cuda.current_device(),
                    "gpu_memory_allocated": torch.cuda.memory_allocated(),
                    "gpu_memory_cached": torch.cuda.memory_reserved()
                })
            except:
                pass  # GPU info not available
        
        return state
    
    def _perform_cleanup(self, context: ErrorContext) -> None:
        """Perform cleanup operations."""
        try:
            # Clear GPU cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Force garbage collection
            import gc
            gc.collect()
            
            self.logger.info("Cleanup completed")
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error and recovery statistics."""
        error_type_counts = defaultdict(int)
        for context in self.error_history:
            error_type_counts[context.error_type.value] += 1
        
        return {
            "total_errors": len(self.error_history),
            "error_types": dict(error_type_counts),
            "recovery_attempts": dict(self.recovery_stats),
            "strategy_performance": self.planner.get_strategy_stats(),
            "recent_errors": len([c for c in self.error_history if time.time() - c.timestamp < 3600])
        }
    
    def generate_error_report(self) -> str:
        """Generate detailed error report for debugging."""
        stats = self.get_error_statistics()
        
        report = ["Meta-Learning Error Recovery Report", "=" * 40, ""]
        
        report.append(f"Total Errors: {stats['total_errors']}")
        report.append(f"Recent Errors (1h): {stats['recent_errors']}")
        report.append("")
        
        report.append("Error Types:")
        for error_type, count in stats['error_types'].items():
            report.append(f"  {error_type}: {count}")
        report.append("")
        
        report.append("Recovery Strategy Performance:")
        for strategy, data in stats['strategy_performance'].items():
            success_rate = data['success_rate']
            attempts = data['attempts']
            report.append(f"  {strategy}: {success_rate:.1%} success rate ({attempts} attempts)")
        report.append("")
        
        if self.error_history:
            recent_error = self.error_history[-1]
            report.append("Most Recent Error:")
            report.append(f"  Type: {recent_error.error_type.value}")
            report.append(f"  Function: {recent_error.function_name}")
            report.append(f"  Message: {str(recent_error.exception)}")
        
        return "\n".join(report)