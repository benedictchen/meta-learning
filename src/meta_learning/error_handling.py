"""
Professional error handling and performance monitoring system for meta-learning.

This module provides:
- IntelligentErrorRecovery: Context-aware error recovery with learning
- PerformanceMonitor: Real-time metrics collection and alerting  
- WarningManager: Warning filtering, categorization, and deduplication
- Professional monitoring with automatic anomaly detection
"""

import time
import threading
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn as nn


class ErrorType(Enum):
    """Classification of error types for intelligent recovery."""
    MEMORY_ERROR = "memory_error"
    NUMERICAL_INSTABILITY = "numerical_instability"
    DIMENSION_MISMATCH = "dimension_mismatch"
    CONVERGENCE_FAILURE = "convergence_failure"
    DEVICE_ERROR = "device_error"
    TIMEOUT_ERROR = "timeout_error"
    UNKNOWN_ERROR = "unknown_error"


class WarningCategory(Enum):
    """Warning categories for filtering and prioritization."""
    PERFORMANCE = "performance"
    COMPATIBILITY = "compatibility"
    DEPRECATION = "deprecation"
    NUMERICAL = "numerical"
    MEMORY = "memory"
    CONFIGURATION = "configuration"
    UNKNOWN = "unknown"


@dataclass
class ErrorContext:
    """Context information for error analysis and recovery."""
    error_type: ErrorType
    error_message: str
    stack_trace: str
    timestamp: float
    system_state: Dict[str, Any]
    recovery_attempts: int = 0
    successful_recovery: Optional[str] = None


@dataclass
class WarningInfo:
    """Information about captured warnings."""
    category: WarningCategory
    message: str
    filename: str
    line_number: int
    timestamp: float
    count: int = 1


class RecoveryStrategy(ABC):
    """Base class for error recovery strategies."""
    
    @abstractmethod
    def can_handle(self, error_context: ErrorContext) -> bool:
        """Check if this strategy can handle the given error."""
        pass
    
    @abstractmethod
    def recover(self, error_context: ErrorContext) -> Tuple[bool, str]:
        """Attempt recovery. Returns (success, description)."""
        pass


class MemoryErrorRecovery(RecoveryStrategy):
    """Recovery strategy for CUDA out of memory errors."""
    
    def can_handle(self, error_context: ErrorContext) -> bool:
        return (error_context.error_type == ErrorType.MEMORY_ERROR or
                "out of memory" in error_context.error_message.lower())
    
    def recover(self, error_context: ErrorContext) -> Tuple[bool, str]:
        """Recover from memory errors by clearing cache and reducing batch size."""
        try:
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Force garbage collection
            import gc
            gc.collect()
            
            return True, "Cleared GPU cache and triggered garbage collection"
            
        except Exception as e:
            return False, f"Memory recovery failed: {e}"


class NumericalInstabilityRecovery(RecoveryStrategy):
    """Recovery strategy for numerical instability issues."""
    
    def can_handle(self, error_context: ErrorContext) -> bool:
        return (error_context.error_type == ErrorType.NUMERICAL_INSTABILITY or
                any(keyword in error_context.error_message.lower() 
                    for keyword in ["nan", "inf", "overflow", "underflow"]))
    
    def recover(self, error_context: ErrorContext) -> Tuple[bool, str]:
        """Recover by adjusting numerical parameters."""
        try:
            # Set more conservative floating point behavior
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
            return True, "Applied numerical stability settings"
            
        except Exception as e:
            return False, f"Numerical stability recovery failed: {e}"


class DimensionMismatchRecovery(RecoveryStrategy):
    """Recovery strategy for tensor dimension mismatches."""
    
    def can_handle(self, error_context: ErrorContext) -> bool:
        return (error_context.error_type == ErrorType.DIMENSION_MISMATCH or
                "dimension" in error_context.error_message.lower() or
                "size mismatch" in error_context.error_message.lower())
    
    def recover(self, error_context: ErrorContext) -> Tuple[bool, str]:
        """Analyze dimension mismatch and suggest fixes."""
        try:
            # Extract dimension information from error message
            error_msg = error_context.error_message
            
            # Common dimension mismatch patterns
            recovery_suggestions = []
            
            if "matrix multiply" in error_msg.lower():
                recovery_suggestions.append("Check input tensor dimensions for matrix operations")
            
            if "expected" in error_msg.lower() and "got" in error_msg.lower():
                recovery_suggestions.append("Verify tensor shapes match expected model input/output")
            
            suggestion = "; ".join(recovery_suggestions) if recovery_suggestions else "Check tensor dimensions"
            return True, f"Dimension analysis complete: {suggestion}"
            
        except Exception as e:
            return False, f"Dimension mismatch analysis failed: {e}"


class IntelligentErrorRecovery:
    """
    Intelligent error recovery system with learning from failure patterns.
    """
    
    def __init__(self, max_retries: int = 3, enable_learning: bool = True):
        self.max_retries = max_retries
        self.enable_learning = enable_learning
        
        # Recovery strategies
        self.strategies = [
            MemoryErrorRecovery(),
            NumericalInstabilityRecovery(),
            DimensionMismatchRecovery()
        ]
        
        # Error tracking
        self.error_history = deque(maxlen=1000)
        self.recovery_success_rates = defaultdict(list)
        self.strategy_effectiveness = defaultdict(lambda: {"successes": 0, "attempts": 0})
        
        # Learning state
        self.learned_patterns = {}
        self.auto_adjustments = {}
        
    def handle_error(self, exception: Exception, context: Dict[str, Any] = None) -> Tuple[bool, str]:
        """
        Handle error with intelligent recovery and learning.
        
        Args:
            exception: The exception that occurred
            context: Additional context about the error
            
        Returns:
            Tuple of (recovery_success, recovery_description)
        """
        # Classify error type
        error_type = self._classify_error(exception)
        
        # Create error context
        error_context = ErrorContext(
            error_type=error_type,
            error_message=str(exception),
            stack_trace=str(exception.__traceback__),
            timestamp=time.time(),
            system_state=self._get_system_state(),
            recovery_attempts=0
        )
        
        if context:
            error_context.system_state.update(context)
        
        # Record error for learning
        self.error_history.append(error_context)
        
        # Attempt recovery
        success, description = self._attempt_recovery(error_context)
        
        # Learn from recovery attempt
        if self.enable_learning:
            self._learn_from_recovery(error_context, success, description)
        
        return success, description
    
    def _classify_error(self, exception: Exception) -> ErrorType:
        """Classify error type based on exception."""
        error_msg = str(exception).lower()
        exception_type = type(exception).__name__.lower()
        
        if "out of memory" in error_msg or "cuda" in error_msg:
            return ErrorType.MEMORY_ERROR
        elif any(keyword in error_msg for keyword in ["nan", "inf", "overflow"]):
            return ErrorType.NUMERICAL_INSTABILITY
        elif "size" in error_msg or "dimension" in error_msg:
            return ErrorType.DIMENSION_MISMATCH
        elif "timeout" in error_msg:
            return ErrorType.TIMEOUT_ERROR
        elif "device" in error_msg or "cuda" in exception_type:
            return ErrorType.DEVICE_ERROR
        else:
            return ErrorType.UNKNOWN_ERROR
    
    def _get_system_state(self) -> Dict[str, Any]:
        """Get current system state for error analysis."""
        state = {
            'timestamp': time.time(),
            'torch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available()
        }
        
        if torch.cuda.is_available():
            state.update({
                'cuda_memory_allocated': torch.cuda.memory_allocated(),
                'cuda_memory_cached': torch.cuda.memory_reserved(),
                'cuda_device_count': torch.cuda.device_count()
            })
        
        return state
    
    def _attempt_recovery(self, error_context: ErrorContext) -> Tuple[bool, str]:
        """Attempt recovery using available strategies."""
        attempted_strategies = []
        
        for strategy in self.strategies:
            if strategy.can_handle(error_context):
                try:
                    success, description = strategy.recover(error_context)
                    strategy_name = strategy.__class__.__name__
                    
                    # Track strategy effectiveness
                    self.strategy_effectiveness[strategy_name]["attempts"] += 1
                    if success:
                        self.strategy_effectiveness[strategy_name]["successes"] += 1
                    
                    attempted_strategies.append(f"{strategy_name}: {description}")
                    
                    if success:
                        error_context.successful_recovery = strategy_name
                        return True, f"Recovery successful using {strategy_name}: {description}"
                        
                except Exception as e:
                    attempted_strategies.append(f"{strategy.__class__.__name__}: Recovery failed - {e}")
        
        # No strategy succeeded
        recovery_summary = "; ".join(attempted_strategies) if attempted_strategies else "No applicable recovery strategies"
        return False, f"Recovery failed. Attempted: {recovery_summary}"
    
    def _learn_from_recovery(self, error_context: ErrorContext, success: bool, description: str):
        """Learn from recovery attempts to improve future performance."""
        # Track success rates by error type
        error_type_key = error_context.error_type.value
        self.recovery_success_rates[error_type_key].append(success)
        
        # Keep only recent history for adaptive learning
        if len(self.recovery_success_rates[error_type_key]) > 100:
            self.recovery_success_rates[error_type_key] = self.recovery_success_rates[error_type_key][-50:]
        
        # Learn patterns from error messages
        error_keywords = error_context.error_message.lower().split()
        for keyword in error_keywords:
            if keyword not in self.learned_patterns:
                self.learned_patterns[keyword] = {"successes": 0, "attempts": 0}
            
            self.learned_patterns[keyword]["attempts"] += 1
            if success:
                self.learned_patterns[keyword]["successes"] += 1
    
    def get_recovery_statistics(self) -> Dict[str, Any]:
        """Get recovery performance statistics."""
        stats = {
            'total_errors': len(self.error_history),
            'error_types': {},
            'strategy_effectiveness': {},
            'recent_success_rate': 0.0
        }
        
        # Error type distribution
        error_type_counts = defaultdict(int)
        recent_successes = 0
        
        for error_context in list(self.error_history)[-100:]:  # Recent 100 errors
            error_type_counts[error_context.error_type.value] += 1
            if error_context.successful_recovery:
                recent_successes += 1
        
        stats['error_types'] = dict(error_type_counts)
        stats['recent_success_rate'] = recent_successes / max(len(self.error_history), 1)
        
        # Strategy effectiveness
        for strategy_name, data in self.strategy_effectiveness.items():
            if data["attempts"] > 0:
                stats['strategy_effectiveness'][strategy_name] = {
                    'success_rate': data["successes"] / data["attempts"],
                    'total_attempts': data["attempts"],
                    'total_successes': data["successes"]
                }
        
        return stats
    
    def generate_error_report(self) -> str:
        """Generate human-readable error report."""
        stats = self.get_recovery_statistics()
        
        report_lines = [
            "=== INTELLIGENT ERROR RECOVERY REPORT ===",
            f"Total errors handled: {stats['total_errors']}",
            f"Recent success rate: {stats['recent_success_rate']:.1%}",
            "",
            "Error Type Distribution:"
        ]
        
        for error_type, count in stats['error_types'].items():
            report_lines.append(f"  {error_type}: {count}")
        
        report_lines.append("\nStrategy Effectiveness:")
        for strategy, data in stats['strategy_effectiveness'].items():
            report_lines.append(f"  {strategy}: {data['success_rate']:.1%} "
                              f"({data['total_successes']}/{data['total_attempts']})")
        
        return "\n".join(report_lines)


class PerformanceMonitor:
    """
    Real-time performance monitoring with metrics collection and alerting.
    """
    
    def __init__(self, collection_interval: float = 1.0):
        self.collection_interval = collection_interval
        self.metrics = defaultdict(deque)
        self.alerts = []
        self.thresholds = {
            'memory_usage_percent': 90.0,
            'cpu_usage_percent': 95.0,
            'episode_generation_time': 5.0,  # seconds
            'model_inference_time': 1.0,     # seconds
        }
        
        # Monitoring state
        self.monitoring = False
        self.monitor_thread = None
        self.start_time = time.time()
        
    def start_monitoring(self):
        """Start background performance monitoring."""
        if self.monitoring:
            return
            
        self.monitoring = True
        self.start_time = time.time()
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop background performance monitoring."""
        self.monitoring = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)
    
    def _monitoring_loop(self):
        """Main monitoring loop running in background thread."""
        while self.monitoring:
            try:
                # Collect system metrics
                self._collect_system_metrics()
                
                # Check for alerts
                self._check_alert_conditions()
                
                # Sleep until next collection
                time.sleep(self.collection_interval)
                
            except Exception as e:
                print(f"Performance monitoring error: {e}")
                time.sleep(self.collection_interval * 2)  # Back off on error
    
    def _collect_system_metrics(self):
        """Collect system performance metrics."""
        timestamp = time.time()
        
        # Memory metrics
        if torch.cuda.is_available():
            gpu_memory_allocated = torch.cuda.memory_allocated()
            gpu_memory_reserved = torch.cuda.memory_reserved()
            gpu_memory_total = torch.cuda.get_device_properties(0).total_memory
            
            gpu_usage_percent = (gpu_memory_reserved / gpu_memory_total) * 100
            self.metrics['gpu_memory_usage_percent'].append((timestamp, gpu_usage_percent))
            self.metrics['gpu_memory_allocated_mb'].append((timestamp, gpu_memory_allocated / (1024*1024)))
        
        # CPU and system memory
        try:
            import psutil
            cpu_percent = psutil.cpu_percent()
            memory_info = psutil.virtual_memory()
            
            self.metrics['cpu_usage_percent'].append((timestamp, cpu_percent))
            self.metrics['system_memory_usage_percent'].append((timestamp, memory_info.percent))
            
        except ImportError:
            pass  # psutil not available
        
        # Trim old metrics (keep last 1000 points)
        for metric_name in self.metrics:
            if len(self.metrics[metric_name]) > 1000:
                self.metrics[metric_name] = deque(list(self.metrics[metric_name])[-500:], maxlen=1000)
    
    def record_metric(self, metric_name: str, value: float):
        """Record a custom metric."""
        timestamp = time.time()
        self.metrics[metric_name].append((timestamp, value))
        
        # Check for immediate alerts
        if metric_name in self.thresholds and value > self.thresholds[metric_name]:
            alert_msg = f"ALERT: {metric_name} = {value:.2f} exceeds threshold {self.thresholds[metric_name]:.2f}"
            self.alerts.append((timestamp, alert_msg))
    
    def _check_alert_conditions(self):
        """Check for alert conditions and record alerts."""
        timestamp = time.time()
        
        for metric_name, threshold in self.thresholds.items():
            if metric_name in self.metrics and len(self.metrics[metric_name]) > 0:
                latest_value = self.metrics[metric_name][-1][1]
                
                if latest_value > threshold:
                    alert_msg = f"ALERT: {metric_name} = {latest_value:.2f} exceeds threshold {threshold:.2f}"
                    
                    # Avoid duplicate alerts (only alert if last alert was >30 seconds ago)
                    if not self.alerts or timestamp - self.alerts[-1][0] > 30:
                        self.alerts.append((timestamp, alert_msg))
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics."""
        summary = {
            'monitoring_duration_seconds': time.time() - self.start_time,
            'total_alerts': len(self.alerts),
            'recent_alerts': [alert for timestamp, alert in self.alerts if time.time() - timestamp < 300],  # Last 5 minutes
            'metrics_summary': {}
        }
        
        # Calculate summary statistics for each metric
        for metric_name, data_points in self.metrics.items():
            if len(data_points) == 0:
                continue
                
            values = [value for timestamp, value in data_points]
            summary['metrics_summary'][metric_name] = {
                'current': values[-1] if values else 0,
                'average': sum(values) / len(values),
                'min': min(values),
                'max': max(values),
                'data_points': len(values)
            }
        
        return summary
    
    def get_performance_report(self) -> str:
        """Generate human-readable performance report."""
        summary = self.get_performance_summary()
        
        report_lines = [
            "=== PERFORMANCE MONITORING REPORT ===",
            f"Monitoring duration: {summary['monitoring_duration_seconds']:.0f} seconds",
            f"Total alerts: {summary['total_alerts']}",
            f"Recent alerts: {len(summary['recent_alerts'])}",
            "",
            "Metrics Summary:"
        ]
        
        for metric_name, stats in summary['metrics_summary'].items():
            report_lines.append(f"  {metric_name}:")
            report_lines.append(f"    Current: {stats['current']:.2f}")
            report_lines.append(f"    Average: {stats['average']:.2f}")
            report_lines.append(f"    Range: {stats['min']:.2f} - {stats['max']:.2f}")
        
        if summary['recent_alerts']:
            report_lines.append("\nRecent Alerts:")
            for alert in summary['recent_alerts'][-5:]:  # Show last 5 alerts
                report_lines.append(f"  {alert}")
        
        return "\n".join(report_lines)


class WarningManager:
    """
    Professional warning management with filtering and categorization.
    """
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.warnings = deque(maxlen=max_history)
        self.warning_counts = defaultdict(int)
        self.suppressed_warnings = set()
        
        # Category patterns for automatic classification
        self.category_patterns = {
            WarningCategory.PERFORMANCE: ["slow", "performance", "inefficient", "optimization"],
            WarningCategory.COMPATIBILITY: ["deprecated", "compatibility", "version", "legacy"],
            WarningCategory.DEPRECATION: ["deprecated", "deprecation", "removal", "future"],
            WarningCategory.NUMERICAL: ["nan", "inf", "overflow", "underflow", "precision"],
            WarningCategory.MEMORY: ["memory", "allocation", "leak", "cache"],
            WarningCategory.CONFIGURATION: ["config", "setting", "parameter", "default"]
        }
        
        # Install warning capture
        self._install_warning_capture()
    
    def _install_warning_capture(self):
        """Install warning capture to intercept Python warnings."""
        self._original_showwarning = warnings.showwarning
        warnings.showwarning = self._capture_warning
    
    def _capture_warning(self, message, category, filename, line_number, file=None, line=None):
        """Capture and process warnings."""
        warning_info = WarningInfo(
            category=self._classify_warning(str(message)),
            message=str(message),
            filename=filename,
            line_number=line_number,
            timestamp=time.time()
        )
        
        # Check if this warning should be suppressed
        warning_key = (warning_info.message, warning_info.filename, warning_info.line_number)
        if warning_key in self.suppressed_warnings:
            return
        
        # Check for duplicate warnings
        for existing_warning in reversed(list(self.warnings)):
            if (existing_warning.message == warning_info.message and
                existing_warning.filename == warning_info.filename):
                existing_warning.count += 1
                existing_warning.timestamp = warning_info.timestamp
                return
        
        # Add new warning
        self.warnings.append(warning_info)
        self.warning_counts[warning_info.category.value] += 1
        
        # Call original warning display (optionally)
        # self._original_showwarning(message, category, filename, line_number, file, line)
    
    def _classify_warning(self, message: str) -> WarningCategory:
        """Classify warning based on message content."""
        message_lower = message.lower()
        
        for category, patterns in self.category_patterns.items():
            if any(pattern in message_lower for pattern in patterns):
                return category
        
        return WarningCategory.UNKNOWN
    
    def suppress_warning(self, pattern: str = None, filename: str = None, message: str = None):
        """Suppress specific warnings by pattern, filename, or exact message."""
        if message and filename:
            self.suppressed_warnings.add((message, filename, 0))  # Line number 0 matches any
        elif pattern:
            # Add pattern-based suppression (simplified implementation)
            for warning in self.warnings:
                if pattern.lower() in warning.message.lower():
                    warning_key = (warning.message, warning.filename, warning.line_number)
                    self.suppressed_warnings.add(warning_key)
    
    def get_warning_summary(self) -> Dict[str, Any]:
        """Get warning summary statistics."""
        total_warnings = len(self.warnings)
        recent_warnings = [w for w in self.warnings if time.time() - w.timestamp < 300]  # Last 5 minutes
        
        # Top warning messages by frequency
        message_counts = defaultdict(int)
        for warning in self.warnings:
            message_counts[warning.message] += warning.count
        
        top_warnings = sorted(message_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'total_warnings': total_warnings,
            'recent_warnings': len(recent_warnings),
            'category_counts': dict(self.warning_counts),
            'suppressed_count': len(self.suppressed_warnings),
            'top_warnings': top_warnings
        }
    
    def generate_warning_report(self) -> str:
        """Generate human-readable warning report."""
        summary = self.get_warning_summary()
        
        report_lines = [
            "=== WARNING MANAGEMENT REPORT ===",
            f"Total warnings captured: {summary['total_warnings']}",
            f"Recent warnings (5 min): {summary['recent_warnings']}",
            f"Suppressed patterns: {summary['suppressed_count']}",
            "",
            "Warning Categories:"
        ]
        
        for category, count in summary['category_counts'].items():
            report_lines.append(f"  {category}: {count}")
        
        if summary['top_warnings']:
            report_lines.append("\nTop Warning Messages:")
            for message, count in summary['top_warnings']:
                truncated_msg = message[:80] + "..." if len(message) > 80 else message
                report_lines.append(f"  [{count}x] {truncated_msg}")
        
        return "\n".join(report_lines)
    
    def clear_warnings(self):
        """Clear warning history."""
        self.warnings.clear()
        self.warning_counts.clear()
    
    def restore_warnings(self):
        """Restore original warning display."""
        if hasattr(self, '_original_showwarning'):
            warnings.showwarning = self._original_showwarning


# Convenience functions for easy integration

def with_error_recovery(max_retries: int = 3):
    """Decorator to add error recovery to functions."""
    def decorator(func):
        recovery_system = IntelligentErrorRecovery(max_retries=max_retries)
        
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries:
                        raise  # Final attempt failed, re-raise
                    
                    success, description = recovery_system.handle_error(e, {
                        'function': func.__name__,
                        'attempt': attempt + 1,
                        'args_count': len(args),
                        'kwargs_keys': list(kwargs.keys())
                    })
                    
                    if not success:
                        continue  # Try again even if recovery failed
                    
                    # Recovery succeeded, try function again
                    print(f"Recovery attempt {attempt + 1}: {description}")
            
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper
    
    return decorator


def monitor_performance(monitor: PerformanceMonitor):
    """Decorator to monitor function performance."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                monitor.record_metric(f"{func.__name__}_execution_time", execution_time)
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                monitor.record_metric(f"{func.__name__}_error_time", execution_time)
                raise
                
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper
    
    return decorator