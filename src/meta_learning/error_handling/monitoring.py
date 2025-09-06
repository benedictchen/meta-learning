"""
Real-time performance monitoring and alerting system.

Provides comprehensive metrics collection, anomaly detection,
and resource utilization optimization for meta-learning pipelines.
"""

import time
import threading
from typing import Dict, Any, List, Callable, Optional, Union, Tuple
from collections import deque, defaultdict
from enum import Enum
import warnings
import statistics

import torch
import numpy as np


class MetricType(Enum):
    """Types of metrics to track."""
    ACCURACY = "accuracy"
    LOSS = "loss"
    TRAINING_TIME = "training_time"
    MEMORY_USAGE = "memory_usage"
    GPU_UTILIZATION = "gpu_utilization"
    EPISODE_GENERATION_RATE = "episode_generation_rate"
    CACHE_HIT_RATE = "cache_hit_rate"
    ERROR_RATE = "error_rate"
    CUSTOM = "custom"


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class Alert:
    """Performance alert with context."""
    
    def __init__(self, 
                 level: AlertLevel,
                 metric: str,
                 message: str,
                 value: float,
                 threshold: float,
                 timestamp: Optional[float] = None):
        self.level = level
        self.metric = metric
        self.message = message
        self.value = value
        self.threshold = threshold
        self.timestamp = timestamp or time.time()
    
    def __str__(self) -> str:
        return f"[{self.level.value.upper()}] {self.metric}: {self.message} (value={self.value:.3f}, threshold={self.threshold:.3f})"


class MetricsCollector:
    """Thread-safe metrics collection with statistical analysis."""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.metrics = defaultdict(lambda: deque(maxlen=max_history))
        self.metric_timestamps = defaultdict(lambda: deque(maxlen=max_history))
        self.lock = threading.RLock()
        
        # Statistical caches for performance
        self.stats_cache = {}
        self.cache_timestamps = {}
        self.cache_ttl = 5.0  # Cache TTL in seconds
    
    def record(self, metric_name: str, value: float, timestamp: Optional[float] = None) -> None:
        """Record a metric value with timestamp."""
        if timestamp is None:
            timestamp = time.time()
        
        with self.lock:
            self.metrics[metric_name].append(value)
            self.metric_timestamps[metric_name].append(timestamp)
            
            # Invalidate cache for this metric
            if metric_name in self.stats_cache:
                del self.stats_cache[metric_name]
                del self.cache_timestamps[metric_name]
    
    def get_recent_values(self, metric_name: str, seconds: int = 60) -> List[float]:
        """Get metric values from the last N seconds."""
        with self.lock:
            if metric_name not in self.metrics:
                return []
            
            current_time = time.time()
            cutoff_time = current_time - seconds
            
            values = []
            timestamps = self.metric_timestamps[metric_name]
            metric_values = self.metrics[metric_name]
            
            for i, ts in enumerate(timestamps):
                if ts >= cutoff_time:
                    values.append(metric_values[i])
            
            return values
    
    def get_statistics(self, metric_name: str, use_cache: bool = True) -> Dict[str, float]:
        """Get comprehensive statistics for a metric."""
        # Check cache first
        if use_cache and metric_name in self.stats_cache:
            cache_time = self.cache_timestamps.get(metric_name, 0)
            if time.time() - cache_time < self.cache_ttl:
                return self.stats_cache[metric_name]
        
        with self.lock:
            values = list(self.metrics[metric_name])
            
            if not values:
                return {
                    "count": 0, "mean": 0.0, "std": 0.0, "min": 0.0, 
                    "max": 0.0, "median": 0.0, "percentile_95": 0.0
                }
            
            stats = {
                "count": len(values),
                "mean": statistics.mean(values),
                "min": min(values),
                "max": max(values),
                "median": statistics.median(values)
            }
            
            if len(values) > 1:
                stats["std"] = statistics.stdev(values)
                stats["percentile_95"] = np.percentile(values, 95)
            else:
                stats["std"] = 0.0
                stats["percentile_95"] = values[0]
            
            # Cache results
            if use_cache:
                self.stats_cache[metric_name] = stats
                self.cache_timestamps[metric_name] = time.time()
            
            return stats
    
    def get_trend(self, metric_name: str, window_size: int = 10) -> str:
        """Get trend direction for metric (increasing, decreasing, stable)."""
        with self.lock:
            values = list(self.metrics[metric_name])
            
            if len(values) < window_size:
                return "insufficient_data"
            
            recent_values = values[-window_size:]
            first_half = recent_values[:window_size//2]
            second_half = recent_values[window_size//2:]
            
            first_mean = statistics.mean(first_half)
            second_mean = statistics.mean(second_half)
            
            relative_change = (second_mean - first_mean) / max(abs(first_mean), 1e-8)
            
            if relative_change > 0.05:  # 5% increase
                return "increasing"
            elif relative_change < -0.05:  # 5% decrease
                return "decreasing"
            else:
                return "stable"
    
    def clear_metric(self, metric_name: str) -> None:
        """Clear all data for a specific metric."""
        with self.lock:
            if metric_name in self.metrics:
                self.metrics[metric_name].clear()
                self.metric_timestamps[metric_name].clear()
            
            if metric_name in self.stats_cache:
                del self.stats_cache[metric_name]
                del self.cache_timestamps[metric_name]
    
    def get_all_metrics(self) -> List[str]:
        """Get list of all tracked metrics."""
        with self.lock:
            return list(self.metrics.keys())


class AnomalyDetector:
    """Statistical anomaly detection for performance metrics."""
    
    def __init__(self, 
                 sensitivity: float = 2.0,
                 min_samples: int = 20):
        """
        Initialize anomaly detector.
        
        Args:
            sensitivity: Number of standard deviations for anomaly threshold
            min_samples: Minimum samples needed for detection
        """
        self.sensitivity = sensitivity
        self.min_samples = min_samples
    
    def detect_anomalies(self, values: List[float]) -> List[Tuple[int, float]]:
        """
        Detect anomalies in metric values.
        
        Returns:
            List of (index, value) tuples for anomalous values
        """
        if len(values) < self.min_samples:
            return []
        
        mean = statistics.mean(values)
        try:
            std = statistics.stdev(values)
        except statistics.StatisticsError:
            return []  # All values identical
        
        if std == 0:
            return []
        
        anomalies = []
        threshold = self.sensitivity * std
        
        for i, value in enumerate(values):
            if abs(value - mean) > threshold:
                anomalies.append((i, value))
        
        return anomalies
    
    def is_anomalous(self, value: float, reference_values: List[float]) -> bool:
        """Check if a single value is anomalous compared to reference."""
        if len(reference_values) < self.min_samples:
            return False
        
        mean = statistics.mean(reference_values)
        try:
            std = statistics.stdev(reference_values)
        except statistics.StatisticsError:
            return False
        
        if std == 0:
            return False
        
        return abs(value - mean) > self.sensitivity * std


class AlertSystem:
    """Real-time alerting system with configurable thresholds."""
    
    def __init__(self):
        self.thresholds = {}
        self.alert_callbacks = defaultdict(list)
        self.alert_history = deque(maxlen=1000)
        self.suppressed_alerts = set()
        self.anomaly_detector = AnomalyDetector()
    
    def set_threshold(self, 
                     metric: str,
                     level: AlertLevel,
                     threshold: float,
                     comparison: str = "greater") -> None:
        """
        Set alert threshold for metric.
        
        Args:
            metric: Metric name
            level: Alert severity level
            threshold: Threshold value
            comparison: "greater", "less", "equal"
        """
        self.thresholds[metric] = {
            "level": level,
            "threshold": threshold,
            "comparison": comparison
        }
    
    def add_alert_callback(self, level: AlertLevel, callback: Callable[[Alert], None]) -> None:
        """Add callback function for specific alert level."""
        self.alert_callbacks[level].append(callback)
    
    def check_thresholds(self, metric: str, value: float) -> Optional[Alert]:
        """Check if metric value triggers any alerts."""
        if metric not in self.thresholds:
            return None
        
        threshold_config = self.thresholds[metric]
        threshold = threshold_config["threshold"]
        comparison = threshold_config["comparison"]
        level = threshold_config["level"]
        
        triggered = False
        if comparison == "greater" and value > threshold:
            triggered = True
        elif comparison == "less" and value < threshold:
            triggered = True
        elif comparison == "equal" and abs(value - threshold) < 1e-8:
            triggered = True
        
        if triggered:
            alert_key = f"{metric}:{level.value}"
            if alert_key not in self.suppressed_alerts:
                message = f"Metric {metric} {comparison} threshold {threshold}"
                alert = Alert(level, metric, message, value, threshold)
                self._handle_alert(alert)
                return alert
        
        return None
    
    def check_anomalies(self, metric: str, current_value: float, reference_values: List[float]) -> Optional[Alert]:
        """Check for anomalous metric values."""
        if self.anomaly_detector.is_anomalous(current_value, reference_values):
            message = f"Anomalous value detected for {metric}"
            alert = Alert(AlertLevel.WARNING, metric, message, current_value, 0.0)
            self._handle_alert(alert)
            return alert
        
        return None
    
    def _handle_alert(self, alert: Alert) -> None:
        """Handle triggered alert."""
        self.alert_history.append(alert)
        
        # Call registered callbacks
        for callback in self.alert_callbacks[alert.level]:
            try:
                callback(alert)
            except Exception as e:
                print(f"Alert callback error: {e}")
        
        # Default handling
        if alert.level in [AlertLevel.CRITICAL, AlertLevel.EMERGENCY]:
            print(f"Alert: {alert}")
        elif alert.level == AlertLevel.WARNING:
            warnings.warn(str(alert), category=UserWarning)
    
    def suppress_alert(self, metric: str, level: AlertLevel, duration: float = 300) -> None:
        """Temporarily suppress alerts for metric."""
        alert_key = f"{metric}:{level.value}"
        self.suppressed_alerts.add(alert_key)
        
        # Auto-unsuppress after duration
        def unsuppress():
            time.sleep(duration)
            self.suppressed_alerts.discard(alert_key)
        
        threading.Thread(target=unsuppress, daemon=True).start()
    
    def get_recent_alerts(self, seconds: int = 3600) -> List[Alert]:
        """Get alerts from the last N seconds."""
        cutoff_time = time.time() - seconds
        return [alert for alert in self.alert_history if alert.timestamp >= cutoff_time]


class PerformanceMonitor:
    """
    Comprehensive performance monitoring system.
    
    Features:
    - Real-time metrics collection
    - Automatic anomaly detection
    - Resource utilization optimization
    - Failure prediction and prevention
    """
    
    def __init__(self, 
                 collection_interval: float = 1.0,
                 enable_auto_alerts: bool = True):
        """
        Initialize performance monitor.
        
        Args:
            collection_interval: Seconds between automatic collections
            enable_auto_alerts: Enable automatic alert generation
        """
        self.collection_interval = collection_interval
        self.enable_auto_alerts = enable_auto_alerts
        
        # Initialize components
        self.metrics = MetricsCollector()
        self.alerts = AlertSystem()
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # Auto-configure common alerts
        if enable_auto_alerts:
            self._setup_default_alerts()
    
    def _setup_default_alerts(self) -> None:
        """Setup common performance alerts."""
        # Memory usage alerts
        self.alerts.set_threshold("memory_usage", AlertLevel.WARNING, 0.8, "greater")
        self.alerts.set_threshold("memory_usage", AlertLevel.CRITICAL, 0.95, "greater")
        
        # Error rate alerts
        self.alerts.set_threshold("error_rate", AlertLevel.WARNING, 0.1, "greater")
        self.alerts.set_threshold("error_rate", AlertLevel.CRITICAL, 0.25, "greater")
        
        # Performance alerts
        self.alerts.set_threshold("training_time", AlertLevel.WARNING, 300, "greater")  # 5 minutes
        
        # Add default callback for critical alerts
        def critical_alert_handler(alert: Alert):
            print(f"🚨 CRITICAL ALERT: {alert}")
        
        self.alerts.add_alert_callback(AlertLevel.CRITICAL, critical_alert_handler)
    
    def record_metric(self, metric_type: Union[MetricType, str], value: float) -> None:
        """Record a performance metric."""
        metric_name = metric_type.value if isinstance(metric_type, MetricType) else metric_type
        self.metrics.record(metric_name, value)
        
        # Check for alerts
        if self.enable_auto_alerts:
            alert = self.alerts.check_thresholds(metric_name, value)
            
            # Check for anomalies
            recent_values = self.metrics.get_recent_values(metric_name, 300)  # Last 5 minutes
            if len(recent_values) > 20:
                self.alerts.check_anomalies(metric_name, value, recent_values[:-1])
    
    def start_monitoring(self) -> None:
        """Start automatic system monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
    
    def stop_monitoring(self) -> None:
        """Stop automatic monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                self._collect_system_metrics()
                time.sleep(self.collection_interval)
            except Exception as e:
                print(f"Monitoring error: {e}")
                time.sleep(self.collection_interval)
    
    def _collect_system_metrics(self) -> None:
        """Collect system-level metrics."""
        # GPU metrics
        if torch.cuda.is_available():
            try:
                memory_allocated = torch.cuda.memory_allocated()
                memory_reserved = torch.cuda.memory_reserved()
                max_memory = torch.cuda.max_memory_allocated()
                
                if max_memory > 0:
                    memory_utilization = memory_allocated / max_memory
                    self.record_metric(MetricType.MEMORY_USAGE, memory_utilization)
                
                # GPU utilization (requires nvidia-ml-py if available)
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
                    gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    self.record_metric(MetricType.GPU_UTILIZATION, gpu_util.gpu / 100.0)
                except ImportError:
                    pass  # nvidia-ml-py not available
            except:
                pass  # GPU metrics not available
    
    def get_performance_report(self) -> str:
        """Generate comprehensive performance report."""
        all_metrics = self.metrics.get_all_metrics()
        
        report = ["Performance Monitoring Report", "=" * 35, ""]
        
        if not all_metrics:
            report.append("No metrics collected yet.")
            return "\n".join(report)
        
        # System metrics
        report.append("System Metrics:")
        system_metrics = [m for m in all_metrics if m in [mt.value for mt in MetricType]]
        
        for metric in system_metrics:
            stats = self.metrics.get_statistics(metric)
            trend = self.metrics.get_trend(metric)
            report.append(f"  {metric}:")
            report.append(f"    Current: {stats['mean']:.3f} (trend: {trend})")
            report.append(f"    Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
            report.append(f"    Std Dev: {stats['std']:.3f}")
        
        report.append("")
        
        # Recent alerts
        recent_alerts = self.alerts.get_recent_alerts(3600)  # Last hour
        if recent_alerts:
            report.append("Recent Alerts (1h):")
            for alert in recent_alerts[-5:]:  # Last 5 alerts
                report.append(f"  {alert}")
        
        report.append("")
        
        # Performance summary
        if "training_time" in all_metrics:
            training_stats = self.metrics.get_statistics("training_time")
            report.append("Performance Summary:")
            report.append(f"  Avg Training Time: {training_stats['mean']:.2f}s")
            report.append(f"  Training Time 95th percentile: {training_stats.get('percentile_95', 0):.2f}s")
        
        if "error_rate" in all_metrics:
            error_stats = self.metrics.get_statistics("error_rate")
            report.append(f"  Current Error Rate: {error_stats['mean']:.1%}")
        
        return "\n".join(report)
    
    def set_alert_threshold(self, metric: str, level: AlertLevel, threshold: float, comparison: str = "greater") -> None:
        """Set custom alert threshold."""
        self.alerts.set_threshold(metric, level, threshold, comparison)
    
    def add_alert_callback(self, level: AlertLevel, callback: Callable[[Alert], None]) -> None:
        """Add custom alert callback."""
        self.alerts.add_alert_callback(level, callback)
    
    def clear_metrics(self, metric_name: Optional[str] = None) -> None:
        """Clear metrics data."""
        if metric_name:
            self.metrics.clear_metric(metric_name)
        else:
            # Clear all metrics
            for metric in self.metrics.get_all_metrics():
                self.metrics.clear_metric(metric)