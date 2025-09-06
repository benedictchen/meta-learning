"""
Real-time Performance Monitoring with AI Auto-tuning

Provides intelligent performance optimization through continuous monitoring,
bottleneck detection, and automatic parameter adjustment.
"""
from __future__ import annotations

import psutil
import threading
import time
from collections import deque, defaultdict
from typing import Any, Dict, List, Optional, Tuple, Callable
import json

import numpy as np
import torch
import torch.nn as nn


class PerformanceMetrics:
    """
    Comprehensive performance metrics collection.
    """
    
    def __init__(self, history_length: int = 1000):
        self.history_length = history_length
        
        # Time-series metrics
        self.metrics = defaultdict(lambda: deque(maxlen=history_length))
        self.timestamps = deque(maxlen=history_length)
        
        # System metrics
        self.system_metrics = defaultdict(lambda: deque(maxlen=history_length))
        
        # Lock for thread safety
        self.lock = threading.Lock()
    
    def record_metric(self, name: str, value: float):
        """Record a performance metric."""
        with self.lock:
            current_time = time.time()
            self.metrics[name].append(value)
            if len(self.timestamps) == 0 or self.timestamps[-1] != current_time:
                self.timestamps.append(current_time)
    
    def record_system_metrics(self):
        """Record current system performance metrics."""
        with self.lock:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=None)
            self.system_metrics['cpu_usage'].append(cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.system_metrics['memory_usage'].append(memory.percent)
            self.system_metrics['available_memory_gb'].append(memory.available / (1024**3))
            
            # GPU metrics (if available)
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    # Memory usage
                    mem_allocated = torch.cuda.memory_allocated(i) / (1024**3)
                    mem_reserved = torch.cuda.memory_reserved(i) / (1024**3)
                    
                    self.system_metrics[f'gpu_{i}_memory_allocated_gb'].append(mem_allocated)
                    self.system_metrics[f'gpu_{i}_memory_reserved_gb'].append(mem_reserved)
                    
                    # Utilization (approximated by memory usage)
                    total_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                    utilization = (mem_reserved / total_memory) * 100
                    self.system_metrics[f'gpu_{i}_utilization'].append(utilization)
    
    def get_recent_stats(self, metric_name: str, window_size: int = 100) -> Dict[str, float]:
        """Get recent statistics for a metric."""
        with self.lock:
            if metric_name not in self.metrics:
                return {}
            
            values = list(self.metrics[metric_name])[-window_size:]
            if not values:
                return {}
            
            return {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values),
                'p95': np.percentile(values, 95),
                'p99': np.percentile(values, 99),
                'trend': np.polyfit(range(len(values)), values, 1)[0] if len(values) > 1 else 0.0
            }
    
    def detect_anomalies(self, metric_name: str, threshold_std: float = 2.0) -> List[int]:
        """Detect anomalous values in metric history."""
        with self.lock:
            if metric_name not in self.metrics:
                return []
            
            values = np.array(list(self.metrics[metric_name]))
            if len(values) < 10:
                return []
            
            # Use moving statistics to detect anomalies
            window_size = min(50, len(values) // 2)
            anomalies = []
            
            for i in range(window_size, len(values)):
                window = values[i-window_size:i]
                mean = np.mean(window)
                std = np.std(window)
                
                if std > 0 and abs(values[i] - mean) > threshold_std * std:
                    anomalies.append(i)
            
            return anomalies


class AutoTuner(nn.Module):
    """
    AI-powered automatic performance tuning system.
    
    Uses neural network to predict optimal parameters based on current
    system state and performance metrics.
    """
    
    def __init__(self, input_dim: int = 20, hidden_dim: int = 64, output_dim: int = 10):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()  # Output in [0,1] range for parameter scaling
        )
        
        # Parameter mappings
        self.param_mappings = {
            0: ('cache_size', 100, 10000),           # Cache size range
            1: ('memory_budget_gb', 0.5, 8.0),       # Memory budget range
            2: ('chunk_size_kb', 64, 8192),          # Chunk size range
            3: ('max_workers', 1, 16),               # Worker thread count
            4: ('prefetch_horizon', 5, 50),          # Prefetch horizon
            5: ('compression_level', 0, 9),          # Compression level
            6: ('batch_size', 8, 128),               # Batch size
            7: ('learning_rate', 0.0001, 0.01),     # Learning rate
            8: ('update_frequency', 10, 1000),       # Update frequency
            9: ('eviction_threshold', 0.7, 0.95),   # Cache eviction threshold
        }
        
        # Training data
        self.training_data = []
        self.performance_history = deque(maxlen=1000)
    
    def extract_features(self, metrics: PerformanceMetrics, 
                        current_config: Dict[str, Any]) -> torch.Tensor:
        """Extract features for auto-tuning prediction."""
        features = []
        
        # System metrics (normalized)
        cpu_stats = metrics.get_recent_stats('cpu_usage', 20)
        memory_stats = metrics.get_recent_stats('memory_usage', 20)
        
        features.extend([
            cpu_stats.get('mean', 0) / 100.0,      # CPU usage [0,1]
            cpu_stats.get('std', 0) / 100.0,       # CPU variance [0,1]
            memory_stats.get('mean', 0) / 100.0,   # Memory usage [0,1]
            memory_stats.get('p95', 0) / 100.0,    # Memory p95 [0,1]
        ])
        
        # Performance metrics
        access_time_stats = metrics.get_recent_stats('access_time', 50)
        cache_hit_stats = metrics.get_recent_stats('cache_hit_rate', 50)
        
        features.extend([
            min(access_time_stats.get('mean', 0) / 0.1, 1.0),    # Access time (capped at 100ms)
            min(access_time_stats.get('p95', 0) / 0.5, 1.0),     # Access time p95 (capped at 500ms)
            cache_hit_stats.get('mean', 0),                      # Cache hit rate [0,1]
            max(0, min(access_time_stats.get('trend', 0) * 1000, 1.0)),  # Trend (scaled)
        ])
        
        # GPU metrics (if available)
        if torch.cuda.is_available():
            gpu_util_stats = metrics.get_recent_stats('gpu_0_utilization', 20)
            gpu_memory_stats = metrics.get_recent_stats('gpu_0_memory_allocated_gb', 20)
            
            features.extend([
                gpu_util_stats.get('mean', 0) / 100.0,     # GPU utilization [0,1]
                min(gpu_memory_stats.get('mean', 0) / 8.0, 1.0),  # GPU memory (capped at 8GB)
            ])
        else:
            features.extend([0.0, 0.0])  # No GPU
        
        # Current configuration (normalized)
        for i in range(10):
            if i in self.param_mappings:
                param_name, min_val, max_val = self.param_mappings[i]
                current_val = current_config.get(param_name, (min_val + max_val) / 2)
                normalized = (current_val - min_val) / (max_val - min_val)
                features.append(max(0, min(1, normalized)))
            else:
                features.append(0.5)  # Default neutral value
        
        # Pad to input_dim if needed
        while len(features) < 20:
            features.append(0.0)
        
        return torch.tensor(features[:20], dtype=torch.float32)
    
    def predict_optimal_params(self, metrics: PerformanceMetrics,
                             current_config: Dict[str, Any]) -> Dict[str, Any]:
        """Predict optimal parameters based on current metrics."""
        features = self.extract_features(metrics, current_config)
        
        with torch.no_grad():
            self.eval()
            predictions = self.network(features.unsqueeze(0)).squeeze(0)
        
        # Convert predictions to actual parameter values
        optimal_params = {}
        for i, pred in enumerate(predictions):
            if i in self.param_mappings:
                param_name, min_val, max_val = self.param_mappings[i]
                # Scale prediction to parameter range
                param_value = min_val + pred.item() * (max_val - min_val)
                
                # Apply type constraints
                if param_name in ['cache_size', 'chunk_size_kb', 'max_workers', 
                                 'prefetch_horizon', 'compression_level', 'batch_size', 'update_frequency']:
                    optimal_params[param_name] = int(round(param_value))
                else:
                    optimal_params[param_name] = float(param_value)
        
        return optimal_params
    
    def record_performance(self, config: Dict[str, Any], performance_score: float):
        """Record performance for a given configuration."""
        self.performance_history.append((config.copy(), performance_score))
        
        # Add to training data if we have recent history
        if len(self.performance_history) >= 2:
            # Use improvement as target
            prev_score = self.performance_history[-2][1]
            improvement = max(-1, min(1, (performance_score - prev_score) / (abs(prev_score) + 1e-6)))
            
            # Create training sample: features -> improvement
            features = self.extract_features(None, config)  # Would need metrics for proper training
            self.training_data.append((features, improvement))
            
            # Keep training data manageable
            if len(self.training_data) > 1000:
                self.training_data = self.training_data[-500:]
    
    def update_model(self, learning_rate: float = 0.001):
        """Update the auto-tuning model with recent performance data."""
        if len(self.training_data) < 20:
            return
        
        # Prepare training data
        features = torch.stack([x[0] for x in self.training_data[-200:]])
        targets = torch.tensor([x[1] for x in self.training_data[-200:]], dtype=torch.float32)
        
        # Simple gradient descent update
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        self.train()
        for _ in range(10):  # Mini-training session
            optimizer.zero_grad()
            predictions = self.network(features)
            
            # Predict improvement (single output for this simple case)
            improvement_pred = predictions.mean(dim=1)  # Average across parameter predictions
            loss = criterion(improvement_pred, targets)
            
            loss.backward()
            optimizer.step()


class RealTimePerformanceMonitor:
    """
    Comprehensive real-time performance monitoring with AI auto-tuning.
    
    Features:
    - Continuous metrics collection and analysis
    - Automatic bottleneck detection and alerting
    - AI-powered parameter optimization
    - Real-time performance dashboards
    - Automated tuning recommendations
    """
    
    def __init__(self, monitor_interval: float = 1.0, enable_auto_tuning: bool = True):
        """
        Initialize performance monitor.
        
        Args:
            monitor_interval: Monitoring frequency in seconds
            enable_auto_tuning: Whether to enable AI auto-tuning
        """
        self.monitor_interval = monitor_interval
        self.enable_auto_tuning = enable_auto_tuning
        
        # Core components
        self.metrics = PerformanceMetrics()
        self.auto_tuner = AutoTuner() if enable_auto_tuning else None
        
        # Monitoring state
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.callbacks: List[Callable] = []
        
        # Configuration tracking
        self.current_config = {
            'cache_size': 1000,
            'memory_budget_gb': 2.0,
            'chunk_size_kb': 1024,
            'max_workers': 4,
            'prefetch_horizon': 10,
            'compression_level': 6,
            'batch_size': 32,
            'learning_rate': 0.001,
            'update_frequency': 100,
            'eviction_threshold': 0.8
        }
        
        # Performance tracking
        self.performance_score_history = deque(maxlen=100)
        self.last_tuning_time = 0.0
        self.tuning_interval = 60.0  # Re-tune every minute
        
        # Alerts and recommendations
        self.alerts = deque(maxlen=50)
        self.recommendations = deque(maxlen=20)
    
    def add_callback(self, callback: Callable[[Dict], None]):
        """Add callback for performance updates."""
        self.callbacks.append(callback)
    
    def start_monitoring(self):
        """Start real-time monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop real-time monitoring."""
        self.monitoring_active = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5.0)
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Collect system metrics
                self.metrics.record_system_metrics()
                
                # Calculate overall performance score
                performance_score = self._calculate_performance_score()
                self.performance_score_history.append(performance_score)
                
                # Check for bottlenecks and anomalies
                self._detect_bottlenecks()
                
                # Auto-tuning (if enabled and interval reached)
                if (self.enable_auto_tuning and self.auto_tuner and 
                    time.time() - self.last_tuning_time > self.tuning_interval):
                    self._perform_auto_tuning()
                
                # Notify callbacks
                status = self.get_status()
                for callback in self.callbacks:
                    try:
                        callback(status)
                    except Exception as e:
                        import warnings
                        warnings.warn(f"Performance monitor callback failed: {e}")
                
            except Exception as e:
                import warnings
                warnings.warn(f"Performance monitoring error: {e}")
            
            time.sleep(self.monitor_interval)
    
    def _calculate_performance_score(self) -> float:
        """Calculate overall performance score (0-1, higher is better)."""
        score_components = []
        
        # System utilization (moderate usage is optimal)
        cpu_stats = self.metrics.get_recent_stats('cpu_usage', 10)
        if cpu_stats:
            cpu_score = 1.0 - abs(cpu_stats['mean'] - 70) / 100.0  # Optimal around 70%
            score_components.append(cpu_score * 0.2)
        
        memory_stats = self.metrics.get_recent_stats('memory_usage', 10)
        if memory_stats:
            memory_score = 1.0 - abs(memory_stats['mean'] - 60) / 100.0  # Optimal around 60%
            score_components.append(memory_score * 0.2)
        
        # Performance metrics
        access_time_stats = self.metrics.get_recent_stats('access_time', 20)
        if access_time_stats:
            access_time_score = max(0, 1.0 - access_time_stats['mean'] / 0.1)  # Lower is better
            score_components.append(access_time_score * 0.3)
        
        cache_hit_stats = self.metrics.get_recent_stats('cache_hit_rate', 20)
        if cache_hit_stats:
            cache_score = cache_hit_stats['mean']  # Higher is better
            score_components.append(cache_score * 0.3)
        
        return np.mean(score_components) if score_components else 0.5
    
    def _detect_bottlenecks(self):
        """Detect and alert on performance bottlenecks."""
        current_time = time.time()
        
        # CPU bottleneck
        cpu_stats = self.metrics.get_recent_stats('cpu_usage', 20)
        if cpu_stats and cpu_stats['mean'] > 90:
            self._add_alert('high_cpu', f"High CPU usage: {cpu_stats['mean']:.1f}%", current_time)
            self._add_recommendation('reduce_workers', "Consider reducing max_workers to decrease CPU load")
        
        # Memory bottleneck
        memory_stats = self.metrics.get_recent_stats('memory_usage', 20)
        if memory_stats and memory_stats['mean'] > 85:
            self._add_alert('high_memory', f"High memory usage: {memory_stats['mean']:.1f}%", current_time)
            self._add_recommendation('reduce_cache', "Consider reducing cache_size or memory_budget_gb")
        
        # Slow access times
        access_stats = self.metrics.get_recent_stats('access_time', 30)
        if access_stats and access_stats['p95'] > 0.2:  # 200ms
            self._add_alert('slow_access', f"Slow access times: {access_stats['p95']:.3f}s", current_time)
            self._add_recommendation('increase_prefetch', "Consider increasing prefetch_horizon")
        
        # Low cache hit rate
        cache_stats = self.metrics.get_recent_stats('cache_hit_rate', 50)
        if cache_stats and cache_stats['mean'] < 0.5:
            self._add_alert('low_cache_hit', f"Low cache hit rate: {cache_stats['mean']:.2f}", current_time)
            self._add_recommendation('tune_cache', "Consider adjusting cache size or eviction policy")
    
    def _add_alert(self, alert_type: str, message: str, timestamp: float):
        """Add performance alert."""
        alert = {
            'type': alert_type,
            'message': message,
            'timestamp': timestamp,
            'severity': 'warning'
        }
        self.alerts.append(alert)
    
    def _add_recommendation(self, rec_type: str, message: str):
        """Add performance recommendation."""
        # Avoid duplicate recommendations
        for rec in self.recommendations:
            if rec['type'] == rec_type:
                return
        
        recommendation = {
            'type': rec_type,
            'message': message,
            'timestamp': time.time()
        }
        self.recommendations.append(recommendation)
    
    def _perform_auto_tuning(self):
        """Perform AI-powered auto-tuning."""
        if not self.auto_tuner:
            return
        
        try:
            # Get current performance score
            current_score = (
                np.mean(list(self.performance_score_history)[-10:])
                if self.performance_score_history else 0.5
            )
            
            # Record current performance
            self.auto_tuner.record_performance(self.current_config, current_score)
            
            # Get optimal parameters
            optimal_params = self.auto_tuner.predict_optimal_params(
                self.metrics, self.current_config
            )
            
            # Apply conservative updates (only change one parameter at a time)
            best_param = None
            best_diff = 0
            
            for param_name, optimal_value in optimal_params.items():
                current_value = self.current_config.get(param_name, 0)
                diff = abs(optimal_value - current_value) / max(abs(current_value), 1e-6)
                
                if diff > best_diff and diff > 0.1:  # At least 10% change
                    best_diff = diff
                    best_param = (param_name, optimal_value)
            
            # Apply the best parameter change
            if best_param:
                param_name, new_value = best_param
                old_value = self.current_config.get(param_name, 0)
                self.current_config[param_name] = new_value
                
                self._add_recommendation(
                    'auto_tune',
                    f"Auto-tuned {param_name}: {old_value:.3f} → {new_value:.3f}"
                )
            
            # Update model with recent performance
            self.auto_tuner.update_model()
            
            self.last_tuning_time = time.time()
            
        except Exception as e:
            import warnings
            warnings.warn(f"Auto-tuning failed: {e}")
    
    def record_metric(self, name: str, value: float):
        """Record a custom metric."""
        self.metrics.record_metric(name, value)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current monitoring status."""
        current_time = time.time()
        
        # Recent performance score
        recent_score = (
            np.mean(list(self.performance_score_history)[-10:])
            if self.performance_score_history else 0.0
        )
        
        # Active alerts (last 5 minutes)
        active_alerts = [
            alert for alert in self.alerts
            if current_time - alert['timestamp'] < 300
        ]
        
        # Recent recommendations (last 10 minutes)  
        recent_recommendations = [
            rec for rec in self.recommendations
            if current_time - rec['timestamp'] < 600
        ]
        
        return {
            'monitoring_active': self.monitoring_active,
            'performance_score': recent_score,
            'current_config': self.current_config.copy(),
            'system_metrics': {
                'cpu_usage': self.metrics.get_recent_stats('cpu_usage', 5),
                'memory_usage': self.metrics.get_recent_stats('memory_usage', 5),
                'gpu_utilization': self.metrics.get_recent_stats('gpu_0_utilization', 5),
            },
            'performance_metrics': {
                'access_time': self.metrics.get_recent_stats('access_time', 20),
                'cache_hit_rate': self.metrics.get_recent_stats('cache_hit_rate', 20),
            },
            'alerts': active_alerts,
            'recommendations': recent_recommendations,
            'auto_tuning_enabled': self.enable_auto_tuning,
            'last_tuning': self.last_tuning_time
        }
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get formatted data for performance dashboard."""
        status = self.get_status()
        
        # Time series data for charts
        with self.metrics.lock:
            recent_timestamps = list(self.metrics.timestamps)[-100:]
            
            chart_data = {}
            for metric_name in ['cpu_usage', 'memory_usage', 'access_time', 'cache_hit_rate']:
                if metric_name in self.metrics.metrics:
                    values = list(self.metrics.metrics[metric_name])[-len(recent_timestamps):]
                    chart_data[metric_name] = {
                        'timestamps': recent_timestamps[-len(values):],
                        'values': values
                    }
        
        return {
            'status': status,
            'charts': chart_data,
            'summary': {
                'overall_health': 'good' if status['performance_score'] > 0.7 else 
                                 'warning' if status['performance_score'] > 0.4 else 'critical',
                'active_alerts_count': len(status['alerts']),
                'recommendations_count': len(status['recommendations'])
            }
        }


def create_performance_monitor(enable_auto_tuning: bool = True) -> RealTimePerformanceMonitor:
    """Create performance monitor with optimal defaults."""
    return RealTimePerformanceMonitor(
        monitor_interval=1.0,
        enable_auto_tuning=enable_auto_tuning
    )