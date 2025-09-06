"""
Performance Monitoring for Meta-Learning Systems.

This module provides advanced performance monitoring capabilities with
real-time trend analysis and adaptive optimization suggestions.

Classes:
    PerformanceMonitor: Monitors performance metrics, tracks trends,
                       and provides optimization suggestions.

The monitor tracks key metrics over time, detects performance anomalies,
and suggests adaptive optimizations based on observed trends.
"""

from typing import Dict, List, Any, Optional
import time
import warnings
import numpy as np


class PerformanceMonitor:
    """Advanced performance monitoring and adaptive optimization.
    
    Continuously monitors performance metrics, analyzes trends, and provides
    real-time optimization suggestions to improve meta-learning performance.
    
    Attributes:
        metrics_history (List): Historical performance metrics
        performance_trends (Dict): Computed trends for key metrics
        alert_thresholds (Dict): Thresholds for performance alerts
    """
    
    def __init__(self):
        """Initialize the performance monitor."""
        self.metrics_history = []
        self.performance_trends = {}
        self.alert_thresholds = {
            'accuracy_drop': 0.1,
            'loss_spike': 2.0,
            'training_time': 300.0  # 5 minutes
        }
        
    def record_metrics(self, metrics: Dict[str, Any], timestamp: Optional[float] = None):
        """Record performance metrics.
        
        Stores metrics with timestamp and triggers trend analysis and alert checking.
        
        Args:
            metrics: Dictionary of performance metrics (accuracy, loss, training_time, etc.)
            timestamp: Optional timestamp (defaults to current time)
        """
        if timestamp is None:
            timestamp = time.time()
        
        metrics_entry = {
            'timestamp': timestamp,
            'metrics': metrics.copy()
        }
        self.metrics_history.append(metrics_entry)
        
        # Keep recent history
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-500:]
        
        self._update_trends()
        self._check_alerts(metrics)
    
    def _update_trends(self):
        """Update performance trend analysis.
        
        Computes linear trends for key metrics over recent history
        to identify performance patterns.
        """
        if len(self.metrics_history) < 10:
            return
        
        recent_metrics = self.metrics_history[-10:]
        
        for metric_name in ['accuracy', 'loss', 'training_time']:
            values = []
            timestamps = []
            
            for entry in recent_metrics:
                if metric_name in entry['metrics']:
                    values.append(entry['metrics'][metric_name])
                    timestamps.append(entry['timestamp'])
            
            if len(values) >= 5:
                # Simple linear trend analysis
                x = np.array(timestamps)
                y = np.array(values)
                
                if len(x) > 1:
                    trend = np.polyfit(x - x[0], y, 1)[0]  # Slope
                    self.performance_trends[metric_name] = {
                        'trend': trend,
                        'recent_mean': np.mean(y),
                        'recent_std': np.std(y)
                    }
    
    def _check_alerts(self, metrics: Dict[str, Any]):
        """Check for performance alerts.
        
        Compares current metrics against historical trends to detect
        significant performance degradation.
        
        Args:
            metrics: Current performance metrics
        """
        alerts = []
        
        # Check for accuracy drops
        if 'accuracy' in metrics and 'accuracy' in self.performance_trends:
            recent_acc = metrics['accuracy']
            trend_acc = self.performance_trends['accuracy']['recent_mean']
            
            if trend_acc - recent_acc > self.alert_thresholds['accuracy_drop']:
                alerts.append({
                    'type': 'accuracy_drop',
                    'message': f"Accuracy dropped from {trend_acc:.3f} to {recent_acc:.3f}",
                    'severity': 'high'
                })
        
        # Check for loss spikes
        if 'loss' in metrics and 'loss' in self.performance_trends:
            recent_loss = metrics['loss']
            trend_loss = self.performance_trends['loss']['recent_mean']
            
            if recent_loss > trend_loss * (1 + self.alert_thresholds['loss_spike']):
                alerts.append({
                    'type': 'loss_spike',
                    'message': f"Loss spiked from {trend_loss:.3f} to {recent_loss:.3f}",
                    'severity': 'medium'
                })
        
        # Check for training time issues
        if 'training_time' in metrics:
            if metrics['training_time'] > self.alert_thresholds['training_time']:
                alerts.append({
                    'type': 'slow_training',
                    'message': f"Training time exceeded {self.alert_thresholds['training_time']:.1f}s",
                    'severity': 'low'
                })
        
        # Log alerts
        for alert in alerts:
            if alert['severity'] == 'high':
                warnings.warn(f"Performance Alert: {alert['message']}")
    
    def get_optimization_suggestions(self) -> List[str]:
        """Get adaptive optimization suggestions based on performance trends.
        
        Analyzes performance trends and provides actionable suggestions
        for improving meta-learning performance.
        
        Returns:
            List of optimization suggestions as strings
        """
        suggestions = []
        
        if 'accuracy' in self.performance_trends:
            acc_trend = self.performance_trends['accuracy']['trend']
            
            if acc_trend < -0.001:  # Decreasing accuracy
                suggestions.append("Consider reducing learning rate or increasing regularization")
            elif acc_trend > 0.001:  # Improving accuracy
                suggestions.append("Performance is improving - consider increasing learning rate")
        
        if 'loss' in self.performance_trends:
            loss_trend = self.performance_trends['loss']['trend']
            
            if loss_trend > 0.01:  # Increasing loss
                suggestions.append("Loss is increasing - check for overfitting or reduce learning rate")
            elif loss_trend < -0.01:  # Decreasing loss  
                suggestions.append("Loss is decreasing well - current settings are effective")
        
        if 'training_time' in self.performance_trends:
            time_trend = self.performance_trends['training_time']['trend']
            
            if time_trend > 1.0:  # Training time increasing
                suggestions.append("Training time is increasing - consider reducing model complexity")
        
        return suggestions
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary.
        
        Returns:
            Dictionary with current trends, recent averages, and alert status
        """
        summary = {
            'total_episodes': len(self.metrics_history),
            'trends': self.performance_trends.copy(),
            'recent_performance': {},
            'alerts_triggered': 0
        }
        
        # Get recent averages
        if len(self.metrics_history) > 0:
            recent_entries = self.metrics_history[-10:]
            
            for metric in ['accuracy', 'loss', 'training_time']:
                values = []
                for entry in recent_entries:
                    if metric in entry['metrics']:
                        values.append(entry['metrics'][metric])
                
                if values:
                    summary['recent_performance'][metric] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'count': len(values)
                    }
        
        return summary
    
    def set_alert_threshold(self, metric: str, threshold: float):
        """Set custom alert threshold for a metric.
        
        Args:
            metric: Metric name ('accuracy_drop', 'loss_spike', 'training_time')
            threshold: New threshold value
        """
        self.alert_thresholds[metric] = threshold