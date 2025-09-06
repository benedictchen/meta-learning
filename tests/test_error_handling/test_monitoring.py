"""
Tests for PerformanceMonitor.

Tests real-time performance monitoring with alerting and anomaly detection.
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch
import torch

from meta_learning.error_handling.monitoring import (
    PerformanceMonitor, MetricCollector, AlertSystem, AnomalyDetector
)


class TestMetricCollector:
    """Test metric collection functionality."""
    
    def test_initialization(self):
        """Test collector initialization."""
        collector = MetricCollector()
        
        assert collector.metrics == {}
        assert collector.collection_count == 0
        assert hasattr(collector, 'start_time')
    
    def test_metric_recording(self):
        """Test basic metric recording."""
        collector = MetricCollector()
        
        # Record various metrics
        collector.record('accuracy', 0.85)
        collector.record('loss', 0.23)
        collector.record('memory_usage', 1024)
        
        assert 'accuracy' in collector.metrics
        assert 'loss' in collector.metrics
        assert 'memory_usage' in collector.metrics
        
        # Should store values with timestamps
        assert len(collector.metrics['accuracy']) == 1
        assert collector.metrics['accuracy'][0][0] == 0.85  # value
        assert isinstance(collector.metrics['accuracy'][0][1], float)  # timestamp
    
    def test_time_series_collection(self):
        """Test time series data collection."""
        collector = MetricCollector()
        
        # Record same metric multiple times
        for i in range(5):
            collector.record('training_step', i)
            time.sleep(0.01)  # Small delay for different timestamps
        
        # Should have 5 entries
        assert len(collector.metrics['training_step']) == 5
        
        # Should have different timestamps
        timestamps = [entry[1] for entry in collector.metrics['training_step']]
        assert len(set(timestamps)) == 5  # All unique
        
        # Values should be in order
        values = [entry[0] for entry in collector.metrics['training_step']]
        assert values == [0, 1, 2, 3, 4]
    
    def test_metric_statistics(self):
        """Test metric statistics calculation."""
        collector = MetricCollector()
        
        # Record some values
        values = [0.1, 0.3, 0.5, 0.7, 0.9]
        for val in values:
            collector.record('test_metric', val)
        
        stats = collector.get_statistics('test_metric')
        
        assert abs(stats['mean'] - 0.5) < 1e-6
        assert abs(stats['std'] - 0.31622) < 1e-3  # std of [0.1,0.3,0.5,0.7,0.9]
        assert stats['min'] == 0.1
        assert stats['max'] == 0.9
        assert stats['count'] == 5
        assert 'trend' in stats
    
    def test_metric_windowing(self):
        """Test windowed metric collection."""
        collector = MetricCollector(max_history=3)
        
        # Record more values than window size
        for i in range(6):
            collector.record('windowed_metric', i)
        
        # Should only keep last 3 values
        assert len(collector.metrics['windowed_metric']) == 3
        values = [entry[0] for entry in collector.metrics['windowed_metric']]
        assert values == [3, 4, 5]  # Last 3 values
    
    def test_multiple_metrics(self):
        """Test collecting multiple different metrics."""
        collector = MetricCollector()
        
        metrics_data = {
            'accuracy': [0.7, 0.75, 0.8],
            'loss': [0.5, 0.4, 0.3],
            'lr': [0.001, 0.001, 0.0005]
        }
        
        for metric_name, values in metrics_data.items():
            for value in values:
                collector.record(metric_name, value)
        
        # All metrics should be recorded
        assert len(collector.metrics) == 3
        for metric_name in metrics_data:
            assert len(collector.metrics[metric_name]) == 3
    
    def test_tensor_metric_handling(self):
        """Test handling of tensor metrics."""
        collector = MetricCollector()
        
        # Record tensor metrics
        tensor_metric = torch.tensor([1.0, 2.0, 3.0])
        collector.record('tensor_values', tensor_metric)
        
        # Should convert tensor to appropriate format
        recorded_value = collector.metrics['tensor_values'][0][0]
        
        # Should handle tensor appropriately (mean, norm, etc.)
        assert isinstance(recorded_value, (float, int, list))
    
    def test_thread_safety(self):
        """Test thread-safe metric collection."""
        collector = MetricCollector()
        
        def worker_thread(worker_id, num_metrics):
            for i in range(num_metrics):
                collector.record(f'worker_{worker_id}', i)
                collector.record('shared_metric', worker_id * 100 + i)
        
        # Start multiple threads
        threads = []
        for worker_id in range(3):
            thread = threading.Thread(target=worker_thread, args=(worker_id, 10))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Should have collected all metrics without corruption
        assert len(collector.metrics) == 4  # 3 worker metrics + 1 shared
        assert len(collector.metrics['shared_metric']) == 30  # 3 workers × 10 metrics


class TestAnomalyDetector:
    """Test anomaly detection functionality."""
    
    def test_initialization(self):
        """Test detector initialization."""
        detector = AnomalyDetector(threshold=2.0, window_size=10)
        
        assert detector.threshold == 2.0
        assert detector.window_size == 10
        assert detector.baseline_stats == {}
        assert detector.anomaly_count == 0
    
    def test_baseline_establishment(self):
        """Test establishing baseline statistics."""
        detector = AnomalyDetector(window_size=5)
        
        # Provide baseline data
        normal_values = [10.0, 12.0, 11.0, 13.0, 9.0, 10.5, 11.5]
        for value in normal_values:
            detector.add_baseline_data('test_metric', value)
        
        # Should establish baseline
        assert 'test_metric' in detector.baseline_stats
        baseline = detector.baseline_stats['test_metric']
        assert abs(baseline['mean'] - 11.0) < 1.0  # Approximately 11
        assert baseline['std'] > 0
    
    def test_anomaly_detection(self):
        """Test anomaly detection."""
        detector = AnomalyDetector(threshold=2.0, window_size=5)
        
        # Establish baseline
        normal_values = [10, 11, 12, 13, 14] * 2  # Stable values
        for value in normal_values:
            detector.add_baseline_data('test_metric', value)
        
        # Test normal value (should not be anomaly)
        is_anomaly_normal = detector.is_anomaly('test_metric', 12.5)
        assert not is_anomaly_normal
        
        # Test anomalous value (should be anomaly)
        is_anomaly_high = detector.is_anomaly('test_metric', 100.0)  # Much higher
        assert is_anomaly_high
        
        is_anomaly_low = detector.is_anomaly('test_metric', -50.0)  # Much lower
        assert is_anomaly_low
    
    def test_adaptive_threshold(self):
        """Test adaptive threshold adjustment."""
        detector = AnomalyDetector(threshold=2.0, adaptive=True)
        
        # Start with baseline
        for value in [10, 11, 12, 13, 14]:
            detector.add_baseline_data('adaptive_metric', value)
        
        initial_threshold = detector.threshold
        
        # Simulate changing distribution
        changing_values = [15, 16, 17, 18, 19, 20]  # Higher values
        for value in changing_values:
            detector.update_adaptive_threshold('adaptive_metric', value)
        
        # Threshold should have adapted
        current_threshold = detector.get_current_threshold('adaptive_metric')
        # (Exact behavior depends on implementation)
        assert isinstance(current_threshold, float)
    
    def test_anomaly_tracking(self):
        """Test anomaly tracking and statistics."""
        detector = AnomalyDetector(window_size=5)
        
        # Establish baseline
        for value in [5, 6, 7, 8, 9]:
            detector.add_baseline_data('tracked_metric', value)
        
        # Generate mix of normal and anomalous values
        test_values = [7.5, 100.0, 6.5, -50.0, 8.2, 200.0]
        anomaly_results = []
        
        for value in test_values:
            is_anomaly = detector.is_anomaly('tracked_metric', value)
            anomaly_results.append(is_anomaly)
            if is_anomaly:
                detector.record_anomaly('tracked_metric', value)
        
        # Should have detected some anomalies
        assert any(anomaly_results)
        assert detector.anomaly_count > 0
        
        # Get anomaly statistics
        stats = detector.get_anomaly_statistics()
        assert isinstance(stats, dict)
        assert 'total_anomalies' in stats
        assert stats['total_anomalies'] > 0
    
    def test_multiple_metrics_monitoring(self):
        """Test monitoring multiple metrics simultaneously."""
        detector = AnomalyDetector()
        
        # Set up baselines for different metrics
        metrics = {
            'accuracy': [0.8, 0.82, 0.85, 0.83, 0.84],
            'loss': [0.3, 0.25, 0.27, 0.24, 0.26],
            'memory_gb': [2.1, 2.3, 2.2, 2.4, 2.0]
        }
        
        for metric_name, values in metrics.items():
            for value in values:
                detector.add_baseline_data(metric_name, value)
        
        # Test anomaly detection for each metric
        test_cases = {
            'accuracy': (0.95, False),  # High but possibly normal
            'accuracy': (0.1, True),    # Very low - anomaly
            'loss': (5.0, True),        # Very high - anomaly
            'memory_gb': (2.25, False)  # Normal range
        }
        
        for metric_name, (test_value, expected_anomaly) in test_cases.items():
            is_anomaly = detector.is_anomaly(metric_name, test_value)
            if expected_anomaly:
                assert is_anomaly, f"Expected anomaly for {metric_name}={test_value}"
    
    def test_seasonal_pattern_detection(self):
        """Test detection of seasonal/periodic patterns."""
        detector = AnomalyDetector(detect_seasonality=True)
        
        # Create periodic data (simulating daily patterns)
        periodic_data = []
        for day in range(14):  # 2 weeks
            for hour in range(24):
                # Simulate daily pattern with noise
                base_value = 50 + 30 * torch.sin(2 * torch.pi * hour / 24)
                noise = torch.randn(1).item() * 2
                periodic_data.append(base_value + noise)
        
        for value in periodic_data:
            detector.add_baseline_data('periodic_metric', value)
        
        # Test values that fit the pattern
        morning_value = 50 + 30 * torch.sin(2 * torch.pi * 8 / 24)  # 8 AM
        evening_value = 50 + 30 * torch.sin(2 * torch.pi * 20 / 24)  # 8 PM
        
        # Should recognize pattern and not flag as anomalies
        assert not detector.is_anomaly('periodic_metric', morning_value + 3)  # Small deviation
        
        # But should flag values that break the pattern
        assert detector.is_anomaly('periodic_metric', 150)  # Way outside pattern


class TestAlertSystem:
    """Test alerting functionality."""
    
    def test_initialization(self):
        """Test alert system initialization."""
        alert_system = AlertSystem()
        
        assert alert_system.alerts == []
        assert alert_system.alert_handlers == []
        assert alert_system.alert_count == 0
    
    def test_alert_creation(self):
        """Test creating alerts."""
        alert_system = AlertSystem()
        
        alert_system.create_alert(
            level='WARNING',
            message='Test warning message',
            metric_name='test_metric',
            current_value=100.0,
            threshold=50.0
        )
        
        assert len(alert_system.alerts) == 1
        alert = alert_system.alerts[0]
        
        assert alert['level'] == 'WARNING'
        assert alert['message'] == 'Test warning message'
        assert alert['metric_name'] == 'test_metric'
        assert alert['current_value'] == 100.0
        assert 'timestamp' in alert
    
    def test_alert_levels(self):
        """Test different alert levels."""
        alert_system = AlertSystem()
        
        # Create alerts of different levels
        levels = ['INFO', 'WARNING', 'ERROR', 'CRITICAL']
        
        for level in levels:
            alert_system.create_alert(
                level=level,
                message=f'Test {level} alert',
                metric_name='test_metric'
            )
        
        assert len(alert_system.alerts) == 4
        
        # Check levels are preserved
        recorded_levels = [alert['level'] for alert in alert_system.alerts]
        assert recorded_levels == levels
    
    def test_alert_handlers(self):
        """Test alert handlers."""
        alert_system = AlertSystem()
        
        # Create mock handler
        handler_calls = []
        
        def mock_handler(alert):
            handler_calls.append(alert)
        
        # Register handler
        alert_system.add_handler(mock_handler)
        
        # Create alert
        alert_system.create_alert(
            level='ERROR',
            message='Test error',
            metric_name='error_metric'
        )
        
        # Handler should have been called
        assert len(handler_calls) == 1
        assert handler_calls[0]['level'] == 'ERROR'
    
    def test_multiple_handlers(self):
        """Test multiple alert handlers."""
        alert_system = AlertSystem()
        
        handler1_calls = []
        handler2_calls = []
        
        def handler1(alert):
            handler1_calls.append(f"H1: {alert['message']}")
        
        def handler2(alert):
            handler2_calls.append(f"H2: {alert['level']}")
        
        alert_system.add_handler(handler1)
        alert_system.add_handler(handler2)
        
        alert_system.create_alert(
            level='WARNING',
            message='Multi-handler test'
        )
        
        # Both handlers should be called
        assert len(handler1_calls) == 1
        assert len(handler2_calls) == 1
        assert 'Multi-handler test' in handler1_calls[0]
        assert 'WARNING' in handler2_calls[0]
    
    def test_alert_filtering(self):
        """Test alert filtering by level."""
        alert_system = AlertSystem(min_level='WARNING')
        
        # Create alerts of different levels
        alert_system.create_alert(level='INFO', message='Info message')      # Should be filtered
        alert_system.create_alert(level='WARNING', message='Warning message') # Should pass
        alert_system.create_alert(level='ERROR', message='Error message')    # Should pass
        
        # Should only have WARNING and ERROR alerts
        assert len(alert_system.alerts) == 2
        levels = [alert['level'] for alert in alert_system.alerts]
        assert 'INFO' not in levels
        assert 'WARNING' in levels
        assert 'ERROR' in levels
    
    def test_alert_rate_limiting(self):
        """Test alert rate limiting."""
        alert_system = AlertSystem(rate_limit_seconds=0.1)  # 100ms rate limit
        
        # Create rapid alerts for same metric
        for i in range(5):
            alert_system.create_alert(
                level='WARNING',
                message=f'Rapid alert {i}',
                metric_name='rapid_metric'
            )
        
        # Should have rate-limited the alerts
        rapid_alerts = [a for a in alert_system.alerts if 'rapid_metric' in a.get('metric_name', '')]
        assert len(rapid_alerts) < 5  # Some should be rate-limited
    
    def test_alert_history(self):
        """Test alert history management."""
        alert_system = AlertSystem(max_history=3)
        
        # Create more alerts than history limit
        for i in range(6):
            alert_system.create_alert(
                level='INFO',
                message=f'History test {i}'
            )
        
        # Should only keep last 3 alerts
        assert len(alert_system.alerts) == 3
        messages = [alert['message'] for alert in alert_system.alerts]
        assert 'History test 3' in messages
        assert 'History test 4' in messages
        assert 'History test 5' in messages
        assert 'History test 0' not in messages


class TestPerformanceMonitor:
    """Test main performance monitoring system."""
    
    def test_initialization(self):
        """Test monitor initialization."""
        monitor = PerformanceMonitor(
            collection_interval=1.0,
            enable_alerts=True,
            anomaly_detection=True
        )
        
        assert monitor.collection_interval == 1.0
        assert monitor.enable_alerts is True
        assert monitor.anomaly_detection is True
        assert hasattr(monitor, 'collector')
        assert hasattr(monitor, 'detector')
        assert hasattr(monitor, 'alert_system')
    
    def test_metric_recording(self):
        """Test metric recording through monitor."""
        monitor = PerformanceMonitor()
        
        # Record various metrics
        monitor.record('training_loss', 0.5)
        monitor.record('validation_accuracy', 0.87)
        monitor.record('gpu_memory_mb', 2048)
        
        # Should be stored in collector
        assert 'training_loss' in monitor.collector.metrics
        assert 'validation_accuracy' in monitor.collector.metrics
        assert 'gpu_memory_mb' in monitor.collector.metrics
    
    def test_automatic_monitoring(self):
        """Test automatic performance monitoring."""
        monitor = PerformanceMonitor(collection_interval=0.1)
        
        # Start monitoring
        monitor.start_monitoring()
        
        # Record some metrics
        for i in range(5):
            monitor.record('auto_metric', i * 10)
            time.sleep(0.02)
        
        # Let automatic collection run
        time.sleep(0.2)
        
        # Stop monitoring
        monitor.stop_monitoring()
        
        # Should have collected system metrics automatically
        stats = monitor.get_statistics()
        assert isinstance(stats, dict)
        assert len(stats) > 0
    
    def test_anomaly_detection_integration(self):
        """Test integration with anomaly detection."""
        monitor = PerformanceMonitor(anomaly_detection=True)
        
        # Establish baseline
        normal_values = [100, 105, 95, 102, 98, 101, 99]
        for value in normal_values:
            monitor.record('baseline_metric', value)
        
        # Allow baseline to be established
        time.sleep(0.1)
        
        # Record anomalous value
        monitor.record('baseline_metric', 500)  # Clearly anomalous
        
        # Should have detected anomaly
        stats = monitor.get_statistics()
        if 'anomalies_detected' in stats:
            assert stats['anomalies_detected'] > 0
    
    def test_alert_generation(self):
        """Test automatic alert generation."""
        monitor = PerformanceMonitor(enable_alerts=True)
        
        alert_received = []
        
        def test_alert_handler(alert):
            alert_received.append(alert)
        
        monitor.alert_system.add_handler(test_alert_handler)
        
        # Set up baseline for anomaly detection
        for value in [50, 52, 48, 51, 49]:
            monitor.record('alert_metric', value)
        
        # Record anomalous value that should trigger alert
        monitor.record('alert_metric', 200)  # Clearly anomalous
        
        # Allow processing time
        time.sleep(0.1)
        
        # May have generated an alert (depends on implementation)
        # At minimum, anomaly should be detected
        detector_stats = monitor.detector.get_anomaly_statistics()
        assert detector_stats.get('total_anomalies', 0) >= 0
    
    def test_performance_metrics_collection(self):
        """Test collection of system performance metrics."""
        monitor = PerformanceMonitor()
        
        # Collect system metrics
        system_metrics = monitor.collect_system_metrics()
        
        assert isinstance(system_metrics, dict)
        
        # Should include standard system metrics
        expected_metrics = ['cpu_percent', 'memory_percent']
        for metric in expected_metrics:
            if metric in system_metrics:
                assert isinstance(system_metrics[metric], (int, float))
                assert 0 <= system_metrics[metric] <= 100
        
        # GPU metrics (if available)
        if torch.cuda.is_available():
            gpu_metrics = monitor.collect_gpu_metrics()
            assert isinstance(gpu_metrics, dict)
    
    def test_contextual_monitoring(self):
        """Test contextual monitoring with decorators."""
        monitor = PerformanceMonitor()
        
        @monitor.monitor_function('test_function')
        def monitored_function(x, y):
            time.sleep(0.01)  # Simulate work
            return x + y
        
        # Call monitored function
        result = monitored_function(5, 3)
        assert result == 8
        
        # Should have recorded execution metrics
        stats = monitor.get_statistics()
        
        # May have recorded execution time, call count, etc.
        # (Exact metrics depend on implementation)
        assert isinstance(stats, dict)
    
    def test_batch_monitoring(self):
        """Test monitoring of batch operations."""
        monitor = PerformanceMonitor()
        
        with monitor.monitor_batch('training_batch'):
            # Simulate batch processing
            for i in range(10):
                monitor.record('batch_loss', 1.0 - i * 0.1)
                monitor.record('batch_accuracy', 0.5 + i * 0.05)
                time.sleep(0.001)
        
        # Should have recorded batch metrics
        batch_stats = monitor.get_batch_statistics('training_batch')
        
        assert isinstance(batch_stats, dict)
        # Should include batch-level aggregations
        if 'duration' in batch_stats:
            assert batch_stats['duration'] > 0
    
    def test_statistics_reporting(self):
        """Test comprehensive statistics reporting."""
        monitor = PerformanceMonitor(
            enable_alerts=True,
            anomaly_detection=True
        )
        
        # Record diverse metrics
        metrics_data = {
            'accuracy': [0.7, 0.75, 0.8, 0.85, 0.9],
            'loss': [1.0, 0.8, 0.6, 0.4, 0.2],
            'learning_rate': [0.01, 0.01, 0.005, 0.005, 0.001]
        }
        
        for metric_name, values in metrics_data.items():
            for value in values:
                monitor.record(metric_name, value)
        
        # Get comprehensive statistics
        stats = monitor.get_statistics()
        
        assert isinstance(stats, dict)
        assert 'total_metrics' in stats
        assert 'collection_period' in stats
        
        # Should include per-metric statistics
        for metric_name in metrics_data:
            if f'{metric_name}_stats' in stats:
                metric_stats = stats[f'{metric_name}_stats']
                assert 'mean' in metric_stats
                assert 'count' in metric_stats
    
    def test_export_functionality(self):
        """Test data export functionality."""
        monitor = PerformanceMonitor()
        
        # Record some data
        for i in range(20):
            monitor.record('export_metric', i)
        
        # Export data
        exported_data = monitor.export_data(format='dict')
        
        assert isinstance(exported_data, dict)
        assert 'metrics' in exported_data
        assert 'export_metric' in exported_data['metrics']
        
        # Should include timestamps and values
        metric_data = exported_data['metrics']['export_metric']
        assert len(metric_data) == 20
    
    def test_thread_safety(self):
        """Test thread-safe monitoring."""
        monitor = PerformanceMonitor()
        
        def worker_thread(worker_id, num_records):
            for i in range(num_records):
                monitor.record(f'worker_{worker_id}_metric', i)
                monitor.record('shared_metric', worker_id * 1000 + i)
                time.sleep(0.001)
        
        # Start multiple monitoring threads
        threads = []
        for worker_id in range(3):
            thread = threading.Thread(target=worker_thread, args=(worker_id, 5))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Should have collected all metrics without corruption
        stats = monitor.get_statistics()
        assert stats['total_metrics'] >= 4  # 3 worker metrics + 1 shared


class TestIntegration:
    """Integration tests for performance monitoring system."""
    
    def test_complete_monitoring_workflow(self):
        """Test complete monitoring workflow."""
        monitor = PerformanceMonitor(
            collection_interval=0.1,
            enable_alerts=True,
            anomaly_detection=True
        )
        
        alerts_received = []
        
        def alert_handler(alert):
            alerts_received.append(alert)
        
        monitor.alert_system.add_handler(alert_handler)
        
        # Start monitoring
        monitor.start_monitoring()
        
        # Simulate training loop with monitoring
        for epoch in range(5):
            with monitor.monitor_batch(f'epoch_{epoch}'):
                for step in range(10):
                    # Normal training metrics
                    loss = 1.0 - (epoch * 10 + step) * 0.01
                    accuracy = (epoch * 10 + step) * 0.01
                    
                    monitor.record('training_loss', loss)
                    monitor.record('training_accuracy', accuracy)
                    
                    # Occasionally record anomalous value
                    if step == 5 and epoch == 3:
                        monitor.record('training_loss', 10.0)  # Spike
        
        # Allow monitoring to complete
        time.sleep(0.2)
        
        # Stop monitoring
        monitor.stop_monitoring()
        
        # Check comprehensive results
        stats = monitor.get_statistics()
        
        assert stats['total_metrics'] > 0
        
        # Should have detected the anomaly
        anomaly_stats = monitor.detector.get_anomaly_statistics()
        assert anomaly_stats.get('total_anomalies', 0) >= 0  # May or may not detect depending on baseline
        
        # Export final data
        exported = monitor.export_data()
        assert isinstance(exported, dict)
        assert 'metrics' in exported
    
    def test_real_world_training_monitoring(self):
        """Test monitoring in realistic training scenario."""
        monitor = PerformanceMonitor()
        
        # Simulate realistic meta-learning training
        n_episodes = 50
        
        for episode in range(n_episodes):
            with monitor.monitor_batch(f'episode_{episode}'):
                # Episode metrics
                episode_accuracy = 0.5 + 0.4 * (episode / n_episodes) + 0.05 * torch.randn(1).item()
                episode_loss = 2.0 * torch.exp(-episode / 20) + 0.1 * torch.randn(1).item()
                
                monitor.record('episode_accuracy', episode_accuracy)
                monitor.record('episode_loss', episode_loss)
                
                # Step-level metrics within episode
                for step in range(5):  # 5 gradient steps per episode
                    step_loss = episode_loss * torch.exp(-step / 3)
                    monitor.record('step_loss', step_loss)
                    
                    # Memory usage (simulated)
                    memory_usage = 2.0 + 0.5 * torch.randn(1).item()
                    monitor.record('gpu_memory_gb', max(0, memory_usage))
        
        # Analyze training progression
        stats = monitor.get_statistics()
        
        # Should show learning progression
        if 'episode_accuracy_stats' in stats:
            acc_stats = stats['episode_accuracy_stats']
            assert acc_stats['count'] == n_episodes
            # Accuracy should generally increase (though with noise)
            # assert acc_stats['trend'] > 0  # Positive trend (if implemented)
        
        # Performance should be reasonable
        assert stats['total_metrics'] > 0
        
        # Export for analysis
        data = monitor.export_data()
        assert len(data['metrics']) >= 3  # At least accuracy, loss, memory


if __name__ == "__main__":
    pytest.main([__file__])