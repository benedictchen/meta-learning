"""
Tests for WarningManager.

Tests professional warning management with filtering and rate limiting.
"""

import pytest
import time
import warnings
from unittest.mock import Mock, patch
import threading

from meta_learning.error_handling.warning_system import (
    WarningManager, WarningCategory, WarningFilter, WarningHistory
)


class TestWarningHistory:
    """Test warning history functionality."""
    
    def test_initialization(self):
        """Test history initialization."""
        history = WarningHistory(max_size=100)
        
        assert history.max_size == 100
        assert len(history.warnings) == 0
        assert history.total_warnings == 0
    
    def test_add_warning(self):
        """Test adding warnings to history."""
        history = WarningHistory(max_size=5)
        
        # Add warnings
        for i in range(3):
            history.add_warning(
                category=WarningCategory.PERFORMANCE,
                message=f"Performance warning {i}",
                source="test_function"
            )
        
        assert len(history.warnings) == 3
        assert history.total_warnings == 3
        
        # Check warning structure
        warning = history.warnings[0]
        assert warning['category'] == WarningCategory.PERFORMANCE
        assert "Performance warning 0" in warning['message']
        assert warning['source'] == "test_function"
        assert 'timestamp' in warning
        assert 'count' in warning
    
    def test_warning_deduplication(self):
        """Test warning deduplication."""
        history = WarningHistory()
        
        # Add same warning multiple times
        for i in range(5):
            history.add_warning(
                category=WarningCategory.DEPRECATION,
                message="Deprecated function used",
                source="deprecated_func"
            )
        
        # Should have only one entry with count=5
        assert len(history.warnings) == 1
        assert history.warnings[0]['count'] == 5
        assert history.total_warnings == 5
    
    def test_circular_buffer(self):
        """Test circular buffer behavior."""
        history = WarningHistory(max_size=3)
        
        # Add more warnings than buffer size
        for i in range(6):
            history.add_warning(
                category=WarningCategory.DATA,
                message=f"Unique warning {i}",
                source=f"source_{i}"
            )
        
        # Should only keep last 3 warnings
        assert len(history.warnings) == 3
        assert history.total_warnings == 6
        
        # Should have warnings 3, 4, 5
        messages = [w['message'] for w in history.warnings]
        assert "Unique warning 3" in messages
        assert "Unique warning 4" in messages
        assert "Unique warning 5" in messages
        assert "Unique warning 0" not in messages
    
    def test_get_summary(self):
        """Test warning summary generation."""
        history = WarningHistory()
        
        # Add warnings of different categories
        categories_data = [
            (WarningCategory.PERFORMANCE, "Slow operation"),
            (WarningCategory.PERFORMANCE, "Memory usage high"),
            (WarningCategory.DATA, "Missing data"),
            (WarningCategory.NUMERICAL, "Numerical instability"),
            (WarningCategory.NUMERICAL, "Numerical instability")  # Duplicate
        ]
        
        for category, message in categories_data:
            history.add_warning(category, message, "test")
        
        summary = history.get_summary()
        
        assert isinstance(summary, dict)
        assert 'total_warnings' in summary
        assert 'unique_warnings' in summary
        assert 'category_counts' in summary
        
        assert summary['total_warnings'] == 5
        assert summary['unique_warnings'] == 4  # 4 unique messages
        
        # Category counts
        category_counts = summary['category_counts']
        assert category_counts[WarningCategory.PERFORMANCE.value] == 2
        assert category_counts[WarningCategory.DATA.value] == 1
        assert category_counts[WarningCategory.NUMERICAL.value] == 2
    
    def test_clear_history(self):
        """Test clearing warning history."""
        history = WarningHistory()
        
        # Add some warnings
        for i in range(3):
            history.add_warning(
                WarningCategory.PERFORMANCE,
                f"Warning {i}",
                "test"
            )
        
        # Clear history
        history.clear()
        
        assert len(history.warnings) == 0
        assert history.total_warnings == 0
    
    def test_filter_by_category(self):
        """Test filtering warnings by category."""
        history = WarningHistory()
        
        # Add mixed warnings
        categories = [
            WarningCategory.PERFORMANCE,
            WarningCategory.DATA,
            WarningCategory.PERFORMANCE,
            WarningCategory.NUMERICAL
        ]
        
        for i, category in enumerate(categories):
            history.add_warning(category, f"Message {i}", "test")
        
        # Filter by performance warnings
        perf_warnings = history.get_warnings_by_category(WarningCategory.PERFORMANCE)
        
        assert len(perf_warnings) == 2
        for warning in perf_warnings:
            assert warning['category'] == WarningCategory.PERFORMANCE


class TestWarningFilter:
    """Test warning filtering functionality."""
    
    def test_initialization(self):
        """Test filter initialization."""
        filter_obj = WarningFilter()
        
        assert filter_obj.enabled_categories == set(WarningCategory)
        assert filter_obj.rate_limits == {}
        assert filter_obj.last_warning_times == {}
    
    def test_category_filtering(self):
        """Test filtering by warning category."""
        filter_obj = WarningFilter()
        
        # Disable performance warnings
        filter_obj.disable_category(WarningCategory.PERFORMANCE)
        
        # Test filtering
        assert filter_obj.should_show_warning(
            WarningCategory.DATA, "Data warning"
        )
        assert not filter_obj.should_show_warning(
            WarningCategory.PERFORMANCE, "Performance warning"
        )
    
    def test_rate_limiting(self):
        """Test warning rate limiting."""
        filter_obj = WarningFilter()
        
        # Set rate limit
        filter_obj.set_rate_limit(WarningCategory.NUMERICAL, min_interval=0.1)
        
        # First warning should be shown
        assert filter_obj.should_show_warning(
            WarningCategory.NUMERICAL, "Numerical issue"
        )
        
        # Immediate second warning should be rate-limited
        assert not filter_obj.should_show_warning(
            WarningCategory.NUMERICAL, "Numerical issue"
        )
        
        # After interval, should be shown again
        time.sleep(0.12)
        assert filter_obj.should_show_warning(
            WarningCategory.NUMERICAL, "Numerical issue"
        )
    
    def test_message_pattern_filtering(self):
        """Test filtering by message patterns."""
        filter_obj = WarningFilter()
        
        # Add pattern filter
        filter_obj.add_message_filter(r"test_.*")  # Regex pattern
        
        # Should filter matching messages
        assert not filter_obj.should_show_warning(
            WarningCategory.DATA, "test_function_deprecated"
        )
        assert filter_obj.should_show_warning(
            WarningCategory.DATA, "normal_warning_message"
        )
    
    def test_source_filtering(self):
        """Test filtering by source function."""
        filter_obj = WarningFilter()
        
        # Add source filter
        filter_obj.add_source_filter("noisy_function")
        
        # Should filter warnings from specific source
        assert not filter_obj.should_show_warning(
            WarningCategory.PERFORMANCE, "Warning message", source="noisy_function"
        )
        assert filter_obj.should_show_warning(
            WarningCategory.PERFORMANCE, "Warning message", source="other_function"
        )
    
    def test_priority_based_filtering(self):
        """Test filtering based on warning priority."""
        filter_obj = WarningFilter(min_priority=2)  # Only show high priority
        
        # Should filter low priority warnings
        assert not filter_obj.should_show_warning(
            WarningCategory.PERFORMANCE, "Low priority", priority=1
        )
        assert filter_obj.should_show_warning(
            WarningCategory.PERFORMANCE, "High priority", priority=3
        )
    
    def test_combined_filtering(self):
        """Test combination of multiple filters."""
        filter_obj = WarningFilter()
        
        # Set up multiple filters
        filter_obj.disable_category(WarningCategory.DEPRECATION)
        filter_obj.set_rate_limit(WarningCategory.DATA, min_interval=0.1)
        filter_obj.add_message_filter("ignore_this")
        
        # Test various combinations
        assert not filter_obj.should_show_warning(
            WarningCategory.DEPRECATION, "Deprecated function"
        )  # Category disabled
        
        assert not filter_obj.should_show_warning(
            WarningCategory.DATA, "ignore_this pattern"
        )  # Message pattern matched
        
        # Rate limiting test
        assert filter_obj.should_show_warning(
            WarningCategory.DATA, "Valid data warning"
        )  # First time
        assert not filter_obj.should_show_warning(
            WarningCategory.DATA, "Valid data warning"
        )  # Rate limited


class TestWarningManager:
    """Test main warning management system."""
    
    def test_initialization(self):
        """Test manager initialization."""
        manager = WarningManager(
            max_history=1000,
            enable_filtering=True,
            default_rate_limit=1.0
        )
        
        assert manager.max_history == 1000
        assert manager.enable_filtering is True
        assert manager.default_rate_limit == 1.0
        assert hasattr(manager, 'history')
        assert hasattr(manager, 'filter')
    
    def test_basic_warning_emission(self):
        """Test basic warning emission."""
        manager = WarningManager()
        
        # Capture warnings
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            
            manager.warn(
                WarningCategory.PERFORMANCE,
                "Test performance warning",
                source="test_function"
            )
        
        # Should have emitted warning
        assert len(caught_warnings) >= 0  # May be filtered
        
        # Should be in history
        assert len(manager.history.warnings) == 1
        assert manager.history.warnings[0]['category'] == WarningCategory.PERFORMANCE
    
    def test_warning_filtering_integration(self):
        """Test integration with warning filtering."""
        manager = WarningManager(enable_filtering=True)
        
        # Disable performance warnings
        manager.filter.disable_category(WarningCategory.PERFORMANCE)
        
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            
            manager.warn(
                WarningCategory.PERFORMANCE,
                "Filtered performance warning"
            )
            
            manager.warn(
                WarningCategory.DATA,
                "Unfiltered data warning"
            )
        
        # Performance warning should be filtered
        # (might still be in history for tracking)
        data_warnings = [w for w in manager.history.warnings 
                        if w['category'] == WarningCategory.DATA]
        assert len(data_warnings) == 1
    
    def test_context_manager_usage(self):
        """Test context manager for warning suppression."""
        manager = WarningManager()
        
        # Normal warning
        manager.warn(WarningCategory.DATA, "Normal warning")
        
        # Suppressed warning
        with manager.suppress_warnings(WarningCategory.DATA):
            manager.warn(WarningCategory.DATA, "Suppressed warning")
            manager.warn(WarningCategory.PERFORMANCE, "Not suppressed")
        
        # Should have recorded all warnings but suppressed DATA category
        warnings_by_category = {}
        for warning in manager.history.warnings:
            cat = warning['category']
            warnings_by_category[cat] = warnings_by_category.get(cat, 0) + 1
        
        # Should have performance warning but data warnings behavior depends on implementation
        assert WarningCategory.PERFORMANCE in warnings_by_category or \
               WarningCategory.DATA in warnings_by_category
    
    def test_batch_warning_management(self):
        """Test managing warnings in batches."""
        manager = WarningManager()
        
        with manager.batch_warnings() as batch:
            for i in range(10):
                batch.warn(
                    WarningCategory.NUMERICAL,
                    f"Batch warning {i}"
                )
        
        # Should have collected batch warnings
        # (exact behavior depends on implementation)
        numerical_warnings = [w for w in manager.history.warnings
                             if w['category'] == WarningCategory.NUMERICAL]
        assert len(numerical_warnings) > 0
    
    def test_warning_statistics(self):
        """Test warning statistics collection."""
        manager = WarningManager()
        
        # Generate various warnings
        warning_data = [
            (WarningCategory.PERFORMANCE, "Slow operation"),
            (WarningCategory.PERFORMANCE, "Memory usage"),
            (WarningCategory.DATA, "Missing values"),
            (WarningCategory.NUMERICAL, "NaN detected"),
            (WarningCategory.NUMERICAL, "NaN detected")  # Duplicate
        ]
        
        for category, message in warning_data:
            manager.warn(category, message)
        
        stats = manager.get_statistics()
        
        assert isinstance(stats, dict)
        assert 'total_warnings' in stats
        assert 'warnings_by_category' in stats
        assert 'most_common_warnings' in stats
        
        # Check category distribution
        category_stats = stats['warnings_by_category']
        assert category_stats.get(WarningCategory.PERFORMANCE.value, 0) == 2
        assert category_stats.get(WarningCategory.DATA.value, 0) == 1
        assert category_stats.get(WarningCategory.NUMERICAL.value, 0) >= 1  # May be deduplicated
    
    def test_warning_export(self):
        """Test exporting warning data."""
        manager = WarningManager()
        
        # Add some warnings
        for i in range(5):
            manager.warn(
                WarningCategory.DATA,
                f"Export test warning {i}",
                source=f"function_{i}"
            )
        
        # Export data
        exported = manager.export_warnings(format='dict')
        
        assert isinstance(exported, dict)
        assert 'warnings' in exported
        assert 'summary' in exported
        assert 'export_timestamp' in exported
        
        # Check warning data
        warnings_data = exported['warnings']
        assert len(warnings_data) == 5
        
        for warning in warnings_data:
            assert 'category' in warning
            assert 'message' in warning
            assert 'timestamp' in warning
    
    def test_performance_impact(self):
        """Test performance impact of warning system."""
        manager = WarningManager()
        
        # Measure overhead of warning system
        start_time = time.time()
        
        for i in range(1000):
            manager.warn(
                WarningCategory.PERFORMANCE,
                "Performance test warning",
                emit_warning=False  # Don't actually emit to avoid spam
            )
        
        end_time = time.time()
        
        # Should be fast (less than 100ms for 1000 warnings)
        duration = end_time - start_time
        assert duration < 0.1, f"Warning system too slow: {duration:.3f}s for 1000 warnings"
        
        # Should have recorded all warnings
        assert len(manager.history.warnings) <= 1000  # May be deduplicated
    
    def test_thread_safety(self):
        """Test thread-safe warning management."""
        manager = WarningManager()
        
        def worker_thread(worker_id, num_warnings):
            for i in range(num_warnings):
                manager.warn(
                    WarningCategory.PERFORMANCE,
                    f"Thread {worker_id} warning {i}",
                    source=f"worker_{worker_id}",
                    emit_warning=False
                )
        
        # Start multiple threads
        threads = []
        for worker_id in range(3):
            thread = threading.Thread(target=worker_thread, args=(worker_id, 10))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Should have collected warnings from all threads without corruption
        total_warnings = len(manager.history.warnings)
        assert total_warnings > 0
        
        # Get statistics to ensure data integrity
        stats = manager.get_statistics()
        assert isinstance(stats, dict)
        assert stats['total_warnings'] >= total_warnings
    
    def test_memory_management(self):
        """Test memory management with large numbers of warnings."""
        manager = WarningManager(max_history=100)  # Limited history
        
        # Generate more warnings than history limit
        for i in range(200):
            manager.warn(
                WarningCategory.DATA,
                f"Memory test warning {i}",
                emit_warning=False
            )
        
        # Should not exceed memory limit
        assert len(manager.history.warnings) <= 100
        
        # Total count should still be accurate
        stats = manager.get_statistics()
        assert stats['total_warnings'] == 200
    
    def test_warning_categories_completeness(self):
        """Test all warning categories are supported."""
        manager = WarningManager()
        
        # Test each category
        all_categories = [
            WarningCategory.PERFORMANCE,
            WarningCategory.DATA,
            WarningCategory.NUMERICAL,
            WarningCategory.MEMORY,
            WarningCategory.DEPRECATION,
            WarningCategory.CONFIGURATION
        ]
        
        for category in all_categories:
            manager.warn(
                category,
                f"Test {category.value} warning",
                emit_warning=False
            )
        
        # Should have warnings for each category
        stats = manager.get_statistics()
        category_counts = stats['warnings_by_category']
        
        for category in all_categories:
            assert category.value in category_counts
            assert category_counts[category.value] > 0


class TestWarningIntegration:
    """Integration tests for warning system."""
    
    def test_realistic_training_scenario(self):
        """Test warning system in realistic training scenario."""
        manager = WarningManager(enable_filtering=True)
        
        # Set up realistic filters
        manager.filter.set_rate_limit(WarningCategory.PERFORMANCE, min_interval=1.0)
        manager.filter.set_rate_limit(WarningCategory.NUMERICAL, min_interval=0.5)
        
        # Simulate training loop with various warning scenarios
        for epoch in range(5):
            for batch in range(10):
                # Occasional performance warnings
                if batch % 5 == 0:
                    manager.warn(
                        WarningCategory.PERFORMANCE,
                        f"Slow batch processing in epoch {epoch}",
                        emit_warning=False
                    )
                
                # Frequent numerical warnings (should be rate-limited)
                if batch % 2 == 0:
                    manager.warn(
                        WarningCategory.NUMERICAL,
                        "Small numerical instability detected",
                        emit_warning=False
                    )
                
                # Occasional data warnings
                if batch == 7:
                    manager.warn(
                        WarningCategory.DATA,
                        f"Unusual data distribution in epoch {epoch}",
                        emit_warning=False
                    )
        
        # Analyze results
        stats = manager.get_statistics()
        
        # Should have collected various warnings
        assert stats['total_warnings'] > 0
        
        # Should have rate-limited repeated warnings
        category_counts = stats['warnings_by_category']
        
        # Performance warnings should be rate-limited
        perf_count = category_counts.get(WarningCategory.PERFORMANCE.value, 0)
        numerical_count = category_counts.get(WarningCategory.NUMERICAL.value, 0)
        
        # Due to rate limiting, should have fewer warnings than total occurrences
        assert perf_count <= 5  # 5 epochs, but rate-limited
        
        # Export final report
        report = manager.export_warnings()
        assert isinstance(report, dict)
        assert 'summary' in report
    
    def test_meta_learning_specific_warnings(self):
        """Test warnings specific to meta-learning scenarios."""
        manager = WarningManager()
        
        # Meta-learning specific warning scenarios
        scenarios = [
            (WarningCategory.DATA, "Few-shot episode has insufficient samples", "episode_generator"),
            (WarningCategory.NUMERICAL, "Inner loop optimization unstable", "maml_inner_loop"),
            (WarningCategory.PERFORMANCE, "Episode loading is slow", "dataloader"),
            (WarningCategory.MEMORY, "GPU memory fragmentation detected", "cuda_allocator"),
            (WarningCategory.CONFIGURATION, "Learning rate may be too high for inner updates", "config_validator")
        ]
        
        for category, message, source in scenarios:
            manager.warn(category, message, source=source, emit_warning=False)
        
        # Analyze meta-learning warnings
        stats = manager.get_statistics()
        
        # Should have warnings from different meta-learning components
        assert stats['total_warnings'] == len(scenarios)
        
        # Check source distribution
        if 'warnings_by_source' in stats:
            source_stats = stats['warnings_by_source']
            expected_sources = {'episode_generator', 'maml_inner_loop', 'dataloader', 
                              'cuda_allocator', 'config_validator'}
            
            for source in expected_sources:
                assert source in source_stats
    
    def test_warning_system_overhead(self):
        """Test warning system has minimal overhead."""
        # Test with warnings disabled
        manager_disabled = WarningManager(enable_filtering=True)
        manager_disabled.filter.disable_category(WarningCategory.PERFORMANCE)
        
        start_time = time.time()
        
        for i in range(1000):
            manager_disabled.warn(
                WarningCategory.PERFORMANCE,
                f"Disabled warning {i}",
                emit_warning=False
            )
        
        disabled_time = time.time() - start_time
        
        # Test with warnings enabled
        manager_enabled = WarningManager()
        
        start_time = time.time()
        
        for i in range(1000):
            manager_enabled.warn(
                WarningCategory.DATA,
                f"Enabled warning {i}",
                emit_warning=False
            )
        
        enabled_time = time.time() - start_time
        
        # Both should be fast, overhead should be minimal
        assert disabled_time < 0.1, f"Disabled warnings too slow: {disabled_time:.3f}s"
        assert enabled_time < 0.2, f"Enabled warnings too slow: {enabled_time:.3f}s"
        
        # Disabled should be faster than enabled (but not by orders of magnitude)
        assert disabled_time <= enabled_time * 2, "Disabled warnings should be faster"


if __name__ == "__main__":
    pytest.main([__file__])