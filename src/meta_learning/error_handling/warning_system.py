"""
Professional warning management system with filtering and categorization.

Provides intelligent warning filtering, categorization, and suppression
to reduce noise while maintaining important notifications.
"""

import warnings
import time
from enum import Enum
from typing import Dict, Any, List, Set, Optional, Callable
from collections import defaultdict, deque
from dataclasses import dataclass


class WarningLevel(Enum):
    """Warning severity levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    DEPRECATION = "deprecation"
    PERFORMANCE = "performance"
    COMPATIBILITY = "compatibility"


@dataclass
class WarningRecord:
    """Record of a warning event."""
    level: WarningLevel
    category: str
    message: str
    filename: str
    lineno: int
    timestamp: float
    count: int = 1


class WarningFilter:
    """Intelligent warning filtering system."""
    
    def __init__(self):
        self.suppressed_categories = set()
        self.suppressed_messages = set()
        self.rate_limits = {}  # category -> (max_per_minute, current_count, reset_time)
        self.warning_patterns = {}
    
    def suppress_category(self, category: str) -> None:
        """Suppress all warnings from a category."""
        self.suppressed_categories.add(category)
    
    def suppress_message(self, message_pattern: str) -> None:
        """Suppress warnings matching message pattern."""
        self.suppressed_messages.add(message_pattern)
    
    def set_rate_limit(self, category: str, max_per_minute: int) -> None:
        """Set rate limit for warning category."""
        self.rate_limits[category] = [max_per_minute, 0, time.time() + 60]
    
    def should_show_warning(self, category: str, message: str) -> bool:
        """Check if warning should be shown based on filters."""
        # Check category suppression
        if category in self.suppressed_categories:
            return False
        
        # Check message suppression
        for pattern in self.suppressed_messages:
            if pattern in message:
                return False
        
        # Check rate limits
        if category in self.rate_limits:
            max_count, current_count, reset_time = self.rate_limits[category]
            
            current_time = time.time()
            if current_time > reset_time:
                # Reset rate limit window
                self.rate_limits[category][1] = 0
                self.rate_limits[category][2] = current_time + 60
                current_count = 0
            
            if current_count >= max_count:
                return False
            
            # Increment count
            self.rate_limits[category][1] += 1
        
        return True


class WarningManager:
    """
    Professional warning management system.
    
    Features:
    - Intelligent categorization and filtering
    - Rate limiting to prevent spam
    - Warning aggregation and reporting
    - Integration with Python warnings system
    """
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.warning_history = deque(maxlen=max_history)
        self.warning_counts = defaultdict(int)
        self.filter = WarningFilter()
        self.callbacks = defaultdict(list)
        
        # Warning categorization patterns
        self.category_patterns = {
            WarningLevel.DEPRECATION: [
                "deprecated", "deprecation", "will be removed", "no longer supported"
            ],
            WarningLevel.PERFORMANCE: [
                "performance", "slow", "inefficient", "optimization", "memory"
            ],
            WarningLevel.COMPATIBILITY: [
                "compatibility", "version", "upgrade", "downgrade", "incompatible"
            ]
        }
        
        # Install warning handler
        self._install_warning_handler()
    
    def _install_warning_handler(self) -> None:
        """Install custom warning handler."""
        self._original_showwarning = warnings.showwarning
        warnings.showwarning = self._handle_warning
    
    def _handle_warning(self, 
                       message, 
                       category, 
                       filename, 
                       lineno, 
                       file=None, 
                       line=None) -> None:
        """Custom warning handler."""
        warning_level = self._categorize_warning(str(message), category.__name__)
        category_name = category.__name__
        
        # Check if warning should be shown
        if not self.filter.should_show_warning(category_name, str(message)):
            return
        
        # Create warning record
        record = WarningRecord(
            level=warning_level,
            category=category_name,
            message=str(message),
            filename=filename,
            lineno=lineno,
            timestamp=time.time()
        )
        
        # Check if we've seen this warning before
        warning_key = f"{category_name}:{message}:{filename}:{lineno}"
        if warning_key in self.warning_counts:
            self.warning_counts[warning_key] += 1
            # Update existing record
            for existing_record in reversed(self.warning_history):
                if (existing_record.category == category_name and 
                    existing_record.message == str(message) and
                    existing_record.filename == filename and
                    existing_record.lineno == lineno):
                    existing_record.count = self.warning_counts[warning_key]
                    break
        else:
            self.warning_counts[warning_key] = 1
            self.warning_history.append(record)
        
        # Call callbacks
        for callback in self.callbacks[warning_level]:
            try:
                callback(record)
            except Exception as e:
                print(f"Warning callback error: {e}")
        
        # Decide whether to show the warning
        if self._should_display_warning(record):
            # Use original warning display
            self._original_showwarning(message, category, filename, lineno, file, line)
    
    def _categorize_warning(self, message: str, category: str) -> WarningLevel:
        """Categorize warning based on message content."""
        message_lower = message.lower()
        
        for level, patterns in self.category_patterns.items():
            for pattern in patterns:
                if pattern in message_lower:
                    return level
        
        # Default categorization based on warning type
        if "DeprecationWarning" in category:
            return WarningLevel.DEPRECATION
        elif "PerformanceWarning" in category:
            return WarningLevel.PERFORMANCE
        elif "UserWarning" in category:
            return WarningLevel.WARNING
        else:
            return WarningLevel.INFO
    
    def _should_display_warning(self, record: WarningRecord) -> bool:
        """Determine if warning should be displayed to user."""
        # Always show critical warnings
        if record.level in [WarningLevel.DEPRECATION, WarningLevel.WARNING]:
            return True
        
        # Show performance warnings but limit frequency
        if record.level == WarningLevel.PERFORMANCE and record.count <= 3:
            return True
        
        # Show info warnings occasionally
        if record.level == WarningLevel.INFO and record.count == 1:
            return True
        
        return False
    
    def emit_warning(self, 
                    level: WarningLevel,
                    message: str,
                    category: Optional[str] = None) -> None:
        """Emit a custom warning through the system."""
        if category is None:
            category = "MetaLearningWarning"
        
        # Create synthetic warning
        record = WarningRecord(
            level=level,
            category=category,
            message=message,
            filename="<synthetic>",
            lineno=0,
            timestamp=time.time()
        )
        
        if self.filter.should_show_warning(category, message):
            self.warning_history.append(record)
            
            # Call callbacks
            for callback in self.callbacks[level]:
                try:
                    callback(record)
                except Exception:
                    pass
            
            # Display if appropriate
            if self._should_display_warning(record):
                print(f"[{level.value.upper()}] {category}: {message}")
    
    def add_callback(self, level: WarningLevel, callback: Callable[[WarningRecord], None]) -> None:
        """Add callback for specific warning level."""
        self.callbacks[level].append(callback)
    
    def suppress_category(self, category: str) -> None:
        """Suppress warnings from specific category."""
        self.filter.suppress_category(category)
    
    def suppress_message_pattern(self, pattern: str) -> None:
        """Suppress warnings matching message pattern."""
        self.filter.suppress_message(pattern)
    
    def set_rate_limit(self, category: str, max_per_minute: int) -> None:
        """Set rate limit for warning category."""
        self.filter.set_rate_limit(category, max_per_minute)
    
    def get_warning_summary(self) -> Dict[str, Any]:
        """Get summary of warning activity."""
        if not self.warning_history:
            return {"total_warnings": 0}
        
        # Count by level
        level_counts = defaultdict(int)
        category_counts = defaultdict(int)
        
        for record in self.warning_history:
            level_counts[record.level.value] += record.count
            category_counts[record.category] += record.count
        
        # Recent warnings (last hour)
        cutoff_time = time.time() - 3600
        recent_warnings = [
            record for record in self.warning_history 
            if record.timestamp >= cutoff_time
        ]
        
        return {
            "total_warnings": sum(record.count for record in self.warning_history),
            "unique_warnings": len(self.warning_history),
            "by_level": dict(level_counts),
            "by_category": dict(category_counts),
            "recent_warnings": len(recent_warnings),
            "most_frequent": self._get_most_frequent_warnings(5)
        }
    
    def _get_most_frequent_warnings(self, limit: int) -> List[Dict[str, Any]]:
        """Get most frequently occurring warnings."""
        warnings_by_frequency = sorted(
            self.warning_history,
            key=lambda r: r.count,
            reverse=True
        )
        
        return [
            {
                "message": record.message[:100] + ("..." if len(record.message) > 100 else ""),
                "category": record.category,
                "level": record.level.value,
                "count": record.count
            }
            for record in warnings_by_frequency[:limit]
        ]
    
    def generate_warning_report(self) -> str:
        """Generate detailed warning report."""
        summary = self.get_warning_summary()
        
        if summary["total_warnings"] == 0:
            return "No warnings recorded."
        
        report = ["Warning System Report", "=" * 25, ""]
        
        report.append(f"Total Warnings: {summary['total_warnings']}")
        report.append(f"Unique Warnings: {summary['unique_warnings']}")
        report.append(f"Recent Warnings (1h): {summary['recent_warnings']}")
        report.append("")
        
        report.append("By Level:")
        for level, count in summary['by_level'].items():
            report.append(f"  {level}: {count}")
        report.append("")
        
        report.append("By Category:")
        for category, count in sorted(summary['by_category'].items(), key=lambda x: x[1], reverse=True)[:5]:
            report.append(f"  {category}: {count}")
        report.append("")
        
        if summary['most_frequent']:
            report.append("Most Frequent Warnings:")
            for warning in summary['most_frequent']:
                report.append(f"  [{warning['level']}] {warning['category']}: {warning['message']} (x{warning['count']})")
        
        return "\n".join(report)
    
    def clear_history(self) -> None:
        """Clear warning history."""
        self.warning_history.clear()
        self.warning_counts.clear()
    
    def restore_warnings(self) -> None:
        """Restore original warning system."""
        if hasattr(self, '_original_showwarning'):
            warnings.showwarning = self._original_showwarning


def create_warning_filter(suppress_categories: Optional[List[str]] = None,
                         suppress_patterns: Optional[List[str]] = None,
                         rate_limits: Optional[Dict[str, int]] = None) -> WarningManager:
    """
    Create configured warning manager.
    
    Args:
        suppress_categories: Categories to suppress completely
        suppress_patterns: Message patterns to suppress
        rate_limits: Rate limits per category (category -> max_per_minute)
    """
    manager = WarningManager()
    
    if suppress_categories:
        for category in suppress_categories:
            manager.suppress_category(category)
    
    if suppress_patterns:
        for pattern in suppress_patterns:
            manager.suppress_message_pattern(pattern)
    
    if rate_limits:
        for category, limit in rate_limits.items():
            manager.set_rate_limit(category, limit)
    
    return manager