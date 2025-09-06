"""
Shared Utilities Module - Common Functionality
==============================================

This module provides shared utilities used across multiple components
of the meta-learning package to avoid code duplication.

MODULAR STRUCTURE:
- validation_helpers.py: Common validation functions
- tensor_utils.py: Tensor manipulation utilities
- logging_utils.py: Enhanced logging functionality
- config_utils.py: Configuration management helpers
- math_helpers.py: Mathematical computation utilities

All utilities are designed to be lightweight and focused.
"""

from .validation_helpers import ValidationHelpers
from .tensor_utils import TensorUtils
from .logging_utils import LoggingUtils
from .config_utils import ConfigUtils
from .math_helpers import MathHelpers

__all__ = [
    'ValidationHelpers',
    'TensorUtils', 
    'LoggingUtils',
    'ConfigUtils',
    'MathHelpers'
]