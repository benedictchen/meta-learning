"""
Shared Utilities Module - Common Functionality
==============================================

This module provides shared utilities used across multiple components
of the meta-learning package to avoid code duplication.

MODULAR STRUCTURE:
- tensor_utils.py: Tensor manipulation utilities (IMPLEMENTED)
- validation_helpers.py: Common validation functions (TODO)
- logging_utils.py: Enhanced logging functionality (TODO)
- config_utils.py: Configuration management helpers (TODO)
- math_helpers.py: Mathematical computation utilities (TODO)

All utilities are designed to be lightweight and focused.
"""

# Import only implemented modules
from .tensor_utils import TensorUtils

__all__ = [
    'TensorUtils'
]

# TODO: Add other utilities as they are implemented
# from .validation_helpers import ValidationHelpers
# from .logging_utils import LoggingUtils  
# from .config_utils import ConfigUtils
# from .math_helpers import MathHelpers