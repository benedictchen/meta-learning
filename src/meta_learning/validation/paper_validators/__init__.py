"""
Paper Validators Module - Modular Research Accuracy Validation
==============================================================

This module provides modular validation components for testing implementations
against their original research papers. Each algorithm gets its own validator.

MODULAR STRUCTURE:
- maml_validator.py: MAML vs Finn et al. (2017) validation
- protonet_validator.py: ProtoNet vs Snell et al. (2017) validation  
- meta_sgd_validator.py: Meta-SGD vs Li et al. (2017) validation
- lora_validator.py: LoRA vs Hu et al. (2021) validation
- paper_reference.py: Research paper reference utilities
- validation_utils.py: Shared validation utilities

Each validator is focused on one paper and keeps file sizes manageable.
"""

from .paper_reference import ResearchPaperReference
from .validation_utils import ValidationUtils

__all__ = [
    'ResearchPaperReference',
    'ValidationUtils'
]