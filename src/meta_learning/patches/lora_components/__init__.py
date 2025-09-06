"""
LoRA Components Module - Modular LoRA Integration
=================================================

This module provides modular LoRA (Low-Rank Adaptation) components for
MAML enhancement, broken down into focused, manageable files.

MODULAR STRUCTURE:
- lora_layers.py: Core LoRA layer implementations (LoRALayer, LoRALinear)
- lora_adapters.py: MAML-specific LoRA adaptation (MAMLLoRAAdapter)
- lora_trainers.py: LLM meta-learning trainers (LLMMetaLearningTrainer)
- lora_utils.py: Utility functions and factory methods

Each component is focused and keeps file sizes under 300 lines.
"""

from .lora_layers import LoRALayer, LoRALinear
from .lora_adapters import MAMLLoRAAdapter
from .lora_trainers import LLMMetaLearningTrainer
from .lora_utils import create_lora_enhanced_maml, apply_lora_patches_to_existing_maml

__all__ = [
    'LoRALayer',
    'LoRALinear', 
    'MAMLLoRAAdapter',
    'LLMMetaLearningTrainer',
    'create_lora_enhanced_maml',
    'apply_lora_patches_to_existing_maml'
]