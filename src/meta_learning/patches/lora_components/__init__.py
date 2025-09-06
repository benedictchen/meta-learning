"""
LoRA Components Module - Modular LoRA Integration
=================================================

This module provides modular LoRA (Low-Rank Adaptation) components for
MAML enhancement, broken down into focused, manageable files.

MODULAR STRUCTURE:
- lora_layers.py: Core LoRA layer implementations (IMPLEMENTED)
- lora_adapters.py: MAML-specific LoRA adaptation (TODO)
- lora_trainers.py: LLM meta-learning trainers (TODO)
- lora_utils.py: Utility functions and factory methods (TODO)

Each component is focused and keeps file sizes under 300 lines.
"""

# Import only implemented modules
from .lora_layers import LoRALayer, LoRALinear

__all__ = [
    'LoRALayer',
    'LoRALinear'
]

# TODO: Add other components as they are implemented
# from .lora_adapters import MAMLLoRAAdapter
# from .lora_trainers import LLMMetaLearningTrainer
# from .lora_utils import create_lora_enhanced_maml, apply_lora_patches_to_existing_maml