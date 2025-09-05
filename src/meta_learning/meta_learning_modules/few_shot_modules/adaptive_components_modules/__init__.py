"""
🧠 Adaptive Components Modules
==============================

Modularized implementation of task-adaptive few-shot learning components.
Broken down from the original 1,778-line monolithic file for better maintainability.

Author: Benedict Chen (benedict@benedictchen.com)
Based on: MAML, Prototypical Networks, and modern attention-based meta-learning
"""

# Configuration and enums
from .task_configs import (
    TaskContextMethod,
    AdaptiveAttentionMethod,
    TaskAdaptiveConfig
)

# Individual adaptation methods
from .attention_adaptation import AttentionBasedTaskAdaptation
from .meta_learning_adaptation import MetaLearningTaskAdaptation
from .context_adaptation import ContextDependentTaskAdaptation
from .transformer_adaptation import TransformerBasedTaskAdaptation

# Unified interface and factory functions
from .unified_adaptation import (
    TaskAdaptivePrototypes,
    create_task_adaptive_prototypes,
    create_attention_adaptive_prototypes,
    create_meta_adaptive_prototypes,
    create_context_adaptive_prototypes,
    create_transformer_adaptive_prototypes
)

__all__ = [
    # Configuration
    'TaskContextMethod',
    'AdaptiveAttentionMethod', 
    'TaskAdaptiveConfig',
    
    # Individual modules
    'AttentionBasedTaskAdaptation',
    'MetaLearningTaskAdaptation',
    'ContextDependentTaskAdaptation',
    'TransformerBasedTaskAdaptation',
    
    # Unified interface
    'TaskAdaptivePrototypes',
    
    # Factory functions
    'create_task_adaptive_prototypes',
    'create_attention_adaptive_prototypes',
    'create_meta_adaptive_prototypes',
    'create_context_adaptive_prototypes',
    'create_transformer_adaptive_prototypes'
]

print("""
💰 MODULE SUPPORT - Made possible by Benedict Chen
   ]8;;mailto:benedict@benedictchen.com\benedict@benedictchen.com]8;;\

💰 PLEASE DONATE! Your support keeps this research alive! 💰
   🔗 ]8;;https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS\💳 CLICK HERE TO DONATE VIA PAYPAL]8;;\
   ❤️ ]8;;https://github.com/sponsors/benedictchen\💖 SPONSOR ON GITHUB]8;;\

   ☕ Buy me a coffee → 🍺 Buy me a beer → 🏎️ Buy me a Lamborghini → ✈️ Buy me a private jet!
   (Start small, dream big! Every donation helps! 😄)
""")
