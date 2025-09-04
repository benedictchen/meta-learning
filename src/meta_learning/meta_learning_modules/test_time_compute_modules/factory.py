"""
📋 Factory
===========

🔬 Research Foundation:  
======================
Based on meta-learning and few-shot learning research:
- Finn, C., Abbeel, P. & Levine, S. (2017). "Model-Agnostic Meta-Learning for Fast Adaptation"
- Snell, J., Swersky, K. & Zemel, R. (2017). "Prototypical Networks for Few-shot Learning"
- Nichol, A., Achiam, J. & Schulman, J. (2018). "On First-Order Meta-Learning Algorithms"
🎯 ELI5 Summary:
This file is an important component in our AI research system! Like different organs 
in your body that work together to keep you healthy, this file has a specific job that 
helps the overall algorithm work correctly and efficiently.

🧪 Technical Details:
===================
Implementation details and technical specifications for this component.
Designed to work seamlessly within the research framework while
maintaining high performance and accuracy standards.

📋 Component Integration:
========================
    ┌──────────┐
    │   This   │
    │Component │ ←→ Other Components
    └──────────┘
         ↑↓
    System Integration

"""
"""
Test-Time Compute Configuration Factory
======================================

Factory functions for creating test-time compute configurations.
Extracted from the monolithic test_time_compute.py for better organization.

Each factory function creates a configuration optimized for specific research methods
and use cases based on published literature.

Author: Benedict Chen (benedict@benedictchen.com)
"""

from .config import TestTimeComputeConfig


def create_process_reward_config() -> TestTimeComputeConfig:
    """
    Create configuration for Process Reward Model verification (Snell et al. 2024).
    
    Research method: Enables process reward model with optimal settings.
    Based on: "Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters"
    """
    config = TestTimeComputeConfig()
    config.compute_strategy = "snell2024"
    config.use_process_reward = True
    config.use_process_reward_model = True
    config.prm_verification_steps = 5
    config.prm_scoring_method = "weighted"
    config.prm_step_penalty = 0.05
    config.reward_weight = 0.4
    return config


def create_consistency_verification_config() -> TestTimeComputeConfig:
    """
    Create configuration for Consistency-Based Verification.
    
    Research method: Enables test-time training with consistency checks.
    Based on: "The Surprising Effectiveness of Test-Time Training for Few-Shot Learning"
    """
    config = TestTimeComputeConfig()
    config.compute_strategy = "akyurek2024"
    config.use_test_time_training = True
    config.ttt_learning_rate = 5e-5
    config.ttt_adaptation_steps = 3
    config.ttt_optimizer = "adamw"
    config.adaptation_weight = 0.6
    config.prm_verification_steps = 4
    return config


def create_gradient_verification_config() -> TestTimeComputeConfig:
    """
    Create configuration for Gradient-Based Step Verification.
    
    Research method: Enables gradient-based reasoning quality assessment.
    """
    config = TestTimeComputeConfig()
    config.compute_strategy = "hybrid"
    config.use_gradient_verification = True
    config.use_chain_of_thought = True
    config.cot_reasoning_steps = 3
    config.cot_temperature = 0.8
    return config


def create_attention_reasoning_config() -> TestTimeComputeConfig:
    """
    Create configuration for Attention-Based Reasoning Path Generation.
    
    Research method: Uses attention mechanisms for reasoning chain generation.
    Based on OpenAI o1 system architecture principles.
    """
    config = TestTimeComputeConfig()
    config.compute_strategy = "openai_o1"
    config.use_chain_of_thought = True
    config.cot_method = "attention_based"
    config.cot_reasoning_steps = 5
    config.cot_temperature = 0.7
    config.cot_self_consistency = True
    config.reasoning_weight = 0.5
    return config


def create_feature_reasoning_config() -> TestTimeComputeConfig:
    """
    Create configuration for Feature-Based Reasoning Decomposition.
    
    Research method: Decomposes reasoning into feature-level operations.
    """
    config = TestTimeComputeConfig()
    config.compute_strategy = "hybrid"
    config.use_chain_of_thought = True
    config.cot_method = "feature_based"
    config.cot_reasoning_steps = 4
    config.cot_temperature = 0.6
    config.cot_self_consistency = True
    return config


def create_prototype_reasoning_config() -> TestTimeComputeConfig:
    """
    Create configuration for Prototype-Distance Reasoning Steps.
    
    Research method: Uses prototype-based distance calculations for reasoning.
    """
    config = TestTimeComputeConfig()
    config.compute_strategy = "hybrid"
    config.use_chain_of_thought = True
    config.cot_method = "prototype_based"
    config.cot_reasoning_steps = 3
    config.cot_temperature = 0.9
    config.cot_self_consistency = True
    return config


def create_comprehensive_config() -> TestTimeComputeConfig:
    """
    Create configuration that enables all implemented research solutions.
    
    COMPREHENSIVE: Combines all research-accurate methods with balanced settings.
    This configuration provides the most thorough test-time compute scaling but
    requires more computational resources.
    """
    config = TestTimeComputeConfig()
    
    # Enable all strategies
    config.compute_strategy = "hybrid"
    
    # Process reward model (Snell et al. 2024)
    config.use_process_reward = True
    config.use_process_reward_model = True
    config.prm_verification_steps = 3
    config.prm_scoring_method = "weighted"
    config.reward_weight = 0.3
    
    # Test-time training (Akyürek et al. 2024)
    config.use_test_time_training = True
    config.ttt_learning_rate = 1e-4
    config.ttt_adaptation_steps = 2
    config.adaptation_weight = 0.4
    
    # Gradient verification
    config.use_gradient_verification = True
    
    # Chain-of-thought reasoning (OpenAI o1 style)
    config.use_chain_of_thought = True
    config.cot_method = "attention_based"  # Default, can be changed
    config.cot_reasoning_steps = 4
    config.cot_temperature = 0.7
    config.cot_self_consistency = True
    config.reasoning_weight = 0.5
    
    # Optimal allocation and distribution updates
    config.use_optimal_allocation = True
    config.use_adaptive_distribution = True
    
    # Enhanced ensemble methods
    config.ensemble_method = "weighted_average"
    config.confidence_weighting = True
    config.diversity_weighting = True
    
    return config


def create_fast_config() -> TestTimeComputeConfig:
    """
    Create a fast configuration with minimal overhead but still research-accurate.
    
    OPTIMIZED: Balanced performance vs accuracy for production use.
    Enables key methods while minimizing computational overhead.
    """
    config = TestTimeComputeConfig()
    config.compute_strategy = "snell2024"
    config.max_compute_budget = 100
    config.min_compute_steps = 3
    
    # Enable one primary method for efficiency
    config.use_chain_of_thought = True
    config.cot_method = "prototype_based"  # Fastest method
    config.cot_reasoning_steps = 2
    config.cot_temperature = 0.8
    
    # Simplified verification
    config.use_process_reward = True
    config.prm_verification_steps = 2
    config.prm_scoring_method = "average"
    
    return config


# Export all factory functions
__all__ = [
    'create_process_reward_config',
    'create_consistency_verification_config',
    'create_gradient_verification_config',
    'create_attention_reasoning_config',
    'create_feature_reasoning_config', 
    'create_prototype_reasoning_config',
    'create_comprehensive_config',
    'create_fast_config'
]