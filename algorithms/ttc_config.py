"""
⚙️ Config
==========

🔬 Research Foundation:  
======================
Based on meta-learning and few-shot learning research:
- Finn, C., Abbeel, P. & Levine, S. (2017). "Model-Agnostic Meta-Learning for Fast Adaptation"
- Snell, J., Swersky, K. & Zemel, R. (2017). "Prototypical Networks for Few-shot Learning"
- Nichol, A., Achiam, J. & Schulman, J. (2018). "On First-Order Meta-Learning Algorithms"
🎯 ELI5 Summary:
Think of this like a control panel for our algorithm! Just like how your TV remote 
has different buttons for volume, channels, and brightness, this file has all the settings 
that control how our AI algorithm behaves. Researchers can adjust these settings to get 
the best results for their specific problem.

🧪 Technical Details:
===================
Implementation details and technical specifications for this component.
Designed to work seamlessly within the research framework while
maintaining high performance and accuracy standards.

⚙️ Configuration Architecture:
==============================
    ┌─────────────────────────┐
    │    USER SETTINGS        │
    ├─────────────────────────┤
    │ • Algorithm Parameters  │
    │ • Performance Options   │
    │ • Research Preferences  │
    │ • Output Formats        │
    └─────────────────────────┘
              ↓
    ┌─────────────────────────┐
    │      ALGORITHM          │
    │    (Configured)         │
    └─────────────────────────┘

"""
"""
Test-Time Compute Configuration
==============================

Configuration classes for test-time compute scaling algorithms.
Extracted from the monolithic test_time_compute.py for better organization.

Based on research from:
- Snell et al. (2024): "Scaling LLM Test-Time Compute Optimally"
- Akyürek et al. (2024): "The Surprising Effectiveness of Test-Time Training"  
- OpenAI o1 system: Reinforcement learning for test-time reasoning

Author: Benedict Chen (benedict@benedictchen.com)
"""

from dataclasses import dataclass
from .strategies import (
    TestTimeComputeStrategy,
    StateFallbackMethod, 
    StateForwardMethod,
    VerificationFallbackMethod
)


@dataclass
class TestTimeComputeConfig:
    """Configuration for test-time compute scaling with research-accurate options."""
    
    # Core Test-Time Compute Strategy
    compute_strategy: TestTimeComputeStrategy = TestTimeComputeStrategy.BASIC
    
    # State Encoding and Forward Pass
    state_encoding_fallback: StateFallbackMethod = StateFallbackMethod.LEARNED
    state_forward_method: StateForwardMethod = StateForwardMethod.ATTENTION
    
    # Verification Methods
    verification_fallback_method: VerificationFallbackMethod = VerificationFallbackMethod.SIMPLE
    
    # Compute Allocation (Snell et al. 2024)
    max_compute_budget: int = 1000
    min_compute_steps: int = 10
    confidence_threshold: float = 0.95
    
    # Process-based Reward Model (Snell et al. 2024)
    use_process_reward: bool = False
    use_process_reward_model: bool = False  
    prm_verification_steps: int = 3
    prm_scoring_method: str = "product"  # "product", "average", "weighted"
    prm_step_penalty: float = 0.1
    reward_weight: float = 0.3
    
    # Process reward solution selection
    process_reward_solution: str = "research_accurate"  # "research_accurate", "simplified", "placeholder"
    
    # Research-accurate Process-based Reward Model options
    prm_quality_estimation_method: str = "step_wise"  # "step_wise", "confidence_based", "gradient_based"
    prm_aggregation_weights: str = "linear_decay"     # "linear_decay", "exponential_decay", "uniform"
    prm_stability_check: bool = True
    prm_numerical_regularization: float = 1e-8
    
    # Simplified Quality Estimation options
    simplified_consistency_method: str = "variance"   # "variance", "entropy", "agreement" 
    simplified_confidence_threshold: float = 0.7
    simplified_smoothing_factor: float = 0.1
    
    # Placeholder options
    placeholder_return_strategy: str = "zeros"        # "zeros", "random", "confidence_based"
    placeholder_shape_matching: bool = True
    placeholder_warning_level: str = "warn"          # "warn", "debug", "silent"
    
    # Test-Time Training (Akyürek et al. 2024)
    use_test_time_training: bool = False
    ttt_learning_rate: float = 1e-4
    ttt_adaptation_steps: int = 3
    ttt_optimizer: str = "adam"  # "adam", "sgd", "adamw"  
    ttt_weight_decay: float = 1e-5
    adaptation_weight: float = 0.4
    
    # Chain-of-Thought Reasoning (OpenAI o1 style)
    use_chain_of_thought: bool = False
    cot_reasoning_steps: int = 5
    cot_temperature: float = 0.7
    cot_self_consistency: bool = True
    reasoning_weight: float = 0.5
    cot_method: str = "attention_based"  # "attention_based", "feature_based", "prototype_based"
    
    # Additional verification options
    use_gradient_verification: bool = False  # Enable gradient-based step verification
    
    # Consistency fallback options
    consistency_fallback_method: str = "confidence"  # "confidence", "variance", "loss", "raise_error"
    consistency_multiple_passes: int = 3  # Number of forward passes for variance estimation
    consistency_min_score: float = 0.0  # Minimum allowable consistency score
    consistency_max_score: float = 1.0  # Maximum allowable consistency score
    require_support_labels: bool = False  # Whether to require support labels for fallback methods
    
    # Bootstrap sampling
    use_bootstrap_sampling: bool = True
    
    # Compute-Optimal Allocation (Snell et al. 2024)
    use_optimal_allocation: bool = False
    allocation_strategy: str = "difficulty_weighted"  # "uniform", "difficulty_weighted", "performance_based"
    difficulty_estimation_method: str = "entropy"  # "entropy", "confidence", "gradient_norm"
    
    # Adaptive Distribution Updates (Snell et al. 2024)
    use_adaptive_distribution: bool = False
    distribution_update_method: str = "confidence_based"  # "confidence_based", "step_based", "hybrid"
    sharpening_factor: float = 1.1
    
    # Ensemble configuration
    ensemble_method: str = "weighted_average"  # "simple_average", "weighted_average", "majority_vote"
    confidence_weighting: bool = True
    diversity_weighting: bool = False
    ensemble_size: int = 5
    
    # Backward compatibility parameters
    compute_allocation_strategy: str = "adaptive"  # adaptive, fixed, exponential
    early_stopping_patience: int = 50
    early_stopping: bool = False  
    difficulty_adaptive: bool = False  
    temperature_scaling: float = 1.0
    
    # Legacy compute strategy selection
    compute_strategy: str = "basic"  # "basic", "snell2024", "akyurek2024", "openai_o1", "hybrid"


# Export configuration class
__all__ = ['TestTimeComputeConfig']