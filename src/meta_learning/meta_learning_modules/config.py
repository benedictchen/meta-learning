"""
⚙️ Config
==========

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
Meta-Learning Configuration System ⚙️🔧
=======================================

🎯 **ELI5 Explanation**:
Think of this like a master control panel for your AI experiments!
Just like a sound engineer has a mixing board with lots of knobs and sliders to control different aspects of music,
this configuration system has settings for different parts of your meta-learning system:

- 🎛️ **Component Configs**: Like individual instrument volume controls (dataset settings, hardware settings)
- 🎚️ **Global Settings**: Like master volume and EQ (learning rates, batch sizes)
- 🎧 **Smart Defaults**: Like pre-saved mixing presets that work well together
- ⚙️ **Easy Tweaking**: Change one setting without breaking everything else

📊 **Configuration Architecture**:
```
MetaLearningConfig (Master Control Panel)
├── 📊 dataset: DatasetConfig (Data Loading Settings)
├── 🧠 hierarchical: HierarchicalConfig (Multi-level Learning)
├── ⚡ test_time_compute: TestTimeComputeConfig (Runtime Optimization)
├── 🎯 adaptive: TaskAdaptiveConfig (Dynamic Task Adaptation)
├── 🔧 maml: MAMLConfig (Meta-Learning Algorithm Settings)
└── 💾 hardware: HardwareConfig (GPU/CPU Optimization)
```

🔬 **Research-Accurate Defaults**:
All default values come from the original research papers:
- **MAML Learning Rates**: Chelsea Finn et al. (2017) - inner_lr=0.01, meta_lr=0.001
- **Batch Sizes**: Follow established few-shot learning protocols
- **Temperature Scaling**: Based on calibration research (Platt 1999, Guo et al. 2017)

💡 **Best Practice Design**:
Follows modular configuration patterns where each component manages its own settings,
preventing configuration conflicts and making experimentation easier.

Author: Benedict Chen (benedict@benedictchen.com)

This module provides a modular configuration system that composes configurations
from different components following Python best practices.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any

# Import component-specific configurations
from .few_shot_modules.hierarchical_components import HierarchicalConfig
from .few_shot_modules.adaptive_components import TaskAdaptiveConfig  
from .test_time_compute import TestTimeComputeConfig
from .utils_modules.configurations import DatasetConfig


@dataclass
class MetaLearningConfig:
    """
    Main configuration that composes all component-specific configurations.
    
    This follows Python best practices for configuration system design where
    each module manages its own configuration and the main config composes them.
    """
    
    # Component Configurations
    hierarchical: HierarchicalConfig = None
    adaptive: TaskAdaptiveConfig = None
    test_time_compute: TestTimeComputeConfig = None
    dataset: DatasetConfig = None
    
    # MAML Configuration
    inner_lr: float = 0.01
    inner_steps: int = 5
    first_order: bool = False               # Use second-order gradients
    
    # Meta-Learning Configuration  
    meta_lr: float = 0.001
    meta_batch_size: int = 4
    
    # Advanced MAML Features
    use_maml_en_llm: bool = False          # MAML for large language models
    adaptive_inner_lr: bool = False         # Adaptive learning rates
    
    # Prototypical Networks Configuration
    use_squared_euclidean: bool = True       # Snell et al. 2017 standard
    distance_temperature: float = 1.0
    prototype_method: str = "mean"           # Snell Equation 1
    multi_scale_features: bool = False
    enable_research_extensions: bool = True
    track_prototype_statistics: bool = False
    
    # Numerical Stability
    max_grad_norm: float = 1.0             # Gradient clipping
    gradient_accumulation_steps: int = 1
    use_mixed_precision: bool = False
    numerical_epsilon: float = 1e-8
    check_for_nan: bool = True
    
    # Logging and Debugging
    log_level: str = "INFO"
    log_performance_metrics: bool = True
    enable_debug_mode: bool = False
    save_intermediate_results: bool = False
    validate_tensor_shapes: bool = True
    
    # Hardware Optimization
    device: str = "auto"                   # auto, cpu, cuda, mps
    use_data_parallel: bool = False
    optimize_memory_usage: bool = True
    clear_cache_between_tasks: bool = True
    monitor_gpu_usage: bool = False
    log_memory_usage: bool = False

    def __post_init__(self):
        """Initialize component configurations with defaults if not provided."""
        if self.hierarchical is None:
            self.hierarchical = HierarchicalConfig()
        if self.adaptive is None:
            self.adaptive = TaskAdaptiveConfig()
        if self.test_time_compute is None:
            self.test_time_compute = TestTimeComputeConfig()
        if self.dataset is None:
            self.dataset = DatasetConfig()


def create_research_accurate_config() -> MetaLearningConfig:
    """
    Create configuration optimized for research accuracy.
    
    Returns:
        MetaLearningConfig: Configuration with all research-based solutions enabled
    """
    hierarchical_config = HierarchicalConfig(
        use_exact_information_theory=True,
        validate_against_papers=True,
        log_theoretical_violations=True
    )
    
    adaptive_config = TaskAdaptiveConfig(
        method="attention_based"
    )
    
    test_time_compute_config = TestTimeComputeConfig(
        use_process_reward=True,
        use_chain_of_thought=True
    )
    
    dataset_config = DatasetConfig(
        require_user_confirmation_for_synthetic=True
    )
    
    return MetaLearningConfig(
        hierarchical=hierarchical_config,
        adaptive=adaptive_config,
        test_time_compute=test_time_compute_config,
        dataset=dataset_config,
        
        # Research-accurate settings
        first_order=False,  # Use second-order gradients
        adaptive_inner_lr=True,
        use_squared_euclidean=True,
        prototype_method="mean",
        
        # Maximum stability and validation
        check_for_nan=True,
        validate_tensor_shapes=True,
        enable_debug_mode=False,
        log_performance_metrics=True
    )


def create_performance_optimized_config() -> MetaLearningConfig:
    """
    Create configuration optimized for performance/speed.
    
    Returns:
        MetaLearningConfig: Configuration with fast approximations
    """
    hierarchical_config = HierarchicalConfig(
        use_exact_information_theory=False,
        validate_against_papers=False
    )
    
    adaptive_config = TaskAdaptiveConfig(
        method="attention_based"
    )
    
    test_time_compute_config = TestTimeComputeConfig(
        use_process_reward=False,
        use_chain_of_thought=False
    )
    
    dataset_config = DatasetConfig()
    
    return MetaLearningConfig(
        hierarchical=hierarchical_config,
        adaptive=adaptive_config,
        test_time_compute=test_time_compute_config,
        dataset=dataset_config,
        
        # Performance-optimized settings
        first_order=True,  # First-order MAML for speed
        adaptive_inner_lr=False,
        
        # Minimal validation
        check_for_nan=False,
        validate_tensor_shapes=False,
        log_performance_metrics=False,
        
        # Performance optimizations
        use_mixed_precision=True,
        optimize_memory_usage=True,
        clear_cache_between_tasks=True
    )


def create_debugging_config() -> MetaLearningConfig:
    """
    Create configuration optimized for debugging and development.
    
    Returns:
        MetaLearningConfig: Configuration with maximum debugging features
    """
    hierarchical_config = HierarchicalConfig(
        validate_against_papers=True,
        log_theoretical_violations=True
    )
    
    adaptive_config = TaskAdaptiveConfig(
        method="attention_based"
    )
    
    test_time_compute_config = TestTimeComputeConfig()
    
    dataset_config = DatasetConfig(
        require_user_confirmation_for_synthetic=True
    )
    
    return MetaLearningConfig(
        hierarchical=hierarchical_config,
        adaptive=adaptive_config,
        test_time_compute=test_time_compute_config,
        dataset=dataset_config,
        
        # Full debugging features
        enable_debug_mode=True,
        save_intermediate_results=True,
        validate_tensor_shapes=True,
        check_for_nan=True,
        
        # Performance monitoring
        monitor_gpu_usage=True,
        log_memory_usage=True,
        
        # Detailed logging
        log_level="DEBUG",
        log_performance_metrics=True
    )


def get_available_configurations() -> Dict[str, List[str]]:
    """
    Get a dictionary of all available configuration options organized by component.
    
    Returns:
        Dict[str, List[str]]: Dictionary mapping components to their available options
    """
    return {
        "Configuration Presets": [
            "research_accurate",
            "performance_optimized", 
            "debugging_focused"
        ],
        
        "Component Configurations": [
            "hierarchical - HierarchicalConfig",
            "adaptive - TaskAdaptiveConfig",
            "test_time_compute - TestTimeComputeConfig", 
            "dataset - DatasetConfig"
        ],
        
        "Research Papers Implemented": [
            "Cover & Thomas 'Elements of Information Theory' (2006)",
            "Shannon 'A Mathematical Theory of Communication' (1948)",
            "Belghazi et al. 'Mutual Information Neural Estimation' (2018)",
            "Vaswani et al. 'Attention Is All You Need' (2017)",
            "Hinton et al. 'Learning Multiple Layers of Representation' (2007)",
            "Snell et al. 'Prototypical Networks for Few-shot Learning' (2017)",
            "Finn et al. 'Model-Agnostic Meta-Learning' (2017)",
            "Snell et al. 'Test-Time Compute Scaling' (2024)"
        ]
    }


def validate_config(config: MetaLearningConfig) -> List[str]:
    """
    Validate a meta-learning configuration and return any warnings or issues.
    
    Args:
        config: Configuration to validate
        
    Returns:
        List[str]: List of validation warnings/issues
    """
    warnings = []
    
    # Check for synthetic data usage
    if hasattr(config.dataset, 'dataset_method') and config.dataset.dataset_method.value == "synthetic":
        if not config.dataset.require_user_confirmation_for_synthetic:
            warnings.append("⚠️  Synthetic data enabled without user confirmation requirement")
        warnings.append("🚨 RESEARCH INVALID: Synthetic data will make results meaningless")
    
    # Check for performance vs accuracy trade-offs
    if not config.hierarchical.use_exact_information_theory:
        warnings.append("⚠️  Using approximations instead of exact information theory")
    
    if config.first_order and config.hierarchical.validate_against_papers:
        warnings.append("⚠️  First-order MAML may not match paper equations exactly")
    
    # Check for debugging overhead
    if config.hierarchical.validate_against_papers and not config.enable_debug_mode:
        warnings.append("💡 Consider enabling debug mode for paper validation")
    
    # Check temperature parameters
    if config.hierarchical.attention_temperature <= 0:
        warnings.append("❌ Attention temperature must be positive")
    
    if config.hierarchical.hierarchy_temperature <= 0:
        warnings.append("❌ Hierarchy temperature must be positive")
    
    # Check compute budget
    if config.test_time_compute.min_compute_steps > config.test_time_compute.max_compute_budget:
        warnings.append("❌ Min compute steps exceeds max budget")
    
    return warnings


def print_configuration_summary():
    """Print a comprehensive summary of all available configuration options."""
    
    print("🔧 META-LEARNING CONFIGURATION SUMMARY")
    print("=" * 60)
    print()
    
    configurations = get_available_configurations()
    
    for category, items in configurations.items():
        print(f"📂 {category}:")
        for item in items:
            print(f"   ✅ {item}")
        print()
    
    print("🎯 USAGE EXAMPLES:")
    print("   # Research-accurate configuration")
    print("   config = create_research_accurate_config()")
    print()
    print("   # Performance-optimized configuration") 
    print("   config = create_performance_optimized_config()")
    print()
    print("   # Custom configuration")
    print("   config = MetaLearningConfig(")
    print("       hierarchical=HierarchicalConfig(use_exact_information_theory=True),")
    print("       adaptive=TaskAdaptiveConfig(method='attention_based'),")
    print("       test_time_compute=TestTimeComputeConfig(use_process_reward=True)")
    print("   )")
    print()
    print("🔍 VALIDATE YOUR CONFIG:")
    print("   warnings = validate_config(config)")
    print("   if warnings:")
    print("       for warning in warnings:")
    print("           print(warning)")
    

if __name__ == "__main__":
    print_configuration_summary()