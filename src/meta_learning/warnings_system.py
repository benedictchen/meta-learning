"""
Warning system for meta-learning configurations.

This module provides a centralized warning system to help users identify
potentially suboptimal configurations and common pitfalls.
"""

import warnings
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum


class WarningLevel(Enum):
    """Warning severity levels."""
    INFO = "info"
    WARNING = "warning" 
    ERROR = "error"


@dataclass
class ConfigurationWarning:
    """Structured warning information."""
    level: WarningLevel
    category: str
    message: str
    suggestion: Optional[str] = None
    parameter: Optional[str] = None
    value: Optional[Any] = None


class MetaLearningWarnings:
    """Centralized warning system for meta-learning configurations."""
    
    def __init__(self, enabled: bool = True):
        """
        Initialize warning system.
        
        Args:
            enabled: Whether warnings are enabled
        """
        self.enabled = enabled
        self._warned_configurations = set()
    
    def warn_if_suboptimal_few_shot(
        self,
        n_way: int,
        k_shot: int,
        n_query: Optional[int] = None
    ) -> List[ConfigurationWarning]:
        """
        Check for suboptimal few-shot learning configurations.
        
        Args:
            n_way: Number of classes
            k_shot: Number of support examples per class
            n_query: Number of query examples per class
            
        Returns:
            List of configuration warnings
        """
        warnings_list = []
        
        # Very challenging few-shot setting
        if n_way >= 10 and k_shot == 1:
            warnings_list.append(ConfigurationWarning(
                level=WarningLevel.WARNING,
                category="few_shot_difficulty",
                message=f"{n_way}-way 1-shot is very challenging",
                suggestion="Consider increasing k_shot or reducing n_way for better performance",
                parameter="n_way,k_shot",
                value=(n_way, k_shot)
            ))
        
        # Extremely high n_way
        if n_way > 20:
            warnings_list.append(ConfigurationWarning(
                level=WarningLevel.WARNING,
                category="high_n_way",
                message=f"n_way={n_way} is unusually high for few-shot learning",
                suggestion="Most benchmarks use n_way <= 20. Consider reducing for faster training",
                parameter="n_way",
                value=n_way
            ))
        
        # Extremely high k_shot
        if k_shot > 10:
            warnings_list.append(ConfigurationWarning(
                level=WarningLevel.INFO,
                category="high_k_shot",
                message=f"k_shot={k_shot} is higher than typical few-shot scenarios",
                suggestion="Consider if this is still 'few-shot' learning vs standard classification",
                parameter="k_shot", 
                value=k_shot
            ))
        
        # Imbalanced query set
        if n_query is not None and n_query < 5:
            warnings_list.append(ConfigurationWarning(
                level=WarningLevel.WARNING,
                category="small_query_set",
                message=f"n_query={n_query} may give unreliable accuracy estimates",
                suggestion="Consider using n_query >= 10 for more stable evaluation",
                parameter="n_query",
                value=n_query
            ))
        
        self._emit_warnings(warnings_list)
        return warnings_list
    
    def warn_if_suboptimal_distance_config(
        self,
        distance: str,
        tau: float
    ) -> List[ConfigurationWarning]:
        """
        Check for suboptimal distance metric configurations.
        
        Args:
            distance: Distance metric name
            tau: Temperature parameter
            
        Returns:
            List of configuration warnings
        """
        warnings_list = []
        
        # Temperature too low for cosine distance
        if distance == "cosine" and tau < 0.1:
            warnings_list.append(ConfigurationWarning(
                level=WarningLevel.WARNING,
                category="low_temperature",
                message=f"tau={tau} is very low for cosine distance",
                suggestion="Low temperature may cause overconfident predictions. Consider tau >= 0.5",
                parameter="tau",
                value=tau
            ))
        
        # Temperature too high for cosine distance
        elif distance == "cosine" and tau > 10:
            warnings_list.append(ConfigurationWarning(
                level=WarningLevel.INFO,
                category="high_temperature",
                message=f"tau={tau} is very high for cosine distance",
                suggestion="High temperature may cause underconfident predictions. Consider tau <= 5.0",
                parameter="tau",
                value=tau
            ))
        
        # Temperature issues for squared Euclidean
        elif distance == "sqeuclidean":
            if tau < 0.01:
                warnings_list.append(ConfigurationWarning(
                    level=WarningLevel.ERROR,
                    category="numerical_instability",
                    message=f"tau={tau} is extremely low for squared Euclidean distance",
                    suggestion="May cause numerical instability. Consider tau >= 0.1",
                    parameter="tau",
                    value=tau
                ))
            elif tau > 100:
                warnings_list.append(ConfigurationWarning(
                    level=WarningLevel.WARNING,
                    category="high_temperature",
                    message=f"tau={tau} is very high for squared Euclidean distance",
                    suggestion="Consider tau <= 10.0 for better calibration",
                    parameter="tau",
                    value=tau
                ))
        
        self._emit_warnings(warnings_list)
        return warnings_list
    
    def warn_if_suboptimal_maml_config(
        self,
        inner_lr: float,
        inner_steps: int,
        outer_lr: float
    ) -> List[ConfigurationWarning]:
        """
        Check for suboptimal MAML configurations.
        
        Args:
            inner_lr: Inner loop learning rate
            inner_steps: Number of inner loop steps
            outer_lr: Outer loop learning rate
            
        Returns:
            List of configuration warnings
        """
        warnings_list = []
        
        # Inner learning rate issues
        if inner_lr > 0.1:
            warnings_list.append(ConfigurationWarning(
                level=WarningLevel.WARNING,
                category="high_inner_lr",
                message=f"inner_lr={inner_lr} is very high for MAML",
                suggestion="High inner learning rates can cause instability. Consider inner_lr <= 0.01",
                parameter="inner_lr",
                value=inner_lr
            ))
        elif inner_lr < 1e-5:
            warnings_list.append(ConfigurationWarning(
                level=WarningLevel.WARNING,
                category="low_inner_lr",
                message=f"inner_lr={inner_lr} is very low for MAML",
                suggestion="Very low learning rates may prevent effective adaptation",
                parameter="inner_lr",
                value=inner_lr
            ))
        
        # Inner steps
        if inner_steps > 10:
            warnings_list.append(ConfigurationWarning(
                level=WarningLevel.INFO,
                category="many_inner_steps",
                message=f"inner_steps={inner_steps} is higher than typical MAML",
                suggestion="Most MAML papers use 1-5 inner steps. More steps increase computation",
                parameter="inner_steps",
                value=inner_steps
            ))
        elif inner_steps < 1:
            warnings_list.append(ConfigurationWarning(
                level=WarningLevel.ERROR,
                category="no_adaptation",
                message=f"inner_steps={inner_steps} means no adaptation",
                suggestion="MAML requires at least 1 inner step for adaptation",
                parameter="inner_steps", 
                value=inner_steps
            ))
        
        # Learning rate ratio
        if outer_lr > inner_lr * 10:
            warnings_list.append(ConfigurationWarning(
                level=WarningLevel.WARNING,
                category="lr_ratio_imbalance",
                message=f"outer_lr/inner_lr ratio is very high ({outer_lr/inner_lr:.1f})",
                suggestion="Large ratio may cause meta-learning instability",
                parameter="outer_lr,inner_lr",
                value=(outer_lr, inner_lr)
            ))
        
        self._emit_warnings(warnings_list)
        return warnings_list
    
    def warn_if_suboptimal_model_config(
        self,
        model_info: Dict[str, Any]
    ) -> List[ConfigurationWarning]:
        """
        Check for suboptimal model configurations.
        
        Args:
            model_info: Dictionary with model information
            
        Returns:
            List of configuration warnings
        """
        warnings_list = []
        
        # Check for BatchNorm layers
        if model_info.get("has_batchnorm", False):
            warnings_list.append(ConfigurationWarning(
                level=WarningLevel.WARNING,
                category="batchnorm_episodic",
                message="Model contains BatchNorm layers",
                suggestion="Consider using GroupNorm or LayerNorm for few-shot learning to avoid statistics leakage",
                parameter="model_architecture",
                value="contains_batchnorm"
            ))
        
        # Check parameter count
        param_count = model_info.get("parameter_count", 0)
        if param_count == 0:
            warnings_list.append(ConfigurationWarning(
                level=WarningLevel.WARNING,
                category="no_parameters",
                message="Model has no trainable parameters",
                suggestion="This may be intended (e.g., Identity layer) but could indicate a configuration error",
                parameter="parameter_count",
                value=param_count
            ))
        elif param_count > 1e7:  # 10M parameters
            warnings_list.append(ConfigurationWarning(
                level=WarningLevel.INFO,
                category="large_model",
                message=f"Model has {param_count:,} parameters",
                suggestion="Large models may overfit on few-shot tasks. Consider regularization",
                parameter="parameter_count",
                value=param_count
            ))
        
        self._emit_warnings(warnings_list)
        return warnings_list
    
    def warn_once(self, key: str, message: str, category: str = "configuration") -> None:
        """
        Emit a warning only once per configuration.
        
        Args:
            key: Unique key for this warning
            message: Warning message
            category: Warning category
        """
        if key not in self._warned_configurations and self.enabled:
            self._warned_configurations.add(key)
            warnings.warn(message, UserWarning, stacklevel=3)
    
    def _emit_warnings(self, warnings_list: List[ConfigurationWarning]) -> None:
        """
        Emit warnings to the Python warnings system.
        
        Args:
            warnings_list: List of warnings to emit
        """
        if not self.enabled:
            return
        
        for warning_obj in warnings_list:
            # Create unique key for this warning
            key = f"{warning_obj.category}_{warning_obj.parameter}_{warning_obj.value}"
            
            if key not in self._warned_configurations:
                self._warned_configurations.add(key)
                
                # Format message with suggestion
                message = warning_obj.message
                if warning_obj.suggestion:
                    message += f". {warning_obj.suggestion}"
                
                # Import ConfigurationWarning from validation module
                try:
                    from .validation import ConfigurationWarning
                    warning_category = ConfigurationWarning
                except ImportError:
                    warning_category = UserWarning
                
                # Choose appropriate warning category  
                if warning_obj.level == WarningLevel.ERROR:
                    warnings.warn(message, warning_category, stacklevel=4)
                else:
                    warnings.warn(message, warning_category, stacklevel=4)
    
    def reset(self) -> None:
        """Reset warning history."""
        self._warned_configurations.clear()
    
    def disable(self) -> None:
        """Disable all warnings."""
        self.enabled = False
    
    def enable(self) -> None:
        """Enable warnings."""
        self.enabled = True


# Global warning system instance
_warning_system = MetaLearningWarnings()


def get_warning_system() -> MetaLearningWarnings:
    """Get the global warning system instance."""
    return _warning_system


def warn_if_suboptimal_config(**kwargs) -> List[ConfigurationWarning]:
    """
    Convenience function to check multiple configuration aspects.
    
    Args:
        **kwargs: Configuration parameters
        
    Returns:
        List of all configuration warnings
    """
    warnings_list = []
    warning_system = get_warning_system()
    
    # Check few-shot configuration
    if "n_way" in kwargs and "k_shot" in kwargs:
        warnings_list.extend(warning_system.warn_if_suboptimal_few_shot(
            kwargs["n_way"], kwargs["k_shot"], kwargs.get("n_query")
        ))
    
    # Check distance configuration
    if "distance" in kwargs and "tau" in kwargs:
        warnings_list.extend(warning_system.warn_if_suboptimal_distance_config(
            kwargs["distance"], kwargs["tau"]
        ))
    
    # Check MAML configuration
    if all(k in kwargs for k in ["inner_lr", "inner_steps", "outer_lr"]):
        warnings_list.extend(warning_system.warn_if_suboptimal_maml_config(
            kwargs["inner_lr"], kwargs["inner_steps"], kwargs["outer_lr"]
        ))
    
    # Check model configuration
    if "model_info" in kwargs:
        warnings_list.extend(warning_system.warn_if_suboptimal_model_config(
            kwargs["model_info"]
        ))
    
    return warnings_list