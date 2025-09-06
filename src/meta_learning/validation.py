"""
Input validation and error handling utilities for meta-learning algorithms.

This module provides robust input validation, error handling, and warnings
following Python best practices.
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import Counter

import torch
import numpy as np


class ValidationError(ValueError):
    """Raised when input validation fails."""
    pass


class ConfigurationWarning(UserWarning):
    """Warning for suboptimal but valid configurations."""
    pass


def validate_episode_tensors(
    support_x: torch.Tensor,
    support_y: torch.Tensor,
    query_x: torch.Tensor,
    query_y: torch.Tensor
) -> None:
    """
    Validate tensors for a few-shot learning episode.
    
    Args:
        support_x: Support set inputs
        support_y: Support set labels
        query_x: Query set inputs  
        query_y: Query set labels
        
    Raises:
        ValidationError: If tensors are invalid or inconsistent
    """
    # Check tensor types
    for name, tensor in [("support_x", support_x), ("support_y", support_y),
                        ("query_x", query_x), ("query_y", query_y)]:
        if not isinstance(tensor, torch.Tensor):
            raise ValidationError(f"{name} must be a torch.Tensor, got {type(tensor)}")
    
    # Check dimensions
    if support_x.dim() < 2:
        raise ValidationError(f"support_x must have at least 2 dimensions, got {support_x.dim()}")
    if query_x.dim() < 2:
        raise ValidationError(f"query_x must have at least 2 dimensions, got {query_x.dim()}")
    if support_y.dim() != 1:
        raise ValidationError(f"support_y must be 1-dimensional, got {support_y.dim()}")
    if query_y.dim() != 1:
        raise ValidationError(f"query_y must be 1-dimensional, got {query_y.dim()}")
    
    # Check batch size consistency
    if support_x.size(0) != support_y.size(0):
        raise ValidationError(
            f"support_x and support_y batch sizes don't match: "
            f"{support_x.size(0)} vs {support_y.size(0)}"
        )
    if query_x.size(0) != query_y.size(0):
        raise ValidationError(
            f"query_x and query_y batch sizes don't match: "
            f"{query_x.size(0)} vs {query_y.size(0)}"
        )
    
    # Check feature dimension consistency
    if support_x.shape[1:] != query_x.shape[1:]:
        raise ValidationError(
            f"support_x and query_x feature dimensions don't match: "
            f"{support_x.shape[1:]} vs {query_x.shape[1:]}"
        )
    
    # Check label validity
    if support_y.min() < 0:
        raise ValidationError(f"support_y contains negative labels: {support_y.min()}")
    if query_y.min() < 0:
        raise ValidationError(f"query_y contains negative labels: {query_y.min()}")
    
    # Check label consistency between support and query
    support_classes = set(support_y.tolist())
    query_classes = set(query_y.tolist())
    if query_classes - support_classes:
        missing_classes = query_classes - support_classes
        raise ValidationError(
            f"Query set contains classes not in support set: {missing_classes}"
        )


def validate_few_shot_configuration(
    n_way: int,
    k_shot: int, 
    n_query: Optional[int] = None
) -> None:
    """
    Validate few-shot learning configuration parameters.
    
    Args:
        n_way: Number of classes
        k_shot: Number of support examples per class
        n_query: Number of query examples per class (optional)
        
    Raises:
        ValidationError: If configuration is invalid
    """
    if not isinstance(n_way, int) or n_way <= 0:
        raise ValidationError(f"n_way must be a positive integer, got {n_way}")
    if not isinstance(k_shot, int) or k_shot <= 0:
        raise ValidationError(f"k_shot must be a positive integer, got {k_shot}")
    if n_query is not None and (not isinstance(n_query, int) or n_query <= 0):
        raise ValidationError(f"n_query must be a positive integer, got {n_query}")
    
    # Check reasonable bounds
    if n_way > 100:
        warnings.warn(
            f"n_way={n_way} is unusually high for few-shot learning. "
            "Consider using n_way <= 20 for better performance.",
            ConfigurationWarning
        )
    
    if k_shot > 50:
        warnings.warn(
            f"k_shot={k_shot} is unusually high for few-shot learning. "
            "Consider using k_shot <= 10 for typical few-shot scenarios.",
            ConfigurationWarning
        )


def validate_distance_metric(distance: str) -> None:
    """
    Validate distance metric parameter.
    
    Args:
        distance: Distance metric name
        
    Raises:
        ValidationError: If distance metric is invalid
    """
    valid_distances = {"sqeuclidean", "euclidean", "cosine", "manhattan", "dot"}
    if distance not in valid_distances:
        raise ValidationError(
            f"Invalid distance metric '{distance}'. "
            f"Must be one of: {valid_distances}"
        )


def validate_temperature_parameter(tau: float, distance: str) -> None:
    """
    Validate temperature parameter for distance-based methods.
    
    Args:
        tau: Temperature parameter
        distance: Distance metric being used
        
    Raises:
        ValidationError: If temperature is invalid
    """
    if not isinstance(tau, (int, float)) or tau <= 0:
        raise ValidationError(f"tau must be a positive number, got {tau}")
    
    # Provide guidance based on distance metric
    if distance == "cosine" and tau > 10:
        warnings.warn(
            f"tau={tau} is very high for cosine distance. "
            "Consider tau in range [0.1, 5.0] for better calibration.",
            ConfigurationWarning
        )
    elif distance == "sqeuclidean" and tau > 100:
        warnings.warn(
            f"tau={tau} is very high for squared Euclidean distance. "
            "Consider tau in range [0.1, 10.0] for better calibration.",
            ConfigurationWarning
        )


def validate_learning_rate(lr: float, context: str = "learning rate") -> None:
    """
    Validate learning rate parameter.
    
    Args:
        lr: Learning rate value
        context: Context for error messages
        
    Raises:
        ValidationError: If learning rate is invalid
    """
    if not isinstance(lr, (int, float)) or lr <= 0:
        raise ValidationError(f"{context} must be a positive number, got {lr}")
    
    if lr > 1.0:
        warnings.warn(
            f"{context}={lr} is very high. "
            "Consider values in range [1e-5, 0.1] for stability.",
            ConfigurationWarning
        )
    elif lr < 1e-6:
        warnings.warn(
            f"{context}={lr} is very low. "
            "Training may be extremely slow.",
            ConfigurationWarning
        )


def validate_model_parameters(model: torch.nn.Module) -> None:
    """
    Validate model parameters and architecture.
    
    Args:
        model: PyTorch model to validate
        
    Raises:
        ValidationError: If model is invalid
    """
    if not isinstance(model, torch.nn.Module):
        raise ValidationError(f"model must be a torch.nn.Module, got {type(model)}")
    
    # Check if model has parameters
    param_count = sum(p.numel() for p in model.parameters())
    if param_count == 0:
        warnings.warn(
            "Model has no parameters. This may be intended (e.g., Identity) "
            "but could indicate a configuration error.",
            ConfigurationWarning
        )
    
    # Check for common issues
    has_batchnorm = any(isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d))
                       for m in model.modules())
    if has_batchnorm:
        warnings.warn(
            "Model contains BatchNorm layers. Consider using GroupNorm or LayerNorm "
            "for few-shot learning to avoid episodic statistics leakage.",
            ConfigurationWarning
        )


def check_episode_quality(
    support_x: torch.Tensor,
    support_y: torch.Tensor,
    query_x: torch.Tensor,
    query_y: torch.Tensor
) -> Dict[str, Any]:
    """
    Analyze episode quality and return metrics.
    
    Args:
        support_x: Support set inputs
        support_y: Support set labels
        query_x: Query set inputs
        query_y: Query set labels
        
    Returns:
        Dictionary with quality metrics
    """
    metrics = {}
    
    # Class balance in support set
    support_counts = Counter(support_y.tolist())
    metrics["support_class_balance"] = dict(support_counts)
    metrics["support_is_balanced"] = len(set(support_counts.values())) == 1
    
    # Class balance in query set
    query_counts = Counter(query_y.tolist())
    metrics["query_class_balance"] = dict(query_counts)
    metrics["query_is_balanced"] = len(set(query_counts.values())) == 1
    
    # Number of classes
    metrics["n_way"] = len(support_counts)
    metrics["k_shot"] = min(support_counts.values())
    
    # Data quality checks
    if support_x.dtype == torch.float:
        metrics["support_has_nan"] = torch.isnan(support_x).any().item()
        metrics["support_has_inf"] = torch.isinf(support_x).any().item()
        metrics["query_has_nan"] = torch.isnan(query_x).any().item()
        metrics["query_has_inf"] = torch.isinf(query_x).any().item()
    
    return metrics


def warn_if_suboptimal_config(
    n_way: int,
    k_shot: int,
    distance: str,
    tau: float,
    **kwargs
) -> None:
    """
    Warn about potentially suboptimal configurations.
    
    Args:
        n_way: Number of classes
        k_shot: Number of support examples per class
        distance: Distance metric
        tau: Temperature parameter
        **kwargs: Additional configuration parameters
    """
    # Very challenging few-shot setting
    if n_way >= 10 and k_shot == 1:
        warnings.warn(
            f"{n_way}-way 1-shot is very challenging. "
            "Consider increasing k_shot or reducing n_way for better performance.",
            ConfigurationWarning
        )
    
    # Suboptimal distance-temperature combinations
    if distance == "cosine" and tau < 0.1:
        warnings.warn(
            f"tau={tau} is very low for cosine distance, which may cause "
            "overconfident predictions. Consider tau >= 0.5.",
            ConfigurationWarning
        )
    
    if distance == "sqeuclidean" and tau < 0.01:
        warnings.warn(
            f"tau={tau} is very low for squared Euclidean distance, which may cause "
            "numerical instability. Consider tau >= 0.1.",
            ConfigurationWarning
        )


def validate_maml_config(
    inner_lr: float,
    inner_steps: int, 
    outer_lr: float,
    first_order: bool = False,
    allow_unused: bool = None
) -> None:
    """
    Validate MAML hyperparameter configuration.
    
    Args:
        inner_lr: Inner loop learning rate
        inner_steps: Number of inner loop gradient steps
        outer_lr: Outer loop learning rate
        first_order: Whether to use first-order approximation
        allow_unused: Whether to allow unused parameters in gradient computation
        
    Raises:
        ValidationError: If configuration is invalid
    """
    # Validate inner learning rate
    if not isinstance(inner_lr, (int, float)) or inner_lr <= 0:
        raise ValidationError(f"inner_lr must be a positive number, got {inner_lr}")
    
    # Validate inner steps
    if not isinstance(inner_steps, int) or inner_steps <= 0:
        raise ValidationError(f"inner_steps must be a positive integer, got {inner_steps}")
    
    # Validate outer learning rate
    if not isinstance(outer_lr, (int, float)) or outer_lr <= 0:
        raise ValidationError(f"outer_lr must be a positive number, got {outer_lr}")
    
    # Configuration warnings
    if inner_lr > 0.1:
        warnings.warn(
            f"inner_lr={inner_lr} is very high for MAML. "
            "High inner learning rates can cause instability. Consider inner_lr <= 0.01.",
            ConfigurationWarning
        )
    elif inner_lr < 1e-5:
        warnings.warn(
            f"inner_lr={inner_lr} is very low for MAML. "
            "Very low learning rates may prevent effective adaptation.",
            ConfigurationWarning
        )
    
    if inner_steps > 10:
        warnings.warn(
            f"inner_steps={inner_steps} is higher than typical MAML configurations. "
            "Most MAML implementations use 1-5 inner steps. More steps increase computation.",
            ConfigurationWarning
        )
    
    if outer_lr > inner_lr * 10:
        warnings.warn(
            f"outer_lr/inner_lr ratio is very high ({outer_lr/inner_lr:.1f}). "
            "Large ratios may cause meta-learning instability.",
            ConfigurationWarning
        )


def validate_uncertainty_config(
    method: str,
    n_samples: int = 10,
    dropout_rate: float = 0.1
) -> None:
    """
    Validate uncertainty estimation configuration.
    
    Args:
        method: Uncertainty estimation method
        n_samples: Number of samples for stochastic methods
        dropout_rate: Dropout rate for Monte Carlo dropout
        
    Raises:
        ValidationError: If configuration is invalid
    """
    valid_methods = {"monte_carlo_dropout", "deep_ensemble", "evidential"}
    if method not in valid_methods:
        raise ValidationError(
            f"Invalid uncertainty method '{method}'. "
            f"Must be one of: {valid_methods}"
        )
    
    if not isinstance(n_samples, int) or n_samples <= 0:
        raise ValidationError(f"n_samples must be a positive integer, got {n_samples}")
    
    if not isinstance(dropout_rate, (int, float)) or not (0.0 <= dropout_rate <= 1.0):
        raise ValidationError(f"dropout_rate must be in [0, 1], got {dropout_rate}")
    
    # Configuration warnings
    if method == "monte_carlo_dropout":
        if n_samples < 5:
            warnings.warn(
                f"n_samples={n_samples} is low for Monte Carlo dropout. "
                "Consider using at least 10 samples for reliable uncertainty estimates.",
                ConfigurationWarning
            )
        if dropout_rate < 0.05:
            warnings.warn(
                f"dropout_rate={dropout_rate} is very low. "
                "Low dropout rates may not provide meaningful uncertainty estimates.",
                ConfigurationWarning
            )


def validate_optimizer_config(
    optimizer_type: str,
    lr: float,
    weight_decay: float = 0.0,
    momentum: float = 0.0,
    **optimizer_kwargs
) -> None:
    """
    Validate optimizer configuration.
    
    Args:
        optimizer_type: Type of optimizer
        lr: Learning rate
        weight_decay: L2 regularization strength
        momentum: Momentum coefficient
        **optimizer_kwargs: Additional optimizer parameters
        
    Raises:
        ValidationError: If configuration is invalid
    """
    valid_optimizers = {"sgd", "adam", "adamw", "rmsprop", "adagrad"}
    if optimizer_type.lower() not in valid_optimizers:
        raise ValidationError(
            f"Invalid optimizer type '{optimizer_type}'. "
            f"Must be one of: {valid_optimizers}"
        )
    
    validate_learning_rate(lr, "optimizer learning rate")
    
    if not isinstance(weight_decay, (int, float)) or weight_decay < 0:
        raise ValidationError(f"weight_decay must be non-negative, got {weight_decay}")
    
    if not isinstance(momentum, (int, float)) or not (0.0 <= momentum <= 1.0):
        raise ValidationError(f"momentum must be in [0, 1], got {momentum}")
    
    # Optimizer-specific validation
    if optimizer_type.lower() == "adam" or optimizer_type.lower() == "adamw":
        beta1 = optimizer_kwargs.get("beta1", 0.9)
        beta2 = optimizer_kwargs.get("beta2", 0.999)
        
        if not (0.0 <= beta1 <= 1.0):
            raise ValidationError(f"adam beta1 must be in [0, 1], got {beta1}")
        if not (0.0 <= beta2 <= 1.0):
            raise ValidationError(f"adam beta2 must be in [0, 1], got {beta2}")


def validate_episodic_config(
    n_tasks: int,
    meta_batch_size: int,
    train_episodes: int,
    val_episodes: int,
    test_episodes: int
) -> None:
    """
    Validate episodic training configuration.
    
    Args:
        n_tasks: Total number of tasks
        meta_batch_size: Number of tasks per meta-batch
        train_episodes: Number of training episodes per task
        val_episodes: Number of validation episodes per task
        test_episodes: Number of test episodes per task
        
    Raises:
        ValidationError: If configuration is invalid
    """
    if not isinstance(n_tasks, int) or n_tasks <= 0:
        raise ValidationError(f"n_tasks must be a positive integer, got {n_tasks}")
    
    if not isinstance(meta_batch_size, int) or meta_batch_size <= 0:
        raise ValidationError(f"meta_batch_size must be a positive integer, got {meta_batch_size}")
    
    if meta_batch_size > n_tasks:
        raise ValidationError(
            f"meta_batch_size ({meta_batch_size}) cannot exceed n_tasks ({n_tasks})"
        )
    
    for name, value in [("train_episodes", train_episodes), 
                       ("val_episodes", val_episodes),
                       ("test_episodes", test_episodes)]:
        if not isinstance(value, int) or value <= 0:
            raise ValidationError(f"{name} must be a positive integer, got {value}")
    
    # Configuration warnings
    if val_episodes < 100:
        warnings.warn(
            f"val_episodes={val_episodes} may be too small for reliable validation. "
            "Consider using at least 100 episodes for stable estimates.",
            ConfigurationWarning
        )
    
    if test_episodes < 500:
        warnings.warn(
            f"test_episodes={test_episodes} may be too small for reliable testing. "
            "Consider using at least 500 episodes for stable final evaluation.",
            ConfigurationWarning
        )


def validate_regularization_config(
    dropout_rate: float = 0.0,
    weight_decay: float = 0.0,
    gradient_clip_norm: Optional[float] = None,
    prototype_shrinkage: float = 0.0
) -> None:
    """
    Validate regularization configuration.
    
    Args:
        dropout_rate: Dropout probability
        weight_decay: L2 regularization strength
        gradient_clip_norm: Gradient clipping threshold
        prototype_shrinkage: Prototype shrinkage factor
        
    Raises:
        ValidationError: If configuration is invalid
    """
    if not isinstance(dropout_rate, (int, float)) or not (0.0 <= dropout_rate <= 1.0):
        raise ValidationError(f"dropout_rate must be in [0, 1], got {dropout_rate}")
    
    if not isinstance(weight_decay, (int, float)) or weight_decay < 0:
        raise ValidationError(f"weight_decay must be non-negative, got {weight_decay}")
    
    if gradient_clip_norm is not None:
        if not isinstance(gradient_clip_norm, (int, float)) or gradient_clip_norm <= 0:
            raise ValidationError(f"gradient_clip_norm must be positive, got {gradient_clip_norm}")
    
    if not isinstance(prototype_shrinkage, (int, float)) or not (0.0 <= prototype_shrinkage <= 1.0):
        raise ValidationError(f"prototype_shrinkage must be in [0, 1], got {prototype_shrinkage}")


class ValidationContext:
    """Context manager for validation settings."""
    
    def __init__(self, strict: bool = True, warnings_enabled: bool = True):
        """
        Initialize validation context.
        
        Args:
            strict: Whether to raise errors on validation failures
            warnings_enabled: Whether to emit warnings
        """
        self.strict = strict
        self.warnings_enabled = warnings_enabled
        self._old_warnings_filter = None
    
    def __enter__(self):
        if not self.warnings_enabled:
            self._old_warnings_filter = warnings.filters[:]
            warnings.filterwarnings("ignore", category=ConfigurationWarning)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._old_warnings_filter is not None:
            warnings.filters[:] = self._old_warnings_filter


def validate_complete_config(config: Dict[str, Any]) -> None:
    """
    Comprehensive validation of a complete meta-learning configuration.
    
    Args:
        config: Configuration dictionary containing all parameters
        
    Raises:
        ValidationError: If any part of configuration is invalid
    """
    # Few-shot configuration
    if all(k in config for k in ["n_way", "k_shot"]):
        validate_few_shot_configuration(
            config["n_way"], 
            config["k_shot"], 
            config.get("n_query")
        )
    
    # Distance and temperature
    if "distance" in config:
        validate_distance_metric(config["distance"])
        if "tau" in config:
            validate_temperature_parameter(config["tau"], config["distance"])
    
    # MAML configuration
    if all(k in config for k in ["inner_lr", "inner_steps", "outer_lr"]):
        validate_maml_config(
            config["inner_lr"],
            config["inner_steps"], 
            config["outer_lr"],
            config.get("first_order", False),
            config.get("allow_unused")
        )
    
    # Uncertainty configuration
    if "uncertainty_method" in config:
        validate_uncertainty_config(
            config["uncertainty_method"],
            config.get("n_uncertainty_samples", 10),
            config.get("dropout_rate", 0.1)
        )
    
    # Check for suboptimal configurations and warn
    try:
        from meta_learning.warnings_system import warn_if_suboptimal_config
        warn_if_suboptimal_config(**config)
    except ImportError:
        # Warning system not available, skip warnings
        pass
    
    # Optimizer configuration
    if "optimizer" in config:
        validate_optimizer_config(
            config["optimizer"],
            config.get("lr", config.get("outer_lr", 1e-3)),
            config.get("weight_decay", 0.0),
            config.get("momentum", 0.0),
            **{k: v for k, v in config.items() if k.startswith("optimizer_")}
        )
    
    # Episodic training configuration
    if all(k in config for k in ["n_tasks", "meta_batch_size"]):
        validate_episodic_config(
            config["n_tasks"],
            config["meta_batch_size"],
            config.get("train_episodes", 100),
            config.get("val_episodes", 100),
            config.get("test_episodes", 500)
        )
    
    # Regularization configuration
    validate_regularization_config(
        config.get("dropout_rate", 0.0),
        config.get("weight_decay", 0.0),
        config.get("gradient_clip_norm"),
        config.get("prototype_shrinkage", 0.0)
    )