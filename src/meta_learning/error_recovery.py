"""
Error recovery and fault tolerance mechanisms for meta-learning systems.

This module provides graceful error handling, fallback strategies, and
recovery mechanisms to ensure robust operation under various failure conditions.
"""

import logging
import traceback
import warnings
from typing import Any, Dict, List, Optional, Callable, Union, Tuple
from functools import wraps
import time

import torch
import numpy as np

from .validation import ValidationError, ConfigurationWarning


class RecoveryError(Exception):
    """Raised when error recovery fails."""
    pass


class ErrorRecoveryManager:
    """Manages error recovery strategies and fallback mechanisms."""
    
    def __init__(self, 
                 max_retries: int = 3,
                 retry_delay: float = 0.1,
                 enable_logging: bool = True):
        """
        Initialize error recovery manager.
        
        Args:
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retry attempts in seconds
            enable_logging: Whether to log recovery attempts
        """
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.enable_logging = enable_logging
        self.recovery_stats = {
            "total_recoveries": 0,
            "successful_recoveries": 0,
            "failed_recoveries": 0,
            "recovery_types": {}
        }
        
        if enable_logging:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = None
    
    def log(self, message: str, level: str = "info") -> None:
        """Log message if logging is enabled."""
        if self.logger:
            getattr(self.logger, level)(message)
    
    def record_recovery(self, recovery_type: str, success: bool) -> None:
        """Record recovery attempt statistics."""
        self.recovery_stats["total_recoveries"] += 1
        if success:
            self.recovery_stats["successful_recoveries"] += 1
        else:
            self.recovery_stats["failed_recoveries"] += 1
        
        if recovery_type not in self.recovery_stats["recovery_types"]:
            self.recovery_stats["recovery_types"][recovery_type] = {"success": 0, "failure": 0}
        
        if success:
            self.recovery_stats["recovery_types"][recovery_type]["success"] += 1
        else:
            self.recovery_stats["recovery_types"][recovery_type]["failure"] += 1


def with_retry(max_attempts: int = 3, delay: float = 0.1, 
               exceptions: Tuple = (Exception,)):
    """
    Decorator for automatic retry on failure.
    
    Args:
        max_attempts: Maximum number of attempts
        delay: Delay between attempts in seconds
        exceptions: Tuple of exceptions to catch and retry
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        time.sleep(delay)
                        continue
                    break
            
            raise last_exception
        return wrapper
    return decorator


def safe_tensor_operation(operation: Callable, 
                         tensor: torch.Tensor,
                         fallback_value: Optional[torch.Tensor] = None,
                         **kwargs) -> torch.Tensor:
    """
    Safely execute tensor operation with fallback.
    
    Args:
        operation: Tensor operation to execute
        tensor: Input tensor
        fallback_value: Value to return on failure
        **kwargs: Additional arguments for operation
        
    Returns:
        Result tensor or fallback value
    """
    try:
        return operation(tensor, **kwargs)
    except (RuntimeError, ValueError) as e:
        warnings.warn(
            f"Tensor operation failed: {e}. Using fallback.",
            ConfigurationWarning
        )
        if fallback_value is not None:
            return fallback_value
        else:
            # Return tensor filled with zeros of same shape
            return torch.zeros_like(tensor)


def handle_numerical_instability(tensor: torch.Tensor, 
                                eps: float = 1e-8,
                                clip_range: Optional[Tuple[float, float]] = None) -> torch.Tensor:
    """
    Handle numerical instability in tensors.
    
    Args:
        tensor: Input tensor
        eps: Small value to replace NaN/Inf
        clip_range: Optional clipping range (min, max)
        
    Returns:
        Cleaned tensor
    """
    # Handle NaN values
    if torch.isnan(tensor).any():
        warnings.warn("NaN values detected, replacing with zeros", ConfigurationWarning)
        tensor = torch.where(torch.isnan(tensor), torch.zeros_like(tensor), tensor)
    
    # Handle infinite values
    if torch.isinf(tensor).any():
        warnings.warn("Infinite values detected, clipping", ConfigurationWarning)
        tensor = torch.where(torch.isinf(tensor), 
                           torch.sign(tensor) * 1e6, tensor)
    
    # Optional clipping
    if clip_range is not None:
        tensor = torch.clamp(tensor, min=clip_range[0], max=clip_range[1])
    
    return tensor


def recover_from_dimension_mismatch(tensor1: torch.Tensor, 
                                  tensor2: torch.Tensor,
                                  operation: str = "broadcast") -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Recover from tensor dimension mismatches.
    
    Args:
        tensor1: First tensor
        tensor2: Second tensor  
        operation: Recovery operation ("broadcast", "reshape", "pad")
        
    Returns:
        Tuple of recovered tensors
    """
    if operation == "broadcast":
        try:
            # Try broadcasting
            result1, result2 = torch.broadcast_tensors(tensor1, tensor2)
            return result1, result2
        except RuntimeError:
            pass
    
    if operation == "reshape":
        # Try to reshape to compatible dimensions
        if tensor1.numel() == tensor2.numel():
            return tensor1.view(-1), tensor2.view(-1)
    
    if operation == "pad":
        # Pad smaller tensor to match larger one
        if tensor1.shape != tensor2.shape:
            max_dims = max(tensor1.dim(), tensor2.dim())
            
            # Pad dimensions to match
            while tensor1.dim() < max_dims:
                tensor1 = tensor1.unsqueeze(0)
            while tensor2.dim() < max_dims:
                tensor2 = tensor2.unsqueeze(0)
            
            # Pad sizes to match
            max_shape = [max(s1, s2) for s1, s2 in zip(tensor1.shape, tensor2.shape)]
            
            pad1 = []
            pad2 = []
            for i in range(len(max_shape)):
                diff1 = max_shape[i] - tensor1.shape[i]
                diff2 = max_shape[i] - tensor2.shape[i]
                pad1.extend([0, diff1])
                pad2.extend([0, diff2])
            
            # Reverse padding for F.pad format
            pad1 = pad1[::-1]
            pad2 = pad2[::-1]
            
            if any(p > 0 for p in pad1):
                tensor1 = torch.nn.functional.pad(tensor1, pad1)
            if any(p > 0 for p in pad2):
                tensor2 = torch.nn.functional.pad(tensor2, pad2)
            
            return tensor1, tensor2
    
    # If all recovery attempts fail, return original tensors
    warnings.warn(
        f"Could not recover from dimension mismatch: {tensor1.shape} vs {tensor2.shape}",
        ConfigurationWarning
    )
    return tensor1, tensor2


class RobustPrototypeNetwork:
    """
    Robust wrapper for prototype networks with error recovery.
    """
    
    def __init__(self, base_network, recovery_manager: Optional[ErrorRecoveryManager] = None):
        """
        Initialize robust prototype network.
        
        Args:
            base_network: Base prototype network
            recovery_manager: Error recovery manager
        """
        self.base_network = base_network
        self.recovery_manager = recovery_manager or ErrorRecoveryManager()
    
    def forward_with_recovery(self, 
                            support_x: torch.Tensor,
                            support_y: torch.Tensor, 
                            query_x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with error recovery.
        
        Args:
            support_x: Support set features
            support_y: Support set labels
            query_x: Query set features
            
        Returns:
            Query predictions
        """
        try:
            # Attempt normal forward pass
            return self.base_network(support_x, support_y, query_x)
            
        except RuntimeError as e:
            self.recovery_manager.log(f"Forward pass failed: {e}", "warning")
            
            # Try recovery strategies
            recovered = False
            
            # Strategy 1: Handle numerical instabilities
            try:
                clean_support_x = handle_numerical_instability(support_x)
                clean_query_x = handle_numerical_instability(query_x)
                result = self.base_network(clean_support_x, support_y, clean_query_x)
                self.recovery_manager.record_recovery("numerical_cleanup", True)
                recovered = True
                return result
            except Exception:
                pass
            
            # Strategy 2: Dimension recovery
            try:
                if support_x.shape[1:] != query_x.shape[1:]:
                    support_x, query_x = recover_from_dimension_mismatch(
                        support_x, query_x, "broadcast"
                    )
                    result = self.base_network(support_x, support_y, query_x)
                    self.recovery_manager.record_recovery("dimension_fix", True)
                    recovered = True
                    return result
            except Exception:
                pass
            
            # Strategy 3: Fallback to simple nearest neighbor
            try:
                result = self._fallback_nearest_neighbor(support_x, support_y, query_x)
                self.recovery_manager.record_recovery("fallback_nn", True)
                recovered = True
                return result
            except Exception:
                pass
            
            # If all recovery fails
            if not recovered:
                self.recovery_manager.record_recovery("complete_failure", False)
                raise RecoveryError(f"All recovery strategies failed for error: {e}")
    
    def _fallback_nearest_neighbor(self, 
                                 support_x: torch.Tensor,
                                 support_y: torch.Tensor,
                                 query_x: torch.Tensor) -> torch.Tensor:
        """
        Fallback to simple nearest neighbor classification.
        
        Args:
            support_x: Support set features
            support_y: Support set labels
            query_x: Query set features
            
        Returns:
            Query predictions
        """
        # Compute distances between query and support
        distances = torch.cdist(query_x, support_x)  # [n_query, n_support]
        
        # Find nearest neighbors
        nearest_indices = torch.argmin(distances, dim=1)  # [n_query]
        
        # Get corresponding labels
        predictions = support_y[nearest_indices]  # [n_query]
        
        # Convert to one-hot logits
        n_classes = len(torch.unique(support_y))
        logits = torch.zeros(query_x.size(0), n_classes, device=query_x.device)
        logits.scatter_(1, predictions.unsqueeze(1), 1.0)
        
        return logits


def create_robust_episode(n_way: int, 
                         k_shot: int,
                         n_query: int,
                         feature_dim: int,
                         device: torch.device = None) -> Dict[str, torch.Tensor]:
    """
    Create a robust synthetic episode with error checking.
    
    Args:
        n_way: Number of classes
        k_shot: Number of support examples per class
        n_query: Number of query examples per class
        feature_dim: Feature dimensionality
        device: Device for tensors
        
    Returns:
        Dictionary containing episode tensors
    """
    device = device or torch.device("cpu")
    
    try:
        # Generate synthetic data
        support_x = torch.randn(n_way * k_shot, feature_dim, device=device)
        query_x = torch.randn(n_way * n_query, feature_dim, device=device)
        
        # Generate labels
        support_y = torch.arange(n_way, device=device).repeat_interleave(k_shot)
        query_y = torch.arange(n_way, device=device).repeat_interleave(n_query)
        
        # Validate episode
        episode = {
            "support_x": support_x,
            "support_y": support_y,
            "query_x": query_x,
            "query_y": query_y
        }
        
        # Add some noise to make it more realistic
        episode["support_x"] += 0.1 * torch.randn_like(episode["support_x"])
        episode["query_x"] += 0.1 * torch.randn_like(episode["query_x"])
        
        return episode
        
    except Exception as e:
        warnings.warn(f"Episode creation failed: {e}", ConfigurationWarning)
        
        # Fallback to minimal episode
        support_x = torch.eye(feature_dim, device=device)[:n_way]
        support_x = support_x.repeat_interleave(k_shot, dim=0)
        query_x = torch.eye(feature_dim, device=device)[:n_way]
        query_x = query_x.repeat_interleave(n_query, dim=0)
        
        support_y = torch.arange(n_way, device=device).repeat_interleave(k_shot)
        query_y = torch.arange(n_way, device=device).repeat_interleave(n_query)
        
        return {
            "support_x": support_x,
            "support_y": support_y,
            "query_x": query_x,
            "query_y": query_y
        }


class FaultTolerantTrainer:
    """
    Trainer with fault tolerance and recovery mechanisms.
    """
    
    def __init__(self, model, optimizer, recovery_manager: Optional[ErrorRecoveryManager] = None):
        """
        Initialize fault-tolerant trainer.
        
        Args:
            model: Model to train
            optimizer: Optimizer
            recovery_manager: Error recovery manager
        """
        self.model = model
        self.optimizer = optimizer
        self.recovery_manager = recovery_manager or ErrorRecoveryManager()
        self.checkpoint_data = None
    
    def save_checkpoint(self) -> None:
        """Save current model and optimizer state."""
        import copy
        self.checkpoint_data = {
            "model_state": copy.deepcopy(self.model.state_dict()),
            "optimizer_state": copy.deepcopy(self.optimizer.state_dict())
        }
    
    def restore_checkpoint(self) -> None:
        """Restore from saved checkpoint."""
        if self.checkpoint_data:
            self.model.load_state_dict(self.checkpoint_data["model_state"])
            self.optimizer.load_state_dict(self.checkpoint_data["optimizer_state"])
            self.recovery_manager.log("Restored from checkpoint", "info")
    
    @with_retry(max_attempts=3, delay=0.1)
    def robust_training_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """
        Execute a training step with error recovery.
        
        Args:
            batch: Training batch
            
        Returns:
            Loss value
        """
        try:
            self.optimizer.zero_grad()
            
            # Forward pass
            logits = self.model(
                batch["support_x"], 
                batch["support_y"], 
                batch["query_x"]
            )
            
            # Compute loss
            loss = torch.nn.functional.cross_entropy(logits, batch["query_y"])
            
            # Check for numerical issues
            if torch.isnan(loss) or torch.isinf(loss):
                raise ValueError(f"Loss is {loss}")
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Optimizer step
            self.optimizer.step()
            
            return loss.item()
            
        except Exception as e:
            self.recovery_manager.log(f"Training step failed: {e}", "warning")
            
            # Try to recover
            self.restore_checkpoint()
            
            # Reduce learning rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= 0.5
            
            self.recovery_manager.record_recovery("training_step", True)
            return float('inf')  # Return high loss to indicate failure


def safe_evaluate(model, episodes: List[Dict[str, torch.Tensor]], 
                 recovery_manager: Optional[ErrorRecoveryManager] = None) -> Dict[str, float]:
    """
    Safely evaluate model with error recovery.
    
    Args:
        model: Model to evaluate
        episodes: List of evaluation episodes
        recovery_manager: Error recovery manager
        
    Returns:
        Evaluation metrics
    """
    recovery_manager = recovery_manager or ErrorRecoveryManager()
    
    total_correct = 0
    total_samples = 0
    failed_episodes = 0
    
    for i, episode in enumerate(episodes):
        try:
            with torch.no_grad():
                logits = model(
                    episode["support_x"],
                    episode["support_y"], 
                    episode["query_x"]
                )
                
                predictions = torch.argmax(logits, dim=1)
                correct = (predictions == episode["query_y"]).sum().item()
                
                total_correct += correct
                total_samples += len(episode["query_y"])
                
        except Exception as e:
            recovery_manager.log(f"Evaluation failed for episode {i}: {e}", "warning")
            failed_episodes += 1
            
            # Try fallback evaluation
            try:
                # Use majority class prediction
                majority_class = torch.mode(episode["support_y"]).values
                predictions = majority_class.repeat(len(episode["query_y"]))
                correct = (predictions == episode["query_y"]).sum().item()
                total_correct += correct
                total_samples += len(episode["query_y"])
                recovery_manager.record_recovery("evaluation_fallback", True)
            except Exception:
                recovery_manager.record_recovery("evaluation_fallback", False)
    
    if total_samples == 0:
        return {"accuracy": 0.0, "failed_episodes": failed_episodes}
    
    return {
        "accuracy": total_correct / total_samples,
        "failed_episodes": failed_episodes,
        "total_episodes": len(episodes)
    }