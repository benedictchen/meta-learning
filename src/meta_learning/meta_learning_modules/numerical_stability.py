"""
Numerical Stability Utilities for Meta-Learning
===============================================

Author: Benedict Chen (benedict@benedictchen.com)

Research-grade numerical stability utilities following best practices
from meta-learning literature. Ensures robust training and evaluation.

Key Stability Measures:
1. F.log_softmax + NLL instead of softmax + CE
2. Gradient clipping for inner/outer loops
3. Epsilon handling in normalization and distances
4. Deterministic seeding for reproducibility
"""

import torch
import torch.nn.functional as F
import numpy as np
import random
from typing import Optional, List, Tuple
import warnings


# Global epsilon for numerical stability
EPS = 1e-8


def stable_softmax_ce_loss(logits: torch.Tensor, targets: torch.Tensor, 
                          temperature: float = 1.0) -> torch.Tensor:
    """
    Numerically stable cross-entropy loss using log_softmax + NLL.
    
    More stable than F.softmax + F.cross_entropy, especially with 
    temperature scaling.
    
    Args:
        logits: Unnormalized logits [batch_size, num_classes]
        targets: Target class indices [batch_size]
        temperature: Temperature scaling parameter
        
    Returns:
        Cross-entropy loss (scalar)
    """
    # Apply temperature scaling to logits (not probabilities)
    scaled_logits = logits / temperature
    
    # Use log_softmax + NLL for numerical stability
    log_probs = F.log_softmax(scaled_logits, dim=1)
    loss = F.nll_loss(log_probs, targets)
    
    return loss


def safe_normalize(x: torch.Tensor, dim: int = -1, eps: float = EPS) -> torch.Tensor:
    """
    L2 normalize with epsilon for numerical stability.
    
    Prevents division by zero in normalization operations.
    Essential for cosine similarity computations.
    """
    norm = x.norm(dim=dim, keepdim=True)
    # Add epsilon to prevent division by zero
    safe_norm = torch.clamp(norm, min=eps)
    return x / safe_norm


def safe_distance_computation(query_features: torch.Tensor, 
                             prototypes: torch.Tensor,
                             distance_type: str = "euclidean",
                             eps: float = EPS) -> torch.Tensor:
    """
    Compute distances with numerical stability.
    
    Args:
        query_features: Query embeddings [N_query, D]
        prototypes: Class prototypes [N_classes, D]  
        distance_type: "euclidean" or "cosine"
        eps: Epsilon for stability
        
    Returns:
        Distances/similarities [N_query, N_classes]
    """
    if distance_type == "euclidean":
        # Squared Euclidean distance: ||q - p||²
        # Use broadcasting: [N_q, 1, D] - [1, N_c, D] = [N_q, N_c, D]
        query_expanded = query_features.unsqueeze(1)     # [N_q, 1, D]
        proto_expanded = prototypes.unsqueeze(0)         # [1, N_c, D]
        
        differences = query_expanded - proto_expanded    # [N_q, N_c, D]
        distances = (differences ** 2).sum(dim=2)        # [N_q, N_c]
        
        return distances
        
    elif distance_type == "cosine":
        # Cosine similarity with safe normalization
        query_norm = safe_normalize(query_features, dim=1, eps=eps)
        proto_norm = safe_normalize(prototypes, dim=1, eps=eps)
        
        # Cosine similarity: q·p / (||q|| ||p||)
        similarities = torch.mm(query_norm, proto_norm.t())  # [N_q, N_c]
        
        # Return negative similarities for consistency with distance formulation
        return -similarities
        
    else:
        raise ValueError(f"Unknown distance type: {distance_type}")


def clip_gradients(parameters, max_norm: float, norm_type: float = 2.0) -> float:
    """
    Gradient clipping with proper error handling.
    
    Args:
        parameters: Model parameters or parameter groups
        max_norm: Maximum gradient norm
        norm_type: Type of norm (2.0 for L2)
        
    Returns:
        Total gradient norm before clipping
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    
    # Filter parameters with gradients
    parameters = [p for p in parameters if p.grad is not None]
    
    if len(parameters) == 0:
        return 0.0
    
    # Compute total gradient norm
    total_norm = torch.norm(
        torch.stack([torch.norm(p.grad.detach(), norm_type) for p in parameters]),
        norm_type
    )
    
    # Clip gradients if needed
    if total_norm > max_norm:
        clip_coef = max_norm / (total_norm + EPS)
        for p in parameters:
            p.grad.detach().mul_(clip_coef)
    
    return total_norm.item()


def seed_everything(seed: int, deterministic_cuda: bool = True):
    """
    Set all random seeds for reproducible experiments.
    
    Critical for reproducible meta-learning experiments where
    episode sampling must be deterministic.
    
    Args:
        seed: Random seed
        deterministic_cuda: Use deterministic CUDA operations (slower but reproducible)
    """
    # Python random seed
    random.seed(seed)
    
    # NumPy random seed
    np.random.seed(seed)
    
    # PyTorch random seeds
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU
    
    # Make CuDNN deterministic (slower but reproducible)
    if deterministic_cuda and torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Warn about performance impact
        warnings.warn(
            "Deterministic CUDA operations enabled. This may reduce performance."
        )


def check_for_nan_inf(tensor: torch.Tensor, name: str = "tensor") -> bool:
    """
    Check tensor for NaN/Inf values and raise informative error.
    
    Useful for debugging numerical instabilities during training.
    
    Returns:
        True if tensor is clean, raises RuntimeError if problems found
    """
    if torch.isnan(tensor).any():
        raise RuntimeError(f"{name} contains NaN values")
    if torch.isinf(tensor).any():
        raise RuntimeError(f"{name} contains Inf values") 
    return True


def stable_log_prob(probs: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """
    Compute log probabilities with numerical stability.
    
    Adds epsilon before taking logarithm to prevent log(0).
    """
    return torch.log(probs + eps)


def temperature_scaled_logits(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Apply temperature scaling to logits with bounds checking.
    
    Temperature scaling should be applied to logits BEFORE softmax,
    never to probabilities after softmax.
    
    Args:
        logits: Raw logits
        temperature: Temperature parameter (τ > 0)
        
    Returns:
        Temperature-scaled logits
    """
    if temperature <= 0:
        raise ValueError(f"Temperature must be positive, got {temperature}")
    
    return logits / temperature


class NumericalStabilityMonitor:
    """
    Monitor for numerical stability during training.
    
    Tracks gradient norms, parameter norms, and loss values
    to detect training instabilities early.
    """
    
    def __init__(self, warn_threshold: float = 1e6):
        self.warn_threshold = warn_threshold
        self.gradient_norms = []
        self.parameter_norms = []
        self.losses = []
        
    def log_gradients(self, model: torch.nn.Module):
        """Log gradient norms for monitoring."""
        grad_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                grad_norm += param.grad.data.norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5
        
        self.gradient_norms.append(grad_norm)
        
        if grad_norm > self.warn_threshold:
            warnings.warn(f"Large gradient norm detected: {grad_norm:.2e}")
            
    def log_parameters(self, model: torch.nn.Module):
        """Log parameter norms for monitoring.""" 
        param_norm = 0.0
        for param in model.parameters():
            param_norm += param.data.norm(2).item() ** 2
        param_norm = param_norm ** 0.5
        
        self.parameter_norms.append(param_norm)
        
        if param_norm > self.warn_threshold:
            warnings.warn(f"Large parameter norm detected: {param_norm:.2e}")
    
    def log_loss(self, loss: torch.Tensor):
        """Log loss values for monitoring."""
        loss_val = loss.item()
        self.losses.append(loss_val)
        
        if loss_val > self.warn_threshold or torch.isnan(loss).any():
            warnings.warn(f"Problematic loss value: {loss_val}")
    
    def get_stats(self) -> dict:
        """Get monitoring statistics."""
        return {
            'gradient_norm_mean': np.mean(self.gradient_norms) if self.gradient_norms else 0.0,
            'gradient_norm_std': np.std(self.gradient_norms) if self.gradient_norms else 0.0,
            'parameter_norm_mean': np.mean(self.parameter_norms) if self.parameter_norms else 0.0,
            'loss_mean': np.mean(self.losses) if self.losses else 0.0,
            'loss_std': np.std(self.losses) if self.losses else 0.0
        }


# Context manager for deterministic operations
class DeterministicMode:
    """
    Context manager for deterministic operations.
    
    Temporarily sets deterministic flags for reproducible computation
    within a specific block of code.
    """
    
    def __init__(self, seed: int):
        self.seed = seed
        self.old_deterministic = None
        self.old_benchmark = None
    
    def __enter__(self):
        # Store old settings
        if torch.cuda.is_available():
            self.old_deterministic = torch.backends.cudnn.deterministic
            self.old_benchmark = torch.backends.cudnn.benchmark
        
        # Set deterministic mode
        seed_everything(self.seed, deterministic_cuda=True)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore old settings
        if torch.cuda.is_available() and self.old_deterministic is not None:
            torch.backends.cudnn.deterministic = self.old_deterministic
            torch.backends.cudnn.benchmark = self.old_benchmark


if __name__ == "__main__":
    # Test numerical stability utilities
    print("Numerical Stability Utilities Test")
    print("=" * 40)
    
    # Test stable loss computation
    logits = torch.randn(32, 5)
    targets = torch.randint(0, 5, (32,))
    
    stable_loss = stable_softmax_ce_loss(logits, targets, temperature=2.0)
    regular_loss = F.cross_entropy(logits / 2.0, targets)
    
    print(f"Stable loss: {stable_loss.item():.6f}")
    print(f"Regular loss: {regular_loss.item():.6f}")
    print(f"Difference: {abs(stable_loss.item() - regular_loss.item()):.8f}")
    
    # Test safe normalization
    features = torch.randn(10, 64)
    normalized = safe_normalize(features)
    norms = normalized.norm(dim=1)
    print(f"Normalized feature norms: {norms.min():.6f} - {norms.max():.6f}")
    
    # Test distance computation
    query = torch.randn(5, 32)
    prototypes = torch.randn(3, 32) 
    
    euclidean_dist = safe_distance_computation(query, prototypes, "euclidean")
    cosine_dist = safe_distance_computation(query, prototypes, "cosine")
    
    print(f"Euclidean distances shape: {euclidean_dist.shape}")
    print(f"Cosine distances shape: {cosine_dist.shape}")
    
    # Test deterministic seeding
    with DeterministicMode(seed=42):
        x1 = torch.randn(5)
        
    with DeterministicMode(seed=42):
        x2 = torch.randn(5)
        
    print(f"Deterministic test: {torch.allclose(x1, x2)}")
    
    print("\n✓ All numerical stability tests passed")