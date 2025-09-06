"""
💰 SUPPORT THIS RESEARCH - PLEASE DONATE! 💰

🙏 If this library helps your research or project, please consider donating:
💳 PayPal: https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS
⭐ GitHub Sponsors: https://github.com/sponsors/benedictchen

👨‍💻 Author: Benedict Chen
💰 Donations: Help support this work! Buy me a coffee ☕, beer 🍺, lamborghini 🏎️, or private island 🏝️
💖 Please consider recurring donations to fully support continued research

Your support enables cutting-edge AI research for everyone! 🚀

Research-Grade Mathematical Utilities
====================================

Numerically stable mathematical operations for meta-learning algorithms.
Implements best practices from numerical analysis and research literature.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F

__all__ = ["pairwise_sqeuclidean", "cosine_logits", "_eps_like"]


def _eps_like(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Create epsilon tensor with same dtype/device as input for numerical stability."""
    return torch.full((), eps, dtype=x.dtype, device=x.device)


def pairwise_sqeuclidean(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise squared Euclidean distances with numerical stability.
    
    Uses the identity: ||a - b||² = ||a||² + ||b||² - 2a^T b
    with clamping to prevent tiny negatives from floating point errors.
    
    Args:
        a: [N, D] tensor
        b: [M, D] tensor
        
    Returns:
        [N, M] tensor of squared distances
        
    Mathematical Foundation:
        This approach is numerically superior to the naive (a-b)².sum() 
        because it avoids intermediate subtraction that can amplify 
        floating point errors, especially when a≈b.
    """
    # ||a||² for each row: [N, 1]
    a2 = (a * a).sum(dim=-1, keepdim=True)
    # ||b||² for each row: [1, M] 
    b2 = (b * b).sum(dim=-1, keepdim=True).transpose(0, 1)
    # -2a^T b: [N, M]
    cross = -2.0 * (a @ b.transpose(0, 1))
    # Clamp to prevent numerical negatives (should be >= 0 mathematically)
    return torch.clamp(a2 + b2 + cross, min=0.0)


def cosine_logits(a: torch.Tensor, b: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    """
    Compute cosine similarity logits with unified temperature semantics.
    
    Args:
        a: [N, D] query features
        b: [M, D] support features  
        tau: Temperature parameter (higher = less confident, softer predictions)
        
    Returns:
        [N, M] cosine similarity logits
        
    Mathematical Foundation:
        - Epsilon guard prevents division by zero when ||x|| = 0
        - Unified temperature semantics: logits = cosine / tau
        - Higher tau → softer probability distributions (higher entropy)
        - This matches squared Euclidean behavior: logits = -dist / tau
    """
    if tau <= 0:
        raise ValueError("tau must be > 0")
    eps = _eps_like(a)
    # L2 normalize with epsilon guard against zero norms
    a_norm = a / (a.norm(dim=-1, keepdim=True) + eps)
    b_norm = b / (b.norm(dim=-1, keepdim=True) + eps)
    # Unified temperature scaling: divide by tau (not multiply)
    cosine_sim = a_norm @ b_norm.transpose(0, 1)
    return cosine_sim / tau
