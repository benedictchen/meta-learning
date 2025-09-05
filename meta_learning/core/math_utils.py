from __future__ import annotations
import torch
import torch.nn.functional as F

__all__ = ["pairwise_sqeuclidean", "cosine_logits"]

def pairwise_sqeuclidean(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a[:, None] - b[None, :]).pow(2).sum(dim=-1)

def cosine_logits(a: torch.Tensor, b: torch.Tensor, tau: float = 10.0) -> torch.Tensor:
    a = F.normalize(a, p=2, dim=-1)
    b = F.normalize(b, p=2, dim=-1)
    return tau * (a @ b.t())
