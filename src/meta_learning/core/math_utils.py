import torch
import torch.nn.functional as F

__all__ = ["pairwise_sqeuclidean", "cosine_logits"]

def pairwise_sqeuclidean(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute pairwise squared Euclidean distances between two sets of vectors.
    
    Args:
        a: Tensor of shape (N, D) - first set of vectors
        b: Tensor of shape (M, D) - second set of vectors
        
    Returns:
        Tensor of shape (N, M) containing squared Euclidean distances
        
    Based on Snell et al. 2017: d(f_φ(x), c_k) = ||f_φ(x) - c_k||²
    """
    return (a[:, None] - b[None, :]).pow(2).sum(dim=-1)

def cosine_logits(a: torch.Tensor, b: torch.Tensor, tau: float = 10.0) -> torch.Tensor:
    """Compute cosine similarity logits with temperature scaling.
    
    Args:
        a: Tensor of shape (N, D) - query embeddings
        b: Tensor of shape (M, D) - prototype embeddings  
        tau: Temperature parameter for scaling
        
    Returns:
        Tensor of shape (N, M) containing scaled cosine similarities
    """
    a = F.normalize(a, p=2, dim=-1)
    b = F.normalize(b, p=2, dim=-1)
    return tau * (a @ b.t())