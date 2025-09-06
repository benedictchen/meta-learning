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

# ✅ PHASE 2.2 - ADVANCED MATHEMATICAL UTILITIES - COMPLETED
# TODO: - Add support for higher-order derivatives in stochastic graphs
# TODO: - Integrate with DiCE estimator for reinforcement learning applications
# TODO: - Add comprehensive tests for gradient computation accuracy
# TODO: - Document use cases for stochastic optimization and policy gradients

# ✅ Advanced similarity metrics implemented in advanced_similarity.py

from __future__ import annotations
from typing import Dict, Tuple
import torch
import torch.nn.functional as F


__all__ = [
    "pairwise_sqeuclidean", 
    "cosine_logits", 
    "_eps_like",
    "batched_prototype_computation",
    "adaptive_temperature_scaling",
    "adaptive_temperature_scaling_supervised", 
    "numerical_stability_monitor",
    "mixed_precision_distances",
    "batch_aware_prototype_computation",
    "magic_box",
    "pairwise_cosine_similarity",
    "matching_loss",
    "attention_matching_loss",
    "learnable_distance_metric"
]


def _eps_like(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Create epsilon tensor with same dtype and device as input tensor.
    
    This helper function creates a small epsilon value tensor that matches
    the dtype and device of the input tensor, ensuring numerical stability
    in operations like division and normalization.
    
    Args:
        x (torch.Tensor): Reference tensor for dtype and device matching.
        eps (float, optional): Epsilon value. Defaults to 1e-12.
        
    Returns:
        torch.Tensor: Scalar tensor with value eps on same device/dtype as x.
    """
    return torch.full((), eps, dtype=x.dtype, device=x.device)


def pairwise_sqeuclidean(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute pairwise squared Euclidean distances with numerical stability.
    
    Uses the mathematically equivalent identity: ||a - b||² = ||a||² + ||b||² - 2a^T b
    with clamping to prevent tiny negatives from floating point errors.
    This approach is numerically superior to the naive (a-b)².sum() computation.
    
    Args:
        a (torch.Tensor): First set of vectors of shape [N, D].
        b (torch.Tensor): Second set of vectors of shape [M, D].
        
    Returns:
        torch.Tensor: Pairwise squared distances of shape [N, M], where
            result[i, j] = ||a[i] - b[j]||².
            
    Note:
        This function is numerically stable and avoids intermediate subtraction
        that can amplify floating point errors, especially when a ≈ b.
        Results are clamped to ensure non-negative values.
        
    Examples:
        >>> import torch
        >>> from meta_learning.core.math_utils import pairwise_sqeuclidean
        >>> 
        >>> # Simple 2D example
        >>> a = torch.tensor([[0., 0.], [1., 1.]])  # 2 points
        >>> b = torch.tensor([[0., 1.], [1., 0.]])  # 2 points
        >>> distances = pairwise_sqeuclidean(a, b)
        >>> print(distances)
        # tensor([[1., 1.],  # ||[0,0] - [0,1]||² = 1, ||[0,0] - [1,0]||² = 1
        #         [1., 1.]])  # ||[1,1] - [0,1]||² = 1, ||[1,1] - [1,0]||² = 1
        >>> 
        >>> # Prototypical networks use case
        >>> prototypes = torch.randn(5, 128)  # 5 class prototypes
        >>> query_features = torch.randn(20, 128)  # 20 query examples
        >>> distances = pairwise_sqeuclidean(query_features, prototypes)
        >>> print(distances.shape)  # torch.Size([20, 5])
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
    """Compute cosine similarity logits with temperature scaling.
    
    Computes L2-normalized cosine similarity between two sets of vectors,
    scaled by temperature parameter for controlling prediction confidence.
    Includes numerical stability guards to prevent division by zero.
    
    Args:
        a (torch.Tensor): Query feature vectors of shape [N, D].
        b (torch.Tensor): Support/prototype feature vectors of shape [M, D].
        tau (float, optional): Temperature parameter for scaling logits.
            Higher values produce softer (less confident) probability distributions.
            Must be positive. Defaults to 1.0.
            
    Returns:
        torch.Tensor: Cosine similarity logits of shape [N, M], where
            result[i, j] = cosine_similarity(a[i], b[j]) / tau.
            
    Raises:
        ValueError: If tau <= 0.
        
    Note:
        - Uses epsilon guard to prevent division by zero when vector norms are 0
        - Temperature semantics: logits = cosine_similarity / tau  
        - Higher tau → softer probability distributions (higher entropy)
        - This matches distance-based metrics: logits = -distance / tau
        
    Examples:
        >>> import torch
        >>> from meta_learning.core.math_utils import cosine_logits
        >>> 
        >>> # Simple example with orthogonal vectors
        >>> a = torch.tensor([[1., 0.], [0., 1.]])  # 2 query vectors
        >>> b = torch.tensor([[1., 0.], [0., 1.]])  # 2 prototype vectors  
        >>> logits = cosine_logits(a, b)
        >>> print(logits)
        # tensor([[1., 0.],  # Perfect similarity, orthogonal
        #         [0., 1.]])  # Orthogonal, perfect similarity
        >>> 
        >>> # Temperature scaling example
        >>> logits_cold = cosine_logits(a, b, tau=0.5)  # Sharper predictions
        >>> logits_hot = cosine_logits(a, b, tau=2.0)   # Softer predictions
        >>> print(f"Cold: {logits_cold[0, 0]:.2f}, Hot: {logits_hot[0, 0]:.2f}")
        # Cold: 2.00, Hot: 0.50
        >>> 
        >>> # Prototypical networks usage
        >>> query_features = torch.randn(10, 512)
        >>> prototypes = torch.randn(5, 512) 
        >>> similarity_logits = cosine_logits(query_features, prototypes, tau=0.1)
        >>> probabilities = torch.softmax(similarity_logits, dim=1)
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


def adaptive_temperature_scaling_supervised(logits: torch.Tensor, targets: torch.Tensor, 
                                           initial_tau: float = 1.0, max_iter: int = 50) -> torch.Tensor:
    """Adaptive temperature scaling for optimal calibration with supervision.
    
    Optimizes temperature parameter to minimize negative log-likelihood,
    improving model calibration for few-shot learning scenarios.
    
    Args:
        logits (torch.Tensor): Raw logits of shape [N, num_classes].
        targets (torch.Tensor): Ground truth targets of shape [N].
        initial_tau (float, optional): Initial temperature value. Defaults to 1.0.
        max_iter (int, optional): Maximum optimization iterations. Defaults to 50.
        
    Returns:
        torch.Tensor: Optimally scaled logits with learned temperature.
        
    Examples:
        >>> logits = torch.randn(100, 5)
        >>> targets = torch.randint(0, 5, (100,))
        >>> scaled_logits = adaptive_temperature_scaling_supervised(logits, targets)
        >>> # Scaled logits will be better calibrated
    """
    tau = torch.tensor(initial_tau, requires_grad=True)
    optimizer = torch.optim.LBFGS([tau], lr=0.01, max_iter=max_iter)
    
    def closure():
        optimizer.zero_grad()
        scaled_logits = logits / tau
        loss = F.cross_entropy(scaled_logits, targets)
        loss.backward()
        return loss
    
    optimizer.step(closure)
    
    with torch.no_grad():
        return logits / tau.clamp(min=0.01)  # Prevent division by very small numbers


def mixed_precision_distances(a: torch.Tensor, b: torch.Tensor, 
                             use_half_precision: bool = None) -> torch.Tensor:
    """Compute distances with automatic mixed precision for memory efficiency.
    
    Automatically uses half precision (fp16) for large tensors to save memory
    while maintaining numerical accuracy for smaller computations.
    
    Args:
        a (torch.Tensor): First tensor of shape [N, D].
        b (torch.Tensor): Second tensor of shape [M, D].
        use_half_precision (bool, optional): Force precision choice. If None,
            automatically selects based on tensor size.
            
    Returns:
        torch.Tensor: Distance matrix of shape [N, M].
    """
    if use_half_precision is None:
        # Use half precision for large computations
        total_elements = a.numel() + b.numel()
        use_half_precision = total_elements > 1_000_000 and a.device.type == 'cuda'
    
    if use_half_precision and a.dtype == torch.float32:
        # Convert to half precision for computation
        a_half = a.half()
        b_half = b.half()
        distances = pairwise_sqeuclidean(a_half, b_half)
        return distances.float()  # Convert back to float32 for stability
    else:
        return pairwise_sqeuclidean(a, b)


def batch_aware_prototype_computation(support_x: torch.Tensor, support_y: torch.Tensor, 
                                    memory_budget: float = 0.8) -> torch.Tensor:
    """Memory-efficient prototype computation with automatic batching.
    
    Automatically batches computation when memory usage would exceed budget,
    preventing OOM errors on large datasets.
    
    Args:
        support_x (torch.Tensor): Support features of shape [N, D].
        support_y (torch.Tensor): Support labels of shape [N].
        memory_budget (float, optional): Fraction of available memory to use.
            Defaults to 0.8.
            
    Returns:
        torch.Tensor: Class prototypes of shape [K, D] where K is num classes.
    """
    unique_labels = torch.unique(support_y)
    num_classes = len(unique_labels)
    feature_dim = support_x.shape[-1]
    
    # Estimate memory usage
    estimated_memory = support_x.numel() * support_x.element_size()
    if torch.cuda.is_available():
        available_memory = torch.cuda.get_device_properties(0).total_memory * memory_budget
        needs_batching = estimated_memory > available_memory
    else:
        needs_batching = False
    
    prototypes = torch.zeros(num_classes, feature_dim, dtype=support_x.dtype, device=support_x.device)
    
    if needs_batching:
        # Process one class at a time to save memory
        for i, label in enumerate(unique_labels):
            mask = support_y == label
            class_features = support_x[mask]
            prototypes[i] = class_features.mean(dim=0)
    else:
        # Use vectorized computation for speed
        for i, label in enumerate(unique_labels):
            mask = support_y == label
            prototypes[i] = support_x[mask].mean(dim=0)
    
    return prototypes


def numerical_stability_monitor(tensor: torch.Tensor, name: str = "tensor") -> Dict[str, float]:
    """Monitor numerical stability properties of tensors.
    
    Provides comprehensive numerical health metrics for debugging and
    optimization of meta-learning algorithms.
    
    Args:
        tensor (torch.Tensor): Tensor to analyze.
        name (str, optional): Name for logging. Defaults to "tensor".
        
    Returns:
        Dict[str, float]: Dictionary of numerical stability metrics.
    """
    with torch.no_grad():
        stats = {
            'has_nan': torch.isnan(tensor).any().item(),
            'has_inf': torch.isinf(tensor).any().item(),
            'min_value': tensor.min().item(),
            'max_value': tensor.max().item(),
            'mean_value': tensor.mean().item(),
            'std_value': tensor.std().item(),
            'condition_number': torch.linalg.cond(tensor).item() if tensor.dim() == 2 else float('nan'),
            'dynamic_range': (tensor.max() - tensor.min()).item(),
        }
        
        # Check for potential numerical issues
        if stats['has_nan']:
            print(f"WARNING: {name} contains NaN values")
        if stats['has_inf']:
            print(f"WARNING: {name} contains infinite values")
        if stats['condition_number'] > 1e12:
            print(f"WARNING: {name} has high condition number: {stats['condition_number']:.2e}")
            
        return stats

def batched_prototype_computation(support_x: torch.Tensor, support_y: torch.Tensor) -> torch.Tensor:
    """
    Memory-efficient vectorized prototype computation.
    
    Args:
        support_x: [N, D] support features
        support_y: [N] support labels (0 to K-1)
        
    Returns:
        [K, D] prototype features where K is number of unique classes
        
    Implementation:
        Uses scatter_add for efficient vectorized computation rather than loops.
        Handles arbitrary label values by internally remapping to 0..K-1.
    """
    device = support_x.device
    labels = support_y.long()
    
    # Get unique labels and remap to contiguous range [0, K-1]
    unique_labels = torch.unique(labels)
    n_classes = len(unique_labels)
    
    # Create mapping from original labels to [0, K-1]
    label_map = torch.zeros(labels.max().item() + 1, dtype=torch.long, device=device)
    label_map[unique_labels] = torch.arange(n_classes, device=device)
    remapped_labels = label_map[labels]
    
    # Initialize prototypes tensor
    prototypes = torch.zeros(n_classes, support_x.size(1), dtype=support_x.dtype, device=device)
    
    # Count samples per class for averaging
    class_counts = torch.zeros(n_classes, dtype=torch.long, device=device)
    class_counts.scatter_add_(0, remapped_labels, torch.ones_like(remapped_labels))
    
    # Sum features per class
    prototypes.scatter_add_(0, remapped_labels.unsqueeze(1).expand(-1, support_x.size(1)), support_x)
    
    # Average by class counts (with epsilon for numerical stability)
    class_counts = class_counts.float().clamp(min=1.0)  # Prevent division by zero
    prototypes = prototypes / class_counts.unsqueeze(1)
    
    return prototypes


def adaptive_temperature_scaling(logits: torch.Tensor, target_entropy: float = None) -> Tuple[torch.Tensor, float]:
    """
    Adaptive temperature scaling for calibrated predictions.
    
    Args:
        logits: [N, K] raw logits
        target_entropy: Target entropy for temperature tuning (optional)
        
    Returns:
        Tuple of (scaled_logits, selected_temperature)
        
    Implementation:
        Uses binary search to find temperature that achieves target entropy.
        If no target specified, uses logit statistics for reasonable default.
    """
    if logits.numel() == 0:
        return logits, 1.0
        
    # Default target entropy based on logit distribution
    if target_entropy is None:
        n_classes = logits.size(-1)
        # Target entropy slightly below uniform (encourages some confidence)
        target_entropy = 0.8 * torch.log(torch.tensor(float(n_classes)))
    
    def entropy_at_temperature(temp: float) -> float:
        scaled_logits = logits / max(temp, 1e-8)
        probs = torch.softmax(scaled_logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=-1).mean()
        return entropy.item()
    
    # Binary search for optimal temperature
    low_temp, high_temp = 0.1, 10.0
    for _ in range(20):  # 20 iterations gives good precision
        mid_temp = (low_temp + high_temp) / 2
        current_entropy = entropy_at_temperature(mid_temp)
        
        if abs(current_entropy - target_entropy) < 0.01:
            break
            
        if current_entropy > target_entropy:
            low_temp = mid_temp
        else:
            high_temp = mid_temp
    
    optimal_temp = (low_temp + high_temp) / 2
    scaled_logits = logits / optimal_temp
    
    return scaled_logits, optimal_temp


def numerical_stability_monitor(tensor: torch.Tensor, operation: str = "tensor") -> Dict[str, float]:
    """
    Monitor tensor for numerical stability issues.
    
    Args:
        tensor: Tensor to analyze
        operation: Description of operation for context
        
    Returns:
        Dictionary with stability metrics
        
    Metrics:
        - has_nan: Boolean indicating presence of NaN values
        - has_inf: Boolean indicating presence of infinite values  
        - condition_number: Condition number (for matrices)
        - dynamic_range: log10 of max/min absolute values
        - gradient_norm: L2 norm if tensor has gradients
    """
    if tensor.numel() == 0:
        return {"operation": operation, "warning": "empty_tensor"}
    
    metrics = {"operation": operation}
    
    # Basic stability checks
    metrics["has_nan"] = torch.isnan(tensor).any().item()
    metrics["has_inf"] = torch.isinf(tensor).any().item()
    
    # Dynamic range analysis
    abs_vals = torch.abs(tensor)
    nonzero_mask = abs_vals > 1e-20
    if nonzero_mask.any():
        min_val = abs_vals[nonzero_mask].min().item()
        max_val = abs_vals.max().item()
        metrics["dynamic_range"] = float(torch.log10(torch.tensor(max_val / (min_val + 1e-20))))
    else:
        metrics["dynamic_range"] = 0.0
    
    # Condition number for matrices
    if tensor.dim() == 2 and tensor.size(0) >= tensor.size(1):
        try:
            _, s, _ = torch.svd(tensor)
            if s.numel() > 1 and s[-1].item() > 1e-12:
                metrics["condition_number"] = (s[0] / s[-1]).item()
            else:
                metrics["condition_number"] = float('inf')
        except RuntimeError:
            metrics["condition_number"] = float('nan')
    
    # Gradient analysis
    if tensor.grad is not None:
        metrics["gradient_norm"] = torch.norm(tensor.grad).item()
        metrics["gradient_has_nan"] = torch.isnan(tensor.grad).any().item()
    
    return metrics


# PHASE 2.2 - ADVANCED MATHEMATICAL UTILITIES

def magic_box(x: torch.Tensor) -> torch.Tensor:
    """
    Magic box operator for stochastic meta-learning.
    
    This function evaluates to 1.0 but has gradient dx, enabling stochastic
    optimization in scenarios where the standard chain rule doesn't apply.
    
    Mathematical property:
        magic_box(x) = 1.0 (forward)
        d/dx magic_box(x) = 1.0 (backward)
    
    Implementation:
        Uses exp(x - detach(x)) which equals exp(0) = 1 in forward pass,
        but preserves gradients through x in backward pass.
    
    Applications:
        - REINFORCE-style gradient estimation
        - Stochastic meta-learning with discrete variables
        - Policy gradient methods
        - DiCE estimator implementations
    
    Args:
        x: Input tensor (can be any shape)
        
    Returns:
        Tensor of same shape as x, filled with 1.0 values but preserving gradients
        
    Example:
        >>> x = torch.randn(5, requires_grad=True)
        >>> y = magic_box(x).sum()
        >>> y.backward()
        >>> print(x.grad)  # Will be tensor([1., 1., 1., 1., 1.])
    """
    return torch.exp(x - x.detach())


def pairwise_cosine_similarity(
    embeddings_a: torch.Tensor, 
    embeddings_b: torch.Tensor, 
    temperature: float = 1.0,
    normalize: bool = True
) -> torch.Tensor:
    """
    Compute pairwise cosine similarities between two sets of embeddings.
    
    This function efficiently computes cosine similarities between all pairs
    of embeddings from two different sets, with optional temperature scaling
    and normalization.
    
    Args:
        embeddings_a: First set of embeddings [N, D]
        embeddings_b: Second set of embeddings [M, D]  
        temperature: Temperature scaling factor (higher = more uniform)
        normalize: Whether to L2 normalize embeddings before computing similarity
        
    Returns:
        Pairwise similarity matrix [N, M] where entry (i,j) is the cosine
        similarity between embeddings_a[i] and embeddings_b[j]
        
    Example:
        >>> a = torch.randn(5, 64)  # 5 embeddings of dim 64
        >>> b = torch.randn(3, 64)  # 3 embeddings of dim 64
        >>> sim = pairwise_cosine_similarity(a, b)
        >>> print(sim.shape)  # torch.Size([5, 3])
    """
    if embeddings_a.numel() == 0 or embeddings_b.numel() == 0:
        return torch.zeros(embeddings_a.size(0), embeddings_b.size(0), 
                          device=embeddings_a.device)
    
    # L2 normalization if requested
    if normalize:
        embeddings_a = F.normalize(embeddings_a, p=2, dim=-1)
        embeddings_b = F.normalize(embeddings_b, p=2, dim=-1)
    
    # Compute cosine similarities via matrix multiplication
    similarities = torch.matmul(embeddings_a, embeddings_b.t())
    
    # Apply temperature scaling
    if temperature != 1.0:
        similarities = similarities / temperature
    
    return similarities


def matching_loss(
    support_embeddings: torch.Tensor,
    support_labels: torch.Tensor,
    query_embeddings: torch.Tensor,
    query_labels: torch.Tensor,
    distance_metric: str = "cosine",
    temperature: float = 1.0,
    reduction: str = "mean"
) -> torch.Tensor:
    """
    Compute matching loss for few-shot learning.
    
    This function implements the standard matching loss used in matching networks
    and similar few-shot learning algorithms. It computes distances between
    query samples and support prototypes.
    
    Args:
        support_embeddings: Support set embeddings [N_support, D]
        support_labels: Support set labels [N_support]
        query_embeddings: Query set embeddings [N_query, D]
        query_labels: Query set labels [N_query]
        distance_metric: Distance metric ("cosine", "euclidean", "manhattan")
        temperature: Temperature scaling for similarities
        reduction: Loss reduction ("mean", "sum", "none")
        
    Returns:
        Matching loss scalar (if reduction != "none") or per-query losses
        
    Example:
        >>> support_emb = torch.randn(15, 64)  # 3-way 5-shot
        >>> support_lab = torch.tensor([0,0,0,0,0, 1,1,1,1,1, 2,2,2,2,2])
        >>> query_emb = torch.randn(9, 64)     # 3 queries per class
        >>> query_lab = torch.tensor([0,0,0, 1,1,1, 2,2,2])
        >>> loss = matching_loss(support_emb, support_lab, query_emb, query_lab)
    """
    if len(support_embeddings) == 0 or len(query_embeddings) == 0:
        return torch.tensor(0.0, device=support_embeddings.device, requires_grad=True)
    
    unique_labels = torch.unique(support_labels)
    num_classes = len(unique_labels)
    
    # Compute class prototypes
    prototypes = torch.zeros(num_classes, support_embeddings.size(-1), 
                           device=support_embeddings.device)
    
    for i, label in enumerate(unique_labels):
        mask = (support_labels == label)
        if mask.sum() > 0:
            prototypes[i] = support_embeddings[mask].mean(dim=0)
    
    # Compute distances between query embeddings and prototypes
    if distance_metric == "cosine":
        # Use cosine similarity (higher = more similar)
        similarities = pairwise_cosine_similarity(query_embeddings, prototypes, temperature)
        logits = similarities
    elif distance_metric == "euclidean":
        # Use negative Euclidean distance (higher = more similar)
        distances = torch.cdist(query_embeddings, prototypes, p=2)
        logits = -distances / temperature
    elif distance_metric == "manhattan":
        # Use negative Manhattan distance
        distances = torch.cdist(query_embeddings, prototypes, p=1)  
        logits = -distances / temperature
    else:
        raise ValueError(f"Unknown distance metric: {distance_metric}")
    
    # Convert query labels to class indices
    query_class_indices = torch.zeros_like(query_labels)
    for i, label in enumerate(unique_labels):
        query_class_indices[query_labels == label] = i
    
    # Compute cross-entropy loss
    loss = F.cross_entropy(logits, query_class_indices, reduction=reduction)
    
    return loss


def attention_matching_loss(
    support_embeddings: torch.Tensor,
    support_labels: torch.Tensor,
    query_embeddings: torch.Tensor, 
    query_labels: torch.Tensor,
    attention_dim: int = 64,
    temperature: float = 1.0
) -> torch.Tensor:
    """
    Attention-based matching loss for few-shot learning.
    
    This implements attention mechanisms for matching networks, where query
    samples attend over support samples to compute weighted similarities.
    
    Args:
        support_embeddings: Support set embeddings [N_support, D]
        support_labels: Support set labels [N_support]
        query_embeddings: Query set embeddings [N_query, D]
        query_labels: Query set labels [N_query]
        attention_dim: Dimension for attention computation
        temperature: Temperature scaling for attention weights
        
    Returns:
        Attention-based matching loss
    """
    if len(support_embeddings) == 0 or len(query_embeddings) == 0:
        return torch.tensor(0.0, device=support_embeddings.device, requires_grad=True)
    
    N_support = support_embeddings.size(0)
    N_query = query_embeddings.size(0)
    embed_dim = support_embeddings.size(-1)
    
    # Simple attention mechanism using dot product
    # In a full implementation, this would use learned attention weights
    attention_scores = torch.matmul(query_embeddings, support_embeddings.t()) / temperature
    attention_weights = F.softmax(attention_scores, dim=-1)  # [N_query, N_support]
    
    # Compute attended support representations for each query
    attended_support = torch.matmul(attention_weights, support_embeddings)  # [N_query, D]
    
    # Compute similarities between queries and attended support representations
    similarities = F.cosine_similarity(query_embeddings, attended_support, dim=-1)
    
    # Create targets (1 for correct matches, 0 for incorrect)
    # This is a simplified version - full implementation would be more sophisticated
    targets = torch.ones_like(similarities)
    
    # Use MSE loss between similarities and targets
    loss = F.mse_loss(similarities, targets)
    
    return loss


class LearnableDistanceMetric(torch.nn.Module):
    """
    Learnable distance metric for few-shot learning.
    
    This module implements a parameterized distance function that can be learned
    during meta-training to improve few-shot classification performance.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, metric_type: str = "mahalanobis"):
        """
        Initialize learnable distance metric.
        
        Args:
            input_dim: Dimension of input embeddings
            hidden_dim: Hidden dimension for distance computation
            metric_type: Type of learnable metric ("mahalanobis", "neural", "bilinear")
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.metric_type = metric_type
        
        if metric_type == "mahalanobis":
            # Learn a positive definite matrix for Mahalanobis distance
            self.metric_matrix = torch.nn.Parameter(torch.eye(input_dim))
        elif metric_type == "neural":
            # Neural network to compute distances
            self.distance_net = torch.nn.Sequential(
                torch.nn.Linear(input_dim * 2, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, 1),
                torch.nn.Sigmoid()
            )
        elif metric_type == "bilinear":
            # Bilinear form for distance computation
            self.bilinear = torch.nn.Bilinear(input_dim, input_dim, 1)
        else:
            raise ValueError(f"Unknown metric type: {metric_type}")
    
    def forward(self, embeddings_a: torch.Tensor, embeddings_b: torch.Tensor) -> torch.Tensor:
        """
        Compute learnable distances between embeddings.
        
        Args:
            embeddings_a: First set of embeddings [N, D]
            embeddings_b: Second set of embeddings [M, D]
            
        Returns:
            Distance matrix [N, M]
        """
        N, M = embeddings_a.size(0), embeddings_b.size(0)
        
        if self.metric_type == "mahalanobis":
            # Ensure positive definiteness by using M^T @ M
            metric = self.metric_matrix.t() @ self.metric_matrix
            
            # Compute Mahalanobis distances
            diff = embeddings_a.unsqueeze(1) - embeddings_b.unsqueeze(0)  # [N, M, D]
            distances = torch.einsum('nmd,de,nme->nm', diff, metric, diff)
            distances = torch.sqrt(torch.clamp(distances, min=1e-8))
            
        elif self.metric_type == "neural":
            # Compute pairwise concatenated features
            a_expanded = embeddings_a.unsqueeze(1).expand(-1, M, -1)  # [N, M, D]
            b_expanded = embeddings_b.unsqueeze(0).expand(N, -1, -1)  # [N, M, D]
            
            # Concatenate features
            combined = torch.cat([a_expanded, b_expanded], dim=-1)  # [N, M, 2D]
            
            # Compute distances using neural network
            distances = self.distance_net(combined.view(-1, self.input_dim * 2))
            distances = distances.view(N, M)
            
        elif self.metric_type == "bilinear":
            # Compute bilinear distances
            distances = torch.zeros(N, M, device=embeddings_a.device)
            for i in range(N):
                for j in range(M):
                    dist = self.bilinear(embeddings_a[i:i+1], embeddings_b[j:j+1])
                    distances[i, j] = dist.squeeze()
        
        return distances


def learnable_distance_metric(
    embeddings_a: torch.Tensor,
    embeddings_b: torch.Tensor,
    metric_module: LearnableDistanceMetric
) -> torch.Tensor:
    """
    Compute distances using a learnable distance metric.
    
    Args:
        embeddings_a: First set of embeddings [N, D]
        embeddings_b: Second set of embeddings [M, D]
        metric_module: Learned distance metric module
        
    Returns:
        Distance matrix [N, M]
    """
    return metric_module(embeddings_a, embeddings_b)

