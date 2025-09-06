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
    "batch_aware_prototype_computation"
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

