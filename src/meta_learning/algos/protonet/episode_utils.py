"""Episode creation utilities for few-shot learning."""

import torch
import numpy as np
from typing import Tuple, List, Optional


def make_episode(
    data: torch.Tensor,
    labels: torch.Tensor,
    n_way: int,
    n_shot: int,
    n_query: int,
    seed: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create a few-shot learning episode from data.
    
    Args:
        data: Full dataset (N, ...)
        labels: Labels for data (N,)
        n_way: Number of classes in episode
        n_shot: Number of support samples per class
        n_query: Number of query samples per class
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (support_data, support_labels, query_data, query_labels)
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    unique_labels = torch.unique(labels)
    if len(unique_labels) < n_way:
        raise ValueError(f"Not enough classes: need {n_way}, got {len(unique_labels)}")
    
    # Sample classes
    selected_classes = np.random.choice(
        unique_labels.numpy(), size=n_way, replace=False
    )
    
    support_data = []
    support_labels = []
    query_data = []
    query_labels = []
    
    for class_idx, cls in enumerate(selected_classes):
        # Get all samples for this class
        class_mask = labels == cls
        class_data = data[class_mask]
        
        if len(class_data) < n_shot + n_query:
            raise ValueError(f"Class {cls} has only {len(class_data)} samples, "
                           f"need {n_shot + n_query}")
        
        # Sample support and query indices
        indices = np.random.choice(
            len(class_data), size=n_shot + n_query, replace=False
        )
        
        support_indices = indices[:n_shot]
        query_indices = indices[n_shot:]
        
        # Add to episode
        support_data.append(class_data[support_indices])
        support_labels.extend([class_idx] * n_shot)
        
        query_data.append(class_data[query_indices])
        query_labels.extend([class_idx] * n_query)
    
    return (
        torch.cat(support_data),
        torch.tensor(support_labels),
        torch.cat(query_data),
        torch.tensor(query_labels)
    )


def fit_episode(model, support_data, support_labels, query_data, query_labels):
    """Evaluate model on a single episode.
    
    Args:
        model: ProtoHead model
        support_data: Support set data
        support_labels: Support set labels
        query_data: Query set data
        query_labels: Query set labels (for evaluation)
        
    Returns:
        Dictionary with accuracy and loss
    """
    with torch.no_grad():
        logits = model(support_data, support_labels, query_data)
        predictions = logits.argmax(dim=1)
        accuracy = (predictions == query_labels).float().mean().item()
        
        loss = torch.nn.functional.cross_entropy(logits, query_labels)
        
    return {
        'accuracy': accuracy,
        'loss': loss.item(),
        'predictions': predictions
    }