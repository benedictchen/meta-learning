from __future__ import annotations

"""
Core utility functions for meta-learning.

This module contains utility functions that don't depend on complex data structures,
preventing circular imports while providing essential functionality.
"""

from typing import Tuple
import torch
from ..shared.types import Episode


def partition_task(data: torch.Tensor, labels: torch.Tensor, shots: int = 1) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """
    Partition a classification task into support and query sets.
    
    This is the essential function that learn2learn provides that we were missing.
    The support set will contain `shots` samples per class, the query will take 
    the remaining samples.
    
    Args:
        data: Data to be partitioned into support and query [N, ...]
        labels: Labels of each data sample [N]
        shots: Number of data samples per class in the support set
        
    Returns:
        ((support_data, support_labels), (query_data, query_labels))
        
    Example:
        >>> X, y = taskset.sample()
        >>> (X_support, y_support), (X_query, y_query) = partition_task(X, y, shots=5)
    """
    assert data.size(0) == labels.size(0)
    unique_labels = labels.unique()
    ways = unique_labels.numel()
    data_shape = data.shape[1:]
    num_support = ways * shots
    num_query = data.size(0) - num_support
    assert num_query % ways == 0, 'Only query_shot == support_shot supported.'
    query_shots = num_query // ways
    
    support_data = torch.empty(
        (num_support,) + data_shape,
        device=data.device,
        dtype=data.dtype,
    )
    support_labels = torch.empty(
        num_support,
        device=labels.device,
        dtype=labels.dtype,
    )
    query_data = torch.empty(
        (num_query, ) + data_shape,
        device=data.device,
        dtype=data.dtype,
    )
    query_labels = torch.empty(
        num_query,
        device=labels.device,
        dtype=labels.dtype,
    )
    
    for i, label in enumerate(unique_labels):
        support_start = i * shots
        support_end = support_start + shots
        query_start = i * query_shots
        query_end = query_start + query_shots

        # Filter data
        label_data = data[labels == label]
        num_label_data = label_data.size(0)
        assert num_label_data == shots + query_shots, \
            'Only same number of query per label supported.'

        # Set value of labels
        support_labels[support_start:support_end].fill_(label)
        query_labels[query_start:query_end].fill_(label)

        # Set value of data
        support_data[support_start:support_end].copy_(label_data[:shots])
        query_data[query_start:query_end].copy_(label_data[shots:])

    return (support_data, support_labels), (query_data, query_labels)


def remap_labels(y_support: torch.Tensor, y_query: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Remap labels to contiguous integers starting from 0.
    
    This function takes support and query labels that may have arbitrary integer
    values and remaps them to contiguous integers [0, 1, 2, ..., n_classes-1].
    The mapping is determined by the unique classes present in the support set.
    
    Args:
        y_support (torch.Tensor): Support set labels of shape [n_support].
            Can contain any integer values.
        y_query (torch.Tensor): Query set labels of shape [n_query].
            Must be a subset of classes in y_support.
            
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - Remapped support labels of shape [n_support] with values [0, n_classes-1]
            - Remapped query labels of shape [n_query] with values [0, n_classes-1]
            
    Raises:
        KeyError: If query labels contain classes not present in support set.
        
    Examples:
        >>> import torch
        >>> from meta_learning.core.utils import remap_labels
        >>> 
        >>> # Labels with arbitrary values
        >>> y_support = torch.tensor([10, 25, 10, 37, 25])
        >>> y_query = torch.tensor([10, 10, 25, 37, 37, 25])
        >>> 
        >>> # Remap to contiguous [0, 1, 2]
        >>> support_remapped, query_remapped = remap_labels(y_support, y_query)
        >>> print(support_remapped)  # tensor([0, 1, 0, 2, 1])
        >>> print(query_remapped)    # tensor([0, 0, 1, 2, 2, 1])
        >>> 
        >>> # Verify mapping consistency
        >>> print(torch.unique(support_remapped))  # tensor([0, 1, 2])
    """
    classes = torch.unique(y_support)
    mapping = {c.item(): i for i, c in enumerate(classes)}
    ys = torch.tensor([mapping[int(c.item())] for c in y_support], device=y_support.device)
    yq = torch.tensor([mapping[int(c.item())] for c in y_query], device=y_query.device)
    return ys.long(), yq.long()


def create_episode_from_partition(data: torch.Tensor, labels: torch.Tensor, shots: int = 1) -> Episode:
    """Create Episode from data using partition_task utility.
    
    This is a factory function that combines partition_task with Episode creation,
    providing a clean interface for creating episodes from raw data.
    
    Args:
        data: Data tensor [N, ...]
        labels: Label tensor [N]
        shots: Number of support shots per class
        
    Returns:
        Episode created by partitioning the data
        
    Example:
        >>> data = torch.randn(50, 32)  # 50 samples, 32 features
        >>> labels = torch.randint(0, 5, (50,))  # 5 classes
        >>> episode = create_episode_from_partition(data, labels, shots=2)
    """
    (support_data, support_labels), (query_data, query_labels) = partition_task(data, labels, shots)
    return Episode(support_data, support_labels, query_data, query_labels)