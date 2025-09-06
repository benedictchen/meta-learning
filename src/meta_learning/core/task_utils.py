"""
💰 SUPPORT THIS RESEARCH - PLEASE DONATE! 💰

🙏 If this library helps your research, please donate:
💳 PayPal: https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS
⭐ GitHub Sponsors: https://github.com/sponsors/benedictchen

Task Utilities for Meta-Learning
===============================

Core utilities for task partitioning and episode creation.
Extracted to break circular dependencies between Episode and data utilities.
"""
import torch
from typing import Tuple


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


def remap_labels(labels: torch.Tensor) -> torch.Tensor:
    """
    Remap labels to be consecutive integers starting from 0.
    
    Args:
        labels: Input labels that may not be consecutive
        
    Returns:
        Remapped labels as consecutive integers
        
    Example:
        >>> labels = torch.tensor([3, 7, 3, 7, 11])
        >>> remapped = remap_labels(labels)
        >>> print(remapped)  # tensor([0, 1, 0, 1, 2])
    """
    unique_labels = labels.unique(sorted=True)
    remapped = torch.zeros_like(labels)
    for i, label in enumerate(unique_labels):
        remapped[labels == label] = i
    return remapped