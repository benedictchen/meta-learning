from __future__ import annotations

"""
Shared type definitions for meta-learning package.

This module contains core data structures that are used throughout the package
but have no dependencies on other modules, preventing circular imports.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import torch


@dataclass 
class Episode:
    """A few-shot learning episode containing support and query sets.
    
    An Episode represents a single few-shot learning task, containing a support
    set for adaptation and a query set for evaluation. This is the fundamental
    data structure used throughout the meta-learning library.
    
    Attributes:
        support_x (torch.Tensor): Support set inputs of shape [n_support, ...].
        support_y (torch.Tensor): Support set labels of shape [n_support] with
            values in range [0, n_classes-1].
        query_x (torch.Tensor): Query set inputs of shape [n_query, ...].
        query_y (torch.Tensor): Query set labels of shape [n_query] with
            values in range [0, n_classes-1].
            
    Note:
        Labels must be contiguous integers starting from 0. Use remap_labels()
        if your data has non-contiguous or non-zero-based labels.
        
    Examples:
        >>> import torch
        >>> from meta_learning.shared.types import Episode
        >>> 
        >>> # Create a simple 5-way 1-shot episode
        >>> support_x = torch.randn(5, 64)  # 5 support examples, 64 features
        >>> support_y = torch.arange(5)     # Labels [0, 1, 2, 3, 4]
        >>> query_x = torch.randn(15, 64)   # 15 query examples
        >>> query_y = torch.repeat_interleave(torch.arange(5), 3)  # [0,0,0,1,1,1,...]
        >>> 
        >>> episode = Episode(support_x, support_y, query_x, query_y)
        >>> print(f"Classes: {episode.num_classes}")  # 5
        >>> print(f"Total samples: {episode.num_samples}")  # 20
    """
    support_x: torch.Tensor
    support_y: torch.Tensor
    query_x: torch.Tensor
    query_y: torch.Tensor
    
    def validate(self, *, expect_n_classes: Optional[int] = None) -> None:
        """Validate episode structure and constraints."""
        assert self.support_x.shape[0] == self.support_y.shape[0], "support X/Y mismatch"
        assert self.query_x.shape[0] == self.query_y.shape[0], "query X/Y mismatch"
        assert self.support_y.dtype == torch.int64 and self.query_y.dtype == torch.int64, "labels must be int64"
        assert self.support_y.dim() == 1 and self.query_y.dim() == 1, "labels must be 1D"
        classes = torch.unique(self.support_y)
        assert torch.all(torch.isin(torch.unique(self.query_y), classes)), "query labels not subset of support"
        if expect_n_classes is not None:
            assert classes.numel() == expect_n_classes, f"expected {expect_n_classes} classes, got {classes.numel()}"
        remapped = torch.sort(classes).values
        assert torch.equal(remapped, torch.arange(remapped.numel(), device=remapped.device)), "labels must be [0..C-1] contiguous"

    def to_device(self, device: torch.device, pin_memory: bool = False) -> Episode:
        """Move Episode to specified device with memory optimization."""
        def _to_device_with_pinning(tensor: torch.Tensor) -> torch.Tensor:
            if pin_memory and device.type == 'cuda' and tensor.device.type == 'cpu':
                return tensor.pin_memory().to(device, non_blocking=True)
            return tensor.to(device)
        
        return Episode(
            support_x=_to_device_with_pinning(self.support_x),
            support_y=_to_device_with_pinning(self.support_y), 
            query_x=_to_device_with_pinning(self.query_x),
            query_y=_to_device_with_pinning(self.query_y)
        )
    
    @property
    def num_classes(self) -> int:
        """Number of classes in episode (TorchMeta Task compatibility)."""
        return len(torch.unique(self.support_y))
    
    @property 
    def num_samples(self) -> int:
        """Total number of samples in episode."""
        return len(self.support_x) + len(self.query_x)
    
    def __len__(self) -> int:
        """Length for TorchMeta Task compatibility."""
        return self.num_samples
    
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get item for TorchMeta Task compatibility."""
        total_support = len(self.support_x)
        if index < total_support:
            return self.support_x[index], self.support_y[index]
        else:
            query_idx = index - total_support
            return self.query_x[query_idx], self.query_y[query_idx]

    @classmethod
    def from_partition(cls, data: torch.Tensor, labels: torch.Tensor, shots: int = 1) -> Episode:
        """Create Episode using partition_task approach.
        
        Args:
            data: Data tensor [N, ...]  
            labels: Label tensor [N]
            shots: Number of support shots per class
            
        Returns:
            Episode created by partitioning the data
        """
        # Import locally to avoid circular imports
        from ..core.utils import partition_task
        
        (support_data, support_labels), (query_data, query_labels) = partition_task(data, labels, shots)
        return cls(support_data, support_labels, query_data, query_labels)

    @classmethod
    def from_raw_data(cls, data: torch.Tensor, labels: torch.Tensor, 
                      n_shot: int = 1, n_query: int = 15, random_state: int = None) -> Episode:
        """Create Episode from raw data using balanced class sampling."""
        if random_state is not None:
            torch.manual_seed(random_state)
        
        unique_labels = torch.unique(labels)
        n_classes = len(unique_labels)
        
        support_data, support_labels = [], []
        query_data, query_labels = [], []
        
        for class_idx, label in enumerate(unique_labels):
            class_mask = labels == label
            class_data = data[class_mask]
            
            if len(class_data) < n_shot + n_query:
                raise ValueError(f"Class {label} has only {len(class_data)} samples, "
                               f"need {n_shot + n_query}")
            
            # Random permutation for sampling
            perm = torch.randperm(len(class_data))
            
            # Support set
            support_indices = perm[:n_shot]
            support_data.append(class_data[support_indices])
            support_labels.extend([class_idx] * n_shot)
            
            # Query set  
            query_indices = perm[n_shot:n_shot + n_query]
            query_data.append(class_data[query_indices])
            query_labels.extend([class_idx] * n_query)
        
        return cls(
            support_x=torch.cat(support_data),
            support_y=torch.tensor(support_labels, device=data.device),
            query_x=torch.cat(query_data),
            query_y=torch.tensor(query_labels, device=data.device)
        )