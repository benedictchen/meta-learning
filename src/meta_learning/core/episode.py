from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import torch

# Import shared utilities to break circular dependency
from .task_utils import partition_task, remap_labels


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
    """
    support_x: torch.Tensor
    support_y: torch.Tensor
    query_x: torch.Tensor
    query_y: torch.Tensor
    
    @classmethod
    def from_partition(
        cls,
        data: torch.Tensor,
        labels: torch.Tensor,
        shots: int = 1,
        remap: bool = True
    ) -> Episode:
        """Create episode from data by partitioning into support and query sets.
        
        Args:
            data: Data tensor to partition [N, ...]
            labels: Label tensor [N] 
            shots: Number of support samples per class
            remap: Whether to remap labels to consecutive integers
            
        Returns:
            Episode with partitioned data
            
        Example:
            >>> X, y = taskset.sample()
            >>> episode = Episode.from_partition(X, y, shots=5)
        """
        (support_x, support_y), (query_x, query_y) = partition_task(data, labels, shots)
        
        if remap:
            # Remap labels to be consecutive integers starting from 0
            unique_labels = support_y.unique(sorted=True)
            label_map = {label.item(): i for i, label in enumerate(unique_labels)}
            
            support_y_remapped = torch.tensor([label_map[label.item()] for label in support_y], 
                                            dtype=support_y.dtype, device=support_y.device)
            query_y_remapped = torch.tensor([label_map[label.item()] for label in query_y],
                                           dtype=query_y.dtype, device=query_y.device)
                                           
            return cls(support_x, support_y_remapped, query_x, query_y_remapped)
        else:
            return cls(support_x, support_y, query_x, query_y)
    
    @property
    def n_classes(self) -> int:
        """Number of classes in the episode."""
        return len(self.support_y.unique())
    
    @property
    def n_support(self) -> int:
        """Total number of support samples."""
        return len(self.support_x)
    
    @property
    def n_query(self) -> int:
        """Total number of query samples."""
        return len(self.query_x)
    
    @property
    def shots(self) -> int:
        """Number of support samples per class."""
        return self.n_support // self.n_classes
    
    def to(self, device: torch.device) -> Episode:
        """Move episode to specified device."""
        return Episode(
            self.support_x.to(device),
            self.support_y.to(device),
            self.query_x.to(device),
            self.query_y.to(device)
        )
    
    def __len__(self) -> int:
        """Total number of samples in episode."""
        return len(self.support_x) + len(self.query_x)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get sample at index (support first, then query)."""
        total_support = len(self.support_x)
        if index < total_support:
            return self.support_x[index], self.support_y[index]
        else:
            query_idx = index - total_support
            return self.query_x[query_idx], self.query_y[query_idx]
    
    def validate(self, *, expect_n_classes: Optional[int] = None) -> None:
        """Validate episode data consistency and format.
        
        Args:
            expect_n_classes: Expected number of classes (optional)
            
        Raises:
            AssertionError: If validation fails
        """
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


def remap_labels(y_support: torch.Tensor, y_query: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Remap labels to consecutive integers starting from 0.
    
    Args:
        y_support: Support labels tensor
        y_query: Query labels tensor
        
    Returns:
        Tuple of (remapped_support_labels, remapped_query_labels)
    """
    classes = torch.unique(y_support)
    mapping = {c.item(): i for i, c in enumerate(classes)}
    ys = torch.tensor([mapping[int(c.item())] for c in y_support], device=y_support.device)
    yq = torch.tensor([mapping[int(c.item())] for c in y_query], device=y_query.device)
    return ys.long(), yq.long()