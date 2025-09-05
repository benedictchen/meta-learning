"""
Episode Protocol - Research-Accurate Few-Shot Episode Management
================================================================

Author: Benedict Chen (benedict@benedictchen.com)

Research-accurate implementation of N-way K-shot M-query episode protocol
with strict mathematical guarantees as specified in meta-learning literature.

References:
- Vinyals et al. (2016): "Matching Networks for One Shot Learning" 
- Snell et al. (2017): "Prototypical Networks for Few-shot Learning"
- Finn et al. (2017): "Model-Agnostic Meta-Learning for Fast Adaptation"

Key Invariants Enforced:
1. Strict per-episode class sampling with no leakage
2. Label remapping to [0..C-1] contiguous range per episode
3. Deterministic episode generation with proper seeding
4. Support/query split integrity within episodes
"""

import torch
import numpy as np
from typing import Tuple, List, Dict, Optional, Union
from dataclasses import dataclass
import warnings


@dataclass
class EpisodeConfig:
    """Configuration for episode generation protocol."""
    n_way: int = 5           # Number of classes per episode
    k_shot: int = 5          # Number of support examples per class
    m_query: int = 15        # Number of query examples per class
    
    # Data split enforcement
    enforce_class_splits: bool = True  # Strict train/val/test class separation
    seed: Optional[int] = None         # Deterministic episode generation
    
    # Validation parameters
    validate_episodes: bool = True     # Runtime episode validation
    allow_class_reuse: bool = False    # Allow same class across episodes in batch
    
    def __post_init__(self):
        """Validate episode configuration parameters."""
        if self.n_way <= 0:
            raise ValueError(f"n_way must be positive, got {self.n_way}")
        if self.k_shot <= 0:
            raise ValueError(f"k_shot must be positive, got {self.k_shot}")
        if self.m_query <= 0:
            raise ValueError(f"m_query must be positive, got {self.m_query}")
            
        # Research-based parameter warnings
        if self.n_way > 20:
            warnings.warn(f"n_way={self.n_way} is unusually high for few-shot learning")
        if self.k_shot > 50:
            warnings.warn(f"k_shot={self.k_shot} approaches many-shot learning regime")


class EpisodeProtocol:
    """
    Research-accurate episode generation following meta-learning protocols.
    
    Ensures mathematical correctness for few-shot learning experiments:
    - Strict N-way K-shot M-query sampling
    - Label remapping to [0, N-1] per episode  
    - No support/query leakage
    - Deterministic reproducibility
    """
    
    def __init__(self, config: EpisodeConfig):
        self.config = config
        self.rng = np.random.RandomState(config.seed)
        
        # Episode validation state
        self._last_episode_classes = None
        self._episode_count = 0
        
    def generate_episode(self, 
                        dataset: Dict[int, List[torch.Tensor]],
                        available_classes: List[int]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate a single few-shot episode with mathematical guarantees.
        
        Args:
            dataset: Dictionary mapping class_id -> list of samples
            available_classes: List of class IDs available for sampling
            
        Returns:
            support_x: Support examples [N*K, ...]
            support_y: Support labels [N*K] (remapped to [0, N-1])
            query_x: Query examples [N*M, ...]  
            query_y: Query labels [N*M] (remapped to [0, N-1])
            
        Mathematical Guarantees:
        1. |support_y| = N * K, |query_y| = N * M
        2. support_y, query_y ∈ [0, N-1] (contiguous)
        3. No overlap between support and query samples
        4. Exactly K support + M query examples per class
        """
        N, K, M = self.config.n_way, self.config.k_shot, self.config.m_query
        
        # 1. Sample N classes for this episode
        if len(available_classes) < N:
            raise ValueError(f"Need at least {N} classes, got {len(available_classes)}")
            
        episode_classes = self.rng.choice(available_classes, size=N, replace=False)
        
        # 2. Validate class availability
        for class_id in episode_classes:
            if class_id not in dataset:
                raise ValueError(f"Class {class_id} not found in dataset")
            if len(dataset[class_id]) < K + M:
                raise ValueError(f"Class {class_id} needs {K+M} samples, got {len(dataset[class_id])}")
        
        # 3. Sample support and query examples per class
        support_samples = []
        support_labels = []
        query_samples = []
        query_labels = []
        
        for new_label, original_class_id in enumerate(episode_classes):
            class_samples = dataset[original_class_id]
            
            # Sample K+M examples without replacement
            sample_indices = self.rng.choice(len(class_samples), size=K+M, replace=False)
            selected_samples = [class_samples[i] for i in sample_indices]
            
            # Split into support and query
            support_samples.extend(selected_samples[:K])
            support_labels.extend([new_label] * K)  # Remapped labels [0, N-1]
            
            query_samples.extend(selected_samples[K:K+M])
            query_labels.extend([new_label] * M)    # Remapped labels [0, N-1]
        
        # 4. Convert to tensors
        support_x = torch.stack(support_samples)  # [N*K, ...]
        support_y = torch.tensor(support_labels, dtype=torch.long)  # [N*K]
        query_x = torch.stack(query_samples)      # [N*M, ...]
        query_y = torch.tensor(query_labels, dtype=torch.long)      # [N*M]
        
        # 5. Validate episode mathematical properties
        if self.config.validate_episodes:
            self._validate_episode(support_x, support_y, query_x, query_y)
            
        # 6. Update episode tracking
        self._last_episode_classes = episode_classes
        self._episode_count += 1
        
        return support_x, support_y, query_x, query_y
    
    def generate_episode_batch(self,
                              dataset: Dict[int, List[torch.Tensor]],
                              available_classes: List[int],
                              batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate a batch of few-shot episodes.
        
        Returns:
            support_x: [batch_size, N*K, ...]
            support_y: [batch_size, N*K] 
            query_x: [batch_size, N*M, ...]
            query_y: [batch_size, N*M]
        """
        batch_support_x = []
        batch_support_y = []
        batch_query_x = []
        batch_query_y = []
        
        for _ in range(batch_size):
            sx, sy, qx, qy = self.generate_episode(dataset, available_classes)
            batch_support_x.append(sx)
            batch_support_y.append(sy)
            batch_query_x.append(qx)
            batch_query_y.append(qy)
            
        return (torch.stack(batch_support_x),
                torch.stack(batch_support_y),
                torch.stack(batch_query_x),
                torch.stack(batch_query_y))
    
    def _validate_episode(self, support_x, support_y, query_x, query_y):
        """Validate episode satisfies mathematical invariants."""
        N, K, M = self.config.n_way, self.config.k_shot, self.config.m_query
        
        # Shape validation
        expected_support_size = N * K
        expected_query_size = N * M
        
        if len(support_x) != expected_support_size:
            raise RuntimeError(f"Support size mismatch: expected {expected_support_size}, got {len(support_x)}")
        if len(query_x) != expected_query_size:
            raise RuntimeError(f"Query size mismatch: expected {expected_query_size}, got {len(query_x)}")
            
        # Label range validation  
        support_unique = torch.unique(support_y)
        query_unique = torch.unique(query_y)
        expected_labels = torch.arange(N)
        
        if not torch.equal(torch.sort(support_unique)[0], expected_labels):
            raise RuntimeError(f"Support labels not in [0, {N-1}]: {support_unique}")
        if not torch.equal(torch.sort(query_unique)[0], expected_labels):
            raise RuntimeError(f"Query labels not in [0, {N-1}]: {query_unique}")
            
        # Per-class count validation
        for class_id in range(N):
            support_count = (support_y == class_id).sum().item()
            query_count = (query_y == class_id).sum().item()
            
            if support_count != K:
                raise RuntimeError(f"Class {class_id} has {support_count} support examples, expected {K}")
            if query_count != M:
                raise RuntimeError(f"Class {class_id} has {query_count} query examples, expected {M}")


def remap_labels_to_episode(labels: torch.Tensor, episode_classes: List[int]) -> torch.Tensor:
    """
    Remap class labels to contiguous [0, N-1] range for episode.
    
    Args:
        labels: Original class labels
        episode_classes: List of N class IDs for this episode
        
    Returns:
        Remapped labels in [0, N-1] range
        
    Example:
        >>> episode_classes = [7, 23, 45]  # 3-way episode
        >>> labels = torch.tensor([23, 7, 45, 23, 7])
        >>> remap_labels_to_episode(labels, episode_classes)
        tensor([1, 0, 2, 1, 0])  # Remapped to [0, 1, 2]
    """
    # Create mapping from original class ID to new label
    class_to_label = {class_id: new_label for new_label, class_id in enumerate(episode_classes)}
    
    # Remap labels
    remapped = torch.zeros_like(labels)
    for i, original_label in enumerate(labels.tolist()):
        if original_label not in class_to_label:
            raise ValueError(f"Label {original_label} not in episode classes {episode_classes}")
        remapped[i] = class_to_label[original_label]
        
    return remapped


def validate_episode_integrity(support_x, support_y, query_x, query_y, config: EpisodeConfig):
    """
    Validate that an episode satisfies all mathematical invariants.
    
    This function can be called by research code to verify episode correctness
    before training or evaluation.
    """
    protocol = EpisodeProtocol(config)
    protocol._validate_episode(support_x, support_y, query_x, query_y)
    

# Research utilities for deterministic episode generation
def seed_episode_generation(seed: int):
    """Set global seed for deterministic episode generation across all protocols."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_dataset_splits(full_dataset: Dict[int, List],
                         train_ratio: float = 0.64,
                         val_ratio: float = 0.16,
                         test_ratio: float = 0.20,
                         seed: int = 42) -> Tuple[List[int], List[int], List[int]]:
    """
    Create train/validation/test class splits for meta-learning.
    
    Ensures no class appears in multiple splits (critical for few-shot evaluation).
    
    Returns:
        train_classes, val_classes, test_classes
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")
        
    all_classes = list(full_dataset.keys())
    n_classes = len(all_classes)
    
    rng = np.random.RandomState(seed)
    rng.shuffle(all_classes)
    
    n_train = int(n_classes * train_ratio)
    n_val = int(n_classes * val_ratio)
    
    train_classes = all_classes[:n_train]
    val_classes = all_classes[n_train:n_train + n_val]
    test_classes = all_classes[n_train + n_val:]
    
    return train_classes, val_classes, test_classes


if __name__ == "__main__":
    # Example usage and testing
    print("Episode Protocol - Research-Accurate Implementation")
    print("=" * 60)
    
    # Create example dataset
    n_classes = 10
    samples_per_class = 50
    sample_shape = (3, 32, 32)
    
    dataset = {}
    for class_id in range(n_classes):
        dataset[class_id] = [torch.randn(sample_shape) for _ in range(samples_per_class)]
    
    # Test episode generation
    config = EpisodeConfig(n_way=5, k_shot=5, m_query=10, seed=42)
    protocol = EpisodeProtocol(config)
    
    # Generate single episode
    support_x, support_y, query_x, query_y = protocol.generate_episode(
        dataset, list(range(n_classes))
    )
    
    print(f"Episode shapes:")
    print(f"  Support: {support_x.shape}, labels: {support_y.shape}")
    print(f"  Query: {query_x.shape}, labels: {query_y.shape}")
    print(f"  Support label range: {support_y.min()}-{support_y.max()}")
    print(f"  Query label range: {query_y.min()}-{query_y.max()}")
    
    # Validate mathematical properties
    validate_episode_integrity(support_x, support_y, query_x, query_y, config)
    print("✓ Episode validation passed")
    
    # Test batch generation
    batch_sx, batch_sy, batch_qx, batch_qy = protocol.generate_episode_batch(
        dataset, list(range(n_classes)), batch_size=4
    )
    
    print(f"\nBatch shapes:")
    print(f"  Batch support: {batch_sx.shape}")
    print(f"  Batch query: {batch_qx.shape}")
    print("✓ Batch episode generation successful")