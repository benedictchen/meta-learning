
# Import existing implementations
from .datasets import MiniImageNetDataset
from .iterators import AdaptiveBatchSampler, MemoryAwareIterator, BalancedTaskGenerator, EpisodeIterator
from ..shared.types import Episode

# Import utility functions  
try:
    from .utils import partition_task_enhanced, download_with_progress, validate_episode
except ImportError:
    # Fallback implementations if files don't exist yet
    def partition_task_enhanced(*args, **kwargs):
        """Enhanced task partitioning (fallback implementation)."""
        from ..data.utils import partition_task
        return partition_task(*args, **kwargs)
    
    def download_with_progress(url: str, path: str) -> bool:
        """Download with progress bar (fallback implementation)."""
        import requests
        try:
            response = requests.get(url)
            with open(path, 'wb') as f:
                f.write(response.content)
            return True
        except Exception:
            return False
    
    def validate_episode(episode) -> bool:
        """Validate episode structure (fallback implementation)."""
        return hasattr(episode, 'support_x') and hasattr(episode, 'query_x')

# Import acceleration components
try:
    from .acceleration import OnDeviceDataset, InfiniteIterator
except ImportError:
    # Fallback implementations
    class OnDeviceDataset:
        """On-device dataset acceleration (fallback implementation)."""
        def __init__(self, dataset, device='cpu'):
            self.dataset = dataset
            self.device = device
        
        def __getitem__(self, idx):
            return self.dataset[idx]
        
        def __len__(self):
            return len(self.dataset)
    
    class InfiniteIterator:
        """Infinite iterator (fallback implementation)."""
        def __init__(self, dataloader):
            self.dataloader = dataloader
            self.iterator = iter(self.dataloader)
        
        def __iter__(self):
            return self
        
        def __next__(self):
            try:
                return next(self.iterator)
            except StopIteration:
                self.iterator = iter(self.dataloader)
                return next(self.iterator)


# Additional utility functions
def create_episode_from_data(support_data, support_labels, query_data, query_labels):
    """Create episode from raw data tensors."""
    return Episode(
        support_x=support_data,
        support_y=support_labels, 
        query_x=query_data,
        query_y=query_labels
    )

def compute_episode_statistics(episode):
    """Compute statistics for an episode."""
    import torch
    
    stats = {}
    
    # Support set statistics
    stats['n_support'] = len(episode.support_x)
    stats['support_classes'] = torch.unique(episode.support_y).tolist()
    stats['n_support_classes'] = len(stats['support_classes'])
    
    # Query set statistics  
    stats['n_query'] = len(episode.query_x)
    stats['query_classes'] = torch.unique(episode.query_y).tolist()
    stats['n_query_classes'] = len(stats['query_classes'])
    
    # Data shape information
    stats['data_shape'] = list(episode.support_x.shape[1:])
    stats['support_shape'] = list(episode.support_x.shape)
    stats['query_shape'] = list(episode.query_x.shape)
    
    return stats

def split_episode(episode, query_ratio=0.5):
    """Split an episode into smaller episodes."""
    import torch
    
    n_query_new = max(1, int(len(episode.query_x) * query_ratio))
    
    # Random split of query set
    indices = torch.randperm(len(episode.query_x))
    query1_indices = indices[:n_query_new]
    query2_indices = indices[n_query_new:]
    
    episode1 = create_episode_from_data(
        episode.support_x,
        episode.support_y,
        episode.query_x[query1_indices],
        episode.query_y[query1_indices]
    )
    
    episode2 = create_episode_from_data(
        episode.support_x,
        episode.support_y, 
        episode.query_x[query2_indices],
        episode.query_y[query2_indices]
    )
    
    return episode1, episode2

def merge_episodes(*episodes):
    """Combine multiple episodes for batch processing."""
    import torch
    
    if not episodes:
        raise ValueError("At least one episode must be provided")
    
    # Collect all support and query data
    all_support_x = []
    all_support_y = []
    all_query_x = []
    all_query_y = []
    
    for episode in episodes:
        all_support_x.append(episode.support_x)
        all_support_y.append(episode.support_y)
        all_query_x.append(episode.query_x)
        all_query_y.append(episode.query_y)
    
    # Concatenate all tensors
    merged_episode = create_episode_from_data(
        torch.cat(all_support_x, dim=0),
        torch.cat(all_support_y, dim=0),
        torch.cat(all_query_x, dim=0),
        torch.cat(all_query_y, dim=0)
    )
    
    return merged_episode

def balance_episode(episode, target_shots_per_class=None):
    """Ensure class balance within episodes."""
    import torch
    
    # Get class information
    unique_classes = torch.unique(episode.support_y)
    n_classes = len(unique_classes)
    
    if target_shots_per_class is None:
        # Find minimum shots per class
        shots_per_class = []
        for cls in unique_classes:
            shots_per_class.append((episode.support_y == cls).sum().item())
        target_shots_per_class = min(shots_per_class)
    
    # Sample balanced support set
    balanced_support_x = []
    balanced_support_y = []
    
    for cls in unique_classes:
        cls_mask = episode.support_y == cls
        cls_indices = torch.where(cls_mask)[0]
        
        if len(cls_indices) < target_shots_per_class:
            # Oversample if needed
            selected_indices = torch.randint(
                0, len(cls_indices), 
                (target_shots_per_class,)
            )
            selected_indices = cls_indices[selected_indices]
        else:
            # Subsample to target
            selected_indices = cls_indices[torch.randperm(len(cls_indices))[:target_shots_per_class]]
        
        balanced_support_x.append(episode.support_x[selected_indices])
        balanced_support_y.append(episode.support_y[selected_indices])
    
    return create_episode_from_data(
        torch.cat(balanced_support_x, dim=0),
        torch.cat(balanced_support_y, dim=0),
        episode.query_x,
        episode.query_y
    )

def augment_episode(episode, augmentation_fn=None):
    """Apply data augmentation to episodes."""
    import torch
    
    if augmentation_fn is None:
        # Default: simple noise augmentation
        def default_augmentation(x):
            noise = torch.randn_like(x) * 0.01
            return x + noise
        augmentation_fn = default_augmentation
    
    # Apply augmentation to support set only (preserve query for evaluation)
    augmented_support_x = augmentation_fn(episode.support_x)
    
    return create_episode_from_data(
        augmented_support_x,
        episode.support_y,
        episode.query_x,
        episode.query_y
    )

# Export everything
__all__ = [
    # Datasets
    "MiniImageNetDataset",
    
    # Iterators
    "EpisodeIterator",
    "AdaptiveBatchSampler", 
    "MemoryAwareIterator",
    "BalancedTaskGenerator",
    
    # Acceleration
    "OnDeviceDataset",
    "InfiniteIterator",
    
    # Utilities
    "partition_task_enhanced",
    "download_with_progress",
    "validate_episode",
    "create_episode_from_data",
    "compute_episode_statistics", 
    "split_episode",
    "merge_episodes",
    "balance_episode",
    "augment_episode",
]