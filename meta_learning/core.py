"""
Core meta-learning algorithms with research-accurate implementations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


class ProtoHead(nn.Module):
    """
    Prototypical Networks head for few-shot classification.
    
    Based on "Prototypical Networks for Few-shot Learning" (Snell et al., 2017).
    Computes class prototypes from support set and classifies queries via distance.
    """
    
    def __init__(self, feature_extractor: nn.Module, distance_metric: str = "euclidean"):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.distance_metric = distance_metric
        
    def forward(self, support_x: torch.Tensor, support_y: torch.Tensor, 
                query_x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        Forward pass for prototypical classification.
        
        Args:
            support_x: Support images [N_support, C, H, W]
            support_y: Support labels [N_support]
            query_x: Query images [N_query, C, H, W]
            temperature: Temperature scaling for logits
            
        Returns:
            logits: Classification logits [N_query, N_way]
        """
        # Extract features
        support_features = self.feature_extractor(support_x)  # [N_support, D]
        query_features = self.feature_extractor(query_x)      # [N_query, D]
        
        # Compute prototypes (class centroids)
        n_way = len(torch.unique(support_y))
        prototypes = []
        
        for class_id in range(n_way):
            class_mask = (support_y == class_id)
            class_features = support_features[class_mask]  # [N_shot, D]
            prototype = class_features.mean(dim=0)         # [D]
            prototypes.append(prototype)
            
        prototypes = torch.stack(prototypes)  # [N_way, D]
        
        # Compute distances
        if self.distance_metric == "euclidean":
            # Euclidean distance: ||q - p||^2
            distances = torch.cdist(query_features, prototypes, p=2.0)  # [N_query, N_way]
        elif self.distance_metric == "squared_euclidean":
            distances = torch.cdist(query_features, prototypes, p=2.0) ** 2
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
        
        # Convert distances to logits (negative distances)
        logits = -distances / temperature
        return logits


class Conv4(nn.Module):
    """
    4-layer convolutional backbone commonly used in few-shot learning.
    
    Standard architecture from meta-learning literature with proper
    initialization and batch normalization.
    """
    
    def __init__(self, input_channels: int = 3, hidden_dim: int = 64):
        super().__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(input_channels, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 2  
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 3
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 4
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        
        self.flatten = nn.Flatten()
        
        # Initialize weights
        self._initialize_weights()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from input images."""
        x = self.features(x)
        x = self.flatten(x)
        return x
        
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class PrototypicalConfig:
    """Configuration class for Prototypical Networks."""
    
    def __init__(self, distance_metric: str = "euclidean", 
                 temperature: float = 1.0, 
                 use_cosine_classifier: bool = False):
        self.distance_metric = distance_metric
        self.temperature = temperature
        self.use_cosine_classifier = use_cosine_classifier
        

class PrototypicalNetworks(nn.Module):
    """
    Complete Prototypical Networks implementation with configuration.
    """
    
    def __init__(self, feature_extractor: nn.Module, config: PrototypicalConfig):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.config = config
        self.proto_head = ProtoHead(feature_extractor, config.distance_metric)
        
    def forward(self, support_x: torch.Tensor, support_y: torch.Tensor,
                query_x: torch.Tensor) -> torch.Tensor:
        """Forward pass through prototypical network."""
        return self.proto_head(support_x, support_y, query_x, self.config.temperature)


def make_episode(dataset, n_way: int, k_shot: int, n_query: int, 
                 n_episodes: int = 1) -> Union[Dict, List[Dict]]:
    """
    Create few-shot learning episodes from a dataset.
    
    Args:
        dataset: Dataset with __getitem__ and labels
        n_way: Number of classes per episode
        k_shot: Number of support examples per class  
        n_query: Number of query examples per class
        n_episodes: Number of episodes to generate
        
    Returns:
        Episode dict or list of episode dicts with support/query splits
    """
    episodes = []
    
    # Get all unique classes from dataset
    if hasattr(dataset, 'labels'):
        all_labels = dataset.labels
    else:
        # Extract labels by iterating through dataset
        all_labels = []
        for i in range(len(dataset)):
            _, label = dataset[i]
            all_labels.append(label)
        all_labels = np.array(all_labels)
    
    unique_classes = np.unique(all_labels)
    
    for episode_idx in range(n_episodes):
        # Sample n_way classes
        episode_classes = np.random.choice(unique_classes, n_way, replace=False)
        
        support_x, support_y = [], []
        query_x, query_y = [], []
        
        for class_idx, class_label in enumerate(episode_classes):
            # Get all indices for this class
            class_indices = np.where(all_labels == class_label)[0]
            
            # Sample k_shot + n_query examples
            selected_indices = np.random.choice(
                class_indices, k_shot + n_query, replace=False
            )
            
            # Split into support and query
            support_indices = selected_indices[:k_shot]
            query_indices = selected_indices[k_shot:]
            
            # Collect support examples
            for idx in support_indices:
                x, _ = dataset[idx]
                support_x.append(x)
                support_y.append(class_idx)  # Relabel to 0, 1, ..., n_way-1
                
            # Collect query examples  
            for idx in query_indices:
                x, _ = dataset[idx]
                query_x.append(x)
                query_y.append(class_idx)
        
        # Convert to tensors
        if isinstance(support_x[0], torch.Tensor):
            support_x = torch.stack(support_x)
            query_x = torch.stack(query_x)
        else:
            support_x = torch.tensor(np.array(support_x))
            query_x = torch.tensor(np.array(query_x))
            
        support_y = torch.tensor(support_y)
        query_y = torch.tensor(query_y)
        
        episode = {
            'support_x': support_x,
            'support_y': support_y, 
            'query_x': query_x,
            'query_y': query_y,
            'n_way': n_way,
            'k_shot': k_shot,
            'n_query': n_query
        }
        
        episodes.append(episode)
    
    return episodes[0] if n_episodes == 1 else episodes


def get_dataset(name: str, split: str = "train"):
    """
    Load a few-shot learning dataset by name.
    
    Args:
        name: Dataset name ("omniglot", "miniimagenet", etc.)
        split: Data split ("train", "val", "test")
        
    Returns:
        Dataset object with __getitem__ and labels
    """
    if name.lower() == "omniglot":
        return SyntheticOmniglot(split=split)
    else:
        raise ValueError(f"Dataset {name} not implemented yet. Use 'omniglot'.")


class SyntheticOmniglot:
    """
    Synthetic Omniglot-like dataset for testing and demos.
    
    Generates random 28x28 binary images with consistent class structure
    suitable for few-shot learning evaluation.
    """
    
    def __init__(self, split: str = "train", n_classes: int = 100, 
                 examples_per_class: int = 20):
        self.split = split
        self.n_classes = n_classes
        self.examples_per_class = examples_per_class
        
        # Generate synthetic data
        np.random.seed(42 + hash(split) % 1000)  # Consistent per split
        
        # Create class templates (characteristic patterns)
        self.class_templates = []
        for _ in range(n_classes):
            template = np.random.rand(28, 28) > 0.7  # Sparse binary pattern
            self.class_templates.append(template)
        
        # Generate examples with variation
        self.data = []
        self.labels = []
        
        for class_idx in range(n_classes):
            template = self.class_templates[class_idx]
            
            for example_idx in range(examples_per_class):
                # Add noise and variation to template
                noise = np.random.rand(28, 28) > 0.9
                example = template.copy()
                example = example ^ noise  # XOR noise
                
                # Convert to tensor format [C, H, W]
                example_tensor = torch.FloatTensor(example).unsqueeze(0)
                
                self.data.append(example_tensor)
                self.labels.append(class_idx)
        
        self.labels = np.array(self.labels)
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]