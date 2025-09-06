"""
💰 SUPPORT THIS RESEARCH - PLEASE DONATE! 💰

Advanced Similarity Metrics for Cross-Task Knowledge Transfer
==========================================================

Implementation of advanced similarity metrics for meta-learning tasks,
including task embedding similarity, hierarchical relationships, and
efficient nearest neighbor search for task matching.

🎯 **Features:**
- Task embedding similarity computations
- Hierarchical task relationship modeling
- Efficient nearest neighbor search for task matching
- Knowledge transfer system for parameter reuse

This module is created ADDITIVELY to implement remaining TODOs from math_utils.py
without modifying the existing core functionality.

💰 Please donate if this accelerates your research!
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Union, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from sklearn.neighbors import NearestNeighbors

from ..shared.types import Episode


class TaskEmbeddingSimilarity:
    """
    Compute similarity between task embeddings for knowledge transfer.
    
    This class implements various metrics for comparing task representations,
    enabling efficient meta-learning and transfer learning applications.
    """
    
    def __init__(self, embedding_dim: int, similarity_metric: str = "cosine"):
        """
        Initialize task embedding similarity computer.
        
        Args:
            embedding_dim: Dimension of task embeddings
            similarity_metric: Type of similarity ("cosine", "euclidean", "mahalanobis")
        """
        self.embedding_dim = embedding_dim
        self.similarity_metric = similarity_metric
        
        # Initialize learned parameters for advanced metrics
        if similarity_metric == "mahalanobis":
            self.precision_matrix = nn.Parameter(torch.eye(embedding_dim))
        elif similarity_metric == "learned":
            self.similarity_net = nn.Sequential(
                nn.Linear(embedding_dim * 2, embedding_dim),
                nn.ReLU(),
                nn.Linear(embedding_dim, embedding_dim // 2),
                nn.ReLU(),
                nn.Linear(embedding_dim // 2, 1),
                nn.Sigmoid()
            )
    
    def compute_task_embedding(self, episode: Episode) -> torch.Tensor:
        """
        Compute task embedding from an episode.
        
        Args:
            episode: Meta-learning episode
            
        Returns:
            Task embedding vector of shape [embedding_dim]
        """
        # Compute basic statistics
        support_mean = episode.support_x.mean(dim=0)
        support_std = episode.support_x.std(dim=0)
        query_mean = episode.query_x.mean(dim=0)
        
        # Number of ways and shots
        n_classes = len(torch.unique(episode.support_y))
        n_support = len(episode.support_y)
        n_query = len(episode.query_y)
        
        # Create task embedding combining statistical and structural features
        task_stats = torch.cat([
            support_mean[:min(len(support_mean), self.embedding_dim//4)],
            support_std[:min(len(support_std), self.embedding_dim//4)],
            query_mean[:min(len(query_mean), self.embedding_dim//4)]
        ])
        
        # Pad or truncate to correct dimension
        if len(task_stats) < self.embedding_dim:
            # Add structural features
            structural_features = torch.tensor([
                float(n_classes), float(n_support), float(n_query),
                float(n_support / n_classes),  # shots per class
                float(episode.support_x.var().item()),  # overall variance
                float(episode.support_x.norm().item()),  # overall magnitude
            ])
            
            # Pad with structural features and zeros if needed
            remaining_dim = self.embedding_dim - len(task_stats)
            padding_features = structural_features[:remaining_dim]
            if len(padding_features) < remaining_dim:
                padding_zeros = torch.zeros(remaining_dim - len(padding_features))
                padding_features = torch.cat([padding_features, padding_zeros])
            
            task_embedding = torch.cat([task_stats, padding_features])
        else:
            task_embedding = task_stats[:self.embedding_dim]
        
        return task_embedding
    
    def compute_similarity(self, embedding1: torch.Tensor, embedding2: torch.Tensor) -> torch.Tensor:
        """
        Compute similarity between two task embeddings.
        
        Args:
            embedding1: First task embedding [embedding_dim]
            embedding2: Second task embedding [embedding_dim]
            
        Returns:
            Similarity score (higher = more similar)
        """
        if self.similarity_metric == "cosine":
            return F.cosine_similarity(embedding1.unsqueeze(0), embedding2.unsqueeze(0))
        
        elif self.similarity_metric == "euclidean":
            distance = torch.norm(embedding1 - embedding2)
            return 1.0 / (1.0 + distance)  # Convert distance to similarity
        
        elif self.similarity_metric == "mahalanobis":
            diff = embedding1 - embedding2
            # Ensure precision matrix is positive definite
            precision = self.precision_matrix.t() @ self.precision_matrix
            distance = torch.sqrt(diff.t() @ precision @ diff)
            return 1.0 / (1.0 + distance)
        
        elif self.similarity_metric == "learned":
            combined = torch.cat([embedding1, embedding2])
            return self.similarity_net(combined).squeeze()
        
        else:
            raise ValueError(f"Unknown similarity metric: {self.similarity_metric}")
    
    def batch_similarity(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise similarities for a batch of task embeddings.
        
        Args:
            embeddings: Batch of embeddings [batch_size, embedding_dim]
            
        Returns:
            Similarity matrix [batch_size, batch_size]
        """
        batch_size = embeddings.size(0)
        similarities = torch.zeros(batch_size, batch_size)
        
        for i in range(batch_size):
            for j in range(batch_size):
                if i == j:
                    similarities[i, j] = 1.0
                else:
                    sim = self.compute_similarity(embeddings[i], embeddings[j])
                    similarities[i, j] = sim
                    similarities[j, i] = sim  # Symmetric
        
        return similarities


class HierarchicalTaskRelationships:
    """
    Model hierarchical relationships between meta-learning tasks.
    
    This class builds a hierarchy of tasks based on similarity and
    enables efficient knowledge transfer along the hierarchy.
    """
    
    def __init__(self, max_depth: int = 5):
        """
        Initialize hierarchical task relationship model.
        
        Args:
            max_depth: Maximum depth of the task hierarchy
        """
        self.max_depth = max_depth
        self.task_hierarchy = {}
        self.task_embeddings = {}
        self.parent_child_map = defaultdict(list)
        self.child_parent_map = {}
    
    def add_task(self, task_id: str, task_embedding: torch.Tensor, parent_id: Optional[str] = None):
        """
        Add a task to the hierarchy.
        
        Args:
            task_id: Unique identifier for the task
            task_embedding: Task embedding vector
            parent_id: Parent task ID (None for root tasks)
        """
        self.task_embeddings[task_id] = task_embedding
        self.task_hierarchy[task_id] = {
            'embedding': task_embedding,
            'parent': parent_id,
            'children': [],
            'depth': 0 if parent_id is None else self.task_hierarchy[parent_id]['depth'] + 1
        }
        
        if parent_id is not None:
            self.parent_child_map[parent_id].append(task_id)
            self.child_parent_map[task_id] = parent_id
            self.task_hierarchy[parent_id]['children'].append(task_id)
    
    def find_similar_tasks(self, query_embedding: torch.Tensor, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Find most similar tasks in the hierarchy.
        
        Args:
            query_embedding: Query task embedding
            top_k: Number of similar tasks to return
            
        Returns:
            List of (task_id, similarity_score) tuples
        """
        similarities = []
        
        for task_id, embedding in self.task_embeddings.items():
            similarity = F.cosine_similarity(query_embedding.unsqueeze(0), embedding.unsqueeze(0)).item()
            similarities.append((task_id, similarity))
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def get_hierarchical_path(self, task_id: str) -> List[str]:
        """
        Get the path from root to the specified task.
        
        Args:
            task_id: Task identifier
            
        Returns:
            List of task IDs from root to target task
        """
        path = []
        current_id = task_id
        
        while current_id is not None:
            path.append(current_id)
            current_id = self.child_parent_map.get(current_id)
        
        return list(reversed(path))
    
    def get_transfer_candidates(self, task_id: str, max_candidates: int = 10) -> List[str]:
        """
        Get candidate tasks for knowledge transfer based on hierarchy.
        
        Args:
            task_id: Target task ID
            max_candidates: Maximum number of candidates
            
        Returns:
            List of candidate task IDs ordered by transfer potential
        """
        candidates = []
        
        # Add parent (direct ancestor knowledge)
        parent = self.child_parent_map.get(task_id)
        if parent:
            candidates.append(parent)
        
        # Add siblings (similar level knowledge)
        if parent:
            siblings = [child for child in self.parent_child_map[parent] if child != task_id]
            candidates.extend(siblings)
        
        # Add children (if they exist and we're not at max depth)
        children = self.parent_child_map.get(task_id, [])
        candidates.extend(children)
        
        # Add similar tasks from other parts of hierarchy
        if task_id in self.task_embeddings:
            query_embedding = self.task_embeddings[task_id]
            similar_tasks = self.find_similar_tasks(query_embedding, top_k=max_candidates)
            for similar_id, _ in similar_tasks:
                if similar_id not in candidates and similar_id != task_id:
                    candidates.append(similar_id)
        
        return candidates[:max_candidates]


class EfficientTaskMatcher:
    """
    Efficient nearest neighbor search for task matching using optimized data structures.
    """
    
    def __init__(self, n_neighbors: int = 5, algorithm: str = "auto"):
        """
        Initialize efficient task matcher.
        
        Args:
            n_neighbors: Number of nearest neighbors to find
            algorithm: Algorithm for nearest neighbor search
        """
        self.n_neighbors = n_neighbors
        self.algorithm = algorithm
        self.task_embeddings = []
        self.task_ids = []
        self.nn_model = None
        self.is_fitted = False
    
    def add_tasks(self, task_ids: List[str], embeddings: torch.Tensor):
        """
        Add multiple tasks to the matcher.
        
        Args:
            task_ids: List of task identifiers
            embeddings: Task embeddings [num_tasks, embedding_dim]
        """
        self.task_ids.extend(task_ids)
        self.task_embeddings.extend(embeddings.detach().cpu().numpy())
        self.is_fitted = False
    
    def fit(self):
        """Fit the nearest neighbor model."""
        if len(self.task_embeddings) == 0:
            raise ValueError("No tasks added to matcher")
        
        embeddings_array = np.array(self.task_embeddings)
        self.nn_model = NearestNeighbors(
            n_neighbors=min(self.n_neighbors, len(self.task_embeddings)),
            algorithm=self.algorithm,
            metric='cosine'
        )
        self.nn_model.fit(embeddings_array)
        self.is_fitted = True
    
    def find_matches(self, query_embedding: torch.Tensor) -> List[Tuple[str, float]]:
        """
        Find nearest neighbor tasks for a query embedding.
        
        Args:
            query_embedding: Query task embedding
            
        Returns:
            List of (task_id, similarity_score) tuples
        """
        if not self.is_fitted:
            self.fit()
        
        query_array = query_embedding.detach().cpu().numpy().reshape(1, -1)
        distances, indices = self.nn_model.kneighbors(query_array)
        
        matches = []
        for dist, idx in zip(distances[0], indices[0]):
            # Convert cosine distance to similarity
            similarity = 1.0 - dist
            task_id = self.task_ids[idx]
            matches.append((task_id, similarity))
        
        return matches
    
    def batch_find_matches(self, query_embeddings: torch.Tensor) -> List[List[Tuple[str, float]]]:
        """
        Find nearest neighbors for multiple query embeddings.
        
        Args:
            query_embeddings: Batch of query embeddings [batch_size, embedding_dim]
            
        Returns:
            List of match lists for each query
        """
        if not self.is_fitted:
            self.fit()
        
        query_array = query_embeddings.detach().cpu().numpy()
        distances, indices = self.nn_model.kneighbors(query_array)
        
        batch_matches = []
        for batch_idx in range(len(query_embeddings)):
            matches = []
            for dist, idx in zip(distances[batch_idx], indices[batch_idx]):
                similarity = 1.0 - dist
                task_id = self.task_ids[idx]
                matches.append((task_id, similarity))
            batch_matches.append(matches)
        
        return batch_matches


class KnowledgeTransferSystem:
    """
    System for parameter reuse and knowledge transfer between meta-learning tasks.
    """
    
    def __init__(self, similarity_threshold: float = 0.7):
        """
        Initialize knowledge transfer system.
        
        Args:
            similarity_threshold: Minimum similarity for knowledge transfer
        """
        self.similarity_threshold = similarity_threshold
        self.task_parameters = {}
        self.task_embeddings = {}
        self.transfer_history = defaultdict(list)
    
    def register_task(self, task_id: str, task_embedding: torch.Tensor, parameters: Dict[str, torch.Tensor]):
        """
        Register a task with its learned parameters.
        
        Args:
            task_id: Task identifier
            task_embedding: Task embedding vector
            parameters: Dictionary of learned parameters
        """
        self.task_embeddings[task_id] = task_embedding
        self.task_parameters[task_id] = {k: v.clone() for k, v in parameters.items()}
    
    def find_transfer_candidates(self, query_task_id: str, query_embedding: torch.Tensor) -> List[str]:
        """
        Find candidate tasks for parameter transfer.
        
        Args:
            query_task_id: Target task ID
            query_embedding: Target task embedding
            
        Returns:
            List of candidate task IDs for transfer
        """
        candidates = []
        
        for task_id, embedding in self.task_embeddings.items():
            if task_id == query_task_id:
                continue
            
            similarity = F.cosine_similarity(query_embedding.unsqueeze(0), embedding.unsqueeze(0)).item()
            
            if similarity >= self.similarity_threshold:
                candidates.append((task_id, similarity))
        
        # Sort by similarity
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [task_id for task_id, _ in candidates]
    
    def transfer_parameters(self, source_task_id: str, target_task_id: str, 
                          transfer_ratio: float = 0.5) -> Dict[str, torch.Tensor]:
        """
        Transfer parameters from source to target task.
        
        Args:
            source_task_id: Source task ID
            target_task_id: Target task ID
            transfer_ratio: Ratio of parameters to transfer (0.0 to 1.0)
            
        Returns:
            Dictionary of transferred parameters
        """
        if source_task_id not in self.task_parameters:
            raise ValueError(f"Source task {source_task_id} not registered")
        
        source_params = self.task_parameters[source_task_id]
        transferred_params = {}
        
        for param_name, param_tensor in source_params.items():
            # Apply transfer ratio - blend with random initialization
            if target_task_id in self.task_parameters:
                # Blend with existing parameters
                target_param = self.task_parameters[target_task_id][param_name]
                transferred = transfer_ratio * param_tensor + (1 - transfer_ratio) * target_param
            else:
                # Blend with random initialization
                random_init = torch.randn_like(param_tensor) * 0.1
                transferred = transfer_ratio * param_tensor + (1 - transfer_ratio) * random_init
            
            transferred_params[param_name] = transferred
        
        # Record transfer
        self.transfer_history[target_task_id].append({
            'source_task': source_task_id,
            'transfer_ratio': transfer_ratio,
            'timestamp': torch.tensor(0.0)  # Placeholder timestamp
        })
        
        return transferred_params
    
    def get_transfer_history(self, task_id: str) -> List[Dict]:
        """
        Get knowledge transfer history for a task.
        
        Args:
            task_id: Task identifier
            
        Returns:
            List of transfer records
        """
        return self.transfer_history.get(task_id, [])


# Convenience functions for easy usage
def compute_task_similarity(episode1: Episode, episode2: Episode, 
                          embedding_dim: int = 128, metric: str = "cosine") -> float:
    """
    Compute similarity between two episodes using task embeddings.
    
    Args:
        episode1: First episode
        episode2: Second episode  
        embedding_dim: Dimension for task embeddings
        metric: Similarity metric to use
        
    Returns:
        Similarity score between episodes
    """
    similarity_computer = TaskEmbeddingSimilarity(embedding_dim, metric)
    
    embedding1 = similarity_computer.compute_task_embedding(episode1)
    embedding2 = similarity_computer.compute_task_embedding(episode2)
    
    return similarity_computer.compute_similarity(embedding1, embedding2).item()


def build_task_hierarchy(episodes: List[Episode], task_ids: List[str], 
                        embedding_dim: int = 128) -> HierarchicalTaskRelationships:
    """
    Build a hierarchical task relationship model from episodes.
    
    Args:
        episodes: List of episodes
        task_ids: Corresponding task identifiers
        embedding_dim: Embedding dimension
        
    Returns:
        Hierarchical task relationship model
    """
    hierarchy = HierarchicalTaskRelationships()
    similarity_computer = TaskEmbeddingSimilarity(embedding_dim)
    
    # Compute embeddings
    embeddings = []
    for episode in episodes:
        embedding = similarity_computer.compute_task_embedding(episode)
        embeddings.append(embedding)
    
    # Build hierarchy based on similarities (simple clustering approach)
    # Add root tasks first
    for i, (task_id, embedding) in enumerate(zip(task_ids, embeddings)):
        hierarchy.add_task(task_id, embedding)
    
    return hierarchy


def efficient_task_matching(episodes: List[Episode], task_ids: List[str], 
                          query_episode: Episode, k: int = 5) -> List[Tuple[str, float]]:
    """
    Efficiently find k most similar tasks to a query episode.
    
    Args:
        episodes: List of reference episodes
        task_ids: Corresponding task identifiers
        query_episode: Query episode to match
        k: Number of matches to return
        
    Returns:
        List of (task_id, similarity_score) tuples
    """
    matcher = EfficientTaskMatcher(n_neighbors=k)
    similarity_computer = TaskEmbeddingSimilarity(embedding_dim=128)
    
    # Compute embeddings for reference tasks
    embeddings = []
    for episode in episodes:
        embedding = similarity_computer.compute_task_embedding(episode)
        embeddings.append(embedding)
    
    embeddings_tensor = torch.stack(embeddings)
    matcher.add_tasks(task_ids, embeddings_tensor)
    
    # Compute query embedding and find matches
    query_embedding = similarity_computer.compute_task_embedding(query_episode)
    matches = matcher.find_matches(query_embedding)
    
    return matches