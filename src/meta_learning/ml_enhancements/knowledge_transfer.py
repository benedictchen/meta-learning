"""
Cross-Task Knowledge Transfer for Meta-Learning.

This module provides capabilities for transferring knowledge between
different meta-learning tasks based on task similarity and past performance.

Classes:
    CrossTaskKnowledgeTransfer: Manages task embeddings, similarity matching,
                               and knowledge transfer between tasks.

The system computes task embeddings, finds similar historical tasks,
and transfers successful hyperparameters and configurations.
"""

from typing import Dict, List, Tuple, Any
import time
import torch
import numpy as np

from ..core.episode import Episode


class CrossTaskKnowledgeTransfer:
    """Cross-task knowledge transfer and meta-optimization.
    
    Maintains a knowledge base of task embeddings and successful configurations
    to enable transfer learning between similar meta-learning tasks.
    
    Attributes:
        task_embeddings (Dict): Stored task embeddings for similarity matching
        knowledge_base (Dict): Stored configurations and results from past tasks
        transfer_history (List): History of knowledge transfer operations
    """
    
    def __init__(self):
        """Initialize the knowledge transfer system."""
        self.task_embeddings = {}
        self.knowledge_base = {}
        self.transfer_history = []
        
    def compute_task_embedding(self, episode: Episode) -> np.ndarray:
        """Compute task embedding for similarity matching.
        
        Creates a fixed-length embedding vector that captures task characteristics
        including support set size, class distribution, and feature statistics.
        
        Args:
            episode: The meta-learning episode to embed
            
        Returns:
            23-dimensional embedding vector containing:
            - Task statistics (3 dims): n_support, n_classes, class_balance
            - Feature means (10 dims): Mean feature values
            - Feature stds (10 dims): Feature standard deviations
        """
        support_x, support_y = episode.support_x, episode.support_y
        
        # Basic task statistics
        n_support = len(support_y)
        n_classes = len(torch.unique(support_y))
        class_balance = (torch.bincount(support_y).min().float() / torch.bincount(support_y).max().float()).item()
        
        # Feature statistics
        support_flat = support_x.view(support_x.size(0), -1)
        mean_features = support_flat.mean(dim=0).cpu().numpy()[:10]  # Take first 10 dims
        std_features = support_flat.std(dim=0).cpu().numpy()[:10]
        
        # Pad if necessary
        if len(mean_features) < 10:
            mean_features = np.pad(mean_features, (0, 10 - len(mean_features)), 'constant')
        if len(std_features) < 10:
            std_features = np.pad(std_features, (0, 10 - len(std_features)), 'constant')
        
        embedding = np.concatenate([
            [n_support, n_classes, class_balance],
            mean_features,
            std_features
        ])
        
        return embedding
    
    def find_similar_tasks(self, episode: Episode, top_k: int = 5) -> List[Tuple[str, float]]:
        """Find similar tasks for knowledge transfer.
        
        Uses cosine similarity between task embeddings to identify
        the most similar historical tasks.
        
        Args:
            episode: Current episode to find similarities for
            top_k: Number of top similar tasks to return
            
        Returns:
            List of (task_id, similarity_score) tuples, sorted by similarity
        """
        current_embedding = self.compute_task_embedding(episode)
        
        similarities = []
        for task_id, stored_embedding in self.task_embeddings.items():
            if np.linalg.norm(current_embedding) > 0 and np.linalg.norm(stored_embedding) > 0:
                sim = np.dot(current_embedding, stored_embedding) / (
                    np.linalg.norm(current_embedding) * np.linalg.norm(stored_embedding)
                )
                similarities.append((task_id, sim))
        
        # Sort by similarity and return top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def transfer_knowledge(self, episode: Episode, algorithm_config: Dict[str, Any]) -> Dict[str, Any]:
        """Transfer knowledge from similar tasks.
        
        Finds the most similar historical task and transfers its successful
        hyperparameters if the similarity is high enough and the task was successful.
        
        Args:
            episode: Current episode to apply transfer to
            algorithm_config: Base algorithm configuration
            
        Returns:
            Modified algorithm configuration with transferred knowledge,
            or original config if no suitable transfer found
        """
        similar_tasks = self.find_similar_tasks(episode)
        
        if not similar_tasks:
            return algorithm_config  # No similar tasks found
        
        # Transfer hyperparameters from most similar task
        best_task_id, similarity = similar_tasks[0]
        
        if similarity > 0.8 and best_task_id in self.knowledge_base:  # High similarity threshold
            stored_knowledge = self.knowledge_base[best_task_id]
            
            # Transfer successful hyperparameters
            if stored_knowledge.get('accuracy', 0) > 0.7:  # Only transfer from successful tasks
                transferred_config = algorithm_config.copy()
                
                # Transfer learning rate with decay based on similarity
                if 'learning_rate' in stored_knowledge:
                    original_lr = stored_knowledge['learning_rate']
                    transferred_config['learning_rate'] = original_lr * similarity
                
                # Transfer inner steps
                if 'inner_steps' in stored_knowledge:
                    transferred_config['inner_steps'] = stored_knowledge['inner_steps']
                
                self.transfer_history.append({
                    'from_task': best_task_id,
                    'similarity': similarity,
                    'transferred_params': list(transferred_config.keys())
                })
                
                return transferred_config
        
        return algorithm_config
    
    def store_task_knowledge(self, episode: Episode, task_id: str, result: Dict[str, Any]):
        """Store successful task knowledge for future transfer.
        
        Saves the task embedding and performance results for later use
        in knowledge transfer to similar tasks.
        
        Args:
            episode: The completed episode
            task_id: Unique identifier for this task
            result: Dictionary with results including accuracy and hyperparameters
        """
        embedding = self.compute_task_embedding(episode)
        self.task_embeddings[task_id] = embedding
        
        self.knowledge_base[task_id] = {
            'accuracy': result.get('accuracy', 0.0),
            'learning_rate': result.get('learning_rate', 0.01),
            'inner_steps': result.get('inner_steps', 1),
            'timestamp': time.time()
        }
        
        # Cleanup old entries
        if len(self.knowledge_base) > 500:
            # Remove oldest entries
            sorted_tasks = sorted(self.knowledge_base.items(), key=lambda x: x[1]['timestamp'])
            tasks_to_remove = sorted_tasks[:100]  # Remove oldest 100
            for task_id, _ in tasks_to_remove:
                if task_id in self.task_embeddings:
                    del self.task_embeddings[task_id]
                del self.knowledge_base[task_id]
    
    def get_transfer_stats(self) -> Dict[str, Any]:
        """Get statistics about knowledge transfer operations.
        
        Returns:
            Dictionary with transfer statistics including count,
            average similarity, and success rate
        """
        if not self.transfer_history:
            return {'total_transfers': 0, 'avg_similarity': 0.0}
            
        similarities = [t['similarity'] for t in self.transfer_history]
        return {
            'total_transfers': len(self.transfer_history),
            'avg_similarity': np.mean(similarities),
            'max_similarity': np.max(similarities),
            'min_similarity': np.min(similarities),
            'knowledge_base_size': len(self.knowledge_base)
        }