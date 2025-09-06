"""
Failure Prediction Model for Meta-Learning Algorithms.

This module provides ML-based failure prediction capabilities to anticipate
when meta-learning algorithms are likely to fail on specific episodes.

Classes:
    FailurePredictionModel: Predicts algorithm failure probability using
                           similarity-based learning from historical data.

The model extracts features from episodes and algorithm states to build
a similarity-based predictor that learns from past successes and failures.
"""

from typing import Dict, Any
import torch
import numpy as np

from ..core.episode import Episode


class FailurePredictionModel:
    """ML-based failure prediction for meta-learning algorithms.
    
    Uses historical episode features and algorithm states to predict the 
    probability that an algorithm will fail on a given episode. Employs
    similarity-based learning with exponential weighting.
    
    Attributes:
        feature_history (List[np.ndarray]): Historical feature vectors
        failure_history (List[float]): Historical failure outcomes (0.0 or 1.0)  
        prediction_threshold (float): Threshold for failure prediction
    """
    
    def __init__(self):
        """Initialize the failure prediction model."""
        self.feature_history = []
        self.failure_history = []
        self.prediction_threshold = 0.7
        
    def extract_features(self, episode: Episode, algorithm_state: Dict[str, Any]) -> np.ndarray:
        """Extract features for failure prediction.
        
        Combines task complexity features (support set size, class balance,
        feature diversity) with algorithm state features (learning rate,
        training progress) to create a comprehensive feature vector.
        
        Args:
            episode: The meta-learning episode
            algorithm_state: Current algorithm state and hyperparameters
            
        Returns:
            Feature vector as numpy array with 8 dimensions:
            [n_support, n_classes, class_balance, avg_distance,
             learning_rate, inner_steps, avg_loss, training_progress]
        """
        # Task complexity features
        support_x, support_y = episode.support_x, episode.support_y
        n_support, n_classes = len(support_y), len(torch.unique(support_y))
        class_balance = (torch.bincount(support_y).min().float() / torch.bincount(support_y).max().float()).item()
        
        # Feature diversity
        support_flat = support_x.view(support_x.size(0), -1)
        pairwise_distances = torch.pdist(support_flat)
        avg_distance = pairwise_distances.mean().item() if len(pairwise_distances) > 0 else 0.0
        
        # Algorithm state features
        learning_rate = algorithm_state.get('learning_rate', 0.01)
        inner_steps = algorithm_state.get('inner_steps', 1)
        loss_history = algorithm_state.get('loss_history', [])
        avg_loss = np.mean(loss_history) if loss_history else 0.0
        
        return np.array([
            n_support, n_classes, class_balance, avg_distance,
            learning_rate, inner_steps, avg_loss,
            len(loss_history)  # Training progress indicator
        ])
    
    def predict_failure_risk(self, episode: Episode, algorithm_state: Dict[str, Any]) -> float:
        """Predict probability of algorithm failure.
        
        Uses similarity-based prediction by comparing the current episode
        features with historical episodes and weighting by similarity.
        
        Args:
            episode: The episode to predict failure risk for
            algorithm_state: Current algorithm state
            
        Returns:
            Failure risk probability between 0.0 and 1.0
        """
        features = self.extract_features(episode, algorithm_state)
        
        if len(self.feature_history) < 10:  # Not enough data for prediction
            return 0.5  # Neutral prediction
        
        # Simple similarity-based prediction (in practice would use trained ML model)
        feature_matrix = np.array(self.feature_history)
        failure_array = np.array(self.failure_history)
        
        # Find similar episodes using cosine similarity
        similarities = []
        for hist_features in feature_matrix:
            if np.linalg.norm(features) > 0 and np.linalg.norm(hist_features) > 0:
                sim = np.dot(features, hist_features) / (np.linalg.norm(features) * np.linalg.norm(hist_features))
                similarities.append(sim)
            else:
                similarities.append(0.0)
        
        similarities = np.array(similarities)
        
        # Weight by similarity and predict
        if len(similarities) > 0 and similarities.max() > 0.1:
            weights = np.exp(similarities * 5)  # Exponential weighting
            weights = weights / weights.sum()
            failure_risk = np.sum(weights * failure_array)
            return failure_risk
        
        return 0.5  # Default neutral prediction
    
    def update_with_outcome(self, episode: Episode, algorithm_state: Dict[str, Any], failed: bool):
        """Update model with episode outcome.
        
        Adds the episode features and outcome to the historical data
        for future predictions. Maintains a sliding window of recent data.
        
        Args:
            episode: The completed episode
            algorithm_state: Algorithm state during the episode
            failed: Whether the algorithm failed on this episode
        """
        features = self.extract_features(episode, algorithm_state)
        self.feature_history.append(features)
        self.failure_history.append(1.0 if failed else 0.0)
        
        # Keep only recent history
        if len(self.feature_history) > 1000:
            self.feature_history = self.feature_history[-500:]
            self.failure_history = self.failure_history[-500:]