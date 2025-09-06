"""
Ridge Regression Integration Helpers
====================================

Integration utilities for ridge regression with Episode data structures
and meta-learning frameworks.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from typing import Optional, Dict, Any, Tuple, Union

from .ridge_core import RidgeRegression, ridge_regression_solve
from ...core.episode import Episode


class RidgeEpisodeClassifier(nn.Module):
    """
    Ridge regression classifier designed for Episode-based few-shot learning.
    
    Provides a clean interface for training and predicting on Episodes
    with automatic label conversion and performance tracking.
    """
    
    def __init__(
        self,
        reg_lambda: float = 0.01,
        use_woodbury: Optional[bool] = None,
        preprocessing: str = 'standardize',
        lambda_selection: str = 'cv'
    ):
        """
        Initialize Episode-based ridge classifier.
        
        Args:
            reg_lambda: Regularization parameter
            use_woodbury: Whether to use Woodbury formula (auto if None)
            preprocessing: Feature preprocessing method
            lambda_selection: Lambda selection method ('fixed', 'cv', 'gcv')
        """
        super().__init__()
        self.ridge_model = RidgeRegression(
            reg_lambda=reg_lambda,
            use_woodbury=use_woodbury,
            bias=True,
            preprocessing=preprocessing,
            lambda_selection=lambda_selection
        )
        self.num_classes = None
        
    def forward(self, episode: Episode) -> torch.Tensor:
        """
        Forward pass using Episode data structure.
        
        Args:
            episode: Episode containing support and query data
            
        Returns:
            Query predictions (logits for classification)
        """
        # Extract support data
        support_embeddings = episode.support_x
        support_labels = episode.support_y
        query_embeddings = episode.query_x
        
        # Determine number of classes
        self.num_classes = len(torch.unique(support_labels))
        
        # Convert labels to one-hot for regression
        targets = F.one_hot(support_labels, num_classes=self.num_classes).float()
        
        # Fit ridge regression
        self.ridge_model.fit(support_embeddings, targets)
        
        # Predict on query set
        query_logits = self.ridge_model.predict(query_embeddings)
        
        return query_logits
    
    def fit_episode(self, episode: Episode) -> Dict[str, Any]:
        """
        Fit the model on episode support set and return fitting statistics.
        
        Args:
            episode: Episode with support data
            
        Returns:
            Dictionary with fitting statistics and performance metrics
        """
        support_embeddings = episode.support_x
        support_labels = episode.support_y
        
        # Determine number of classes
        self.num_classes = len(torch.unique(support_labels))
        
        # Convert to one-hot
        targets = F.one_hot(support_labels, num_classes=self.num_classes).float()
        
        # Fit and get statistics
        fit_info = self.ridge_model.fit(support_embeddings, targets)
        fit_info['num_classes'] = self.num_classes
        
        return fit_info
    
    def predict_episode(self, episode: Episode, return_uncertainty: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Predict on episode query set.
        
        Args:
            episode: Episode with query data
            return_uncertainty: Whether to return uncertainty estimates
            
        Returns:
            Query predictions or (predictions, uncertainties)
        """
        if self.ridge_model.weights is None:
            raise RuntimeError("Model must be fitted before prediction")
            
        return self.ridge_model.predict(
            episode.query_x, 
            return_uncertainty=return_uncertainty
        )


def create_ridge_classifier(
    reg_lambda: float = 0.01,
    preprocessing: str = 'standardize',
    lambda_selection: str = 'cv'
) -> RidgeEpisodeClassifier:
    """
    Create a ridge classifier optimized for few-shot learning episodes.
    
    Args:
        reg_lambda: Regularization parameter
        preprocessing: Feature preprocessing method  
        lambda_selection: Lambda selection method
        
    Returns:
        Configured RidgeEpisodeClassifier
    """
    return RidgeEpisodeClassifier(
        reg_lambda=reg_lambda,
        preprocessing=preprocessing,
        lambda_selection=lambda_selection
    )


def ridge_episode_loss(
    episode: Episode,
    reg_lambda: float = 0.01,
    loss_type: str = 'cross_entropy',
    **kwargs
) -> torch.Tensor:
    """
    Compute ridge regression loss on an episode.
    
    Args:
        episode: Episode with support and query data
        reg_lambda: Regularization parameter
        loss_type: Type of loss ('cross_entropy', 'mse', 'accuracy')
        **kwargs: Additional arguments for ridge regression
        
    Returns:
        Loss tensor
    """
    # Create and fit classifier
    classifier = create_ridge_classifier(reg_lambda=reg_lambda, **kwargs)
    
    # Fit on support set
    classifier.fit_episode(episode)
    
    # Predict on query set
    query_logits = classifier.predict_episode(episode)
    query_targets = episode.query_y
    
    # Compute loss based on type
    if loss_type == 'cross_entropy':
        loss = F.cross_entropy(query_logits, query_targets)
    elif loss_type == 'mse':
        # Convert targets to one-hot for MSE
        num_classes = len(torch.unique(query_targets))
        target_onehot = F.one_hot(query_targets, num_classes=num_classes).float()
        loss = F.mse_loss(query_logits, target_onehot)
    elif loss_type == 'accuracy':
        # Return negative accuracy as "loss" (for minimization)
        pred_classes = torch.argmax(query_logits, dim=1)
        accuracy = (pred_classes == query_targets).float().mean()
        loss = 1.0 - accuracy
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    
    return loss


def ridge_episode_accuracy(
    episode: Episode,
    reg_lambda: float = 0.01,
    **kwargs
) -> float:
    """
    Compute accuracy of ridge classifier on an episode.
    
    Args:
        episode: Episode with support and query data
        reg_lambda: Regularization parameter
        **kwargs: Additional arguments for ridge regression
        
    Returns:
        Accuracy as float
    """
    classifier = create_ridge_classifier(reg_lambda=reg_lambda, **kwargs)
    classifier.fit_episode(episode)
    
    query_logits = classifier.predict_episode(episode)
    query_targets = episode.query_y
    
    pred_classes = torch.argmax(query_logits, dim=1)
    accuracy = (pred_classes == query_targets).float().mean().item()
    
    return accuracy


def ridge_cross_episode_validation(
    episodes: list,
    lambda_candidates: list = None,
    n_folds: int = 5,
    **kwargs
) -> float:
    """
    Cross-validation for regularization parameter using multiple episodes.
    
    Args:
        episodes: List of Episode objects
        lambda_candidates: List of lambda values to test
        n_folds: Number of CV folds
        **kwargs: Additional arguments for ridge regression
        
    Returns:
        Optimal lambda value
    """
    if lambda_candidates is None:
        lambda_candidates = [1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0, 100.0]
    
    n_episodes = len(episodes)
    if n_episodes < n_folds:
        n_folds = max(2, n_episodes // 2)
    
    fold_size = n_episodes // n_folds
    best_lambda = lambda_candidates[0]
    best_accuracy = 0.0
    
    for reg_lambda in lambda_candidates:
        fold_accuracies = []
        
        for fold in range(n_folds):
            # Create validation fold
            start_idx = fold * fold_size
            end_idx = start_idx + fold_size if fold < n_folds - 1 else n_episodes
            
            val_episodes = episodes[start_idx:end_idx]
            train_episodes = episodes[:start_idx] + episodes[end_idx:]
            
            # Compute average accuracy on validation episodes
            val_accs = []
            for episode in val_episodes:
                try:
                    acc = ridge_episode_accuracy(episode, reg_lambda=reg_lambda, **kwargs)
                    val_accs.append(acc)
                except Exception:
                    # Skip episodes that cause numerical issues
                    continue
            
            if val_accs:
                fold_accuracies.append(sum(val_accs) / len(val_accs))
        
        if fold_accuracies:
            avg_accuracy = sum(fold_accuracies) / len(fold_accuracies)
            if avg_accuracy > best_accuracy:
                best_accuracy = avg_accuracy
                best_lambda = reg_lambda
    
    return best_lambda


def analyze_episode_difficulty(
    episode: Episode,
    reg_lambda: float = 0.01
) -> Dict[str, float]:
    """
    Analyze the difficulty of an episode using ridge regression metrics.
    
    Args:
        episode: Episode to analyze
        reg_lambda: Regularization parameter
        
    Returns:
        Dictionary with difficulty metrics
    """
    support_embeddings = episode.support_x
    support_labels = episode.support_y
    
    # Basic statistics
    n_samples, n_features = support_embeddings.shape
    n_classes = len(torch.unique(support_labels))
    
    # Compute class prototypes and separability
    class_means = []
    class_vars = []
    
    for class_id in range(n_classes):
        class_mask = support_labels == class_id
        class_embeddings = support_embeddings[class_mask]
        
        if class_embeddings.shape[0] > 0:
            class_mean = class_embeddings.mean(dim=0)
            class_var = class_embeddings.var(dim=0).mean().item()
            class_means.append(class_mean)
            class_vars.append(class_var)
    
    # Compute inter-class distances
    inter_class_dists = []
    if len(class_means) > 1:
        for i in range(len(class_means)):
            for j in range(i + 1, len(class_means)):
                dist = torch.norm(class_means[i] - class_means[j]).item()
                inter_class_dists.append(dist)
    
    # Fit ridge regression and get condition number
    try:
        classifier = create_ridge_classifier(reg_lambda=reg_lambda)
        fit_info = classifier.fit_episode(episode)
        ridge_loss = ridge_episode_loss(episode, reg_lambda=reg_lambda).item()
        ridge_acc = ridge_episode_accuracy(episode, reg_lambda=reg_lambda)
    except Exception:
        ridge_loss = float('inf')
        ridge_acc = 0.0
        fit_info = {}
    
    return {
        'n_samples': n_samples,
        'n_features': n_features,
        'n_classes': n_classes,
        'samples_per_class': n_samples / n_classes,
        'avg_intra_class_var': sum(class_vars) / len(class_vars) if class_vars else 0.0,
        'avg_inter_class_dist': sum(inter_class_dists) / len(inter_class_dists) if inter_class_dists else 0.0,
        'ridge_loss': ridge_loss,
        'ridge_accuracy': ridge_acc,
        'condition_number': fit_info.get('condition_number', float('inf')),
        'difficulty_score': ridge_loss / (ridge_acc + 1e-8)  # Higher = more difficult
    }