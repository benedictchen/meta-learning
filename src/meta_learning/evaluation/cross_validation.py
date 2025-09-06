#!/usr/bin/env python3
"""
Cross-Validation for Meta-Learning

Implements cross-validation strategies specific to meta-learning scenarios
where tasks are the fundamental unit of data rather than individual examples.

Author: Benedict Chen (benedict@benedictchen.com)
License: Custom Non-Commercial License with Donation Requirements
"""

import numpy as np
import torch
from typing import List, Dict, Iterator, Optional, Tuple, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class CrossValidationResult:
    """Results from cross-validation evaluation."""
    fold_scores: List[float]
    mean_score: float
    std_score: float
    confidence_interval: Tuple[float, float]
    fold_details: Dict[int, Dict[str, Any]]


class MetaLearningCrossValidator:
    """
    Cross-validation for meta-learning experiments.
    
    Handles task-level splitting and evaluation for meta-learning models.
    """
    
    def __init__(
        self,
        n_folds: int = 5,
        shuffle: bool = True,
        random_state: Optional[int] = None,
        stratify: bool = False
    ):
        """
        Initialize cross-validator.
        
        Args:
            n_folds: Number of cross-validation folds
            shuffle: Whether to shuffle tasks before splitting
            random_state: Random seed for reproducibility
            stratify: Whether to stratify tasks by class distribution
        """
        self.n_folds = n_folds
        self.shuffle = shuffle
        self.random_state = random_state
        self.stratify = stratify
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def split_tasks(self, tasks: List[Any]) -> Iterator[Tuple[List[int], List[int]]]:
        """
        Split tasks into train/validation folds.
        
        Args:
            tasks: List of meta-learning tasks
            
        Yields:
            Tuple of (train_indices, val_indices) for each fold
        """
        n_tasks = len(tasks)
        indices = np.arange(n_tasks)
        
        if self.shuffle:
            np.random.shuffle(indices)
        
        fold_sizes = np.full(self.n_folds, n_tasks // self.n_folds, dtype=int)
        fold_sizes[:n_tasks % self.n_folds] += 1
        
        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            val_indices = indices[start:stop]
            train_indices = np.concatenate([indices[:start], indices[stop:]])
            yield train_indices.tolist(), val_indices.tolist()
            current = stop
    
    def cross_validate(
        self,
        model: Any,
        tasks: List[Any],
        eval_fn: callable,
        fit_fn: Optional[callable] = None
    ) -> CrossValidationResult:
        """
        Perform cross-validation on meta-learning model.
        
        Args:
            model: Meta-learning model to evaluate
            tasks: List of meta-learning tasks
            eval_fn: Function to evaluate model performance
            fit_fn: Optional function to fit model on training tasks
            
        Returns:
            CrossValidationResult with detailed metrics
        """
        fold_scores = []
        fold_details = {}
        
        for fold_idx, (train_indices, val_indices) in enumerate(self.split_tasks(tasks)):
            train_tasks = [tasks[i] for i in train_indices]
            val_tasks = [tasks[i] for i in val_indices]
            
            # Fit model on training tasks if fit function provided
            if fit_fn is not None:
                fold_model = fit_fn(model, train_tasks)
            else:
                fold_model = model
            
            # Evaluate on validation tasks
            fold_score = eval_fn(fold_model, val_tasks)
            fold_scores.append(fold_score)
            
            fold_details[fold_idx] = {
                'score': fold_score,
                'n_train_tasks': len(train_tasks),
                'n_val_tasks': len(val_tasks),
                'train_indices': train_indices,
                'val_indices': val_indices
            }
        
        # Calculate statistics
        mean_score = np.mean(fold_scores)
        std_score = np.std(fold_scores, ddof=1) if len(fold_scores) > 1 else 0.0
        
        # Calculate confidence interval (95%)
        if len(fold_scores) > 1:
            from scipy.stats import t
            alpha = 0.05
            dof = len(fold_scores) - 1
            t_val = t.ppf(1 - alpha/2, dof)
            margin = t_val * std_score / np.sqrt(len(fold_scores))
            ci = (mean_score - margin, mean_score + margin)
        else:
            ci = (mean_score, mean_score)
        
        return CrossValidationResult(
            fold_scores=fold_scores,
            mean_score=mean_score,
            std_score=std_score,
            confidence_interval=ci,
            fold_details=fold_details
        )
    
    def stratified_split_tasks(self, tasks: List[Any], labels: List[int]) -> Iterator[Tuple[List[int], List[int]]]:
        """
        Perform stratified splitting of tasks by label distribution.
        
        Args:
            tasks: List of meta-learning tasks
            labels: Task labels for stratification
            
        Yields:
            Tuple of (train_indices, val_indices) for each fold
        """
        from sklearn.model_selection import StratifiedKFold
        
        skf = StratifiedKFold(
            n_splits=self.n_folds,
            shuffle=self.shuffle,
            random_state=self.random_state
        )
        
        for train_indices, val_indices in skf.split(tasks, labels):
            yield train_indices.tolist(), val_indices.tolist()