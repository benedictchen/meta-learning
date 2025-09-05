"""
💰 SUPPORT THIS RESEARCH - PLEASE DONATE! 💰

🙏 If this library helps your research or project, please consider donating:
💳 PayPal: https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS
⭐ GitHub Sponsors: https://github.com/sponsors/benedictchen

👨‍💻 Author: Benedict Chen
💰 Donations: Help support this work! Buy me a coffee ☕, beer 🍺, lamborghini 🏎️, or private island 🏝️
💖 Please consider recurring donations to fully support continued research

Your support enables cutting-edge AI research for everyone! 🚀

Meta-Learning Toolkit - High-Level API
=================================================================

This module provides a high-level, user-friendly API for meta-learning research.
It wraps the complex low-level algorithms into simple, one-liner interfaces.

Main Components:
- MetaLearningToolkit: Main class for algorithm management
- create_meta_learning_toolkit(): Convenience function for quick setup
- quick_evaluation(): Simple evaluation interface

Supported Algorithms:
- MAML (Model-Agnostic Meta-Learning) with research-accurate implementation
- Test-Time Compute Scaling (2024 breakthrough algorithm)
- Deterministic training setup for reproducible research
- BatchNorm policy fixes for few-shot learning
- Comprehensive evaluation harness with 95% confidence intervals

Usage:
    >>> from meta_learning import create_meta_learning_toolkit
    >>> toolkit = create_meta_learning_toolkit(model, algorithm='maml')
    >>> results = toolkit.train_episode(episode)

Author: Benedict Chen (benedict@benedictchen.com)
License: Custom Non-Commercial License with Donation Requirements

💰 SUPPORT THIS RESEARCH - PLEASE DONATE! 💰
"""

from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import recovered breakthrough algorithms - corrected for actual structure  
import sys
import os
# Add the parent directory to path to access algorithms, research_patches, evaluation
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from algorithms.ttc_scaler import TestTimeComputeScaler
from algorithms.ttc_config import TestTimeComputeConfig
from algorithms.maml_research_accurate import (
    ResearchMAML, MAMLConfig, MAMLVariant, FunctionalModule
)
from research_patches.batch_norm_policy import EpisodicBatchNormPolicy
from research_patches.determinism_hooks import DeterminismManager, setup_deterministic_environment
from evaluation.few_shot_evaluation_harness import FewShotEvaluationHarness
from .core.episode import Episode, remap_labels

class MetaLearningToolkit:
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Initialize recovered breakthrough components
        self.test_time_scaler = None
        self.maml_learner = None  
        self.batch_norm_policy = EpisodicBatchNormPolicy()
        self.determinism_manager = DeterminismManager()
        self.evaluation_harness = None
        
    def create_test_time_compute_scaler(
        self, 
        base_model: nn.Module,
        config: Optional[TestTimeComputeConfig] = None
    ) -> TestTimeComputeScaler:
        """Create Test-Time Compute Scaler."""
        if config is None:
            config = TestTimeComputeConfig()
        
        self.test_time_scaler = TestTimeComputeScaler(base_model, config)
        return self.test_time_scaler
    
    def create_research_maml(
        self, 
        model: nn.Module,
        config: Optional[MAMLConfig] = None
    ) -> ResearchMAML:
        """Create MAML implementation."""
        if config is None:
            config = MAMLConfig(
                variant=MAMLVariant.MAML,
                inner_lr=0.01,
                outer_lr=0.001,
                inner_steps=1,
                first_order=False
            )
        
        self.maml_learner = ResearchMAML(model, config)
        return self.maml_learner
    
    def apply_batch_norm_fixes(self, model: nn.Module) -> nn.Module:
        """Apply research-accurate BatchNorm fixes for few-shot learning."""
        return self.batch_norm_policy.apply_to_model(model)
    
    def setup_deterministic_training(self, seed: int = 42) -> None:
        """Setup deterministic training for reproducible research."""
        setup_deterministic_environment(seed)
        self.determinism_manager.enable_full_determinism()
    
    def create_evaluation_harness(self, **kwargs) -> FewShotEvaluationHarness:
        """Create proper episodic evaluation harness with 95% CI."""
        self.evaluation_harness = FewShotEvaluationHarness(**kwargs)
        return self.evaluation_harness
    
    def train_episode(
        self, 
        episode: Episode,
        algorithm: str = "maml"
    ) -> Dict[str, Any]:
        """Train on a single episode using specified algorithm."""
        episode.validate()
        
        if algorithm == "maml" and self.maml_learner is not None:
            return self._train_maml_episode(episode)
        elif algorithm == "test_time_compute" and self.test_time_scaler is not None:
            return self._train_test_time_episode(episode)
        else:
            raise ValueError(f"Algorithm {algorithm} not initialized")
    
    def _train_maml_episode(self, episode: Episode) -> Dict[str, Any]:
        """Train using research-accurate MAML."""
        # Define loss function for MAML
        loss_fn = F.cross_entropy
        
        # Format episode as task batch (single task)
        task_batch = [(episode.support_x, episode.support_y, episode.query_x, episode.query_y)]
        
        # Use MAML forward method with task batch and loss function
        # This computes: inner adaptation on support → query loss on adapted params
        meta_loss = self.maml_learner(task_batch, loss_fn)
        
        # For metrics, compute support and query losses separately
        # NOTE: This re-does the inner loop adaptation (could be optimized to reuse adapted_params)
        support_logits = self.maml_learner.model(episode.support_x)
        support_loss = loss_fn(support_logits, episode.support_y)
        
        # Compute inner loop adaptation for query evaluation
        adapted_params = self.maml_learner.inner_loop(episode.support_x, episode.support_y, loss_fn)
        
        # Evaluate on query set with adapted parameters
        if adapted_params is not None and len(adapted_params) > 0:
            query_logits = FunctionalModule.functional_forward(
                self.maml_learner.model, 
                episode.query_x, 
                adapted_params
            )
        else:
            # Fallback to base model - this should only happen with zero inner steps
            import warnings
            inner_steps = getattr(self.maml_learner.config, 'inner_steps', 'unknown')
            warnings.warn(
                f"MAML falling back to base model (no adapted parameters). "
                f"This should only occur with inner_steps=0. "
                f"Current inner_steps: {inner_steps}. "
                f"Check MAML configuration if this is unexpected.",
                UserWarning,
                stacklevel=3
            )
            query_logits = self.maml_learner.model(episode.query_x)
        
        query_loss = F.cross_entropy(query_logits, episode.query_y)
        
        return {
            "query_loss": query_loss.item(),
            "query_accuracy": (query_logits.argmax(-1) == episode.query_y).float().mean().item(),
            "support_loss": support_loss.item(),
            "meta_loss": meta_loss.item()
        }
    
    def _train_test_time_episode(self, episode: Episode) -> Dict[str, Any]:
        """Train using Test-Time Compute Scaling."""
        predictions, metrics = self.test_time_scaler.scale_compute(
            support_set=episode.support_x,
            support_labels=episode.support_y, 
            query_set=episode.query_x,
            task_context={"n_classes": len(torch.unique(episode.support_y))}
        )
        
        accuracy = (predictions.argmax(-1) == episode.query_y).float().mean().item()
        
        return {
            "query_accuracy": accuracy,
            "compute_scaling_metrics": metrics,
            "predictions": predictions
        }


# Convenience functions for quick setup
def create_meta_learning_toolkit(
    model: nn.Module,
    algorithm: str = "maml",
    seed: int = 42,
    **kwargs
) -> MetaLearningToolkit:
    """
    Quick setup for meta-learning toolkit with algorithms.
    
    Args:
        model: Base neural network model
        algorithm: "maml" or "test_time_compute"  
        seed: Random seed for reproducible research
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured MetaLearningToolkit instance
    """
    meta_learner = MetaLearningToolkit()
    
    # Setup deterministic training
    meta_learner.setup_deterministic_training(seed)
    
    # Apply research-accurate BatchNorm fixes
    model = meta_learner.apply_batch_norm_fixes(model)
    
    # Initialize requested algorithm
    if algorithm == "maml":
        maml_config = MAMLConfig(**kwargs)
        meta_learner.create_research_maml(model, maml_config)
    elif algorithm == "test_time_compute":
        ttc_config = TestTimeComputeConfig(**kwargs)
        meta_learner.create_test_time_compute_scaler(model, ttc_config)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    return meta_learner

def quick_evaluation(
    model: nn.Module,
    episodes: list,
    algorithm: str = "maml",
    **kwargs
) -> Dict[str, Any]:
    """
    Quick evaluation using evaluation harness.
    """
    meta_learner = create_meta_learning_toolkit(model, algorithm, **kwargs)
    harness = meta_learner.create_evaluation_harness()
    
    return harness.evaluate_on_episodes(episodes, meta_learner.train_episode)


# Export key recovered functionality
__all__ = [
    "MetaLearningToolkit",
    "create_meta_learning_toolkit", 
    "quick_evaluation",
    "Episode",
    "remap_labels",
    "TestTimeComputeScaler",
    "ResearchMAML",
    "MAMLVariant",
    "EpisodicBatchNormPolicy",
    "FewShotEvaluationHarness"
]