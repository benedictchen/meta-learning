"""
💰 SUPPORT THIS RESEARCH - PLEASE DONATE! 💰

🙏 If this library helps your research or project, please consider donating:
💳 PayPal: https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS
⭐ GitHub Sponsors: https://github.com/sponsors/benedictchen

👨‍💻 Author: Benedict Chen
💰 Donations: Help support this work! Buy me a coffee ☕, beer 🍺, lamborghini 🏎️, or private island 🏝️
💖 Please consider recurring donations to fully support continued research

Your support enables cutting-edge AI research for everyone! 🚀

Meta-Learning v2 - Clean Architecture with Preserved Breakthroughs
=================================================================

This is the v2 rewrite that preserves ALL essential research functionality
while eliminating bloat and maintaining clean architecture.

RECOVERED BREAKTHROUGH ALGORITHMS:
- TestTimeComputeScaler (2024 world-first implementation)
- Research-accurate MAML with all variants (MAML, FOMAML, ANIL, BOIL, Reptile)  
- BatchNorm research patches for few-shot learning accuracy
- Proper episodic evaluation harness with 95% CI
- Determinism hooks for reproducible research

Author: Benedict Chen (benedict@benedictchen.com)
License: Custom Non-Commercial License with Donation Requirements

💰 SUPPORT THIS RESEARCH - PLEASE DONATE! 💰
"""

from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn

# Import recovered breakthrough algorithms
from .algorithms.ttc_scaler import TestTimeComputeScaler
from .algorithms.ttc_config import TestTimeComputeConfig
from .algorithms.maml_research_accurate import (
    ResearchMAML, MAMLConfig, MAMLVariant, FunctionalModule
)
from .research_patches.batch_norm_policy import EpisodicBatchNormPolicy
from .research_patches.determinism_hooks import DeterminismManager, seed_everything
from .evaluation.few_shot_evaluation_harness import FewShotEvaluationHarness
from .core.episode import Episode, remap_labels

class MetaLearningV2:
    """
    Clean v2 interface preserving all breakthrough research functionality.
    
    Combines the essential algorithms recovered from the pre-v3 implementation
    with a streamlined architecture that eliminates bloat.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize MetaLearning v2 with recovered algorithms."""
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
        """Create Test-Time Compute Scaler (2024 breakthrough)."""
        if config is None:
            config = TestTimeComputeConfig()
        
        self.test_time_scaler = TestTimeComputeScaler(base_model, config)
        return self.test_time_scaler
    
    def create_research_maml(
        self, 
        model: nn.Module,
        config: Optional[MAMLConfig] = None
    ) -> ResearchMAML:
        """Create research-accurate MAML implementation."""
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
        seed_everything(seed)
        self.determinism_manager.configure_deterministic()
    
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
        # Use functional parameter updates to avoid in-place mutations
        support_loss = F.cross_entropy(
            self.maml_learner(episode.support_x), 
            episode.support_y
        )
        
        # Compute inner loop adaptation
        adapted_params = self.maml_learner.inner_update(support_loss)
        
        # Evaluate on query set with adapted parameters
        query_logits = FunctionalModule.functional_forward(
            self.maml_learner.model, 
            episode.query_x, 
            adapted_params
        )
        query_loss = F.cross_entropy(query_logits, episode.query_y)
        
        return {
            "query_loss": query_loss.item(),
            "query_accuracy": (query_logits.argmax(-1) == episode.query_y).float().mean().item(),
            "support_loss": support_loss.item()
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
def create_v2_meta_learner(
    model: nn.Module,
    algorithm: str = "maml",
    seed: int = 42,
    **kwargs
) -> MetaLearningV2:
    """
    Quick setup for v2 meta-learner with breakthrough algorithms.
    
    Args:
        model: Base neural network model
        algorithm: "maml" or "test_time_compute"  
        seed: Random seed for reproducible research
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured MetaLearningV2 instance
    """
    meta_learner = MetaLearningV2()
    
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
    Quick evaluation using recovered evaluation harness.
    
    Provides proper 95% confidence intervals and research-accurate metrics.
    """
    meta_learner = create_v2_meta_learner(model, algorithm, **kwargs)
    harness = meta_learner.create_evaluation_harness()
    
    return harness.evaluate_on_episodes(episodes, meta_learner.train_episode)


# Export key recovered functionality
__all__ = [
    "MetaLearningV2",
    "create_v2_meta_learner", 
    "quick_evaluation",
    "Episode",
    "remap_labels",
    "TestTimeComputeScaler",
    "ResearchMAML",
    "MAMLVariant",
    "EpisodicBatchNormPolicy",
    "FewShotEvaluationHarness"
]