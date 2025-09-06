"""
Shared types and data structures for the meta-learning package.

This module contains common data structures, enums, and type definitions
used across multiple modules to ensure consistency and reduce coupling.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import torch

# Re-export Episode from core for convenience
from ..core.episode import Episode

__all__ = ['Episode', 'DifficultyMetric', 'TaskDifficultyProfile']


class DifficultyMetric(Enum):
    """Types of task difficulty metrics."""
    # Statistical complexity
    FISHER_DISCRIMINANT_RATIO = "fisher_discriminant_ratio"
    VOLUME_RATIO = "volume_ratio" 
    FEATURE_EFFICIENCY = "feature_efficiency"
    
    # Geometrical complexity  
    CLASS_SEPARABILITY = "class_separability"
    NEIGHBORHOOD_SEPARABILITY = "neighborhood_separability"
    BOUNDARY_COMPLEXITY = "boundary_complexity"
    
    # Learning dynamics
    CONVERGENCE_RATE = "convergence_rate"
    GRADIENT_VARIANCE = "gradient_variance"
    LOSS_LANDSCAPE_SMOOTHNESS = "loss_landscape_smoothness"
    
    # Meta-learning specific
    ADAPTATION_DIFFICULTY = "adaptation_difficulty"
    FEW_SHOT_TRANSFERABILITY = "few_shot_transferability"
    GENERALIZATION_GAP = "generalization_gap"


@dataclass
class TaskDifficultyProfile:
    """Complete difficulty profile for a task."""
    task_id: str
    difficulty_scores: Dict[DifficultyMetric, float] = field(default_factory=dict)
    overall_difficulty: float = 0.0
    difficulty_ranking: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Compute overall difficulty score."""
        if self.difficulty_scores:
            # Weighted average of different difficulty aspects
            weights = {
                DifficultyMetric.CLASS_SEPARABILITY: 0.25,
                DifficultyMetric.ADAPTATION_DIFFICULTY: 0.25,
                DifficultyMetric.CONVERGENCE_RATE: 0.2,
                DifficultyMetric.GENERALIZATION_GAP: 0.2,
                DifficultyMetric.FEATURE_EFFICIENCY: 0.1
            }
            
            weighted_sum = 0.0
            total_weight = 0.0
            
            for metric, score in self.difficulty_scores.items():
                weight = weights.get(metric, 0.05)  # Default small weight
                weighted_sum += weight * score
                total_weight += weight
            
            if total_weight > 0:
                self.overall_difficulty = weighted_sum / total_weight