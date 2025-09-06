"""
Main task difficulty assessment coordinator.

This module provides the main TaskDifficultyAssessor class that coordinates
all difficulty analysis components and provides high-level interfaces for
comprehensive task difficulty assessment.

The assessor combines:
- Statistical complexity analysis
- Learning dynamics analysis  
- Meta-learning specific measures

Usage:
    assessor = TaskDifficultyAssessor()
    profile = assessor.assess_episode_difficulty(episode, model)
    print(f"Overall difficulty: {profile.overall_difficulty:.3f}")
"""

from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import logging

from ...shared.types import Episode, DifficultyMetric, TaskDifficultyProfile
from .complexity_analyzer import ComplexityAnalyzer
from .learning_dynamics import LearningDynamicsAnalyzer
from .meta_analyzer import MetaLearningSpecificAnalyzer


class TaskDifficultyAssessor:
    """Main coordinator for task difficulty assessment."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the task difficulty assessor.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.complexity_analyzer = ComplexityAnalyzer()
        self.dynamics_analyzer = LearningDynamicsAnalyzer()
        self.meta_analyzer = MetaLearningSpecificAnalyzer()
        self.logger = logging.getLogger(__name__)
        
    def assess_episode_difficulty(
        self,
        episode: Episode,
        model: Optional[nn.Module] = None,
        include_learning_dynamics: bool = True
    ) -> TaskDifficultyProfile:
        """
        Comprehensive difficulty assessment for a single episode.
        
        Args:
            episode: Episode to assess
            model: Optional model for learning dynamics analysis
            include_learning_dynamics: Whether to include learning-based metrics
            
        Returns:
            Complete difficulty profile with all available metrics
        """
        task_id = getattr(episode, 'task_id', f"episode_{hash(str(episode))}")
        
        difficulty_scores = {}
        metadata = {}
        
        try:
            # Statistical complexity measures (always computed)
            complexity_measures = self.complexity_analyzer.compute_all_complexity_measures(
                episode.support_x, episode.support_y
            )
            
            # Map complexity measures to difficulty metrics
            if 'fisher_discriminant_ratio' in complexity_measures:
                difficulty_scores[DifficultyMetric.FISHER_DISCRIMINANT_RATIO] = \
                    complexity_measures['fisher_discriminant_ratio']
                    
            if 'class_separability' in complexity_measures:
                difficulty_scores[DifficultyMetric.CLASS_SEPARABILITY] = \
                    complexity_measures['class_separability']
                    
            if 'neighborhood_separability' in complexity_measures:
                difficulty_scores[DifficultyMetric.NEIGHBORHOOD_SEPARABILITY] = \
                    complexity_measures['neighborhood_separability']
                    
            if 'feature_efficiency' in complexity_measures:
                difficulty_scores[DifficultyMetric.FEATURE_EFFICIENCY] = \
                    complexity_measures['feature_efficiency']
                    
            if 'boundary_complexity' in complexity_measures:
                difficulty_scores[DifficultyMetric.BOUNDARY_COMPLEXITY] = \
                    complexity_measures['boundary_complexity']
            
            # Meta-learning specific measures (always computed if possible)
            try:
                meta_measures = self.meta_analyzer.compute_all_meta_learning_measures(
                    model if model is not None else self._create_dummy_model(episode),
                    episode.support_x, episode.support_y,
                    episode.query_x, episode.query_y
                )
                
                if 'few_shot_transferability' in meta_measures:
                    difficulty_scores[DifficultyMetric.FEW_SHOT_TRANSFERABILITY] = \
                        meta_measures['few_shot_transferability']
                        
                if model is not None:  # Only include model-dependent measures if model provided
                    if 'adaptation_difficulty' in meta_measures:
                        difficulty_scores[DifficultyMetric.ADAPTATION_DIFFICULTY] = \
                            meta_measures['adaptation_difficulty']
                            
                    if 'generalization_gap' in meta_measures:
                        difficulty_scores[DifficultyMetric.GENERALIZATION_GAP] = \
                            meta_measures['generalization_gap']
                        
            except Exception as e:
                self.logger.warning(f"Error computing meta-learning measures: {e}")
            
            # Learning dynamics measures (if model provided and requested)
            if model is not None and include_learning_dynamics:
                try:
                    dynamics_measures = self.dynamics_analyzer.compute_all_dynamics_measures(
                        model, episode.support_x, episode.support_y
                    )
                    
                    if 'convergence_rate' in dynamics_measures:
                        difficulty_scores[DifficultyMetric.CONVERGENCE_RATE] = \
                            dynamics_measures['convergence_rate']
                            
                    if 'gradient_variance' in dynamics_measures:
                        difficulty_scores[DifficultyMetric.GRADIENT_VARIANCE] = \
                            dynamics_measures['gradient_variance']
                            
                    if 'loss_landscape_smoothness' in dynamics_measures:
                        difficulty_scores[DifficultyMetric.LOSS_LANDSCAPE_SMOOTHNESS] = \
                            dynamics_measures['loss_landscape_smoothness']
                            
                except Exception as e:
                    self.logger.warning(f"Error computing learning dynamics: {e}")
            
            # Collect metadata
            metadata = {
                "n_way": len(torch.unique(episode.support_y)),
                "n_shot": len(episode.support_y) // len(torch.unique(episode.support_y)),
                "n_query": len(episode.query_y),
                "support_shape": list(episode.support_x.shape),
                "query_shape": list(episode.query_x.shape),
                "has_model": model is not None,
                "included_learning_dynamics": include_learning_dynamics and model is not None
            }
            
        except Exception as e:
            self.logger.error(f"Error assessing difficulty for {task_id}: {e}")
            # Provide default difficulty scores
            for metric in DifficultyMetric:
                difficulty_scores[metric] = 0.5
        
        profile = TaskDifficultyProfile(
            task_id=task_id,
            difficulty_scores=difficulty_scores,
            metadata=metadata
        )
        
        return profile
    
    def assess_episode_batch_difficulty(
        self,
        episodes: List[Episode],
        model: Optional[nn.Module] = None,
        include_learning_dynamics: bool = True
    ) -> List[TaskDifficultyProfile]:
        """
        Assess difficulty for multiple episodes.
        
        Args:
            episodes: List of episodes to assess
            model: Optional model for learning dynamics analysis
            include_learning_dynamics: Whether to include learning-based metrics
            
        Returns:
            List of difficulty profiles with rankings
        """
        profiles = []
        
        for i, episode in enumerate(episodes):
            if i % 10 == 0:
                self.logger.info(f"Assessing difficulty for episode {i+1}/{len(episodes)}")
            
            profile = self.assess_episode_difficulty(
                episode, model, include_learning_dynamics
            )
            profiles.append(profile)
        
        # Add difficulty rankings
        self._add_difficulty_rankings(profiles)
        
        return profiles
    
    def _add_difficulty_rankings(self, profiles: List[TaskDifficultyProfile]) -> None:
        """Add difficulty rankings to profiles."""
        # Sort by overall difficulty
        sorted_profiles = sorted(profiles, key=lambda p: p.overall_difficulty)
        
        # Add rankings (1 = easiest, len(profiles) = hardest)
        for rank, profile in enumerate(sorted_profiles):
            profile.difficulty_ranking = rank + 1
    
    def analyze_difficulty_distribution(
        self,
        profiles: List[TaskDifficultyProfile]
    ) -> Dict[str, Any]:
        """
        Analyze difficulty distribution across tasks.
        
        Args:
            profiles: List of difficulty profiles
            
        Returns:
            Dictionary with distribution statistics and analysis
        """
        if not profiles:
            return {}
        
        analysis = {}
        
        # Overall difficulty statistics
        difficulties = [p.overall_difficulty for p in profiles]
        analysis["overall_difficulty"] = {
            "mean": np.mean(difficulties),
            "std": np.std(difficulties),
            "min": np.min(difficulties),
            "max": np.max(difficulties),
            "median": np.median(difficulties),
            "quartiles": np.percentile(difficulties, [25, 50, 75]).tolist()
        }
        
        # Per-metric analysis
        analysis["per_metric"] = {}
        for metric in DifficultyMetric:
            scores = [
                p.difficulty_scores.get(metric, 0) 
                for p in profiles 
                if metric in p.difficulty_scores
            ]
            
            if scores:
                analysis["per_metric"][metric.value] = {
                    "mean": np.mean(scores),
                    "std": np.std(scores),
                    "correlation_with_overall": np.corrcoef(
                        scores, 
                        [p.overall_difficulty for p in profiles if metric in p.difficulty_scores]
                    )[0, 1] if len(scores) > 1 else 0.0
                }
        
        # Task complexity patterns
        n_ways = [p.metadata.get("n_way", 0) for p in profiles]
        n_shots = [p.metadata.get("n_shot", 0) for p in profiles]
        
        if len(set(n_ways)) > 1:
            analysis["n_way_effect"] = self._analyze_factor_effect(n_ways, difficulties)
        if len(set(n_shots)) > 1:
            analysis["n_shot_effect"] = self._analyze_factor_effect(n_shots, difficulties)
        
        return analysis
    
    def _analyze_factor_effect(
        self,
        factor_values: List[int],
        difficulties: List[float]
    ) -> Dict[str, float]:
        """Analyze the effect of a factor on difficulty."""
        from scipy.stats import pearsonr
        
        correlation, p_value = pearsonr(factor_values, difficulties)
        
        return {
            "correlation": correlation,
            "p_value": p_value,
            "strength": "strong" if abs(correlation) > 0.7 else "moderate" if abs(correlation) > 0.4 else "weak"
        }
    
    def get_most_difficult_episodes(
        self,
        profiles: List[TaskDifficultyProfile],
        n_episodes: int = 10
    ) -> List[TaskDifficultyProfile]:
        """
        Get the most difficult episodes from a batch.
        
        Args:
            profiles: List of difficulty profiles
            n_episodes: Number of most difficult episodes to return
            
        Returns:
            List of most difficult episodes, sorted by difficulty
        """
        sorted_profiles = sorted(
            profiles, 
            key=lambda p: p.overall_difficulty, 
            reverse=True
        )
        return sorted_profiles[:n_episodes]
    
    def get_easiest_episodes(
        self,
        profiles: List[TaskDifficultyProfile],
        n_episodes: int = 10
    ) -> List[TaskDifficultyProfile]:
        """
        Get the easiest episodes from a batch.
        
        Args:
            profiles: List of difficulty profiles
            n_episodes: Number of easiest episodes to return
            
        Returns:
            List of easiest episodes, sorted by difficulty
        """
        sorted_profiles = sorted(
            profiles, 
            key=lambda p: p.overall_difficulty
        )
        return sorted_profiles[:n_episodes]
    
    def _create_dummy_model(self, episode: Episode) -> nn.Module:
        """Create a simple dummy model for episodes without a provided model."""
        input_shape = episode.support_x.shape[1:]
        n_classes = len(torch.unique(episode.support_y))
        
        # Simple linear model
        if len(input_shape) == 1:
            # Vector input
            return nn.Linear(input_shape[0], n_classes)
        else:
            # Image or tensor input - flatten and use linear
            flattened_size = np.prod(input_shape)
            return nn.Sequential(
                nn.Flatten(),
                nn.Linear(flattened_size, n_classes)
            )


def assess_episode_difficulty(
    episode: Episode,
    model: Optional[nn.Module] = None,
    **kwargs
) -> TaskDifficultyProfile:
    """
    Convenience function to assess episode difficulty.
    
    Args:
        episode: Episode to assess
        model: Optional model for learning dynamics analysis
        **kwargs: Additional arguments
        
    Returns:
        Task difficulty profile
    """
    assessor = TaskDifficultyAssessor()
    return assessor.assess_episode_difficulty(episode, model, **kwargs)


def compare_episode_difficulties(
    episodes: List[Episode],
    model: Optional[nn.Module] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Compare difficulties across multiple episodes.
    
    Args:
        episodes: List of episodes to compare
        model: Optional model for learning dynamics analysis
        **kwargs: Additional arguments
        
    Returns:
        Comparison results with statistics and rankings
    """
    assessor = TaskDifficultyAssessor()
    profiles = assessor.assess_episode_batch_difficulty(episodes, model, **kwargs)
    
    return {
        "profiles": profiles,
        "distribution_analysis": assessor.analyze_difficulty_distribution(profiles),
        "most_difficult": assessor.get_most_difficult_episodes(profiles, 5),
        "easiest": assessor.get_easiest_episodes(profiles, 5)
    }


if __name__ == "__main__":
    # Demonstration
    import torch.nn as nn
    
    torch.manual_seed(42)
    
    # Create synthetic episodes with varying difficulty
    episodes = []
    for i in range(20):
        n_way = np.random.choice([3, 5])
        n_shot = np.random.choice([1, 5])
        
        # Create synthetic data with controlled separability
        difficulty_factor = np.random.uniform(0.1, 2.0)  # Easy to hard
        
        support_x = []
        support_y = []
        
        for cls in range(n_way):
            # Create class-specific data with controlled separability
            class_center = np.random.randn(84) * 2
            class_data = np.random.randn(n_shot, 84) * difficulty_factor + class_center
            
            support_x.append(class_data)
            support_y.extend([cls] * n_shot)
        
        support_x = torch.tensor(np.vstack(support_x), dtype=torch.float32)
        support_y = torch.tensor(support_y, dtype=torch.long)
        
        # Query set
        query_x = torch.randn(n_way * 5, 84)
        query_y = torch.repeat_interleave(torch.arange(n_way), 5)
        
        episode = Episode(support_x, support_y, query_x, query_y)
        episode.task_id = f"synthetic_task_{i}"
        episodes.append(episode)
    
    # Create simple model for testing
    model = nn.Sequential(
        nn.Linear(84, 64),
        nn.ReLU(),
        nn.Linear(64, 5)  # Max 5 classes
    )
    
    # Run comparison
    results = compare_episode_difficulties(episodes, model)
    
    print("Task Difficulty Analysis Results:")
    print(f"Mean difficulty: {results['distribution_analysis']['overall_difficulty']['mean']:.3f}")
    print(f"Std difficulty: {results['distribution_analysis']['overall_difficulty']['std']:.3f}")
    
    print("\nTop 5 Most Difficult Tasks:")
    for i, profile in enumerate(results['most_difficult']):
        print(f"{i+1}. {profile.task_id}: {profile.overall_difficulty:.3f}")
    
    print("\nTop 5 Easiest Tasks:")
    for i, profile in enumerate(results['easiest']):
        print(f"{i+1}. {profile.task_id}: {profile.overall_difficulty:.3f}")