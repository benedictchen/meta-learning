"""
Task difficulty assessment for meta-learning evaluation.

This module provides comprehensive analysis of task difficulty across multiple dimensions:
- Statistical complexity measures (class separability, feature complexity)
- Learning dynamics analysis (convergence rate, gradient variance)
- Dataset-specific measures (intra/inter-class variance, margin analysis)
- Meta-learning specific measures (adaptation difficulty, generalization gap)

Key classes:
- TaskDifficultyAssessor: Main difficulty assessment coordinator
- ComplexityAnalyzer: Statistical complexity measures
- LearningDynamicsAnalyzer: Learning-based difficulty metrics
- DatasetComplexityMeasures: Dataset-specific complexity analysis

Research foundations:
- Ho & Basu (2002): Data Complexity measures in pattern recognition
- Lorena et al. (2019): Complex Networks measures for classification
- García et al. (2016): An insight into classification with imbalanced data
- Muñoz et al. (2018): Instance spaces for machine learning classification
"""

from typing import Dict, List, Any, Tuple, Optional, Union, Callable
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import warnings
from scipy.stats import entropy, pearsonr, spearmanr
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import logging

from ..core.episode import Episode


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


class ComplexityAnalyzer:
    """Statistical and geometrical complexity measures."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def fisher_discriminant_ratio(self, X: torch.Tensor, y: torch.Tensor) -> float:
        """
        Fisher's discriminant ratio: between-class variance / within-class variance.
        Higher values indicate easier separation.
        """
        classes = torch.unique(y)
        n_classes = len(classes)
        
        if n_classes < 2:
            return 0.0
            
        # Convert to numpy for easier computation
        X_np = X.cpu().numpy()
        y_np = y.cpu().numpy()
        
        # Overall mean
        overall_mean = np.mean(X_np, axis=0)
        
        # Between-class variance
        between_var = 0.0
        for cls in classes:
            mask = (y_np == cls.item())
            class_mean = np.mean(X_np[mask], axis=0)
            n_samples = np.sum(mask)
            between_var += n_samples * np.sum((class_mean - overall_mean) ** 2)
        
        # Within-class variance
        within_var = 0.0
        for cls in classes:
            mask = (y_np == cls.item())
            class_samples = X_np[mask]
            class_mean = np.mean(class_samples, axis=0)
            within_var += np.sum((class_samples - class_mean) ** 2)
        
        if within_var == 0:
            return float('inf') if between_var > 0 else 1.0
            
        fdr = between_var / within_var
        
        # Normalize to [0, 1] scale (higher = easier)
        # Use sigmoid transformation to bound the ratio
        normalized_fdr = 1 / (1 + np.exp(-np.log(fdr + 1e-8)))
        
        # Return difficulty (invert so higher = more difficult)
        return 1.0 - normalized_fdr
    
    def class_separability(self, X: torch.Tensor, y: torch.Tensor) -> float:
        """
        Measure class separability using silhouette score.
        Higher silhouette = better separation = lower difficulty.
        """
        try:
            X_np = X.cpu().numpy()
            y_np = y.cpu().numpy()
            
            if len(np.unique(y_np)) < 2:
                return 1.0  # Maximum difficulty for single class
            
            # Compute silhouette score
            silhouette = silhouette_score(X_np, y_np)
            
            # Convert to difficulty (silhouette ranges from -1 to 1)
            # Higher silhouette = easier task, so invert
            difficulty = (1.0 - silhouette) / 2.0  # Maps to [0, 1]
            
            return difficulty
            
        except Exception as e:
            self.logger.warning(f"Error computing class separability: {e}")
            return 0.5  # Default moderate difficulty
    
    def neighborhood_separability(self, X: torch.Tensor, y: torch.Tensor, k: int = 3) -> float:
        """
        Measure how well classes are separated in local neighborhoods.
        """
        X_np = X.cpu().numpy()
        y_np = y.cpu().numpy()
        n_samples = len(X_np)
        
        if n_samples <= k:
            return 0.5
            
        # Find k-nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=k+1).fit(X_np)  # +1 to exclude self
        indices = nbrs.kneighbors(X_np, return_distance=False)[:, 1:]  # Exclude self
        
        # Count misclassified neighbors
        total_neighbors = 0
        misclassified = 0
        
        for i in range(n_samples):
            neighbor_labels = y_np[indices[i]]
            true_label = y_np[i]
            
            total_neighbors += len(neighbor_labels)
            misclassified += np.sum(neighbor_labels != true_label)
        
        # Higher misclassification rate = higher difficulty
        if total_neighbors == 0:
            return 0.5
            
        difficulty = misclassified / total_neighbors
        return difficulty
    
    def feature_efficiency(self, X: torch.Tensor, y: torch.Tensor) -> float:
        """
        Measure how efficiently features separate classes using PCA.
        """
        try:
            X_np = X.cpu().numpy()
            y_np = y.cpu().numpy()
            
            # Apply PCA
            pca = PCA()
            X_pca = pca.fit_transform(X_np)
            
            # Compute class separability in each PC component
            separabilities = []
            for i in range(min(5, X_pca.shape[1])):  # Check first 5 components
                component = X_pca[:, i:i+1]
                sep = self._component_separability(component, y_np)
                separabilities.append(sep)
            
            if not separabilities:
                return 0.5
                
            # Weight by explained variance
            weights = pca.explained_variance_ratio_[:len(separabilities)]
            weighted_sep = np.average(separabilities, weights=weights)
            
            # Convert to difficulty (higher separability = lower difficulty)
            difficulty = 1.0 - weighted_sep
            return difficulty
            
        except Exception as e:
            self.logger.warning(f"Error computing feature efficiency: {e}")
            return 0.5
    
    def _component_separability(self, component: np.ndarray, y: np.ndarray) -> float:
        """Measure separability in a single component."""
        classes = np.unique(y)
        if len(classes) < 2:
            return 0.0
            
        # Compute between/within class variance in this component
        overall_var = np.var(component)
        if overall_var == 0:
            return 0.0
            
        within_var = 0.0
        for cls in classes:
            mask = (y == cls)
            class_component = component[mask]
            if len(class_component) > 1:
                within_var += np.var(class_component) * np.sum(mask)
        
        within_var /= len(y)
        
        separability = 1 - (within_var / overall_var)
        return max(0.0, separability)


class LearningDynamicsAnalyzer:
    """Analyze learning dynamics to assess task difficulty."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def convergence_rate(
        self, 
        model: nn.Module,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        n_steps: int = 10,
        lr: float = 0.01
    ) -> float:
        """
        Measure convergence rate during few-shot adaptation.
        Slower convergence = higher difficulty.
        """
        model.eval()
        
        # Clone model for temporary training
        temp_model = self._clone_model(model)
        temp_model.train()
        
        optimizer = torch.optim.SGD(temp_model.parameters(), lr=lr)
        
        losses = []
        for step in range(n_steps):
            optimizer.zero_grad()
            logits = temp_model(support_x)
            loss = F.cross_entropy(logits, support_y)
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
        
        if len(losses) < 2:
            return 0.5
            
        # Measure convergence rate as relative improvement
        initial_loss = losses[0]
        final_loss = losses[-1]
        
        if initial_loss == 0:
            return 0.0
            
        improvement_rate = (initial_loss - final_loss) / initial_loss
        
        # Convert to difficulty (slower convergence = higher difficulty)
        # Sigmoid transformation to handle negative improvements
        difficulty = 1 / (1 + np.exp(5 * (improvement_rate - 0.2)))
        
        return float(difficulty)
    
    def gradient_variance(
        self,
        model: nn.Module, 
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        n_samples: int = 10
    ) -> float:
        """
        Measure gradient variance across multiple forward passes.
        Higher variance = more difficult optimization.
        """
        model.eval()
        
        gradients = []
        for _ in range(n_samples):
            model.zero_grad()
            logits = model(support_x)
            loss = F.cross_entropy(logits, support_y)
            loss.backward()
            
            # Collect gradients
            grad_norm = 0.0
            for param in model.parameters():
                if param.grad is not None:
                    grad_norm += torch.norm(param.grad).item() ** 2
            
            gradients.append(grad_norm ** 0.5)
        
        if len(gradients) < 2:
            return 0.0
            
        # Compute coefficient of variation
        mean_grad = np.mean(gradients)
        std_grad = np.std(gradients)
        
        if mean_grad == 0:
            return 0.0
            
        cv = std_grad / mean_grad
        
        # Normalize to [0, 1] range
        difficulty = min(1.0, cv / 2.0)  # Assume CV > 2 is maximum difficulty
        
        return difficulty
    
    def loss_landscape_smoothness(
        self,
        model: nn.Module,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        n_perturbations: int = 20,
        perturbation_scale: float = 0.01
    ) -> float:
        """
        Measure loss landscape smoothness around current parameters.
        Rougher landscape = higher difficulty.
        """
        model.eval()
        
        # Get baseline loss
        baseline_logits = model(support_x)
        baseline_loss = F.cross_entropy(baseline_logits, support_y).item()
        
        loss_variations = []
        
        with torch.no_grad():
            original_params = [p.clone() for p in model.parameters()]
            
            for _ in range(n_perturbations):
                # Add random perturbation
                for param, orig_param in zip(model.parameters(), original_params):
                    noise = torch.randn_like(param) * perturbation_scale
                    param.data = orig_param + noise
                
                # Compute perturbed loss
                perturbed_logits = model(support_x)
                perturbed_loss = F.cross_entropy(perturbed_logits, support_y).item()
                
                loss_variations.append(abs(perturbed_loss - baseline_loss))
            
            # Restore original parameters
            for param, orig_param in zip(model.parameters(), original_params):
                param.data = orig_param.clone()
        
        if not loss_variations:
            return 0.5
            
        # Higher variance = rougher landscape = higher difficulty
        variance = np.var(loss_variations)
        difficulty = min(1.0, variance / 0.1)  # Normalize assuming max variance of 0.1
        
        return difficulty
    
    def _clone_model(self, model: nn.Module) -> nn.Module:
        """Create a deep copy of the model."""
        import copy
        return copy.deepcopy(model)


class MetaLearningSpecificAnalyzer:
    """Meta-learning specific difficulty measures."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def adaptation_difficulty(
        self,
        model: nn.Module,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        query_x: torch.Tensor,
        query_y: torch.Tensor,
        n_adaptation_steps: int = 5
    ) -> float:
        """
        Measure how difficult it is to adapt from support to query set.
        """
        model.eval()
        
        # Initial performance on query set (before adaptation)
        with torch.no_grad():
            initial_logits = model(query_x)
            initial_acc = (initial_logits.argmax(-1) == query_y).float().mean().item()
        
        # Adapt on support set
        temp_model = self._clone_model(model)
        temp_model.train()
        
        optimizer = torch.optim.SGD(temp_model.parameters(), lr=0.01)
        
        for _ in range(n_adaptation_steps):
            optimizer.zero_grad()
            support_logits = temp_model(support_x)
            loss = F.cross_entropy(support_logits, support_y)
            loss.backward()
            optimizer.step()
        
        # Final performance on query set (after adaptation)
        temp_model.eval()
        with torch.no_grad():
            final_logits = temp_model(query_x)
            final_acc = (final_logits.argmax(-1) == query_y).float().mean().item()
        
        # Measure adaptation effectiveness
        improvement = final_acc - initial_acc
        
        # Convert to difficulty (less improvement = higher difficulty)
        # Use sigmoid to bound the values
        difficulty = 1 / (1 + np.exp(10 * (improvement - 0.1)))
        
        return float(difficulty)
    
    def generalization_gap(
        self,
        model: nn.Module,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        query_x: torch.Tensor,
        query_y: torch.Tensor,
        n_adaptation_steps: int = 5
    ) -> float:
        """
        Measure generalization gap between support and query performance.
        """
        temp_model = self._clone_model(model)
        temp_model.train()
        
        optimizer = torch.optim.SGD(temp_model.parameters(), lr=0.01)
        
        # Adapt on support set
        for _ in range(n_adaptation_steps):
            optimizer.zero_grad()
            support_logits = temp_model(support_x)
            loss = F.cross_entropy(support_logits, support_y)
            loss.backward()
            optimizer.step()
        
        temp_model.eval()
        with torch.no_grad():
            # Performance on support set
            support_logits = temp_model(support_x)
            support_acc = (support_logits.argmax(-1) == support_y).float().mean().item()
            
            # Performance on query set
            query_logits = temp_model(query_x)
            query_acc = (query_logits.argmax(-1) == query_y).float().mean().item()
        
        # Generalization gap (higher gap = higher difficulty)
        gap = max(0, support_acc - query_acc)
        
        return gap
    
    def few_shot_transferability(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        n_shot: int = 1
    ) -> float:
        """
        Measure how well few-shot examples represent the full class distribution.
        """
        classes = torch.unique(y)
        n_classes = len(classes)
        
        if n_classes < 2 or n_shot >= len(y) // n_classes:
            return 0.5
        
        # For each class, sample n_shot examples and measure representativeness
        representativeness_scores = []
        
        for cls in classes:
            class_mask = (y == cls)
            class_samples = X[class_mask]
            
            if len(class_samples) <= n_shot:
                representativeness_scores.append(1.0)  # Perfect if we have all samples
                continue
            
            # Sample n_shot examples
            indices = torch.randperm(len(class_samples))[:n_shot]
            few_shot_samples = class_samples[indices]
            
            # Measure how well they represent the full class distribution
            representativeness = self._compute_representativeness(
                few_shot_samples, class_samples
            )
            representativeness_scores.append(representativeness)
        
        avg_representativeness = np.mean(representativeness_scores)
        
        # Convert to difficulty (lower representativeness = higher difficulty)
        difficulty = 1.0 - avg_representativeness
        
        return difficulty
    
    def _compute_representativeness(
        self,
        few_shot_samples: torch.Tensor,
        all_samples: torch.Tensor
    ) -> float:
        """Compute how well few-shot samples represent the full distribution."""
        # Simple approach: compare centroids and spread
        few_shot_mean = torch.mean(few_shot_samples, dim=0)
        all_samples_mean = torch.mean(all_samples, dim=0)
        
        # Distance between centroids (normalized)
        centroid_distance = torch.norm(few_shot_mean - all_samples_mean).item()
        centroid_similarity = 1 / (1 + centroid_distance)
        
        # Spread comparison
        few_shot_std = torch.std(few_shot_samples, dim=0).mean().item()
        all_samples_std = torch.std(all_samples, dim=0).mean().item()
        
        if all_samples_std == 0:
            spread_similarity = 1.0 if few_shot_std == 0 else 0.0
        else:
            spread_ratio = min(few_shot_std / all_samples_std, all_samples_std / few_shot_std)
            spread_similarity = spread_ratio
        
        # Combine measures
        representativeness = 0.6 * centroid_similarity + 0.4 * spread_similarity
        
        return representativeness
    
    def _clone_model(self, model: nn.Module) -> nn.Module:
        """Create a deep copy of the model."""
        import copy
        return copy.deepcopy(model)


class TaskDifficultyAssessor:
    """Main coordinator for task difficulty assessment."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
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
            Complete difficulty profile
        """
        task_id = getattr(episode, 'task_id', f"episode_{hash(str(episode))}")
        
        difficulty_scores = {}
        metadata = {}
        
        try:
            # Statistical complexity measures
            difficulty_scores[DifficultyMetric.FISHER_DISCRIMINANT_RATIO] = \
                self.complexity_analyzer.fisher_discriminant_ratio(episode.support_x, episode.support_y)
            
            difficulty_scores[DifficultyMetric.CLASS_SEPARABILITY] = \
                self.complexity_analyzer.class_separability(episode.support_x, episode.support_y)
            
            difficulty_scores[DifficultyMetric.NEIGHBORHOOD_SEPARABILITY] = \
                self.complexity_analyzer.neighborhood_separability(episode.support_x, episode.support_y)
            
            difficulty_scores[DifficultyMetric.FEATURE_EFFICIENCY] = \
                self.complexity_analyzer.feature_efficiency(episode.support_x, episode.support_y)
            
            # Meta-learning specific measures
            difficulty_scores[DifficultyMetric.FEW_SHOT_TRANSFERABILITY] = \
                self.meta_analyzer.few_shot_transferability(episode.support_x, episode.support_y)
            
            # Learning dynamics measures (if model provided)
            if model is not None and include_learning_dynamics:
                difficulty_scores[DifficultyMetric.CONVERGENCE_RATE] = \
                    self.dynamics_analyzer.convergence_rate(
                        model, episode.support_x, episode.support_y
                    )
                
                difficulty_scores[DifficultyMetric.GRADIENT_VARIANCE] = \
                    self.dynamics_analyzer.gradient_variance(
                        model, episode.support_x, episode.support_y
                    )
                
                difficulty_scores[DifficultyMetric.ADAPTATION_DIFFICULTY] = \
                    self.meta_analyzer.adaptation_difficulty(
                        model, episode.support_x, episode.support_y,
                        episode.query_x, episode.query_y
                    )
                
                difficulty_scores[DifficultyMetric.GENERALIZATION_GAP] = \
                    self.meta_analyzer.generalization_gap(
                        model, episode.support_x, episode.support_y,
                        episode.query_x, episode.query_y
                    )
            
            # Collect metadata
            metadata = {
                "n_way": len(torch.unique(episode.support_y)),
                "n_shot": len(episode.support_y) // len(torch.unique(episode.support_y)),
                "n_query": len(episode.query_y),
                "support_shape": list(episode.support_x.shape),
                "query_shape": list(episode.query_x.shape)
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
        """Assess difficulty for multiple episodes."""
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
        
        # Add rankings
        for rank, profile in enumerate(sorted_profiles):
            profile.difficulty_ranking = rank + 1
    
    def analyze_difficulty_distribution(
        self,
        profiles: List[TaskDifficultyProfile]
    ) -> Dict[str, Any]:
        """Analyze difficulty distribution across tasks."""
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
                        scores, [p.overall_difficulty for p in profiles if metric in p.difficulty_scores]
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
        correlation, p_value = pearsonr(factor_values, difficulties)
        
        return {
            "correlation": correlation,
            "p_value": p_value,
            "strength": "strong" if abs(correlation) > 0.7 else "moderate" if abs(correlation) > 0.4 else "weak"
        }


def demonstrate_task_difficulty_assessment():
    """Demonstrate task difficulty assessment."""
    # Create mock episodes with varying difficulty
    episodes = []
    
    for i in range(20):
        n_way = np.random.choice([3, 5])
        n_shot = np.random.choice([1, 5])
        
        # Create synthetic data with varying separability
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
    
    # Create difficulty assessor
    assessor = TaskDifficultyAssessor()
    
    # Assess difficulties
    print("Assessing task difficulties...")
    profiles = assessor.assess_episode_batch_difficulty(episodes, include_learning_dynamics=False)
    
    # Analyze results
    analysis = assessor.analyze_difficulty_distribution(profiles)
    
    print("\nDifficulty Distribution Analysis:")
    print(f"Mean difficulty: {analysis['overall_difficulty']['mean']:.3f}")
    print(f"Std difficulty: {analysis['overall_difficulty']['std']:.3f}")
    print(f"Difficulty range: [{analysis['overall_difficulty']['min']:.3f}, {analysis['overall_difficulty']['max']:.3f}]")
    
    print("\nTop 5 Most Difficult Tasks:")
    sorted_profiles = sorted(profiles, key=lambda p: p.overall_difficulty, reverse=True)
    for i, profile in enumerate(sorted_profiles[:5]):
        print(f"{i+1}. {profile.task_id}: {profile.overall_difficulty:.3f}")
    
    print("\nTop 5 Easiest Tasks:")
    for i, profile in enumerate(sorted_profiles[-5:]):
        print(f"{i+1}. {profile.task_id}: {profile.overall_difficulty:.3f}")
    
    return profiles, analysis


if __name__ == "__main__":
    # Run demonstration
    demonstrate_task_difficulty_assessment()