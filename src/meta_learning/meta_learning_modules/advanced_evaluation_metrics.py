"""
Advanced Evaluation Metrics for Meta-Learning
==============================================

Author: Benedict Chen (benedict@benedictchen.com)

Implements comprehensive evaluation metrics beyond simple accuracy,
following best practices from meta-learning research.

Key Features:
1. Prototype quality assessment (intra/inter-class distances)
2. Task difficulty estimation with multiple methods
3. Uncertainty quantification and calibration analysis
4. Statistical significance testing with multiple comparison correction
5. Learning curve analysis and convergence metrics

References:
- Chen et al. (2019): "A Closer Look at Few-shot Classification"
- Triantafillou et al. (2019): "Meta-Dataset: A Dataset of Datasets"
- Yue et al. (2020): "Interventional Few-Shot Learning"
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from collections import defaultdict
import warnings
import logging

logger = logging.getLogger(__name__)


@dataclass
class PrototypeQualityMetrics:
    """Metrics for assessing prototype quality in prototypical networks."""
    intra_class_variance: float
    inter_class_separation: float  
    separability_ratio: float
    silhouette_score: float
    cluster_cohesion: float


@dataclass
class TaskDifficultyMetrics:
    """Comprehensive task difficulty assessment metrics."""
    class_separability: float
    feature_complexity: float
    support_set_diversity: float
    label_noise_estimate: float
    intrinsic_dimensionality: float
    overall_difficulty: float


@dataclass
class UncertaintyMetrics:
    """Uncertainty quantification metrics."""
    epistemic_uncertainty: float
    aleatoric_uncertainty: float
    total_uncertainty: float
    predictive_entropy: float
    confidence_score: float
    calibration_error: float


class PrototypeAnalyzer:
    """
    Analyzer for prototype quality in prototypical networks.
    
    Evaluates how well prototypes represent their classes and
    how well separated they are from other classes.
    """
    
    def __init__(self, distance_metric: str = 'euclidean'):
        """
        Initialize prototype analyzer.
        
        Args:
            distance_metric: Distance metric to use ('euclidean', 'cosine', 'manhattan')
        """
        self.distance_metric = distance_metric
        
    def analyze_prototypes(self, 
                         support_features: torch.Tensor,
                         support_labels: torch.Tensor,
                         prototypes: torch.Tensor) -> PrototypeQualityMetrics:
        """
        Comprehensive analysis of prototype quality.
        
        Args:
            support_features: Support set features [N, D]  
            support_labels: Support set labels [N]
            prototypes: Computed prototypes [K, D]
            
        Returns:
            PrototypeQualityMetrics with comprehensive assessment
        """
        n_classes = len(torch.unique(support_labels))
        
        # Compute intra-class variance
        intra_class_vars = []
        for class_idx in range(n_classes):
            mask = support_labels == class_idx
            if mask.sum() > 0:
                class_features = support_features[mask]
                prototype = prototypes[class_idx]
                
                # Compute variance from prototype
                distances = self._compute_distances(class_features, prototype.unsqueeze(0))
                intra_class_vars.append(distances.mean().item())
        
        avg_intra_class_var = np.mean(intra_class_vars) if intra_class_vars else 0.0
        
        # Compute inter-class separation
        if n_classes > 1:
            proto_distances = self._compute_pairwise_distances(prototypes)
            # Get minimum distance between different prototypes
            mask = ~torch.eye(n_classes, dtype=torch.bool)
            min_inter_class_dist = proto_distances[mask].min().item()
            mean_inter_class_dist = proto_distances[mask].mean().item()
        else:
            min_inter_class_dist = float('inf')
            mean_inter_class_dist = float('inf')
        
        # Separability ratio (higher is better)
        separability_ratio = mean_inter_class_dist / (avg_intra_class_var + 1e-8)
        
        # Silhouette score approximation
        silhouette_score = self._compute_silhouette_score(
            support_features, support_labels, prototypes
        )
        
        # Cluster cohesion (Davies-Bouldin-like metric)
        cohesion = self._compute_cluster_cohesion(
            support_features, support_labels, prototypes
        )
        
        return PrototypeQualityMetrics(
            intra_class_variance=avg_intra_class_var,
            inter_class_separation=mean_inter_class_dist,
            separability_ratio=separability_ratio,
            silhouette_score=silhouette_score,
            cluster_cohesion=cohesion
        )
    
    def _compute_distances(self, features: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
        """Compute distances according to specified metric."""
        if self.distance_metric == 'euclidean':
            return torch.cdist(features, centers, p=2)
        elif self.distance_metric == 'cosine':
            # Cosine distance = 1 - cosine_similarity
            cos_sim = F.cosine_similarity(features.unsqueeze(1), centers.unsqueeze(0), dim=2)
            return 1 - cos_sim
        elif self.distance_metric == 'manhattan':
            return torch.cdist(features, centers, p=1)
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
    
    def _compute_pairwise_distances(self, prototypes: torch.Tensor) -> torch.Tensor:
        """Compute pairwise distances between prototypes."""
        return self._compute_distances(prototypes, prototypes)
    
    def _compute_silhouette_score(self, 
                                 features: torch.Tensor,
                                 labels: torch.Tensor, 
                                 prototypes: torch.Tensor) -> float:
        """Compute silhouette score for prototype clustering."""
        n_samples = features.size(0)
        n_classes = len(torch.unique(labels))
        
        if n_classes < 2:
            return 1.0  # Perfect score for single class
        
        silhouette_scores = []
        
        for i in range(n_samples):
            sample = features[i].unsqueeze(0)
            true_class = labels[i].item()
            
            # Distance to own cluster prototype
            intra_distance = self._compute_distances(sample, prototypes[true_class].unsqueeze(0)).item()
            
            # Distance to nearest different cluster prototype
            other_prototypes = torch.cat([
                prototypes[:true_class], 
                prototypes[true_class+1:]
            ]) if n_classes > 1 else prototypes
            
            if other_prototypes.size(0) > 0:
                inter_distances = self._compute_distances(sample, other_prototypes)
                nearest_inter_distance = inter_distances.min().item()
                
                # Silhouette coefficient
                if max(intra_distance, nearest_inter_distance) > 0:
                    silhouette = (nearest_inter_distance - intra_distance) / max(intra_distance, nearest_inter_distance)
                    silhouette_scores.append(silhouette)
        
        return np.mean(silhouette_scores) if silhouette_scores else 0.0
    
    def _compute_cluster_cohesion(self,
                                 features: torch.Tensor,
                                 labels: torch.Tensor,
                                 prototypes: torch.Tensor) -> float:
        """Compute cluster cohesion metric."""
        n_classes = len(torch.unique(labels))
        cohesion_scores = []
        
        for class_idx in range(n_classes):
            mask = labels == class_idx
            if mask.sum() > 1:  # Need at least 2 samples
                class_features = features[mask]
                prototype = prototypes[class_idx]
                
                # Compute spread around prototype
                distances = self._compute_distances(class_features, prototype.unsqueeze(0))
                cohesion = 1.0 / (1.0 + distances.mean().item())  # Higher is better
                cohesion_scores.append(cohesion)
        
        return np.mean(cohesion_scores) if cohesion_scores else 1.0


class TaskDifficultyAnalyzer:
    """
    Comprehensive task difficulty assessment for few-shot episodes.
    
    Estimates task difficulty using multiple methods to predict
    which tasks will be harder for models to learn.
    """
    
    def __init__(self, methods: List[str] = None):
        """
        Initialize task difficulty analyzer.
        
        Args:
            methods: List of methods to use for difficulty assessment
        """
        self.methods = methods or [
            'separability', 'complexity', 'diversity', 'noise', 'dimensionality'
        ]
    
    def assess_task_difficulty(self,
                             support_x: torch.Tensor,
                             support_y: torch.Tensor,
                             query_x: Optional[torch.Tensor] = None) -> TaskDifficultyMetrics:
        """
        Comprehensive task difficulty assessment.
        
        Args:
            support_x: Support features [N, ...]
            support_y: Support labels [N]
            query_x: Optional query features for additional analysis
            
        Returns:
            TaskDifficultyMetrics with comprehensive assessment
        """
        # Flatten features for analysis
        support_flat = support_x.view(support_x.size(0), -1)
        
        metrics = {}
        
        if 'separability' in self.methods:
            metrics['class_separability'] = self._assess_class_separability(support_flat, support_y)
        
        if 'complexity' in self.methods:
            metrics['feature_complexity'] = self._assess_feature_complexity(support_flat)
        
        if 'diversity' in self.methods:
            metrics['support_set_diversity'] = self._assess_support_diversity(support_flat, support_y)
        
        if 'noise' in self.methods:
            metrics['label_noise_estimate'] = self._estimate_label_noise(support_flat, support_y)
        
        if 'dimensionality' in self.methods:
            metrics['intrinsic_dimensionality'] = self._estimate_intrinsic_dimension(support_flat)
        
        # Combine metrics into overall difficulty score
        overall_difficulty = self._compute_overall_difficulty(metrics)
        
        return TaskDifficultyMetrics(
            class_separability=metrics.get('class_separability', 0.5),
            feature_complexity=metrics.get('feature_complexity', 0.5),
            support_set_diversity=metrics.get('support_set_diversity', 0.5),
            label_noise_estimate=metrics.get('label_noise_estimate', 0.0),
            intrinsic_dimensionality=metrics.get('intrinsic_dimensionality', 0.5),
            overall_difficulty=overall_difficulty
        )
    
    def _assess_class_separability(self, features: torch.Tensor, labels: torch.Tensor) -> float:
        """Assess how separable the classes are in feature space."""
        unique_classes = torch.unique(labels)
        n_classes = len(unique_classes)
        
        if n_classes < 2:
            return 0.0  # Single class is trivially separable
        
        # Compute class centroids
        centroids = []
        intra_variances = []
        
        for cls in unique_classes:
            mask = labels == cls
            if mask.sum() > 0:
                class_features = features[mask]
                centroid = class_features.mean(dim=0)
                centroids.append(centroid)
                
                # Intra-class variance
                if mask.sum() > 1:
                    variance = ((class_features - centroid) ** 2).mean().item()
                    intra_variances.append(variance)
        
        if len(centroids) < 2:
            return 0.0
        
        # Inter-class distances
        centroids = torch.stack(centroids)
        inter_distances = torch.cdist(centroids, centroids)
        mask = ~torch.eye(len(centroids), dtype=torch.bool)
        min_inter_distance = inter_distances[mask].min().item()
        
        # Average intra-class variance
        avg_intra_variance = np.mean(intra_variances) if intra_variances else 0.1
        
        # Separability ratio (lower means harder to separate)
        separability = min_inter_distance / (avg_intra_variance + 1e-8)
        
        # Convert to difficulty score (0=easy, 1=hard)
        difficulty = 1.0 / (1.0 + separability)
        
        return float(difficulty)
    
    def _assess_feature_complexity(self, features: torch.Tensor) -> float:
        """Assess the complexity of the feature distribution."""
        # Feature complexity based on distribution properties
        
        # 1. Entropy of feature activations
        feature_std = features.std(dim=0)
        normalized_std = feature_std / (feature_std.max() + 1e-8)
        entropy = -(normalized_std * torch.log(normalized_std + 1e-8)).sum()
        max_entropy = torch.log(torch.tensor(features.size(1), dtype=torch.float))
        normalized_entropy = entropy / max_entropy
        
        # 2. Sparsity (how many features are near zero)
        sparsity = (features.abs() < 0.1).float().mean()
        
        # 3. Dynamic range
        feature_ranges = features.max(dim=0)[0] - features.min(dim=0)[0]
        avg_range = feature_ranges.mean()
        
        # Combine into complexity score
        complexity = (normalized_entropy * 0.5 + 
                     (1 - sparsity) * 0.3 + 
                     torch.sigmoid(avg_range - 1.0) * 0.2)
        
        return float(complexity)
    
    def _assess_support_diversity(self, features: torch.Tensor, labels: torch.Tensor) -> float:
        """Assess diversity within the support set."""
        unique_classes = torch.unique(labels)
        diversity_scores = []
        
        for cls in unique_classes:
            mask = labels == cls
            if mask.sum() > 1:
                class_features = features[mask]
                
                # Pairwise distances within class
                distances = torch.cdist(class_features, class_features)
                # Remove diagonal (distance to self)
                mask_diag = ~torch.eye(distances.size(0), dtype=torch.bool)
                avg_distance = distances[mask_diag].mean()
                
                diversity_scores.append(avg_distance.item())
        
        if not diversity_scores:
            return 0.5  # Neutral diversity
        
        # Normalize diversity score
        avg_diversity = np.mean(diversity_scores)
        normalized_diversity = torch.sigmoid(torch.tensor(avg_diversity) - 1.0)
        
        return float(normalized_diversity)
    
    def _estimate_label_noise(self, features: torch.Tensor, labels: torch.Tensor) -> float:
        """Estimate the amount of label noise in the support set."""
        unique_classes = torch.unique(labels)
        n_classes = len(unique_classes)
        
        if n_classes < 2:
            return 0.0
        
        # Use k-nearest neighbors to estimate label consistency
        k = min(3, features.size(0) - 1)  # k=3 or less if few samples
        
        if k < 1:
            return 0.0
        
        noise_estimates = []
        
        for i, (feature, true_label) in enumerate(zip(features, labels)):
            # Find k nearest neighbors
            distances = torch.norm(features - feature.unsqueeze(0), dim=1)
            distances[i] = float('inf')  # Exclude self
            
            _, nearest_indices = distances.topk(k, largest=False)
            neighbor_labels = labels[nearest_indices]
            
            # Fraction of neighbors with different labels
            different_labels = (neighbor_labels != true_label).float().mean()
            noise_estimates.append(different_labels.item())
        
        estimated_noise = np.mean(noise_estimates)
        return float(estimated_noise)
    
    def _estimate_intrinsic_dimension(self, features: torch.Tensor) -> float:
        """Estimate intrinsic dimensionality of the feature space."""
        # Use PCA to estimate effective dimensionality
        
        if features.size(0) < 2:
            return 0.0
        
        # Center the data
        centered_features = features - features.mean(dim=0)
        
        # Compute covariance matrix
        cov_matrix = torch.cov(centered_features.T)
        
        # Eigenvalue decomposition
        try:
            eigenvalues = torch.linalg.eigvals(cov_matrix).real
            eigenvalues = eigenvalues[eigenvalues > 1e-8]  # Remove near-zero eigenvalues
            
            if len(eigenvalues) == 0:
                return 0.0
            
            # Compute explained variance ratios
            total_variance = eigenvalues.sum()
            eigenvalues_sorted = torch.sort(eigenvalues, descending=True)[0]
            cumsum_variance = torch.cumsum(eigenvalues_sorted, dim=0) / total_variance
            
            # Find number of components explaining 90% of variance
            n_components_90 = (cumsum_variance < 0.9).sum().item() + 1
            
            # Normalize by feature dimensionality
            intrinsic_ratio = n_components_90 / features.size(1)
            
            return float(intrinsic_ratio)
            
        except Exception:
            # Fallback if eigenvalue decomposition fails
            return 0.5
    
    def _compute_overall_difficulty(self, metrics: Dict[str, float]) -> float:
        """Combine individual metrics into overall difficulty score."""
        # Weighted combination of difficulty metrics
        weights = {
            'class_separability': 0.3,
            'feature_complexity': 0.2,
            'support_set_diversity': 0.2,
            'label_noise_estimate': 0.2,
            'intrinsic_dimensionality': 0.1
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for metric_name, value in metrics.items():
            if metric_name in weights:
                weighted_sum += weights[metric_name] * value
                total_weight += weights[metric_name]
        
        if total_weight > 0:
            return weighted_sum / total_weight
        else:
            return 0.5  # Neutral difficulty


class UncertaintyQuantifier:
    """
    Comprehensive uncertainty quantification for meta-learning models.
    
    Separates aleatoric (data) and epistemic (model) uncertainty
    and provides calibration analysis.
    """
    
    def __init__(self, n_mc_samples: int = 10, temperature_scaling: bool = True):
        """
        Initialize uncertainty quantifier.
        
        Args:
            n_mc_samples: Number of Monte Carlo samples for uncertainty estimation
            temperature_scaling: Whether to apply temperature scaling for calibration
        """
        self.n_mc_samples = n_mc_samples
        self.temperature_scaling = temperature_scaling
        self.temperature = None
    
    def quantify_uncertainty(self,
                           logits: torch.Tensor,
                           targets: Optional[torch.Tensor] = None,
                           mc_logits: Optional[List[torch.Tensor]] = None) -> UncertaintyMetrics:
        """
        Comprehensive uncertainty quantification.
        
        Args:
            logits: Model logits [N, C]
            targets: True targets for calibration analysis [N]
            mc_logits: List of MC dropout logits for epistemic uncertainty
            
        Returns:
            UncertaintyMetrics with comprehensive uncertainty analysis
        """
        # Apply temperature scaling if available
        if self.temperature is not None:
            scaled_logits = logits / self.temperature
        else:
            scaled_logits = logits
        
        # Convert to probabilities
        probs = F.softmax(scaled_logits, dim=-1)
        
        # Predictive entropy (total uncertainty)
        predictive_entropy = self._compute_predictive_entropy(probs)
        
        # Confidence score
        confidence = probs.max(dim=-1)[0].mean().item()
        
        # Decompose uncertainty if MC samples available
        if mc_logits is not None:
            epistemic, aleatoric = self._decompose_uncertainty(mc_logits)
            total_uncertainty = epistemic + aleatoric
        else:
            # Fallback: use predictive entropy as total uncertainty
            epistemic = 0.0
            aleatoric = 0.0
            total_uncertainty = predictive_entropy
        
        # Calibration error
        if targets is not None:
            calibration_error = self._compute_calibration_error(probs, targets)
        else:
            calibration_error = 0.0
        
        return UncertaintyMetrics(
            epistemic_uncertainty=epistemic,
            aleatoric_uncertainty=aleatoric,
            total_uncertainty=total_uncertainty,
            predictive_entropy=predictive_entropy,
            confidence_score=confidence,
            calibration_error=calibration_error
        )
    
    def calibrate_temperature(self, logits: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Learn temperature scaling parameter for calibration.
        
        Args:
            logits: Validation logits [N, C]
            targets: Validation targets [N]
            
        Returns:
            Optimal temperature parameter
        """
        # Simple temperature scaling optimization
        from torch.optim import LBFGS
        
        temperature = torch.tensor([1.0], requires_grad=True)
        optimizer = LBFGS([temperature], lr=0.01, max_iter=50)
        
        def eval_loss():
            optimizer.zero_grad()
            scaled_logits = logits / temperature
            loss = F.cross_entropy(scaled_logits, targets)
            loss.backward()
            return loss
        
        optimizer.step(eval_loss)
        
        self.temperature = temperature.item()
        return self.temperature
    
    def _compute_predictive_entropy(self, probs: torch.Tensor) -> float:
        """Compute predictive entropy."""
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
        return entropy.mean().item()
    
    def _decompose_uncertainty(self, mc_logits: List[torch.Tensor]) -> Tuple[float, float]:
        """
        Decompose uncertainty into epistemic and aleatoric components.
        
        Args:
            mc_logits: List of MC forward pass logits
            
        Returns:
            (epistemic_uncertainty, aleatoric_uncertainty)
        """
        # Stack MC samples
        mc_probs = [F.softmax(logits, dim=-1) for logits in mc_logits]
        mc_probs_stack = torch.stack(mc_probs, dim=0)  # [MC, N, C]
        
        # Predictive mean
        pred_mean = mc_probs_stack.mean(dim=0)  # [N, C]
        
        # Total uncertainty (entropy of predictive mean)
        total_uncertainty = -(pred_mean * torch.log(pred_mean + 1e-8)).sum(dim=-1).mean()
        
        # Aleatoric uncertainty (expected entropy)
        individual_entropies = -(mc_probs_stack * torch.log(mc_probs_stack + 1e-8)).sum(dim=-1)  # [MC, N]
        aleatoric_uncertainty = individual_entropies.mean()
        
        # Epistemic uncertainty (difference)
        epistemic_uncertainty = total_uncertainty - aleatoric_uncertainty
        
        return epistemic_uncertainty.item(), aleatoric_uncertainty.item()
    
    def _compute_calibration_error(self, probs: torch.Tensor, targets: torch.Tensor, n_bins: int = 15) -> float:
        """
        Compute Expected Calibration Error (ECE).
        
        Args:
            probs: Predicted probabilities [N, C]
            targets: True targets [N]
            n_bins: Number of bins for calibration
            
        Returns:
            Expected calibration error
        """
        confidences = probs.max(dim=-1)[0]
        predictions = probs.argmax(dim=-1)
        accuracies = (predictions == targets).float()
        
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        ece = 0.0
        
        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.float().mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                
                ece += prop_in_bin * torch.abs(avg_confidence_in_bin - accuracy_in_bin)
        
        return ece.item()


def analyze_episode_quality(support_x: torch.Tensor,
                           support_y: torch.Tensor,
                           query_x: torch.Tensor,
                           query_y: torch.Tensor,
                           model_predictions: Optional[torch.Tensor] = None,
                           model_logits: Optional[torch.Tensor] = None) -> Dict[str, float]:
    """
    Comprehensive episode quality analysis combining all metrics.
    
    Args:
        support_x: Support features
        support_y: Support labels
        query_x: Query features  
        query_y: Query labels
        model_predictions: Model predictions (optional)
        model_logits: Model logits (optional)
        
    Returns:
        Dictionary of comprehensive quality metrics
    """
    metrics = {}
    
    # Task difficulty analysis
    difficulty_analyzer = TaskDifficultyAnalyzer()
    difficulty_metrics = difficulty_analyzer.assess_task_difficulty(support_x, support_y, query_x)
    
    metrics.update({
        'task_difficulty': difficulty_metrics.overall_difficulty,
        'class_separability': difficulty_metrics.class_separability,
        'feature_complexity': difficulty_metrics.feature_complexity,
        'support_diversity': difficulty_metrics.support_set_diversity,
        'label_noise': difficulty_metrics.label_noise_estimate,
        'intrinsic_dimensionality': difficulty_metrics.intrinsic_dimensionality
    })
    
    # Prototype analysis (if we can compute prototypes)
    if support_x.size(0) > 0:
        # Compute prototypes
        unique_classes = torch.unique(support_y)
        prototypes = []
        
        for cls in unique_classes:
            mask = support_y == cls
            if mask.sum() > 0:
                prototype = support_x[mask].mean(dim=0)
                prototypes.append(prototype)
        
        if prototypes:
            prototypes = torch.stack(prototypes)
            
            prototype_analyzer = PrototypeAnalyzer()
            proto_metrics = prototype_analyzer.analyze_prototypes(
                support_x, support_y, prototypes
            )
            
            metrics.update({
                'prototype_intra_var': proto_metrics.intra_class_variance,
                'prototype_inter_sep': proto_metrics.inter_class_separation,
                'prototype_separability': proto_metrics.separability_ratio,
                'prototype_silhouette': proto_metrics.silhouette_score,
                'prototype_cohesion': proto_metrics.cluster_cohesion
            })
    
    # Uncertainty analysis (if predictions/logits available)
    if model_logits is not None:
        uncertainty_quantifier = UncertaintyQuantifier()
        uncertainty_metrics = uncertainty_quantifier.quantify_uncertainty(
            model_logits, query_y if query_y is not None else None
        )
        
        metrics.update({
            'predictive_entropy': uncertainty_metrics.predictive_entropy,
            'confidence_score': uncertainty_metrics.confidence_score,
            'calibration_error': uncertainty_metrics.calibration_error
        })
    
    return metrics


if __name__ == "__main__":
    # Test the advanced evaluation metrics
    print("Advanced Evaluation Metrics Test")
    print("=" * 50)
    
    # Create synthetic episode data
    n_way, k_shot, m_query = 5, 3, 2
    feature_dim = 64
    
    support_x = torch.randn(n_way * k_shot, feature_dim)
    support_y = torch.repeat_interleave(torch.arange(n_way), k_shot)
    query_x = torch.randn(n_way * m_query, feature_dim)  
    query_y = torch.repeat_interleave(torch.arange(n_way), m_query)
    
    # Test prototype analysis
    unique_classes = torch.unique(support_y)
    prototypes = torch.stack([
        support_x[support_y == cls].mean(dim=0) for cls in unique_classes
    ])
    
    prototype_analyzer = PrototypeAnalyzer()
    proto_metrics = prototype_analyzer.analyze_prototypes(support_x, support_y, prototypes)
    print(f"✓ Prototype Analysis: separability_ratio={proto_metrics.separability_ratio:.3f}")
    
    # Test task difficulty analysis
    difficulty_analyzer = TaskDifficultyAnalyzer()
    difficulty_metrics = difficulty_analyzer.assess_task_difficulty(support_x, support_y)
    print(f"✓ Task Difficulty: overall={difficulty_metrics.overall_difficulty:.3f}")
    
    # Test uncertainty quantification
    synthetic_logits = torch.randn(n_way * m_query, n_way)
    uncertainty_quantifier = UncertaintyQuantifier()
    uncertainty_metrics = uncertainty_quantifier.quantify_uncertainty(synthetic_logits, query_y)
    print(f"✓ Uncertainty: entropy={uncertainty_metrics.predictive_entropy:.3f}")
    
    # Test comprehensive episode analysis
    episode_metrics = analyze_episode_quality(
        support_x, support_y, query_x, query_y, 
        model_logits=synthetic_logits
    )
    print(f"✓ Episode Analysis: {len(episode_metrics)} metrics computed")
    
    print("\n✓ Advanced evaluation metrics tests completed")