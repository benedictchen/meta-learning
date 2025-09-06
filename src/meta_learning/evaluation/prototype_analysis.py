# Advanced prototype quality analysis for meta-learning

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class PrototypeQualityMetrics:
    """Container for prototype quality metrics."""
    intra_class_variance: float
    inter_class_distance: float
    silhouette_score: float
    prototype_coherence: float
    class_separation_ratio: float
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for serialization."""
        return {
            "intra_class_variance": self.intra_class_variance,
            "inter_class_distance": self.inter_class_distance,
            "silhouette_score": self.silhouette_score,
            "prototype_coherence": self.prototype_coherence,
            "class_separation_ratio": self.class_separation_ratio
        }


class PrototypeAnalyzer:
    """Analyze quality and characteristics of prototypes in few-shot learning."""
    
    @staticmethod
    def compute_intra_class_variance(support_features: torch.Tensor,
                                   support_labels: torch.Tensor) -> float:
        """
        Compute average intra-class variance across all classes.
        
        Lower variance indicates more coherent prototypes.
        
        Args:
            support_features: Support set features [N, D]
            support_labels: Support set labels [N]
            
        Returns:
            Average intra-class variance
        """
        unique_labels = support_labels.unique()
        total_variance = 0.0
        
        for label in unique_labels:
            class_mask = support_labels == label
            class_features = support_features[class_mask]
            
            if len(class_features) > 1:
                # Compute variance along each dimension, then average
                class_variance = torch.var(class_features, dim=0).mean()
                total_variance += class_variance.item()
        
        return total_variance / len(unique_labels)
    
    @staticmethod
    def compute_inter_class_distance(prototypes: torch.Tensor,
                                   distance_metric: str = "euclidean") -> float:
        """
        Compute average pairwise distance between prototypes.
        
        Higher distance indicates better class separation.
        
        Args:
            prototypes: Class prototypes [n_way, D]
            distance_metric: Distance metric to use
            
        Returns:
            Average inter-prototype distance
        """
        if distance_metric == "euclidean":
            # Pairwise Euclidean distances
            distances = torch.cdist(prototypes, prototypes, p=2)
        elif distance_metric == "cosine":
            # Cosine similarity converted to distance
            similarities = F.cosine_similarity(
                prototypes.unsqueeze(1), 
                prototypes.unsqueeze(0), 
                dim=2
            )
            distances = 1 - similarities
        else:
            raise ValueError(f"Unknown distance metric: {distance_metric}")
        
        # Extract upper triangular part (excluding diagonal)
        mask = torch.triu(torch.ones_like(distances), diagonal=1).bool()
        inter_distances = distances[mask]
        
        return inter_distances.mean().item()
    
    @staticmethod
    def compute_silhouette_score(features: torch.Tensor,
                               labels: torch.Tensor) -> float:
        """
        Compute silhouette score for prototype quality assessment.
        
        Higher scores indicate better separated and cohesive clusters.
        
        Args:
            features: All features [N, D]
            labels: Corresponding labels [N]
            
        Returns:
            Average silhouette score
        """
        unique_labels = labels.unique()
        n_classes = len(unique_labels)
        
        if n_classes < 2:
            return 0.0
        
        silhouette_scores = []
        
        for i, point in enumerate(features):
            current_label = labels[i]
            
            # Intra-cluster distance (average distance to points in same cluster)
            same_cluster_mask = labels == current_label
            same_cluster_points = features[same_cluster_mask]
            
            if len(same_cluster_points) > 1:
                # Exclude the point itself
                other_same_cluster = torch.cat([
                    same_cluster_points[:i] if i > 0 else torch.empty(0, features.size(1)),
                    same_cluster_points[i+1:] if i < len(same_cluster_points)-1 else torch.empty(0, features.size(1))
                ])
                if len(other_same_cluster) > 0:
                    intra_distance = torch.cdist(point.unsqueeze(0), other_same_cluster, p=2).mean()
                else:
                    intra_distance = 0.0
            else:
                intra_distance = 0.0
            
            # Inter-cluster distance (minimum average distance to points in other clusters)
            min_inter_distance = float('inf')
            
            for other_label in unique_labels:
                if other_label != current_label:
                    other_cluster_mask = labels == other_label
                    other_cluster_points = features[other_cluster_mask]
                    
                    if len(other_cluster_points) > 0:
                        inter_distance = torch.cdist(point.unsqueeze(0), other_cluster_points, p=2).mean()
                        min_inter_distance = min(min_inter_distance, inter_distance.item())
            
            # Silhouette score for this point
            if min_inter_distance == float('inf'):
                silhouette = 0.0
            else:
                silhouette = (min_inter_distance - intra_distance) / max(min_inter_distance, intra_distance)
                silhouette_scores.append(silhouette)
        
        return np.mean(silhouette_scores) if silhouette_scores else 0.0
    
    @staticmethod
    def compute_prototype_coherence(support_features: torch.Tensor,
                                  support_labels: torch.Tensor,
                                  prototypes: torch.Tensor) -> float:
        """
        Compute coherence of prototypes (how well they represent their classes).
        
        Measures how close each prototype is to its class members.
        
        Args:
            support_features: Support set features [N, D]
            support_labels: Support set labels [N]
            prototypes: Class prototypes [n_way, D]
            
        Returns:
            Average prototype coherence score
        """
        unique_labels = support_labels.unique()
        coherence_scores = []
        
        for i, label in enumerate(unique_labels):
            class_mask = support_labels == label
            class_features = support_features[class_mask]
            prototype = prototypes[i]
            
            # Average distance from prototype to class members
            distances = torch.cdist(prototype.unsqueeze(0), class_features, p=2)
            avg_distance = distances.mean()
            
            # Convert to coherence score (lower distance = higher coherence)
            coherence = 1.0 / (1.0 + avg_distance.item())
            coherence_scores.append(coherence)
        
        return np.mean(coherence_scores)
    
    @staticmethod
    def compute_class_separation_ratio(prototypes: torch.Tensor,
                                     support_features: torch.Tensor,
                                     support_labels: torch.Tensor) -> float:
        """
        Compute ratio of inter-class to intra-class distances.
        
        Higher ratios indicate better class separation.
        
        Args:
            prototypes: Class prototypes [n_way, D]
            support_features: Support set features [N, D]
            support_labels: Support set labels [N]
            
        Returns:
            Class separation ratio
        """
        # Inter-class distances
        inter_class_dist = PrototypeAnalyzer.compute_inter_class_distance(prototypes)
        
        # Intra-class variance (proxy for intra-class distance)
        intra_class_var = PrototypeAnalyzer.compute_intra_class_variance(
            support_features, support_labels
        )
        
        # Avoid division by zero
        if intra_class_var == 0:
            return float('inf')
        
        return inter_class_dist / np.sqrt(intra_class_var)
    
    @staticmethod
    def analyze_prototype_quality(support_features: torch.Tensor,
                                support_labels: torch.Tensor,
                                prototypes: torch.Tensor) -> PrototypeQualityMetrics:
        """
        Comprehensive prototype quality analysis.
        
        Args:
            support_features: Support set features [N, D]
            support_labels: Support set labels [N] 
            prototypes: Class prototypes [n_way, D]
            
        Returns:
            Comprehensive prototype quality metrics
        """
        intra_var = PrototypeAnalyzer.compute_intra_class_variance(
            support_features, support_labels
        )
        
        inter_dist = PrototypeAnalyzer.compute_inter_class_distance(prototypes)
        
        silhouette = PrototypeAnalyzer.compute_silhouette_score(
            support_features, support_labels
        )
        
        coherence = PrototypeAnalyzer.compute_prototype_coherence(
            support_features, support_labels, prototypes
        )
        
        separation_ratio = PrototypeAnalyzer.compute_class_separation_ratio(
            prototypes, support_features, support_labels
        )
        
        return PrototypeQualityMetrics(
            intra_class_variance=intra_var,
            inter_class_distance=inter_dist,
            silhouette_score=silhouette,
            prototype_coherence=coherence,
            class_separation_ratio=separation_ratio
        )


class TaskDifficultyAnalyzer:
    """Analyze difficulty of few-shot learning tasks."""
    
    @staticmethod
    def compute_task_difficulty(support_features: torch.Tensor,
                              support_labels: torch.Tensor,
                              query_features: torch.Tensor) -> Dict[str, float]:
        """
        Compute various measures of task difficulty.
        
        Args:
            support_features: Support set features [N_support, D]
            support_labels: Support set labels [N_support]
            query_features: Query set features [N_query, D]
            
        Returns:
            Dictionary of difficulty metrics
        """
        # Compute prototypes
        unique_labels = support_labels.unique()
        prototypes = []
        
        for label in unique_labels:
            class_mask = support_labels == label
            class_features = support_features[class_mask]
            prototype = class_features.mean(dim=0)
            prototypes.append(prototype)
        
        prototypes = torch.stack(prototypes)
        
        # 1. Inter-class similarity (higher = more difficult)
        inter_similarities = F.cosine_similarity(
            prototypes.unsqueeze(1),
            prototypes.unsqueeze(0),
            dim=2
        )
        # Get off-diagonal elements
        mask = ~torch.eye(len(unique_labels), dtype=bool)
        max_similarity = inter_similarities[mask].max().item()
        
        # 2. Intra-class variance (higher = more difficult)
        analyzer = PrototypeAnalyzer()
        intra_variance = analyzer.compute_intra_class_variance(
            support_features, support_labels
        )
        
        # 3. Query-prototype distances (statistics)
        query_to_prototype_distances = torch.cdist(query_features, prototypes, p=2)
        min_distances = query_to_prototype_distances.min(dim=1)[0]
        
        return {
            "max_inter_class_similarity": max_similarity,
            "intra_class_variance": intra_variance,
            "min_query_distance_mean": min_distances.mean().item(),
            "min_query_distance_std": min_distances.std().item(),
            "difficulty_score": max_similarity * np.log(1 + intra_variance)
        }


def analyze_episode_quality(support_features: torch.Tensor,
                           support_labels: torch.Tensor,
                           query_features: torch.Tensor,
                           prototypes: Optional[torch.Tensor] = None) -> Dict[str, float]:
    """
    Comprehensive analysis of episode quality.
    
    Args:
        support_features: Support set features
        support_labels: Support set labels
        query_features: Query set features
        prototypes: Pre-computed prototypes (optional)
        
    Returns:
        Dictionary of quality metrics
    """
    # Compute prototypes if not provided
    if prototypes is None:
        unique_labels = support_labels.unique()
        prototype_list = []
        
        for label in unique_labels:
            class_mask = support_labels == label
            class_features = support_features[class_mask]
            prototype = class_features.mean(dim=0)
            prototype_list.append(prototype)
        
        prototypes = torch.stack(prototype_list)
    
    # Prototype quality analysis
    prototype_analyzer = PrototypeAnalyzer()
    prototype_metrics = prototype_analyzer.analyze_prototype_quality(
        support_features, support_labels, prototypes
    )
    
    # Task difficulty analysis
    difficulty_analyzer = TaskDifficultyAnalyzer()
    difficulty_metrics = difficulty_analyzer.compute_task_difficulty(
        support_features, support_labels, query_features
    )
    
    # Combine all metrics
    combined_metrics = prototype_metrics.to_dict()
    combined_metrics.update(difficulty_metrics)
    
    return combined_metrics


class PrototypeVisualizer:
    """Visualization tools for prototype analysis."""
    
    @staticmethod
    def plot_prototype_distribution(prototypes: torch.Tensor,
                                  labels: List[str] = None,
                                  save_path: str = None) -> Dict[str, any]:
        """
        Create visualization of prototype distribution using dimensionality reduction.
        
        Args:
            prototypes: Class prototypes [n_way, D]
            labels: Class labels for plotting
            save_path: Path to save plot (optional)
            
        Returns:
            Dictionary with plot data for external visualization
        """
        # Dimensionality reduction for visualization
        if prototypes.shape[1] > 2:
            # Use PCA for dimensionality reduction
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            prototypes_2d = pca.fit_transform(prototypes.detach().cpu().numpy())
        else:
            prototypes_2d = prototypes.detach().cpu().numpy()
        
        # Prepare plot data
        plot_data = {
            'coordinates': prototypes_2d,
            'labels': labels or [f'Class {i}' for i in range(len(prototypes))],
            'n_classes': len(prototypes),
            'explained_variance': None
        }
        
        if prototypes.shape[1] > 2:
            plot_data['explained_variance'] = pca.explained_variance_ratio_
            plot_data['total_explained_variance'] = pca.explained_variance_ratio_.sum()
        
        # Generate visualization instructions
        plot_data['plot_instructions'] = {
            'x': prototypes_2d[:, 0],
            'y': prototypes_2d[:, 1],
            'title': 'Prototype Distribution in 2D Space',
            'xlabel': f'PC1 ({plot_data.get("explained_variance", [0])[0]:.2%} variance)' if plot_data['explained_variance'] is not None else 'Dimension 1',
            'ylabel': f'PC2 ({plot_data.get("explained_variance", [0])[1]:.2%} variance)' if plot_data['explained_variance'] is not None else 'Dimension 2'
        }
        
        return plot_data
    
    @staticmethod
    def analyze_prototype_clusters(prototypes: torch.Tensor,
                                 support_features: torch.Tensor,
                                 support_labels: torch.Tensor) -> Dict[str, any]:
        """
        Analyze clustering properties of prototypes.
        
        Args:
            prototypes: Class prototypes [n_way, D]
            support_features: Support features [N, D]
            support_labels: Support labels [N]
            
        Returns:
            Clustering analysis results
        """
        # Compute distances between prototypes
        pairwise_distances = torch.cdist(prototypes, prototypes, p=2)
        
        # Find closest and farthest prototype pairs
        mask = torch.triu(torch.ones_like(pairwise_distances), diagonal=1).bool()
        distances = pairwise_distances[mask]
        
        min_distance = distances.min().item()
        max_distance = distances.max().item()
        mean_distance = distances.mean().item()
        std_distance = distances.std().item()
        
        # Analyze within-cluster vs between-cluster distances
        within_cluster_distances = []
        unique_labels = support_labels.unique()
        
        for i, label in enumerate(unique_labels):
            class_mask = support_labels == label
            class_features = support_features[class_mask]
            prototype = prototypes[i]
            
            # Distances from prototype to class members
            distances_to_prototype = torch.cdist(prototype.unsqueeze(0), class_features, p=2).squeeze()
            within_cluster_distances.extend(distances_to_prototype.tolist())
        
        within_cluster_mean = np.mean(within_cluster_distances)
        within_cluster_std = np.std(within_cluster_distances)
        
        # Cluster quality metrics
        cluster_analysis = {
            'prototype_distances': {
                'min': min_distance,
                'max': max_distance,
                'mean': mean_distance,
                'std': std_distance
            },
            'within_cluster': {
                'mean_distance': within_cluster_mean,
                'std_distance': within_cluster_std
            },
            'cluster_quality': {
                'separation_ratio': mean_distance / (within_cluster_mean + 1e-8),
                'compactness': 1.0 / (1.0 + within_cluster_mean),
                'separation': mean_distance
            }
        }
        
        return cluster_analysis


class PrototypeStabilityAnalyzer:
    """Analyze prototype stability over time/iterations."""
    
    def __init__(self, history_length: int = 100):
        """
        Initialize stability analyzer.
        
        Args:
            history_length: Number of iterations to keep in history
        """
        self.history_length = history_length
        self.prototype_history = []
        self.stability_metrics = []
    
    def add_prototypes(self, prototypes: torch.Tensor, iteration: int = None):
        """
        Add prototypes to history for stability analysis.
        
        Args:
            prototypes: Current prototypes [n_way, D]
            iteration: Optional iteration number
        """
        prototype_entry = {
            'prototypes': prototypes.detach().cpu().clone(),
            'iteration': iteration or len(self.prototype_history),
            'timestamp': len(self.prototype_history)
        }
        
        self.prototype_history.append(prototype_entry)
        
        # Keep only recent history
        if len(self.prototype_history) > self.history_length:
            self.prototype_history = self.prototype_history[-self.history_length:]
    
    def compute_stability_metrics(self) -> Dict[str, float]:
        """
        Compute prototype stability metrics over history.
        
        Returns:
            Dictionary of stability metrics
        """
        if len(self.prototype_history) < 2:
            return {'error': 'Need at least 2 prototype snapshots for stability analysis'}
        
        # Compute pairwise distances between consecutive prototypes
        consecutive_distances = []
        magnitude_changes = []
        
        for i in range(1, len(self.prototype_history)):
            prev_prototypes = self.prototype_history[i-1]['prototypes']
            curr_prototypes = self.prototype_history[i]['prototypes']
            
            # Ensure same shape (may vary with different episodes)
            if prev_prototypes.shape == curr_prototypes.shape:
                # Distance between corresponding prototypes
                distances = torch.norm(curr_prototypes - prev_prototypes, dim=1)
                consecutive_distances.extend(distances.tolist())
                
                # Magnitude changes
                prev_magnitudes = torch.norm(prev_prototypes, dim=1)
                curr_magnitudes = torch.norm(curr_prototypes, dim=1)
                magnitude_change = torch.abs(curr_magnitudes - prev_magnitudes)
                magnitude_changes.extend(magnitude_change.tolist())
        
        if not consecutive_distances:
            return {'error': 'No compatible prototype pairs found for stability analysis'}
        
        # Stability metrics
        stability_metrics = {
            'mean_displacement': np.mean(consecutive_distances),
            'std_displacement': np.std(consecutive_distances),
            'max_displacement': np.max(consecutive_distances),
            'stability_score': 1.0 / (1.0 + np.mean(consecutive_distances)),
            'magnitude_change_mean': np.mean(magnitude_changes),
            'magnitude_change_std': np.std(magnitude_changes),
            'snapshots_analyzed': len(self.prototype_history)
        }
        
        return stability_metrics
    
    def detect_instability_events(self, threshold_multiplier: float = 2.0) -> List[Dict]:
        """
        Detect episodes where prototypes changed significantly.
        
        Args:
            threshold_multiplier: Multiplier for standard deviation threshold
            
        Returns:
            List of instability events
        """
        stability_metrics = self.compute_stability_metrics()
        
        if 'error' in stability_metrics:
            return []
        
        threshold = stability_metrics['mean_displacement'] + \
                   threshold_multiplier * stability_metrics['std_displacement']
        
        instability_events = []
        
        for i in range(1, len(self.prototype_history)):
            prev_prototypes = self.prototype_history[i-1]['prototypes']
            curr_prototypes = self.prototype_history[i]['prototypes']
            
            if prev_prototypes.shape == curr_prototypes.shape:
                distances = torch.norm(curr_prototypes - prev_prototypes, dim=1)
                max_displacement = distances.max().item()
                
                if max_displacement > threshold:
                    instability_events.append({
                        'iteration': self.prototype_history[i]['iteration'],
                        'timestamp': self.prototype_history[i]['timestamp'],
                        'displacement': max_displacement,
                        'threshold': threshold,
                        'severity': max_displacement / threshold
                    })
        
        return instability_events
    
    def get_stability_trend(self) -> Dict[str, List[float]]:
        """
        Get trend of stability metrics over time.
        
        Returns:
            Dictionary with time series of stability metrics
        """
        if len(self.prototype_history) < 3:
            return {}
        
        displacements = []
        iterations = []
        
        for i in range(1, len(self.prototype_history)):
            prev_prototypes = self.prototype_history[i-1]['prototypes']
            curr_prototypes = self.prototype_history[i]['prototypes']
            
            if prev_prototypes.shape == curr_prototypes.shape:
                distances = torch.norm(curr_prototypes - prev_prototypes, dim=1)
                mean_displacement = distances.mean().item()
                
                displacements.append(mean_displacement)
                iterations.append(self.prototype_history[i]['iteration'])
        
        return {
            'iterations': iterations,
            'displacements': displacements,
            'smoothed_displacements': self._smooth_signal(displacements) if len(displacements) > 5 else displacements
        }
    
    def _smooth_signal(self, signal: List[float], window_size: int = 5) -> List[float]:
        """Apply simple moving average smoothing."""
        if len(signal) < window_size:
            return signal
        
        smoothed = []
        for i in range(len(signal)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(signal), i + window_size // 2 + 1)
            window_mean = np.mean(signal[start_idx:end_idx])
            smoothed.append(window_mean)
        
        return smoothed