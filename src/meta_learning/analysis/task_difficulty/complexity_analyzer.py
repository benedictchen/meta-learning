"""
Statistical and geometrical complexity measures for task difficulty assessment.

This module implements various statistical measures to assess the inherent complexity
of classification tasks, focusing on class separability and feature efficiency.

Key measures:
- Fisher's discriminant ratio (between/within class variance)
- Class separability using silhouette score  
- Neighborhood separability analysis
- Feature efficiency via PCA analysis

Research foundations:
- Fisher, R.A. (1936): The use of multiple measurements in taxonomic problems
- Ho & Basu (2002): Data Complexity measures in pattern recognition
- Rousseeuw (1987): Silhouettes: A graphical aid to the interpretation of cluster analysis
"""

from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import torch
import logging
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score


class ComplexityAnalyzer:
    """Statistical and geometrical complexity measures."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def fisher_discriminant_ratio(self, X: torch.Tensor, y: torch.Tensor) -> float:
        """
        Fisher's discriminant ratio: between-class variance / within-class variance.
        
        Higher values indicate easier separation between classes. This metric
        measures how well classes are separated relative to their internal variance.
        
        Args:
            X: Feature tensor [N, D]
            y: Label tensor [N]
            
        Returns:
            Difficulty score (0=easy, 1=difficult) - inverted from original ratio
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
        
        The silhouette score measures how similar objects are to their own cluster
        compared to other clusters. Higher scores indicate better separated clusters.
        
        Args:
            X: Feature tensor [N, D] 
            y: Label tensor [N]
            
        Returns:
            Difficulty score (0=easy, 1=difficult)
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
        
        Analyzes k-nearest neighbors for each point to determine if they belong
        to the same class. Higher misclassification rate indicates higher difficulty.
        
        Args:
            X: Feature tensor [N, D]
            y: Label tensor [N] 
            k: Number of nearest neighbors to analyze
            
        Returns:
            Difficulty score (0=easy, 1=difficult)
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
        
        Analyzes class separability in principal component space to determine
        how well the most important features discriminate between classes.
        
        Args:
            X: Feature tensor [N, D]
            y: Label tensor [N]
            
        Returns:
            Difficulty score (0=easy, 1=difficult)
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
        """
        Measure separability in a single component.
        
        Args:
            component: Single component values [N, 1]
            y: Labels [N]
            
        Returns:
            Separability score (0=poor, 1=perfect)
        """
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
    
    def boundary_complexity(self, X: torch.Tensor, y: torch.Tensor) -> float:
        """
        Estimate decision boundary complexity using density-based measures.
        
        This is a simplified version that estimates boundary complexity by
        analyzing the distribution of points near class boundaries.
        
        Args:
            X: Feature tensor [N, D]
            y: Label tensor [N]
            
        Returns:
            Difficulty score (0=simple boundary, 1=complex boundary)
        """
        try:
            X_np = X.cpu().numpy()
            y_np = y.cpu().numpy()
            classes = np.unique(y_np)
            
            if len(classes) < 2:
                return 0.0
            
            # Find points near class boundaries using k-NN
            k = min(5, len(X_np) // 4)
            nbrs = NearestNeighbors(n_neighbors=k).fit(X_np)
            indices = nbrs.kneighbors(X_np, return_distance=False)
            
            boundary_points = 0
            total_points = len(X_np)
            
            for i in range(total_points):
                neighbor_labels = y_np[indices[i]]
                current_label = y_np[i]
                
                # Point is near boundary if neighbors have different labels
                if np.any(neighbor_labels != current_label):
                    boundary_points += 1
            
            # Higher proportion of boundary points = more complex boundary
            complexity = boundary_points / total_points
            return complexity
            
        except Exception as e:
            self.logger.warning(f"Error computing boundary complexity: {e}")
            return 0.5
    
    def compute_all_complexity_measures(
        self, 
        X: torch.Tensor, 
        y: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute all complexity measures for the given data.
        
        Args:
            X: Feature tensor [N, D]
            y: Label tensor [N]
            
        Returns:
            Dictionary of complexity measures
        """
        measures = {}
        
        try:
            measures['fisher_discriminant_ratio'] = self.fisher_discriminant_ratio(X, y)
        except Exception as e:
            self.logger.warning(f"Error computing Fisher ratio: {e}")
            measures['fisher_discriminant_ratio'] = 0.5
            
        try:
            measures['class_separability'] = self.class_separability(X, y)
        except Exception as e:
            self.logger.warning(f"Error computing class separability: {e}")
            measures['class_separability'] = 0.5
            
        try:
            measures['neighborhood_separability'] = self.neighborhood_separability(X, y)
        except Exception as e:
            self.logger.warning(f"Error computing neighborhood separability: {e}")
            measures['neighborhood_separability'] = 0.5
            
        try:
            measures['feature_efficiency'] = self.feature_efficiency(X, y)
        except Exception as e:
            self.logger.warning(f"Error computing feature efficiency: {e}")
            measures['feature_efficiency'] = 0.5
            
        try:
            measures['boundary_complexity'] = self.boundary_complexity(X, y)
        except Exception as e:
            self.logger.warning(f"Error computing boundary complexity: {e}")
            measures['boundary_complexity'] = 0.5
            
        return measures


def analyze_data_complexity(X: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
    """
    Convenience function to analyze data complexity.
    
    Args:
        X: Feature tensor [N, D]
        y: Label tensor [N]
        
    Returns:
        Dictionary of complexity measures
    """
    analyzer = ComplexityAnalyzer()
    return analyzer.compute_all_complexity_measures(X, y)


if __name__ == "__main__":
    # Simple demonstration
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create synthetic data with varying complexity
    n_samples, n_features = 100, 10
    n_classes = 3
    
    # Easy case: well-separated classes
    easy_X = torch.randn(n_samples, n_features)
    easy_y = torch.repeat_interleave(torch.arange(n_classes), n_samples // n_classes)
    easy_X += easy_y.unsqueeze(1) * 3  # Add class-specific offsets
    
    # Hard case: overlapping classes  
    hard_X = torch.randn(n_samples, n_features) * 2
    hard_y = torch.repeat_interleave(torch.arange(n_classes), n_samples // n_classes)
    hard_X += easy_y.unsqueeze(1) * 0.5  # Small class-specific offsets
    
    analyzer = ComplexityAnalyzer()
    
    print("Easy data complexity:")
    easy_measures = analyzer.compute_all_complexity_measures(easy_X, easy_y)
    for measure, value in easy_measures.items():
        print(f"  {measure}: {value:.3f}")
    
    print("\nHard data complexity:")
    hard_measures = analyzer.compute_all_complexity_measures(hard_X, hard_y)
    for measure, value in hard_measures.items():
        print(f"  {measure}: {value:.3f}")