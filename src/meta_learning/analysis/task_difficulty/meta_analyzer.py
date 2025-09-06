"""
Meta-learning specific difficulty measures.

This module provides difficulty assessment measures specifically designed for
meta-learning scenarios, focusing on adaptation effectiveness, generalization
gaps, and few-shot transferability.

Key measures:
- Adaptation difficulty (how hard it is to adapt from support to query)
- Generalization gap (difference between support and query performance)
- Few-shot transferability (how well few-shot samples represent classes)
- Cross-task transfer difficulty

Research foundations:
- Finn et al. (2017): Model-Agnostic Meta-Learning for Fast Adaptation
- Hospedales et al. (2021): Meta-Learning in Neural Networks: A Survey
- Wang et al. (2020): Generalizing from a Few Examples: A Survey on Few-Shot Learning
"""

from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import copy


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
        
        This metric captures the core challenge of meta-learning: how effectively
        can a model adapt to new tasks using only the support set, and how well
        does this adaptation generalize to the query set.
        
        Args:
            model: Neural network model
            support_x: Support set features [N_support, ...]
            support_y: Support set labels [N_support]
            query_x: Query set features [N_query, ...]
            query_y: Query set labels [N_query]
            n_adaptation_steps: Number of adaptation steps
            
        Returns:
            Difficulty score (0=easy adaptation, 1=difficult adaptation)
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
        # Use sigmoid to bound the values and handle negative improvements
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
        
        A large gap indicates overfitting to the support set, which suggests
        the task is difficult or the adaptation process is not optimal.
        
        Args:
            model: Neural network model
            support_x: Support set features [N_support, ...]
            support_y: Support set labels [N_support]
            query_x: Query set features [N_query, ...]
            query_y: Query set labels [N_query]
            n_adaptation_steps: Number of adaptation steps
            
        Returns:
            Generalization gap (0=no gap, 1=large gap)
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
        
        In few-shot learning, we have very limited examples per class. This metric
        assesses how representative these few examples are of the entire class
        distribution.
        
        Args:
            X: All features for the class [N, D]
            y: All labels [N]
            n_shot: Number of shots per class
            
        Returns:
            Difficulty score (0=representative samples, 1=poor samples)
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
    
    def cross_task_transfer_difficulty(
        self,
        source_model: nn.Module,
        source_support_x: torch.Tensor,
        source_support_y: torch.Tensor,
        target_support_x: torch.Tensor,
        target_support_y: torch.Tensor,
        target_query_x: torch.Tensor,
        target_query_y: torch.Tensor,
        n_adaptation_steps: int = 5
    ) -> float:
        """
        Measure difficulty of transferring from one task to another.
        
        This assesses how well a model adapted to one task can be further
        adapted to a related task, which is important for continual meta-learning.
        
        Args:
            source_model: Model pre-adapted to source task
            source_support_x: Source task support features
            source_support_y: Source task support labels  
            target_support_x: Target task support features
            target_support_y: Target task support labels
            target_query_x: Target task query features
            target_query_y: Target task query labels
            n_adaptation_steps: Adaptation steps for target task
            
        Returns:
            Transfer difficulty score (0=easy transfer, 1=difficult transfer)
        """
        # First adapt to source task
        source_adapted_model = self._clone_model(source_model)
        source_adapted_model.train()
        
        optimizer = torch.optim.SGD(source_adapted_model.parameters(), lr=0.01)
        for _ in range(n_adaptation_steps):
            optimizer.zero_grad()
            logits = source_adapted_model(source_support_x)
            loss = F.cross_entropy(logits, source_support_y)
            loss.backward()
            optimizer.step()
        
        # Test direct transfer (no additional adaptation)
        source_adapted_model.eval()
        with torch.no_grad():
            direct_transfer_logits = source_adapted_model(target_query_x)
            direct_transfer_acc = (direct_transfer_logits.argmax(-1) == target_query_y).float().mean().item()
        
        # Now adapt to target task
        target_adapted_model = self._clone_model(source_adapted_model)
        target_adapted_model.train()
        
        optimizer = torch.optim.SGD(target_adapted_model.parameters(), lr=0.01)
        for _ in range(n_adaptation_steps):
            optimizer.zero_grad()
            logits = target_adapted_model(target_support_x)
            loss = F.cross_entropy(logits, target_support_y)
            loss.backward()
            optimizer.step()
        
        # Test adapted performance
        target_adapted_model.eval()
        with torch.no_grad():
            adapted_logits = target_adapted_model(target_query_x)
            adapted_acc = (adapted_logits.argmax(-1) == target_query_y).float().mean().item()
        
        # Compare with baseline (adapting from original model)
        baseline_model = self._clone_model(source_model)
        baseline_model.train()
        
        optimizer = torch.optim.SGD(baseline_model.parameters(), lr=0.01)
        for _ in range(n_adaptation_steps):
            optimizer.zero_grad()
            logits = baseline_model(target_support_x)
            loss = F.cross_entropy(logits, target_support_y)
            loss.backward()
            optimizer.step()
        
        baseline_model.eval()
        with torch.no_grad():
            baseline_logits = baseline_model(target_query_x)
            baseline_acc = (baseline_logits.argmax(-1) == target_query_y).float().mean().item()
        
        # Transfer benefit: how much better is transfer compared to learning from scratch
        transfer_benefit = adapted_acc - baseline_acc
        
        # Transfer difficulty: how much adaptation was needed after direct transfer
        adaptation_needed = adapted_acc - direct_transfer_acc
        
        # Combine measures: high difficulty if little transfer benefit and much adaptation needed
        if adapted_acc == 0:
            return 1.0
            
        transfer_efficiency = transfer_benefit / (adaptation_needed + 1e-8)
        difficulty = 1 / (1 + np.exp(transfer_efficiency - 0.1))
        
        return float(difficulty)
    
    def support_query_domain_shift(
        self,
        support_x: torch.Tensor,
        query_x: torch.Tensor,
        metric: str = 'mmd'
    ) -> float:
        """
        Measure domain shift between support and query sets.
        
        Large domain shifts make few-shot learning more difficult as the
        support set becomes less representative of the query distribution.
        
        Args:
            support_x: Support set features [N_support, D]
            query_x: Query set features [N_query, D]
            metric: Distance metric ('mmd', 'wasserstein', 'kl')
            
        Returns:
            Domain shift score (0=no shift, 1=large shift)
        """
        if metric == 'mmd':
            return self._maximum_mean_discrepancy(support_x, query_x)
        elif metric == 'wasserstein':
            return self._wasserstein_distance(support_x, query_x)
        else:
            # Simple statistical distance
            return self._statistical_distance(support_x, query_x)
    
    def _maximum_mean_discrepancy(
        self, 
        X: torch.Tensor, 
        Y: torch.Tensor,
        kernel: str = 'rbf',
        sigma: float = 1.0
    ) -> float:
        """Compute MMD between two distributions."""
        def kernel_matrix(X, Y, kernel, sigma):
            if kernel == 'rbf':
                # RBF kernel
                XX = torch.sum(X**2, dim=1, keepdim=True)
                YY = torch.sum(Y**2, dim=1, keepdim=True)
                XY = torch.mm(X, Y.t())
                
                K = torch.exp(-(XX - 2*XY + YY.t()) / (2 * sigma**2))
                return K
            else:
                # Linear kernel
                return torch.mm(X, Y.t())
        
        XX = kernel_matrix(X, X, kernel, sigma)
        YY = kernel_matrix(Y, Y, kernel, sigma) 
        XY = kernel_matrix(X, Y, kernel, sigma)
        
        mmd = XX.mean() + YY.mean() - 2 * XY.mean()
        return max(0.0, mmd.item())
    
    def _wasserstein_distance(self, X: torch.Tensor, Y: torch.Tensor) -> float:
        """Approximate Wasserstein distance using sliced Wasserstein."""
        n_projections = 50
        projections = torch.randn(X.shape[1], n_projections)
        projections = F.normalize(projections, dim=0)
        
        X_proj = torch.mm(X, projections)
        Y_proj = torch.mm(Y, projections)
        
        distances = []
        for i in range(n_projections):
            X_sorted, _ = torch.sort(X_proj[:, i])
            Y_sorted, _ = torch.sort(Y_proj[:, i])
            
            # Pad to same length
            min_len = min(len(X_sorted), len(Y_sorted))
            X_sorted = X_sorted[:min_len]
            Y_sorted = Y_sorted[:min_len]
            
            dist = torch.mean(torch.abs(X_sorted - Y_sorted)).item()
            distances.append(dist)
        
        return np.mean(distances)
    
    def _statistical_distance(self, X: torch.Tensor, Y: torch.Tensor) -> float:
        """Simple statistical distance between feature distributions."""
        X_mean = torch.mean(X, dim=0)
        Y_mean = torch.mean(Y, dim=0)
        
        X_std = torch.std(X, dim=0) + 1e-8
        Y_std = torch.std(Y, dim=0) + 1e-8
        
        # Normalized mean difference
        mean_diff = torch.norm(X_mean - Y_mean) / (torch.norm(X_mean) + torch.norm(Y_mean) + 1e-8)
        
        # Relative std difference
        std_diff = torch.mean(torch.abs(X_std - Y_std) / (X_std + Y_std + 1e-8))
        
        # Combine measures
        distance = 0.7 * mean_diff.item() + 0.3 * std_diff.item()
        return min(1.0, distance)
    
    def _compute_representativeness(
        self,
        few_shot_samples: torch.Tensor,
        all_samples: torch.Tensor
    ) -> float:
        """
        Compute how well few-shot samples represent the full distribution.
        
        Uses multiple measures including centroid distance and spread similarity.
        """
        # Centroid similarity
        few_shot_mean = torch.mean(few_shot_samples, dim=0)
        all_samples_mean = torch.mean(all_samples, dim=0)
        
        centroid_distance = torch.norm(few_shot_mean - all_samples_mean).item()
        centroid_similarity = 1 / (1 + centroid_distance)
        
        # Spread similarity
        few_shot_std = torch.std(few_shot_samples, dim=0).mean().item()
        all_samples_std = torch.std(all_samples, dim=0).mean().item()
        
        if all_samples_std == 0:
            spread_similarity = 1.0 if few_shot_std == 0 else 0.0
        else:
            spread_ratio = min(few_shot_std / all_samples_std, all_samples_std / few_shot_std)
            spread_similarity = spread_ratio
        
        # Coverage similarity (how well few-shot samples cover the range)
        if len(few_shot_samples) > 1 and len(all_samples) > 1:
            few_shot_range = torch.max(few_shot_samples, dim=0)[0] - torch.min(few_shot_samples, dim=0)[0]
            all_samples_range = torch.max(all_samples, dim=0)[0] - torch.min(all_samples, dim=0)[0]
            
            range_similarity = torch.mean(
                torch.minimum(few_shot_range, all_samples_range) / 
                (torch.maximum(few_shot_range, all_samples_range) + 1e-8)
            ).item()
        else:
            range_similarity = 0.5
        
        # Weighted combination
        representativeness = (
            0.4 * centroid_similarity + 
            0.3 * spread_similarity + 
            0.3 * range_similarity
        )
        
        return representativeness
    
    def compute_all_meta_learning_measures(
        self,
        model: nn.Module,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        query_x: torch.Tensor,
        query_y: torch.Tensor,
        **kwargs
    ) -> Dict[str, float]:
        """
        Compute all meta-learning specific measures.
        
        Args:
            model: Neural network model
            support_x: Support set features
            support_y: Support set labels
            query_x: Query set features  
            query_y: Query set labels
            **kwargs: Additional arguments
            
        Returns:
            Dictionary of meta-learning measures
        """
        measures = {}
        
        try:
            measures['adaptation_difficulty'] = self.adaptation_difficulty(
                model, support_x, support_y, query_x, query_y
            )
        except Exception as e:
            self.logger.warning(f"Error computing adaptation difficulty: {e}")
            measures['adaptation_difficulty'] = 0.5
            
        try:
            measures['generalization_gap'] = self.generalization_gap(
                model, support_x, support_y, query_x, query_y
            )
        except Exception as e:
            self.logger.warning(f"Error computing generalization gap: {e}")
            measures['generalization_gap'] = 0.5
            
        try:
            # Combine support and query for transferability analysis
            all_x = torch.cat([support_x, query_x], dim=0)
            all_y = torch.cat([support_y, query_y], dim=0)
            n_shot = len(support_y) // len(torch.unique(support_y))
            
            measures['few_shot_transferability'] = self.few_shot_transferability(
                all_x, all_y, n_shot
            )
        except Exception as e:
            self.logger.warning(f"Error computing transferability: {e}")
            measures['few_shot_transferability'] = 0.5
            
        try:
            measures['support_query_domain_shift'] = self.support_query_domain_shift(
                support_x, query_x
            )
        except Exception as e:
            self.logger.warning(f"Error computing domain shift: {e}")
            measures['support_query_domain_shift'] = 0.5
            
        return measures
    
    def _clone_model(self, model: nn.Module) -> nn.Module:
        """Create a deep copy of the model."""
        return copy.deepcopy(model)


def analyze_meta_learning_difficulty(
    model: nn.Module,
    support_x: torch.Tensor,
    support_y: torch.Tensor, 
    query_x: torch.Tensor,
    query_y: torch.Tensor,
    **kwargs
) -> Dict[str, float]:
    """
    Convenience function to analyze meta-learning difficulty.
    
    Args:
        model: Neural network model
        support_x: Support set features
        support_y: Support set labels
        query_x: Query set features
        query_y: Query set labels
        **kwargs: Additional arguments
        
    Returns:
        Dictionary of meta-learning difficulty measures
    """
    analyzer = MetaLearningSpecificAnalyzer()
    return analyzer.compute_all_meta_learning_measures(
        model, support_x, support_y, query_x, query_y, **kwargs
    )


if __name__ == "__main__":
    # Simple demonstration
    import torch.nn as nn
    
    torch.manual_seed(42)
    
    # Create a simple model
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(), 
        nn.Linear(20, 3)
    )
    
    # Create synthetic few-shot data
    n_way, n_shot, n_query = 3, 5, 15
    
    support_x = torch.randn(n_way * n_shot, 10)
    support_y = torch.repeat_interleave(torch.arange(n_way), n_shot)
    
    query_x = torch.randn(n_way * n_query, 10) 
    query_y = torch.repeat_interleave(torch.arange(n_way), n_query)
    
    analyzer = MetaLearningSpecificAnalyzer()
    
    print("Meta-Learning Difficulty Analysis:")
    measures = analyzer.compute_all_meta_learning_measures(
        model, support_x, support_y, query_x, query_y
    )
    
    for measure, value in measures.items():
        print(f"  {measure}: {value:.3f}")
    
    print(f"\nOverall meta-learning difficulty: {np.mean(list(measures.values())):.3f}")