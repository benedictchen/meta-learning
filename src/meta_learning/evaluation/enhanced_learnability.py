"""
💰 SUPPORT THIS RESEARCH - PLEASE DONATE! 💰

🙏 If this enhanced learnability analysis helps your research, please donate:
💳 PayPal: https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS
⭐ GitHub Sponsors: https://github.com/sponsors/benedictchen

👨‍💻 Author: Benedict Chen
💰 Donations: Help support this work! Buy me a coffee ☕, beer 🍺, lamborghini 🏎️, or private island 🏝️
💖 Please consider recurring donations to fully support continued research

Your support enables cutting-edge AI research for everyone! 🚀

Enhanced Learnability Analyzer with Hardness Metrics Integration
================================================================

This module enhances the base LearnabilityAnalyzer with advanced hardness metrics,
providing more sophisticated task difficulty assessment and complexity analysis.

💰 Please donate if this accelerates your research!
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Any, Union
import torch
import torch.nn.functional as F
import numpy as np
from ..core.episode import Episode
from .task_analysis import hardness_metric
from ..eval import LearnabilityAnalyzer


class EnhancedLearnabilityAnalyzer(LearnabilityAnalyzer):
    """
    Enhanced LearnabilityAnalyzer with integrated hardness metrics.
    
    Extends the base LearnabilityAnalyzer with:
    - Advanced hardness metric integration
    - Multi-scale difficulty assessment
    - Task complexity profiling
    - Comparative analysis capabilities
    - Curriculum learning support
    """
    
    def __init__(self):
        super().__init__()
        self.hardness_cache = {}
        self.complexity_profiles = {}
        
    def compute_enhanced_task_difficulty(self, episode: Episode) -> Dict[str, float]:
        """
        Compute enhanced task difficulty metrics with hardness integration.
        
        Combines base difficulty metrics with hardness_metric() for comprehensive
        assessment of task complexity and learnability.
        
        Args:
            episode: Episode to analyze
            
        Returns:
            Dictionary with enhanced difficulty metrics
        """
        # Get base difficulty metrics
        base_metrics = self.compute_task_difficulty(episode)
        
        # Compute hardness metric
        num_classes = len(torch.unique(episode.support_y))
        hardness_score = hardness_metric(episode, num_classes)
        
        # Enhanced metrics combining both approaches
        enhanced_metrics = {
            **base_metrics,
            'hardness_score': hardness_score,
            'composite_difficulty': self._compute_composite_difficulty(base_metrics, hardness_score),
            'separability_ratio': self._compute_separability_ratio(episode),
            'task_complexity_profile': self._compute_complexity_profile(episode, hardness_score)
        }
        
        return enhanced_metrics
    
    def _compute_composite_difficulty(self, base_metrics: Dict[str, float], hardness_score: float) -> float:
        """Compute composite difficulty score from multiple metrics."""
        base_difficulty = base_metrics.get('difficulty_score', 0.5)
        class_balance_penalty = 1.0 - base_metrics.get('class_balance', 0.5)
        
        # Weighted combination of difficulty indicators
        composite = (
            0.4 * hardness_score +
            0.3 * base_difficulty +
            0.2 * class_balance_penalty +
            0.1 * min(1.0, base_metrics.get('intra_class_variance', 0.0))
        )
        
        return min(1.0, max(0.0, composite))
    
    def _compute_separability_ratio(self, episode: Episode) -> float:
        """Compute ratio of inter-class to intra-class distances."""
        support_x = episode.support_x.view(episode.support_x.size(0), -1)
        support_y = episode.support_y
        
        # Compute class prototypes
        unique_classes = torch.unique(support_y)
        prototypes = []
        intra_class_distances = []
        
        for cls in unique_classes:
            mask = support_y == cls
            class_samples = support_x[mask]
            
            if len(class_samples) > 0:
                prototype = class_samples.mean(dim=0)
                prototypes.append(prototype)
                
                # Intra-class distances
                if len(class_samples) > 1:
                    intra_dists = torch.cdist(class_samples, class_samples[None, :].expand(1, -1, -1))
                    intra_class_distances.extend(intra_dists.flatten().tolist())
        
        if len(prototypes) < 2 or len(intra_class_distances) == 0:
            return 0.5  # Default moderate separability
        
        # Inter-class distances
        prototypes = torch.stack(prototypes)
        inter_dists = torch.pdist(prototypes)
        
        avg_inter = inter_dists.mean().item()
        avg_intra = np.mean([d for d in intra_class_distances if d > 0])
        
        if avg_intra == 0:
            return 1.0  # Perfect separability
        
        separability_ratio = avg_inter / (avg_intra + 1e-8)
        return min(1.0, separability_ratio / 5.0)  # Normalize to [0,1]
    
    def _compute_complexity_profile(self, episode: Episode, hardness_score: float) -> Dict[str, float]:
        """Compute detailed complexity profile for the task."""
        support_x = episode.support_x
        query_x = episode.query_x
        support_y = episode.support_y
        
        num_classes = len(torch.unique(support_y))
        shots_per_class = len(support_y) / num_classes
        
        # Dimensionality complexity
        feature_dim = support_x.view(support_x.size(0), -1).size(1)
        dim_complexity = min(1.0, feature_dim / 1000.0)  # Normalize by typical feature dimension
        
        # Support set complexity
        support_complexity = min(1.0, hardness_score * (1.0 + 1.0/shots_per_class))
        
        # Query-support alignment
        if len(query_x) > 0:
            support_flat = support_x.view(support_x.size(0), -1)
            query_flat = query_x.view(query_x.size(0), -1)
            
            # Compare distributions
            support_mean = support_flat.mean(dim=0)
            query_mean = query_flat.mean(dim=0)
            alignment_score = F.cosine_similarity(support_mean, query_mean, dim=0).item()
            distribution_gap = 1.0 - abs(alignment_score)
        else:
            distribution_gap = 0.0
        
        return {
            'dimensionality_complexity': dim_complexity,
            'support_complexity': support_complexity,
            'class_count_complexity': min(1.0, num_classes / 20.0),
            'shot_count_complexity': max(0.0, 1.0 - shots_per_class / 10.0),
            'distribution_gap': distribution_gap
        }
    
    def analyze_hardness_distribution(self, episodes: List[Episode]) -> Dict[str, Any]:
        """
        Analyze hardness distribution across multiple episodes.
        
        Args:
            episodes: List of episodes to analyze
            
        Returns:
            Dictionary with hardness distribution statistics
        """
        hardness_scores = []
        difficulty_scores = []
        separability_ratios = []
        
        for episode in episodes:
            enhanced_metrics = self.compute_enhanced_task_difficulty(episode)
            hardness_scores.append(enhanced_metrics['hardness_score'])
            difficulty_scores.append(enhanced_metrics['composite_difficulty'])
            separability_ratios.append(enhanced_metrics['separability_ratio'])
        
        return {
            'hardness_statistics': {
                'mean': np.mean(hardness_scores),
                'std': np.std(hardness_scores),
                'min': np.min(hardness_scores),
                'max': np.max(hardness_scores),
                'percentiles': {
                    '25': np.percentile(hardness_scores, 25),
                    '50': np.percentile(hardness_scores, 50),
                    '75': np.percentile(hardness_scores, 75)
                }
            },
            'difficulty_statistics': {
                'mean': np.mean(difficulty_scores),
                'std': np.std(difficulty_scores),
                'correlation_with_hardness': np.corrcoef(hardness_scores, difficulty_scores)[0, 1]
            },
            'separability_statistics': {
                'mean': np.mean(separability_ratios),
                'std': np.std(separability_ratios),
                'correlation_with_hardness': np.corrcoef(hardness_scores, separability_ratios)[0, 1]
            },
            'episode_count': len(episodes)
        }
    
    def generate_curriculum_ordering(self, episodes: List[Episode], strategy: str = 'gradual') -> List[int]:
        """
        Generate curriculum learning order based on task difficulty.
        
        Args:
            episodes: Episodes to order
            strategy: Curriculum strategy ('gradual', 'mixed', 'hard_first')
            
        Returns:
            List of episode indices in curriculum order
        """
        # Compute difficulty for all episodes
        difficulties = []
        for i, episode in enumerate(episodes):
            enhanced_metrics = self.compute_enhanced_task_difficulty(episode)
            difficulties.append((i, enhanced_metrics['composite_difficulty']))
        
        if strategy == 'gradual':
            # Easy to hard
            ordered = sorted(difficulties, key=lambda x: x[1])
        elif strategy == 'hard_first':
            # Hard to easy
            ordered = sorted(difficulties, key=lambda x: x[1], reverse=True)
        elif strategy == 'mixed':
            # Interleave easy and hard
            sorted_diffs = sorted(difficulties, key=lambda x: x[1])
            ordered = []
            for i in range(len(sorted_diffs)):
                if i % 2 == 0:
                    # Take from easy end
                    ordered.append(sorted_diffs[i // 2])
                else:
                    # Take from hard end
                    ordered.append(sorted_diffs[-(i // 2 + 1)])
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        return [idx for idx, _ in ordered]
    
    def compare_task_difficulties(self, episodes1: List[Episode], episodes2: List[Episode]) -> Dict[str, Any]:
        """
        Compare difficulty distributions between two sets of episodes.
        
        Args:
            episodes1: First set of episodes
            episodes2: Second set of episodes
            
        Returns:
            Dictionary with comparison statistics
        """
        analysis1 = self.analyze_hardness_distribution(episodes1)
        analysis2 = self.analyze_hardness_distribution(episodes2)
        
        # Statistical comparison
        hardness1 = [self.compute_enhanced_task_difficulty(ep)['hardness_score'] for ep in episodes1]
        hardness2 = [self.compute_enhanced_task_difficulty(ep)['hardness_score'] for ep in episodes2]
        
        # Simple t-test approximation
        mean1, std1, n1 = np.mean(hardness1), np.std(hardness1), len(hardness1)
        mean2, std2, n2 = np.mean(hardness2), np.std(hardness2), len(hardness2)
        
        pooled_se = np.sqrt((std1**2 / n1) + (std2**2 / n2))
        t_stat = (mean1 - mean2) / (pooled_se + 1e-8)
        
        return {
            'set1_analysis': analysis1,
            'set2_analysis': analysis2,
            'comparison': {
                'mean_difference': mean1 - mean2,
                't_statistic': t_stat,
                'effect_size': (mean1 - mean2) / (np.sqrt((std1**2 + std2**2) / 2) + 1e-8),
                'harder_set': 1 if mean1 > mean2 else 2
            }
        }
    
    def get_task_difficulty_recommendations(self, episode: Episode) -> Dict[str, Any]:
        """
        Get recommendations based on task difficulty analysis.
        
        Args:
            episode: Episode to analyze
            
        Returns:
            Dictionary with recommendations and reasoning
        """
        enhanced_metrics = self.compute_enhanced_task_difficulty(episode)
        
        hardness_score = enhanced_metrics['hardness_score']
        composite_difficulty = enhanced_metrics['composite_difficulty']
        separability_ratio = enhanced_metrics['separability_ratio']
        
        # Generate recommendations
        recommendations = {}
        
        if hardness_score > 0.7:
            recommendations['algorithm_preference'] = 'complex'
            recommendations['suggested_algorithms'] = ['maml', 'ttcs']
            recommendations['reasoning'] = 'High hardness score suggests need for adaptive algorithms'
        elif hardness_score < 0.3:
            recommendations['algorithm_preference'] = 'simple'
            recommendations['suggested_algorithms'] = ['ridge_regression', 'protonet']
            recommendations['reasoning'] = 'Low hardness score allows for simpler, efficient algorithms'
        else:
            recommendations['algorithm_preference'] = 'moderate'
            recommendations['suggested_algorithms'] = ['protonet', 'maml', 'ridge_regression']
            recommendations['reasoning'] = 'Moderate hardness allows for multiple algorithm choices'
        
        if separability_ratio < 0.3:
            recommendations['attention_required'] = 'high'
            recommendations['notes'] = 'Poor class separability - consider attention mechanisms'
        
        if composite_difficulty > 0.8:
            recommendations['training_strategy'] = 'careful'
            recommendations['suggested_passes'] = 'high'
            recommendations['notes'] = recommendations.get('notes', '') + ' Very difficult task - use more training passes'
        
        return {
            'metrics': enhanced_metrics,
            'recommendations': recommendations
        }


# TODO: Connect with TestTimeComputeScaler for adaptive pass allocation based on task hardness
# TODO: Integrate with algorithm selector for hardness-based algorithm recommendation
# TODO: Add curriculum learning integration with difficulty-based episode scheduling
# TODO: Support cross-task difficulty comparison for meta-dataset analysis