"""
💰 SUPPORT THIS RESEARCH - PLEASE DONATE! 💰

🙏 If this hardness-aware algorithm selection helps your research, please donate:
💳 PayPal: https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS
⭐ GitHub Sponsors: https://github.com/sponsors/benedictchen

👨‍💻 Author: Benedict Chen
💰 Donations: Help support this work! Buy me a coffee ☕, beer 🍺, lamborghini 🏎️, or private island 🏝️
💖 Please consider recurring donations to fully support continued research

Your support enables cutting-edge AI research for everyone! 🚀

Hardness-Aware Algorithm Selector
=================================

This module integrates task hardness metrics with algorithm selection,
providing intelligent algorithm recommendation based on task difficulty
and complexity analysis.

💰 Please donate if this accelerates your research!
"""

from __future__ import annotations
from typing import Dict, List, Any, Optional, Tuple
import torch
import numpy as np
from ..shared.types import Episode
from ..evaluation.enhanced_learnability import EnhancedLearnabilityAnalyzer
from .algorithm_registry import algorithm_registry, TaskDifficulty, AlgorithmType
from .algorithm_selection import AlgorithmSelector


class HardnessAwareSelector(AlgorithmSelector):
    """
    Algorithm selector enhanced with task hardness analysis.
    
    Extends AlgorithmSelector with:
    - Task hardness-based algorithm recommendation  
    - Difficulty-aware performance prediction
    - Adaptive selection strategies based on task complexity
    - Integration with EnhancedLearnabilityAnalyzer
    """
    
    def __init__(self, selection_strategy: str = "hardness_adaptive"):
        super().__init__(selection_strategy)
        self.learnability_analyzer = EnhancedLearnabilityAnalyzer()
        self.hardness_history = {}  # Track hardness vs performance correlations
        self.difficulty_thresholds = {
            'very_easy': 0.2,
            'easy': 0.4, 
            'medium': 0.6,
            'hard': 0.8,
            'very_hard': 1.0
        }
        
    def select_algorithm_with_hardness(self, episode: Episode) -> Dict[str, Any]:
        """
        Select algorithm with comprehensive hardness analysis.
        
        Args:
            episode: Episode to analyze and select algorithm for
            
        Returns:
            Dictionary with selected algorithm and analysis details
        """
        # Compute enhanced difficulty metrics
        difficulty_analysis = self.learnability_analyzer.compute_enhanced_task_difficulty(episode)
        hardness_score = difficulty_analysis['hardness_score']
        composite_difficulty = difficulty_analysis['composite_difficulty']
        
        # Map difficulty to TaskDifficulty enum
        task_difficulty = self._map_hardness_to_difficulty(hardness_score)
        
        # Get hardness-aware recommendations
        recommendations = self.learnability_analyzer.get_task_difficulty_recommendations(episode)
        
        # Select algorithm using enhanced strategy
        if self.selection_strategy == "hardness_adaptive":
            selected_algorithm = self._hardness_adaptive_selection(
                episode, hardness_score, composite_difficulty, recommendations
            )
        else:
            # Fallback to base selection
            selected_algorithm = self.select_algorithm(episode)
        
        return {
            'selected_algorithm': selected_algorithm,
            'hardness_score': hardness_score,
            'composite_difficulty': composite_difficulty,
            'task_difficulty': task_difficulty.value,
            'difficulty_analysis': difficulty_analysis,
            'recommendations': recommendations,
            'selection_reasoning': self._generate_hardness_reasoning(
                episode, selected_algorithm, hardness_score, recommendations
            )
        }
    
    def _map_hardness_to_difficulty(self, hardness_score: float) -> TaskDifficulty:
        """Map continuous hardness score to discrete difficulty level."""
        if hardness_score <= self.difficulty_thresholds['very_easy']:
            return TaskDifficulty.VERY_EASY
        elif hardness_score <= self.difficulty_thresholds['easy']:
            return TaskDifficulty.EASY
        elif hardness_score <= self.difficulty_thresholds['medium']:
            return TaskDifficulty.MEDIUM
        elif hardness_score <= self.difficulty_thresholds['hard']:
            return TaskDifficulty.HARD
        else:
            return TaskDifficulty.VERY_HARD
    
    def _hardness_adaptive_selection(
        self,
        episode: Episode,
        hardness_score: float,
        composite_difficulty: float,
        recommendations: Dict[str, Any]
    ) -> str:
        """Select algorithm using hardness-adaptive strategy."""
        
        # Get episode characteristics
        n_support = len(episode.support_y)
        n_classes = len(torch.unique(episode.support_y))
        n_shots_per_class = n_support // n_classes
        
        # High hardness (>0.6) - prefer adaptive algorithms  
        if hardness_score > 0.6:
            # Very hard tasks benefit from test-time adaptation
            if composite_difficulty > 0.7 and n_shots_per_class <= 5:
                return 'ttcs'
            # Hard tasks with sufficient data can use MAML
            elif n_shots_per_class >= 3:
                return 'maml'
            else:
                return 'matching_networks'  # Attention helps with hard tasks
        
        # Medium hardness (0.35-0.6) - balanced approach  
        elif hardness_score > 0.35:
            # Sufficient data allows ridge regression
            if n_shots_per_class >= 5 and n_classes <= 10:
                return 'ridge_regression'
            # Otherwise use prototypical networks
            else:
                return 'protonet'
        
        # Low hardness (<=0.35) - prefer efficient algorithms
        else:
            # Easy tasks can use simple algorithms
            if n_shots_per_class >= 3:
                return 'ridge_regression'
            else:
                return 'protonet'
    
    def update_hardness_performance(
        self, 
        algorithm: str, 
        episode: Episode, 
        accuracy: float,
        hardness_score: Optional[float] = None
    ):
        """Update performance tracking with hardness correlation."""
        # Update base performance tracking (handle unregistered algorithms)
        if algorithm not in self.algorithm_performance:
            self.algorithm_performance[algorithm] = []
        
        self.update_performance(algorithm, episode, accuracy)
        
        # Compute hardness if not provided
        if hardness_score is None:
            difficulty_analysis = self.learnability_analyzer.compute_enhanced_task_difficulty(episode)
            hardness_score = difficulty_analysis['hardness_score']
        
        # Track hardness-performance correlation
        if algorithm not in self.hardness_history:
            self.hardness_history[algorithm] = []
        
        self.hardness_history[algorithm].append({
            'hardness_score': hardness_score,
            'accuracy': accuracy,
            'n_support': len(episode.support_y),
            'n_classes': len(torch.unique(episode.support_y))
        })
        
        # Keep recent history
        if len(self.hardness_history[algorithm]) > 100:
            self.hardness_history[algorithm] = self.hardness_history[algorithm][-50:]
    
    def get_hardness_performance_analysis(self, algorithm: str) -> Dict[str, Any]:
        """Get hardness-performance correlation analysis for an algorithm."""
        if algorithm not in self.hardness_history or not self.hardness_history[algorithm]:
            return {'status': 'no_data'}
        
        history = self.hardness_history[algorithm]
        hardness_scores = [h['hardness_score'] for h in history]
        accuracies = [h['accuracy'] for h in history]
        
        # Compute correlation
        if len(hardness_scores) > 1:
            correlation = np.corrcoef(hardness_scores, accuracies)[0, 1]
        else:
            correlation = 0.0
        
        # Analyze performance across difficulty ranges
        easy_tasks = [h for h in history if h['hardness_score'] <= 0.3]
        medium_tasks = [h for h in history if 0.3 < h['hardness_score'] <= 0.7]
        hard_tasks = [h for h in history if h['hardness_score'] > 0.7]
        
        analysis = {
            'correlation': correlation,
            'total_episodes': len(history),
            'mean_hardness': np.mean(hardness_scores),
            'mean_accuracy': np.mean(accuracies)
        }
        
        if easy_tasks:
            analysis['easy_performance'] = {
                'count': len(easy_tasks),
                'mean_accuracy': np.mean([t['accuracy'] for t in easy_tasks]),
                'mean_hardness': np.mean([t['hardness_score'] for t in easy_tasks])
            }
        
        if medium_tasks:
            analysis['medium_performance'] = {
                'count': len(medium_tasks),
                'mean_accuracy': np.mean([t['accuracy'] for t in medium_tasks]),
                'mean_hardness': np.mean([t['hardness_score'] for t in medium_tasks])
            }
        
        if hard_tasks:
            analysis['hard_performance'] = {
                'count': len(hard_tasks),
                'mean_accuracy': np.mean([t['accuracy'] for t in hard_tasks]),
                'mean_hardness': np.mean([t['hardness_score'] for t in hard_tasks])
            }
        
        return analysis
    
    def recommend_algorithms_by_hardness(
        self,
        episode: Episode,
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Recommend algorithms with hardness-based ranking.
        
        Args:
            episode: Episode to analyze
            top_k: Number of algorithms to recommend
            
        Returns:
            List of algorithm recommendations with hardness analysis
        """
        # Get difficulty analysis
        difficulty_analysis = self.learnability_analyzer.compute_enhanced_task_difficulty(episode)
        hardness_score = difficulty_analysis['hardness_score']
        
        # Get all suitable algorithms from registry
        n_support = len(episode.support_y)
        n_classes = len(torch.unique(episode.support_y))
        n_shots_per_class = n_support // n_classes
        task_difficulty = self._map_hardness_to_difficulty(hardness_score)
        
        suitable_algorithms = self.registry.get_suitable_algorithms(
            n_shot=n_shots_per_class,
            n_classes=n_classes,
            task_difficulty=task_difficulty,
            max_algorithms=top_k * 2
        )
        
        # Score algorithms based on hardness and performance history
        algorithm_scores = []
        
        for metadata in suitable_algorithms:
            base_score = metadata.selection_priority
            
            # Adjust score based on hardness-specific performance
            hardness_analysis = self.get_hardness_performance_analysis(metadata.name)
            
            if hardness_analysis.get('status') != 'no_data':
                # Factor in historical hardness performance
                hardness_factor = 1.0
                
                if hardness_score <= 0.3 and 'easy_performance' in hardness_analysis:
                    # Boost score for algorithms that perform well on easy tasks
                    easy_acc = hardness_analysis['easy_performance']['mean_accuracy']
                    hardness_factor = 0.8 + 0.4 * easy_acc  # [0.8, 1.2]
                elif hardness_score > 0.7 and 'hard_performance' in hardness_analysis:
                    # Boost score for algorithms that perform well on hard tasks
                    hard_acc = hardness_analysis['hard_performance']['mean_accuracy']
                    hardness_factor = 0.8 + 0.4 * hard_acc  # [0.8, 1.2]
                
                adjusted_score = base_score * hardness_factor
            else:
                adjusted_score = base_score
            
            algorithm_scores.append({
                'algorithm': metadata.name,
                'score': adjusted_score,
                'base_score': base_score,
                'hardness_analysis': hardness_analysis,
                'algorithm_type': metadata.algorithm_type.value,
                'description': metadata.description
            })
        
        # Sort by adjusted score and return top-k
        algorithm_scores.sort(key=lambda x: x['score'], reverse=True)
        return algorithm_scores[:top_k]
    
    def _generate_hardness_reasoning(
        self,
        episode: Episode,
        selected_algorithm: str,
        hardness_score: float,
        recommendations: Dict[str, Any]
    ) -> str:
        """Generate human-readable reasoning for hardness-aware selection."""
        n_classes = len(torch.unique(episode.support_y))
        n_support = len(episode.support_y)
        n_shots_per_class = n_support // n_classes
        
        reasoning = f"Selected {selected_algorithm} for {n_classes}-way {n_shots_per_class}-shot task "
        reasoning += f"with hardness score {hardness_score:.3f}. "
        
        if hardness_score > 0.7:
            reasoning += "High task hardness requires adaptive algorithms with strong generalization. "
        elif hardness_score > 0.3:
            reasoning += "Medium task hardness allows for balanced algorithm choice. "
        else:
            reasoning += "Low task hardness enables efficient, simple algorithms. "
        
        # Add algorithm-specific reasoning
        if selected_algorithm == 'ttcs':
            reasoning += "TTCS provides test-time adaptation for challenging scenarios."
        elif selected_algorithm == 'maml':
            reasoning += "MAML offers gradient-based adaptation suitable for this difficulty level."
        elif selected_algorithm == 'ridge_regression':
            reasoning += "Ridge regression provides stable, efficient solution for this task complexity."
        elif selected_algorithm == 'protonet':
            reasoning += "Prototypical networks offer robust metric-based learning."
        elif selected_algorithm == 'matching_networks':
            reasoning += "Matching networks leverage attention for complex pattern recognition."
        
        return reasoning
    
    def generate_hardness_curriculum(
        self,
        episodes: List[Episode],
        strategy: str = 'adaptive_gradual'
    ) -> List[Tuple[int, str]]:
        """
        Generate curriculum with hardness-aware algorithm assignment.
        
        Args:
            episodes: Episodes to order and assign algorithms
            strategy: Curriculum strategy
            
        Returns:
            List of (episode_index, recommended_algorithm) tuples
        """
        # Get curriculum ordering from learnability analyzer
        episode_ordering = self.learnability_analyzer.generate_curriculum_ordering(
            episodes, strategy='gradual' if 'gradual' in strategy else 'mixed'
        )
        
        curriculum = []
        
        for episode_idx in episode_ordering:
            episode = episodes[episode_idx]
            
            if strategy == 'adaptive_gradual':
                # Use hardness-aware selection for each episode
                selection_result = self.select_algorithm_with_hardness(episode)
                recommended_algorithm = selection_result['selected_algorithm']
            elif strategy == 'progressive_complexity':
                # Start with simple algorithms, progress to complex
                hardness_analysis = self.learnability_analyzer.compute_enhanced_task_difficulty(episode)
                hardness_score = hardness_analysis['hardness_score']
                
                if hardness_score <= 0.3:
                    recommended_algorithm = 'ridge_regression'
                elif hardness_score <= 0.6:
                    recommended_algorithm = 'protonet'
                else:
                    recommended_algorithm = 'maml'
            else:
                # Fallback to basic selection
                recommended_algorithm = self.select_algorithm(episode)
            
            curriculum.append((episode_idx, recommended_algorithm))
        
        return curriculum


# TODO: Connect with TestTimeComputeScaler for adaptive pass allocation based on task hardness
# TODO: Add performance prediction based on hardness-algorithm compatibility matrices  
# TODO: Integrate with curriculum learning framework for hardness-progressive training
# TODO: Support multi-objective optimization balancing accuracy and computational efficiency