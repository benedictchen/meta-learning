"""
Task Analysis Visualization.

This module provides specialized visualization tools for analyzing task
characteristics, difficulty distributions, and task embeddings.

Classes:
    TaskAnalysisPlots: Creates task-specific analysis visualizations including
                      difficulty distributions and task embeddings.
"""

from typing import Dict, List, Optional
from collections import defaultdict
import logging
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from .config import VisualizationConfig
from ..core.episode import Episode
from ..shared.types import TaskDifficultyProfile


class TaskAnalysisPlots:
    """Task-specific analysis visualization.
    
    Provides specialized visualization tools for analyzing task characteristics,
    difficulty patterns, and task relationships through dimensionality reduction.
    
    Attributes:
        config: Visualization configuration settings
        logger: Logger instance for debugging
    """
    
    def __init__(self, config: VisualizationConfig):
        """Initialize the task analysis visualizer.
        
        Args:
            config: Visualization configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def plot_difficulty_distribution(
        self,
        difficulty_profiles: List[TaskDifficultyProfile],
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot task difficulty distribution analysis.
        
        Creates a comprehensive 2x2 subplot analysis of task difficulty including
        overall distribution, parameter-based grouping, metric correlations,
        and difficulty-performance relationships.
        
        Args:
            difficulty_profiles: List of task difficulty profiles
            title: Overall plot title (auto-generated if None)
            save_path: Path to save figure (optional)
            
        Returns:
            Matplotlib figure with 4 analysis subplots
        """
        plt.style.use(self.config.style)
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Extract difficulty scores
        overall_difficulties = [p.overall_difficulty for p in difficulty_profiles]
        
        # 1. Overall difficulty histogram
        ax1.hist(overall_difficulties, bins=20, alpha=0.7, edgecolor='black')
        ax1.set_xlabel("Overall Difficulty")
        ax1.set_ylabel("Frequency")
        ax1.set_title("Overall Task Difficulty Distribution")
        ax1.axvline(np.mean(overall_difficulties), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(overall_difficulties):.3f}')
        ax1.legend()
        
        # 2. Difficulty by task parameters
        n_ways = [p.metadata.get("n_way", 0) for p in difficulty_profiles]
        n_shots = [p.metadata.get("n_shot", 0) for p in difficulty_profiles]
        
        if len(set(n_ways)) > 1:
            for n_way in set(n_ways):
                mask = np.array(n_ways) == n_way
                difficulties_subset = np.array(overall_difficulties)[mask]
                ax2.hist(difficulties_subset, alpha=0.6, label=f"{n_way}-way", bins=15)
            
            ax2.set_xlabel("Difficulty")
            ax2.set_ylabel("Frequency")
            ax2.set_title("Difficulty by N-Way")
            ax2.legend()
        else:
            ax2.text(0.5, 0.5, "Insufficient N-Way\nvariability", 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title("Difficulty by N-Way")
        
        # 3. Metric correlation heatmap
        metric_data = self._extract_metric_data(difficulty_profiles)
        if metric_data and len(metric_data) > 1:
            corr_matrix = np.corrcoef([metric_data[metric] for metric in metric_data.keys()])
            
            im = ax3.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            metrics = list(metric_data.keys())
            ax3.set_xticks(range(len(metrics)))
            ax3.set_yticks(range(len(metrics)))
            ax3.set_xticklabels([m.replace('_', '\n') for m in metrics], rotation=45)
            ax3.set_yticklabels([m.replace('_', '\n') for m in metrics])
            ax3.set_title("Difficulty Metric Correlations")
            
            plt.colorbar(im, ax=ax3)
        else:
            ax3.text(0.5, 0.5, "Insufficient metric\ndata for correlation", 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title("Difficulty Metric Correlations")
        
        # 4. Difficulty vs Performance scatter (if available)
        if hasattr(difficulty_profiles[0], 'performance_score'):
            performance_scores = [p.performance_score for p in difficulty_profiles]
            ax4.scatter(overall_difficulties, performance_scores, alpha=0.6)
            ax4.set_xlabel("Task Difficulty")
            ax4.set_ylabel("Performance Score")
            ax4.set_title("Difficulty vs Performance")
            
            # Add trend line
            z = np.polyfit(overall_difficulties, performance_scores, 1)
            p = np.poly1d(z)
            ax4.plot(overall_difficulties, p(overall_difficulties), "r--", alpha=0.8)
        else:
            ax4.text(0.5, 0.5, "Performance data\nnot available", 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title("Performance Analysis")
        
        if self.config.tight_layout:
            plt.tight_layout()
            
        if save_path:
            fig.savefig(save_path, dpi=self.config.save_dpi, format=self.config.save_format)
            
        return fig
    
    def plot_task_embeddings(
        self,
        episodes: List[Episode],
        difficulty_profiles: Optional[List[TaskDifficultyProfile]] = None,
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot task embeddings using dimensionality reduction.
        
        Creates PCA and t-SNE visualizations of task embeddings, optionally
        colored by task difficulty to reveal task clustering patterns.
        
        Args:
            episodes: List of episodes to embed
            difficulty_profiles: Optional difficulty profiles for coloring
            title: Plot title (auto-generated if None)
            save_path: Path to save figure (optional)
            
        Returns:
            Matplotlib figure with PCA and t-SNE subplots
        """
        plt.style.use(self.config.style)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Create task embeddings (simple: mean and std of support set)
        embeddings = []
        for episode in episodes:
            support_mean = torch.mean(episode.support_x.flatten(1), dim=0)
            support_std = torch.std(episode.support_x.flatten(1), dim=0)
            embedding = torch.cat([support_mean, support_std])
            embeddings.append(embedding.cpu().numpy())
        
        embeddings = np.array(embeddings)
        
        # Handle edge case of insufficient data
        if len(embeddings) < 2:
            ax1.text(0.5, 0.5, "Insufficient data\nfor embedding", 
                    ha='center', va='center', transform=ax1.transAxes)
            ax2.text(0.5, 0.5, "Insufficient data\nfor embedding", 
                    ha='center', va='center', transform=ax2.transAxes)
            return fig
        
        # PCA projection
        pca = PCA(n_components=2)
        pca_embeddings = pca.fit_transform(embeddings)
        
        # t-SNE projection (with safety check for perplexity)
        perplexity = min(30, len(embeddings)-1, max(5, len(embeddings)//3))
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        tsne_embeddings = tsne.fit_transform(embeddings)
        
        # Color by difficulty if available
        if difficulty_profiles and len(difficulty_profiles) == len(episodes):
            difficulties = [p.overall_difficulty for p in difficulty_profiles]
            scatter_kwargs = {'c': difficulties, 'cmap': 'viridis', 's': 60, 'alpha': 0.7}
            
            # PCA plot
            scatter1 = ax1.scatter(pca_embeddings[:, 0], pca_embeddings[:, 1], **scatter_kwargs)
            ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
            ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
            ax1.set_title("PCA: Task Embeddings by Difficulty")
            plt.colorbar(scatter1, ax=ax1, label="Difficulty")
            
            # t-SNE plot
            scatter2 = ax2.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], **scatter_kwargs)
            ax2.set_xlabel("t-SNE 1")
            ax2.set_ylabel("t-SNE 2") 
            ax2.set_title("t-SNE: Task Embeddings by Difficulty")
            plt.colorbar(scatter2, ax=ax2, label="Difficulty")
        else:
            # Just plot embeddings without coloring
            ax1.scatter(pca_embeddings[:, 0], pca_embeddings[:, 1], alpha=0.7)
            ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
            ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
            ax1.set_title("PCA: Task Embeddings")
            
            ax2.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], alpha=0.7)
            ax2.set_xlabel("t-SNE 1")
            ax2.set_ylabel("t-SNE 2")
            ax2.set_title("t-SNE: Task Embeddings")
        
        if self.config.tight_layout:
            plt.tight_layout()
            
        if save_path:
            fig.savefig(save_path, dpi=self.config.save_dpi, format=self.config.save_format)
            
        return fig
    
    def _extract_metric_data(self, profiles: List[TaskDifficultyProfile]) -> Dict[str, List[float]]:
        """Extract metric data for correlation analysis.
        
        Extracts available difficulty metrics from profiles and filters out
        metrics with insufficient coverage.
        
        Args:
            profiles: List of difficulty profiles
            
        Returns:
            Dictionary mapping metric names to value lists
        """
        metric_data = defaultdict(list)
        
        # Get all available metrics
        all_metrics = set()
        for profile in profiles:
            all_metrics.update(profile.difficulty_scores.keys())
        
        # Extract data for each metric
        for metric in all_metrics:
            for profile in profiles:
                if metric in profile.difficulty_scores:
                    metric_data[metric.value].append(profile.difficulty_scores[metric])
        
        # Filter out metrics with insufficient data
        filtered_data = {}
        for metric, values in metric_data.items():
            if len(values) >= len(profiles) * 0.8:  # At least 80% coverage
                filtered_data[metric] = values
        
        return filtered_data
    
    def plot_task_parameter_analysis(
        self,
        difficulty_profiles: List[TaskDifficultyProfile],
        parameter: str = "n_shot",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot analysis by task parameter.
        
        Creates box plots showing difficulty distribution grouped by
        a specific task parameter.
        
        Args:
            difficulty_profiles: List of difficulty profiles
            parameter: Parameter to group by ('n_shot', 'n_way', etc.)
            save_path: Path to save figure (optional)
            
        Returns:
            Matplotlib figure with parameter analysis
        """
        plt.style.use(self.config.style)
        fig, ax = plt.subplots(figsize=self.config.figure_size, dpi=self.config.dpi)
        
        # Group by parameter
        parameter_groups = defaultdict(list)
        for profile in difficulty_profiles:
            param_value = profile.metadata.get(parameter, "unknown")
            parameter_groups[str(param_value)].append(profile.overall_difficulty)
        
        # Create box plot
        if len(parameter_groups) > 1:
            groups = list(parameter_groups.keys())
            values = [parameter_groups[group] for group in groups]
            
            ax.boxplot(values, labels=groups)
            ax.set_xlabel(parameter.replace('_', ' ').title())
            ax.set_ylabel("Task Difficulty")
            ax.set_title(f"Difficulty Distribution by {parameter.replace('_', ' ').title()}")
            
            if self.config.grid:
                ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, f"Insufficient {parameter}\nvariability", 
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"Difficulty by {parameter.replace('_', ' ').title()}")
        
        if self.config.tight_layout:
            plt.tight_layout()
            
        if save_path:
            fig.savefig(save_path, dpi=self.config.save_dpi, format=self.config.save_format)
            
        return fig