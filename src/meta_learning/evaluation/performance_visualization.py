"""
Performance visualization and reporting for meta-learning evaluation.

This module provides comprehensive visualization capabilities for meta-learning experiments:
- Learning curves and convergence analysis
- Task difficulty visualizations
- Cross-validation results with confidence intervals
- Algorithm comparison plots
- Statistical significance visualization
- Interactive performance dashboards

Key classes:
- PerformanceVisualizer: Main visualization coordinator
- LearningCurveAnalyzer: Learning dynamics visualization
- StatisticalComparison: Statistical comparison plots
- TaskAnalysisPlots: Task-specific analysis visualization
- InteractiveDashboard: Interactive analysis dashboard

Research foundations:
- Hospedales et al. (2021): Meta-Learning in Neural Networks: A Survey
- Vanschoren (2018): Meta-Learning: A Survey  
- Best practices from ML evaluation literature
"""

from typing import Dict, List, Any, Tuple, Optional, Union, Callable
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, field
from collections import defaultdict
import pandas as pd
from pathlib import Path
import warnings
import logging

# Statistical visualization
from scipy import stats
from scipy.stats import t, norm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from .statistical_testing import StatisticalTestSuite
from .task_difficulty import TaskDifficultyProfile, DifficultyMetric
from ..core.episode import Episode


@dataclass
class VisualizationConfig:
    """Configuration for performance visualization."""
    figure_size: Tuple[int, int] = (10, 6)
    dpi: int = 150
    style: str = "seaborn-v0_8"
    color_palette: str = "husl"
    save_format: str = "png"
    save_dpi: int = 300
    interactive: bool = False
    
    # Plot-specific options
    show_confidence_intervals: bool = True
    confidence_level: float = 0.95
    show_statistical_significance: bool = True
    significance_level: float = 0.05
    
    # Layout options
    tight_layout: bool = True
    grid: bool = True
    legend: bool = True


class LearningCurveAnalyzer:
    """Analyze and visualize learning curves."""
    
    def __init__(self, config: VisualizationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def plot_learning_curves(
        self,
        results: Dict[str, List[Dict[str, float]]],
        metric: str = "accuracy",
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot learning curves for multiple algorithms/configurations.
        
        Args:
            results: Dict mapping algorithm names to list of episode results
            metric: Metric to plot
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        plt.style.use(self.config.style)
        fig, ax = plt.subplots(figsize=self.config.figure_size, dpi=self.config.dpi)
        
        colors = plt.cm.get_cmap(self.config.color_palette)(np.linspace(0, 1, len(results)))
        
        for i, (algorithm, episode_results) in enumerate(results.items()):
            # Extract metric values
            values = [result.get(metric, 0) for result in episode_results]
            episodes = list(range(1, len(values) + 1))
            
            # Compute running average and confidence intervals
            running_mean, running_ci = self._compute_running_statistics(values)
            
            # Plot main curve
            ax.plot(episodes, running_mean, label=algorithm, color=colors[i], linewidth=2)
            
            # Add confidence intervals
            if self.config.show_confidence_intervals and running_ci is not None:
                ci_lower, ci_upper = running_ci
                ax.fill_between(
                    episodes, ci_lower, ci_upper, 
                    alpha=0.2, color=colors[i]
                )
        
        # Formatting
        ax.set_xlabel("Episode")
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(title or f"Learning Curves: {metric.replace('_', ' ').title()}")
        
        if self.config.grid:
            ax.grid(True, alpha=0.3)
        if self.config.legend:
            ax.legend()
        
        if self.config.tight_layout:
            plt.tight_layout()
        
        # Save if requested
        if save_path:
            fig.savefig(save_path, dpi=self.config.save_dpi, format=self.config.save_format)
            
        return fig
    
    def plot_convergence_analysis(
        self,
        training_curves: Dict[str, List[float]],
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot convergence analysis with derivative and smoothing."""
        plt.style.use(self.config.style)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(self.config.figure_size[0], self.config.figure_size[1]*1.5))
        
        colors = plt.cm.get_cmap(self.config.color_palette)(np.linspace(0, 1, len(training_curves)))
        
        for i, (algorithm, curve) in enumerate(training_curves.items()):
            episodes = list(range(1, len(curve) + 1))
            
            # Original curve (smoothed)
            smoothed_curve = self._smooth_curve(curve)
            ax1.plot(episodes, smoothed_curve, label=algorithm, color=colors[i], linewidth=2)
            
            # Derivative (convergence rate)
            if len(curve) > 1:
                derivative = np.diff(smoothed_curve)
                ax2.plot(episodes[1:], derivative, label=f"{algorithm} (rate)", color=colors[i], linewidth=2)
        
        # Formatting
        ax1.set_ylabel("Performance")
        ax1.set_title(title or "Performance Convergence")
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Improvement Rate")
        ax2.set_title("Convergence Rate")
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        if self.config.tight_layout:
            plt.tight_layout()
            
        if save_path:
            fig.savefig(save_path, dpi=self.config.save_dpi, format=self.config.save_format)
            
        return fig
    
    def _compute_running_statistics(
        self, 
        values: List[float]
    ) -> Tuple[List[float], Optional[Tuple[List[float], List[float]]]]:
        """Compute running mean and confidence intervals."""
        running_mean = []
        ci_lower, ci_upper = [], []
        
        for i in range(1, len(values) + 1):
            window_values = values[:i]
            mean_val = np.mean(window_values)
            running_mean.append(mean_val)
            
            if i > 1 and self.config.show_confidence_intervals:
                std_val = np.std(window_values, ddof=1)
                n = len(window_values)
                
                # t-distribution confidence interval
                alpha = 1 - self.config.confidence_level
                t_val = t.ppf(1 - alpha/2, n-1)
                margin = t_val * std_val / np.sqrt(n)
                
                ci_lower.append(mean_val - margin)
                ci_upper.append(mean_val + margin)
            else:
                ci_lower.append(mean_val)
                ci_upper.append(mean_val)
        
        if self.config.show_confidence_intervals:
            return running_mean, (ci_lower, ci_upper)
        else:
            return running_mean, None
    
    def _smooth_curve(self, curve: List[float], window: int = 5) -> np.ndarray:
        """Apply moving average smoothing."""
        if len(curve) <= window:
            return np.array(curve)
        
        padded_curve = np.concatenate([
            np.full(window//2, curve[0]),
            curve,
            np.full(window//2, curve[-1])
        ])
        
        smoothed = np.convolve(padded_curve, np.ones(window)/window, mode='valid')
        return smoothed


class StatisticalComparison:
    """Statistical comparison visualization."""
    
    def __init__(self, config: VisualizationConfig):
        self.config = config
        self.statistical_suite = StatisticalTestSuite()
        self.logger = logging.getLogger(__name__)
        
    def plot_algorithm_comparison(
        self,
        results: Dict[str, List[float]],
        metric_name: str = "Accuracy",
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Create comprehensive algorithm comparison plot."""
        plt.style.use(self.config.style)
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        algorithms = list(results.keys())
        values = [results[alg] for alg in algorithms]
        
        # 1. Box plot comparison
        ax1.boxplot(values, labels=algorithms)
        ax1.set_title(f"{metric_name} Distribution by Algorithm")
        ax1.set_ylabel(metric_name)
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Violin plot for density
        parts = ax2.violinplot(values, positions=range(1, len(algorithms)+1))
        ax2.set_xticks(range(1, len(algorithms)+1))
        ax2.set_xticklabels(algorithms, rotation=45)
        ax2.set_title(f"{metric_name} Density Distribution")
        ax2.set_ylabel(metric_name)
        
        # 3. Mean with confidence intervals
        means = [np.mean(vals) for vals in values]
        cis = [self.statistical_suite.confidence_interval(vals, self.config.confidence_level) 
               for vals in values]
        
        x_pos = range(len(algorithms))
        ax3.bar(x_pos, means, alpha=0.7, capsize=5)
        
        # Add error bars
        ci_lowers = [ci[0] for ci in cis]
        ci_uppers = [ci[1] for ci in cis]
        errors = [[mean - ci_low for mean, ci_low in zip(means, ci_lowers)],
                 [ci_up - mean for mean, ci_up in zip(means, ci_uppers)]]
        
        ax3.errorbar(x_pos, means, yerr=errors, fmt='none', color='black', capsize=5)
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(algorithms, rotation=45)
        ax3.set_title(f"Mean {metric_name} with {int(self.config.confidence_level*100)}% CI")
        ax3.set_ylabel(metric_name)
        
        # 4. Statistical significance heatmap
        if len(algorithms) > 1:
            significance_matrix = self._compute_significance_matrix(results)
            im = ax4.imshow(significance_matrix, cmap='RdYlBu_r', vmin=0, vmax=0.05)
            ax4.set_xticks(range(len(algorithms)))
            ax4.set_yticks(range(len(algorithms)))
            ax4.set_xticklabels(algorithms, rotation=45)
            ax4.set_yticklabels(algorithms)
            ax4.set_title("Statistical Significance (p-values)")
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax4)
            cbar.set_label("p-value")
            
            # Add text annotations
            for i in range(len(algorithms)):
                for j in range(len(algorithms)):
                    if i != j:
                        text = f"{significance_matrix[i, j]:.3f}"
                        ax4.text(j, i, text, ha="center", va="center", 
                                color="white" if significance_matrix[i, j] < 0.025 else "black")
        
        if self.config.tight_layout:
            plt.tight_layout()
            
        if save_path:
            fig.savefig(save_path, dpi=self.config.save_dpi, format=self.config.save_format)
            
        return fig
    
    def plot_confidence_intervals(
        self,
        results: Dict[str, List[float]],
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot confidence intervals comparison."""
        plt.style.use(self.config.style)
        fig, ax = plt.subplots(figsize=self.config.figure_size, dpi=self.config.dpi)
        
        algorithms = list(results.keys())
        means = []
        ci_lowers = []
        ci_uppers = []
        
        for algorithm in algorithms:
            values = results[algorithm]
            mean_val = np.mean(values)
            ci_lower, ci_upper = self.statistical_suite.confidence_interval(
                values, self.config.confidence_level
            )
            
            means.append(mean_val)
            ci_lowers.append(ci_lower)
            ci_uppers.append(ci_upper)
        
        # Sort by mean performance
        sorted_indices = np.argsort(means)[::-1]  # Descending order
        sorted_algorithms = [algorithms[i] for i in sorted_indices]
        sorted_means = [means[i] for i in sorted_indices]
        sorted_ci_lowers = [ci_lowers[i] for i in sorted_indices]
        sorted_ci_uppers = [ci_uppers[i] for i in sorted_indices]
        
        # Create horizontal plot
        y_pos = np.arange(len(sorted_algorithms))
        
        # Plot confidence intervals
        for i, (mean_val, ci_low, ci_up) in enumerate(zip(sorted_means, sorted_ci_lowers, sorted_ci_uppers)):
            ax.barh(y_pos[i], ci_up - ci_low, left=ci_low, alpha=0.3, height=0.6)
            ax.plot(mean_val, y_pos[i], 'o', markersize=8, color='red')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_algorithms)
        ax.set_xlabel("Performance")
        ax.set_title(title or f"Algorithm Performance with {int(self.config.confidence_level*100)}% Confidence Intervals")
        
        if self.config.grid:
            ax.grid(True, alpha=0.3, axis='x')
        
        if self.config.tight_layout:
            plt.tight_layout()
            
        if save_path:
            fig.savefig(save_path, dpi=self.config.save_dpi, format=self.config.save_format)
            
        return fig
    
    def _compute_significance_matrix(self, results: Dict[str, List[float]]) -> np.ndarray:
        """Compute matrix of statistical significance p-values."""
        algorithms = list(results.keys())
        n = len(algorithms)
        significance_matrix = np.ones((n, n))
        
        for i, alg1 in enumerate(algorithms):
            for j, alg2 in enumerate(algorithms):
                if i != j:
                    # Perform t-test
                    _, p_value = stats.ttest_ind(results[alg1], results[alg2])
                    significance_matrix[i, j] = p_value
        
        return significance_matrix


class TaskAnalysisPlots:
    """Task-specific analysis visualization."""
    
    def __init__(self, config: VisualizationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def plot_difficulty_distribution(
        self,
        difficulty_profiles: List[TaskDifficultyProfile],
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot task difficulty distribution analysis."""
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
        
        # 3. Metric correlation heatmap
        metric_data = self._extract_metric_data(difficulty_profiles)
        if metric_data:
            corr_matrix = np.corrcoef([metric_data[metric] for metric in metric_data.keys()])
            
            im = ax3.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            metrics = list(metric_data.keys())
            ax3.set_xticks(range(len(metrics)))
            ax3.set_yticks(range(len(metrics)))
            ax3.set_xticklabels([m.replace('_', '\n') for m in metrics], rotation=45)
            ax3.set_yticklabels([m.replace('_', '\n') for m in metrics])
            ax3.set_title("Difficulty Metric Correlations")
            
            plt.colorbar(im, ax=ax3)
        
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
        """Plot task embeddings using dimensionality reduction."""
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
        
        # PCA projection
        pca = PCA(n_components=2)
        pca_embeddings = pca.fit_transform(embeddings)
        
        # t-SNE projection  
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
        tsne_embeddings = tsne.fit_transform(embeddings)
        
        # Color by difficulty if available
        if difficulty_profiles:
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
        """Extract metric data for correlation analysis."""
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


class PerformanceVisualizer:
    """Main coordinator for performance visualization."""
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.config = config or VisualizationConfig()
        self.curve_analyzer = LearningCurveAnalyzer(self.config)
        self.statistical_comparison = StatisticalComparison(self.config)
        self.task_analyzer = TaskAnalysisPlots(self.config)
        self.logger = logging.getLogger(__name__)
        
        # Set matplotlib style
        plt.style.use(self.config.style)
        sns.set_palette(self.config.color_palette)
        
    def create_comprehensive_report(
        self,
        experiment_results: Dict[str, Any],
        output_dir: str,
        report_name: str = "meta_learning_evaluation"
    ) -> Dict[str, str]:
        """
        Create a comprehensive visualization report.
        
        Args:
            experiment_results: Dictionary containing all experiment data
            output_dir: Directory to save visualizations
            report_name: Base name for report files
            
        Returns:
            Dictionary mapping plot names to file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        saved_plots = {}
        
        try:
            # 1. Learning curves (if available)
            if "learning_curves" in experiment_results:
                self.logger.info("Creating learning curves...")
                fig = self.curve_analyzer.plot_learning_curves(
                    experiment_results["learning_curves"],
                    title="Meta-Learning Algorithm Comparison"
                )
                
                save_path = output_path / f"{report_name}_learning_curves.{self.config.save_format}"
                fig.savefig(save_path, dpi=self.config.save_dpi, format=self.config.save_format)
                saved_plots["learning_curves"] = str(save_path)
                plt.close(fig)
            
            # 2. Algorithm comparison
            if "algorithm_results" in experiment_results:
                self.logger.info("Creating algorithm comparison...")
                fig = self.statistical_comparison.plot_algorithm_comparison(
                    experiment_results["algorithm_results"],
                    title="Algorithm Performance Comparison"
                )
                
                save_path = output_path / f"{report_name}_algorithm_comparison.{self.config.save_format}"
                fig.savefig(save_path, dpi=self.config.save_dpi, format=self.config.save_format)
                saved_plots["algorithm_comparison"] = str(save_path)
                plt.close(fig)
            
            # 3. Confidence intervals
            if "final_results" in experiment_results:
                self.logger.info("Creating confidence interval plot...")
                fig = self.statistical_comparison.plot_confidence_intervals(
                    experiment_results["final_results"],
                    title="Final Performance with Confidence Intervals"
                )
                
                save_path = output_path / f"{report_name}_confidence_intervals.{self.config.save_format}"
                fig.savefig(save_path, dpi=self.config.save_dpi, format=self.config.save_format)
                saved_plots["confidence_intervals"] = str(save_path)
                plt.close(fig)
            
            # 4. Task difficulty analysis (if available)
            if "difficulty_profiles" in experiment_results:
                self.logger.info("Creating task difficulty analysis...")
                fig = self.task_analyzer.plot_difficulty_distribution(
                    experiment_results["difficulty_profiles"],
                    title="Task Difficulty Analysis"
                )
                
                save_path = output_path / f"{report_name}_task_difficulty.{self.config.save_format}"
                fig.savefig(save_path, dpi=self.config.save_dpi, format=self.config.save_format)
                saved_plots["task_difficulty"] = str(save_path)
                plt.close(fig)
            
            # 5. Task embeddings (if episodes available)
            if "episodes" in experiment_results:
                self.logger.info("Creating task embeddings...")
                difficulty_profiles = experiment_results.get("difficulty_profiles")
                fig = self.task_analyzer.plot_task_embeddings(
                    experiment_results["episodes"],
                    difficulty_profiles,
                    title="Task Embedding Analysis"
                )
                
                save_path = output_path / f"{report_name}_task_embeddings.{self.config.save_format}"
                fig.savefig(save_path, dpi=self.config.save_dpi, format=self.config.save_format)
                saved_plots["task_embeddings"] = str(save_path)
                plt.close(fig)
            
            # 6. Convergence analysis (if training curves available)
            if "training_curves" in experiment_results:
                self.logger.info("Creating convergence analysis...")
                fig = self.curve_analyzer.plot_convergence_analysis(
                    experiment_results["training_curves"],
                    title="Training Convergence Analysis"
                )
                
                save_path = output_path / f"{report_name}_convergence.{self.config.save_format}"
                fig.savefig(save_path, dpi=self.config.save_dpi, format=self.config.save_format)
                saved_plots["convergence_analysis"] = str(save_path)
                plt.close(fig)
            
            self.logger.info(f"Saved {len(saved_plots)} visualizations to {output_dir}")
            
        except Exception as e:
            self.logger.error(f"Error creating visualization report: {e}")
            raise
        
        return saved_plots
    
    def create_summary_dashboard(
        self,
        experiment_results: Dict[str, Any],
        save_path: str
    ) -> plt.Figure:
        """Create a summary dashboard with key visualizations."""
        fig = plt.figure(figsize=(20, 16))
        
        # Create a complex subplot layout
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        try:
            # Algorithm comparison (top left, spanning 2x2)
            if "algorithm_results" in experiment_results:
                ax1 = fig.add_subplot(gs[0:2, 0:2])
                self._plot_algorithm_comparison_subplot(ax1, experiment_results["algorithm_results"])
            
            # Learning curves (top right, spanning 2x2)
            if "learning_curves" in experiment_results:
                ax2 = fig.add_subplot(gs[0:2, 2:4])
                self._plot_learning_curves_subplot(ax2, experiment_results["learning_curves"])
            
            # Task difficulty (bottom left)
            if "difficulty_profiles" in experiment_results:
                ax3 = fig.add_subplot(gs[2, 0:2])
                self._plot_difficulty_histogram_subplot(ax3, experiment_results["difficulty_profiles"])
            
            # Performance statistics table (bottom right)
            if "final_results" in experiment_results:
                ax4 = fig.add_subplot(gs[2, 2:4])
                self._create_statistics_table_subplot(ax4, experiment_results["final_results"])
            
            # Cross-validation results (bottom, spanning full width)
            if "cv_results" in experiment_results:
                ax5 = fig.add_subplot(gs[3, :])
                self._plot_cv_results_subplot(ax5, experiment_results["cv_results"])
            
            plt.suptitle("Meta-Learning Evaluation Dashboard", fontsize=16, y=0.98)
            
            if save_path:
                fig.savefig(save_path, dpi=self.config.save_dpi, format=self.config.save_format)
                
        except Exception as e:
            self.logger.error(f"Error creating dashboard: {e}")
            
        return fig
    
    def _plot_algorithm_comparison_subplot(self, ax, results: Dict[str, List[float]]):
        """Create algorithm comparison subplot."""
        algorithms = list(results.keys())
        values = [results[alg] for alg in algorithms]
        
        ax.boxplot(values, labels=algorithms)
        ax.set_title("Algorithm Performance")
        ax.set_ylabel("Accuracy")
        ax.tick_params(axis='x', rotation=45)
    
    def _plot_learning_curves_subplot(self, ax, curves: Dict[str, List[Dict[str, float]]]):
        """Create learning curves subplot."""
        for algorithm, episode_results in curves.items():
            accuracies = [result.get("accuracy", 0) for result in episode_results]
            episodes = list(range(1, len(accuracies) + 1))
            ax.plot(episodes, accuracies, label=algorithm, linewidth=2)
        
        ax.set_xlabel("Episode")
        ax.set_ylabel("Accuracy")
        ax.set_title("Learning Curves")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_difficulty_histogram_subplot(self, ax, profiles: List[TaskDifficultyProfile]):
        """Create difficulty histogram subplot."""
        difficulties = [p.overall_difficulty for p in profiles]
        ax.hist(difficulties, bins=15, alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(difficulties), color='red', linestyle='--', 
                  label=f'Mean: {np.mean(difficulties):.3f}')
        ax.set_xlabel("Task Difficulty")
        ax.set_ylabel("Frequency")
        ax.set_title("Task Difficulty Distribution")
        ax.legend()
    
    def _create_statistics_table_subplot(self, ax, results: Dict[str, List[float]]):
        """Create performance statistics table."""
        ax.axis('off')
        
        # Compute statistics
        stats_data = []
        for algorithm, values in results.items():
            stats_data.append([
                algorithm,
                f"{np.mean(values):.3f}",
                f"{np.std(values):.3f}",
                f"{np.min(values):.3f}",
                f"{np.max(values):.3f}"
            ])
        
        # Create table
        table = ax.table(
            cellText=stats_data,
            colLabels=["Algorithm", "Mean", "Std", "Min", "Max"],
            cellLoc="center",
            loc="center"
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        ax.set_title("Performance Statistics")
    
    def _plot_cv_results_subplot(self, ax, cv_results: Dict[str, Any]):
        """Create cross-validation results subplot."""
        if "cv_metrics" in cv_results:
            metrics = cv_results["cv_metrics"]
            metric_names = [k for k in metrics.keys() if k.endswith("_mean")]
            
            values = [metrics[name] for name in metric_names]
            labels = [name.replace("_mean", "").replace("_", " ").title() for name in metric_names]
            
            bars = ax.bar(range(len(values)), values, alpha=0.7)
            ax.set_xticks(range(len(values)))
            ax.set_xticklabels(labels, rotation=45)
            ax.set_title("Cross-Validation Results")
            ax.set_ylabel("Performance")
        
        ax.grid(True, alpha=0.3, axis='y')


def demonstrate_performance_visualization():
    """Demonstrate performance visualization capabilities."""
    # Create mock experiment results
    np.random.seed(42)
    
    experiment_results = {
        "algorithm_results": {
            "MAML": np.random.normal(0.75, 0.05, 50).tolist(),
            "ProtoNet": np.random.normal(0.72, 0.04, 50).tolist(),
            "MatchingNet": np.random.normal(0.68, 0.06, 50).tolist(),
        },
        "learning_curves": {
            "MAML": [{"accuracy": 0.5 + 0.25 * (1 - np.exp(-i/10)) + np.random.normal(0, 0.02)} 
                    for i in range(100)],
            "ProtoNet": [{"accuracy": 0.45 + 0.27 * (1 - np.exp(-i/8)) + np.random.normal(0, 0.02)} 
                        for i in range(100)],
        }
    }
    
    # Create visualizer
    config = VisualizationConfig(figure_size=(12, 8))
    visualizer = PerformanceVisualizer(config)
    
    # Create sample plots
    print("Creating learning curves...")
    fig1 = visualizer.curve_analyzer.plot_learning_curves(
        experiment_results["learning_curves"],
        title="Sample Learning Curves"
    )
    
    print("Creating algorithm comparison...")
    fig2 = visualizer.statistical_comparison.plot_algorithm_comparison(
        experiment_results["algorithm_results"],
        title="Sample Algorithm Comparison"
    )
    
    print("Creating confidence intervals...")
    fig3 = visualizer.statistical_comparison.plot_confidence_intervals(
        experiment_results["algorithm_results"],
        title="Sample Confidence Intervals"
    )
    
    plt.show()
    
    return visualizer


if __name__ == "__main__":
    # Run demonstration
    demonstrate_performance_visualization()