"""
Statistical Comparison Visualization.

This module provides comprehensive statistical visualization tools for comparing
meta-learning algorithms with proper statistical analysis.

Classes:
    StatisticalComparison: Creates statistical comparison plots including
                          box plots, confidence intervals, and significance testing.
"""

from typing import Dict, List, Optional
import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from .config import VisualizationConfig
from ..evaluation.statistical_testing import StatisticalTestSuite


class StatisticalComparison:
    """Statistical comparison visualization.
    
    Provides comprehensive statistical visualization tools including distribution
    comparisons, confidence intervals, and significance testing matrices.
    
    Attributes:
        config: Visualization configuration settings
        statistical_suite: Statistical testing suite for computations
        logger: Logger instance for debugging
    """
    
    def __init__(self, config: VisualizationConfig):
        """Initialize the statistical comparison visualizer.
        
        Args:
            config: Visualization configuration
        """
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
        """Create comprehensive algorithm comparison plot.
        
        Creates a 2x2 subplot figure with box plots, violin plots,
        mean comparisons with confidence intervals, and significance heatmap.
        
        Args:
            results: Dict mapping algorithm names to performance values
            metric_name: Name of the metric being compared
            title: Overall plot title (auto-generated if None)
            save_path: Path to save figure (optional)
            
        Returns:
            Matplotlib figure with 4 comparison subplots
        """
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
        """Plot confidence intervals comparison.
        
        Creates a horizontal bar chart showing confidence intervals
        for each algorithm, sorted by performance.
        
        Args:
            results: Dict mapping algorithm names to performance values
            title: Plot title (auto-generated if None)
            save_path: Path to save figure (optional)
            
        Returns:
            Matplotlib figure with horizontal confidence interval bars
        """
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
        """Compute matrix of statistical significance p-values.
        
        Performs pairwise t-tests between all algorithms and returns
        a matrix of p-values for significance testing.
        
        Args:
            results: Dict mapping algorithm names to performance values
            
        Returns:
            Square matrix of p-values from pairwise comparisons
        """
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
    
    def plot_performance_distribution(
        self,
        results: Dict[str, List[float]],
        metric_name: str = "Accuracy",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot performance distribution histograms.
        
        Creates overlaid histograms showing the distribution of performance
        for each algorithm.
        
        Args:
            results: Dict mapping algorithm names to performance values
            metric_name: Name of the metric being plotted
            save_path: Path to save figure (optional)
            
        Returns:
            Matplotlib figure with overlaid histograms
        """
        plt.style.use(self.config.style)
        fig, ax = plt.subplots(figsize=self.config.figure_size, dpi=self.config.dpi)
        
        colors = plt.cm.get_cmap(self.config.color_palette)(np.linspace(0, 1, len(results)))
        
        for i, (algorithm, values) in enumerate(results.items()):
            ax.hist(values, alpha=0.6, label=algorithm, color=colors[i], 
                   bins=20, density=True)
        
        ax.set_xlabel(metric_name)
        ax.set_ylabel("Density")
        ax.set_title(f"{metric_name} Distribution by Algorithm")
        
        if self.config.grid:
            ax.grid(True, alpha=0.3)
        if self.config.legend:
            ax.legend()
        
        if self.config.tight_layout:
            plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.config.save_dpi, format=self.config.save_format)
            
        return fig