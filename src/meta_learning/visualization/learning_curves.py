"""
Learning Curve Analysis and Visualization.

This module provides tools for analyzing and visualizing learning curves
in meta-learning experiments, including convergence analysis and statistical metrics.

Classes:
    LearningCurveAnalyzer: Creates learning curve plots with confidence intervals
                          and convergence analysis.
"""

from typing import Dict, List, Optional, Tuple
import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t

from .config import VisualizationConfig


class LearningCurveAnalyzer:
    """Analyze and visualize learning curves.
    
    Provides comprehensive learning curve visualization with running statistics,
    confidence intervals, and convergence analysis for meta-learning experiments.
    
    Attributes:
        config: Visualization configuration settings
        logger: Logger instance for debugging
    """
    
    def __init__(self, config: VisualizationConfig):
        """Initialize the learning curve analyzer.
        
        Args:
            config: Visualization configuration
        """
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
        
        Creates a line plot showing performance over episodes with optional
        confidence intervals and statistical smoothing.
        
        Args:
            results: Dict mapping algorithm names to list of episode results
            metric: Metric to plot (e.g., 'accuracy', 'loss')
            title: Plot title (auto-generated if None)
            save_path: Path to save figure (optional)
            
        Returns:
            Matplotlib figure object
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
        """Plot convergence analysis with derivative and smoothing.
        
        Creates a dual-panel plot showing both the smoothed performance curves
        and their derivatives (convergence rates).
        
        Args:
            training_curves: Dict mapping algorithm names to performance curves
            title: Plot title (auto-generated if None)
            save_path: Path to save figure (optional)
            
        Returns:
            Matplotlib figure with dual panels
        """
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
        """Compute running mean and confidence intervals.
        
        Calculates running statistics with proper statistical confidence intervals
        using t-distribution for small sample sizes.
        
        Args:
            values: List of performance values
            
        Returns:
            Tuple of (running_means, (ci_lower, ci_upper)) or (running_means, None)
        """
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
        """Apply moving average smoothing.
        
        Smooths the curve using a moving average window with proper edge handling.
        
        Args:
            curve: Raw performance curve
            window: Smoothing window size
            
        Returns:
            Smoothed curve as numpy array
        """
        if len(curve) <= window:
            return np.array(curve)
        
        padded_curve = np.concatenate([
            np.full(window//2, curve[0]),
            curve,
            np.full(window//2, curve[-1])
        ])
        
        smoothed = np.convolve(padded_curve, np.ones(window)/window, mode='valid')
        return smoothed