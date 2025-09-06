"""
Configuration for Performance Visualization.

This module provides configuration classes and settings for all visualization
components in the meta-learning framework.

Classes:
    VisualizationConfig: Main configuration for visualization settings
                        including figure properties, styling, and plot options.
"""

from typing import Tuple
from dataclasses import dataclass


@dataclass
class VisualizationConfig:
    """Configuration for performance visualization.
    
    Centralizes all visualization settings including figure properties,
    styling options, and plot-specific configurations.
    
    Attributes:
        figure_size: Default figure size in inches (width, height)
        dpi: Display resolution for plots
        style: Matplotlib style to use
        color_palette: Color palette name for plots
        save_format: Default format for saved figures
        save_dpi: Resolution for saved figures
        interactive: Whether to enable interactive plots
        show_confidence_intervals: Whether to show confidence intervals
        confidence_level: Confidence level for intervals (0.0-1.0)
        show_statistical_significance: Whether to mark statistical significance
        significance_level: P-value threshold for significance
        tight_layout: Whether to use tight layout
        grid: Whether to show grid lines
        legend: Whether to show legend
    """
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