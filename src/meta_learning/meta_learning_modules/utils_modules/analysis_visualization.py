"""
Analysis and Visualization Functions for Meta-Learning 📊📈🎨
============================================================

🎯 **ELI5 Explanation**:
Think of this like an art teacher who helps you create beautiful charts and graphs to show off your AI results!
Just like how artists use different colors and styles to tell stories, this module helps you:
- 📊 **Create Pretty Charts**: Turn boring numbers into colorful, easy-to-understand graphs
- 📈 **Show Comparisons**: Like before/after photos, but for AI algorithm performance  
- 🎨 **Make Reports**: Generate professional-looking research reports with all the right charts
- 💾 **Save Everything**: Keep your results organized like a photo album

📊 **Visualization Types Available**:
```
Raw Results:           Visualization:           Beautiful Charts:
┌─────────────┐       ┌───────────────┐       ┌─────────────────┐
│ Accuracy:   │       │               │       │  📊 Bar Charts  │
│ [0.85, 0.87,│  ──→  │ Visualization │  ──→  │  📈 Line Plots  │
│  0.83, 0.89]│       │ Engine        │       │  📦 Box Plots   │
└─────────────┘       └───────────────┘       │  🔥 Heat Maps   │
                                              └─────────────────┘
```

🎨 **Chart Types for Different Insights**:
- 📊 **Bar Charts**: Compare different algorithms side-by-side
- 📈 **Line Plots**: Show learning progress over time  
- 📦 **Box Plots**: Display performance distributions and outliers
- 🔥 **Heat Maps**: Visualize performance across different conditions
- 📐 **Confidence Intervals**: Show uncertainty and reliability of results

🔬 **Research-Quality Visualizations**:
Follows best practices for scientific visualization from:
- **Edward Tufte**: "The Visual Display of Quantitative Information" 
- **Matplotlib Best Practices**: Clear, publication-ready figures
- **Seaborn Statistical Plots**: Professional statistical visualizations

Author: Benedict Chen (benedict@benedictchen.com)

This module contains visualization and I/O functions for meta-learning results,
providing comprehensive analysis tools for research.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any
import json
import pickle
from pathlib import Path
import logging
from .statistical_evaluation import compute_confidence_interval

logger = logging.getLogger(__name__)


def visualize_meta_learning_results(
    results: Dict[str, List[float]],
    title: str = "Meta-Learning Results",
    save_path: Optional[str] = None
):
    """
    Create comprehensive visualizations for meta-learning results.
    
    Args:
        results: Dictionary with algorithm names as keys and accuracy lists as values
        title: Plot title
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(title, fontsize=16)
    
    # 1. Accuracy comparison (box plot)
    ax1 = axes[0, 0]
    data_for_boxplot = [results[alg] for alg in results.keys()]
    labels = list(results.keys())
    
    ax1.boxplot(data_for_boxplot, labels=labels)
    ax1.set_title("Accuracy Distribution")
    ax1.set_ylabel("Accuracy")
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. Learning curves
    ax2 = axes[0, 1]
    for alg_name, accuracies in results.items():
        # Compute running average
        running_avg = np.cumsum(accuracies) / np.arange(1, len(accuracies) + 1)
        ax2.plot(running_avg, label=alg_name, alpha=0.7)
    
    ax2.set_title("Learning Curves (Running Average)")
    ax2.set_xlabel("Task Number")
    ax2.set_ylabel("Cumulative Average Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Statistical comparison
    ax3 = axes[1, 0]
    means = [np.mean(results[alg]) for alg in results.keys()]
    stds = [np.std(results[alg]) for alg in results.keys()]
    
    ax3.barh(labels, means, xerr=stds, capsize=5)
    ax3.set_title("Mean Accuracy ± Standard Deviation")
    ax3.set_xlabel("Accuracy")
    
    # 4. Confidence intervals
    ax4 = axes[1, 1]
    ci_data = {}
    for alg_name, accuracies in results.items():
        mean_val, lower, upper = compute_confidence_interval(accuracies)
        ci_data[alg_name] = (mean_val, lower, upper)
    
    alg_names = list(ci_data.keys())
    means = [ci_data[alg][0] for alg in alg_names]
    lowers = [ci_data[alg][1] for alg in alg_names]
    uppers = [ci_data[alg][2] for alg in alg_names]
    
    y_pos = np.arange(len(alg_names))
    ax4.barh(y_pos, means, xerr=[np.array(means) - np.array(lowers), 
                                  np.array(uppers) - np.array(means)],
             capsize=5)
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(alg_names)
    ax4.set_title("95% Confidence Intervals")
    ax4.set_xlabel("Accuracy")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved visualization to {save_path}")
    
    plt.show()


def save_meta_learning_results(
    results: Dict[str, Any],
    filepath: str,
    format: str = "json"
):
    """
    Save meta-learning results to file.
    
    Args:
        results: Results dictionary to save
        filepath: Path to save file
        format: File format ("json", "pickle")
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if format == "json":
        # Convert torch tensors to lists for JSON serialization
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, torch.Tensor):
                serializable_results[key] = value.tolist()
            elif isinstance(value, np.ndarray):
                serializable_results[key] = value.tolist()
            else:
                serializable_results[key] = value
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
    
    elif format == "pickle":
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)
    
    logger.info(f"Saved results to {filepath}")


def load_meta_learning_results(filepath: str, format: str = "auto") -> Dict[str, Any]:
    """
    Load meta-learning results from file.
    
    Args:
        filepath: Path to load from
        format: File format ("json", "pickle", "auto")
        
    Returns:
        Loaded results dictionary
    """
    filepath = Path(filepath)
    
    if format == "auto":
        format = filepath.suffix[1:]  # Remove the dot
    
    if format == "json":
        with open(filepath, 'r') as f:
            results = json.load(f)
    elif format in ["pickle", "pkl"]:
        with open(filepath, 'rb') as f:
            results = pickle.load(f)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    logger.info(f"Loaded results from {filepath}")
    return results