#!/usr/bin/env python3
"""
Model Calibration Analysis for Meta-Learning

Implements comprehensive calibration analysis for meta-learning models including:
- Reliability diagrams and calibration curves
- Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)
- Temperature scaling and Platt scaling calibration
- Brier score decomposition and sharpness analysis

Research Standards Implemented:
- Guo et al. (2017): "On Calibration of Modern Neural Networks"
- Platt (1999): "Probabilistic outputs for support vector machines"
- Murphy (1973): "A new vector partition of the probability score"
- Ovadia et al. (2019): "Can you trust your model's uncertainty?"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from scipy import optimize
import warnings
from collections import defaultdict
import matplotlib.pyplot as plt


@dataclass
class CalibrationMetrics:
    """Container for calibration analysis results."""
    ece: float  # Expected Calibration Error
    mce: float  # Maximum Calibration Error
    ace: float  # Average Calibration Error
    brier_score: float
    reliability: float
    sharpness: float
    confidence_histogram: Dict[str, np.ndarray]
    calibration_curve: Dict[str, np.ndarray]
    overconfidence_rate: float
    underconfidence_rate: float


class CalibrationAnalyzer:
    """
    Comprehensive calibration analysis for meta-learning models.
    
    Analyzes how well predicted probabilities match actual outcomes.
    """
    
    def __init__(self, n_bins: int = 15, temperature_search_range: Tuple[float, float] = (0.1, 10.0)):
        """
        Initialize calibration analyzer.
        
        Args:
            n_bins: Number of bins for reliability diagrams
            temperature_search_range: Range for temperature scaling search
        """
        self.n_bins = n_bins
        self.temperature_search_range = temperature_search_range
        self.fitted_calibrator = None
    
    def analyze_calibration(self, 
                          logits: torch.Tensor,
                          labels: torch.Tensor,
                          return_detailed: bool = True) -> CalibrationMetrics:
        """
        Comprehensive calibration analysis.
        
        Args:
            logits: Model logits [N, C] where C is number of classes
            labels: True labels [N] 
            return_detailed: Whether to return detailed breakdown
            
        Returns:
            CalibrationMetrics object with analysis results
        """
        if isinstance(logits, torch.Tensor):
            logits = logits.cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        
        # Convert logits to probabilities
        probabilities = self._logits_to_probabilities(logits)
        confidences = np.max(probabilities, axis=1)
        predictions = np.argmax(probabilities, axis=1)
        accuracies = (predictions == labels).astype(float)
        
        # Core calibration metrics
        ece = self._expected_calibration_error(confidences, accuracies)
        mce = self._maximum_calibration_error(confidences, accuracies)
        ace = self._average_calibration_error(confidences, accuracies)
        
        # Brier score and decomposition
        brier_score = self._brier_score(probabilities, labels)
        reliability, sharpness = self._brier_decomposition(probabilities, labels, confidences, accuracies)
        
        # Calibration curve data
        calibration_data = self._compute_calibration_curve(confidences, accuracies)
        
        # Over/underconfidence analysis
        overconf_rate, underconf_rate = self._confidence_bias_analysis(confidences, accuracies)
        
        return CalibrationMetrics(
            ece=ece,
            mce=mce, 
            ace=ace,
            brier_score=brier_score,
            reliability=reliability,
            sharpness=sharpness,
            confidence_histogram=calibration_data['histogram'],
            calibration_curve=calibration_data['curve'],
            overconfidence_rate=overconf_rate,
            underconfidence_rate=underconf_rate
        )
    
    def _logits_to_probabilities(self, logits: np.ndarray) -> np.ndarray:
        """Convert logits to probabilities using softmax."""
        # Numerical stability
        shifted_logits = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(shifted_logits)
        probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        return probabilities
    
    def _expected_calibration_error(self, confidences: np.ndarray, accuracies: np.ndarray) -> float:
        """
        Calculate Expected Calibration Error (ECE).
        
        ECE measures the difference between confidence and accuracy across bins.
        """
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def _maximum_calibration_error(self, confidences: np.ndarray, accuracies: np.ndarray) -> float:
        """
        Calculate Maximum Calibration Error (MCE).
        
        MCE is the maximum difference between confidence and accuracy across bins.
        """
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        mce = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            
            if in_bin.sum() > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                bin_error = np.abs(avg_confidence_in_bin - accuracy_in_bin)
                mce = max(mce, bin_error)
        
        return mce
    
    def _average_calibration_error(self, confidences: np.ndarray, accuracies: np.ndarray) -> float:
        """
        Calculate Average Calibration Error (ACE).
        
        ACE is the average difference between confidence and accuracy across bins.
        """
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        errors = []
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            
            if in_bin.sum() > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                errors.append(np.abs(avg_confidence_in_bin - accuracy_in_bin))
        
        return np.mean(errors) if errors else 0.0
    
    def _brier_score(self, probabilities: np.ndarray, labels: np.ndarray) -> float:
        """
        Calculate Brier score.
        
        Brier score measures the mean squared difference between predicted probabilities
        and actual outcomes.
        """
        n_classes = probabilities.shape[1]
        one_hot_labels = np.eye(n_classes)[labels]
        
        brier = np.mean(np.sum((probabilities - one_hot_labels) ** 2, axis=1))
        return brier
    
    def _brier_decomposition(self, 
                           probabilities: np.ndarray, 
                           labels: np.ndarray,
                           confidences: np.ndarray, 
                           accuracies: np.ndarray) -> Tuple[float, float]:
        """
        Decompose Brier score into reliability and sharpness components.
        
        Reliability: How close predicted probabilities are to actual frequencies
        Sharpness: How much predicted probabilities differ from base rate
        """
        # Overall accuracy (base rate)
        base_rate = np.mean(accuracies)
        
        # Bin-wise analysis
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        reliability = 0
        sharpness = 0
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            
            if in_bin.sum() > 0:
                prop_in_bin = in_bin.mean()
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                
                # Reliability component
                reliability += prop_in_bin * (avg_confidence_in_bin - accuracy_in_bin) ** 2
                
                # Sharpness component
                sharpness += prop_in_bin * (accuracy_in_bin - base_rate) ** 2
        
        return reliability, sharpness
    
    def _compute_calibration_curve(self, 
                                 confidences: np.ndarray, 
                                 accuracies: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute calibration curve data for plotting."""
        
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        bin_centers = []
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            
            if in_bin.sum() > 0:
                bin_centers.append((bin_lower + bin_upper) / 2)
                bin_accuracies.append(accuracies[in_bin].mean())
                bin_confidences.append(confidences[in_bin].mean())
                bin_counts.append(in_bin.sum())
            else:
                bin_centers.append((bin_lower + bin_upper) / 2)
                bin_accuracies.append(0)
                bin_confidences.append(0)
                bin_counts.append(0)
        
        return {
            'curve': {
                'bin_centers': np.array(bin_centers),
                'bin_accuracies': np.array(bin_accuracies),
                'bin_confidences': np.array(bin_confidences)
            },
            'histogram': {
                'bin_centers': np.array(bin_centers),
                'bin_counts': np.array(bin_counts),
                'bin_boundaries': bin_boundaries
            }
        }
    
    def _confidence_bias_analysis(self, 
                                confidences: np.ndarray, 
                                accuracies: np.ndarray) -> Tuple[float, float]:
        """Analyze overconfidence and underconfidence rates."""
        
        # Overconfidence: confidence > accuracy
        overconfident_mask = confidences > accuracies
        overconfidence_rate = np.mean(overconfident_mask)
        
        # Underconfidence: confidence < accuracy  
        underconfident_mask = confidences < accuracies
        underconfidence_rate = np.mean(underconfident_mask)
        
        return overconfidence_rate, underconfidence_rate


class TemperatureScaling:
    """
    Temperature scaling for model calibration.
    
    Post-hoc calibration method that learns a single temperature parameter
    to calibrate model predictions.
    """
    
    def __init__(self):
        """Initialize temperature scaling."""
        self.temperature = 1.0
        self.fitted = False
    
    def fit(self, logits: torch.Tensor, labels: torch.Tensor, lr: float = 0.01, max_iter: int = 50):
        """
        Fit temperature scaling parameter.
        
        Args:
            logits: Validation logits for calibration [N, C]
            labels: Validation labels [N]
            lr: Learning rate for optimization
            max_iter: Maximum optimization iterations
        """
        if isinstance(logits, np.ndarray):
            logits = torch.from_numpy(logits).float()
        if isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels).long()
        
        # Initialize temperature parameter
        temperature = nn.Parameter(torch.ones(1) * 1.5)
        optimizer = torch.optim.LBFGS([temperature], lr=lr, max_iter=max_iter)
        
        def eval_loss():
            optimizer.zero_grad()
            scaled_logits = logits / temperature
            loss = F.cross_entropy(scaled_logits, labels)
            loss.backward()
            return loss
        
        # Optimize temperature
        optimizer.step(eval_loss)
        
        self.temperature = temperature.item()
        self.fitted = True
        
        return self.temperature
    
    def transform(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply temperature scaling to logits.
        
        Args:
            logits: Input logits [N, C]
            
        Returns:
            Temperature-scaled logits [N, C]
        """
        if not self.fitted:
            warnings.warn("Temperature scaling not fitted. Using temperature=1.0")
            return logits
        
        return logits / self.temperature
    
    def predict_proba(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Get calibrated probabilities.
        
        Args:
            logits: Input logits [N, C]
            
        Returns:
            Calibrated probabilities [N, C]
        """
        scaled_logits = self.transform(logits)
        return F.softmax(scaled_logits, dim=1)


class PlattScaling:
    """
    Platt scaling for binary calibration.
    
    Fits a sigmoid function to map scores to probabilities.
    Extends to multiclass via one-vs-rest approach.
    """
    
    def __init__(self):
        """Initialize Platt scaling."""
        self.calibrators = {}
        self.fitted = False
    
    def fit(self, scores: np.ndarray, labels: np.ndarray):
        """
        Fit Platt scaling parameters.
        
        Args:
            scores: Model scores/logits [N, C] or [N] for binary
            labels: True labels [N]
        """
        if scores.ndim == 1:
            # Binary case
            self._fit_binary(scores, labels)
        else:
            # Multiclass case - fit one calibrator per class
            n_classes = scores.shape[1]
            
            for class_idx in range(n_classes):
                # One-vs-rest setup
                binary_labels = (labels == class_idx).astype(int)
                class_scores = scores[:, class_idx]
                
                self.calibrators[class_idx] = self._fit_binary_calibrator(class_scores, binary_labels)
        
        self.fitted = True
    
    def _fit_binary(self, scores: np.ndarray, labels: np.ndarray):
        """Fit binary Platt scaling."""
        self.calibrators[0] = self._fit_binary_calibrator(scores, labels)
    
    def _fit_binary_calibrator(self, scores: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """
        Fit sigmoid parameters for binary calibration.
        
        Minimizes negative log-likelihood of sigmoid(A*score + B).
        """
        # Prior counts for regularization (Platt's method)
        prior0 = np.sum(labels == 0)
        prior1 = np.sum(labels == 1)
        
        # Target probabilities with smoothing
        hiTarget = (prior1 + 1.0) / (prior1 + 2.0)
        loTarget = 1.0 / (prior0 + 2.0)
        
        # Convert labels to target probabilities
        targets = np.where(labels == 1, hiTarget, loTarget)
        
        # Optimization function
        def objective(params):
            A, B = params
            predictions = 1.0 / (1.0 + np.exp(A * scores + B))
            predictions = np.clip(predictions, 1e-7, 1-1e-7)  # Numerical stability
            
            # Negative log-likelihood
            nll = -np.sum(targets * np.log(predictions) + (1-targets) * np.log(1-predictions))
            return nll
        
        # Optimize A and B parameters
        result = optimize.minimize(objective, x0=[0.0, 0.0], method='BFGS')
        
        return {'A': result.x[0], 'B': result.x[1]}
    
    def transform(self, scores: np.ndarray) -> np.ndarray:
        """
        Apply Platt scaling to scores.
        
        Args:
            scores: Input scores [N, C] or [N] for binary
            
        Returns:
            Calibrated probabilities [N, C] or [N] for binary
        """
        if not self.fitted:
            raise ValueError("Platt scaling not fitted")
        
        if scores.ndim == 1:
            # Binary case
            calibrator = self.calibrators[0]
            return self._apply_sigmoid(scores, calibrator['A'], calibrator['B'])
        else:
            # Multiclass case
            n_classes = scores.shape[1]
            calibrated_probs = np.zeros_like(scores)
            
            for class_idx in range(n_classes):
                if class_idx in self.calibrators:
                    calibrator = self.calibrators[class_idx]
                    class_scores = scores[:, class_idx]
                    calibrated_probs[:, class_idx] = self._apply_sigmoid(
                        class_scores, calibrator['A'], calibrator['B']
                    )
                else:
                    # Fallback for classes not seen during training
                    calibrated_probs[:, class_idx] = 0.5
            
            # Normalize to valid probability distribution
            calibrated_probs = calibrated_probs / np.sum(calibrated_probs, axis=1, keepdims=True)
            
            return calibrated_probs
    
    def _apply_sigmoid(self, scores: np.ndarray, A: float, B: float) -> np.ndarray:
        """Apply sigmoid transformation with parameters A and B."""
        linear_combination = A * scores + B
        # Numerical stability
        linear_combination = np.clip(linear_combination, -500, 500)
        return 1.0 / (1.0 + np.exp(linear_combination))


class CalibrationVisualizer:
    """Visualization tools for calibration analysis."""
    
    @staticmethod
    def plot_reliability_diagram(calibration_metrics: CalibrationMetrics,
                               title: str = "Reliability Diagram",
                               figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """
        Plot reliability diagram showing calibration curve.
        
        Args:
            calibration_metrics: CalibrationMetrics object
            title: Plot title
            figsize: Figure size
            
        Returns:
            matplotlib Figure object
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Extract data
        curve_data = calibration_metrics.calibration_curve
        hist_data = calibration_metrics.confidence_histogram
        
        bin_centers = curve_data['bin_centers']
        bin_accuracies = curve_data['bin_accuracies'] 
        bin_confidences = curve_data['bin_confidences']
        bin_counts = hist_data['bin_counts']
        
        # Plot 1: Reliability diagram
        ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration')
        ax1.scatter(bin_confidences, bin_accuracies, s=bin_counts*0.5, 
                   alpha=0.7, c='blue', label='Bin accuracy')
        ax1.plot(bin_confidences, bin_accuracies, 'b-', alpha=0.8)
        
        ax1.set_xlabel('Mean Predicted Probability')
        ax1.set_ylabel('Fraction of Positives')
        ax1.set_title(f'{title}\nECE: {calibration_metrics.ece:.3f}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Confidence histogram
        ax2.bar(bin_centers, bin_counts, width=0.05, alpha=0.7, color='orange')
        ax2.set_xlabel('Confidence')
        ax2.set_ylabel('Count')
        ax2.set_title('Confidence Distribution')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_calibration_comparison(before_metrics: CalibrationMetrics,
                                  after_metrics: CalibrationMetrics,
                                  method_name: str = "Calibration",
                                  figsize: Tuple[int, int] = (12, 5)) -> plt.Figure:
        """
        Compare calibration before and after calibration method.
        
        Args:
            before_metrics: Calibration metrics before calibration
            after_metrics: Calibration metrics after calibration
            method_name: Name of calibration method
            figsize: Figure size
            
        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Before calibration
        before_curve = before_metrics.calibration_curve
        axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes[0].scatter(before_curve['bin_confidences'], before_curve['bin_accuracies'], 
                       alpha=0.7, c='red', label='Before')
        axes[0].plot(before_curve['bin_confidences'], before_curve['bin_accuracies'], 'r-', alpha=0.8)
        axes[0].set_xlabel('Mean Predicted Probability')
        axes[0].set_ylabel('Fraction of Positives')
        axes[0].set_title(f'Before {method_name}\nECE: {before_metrics.ece:.3f}')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # After calibration
        after_curve = after_metrics.calibration_curve
        axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes[1].scatter(after_curve['bin_confidences'], after_curve['bin_accuracies'], 
                       alpha=0.7, c='green', label='After')
        axes[1].plot(after_curve['bin_confidences'], after_curve['bin_accuracies'], 'g-', alpha=0.8)
        axes[1].set_xlabel('Mean Predicted Probability')
        axes[1].set_ylabel('Fraction of Positives')
        axes[1].set_title(f'After {method_name}\nECE: {after_metrics.ece:.3f}')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_calibration_metrics_summary(calibration_metrics: CalibrationMetrics,
                                       figsize: Tuple[int, int] = (14, 10)) -> plt.Figure:
        """
        Create comprehensive calibration metrics summary plot.
        
        Args:
            calibration_metrics: CalibrationMetrics object
            figsize: Figure size
            
        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 1. Reliability diagram
        curve_data = calibration_metrics.calibration_curve
        axes[0, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes[0, 0].scatter(curve_data['bin_confidences'], curve_data['bin_accuracies'], 
                          alpha=0.7, c='blue')
        axes[0, 0].plot(curve_data['bin_confidences'], curve_data['bin_accuracies'], 'b-', alpha=0.8)
        axes[0, 0].set_xlabel('Confidence')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_title('Reliability Diagram')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Metrics bar chart
        metrics_names = ['ECE', 'MCE', 'ACE', 'Brier Score', 'Reliability', 'Sharpness']
        metrics_values = [
            calibration_metrics.ece,
            calibration_metrics.mce,
            calibration_metrics.ace,
            calibration_metrics.brier_score,
            calibration_metrics.reliability,
            calibration_metrics.sharpness
        ]
        
        axes[0, 1].bar(metrics_names, metrics_values, alpha=0.7, color='orange')
        axes[0, 1].set_ylabel('Value')
        axes[0, 1].set_title('Calibration Metrics')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Confidence histogram
        hist_data = calibration_metrics.confidence_histogram
        axes[1, 0].bar(hist_data['bin_centers'], hist_data['bin_counts'], 
                      width=0.05, alpha=0.7, color='green')
        axes[1, 0].set_xlabel('Confidence')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Confidence Distribution')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Bias analysis
        bias_categories = ['Well-Calibrated', 'Overconfident', 'Underconfident']
        well_calibrated_rate = 1 - calibration_metrics.overconfidence_rate - calibration_metrics.underconfidence_rate
        bias_rates = [well_calibrated_rate, calibration_metrics.overconfidence_rate, 
                     calibration_metrics.underconfidence_rate]
        
        colors = ['green', 'red', 'blue']
        axes[1, 1].pie(bias_rates, labels=bias_categories, colors=colors, autopct='%1.1f%%')
        axes[1, 1].set_title('Confidence Bias Analysis')
        
        plt.tight_layout()
        return fig


def calibrate_model_predictions(logits: torch.Tensor,
                               labels: torch.Tensor,
                               method: str = "temperature",
                               validation_split: float = 0.2) -> Tuple[torch.Tensor, CalibrationMetrics, CalibrationMetrics]:
    """
    Convenience function to calibrate model predictions.
    
    Args:
        logits: Model logits [N, C]
        labels: True labels [N]
        method: Calibration method ("temperature" or "platt")
        validation_split: Fraction of data to use for calibration fitting
        
    Returns:
        Tuple of (calibrated_logits, before_metrics, after_metrics)
    """
    # Split data
    n_samples = len(logits)
    n_val = int(n_samples * validation_split)
    
    # Random split
    indices = np.random.permutation(n_samples)
    cal_indices = indices[:n_val]
    test_indices = indices[n_val:]
    
    cal_logits = logits[cal_indices]
    cal_labels = labels[cal_indices]
    test_logits = logits[test_indices]
    test_labels = labels[test_indices]
    
    # Analyze before calibration
    analyzer = CalibrationAnalyzer()
    before_metrics = analyzer.analyze_calibration(test_logits, test_labels)
    
    # Apply calibration
    if method == "temperature":
        calibrator = TemperatureScaling()
        calibrator.fit(cal_logits, cal_labels)
        calibrated_logits = calibrator.transform(test_logits)
    elif method == "platt":
        calibrator = PlattScaling()
        calibrator.fit(cal_logits.numpy(), cal_labels.numpy())
        calibrated_probs = calibrator.transform(test_logits.numpy())
        # Convert back to logits for consistency
        calibrated_probs = np.clip(calibrated_probs, 1e-7, 1-1e-7)
        calibrated_logits = torch.from_numpy(np.log(calibrated_probs))
    else:
        raise ValueError(f"Unknown calibration method: {method}")
    
    # Analyze after calibration
    after_metrics = analyzer.analyze_calibration(calibrated_logits, test_labels)
    
    return calibrated_logits, before_metrics, after_metrics