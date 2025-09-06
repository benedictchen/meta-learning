"""
💰 SUPPORT THIS RESEARCH - PLEASE DONATE! 💰

🙏 If this library helps your research, please donate:
💳 PayPal: https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS
⭐ GitHub Sponsors: https://github.com/sponsors/benedictchen

Comprehensive Meta-Learning Evaluation Metrics
==============================================

Professional evaluation metrics for meta-learning with statistical rigor.
Implements metrics beyond what's available in existing libraries.
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import scipy.stats
from sklearn.metrics import roc_auc_score, brier_score_loss


@dataclass
class EvaluationResults:
    """Container for comprehensive evaluation results."""
    accuracy: float
    confidence_interval: Tuple[float, float] 
    per_class_accuracy: Dict[int, float]
    calibration_metrics: Dict[str, float]
    uncertainty_metrics: Dict[str, float]
    prototype_quality: Dict[str, float]
    task_difficulty: float
    statistical_significance: Dict[str, float]


class AccuracyCalculator:
    """Advanced accuracy calculation with statistical analysis."""
    
    @staticmethod
    def calculate_accuracy_with_ci(predictions: torch.Tensor, 
                                  targets: torch.Tensor, 
                                  confidence: float = 0.95) -> Tuple[float, Tuple[float, float]]:
        """
        Calculate accuracy with confidence intervals.
        
        Args:
            predictions: Predicted class labels [N]
            targets: True class labels [N]
            confidence: Confidence level (default 0.95 for 95% CI)
            
        Returns:
            Tuple of (accuracy, (lower_bound, upper_bound))
        """
        correct = (predictions == targets).float()
        accuracy = correct.mean().item()
        n = len(correct)
        
        # Wilson score interval for binomial proportion
        z = scipy.stats.norm.ppf((1 + confidence) / 2)
        p = accuracy
        denominator = 1 + (z**2) / n
        centre_adjusted_probability = (p + (z**2) / (2*n)) / denominator
        adjusted_standard_deviation = torch.sqrt((p*(1-p) + (z**2)/(4*n)) / n) / denominator
        
        lower_bound = centre_adjusted_probability - z * adjusted_standard_deviation
        upper_bound = centre_adjusted_probability + z * adjusted_standard_deviation
        
        return accuracy, (lower_bound.item(), upper_bound.item())
    
    @staticmethod 
    def per_class_accuracy(predictions: torch.Tensor, 
                          targets: torch.Tensor) -> Dict[int, float]:
        """Calculate per-class accuracy breakdown."""
        unique_classes = torch.unique(targets)
        per_class_acc = {}
        
        for class_idx in unique_classes:
            class_mask = targets == class_idx
            class_predictions = predictions[class_mask]
            class_targets = targets[class_mask]
            
            if len(class_targets) > 0:
                accuracy = (class_predictions == class_targets).float().mean().item()
                per_class_acc[class_idx.item()] = accuracy
            else:
                per_class_acc[class_idx.item()] = 0.0
                
        return per_class_acc


class CalibrationAnalyzer:
    """Model calibration analysis and metrics."""
    
    @staticmethod
    def expected_calibration_error(logits: torch.Tensor, 
                                  targets: torch.Tensor, 
                                  n_bins: int = 15) -> float:
        """
        Calculate Expected Calibration Error (ECE).
        
        Args:
            logits: Model output logits [N, C]
            targets: True class labels [N]
            n_bins: Number of calibration bins
            
        Returns:
            ECE value
        """
        probs = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(probs, dim=1)
        accuracies = predictions == targets
        
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.float().mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                
        return ece.item()
    
    @staticmethod
    def maximum_calibration_error(logits: torch.Tensor, 
                                 targets: torch.Tensor, 
                                 n_bins: int = 15) -> float:
        """Calculate Maximum Calibration Error (MCE)."""
        probs = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(probs, dim=1)
        accuracies = predictions == targets
        
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        mce = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            
            if in_bin.sum() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                bin_error = torch.abs(avg_confidence_in_bin - accuracy_in_bin)
                mce = max(mce, bin_error.item())
                
        return mce
    
    @staticmethod
    def reliability_diagram_data(logits: torch.Tensor, 
                                targets: torch.Tensor, 
                                n_bins: int = 15) -> Dict[str, np.ndarray]:
        """Generate data for reliability diagram visualization."""
        probs = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(probs, dim=1)
        accuracies = predictions == targets
        
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            
            if in_bin.sum() > 0:
                bin_accuracy = accuracies[in_bin].float().mean().item()
                bin_confidence = confidences[in_bin].mean().item()
                bin_count = in_bin.sum().item()
            else:
                bin_accuracy = 0.0
                bin_confidence = (bin_lower + bin_upper) / 2
                bin_count = 0
                
            bin_accuracies.append(bin_accuracy)
            bin_confidences.append(bin_confidence)
            bin_counts.append(bin_count)
            
        return {
            'accuracies': np.array(bin_accuracies),
            'confidences': np.array(bin_confidences),
            'counts': np.array(bin_counts),
            'bin_boundaries': bin_boundaries.numpy()
        }


class UncertaintyQuantifier:
    """Uncertainty quantification metrics for meta-learning."""
    
    @staticmethod
    def predictive_entropy(logits: torch.Tensor) -> torch.Tensor:
        """Calculate predictive entropy as uncertainty measure."""
        probs = F.softmax(logits, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
        return entropy
    
    @staticmethod
    def mutual_information(logits_samples: torch.Tensor) -> torch.Tensor:
        """
        Calculate mutual information for epistemic uncertainty.
        
        Args:
            logits_samples: Multiple forward passes [N_samples, N, C]
            
        Returns:
            Mutual information per sample [N]
        """
        # Average over samples
        mean_probs = F.softmax(logits_samples, dim=2).mean(dim=0)  # [N, C]
        
        # Entropy of mean
        entropy_mean = -torch.sum(mean_probs * torch.log(mean_probs + 1e-8), dim=1)
        
        # Mean entropy
        sample_entropies = UncertaintyQuantifier.predictive_entropy(logits_samples)  # [N_samples, N]
        mean_entropy = sample_entropies.mean(dim=0)  # [N]
        
        # Mutual information = Entropy(mean) - Mean(entropy)
        mutual_info = entropy_mean - mean_entropy
        return mutual_info
    
    @staticmethod
    def uncertainty_metrics(logits: torch.Tensor, 
                          targets: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """Calculate comprehensive uncertainty metrics."""
        metrics = {}
        
        # Predictive uncertainty
        entropy = UncertaintyQuantifier.predictive_entropy(logits)
        metrics['mean_predictive_entropy'] = entropy.mean().item()
        metrics['std_predictive_entropy'] = entropy.std().item()
        
        # Confidence-based metrics
        probs = F.softmax(logits, dim=1)
        max_probs, _ = torch.max(probs, dim=1)
        metrics['mean_confidence'] = max_probs.mean().item()
        metrics['confidence_variance'] = max_probs.var().item()
        
        if targets is not None:
            # Calibration-based uncertainty assessment
            correct_mask = (logits.argmax(dim=1) == targets)
            correct_entropy = entropy[correct_mask].mean().item() if correct_mask.sum() > 0 else 0
            incorrect_entropy = entropy[~correct_mask].mean().item() if (~correct_mask).sum() > 0 else 0
            
            metrics['correct_predictions_entropy'] = correct_entropy
            metrics['incorrect_predictions_entropy'] = incorrect_entropy
            metrics['entropy_discrimination'] = incorrect_entropy - correct_entropy
            
        return metrics


class PrototypeAnalyzer:
    """Analyze prototype quality and class separation."""
    
    @staticmethod
    def intra_class_variance(features: torch.Tensor, 
                            labels: torch.Tensor) -> Dict[str, float]:
        """Calculate intra-class variance metrics."""
        unique_labels = torch.unique(labels)
        class_variances = []
        
        for label in unique_labels:
            class_mask = labels == label
            class_features = features[class_mask]
            
            if len(class_features) > 1:
                # Calculate variance within class
                class_mean = class_features.mean(dim=0)
                centered_features = class_features - class_mean
                class_variance = torch.mean(torch.sum(centered_features**2, dim=1))
                class_variances.append(class_variance.item())
        
        return {
            'mean_intra_class_variance': np.mean(class_variances),
            'std_intra_class_variance': np.std(class_variances),
            'max_intra_class_variance': np.max(class_variances),
            'min_intra_class_variance': np.min(class_variances)
        }
    
    @staticmethod
    def inter_class_distances(prototypes: torch.Tensor) -> Dict[str, float]:
        """Calculate inter-class distance metrics."""
        # Pairwise distances between prototypes
        distances = torch.cdist(prototypes, prototypes)
        
        # Remove diagonal (self-distances)
        n_classes = len(prototypes)
        mask = ~torch.eye(n_classes, dtype=torch.bool)
        pairwise_distances = distances[mask]
        
        return {
            'mean_inter_class_distance': pairwise_distances.mean().item(),
            'std_inter_class_distance': pairwise_distances.std().item(),
            'min_inter_class_distance': pairwise_distances.min().item(),
            'max_inter_class_distance': pairwise_distances.max().item()
        }
    
    @staticmethod
    def silhouette_score_approx(features: torch.Tensor, 
                               labels: torch.Tensor) -> float:
        """Calculate approximate silhouette score for prototype quality."""
        unique_labels = torch.unique(labels)
        silhouette_scores = []
        
        for i, sample_features in enumerate(features):
            sample_label = labels[i]
            
            # Intra-cluster distance (a)
            same_class_mask = labels == sample_label
            same_class_features = features[same_class_mask]
            if len(same_class_features) > 1:
                a = torch.mean(torch.norm(sample_features - same_class_features, dim=1))
            else:
                a = 0
            
            # Inter-cluster distance (b)
            b_distances = []
            for other_label in unique_labels:
                if other_label != sample_label:
                    other_class_mask = labels == other_label
                    other_class_features = features[other_class_mask]
                    b_dist = torch.mean(torch.norm(sample_features - other_class_features, dim=1))
                    b_distances.append(b_dist.item())
            
            if b_distances:
                b = min(b_distances)
                silhouette = (b - a) / max(a, b) if max(a, b) > 0 else 0
                silhouette_scores.append(silhouette)
        
        return np.mean(silhouette_scores) if silhouette_scores else 0.0


class StatisticalTester:
    """Statistical significance testing for meta-learning evaluation."""
    
    @staticmethod
    def paired_t_test(results_a: List[float], 
                     results_b: List[float], 
                     alpha: float = 0.05) -> Dict[str, float]:
        """Perform paired t-test between two methods."""
        if len(results_a) != len(results_b):
            raise ValueError("Results lists must have same length")
        
        t_stat, p_value = scipy.stats.ttest_rel(results_a, results_b)
        
        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < alpha,
            'effect_size': (np.mean(results_a) - np.mean(results_b)) / np.sqrt(
                (np.var(results_a, ddof=1) + np.var(results_b, ddof=1)) / 2
            )
        }
    
    @staticmethod
    def bootstrap_confidence_interval(results: List[float], 
                                    n_bootstrap: int = 1000,
                                    confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval."""
        bootstrap_means = []
        n_samples = len(results)
        
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(results, size=n_samples, replace=True)
            bootstrap_means.append(np.mean(bootstrap_sample))
        
        alpha = 1 - confidence
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_bound = np.percentile(bootstrap_means, lower_percentile)
        upper_bound = np.percentile(bootstrap_means, upper_percentile)
        
        return lower_bound, upper_bound


class TaskDifficultyAssessor:
    """Assess task difficulty for meta-learning episodes."""
    
    @staticmethod
    def calculate_task_difficulty(support_features: torch.Tensor,
                                 support_labels: torch.Tensor,
                                 query_features: torch.Tensor,
                                 query_labels: torch.Tensor) -> Dict[str, float]:
        """Calculate comprehensive task difficulty metrics."""
        metrics = {}
        
        # Inter-class separability in support set
        unique_labels = torch.unique(support_labels)
        prototypes = []
        for label in unique_labels:
            class_mask = support_labels == label
            class_features = support_features[class_mask]
            prototype = class_features.mean(dim=0)
            prototypes.append(prototype)
        
        prototypes = torch.stack(prototypes)
        inter_class_metrics = PrototypeAnalyzer.inter_class_distances(prototypes)
        intra_class_metrics = PrototypeAnalyzer.intra_class_variance(support_features, support_labels)
        
        # Separability ratio
        avg_inter_distance = inter_class_metrics['mean_inter_class_distance']
        avg_intra_variance = intra_class_metrics['mean_intra_class_variance']
        separability_ratio = avg_inter_distance / (avg_intra_variance + 1e-8)
        
        metrics['separability_ratio'] = separability_ratio
        metrics['inter_class_distance'] = avg_inter_distance
        metrics['intra_class_variance'] = avg_intra_variance
        
        # Support-query domain gap
        support_centroid = support_features.mean(dim=0)
        query_centroid = query_features.mean(dim=0)
        domain_gap = torch.norm(support_centroid - query_centroid).item()
        metrics['domain_gap'] = domain_gap
        
        # Feature space dimensionality relative to sample size
        n_features = support_features.shape[1]
        n_support_samples = len(support_features)
        dimensionality_ratio = n_features / n_support_samples
        metrics['dimensionality_ratio'] = dimensionality_ratio
        
        # Overall difficulty score (lower is easier)
        difficulty_score = (1.0 / separability_ratio) + domain_gap + dimensionality_ratio
        metrics['overall_difficulty'] = difficulty_score
        
        return metrics


class ComprehensiveEvaluator:
    """Comprehensive meta-learning evaluation system."""
    
    def __init__(self):
        self.accuracy_calc = AccuracyCalculator()
        self.calibration_analyzer = CalibrationAnalyzer()
        self.uncertainty_quantifier = UncertaintyQuantifier()
        self.prototype_analyzer = PrototypeAnalyzer()
        self.statistical_tester = StatisticalTester()
        self.task_assessor = TaskDifficultyAssessor()
    
    def evaluate_episode(self, 
                        logits: torch.Tensor,
                        targets: torch.Tensor,
                        support_features: Optional[torch.Tensor] = None,
                        support_labels: Optional[torch.Tensor] = None,
                        query_features: Optional[torch.Tensor] = None) -> EvaluationResults:
        """
        Comprehensive evaluation of a single episode.
        
        Args:
            logits: Model output logits [N, C]
            targets: True labels [N]
            support_features: Support set features [N_support, D]
            support_labels: Support set labels [N_support]
            query_features: Query set features [N_query, D]
            
        Returns:
            EvaluationResults with comprehensive metrics
        """
        predictions = logits.argmax(dim=1)
        
        # Accuracy with confidence intervals
        accuracy, ci = self.accuracy_calc.calculate_accuracy_with_ci(predictions, targets)
        per_class_acc = self.accuracy_calc.per_class_accuracy(predictions, targets)
        
        # Calibration metrics
        calibration_metrics = {
            'ece': self.calibration_analyzer.expected_calibration_error(logits, targets),
            'mce': self.calibration_analyzer.maximum_calibration_error(logits, targets)
        }
        
        # Uncertainty metrics
        uncertainty_metrics = self.uncertainty_quantifier.uncertainty_metrics(logits, targets)
        
        # Prototype quality (if features available)
        prototype_quality = {}
        task_difficulty = 0.0
        
        if support_features is not None and support_labels is not None:
            prototype_quality = self.prototype_analyzer.intra_class_variance(support_features, support_labels)
            
            if query_features is not None:
                difficulty_metrics = self.task_assessor.calculate_task_difficulty(
                    support_features, support_labels, query_features, targets
                )
                task_difficulty = difficulty_metrics['overall_difficulty']
                prototype_quality.update(difficulty_metrics)
        
        # Statistical significance placeholder (needs multiple runs)
        statistical_significance = {'single_episode': True}
        
        return EvaluationResults(
            accuracy=accuracy,
            confidence_interval=ci,
            per_class_accuracy=per_class_acc,
            calibration_metrics=calibration_metrics,
            uncertainty_metrics=uncertainty_metrics,
            prototype_quality=prototype_quality,
            task_difficulty=task_difficulty,
            statistical_significance=statistical_significance
        )
    
    def aggregate_results(self, 
                         episode_results: List[EvaluationResults],
                         baseline_results: Optional[List[EvaluationResults]] = None) -> Dict:
        """
        Aggregate results across multiple episodes with statistical analysis.
        
        Args:
            episode_results: List of evaluation results from multiple episodes
            baseline_results: Optional baseline results for comparison
            
        Returns:
            Aggregated statistics and comparisons
        """
        accuracies = [r.accuracy for r in episode_results]
        
        aggregated = {
            'mean_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies),
            'median_accuracy': np.median(accuracies),
            'accuracy_95_ci': self.statistical_tester.bootstrap_confidence_interval(accuracies),
            'n_episodes': len(episode_results)
        }
        
        # Aggregate calibration metrics
        ece_values = [r.calibration_metrics['ece'] for r in episode_results]
        mce_values = [r.calibration_metrics['mce'] for r in episode_results]
        
        aggregated['calibration'] = {
            'mean_ece': np.mean(ece_values),
            'mean_mce': np.mean(mce_values),
            'std_ece': np.std(ece_values),
            'std_mce': np.std(mce_values)
        }
        
        # Aggregate uncertainty metrics
        if episode_results[0].uncertainty_metrics:
            uncertainty_keys = episode_results[0].uncertainty_metrics.keys()
            aggregated['uncertainty'] = {}
            
            for key in uncertainty_keys:
                values = [r.uncertainty_metrics[key] for r in episode_results]
                aggregated['uncertainty'][f'mean_{key}'] = np.mean(values)
                aggregated['uncertainty'][f'std_{key}'] = np.std(values)
        
        # Statistical comparison with baseline
        if baseline_results is not None:
            baseline_accuracies = [r.accuracy for r in baseline_results]
            comparison = self.statistical_tester.paired_t_test(accuracies, baseline_accuracies)
            aggregated['baseline_comparison'] = comparison
        
        return aggregated


import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import warnings


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics with proper typing."""
    accuracy: float
    loss: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    per_class_accuracy: Optional[Dict[int, float]] = None
    calibration_error: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Union[float, Dict, Tuple]]:
        """Convert to dictionary for serialization."""
        result = {"accuracy": self.accuracy}
        
        if self.loss is not None:
            result["loss"] = self.loss
        if self.confidence_interval is not None:
            result["confidence_interval"] = self.confidence_interval
        if self.per_class_accuracy is not None:
            result["per_class_accuracy"] = self.per_class_accuracy
        if self.calibration_error is not None:
            result["calibration_error"] = self.calibration_error
            
        return result


class AccuracyCalculator:
    """Calculate various accuracy metrics for few-shot learning."""
    
    @staticmethod
    def compute_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Compute standard accuracy.
        
        Args:
            predictions: Probability predictions or logits [N, C]
            targets: Ground truth labels [N]
            
        Returns:
            Accuracy as float between 0 and 1
        """
        if predictions.dim() > 1:
            predicted_classes = torch.argmax(predictions, dim=1)
        else:
            predicted_classes = predictions
        
        correct = (predicted_classes == targets).float()
        return correct.mean().item()
    
    @staticmethod  
    def compute_accuracy_with_ci(predictions: torch.Tensor, targets: torch.Tensor, 
                               confidence: float = 0.95) -> Tuple[float, Tuple[float, float]]:
        """Compute accuracy with confidence interval."""
        accuracy = AccuracyCalculator.compute_accuracy(predictions, targets)
        
        # Compute binomial confidence interval
        n = len(targets)
        z_score = 1.96 if confidence == 0.95 else 2.58  # 95% or 99% CI
        margin = z_score * np.sqrt(accuracy * (1 - accuracy) / n)
        
        ci_lower = max(0, accuracy - margin)
        ci_upper = min(1, accuracy + margin)
        
        return accuracy, (ci_lower, ci_upper)
    
    @staticmethod
    def compute_per_class_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> Dict[int, float]:
        """Compute per-class accuracy."""
        if predictions.dim() > 1:
            predicted_classes = torch.argmax(predictions, dim=1)
        else:
            predicted_classes = predictions
            
        per_class_acc = {}
        unique_classes = torch.unique(targets)
        
        for class_id in unique_classes:
            class_mask = targets == class_id
            if class_mask.sum() > 0:
                class_correct = (predicted_classes[class_mask] == targets[class_mask]).float()
                per_class_acc[class_id.item()] = class_correct.mean().item()
        
        return per_class_acc


class CalibrationCalculator:
    """Calculate calibration metrics for model predictions."""
    
    @staticmethod
    def compute_expected_calibration_error(predictions: torch.Tensor, targets: torch.Tensor,
                                         n_bins: int = 15) -> float:
        """
        Compute Expected Calibration Error (ECE).
        
        Args:
            predictions: Probability predictions [N, C] 
            targets: Ground truth labels [N]
            n_bins: Number of confidence bins
            
        Returns:
            ECE value as float
        """
        predicted_probs = torch.softmax(predictions, dim=1) if predictions.dim() > 1 else predictions
        confidences = torch.max(predicted_probs, dim=1)[0] if predicted_probs.dim() > 1 else predicted_probs
        predicted_classes = torch.argmax(predicted_probs, dim=1) if predicted_probs.dim() > 1 else predicted_probs
        
        accuracies = (predicted_classes == targets).float()
        
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]  
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find predictions in this confidence bin
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.float().mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece.item()
    
    @staticmethod
    def compute_maximum_calibration_error(predictions: torch.Tensor, targets: torch.Tensor,
                                        n_bins: int = 15) -> float:
        """Compute Maximum Calibration Error (MCE)."""
        predicted_probs = torch.softmax(predictions, dim=1) if predictions.dim() > 1 else predictions  
        confidences = torch.max(predicted_probs, dim=1)[0] if predicted_probs.dim() > 1 else predicted_probs
        predicted_classes = torch.argmax(predicted_probs, dim=1) if predicted_probs.dim() > 1 else predicted_probs
        
        accuracies = (predicted_classes == targets).float()
        
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        max_error = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            
            if in_bin.sum() > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                error = torch.abs(avg_confidence_in_bin - accuracy_in_bin)
                max_error = max(max_error, error.item())
                
        return max_error
    
    @staticmethod
    def get_reliability_diagram_data(predictions: torch.Tensor, targets: torch.Tensor,
                                   n_bins: int = 15) -> Tuple[np.ndarray, ...]:
        """Get data for reliability diagram plotting."""
        predicted_probs = torch.softmax(predictions, dim=1) if predictions.dim() > 1 else predictions
        confidences = torch.max(predicted_probs, dim=1)[0] if predicted_probs.dim() > 1 else predicted_probs  
        predicted_classes = torch.argmax(predicted_probs, dim=1) if predicted_probs.dim() > 1 else predicted_probs
        
        accuracies = (predicted_classes == targets).float()
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        bin_accs = []
        bin_confs = []  
        bin_sizes = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.float().mean().item()
            
            if in_bin.sum() > 0:
                accuracy_in_bin = accuracies[in_bin].mean().item()
                avg_confidence_in_bin = confidences[in_bin].mean().item() 
                bin_accs.append(accuracy_in_bin)
                bin_confs.append(avg_confidence_in_bin)
                bin_sizes.append(prop_in_bin)
            else:
                bin_accs.append(np.nan)
                bin_confs.append(np.nan)
                bin_sizes.append(0)
        
        return (bin_boundaries, bin_lowers, bin_uppers, 
                np.array(bin_accs), np.array(bin_confs), np.array(bin_sizes))


class UncertaintyCalculator:
    """Calculate uncertainty metrics for model predictions."""
    
    @staticmethod
    def compute_predictive_entropy(predictions: torch.Tensor) -> torch.Tensor:
        """Compute predictive entropy for uncertainty estimation."""
        probs = torch.softmax(predictions, dim=-1) if predictions.dim() > 1 else predictions
        log_probs = torch.log(probs + 1e-12)  # Add epsilon for numerical stability
        entropy = -(probs * log_probs).sum(dim=-1)
        return entropy
    
    @staticmethod
    def compute_mutual_information(mc_predictions: torch.Tensor) -> torch.Tensor:
        """Compute mutual information from Monte Carlo predictions."""
        # mc_predictions shape: [n_samples, n_data, n_classes]
        # Average predictions across samples
        mean_probs = torch.mean(mc_predictions, dim=0)  # [n_data, n_classes]
        
        # Total entropy (aleatoric + epistemic)
        total_entropy = UncertaintyCalculator.compute_predictive_entropy(mean_probs)
        
        # Expected entropy (aleatoric only)
        sample_entropies = UncertaintyCalculator.compute_predictive_entropy(mc_predictions)  # [n_samples, n_data]
        expected_entropy = torch.mean(sample_entropies, dim=0)  # [n_data]
        
        # Mutual information (epistemic uncertainty)  
        mutual_info = total_entropy - expected_entropy
        return mutual_info
    
    @staticmethod
    def compute_confidence(predictions: torch.Tensor) -> torch.Tensor:
        """Compute confidence as maximum predicted probability."""
        probs = torch.softmax(predictions, dim=-1) if predictions.dim() > 1 else predictions
        confidences = torch.max(probs, dim=-1)[0] if probs.dim() > 1 else probs
        return confidences
    
    @staticmethod  
    def decompose_uncertainty(mc_predictions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decompose uncertainty into aleatoric and epistemic components."""
        # mc_predictions shape: [n_samples, n_data, n_classes]
        
        # Mean predictions across samples
        mean_probs = torch.mean(mc_predictions, dim=0)  # [n_data, n_classes]
        
        # Total uncertainty (entropy of mean)
        total_entropy = UncertaintyCalculator.compute_predictive_entropy(mean_probs)
        
        # Aleatoric uncertainty (expected entropy)
        sample_entropies = UncertaintyCalculator.compute_predictive_entropy(mc_predictions)  # [n_samples, n_data]
        aleatoric_entropy = torch.mean(sample_entropies, dim=0)  # [n_data]
        
        # Epistemic uncertainty (difference)
        epistemic_entropy = total_entropy - aleatoric_entropy
        
        return total_entropy, aleatoric_entropy, epistemic_entropy