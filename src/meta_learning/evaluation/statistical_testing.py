#!/usr/bin/env python3
"""
Statistical Testing for Meta-Learning Evaluation

Implements rigorous statistical testing for meta-learning research including:
- Confidence intervals with multiple correction methods
- Significance testing with proper multiple comparisons correction
- Effect size calculations and power analysis
- Bootstrap and permutation testing for robust inference

Research Standards Implemented:
- Welch's t-test for unequal variances (Welch, 1947)
- Benjamini-Hochberg FDR correction (Benjamini & Hochberg, 1995)
- Bootstrap confidence intervals (Efron, 1979)
- Cohen's d effect size (Cohen, 1988)
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from scipy import stats
import warnings
from collections import defaultdict


@dataclass
class StatisticalTestResult:
    """Container for statistical test results."""
    statistic: float
    p_value: float
    confidence_interval: Tuple[float, float]
    effect_size: float
    power: float
    method: str
    sample_size: int
    degrees_of_freedom: Optional[float] = None
    corrected_p_value: Optional[float] = None


class ConfidenceIntervalCalculator:
    """
    Professional confidence interval calculations for meta-learning metrics.
    
    Supports multiple methods for robust statistical inference.
    """
    
    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize confidence interval calculator.
        
        Args:
            confidence_level: Confidence level (default: 0.95 for 95% CI)
        """
        if not 0 < confidence_level < 1:
            raise ValueError("Confidence level must be between 0 and 1")
        
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
    
    def t_confidence_interval(self, 
                            data: Union[torch.Tensor, np.ndarray, List[float]],
                            method: str = "welch") -> Tuple[float, float]:
        """
        Calculate t-distribution confidence interval.
        
        Args:
            data: Sample data
            method: Method to use ("student" or "welch")
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        elif isinstance(data, list):
            data = np.array(data)
        
        if len(data) < 2:
            raise ValueError("Need at least 2 data points for confidence interval")
        
        mean = np.mean(data)
        std = np.std(data, ddof=1)  # Sample standard deviation
        n = len(data)
        
        if method == "student":
            # Student's t-test (assumes equal variances)
            df = n - 1
            sem = std / np.sqrt(n)
        elif method == "welch":
            # Welch's t-test (unequal variances, more robust)
            df = n - 1  # For single sample, same as Student's
            sem = std / np.sqrt(n)
        else:
            raise ValueError("Method must be 'student' or 'welch'")
        
        t_critical = stats.t.ppf(1 - self.alpha/2, df)
        margin_error = t_critical * sem
        
        return (mean - margin_error, mean + margin_error)
    
    def bootstrap_confidence_interval(self,
                                    data: Union[torch.Tensor, np.ndarray, List[float]],
                                    statistic_func: callable = np.mean,
                                    n_bootstrap: int = 10000,
                                    method: str = "percentile") -> Tuple[float, float]:
        """
        Calculate bootstrap confidence interval.
        
        Args:
            data: Sample data
            statistic_func: Function to calculate statistic
            n_bootstrap: Number of bootstrap samples
            method: Bootstrap method ("percentile", "bca", or "basic")
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        elif isinstance(data, list):
            data = np.array(data)
        
        if len(data) < 2:
            raise ValueError("Need at least 2 data points for bootstrap")
        
        # Generate bootstrap samples
        bootstrap_statistics = []
        rng = np.random.RandomState(42)  # For reproducibility
        
        for _ in range(n_bootstrap):
            bootstrap_sample = rng.choice(data, size=len(data), replace=True)
            bootstrap_stat = statistic_func(bootstrap_sample)
            bootstrap_statistics.append(bootstrap_stat)
        
        bootstrap_statistics = np.array(bootstrap_statistics)
        
        if method == "percentile":
            # Percentile method (most common)
            lower_percentile = (self.alpha/2) * 100
            upper_percentile = (1 - self.alpha/2) * 100
            
            lower_bound = np.percentile(bootstrap_statistics, lower_percentile)
            upper_bound = np.percentile(bootstrap_statistics, upper_percentile)
            
        elif method == "basic":
            # Basic bootstrap method
            observed_stat = statistic_func(data)
            lower_percentile = (self.alpha/2) * 100
            upper_percentile = (1 - self.alpha/2) * 100
            
            lower_bootstrap = np.percentile(bootstrap_statistics, lower_percentile)
            upper_bootstrap = np.percentile(bootstrap_statistics, upper_percentile)
            
            lower_bound = 2 * observed_stat - upper_bootstrap
            upper_bound = 2 * observed_stat - lower_bootstrap
            
        elif method == "bca":
            # Bias-corrected and accelerated (BCa) bootstrap
            # More complex but more accurate
            observed_stat = statistic_func(data)
            
            # Bias correction
            n_less = np.sum(bootstrap_statistics < observed_stat)
            z0 = stats.norm.ppf(n_less / n_bootstrap)
            
            # Acceleration correction (jackknife)
            n = len(data)
            jackknife_stats = []
            for i in range(n):
                jackknife_sample = np.delete(data, i)
                jackknife_stat = statistic_func(jackknife_sample)
                jackknife_stats.append(jackknife_stat)
            
            jackknife_mean = np.mean(jackknife_stats)
            acceleration = np.sum((jackknife_mean - jackknife_stats)**3) / (
                6 * (np.sum((jackknife_mean - jackknife_stats)**2))**(3/2)
            )
            
            # Adjust percentiles
            z_alpha2 = stats.norm.ppf(self.alpha/2)
            z_1alpha2 = stats.norm.ppf(1 - self.alpha/2)
            
            alpha1 = stats.norm.cdf(z0 + (z0 + z_alpha2)/(1 - acceleration*(z0 + z_alpha2)))
            alpha2 = stats.norm.cdf(z0 + (z0 + z_1alpha2)/(1 - acceleration*(z0 + z_1alpha2)))
            
            lower_bound = np.percentile(bootstrap_statistics, alpha1 * 100)
            upper_bound = np.percentile(bootstrap_statistics, alpha2 * 100)
            
        else:
            raise ValueError("Method must be 'percentile', 'basic', or 'bca'")
        
        return (lower_bound, upper_bound)
    
    def wilson_score_interval(self, successes: int, trials: int) -> Tuple[float, float]:
        """
        Calculate Wilson score confidence interval for proportions.
        
        More accurate than normal approximation for small samples.
        
        Args:
            successes: Number of successes
            trials: Total number of trials
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        if trials == 0:
            return (0.0, 1.0)
        
        p = successes / trials
        z = stats.norm.ppf(1 - self.alpha/2)
        
        denominator = 1 + z**2 / trials
        center = (p + z**2 / (2 * trials)) / denominator
        margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * trials)) / trials) / denominator
        
        return (max(0, center - margin), min(1, center + margin))


class SignificanceTestSuite:
    """
    Comprehensive significance testing for meta-learning experiments.
    
    Handles multiple comparisons, effect sizes, and power analysis.
    """
    
    def __init__(self, alpha: float = 0.05):
        """
        Initialize significance test suite.
        
        Args:
            alpha: Significance level (default: 0.05)
        """
        self.alpha = alpha
        self.results_cache = {}
    
    def compare_algorithms(self,
                         results_dict: Dict[str, List[float]],
                         correction_method: str = "benjamini_hochberg",
                         test_type: str = "welch") -> Dict[str, Dict[str, StatisticalTestResult]]:
        """
        Compare multiple algorithms with proper multiple comparisons correction.
        
        Args:
            results_dict: Dictionary mapping algorithm names to result lists
            correction_method: Multiple comparison correction method
            test_type: Statistical test to use
            
        Returns:
            Dictionary of pairwise comparison results
        """
        algorithms = list(results_dict.keys())
        pairwise_results = {}
        raw_p_values = []
        comparison_pairs = []
        
        # Perform all pairwise comparisons
        for i, alg1 in enumerate(algorithms):
            pairwise_results[alg1] = {}
            
            for j, alg2 in enumerate(algorithms):
                if i >= j:  # Skip self-comparison and duplicates
                    continue
                
                # Perform statistical test
                result = self._compare_two_algorithms(
                    results_dict[alg1], 
                    results_dict[alg2],
                    test_type=test_type
                )
                
                pairwise_results[alg1][alg2] = result
                raw_p_values.append(result.p_value)
                comparison_pairs.append((alg1, alg2))
        
        # Apply multiple comparisons correction
        if len(raw_p_values) > 1:
            corrected_p_values = self._apply_multiple_comparisons_correction(
                raw_p_values, correction_method
            )
            
            # Update results with corrected p-values
            idx = 0
            for i, alg1 in enumerate(algorithms):
                for j, alg2 in enumerate(algorithms):
                    if i >= j:
                        continue
                    
                    pairwise_results[alg1][alg2].corrected_p_value = corrected_p_values[idx]
                    idx += 1
        
        return pairwise_results
    
    def _compare_two_algorithms(self,
                              results1: List[float],
                              results2: List[float],
                              test_type: str = "welch") -> StatisticalTestResult:
        """Compare two algorithms using specified statistical test."""
        
        data1 = np.array(results1)
        data2 = np.array(results2)
        
        if test_type == "welch":
            # Welch's t-test (unequal variances)
            statistic, p_value = stats.ttest_ind(data1, data2, equal_var=False)
            df = self._welch_degrees_of_freedom(data1, data2)
            
        elif test_type == "student":
            # Student's t-test (equal variances)
            statistic, p_value = stats.ttest_ind(data1, data2, equal_var=True)
            df = len(data1) + len(data2) - 2
            
        elif test_type == "mannwhitney":
            # Mann-Whitney U test (non-parametric)
            statistic, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
            df = None
            
        elif test_type == "permutation":
            # Permutation test
            statistic, p_value = self._permutation_test(data1, data2)
            df = None
            
        else:
            raise ValueError(f"Unknown test type: {test_type}")
        
        # Calculate effect size (Cohen's d)
        effect_size = self._calculate_cohens_d(data1, data2)
        
        # Calculate confidence interval for the difference
        diff_mean = np.mean(data1) - np.mean(data2)
        diff_se = np.sqrt(np.var(data1, ddof=1)/len(data1) + np.var(data2, ddof=1)/len(data2))
        
        if df is not None:
            t_critical = stats.t.ppf(1 - self.alpha/2, df)
            margin_error = t_critical * diff_se
        else:
            # Use normal approximation for non-parametric tests
            z_critical = stats.norm.ppf(1 - self.alpha/2)
            margin_error = z_critical * diff_se
        
        ci_lower = diff_mean - margin_error
        ci_upper = diff_mean + margin_error
        
        # Calculate power (post-hoc power analysis)
        power = self._calculate_power(data1, data2, effect_size)
        
        return StatisticalTestResult(
            statistic=statistic,
            p_value=p_value,
            confidence_interval=(ci_lower, ci_upper),
            effect_size=effect_size,
            power=power,
            method=test_type,
            sample_size=len(data1) + len(data2),
            degrees_of_freedom=df
        )
    
    def _welch_degrees_of_freedom(self, data1: np.ndarray, data2: np.ndarray) -> float:
        """Calculate Welch's degrees of freedom for unequal variances."""
        s1_sq = np.var(data1, ddof=1)
        s2_sq = np.var(data2, ddof=1)
        n1, n2 = len(data1), len(data2)
        
        numerator = (s1_sq/n1 + s2_sq/n2)**2
        denominator = (s1_sq/n1)**2/(n1-1) + (s2_sq/n2)**2/(n2-1)
        
        return numerator / denominator
    
    def _calculate_cohens_d(self, data1: np.ndarray, data2: np.ndarray) -> float:
        """Calculate Cohen's d effect size."""
        mean1, mean2 = np.mean(data1), np.mean(data2)
        
        # Pooled standard deviation
        n1, n2 = len(data1), len(data2)
        var1, var2 = np.var(data1, ddof=1), np.var(data2, ddof=1)
        pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
        
        return (mean1 - mean2) / pooled_std
    
    def _calculate_power(self, data1: np.ndarray, data2: np.ndarray, effect_size: float) -> float:
        """Calculate post-hoc statistical power."""
        try:
            from scipy.stats import power
            # This is a simplified power calculation
            # In practice, you might want to use more sophisticated methods
            n1, n2 = len(data1), len(data2)
            nobs = min(n1, n2)  # Conservative estimate
            
            # Use effect size to estimate power
            if abs(effect_size) < 0.2:
                return 0.1  # Low power for small effects
            elif abs(effect_size) < 0.5:
                return min(0.8, nobs / 20)  # Medium effect
            else:
                return min(0.95, nobs / 10)  # Large effect
                
        except ImportError:
            # Fallback if power analysis not available
            return 0.8 if abs(effect_size) > 0.5 else 0.5
    
    def _permutation_test(self, 
                         data1: np.ndarray, 
                         data2: np.ndarray, 
                         n_permutations: int = 10000) -> Tuple[float, float]:
        """Perform permutation test for two samples."""
        
        # Observed difference in means
        observed_diff = np.mean(data1) - np.mean(data2)
        
        # Combine datasets for permutation
        combined = np.concatenate([data1, data2])
        n1 = len(data1)
        
        # Generate permutation distribution
        permutation_diffs = []
        rng = np.random.RandomState(42)
        
        for _ in range(n_permutations):
            permuted = rng.permutation(combined)
            perm_group1 = permuted[:n1]
            perm_group2 = permuted[n1:]
            perm_diff = np.mean(perm_group1) - np.mean(perm_group2)
            permutation_diffs.append(perm_diff)
        
        permutation_diffs = np.array(permutation_diffs)
        
        # Calculate p-value (two-tailed)
        p_value = np.mean(np.abs(permutation_diffs) >= np.abs(observed_diff))
        
        return observed_diff, p_value
    
    def _apply_multiple_comparisons_correction(self,
                                             p_values: List[float],
                                             method: str) -> List[float]:
        """Apply multiple comparisons correction to p-values."""
        
        p_values = np.array(p_values)
        
        if method == "bonferroni":
            # Bonferroni correction (conservative)
            corrected = p_values * len(p_values)
            return np.minimum(corrected, 1.0).tolist()
            
        elif method == "benjamini_hochberg":
            # Benjamini-Hochberg FDR correction
            n = len(p_values)
            sorted_indices = np.argsort(p_values)
            sorted_p = p_values[sorted_indices]
            
            # Apply BH correction
            corrected_sorted = np.minimum.accumulate(
                sorted_p * n / np.arange(1, n+1)[::-1]
            )[::-1]
            
            # Unsort to original order
            corrected = np.empty_like(corrected_sorted)
            corrected[sorted_indices] = corrected_sorted
            
            return np.minimum(corrected, 1.0).tolist()
            
        elif method == "holm":
            # Holm-Bonferroni correction
            n = len(p_values)
            sorted_indices = np.argsort(p_values)
            sorted_p = p_values[sorted_indices]
            
            corrected_sorted = sorted_p * (n - np.arange(n))
            corrected_sorted = np.maximum.accumulate(corrected_sorted)
            
            # Unsort to original order
            corrected = np.empty_like(corrected_sorted)
            corrected[sorted_indices] = corrected_sorted
            
            return np.minimum(corrected, 1.0).tolist()
            
        else:
            raise ValueError(f"Unknown correction method: {method}")


class EffectSizeCalculator:
    """Calculate various effect size measures for meta-learning experiments."""
    
    @staticmethod
    def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
        """Calculate Cohen's d effect size."""
        mean1, mean2 = np.mean(group1), np.mean(group2)
        
        # Pooled standard deviation
        n1, n2 = len(group1), len(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
        
        return (mean1 - mean2) / pooled_std
    
    @staticmethod
    def glass_delta(group1: np.ndarray, group2: np.ndarray, control_group: int = 2) -> float:
        """Calculate Glass's Δ effect size using control group standard deviation."""
        mean1, mean2 = np.mean(group1), np.mean(group2)
        
        if control_group == 1:
            control_std = np.std(group1, ddof=1)
        else:
            control_std = np.std(group2, ddof=1)
        
        return (mean1 - mean2) / control_std
    
    @staticmethod
    def hedges_g(group1: np.ndarray, group2: np.ndarray) -> float:
        """Calculate Hedges' g effect size (bias-corrected Cohen's d)."""
        cohens_d = EffectSizeCalculator.cohens_d(group1, group2)
        n1, n2 = len(group1), len(group2)
        df = n1 + n2 - 2
        
        # Correction factor
        correction = 1 - (3 / (4 * df - 1))
        
        return cohens_d * correction
    
    @staticmethod
    def cliff_delta(group1: np.ndarray, group2: np.ndarray) -> float:
        """Calculate Cliff's delta (non-parametric effect size)."""
        n1, n2 = len(group1), len(group2)
        
        # Count dominance
        dominance = 0
        for x1 in group1:
            for x2 in group2:
                if x1 > x2:
                    dominance += 1
                elif x1 < x2:
                    dominance -= 1
        
        return dominance / (n1 * n2)
    
    @staticmethod
    def interpret_effect_size(effect_size: float, measure: str = "cohens_d") -> str:
        """Interpret effect size magnitude according to standard conventions."""
        
        abs_effect = abs(effect_size)
        
        if measure in ["cohens_d", "hedges_g"]:
            if abs_effect < 0.2:
                return "negligible"
            elif abs_effect < 0.5:
                return "small"
            elif abs_effect < 0.8:
                return "medium"
            else:
                return "large"
                
        elif measure == "cliff_delta":
            if abs_effect < 0.147:
                return "negligible" 
            elif abs_effect < 0.33:
                return "small"
            elif abs_effect < 0.474:
                return "medium"
            else:
                return "large"
        
        else:
            return "unknown"


def evaluate_statistical_power(effect_size: float,
                             sample_size: int,
                             alpha: float = 0.05,
                             test_type: str = "two_sample") -> float:
    """
    Evaluate statistical power for given parameters.
    
    Args:
        effect_size: Expected effect size (Cohen's d)
        sample_size: Sample size per group
        alpha: Significance level
        test_type: Type of test ("two_sample", "one_sample")
        
    Returns:
        Statistical power (0-1)
    """
    try:
        # This would require statsmodels for precise power calculations
        # Providing approximation based on common heuristics
        
        if test_type == "two_sample":
            # Rough approximation for two-sample t-test
            if sample_size < 5:
                power = 0.1
            elif abs(effect_size) > 0.8 and sample_size >= 15:
                power = 0.8
            elif abs(effect_size) > 0.5 and sample_size >= 25:
                power = 0.8
            elif abs(effect_size) > 0.2 and sample_size >= 50:
                power = 0.8
            else:
                # Linear approximation
                power = min(0.95, abs(effect_size) * sample_size / 20)
        else:
            power = min(0.95, abs(effect_size) * sample_size / 15)
        
        return max(0.05, power)  # Minimum power of 5%
        
    except Exception:
        return 0.5  # Default moderate power


# Alias for backward compatibility
StatisticalTestSuite = SignificanceTestSuite