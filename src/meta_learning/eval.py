from __future__ import annotations
from typing import Callable, Dict, Any, Iterable, Optional, List, Tuple, Union
import time, json, os, math, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from enum import Enum
from .core.episode import Episode

def _get_t_critical(df: int) -> float:
    """Get t-critical value for 95% confidence interval given degrees of freedom.
    
    This approximates scipy.stats.t.ppf(0.975, df) for common sample sizes.
    For research-grade statistical inference with small samples.
    """
    # Pre-computed t-values for 95% CI (two-tailed, α=0.05)
    t_table = {
        1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36, 8: 2.31, 9: 2.26,
        10: 2.23, 11: 2.20, 12: 2.18, 13: 2.16, 14: 2.14, 15: 2.13, 16: 2.12, 17: 2.11, 
        18: 2.10, 19: 2.09, 20: 2.09, 21: 2.08, 22: 2.07, 23: 2.07, 24: 2.06, 25: 2.06,
        26: 2.06, 27: 2.05, 28: 2.05, 29: 2.05
    }
    
    if df in t_table:
        return t_table[df]
    elif df >= 30:
        return 1.96  # Normal approximation for large samples
    else:
        # Linear interpolation for missing values (simple approximation)
        return 2.78 - (df - 4) * 0.02  # Rough approximation

def evaluate(run_logits: Callable[[Episode], torch.Tensor], episodes: Iterable[Episode], *, outdir: Optional[str] = None, dump_preds: bool = False) -> Dict[str, Any]:
    """Basic evaluation function with statistical confidence intervals."""
    accs = []; preds_dump = []
    t0 = time.time(); episodes = list(episodes)
    for ep in episodes:
        logits = run_logits(ep)
        pred = logits.argmax(dim=1)
        accs.append((pred == ep.query_y).float().mean().item())
        if dump_preds:
            preds_dump.append({"pred": pred.tolist(), "y": ep.query_y.tolist()})
    dt = time.time()-t0
    n = len(accs); mean = sum(accs)/n
    var = sum((x-mean)**2 for x in accs)/(n-1 if n>1 else 1); sd = var**0.5
    
    # Use t-distribution for small sample sizes (research-grade statistics)
    if n > 1:
        if n < 30:
            t_critical = _get_t_critical(n-1) 
        else:
            t_critical = 1.96  # Normal approximation for large samples
        
        ci = t_critical * sd / math.sqrt(n)
        se = sd / math.sqrt(n)  # Standard error
    else:
        ci = 0.0
        se = 0.0
        
    res = {"episodes": n, "mean_acc": mean, "ci95": ci, "std_err": se, "elapsed_s": dt}
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        with open(os.path.join(outdir, "metrics.json"), "w") as f:
            json.dump(res, f, indent=2)
        if dump_preds:
            with open(os.path.join(outdir, "preds.jsonl"), "w") as f:
                for p in preds_dump: f.write(json.dumps(p)+"\n")
    return res

# Evaluation metrics enum
class MetricType(Enum):
    ACCURACY = "accuracy"
    LOSS = "loss" 
    F1_SCORE = "f1_score"
    PRECISION = "precision"
    RECALL = "recall"
    AUC = "auc"

@dataclass
class EvaluationConfig:
    """Configuration for meta-learning evaluation."""
    metrics: List[MetricType]
    confidence_level: float = 0.95
    bootstrap_samples: int = 1000
    statistical_tests: List[str] = None
    visualization: bool = False
    save_predictions: bool = False
    
    def __post_init__(self):
        if self.statistical_tests is None:
            self.statistical_tests = ["t_test", "wilcoxon"]

class Accuracy:
    """Advanced accuracy computation with uncertainty estimation."""
    
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        
    def compute(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """Compute accuracy with confidence intervals."""
        correct = (predictions == targets).float()
        accuracy = correct.mean().item()
        n = len(correct)
        
        if n > 1:
            std = correct.std().item()
            se = std / math.sqrt(n)
            
            # Use t-distribution for small samples
            if n < 30:
                t_critical = _get_t_critical(n-1)
            else:
                t_critical = 1.96
                
            ci = t_critical * se
        else:
            ci = 0.0
            se = 0.0
            
        return {
            "accuracy": accuracy,
            "std_error": se,
            "confidence_interval": ci,
            "n_samples": n
        }
    
    def bootstrap_confidence_interval(self, predictions: torch.Tensor, targets: torch.Tensor, n_bootstrap: int = 1000) -> Tuple[float, float]:
        """Bootstrap confidence interval for accuracy."""
        correct = (predictions == targets).float()
        n = len(correct)
        bootstrap_accs = []
        
        for _ in range(n_bootstrap):
            indices = torch.randint(0, n, (n,))
            bootstrap_acc = correct[indices].mean().item()
            bootstrap_accs.append(bootstrap_acc)
            
        bootstrap_accs = np.array(bootstrap_accs)
        alpha = 1 - self.confidence_level
        lower = np.percentile(bootstrap_accs, 100 * alpha / 2)
        upper = np.percentile(bootstrap_accs, 100 * (1 - alpha / 2))
        
        return lower, upper

class UncertaintyEvaluator:
    """Evaluator for uncertainty quantification in meta-learning."""
    
    def __init__(self):
        self.entropy_threshold = 1.0
        
    def compute_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute predictive entropy."""
        probs = F.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
        return entropy
    
    def compute_confidence(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute prediction confidence (max probability)."""
        probs = F.softmax(logits, dim=-1)
        confidence = torch.max(probs, dim=-1)[0]
        return confidence
    
    def calibration_analysis(self, logits: torch.Tensor, targets: torch.Tensor, n_bins: int = 10) -> Dict[str, Any]:
        """Compute calibration metrics."""
        probs = F.softmax(logits, dim=-1)
        confidences = torch.max(probs, dim=-1)[0]
        predictions = torch.argmax(probs, dim=-1)
        accuracies = (predictions == targets).float()
        
        # Bin by confidence
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.float().mean().item()
            bin_counts.append(prop_in_bin)
            
            if in_bin.any():
                accuracy_in_bin = accuracies[in_bin].mean().item()
                avg_confidence_in_bin = confidences[in_bin].mean().item()
                bin_accuracies.append(accuracy_in_bin)
                bin_confidences.append(avg_confidence_in_bin)
            else:
                bin_accuracies.append(0.0)
                bin_confidences.append(0.0)
        
        # Expected Calibration Error (ECE)
        ece = sum(count * abs(acc - conf) for count, acc, conf in zip(bin_counts, bin_accuracies, bin_confidences))
        
        return {
            "ece": ece,
            "bin_accuracies": bin_accuracies,
            "bin_confidences": bin_confidences,
            "bin_counts": bin_counts
        }

class StatisticalTestSuite:
    """Statistical testing suite for meta-learning evaluation."""
    
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        
    def paired_t_test(self, scores1: List[float], scores2: List[float]) -> Dict[str, float]:
        """Paired t-test for comparing two algorithms."""
        if len(scores1) != len(scores2):
            raise ValueError("Score lists must have same length")
            
        differences = np.array(scores1) - np.array(scores2)
        n = len(differences)
        
        if n < 2:
            return {"t_statistic": 0.0, "p_value": 1.0, "significant": False}
            
        mean_diff = np.mean(differences)
        std_diff = np.std(differences, ddof=1)
        se_diff = std_diff / np.sqrt(n)
        
        t_statistic = mean_diff / se_diff if se_diff > 0 else 0.0
        
        # Approximate p-value using t-distribution
        df = n - 1
        t_critical = _get_t_critical(df)
        p_value = 2 * (1 - self._t_cdf(abs(t_statistic), df))
        
        return {
            "t_statistic": t_statistic,
            "p_value": p_value,
            "significant": p_value < self.alpha,
            "mean_difference": mean_diff
        }
    
    def _t_cdf(self, t: float, df: int) -> float:
        """Approximate t-distribution CDF."""
        # Simple approximation - in practice would use scipy
        if df >= 30:
            # Normal approximation
            return 0.5 * (1 + math.erf(t / math.sqrt(2)))
        else:
            # Rough approximation for t-distribution
            return 0.5 * (1 + math.erf(t / math.sqrt(2 * (1 + t**2 / df))))
    
    def wilcoxon_signed_rank_test(self, scores1: List[float], scores2: List[float]) -> Dict[str, float]:
        """Wilcoxon signed-rank test (approximate implementation)."""
        differences = np.array(scores1) - np.array(scores2)
        differences = differences[differences != 0]  # Remove zeros
        
        if len(differences) == 0:
            return {"w_statistic": 0.0, "p_value": 1.0, "significant": False}
        
        abs_diff = np.abs(differences)
        ranks = np.argsort(np.argsort(abs_diff)) + 1  # Simple ranking
        
        signed_ranks = ranks * np.sign(differences)
        w_statistic = np.sum(signed_ranks[signed_ranks > 0])
        
        # Approximate p-value (simplified)
        n = len(differences)
        expected_w = n * (n + 1) / 4
        var_w = n * (n + 1) * (2 * n + 1) / 24
        
        if var_w > 0:
            z = (w_statistic - expected_w) / np.sqrt(var_w)
            p_value = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
        else:
            p_value = 1.0
            
        return {
            "w_statistic": w_statistic,
            "p_value": p_value,
            "significant": p_value < self.alpha
        }

class LearnabilityAnalyzer:
    """Analyzer for task learnability and difficulty."""
    
    def __init__(self):
        self.feature_cache = {}
        
    def compute_task_difficulty(self, episode: Episode) -> Dict[str, float]:
        """Compute various difficulty metrics for a task."""
        support_x, support_y = episode.support_x, episode.support_y
        query_x, query_y = episode.query_x, episode.query_y
        
        # Class balance
        class_counts = torch.bincount(support_y)
        class_balance = (class_counts.min().float() / class_counts.max().float()).item()
        
        # Feature diversity (approximate)
        support_flat = support_x.view(support_x.size(0), -1)
        pairwise_distances = torch.pdist(support_flat)
        avg_distance = pairwise_distances.mean().item()
        
        # Intra-class variance
        intra_class_var = 0.0
        n_classes = len(class_counts)
        for class_idx in range(n_classes):
            class_mask = support_y == class_idx
            if class_mask.sum() > 1:
                class_samples = support_flat[class_mask]
                class_var = torch.var(class_samples, dim=0).mean().item()
                intra_class_var += class_var
        intra_class_var /= n_classes
        
        # Inter-class separation
        class_centers = []
        for class_idx in range(n_classes):
            class_mask = support_y == class_idx
            if class_mask.sum() > 0:
                class_center = support_flat[class_mask].mean(dim=0)
                class_centers.append(class_center)
        
        inter_class_sep = 0.0
        if len(class_centers) > 1:
            class_centers = torch.stack(class_centers)
            inter_class_distances = torch.pdist(class_centers)
            inter_class_sep = inter_class_distances.mean().item()
        
        return {
            "class_balance": class_balance,
            "avg_feature_distance": avg_distance,
            "intra_class_variance": intra_class_var,
            "inter_class_separation": inter_class_sep,
            "difficulty_score": intra_class_var / (inter_class_sep + 1e-8)
        }
    
    def analyze_few_shot_complexity(self, episodes: List[Episode]) -> Dict[str, Any]:
        """Analyze complexity across multiple few-shot episodes."""
        difficulties = []
        class_balances = []
        
        for episode in episodes:
            metrics = self.compute_task_difficulty(episode)
            difficulties.append(metrics["difficulty_score"])
            class_balances.append(metrics["class_balance"])
        
        return {
            "mean_difficulty": np.mean(difficulties),
            "std_difficulty": np.std(difficulties),
            "mean_class_balance": np.mean(class_balances),
            "difficulty_distribution": difficulties
        }

class MetaLearningMetrics:
    """Comprehensive metrics for meta-learning evaluation."""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.accuracy = Accuracy(config.confidence_level)
        self.uncertainty = UncertaintyEvaluator()
        self.stats = StatisticalTestSuite()
        self.learnability = LearnabilityAnalyzer()
        
    def compute_episode_metrics(self, logits: torch.Tensor, targets: torch.Tensor, episode: Episode) -> Dict[str, Any]:
        """Compute all metrics for a single episode."""
        predictions = logits.argmax(dim=-1)
        
        results = {}
        
        # Basic accuracy
        if MetricType.ACCURACY in self.config.metrics:
            acc_results = self.accuracy.compute(predictions, targets)
            results.update(acc_results)
        
        # Loss
        if MetricType.LOSS in self.config.metrics:
            loss = F.cross_entropy(logits, targets)
            results["loss"] = loss.item()
        
        # Uncertainty metrics
        entropy = self.uncertainty.compute_entropy(logits)
        confidence = self.uncertainty.compute_confidence(logits)
        results["mean_entropy"] = entropy.mean().item()
        results["mean_confidence"] = confidence.mean().item()
        
        # Calibration
        calibration = self.uncertainty.calibration_analysis(logits, targets)
        results["calibration"] = calibration
        
        # Task difficulty
        difficulty = self.learnability.compute_task_difficulty(episode)
        results["task_difficulty"] = difficulty
        
        return results

class EvaluationVisualizer:
    """Visualization tools for meta-learning evaluation."""
    
    def __init__(self):
        self.figures = {}
        
    def plot_accuracy_distribution(self, accuracies: List[float], title: str = "Accuracy Distribution") -> None:
        """Plot accuracy distribution."""
        # Placeholder for visualization - would use matplotlib in practice
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        print(f"{title}: Mean={mean_acc:.3f}, Std={std_acc:.3f}")
        
    def plot_calibration_curve(self, calibration_results: Dict[str, Any]) -> None:
        """Plot calibration curve."""
        # Placeholder for calibration plot
        ece = calibration_results["ece"]
        print(f"Expected Calibration Error: {ece:.3f}")
        
    def plot_learning_curves(self, metrics_over_time: List[Dict[str, float]]) -> None:
        """Plot learning curves over episodes."""
        # Placeholder for learning curve visualization
        if metrics_over_time:
            final_acc = metrics_over_time[-1].get("accuracy", 0.0)
            print(f"Final accuracy: {final_acc:.3f}")

class MetaLearningEvaluator:
    """Main evaluator class for comprehensive meta-learning evaluation."""
    
    def __init__(self, config: Optional[EvaluationConfig] = None):
        if config is None:
            config = EvaluationConfig(
                metrics=[MetricType.ACCURACY, MetricType.LOSS],
                confidence_level=0.95,
                bootstrap_samples=1000
            )
        
        self.config = config
        self.metrics = MetaLearningMetrics(config)
        self.visualizer = EvaluationVisualizer()
        
    def evaluate_episodes(self, run_logits: Callable[[Episode], torch.Tensor], episodes: List[Episode]) -> Dict[str, Any]:
        """Evaluate model on multiple episodes."""
        all_results = []
        all_accuracies = []
        
        start_time = time.time()
        
        for i, episode in enumerate(episodes):
            # Get model predictions
            logits = run_logits(episode)
            
            # Compute metrics
            episode_results = self.metrics.compute_episode_metrics(logits, episode.query_y, episode)
            episode_results["episode_idx"] = i
            all_results.append(episode_results)
            
            if "accuracy" in episode_results:
                all_accuracies.append(episode_results["accuracy"])
        
        elapsed_time = time.time() - start_time
        
        # Aggregate results
        aggregate_results = self._aggregate_results(all_results)
        aggregate_results["elapsed_time"] = elapsed_time
        aggregate_results["n_episodes"] = len(episodes)
        
        # Statistical analysis
        if len(all_accuracies) > 1:
            mean_acc = np.mean(all_accuracies)
            std_acc = np.std(all_accuracies, ddof=1)
            se_acc = std_acc / np.sqrt(len(all_accuracies))
            
            # Confidence interval
            if len(all_accuracies) < 30:
                t_critical = _get_t_critical(len(all_accuracies) - 1)
            else:
                t_critical = 1.96
            
            ci = t_critical * se_acc
            
            aggregate_results.update({
                "mean_accuracy": mean_acc,
                "std_accuracy": std_acc,
                "se_accuracy": se_acc,
                "ci95_accuracy": ci
            })
        
        # Visualization
        if self.config.visualization and all_accuracies:
            self.visualizer.plot_accuracy_distribution(all_accuracies)
        
        return {
            "aggregate": aggregate_results,
            "episodes": all_results if self.config.save_predictions else []
        }
    
    def _aggregate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results across episodes."""
        aggregate = {}
        
        # Aggregate numerical metrics
        numerical_keys = ["accuracy", "loss", "mean_entropy", "mean_confidence"]
        for key in numerical_keys:
            values = [r[key] for r in results if key in r]
            if values:
                aggregate[f"mean_{key}"] = np.mean(values)
                aggregate[f"std_{key}"] = np.std(values)
        
        # Aggregate task difficulty
        difficulty_scores = [r["task_difficulty"]["difficulty_score"] for r in results if "task_difficulty" in r]
        if difficulty_scores:
            aggregate["mean_task_difficulty"] = np.mean(difficulty_scores)
            aggregate["std_task_difficulty"] = np.std(difficulty_scores)
        
        return aggregate

class TorchMetaEvaluationHarness:
    """Evaluation harness compatible with TorchMeta datasets."""
    
    def __init__(self, evaluator: MetaLearningEvaluator):
        self.evaluator = evaluator
        
    def evaluate_torchmeta_dataset(self, model: nn.Module, dataset, num_episodes: int = 600) -> Dict[str, Any]:
        """Evaluate model on TorchMeta dataset format."""
        episodes = []
        
        # Convert TorchMeta format to our Episode format
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
        
        for i, batch in enumerate(dataloader):
            if i >= num_episodes:
                break
                
            # Extract support and query sets from TorchMeta batch
            # This is a simplified version - actual implementation would depend on TorchMeta structure
            support_x = batch["train"][0][0]  # Assuming TorchMeta structure
            support_y = batch["train"][1][0]
            query_x = batch["test"][0][0]
            query_y = batch["test"][1][0]
            
            episode = Episode(
                support_x=support_x,
                support_y=support_y,
                query_x=query_x,
                query_y=query_y
            )
            episodes.append(episode)
        
        # Define run_logits function
        def run_logits(ep: Episode) -> torch.Tensor:
            model.eval()
            with torch.no_grad():
                return model(ep.query_x)
        
        return self.evaluator.evaluate_episodes(run_logits, episodes)

def evaluate_multiple_seeds(
    model_fn: Callable[[], nn.Module],
    episode_fn: Callable[[], List[Episode]], 
    seeds: List[int],
    config: Optional[EvaluationConfig] = None
) -> Dict[str, Any]:
    """Evaluate across multiple random seeds."""
    all_results = []
    
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        model = model_fn()
        episodes = episode_fn()
        
        evaluator = MetaLearningEvaluator(config)
        
        def run_logits(ep: Episode) -> torch.Tensor:
            model.eval()
            with torch.no_grad():
                return model(ep.query_x)
        
        results = evaluator.evaluate_episodes(run_logits, episodes)
        results["seed"] = seed
        all_results.append(results)
    
    # Aggregate across seeds
    seed_accuracies = [r["aggregate"]["mean_accuracy"] for r in all_results if "mean_accuracy" in r["aggregate"]]
    
    if seed_accuracies:
        mean_acc = np.mean(seed_accuracies)
        std_acc = np.std(seed_accuracies, ddof=1)
        se_acc = std_acc / np.sqrt(len(seed_accuracies))
        
        if len(seed_accuracies) < 30:
            t_critical = _get_t_critical(len(seed_accuracies) - 1)
        else:
            t_critical = 1.96
            
        ci = t_critical * se_acc
        
        return {
            "mean_accuracy": mean_acc,
            "std_accuracy": std_acc,
            "se_accuracy": se_acc,
            "ci95_accuracy": ci,
            "n_seeds": len(seeds),
            "seed_results": all_results
        }
    else:
        return {"seed_results": all_results}

class MetaLearningCrossValidator:
    """Cross-validation for meta-learning evaluation."""
    
    def __init__(self, n_folds: int = 5, shuffle: bool = True):
        self.n_folds = n_folds
        self.shuffle = shuffle
        
    def cross_validate(self, 
                      model_fn: Callable[[], nn.Module],
                      episodes: List[Episode],
                      config: Optional[EvaluationConfig] = None) -> Dict[str, Any]:
        """Perform cross-validation on episodes."""
        if self.shuffle:
            indices = np.random.permutation(len(episodes))
            episodes = [episodes[i] for i in indices]
        
        fold_size = len(episodes) // self.n_folds
        fold_results = []
        
        for fold in range(self.n_folds):
            start_idx = fold * fold_size
            end_idx = (fold + 1) * fold_size if fold < self.n_folds - 1 else len(episodes)
            
            test_episodes = episodes[start_idx:end_idx]
            train_episodes = episodes[:start_idx] + episodes[end_idx:]
            
            # Create and train model (simplified - would need actual training logic)
            model = model_fn()
            
            # Evaluate on test fold
            evaluator = MetaLearningEvaluator(config)
            
            def run_logits(ep: Episode) -> torch.Tensor:
                model.eval()
                with torch.no_grad():
                    return model(ep.query_x)
            
            fold_result = evaluator.evaluate_episodes(run_logits, test_episodes)
            fold_result["fold"] = fold
            fold_results.append(fold_result)
        
        # Aggregate across folds
        fold_accuracies = [r["aggregate"]["mean_accuracy"] for r in fold_results if "mean_accuracy" in r["aggregate"]]
        
        if fold_accuracies:
            mean_cv_acc = np.mean(fold_accuracies)
            std_cv_acc = np.std(fold_accuracies, ddof=1)
            
            return {
                "cv_mean_accuracy": mean_cv_acc,
                "cv_std_accuracy": std_cv_acc,
                "fold_results": fold_results
            }
        else:
            return {"fold_results": fold_results}

def comprehensive_evaluate(
    model: nn.Module,
    episodes: List[Episode],
    seeds: Optional[List[int]] = None,
    cross_validate: bool = False,
    config: Optional[EvaluationConfig] = None
) -> Dict[str, Any]:
    """Comprehensive evaluation with multiple methods."""
    
    if config is None:
        config = EvaluationConfig(
            metrics=[MetricType.ACCURACY, MetricType.LOSS],
            confidence_level=0.95,
            statistical_tests=["t_test", "wilcoxon"],
            visualization=True
        )
    
    results = {}
    
    # Basic evaluation
    evaluator = MetaLearningEvaluator(config)
    
    def run_logits(ep: Episode) -> torch.Tensor:
        model.eval()
        with torch.no_grad():
            return model(ep.query_x)
    
    basic_results = evaluator.evaluate_episodes(run_logits, episodes)
    results["basic_evaluation"] = basic_results
    
    # Multi-seed evaluation
    if seeds:
        def model_fn():
            return model
        def episode_fn():
            return episodes
        
        seed_results = evaluate_multiple_seeds(model_fn, episode_fn, seeds, config)
        results["multi_seed_evaluation"] = seed_results
    
    # Cross-validation
    if cross_validate:
        cv = MetaLearningCrossValidator()
        def model_fn():
            return model
        
        cv_results = cv.cross_validate(model_fn, episodes, config)
        results["cross_validation"] = cv_results
    
    return results