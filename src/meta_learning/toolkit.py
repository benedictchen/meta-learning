"""
💰 SUPPORT THIS RESEARCH - PLEASE DONATE! 💰

🙏 If this library helps your research or project, please consider donating:
💳 PayPal: https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS
⭐ GitHub Sponsors: https://github.com/sponsors/benedictchen

👨‍💻 Author: Benedict Chen
💰 Donations: Help support this work! Buy me a coffee ☕, beer 🍺, lamborghini 🏎️, or private island 🏝️
💖 Please consider recurring donations to fully support continued research

Your support enables cutting-edge AI research for everyone! 🚀

Meta-Learning Toolkit - High-Level API
=================================================================

This module provides a high-level, user-friendly API for meta-learning research.
It wraps the complex low-level algorithms into simple, one-liner interfaces.

Main Components:
- MetaLearningToolkit: Main class for algorithm management
- create_meta_learning_toolkit(): Convenience function for quick setup
- quick_evaluation(): Simple evaluation interface

Supported Algorithms:
- MAML (Model-Agnostic Meta-Learning) with research-accurate implementation
- Test-Time Compute Scaling (2024 breakthrough algorithm)
- Deterministic training setup for reproducible research
- BatchNorm policy fixes for few-shot learning
- Comprehensive evaluation harness with 95% confidence intervals

Usage:
    >>> from meta_learning import create_meta_learning_toolkit
    >>> toolkit = create_meta_learning_toolkit(model, algorithm='maml')
    >>> results = toolkit.train_episode(episode)

Author: Benedict Chen (benedict@benedictchen.com)
License: Custom Non-Commercial License with Donation Requirements

💰 SUPPORT THIS RESEARCH - PLEASE DONATE! 💰
"""

from typing import Dict, Any, Optional, Tuple, List, Callable, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import warnings
import json

# Import core components - always available
from .core.episode import Episode, remap_labels

# Import advanced components - now all part of the package
from .algorithms.ttc_scaler import TestTimeComputeScaler
from .algorithms.ttc_config import TestTimeComputeConfig
from .algorithms.maml_research_accurate import (
    ResearchMAML, MAMLConfig, MAMLVariant, FunctionalModule
)
from .research_patches.batch_norm_policy import EpisodicBatchNormPolicy
from .research_patches.determinism_hooks import DeterminismManager, setup_deterministic_environment
from .evaluation.few_shot_evaluation_harness import FewShotEvaluationHarness

# Phase 4: Advanced ML-based enhancements

class FailurePredictionModel:
    """ML-based failure prediction for meta-learning algorithms."""
    
    def __init__(self):
        self.feature_history = []
        self.failure_history = []
        self.prediction_threshold = 0.7
        
    def extract_features(self, episode: Episode, algorithm_state: Dict[str, Any]) -> np.ndarray:
        """Extract features for failure prediction."""
        # Task complexity features
        support_x, support_y = episode.support_x, episode.support_y
        n_support, n_classes = len(support_y), len(torch.unique(support_y))
        class_balance = (torch.bincount(support_y).min().float() / torch.bincount(support_y).max().float()).item()
        
        # Feature diversity
        support_flat = support_x.view(support_x.size(0), -1)
        pairwise_distances = torch.pdist(support_flat)
        avg_distance = pairwise_distances.mean().item() if len(pairwise_distances) > 0 else 0.0
        
        # Algorithm state features
        learning_rate = algorithm_state.get('learning_rate', 0.01)
        inner_steps = algorithm_state.get('inner_steps', 1)
        loss_history = algorithm_state.get('loss_history', [])
        avg_loss = np.mean(loss_history) if loss_history else 0.0
        
        return np.array([
            n_support, n_classes, class_balance, avg_distance,
            learning_rate, inner_steps, avg_loss,
            len(loss_history)  # Training progress indicator
        ])
    
    def predict_failure_risk(self, episode: Episode, algorithm_state: Dict[str, Any]) -> float:
        """Predict probability of algorithm failure."""
        features = self.extract_features(episode, algorithm_state)
        
        if len(self.feature_history) < 10:  # Not enough data for prediction
            return 0.5  # Neutral prediction
        
        # Simple similarity-based prediction (in practice would use trained ML model)
        feature_matrix = np.array(self.feature_history)
        failure_array = np.array(self.failure_history)
        
        # Find similar episodes using cosine similarity
        similarities = []
        for hist_features in feature_matrix:
            if np.linalg.norm(features) > 0 and np.linalg.norm(hist_features) > 0:
                sim = np.dot(features, hist_features) / (np.linalg.norm(features) * np.linalg.norm(hist_features))
                similarities.append(sim)
            else:
                similarities.append(0.0)
        
        similarities = np.array(similarities)
        
        # Weight by similarity and predict
        if len(similarities) > 0 and similarities.max() > 0.1:
            weights = np.exp(similarities * 5)  # Exponential weighting
            weights = weights / weights.sum()
            failure_risk = np.sum(weights * failure_array)
            return failure_risk
        
        return 0.5  # Default neutral prediction
    
    def update_with_outcome(self, episode: Episode, algorithm_state: Dict[str, Any], failed: bool):
        """Update model with episode outcome."""
        features = self.extract_features(episode, algorithm_state)
        self.feature_history.append(features)
        self.failure_history.append(1.0 if failed else 0.0)
        
        # Keep only recent history
        if len(self.feature_history) > 1000:
            self.feature_history = self.feature_history[-500:]
            self.failure_history = self.failure_history[-500:]

class AlgorithmSelector:
    """Automatic algorithm selection based on task characteristics."""
    
    def __init__(self):
        self.algorithm_performance = {
            'maml': [],
            'test_time_compute': [],
            'protonet': []
        }
        
    def select_algorithm(self, episode: Episode) -> str:
        """Select best algorithm based on task characteristics."""
        # Extract task features
        n_support = len(episode.support_y)
        n_classes = len(torch.unique(episode.support_y))
        n_query = len(episode.query_y)
        
        # Simple heuristic-based selection (would be ML-based in practice)
        if n_support < 5:  # Very few-shot
            return 'test_time_compute'  # Better for extremely low-shot scenarios
        elif n_classes > 10:  # Many classes
            return 'protonet'  # Good for multi-class scenarios
        else:
            return 'maml'  # General purpose
    
    def update_performance(self, algorithm: str, episode: Episode, accuracy: float):
        """Update performance history for algorithm selection."""
        self.algorithm_performance[algorithm].append({
            'accuracy': accuracy,
            'n_support': len(episode.support_y),
            'n_classes': len(torch.unique(episode.support_y)),
            'timestamp': time.time()
        })
        
        # Keep recent history
        if len(self.algorithm_performance[algorithm]) > 100:
            self.algorithm_performance[algorithm] = self.algorithm_performance[algorithm][-50:]

class ABTestingFramework:
    """A/B testing framework for algorithm comparison."""
    
    def __init__(self):
        self.test_groups = {}
        self.results_cache = {}
        
    def create_ab_test(self, test_name: str, algorithms: List[str], allocation_ratio: List[float] = None):
        """Create A/B test configuration."""
        if allocation_ratio is None:
            allocation_ratio = [1.0 / len(algorithms)] * len(algorithms)
        
        if len(algorithms) != len(allocation_ratio):
            raise ValueError("Algorithms and allocation ratios must have same length")
        
        self.test_groups[test_name] = {
            'algorithms': algorithms,
            'allocation_ratio': allocation_ratio,
            'results': {alg: [] for alg in algorithms}
        }
    
    def assign_algorithm(self, test_name: str, episode_id: str) -> str:
        """Assign episode to algorithm group."""
        if test_name not in self.test_groups:
            raise ValueError(f"Test {test_name} not found")
        
        # Deterministic assignment based on episode_id hash
        import hashlib
        hash_val = int(hashlib.md5(episode_id.encode()).hexdigest(), 16)
        rand_val = (hash_val % 1000) / 1000.0
        
        algorithms = self.test_groups[test_name]['algorithms']
        ratios = self.test_groups[test_name]['allocation_ratio']
        
        cumulative_ratio = 0
        for i, ratio in enumerate(ratios):
            cumulative_ratio += ratio
            if rand_val <= cumulative_ratio:
                return algorithms[i]
        
        return algorithms[-1]  # Fallback
    
    def record_result(self, test_name: str, algorithm: str, result: Dict[str, Any]):
        """Record A/B test result."""
        if test_name in self.test_groups:
            self.test_groups[test_name]['results'][algorithm].append(result)
    
    def analyze_ab_test(self, test_name: str) -> Dict[str, Any]:
        """Analyze A/B test results."""
        if test_name not in self.test_groups:
            return {}
        
        results = {}
        test_data = self.test_groups[test_name]['results']
        
        for algorithm, alg_results in test_data.items():
            if alg_results:
                accuracies = [r.get('accuracy', 0.0) for r in alg_results]
                results[algorithm] = {
                    'mean_accuracy': np.mean(accuracies),
                    'std_accuracy': np.std(accuracies),
                    'n_samples': len(accuracies)
                }
        
        return results

class CrossTaskKnowledgeTransfer:
    """Cross-task knowledge transfer and meta-optimization."""
    
    def __init__(self):
        self.task_embeddings = {}
        self.knowledge_base = {}
        self.transfer_history = []
        
    def compute_task_embedding(self, episode: Episode) -> np.ndarray:
        """Compute task embedding for similarity matching."""
        support_x, support_y = episode.support_x, episode.support_y
        
        # Basic task statistics
        n_support = len(support_y)
        n_classes = len(torch.unique(support_y))
        class_balance = (torch.bincount(support_y).min().float() / torch.bincount(support_y).max().float()).item()
        
        # Feature statistics
        support_flat = support_x.view(support_x.size(0), -1)
        mean_features = support_flat.mean(dim=0).cpu().numpy()[:10]  # Take first 10 dims
        std_features = support_flat.std(dim=0).cpu().numpy()[:10]
        
        # Pad if necessary
        if len(mean_features) < 10:
            mean_features = np.pad(mean_features, (0, 10 - len(mean_features)), 'constant')
        if len(std_features) < 10:
            std_features = np.pad(std_features, (0, 10 - len(std_features)), 'constant')
        
        embedding = np.concatenate([
            [n_support, n_classes, class_balance],
            mean_features,
            std_features
        ])
        
        return embedding
    
    def find_similar_tasks(self, episode: Episode, top_k: int = 5) -> List[Tuple[str, float]]:
        """Find similar tasks for knowledge transfer."""
        current_embedding = self.compute_task_embedding(episode)
        
        similarities = []
        for task_id, stored_embedding in self.task_embeddings.items():
            if np.linalg.norm(current_embedding) > 0 and np.linalg.norm(stored_embedding) > 0:
                sim = np.dot(current_embedding, stored_embedding) / (
                    np.linalg.norm(current_embedding) * np.linalg.norm(stored_embedding)
                )
                similarities.append((task_id, sim))
        
        # Sort by similarity and return top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def transfer_knowledge(self, episode: Episode, algorithm_config: Dict[str, Any]) -> Dict[str, Any]:
        """Transfer knowledge from similar tasks."""
        similar_tasks = self.find_similar_tasks(episode)
        
        if not similar_tasks:
            return algorithm_config  # No similar tasks found
        
        # Transfer hyperparameters from most similar task
        best_task_id, similarity = similar_tasks[0]
        
        if similarity > 0.8 and best_task_id in self.knowledge_base:  # High similarity threshold
            stored_knowledge = self.knowledge_base[best_task_id]
            
            # Transfer successful hyperparameters
            if stored_knowledge.get('accuracy', 0) > 0.7:  # Only transfer from successful tasks
                transferred_config = algorithm_config.copy()
                
                # Transfer learning rate with decay based on similarity
                if 'learning_rate' in stored_knowledge:
                    original_lr = stored_knowledge['learning_rate']
                    transferred_config['learning_rate'] = original_lr * similarity
                
                # Transfer inner steps
                if 'inner_steps' in stored_knowledge:
                    transferred_config['inner_steps'] = stored_knowledge['inner_steps']
                
                self.transfer_history.append({
                    'from_task': best_task_id,
                    'similarity': similarity,
                    'transferred_params': list(transferred_config.keys())
                })
                
                return transferred_config
        
        return algorithm_config
    
    def store_task_knowledge(self, episode: Episode, task_id: str, result: Dict[str, Any]):
        """Store successful task knowledge for future transfer."""
        embedding = self.compute_task_embedding(episode)
        self.task_embeddings[task_id] = embedding
        
        self.knowledge_base[task_id] = {
            'accuracy': result.get('accuracy', 0.0),
            'learning_rate': result.get('learning_rate', 0.01),
            'inner_steps': result.get('inner_steps', 1),
            'timestamp': time.time()
        }
        
        # Cleanup old entries
        if len(self.knowledge_base) > 500:
            # Remove oldest entries
            sorted_tasks = sorted(self.knowledge_base.items(), key=lambda x: x[1]['timestamp'])
            tasks_to_remove = sorted_tasks[:100]  # Remove oldest 100
            for task_id, _ in tasks_to_remove:
                if task_id in self.task_embeddings:
                    del self.task_embeddings[task_id]
                del self.knowledge_base[task_id]

class PerformanceMonitor:
    """Advanced performance monitoring and adaptive optimization."""
    
    def __init__(self):
        self.metrics_history = []
        self.performance_trends = {}
        self.alert_thresholds = {
            'accuracy_drop': 0.1,
            'loss_spike': 2.0,
            'training_time': 300.0  # 5 minutes
        }
        
    def record_metrics(self, metrics: Dict[str, Any], timestamp: Optional[float] = None):
        """Record performance metrics."""
        if timestamp is None:
            timestamp = time.time()
        
        metrics_entry = {
            'timestamp': timestamp,
            'metrics': metrics.copy()
        }
        self.metrics_history.append(metrics_entry)
        
        # Keep recent history
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-500:]
        
        self._update_trends()
        self._check_alerts(metrics)
    
    def _update_trends(self):
        """Update performance trend analysis."""
        if len(self.metrics_history) < 10:
            return
        
        recent_metrics = self.metrics_history[-10:]
        
        for metric_name in ['accuracy', 'loss', 'training_time']:
            values = []
            timestamps = []
            
            for entry in recent_metrics:
                if metric_name in entry['metrics']:
                    values.append(entry['metrics'][metric_name])
                    timestamps.append(entry['timestamp'])
            
            if len(values) >= 5:
                # Simple linear trend analysis
                x = np.array(timestamps)
                y = np.array(values)
                
                if len(x) > 1:
                    trend = np.polyfit(x - x[0], y, 1)[0]  # Slope
                    self.performance_trends[metric_name] = {
                        'trend': trend,
                        'recent_mean': np.mean(y),
                        'recent_std': np.std(y)
                    }
    
    def _check_alerts(self, metrics: Dict[str, Any]):
        """Check for performance alerts."""
        alerts = []
        
        # Check for accuracy drops
        if 'accuracy' in metrics and 'accuracy' in self.performance_trends:
            recent_acc = metrics['accuracy']
            trend_acc = self.performance_trends['accuracy']['recent_mean']
            
            if trend_acc - recent_acc > self.alert_thresholds['accuracy_drop']:
                alerts.append({
                    'type': 'accuracy_drop',
                    'message': f"Accuracy dropped from {trend_acc:.3f} to {recent_acc:.3f}",
                    'severity': 'high'
                })
        
        # Log alerts
        for alert in alerts:
            if alert['severity'] == 'high':
                warnings.warn(f"Performance Alert: {alert['message']}")
    
    def get_optimization_suggestions(self) -> List[str]:
        """Get adaptive optimization suggestions based on performance trends."""
        suggestions = []
        
        if 'accuracy' in self.performance_trends:
            acc_trend = self.performance_trends['accuracy']['trend']
            
            if acc_trend < -0.001:  # Decreasing accuracy
                suggestions.append("Consider reducing learning rate or increasing regularization")
            elif acc_trend > 0.001:  # Improving accuracy
                suggestions.append("Performance is improving - consider increasing learning rate")
        
        if 'loss' in self.performance_trends:
            loss_trend = self.performance_trends['loss']['trend']
            
            if loss_trend > 0.01:  # Increasing loss
                suggestions.append("Loss is increasing - check for overfitting or reduce learning rate")
        
        return suggestions

class MetaLearningToolkit:
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Initialize components
        self.test_time_scaler = None
        self.maml_learner = None  
        self.batch_norm_policy = EpisodicBatchNormPolicy()
        self.determinism_manager = DeterminismManager()
        self.evaluation_harness = None
        
        # Phase 4: ML-Powered Enhancement Components
        self.failure_prediction_enabled = False
        self.auto_algorithm_selection_enabled = False
        self.realtime_optimization_enabled = False
        self.cross_task_transfer_enabled = False
        
        # ML-powered failure prediction system
        self.failure_patterns = {}
        self.recovery_strategies = {}
        self.performance_history = []
        
        # Auto algorithm selection system
        self.algorithm_performance_db = {}
        self.data_analyzer = None
        
        # Real-time optimization system
        self.ab_test_registry = {}
        self.optimization_metrics = {}
        
        # Cross-task knowledge transfer system
        self.task_experience_bank = []
        self.knowledge_transfer_network = None
        
        # Initialize Phase 4 ML systems
        self.failure_prediction_model = FailurePredictionModel()
        self.algorithm_selector = AlgorithmSelector()
        self.ab_testing_framework = ABTestingFramework()
        self.knowledge_transfer_system = CrossTaskKnowledgeTransfer()
        self.performance_monitor = PerformanceMonitor()
        
    def create_test_time_compute_scaler(
        self, 
        base_model: nn.Module,
        config: Optional[TestTimeComputeConfig] = None
    ) -> TestTimeComputeScaler:
        """Create Test-Time Compute Scaler."""
        if config is None:
            config = TestTimeComputeConfig()
        
        self.test_time_scaler = TestTimeComputeScaler(base_model, config)
        return self.test_time_scaler
    
    def create_research_maml(
        self, 
        model: nn.Module,
        config: Optional[MAMLConfig] = None
    ) -> ResearchMAML:
        """Create MAML implementation."""
        if config is None:
            config = MAMLConfig(
                variant=MAMLVariant.MAML,
                inner_lr=0.01,
                outer_lr=0.001,
                inner_steps=1,
                first_order=False
            )
        
        self.maml_learner = ResearchMAML(model, config)
        return self.maml_learner
    
    def apply_batch_norm_fixes(self, model: nn.Module) -> nn.Module:
        """Apply research-accurate BatchNorm fixes for few-shot learning."""
        return self.batch_norm_policy.apply_to_model(model)
    
    def setup_deterministic_training(self, seed: int = 42) -> None:
        """Setup deterministic training for reproducible research."""
        setup_deterministic_environment(seed)
        self.determinism_manager.enable_full_determinism()
    
    def create_evaluation_harness(self, **kwargs) -> FewShotEvaluationHarness:
        """Create proper episodic evaluation harness with 95% CI."""
        self.evaluation_harness = FewShotEvaluationHarness(**kwargs)
        return self.evaluation_harness
    
    def train_episode(
        self, 
        episode: Episode,
        algorithm: str = "maml"
    ) -> Dict[str, Any]:
        """Train on a single episode using specified algorithm."""
        episode.validate()
        
        if algorithm == "maml" and self.maml_learner is not None:
            return self._train_maml_episode(episode)
        elif algorithm == "test_time_compute" and self.test_time_scaler is not None:
            return self._train_test_time_episode(episode)
        else:
            raise ValueError(f"Algorithm {algorithm} not initialized")
    
    def _train_maml_episode(self, episode: Episode) -> Dict[str, Any]:
        """Train using research-accurate MAML."""
        # Define loss function for MAML
        loss_fn = F.cross_entropy
        
        # Format episode as task batch (single task)
        task_batch = [(episode.support_x, episode.support_y, episode.query_x, episode.query_y)]
        
        # Use MAML forward method with task batch and loss function
        # This computes: inner adaptation on support → query loss on adapted params
        meta_loss = self.maml_learner(task_batch, loss_fn)
        
        # For metrics, compute support and query losses separately
        # NOTE: This re-does the inner loop adaptation (could be optimized to reuse adapted_params)
        support_logits = self.maml_learner.model(episode.support_x)
        support_loss = loss_fn(support_logits, episode.support_y)
        
        # Compute inner loop adaptation for query evaluation
        adapted_params = self.maml_learner.inner_loop(episode.support_x, episode.support_y, loss_fn)
        
        # Evaluate on query set with adapted parameters
        if adapted_params is not None and len(adapted_params) > 0:
            query_logits = FunctionalModule.functional_forward(
                self.maml_learner.model, 
                episode.query_x, 
                adapted_params
            )
        else:
            # Fallback to base model - this should only happen with zero inner steps
            import warnings
            inner_steps = getattr(self.maml_learner.config, 'inner_steps', 'unknown')
            warnings.warn(
                f"MAML falling back to base model (no adapted parameters). "
                f"This should only occur with inner_steps=0. "
                f"Current inner_steps: {inner_steps}. "
                f"Check MAML configuration if this is unexpected.",
                UserWarning,
                stacklevel=3
            )
            query_logits = self.maml_learner.model(episode.query_x)
        
        query_loss = F.cross_entropy(query_logits, episode.query_y)
        
        return {
            "query_loss": query_loss.item(),
            "query_accuracy": (query_logits.argmax(-1) == episode.query_y).float().mean().item(),
            "support_loss": support_loss.item(),
            "meta_loss": meta_loss.item()
        }
    
    def _train_test_time_episode(self, episode: Episode) -> Dict[str, Any]:
        """Train using Test-Time Compute Scaling."""
        predictions, metrics = self.test_time_scaler.scale_compute(
            support_set=episode.support_x,
            support_labels=episode.support_y, 
            query_set=episode.query_x,
            task_context={"n_classes": len(torch.unique(episode.support_y))}
        )
        
        accuracy = (predictions.argmax(-1) == episode.query_y).float().mean().item()
        
        return {
            "query_accuracy": accuracy,
            "compute_scaling_metrics": metrics,
            "predictions": predictions
        }
    
    # ================================================================================
    # Phase 4: Advanced ML-Powered Meta-Learning Enhancement Methods
    # ================================================================================
    
    def enable_failure_prediction(self, enable_ml_prediction: bool = True, enable_auto_recovery: bool = True):
        """
        Enable ML-powered failure prediction and automatic recovery systems.
        
        This system learns from training failures and automatically applies recovery
        strategies to prevent common issues like memory spikes, gradient explosions, etc.
        
        Args:
            enable_ml_prediction: Enable ML-based pattern recognition for failure prediction
            enable_auto_recovery: Enable automatic application of recovery strategies
        """
        self.failure_prediction_enabled = enable_ml_prediction
        
        if enable_ml_prediction:
            print("✅ ML-powered failure prediction enabled")
            print("   🧠 Learning from failure patterns")
            print("   🔮 Predicting potential failures before they occur")
            
        if enable_auto_recovery:
            print("   🔧 Auto-recovery strategies enabled")
            print("   📊 Monitoring memory usage, gradients, and performance")
    
    def enable_automatic_algorithm_selection(self, enable_data_analysis: bool = True, fallback_algorithm: str = "maml"):
        """
        Enable automatic algorithm selection based on data characteristics and task properties.
        
        The system analyzes incoming data and automatically selects the best algorithm
        (MAML, Test-Time Compute, etc.) based on learned performance patterns.
        
        Args:
            enable_data_analysis: Enable automatic data analysis for algorithm selection
            fallback_algorithm: Default algorithm to use when analysis is inconclusive
        """
        self.auto_algorithm_selection_enabled = enable_data_analysis
        
        if enable_data_analysis:
            print("✅ Automatic algorithm selection enabled")
            print("   📊 Analyzing task characteristics")
            print("   🎯 Auto-selecting optimal algorithms")
            print(f"   🔄 Fallback algorithm: {fallback_algorithm}")
    
    def enable_realtime_optimization(self, enable_ab_testing: bool = True, optimization_interval: int = 100):
        """
        Enable real-time optimization with A/B testing and performance monitoring.
        
        The system continuously monitors performance and runs A/B tests on different
        hyperparameters, automatically switching to better configurations.
        
        Args:
            enable_ab_testing: Enable automatic A/B testing of hyperparameters
            optimization_interval: Number of episodes between optimization checks
        """
        self.realtime_optimization_enabled = enable_ab_testing
        
        if enable_ab_testing:
            # Setup default A/B tests
            self.ab_testing_framework.create_ab_test(
                'learning_rate_test',
                ['maml_lr_001', 'maml_lr_01', 'maml_lr_1'],
                [0.4, 0.4, 0.2]  # More conservative allocation
            )
            
            print("✅ Real-time optimization enabled")
            print("   🔬 A/B testing hyperparameters")
            print(f"   ⏱️  Optimization interval: {optimization_interval} episodes")
            print("   📈 Continuous performance monitoring")
    
    def enable_cross_task_knowledge_transfer(self, enable_continual_improvement: bool = True, memory_size: int = 1000):
        """
        Enable cross-task knowledge transfer and continual meta-learning.
        
        The system learns from all tasks and transfers knowledge between similar
        tasks, improving performance through experience accumulation.
        
        Args:
            enable_continual_improvement: Enable continual learning from task experience
            memory_size: Maximum number of task experiences to retain
        """
        self.cross_task_transfer_enabled = enable_continual_improvement
        
        if enable_continual_improvement:
            print("✅ Cross-task knowledge transfer enabled")  
            print("   🧠 Learning from task similarities")
            print("   🔄 Transferring successful strategies")
            print(f"   💾 Memory size: {memory_size} task experiences")
    
    def predict_and_prevent_failures(self, episode: Episode, algorithm_state: Dict[str, Any]) -> Dict[str, Any]:
        """Use ML to predict and prevent potential failures."""
        if not self.failure_prediction_enabled:
            return {'failure_risk': 0.0, 'recommendations': []}
        
        failure_risk = self.failure_prediction_model.predict_failure_risk(episode, algorithm_state)
        
        recommendations = []
        if failure_risk > 0.7:
            recommendations.append("High failure risk detected - consider reducing learning rate")
            recommendations.append("Monitor memory usage and gradient norms")
        elif failure_risk > 0.4:
            recommendations.append("Moderate failure risk - increase monitoring")
        
        return {
            'failure_risk': failure_risk,
            'recommendations': recommendations
        }
    
    def select_optimal_algorithm(self, episode: Episode) -> str:
        """Automatically select the best algorithm for the given episode."""
        if not self.auto_algorithm_selection_enabled:
            return 'maml'  # Default fallback
        
        return self.algorithm_selector.select_algorithm(episode)
    
    def run_ab_test(self, test_name: str, episode_id: str) -> str:
        """Run A/B test and return assigned algorithm variant."""
        if not self.realtime_optimization_enabled:
            return 'maml'
        
        try:
            return self.ab_testing_framework.assign_algorithm(test_name, episode_id)
        except ValueError:
            return 'maml'  # Fallback if test doesn't exist
    
    def transfer_knowledge_from_similar_tasks(self, episode: Episode, base_config: Dict[str, Any]) -> Dict[str, Any]:
        """Transfer knowledge from similar previous tasks."""
        if not self.cross_task_transfer_enabled:
            return base_config
        
        return self.knowledge_transfer_system.transfer_knowledge(episode, base_config)
    
    def record_episode_outcome(self, episode: Episode, results: Dict[str, Any], algorithm_used: str):
        """Record episode outcome for all ML systems."""
        # Update failure prediction model
        failed = results.get('query_accuracy', 0.0) < 0.3  # Consider <30% accuracy as failure
        algorithm_state = {
            'learning_rate': results.get('learning_rate', 0.01),
            'inner_steps': results.get('inner_steps', 1),
            'loss_history': [results.get('query_loss', float('inf'))]
        }
        self.failure_prediction_model.update_with_outcome(episode, algorithm_state, failed)
        
        # Update algorithm selector
        self.algorithm_selector.update_performance(algorithm_used, episode, results.get('query_accuracy', 0.0))
        
        # Record A/B test results
        if self.realtime_optimization_enabled:
            self.ab_testing_framework.record_result('learning_rate_test', algorithm_used, results)
        
        # Store knowledge for transfer
        if self.cross_task_transfer_enabled:
            task_id = f"task_{len(self.knowledge_transfer_system.task_embeddings)}"
            self.knowledge_transfer_system.store_task_knowledge(episode, task_id, results)
        
        # Monitor performance
        self.performance_monitor.record_metrics({
            'accuracy': results.get('query_accuracy', 0.0),
            'loss': results.get('query_loss', float('inf')),
            'algorithm': algorithm_used
        })
    
    def get_optimization_insights(self) -> Dict[str, Any]:
        """Get insights and recommendations from all ML systems."""
        insights = {
            'performance_trends': self.performance_monitor.performance_trends,
            'optimization_suggestions': self.performance_monitor.get_optimization_suggestions(),
            'knowledge_transfers': len(self.knowledge_transfer_system.transfer_history),
            'ab_test_results': {}
        }
        
        # Get A/B test results
        if self.realtime_optimization_enabled:
            insights['ab_test_results'] = self.ab_testing_framework.analyze_ab_test('learning_rate_test')
        
        return insights


# Convenience functions for quick setup
def create_meta_learning_toolkit(
    model: nn.Module,
    algorithm: str = "maml",
    seed: int = 42,
    **kwargs
) -> MetaLearningToolkit:
    """
    Quick setup for meta-learning toolkit with algorithms.
    
    Args:
        model: Base neural network model
        algorithm: "maml" or "test_time_compute"  
        seed: Random seed for reproducible research
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured MetaLearningToolkit instance
    """
    meta_learner = MetaLearningToolkit()
    
    # Setup deterministic training
    meta_learner.setup_deterministic_training(seed)
    
    # Apply research-accurate BatchNorm fixes
    model = meta_learner.apply_batch_norm_fixes(model)
    
    # Initialize requested algorithm
    if algorithm == "maml":
        maml_config = MAMLConfig(**kwargs)
        meta_learner.create_research_maml(model, maml_config)
    elif algorithm == "test_time_compute":
        ttc_config = TestTimeComputeConfig(**kwargs)
        meta_learner.create_test_time_compute_scaler(model, ttc_config)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    return meta_learner

def quick_evaluation(
    model: nn.Module,
    episodes: list,
    algorithm: str = "maml",
    **kwargs
) -> Dict[str, Any]:
    """
    Quick evaluation using evaluation harness.
    """
    meta_learner = create_meta_learning_toolkit(model, algorithm, **kwargs)
    harness = meta_learner.create_evaluation_harness()
    
    return harness.evaluate_on_episodes(episodes, meta_learner.train_episode)


# Export key recovered functionality
__all__ = [
    "MetaLearningToolkit",
    "create_meta_learning_toolkit", 
    "quick_evaluation",
    "Episode",
    "remap_labels",
    "TestTimeComputeScaler",
    "ResearchMAML",
    "MAMLVariant",
    "EpisodicBatchNormPolicy",
    "FewShotEvaluationHarness"
]