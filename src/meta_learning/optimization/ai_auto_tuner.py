"""
AI-Powered Auto-Tuning System

Provides self-optimizing system that gets faster over time through intelligent
hyperparameter optimization, adaptive configuration, and performance learning.
"""
from __future__ import annotations

import json
import pickle
import threading
import time
from collections import deque, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
import warnings

import torch
import torch.nn as nn
import numpy as np
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, ConstantKernel


class OptimizationObjective(Enum):
    """Optimization objectives for auto-tuning."""
    ACCURACY = "accuracy"
    SPEED = "speed"
    MEMORY = "memory"
    BALANCED = "balanced"


@dataclass
class ConfigurationSpace:
    """Defines the space of possible configurations."""
    learning_rate: Tuple[float, float] = (1e-5, 1e-1)
    batch_size: Tuple[int, int] = (8, 128)
    adaptation_steps: Tuple[int, int] = (1, 10)
    meta_lr: Tuple[float, float] = (1e-4, 1e-2)
    regularization: Tuple[float, float] = (0.0, 0.1)
    dropout: Tuple[float, float] = (0.0, 0.5)
    hidden_dim: Tuple[int, int] = (64, 1024)
    num_layers: Tuple[int, int] = (1, 6)
    
    def to_bounds(self) -> List[Tuple[float, float]]:
        """Convert to optimization bounds."""
        return [
            self.learning_rate,
            (float(self.batch_size[0]), float(self.batch_size[1])),
            (float(self.adaptation_steps[0]), float(self.adaptation_steps[1])),
            self.meta_lr,
            self.regularization,
            self.dropout,
            (float(self.hidden_dim[0]), float(self.hidden_dim[1])),
            (float(self.num_layers[0]), float(self.num_layers[1]))
        ]
    
    def from_vector(self, vector: np.ndarray) -> Dict[str, Any]:
        """Convert optimization vector to configuration dict."""
        return {
            'learning_rate': float(vector[0]),
            'batch_size': int(round(vector[1])),
            'adaptation_steps': int(round(vector[2])),
            'meta_lr': float(vector[3]),
            'regularization': float(vector[4]),
            'dropout': float(vector[5]),
            'hidden_dim': int(round(vector[6])),
            'num_layers': int(round(vector[7]))
        }
    
    def to_vector(self, config: Dict[str, Any]) -> np.ndarray:
        """Convert configuration dict to optimization vector."""
        return np.array([
            config.get('learning_rate', 0.001),
            config.get('batch_size', 32),
            config.get('adaptation_steps', 5),
            config.get('meta_lr', 0.001),
            config.get('regularization', 0.01),
            config.get('dropout', 0.1),
            config.get('hidden_dim', 256),
            config.get('num_layers', 3)
        ])


class PerformancePredictor(nn.Module):
    """Neural network for predicting performance from configuration."""
    
    def __init__(self, config_dim: int = 8, hidden_dim: int = 128, 
                 output_dim: int = 3):  # accuracy, speed, memory
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(config_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # Output normalization layers
        self.accuracy_norm = nn.Sigmoid()  # [0, 1]
        self.speed_norm = nn.Softplus()    # > 0
        self.memory_norm = nn.Softplus()   # > 0
    
    def forward(self, config_vector: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Predict performance metrics from configuration."""
        features = self.network(config_vector)
        
        return {
            'accuracy': self.accuracy_norm(features[..., 0]),
            'speed': self.speed_norm(features[..., 1]),
            'memory': self.memory_norm(features[..., 2])
        }


class BayesianOptimizer:
    """Bayesian optimization for hyperparameter tuning."""
    
    def __init__(self, config_space: ConfigurationSpace, objective: OptimizationObjective):
        self.config_space = config_space
        self.objective = objective
        self.bounds = config_space.to_bounds()
        
        # Gaussian Process for modeling objective
        kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=10
        )
        
        # Observation history
        self.X_observed = []  # Configurations
        self.y_observed = []  # Performance scores
        self.iteration = 0
    
    def acquisition_function(self, X: np.ndarray, xi: float = 0.01) -> np.ndarray:
        """Expected improvement acquisition function."""
        if len(self.y_observed) < 2:
            # Random exploration for first few points
            return np.random.random(X.shape[0])
        
        # Predict mean and std
        mu, sigma = self.gp.predict(X, return_std=True)
        
        # Current best
        f_best = np.max(self.y_observed) if self.objective != OptimizationObjective.SPEED else np.min(self.y_observed)
        
        # Expected improvement
        if self.objective == OptimizationObjective.SPEED:
            # For speed, we want to minimize (lower is better)
            improvement = f_best - mu - xi
        else:
            # For accuracy/memory efficiency, we want to maximize
            improvement = mu - f_best - xi
        
        Z = improvement / (sigma + 1e-9)
        ei = improvement * self._normal_cdf(Z) + sigma * self._normal_pdf(Z)
        
        return ei
    
    def _normal_cdf(self, x):
        """Standard normal CDF approximation."""
        return 0.5 * (1.0 + np.sign(x) * np.sqrt(1.0 - np.exp(-2.0 * x**2 / np.pi)))
    
    def _normal_pdf(self, x):
        """Standard normal PDF."""
        return np.exp(-0.5 * x**2) / np.sqrt(2.0 * np.pi)
    
    def suggest_next_config(self) -> Dict[str, Any]:
        """Suggest next configuration to evaluate."""
        if len(self.X_observed) < 3:
            # Random exploration for first few iterations
            random_vector = np.random.uniform(
                low=[b[0] for b in self.bounds],
                high=[b[1] for b in self.bounds]
            )
            return self.config_space.from_vector(random_vector)
        
        # Optimize acquisition function
        best_ei = -np.inf
        best_config = None
        
        # Multi-start optimization
        for _ in range(10):
            # Random starting point
            x0 = np.random.uniform(
                low=[b[0] for b in self.bounds],
                high=[b[1] for b in self.bounds]
            )
            
            # Optimize
            result = minimize(
                fun=lambda x: -self.acquisition_function(x.reshape(1, -1))[0],
                x0=x0,
                bounds=self.bounds,
                method='L-BFGS-B'
            )
            
            if result.fun < best_ei:
                best_ei = result.fun
                best_config = result.x
        
        return self.config_space.from_vector(best_config) if best_config is not None else self.suggest_next_config()
    
    def update(self, config: Dict[str, Any], performance_score: float):
        """Update optimizer with new observation."""
        config_vector = self.config_space.to_vector(config)
        
        self.X_observed.append(config_vector)
        self.y_observed.append(performance_score)
        self.iteration += 1
        
        # Update Gaussian Process
        if len(self.y_observed) >= 2:
            X = np.array(self.X_observed)
            y = np.array(self.y_observed)
            self.gp.fit(X, y)
    
    def predict_performance(self, config: Dict[str, Any]) -> Tuple[float, float]:
        """Predict performance (mean, std) for a configuration."""
        if len(self.y_observed) < 2:
            return 0.5, 0.5  # High uncertainty
        
        config_vector = self.config_space.to_vector(config).reshape(1, -1)
        mean, std = self.gp.predict(config_vector, return_std=True)
        return float(mean[0]), float(std[0])


class AIAutoTuner:
    """
    AI-powered auto-tuning system that learns and optimizes over time.
    
    Features:
    - Self-optimizing system that gets faster over time
    - Bayesian optimization for efficient hyperparameter search
    - Multi-objective optimization (accuracy, speed, memory)
    - Adaptive configuration based on task characteristics
    - Performance prediction and recommendation system
    """
    
    def __init__(self, config_space: Optional[ConfigurationSpace] = None,
                 objective: OptimizationObjective = OptimizationObjective.BALANCED,
                 save_dir: str = "~/.meta_learning_autotuner"):
        """
        Initialize AI auto-tuner.
        
        Args:
            config_space: Configuration space to optimize over
            objective: Primary optimization objective
            save_dir: Directory to save tuning history and models
        """
        self.config_space = config_space or ConfigurationSpace()
        self.objective = objective
        self.save_dir = Path(save_dir).expanduser()
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Optimization components
        self.bayesian_optimizer = BayesianOptimizer(self.config_space, objective)
        self.performance_predictor = PerformancePredictor()
        self.predictor_optimizer = torch.optim.Adam(self.performance_predictor.parameters(), lr=0.001)
        
        # History and state
        self.optimization_history = []
        self.current_best_config = None
        self.current_best_score = float('-inf') if objective != OptimizationObjective.SPEED else float('inf')
        
        # Task adaptation
        self.task_characteristics = {}
        self.task_specific_configs = {}
        
        # Performance tracking
        self.tuning_stats = {
            'total_evaluations': 0,
            'improvements_found': 0,
            'convergence_iterations': 0,
            'prediction_accuracy': 0.0,
            'time_savings': 0.0
        }
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Load previous state
        self._load_state()
    
    def _calculate_performance_score(self, metrics: Dict[str, float]) -> float:
        """Calculate overall performance score from multiple metrics."""
        accuracy = metrics.get('accuracy', 0.0)
        speed = metrics.get('speed', 1.0)  # iterations per second
        memory_efficiency = metrics.get('memory_efficiency', 1.0)  # 1 / memory_usage
        
        if self.objective == OptimizationObjective.ACCURACY:
            return accuracy
        elif self.objective == OptimizationObjective.SPEED:
            return speed
        elif self.objective == OptimizationObjective.MEMORY:
            return memory_efficiency
        else:  # BALANCED
            # Weighted combination
            return 0.5 * accuracy + 0.3 * (speed / 10.0) + 0.2 * memory_efficiency
    
    def suggest_configuration(self, task_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Suggest optimal configuration for a task.
        
        Args:
            task_info: Optional task characteristics for adaptive tuning
            
        Returns:
            Suggested configuration
        """
        with self.lock:
            # Check if we have task-specific configuration
            if task_info:
                task_key = self._get_task_key(task_info)
                if task_key in self.task_specific_configs:
                    return self.task_specific_configs[task_key].copy()
            
            # Use global best configuration if available
            if self.current_best_config:
                return self.current_best_config.copy()
            
            # Use Bayesian optimization to suggest next configuration
            suggested_config = self.bayesian_optimizer.suggest_next_config()
            
            print(f"Suggested configuration: {suggested_config}")
            return suggested_config
    
    def update_performance(self, config: Dict[str, Any], metrics: Dict[str, float],
                         task_info: Optional[Dict[str, Any]] = None):
        """
        Update tuner with performance results.
        
        Args:
            config: Configuration that was evaluated
            metrics: Performance metrics (accuracy, speed, memory_efficiency)
            task_info: Task characteristics
        """
        with self.lock:
            # Calculate performance score
            performance_score = self._calculate_performance_score(metrics)
            
            # Update optimization history
            evaluation_record = {
                'config': config.copy(),
                'metrics': metrics.copy(),
                'performance_score': performance_score,
                'task_info': task_info.copy() if task_info else None,
                'timestamp': time.time(),
                'iteration': len(self.optimization_history)
            }
            self.optimization_history.append(evaluation_record)
            
            # Update Bayesian optimizer
            self.bayesian_optimizer.update(config, performance_score)
            
            # Check if this is a new best configuration
            is_better = (
                (self.objective != OptimizationObjective.SPEED and performance_score > self.current_best_score) or
                (self.objective == OptimizationObjective.SPEED and performance_score < self.current_best_score)
            )
            
            if is_better:
                self.current_best_config = config.copy()
                self.current_best_score = performance_score
                self.tuning_stats['improvements_found'] += 1
                
                print(f"New best configuration found! Score: {performance_score:.4f}")
            
            # Update task-specific configuration if applicable
            if task_info:
                task_key = self._get_task_key(task_info)
                if task_key not in self.task_specific_configs or is_better:
                    self.task_specific_configs[task_key] = config.copy()
            
            # Update performance predictor
            self._update_performance_predictor()
            
            # Update statistics
            self.tuning_stats['total_evaluations'] += 1
            
            # Save state periodically
            if self.tuning_stats['total_evaluations'] % 10 == 0:
                self._save_state()
    
    def _get_task_key(self, task_info: Dict[str, Any]) -> str:
        """Generate key for task-specific configurations."""
        # Create a stable key from task characteristics
        key_parts = []
        for key in sorted(task_info.keys()):
            value = task_info[key]
            if isinstance(value, (int, float)):
                key_parts.append(f"{key}:{value}")
            else:
                key_parts.append(f"{key}:{str(value)}")
        
        return "_".join(key_parts)
    
    def _update_performance_predictor(self):
        """Update neural network performance predictor."""
        if len(self.optimization_history) < 10:
            return
        
        # Prepare training data
        configs = []
        targets = []
        
        for record in self.optimization_history[-100:]:  # Use recent history
            config_vector = self.config_space.to_vector(record['config'])
            configs.append(config_vector)
            
            # Multi-target prediction
            metrics = record['metrics']
            target = [
                metrics.get('accuracy', 0.0),
                metrics.get('speed', 1.0),
                metrics.get('memory_efficiency', 1.0)
            ]
            targets.append(target)
        
        # Convert to tensors
        X = torch.tensor(np.array(configs), dtype=torch.float32)
        y = torch.tensor(np.array(targets), dtype=torch.float32)
        
        # Train predictor
        self.performance_predictor.train()
        for _ in range(10):  # Mini-training session
            self.predictor_optimizer.zero_grad()
            
            predictions = self.performance_predictor(X)
            
            # Multi-objective loss
            loss = 0
            loss += nn.MSELoss()(predictions['accuracy'], y[:, 0])
            loss += nn.MSELoss()(predictions['speed'], y[:, 1])
            loss += nn.MSELoss()(predictions['memory'], y[:, 2])
            
            loss.backward()
            self.predictor_optimizer.step()
        
        # Update prediction accuracy
        self._evaluate_prediction_accuracy()
    
    def _evaluate_prediction_accuracy(self):
        """Evaluate performance predictor accuracy."""
        if len(self.optimization_history) < 20:
            return
        
        # Use recent data for evaluation
        recent_data = self.optimization_history[-20:]
        configs = []
        actual_scores = []
        
        for record in recent_data:
            config_vector = self.config_space.to_vector(record['config'])
            configs.append(config_vector)
            actual_scores.append(record['performance_score'])
        
        # Predict with current model
        self.performance_predictor.eval()
        with torch.no_grad():
            X = torch.tensor(np.array(configs), dtype=torch.float32)
            predictions = self.performance_predictor(X)
            
            # Use primary objective for accuracy calculation
            if self.objective == OptimizationObjective.ACCURACY:
                predicted_scores = predictions['accuracy'].numpy()
            elif self.objective == OptimizationObjective.SPEED:
                predicted_scores = predictions['speed'].numpy()
            elif self.objective == OptimizationObjective.MEMORY:
                predicted_scores = predictions['memory'].numpy()
            else:  # BALANCED
                predicted_scores = (
                    0.5 * predictions['accuracy'] +
                    0.3 * (predictions['speed'] / 10.0) +
                    0.2 * predictions['memory']
                ).numpy()
        
        # Calculate accuracy (correlation)
        correlation = np.corrcoef(actual_scores, predicted_scores)[0, 1]
        self.tuning_stats['prediction_accuracy'] = max(0.0, correlation) if not np.isnan(correlation) else 0.0
    
    def predict_performance(self, config: Dict[str, Any]) -> Dict[str, float]:
        """Predict performance for a given configuration."""
        config_vector = self.config_space.to_vector(config)
        
        self.performance_predictor.eval()
        with torch.no_grad():
            X = torch.tensor(config_vector, dtype=torch.float32).unsqueeze(0)
            predictions = self.performance_predictor(X)
            
            return {
                'accuracy': float(predictions['accuracy'].item()),
                'speed': float(predictions['speed'].item()),
                'memory_efficiency': float(predictions['memory'].item()),
                'confidence': self.tuning_stats['prediction_accuracy']
            }
    
    def optimize_for_task(self, task_info: Dict[str, Any], 
                         max_evaluations: int = 20,
                         evaluation_fn: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Optimize configuration for a specific task.
        
        Args:
            task_info: Task characteristics
            max_evaluations: Maximum number of configurations to evaluate
            evaluation_fn: Function to evaluate configurations
            
        Returns:
            Best configuration found
        """
        print(f"Optimizing for task: {task_info}")
        
        best_config = None
        best_score = float('-inf') if self.objective != OptimizationObjective.SPEED else float('inf')
        
        for iteration in range(max_evaluations):
            # Get suggestion
            suggested_config = self.suggest_configuration(task_info)
            
            if evaluation_fn:
                # Evaluate configuration
                metrics = evaluation_fn(suggested_config)
                self.update_performance(suggested_config, metrics, task_info)
                
                # Track best
                score = self._calculate_performance_score(metrics)
                is_better = (
                    (self.objective != OptimizationObjective.SPEED and score > best_score) or
                    (self.objective == OptimizationObjective.SPEED and score < best_score)
                )
                
                if is_better:
                    best_config = suggested_config.copy()
                    best_score = score
                
                print(f"  Iteration {iteration + 1}/{max_evaluations}: Score {score:.4f}")
            else:
                # No evaluation function - just return suggestion
                best_config = suggested_config
                break
        
        print(f"✅ Task optimization complete. Best score: {best_score:.4f}")
        return best_config if best_config else {}
    
    def get_optimization_insights(self) -> Dict[str, Any]:
        """Get insights from optimization history."""
        if not self.optimization_history:
            return {'status': 'No optimization history available'}
        
        # Analyze parameter importance
        param_importance = self._analyze_parameter_importance()
        
        # Convergence analysis
        convergence_info = self._analyze_convergence()
        
        # Task-specific insights
        task_insights = self._analyze_task_patterns()
        
        return {
            'total_evaluations': len(self.optimization_history),
            'best_configuration': self.current_best_config,
            'best_score': self.current_best_score,
            'parameter_importance': param_importance,
            'convergence_info': convergence_info,
            'task_insights': task_insights,
            'prediction_accuracy': self.tuning_stats['prediction_accuracy'],
            'improvements_found': self.tuning_stats['improvements_found']
        }
    
    def _analyze_parameter_importance(self) -> Dict[str, float]:
        """Analyze which parameters have the most impact on performance."""
        if len(self.optimization_history) < 10:
            return {}
        
        # Extract data
        configs = []
        scores = []
        
        for record in self.optimization_history:
            config_vector = self.config_space.to_vector(record['config'])
            configs.append(config_vector)
            scores.append(record['performance_score'])
        
        configs = np.array(configs)
        scores = np.array(scores)
        
        # Calculate correlation for each parameter
        param_names = ['learning_rate', 'batch_size', 'adaptation_steps', 'meta_lr', 
                      'regularization', 'dropout', 'hidden_dim', 'num_layers']
        
        importance = {}
        for i, param_name in enumerate(param_names):
            correlation = np.corrcoef(configs[:, i], scores)[0, 1]
            importance[param_name] = abs(correlation) if not np.isnan(correlation) else 0.0
        
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
    
    def _analyze_convergence(self) -> Dict[str, Any]:
        """Analyze optimization convergence."""
        if len(self.optimization_history) < 5:
            return {'status': 'Insufficient data for convergence analysis'}
        
        scores = [record['performance_score'] for record in self.optimization_history]
        
        # Find best score at each iteration
        best_scores = []
        current_best = scores[0]
        
        for score in scores:
            if self.objective != OptimizationObjective.SPEED:
                current_best = max(current_best, score)
            else:
                current_best = min(current_best, score)
            best_scores.append(current_best)
        
        # Detect convergence (no improvement in last 20% of evaluations)
        recent_window = max(5, len(best_scores) // 5)
        recent_improvement = best_scores[-1] - best_scores[-recent_window]
        
        converged = abs(recent_improvement) < 1e-4
        
        return {
            'converged': converged,
            'iterations_to_best': best_scores.index(best_scores[-1]),
            'total_improvement': best_scores[-1] - best_scores[0],
            'recent_improvement': recent_improvement,
            'convergence_rate': abs(recent_improvement) / recent_window if recent_window > 0 else 0.0
        }
    
    def _analyze_task_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in task-specific configurations."""
        task_patterns = defaultdict(list)
        
        for record in self.optimization_history:
            if record['task_info']:
                task_key = self._get_task_key(record['task_info'])
                task_patterns[task_key].append(record)
        
        insights = {}
        for task_key, records in task_patterns.items():
            if len(records) >= 3:
                best_record = max(records, key=lambda r: r['performance_score'])
                avg_score = np.mean([r['performance_score'] for r in records])
                
                insights[task_key] = {
                    'evaluations': len(records),
                    'best_score': best_record['performance_score'],
                    'average_score': avg_score,
                    'best_config': best_record['config']
                }
        
        return insights
    
    def _save_state(self):
        """Save tuning state to disk."""
        try:
            state = {
                'optimization_history': self.optimization_history,
                'current_best_config': self.current_best_config,
                'current_best_score': self.current_best_score,
                'task_specific_configs': self.task_specific_configs,
                'tuning_stats': self.tuning_stats,
                'bayesian_optimizer_state': {
                    'X_observed': self.bayesian_optimizer.X_observed,
                    'y_observed': self.bayesian_optimizer.y_observed,
                    'iteration': self.bayesian_optimizer.iteration
                }
            }
            
            with open(self.save_dir / 'tuning_state.pkl', 'wb') as f:
                pickle.dump(state, f)
            
            # Save performance predictor
            torch.save(self.performance_predictor.state_dict(), 
                      self.save_dir / 'performance_predictor.pt')
            
        except Exception as e:
            warnings.warn(f"Failed to save auto-tuner state: {e}")
    
    def _load_state(self):
        """Load tuning state from disk."""
        try:
            state_file = self.save_dir / 'tuning_state.pkl'
            predictor_file = self.save_dir / 'performance_predictor.pt'
            
            if state_file.exists():
                with open(state_file, 'rb') as f:
                    state = pickle.load(f)
                
                self.optimization_history = state.get('optimization_history', [])
                self.current_best_config = state.get('current_best_config')
                self.current_best_score = state.get('current_best_score', 
                    float('-inf') if self.objective != OptimizationObjective.SPEED else float('inf'))
                self.task_specific_configs = state.get('task_specific_configs', {})
                self.tuning_stats = state.get('tuning_stats', self.tuning_stats)
                
                # Restore Bayesian optimizer state
                bo_state = state.get('bayesian_optimizer_state', {})
                if bo_state:
                    self.bayesian_optimizer.X_observed = bo_state.get('X_observed', [])
                    self.bayesian_optimizer.y_observed = bo_state.get('y_observed', [])
                    self.bayesian_optimizer.iteration = bo_state.get('iteration', 0)
                    
                    # Refit Gaussian Process if we have data
                    if len(self.bayesian_optimizer.y_observed) >= 2:
                        X = np.array(self.bayesian_optimizer.X_observed)
                        y = np.array(self.bayesian_optimizer.y_observed)
                        self.bayesian_optimizer.gp.fit(X, y)
            
            if predictor_file.exists():
                self.performance_predictor.load_state_dict(torch.load(predictor_file))
                
            print(f"📂 Loaded auto-tuner state: {len(self.optimization_history)} evaluations")
                
        except Exception as e:
            warnings.warn(f"Failed to load auto-tuner state: {e}")
    
    def reset_optimization(self):
        """Reset optimization state (keep learned models)."""
        with self.lock:
            self.bayesian_optimizer = BayesianOptimizer(self.config_space, self.objective)
            self.current_best_config = None
            self.current_best_score = float('-inf') if self.objective != OptimizationObjective.SPEED else float('inf')
            
            # Keep performance predictor but reset optimization history
            self.optimization_history = []
            self.tuning_stats = {
                'total_evaluations': 0,
                'improvements_found': 0,
                'convergence_iterations': 0,
                'prediction_accuracy': self.tuning_stats.get('prediction_accuracy', 0.0),
                'time_savings': 0.0
            }
            
            print("🔄 Optimization state reset")


def create_ai_auto_tuner(objective: OptimizationObjective = OptimizationObjective.BALANCED) -> AIAutoTuner:
    """Create AI auto-tuner with optimal defaults."""
    return AIAutoTuner(
        config_space=ConfigurationSpace(),
        objective=objective
    )


def optimize_meta_learning_config(model: nn.Module, train_loader, val_loader,
                                 max_evaluations: int = 20,
                                 objective: OptimizationObjective = OptimizationObjective.BALANCED) -> Dict[str, Any]:
    """
    Convenience function for optimizing meta-learning configuration.
    
    Args:
        model: Model to optimize
        train_loader: Training data loader
        val_loader: Validation data loader  
        max_evaluations: Maximum evaluations
        objective: Optimization objective
        
    Returns:
        Best configuration found
    """
    auto_tuner = create_ai_auto_tuner(objective)
    
    def evaluate_config(config: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate configuration performance."""
        # This would implement actual training and evaluation
        # For now, return mock metrics
        return {
            'accuracy': np.random.uniform(0.6, 0.9),
            'speed': np.random.uniform(1.0, 10.0),
            'memory_efficiency': np.random.uniform(0.5, 1.5)
        }
    
    # Extract task information
    task_info = {
        'model_type': type(model).__name__,
        'num_parameters': sum(p.numel() for p in model.parameters()),
        'dataset_size': len(train_loader.dataset) if hasattr(train_loader, 'dataset') else 1000
    }
    
    return auto_tuner.optimize_for_task(task_info, max_evaluations, evaluate_config)