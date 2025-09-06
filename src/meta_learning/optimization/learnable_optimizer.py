"""
💰 SUPPORT THIS RESEARCH - PLEASE DONATE! 💰

🙏 If this library helps your research or project, please consider donating:
💳 PayPal: https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS
⭐ GitHub Sponsors: https://github.com/sponsors/benedictchen

👨‍💻 Author: Benedict Chen
💰 Donations: Help support this work! Buy me a coffee ☕, beer 🍺, lamborghini 🏎️, or private island 🏝️
💖 Please consider recurring donations to fully support continued research

Your support enables cutting-edge AI research for everyone! 🚀

Learnable Optimizer Implementation
=================================

Meta-optimization algorithms that learn to optimize, including learnable learning rates,
adaptive gradient transforms, and meta-descent methods.
"""

# TODO: PHASE 1.3 - LEARNABLE OPTIMIZER IMPLEMENTATION
# TODO: Create LearnableOptimizer class extending nn.Module
# TODO: - Implement __init__ with model, transform, and learning rate parameters
# TODO: - Add support for per-parameter learnable transforms
# TODO: - Include gradient preprocessing and postprocessing hooks
# TODO: - Add integration with existing optimizers (SGD, Adam, etc.)
# TODO: - Support both meta-learning and standard optimization modes

# TODO: Implement gradient transform system
# TODO: - Create base GradientTransform class with forward/backward methods
# TODO: - Add ScaleTransform for learnable learning rate scaling
# TODO: - Implement BiasTransform for learnable gradient bias addition
# TODO: - Create CompositeTransform for combining multiple transforms
# TODO: - Add regularization options for transform parameters

# TODO: Add meta-optimization step functionality
# TODO: - Implement meta_step() for updating transform parameters
# TODO: - Add support for higher-order gradients through transforms
# TODO: - Include gradient clipping and normalization options
# TODO: - Support batched meta-optimization across multiple tasks
# TODO: - Add convergence monitoring for meta-optimization process

# TODO: Integrate with Phase 4 ML-powered enhancements
# TODO: - Connect with failure prediction for automatic learning rate adjustment
# TODO: - Add hooks for performance monitoring of optimization progress
# TODO: - Integrate with cross-task knowledge transfer for transform sharing
# TODO: - Include A/B testing support for different optimization strategies
# TODO: - Add real-time optimization suggestions based on performance trends

# TODO: Implement advanced features
# TODO: - Add support for momentum and second-order optimization methods
# TODO: - Implement curriculum learning for meta-optimization
# TODO: - Add support for different learning rate schedules
# TODO: - Include Bayesian optimization for hyperparameter tuning
# TODO: - Support multi-objective optimization for competing goals

# TODO: Add comprehensive testing and validation
# TODO: - Test convergence properties on standard optimization benchmarks
# TODO: - Validate meta-learning capabilities on few-shot tasks
# TODO: - Test integration with MAML and other meta-learning algorithms
# TODO: - Benchmark performance against standard optimizers
# TODO: - Add numerical stability tests for edge cases

from __future__ import annotations
from typing import Optional, Dict, Any, List, Callable, Union
import torch
import torch.nn as nn
import warnings
from ..core.utils import clone_module, update_module
from ..shared.types import Episode


class IdentityTransform(nn.Module):
    """Identity transform that passes gradients unchanged."""
    
    def forward(self, gradient: torch.Tensor) -> torch.Tensor:
        return gradient


class LearnableOptimizer(nn.Module):
    """
    Meta-optimizer with learnable gradient transforms.
    
    Learns to optimize by using learnable gradient transforms that can adapt
    their behavior based on meta-learning objectives.
    """
    
    def __init__(
        self,
        model: nn.Module,
        transform: Optional[nn.Module] = None,
        lr: float = 1.0,
        meta_lr: float = 0.001,
        gradient_clip: Optional[float] = None,
        momentum: float = 0.0,
        failure_prediction_model=None,
        adaptive_lr: bool = True
    ):
        super().__init__()
        self.model = model
        self.lr = lr
        self.base_lr = lr  # Store original learning rate
        self.meta_lr = meta_lr
        self.gradient_clip = gradient_clip
        self.momentum = momentum
        
        # Failure prediction integration
        self.failure_prediction_model = failure_prediction_model
        self.adaptive_lr = adaptive_lr
        self.current_episode = None
        self.failure_risk_history = []
        
        # Use identity transform if none provided
        if transform is None:
            transform = IdentityTransform()
        self.transform = transform
        
        # Meta-optimizer for transform parameters (only if there are parameters)
        transform_params = list(self.transform.parameters())
        if transform_params:
            self.meta_optimizer = torch.optim.Adam(transform_params, lr=meta_lr)
        else:
            self.meta_optimizer = None
        
        # Momentum buffers
        self.momentum_buffers = {}
        
        # Performance tracking
        self.step_count = 0
        self.performance_history = []
    
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """Perform optimization step with learnable transforms."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        # Apply gradient transforms and update parameters
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    # Apply learnable transform
                    transformed_grad = self.transform(param.grad)
                    
                    # Apply gradient clipping if specified
                    if self.gradient_clip is not None:
                        torch.nn.utils.clip_grad_norm_([param], self.gradient_clip)
                    
                    # Apply momentum if specified
                    if self.momentum > 0:
                        if name not in self.momentum_buffers:
                            self.momentum_buffers[name] = torch.zeros_like(param.data)
                        
                        buf = self.momentum_buffers[name]
                        buf.mul_(self.momentum).add_(transformed_grad)
                        transformed_grad = buf
                    
                    # Update parameter
                    param.data.add_(transformed_grad, alpha=-self.lr)
        
        self.step_count += 1
        return loss
    
    def meta_step(self, meta_loss: torch.Tensor) -> None:
        """Update transform parameters based on meta-learning objective."""
        # Only perform meta-optimization if transform has parameters
        if self.meta_optimizer is not None:
            # Zero gradients for meta-optimizer
            self.meta_optimizer.zero_grad()
            
            # Compute gradients w.r.t. transform parameters
            meta_loss.backward(retain_graph=True)
            
            # Update transform parameters
            self.meta_optimizer.step()
        
        # Track performance
        self.performance_history.append(meta_loss.item())
        
        # Limit history size
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]
    
    def zero_grad(self) -> None:
        """Zero gradients for both model and transform parameters."""
        self.model.zero_grad()
        self.transform.zero_grad()
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for monitoring."""
        if not self.performance_history:
            return {}
        
        import numpy as np
        history = np.array(self.performance_history)
        
        return {
            'step_count': self.step_count,
            'recent_loss': self.performance_history[-1] if self.performance_history else float('inf'),
            'average_loss': float(np.mean(history)),
            'loss_std': float(np.std(history)),
            'improvement_rate': float(history[0] - history[-1]) if len(history) > 1 else 0.0,
            'convergence_indicator': float(np.mean(np.diff(history[-10:]))) if len(history) > 10 else 0.0
        }
    
    def get_learning_rate(self) -> float:
        """Get current learning rate."""
        return self.lr
    
    def set_learning_rate(self, lr: float) -> None:
        """Set learning rate."""
        self.lr = lr
    
    def set_current_episode(self, episode: Episode) -> None:
        """Set current episode for failure prediction."""
        self.current_episode = episode
    
    def predict_and_adjust_learning_rate(self, algorithm_state: Optional[Dict[str, Any]] = None) -> float:
        """Predict failure risk and adjust learning rate accordingly."""
        if not self.adaptive_lr or self.failure_prediction_model is None or self.current_episode is None:
            return 0.0  # No failure risk prediction available
        
        # Use default algorithm state if none provided
        if algorithm_state is None:
            algorithm_state = {
                'learning_rate': self.lr,
                'step_count': self.step_count,
                'momentum': self.momentum,
                'gradient_clip': self.gradient_clip,
                'recent_performance': self.performance_history[-10:] if self.performance_history else []
            }
        
        # Predict failure risk
        failure_risk = self.failure_prediction_model.predict_failure_risk(
            self.current_episode, algorithm_state
        )
        
        # Store failure risk history
        self.failure_risk_history.append(failure_risk)
        if len(self.failure_risk_history) > 100:
            self.failure_risk_history = self.failure_risk_history[-100:]
        
        # Adaptive learning rate adjustment based on failure risk
        if failure_risk > 0.7:  # High risk - reduce learning rate significantly
            adjustment_factor = 0.5
        elif failure_risk > 0.4:  # Medium risk - reduce learning rate moderately
            adjustment_factor = 0.75
        elif failure_risk < 0.1:  # Very low risk - can increase learning rate slightly
            adjustment_factor = 1.1
        else:  # Normal risk - keep learning rate unchanged
            adjustment_factor = 1.0
        
        # Apply adjustment with bounds
        self.lr = max(0.001, min(self.base_lr * 2.0, self.lr * adjustment_factor))
        
        return failure_risk
    
    def update_failure_prediction_model(self, success: bool) -> None:
        """Update failure prediction model with optimization outcome."""
        if (self.failure_prediction_model is not None and 
            self.current_episode is not None):
            
            algorithm_state = {
                'learning_rate': self.lr,
                'step_count': self.step_count,
                'momentum': self.momentum,
                'gradient_clip': self.gradient_clip,
                'recent_performance': self.performance_history[-10:] if self.performance_history else []
            }
            
            # Update model with outcome (failure = not success)
            self.failure_prediction_model.update_with_outcome(
                self.current_episode, algorithm_state, not success
            )
    
    def get_failure_prediction_metrics(self) -> Dict[str, Any]:
        """Get failure prediction related metrics."""
        if not self.failure_risk_history:
            return {}
        
        import numpy as np
        risks = np.array(self.failure_risk_history)
        
        return {
            'current_failure_risk': self.failure_risk_history[-1] if self.failure_risk_history else 0.0,
            'average_failure_risk': float(np.mean(risks)),
            'max_failure_risk': float(np.max(risks)),
            'min_failure_risk': float(np.min(risks)),
            'failure_risk_trend': float(np.mean(np.diff(risks[-10:]))) if len(risks) > 10 else 0.0,
            'lr_adjustment_ratio': self.lr / self.base_lr,
            'adaptive_lr_enabled': self.adaptive_lr,
            'has_failure_prediction': self.failure_prediction_model is not None
        }


class GradientTransform(nn.Module):
    """
    Base class for learnable gradient transforms.
    
    Provides common functionality for all gradient transforms including
    regularization and parameter initialization.
    """
    
    def __init__(self, regularization: float = 0.01):
        super().__init__()
        self.regularization = regularization
        self._initialized = False
    
    def forward(self, gradient: torch.Tensor) -> torch.Tensor:
        """Apply transform to gradient tensor."""
        # Lazy initialization based on first gradient shape
        if not self._initialized:
            self._initialize(gradient.shape)
            self._initialized = True
        
        return self._apply_transform(gradient)
    
    def _initialize(self, gradient_shape: torch.Size) -> None:
        """Initialize parameters based on gradient shape."""
        pass
    
    def _apply_transform(self, gradient: torch.Tensor) -> torch.Tensor:
        """Apply the actual transformation (to be implemented by subclasses)."""
        raise NotImplementedError
    
    def get_regularization_loss(self) -> torch.Tensor:
        """Compute regularization loss for transform parameters."""
        reg_loss = torch.tensor(0.0)
        for param in self.parameters():
            reg_loss += self.regularization * torch.sum(param ** 2)
        return reg_loss


class ScaleTransform(GradientTransform):
    """
    Learnable scaling transform for gradients.
    
    Applies learnable per-element or global scaling to gradients.
    """
    
    def __init__(self, per_element: bool = False, init_scale: float = 1.0, regularization: float = 0.01):
        super().__init__(regularization)
        self.per_element = per_element
        self.init_scale = init_scale
        self.scale = None
    
    def _initialize(self, gradient_shape: torch.Size) -> None:
        """Initialize scale parameters."""
        if self.per_element:
            # Per-element scaling
            self.scale = nn.Parameter(torch.full(gradient_shape, self.init_scale))
        else:
            # Global scaling
            self.scale = nn.Parameter(torch.tensor(self.init_scale))
    
    def _apply_transform(self, gradient: torch.Tensor) -> torch.Tensor:
        """Apply learnable scaling to gradient."""
        if self.per_element:
            return gradient * self.scale
        else:
            return gradient * self.scale.expand_as(gradient)


class BiasTransform(GradientTransform):
    """
    Learnable bias addition transform for gradients.
    
    Adds learnable bias terms to gradients for improved optimization.
    """
    
    def __init__(self, per_element: bool = False, init_bias: float = 0.0, regularization: float = 0.01):
        super().__init__(regularization)
        self.per_element = per_element
        self.init_bias = init_bias
        self.bias = None
    
    def _initialize(self, gradient_shape: torch.Size) -> None:
        """Initialize bias parameters."""
        if self.per_element:
            # Per-element bias
            self.bias = nn.Parameter(torch.full(gradient_shape, self.init_bias))
        else:
            # Global bias
            self.bias = nn.Parameter(torch.tensor(self.init_bias))
    
    def _apply_transform(self, gradient: torch.Tensor) -> torch.Tensor:
        """Apply learnable bias to gradient."""
        if self.per_element:
            return gradient + self.bias
        else:
            return gradient + self.bias.expand_as(gradient)


class CompositeTransform(GradientTransform):
    """
    Composition of multiple gradient transforms.
    
    Applies multiple transforms in sequence with proper gradient flow.
    """
    
    def __init__(self, transforms: List[GradientTransform]):
        super().__init__(0.0)  # No additional regularization
        self.transforms = nn.ModuleList(transforms)
    
    def _initialize(self, gradient_shape: torch.Size) -> None:
        """Initialize all sub-transforms."""
        # Force initialization of all transforms
        dummy_grad = torch.zeros(gradient_shape)
        for transform in self.transforms:
            dummy_grad = transform(dummy_grad)
    
    def _apply_transform(self, gradient: torch.Tensor) -> torch.Tensor:
        """Apply all transforms in sequence."""
        result = gradient
        for transform in self.transforms:
            result = transform._apply_transform(result)
        return result
    
    def get_regularization_loss(self) -> torch.Tensor:
        """Get combined regularization loss from all transforms."""
        total_loss = torch.tensor(0.0)
        for transform in self.transforms:
            total_loss += transform.get_regularization_loss()
        return total_loss