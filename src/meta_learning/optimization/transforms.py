"""
💰 SUPPORT THIS RESEARCH - PLEASE DONATE! 💰

🙏 If this library helps your research or project, please consider donating:
💳 PayPal: https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS
⭐ GitHub Sponsors: https://github.com/sponsors/benedictchen

👨‍💻 Author: Benedict Chen
💰 Donations: Help support this work! Buy me a coffee ☕, beer 🍺, lamborghini 🏎️, or private island 🏝️
💖 Please consider recurring donations to fully support continued research

Your support enables cutting-edge AI research for everyone! 🚀

Gradient Transform Utilities
============================

Advanced gradient transform classes for learnable optimization,
including scaling, bias, and composite transforms for meta-learning.
"""


# TODO: Add integration with existing meta-learning framework
# TODO: - Integrate with LearnableOptimizer for transform chaining
# TODO: - Connect with FailurePredictionModel for transform monitoring
# TODO: - Add to performance monitoring dashboard
# TODO: - Support A/B testing for transform comparison
# TODO: - Include cross-task knowledge transfer for transform parameters

# TODO: Add comprehensive testing and validation
# TODO: - Test gradient flow preservation through transforms
# TODO: - Validate numerical stability with extreme gradients
# TODO: - Test composition correctness with multiple transforms
# TODO: - Benchmark performance against standard optimizers
# TODO: - Add regression tests for transform parameter evolution

from __future__ import annotations
from typing import Optional, Dict, Any, List, Union, Tuple
import torch
import torch.nn as nn
import warnings
from abc import ABC, abstractmethod


class GradientTransform(nn.Module, ABC):
    """Abstract base class for gradient transforms."""
    
    def __init__(self, regularization_weight: float = 0.0):
        """Initialize base gradient transform.
        
        Args:
            regularization_weight: L2 regularization weight for transform parameters
        """
        super().__init__()
        self.regularization_weight = regularization_weight
        
        # Common parameters for all transforms
        self.num_applications = 0
        self.gradient_norm_history = []
        self.register_buffer('total_applications', torch.tensor(0, dtype=torch.long))
    
    @abstractmethod
    def forward(self, gradient: torch.Tensor) -> torch.Tensor:
        """Apply transform to gradient.
        
        Args:
            gradient: Input gradient tensor
            
        Returns:
            Transformed gradient tensor
        """
        pass
    
    def get_regularization_loss(self) -> torch.Tensor:
        """Compute regularization loss for transform parameters."""
        if self.regularization_weight <= 0:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        
        reg_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        for param in self.parameters():
            reg_loss += torch.sum(param ** 2)
        
        return self.regularization_weight * reg_loss
    
    def update_statistics(self, gradient: torch.Tensor) -> None:
        """Update gradient statistics for monitoring."""
        self.num_applications += 1
        self.total_applications += 1
        
        grad_norm = torch.norm(gradient).item()
        self.gradient_norm_history.append(grad_norm)
        
        # Keep only recent history
        if len(self.gradient_norm_history) > 1000:
            self.gradient_norm_history = self.gradient_norm_history[-1000:]
    
    @abstractmethod
    def forward(self, gradient: torch.Tensor, parameter: torch.Tensor) -> torch.Tensor:
        """
        Apply gradient transform.
        
        Args:
            gradient: Input gradient tensor
            parameter: Associated parameter tensor (for context)
            
        Returns:
            Transformed gradient tensor
        """
        pass
    
    def compute_regularization_loss(self) -> torch.Tensor:
        """
        Compute L2 regularization loss for transform parameters.
        
        Returns:
            Regularization loss scalar tensor
        """
        if self.regularization_weight == 0.0:
            return torch.tensor(0.0)
        
        # Get device from first parameter if available
        device = None
        param_list = list(self.parameters())
        if param_list:
            device = param_list[0].device
        
        # L2 regularization on all learnable parameters
        reg_loss = torch.tensor(0.0, device=device)
        
        for param in param_list:
            if param.requires_grad:
                reg_loss += torch.sum(param ** 2)
        
        return self.regularization_weight * reg_loss
    
    def _update_statistics(self, gradient: torch.Tensor) -> None:
        """Update internal statistics for monitoring."""
        with torch.no_grad():
            self.num_applications += 1
            self.total_applications += 1
            
            # Track gradient norm for analysis
            grad_norm = gradient.norm().item()
            self.gradient_norm_history.append(grad_norm)
            
            # Keep history bounded
            if len(self.gradient_norm_history) > 1000:
                self.gradient_norm_history = self.gradient_norm_history[-500:]
    
    def get_statistics(self) -> Dict[str, float]:
        """Get transform usage and performance statistics."""
        if not self.gradient_norm_history:
            return {'applications': self.num_applications, 'avg_gradient_norm': 0.0}
        
        import numpy as np
        return {
            'applications': self.num_applications,
            'avg_gradient_norm': np.mean(self.gradient_norm_history),
            'gradient_norm_std': np.std(self.gradient_norm_history),
            'recent_gradient_norm': self.gradient_norm_history[-1] if self.gradient_norm_history else 0.0
        }


class ScaleTransform(GradientTransform):
    """
    Learnable scaling transform for gradients.
    
    TODO: Implement learnable scaling with per-parameter or global scaling
    """
    
    def __init__(
        self,
        parameter_shapes: Dict[str, torch.Size],
        init_scale: float = 1.0,
        per_parameter: bool = True,
        regularization_weight: float = 0.01
    ):
        """Initialize learnable scaling transform.
        
        Args:
            parameter_shapes: Dictionary mapping parameter names to shapes
            init_scale: Initial scaling value
            per_parameter: If True, learn separate scales per parameter
            regularization_weight: L2 regularization for scale parameters
        """
        super().__init__(regularization_weight)
        
        self.parameter_shapes = parameter_shapes
        self.init_scale = init_scale
        self.per_parameter = per_parameter
        
        # Initialize learnable scaling parameters
        if per_parameter:
            # Separate scaling parameter for each model parameter
            self.scales = nn.ParameterDict()
            for name, shape in parameter_shapes.items():
                # Initialize scales close to 1.0 for stability
                scale_param = torch.full(shape, init_scale, dtype=torch.float32)
                # Add small random noise to break symmetry
                scale_param += 0.1 * torch.randn_like(scale_param)
                self.scales[name] = nn.Parameter(scale_param)
        else:
            # Global scaling parameter
            self.global_scale = nn.Parameter(
                torch.tensor(init_scale, dtype=torch.float32)
            )
        
        # Constraints to prevent extreme scaling
        self.min_scale = 1e-6
        self.max_scale = 100.0
    
    def forward(self, gradient: torch.Tensor, parameter: torch.Tensor) -> torch.Tensor:
        """
        Apply learnable scaling to gradient.
        
        Args:
            gradient: Input gradient tensor
            parameter: Associated parameter tensor (for parameter identification)
            
        Returns:
            Scaled gradient tensor
        """
        self._update_statistics(gradient)
        
        if self.per_parameter:
            # Find matching scale parameter by shape
            param_shape = tuple(parameter.shape)
            matching_scale = None
            
            # Find scale parameter with matching shape
            for name, scale in self.scales.items():
                if tuple(scale.shape) == param_shape:
                    matching_scale = scale
                    break
            
            if matching_scale is not None:
                # Clamp scaling to prevent extreme values
                clamped_scale = torch.clamp(matching_scale, self.min_scale, self.max_scale)
                scaled_gradient = gradient * clamped_scale
            else:
                # Fallback: use mean of all scales if no exact match
                avg_scale = torch.stack([s.mean() for s in self.scales.values()]).mean()
                clamped_scale = torch.clamp(avg_scale, self.min_scale, self.max_scale)
                scaled_gradient = gradient * clamped_scale
        else:
            # Apply global scaling
            clamped_scale = torch.clamp(self.global_scale, self.min_scale, self.max_scale)
            scaled_gradient = gradient * clamped_scale
        
        return scaled_gradient
    
    def get_current_scales(self) -> Dict[str, torch.Tensor]:
        """Get current scaling values for analysis."""
        if self.per_parameter:
            return {name: torch.clamp(scale, self.min_scale, self.max_scale) 
                   for name, scale in self.scales.items()}
        else:
            return {'global': torch.clamp(self.global_scale, self.min_scale, self.max_scale)}


class BiasTransform(GradientTransform):
    """
    Learnable bias addition transform for gradients.
    
    Adds learnable bias terms to gradients, potentially with momentum.
    """
    
    def __init__(
        self,
        parameter_shapes: Dict[str, torch.Size],
        init_bias: float = 0.0,
        momentum: float = 0.9,
        regularization_weight: float = 0.01
    ):
        """Initialize learnable bias transform.
        
        Args:
            parameter_shapes: Dictionary mapping parameter names to shapes
            init_bias: Initial bias value
            momentum: Momentum factor for bias accumulation
            regularization_weight: L2 regularization for bias parameters
        """
        super().__init__(regularization_weight)
        
        self.parameter_shapes = parameter_shapes
        self.init_bias = init_bias
        self.momentum = momentum
        
        # Initialize learnable bias parameters
        self.biases = nn.ParameterDict()
        for name, shape in parameter_shapes.items():
            bias_param = torch.full(shape, init_bias, dtype=torch.float32)
            self.biases[name] = nn.Parameter(bias_param)
        
        # Initialize momentum buffers (non-learnable)
        self.momentum_buffers = {}
        for name, shape in parameter_shapes.items():
            self.register_buffer(f'momentum_{name}', torch.zeros(shape))
            self.momentum_buffers[name] = getattr(self, f'momentum_{name}')
    
    def forward(self, gradient: torch.Tensor, parameter: torch.Tensor) -> torch.Tensor:
        """
        Apply learnable bias addition to gradient.
        
        Args:
            gradient: Input gradient tensor
            parameter: Associated parameter tensor
            
        Returns:
            Gradient with bias added
        """
        self._update_statistics(gradient)
        
        # Find matching bias parameter by shape
        param_shape = tuple(parameter.shape)
        matching_bias = None
        matching_momentum = None
        
        for name, bias in self.biases.items():
            if tuple(bias.shape) == param_shape:
                matching_bias = bias
                matching_momentum = self.momentum_buffers[name]
                break
        
        if matching_bias is not None and matching_momentum is not None:
            # Apply momentum to bias
            with torch.no_grad():
                matching_momentum.mul_(self.momentum).add_(matching_bias, alpha=1-self.momentum)
            
            # Add bias to gradient
            biased_gradient = gradient + matching_momentum
        else:
            # Fallback: no bias addition if no matching shape
            biased_gradient = gradient
        
        return biased_gradient
    
    def get_current_biases(self) -> Dict[str, torch.Tensor]:
        """Get current bias values for analysis."""
        return {name: bias.clone() for name, bias in self.biases.items()}


class MomentumTransform(GradientTransform):
    """
    Learnable momentum transform for gradients.
    
    Implements learnable momentum with adaptive coefficients.
    """
    
    def __init__(
        self,
        parameter_shapes: Dict[str, torch.Size],
        init_momentum: float = 0.9,
        adaptive: bool = True,
        regularization_weight: float = 0.01
    ):
        """Initialize learnable momentum transform.
        
        Args:
            parameter_shapes: Dictionary mapping parameter names to shapes
            init_momentum: Initial momentum coefficient
            adaptive: If True, learn momentum coefficients
            regularization_weight: L2 regularization weight
        """
        super().__init__(regularization_weight)
        
        self.parameter_shapes = parameter_shapes
        self.init_momentum = init_momentum
        self.adaptive = adaptive
        
        if adaptive:
            # Learnable momentum coefficients per parameter
            self.momentum_coeffs = nn.ParameterDict()
            for name, shape in parameter_shapes.items():
                # Initialize momentum coefficients close to init_momentum
                momentum_coeff = torch.full(shape, init_momentum, dtype=torch.float32)
                # Add small noise to break symmetry
                momentum_coeff += 0.01 * torch.randn_like(momentum_coeff)
                self.momentum_coeffs[name] = nn.Parameter(momentum_coeff)
        else:
            # Fixed momentum coefficient
            self.fixed_momentum = init_momentum
        
        # Initialize momentum buffers
        self.momentum_buffers = {}
        for name, shape in parameter_shapes.items():
            self.register_buffer(f'momentum_{name}', torch.zeros(shape))
            self.momentum_buffers[name] = getattr(self, f'momentum_{name}')
        
        # Constraints for momentum coefficients
        self.min_momentum = 0.0
        self.max_momentum = 0.99
    
    def forward(self, gradient: torch.Tensor, parameter: torch.Tensor) -> torch.Tensor:
        """
        Apply learnable momentum to gradient.
        
        Args:
            gradient: Input gradient tensor
            parameter: Associated parameter tensor
            
        Returns:
            Momentum-transformed gradient
        """
        self._update_statistics(gradient)
        
        # Find matching momentum buffer and coefficient by shape
        param_shape = tuple(parameter.shape)
        matching_buffer = None
        momentum_coeff = self.fixed_momentum if not self.adaptive else None
        
        for name, buffer in self.momentum_buffers.items():
            if tuple(buffer.shape) == param_shape:
                matching_buffer = buffer
                if self.adaptive and name in self.momentum_coeffs:
                    momentum_coeff = torch.clamp(
                        self.momentum_coeffs[name], self.min_momentum, self.max_momentum
                    )
                break
        
        if matching_buffer is not None and momentum_coeff is not None:
            # Update momentum buffer
            with torch.no_grad():
                if self.adaptive:
                    # Per-element momentum coefficients
                    matching_buffer.mul_(momentum_coeff).add_(gradient, alpha=1.0)
                else:
                    # Global momentum coefficient
                    matching_buffer.mul_(momentum_coeff).add_(gradient, alpha=1.0)
            
            # Return momentum buffer as transformed gradient
            return matching_buffer.clone()
        else:
            # Fallback: return original gradient if no matching buffer
            return gradient
    
    def get_current_momentum_coeffs(self) -> Dict[str, torch.Tensor]:
        """Get current momentum coefficients for analysis."""
        if self.adaptive:
            return {name: torch.clamp(coeff, self.min_momentum, self.max_momentum)
                   for name, coeff in self.momentum_coeffs.items()}
        else:
            return {'global': torch.tensor(self.fixed_momentum)}


class AdaptiveTransform(GradientTransform):
    """
    Adaptive gradient transform based on gradient statistics.
    
    Similar to Adam's adaptive learning rates but for general gradient transformation.
    """
    
    def __init__(
        self,
        parameter_shapes: Dict[str, torch.Size],
        adaptation_rate: float = 0.01,
        epsilon: float = 1e-8,
        regularization_weight: float = 0.01
    ):
        """Initialize adaptive transform.
        
        Args:
            parameter_shapes: Dictionary mapping parameter names to shapes
            adaptation_rate: Rate of adaptation to gradient statistics
            epsilon: Small constant for numerical stability
            regularization_weight: L2 regularization weight
        """
        super().__init__(regularization_weight)
        
        self.parameter_shapes = parameter_shapes
        self.adaptation_rate = adaptation_rate
        self.epsilon = epsilon
        
        # Initialize running statistics (exponential moving averages)
        self.mean_buffers = {}
        self.variance_buffers = {}
        
        for name, shape in parameter_shapes.items():
            # Mean of gradients (first moment)
            self.register_buffer(f'mean_{name}', torch.zeros(shape))
            self.mean_buffers[name] = getattr(self, f'mean_{name}')
            
            # Variance of gradients (second moment)
            self.register_buffer(f'var_{name}', torch.zeros(shape))
            self.variance_buffers[name] = getattr(self, f'var_{name}')
        
        # Learnable adaptation parameters
        self.adaptation_weights = nn.ParameterDict()
        for name, shape in parameter_shapes.items():
            # Weight for how much to adapt based on statistics
            adapt_weight = torch.full(shape, adaptation_rate, dtype=torch.float32)
            self.adaptation_weights[name] = nn.Parameter(adapt_weight)
        
        # Decay factors for exponential moving averages
        self.beta1 = 0.9  # For mean
        self.beta2 = 0.999  # For variance
    
    def forward(self, gradient: torch.Tensor, parameter: torch.Tensor) -> torch.Tensor:
        """
        Apply adaptive gradient transformation.
        
        Args:
            gradient: Input gradient tensor
            parameter: Associated parameter tensor
            
        Returns:
            Adaptively transformed gradient
        """
        self._update_statistics(gradient)
        
        # Find matching buffers by shape
        param_shape = tuple(parameter.shape)
        matching_mean = None
        matching_var = None
        matching_weight = None
        
        for name, mean_buf in self.mean_buffers.items():
            if tuple(mean_buf.shape) == param_shape:
                matching_mean = mean_buf
                matching_var = self.variance_buffers[name]
                matching_weight = self.adaptation_weights[name]
                break
        
        if matching_mean is not None and matching_var is not None and matching_weight is not None:
            # Update running statistics
            with torch.no_grad():
                # Update mean (first moment)
                matching_mean.mul_(self.beta1).add_(gradient, alpha=1 - self.beta1)
                
                # Update variance (second moment)
                matching_var.mul_(self.beta2).addcmul_(gradient, gradient, value=1 - self.beta2)
            
            # Compute adaptive transformation
            # Scale gradient by inverse of standard deviation (like Adam)
            std = torch.sqrt(matching_var + self.epsilon)
            normalized_gradient = gradient / std
            
            # Apply learnable adaptation weight
            adapted_gradient = gradient + matching_weight * normalized_gradient
            
            return adapted_gradient
        else:
            # Fallback: return original gradient
            return gradient
    
    def get_adaptation_stats(self) -> Dict[str, Dict[str, float]]:
        """Get current adaptation statistics."""
        stats = {}
        for name in self.parameter_shapes:
            if name in self.mean_buffers and name in self.variance_buffers:
                mean_norm = self.mean_buffers[name].norm().item()
                var_mean = self.variance_buffers[name].mean().item()
                adapt_weight_mean = self.adaptation_weights[name].mean().item()
                
                stats[name] = {
                    'mean_norm': mean_norm,
                    'variance_mean': var_mean,
                    'adaptation_weight_mean': adapt_weight_mean
                }
        return stats


class CompositeTransform(GradientTransform):
    """
    Composition of multiple gradient transforms.
    
    Allows combining multiple transforms either sequentially or in parallel.
    """
    
    def __init__(
        self,
        transforms: List[GradientTransform],
        composition_type: str = "sequential",
        weights: Optional[torch.Tensor] = None
    ):
        """Initialize composite transform.
        
        Args:
            transforms: List of gradient transforms to compose
            composition_type: 'sequential' or 'parallel' composition
            weights: For parallel composition, weights for combining outputs
        """
        super().__init__()
        
        self.transforms = nn.ModuleList(transforms)
        self.composition_type = composition_type
        
        if composition_type == "parallel":
            if weights is None:
                # Equal weights for all transforms
                self.weights = nn.Parameter(torch.ones(len(transforms)) / len(transforms))
            else:
                self.weights = nn.Parameter(weights)
        else:
            self.weights = None
        
        # Validate composition type
        if composition_type not in ["sequential", "parallel"]:
            raise ValueError(f"Unknown composition_type: {composition_type}")
    
    def forward(self, gradient: torch.Tensor, parameter: torch.Tensor) -> torch.Tensor:
        """
        Apply composite transform.
        
        Args:
            gradient: Input gradient tensor
            parameter: Associated parameter tensor
            
        Returns:
            Transformed gradient tensor
        """
        self._update_statistics(gradient)
        
        if self.composition_type == "sequential":
            # Apply transforms sequentially
            current_gradient = gradient
            for transform in self.transforms:
                current_gradient = transform(current_gradient, parameter)
            return current_gradient
        
        elif self.composition_type == "parallel":
            # Apply transforms in parallel and combine
            transformed_gradients = []
            for transform in self.transforms:
                transformed_grad = transform(gradient, parameter)
                transformed_gradients.append(transformed_grad)
            
            # Weighted combination
            if len(transformed_gradients) > 0:
                stacked_grads = torch.stack(transformed_gradients, dim=0)
                # Normalize weights to sum to 1
                normalized_weights = torch.softmax(self.weights, dim=0)
                # Weight and sum
                combined_gradient = torch.sum(
                    stacked_grads * normalized_weights.view(-1, *([1] * (stacked_grads.ndim - 1))),
                    dim=0
                )
                return combined_gradient
            else:
                return gradient
        
        return gradient
    
    def add_transform(self, transform: GradientTransform, position: Optional[int] = None):
        """Add transform to the composition.
        
        Args:
            transform: Transform to add
            position: Position to insert (None for append)
        """
        if position is None:
            self.transforms.append(transform)
        else:
            self.transforms.insert(position, transform)
        
        # Update weights for parallel composition
        if self.composition_type == "parallel":
            new_size = len(self.transforms)
            if self.weights is not None:
                # Extend weights with equal probability for new transform
                with torch.no_grad():
                    old_weights = self.weights.data * (new_size - 1) / new_size
                    new_weight = torch.tensor([1.0 / new_size])
                    updated_weights = torch.cat([old_weights, new_weight])
                    self.weights = nn.Parameter(updated_weights)
    
    def remove_transform(self, index: int):
        """Remove transform from composition.
        
        Args:
            index: Index of transform to remove
        """
        if 0 <= index < len(self.transforms):
            del self.transforms[index]
            
            # Update weights for parallel composition
            if self.composition_type == "parallel" and self.weights is not None:
                with torch.no_grad():
                    # Remove corresponding weight and renormalize
                    mask = torch.ones(len(self.weights), dtype=torch.bool)
                    mask[index] = False
                    remaining_weights = self.weights[mask]
                    # Renormalize
                    self.weights = nn.Parameter(remaining_weights / remaining_weights.sum())
    
    def get_transform_contributions(self, gradient: torch.Tensor, 
                                   parameter: torch.Tensor) -> Dict[int, torch.Tensor]:
        """Get individual transform contributions for analysis."""
        contributions = {}
        
        if self.composition_type == "parallel":
            for i, transform in enumerate(self.transforms):
                contrib = transform(gradient, parameter)
                contributions[i] = contrib
        else:
            # For sequential, show cumulative effect
            current_gradient = gradient
            contributions[0] = current_gradient
            for i, transform in enumerate(self.transforms):
                current_gradient = transform(current_gradient, parameter)
                contributions[i + 1] = current_gradient
        
        return contributions


class NoiseTransform(GradientTransform):
    """
    Gradient noise injection for regularization.
    
    Adds controlled noise to gradients for regularization and exploration.
    """
    
    def __init__(
        self,
        noise_scale: float = 0.01,
        noise_type: str = "gaussian",
        adaptive_scale: bool = True,
        regularization_weight: float = 0.0
    ):
        """Initialize noise transform.
        
        Args:
            noise_scale: Scale of noise injection
            noise_type: Type of noise ('gaussian', 'uniform', 'laplace')
            adaptive_scale: If True, adapt noise scale based on gradient magnitude
            regularization_weight: L2 regularization weight
        """
        super().__init__(regularization_weight)
        
        self.noise_type = noise_type
        self.adaptive_scale = adaptive_scale
        
        # Learnable noise scale parameter
        if adaptive_scale:
            self.noise_scale = nn.Parameter(torch.tensor(noise_scale, dtype=torch.float32))
        else:
            self.fixed_noise_scale = noise_scale
        
        # Supported noise types
        self.supported_noise_types = ['gaussian', 'uniform', 'laplace']
        if noise_type not in self.supported_noise_types:
            raise ValueError(f"Unsupported noise_type: {noise_type}. "
                           f"Supported types: {self.supported_noise_types}")
        
        # Running statistics for adaptive scaling
        self.register_buffer('grad_magnitude_ema', torch.tensor(1.0))
        self.ema_decay = 0.99
    
    def forward(self, gradient: torch.Tensor, parameter: torch.Tensor) -> torch.Tensor:
        """
        Apply noise injection to gradient.
        
        Args:
            gradient: Input gradient tensor
            parameter: Associated parameter tensor
            
        Returns:
            Gradient with noise injected
        """
        self._update_statistics(gradient)
        
        # Update running gradient magnitude for adaptive scaling
        with torch.no_grad():
            current_magnitude = gradient.norm()
            self.grad_magnitude_ema.mul_(self.ema_decay).add_(
                current_magnitude, alpha=1 - self.ema_decay
            )
        
        # Determine noise scale
        if self.adaptive_scale:
            # Scale noise based on current gradient magnitude
            effective_scale = self.noise_scale * self.grad_magnitude_ema
        else:
            effective_scale = self.fixed_noise_scale
        
        # Generate noise based on type
        if self.noise_type == "gaussian":
            noise = torch.randn_like(gradient) * effective_scale
        elif self.noise_type == "uniform":
            noise = (torch.rand_like(gradient) - 0.5) * 2 * effective_scale
        elif self.noise_type == "laplace":
            # Approximate Laplace with difference of exponentials
            u1 = torch.rand_like(gradient)
            u2 = torch.rand_like(gradient)
            noise = effective_scale * torch.sign(u1 - 0.5) * torch.log(2 * torch.min(u1, 1 - u1))
        
        # Add noise to gradient
        noisy_gradient = gradient + noise
        
        return noisy_gradient
    
    def get_current_noise_scale(self) -> torch.Tensor:
        """Get current effective noise scale."""
        if self.adaptive_scale:
            return self.noise_scale * self.grad_magnitude_ema
        else:
            return torch.tensor(self.fixed_noise_scale)


class TemperatureTransform(GradientTransform):
    """
    Temperature-based gradient scaling.
    
    Implements temperature scaling for gradient normalization and control.
    """
    
    def __init__(
        self,
        init_temperature: float = 1.0,
        learnable: bool = True,
        min_temperature: float = 0.1,
        max_temperature: float = 10.0
    ):
        """Initialize temperature transform.
        
        Args:
            init_temperature: Initial temperature value
            learnable: If True, temperature is learnable parameter
            min_temperature: Minimum allowed temperature
            max_temperature: Maximum allowed temperature
        """
        super().__init__()
        
        self.learnable = learnable
        self.min_temperature = min_temperature
        self.max_temperature = max_temperature
        
        if learnable:
            self.temperature = nn.Parameter(torch.tensor(init_temperature, dtype=torch.float32))
        else:
            self.fixed_temperature = init_temperature
    
    def forward(self, gradient: torch.Tensor, parameter: torch.Tensor) -> torch.Tensor:
        """
        Apply temperature-based gradient scaling.
        
        Args:
            gradient: Input gradient tensor
            parameter: Associated parameter tensor
            
        Returns:
            Temperature-scaled gradient tensor
        """
        self._update_statistics(gradient)
        
        # Get current temperature
        if self.learnable:
            current_temp = torch.clamp(self.temperature, self.min_temperature, self.max_temperature)
        else:
            current_temp = self.fixed_temperature
        
        # Apply temperature scaling
        # Higher temperature -> larger gradients (more aggressive updates)
        # Lower temperature -> smaller gradients (more conservative updates)
        scaled_gradient = gradient * current_temp
        
        return scaled_gradient
    
    def get_current_temperature(self) -> torch.Tensor:
        """Get current temperature value."""
        if self.learnable:
            return torch.clamp(self.temperature, self.min_temperature, self.max_temperature)
        else:
            return torch.tensor(self.fixed_temperature)


def create_transform_from_config(config: Dict[str, Any], parameter_shapes: Dict[str, torch.Size]) -> GradientTransform:
    """
    Create gradient transform from configuration dictionary.
    
    Args:
        config: Configuration dictionary with transform specifications
        parameter_shapes: Dictionary mapping parameter names to shapes
        
    Returns:
        Configured gradient transform
    """
    transform_type = config.get('type', 'scale')
    transform_args = config.get('args', {})
    
    # Add parameter_shapes to args if needed
    if 'parameter_shapes' not in transform_args:
        transform_args['parameter_shapes'] = parameter_shapes
    
    # Create transform based on type
    if transform_type == 'scale':
        return ScaleTransform(**transform_args)
    elif transform_type == 'bias':
        return BiasTransform(**transform_args)
    elif transform_type == 'momentum':
        return MomentumTransform(**transform_args)
    elif transform_type == 'adaptive':
        return AdaptiveTransform(**transform_args)
    elif transform_type == 'noise':
        # Remove parameter_shapes from args for NoiseTransform
        noise_args = {k: v for k, v in transform_args.items() if k != 'parameter_shapes'}
        return NoiseTransform(**noise_args)
    elif transform_type == 'temperature':
        # Remove parameter_shapes from args for TemperatureTransform
        temp_args = {k: v for k, v in transform_args.items() if k != 'parameter_shapes'}
        return TemperatureTransform(**temp_args)
    elif transform_type == 'composite':
        # Special handling for composite transforms
        sub_transforms = []
        for sub_config in config.get('sub_transforms', []):
            sub_transform = create_transform_from_config(sub_config, parameter_shapes)
            sub_transforms.append(sub_transform)
        
        composite_args = {k: v for k, v in transform_args.items() if k != 'parameter_shapes'}
        composite_args['transforms'] = sub_transforms
        return CompositeTransform(**composite_args)
    else:
        raise ValueError(f"Unknown transform type: {transform_type}")


def analyze_gradient_statistics(gradients: List[torch.Tensor]) -> Dict[str, float]:
    """
    Analyze gradient statistics for transform optimization.
    
    Args:
        gradients: List of gradient tensors to analyze
        
    Returns:
        Dictionary with gradient statistics
    """
    if not gradients:
        return {}
    
    # Stack gradients for analysis
    if isinstance(gradients[0], torch.Tensor):
        # Flatten all gradients and concatenate
        flattened_grads = [grad.flatten() for grad in gradients]
        all_grads = torch.cat(flattened_grads, dim=0)
    else:
        return {}
    
    # Compute statistics
    stats = {
        'mean': all_grads.mean().item(),
        'std': all_grads.std().item(),
        'min': all_grads.min().item(),
        'max': all_grads.max().item(),
        'median': all_grads.median().item(),
        'l1_norm': all_grads.abs().mean().item(),
        'l2_norm': all_grads.norm().item() / len(all_grads),
        'zero_fraction': (all_grads == 0).float().mean().item(),
        'positive_fraction': (all_grads > 0).float().mean().item(),
        'negative_fraction': (all_grads < 0).float().mean().item(),
    }
    
    # Add percentile information
    sorted_grads = torch.sort(all_grads)[0]
    n = len(sorted_grads)
    if n > 0:
        stats.update({
            'percentile_1': sorted_grads[max(0, int(0.01 * n) - 1)].item(),
            'percentile_5': sorted_grads[max(0, int(0.05 * n) - 1)].item(),
            'percentile_95': sorted_grads[min(n - 1, int(0.95 * n))].item(),
            'percentile_99': sorted_grads[min(n - 1, int(0.99 * n))].item(),
        })
    
    # Add variance and skewness estimates
    variance = all_grads.var().item()
    stats['variance'] = variance
    
    if variance > 1e-8:  # Avoid division by zero
        centered = all_grads - all_grads.mean()
        skewness = (centered ** 3).mean() / (variance ** 1.5)
        kurtosis = (centered ** 4).mean() / (variance ** 2) - 3
        stats['skewness'] = skewness.item() if isinstance(skewness, torch.Tensor) else skewness
        stats['kurtosis'] = kurtosis.item() if isinstance(kurtosis, torch.Tensor) else kurtosis
    else:
        stats['skewness'] = 0.0
        stats['kurtosis'] = 0.0
    
    return stats


class TransformOptimizer:
    """
    Optimizer for gradient transform parameters.
    
    Specialized optimizer for tuning gradient transform parameters based on performance feedback.
    """
    
    def __init__(
        self,
        transforms: List[GradientTransform],
        learning_rate: float = 0.001,
        optimization_frequency: int = 10
    ):
        """Initialize transform optimizer.
        
        Args:
            transforms: List of gradient transforms to optimize
            learning_rate: Learning rate for parameter updates
            optimization_frequency: How often to optimize (every N steps)
        """
        self.transforms = transforms
        self.learning_rate = learning_rate
        self.optimization_frequency = optimization_frequency
        
        # Collect all optimizable parameters from transforms
        self.optimizable_params = []
        for transform in transforms:
            for param in transform.parameters():
                if param.requires_grad:
                    self.optimizable_params.append(param)
        
        # Initialize optimizer for transform parameters
        if self.optimizable_params:
            self.optimizer = torch.optim.Adam(self.optimizable_params, lr=learning_rate)
        else:
            self.optimizer = None
        
        # Performance tracking
        self.step_count = 0
        self.performance_history = []
        self.optimization_history = []
        
        # Performance-based adaptation
        self.best_performance = float('-inf')
        self.best_params = None
        self.patience = 20  # Steps to wait before reverting to best params
        self.no_improvement_count = 0
    
    def step(self, performance_metric: float):
        """Update transform parameters based on performance feedback.
        
        Args:
            performance_metric: Performance metric to optimize (higher is better)
        """
        self.step_count += 1
        self.performance_history.append(performance_metric)
        
        # Check if this is the best performance so far
        if performance_metric > self.best_performance:
            self.best_performance = performance_metric
            self.best_params = self._get_current_params()
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
        
        # Optimize parameters at specified frequency
        if self.step_count % self.optimization_frequency == 0 and self.optimizer is not None:
            self._optimize_transforms(performance_metric)
        
        # Revert to best parameters if no improvement for too long
        if self.no_improvement_count >= self.patience and self.best_params is not None:
            self._restore_best_params()
            self.no_improvement_count = 0
    
    def _optimize_transforms(self, performance_metric: float):
        """Perform optimization step on transform parameters."""
        if not self.optimizable_params or self.optimizer is None:
            return
        
        # Compute loss as negative performance (we want to maximize performance)
        loss = -performance_metric
        
        # Add regularization from transforms
        for transform in self.transforms:
            loss += transform.compute_regularization_loss()
        
        # Gradient step
        self.optimizer.zero_grad()
        
        # Since we can't backpropagate through performance_metric directly,
        # we'll use a simple gradient-free optimization approach
        # This is a simplified version - in practice, you'd use techniques like
        # reinforcement learning or evolutionary strategies
        
        # For now, we'll use random perturbations and keep changes that improve performance
        current_params = self._get_current_params()
        
        # Apply small random perturbations
        perturbation_scale = 0.01 * self.learning_rate
        for param in self.optimizable_params:
            with torch.no_grad():
                perturbation = torch.randn_like(param) * perturbation_scale
                param.add_(perturbation)
        
        # Record optimization step
        self.optimization_history.append({
            'step': self.step_count,
            'performance': performance_metric,
            'loss': loss.item() if hasattr(loss, 'item') else loss,
            'num_params': len(self.optimizable_params)
        })
    
    def _get_current_params(self) -> Dict[int, torch.Tensor]:
        """Get current parameter values."""
        params = {}
        for i, param in enumerate(self.optimizable_params):
            params[i] = param.data.clone()
        return params
    
    def _restore_best_params(self):
        """Restore best parameters found so far."""
        if self.best_params is None:
            return
        
        with torch.no_grad():
            for i, param in enumerate(self.optimizable_params):
                if i in self.best_params:
                    param.data.copy_(self.best_params[i])
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics and progress information."""
        if not self.performance_history:
            return {
                'total_steps': self.step_count,
                'optimizations_performed': len(self.optimization_history),
                'best_performance': self.best_performance,
                'current_performance': 0.0,
                'num_optimizable_params': len(self.optimizable_params)
            }
        
        recent_performance = self.performance_history[-10:] if len(self.performance_history) >= 10 else self.performance_history
        
        stats = {
            'total_steps': self.step_count,
            'optimizations_performed': len(self.optimization_history),
            'best_performance': self.best_performance,
            'current_performance': self.performance_history[-1],
            'recent_avg_performance': sum(recent_performance) / len(recent_performance),
            'num_optimizable_params': len(self.optimizable_params),
            'no_improvement_count': self.no_improvement_count,
            'patience_remaining': max(0, self.patience - self.no_improvement_count)
        }
        
        # Add performance trend
        if len(self.performance_history) >= 5:
            recent_trend = self.performance_history[-5:]
            early_avg = sum(recent_trend[:2]) / 2
            late_avg = sum(recent_trend[-2:]) / 2
            stats['performance_trend'] = late_avg - early_avg
        else:
            stats['performance_trend'] = 0.0
        
        return stats
    
    def reset_optimization(self):
        """Reset optimization state."""
        self.step_count = 0
        self.performance_history.clear()
        self.optimization_history.clear()
        self.best_performance = float('-inf')
        self.best_params = None
        self.no_improvement_count = 0
        
        # Reset optimizer state
        if self.optimizer is not None:
            self.optimizer = torch.optim.Adam(self.optimizable_params, lr=self.learning_rate)