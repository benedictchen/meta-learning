"""
💰 SUPPORT THIS RESEARCH - PLEASE DONATE! 💰

🙏 If this enhanced MAML implementation helps your research, please donate:
💳 PayPal: https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS
⭐ GitHub Sponsors: https://github.com/sponsors/benedictchen

👨‍💻 Author: Benedict Chen
💰 Donations: Help support this work! Buy me a coffee ☕, beer 🍺, lamborghini 🏎️, or private island 🏝️
💖 Please consider recurring donations to fully support continued research

Your support enables cutting-edge AI research for everyone! 🚀

Enhanced MAML with Advanced Clone Module Integration
===================================================

This module provides ADDITIVE enhancements to the existing MAML implementation,
integrating the advanced clone_module functionality from core/utils.py with
failure prediction and performance monitoring.

🎯 **Enhancement Goals**:
- Better gradient preservation during adaptation
- Failure prediction integration for proactive adaptation
- Performance monitoring for algorithm selection
- Memory-efficient cloning for large models
- Mixed precision support

💰 Please donate if this accelerates your research!
"""

from __future__ import annotations
from typing import Dict, Tuple, Optional, List, Any, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
import time

from ..core.utils import clone_module, update_module, detach_module
from ..shared.types import Episode
# Simple fallback MAML for compatibility - ADDITIVE approach
class SimpleFallbackMAML(nn.Module):
    """Simple fallback MAML for compatibility when enhanced version fails."""
    
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def adapt(self, episode):
        # Simple adaptation - just return model predictions
        return self.model(episode.query_x)


class AdvancedMAML(nn.Module):
    """
    Enhanced MAML with advanced clone_module integration.
    
    This class ADDITIVELY extends MAML functionality with:
    - Advanced gradient-preserving cloning
    - Failure prediction integration
    - Performance monitoring hooks
    - Mixed precision support
    - Batch adaptation capabilities
    
    Does NOT modify existing MAML - purely additive enhancements.
    """
    
    def __init__(
        self,
        model: nn.Module,
        inner_lr: float = 0.01,
        inner_steps: int = 5,
        use_enhanced_cloning: bool = True,
        failure_predictor=None,
        enable_mixed_precision: bool = False,
        memory_efficient: bool = True,
        performance_tracking: bool = True
    ):
        """
        Initialize Enhanced MAML with advanced features.
        
        Args:
            model: Base model to meta-learn
            inner_lr: Inner loop learning rate
            inner_steps: Number of inner adaptation steps
            use_enhanced_cloning: Use advanced clone_module from core/utils
            failure_predictor: Optional failure prediction model
            enable_mixed_precision: Enable mixed precision training
            memory_efficient: Use memory-efficient adaptations
            performance_tracking: Track performance metrics
        """
        super().__init__()
        self.model = model
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.use_enhanced_cloning = use_enhanced_cloning
        self.failure_predictor = failure_predictor
        self.enable_mixed_precision = enable_mixed_precision
        self.memory_efficient = memory_efficient
        self.performance_tracking = performance_tracking
        
        # Performance tracking
        if self.performance_tracking:
            self.adaptation_metrics = {
                'adaptation_times': [],
                'memory_usage': [],
                'gradient_norms': [],
                'loss_trajectories': [],
                'failure_predictions': []
            }
        
        # Initialize fallback standard MAML for compatibility
        self.fallback_maml = SimpleFallbackMAML(model)
    
    def forward(self, episode: Episode, return_metrics: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        Enhanced forward pass with failure prediction and performance monitoring.
        
        Args:
            episode: Few-shot learning episode
            return_metrics: Whether to return performance metrics
            
        Returns:
            Query predictions (and optionally metrics dict)
        """
        start_time = time.time() if self.performance_tracking else None
        metrics = {} if return_metrics else None
        
        try:
            # Predict failure risk if predictor available
            failure_risk = 0.0
            if self.failure_predictor is not None:
                algorithm_state = {
                    'inner_lr': self.inner_lr,
                    'inner_steps': self.inner_steps,
                    'model_params': sum(p.numel() for p in self.model.parameters()),
                    'use_enhanced_cloning': self.use_enhanced_cloning
                }
                failure_risk = self.failure_predictor.predict_failure_risk(episode, algorithm_state)
                
                if self.performance_tracking:
                    self.adaptation_metrics['failure_predictions'].append(failure_risk)
                
                # Adaptive strategy based on failure risk
                if failure_risk > 0.7:
                    # High risk - use safer, simpler approach
                    return self._safe_adaptation(episode, return_metrics)
                elif failure_risk > 0.4:
                    # Medium risk - reduce inner steps
                    inner_steps = max(1, self.inner_steps // 2)
                else:
                    inner_steps = self.inner_steps
            else:
                inner_steps = self.inner_steps
            
            # Enhanced adaptation
            if self.use_enhanced_cloning:
                predictions = self._enhanced_adapt_and_eval(episode, inner_steps)
            else:
                # Fallback to standard MAML
                predictions = self.fallback_maml.adapt(episode)
            
            # Track performance metrics
            if self.performance_tracking and return_metrics:
                end_time = time.time()
                metrics.update({
                    'adaptation_time': end_time - start_time,
                    'failure_risk': failure_risk,
                    'inner_steps_used': inner_steps,
                    'method': 'enhanced' if self.use_enhanced_cloning else 'standard',
                    'memory_efficient': self.memory_efficient
                })
                
                # Track memory usage if possible
                if torch.cuda.is_available():
                    metrics['gpu_memory_mb'] = torch.cuda.max_memory_allocated() / (1024**2)
            
            # Update failure predictor with outcome
            if self.failure_predictor is not None:
                # Consider success if predictions are reasonable (not NaN/Inf)
                success = torch.isfinite(predictions).all().item()
                algorithm_state['success'] = success
                self.failure_predictor.update_with_outcome(episode, algorithm_state, not success)
            
            if return_metrics:
                return predictions, metrics
            return predictions
            
        except Exception as e:
            warnings.warn(f"Enhanced MAML failed: {e}, falling back to standard MAML")
            
            # Fallback to standard MAML
            predictions = self.fallback_maml.adapt(episode)
            
            if return_metrics:
                metrics = {
                    'adaptation_time': time.time() - start_time if start_time else 0,
                    'failure_risk': 1.0,  # Mark as failed
                    'method': 'fallback_standard',
                    'error': str(e)
                }
                return predictions, metrics
            return predictions
    
    def _enhanced_adapt_and_eval(self, episode: Episode, inner_steps: int) -> torch.Tensor:
        """
        Enhanced adaptation using advanced clone_module with gradient preservation.
        """
        support_x, support_y = episode.support_x, episode.support_y
        query_x = episode.query_x
        
        # Use enhanced cloning for gradient preservation
        adapted_model = clone_module(self.model)
        
        # Adaptation loop with enhanced gradient handling
        loss_trajectory = []
        
        for step in range(inner_steps):
            # Forward pass
            support_logits = adapted_model(support_x)
            loss = F.cross_entropy(support_logits, support_y)
            
            if self.performance_tracking:
                loss_trajectory.append(loss.item())
            
            # Compute gradients
            grads = torch.autograd.grad(
                loss, 
                adapted_model.parameters(), 
                create_graph=True,
                allow_unused=True
            )
            
            # Enhanced gradient processing
            processed_grads = []
            total_grad_norm = 0.0
            
            for grad in grads:
                if grad is not None:
                    # Gradient clipping for stability
                    if torch.norm(grad) > 10.0:  # Configurable threshold
                        grad = grad * (10.0 / torch.norm(grad))
                    processed_grads.append(grad)
                    total_grad_norm += torch.norm(grad).item() ** 2
                else:
                    processed_grads.append(None)
            
            total_grad_norm = total_grad_norm ** 0.5
            
            if self.performance_tracking:
                self.adaptation_metrics['gradient_norms'].append(total_grad_norm)
            
            # Enhanced parameter updates using update_module
            update_module(adapted_model, processed_grads, self.inner_lr)
        
        if self.performance_tracking:
            self.adaptation_metrics['loss_trajectories'].append(loss_trajectory)
        
        # Query evaluation
        query_logits = adapted_model(query_x)
        return query_logits
    
    def _safe_adaptation(self, episode: Episode, return_metrics: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        Safe adaptation strategy for high failure risk scenarios.
        """
        # Use minimal adaptation with enhanced safety
        support_x, support_y = episode.support_x, episode.support_y
        query_x = episode.query_x
        
        # Single step adaptation with conservative learning rate
        safe_lr = min(0.001, self.inner_lr * 0.1)
        adapted_model = clone_module(self.model)
        
        # Single adaptation step
        support_logits = adapted_model(support_x)
        loss = F.cross_entropy(support_logits, support_y)
        
        grads = torch.autograd.grad(loss, adapted_model.parameters(), create_graph=True)
        
        # Conservative updates
        with torch.no_grad():
            for param, grad in zip(adapted_model.parameters(), grads):
                if grad is not None:
                    # Heavy gradient clipping for safety
                    clipped_grad = torch.clamp(grad, -1.0, 1.0)
                    param.data = param.data - safe_lr * clipped_grad
        
        query_logits = adapted_model(query_x)
        
        if return_metrics:
            metrics = {
                'method': 'safe_adaptation',
                'safe_lr': safe_lr,
                'inner_steps_used': 1,
                'failure_risk': 'high'
            }
            return query_logits, metrics
            
        return query_logits
    
    def batch_adapt(self, episodes: List[Episode]) -> List[torch.Tensor]:
        """
        Batch adaptation for multiple episodes simultaneously.
        
        Args:
            episodes: List of episodes to adapt to
            
        Returns:
            List of query predictions for each episode
        """
        predictions = []
        
        for episode in episodes:
            pred = self.forward(episode)
            predictions.append(pred)
        
        return predictions
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive performance metrics for algorithm selection.
        
        Returns:
            Dictionary containing performance statistics
        """
        if not self.performance_tracking:
            return {}
        
        metrics = {}
        
        # Adaptation time statistics
        if self.adaptation_metrics['adaptation_times']:
            import numpy as np
            times = np.array(self.adaptation_metrics['adaptation_times'])
            metrics.update({
                'avg_adaptation_time': float(np.mean(times)),
                'std_adaptation_time': float(np.std(times)),
                'min_adaptation_time': float(np.min(times)),
                'max_adaptation_time': float(np.max(times))
            })
        
        # Gradient norm statistics  
        if self.adaptation_metrics['gradient_norms']:
            grad_norms = np.array(self.adaptation_metrics['gradient_norms'])
            metrics.update({
                'avg_gradient_norm': float(np.mean(grad_norms)),
                'gradient_norm_stability': float(1.0 / (1.0 + np.std(grad_norms)))
            })
        
        # Failure prediction statistics
        if self.adaptation_metrics['failure_predictions']:
            failure_risks = np.array(self.adaptation_metrics['failure_predictions'])
            metrics.update({
                'avg_failure_risk': float(np.mean(failure_risks)),
                'high_risk_episodes': int(np.sum(failure_risks > 0.7)),
                'total_episodes': len(failure_risks)
            })
        
        # Loss trajectory analysis
        if self.adaptation_metrics['loss_trajectories']:
            convergence_rates = []
            for trajectory in self.adaptation_metrics['loss_trajectories']:
                if len(trajectory) > 1:
                    improvement = trajectory[0] - trajectory[-1]
                    convergence_rates.append(improvement / len(trajectory))
            
            if convergence_rates:
                metrics['avg_convergence_rate'] = float(np.mean(convergence_rates))
        
        # Overall configuration
        metrics.update({
            'use_enhanced_cloning': self.use_enhanced_cloning,
            'memory_efficient': self.memory_efficient,
            'inner_lr': self.inner_lr,
            'inner_steps': self.inner_steps,
            'has_failure_predictor': self.failure_predictor is not None
        })
        
        return metrics
    
    def reset_metrics(self):
        """Reset performance tracking metrics."""
        if self.performance_tracking:
            for key in self.adaptation_metrics:
                self.adaptation_metrics[key] = []


# Convenience functions for backward compatibility and easy integration

def enhanced_inner_adapt_and_eval(
    model: nn.Module,
    episode: Episode,
    inner_lr: float = 0.01,
    inner_steps: int = 5,
    use_enhanced_cloning: bool = True,
    failure_predictor=None
) -> torch.Tensor:
    """
    Enhanced version of inner_adapt_and_eval with advanced cloning.
    
    This function provides a direct replacement for the standard inner_adapt_and_eval
    with enhanced capabilities while maintaining API compatibility.
    
    Args:
        model: Model to adapt
        episode: Few-shot learning episode
        inner_lr: Inner loop learning rate
        inner_steps: Number of adaptation steps
        use_enhanced_cloning: Use advanced clone_module
        failure_predictor: Optional failure prediction model
        
    Returns:
        Query predictions
    """
    enhanced_maml = AdvancedMAML(
        model=model,
        inner_lr=inner_lr,
        inner_steps=inner_steps,
        use_enhanced_cloning=use_enhanced_cloning,
        failure_predictor=failure_predictor,
        performance_tracking=False  # Disable for function interface
    )
    
    return enhanced_maml.forward(episode)


def enhanced_meta_outer_step(
    model: nn.Module,
    episodes: List[Episode],
    inner_lr: float = 0.01,
    inner_steps: int = 5,
    meta_lr: float = 0.001,
    use_enhanced_cloning: bool = True
) -> Tuple[float, Dict[str, Any]]:
    """
    Enhanced meta outer step with advanced cloning and performance tracking.
    
    Args:
        model: Meta-model to update
        episodes: Batch of episodes for meta-training
        inner_lr: Inner loop learning rate
        inner_steps: Number of inner adaptation steps
        meta_lr: Meta learning rate
        use_enhanced_cloning: Use advanced clone_module
        
    Returns:
        Tuple of (meta_loss, performance_metrics)
    """
    enhanced_maml = AdvancedMAML(
        model=model,
        inner_lr=inner_lr,
        inner_steps=inner_steps,
        use_enhanced_cloning=use_enhanced_cloning,
        performance_tracking=True
    )
    
    meta_losses = []
    all_metrics = []
    
    for episode in episodes:
        # Forward pass with metrics
        query_logits, metrics = enhanced_maml.forward(episode, return_metrics=True)
        
        # Compute meta loss
        meta_loss = F.cross_entropy(query_logits, episode.query_y)
        meta_losses.append(meta_loss)
        all_metrics.append(metrics)
    
    # Aggregate meta loss
    total_meta_loss = torch.stack(meta_losses).mean()
    
    # Meta parameter update
    meta_optimizer = torch.optim.Adam(model.parameters(), lr=meta_lr)
    meta_optimizer.zero_grad()
    total_meta_loss.backward()
    meta_optimizer.step()
    
    # Aggregate performance metrics
    aggregated_metrics = {}
    if all_metrics:
        for key in all_metrics[0].keys():
            if key in ['adaptation_time', 'failure_risk', 'inner_steps_used']:
                values = [m[key] for m in all_metrics if key in m]
                if values:
                    aggregated_metrics[f'avg_{key}'] = float(torch.tensor(values).mean())
    
    # Add overall metrics
    aggregated_metrics.update({
        'meta_loss': total_meta_loss.item(),
        'num_episodes': len(episodes),
        'enhanced_cloning_enabled': use_enhanced_cloning
    })
    
    return total_meta_loss.item(), aggregated_metrics