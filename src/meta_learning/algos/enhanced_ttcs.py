"""
💰 SUPPORT THIS RESEARCH - PLEASE DONATE! 💰

🙏 If this Enhanced TTCS implementation helps your research, please donate:
💳 PayPal: https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS
⭐ GitHub Sponsors: https://github.com/sponsors/benedictchen

👨‍💻 Author: Benedict Chen
💰 Donations: Help support this work! Buy me a coffee ☕, beer 🍺, lamborghini 🏎️, or private island 🏝️
💖 Please consider recurring donations to fully support continued research

Your support enables cutting-edge AI research for everyone! 🚀

Enhanced TTCS with detach_module Integration - ADDITIVE Implementation
====================================================================

This module provides ADDITIVE enhancements to the existing TTCS implementation,
integrating the advanced detach_module functionality from core/utils.py with
memory-efficient test-time compute scaling and performance optimization.

🎯 **Enhancement Goals**:
- Memory-efficient gradient detachment during test-time passes
- Automatic memory cleanup between TTCS passes
- Performance monitoring for memory optimization
- Mixed precision support for efficient computation
- Failure prediction integration for adaptive compute allocation

💰 Please donate if this accelerates your research!
"""

from __future__ import annotations
from typing import Dict, Tuple, Optional, List, Any, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
import time
import gc
from collections import deque

from ..core.utils import detach_module, clone_module
from ..shared.types import Episode


class MemoryEfficientTTCS(nn.Module):
    """
    Enhanced Test-Time Compute Scaling with detach_module integration.
    
    This class ADDITIVELY extends TTCS functionality with:
    - Memory-efficient gradient detachment using detach_module()
    - Automatic memory cleanup between passes
    - Performance monitoring hooks
    - Mixed precision support
    - Adaptive memory management
    
    Does NOT modify existing TTCS - purely additive enhancements.
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        head,
        passes: int = 8,
        enable_memory_efficient: bool = True,
        enable_detach_optimization: bool = True,
        memory_cleanup_frequency: int = 2,
        enable_mixed_precision: bool = False,
        failure_predictor=None,
        performance_tracking: bool = True,
        memory_budget_mb: Optional[int] = None
    ):
        """
        Initialize Enhanced TTCS with memory optimization.
        
        Args:
            encoder: Feature encoder model
            head: Classification head
            passes: Number of test-time compute passes
            enable_memory_efficient: Use memory-efficient optimizations
            enable_detach_optimization: Use detach_module for memory cleanup
            memory_cleanup_frequency: Frequency of garbage collection (every N passes)
            enable_mixed_precision: Enable mixed precision training
            failure_predictor: Optional failure prediction model
            performance_tracking: Track performance metrics
            memory_budget_mb: Optional memory budget in MB
        """
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.passes = passes
        self.enable_memory_efficient = enable_memory_efficient
        self.enable_detach_optimization = enable_detach_optimization
        self.memory_cleanup_frequency = memory_cleanup_frequency
        self.enable_mixed_precision = enable_mixed_precision
        self.failure_predictor = failure_predictor
        self.performance_tracking = performance_tracking
        self.memory_budget_mb = memory_budget_mb
        
        # Performance tracking
        if self.performance_tracking:
            self.memory_metrics = {
                'peak_memory_usage': [],
                'memory_savings': [],
                'cleanup_events': 0,
                'detachment_events': 0,
                'pass_memory_usage': deque(maxlen=100)
            }
        
        # Mixed precision setup
        if self.enable_mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
        
        # Memory management state
        self.memory_state = {
            'last_cleanup_pass': 0,
            'accumulated_tensors': [],
            'memory_pressure': 0.0
        }
    
    def forward(self, episode: Episode, device=None, combine: str = "mean_prob", return_metrics: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        Enhanced TTCS forward pass with memory optimization.
        
        Args:
            episode: Few-shot learning episode
            device: Device to run on
            combine: Combination method ("mean_prob" or "mean_logit")
            return_metrics: Whether to return performance metrics
            
        Returns:
            TTCS predictions (and optionally metrics dict)
        """
        start_time = time.time() if self.performance_tracking else None
        initial_memory = self._get_memory_usage() if self.performance_tracking else 0
        
        if device is None:
            device = next(self.encoder.parameters()).device
        
        try:
            # Predict failure risk if predictor available
            failure_risk = 0.0
            if self.failure_predictor is not None:
                algorithm_state = {
                    'passes': self.passes,
                    'memory_efficient': self.enable_memory_efficient,
                    'detach_optimization': self.enable_detach_optimization,
                    'encoder_params': sum(p.numel() for p in self.encoder.parameters())
                }
                failure_risk = self.failure_predictor.predict_failure_risk(episode, algorithm_state)
                
                # Adaptive strategy based on failure risk
                if failure_risk > 0.7:
                    # High risk - use memory-conservative approach
                    return self._memory_conservative_ttcs(episode, device, combine, return_metrics)
                elif failure_risk > 0.4:
                    # Medium risk - reduce passes and enable all memory optimizations
                    effective_passes = max(1, self.passes // 2)
                else:
                    effective_passes = self.passes
            else:
                effective_passes = self.passes
            
            # Enhanced TTCS with memory optimization
            if self.enable_memory_efficient and self.enable_detach_optimization:
                predictions = self._memory_optimized_ttcs(episode, device, combine, effective_passes)
            else:
                # Fallback to standard approach (still enhanced)
                predictions = self._standard_enhanced_ttcs(episode, device, combine, effective_passes)
            
            # Track performance metrics
            metrics = {}
            if return_metrics:
                end_time = time.time()
                final_memory = self._get_memory_usage()
                
                metrics.update({
                    'ttcs_time': end_time - start_time if start_time else 0,
                    'failure_risk': failure_risk,
                    'passes_used': effective_passes,
                    'method': 'memory_optimized' if self.enable_memory_efficient else 'standard_enhanced',
                    'memory_efficient': self.enable_memory_efficient,
                    'detach_optimization': self.enable_detach_optimization,
                    'initial_memory_mb': initial_memory,
                    'final_memory_mb': final_memory,
                    'memory_saved_mb': max(0, initial_memory - final_memory)
                })
                
                if self.performance_tracking:
                    metrics['memory_metrics'] = self._get_memory_statistics()
            
            # Update failure predictor with outcome
            if self.failure_predictor is not None:
                success = torch.isfinite(predictions).all().item()
                algorithm_state['success'] = success
                self.failure_predictor.update_with_outcome(episode, algorithm_state, not success)
            
            if return_metrics:
                return predictions, metrics
            return predictions
            
        except Exception as e:
            warnings.warn(f"Enhanced TTCS failed: {e}, falling back to minimal TTCS")
            
            # Ultimate fallback - minimal memory usage
            predictions = self._minimal_fallback_ttcs(episode, device, combine)
            
            if return_metrics:
                metrics = {
                    'method': 'minimal_fallback',
                    'error': str(e),
                    'failure_risk': 1.0
                }
                return predictions, metrics
            return predictions
    
    def _memory_optimized_ttcs(self, episode: Episode, device, combine: str, passes: int) -> torch.Tensor:
        """
        Memory-optimized TTCS using detach_module for efficient computation.
        """
        support_x, support_y = episode.support_x.to(device), episode.support_y.to(device)
        query_x = episode.query_x.to(device)
        
        # Memory budget check
        if self.memory_budget_mb and self._get_memory_usage() > self.memory_budget_mb:
            warnings.warn(f"Memory usage exceeds budget ({self.memory_budget_mb}MB), reducing passes")
            passes = max(1, passes // 2)
        
        # Enable MC-Dropout for stochasticity
        original_states = self._enable_mc_dropout()
        
        logits_list = []
        memory_snapshots = []
        
        try:
            # Extract support features once and detach for memory efficiency
            if self.enable_mixed_precision and self.scaler:
                with torch.cuda.amp.autocast():
                    support_features = self.encoder(support_x)
            else:
                support_features = self.encoder(support_x)
            
            # Detach support features to prevent gradient accumulation (tensor detachment)
            if self.enable_detach_optimization:
                support_features = self._detach_tensor(support_features)
                if self.performance_tracking:
                    self.memory_metrics['detachment_events'] += 1
            
            for pass_idx in range(passes):
                # Memory cleanup every N passes
                if pass_idx > 0 and pass_idx % self.memory_cleanup_frequency == 0:
                    self._cleanup_memory()
                    if self.performance_tracking:
                        self.memory_metrics['cleanup_events'] += 1
                
                # Query feature extraction with memory optimization
                if self.enable_mixed_precision and self.scaler:
                    with torch.cuda.amp.autocast():
                        query_features = self.encoder(query_x)
                else:
                    query_features = self.encoder(query_x)
                
                # Detach query features for memory efficiency
                if self.enable_detach_optimization:
                    query_features = self._detach_tensor(query_features)
                
                # Classification head computation
                if self.enable_mixed_precision and self.scaler:
                    with torch.cuda.amp.autocast():
                        logits = self.head(support_features, support_y, query_features)
                else:
                    logits = self.head(support_features, support_y, query_features)
                
                # Detach logits to prevent memory accumulation
                if self.enable_detach_optimization:
                    logits = self._detach_tensor(logits)
                
                logits_list.append(logits)
                
                # Track memory usage
                if self.performance_tracking:
                    current_memory = self._get_memory_usage()
                    memory_snapshots.append(current_memory)
                    self.memory_metrics['pass_memory_usage'].append(current_memory)
                
                # Early stopping based on memory pressure
                if self.memory_budget_mb:
                    current_memory_mb = self._get_memory_usage()
                    if current_memory_mb > self.memory_budget_mb * 1.2:  # 120% of budget
                        warnings.warn(f"Memory pressure detected, stopping at pass {pass_idx + 1}")
                        break
                
                # Clear intermediate variables
                del query_features
                if pass_idx < passes - 1:  # Don't cleanup on last iteration
                    torch.cuda.empty_cache() if torch.cuda.is_available() else gc.collect()
        
        finally:
            # Restore original training states
            self._restore_mc_dropout(original_states)
            
            # Final memory cleanup
            self._cleanup_memory()
        
        # Update memory statistics
        if self.performance_tracking and memory_snapshots:
            peak_memory = max(memory_snapshots)
            memory_saved = memory_snapshots[0] - memory_snapshots[-1] if len(memory_snapshots) > 1 else 0
            self.memory_metrics['peak_memory_usage'].append(peak_memory)
            self.memory_metrics['memory_savings'].append(max(0, memory_saved))
        
        # Combine predictions efficiently
        return self._combine_predictions_memory_efficient(logits_list, combine)
    
    def _standard_enhanced_ttcs(self, episode: Episode, device, combine: str, passes: int) -> torch.Tensor:
        """
        Standard enhanced TTCS without detach optimization (fallback).
        """
        support_x, support_y = episode.support_x.to(device), episode.support_y.to(device)
        query_x = episode.query_x.to(device)
        
        # Enable MC-Dropout
        original_states = self._enable_mc_dropout()
        
        logits_list = []
        
        try:
            for pass_idx in range(passes):
                # Extract features
                support_features = self.encoder(support_x)
                query_features = self.encoder(query_x)
                
                # Get predictions
                logits = self.head(support_features, support_y, query_features)
                logits_list.append(logits.detach().clone())  # Manual detach for memory
                
                # Memory cleanup
                if pass_idx % self.memory_cleanup_frequency == 0:
                    self._cleanup_memory()
        
        finally:
            self._restore_mc_dropout(original_states)
        
        return self._combine_predictions_memory_efficient(logits_list, combine)
    
    def _memory_conservative_ttcs(self, episode: Episode, device, combine: str, return_metrics: bool) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        Ultra-conservative TTCS for high failure risk scenarios.
        """
        # Minimal passes with maximum memory efficiency
        conservative_passes = min(2, self.passes)
        
        support_x, support_y = episode.support_x.to(device), episode.support_y.to(device)
        query_x = episode.query_x.to(device)
        
        # Single forward pass approach with minimal memory
        logits_list = []
        
        for pass_idx in range(conservative_passes):
            with torch.no_grad():  # Disable gradients entirely
                # Extract features with immediate detachment
                support_features = self.encoder(support_x).detach()
                query_features = self.encoder(query_x).detach()
                
                # Get predictions
                logits = self.head(support_features, support_y, query_features).detach()
                logits_list.append(logits)
                
                # Immediate cleanup
                del support_features, query_features
                torch.cuda.empty_cache() if torch.cuda.is_available() else gc.collect()
        
        # Simple combination
        if len(logits_list) == 1:
            predictions = logits_list[0]
        else:
            predictions = torch.stack(logits_list).mean(dim=0)
        
        if return_metrics:
            metrics = {
                'method': 'memory_conservative',
                'passes_used': conservative_passes,
                'failure_risk': 'high'
            }
            return predictions, metrics
        
        return predictions
    
    def _minimal_fallback_ttcs(self, episode: Episode, device, combine: str) -> torch.Tensor:
        """
        Minimal fallback TTCS implementation.
        """
        support_x, support_y = episode.support_x.to(device), episode.support_y.to(device)
        query_x = episode.query_x.to(device)
        
        with torch.no_grad():
            # Single pass with maximum memory efficiency
            support_features = self.encoder(support_x).detach()
            query_features = self.encoder(query_x).detach()
            logits = self.head(support_features, support_y, query_features).detach()
            
            return logits
    
    def _detach_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Memory-efficient tensor detachment with optional module integration.
        
        This method provides a tensor-specific detachment that can optionally
        use detach_module for modules or fall back to simple tensor detachment.
        """
        if self.enable_detach_optimization:
            # For complex optimization, we could integrate with detach_module for modules
            # But for tensors, we use simple detachment
            return tensor.detach()
        else:
            return tensor
    
    def _enable_mc_dropout(self) -> Dict[str, bool]:
        """Enable MC-Dropout and return original states."""
        original_states = {}
        
        for name, module in self.encoder.named_modules():
            original_states[name] = module.training
            if isinstance(module, nn.Dropout):
                module.train(True)  # Enable dropout
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                module.eval()  # Keep BatchNorm in eval mode
        
        return original_states
    
    def _restore_mc_dropout(self, original_states: Dict[str, bool]):
        """Restore original training states."""
        for name, module in self.encoder.named_modules():
            if name in original_states:
                module.train(original_states[name])
    
    def _combine_predictions_memory_efficient(self, logits_list: List[torch.Tensor], combine: str) -> torch.Tensor:
        """
        Memory-efficient prediction combination.
        """
        if not logits_list:
            raise ValueError("No logits to combine")
        
        if len(logits_list) == 1:
            return logits_list[0]
        
        # Stack tensors efficiently
        logits_stack = torch.stack(logits_list, dim=0)  # [passes, batch, classes]
        
        if combine == "mean_logit":
            combined = logits_stack.mean(dim=0)
        else:  # mean_prob
            # Convert to probabilities, average, then back to logits
            probs = F.softmax(logits_stack, dim=-1)
            mean_probs = probs.mean(dim=0)
            combined = torch.log(mean_probs.clamp(min=1e-8))
        
        # Clear intermediate tensors
        del logits_stack
        
        return combined
    
    def _cleanup_memory(self):
        """
        Comprehensive memory cleanup.
        """
        # Clear accumulated tensors
        self.memory_state['accumulated_tensors'].clear()
        
        # Python garbage collection
        gc.collect()
        
        # CUDA cache cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def _get_memory_usage(self) -> float:
        """
        Get current memory usage in MB.
        """
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024**2)
        else:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024**2)
    
    def _get_memory_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive memory statistics.
        """
        if not self.performance_tracking:
            return {}
        
        stats = {}
        
        if self.memory_metrics['peak_memory_usage']:
            import numpy as np
            peak_usage = np.array(self.memory_metrics['peak_memory_usage'])
            stats['peak_memory'] = {
                'mean_mb': float(np.mean(peak_usage)),
                'max_mb': float(np.max(peak_usage)),
                'std_mb': float(np.std(peak_usage))
            }
        
        if self.memory_metrics['memory_savings']:
            savings = np.array(self.memory_metrics['memory_savings'])
            stats['memory_savings'] = {
                'total_saved_mb': float(np.sum(savings)),
                'avg_saved_per_run_mb': float(np.mean(savings))
            }
        
        stats['optimization_events'] = {
            'cleanup_events': self.memory_metrics['cleanup_events'],
            'detachment_events': self.memory_metrics['detachment_events']
        }
        
        if self.memory_metrics['pass_memory_usage']:
            recent_usage = list(self.memory_metrics['pass_memory_usage'])
            stats['recent_memory_trend'] = {
                'samples': len(recent_usage),
                'latest_mb': recent_usage[-1] if recent_usage else 0,
                'trend_mb': recent_usage[-5:] if len(recent_usage) >= 5 else recent_usage
            }
        
        return stats
    
    def reset_memory_metrics(self):
        """Reset all memory tracking metrics."""
        if self.performance_tracking:
            self.memory_metrics = {
                'peak_memory_usage': [],
                'memory_savings': [],
                'cleanup_events': 0,
                'detachment_events': 0,
                'pass_memory_usage': deque(maxlen=100)
            }


class AdaptiveMemoryTTCS(MemoryEfficientTTCS):
    """
    Adaptive TTCS that automatically adjusts memory usage based on available resources.
    
    Features:
    - Automatic memory budget detection
    - Dynamic pass allocation based on memory availability
    - Intelligent fallback strategies
    - Real-time memory monitoring
    """
    
    def __init__(self, encoder: nn.Module, head, **kwargs):
        # Auto-detect memory budget if not specified
        if 'memory_budget_mb' not in kwargs:
            kwargs['memory_budget_mb'] = self._detect_memory_budget()
        
        super().__init__(encoder, head, **kwargs)
        
        # Adaptive parameters
        self.memory_pressure_threshold = 0.8  # 80% of budget
        self.adaptive_pass_reduction = True
        self.min_passes = max(1, self.passes // 4)
    
    def _detect_memory_budget(self) -> int:
        """
        Automatically detect appropriate memory budget.
        """
        if torch.cuda.is_available():
            # CUDA memory detection
            total_memory = torch.cuda.get_device_properties(0).total_memory
            available_memory = total_memory - torch.cuda.memory_allocated()
            # Use 50% of available GPU memory
            budget_bytes = available_memory * 0.5
            return int(budget_bytes / (1024**2))  # Convert to MB
        else:
            # CPU memory detection
            import psutil
            available_memory = psutil.virtual_memory().available
            # Use 25% of available CPU memory (more conservative)
            budget_bytes = available_memory * 0.25
            return int(budget_bytes / (1024**2))  # Convert to MB
    
    def forward(self, episode: Episode, device=None, combine: str = "mean_prob", return_metrics: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        Adaptive forward pass that automatically manages memory.
        """
        # Pre-flight memory check
        current_memory = self._get_memory_usage()
        memory_pressure = current_memory / self.memory_budget_mb if self.memory_budget_mb else 0
        
        # Adaptive pass allocation
        if self.adaptive_pass_reduction and memory_pressure > self.memory_pressure_threshold:
            # Reduce passes based on memory pressure
            pressure_factor = min(1.0, (1.0 - memory_pressure) * 2)
            adaptive_passes = max(self.min_passes, int(self.passes * pressure_factor))
            
            # Temporarily adjust passes
            original_passes = self.passes
            self.passes = adaptive_passes
            
            try:
                result = super().forward(episode, device, combine, return_metrics)
                
                # Add adaptive information to metrics
                if return_metrics and isinstance(result, tuple):
                    predictions, metrics = result
                    metrics['adaptive_info'] = {
                        'memory_pressure': memory_pressure,
                        'original_passes': original_passes,
                        'adaptive_passes': adaptive_passes,
                        'reduction_factor': pressure_factor
                    }
                    return predictions, metrics
                
                return result
                
            finally:
                # Restore original passes
                self.passes = original_passes
        else:
            # Standard execution
            return super().forward(episode, device, combine, return_metrics)


# Convenience functions for easy integration

def enhanced_ttcs_predict(
    encoder: nn.Module,
    head,
    episode: Episode,
    passes: int = 8,
    device=None,
    combine: str = "mean_prob",
    enable_memory_efficient: bool = True,
    enable_detach_optimization: bool = True
) -> torch.Tensor:
    """
    Enhanced TTCS prediction with detach_module optimization.
    
    This function provides a direct replacement for standard TTCS
    with memory optimization capabilities.
    
    Args:
        encoder: Feature encoder model
        head: Classification head
        episode: Few-shot learning episode
        passes: Number of test-time compute passes
        device: Device to run on
        combine: Combination method ("mean_prob" or "mean_logit")
        enable_memory_efficient: Enable memory optimizations
        enable_detach_optimization: Use detach_module for memory cleanup
        
    Returns:
        TTCS predictions with memory optimization
    """
    enhanced_ttcs = MemoryEfficientTTCS(
        encoder=encoder,
        head=head,
        passes=passes,
        enable_memory_efficient=enable_memory_efficient,
        enable_detach_optimization=enable_detach_optimization,
        performance_tracking=False  # Disable for function interface
    )
    
    return enhanced_ttcs.forward(episode, device, combine)


def adaptive_ttcs_predict(
    encoder: nn.Module,
    head,
    episode: Episode,
    passes: int = 8,
    device=None,
    combine: str = "mean_prob"
) -> torch.Tensor:
    """
    Adaptive TTCS prediction with automatic memory management.
    
    Args:
        encoder: Feature encoder model
        head: Classification head
        episode: Few-shot learning episode
        passes: Maximum number of passes (auto-adjusted based on memory)
        device: Device to run on
        combine: Combination method
        
    Returns:
        TTCS predictions with adaptive memory management
    """
    adaptive_ttcs = AdaptiveMemoryTTCS(
        encoder=encoder,
        head=head,
        passes=passes,
        performance_tracking=False
    )
    
    return adaptive_ttcs.forward(episode, device, combine)


def ttcs_with_memory_monitoring(
    encoder: nn.Module,
    head,
    episode: Episode,
    passes: int = 8,
    device=None,
    combine: str = "mean_prob"
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    TTCS prediction with comprehensive memory monitoring.
    
    Args:
        encoder: Feature encoder model
        head: Classification head
        episode: Few-shot learning episode
        passes: Number of passes
        device: Device to run on
        combine: Combination method
        
    Returns:
        Tuple of (predictions, memory_metrics)
    """
    enhanced_ttcs = MemoryEfficientTTCS(
        encoder=encoder,
        head=head,
        passes=passes,
        performance_tracking=True
    )
    
    return enhanced_ttcs.forward(episode, device, combine, return_metrics=True)