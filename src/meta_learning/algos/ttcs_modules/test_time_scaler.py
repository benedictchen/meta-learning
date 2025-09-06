"""
TestTimeComputeScaler Class Module
==================================

Main TestTimeComputeScaler class for meta-learning scenarios.
Extracted from large ttcs.py for better maintainability.

FIRST PUBLIC IMPLEMENTATION of Test-Time Compute Scaling!
"""

from __future__ import annotations
import hashlib
import time
import warnings
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import torch
import torch.nn as nn

from .core_predictor import ttcs_predict, ttcs_predict_advanced


class TestTimeComputeScaler:
    """
    Test-Time Compute Scaling implementation for meta-learning.
    
    This is the FIRST PUBLIC IMPLEMENTATION of Test-Time Compute Scaling!
    
    Features:
    - Monte Carlo Dropout for uncertainty estimation
    - Test-Time Augmentation for improved robustness
    - Ensemble predictions across multiple stochastic passes
    - Configurable combining strategies
    - Performance monitoring and optimization
    """
    
    def __init__(self, encoder: nn.Module, head: nn.Module, 
                 passes: int = 8, combine: str = "mean_prob",
                 enable_mc_dropout: bool = True, enable_tta: bool = True,
                 image_size: int = 32, device: Optional[torch.device] = None):
        """
        Initialize TestTimeComputeScaler.
        
        Args:
            encoder: Feature encoder network
            head: Classification head
            passes: Number of stochastic forward passes
            combine: Combination strategy ('mean_prob' or 'mean_logit')
            enable_mc_dropout: Enable Monte Carlo dropout
            enable_tta: Enable test-time augmentation
            image_size: Input image size for TTA
            device: Device to run computation on
        """
        self.encoder = encoder
        self.head = head
        self.passes = passes
        self.combine = combine
        self.enable_mc_dropout = enable_mc_dropout
        self.enable_tta = enable_tta
        self.image_size = image_size
        self.device = device if device is not None else next(encoder.parameters()).device
        
        # Performance tracking
        self.total_predictions = 0
        self.total_time_spent = 0.0
        self.prediction_history = []
        
        # Validation
        self._validate_configuration()
    
    def _validate_configuration(self):
        """Validate TTCS configuration."""
        if self.passes < 1:
            raise ValueError(f"Number of passes must be >= 1, got {self.passes}")
        
        if self.combine not in ["mean_prob", "mean_logit"]:
            raise ValueError(f"Unknown combine strategy: {self.combine}")
        
        if self.enable_mc_dropout and not self._has_dropout_layers():
            warnings.warn(
                "MC-Dropout enabled but no dropout layers found in encoder or head. "
                "TTCS will still work but may have limited uncertainty estimation."
            )
    
    def _has_dropout_layers(self) -> bool:
        """Check if model has dropout layers for MC-Dropout."""
        for module in [self.encoder, self.head]:
            for layer in module.modules():
                if isinstance(layer, nn.Dropout):
                    return True
        return False
    
    def predict(self, episode, return_uncertainty: bool = False, 
                advanced_options: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """
        Make TTCS predictions on episode.
        
        Args:
            episode: Episode with support and query data
            return_uncertainty: Return uncertainty estimates
            advanced_options: Advanced configuration options
            
        Returns:
            Predictions tensor or dict with uncertainty if requested
        """
        start_time = time.time()
        
        if return_uncertainty or advanced_options is not None:
            # Use advanced prediction function
            options = advanced_options or {}
            results = ttcs_predict_advanced(
                encoder=self.encoder,
                head=self.head,
                episode=episode,
                passes=self.passes,
                image_size=self.image_size,
                device=self.device,
                combine=self.combine,
                enable_mc_dropout=self.enable_mc_dropout,
                enable_tta=self.enable_tta,
                uncertainty_estimation=return_uncertainty,
                **options
            )
            
            if return_uncertainty:
                return results  # Return full dict with uncertainty
            else:
                return results['logits']
        else:
            # Use basic prediction function
            logits = ttcs_predict(
                encoder=self.encoder,
                head=self.head,
                episode=episode,
                passes=self.passes,
                image_size=self.image_size,
                device=self.device,
                combine=self.combine,
                enable_mc_dropout=self.enable_mc_dropout,
                enable_tta=self.enable_tta
            )
            
            # Track performance
            prediction_time = time.time() - start_time
            self._update_performance_stats(prediction_time, episode)
            
            return logits
    
    def _update_performance_stats(self, prediction_time: float, episode):
        """Update performance tracking statistics."""
        self.total_predictions += 1
        self.total_time_spent += prediction_time
        
        # Track recent prediction info
        prediction_info = {
            'time': prediction_time,
            'query_size': len(episode.query_x),
            'support_size': len(episode.support_x),
            'timestamp': time.time()
        }
        
        self.prediction_history.append(prediction_info)
        
        # Keep history bounded
        if len(self.prediction_history) > 1000:
            self.prediction_history = self.prediction_history[-800:]
    
    def batch_predict(self, episodes: List, return_uncertainty: bool = False,
                     show_progress: bool = False) -> List[torch.Tensor]:
        """
        Make TTCS predictions on batch of episodes.
        
        Args:
            episodes: List of episodes
            return_uncertainty: Return uncertainty estimates
            show_progress: Show progress bar
            
        Returns:
            List of predictions
        """
        predictions = []
        
        if show_progress:
            try:
                from tqdm import tqdm
                episodes_iter = tqdm(episodes, desc="TTCS Batch Prediction")
            except ImportError:
                episodes_iter = episodes
                if show_progress:
                    print(f"Processing {len(episodes)} episodes...")
        else:
            episodes_iter = episodes
        
        for i, episode in enumerate(episodes_iter):
            pred = self.predict(episode, return_uncertainty=return_uncertainty)
            predictions.append(pred)
            
            if show_progress and 'tqdm' not in locals():
                if (i + 1) % max(1, len(episodes) // 10) == 0:
                    print(f"Processed {i + 1}/{len(episodes)} episodes")
        
        return predictions
    
    def configure(self, **kwargs):
        """
        Update TTCS configuration.
        
        Args:
            **kwargs: Configuration parameters to update
        """
        valid_params = {
            'passes', 'combine', 'enable_mc_dropout', 'enable_tta', 'image_size'
        }
        
        for key, value in kwargs.items():
            if key not in valid_params:
                warnings.warn(f"Unknown configuration parameter: {key}")
                continue
            
            setattr(self, key, value)
        
        # Re-validate after configuration change
        self._validate_configuration()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        if not self.prediction_history:
            return {
                'total_predictions': self.total_predictions,
                'total_time_spent': self.total_time_spent,
                'avg_time_per_prediction': 0.0
            }
        
        times = [p['time'] for p in self.prediction_history]
        query_sizes = [p['query_size'] for p in self.prediction_history]
        
        return {
            'total_predictions': self.total_predictions,
            'total_time_spent': self.total_time_spent,
            'avg_time_per_prediction': np.mean(times),
            'std_time_per_prediction': np.std(times),
            'min_time_per_prediction': np.min(times),
            'max_time_per_prediction': np.max(times),
            'avg_query_size': np.mean(query_sizes),
            'configuration': {
                'passes': self.passes,
                'combine': self.combine,
                'enable_mc_dropout': self.enable_mc_dropout,
                'enable_tta': self.enable_tta,
                'image_size': self.image_size
            },
            'efficiency_metrics': {
                'predictions_per_second': self.total_predictions / self.total_time_spent if self.total_time_spent > 0 else 0.0,
                'time_per_pass': np.mean(times) / self.passes if times else 0.0
            }
        }
    
    def reset_stats(self):
        """Reset performance statistics."""
        self.total_predictions = 0
        self.total_time_spent = 0.0
        self.prediction_history.clear()
    
    def benchmark(self, episode, num_runs: int = 10) -> Dict[str, float]:
        """
        Benchmark TTCS performance on episode.
        
        Args:
            episode: Episode for benchmarking
            num_runs: Number of benchmark runs
            
        Returns:
            Benchmark statistics
        """
        times = []
        
        # Warmup run
        self.predict(episode)
        
        # Benchmark runs
        for _ in range(num_runs):
            start_time = time.time()
            self.predict(episode)
            elapsed_time = time.time() - start_time
            times.append(elapsed_time)
        
        return {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'median_time': np.median(times),
            'total_passes': self.passes,
            'time_per_pass': np.mean(times) / self.passes,
            'query_size': len(episode.query_x),
            'support_size': len(episode.support_x)
        }
    
    def __repr__(self) -> str:
        """String representation of TTCS instance."""
        return (
            f"TestTimeComputeScaler(passes={self.passes}, combine='{self.combine}', "
            f"mc_dropout={self.enable_mc_dropout}, tta={self.enable_tta}, "
            f"predictions={self.total_predictions})"
        )