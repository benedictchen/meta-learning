"""
TTCS Profiler
============

Professional profiling tools for Test-Time Compute Scaling performance analysis.
Extracted from large ttcs.py for better maintainability.

FIRST PUBLIC IMPLEMENTATION of Test-Time Compute Scaling!
"""

from __future__ import annotations
import time
from typing import List, Dict, Any, Optional
import torch
import torch.nn as nn


class TTCSProfiler:
    """
    Professional profiling for TTCS performance.
    
    Features:
    - Track memory usage per pass
    - Monitor uncertainty evolution  
    - Profile compute efficiency
    - Generate optimization recommendations
    """
    
    def __init__(self, enable_memory_tracking: bool = True):
        """
        Initialize TTCS profiler.
        
        Args:
            enable_memory_tracking: Enable CUDA memory tracking
        """
        self.enable_memory_tracking = enable_memory_tracking
        self.profile_data: List[Dict[str, Any]] = []
        self.uncertainty_evolution: List[List[Dict[str, Any]]] = []
        
    def profile_ttcs_run(self, encoder: nn.Module, head: nn.Module, episode, **ttcs_kwargs) -> Dict[str, Any]:
        """
        Profile a complete TTCS run.
        
        Args:
            encoder: Feature encoder network
            head: Classification head
            episode: Episode data
            **ttcs_kwargs: TTCS configuration parameters
            
        Returns:
            Profiling results and predictions
        """
        profile_entry = {
            'timestamp': time.time(),
            'passes': ttcs_kwargs.get('passes', 8),
            'image_size': ttcs_kwargs.get('image_size', 32),
            'combine_method': ttcs_kwargs.get('combine', 'mean_prob')
        }
        
        # Memory tracking
        if self.enable_memory_tracking and torch.cuda.is_available():
            torch.cuda.synchronize()
            initial_memory = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
            profile_entry['initial_memory_mb'] = initial_memory
        
        start_time = time.time()
        
        # Run TTCS with uncertainty evolution tracking
        predictions, uncertainty_evolution = self._profile_with_uncertainty_tracking(
            encoder, head, episode, **ttcs_kwargs
        )
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Complete profile entry
        profile_entry.update({
            'total_time': total_time,
            'time_per_pass': total_time / profile_entry['passes'],
            'query_batch_size': len(episode.query_x),
            'support_batch_size': len(episode.support_x)
        })
        
        # Memory tracking completion
        if self.enable_memory_tracking and torch.cuda.is_available():
            torch.cuda.synchronize()
            final_memory = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
            profile_entry['final_memory_mb'] = final_memory
            profile_entry['memory_increase_mb'] = final_memory - initial_memory
        
        # Store profiling data
        self.profile_data.append(profile_entry)
        self.uncertainty_evolution.append(uncertainty_evolution)
        
        # Limit stored data
        if len(self.profile_data) > 1000:
            self.profile_data = self.profile_data[-500:]
            self.uncertainty_evolution = self.uncertainty_evolution[-500:]
        
        return {
            'predictions': predictions,
            'profile': profile_entry,
            'uncertainty_evolution': uncertainty_evolution
        }
    
    def _profile_with_uncertainty_tracking(self, encoder: nn.Module, head: nn.Module, 
                                         episode, **ttcs_kwargs) -> tuple:
        """
        Run TTCS with detailed uncertainty evolution tracking.
        
        Args:
            encoder: Feature encoder network
            head: Classification head
            episode: Episode data
            **ttcs_kwargs: TTCS configuration
            
        Returns:
            Tuple of (predictions, uncertainty_evolution)
        """
        from .core_predictor import _enable_dropout_for_inference, _disable_dropout_after_inference
        from .augmentation_transforms import create_tta_transforms
        
        passes = ttcs_kwargs.get('passes', 8)
        enable_mc_dropout = ttcs_kwargs.get('enable_mc_dropout', True)
        enable_tta = ttcs_kwargs.get('enable_tta', True)
        image_size = ttcs_kwargs.get('image_size', 32)
        device = ttcs_kwargs.get('device') or next(encoder.parameters()).device
        combine = ttcs_kwargs.get('combine', 'mean_prob')
        
        # Move episode to device
        support_x = episode.support_x.to(device)
        support_y = episode.support_y.to(device)
        query_x = episode.query_x.to(device)
        
        # Setup for MC dropout
        if enable_mc_dropout:
            _enable_dropout_for_inference(encoder)
            _enable_dropout_for_inference(head)
        
        # Setup TTA
        tta_transform = None
        if enable_tta and query_x.dim() == 4:  # Image data
            tta_transform = create_tta_transforms(image_size)
        
        pass_logits = []
        uncertainty_evolution = []
        
        # Multiple stochastic passes with uncertainty tracking
        for pass_idx in range(passes):
            # Apply augmentation if enabled
            if tta_transform is not None:
                # Apply TTA
                support_x_aug = torch.stack([
                    tta_transform(img) if len(img.shape) == 3 else img 
                    for img in support_x.cpu()
                ]).to(device)
                query_x_aug = torch.stack([
                    tta_transform(img) if len(img.shape) == 3 else img 
                    for img in query_x.cpu()
                ]).to(device)
            else:
                support_x_aug = support_x
                query_x_aug = query_x
            
            # Forward pass
            support_features = encoder(support_x_aug)
            query_features = encoder(query_x_aug)
            
            # Get predictions
            if hasattr(head, 'forward'):
                logits = head(support_features, support_y, query_features)
            else:
                # Compute prototypes manually
                unique_classes = torch.unique(support_y)
                prototypes = torch.zeros(len(unique_classes), support_features.size(-1), device=device)
                
                for i, cls in enumerate(unique_classes):
                    class_mask = support_y == cls
                    if class_mask.sum() > 0:
                        prototypes[i] = support_features[class_mask].mean(dim=0)
                
                distances = torch.cdist(query_features, prototypes)
                logits = -distances
            
            pass_logits.append(logits)
            
            # Track uncertainty evolution
            if len(pass_logits) >= 2:
                combined_logits = torch.stack(pass_logits).mean(dim=0)
                probs = torch.softmax(combined_logits, dim=-1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1).mean().item()
                
                uncertainty_evolution.append({
                    'pass': pass_idx + 1,
                    'entropy': entropy,
                    'max_prob': probs.max(dim=-1)[0].mean().item(),
                    'prediction_variance': torch.stack([torch.softmax(logits, dim=-1) for logits in pass_logits]).var(dim=0).mean().item()
                })
        
        # Cleanup dropout
        if enable_mc_dropout:
            _disable_dropout_after_inference(encoder)
            _disable_dropout_after_inference(head)
        
        # Combine predictions
        if combine == "mean_prob":
            probs_list = [torch.softmax(pred, dim=-1) for pred in pass_logits]
            mean_probs = torch.stack(probs_list).mean(dim=0)
            final_logits = torch.log(mean_probs + 1e-8)
        else:  # mean_logit
            final_logits = torch.stack(pass_logits).mean(dim=0)
        
        return final_logits, uncertainty_evolution
    
    def generate_optimization_recommendations(self) -> Dict[str, Any]:
        """
        Generate optimization recommendations based on profiling data.
        
        Returns:
            Dictionary containing recommendations and statistics
        """
        if not self.profile_data:
            return {'error': 'No profiling data available'}
        
        import statistics
        
        # Compute statistics
        times = [entry['total_time'] for entry in self.profile_data]
        passes = [entry['passes'] for entry in self.profile_data]
        
        avg_time = statistics.mean(times)
        avg_passes = statistics.mean(passes)
        
        recommendations = []
        
        # Time-based recommendations
        if avg_time > 2.0:
            recommendations.append("Consider reducing number of passes for faster inference")
        
        if avg_passes > 12:
            recommendations.append("High number of passes may have diminishing returns")
        
        if avg_time < 0.1:
            recommendations.append("Very fast inference - consider increasing passes for better accuracy")
        
        # Memory-based recommendations
        if self.enable_memory_tracking:
            memory_increases = [entry.get('memory_increase_mb', 0) for entry in self.profile_data if 'memory_increase_mb' in entry]
            if memory_increases:
                avg_memory_increase = statistics.mean(memory_increases)
                
                if avg_memory_increase > 500:  # 500MB threshold
                    recommendations.append("High memory usage - consider reducing batch size or passes")
                elif avg_memory_increase < 10:  # Very low memory usage
                    recommendations.append("Low memory usage - could potentially use larger batch sizes")
        
        # Uncertainty evolution recommendations
        if self.uncertainty_evolution:
            avg_entropy_reductions = []
            early_convergence_count = 0
            
            for evolution in self.uncertainty_evolution:
                if len(evolution) >= 2:
                    initial_entropy = evolution[0]['entropy']
                    final_entropy = evolution[-1]['entropy']
                    reduction = initial_entropy - final_entropy
                    avg_entropy_reductions.append(reduction)
                    
                    # Check for early convergence
                    if len(evolution) >= 4:
                        mid_entropy = evolution[len(evolution)//2]['entropy']
                        if abs(mid_entropy - final_entropy) < 0.02:  # Converged early
                            early_convergence_count += 1
            
            if avg_entropy_reductions:
                avg_reduction = statistics.mean(avg_entropy_reductions)
                
                if avg_reduction < 0.1:
                    recommendations.append("Low uncertainty reduction - consider different augmentation strategy")
                elif avg_reduction > 0.5:
                    recommendations.append("High uncertainty reduction - TTCS is working well")
                
                # Early convergence analysis
                early_convergence_rate = early_convergence_count / len(self.uncertainty_evolution)
                if early_convergence_rate > 0.7:
                    recommendations.append("Frequent early convergence - consider reducing passes for efficiency")
        
        # Efficiency recommendations
        if times and passes:
            time_per_pass_values = [t/p for t, p in zip(times, passes) if p > 0]
            if time_per_pass_values:
                avg_time_per_pass = statistics.mean(time_per_pass_values)
                
                if avg_time_per_pass > 0.5:
                    recommendations.append("Slow per-pass time - check model complexity and data size")
        
        return {
            'recommendations': recommendations,
            'statistics': {
                'avg_time_seconds': avg_time,
                'std_time_seconds': statistics.stdev(times) if len(times) > 1 else 0.0,
                'avg_passes': avg_passes,
                'avg_time_per_pass': avg_time / avg_passes if avg_passes > 0 else 0,
                'total_runs_analyzed': len(self.profile_data),
                'efficiency_score': 1.0 / (avg_time + 1e-6)  # Higher is better
            },
            'profiling_runs': len(self.profile_data)
        }
    
    def get_detailed_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive profiling statistics.
        
        Returns:
            Detailed statistics dictionary
        """
        if not self.profile_data:
            return {'error': 'No profiling data available'}
        
        import statistics
        import numpy as np
        
        # Extract metrics
        times = [entry['total_time'] for entry in self.profile_data]
        passes = [entry['passes'] for entry in self.profile_data]
        query_sizes = [entry.get('query_batch_size', 0) for entry in self.profile_data]
        
        stats = {
            'timing': {
                'mean_time': statistics.mean(times),
                'std_time': statistics.stdev(times) if len(times) > 1 else 0.0,
                'min_time': min(times),
                'max_time': max(times),
                'median_time': statistics.median(times)
            },
            'configuration': {
                'mean_passes': statistics.mean(passes),
                'std_passes': statistics.stdev(passes) if len(passes) > 1 else 0.0,
                'min_passes': min(passes),
                'max_passes': max(passes)
            },
            'batch_sizes': {
                'mean_query_size': statistics.mean(query_sizes) if query_sizes else 0,
                'std_query_size': statistics.stdev(query_sizes) if len(query_sizes) > 1 else 0.0
            },
            'total_runs': len(self.profile_data)
        }
        
        # Memory statistics
        if self.enable_memory_tracking:
            memory_increases = [entry.get('memory_increase_mb', 0) for entry in self.profile_data if 'memory_increase_mb' in entry]
            if memory_increases:
                stats['memory'] = {
                    'mean_increase_mb': statistics.mean(memory_increases),
                    'std_increase_mb': statistics.stdev(memory_increases) if len(memory_increases) > 1 else 0.0,
                    'max_increase_mb': max(memory_increases),
                    'total_samples': len(memory_increases)
                }
        
        # Uncertainty evolution statistics
        if self.uncertainty_evolution:
            all_final_entropies = []
            all_entropy_reductions = []
            
            for evolution in self.uncertainty_evolution:
                if evolution:
                    final_entry = evolution[-1]
                    initial_entry = evolution[0]
                    
                    all_final_entropies.append(final_entry['entropy'])
                    all_entropy_reductions.append(initial_entry['entropy'] - final_entry['entropy'])
            
            if all_final_entropies:
                stats['uncertainty'] = {
                    'mean_final_entropy': statistics.mean(all_final_entropies),
                    'std_final_entropy': statistics.stdev(all_final_entropies) if len(all_final_entropies) > 1 else 0.0,
                    'mean_entropy_reduction': statistics.mean(all_entropy_reductions),
                    'std_entropy_reduction': statistics.stdev(all_entropy_reductions) if len(all_entropy_reductions) > 1 else 0.0
                }
        
        return stats
    
    def reset_profiling_data(self):
        """Reset all profiling data."""
        self.profile_data.clear()
        self.uncertainty_evolution.clear()
    
    def export_profiling_data(self, filepath: str):
        """
        Export profiling data to JSON file.
        
        Args:
            filepath: Path to save profiling data
        """
        import json
        
        export_data = {
            'profile_data': self.profile_data,
            'uncertainty_evolution': self.uncertainty_evolution,
            'export_timestamp': time.time(),
            'total_runs': len(self.profile_data)
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)


# Global profiler instance for convenience
ttcs_profiler = TTCSProfiler()