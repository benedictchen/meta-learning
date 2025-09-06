from __future__ import annotations
import torch.nn as nn

# BatchNorm Policy Management for Meta-Learning
# Provides research-accurate BatchNorm handling for few-shot learning scenarios

def freeze_batchnorm_running_stats(model: nn.Module) -> None:
    """
    Freeze BatchNorm running statistics for few-shot learning evaluation.
    
    This prevents contamination of running statistics during episodic evaluation,
    following best practices for meta-learning research.
    
    Args:
        model: Neural network model containing BatchNorm layers
    """
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()
            m.track_running_stats = False

def apply_episodic_bn_policy(model: nn.Module, policy: str = "freeze_stats") -> nn.Module:
    """
    Apply BatchNorm policy with learn2learn compatibility.
    
    Args:
        model: Model to apply policy to
        policy: BN policy ('freeze_stats', 'adaptive', 'reset', 'eval_mode')
        
    Returns:
        Modified model
    """
    bn_layers_modified = 0
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            if policy == "freeze_stats":
                module.eval()
                module.track_running_stats = False
                bn_layers_modified += 1
            elif policy == "eval_mode":
                module.eval()
                bn_layers_modified += 1
            elif policy == "reset":
                if hasattr(module, 'reset_running_stats'):
                    module.reset_running_stats()
                module.train()
                bn_layers_modified += 1
            elif policy == "adaptive":
                # Keep in training mode for episodic adaptation
                module.train()
                module.track_running_stats = True
                bn_layers_modified += 1
    
    if bn_layers_modified == 0:
        import warnings
        warnings.warn(f"No BatchNorm layers found in model. Policy '{policy}' had no effect.")
    
    return model


def validate_bn_compatibility(model: nn.Module) -> dict:
    """
    Validate BN compatibility with meta-learning.
    
    Args:
        model: Model to validate
        
    Returns:
        Compatibility analysis results
    """
    analysis = {
        'has_bn_layers': False,
        'bn_layer_count': 0,
        'bn_types': set(),
        'recommendations': [],
        'warnings': [],
        'compatible': True
    }
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            analysis['has_bn_layers'] = True
            analysis['bn_layer_count'] += 1
            analysis['bn_types'].add(type(module).__name__)
            
            # Check for common issues
            if hasattr(module, 'num_features') and module.num_features < 10:
                analysis['warnings'].append(f"Small number of features ({module.num_features}) in {name}")
            
            if not hasattr(module, 'track_running_stats'):
                analysis['warnings'].append(f"BatchNorm layer {name} missing track_running_stats attribute")
    
    # Generate recommendations
    if not analysis['has_bn_layers']:
        analysis['recommendations'].append("Model has no BatchNorm layers - no BN policy needed")
    else:
        analysis['recommendations'].append("Use 'freeze_stats' policy for few-shot evaluation")
        analysis['recommendations'].append("Use 'adaptive' policy for episodic training")
        
        if analysis['bn_layer_count'] > 10:
            analysis['recommendations'].append("Consider reducing BN layers for better few-shot performance")
    
    # Convert set to list for JSON serialization
    analysis['bn_types'] = list(analysis['bn_types'])
    
    return analysis


class EpisodicBatchNormPolicy:
    """
    Professional BN policy management with learn2learn compatibility.
    
    Features:
    - Support for different BN policies
    - Learn2learn module compatibility
    - Performance monitoring
    - Error recovery and fallbacks
    """
    
    SUPPORTED_POLICIES = ['freeze_stats', 'eval_mode', 'reset', 'adaptive', 'auto']
    
    def __init__(self, default_policy: str = 'freeze_stats', enable_monitoring: bool = False):
        """Initialize BN policy manager."""
        if default_policy not in self.SUPPORTED_POLICIES:
            raise ValueError(f"Policy must be one of {self.SUPPORTED_POLICIES}")
        
        self.default_policy = default_policy
        self.enable_monitoring = enable_monitoring
        self.application_logs = []
        self.policy_cache = {}
    
    def apply_policy(self, model: nn.Module, policy: str = None, 
                    phase: str = 'evaluation') -> dict:
        """
        Apply BN policy with comprehensive logging and error handling.
        
        Args:
            model: Model to apply policy to
            policy: BN policy to apply (uses default if None)
            phase: Training phase ('training', 'evaluation', 'adaptation')
            
        Returns:
            Application results and statistics
        """
        import time
        
        policy = policy or self._auto_select_policy(phase)
        if policy not in self.SUPPORTED_POLICIES:
            raise ValueError(f"Unsupported policy: {policy}")
        
        start_time = time.time()
        
        try:
            # Check if we've seen this model before (caching)
            model_id = id(model)
            if model_id in self.policy_cache and self.policy_cache[model_id]['policy'] == policy:
                cached_result = self.policy_cache[model_id]
                return {
                    'policy_applied': policy,
                    'from_cache': True,
                    'bn_layers_modified': cached_result['bn_layers_modified'],
                    'application_time': 0.0
                }
            
            # Apply policy
            result = self._apply_policy_implementation(model, policy)
            application_time = time.time() - start_time
            
            # Cache result
            self.policy_cache[model_id] = {
                'policy': policy,
                'bn_layers_modified': result['bn_layers_modified']
            }
            
            # Log application
            if self.enable_monitoring:
                log_entry = {
                    'timestamp': time.time(),
                    'policy': policy,
                    'phase': phase,
                    'bn_layers_modified': result['bn_layers_modified'],
                    'application_time': application_time,
                    'model_id': model_id
                }
                self.application_logs.append(log_entry)
            
            return {
                'policy_applied': policy,
                'from_cache': False,
                'bn_layers_modified': result['bn_layers_modified'],
                'application_time': application_time,
                'recommendations': result.get('recommendations', [])
            }
            
        except Exception as e:
            # Error recovery - apply safe fallback
            fallback_policy = 'eval_mode'
            fallback_result = self._apply_policy_implementation(model, fallback_policy)
            
            return {
                'policy_applied': fallback_policy,
                'original_policy': policy,
                'error': str(e),
                'fallback_applied': True,
                'bn_layers_modified': fallback_result['bn_layers_modified'],
                'application_time': time.time() - start_time
            }
    
    def _auto_select_policy(self, phase: str) -> str:
        """Automatically select appropriate BN policy based on phase."""
        phase_policy_map = {
            'training': 'adaptive',
            'evaluation': 'freeze_stats',
            'adaptation': 'reset',
            'inference': 'eval_mode'
        }
        return phase_policy_map.get(phase, self.default_policy)
    
    def _apply_policy_implementation(self, model: nn.Module, policy: str) -> dict:
        """Core policy application logic."""
        bn_layers_modified = 0
        recommendations = []
        bn_layer_info = []
        
        for name, module in model.named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                original_training = module.training
                original_track_stats = getattr(module, 'track_running_stats', None)
                
                # Apply policy
                if policy == "freeze_stats":
                    module.eval()
                    module.track_running_stats = False
                elif policy == "eval_mode":
                    module.eval()
                elif policy == "reset":
                    if hasattr(module, 'reset_running_stats'):
                        module.reset_running_stats()
                    module.train()
                elif policy == "adaptive":
                    module.train()
                    module.track_running_stats = True
                
                bn_layers_modified += 1
                bn_layer_info.append({
                    'name': name,
                    'type': type(module).__name__,
                    'num_features': getattr(module, 'num_features', 'unknown'),
                    'original_training': original_training,
                    'original_track_stats': original_track_stats
                })
        
        # Generate recommendations
        if bn_layers_modified == 0:
            recommendations.append("No BatchNorm layers found - consider this in model architecture")
        elif bn_layers_modified > 15:
            recommendations.append("High number of BN layers may impact few-shot performance")
        
        return {
            'bn_layers_modified': bn_layers_modified,
            'recommendations': recommendations,
            'layer_details': bn_layer_info
        }
    
    def get_monitoring_summary(self) -> dict:
        """Get summary of BN policy applications."""
        if not self.enable_monitoring or not self.application_logs:
            return {'monitoring_enabled': self.enable_monitoring, 'total_applications': 0}
        
        import statistics
        
        applications = len(self.application_logs)
        policies_used = [log['policy'] for log in self.application_logs]
        application_times = [log['application_time'] for log in self.application_logs]
        
        return {
            'monitoring_enabled': True,
            'total_applications': applications,
            'unique_policies_used': list(set(policies_used)),
            'policy_frequency': {policy: policies_used.count(policy) for policy in set(policies_used)},
            'avg_application_time': statistics.mean(application_times) if application_times else 0,
            'cache_hits': len([log for log in self.application_logs if log.get('from_cache', False)]),
            'cache_hit_rate': len([log for log in self.application_logs if log.get('from_cache', False)]) / applications
        }
