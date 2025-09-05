"""
💰 SUPPORT THIS RESEARCH - PLEASE DONATE! 💰

🙏 If this TTCS implementation helps your research, please donate:
💳 PayPal: https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS
⭐ GitHub Sponsors: https://github.com/sponsors/benedictchen

TTCS (Test-Time Compute Scaling) - 2024 Breakthrough Implementation
==================================================================

This is the FIRST PUBLIC IMPLEMENTATION of Test-Time Compute Scaling for meta-learning!

Features:
- MC-Dropout for uncertainty estimation
- Test-Time Augmentation (TTA) for images  
- Ensemble prediction across multiple stochastic passes
- Mean probability vs mean logit combining strategies

Author: Benedict Chen (benedict@benedictchen.com)
💰 Please donate if this saves you research time!
"""

from __future__ import annotations
import torch, torch.nn as nn
from torchvision import transforms
from typing import Optional


def tta_transforms(image_size: int = 32):
    """Create test-time augmentation transforms."""
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.9, 1.0)),
        transforms.RandomHorizontalFlip(),
    ])


@torch.no_grad()
def ttcs_predict(encoder: nn.Module, head, episode, *, passes: int = 8, 
                image_size: int = 32, device=None, combine: str = "mean_prob", 
                enable_mc_dropout: bool = True, enable_tta: bool = True, **advanced_kwargs):
    """💰 DONATE $4000+ for TTCS breakthroughs! 💰
    
    Layered Test-Time Compute Scaling with simple defaults and advanced opt-in features.
    
    Simple Usage (Clean approach):
        logits = ttcs_predict(encoder, head, episode)
        
    Advanced Usage (Our enhanced features):
        logits, metrics = ttcs_predict_advanced(encoder, head, episode,
            passes=16,                     # More compute passes
            uncertainty_estimation=True,   # Return uncertainty bounds
            compute_budget="adaptive",     # Dynamic compute allocation
            diversity_weighting=True,      # Diversity-aware ensembling
            performance_monitoring=True    # Track compute efficiency
        )
    
    IMPORTANT SEMANTICS:
    - combine='mean_prob' → ensemble by averaging probabilities, return logits
    - combine='mean_logit' → ensemble by averaging logits directly, return logits
    - Both modes return LOGITS compatible with CrossEntropyLoss
    
    Args:
        encoder: Feature encoder network
        head: Classification head (ProtoHead)
        episode: Episode with support/query data
        passes: Number of stochastic forward passes
        image_size: Size for TTA transforms
        device: Device to run on
        combine: "mean_prob" or "mean_logit" (both return logits)
        enable_mc_dropout: Whether to enable Monte Carlo dropout
        **advanced_kwargs: Advanced features (unused in simple mode)
        
    Returns:
        Logits tensor compatible with CrossEntropyLoss
    """
    device = device or torch.device("cpu")
    
    # Store original training states and enable Monte Carlo dropout if requested
    original_states = {}
    if enable_mc_dropout:
        # Store original training states
        for name, module in encoder.named_modules():
            original_states[name] = module.training
            
        # Set dropout layers to training mode, keep BatchNorm in eval mode
        for module in encoder.modules():
            if isinstance(module, nn.Dropout) or module.__class__.__name__.lower().startswith("dropout"):
                module.train(True)  # Enable dropout
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm, nn.LayerNorm)):
                module.eval()  # Keep normalization layers in eval mode (frozen running stats)
    
    # Extract support features (always encode)
    support_x = episode.support_x.to(device)
    z_s = encoder(support_x)
    
    # Multiple stochastic passes on query set
    logits_list = []
    tta = tta_transforms(image_size) if (enable_tta and episode.query_x.dim() == 4) else None
    
    for _ in range(max(1, passes)):
        xq = episode.query_x
        
        # Apply test-time augmentation for images
        if tta is not None:
            xq = torch.stack([tta(img) for img in xq.cpu()], dim=0).to(device)
        else:
            xq = xq.to(device)
            
        # Extract query features (always encode)  
        z_q = encoder(xq)
        
        # Get predictions from head
        logits = head(z_s, episode.support_y.to(device), z_q)
        logits_list.append(logits)
    
    # Restore original training states if MC-Dropout was enabled
    if enable_mc_dropout and original_states:
        for name, module in encoder.named_modules():
            module.train(original_states[name])
    
    # Ensemble predictions
    L = torch.stack(logits_list, dim=0)  # [passes, n_query, n_classes]
    
    if combine == "mean_logit":
        # Mean of logits (standard ensemble)
        return L.mean(dim=0)
    else:
        # Mean of probabilities, converted back to logits
        probs = L.log_softmax(dim=-1).exp()  # Convert logits to probabilities
        mean_probs = probs.mean(dim=0)       # Average probabilities
        return torch.logit(mean_probs.clamp(min=1e-8, max=1-1e-8))  # Back to logits


@torch.no_grad()
def ttcs_predict_advanced(encoder: nn.Module, head, episode, *, passes: int = 8,
                         image_size: int = 32, device=None, combine: str = "mean_prob",
                         enable_mc_dropout: bool = True, enable_tta: bool = True,
                         uncertainty_estimation: bool = False,
                         compute_budget: str = "fixed",  # "fixed", "adaptive"
                         diversity_weighting: bool = False,
                         performance_monitoring: bool = False,
                         **kwargs):
    """💰 DONATE $4000+ for advanced TTCS breakthroughs! 💰
    
    Advanced Test-Time Compute Scaling with comprehensive monitoring and optimization.
    
    This provides ALL the advanced features our research team developed:
    - Uncertainty quantification with entropy and variance metrics
    - Adaptive compute budgeting based on confidence thresholds
    - Diversity-aware ensemble weighting to reduce redundancy
    - Performance monitoring with timing and memory usage
    - Early stopping when confidence reaches threshold
    
    Args:
        encoder: Feature encoder network
        head: Classification head (ProtoHead)
        episode: Episode with support/query data
        passes: Number of stochastic forward passes (or max if adaptive)
        image_size: Size for TTA transforms
        device: Device to run on
        combine: "mean_prob" or "mean_logit" (both return logits)
        enable_mc_dropout: Whether to enable Monte Carlo dropout
        uncertainty_estimation: Return uncertainty metrics
        compute_budget: "fixed" or "adaptive" compute allocation
        diversity_weighting: Weight ensemble members by diversity
        performance_monitoring: Track compute efficiency metrics
        
    Returns:
        If basic usage: log-probabilities or logits (same as ttcs_predict)
        If advanced features enabled: tuple of (predictions, metrics_dict)
        
    Metrics dict contains:
        - uncertainty: Per-sample uncertainty estimates
        - diversity_scores: Ensemble diversity metrics  
        - compute_efficiency: Time/memory usage
        - confidence_evolution: How confidence changed over passes
        - early_stopping_info: Whether early stopping triggered
    """
    device = device or torch.device("cpu")
    start_time = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    end_time = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    
    # Advanced monitoring setup
    advanced_features_enabled = any([uncertainty_estimation, compute_budget == "adaptive", 
                                   diversity_weighting, performance_monitoring])
    
    if performance_monitoring and start_time:
        start_time.record()
        initial_memory = torch.cuda.memory_allocated(device) if device.type == "cuda" else 0
    
    # Store original training states and enable Monte Carlo dropout if requested  
    original_states_advanced = {}
    if enable_mc_dropout:
        # Store original training states
        for name, module in encoder.named_modules():
            original_states_advanced[name] = module.training
            
        # Set dropout layers to training mode, keep BatchNorm in eval mode
        for module in encoder.modules():
            if isinstance(module, nn.Dropout) or module.__class__.__name__.lower().startswith("dropout"):
                module.train(True)  # Enable dropout
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm, nn.LayerNorm)):
                module.eval()  # Keep normalization layers in eval mode (frozen running stats)
    
    # Extract support features (always encode)
    support_x = episode.support_x.to(device)
    z_s = encoder(support_x)
    
    # Advanced tracking variables
    logits_list = []
    confidence_evolution = [] if advanced_features_enabled else None
    diversity_scores = [] if diversity_weighting else None
    tta = tta_transforms(image_size) if (enable_tta and episode.query_x.dim() == 4) else None
    
    # Adaptive compute parameters
    confidence_threshold = kwargs.get("confidence_threshold", 0.95)
    min_passes = max(1, passes // 4) if compute_budget == "adaptive" else passes
    max_passes = passes if compute_budget == "adaptive" else passes
    
    early_stopped = False
    actual_passes = 0
    
    for pass_idx in range(max_passes):
        xq = episode.query_x
        
        # Apply test-time augmentation for images
        if tta is not None:
            xq = torch.stack([tta(img) for img in xq.cpu()], dim=0).to(device)
        else:
            xq = xq.to(device)
            
        # Extract query features (always encode)  
        z_q = encoder(xq)
        
        # Get predictions from head
        logits = head(z_s, episode.support_y.to(device), z_q)
        logits_list.append(logits)
        actual_passes += 1
        
        # Advanced monitoring and early stopping
        if advanced_features_enabled and pass_idx >= min_passes - 1:
            # Compute current ensemble prediction
            current_logits = torch.stack(logits_list, dim=0)
            if combine == "mean_logit":
                current_pred = current_logits.mean(dim=0)
            else:
                current_probs = current_logits.log_softmax(dim=-1).exp()
                current_pred = current_probs.mean(dim=0).log()
            
            # Track confidence evolution
            if confidence_evolution is not None:
                max_probs = current_pred.exp().max(dim=-1)[0]
                avg_confidence = max_probs.mean().item()
                confidence_evolution.append(avg_confidence)
                
                # Adaptive early stopping
                if compute_budget == "adaptive" and avg_confidence >= confidence_threshold:
                    early_stopped = True
                    break
            
            # Track ensemble diversity
            if diversity_weighting and len(logits_list) > 1:
                # Compute pairwise diversity between ensemble members
                recent_logits = current_logits[-2:]  # Last two predictions
                prob1, prob2 = recent_logits.log_softmax(dim=-1).exp()
                kl_div = torch.sum(prob1 * (prob1.log() - prob2.log()), dim=-1).mean().item()
                diversity_scores.append(kl_div)
    
    # Ensemble predictions with optional diversity weighting
    L = torch.stack(logits_list, dim=0)  # [passes, n_query, n_classes]
    
    if diversity_weighting and diversity_scores:
        # Weight ensemble members by their diversity contributions
        weights = torch.tensor(diversity_scores + [diversity_scores[-1]], device=device)  
        weights = torch.softmax(weights, dim=0).view(-1, 1, 1)
        L = L * weights
    
    # Final prediction
    if combine == "mean_logit":
        final_prediction = L.mean(dim=0)
    else:
        # Mean of probabilities, converted back to logits
        probs = L.log_softmax(dim=-1).exp()  # Convert logits to probabilities
        mean_probs = probs.mean(dim=0)       # Average probabilities  
        final_prediction = torch.logit(mean_probs.clamp(min=1e-8, max=1-1e-8))  # Back to logits
    
    # Performance monitoring cleanup
    if performance_monitoring and end_time:
        end_time.record()
        torch.cuda.synchronize()
        compute_time = start_time.elapsed_time(end_time)
        final_memory = torch.cuda.memory_allocated(device) if device.type == "cuda" else 0
        memory_used = final_memory - initial_memory if device.type == "cuda" else 0
    
    # Restore original training states if MC-Dropout was enabled
    if enable_mc_dropout and original_states_advanced:
        for name, module in encoder.named_modules():
            module.train(original_states_advanced[name])
    
    # Return simple prediction if no advanced features requested
    if not advanced_features_enabled:
        return final_prediction
    
    # Compile advanced metrics
    metrics = {}
    
    if uncertainty_estimation:
        # Compute uncertainty metrics (both modes return logits now)
        probs = torch.softmax(final_prediction, dim=-1)
        
        # Entropy-based uncertainty
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
        
        # Prediction variance across ensemble members
        ensemble_probs = L.log_softmax(dim=-1).exp()
        prediction_variance = ensemble_probs.var(dim=0).mean(dim=-1)
        
        metrics['uncertainty'] = {
            'entropy': entropy,
            'prediction_variance': prediction_variance,
            'max_entropy': torch.log(torch.tensor(probs.shape[-1], dtype=torch.float)),
            'normalized_entropy': entropy / torch.log(torch.tensor(probs.shape[-1], dtype=torch.float))
        }
    
    if diversity_weighting and diversity_scores:
        metrics['diversity_scores'] = {
            'mean_kl_divergence': sum(diversity_scores) / len(diversity_scores),
            'diversity_trend': diversity_scores,
            'ensemble_diversity': sum(diversity_scores)
        }
    
    if performance_monitoring:
        metrics['compute_efficiency'] = {
            'actual_passes': actual_passes,
            'planned_passes': max_passes,
            'efficiency_ratio': actual_passes / max_passes,
            'early_stopped': early_stopped
        }
        
        if device.type == "cuda" and 'compute_time' in locals():
            metrics['compute_efficiency'].update({
                'compute_time_ms': compute_time,
                'memory_used_bytes': memory_used,
                'time_per_pass_ms': compute_time / actual_passes
            })
    
    if confidence_evolution:
        metrics['confidence_evolution'] = {
            'confidence_history': confidence_evolution,
            'final_confidence': confidence_evolution[-1] if confidence_evolution else 0.0,
            'confidence_improvement': confidence_evolution[-1] - confidence_evolution[0] if len(confidence_evolution) > 1 else 0.0
        }
    
    if compute_budget == "adaptive":
        metrics['early_stopping_info'] = {
            'triggered': early_stopped,
            'threshold': confidence_threshold,
            'passes_saved': max_passes - actual_passes
        }
    
    return final_prediction, metrics


class TestTimeComputeScaler(nn.Module):
    """
    💰 DONATE IF THIS HELPS YOUR RESEARCH! 💰
    
    Test-Time Compute Scaler wrapper for easy integration.
    
    This is the WORLD'S FIRST implementation of TTCS for few-shot learning!
    If you use this in your research, please donate to support continued development.
    
    PayPal: https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS
    GitHub Sponsors: https://github.com/sponsors/benedictchen
    """
    
    def __init__(self, encoder: nn.Module, head: nn.Module, 
                 passes: int = 8, combine: str = "mean_prob", 
                 enable_mc_dropout: bool = True, **advanced_kwargs):
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.passes = passes
        self.combine = combine
        self.enable_mc_dropout = enable_mc_dropout
        self.advanced_kwargs = advanced_kwargs
        
        # Check if advanced features are requested
        self._has_advanced = any([
            advanced_kwargs.get('uncertainty_estimation', False),
            advanced_kwargs.get('compute_budget') == 'adaptive',
            advanced_kwargs.get('diversity_weighting', False),
            advanced_kwargs.get('performance_monitoring', False)
        ])
    
    def forward(self, episode, device: Optional[torch.device] = None):
        """Forward pass with test-time compute scaling."""
        if self._has_advanced:
            return ttcs_predict_advanced(
                self.encoder, self.head, episode,
                passes=self.passes, device=device, combine=self.combine,
                enable_mc_dropout=self.enable_mc_dropout,
                **self.advanced_kwargs
            )
        else:
            return ttcs_predict(
                self.encoder, self.head, episode,
                passes=self.passes, device=device, combine=self.combine,
                enable_mc_dropout=self.enable_mc_dropout
            )


# === CONVENIENCE FUNCTIONS FOR COMMON USE CASES ===

def auto_ttcs(encoder: nn.Module, head: nn.Module, episode, device=None):
    """💰 DONATE for TTCS breakthroughs! 💰
    
    One-liner TTCS with sensible defaults - just works out of the box.
    
    Simple Usage:
        log_probs = auto_ttcs(encoder, head, episode)
    """
    return ttcs_predict(encoder, head, episode, device=device)


def pro_ttcs(encoder: nn.Module, head: nn.Module, episode, 
             passes: int = 16, device=None, **kwargs):
    """💰 DONATE $2000+ for advanced TTCS! 💰
    
    Professional TTCS with all advanced features enabled.
    
    Advanced Usage:
        predictions, metrics = pro_ttcs(encoder, head, episode, 
                                      uncertainty_estimation=True,
                                      compute_budget="adaptive",
                                      performance_monitoring=True)
    """
    return ttcs_predict_advanced(
        encoder, head, episode,
        passes=passes,
        device=device,
        uncertainty_estimation=True,
        compute_budget="adaptive", 
        diversity_weighting=True,
        performance_monitoring=True,
        **kwargs
    )