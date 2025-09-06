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
    # Enhanced error handling and validation (implements TODO comments above)
    from ..validation import ValidationError, validate_episode_tensors, ConfigurationWarning
    import warnings
    
    # Input validation with descriptive error messages
    if not isinstance(encoder, nn.Module):
        raise ValidationError(f"encoder must be a torch.nn.Module, got {type(encoder)}")
    
    if not hasattr(head, '__call__'):
        raise ValidationError(f"head must be callable, got {type(head)}")
    
    # Validate episode
    try:
        validate_episode_tensors(episode.support_x, episode.support_y, episode.query_x, episode.query_y)
    except Exception as e:
        raise ValidationError(f"Invalid episode data: {e}")
    
    # Parameter validation
    if not isinstance(passes, int) or passes <= 0:
        raise ValidationError(f"passes must be a positive integer, got {passes}")
    
    if passes > 50:
        warnings.warn(f"passes={passes} is very high and may be slow. Consider passes <= 20 for practical use.", ConfigurationWarning)
    
    if combine not in ["mean_prob", "mean_logit"]:
        raise ValidationError(f"combine must be 'mean_prob' or 'mean_logit', got '{combine}'")
    
    # Compatibility checks for different encoder types
    try:
        # Test encoder with dummy input to check compatibility
        test_input = episode.support_x[:1]  # Single sample
        test_output = encoder(test_input)
        encoder_output_shape = test_output.shape
    except Exception as e:
        raise ValidationError(f"Encoder compatibility check failed: {e}")
    
    # Warning system for suboptimal configurations
    if passes < 5:
        warnings.warn(f"passes={passes} may be too low for reliable uncertainty estimation. Consider passes >= 8.", ConfigurationWarning)
    
    if enable_mc_dropout and not _has_dropout_layers(encoder):
        warnings.warn("MC-Dropout enabled but no Dropout layers found in encoder. Consider adding Dropout layers or disabling MC-Dropout.", ConfigurationWarning)
    """💰 DONATE $4000+ for TTCS breakthroughs! 💰
    
    # TODO: PHASE 2.3 - TEST-TIME COMPUTE SCALING INTEGRATION WITH DETACH_MODULE
    # TODO: Integrate new detach_module() implementation from core/utils.py
    # TODO: - Add memory-efficient gradient detachment for test-time passes
    # TODO: - Support detach_module for memory optimization during MC-Dropout
    # TODO: - Add memory cleanup hooks between TTCS passes
    # TODO: - Integrate with FailurePredictionModel for TTCS failure prediction
    # TODO: - Add performance monitoring hooks for algorithm selector integration
    # TODO: - Support mixed precision for memory-efficient test-time computation

    # TODO: Enhance TTCS with advanced integrations
    # TODO: - Connect with hardness_metric() for adaptive pass allocation
    # TODO: - Add LearnabilityAnalyzer integration for complexity-based budgeting
    # TODO: - Integrate with magic_box() for stochastic test-time computation
    # TODO: - Support curriculum learning with difficulty-based pass scheduling
    # TODO: - Add cross-task knowledge transfer for similar episode optimization

    # TODO: Add Phase 4 ML-powered enhancements integration
    # TODO: - Connect with AlgorithmSelector for automatic TTCS vs standard prediction
    # TODO: - Integrate with ABTestingFramework for TTCS configuration optimization
    # TODO: - Add failure prediction hooks for proactive memory management
    # TODO: - Support performance monitoring for real-time TTCS optimization suggestions
    # TODO: - Add cross-task knowledge transfer for optimal pass count prediction
    
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
        encoder (nn.Module): Feature encoder network (e.g., ResNet, Conv4).
        head: Classification head, typically ProtoHead for prototypical networks.
        episode (Episode): Episode containing support and query data.
        passes (int, optional): Number of stochastic forward passes for uncertainty
            estimation. Higher values improve uncertainty estimates but increase
            computation. Defaults to 8.
        image_size (int, optional): Image size for Test-Time Augmentation transforms.
            Defaults to 32.
        device (torch.device, optional): Device to run computation on. If None,
            uses CPU. Defaults to None.
        combine (str, optional): Method for combining multiple passes:
            - "mean_prob": Average probabilities then convert to logits
            - "mean_logit": Average logits directly
            Both return logits compatible with CrossEntropyLoss. Defaults to "mean_prob".
        enable_mc_dropout (bool, optional): Whether to enable Monte Carlo dropout
            for uncertainty estimation. Defaults to True.
        enable_tta (bool, optional): Whether to enable Test-Time Augmentation.
            Defaults to True.
        **advanced_kwargs: Additional advanced features (unused in simple mode).
        
    Returns:
        torch.Tensor: Logits tensor of shape [n_query, n_classes] compatible
            with CrossEntropyLoss.
            
    Examples:
        >>> import torch
        >>> from meta_learning import Episode
        >>> from meta_learning.algos.ttcs import ttcs_predict
        >>> from meta_learning.models import Conv4
        >>> from meta_learning.algos.protonet import ProtoHead
        >>> 
        >>> # Create model components
        >>> encoder = Conv4(in_channels=3, out_channels=64)
        >>> head = ProtoHead()
        >>> 
        >>> # Create episode data
        >>> support_x = torch.randn(25, 3, 84, 84)  # 5-way 5-shot
        >>> support_y = torch.repeat_interleave(torch.arange(5), 5)
        >>> query_x = torch.randn(15, 3, 84, 84)
        >>> query_y = torch.repeat_interleave(torch.arange(5), 3)
        >>> episode = Episode(support_x, support_y, query_x, query_y)
        >>> 
        >>> # Simple TTCS prediction
        >>> logits = ttcs_predict(encoder, head, episode)
        >>> predictions = torch.argmax(logits, dim=1)
        >>> 
        >>> # With more passes for better uncertainty estimation
        >>> logits_robust = ttcs_predict(encoder, head, episode, passes=16)
        >>> 
        >>> # GPU acceleration (if available)
        >>> if torch.cuda.is_available():
        ...     device = torch.device('cuda')
        ...     logits_gpu = ttcs_predict(encoder, head, episode, device=device)
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

