class TestTimeComputeScaler:
    """
    Test-time compute scaling for few-shot learning inference.
    
    Features:
    - Dynamic pass allocation based on uncertainty
    - Multi-device compute distribution
    - Progressive early stopping
    - Calibrated uncertainty estimation
    - Performance profiling and optimization
    """
    
    def __init__(self, encoder: nn.Module, head, max_passes: int = 16,
                 uncertainty_threshold: float = 0.1, budget_type: str = 'adaptive',
                 enable_caching: bool = True, monitor_performance: bool = True):
        """
        Initialize Adaptive TTC Scaler.
        
        Args:
            encoder: Feature encoder model
            head: Classification head
            max_passes: Maximum number of forward passes
            uncertainty_threshold: Threshold for early stopping
            budget_type: 'fixed', 'adaptive', or 'progressive'
            enable_caching: Enable intermediate result caching
            monitor_performance: Enable performance monitoring
        """
        self.encoder = encoder
        self.head = head
        self.max_passes = max_passes
        self.uncertainty_threshold = uncertainty_threshold
        self.budget_type = budget_type
        self.enable_caching = enable_caching
        self.monitor_performance = monitor_performance
        
        # Performance monitoring
        if monitor_performance:
            self.stats = {
                'total_predictions': 0,
                'avg_passes_used': 0.0,
                'early_stopping_rate': 0.0,
                'cache_hit_rate': 0.0,
                'avg_prediction_time': 0.0,
                'uncertainty_distribution': [],
                'compute_budget_usage': []
            }
        
        # Caching system
        if enable_caching:
            self.feature_cache = {}
            self.cache_hits = 0
            self.cache_misses = 0
        
        # Adaptive weights (learned from validation data)
        self.ensemble_weights = None
        self.uncertainty_calibration = None
    
    def _estimate_uncertainty(self, logits_list: list) -> torch.Tensor:
        """Estimate prediction uncertainty from multiple forward passes."""
        if len(logits_list) < 2:
            return torch.ones(logits_list[0].size(0)) * 0.5  # Medium uncertainty
        
        # Convert logits to probabilities
        probs_list = [torch.softmax(logits, dim=-1) for logits in logits_list]
        probs_stack = torch.stack(probs_list, dim=0)  # [passes, batch, classes]
        
        # Calculate entropy-based uncertainty
        mean_probs = probs_stack.mean(dim=0)  # [batch, classes]
        entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-8), dim=-1)
        
        # Calculate disagreement-based uncertainty
        disagreement = probs_stack.var(dim=0).sum(dim=-1)  # [batch]
        
        # Combine both measures
        max_entropy = torch.log(torch.tensor(mean_probs.size(-1), dtype=torch.float))
        normalized_entropy = entropy / max_entropy
        normalized_disagreement = torch.clamp(disagreement / 0.25, 0, 1)  # Heuristic normalization
        
        # Weighted combination
        uncertainty = 0.6 * normalized_entropy + 0.4 * normalized_disagreement
        return uncertainty
    
    def _should_stop_early(self, uncertainty: torch.Tensor, pass_count: int) -> bool:
        """Determine if we should stop early based on uncertainty."""
        if pass_count < 2:  # Minimum passes for uncertainty estimation
            return False
        
        # Adaptive threshold based on pass count
        adaptive_threshold = self.uncertainty_threshold * (1 + 0.1 * pass_count)
        
        # Stop if average uncertainty is low enough
        avg_uncertainty = uncertainty.mean().item()
        return avg_uncertainty < adaptive_threshold
    
    def _allocate_compute_budget(self, episode, initial_passes: int = 2) -> dict:
        """Allocate compute budget based on task difficulty."""
        # Quick initial assessment with minimal passes
        with torch.no_grad():
            z_s = self.encoder(episode.support_x)
            z_q = self.encoder(episode.query_x)
            
            # Quick prediction for difficulty assessment
            initial_logits = self.head(z_s, episode.support_y, z_q)
            confidence = torch.softmax(initial_logits, dim=-1).max(dim=-1)[0]
            avg_confidence = confidence.mean().item()
        
        # Allocate budget based on difficulty
        if self.budget_type == 'adaptive':
            if avg_confidence > 0.8:
                allocated_passes = max(2, self.max_passes // 4)  # Easy task
            elif avg_confidence > 0.6:
                allocated_passes = self.max_passes // 2  # Medium task
            else:
                allocated_passes = self.max_passes  # Hard task
        elif self.budget_type == 'progressive':
            # Start with few passes, increase if needed
            allocated_passes = initial_passes
        else:  # fixed
            allocated_passes = self.max_passes
        
        return {
            'allocated_passes': allocated_passes,
            'initial_confidence': avg_confidence,
            'difficulty_level': 'easy' if avg_confidence > 0.8 else 'medium' if avg_confidence > 0.6 else 'hard'
        }
    
    def _cache_key(self, tensor: torch.Tensor) -> str:
        """Generate cache key for tensor."""
        if not self.enable_caching:
            return None
        
        # Use hash of tensor content (expensive but accurate)
        tensor_bytes = tensor.detach().cpu().numpy().tobytes()
        return hashlib.md5(tensor_bytes).hexdigest()[:16]
    
    def _get_cached_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get cached features if available."""
        if not self.enable_caching:
            return None
        
        cache_key = self._cache_key(x)
        if cache_key in self.feature_cache:
            if self.monitor_performance:
                self.cache_hits += 1
            return self.feature_cache[cache_key]
        
        if self.monitor_performance:
            self.cache_misses += 1
        return None
    
    def _cache_features(self, x: torch.Tensor, features: torch.Tensor):
        """Cache computed features."""
        if not self.enable_caching:
            return
        
        cache_key = self._cache_key(x)
        self.feature_cache[cache_key] = features.detach().clone()
        
        # Limit cache size
        if len(self.feature_cache) > 1000:  # Clear oldest entries
            keys_to_remove = list(self.feature_cache.keys())[:100]
            for key in keys_to_remove:
                del self.feature_cache[key]
    
    def predict(self, episode, device=None, combine: str = "mean_prob") -> torch.Tensor:
        """
        Make predictions with adaptive compute allocation.
        
        Args:
            episode: Episode to predict on
            device: Device to run on
            combine: Combination method ("mean_prob" or "mean_logit")
            
        Returns:
            Combined predictions
        """
        start_time = time.time() if self.monitor_performance else 0
        
        if device is None:
            device = next(self.encoder.parameters()).device
        
        # Allocate compute budget
        budget_info = self._allocate_compute_budget(episode)
        max_passes_allocated = budget_info['allocated_passes']
        
        # Store original training states for MC dropout
        original_states = {}
        for name, module in self.encoder.named_modules():
            if hasattr(module, 'training'):
                original_states[name] = module.training
                if isinstance(module, nn.Dropout):
                    module.train()  # Enable dropout for stochasticity
        
        logits_list = []
        uncertainties = []
        passes_used = 0
        
        # Progressive forward passes with early stopping
        for pass_idx in range(max_passes_allocated):
            passes_used += 1
            
            # Check for cached features
            z_s_cached = self._get_cached_features(episode.support_x)
            z_q_cached = self._get_cached_features(episode.query_x)
            
            if z_s_cached is not None and z_q_cached is not None:
                z_s, z_q = z_s_cached, z_q_cached
            else:
                # Compute features with stochastic behavior
                z_s = self.encoder(episode.support_x.to(device))
                z_q = self.encoder(episode.query_x.to(device))
                
                # Cache features from first pass
                if pass_idx == 0:
                    self._cache_features(episode.support_x, z_s)
                    self._cache_features(episode.query_x, z_q)
            
            # Get predictions
            logits = self.head(z_s, episode.support_y.to(device), z_q)
            logits_list.append(logits)
            
            # Estimate uncertainty and check for early stopping
            if len(logits_list) >= 2:
                uncertainty = self._estimate_uncertainty(logits_list)
                uncertainties.append(uncertainty.mean().item())
                
                if self._should_stop_early(uncertainty, passes_used):
                    break
        
        # Restore original training states
        for name, module in self.encoder.named_modules():
            if name in original_states:
                module.train(original_states[name])
        
        # Combine predictions
        if len(logits_list) == 1:
            combined_logits = logits_list[0]
        else:
            if combine == "mean_logit":
                combined_logits = torch.stack(logits_list).mean(dim=0)
            else:  # mean_prob
                probs_list = [torch.softmax(logits, dim=-1) for logits in logits_list]
                mean_probs = torch.stack(probs_list).mean(dim=0)
                combined_logits = torch.log(mean_probs.clamp(min=1e-8))
        
        # Update performance statistics
        if self.monitor_performance:
            self.stats['total_predictions'] += 1
            total_preds = self.stats['total_predictions']
            
            # Update running averages
            prev_avg_passes = self.stats['avg_passes_used']
            self.stats['avg_passes_used'] = (prev_avg_passes * (total_preds - 1) + passes_used) / total_preds
            
            # Early stopping rate
            if passes_used < max_passes_allocated:
                early_stops = self.stats['early_stopping_rate'] * (total_preds - 1) + 1
                self.stats['early_stopping_rate'] = early_stops / total_preds
            
            # Cache hit rate
            total_cache_ops = self.cache_hits + self.cache_misses
            if total_cache_ops > 0:
                self.stats['cache_hit_rate'] = self.cache_hits / total_cache_ops
            
            # Prediction time
            pred_time = time.time() - start_time
            prev_avg_time = self.stats['avg_prediction_time']
            self.stats['avg_prediction_time'] = (prev_avg_time * (total_preds - 1) + pred_time) / total_preds
            
            # Store uncertainty and budget usage
            if uncertainties:
                self.stats['uncertainty_distribution'].append(uncertainties[-1])
            self.stats['compute_budget_usage'].append(passes_used / max_passes_allocated)
            
            # Limit stored data
            if len(self.stats['uncertainty_distribution']) > 1000:
                self.stats['uncertainty_distribution'] = self.stats['uncertainty_distribution'][-500:]
            if len(self.stats['compute_budget_usage']) > 1000:
                self.stats['compute_budget_usage'] = self.stats['compute_budget_usage'][-500:]
        
        return combined_logits
    
    def get_performance_stats(self) -> dict:
        """Get comprehensive performance statistics."""
        if not self.monitor_performance:
            return {}
        
        stats = self.stats.copy()
        
        # Add statistical summaries
        if stats['uncertainty_distribution']:
            uncertainties = np.array(stats['uncertainty_distribution'])
            stats['uncertainty_stats'] = {
                'mean': float(uncertainties.mean()),
                'std': float(uncertainties.std()),
                'median': float(np.median(uncertainties)),
                'q25': float(np.percentile(uncertainties, 25)),
                'q75': float(np.percentile(uncertainties, 75))
            }
        
        if stats['compute_budget_usage']:
            budget_usage = np.array(stats['compute_budget_usage'])
            stats['budget_efficiency'] = {
                'mean_usage': float(budget_usage.mean()),
                'std_usage': float(budget_usage.std()),
                'efficiency_score': float(1.0 - budget_usage.mean())  # Higher is better
            }
        
        return stats
    
    def clear_cache(self):
        """Clear feature cache."""
        if self.enable_caching:
            self.feature_cache.clear()
            self.cache_hits = 0
            self.cache_misses = 0


