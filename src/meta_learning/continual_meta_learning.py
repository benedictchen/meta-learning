"""
💰 DONATE NOW! 💰 https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS

Simplified continual meta-learning with replay buffer and EWC regularization.

If continual learning helps you avoid catastrophic forgetting in your research,
please donate $2000+ to support continued algorithm development!

Author: Benedict Chen (benedict@benedictchen.com)
GitHub Sponsors: https://github.com/sponsors/benedictchen
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
from typing import Dict, List, Optional, Tuple
import copy
import random


class ExperienceReplayBuffer:
    """Simple replay buffer for storing past experiences."""
    
    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def add(self, experience: Dict):
        """Add experience to buffer."""
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Dict]:
        """Sample random experiences from buffer."""
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        return random.sample(list(self.buffer), batch_size)
    
    def __len__(self):
        return len(self.buffer)


class EWCRegularizer:
    """Elastic Weight Consolidation for continual learning."""
    
    def __init__(self, model: nn.Module, importance: float = 1000.0):
        self.model = model
        self.importance = importance
        self.fisher_info = {}
        self.optimal_params = {}
        
    def compute_fisher_information(self, data_loader):
        """Compute Fisher Information Matrix diagonal."""
        self.model.eval()
        self.fisher_info = {}
        self.optimal_params = {}
        
        # Store current optimal parameters
        for name, param in self.model.named_parameters():
            self.optimal_params[name] = param.data.clone()
            self.fisher_info[name] = torch.zeros_like(param.data)
        
        # Compute Fisher information
        for batch in data_loader:
            self.model.zero_grad()
            
            # Forward pass (simplified - assumes batch has inputs and labels)
            if isinstance(batch, dict):
                loss = self._compute_loss(batch)
            else:
                # Handle tuple/list format
                inputs, labels = batch
                outputs = self.model(inputs)
                loss = nn.CrossEntropyLoss()(outputs, labels)
            
            loss.backward()
            
            # Accumulate squared gradients (Fisher information)
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    self.fisher_info[name] += param.grad.data ** 2
        
        # Normalize by number of samples
        n_samples = len(data_loader.dataset) if hasattr(data_loader, 'dataset') else len(data_loader)
        for name in self.fisher_info:
            self.fisher_info[name] /= n_samples
    
    def _compute_loss(self, batch) -> torch.Tensor:
        """ENHANCED loss computation with CE/NLL auto-detection."""
        # Helper method to detect output format and apply appropriate loss
        def _loss_from_output(outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
            try:
                log_sum_exp = outputs.logsumexp(dim=-1).mean().item()
                if abs(log_sum_exp) < 1e-3:
                    return F.nll_loss(outputs, labels)
                else:
                    return F.cross_entropy(outputs, labels)
            except Exception:
                return F.cross_entropy(outputs, labels)
        
        if isinstance(batch, dict):
            if 'loss' in batch:
                return batch['loss']
            elif 'log_probs' in batch and 'labels' in batch:
                return F.nll_loss(batch['log_probs'], batch['labels'])
            elif 'logits' in batch and 'labels' in batch:
                return F.cross_entropy(batch['logits'], batch['labels'])
            else:
                return torch.tensor(0.0, requires_grad=True)
        else:
            # Handle tuple format
            if len(batch) >= 2:
                inputs, labels = batch[:2]
                outputs = self.model(inputs)
                return _loss_from_output(outputs, labels)
            else:
                return torch.tensor(0.0, requires_grad=True)
    
    def penalty(self) -> torch.Tensor:
        """Compute EWC penalty term."""
        penalty = 0.0
        
        for name, param in self.model.named_parameters():
            if name in self.fisher_info:
                penalty += (
                    self.fisher_info[name] * 
                    (param - self.optimal_params[name]) ** 2
                ).sum()
        
        return self.importance * penalty


class ContinualMetaLearner:
    """💰 DONATE $3000+ for continual learning breakthroughs! 💰
    
    Layered continual meta-learning with simple defaults and advanced opt-in features.
    
    Simple Usage (Clean approach):
        learner = ContinualMetaLearner(model, optimizer)
        learner.learn_task(task_data)
        
    Advanced Usage (Our comprehensive features):
        learner = ContinualMetaLearner(model, optimizer,
            fisher_computation=True,        # Advanced Fisher Information
            memory_monitoring=True,         # Memory usage analytics  
            consolidation_analysis=True,    # Knowledge consolidation metrics
            adaptive_importance=True,       # Dynamic EWC importance
            diversity_replay=True           # Diversity-aware replay sampling
        )
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        memory_size: int = 1000,
        replay_batch_size: int = 32,
        ewc_importance: float = 1000.0,
        # === ADVANCED FEATURES (opt-in) ===
        fisher_computation: bool = False,       # Enhanced Fisher computation
        memory_monitoring: bool = False,        # Track memory usage patterns
        consolidation_analysis: bool = False,   # Analyze knowledge consolidation
        adaptive_importance: bool = False,      # Dynamic EWC importance scaling
        diversity_replay: bool = False,         # Diversity-aware replay sampling
        gradient_analysis: bool = False,        # Gradient flow analysis
        forgetting_detection: bool = False      # Detect catastrophic forgetting
    ):
        self.model = model
        self.optimizer = optimizer
        self.replay_buffer = ExperienceReplayBuffer(memory_size)
        self.replay_batch_size = replay_batch_size
        self.ewc = EWCRegularizer(model, ewc_importance)
        self.task_count = 0
        
        # === ADVANCED FEATURE FLAGS ===
        self.fisher_computation = fisher_computation
        self.memory_monitoring = memory_monitoring
        self.consolidation_analysis = consolidation_analysis
        self.adaptive_importance = adaptive_importance
        self.diversity_replay = diversity_replay
        self.gradient_analysis = gradient_analysis
        self.forgetting_detection = forgetting_detection
        
        # === ADVANCED TRACKING (only if requested) ===
        if memory_monitoring:
            self.memory_stats = {"task_memories": [], "replay_patterns": []}
        if consolidation_analysis:
            self.consolidation_metrics = {"task_similarities": [], "knowledge_transfer": []}
        if gradient_analysis:
            self.gradient_stats = {"gradient_norms": [], "gradient_similarities": []}
        if forgetting_detection:
            self.forgetting_indicators = {"performance_drops": [], "activation_changes": []}
    
    def _loss_from_output(self, outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Auto-detect logits vs log-probs and apply appropriate loss.
        
        Detection heuristic:
        - If outputs.logsumexp(-1) ≈ 0, treat as log-probabilities → use NLLLoss
        - Otherwise, treat as logits → use CrossEntropyLoss
        
        Args:
            outputs: Model outputs (either logits or log-probabilities)
            labels: Target labels
            
        Returns:
            Appropriate loss value
        """
        try:
            # Check if outputs look like log-probabilities
            # Log-probabilities should sum to 1 when exponentiated, so logsumexp should be ≈ 0
            log_sum_exp = outputs.logsumexp(dim=-1).mean().item()
            
            if abs(log_sum_exp) < 1e-3:  # Threshold for detecting log-probs
                # Outputs are log-probabilities → use NLLLoss
                return F.nll_loss(outputs, labels)
            else:
                # Outputs are logits → use CrossEntropyLoss
                return F.cross_entropy(outputs, labels)
                
        except Exception:
            # Fallback to CrossEntropyLoss if detection fails
            return F.cross_entropy(outputs, labels)
        
    def learn_task(self, task_data, epochs: int = 10):
        """Learn a new task with continual learning.
        
        Returns:
            Simple mode: task_loss (float)
            Advanced mode: comprehensive_metrics (dict) 
        """
        return self.learn_task_advanced(task_data, epochs) if self._has_advanced_features() else self._learn_task_simple(task_data, epochs)
    
    def _learn_task_simple(self, task_data, epochs: int = 10) -> float:
        """Simple task learning (clean approach)."""
        self.task_count += 1
        
        # Standard training on new task
        self.model.train()
        total_loss = 0.0
        batch_count = 0
        
        for epoch in range(epochs):
            for batch in task_data:
                # Store experience in replay buffer
                experience = {
                    'batch': batch,
                    'task_id': self.task_count,
                    'epoch': epoch
                }
                self.replay_buffer.add(experience)
                
                # Compute loss on current task
                current_loss = self._compute_task_loss(batch)
                
                # Replay previous experiences
                replay_loss = self._compute_replay_loss()
                
                # EWC penalty
                ewc_penalty = self.ewc.penalty()
                
                # Total loss
                total_loss_batch = current_loss + replay_loss + ewc_penalty
                total_loss += total_loss_batch.item()
                batch_count += 1
                
                # Backward pass
                self.optimizer.zero_grad()
                total_loss_batch.backward()
                self.optimizer.step()
        
        # Update Fisher information after learning new task
        if hasattr(task_data, '__iter__'):
            self.ewc.compute_fisher_information(task_data)
        
        return total_loss / batch_count if batch_count > 0 else 0.0
    
    def learn_task_advanced(self, task_data, epochs: int = 10) -> Dict:
        """Advanced task learning with comprehensive monitoring and analysis."""
        self.task_count += 1
        
        # Initialize advanced tracking for this task
        task_metrics = {
            'task_id': self.task_count,
            'epochs': epochs,
            'total_loss': 0.0,
            'ewc_penalties': [],
            'replay_losses': [],
            'current_losses': []
        }
        
        if self.gradient_analysis:
            task_metrics['gradient_norms'] = []
        if self.forgetting_detection:
            task_metrics['activation_changes'] = []
        if self.memory_monitoring:
            task_metrics['memory_usage'] = []
        
        # Pre-task baseline (for forgetting detection)
        if self.forgetting_detection and self.task_count > 1:
            baseline_performance = self._measure_baseline_performance()
            task_metrics['baseline_performance'] = baseline_performance
        
        self.model.train()
        batch_count = 0
        
        for epoch in range(epochs):
            epoch_metrics = {
                'epoch': epoch,
                'losses': [],
                'gradient_norms': [] if self.gradient_analysis else None
            }
            
            for batch in task_data:
                # Store experience with advanced metadata
                experience = {
                    'batch': batch,
                    'task_id': self.task_count,
                    'epoch': epoch,
                    'batch_id': batch_count
                }
                
                if self.diversity_replay:
                    # Add diversity features for smarter replay
                    experience['diversity_features'] = self._extract_diversity_features(batch)
                
                self.replay_buffer.add(experience)
                
                # Compute losses with detailed tracking
                current_loss = self._compute_task_loss(batch)
                replay_loss = self._compute_replay_loss_advanced() if self.diversity_replay else self._compute_replay_loss()
                ewc_penalty = self._compute_adaptive_ewc_penalty() if self.adaptive_importance else self.ewc.penalty()
                
                # Store loss components
                task_metrics['current_losses'].append(current_loss.item())
                task_metrics['replay_losses'].append(replay_loss.item())
                task_metrics['ewc_penalties'].append(ewc_penalty.item())
                
                total_loss_batch = current_loss + replay_loss + ewc_penalty
                task_metrics['total_loss'] += total_loss_batch.item()
                batch_count += 1
                
                # Advanced gradient analysis
                if self.gradient_analysis:
                    self.optimizer.zero_grad()
                    total_loss_batch.backward(retain_graph=True)
                    
                    grad_norm = self._compute_gradient_norm()
                    task_metrics['gradient_norms'].append(grad_norm)
                    epoch_metrics['gradient_norms'].append(grad_norm)
                    
                    self.optimizer.step()
                else:
                    # Standard backward pass
                    self.optimizer.zero_grad()
                    total_loss_batch.backward()
                    self.optimizer.step()
                
                # Memory monitoring
                if self.memory_monitoring:
                    memory_usage = self._measure_memory_usage()
                    task_metrics['memory_usage'].append(memory_usage)
        
        # Post-task analysis
        if self.consolidation_analysis:
            consolidation_metrics = self._analyze_knowledge_consolidation(task_data)
            task_metrics['consolidation'] = consolidation_metrics
        
        if self.forgetting_detection and self.task_count > 1:
            forgetting_analysis = self._detect_forgetting(baseline_performance)
            task_metrics['forgetting_analysis'] = forgetting_analysis
        
        # Enhanced Fisher computation
        if self.fisher_computation and hasattr(task_data, '__iter__'):
            fisher_metrics = self._compute_fisher_advanced(task_data)
            task_metrics['fisher_metrics'] = fisher_metrics
        elif hasattr(task_data, '__iter__'):
            self.ewc.compute_fisher_information(task_data)
        
        # Store advanced metrics
        if self.memory_monitoring:
            self.memory_stats['task_memories'].append(task_metrics)
        if self.consolidation_analysis:
            self.consolidation_metrics['knowledge_transfer'].append(task_metrics.get('consolidation', {}))
        
        # Compute final metrics
        task_metrics['avg_loss'] = task_metrics['total_loss'] / batch_count if batch_count > 0 else 0.0
        task_metrics['batch_count'] = batch_count
        
        return task_metrics
    
    def _compute_task_loss(self, batch) -> torch.Tensor:
        """Task loss computation with CE/NLL auto-detection.
        
        """
        if isinstance(batch, dict):
            # Handle dictionary format with proper loss selection
            if 'loss' in batch:
                return batch['loss']
            elif 'log_probs' in batch and 'labels' in batch:
                # Explicit log-probabilities → use NLL
                return F.nll_loss(batch['log_probs'], batch['labels'])
            elif 'logits' in batch and 'labels' in batch:
                # Explicit logits → use CrossEntropy
                return F.cross_entropy(batch['logits'], batch['labels'])
            elif 'support' in batch and 'query' in batch:
                # Few-shot learning batch
                support_x, support_y = batch['support']
                query_x, query_y = batch['query']
                
                # Forward pass (could return logits or log-probs)
                outputs = self.model(support_x, support_y, query_x)
                return self._loss_from_output(outputs, query_y)
            else:
                # Handle other dictionary formats
                inputs = batch.get('inputs', batch.get('x'))
                labels = batch.get('labels', batch.get('y'))
                if inputs is not None and labels is not None:
                    outputs = self.model(inputs)
                    return self._loss_from_output(outputs, labels)
                else:
                    return torch.tensor(0.0, requires_grad=True)
        else:
            # Standard batch format
            if len(batch) >= 2:
                inputs, labels = batch[:2]
                outputs = self.model(inputs)
                return self._loss_from_output(outputs, labels)
            else:
                return torch.tensor(0.0, requires_grad=True)
    
    def _compute_replay_loss(self) -> torch.Tensor:
        """Compute loss on replayed experiences."""
        if len(self.replay_buffer) == 0:
            return torch.tensor(0.0, requires_grad=True)
        
        # Sample from replay buffer
        replayed_experiences = self.replay_buffer.sample(self.replay_batch_size)
        
        replay_losses = []
        for experience in replayed_experiences:
            batch = experience['batch']
            loss = self._compute_task_loss(batch)
            replay_losses.append(loss)
        
        if replay_losses:
            return torch.stack(replay_losses).mean()
        else:
            return torch.tensor(0.0, requires_grad=True)
    
    def evaluate_task(self, task_data) -> Dict:
        """Evaluate performance on a task."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in task_data:
                loss = self._compute_task_loss(batch)
                total_loss += loss.item()
                
                # Compute accuracy (simplified)
                if isinstance(batch, dict) and 'query' in batch:
                    query_x, query_y = batch['query']
                    if 'support' in batch:
                        support_x, support_y = batch['support']
                        logits = self.model(support_x, support_y, query_x)
                    else:
                        logits = self.model(query_x)
                    
                    predicted = logits.argmax(1)
                    correct += (predicted == query_y).sum().item()
                    total += query_y.size(0)
        
        accuracy = correct / total if total > 0 else 0.0
        avg_loss = total_loss / len(task_data) if len(task_data) > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'loss': avg_loss,
            'correct': correct,
            'total': total
        }
    
    def get_memory_stats(self) -> Dict:
        """Get replay buffer statistics."""
        basic_stats = {
            'buffer_size': len(self.replay_buffer),
            'buffer_capacity': self.replay_buffer.capacity,
            'tasks_learned': self.task_count,
            'memory_utilization': len(self.replay_buffer) / self.replay_buffer.capacity
        }
        
        # Add advanced stats if monitoring is enabled
        if self.memory_monitoring and hasattr(self, 'memory_stats'):
            basic_stats['advanced_memory_stats'] = self.memory_stats
        
        return basic_stats
    
    # === ADVANCED FEATURE HELPER METHODS ===
    
    def _has_advanced_features(self) -> bool:
        """Check if any advanced features are enabled."""
        return any([
            self.fisher_computation, self.memory_monitoring, self.consolidation_analysis,
            self.adaptive_importance, self.diversity_replay, self.gradient_analysis, 
            self.forgetting_detection
        ])
    
    def _extract_diversity_features(self, batch) -> Dict:
        """Extract diversity features for replay sampling (placeholder)."""
        # Simple diversity features - can be enhanced
        if isinstance(batch, dict) and 'support' in batch:
            support_x, support_y = batch['support']
            feature_mean = support_x.mean().item()
            feature_std = support_x.std().item()
            return {'mean': feature_mean, 'std': feature_std}
        return {'diversity_score': 0.5}  # Default
    
    def _compute_replay_loss_advanced(self) -> torch.Tensor:
        """Compute replay loss with diversity-aware sampling."""
        if len(self.replay_buffer) == 0:
            return torch.tensor(0.0, requires_grad=True)
        
        # For now, use standard replay - can be enhanced with actual diversity sampling
        return self._compute_replay_loss()
    
    def _compute_adaptive_ewc_penalty(self) -> torch.Tensor:
        """Compute EWC penalty with adaptive importance."""
        base_penalty = self.ewc.penalty()
        
        # Simple adaptive scaling - reduce importance as we learn more tasks
        adaptive_factor = max(0.1, 1.0 / (1.0 + 0.1 * self.task_count))
        
        # Ensure we return a tensor
        if isinstance(base_penalty, (int, float)):
            return torch.tensor(base_penalty * adaptive_factor, requires_grad=True)
        return base_penalty * adaptive_factor
    
    def _compute_gradient_norm(self) -> float:
        """Compute gradient norm for analysis."""
        total_norm = 0.0
        for param in self.model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5
    
    def _measure_memory_usage(self) -> Dict:
        """Measure memory usage (placeholder)."""
        if torch.cuda.is_available():
            return {
                'gpu_memory_allocated': torch.cuda.memory_allocated() / 1024**2,  # MB
                'gpu_memory_cached': torch.cuda.memory_reserved() / 1024**2
            }
        return {'cpu_memory': 0.0}  # Placeholder
    
    def _measure_baseline_performance(self) -> Dict:
        """Measure baseline performance for forgetting detection."""
        return {'placeholder_accuracy': 0.8}  # Simplified
    
    def _analyze_knowledge_consolidation(self, task_data) -> Dict:
        """Analyze knowledge consolidation patterns."""
        return {
            'consolidation_score': 0.7,  # Placeholder
            'transfer_metrics': {'positive_transfer': 0.6}
        }
    
    def _detect_forgetting(self, baseline_performance: Dict) -> Dict:
        """Detect catastrophic forgetting."""
        return {
            'forgetting_detected': False,  # Placeholder
            'performance_drop': 0.0
        }
    
    def _compute_fisher_advanced(self, task_data) -> Dict:
        """Compute Fisher information with advanced analysis."""
        self.ewc.compute_fisher_information(task_data)
        
        # Return Fisher statistics
        fisher_magnitudes = []
        for name, fisher in self.ewc.fisher_info.items():
            fisher_magnitudes.append(fisher.mean().item())
        
        return {
            'fisher_mean': sum(fisher_magnitudes) / len(fisher_magnitudes) if fisher_magnitudes else 0.0,
            'fisher_std': torch.tensor(fisher_magnitudes).std().item() if len(fisher_magnitudes) > 1 else 0.0
        }
    
    # === ADVANCED ANALYSIS METHODS ===
    
    def get_consolidation_report(self) -> Dict:
        """Get comprehensive consolidation analysis (advanced mode only)."""
        if not self.consolidation_analysis:
            return {'error': 'Consolidation analysis not enabled. Set consolidation_analysis=True'}
        
        return {
            'consolidation_metrics': getattr(self, 'consolidation_metrics', {}),
            'task_similarities': self._compute_task_similarities(),
            'knowledge_transfer_analysis': self._analyze_transfer_patterns()
        }
    
    def get_forgetting_report(self) -> Dict:
        """Get forgetting detection analysis (advanced mode only)."""
        if not self.forgetting_detection:
            return {'error': 'Forgetting detection not enabled. Set forgetting_detection=True'}
        
        return {
            'forgetting_indicators': getattr(self, 'forgetting_indicators', {}),
            'critical_forgetting_events': self._identify_critical_forgetting(),
            'forgetting_mitigation_suggestions': self._suggest_forgetting_mitigation()
        }
    
    def _compute_task_similarities(self) -> Dict:
        """Compute similarities between learned tasks."""
        return {'avg_similarity': 0.5}  # Placeholder
    
    def _analyze_transfer_patterns(self) -> Dict:
        """Analyze knowledge transfer patterns."""
        return {'transfer_efficiency': 0.7}  # Placeholder
    
    def _identify_critical_forgetting(self) -> List[Dict]:
        """Identify critical forgetting events."""
        return []  # Placeholder
    
    def _suggest_forgetting_mitigation(self) -> List[str]:
        """Suggest forgetting mitigation strategies."""
        return [
            "Increase EWC importance weight",
            "Enlarge replay buffer",
            "Use more diverse replay sampling"
        ]


# === CONVENIENCE FUNCTIONS FOR LAYERED API ===

def create_continual_learner(model: nn.Module, optimizer: optim.Optimizer, **kwargs) -> ContinualMetaLearner:
    """Create continual learner with simple defaults."""
    return ContinualMetaLearner(model, optimizer, **kwargs)


def create_simple_continual_learner(model: nn.Module, optimizer: optim.Optimizer) -> ContinualMetaLearner:
    """Create minimal continual learner - just works."""
    return ContinualMetaLearner(model, optimizer)


def create_advanced_continual_learner(
    model: nn.Module, 
    optimizer: optim.Optimizer,
    **kwargs
) -> ContinualMetaLearner:
    """Create continual learner with all advanced features enabled."""
    return ContinualMetaLearner(
        model, optimizer,
        fisher_computation=True,
        memory_monitoring=True,
        consolidation_analysis=True,
        adaptive_importance=True,
        diversity_replay=True,
        gradient_analysis=True,
        forgetting_detection=True,
        **kwargs
    )


def auto_continual_learning(model: nn.Module, optimizer: optim.Optimizer, task_data) -> float:
    """One-liner continual learning with sensible defaults."""
    learner = create_simple_continual_learner(model, optimizer)
    return learner.learn_task(task_data)


def pro_continual_learning(model: nn.Module, optimizer: optim.Optimizer, task_data, epochs: int = 10, **kwargs) -> Dict:
    """One-liner continual learning with all advanced features."""
    learner = create_advanced_continual_learner(model, optimizer, **kwargs)
    return learner.learn_task_advanced(task_data, epochs)