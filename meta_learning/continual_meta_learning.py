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
    
    def _compute_loss(self, batch: Dict) -> torch.Tensor:
        """Compute loss from batch dictionary."""
        # Simplified loss computation - customize based on your task
        if 'loss' in batch:
            return batch['loss']
        elif 'logits' in batch and 'labels' in batch:
            return nn.CrossEntropyLoss()(batch['logits'], batch['labels'])
        else:
            # Default: assume model can handle the batch directly
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
    """Simplified continual meta-learning with replay buffer and EWC."""
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        memory_size: int = 1000,
        replay_batch_size: int = 32,
        ewc_importance: float = 1000.0
    ):
        self.model = model
        self.optimizer = optimizer
        self.replay_buffer = ExperienceReplayBuffer(memory_size)
        self.replay_batch_size = replay_batch_size
        self.ewc = EWCRegularizer(model, ewc_importance)
        self.task_count = 0
        
    def learn_task(self, task_data, epochs: int = 10):
        """Learn a new task with continual learning."""
        self.task_count += 1
        
        # Standard training on new task
        self.model.train()
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
                total_loss = current_loss + replay_loss + ewc_penalty
                
                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
        
        # Update Fisher information after learning new task
        if hasattr(task_data, '__iter__'):
            self.ewc.compute_fisher_information(task_data)
    
    def _compute_task_loss(self, batch) -> torch.Tensor:
        """Compute loss for current task batch."""
        # Simplified - customize based on your meta-learning setup
        if isinstance(batch, dict):
            if 'support' in batch and 'query' in batch:
                # Few-shot learning batch
                support_x, support_y = batch['support']
                query_x, query_y = batch['query']
                
                # Forward pass
                logits = self.model(support_x, support_y, query_x)
                loss = nn.CrossEntropyLoss()(logits, query_y)
                return loss
            else:
                return torch.tensor(0.0, requires_grad=True)
        else:
            # Standard batch format
            inputs, labels = batch
            outputs = self.model(inputs)
            return nn.CrossEntropyLoss()(outputs, labels)
    
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
        return {
            'buffer_size': len(self.replay_buffer),
            'buffer_capacity': self.replay_buffer.capacity,
            'tasks_learned': self.task_count,
            'memory_utilization': len(self.replay_buffer) / self.replay_buffer.capacity
        }