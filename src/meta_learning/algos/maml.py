from __future__ import annotations
from typing import Dict, Tuple, Optional, List
import torch, torch.nn as nn, torch.nn.functional as F
from torch.func import functional_call
from collections import defaultdict, deque
import copy

def _named_params_buffers(model: nn.Module):
    params = {k: v for k, v in model.named_parameters()}
    buffers = {k: v for k, v in model.named_buffers()}
    return params, buffers

def inner_adapt_and_eval(model: nn.Module, loss_fn, support: Tuple[torch.Tensor, torch.Tensor],
                         query: Tuple[torch.Tensor, torch.Tensor], inner_lr: float = 0.4, first_order: bool = False,
                         freeze_bn_stats: bool = True):
    """One-step MAML with functional_call (handles arbitrary modules).
    freeze_bn_stats: if True, uses eval() mode to avoid BN running-stat updates during inner step.
    """
    (x_s, y_s), (x_q, y_q) = support, query
    params, buffers = _named_params_buffers(model)
    if freeze_bn_stats:
        model.eval()
    else:
        model.train()

    # forward on support
    logits_s = functional_call(model, (params, buffers), (x_s,))
    loss_s = loss_fn(logits_s, y_s)

    grads = torch.autograd.grad(loss_s, tuple(params.values()), create_graph=not first_order)
    # SGD update
    new_params = {k: p - inner_lr * g for (k, p), g in zip(params.items(), grads)}
    # evaluate on query
    logits_q = functional_call(model, (new_params, buffers), (x_q,))
    return loss_fn(logits_q, y_q)

def meta_outer_step(model: nn.Module, loss_fn, meta_batch, inner_lr=0.4, first_order=False, optimizer=None, freeze_bn_stats=True):
    losses = []
    for task in meta_batch:
        losses.append(inner_adapt_and_eval(model, loss_fn, task['support'], task['query'],
                                           inner_lr=inner_lr, first_order=first_order, freeze_bn_stats=freeze_bn_stats))
    meta_loss = torch.stack(losses).mean()
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True); meta_loss.backward(); optimizer.step()
    return meta_loss


class ContinualMAML(nn.Module):
    """
    Continual MAML with Elastic Weight Consolidation for few-shot continual learning.
    
    Integrates continual learning capabilities directly into MAML algorithm
    to prevent catastrophic forgetting across task sequences.
    """
    
    def __init__(self, model: nn.Module, memory_size: int = 1000, 
                 consolidation_strength: float = 1000.0, fisher_samples: int = 1000):
        super().__init__()
        self.model = model
        self.memory_size = memory_size
        self.consolidation_strength = consolidation_strength
        self.fisher_samples = fisher_samples
        
        # Episodic memory for experience replay
        self.memory_x = deque(maxlen=memory_size)
        self.memory_y = deque(maxlen=memory_size)
        self.memory_task_ids = deque(maxlen=memory_size)
        
        # EWC components
        self.previous_params = {}
        self.fisher_information = {}
        self.task_count = 0
        
    def add_to_memory(self, x: torch.Tensor, y: torch.Tensor, task_id: int):
        """Add examples to episodic memory using reservoir sampling."""
        batch_size = x.size(0)
        for i in range(batch_size):
            self.memory_x.append(x[i].clone().detach())
            self.memory_y.append(y[i].clone().detach())
            self.memory_task_ids.append(task_id)
    
    def compute_fisher_information(self, dataloader):
        """Compute diagonal Fisher Information Matrix."""
        fisher_dict = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                fisher_dict[name] = torch.zeros_like(param)
        
        self.model.eval()
        sample_count = 0
        
        for batch_x, batch_y in dataloader:
            if sample_count >= self.fisher_samples:
                break
                
            # Forward pass
            logits = self.model(batch_x)
            log_probs = F.log_softmax(logits, dim=-1)
            
            # Sample from predicted distribution
            probs = F.softmax(logits, dim=-1)
            sampled_labels = torch.multinomial(probs, 1).squeeze()
            
            # Compute gradients
            loss = F.nll_loss(log_probs, sampled_labels, reduction='mean')
            grads = torch.autograd.grad(loss, self.model.parameters(), create_graph=False)
            
            # Accumulate squared gradients
            for (name, param), grad in zip(self.model.named_parameters(), grads):
                if param.requires_grad and grad is not None:
                    fisher_dict[name] += grad ** 2
                    
            sample_count += batch_x.size(0)
        
        # Normalize and add numerical stability
        for name in fisher_dict:
            fisher_dict[name] /= sample_count
            fisher_dict[name] += 1e-8
            
        return fisher_dict
    
    def compute_ewc_loss(self) -> torch.Tensor:
        """Compute Elastic Weight Consolidation regularization loss."""
        ewc_loss = 0.0
        
        for task_id in self.previous_params:
            previous_params = self.previous_params[task_id]
            fisher_info = self.fisher_information[task_id]
            
            for name, param in self.model.named_parameters():
                if name in previous_params and name in fisher_info:
                    param_diff = param - previous_params[name]
                    fisher_weight = fisher_info[name]
                    ewc_loss += (fisher_weight * param_diff ** 2).sum()
        
        return self.consolidation_strength * ewc_loss / 2.0
    
    def consolidate_task(self, dataloader, task_id: int):
        """Consolidate knowledge after completing a task."""
        # Store current parameters as important
        previous_params = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                previous_params[name] = param.detach().clone()
        
        # Compute Fisher Information
        fisher_dict = self.compute_fisher_information(dataloader)
        
        # Store for EWC
        self.previous_params[task_id] = previous_params
        self.fisher_information[task_id] = fisher_dict
        self.task_count += 1
    
    def continual_inner_adapt_and_eval(self, support: Tuple[torch.Tensor, torch.Tensor],
                                     query: Tuple[torch.Tensor, torch.Tensor], 
                                     task_id: int, inner_lr: float = 0.4, 
                                     first_order: bool = False) -> torch.Tensor:
        """MAML inner loop with continual learning enhancements."""
        (x_s, y_s), (x_q, y_q) = support, query
        
        # Add support examples to memory
        self.add_to_memory(x_s, y_s, task_id)
        
        # Standard MAML inner adaptation
        params, buffers = _named_params_buffers(self.model)
        self.model.eval()  # Freeze BN stats
        
        # Forward on support
        logits_s = functional_call(self.model, (params, buffers), (x_s,))
        loss_s = F.cross_entropy(logits_s, y_s)
        
        # Add EWC regularization
        ewc_loss = self.compute_ewc_loss()
        total_loss_s = loss_s + ewc_loss
        
        # Compute gradients
        grads = torch.autograd.grad(total_loss_s, tuple(params.values()), 
                                   create_graph=not first_order)
        
        # SGD update
        new_params = {k: p - inner_lr * g for (k, p), g in zip(params.items(), grads)}
        
        # Evaluate on query with experience replay
        logits_q = functional_call(self.model, (new_params, buffers), (x_q,))
        query_loss = F.cross_entropy(logits_q, y_q)
        
        # Add replay loss if memory available
        if len(self.memory_x) > 0:
            # Sample from memory
            memory_size = min(32, len(self.memory_x))
            indices = torch.randperm(len(self.memory_x))[:memory_size]
            
            memory_x_batch = torch.stack([self.memory_x[i] for i in indices])
            memory_y_batch = torch.stack([self.memory_y[i] for i in indices])
            
            # Replay loss with adapted parameters
            memory_logits = functional_call(self.model, (new_params, buffers), (memory_x_batch,))
            replay_loss = F.cross_entropy(memory_logits, memory_y_batch)
            
            # Combine losses
            total_query_loss = query_loss + 0.5 * replay_loss
        else:
            total_query_loss = query_loss
            
        return total_query_loss
    
    def continual_meta_step(self, meta_batch, task_id: int, inner_lr=0.4, 
                           first_order=False, optimizer=None):
        """Meta-learning step with continual learning."""
        losses = []
        for task in meta_batch:
            loss = self.continual_inner_adapt_and_eval(
                task['support'], task['query'], task_id, inner_lr, first_order
            )
            losses.append(loss)
            
        meta_loss = torch.stack(losses).mean()
        
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            meta_loss.backward()
            optimizer.step()
            
        return meta_loss
