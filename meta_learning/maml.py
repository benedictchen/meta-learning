"""
Model-Agnostic Meta-Learning (MAML) implementation.

Based on "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks" 
by Finn et al., ICML 2017.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import copy


class MAMLConfig:
    """Configuration for MAML algorithm."""
    
    def __init__(self, inner_lr: float = 0.01, inner_steps: int = 5, 
                 first_order: bool = False, learn_inner_lr: bool = False):
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.first_order = first_order  # Use first-order approximation (FOMAML)
        self.learn_inner_lr = learn_inner_lr  # Learn inner learning rate


class MAMLLearner(nn.Module):
    """
    MAML learner that can adapt to new tasks quickly.
    
    Key features:
    - Gradient-based meta-learning
    - Few-shot adaptation with inner loop updates
    - Support for first-order approximation (FOMAML)
    - Learnable inner learning rates
    """
    
    def __init__(self, model: nn.Module, config: MAMLConfig):
        super().__init__()
        self.model = model
        self.config = config
        
        # Initialize learnable inner learning rates if requested
        if config.learn_inner_lr:
            self.inner_lrs = nn.ParameterList([
                nn.Parameter(torch.tensor(config.inner_lr))
                for _ in self.model.parameters()
            ])
        else:
            self.inner_lrs = None
            
    def forward(self, support_x: torch.Tensor, support_y: torch.Tensor,
                query_x: torch.Tensor, query_y: Optional[torch.Tensor] = None,
                return_loss: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass with MAML adaptation.
        
        Args:
            support_x: Support examples [N_support, ...]
            support_y: Support labels [N_support]
            query_x: Query examples [N_query, ...]
            query_y: Query labels [N_query] (optional, for loss computation)
            return_loss: Whether to return loss along with predictions
            
        Returns:
            logits: Query predictions [N_query, N_classes]
            loss: Query loss (if return_loss=True and query_y provided)
        """
        # Clone model for inner loop updates
        adapted_model = self._clone_model()
        
        # Inner loop: adapt to support set
        for step in range(self.config.inner_steps):
            # Forward pass on support set
            support_logits = adapted_model(support_x)
            support_loss = F.cross_entropy(support_logits, support_y)
            
            # Compute gradients only if parameters require grad
            param_list = [p for p in adapted_model.parameters() if p.requires_grad]
            if len(param_list) == 0:
                break  # No parameters to update
                
            try:
                gradients = torch.autograd.grad(
                    support_loss, 
                    param_list,
                    create_graph=not self.config.first_order,
                    retain_graph=step < self.config.inner_steps - 1,
                    allow_unused=True
                )
                
                # Update adapted model parameters
                self._update_adapted_model_v2(param_list, gradients)
            except RuntimeError:
                # If gradient computation fails, break the loop
                break
        
        # Forward pass on query set with adapted model
        query_logits = adapted_model(query_x)
        
        if return_loss and query_y is not None:
            query_loss = F.cross_entropy(query_logits, query_y)
            return query_logits, query_loss
        
        return query_logits
    
    def _clone_model(self) -> nn.Module:
        """Create a copy of the model for adaptation."""
        # Create a functional copy that shares parameters but allows gradient computation
        adapted_model = copy.deepcopy(self.model)
        
        # Ensure parameters require gradients for meta-learning
        for param in adapted_model.parameters():
            param.requires_grad_(True)
            
        return adapted_model
    
    def _update_adapted_model(self, adapted_model: nn.Module, gradients: List[torch.Tensor]):
        """Update adapted model parameters using gradients."""
        for i, (param, grad) in enumerate(zip(adapted_model.parameters(), gradients)):
            if grad is not None:
                if self.inner_lrs is not None:
                    lr = self.inner_lrs[i]
                else:
                    lr = self.config.inner_lr
                    
                param.data = param.data - lr * grad
    
    def _update_adapted_model_v2(self, param_list: List[torch.Tensor], gradients: List[torch.Tensor]):
        """Update adapted model parameters using gradients (v2)."""
        for i, (param, grad) in enumerate(zip(param_list, gradients)):
            if grad is not None:
                if self.inner_lrs is not None and i < len(self.inner_lrs):
                    lr = self.inner_lrs[i]
                else:
                    lr = self.config.inner_lr
                    
                param.data = param.data - lr * grad
    
    def meta_loss(self, support_x: torch.Tensor, support_y: torch.Tensor,
                  query_x: torch.Tensor, query_y: torch.Tensor) -> torch.Tensor:
        """
        Compute meta-loss for a single task (for meta-training).
        
        Args:
            support_x: Support examples
            support_y: Support labels  
            query_x: Query examples
            query_y: Query labels
            
        Returns:
            meta_loss: Loss on query set after adaptation
        """
        _, query_loss = self.forward(support_x, support_y, query_x, query_y, return_loss=True)
        return query_loss


def train_maml_step(maml_learner: MAMLLearner, meta_optimizer: torch.optim.Optimizer,
                    episodes: List[Dict], device: torch.device = None) -> float:
    """
    Perform one step of MAML meta-training.
    
    Args:
        maml_learner: MAML learner
        meta_optimizer: Meta-optimizer (e.g., Adam)
        episodes: Batch of episodes for meta-training
        device: Device to run on
        
    Returns:
        avg_meta_loss: Average meta-loss across episodes
    """
    if device is None:
        device = next(maml_learner.parameters()).device
    
    meta_optimizer.zero_grad()
    
    total_loss = 0.0
    for episode in episodes:
        # Move data to device
        support_x = episode['support_x'].to(device)
        support_y = episode['support_y'].to(device)
        query_x = episode['query_x'].to(device) 
        query_y = episode['query_y'].to(device)
        
        # Compute meta-loss for this episode
        meta_loss = maml_learner.meta_loss(support_x, support_y, query_x, query_y)
        
        # Accumulate gradients (meta-gradient)
        meta_loss.backward()
        total_loss += meta_loss.item()
    
    # Meta-update
    meta_optimizer.step()
    
    return total_loss / len(episodes)


def evaluate_maml(maml_learner: MAMLLearner, episodes: List[Dict], 
                  device: torch.device = None) -> float:
    """
    Evaluate MAML on a set of episodes.
    
    Args:
        maml_learner: MAML learner
        episodes: Episodes for evaluation
        device: Device to run on
        
    Returns:
        avg_accuracy: Average accuracy across episodes
    """
    if device is None:
        device = next(maml_learner.parameters()).device
    
    maml_learner.eval()
    total_accuracy = 0.0
    
    with torch.no_grad():
        for episode in episodes:
            support_x = episode['support_x'].to(device)
            support_y = episode['support_y'].to(device)
            query_x = episode['query_x'].to(device)
            query_y = episode['query_y'].to(device)
            
            # Get predictions after adaptation
            query_logits = maml_learner(support_x, support_y, query_x)
            
            # Compute accuracy
            query_pred = query_logits.argmax(dim=-1)
            accuracy = (query_pred == query_y).float().mean().item()
            total_accuracy += accuracy
    
    maml_learner.train()
    return total_accuracy / len(episodes)


# Example usage and testing
if __name__ == "__main__":
    # Test MAML implementation
    from .core import Conv4, make_episode, get_dataset
    
    # Create a simple classification head
    class SimpleClassifier(nn.Module):
        def __init__(self, feature_dim: int = 64, n_classes: int = 5):
            super().__init__()
            self.backbone = Conv4(input_channels=1, hidden_dim=feature_dim)
            self.classifier = nn.Linear(feature_dim, n_classes)
            
        def forward(self, x):
            features = self.backbone(x)
            return self.classifier(features)
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleClassifier().to(device)
    
    config = MAMLConfig(inner_lr=0.01, inner_steps=5, first_order=False)
    maml_learner = MAMLLearner(model, config).to(device)
    
    # Create synthetic episodes
    dataset = get_dataset("omniglot", split="train")
    episodes = [make_episode(dataset, n_way=5, k_shot=1, n_query=15) for _ in range(4)]
    
    # Meta-training step
    meta_optimizer = torch.optim.Adam(maml_learner.parameters(), lr=0.001)
    avg_loss = train_maml_step(maml_learner, meta_optimizer, episodes, device)
    print(f"Meta-training loss: {avg_loss:.4f}")
    
    # Evaluation
    test_episodes = [make_episode(dataset, n_way=5, k_shot=1, n_query=15) for _ in range(10)]
    avg_accuracy = evaluate_maml(maml_learner, test_episodes, device)
    print(f"MAML accuracy: {avg_accuracy:.4f}")