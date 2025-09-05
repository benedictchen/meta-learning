"""
MAML Core Components with Proper Inner/Outer Separation
======================================================

Clean separation of MAML inner adaptation from outer meta-learning loop.
Based on Finn et al. (2017) with proper gradient computation handling.

This module provides the mathematical foundation for MAML with clear contracts
between inner loop adaptation and outer loop meta-updates.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Callable, List, Any
from dataclasses import dataclass


@dataclass 
class InnerLoopConfig:
    """Configuration for inner loop adaptation."""
    inner_lr: float = 0.01
    inner_steps: int = 1
    first_order: bool = False  # Use first-order approximation (FOMAML)
    allow_unused: bool = True  # Allow unused parameters in gradient computation


@dataclass
class InnerLoopResult:
    """Results from inner loop adaptation."""
    adapted_params: Dict[str, torch.Tensor]
    adaptation_loss: torch.Tensor
    support_logits: torch.Tensor
    query_logits: Optional[torch.Tensor] = None
    query_loss: Optional[torch.Tensor] = None


def functional_parameter_update(
    params: Dict[str, torch.Tensor],
    gradients: List[Optional[torch.Tensor]], 
    param_names: List[str],
    learning_rate: float
) -> Dict[str, torch.Tensor]:
    """
    Functional parameter update for MAML inner loop.
    
    Args:
        params: Current parameters 
        gradients: Gradients for each parameter
        param_names: Parameter names in same order as gradients
        learning_rate: Inner learning rate
        
    Returns:
        Updated parameters dictionary
    """
    adapted_params = {}
    for (name, param), grad in zip(params.items(), gradients):
        if grad is not None:
            adapted_params[name] = param - learning_rate * grad
        else:
            adapted_params[name] = param  # No gradient, keep original
    return adapted_params


def compute_inner_gradients(
    model: nn.Module,
    loss: torch.Tensor,
    create_graph: bool = True,
    allow_unused: bool = True
) -> List[Optional[torch.Tensor]]:
    """
    Compute gradients for inner loop adaptation.
    
    Args:
        model: Model to compute gradients for
        loss: Loss tensor to differentiate
        create_graph: Whether to create computational graph (needed for second-order)
        allow_unused: Whether to allow unused parameters
        
    Returns:
        List of gradients for each parameter
    """
    return torch.autograd.grad(
        outputs=loss,
        inputs=model.parameters(),
        create_graph=create_graph,
        allow_unused=allow_unused,
        retain_graph=create_graph
    )


def inner_adaptation_step(
    model: nn.Module,
    support_x: torch.Tensor,
    support_y: torch.Tensor,
    config: InnerLoopConfig
) -> InnerLoopResult:
    """
    Single inner adaptation step for MAML.
    
    Performs one gradient descent step on the support set and returns
    the adapted parameters along with adaptation statistics.
    
    Args:
        model: Base model to adapt
        support_x: Support set inputs
        support_y: Support set labels
        config: Inner loop configuration
        
    Returns:
        InnerLoopResult containing adapted parameters and statistics
    """
    # Forward pass on support set
    support_logits = model(support_x)
    adaptation_loss = F.cross_entropy(support_logits, support_y)
    
    # Compute gradients for adaptation
    gradients = compute_inner_gradients(
        model=model,
        loss=adaptation_loss,
        create_graph=not config.first_order,
        allow_unused=config.allow_unused
    )
    
    # Functional parameter update
    param_names = [name for name, _ in model.named_parameters()]
    adapted_params = functional_parameter_update(
        params=dict(model.named_parameters()),
        gradients=gradients,
        param_names=param_names,
        learning_rate=config.inner_lr
    )
    
    return InnerLoopResult(
        adapted_params=adapted_params,
        adaptation_loss=adaptation_loss,
        support_logits=support_logits
    )


def multi_step_inner_adaptation(
    model: nn.Module,
    support_x: torch.Tensor,
    support_y: torch.Tensor,
    config: InnerLoopConfig
) -> InnerLoopResult:
    """
    Multi-step inner adaptation for MAML.
    
    Performs multiple gradient descent steps on the support set.
    Each step uses the adapted parameters from the previous step.
    
    Args:
        model: Base model to adapt
        support_x: Support set inputs
        support_y: Support set labels
        config: Inner loop configuration
        
    Returns:
        InnerLoopResult with final adapted parameters
    """
    current_params = dict(model.named_parameters())
    
    for step in range(config.inner_steps):
        # Create temporary model with current parameters
        with torch.no_grad():
            for name, param in current_params.items():
                model.state_dict()[name].copy_(param)
        
        # Perform adaptation step
        result = inner_adaptation_step(model, support_x, support_y, config)
        current_params = result.adapted_params
        
        # For multi-step, we only keep the final adaptation loss
        if step == config.inner_steps - 1:
            return result
    
    # This should never be reached due to the loop structure
    return result


def evaluate_adapted_model(
    model: nn.Module,
    adapted_params: Dict[str, torch.Tensor],
    query_x: torch.Tensor,
    query_y: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Evaluate adapted model on query set.
    
    Uses functional forward pass with adapted parameters.
    
    Args:
        model: Base model
        adapted_params: Parameters from inner adaptation
        query_x: Query set inputs
        query_y: Query set labels
        
    Returns:
        Tuple of (query_logits, query_loss)
    """
    # Functional forward pass with adapted parameters
    query_logits = torch.func.functional_call(model, adapted_params, query_x)
    query_loss = F.cross_entropy(query_logits, query_y)
    
    return query_logits, query_loss


def maml_inner_loop(
    model: nn.Module,
    support_x: torch.Tensor,
    support_y: torch.Tensor,
    query_x: torch.Tensor,
    query_y: torch.Tensor,
    config: InnerLoopConfig
) -> InnerLoopResult:
    """
    Complete MAML inner loop: adaptation + query evaluation.
    
    This is the core MAML operation that:
    1. Adapts model parameters using support set
    2. Evaluates adapted model on query set
    3. Returns query loss for outer loop optimization
    
    Args:
        model: Base model to adapt
        support_x: Support set inputs  
        support_y: Support set labels
        query_x: Query set inputs
        query_y: Query set labels
        config: Inner loop configuration
        
    Returns:
        InnerLoopResult with complete adaptation and evaluation results
    """
    # Perform inner adaptation
    if config.inner_steps == 1:
        result = inner_adaptation_step(model, support_x, support_y, config)
    else:
        result = multi_step_inner_adaptation(model, support_x, support_y, config)
    
    # Evaluate on query set
    query_logits, query_loss = evaluate_adapted_model(
        model=model,
        adapted_params=result.adapted_params,
        query_x=query_x,
        query_y=query_y
    )
    
    # Add query evaluation to results
    result.query_logits = query_logits
    result.query_loss = query_loss
    
    return result


class MAMLInnerLoop:
    """
    Stateful MAML inner loop manager.
    
    Encapsulates inner loop configuration and provides clean interface
    for MAML adaptation operations.
    """
    
    def __init__(self, config: Optional[InnerLoopConfig] = None):
        self.config = config or InnerLoopConfig()
        
    def adapt_and_evaluate(
        self,
        model: nn.Module,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        query_x: torch.Tensor,
        query_y: torch.Tensor
    ) -> InnerLoopResult:
        """Perform full inner loop adaptation and evaluation."""
        return maml_inner_loop(
            model=model,
            support_x=support_x,
            support_y=support_y,
            query_x=query_x,
            query_y=query_y,
            config=self.config
        )
        
    def adapt_only(
        self,
        model: nn.Module,
        support_x: torch.Tensor,
        support_y: torch.Tensor
    ) -> InnerLoopResult:
        """Perform only inner adaptation (no query evaluation)."""
        if self.config.inner_steps == 1:
            return inner_adaptation_step(model, support_x, support_y, self.config)
        else:
            return multi_step_inner_adaptation(model, support_x, support_y, self.config)


class MAMLOuterLoop:
    """
    MAML outer loop (meta-learning) manager.
    
    Handles meta-batch processing and meta-parameter updates.
    """
    
    def __init__(self, model: nn.Module, meta_lr: float = 0.001):
        self.model = model
        self.meta_optimizer = torch.optim.Adam(model.parameters(), lr=meta_lr)
        
    def meta_train_step(
        self,
        meta_batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
        inner_loop: MAMLInnerLoop
    ) -> float:
        """
        Perform one meta-training step.
        
        Args:
            meta_batch: List of (support_x, support_y, query_x, query_y) tuples
            inner_loop: Configured inner loop manager
            
        Returns:
            Average meta-loss across tasks
        """
        self.meta_optimizer.zero_grad()
        total_loss = 0.0
        
        for support_x, support_y, query_x, query_y in meta_batch:
            # Inner loop adaptation and evaluation
            result = inner_loop.adapt_and_evaluate(
                model=self.model,
                support_x=support_x,
                support_y=support_y, 
                query_x=query_x,
                query_y=query_y
            )
            
            # Accumulate query loss for meta-update
            total_loss += result.query_loss
        
        # Averaged meta-loss
        meta_loss = total_loss / len(meta_batch)
        
        # Meta-gradient computation and update
        meta_loss.backward()
        self.meta_optimizer.step()
        
        return meta_loss.item()


def create_maml_trainer(
    model: nn.Module,
    inner_lr: float = 0.01,
    meta_lr: float = 0.001,
    inner_steps: int = 1,
    first_order: bool = False
) -> Tuple[MAMLInnerLoop, MAMLOuterLoop]:
    """
    Factory function to create properly configured MAML trainer.
    
    Returns:
        Tuple of (inner_loop_manager, outer_loop_manager)
    """
    inner_config = InnerLoopConfig(
        inner_lr=inner_lr,
        inner_steps=inner_steps,
        first_order=first_order
    )
    
    inner_loop = MAMLInnerLoop(inner_config)
    outer_loop = MAMLOuterLoop(model, meta_lr)
    
    return inner_loop, outer_loop


if __name__ == "__main__":
    # Quick validation
    print("🔧 Testing MAML core components...")
    
    # Create simple test model
    model = nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 3)
    )
    
    # Create test data
    support_x = torch.randn(6, 10)
    support_y = torch.randint(0, 3, (6,))
    query_x = torch.randn(3, 10)
    query_y = torch.randint(0, 3, (3,))
    
    # Test inner loop
    inner_config = InnerLoopConfig(inner_lr=0.01, first_order=False)
    result = maml_inner_loop(model, support_x, support_y, query_x, query_y, inner_config)
    
    assert result.query_loss is not None
    assert torch.isfinite(result.query_loss)
    assert len(result.adapted_params) > 0
    print("✅ Inner loop adaptation working")
    
    # Test trainer creation
    inner_loop, outer_loop = create_maml_trainer(model)
    
    meta_batch = [(support_x, support_y, query_x, query_y)]
    meta_loss = outer_loop.meta_train_step(meta_batch, inner_loop)
    
    assert isinstance(meta_loss, float)
    assert meta_loss > 0
    print("✅ Outer loop meta-training working")
    
    print("🎉 MAML core components validated!")