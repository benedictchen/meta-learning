"""
Mathematically Correct MAML Core Implementation
==============================================

This module provides research-accurate MAML implementation that fixes critical
gradient flow issues in the existing code. Key improvements:

1. ✅ Stateless functional forward (no .data mutation)
2. ✅ True second-order gradients (preserves autograd graph) 
3. ✅ Proper gradient validation (no allow_unused=True masking)
4. ✅ Episodic BatchNorm isolation
5. ✅ ANIL/BOIL parameter partitioning

MATHEMATICAL FOUNDATION:
- MAML: meta_loss = E_τ[L_τ(f_θ')] where θ' = θ - α∇L_τ(f_θ)
- Proper gradient flow: ∇_θ meta_loss flows through inner adaptation
- No .data mutations that break autograd graph

This implementation is additive and preserves existing core functionality.
"""

from typing import Dict, Iterable, Tuple, Optional, List
import torch
from torch import nn
from torch.func import functional_call

# Type alias for parameter dictionaries
ParamDict = Dict[str, torch.Tensor]


def split_params(model: nn.Module, adapt_keys: Iterable[str]) -> Tuple[ParamDict, ParamDict]:
    """
    Split model parameters into adaptable and frozen sets.
    
    Args:
        model: PyTorch model
        adapt_keys: Parameter names to adapt during inner loop
        
    Returns:
        Tuple of (adapt_params, frozen_params) dictionaries
    """
    base_params = {k: p for k, p in model.named_parameters()}
    adapt_params = {k: base_params[k] for k in adapt_keys if k in base_params}
    frozen_params = {k: v for k, v in base_params.items() if k not in adapt_params}
    return adapt_params, frozen_params


def sgd_step(params: ParamDict, grads: ParamDict, lr: float) -> ParamDict:
    """
    Apply SGD update without mutating original parameters.
    
    Preserves gradient graph when grads require gradients.
    
    Args:
        params: Parameter dictionary
        grads: Gradient dictionary  
        lr: Learning rate
        
    Returns:
        Updated parameter dictionary (new tensors)
    """
    return {k: p - lr * grads[k] for k, p in params.items()}


def combine_params(frozen: ParamDict, adapted: ParamDict) -> ParamDict:
    """
    Combine frozen and adapted parameters into single dictionary.
    
    Args:
        frozen: Parameters that weren't adapted
        adapted: Parameters that were adapted
        
    Returns:
        Combined parameter dictionary
    """
    combined = dict(frozen)
    combined.update(adapted)
    return combined


@torch.no_grad()
def set_episodic_bn_mode(model: nn.Module) -> None:
    """
    Set episodic BatchNorm mode to prevent support->query leakage.
    
    Freezes running statistics while keeping affine parameters trainable.
    This ensures proper episodic isolation as required by few-shot learning.
    
    Args:
        model: Model to apply episodic BN policy to
    """
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            # Freeze running statistics (eval mode)
            module.eval()
            # Keep affine parameters trainable
            for param in module.parameters(recurse=False):
                param.requires_grad_(True)


def inner_adapt(
    model: nn.Module,
    support_x: torch.Tensor,
    support_y: torch.Tensor,
    loss_fn,
    inner_steps: int,
    inner_lr: float,
    adapt_param_names: Iterable[str],
    use_second_order: bool = True,
) -> ParamDict:
    """
    Perform inner loop adaptation with proper gradient flow.
    
    This is the corrected version that:
    - Uses functional_call (no .data mutation)  
    - Preserves autograd graph for second-order gradients
    - Enforces episodic BatchNorm isolation
    - Fails fast on disconnected parameters
    
    Args:
        model: Model to adapt
        support_x: Support set inputs [n_support, ...]
        support_y: Support set labels [n_support]
        loss_fn: Loss function (e.g., CrossEntropyLoss)
        inner_steps: Number of inner loop steps
        inner_lr: Inner loop learning rate
        adapt_param_names: Names of parameters to adapt
        use_second_order: True for MAML, False for FOMAML
        
    Returns:
        Dictionary of adapted parameters (preserves gradients if second_order)
    """
    # Apply episodic BatchNorm policy
    set_episodic_bn_mode(model)
    
    # Split parameters into adaptable and frozen
    adapt_params, frozen_params = split_params(model, adapt_param_names)
    
    # Clone adaptable parameters (creates leaf tensors if second_order=True)
    adapted = {k: v.clone().requires_grad_(True) for k, v in adapt_params.items()}
    
    # Inner loop adaptation
    for step in range(inner_steps):
        # Combine parameters for functional forward
        full_params = combine_params(frozen_params, adapted)
        
        # Forward pass using functional_call (no .data mutation!)
        logits = functional_call(model, full_params, (support_x,))
        loss = loss_fn(logits, support_y)
        
        # Compute gradients
        grads = torch.autograd.grad(
            loss,
            adapted.values(),
            create_graph=use_second_order,  # True preserves graph for MAML
            retain_graph=(step < inner_steps - 1),  # Retain for intermediate steps only
            allow_unused=False,  # Fail fast if parameters disconnected
        )
        
        # Apply SGD update (preserves gradient flow)
        grad_dict = {k: g for k, g in zip(adapted.keys(), grads)}
        adapted = sgd_step(adapted, grad_dict, inner_lr)
    
    # Return full parameter set with adaptations
    return combine_params(frozen_params, adapted)


def query_forward(
    model: nn.Module,
    query_x: torch.Tensor,
    adapted_params: ParamDict,
) -> torch.Tensor:
    """
    Forward pass on query set using adapted parameters.
    
    Uses functional_call to preserve gradient flow from adapted parameters
    back to original model parameters.
    
    Args:
        model: Model architecture
        query_x: Query set inputs [n_query, ...]
        adapted_params: Parameters after inner loop adaptation
        
    Returns:
        Query set logits [n_query, n_classes]
    """
    return functional_call(model, adapted_params, (query_x,))


def get_all_param_names(model: nn.Module) -> List[str]:
    """Get all parameter names from a model."""
    return [name for name, _ in model.named_parameters()]


def get_anil_adapt_names(model: nn.Module, head_module_names: List[str]) -> List[str]:
    """
    Get parameter names for ANIL adaptation (head-only).
    
    Args:
        model: Full model
        head_module_names: Names of head modules (e.g., ['classifier', 'head'])
        
    Returns:
        List of parameter names that should be adapted (head only)
    """
    adapt_names = []
    for name, _ in model.named_parameters():
        for head_name in head_module_names:
            if head_name in name:
                adapt_names.append(name)
                break
    return adapt_names


def get_boil_adapt_names(model: nn.Module, head_module_names: List[str]) -> List[str]:
    """
    Get parameter names for BOIL adaptation (backbone-only, freeze head).
    
    Args:
        model: Full model  
        head_module_names: Names of head modules to freeze
        
    Returns:
        List of parameter names that should be adapted (backbone only)
    """
    all_names = get_all_param_names(model)
    head_names = get_anil_adapt_names(model, head_module_names)
    return [name for name in all_names if name not in head_names]


def validate_param_connectivity(
    model: nn.Module,
    adapted_params: ParamDict,
    loss: torch.Tensor,
) -> None:
    """
    Validate that all adapted parameters have gradients.
    
    This catches the common issue where .data mutations break gradient flow.
    
    Args:
        model: Model being trained
        adapted_params: Parameters that should have gradients
        loss: Loss tensor that gradients flow from
        
    Raises:
        RuntimeError: If any adapted parameter lacks gradients
    """
    # Compute gradients to check connectivity
    grads = torch.autograd.grad(
        loss,
        adapted_params.values(),
        allow_unused=True,  # Only for validation
        retain_graph=True,
    )
    
    disconnected = []
    for (name, param), grad in zip(adapted_params.items(), grads):
        if grad is None:
            disconnected.append(name)
    
    if disconnected:
        raise RuntimeError(
            f"Gradient flow broken for parameters: {disconnected}. "
            f"This usually indicates .data mutation or incorrect functional_call usage."
        )


class StatelessMAML:
    """
    Stateless MAML implementation with proper gradient flow.
    
    This class encapsulates the corrected MAML algorithm and can be used
    alongside the existing implementation for comparison and validation.
    """
    
    def __init__(
        self,
        model: nn.Module,
        inner_lr: float = 0.01,
        inner_steps: int = 1,
        use_second_order: bool = True,
        adapt_param_names: Optional[List[str]] = None,
    ):
        """
        Initialize stateless MAML.
        
        Args:
            model: Model to meta-learn
            inner_lr: Inner loop learning rate
            inner_steps: Number of inner loop steps
            use_second_order: True for MAML, False for FOMAML
            adapt_param_names: Parameter names to adapt (None = all)
        """
        self.model = model
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.use_second_order = use_second_order
        self.adapt_param_names = adapt_param_names or get_all_param_names(model)
    
    def forward(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        query_x: torch.Tensor,
        query_y: torch.Tensor,
        loss_fn,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through MAML algorithm.
        
        Args:
            support_x: Support inputs [n_support, ...]
            support_y: Support labels [n_support]
            query_x: Query inputs [n_query, ...]
            query_y: Query labels [n_query] 
            loss_fn: Loss function
            
        Returns:
            Tuple of (query_loss, query_logits)
        """
        # Inner loop adaptation
        adapted_params = inner_adapt(
            self.model,
            support_x,
            support_y,
            loss_fn,
            self.inner_steps,
            self.inner_lr,
            self.adapt_param_names,
            self.use_second_order,
        )
        
        # Query forward pass
        query_logits = query_forward(self.model, query_x, adapted_params)
        query_loss = loss_fn(query_logits, query_y)
        
        return query_loss, query_logits
    
    def adapt_only(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        loss_fn,
    ) -> ParamDict:
        """
        Perform only the inner loop adaptation.
        
        Returns:
            Adapted parameter dictionary
        """
        return inner_adapt(
            self.model,
            support_x,
            support_y,
            loss_fn,
            self.inner_steps,
            self.inner_lr,
            self.adapt_param_names,
            self.use_second_order,
        )