from __future__ import annotations

"""
Core utility functions for meta-learning.

This module contains utility functions that don't depend on complex data structures,
preventing circular imports while providing essential functionality.
"""

# Core utilities for meta-learning: gradient-preserving module operations
# Includes clone_module(), update_module(), detach_module(), and clone_parameters()
# for MAML inner loops, memory optimization, and Meta-SGD support

from typing import Tuple, Optional, Dict, Any, Union
import torch
import torch.nn as nn
import copy
import warnings
from ..shared.types import Episode


def partition_task(data: torch.Tensor, labels: torch.Tensor, shots: int = 1) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """
    Partition a classification task into support and query sets.
    
    The support set will contain `shots` samples per class, the query will take 
    the remaining samples.
    
    Args:
        data: Data to be partitioned into support and query [N, ...]
        labels: Labels of each data sample [N]
        shots: Number of data samples per class in the support set
        
    Returns:
        ((support_data, support_labels), (query_data, query_labels))
        
    Example:
        >>> X, y = taskset.sample()
        >>> (X_support, y_support), (X_query, y_query) = partition_task(X, y, shots=5)
    """
    assert data.size(0) == labels.size(0)
    unique_labels = labels.unique()
    ways = unique_labels.numel()
    data_shape = data.shape[1:]
    num_support = ways * shots
    num_query = data.size(0) - num_support
    assert num_query % ways == 0, 'Only query_shot == support_shot supported.'
    query_shots = num_query // ways
    
    support_data = torch.empty(
        (num_support,) + data_shape,
        device=data.device,
        dtype=data.dtype,
    )
    support_labels = torch.empty(
        num_support,
        device=labels.device,
        dtype=labels.dtype,
    )
    query_data = torch.empty(
        (num_query, ) + data_shape,
        device=data.device,
        dtype=data.dtype,
    )
    query_labels = torch.empty(
        num_query,
        device=labels.device,
        dtype=labels.dtype,
    )
    
    for i, label in enumerate(unique_labels):
        support_start = i * shots
        support_end = support_start + shots
        query_start = i * query_shots
        query_end = query_start + query_shots

        # Filter data
        label_data = data[labels == label]
        num_label_data = label_data.size(0)
        assert num_label_data == shots + query_shots, \
            'Only same number of query per label supported.'

        # Set value of labels
        support_labels[support_start:support_end].fill_(label)
        query_labels[query_start:query_end].fill_(label)

        # Set value of data
        support_data[support_start:support_end].copy_(label_data[:shots])
        query_data[query_start:query_end].copy_(label_data[shots:])

    return (support_data, support_labels), (query_data, query_labels)


def remap_labels(y_support: torch.Tensor, y_query: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Remap labels to contiguous integers starting from 0.
    
    This function takes support and query labels that may have arbitrary integer
    values and remaps them to contiguous integers [0, 1, 2, ..., n_classes-1].
    The mapping is determined by the unique classes present in the support set.
    
    Args:
        y_support (torch.Tensor): Support set labels of shape [n_support].
            Can contain any integer values.
        y_query (torch.Tensor): Query set labels of shape [n_query].
            Must be a subset of classes in y_support.
            
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - Remapped support labels of shape [n_support] with values [0, n_classes-1]
            - Remapped query labels of shape [n_query] with values [0, n_classes-1]
            
    Raises:
        KeyError: If query labels contain classes not present in support set.
        
    Examples:
        >>> import torch
        >>> from meta_learning.core.utils import remap_labels
        >>> 
        >>> # Labels with arbitrary values
        >>> y_support = torch.tensor([10, 25, 10, 37, 25])
        >>> y_query = torch.tensor([10, 10, 25, 37, 37, 25])
        >>> 
        >>> # Remap to contiguous [0, 1, 2]
        >>> support_remapped, query_remapped = remap_labels(y_support, y_query)
        >>> print(support_remapped)  # tensor([0, 1, 0, 2, 1])
        >>> print(query_remapped)    # tensor([0, 0, 1, 2, 2, 1])
        >>> 
        >>> # Verify mapping consistency
        >>> print(torch.unique(support_remapped))  # tensor([0, 1, 2])
    """
    classes = torch.unique(y_support)
    mapping = {c.item(): i for i, c in enumerate(classes)}
    ys = torch.tensor([mapping[int(c.item())] for c in y_support], device=y_support.device)
    yq = torch.tensor([mapping[int(c.item())] for c in y_query], device=y_query.device)
    return ys.long(), yq.long()


def create_episode_from_partition(data: torch.Tensor, labels: torch.Tensor, shots: int = 1) -> Episode:
    """Create Episode from data using partition_task utility.
    
    This is a factory function that combines partition_task with Episode creation,
    providing a clean interface for creating episodes from raw data.
    
    Args:
        data: Data tensor [N, ...]
        labels: Label tensor [N]
        shots: Number of support shots per class
        
    Returns:
        Episode created by partitioning the data
        
    Example:
        >>> data = torch.randn(50, 32)  # 50 samples, 32 features
        >>> labels = torch.randint(0, 5, (50,))  # 5 classes
        >>> episode = create_episode_from_partition(data, labels, shots=2)
    """
    (support_data, support_labels), (query_data, query_labels) = partition_task(data, labels, shots)
    return Episode(support_data, support_labels, query_data, query_labels)


def clone_module(module: nn.Module, memo: Optional[Dict[int, nn.Module]] = None) -> nn.Module:
    """
    Create a differentiable clone of a PyTorch module preserving computational graph.
    
    This function creates a clone of a module where parameters maintain their
    gradient information and can be updated differentiably. Unlike deepcopy,
    this preserves the computational graph for meta-learning algorithms.
    
    Args:
        module: PyTorch module to clone
        memo: Optional memo dict to handle shared parameters efficiently
        
    Returns:
        Cloned module with preserved gradient computation
        
    Examples:
        >>> model = nn.Linear(10, 5)
        >>> cloned = clone_module(model)
        >>> # Parameters can be updated differentiably
        >>> loss = F.mse_loss(cloned(x), y)
        >>> grads = torch.autograd.grad(loss, cloned.parameters())
    """
    if memo is None:
        memo = {}
    
    # Check memo to handle shared parameters
    module_id = id(module)
    if module_id in memo:
        return memo[module_id]
    
    # Create new module instance using deepcopy for structure
    cloned = copy.deepcopy(module)
    
    # Replace parameters with gradient-preserving versions
    def clone_parameter(param: nn.Parameter) -> nn.Parameter:
        if param.requires_grad:
            # Create new parameter that preserves gradient computation
            cloned_param = param.clone().detach().requires_grad_(True)
            # Preserve parameter metadata if any
            if hasattr(param, '_meta'):
                cloned_param._meta = param._meta
            return cloned_param
        else:
            return param.clone().detach()
    
    # Replace parameters recursively
    for name, param in module.named_parameters():
        if param is not None:
            # Navigate to the correct submodule
            *path, param_name = name.split('.')
            current_module = cloned
            for attr_name in path:
                current_module = getattr(current_module, attr_name)
            
            # Replace parameter
            setattr(current_module, param_name, nn.Parameter(clone_parameter(param)))
    
    # Handle buffers (non-trainable parameters)
    for name, buffer in module.named_buffers():
        if buffer is not None:
            # Navigate to the correct submodule
            *path, buffer_name = name.split('.')
            current_module = cloned
            for attr_name in path:
                current_module = getattr(current_module, attr_name)
            
            # Replace buffer
            cloned_buffer = buffer.clone().detach()
            current_module.register_buffer(buffer_name, cloned_buffer)
    
    # Handle special cases for problematic layers
    for module_name, submodule in cloned.named_modules():
        # Handle RNN layers that have flatten_parameters
        if hasattr(submodule, 'flatten_parameters'):
            try:
                submodule.flatten_parameters()
            except RuntimeError:
                # Some RNN configurations can't flatten parameters
                warnings.warn(f"Could not flatten parameters for {module_name}")
    
    memo[module_id] = cloned
    return cloned


def update_module(module: nn.Module, updates: Optional[Dict[str, torch.Tensor]] = None, 
                  memo: Optional[Dict[int, nn.Module]] = None) -> nn.Module:
    """
    Apply differentiable parameter updates to a module.
    
    This function applies parameter updates while preserving gradient flow,
    essential for meta-learning algorithms that require second-order gradients.
    
    Args:
        module: Module to update
        updates: Dict mapping parameter names to update tensors
        memo: Optional memo dict for shared parameter handling
        
    Returns:
        Updated module with preserved gradient computation
        
    Examples:
        >>> model = nn.Linear(10, 5)
        >>> # Compute gradients for inner loop update
        >>> loss = F.mse_loss(model(x), y)
        >>> grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
        >>> # Create updates dict
        >>> updates = {name: -0.01 * grad for (name, param), grad 
        ...           in zip(model.named_parameters(), grads)}
        >>> updated_model = update_module(model, updates)
    """
    if updates is None:
        return module
    
    if memo is None:
        memo = {}
    
    module_id = id(module)
    if module_id in memo:
        return memo[module_id]
    
    try:
        # Create updated module by cloning first
        updated_module = clone_module(module)
        
        # Apply parameter updates
        for param_name, update in updates.items():
            if update is not None:
                # Navigate to parameter
                *path, param_attr = param_name.split('.')
                current_module = updated_module
                for attr_name in path:
                    current_module = getattr(current_module, attr_name)
                
                # Get current parameter
                current_param = getattr(current_module, param_attr)
                if isinstance(current_param, nn.Parameter):
                    # Apply differentiable update: param = param + update
                    new_param = current_param + update
                    setattr(current_module, param_attr, nn.Parameter(new_param))
                else:
                    warnings.warn(f"Skipping update for non-parameter {param_name}")
        
        memo[module_id] = updated_module
        return updated_module
        
    except Exception as e:
        warnings.warn(f"Parameter update failed: {e}, returning original module")
        return module


def detach_module(module: nn.Module, keep_requires_grad: bool = True,
                  memo: Optional[Dict[int, nn.Module]] = None) -> nn.Module:
    """
    Detach all parameters and buffers from computational graph for memory optimization.
    
    This function is useful for test-time compute scaling where you want to
    break gradient computation to save memory while optionally preserving
    the requires_grad flag for future gradient computation.
    
    Args:
        module: Module to detach
        keep_requires_grad: Whether to preserve requires_grad flags
        memo: Optional memo dict for shared parameter handling
        
    Returns:
        Module with detached parameters and buffers
        
    Examples:
        >>> model = nn.Linear(10, 5)
        >>> # After some forward passes with gradient tracking
        >>> detached_model = detach_module(model, keep_requires_grad=True)
        >>> # Parameters are detached but can still compute gradients
    """
    if memo is None:
        memo = {}
    
    module_id = id(module)
    if module_id in memo:
        return memo[module_id]
    
    # Create detached module
    detached = copy.deepcopy(module)
    
    # Detach all parameters
    for name, param in module.named_parameters():
        if param is not None:
            # Navigate to parameter location
            *path, param_name = name.split('.')
            current_module = detached
            for attr_name in path:
                current_module = getattr(current_module, attr_name)
            
            # Create detached parameter
            detached_param = param.detach()
            if keep_requires_grad and param.requires_grad:
                detached_param.requires_grad_(True)
            
            setattr(current_module, param_name, nn.Parameter(detached_param))
    
    # Detach all buffers
    for name, buffer in module.named_buffers():
        if buffer is not None:
            # Navigate to buffer location
            *path, buffer_name = name.split('.')
            current_module = detached
            for attr_name in path:
                current_module = getattr(current_module, attr_name)
            
            # Register detached buffer
            detached_buffer = buffer.detach()
            current_module.register_buffer(buffer_name, detached_buffer)
    
    memo[module_id] = detached
    return detached


def safe_clone_module(module: nn.Module, memo: Optional[Dict[int, nn.Module]] = None) -> nn.Module:
    """
    Safe wrapper for clone_module with comprehensive error handling.
    
    Args:
        module: Module to clone
        memo: Optional memo dict
        
    Returns:
        Cloned module, or original module if cloning fails
    """
    try:
        return clone_module(module, memo)
    except Exception as e:
        warnings.warn(f"clone_module failed: {e}, returning deepcopy fallback")
        try:
            return copy.deepcopy(module)
        except Exception as e2:
            warnings.warn(f"deepcopy fallback also failed: {e2}, returning original module")
            return module


def safe_update_module(module: nn.Module, updates: Optional[Dict[str, torch.Tensor]] = None,
                       memo: Optional[Dict[int, nn.Module]] = None) -> nn.Module:
    """
    Safe wrapper for update_module with comprehensive error handling.
    
    Args:
        module: Module to update  
        updates: Parameter updates
        memo: Optional memo dict
        
    Returns:
        Updated module, or original module if update fails
    """
    try:
        return update_module(module, updates, memo)
    except Exception as e:
        warnings.warn(f"update_module failed: {e}, returning original module")
        return module


def validate_module_updates(module: nn.Module, updates: Dict[str, torch.Tensor]) -> bool:
    """
    Validate that updates dictionary matches module parameters.
    
    Args:
        module: Module to validate against
        updates: Updates dictionary to validate
        
    Returns:
        True if updates are valid, False otherwise
    """
    try:
        param_names = {name for name, _ in module.named_parameters()}
        update_names = set(updates.keys())
        
        # Check for invalid parameter names
        invalid_names = update_names - param_names
        if invalid_names:
            warnings.warn(f"Invalid parameter names in updates: {invalid_names}")
            return False
        
        # Check tensor compatibility
        for name, update in updates.items():
            param = dict(module.named_parameters())[name]
            if param.shape != update.shape:
                warnings.warn(f"Shape mismatch for {name}: param {param.shape} vs update {update.shape}")
                return False
        
        return True
    except Exception as e:
        warnings.warn(f"Validation failed: {e}")
        return False


def clone_parameters(parameter_list: nn.ParameterList) -> nn.ParameterList:
    """
    Clone a ParameterList preserving gradient computation.
    
    Required for Meta-SGD algorithm which learns per-parameter learning rates
    stored as nn.ParameterList. This function creates gradient-preserving clones.
    
    Args:
        parameter_list: nn.ParameterList to clone
        
    Returns:
        Cloned ParameterList with preserved gradient computation
        
    Examples:
        >>> lrs = nn.ParameterList([nn.Parameter(torch.ones(10)) for _ in range(3)])
        >>> cloned_lrs = clone_parameters(lrs)
        >>> # Each cloned parameter preserves gradient flow
    """
    cloned_params = []
    
    for param in parameter_list:
        if param is None:
            cloned_params.append(None)
            continue
            
        if param.requires_grad:
            # Clone with gradient preservation
            cloned_param = param.clone().detach().requires_grad_(True)
            
            # Preserve any metadata attributes
            if hasattr(param, '_meta'):
                cloned_param._meta = param._meta
            if hasattr(param, 'grad_fn'):
                # Preserve gradient function information if needed
                pass
                
        else:
            # Simple clone for non-gradient parameters
            cloned_param = param.clone().detach()
            
        cloned_params.append(nn.Parameter(cloned_param))
    
    return nn.ParameterList(cloned_params)