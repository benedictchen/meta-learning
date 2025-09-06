"""
TODO: ANIL (Almost No Inner Loop) Algorithm Implementation  
========================================================

PRIORITY: HIGH - Missing core MAML variant

ANIL: Raghu et al. (2019) "Rapid Learning or Feature Reuse? Towards Understanding the Effectiveness of MAML"
ICLR 2020

ALGORITHM OVERVIEW:
ANIL is a simplified variant of MAML that only performs gradient updates on the final layer
during the inner loop, while keeping all other layers frozen. This reduces computational
cost while maintaining similar performance to full MAML.

MATHEMATICAL FOUNDATION:
θ_head' = θ_head - α ∇_θ_head L_task(θ_backbone, θ_head)
θ_backbone remains frozen during inner loop adaptation

INTEGRATION TARGET:
- Add to algorithms/__init__.py exports  
- Update CLI to support ANIL training
- Add benchmarking against MAML and Meta-SGD
- Integrate with our existing clone_module utilities in core.utils
"""

import torch
import torch.nn as nn
from torch.autograd import grad
from typing import Optional, List, Dict, Union
from ..core.utils import clone_module, safe_clone_module

class ANIL(nn.Module):
    """
    ANIL (Almost No Inner Loop) implementation.
    
    Only adapts the final classification head during inner loop updates,
    keeping all backbone features frozen. This provides significant computational
    savings compared to full MAML while maintaining competitive performance.
    """
    
    def __init__(self, model: nn.Module, head_lr: float = 1e-3, 
                 first_order: bool = False, allow_unused: Optional[bool] = None,
                 allow_nograd: bool = False):
        """
        Initialize ANIL.
        
        Args:
            model: Base model to meta-learn  
            head_lr: Learning rate for head adaptation
            first_order: Use first-order approximation
            allow_unused: Allow unused parameters in gradient computation
            allow_nograd: Allow parameters without gradients
        """
        super().__init__()
        
        # TODO: STEP 1 - Store base model and identify head/backbone split
        # The key insight of ANIL is to separate the model into:
        # - backbone: Feature extraction layers (frozen during inner loop)
        # - head: Final classification layer (adapted during inner loop)
        # 
        # For common architectures:
        # - Conv4: backbone = features (conv layers), head = classifier (linear layer)
        # - ResNet: backbone = all layers except final fc, head = final fc layer
        # 
        # Implementation approach:
        # self.model = model
        # self.head_lr = head_lr  
        # self.first_order = first_order
        # 
        # # Identify head parameters (typically final linear layer)
        # self.head_params = []
        # self.backbone_params = []
        # for name, param in model.named_parameters():
        #     if 'classifier' in name or 'fc' in name or 'head' in name:
        #         self.head_params.append(param)
        #     else:
        #         self.backbone_params.append(param)
        
        raise NotImplementedError("TODO: Implement ANIL.__init__")
    
    def forward(self, *args, **kwargs):
        """Forward pass through base model."""
        # TODO: Forward through the wrapped model
        # return self.model(*args, **kwargs)
        
        raise NotImplementedError("TODO: Implement ANIL.forward")
    
    def clone(self, first_order: Optional[bool] = None,
             allow_unused: Optional[bool] = None, 
             allow_nograd: Optional[bool] = None):
        """
        Create a copy of the model for adaptation.
        
        Returns:
            Cloned ANIL model ready for head-only adaptation
        """
        # TODO: Clone the model using our existing utilities
        # Based on MAML clone pattern but simpler since we only adapt head:
        # 
        # from ..core.utils import clone_module
        # cloned_model = clone_module(self.model)
        # return ANIL(cloned_model, 
        #             head_lr=self.head_lr,
        #             first_order=first_order or self.first_order,
        #             allow_unused=allow_unused,
        #             allow_nograd=allow_nograd)
        
        raise NotImplementedError("TODO: Implement ANIL.clone")
    
    def adapt(self, loss: torch.Tensor, first_order: Optional[bool] = None,
             allow_unused: Optional[bool] = None, allow_nograd: Optional[bool] = None):
        """
        Perform one step of head-only adaptation.
        
        Args:
            loss: Loss to adapt to (computed on current model state)
            first_order: Override first-order setting  
            allow_unused: Override allow_unused setting
            allow_nograd: Override allow_nograd setting
            
        Returns:
            None (updates model in-place)
        """
        # TODO: STEP 1 - Determine gradient computation mode
        # if first_order is None:
        #     first_order = self.first_order
        # second_order = not first_order
        
        # TODO: STEP 2 - Compute gradients ONLY for head parameters
        # This is the key difference from MAML - we only compute gradients
        # for the final classification layer, not the entire model:
        #
        # gradients = grad(loss,
        #                  self.head_params,  # Only head parameters!
        #                  retain_graph=second_order,
        #                  create_graph=second_order,
        #                  allow_unused=allow_unused or False)
        
        # TODO: STEP 3 - Apply updates only to head parameters
        # for param, grad_val in zip(self.head_params, gradients):
        #     if grad_val is not None:
        #         param.data = param.data - self.head_lr * grad_val
        
        # TODO: STEP 4 - Backbone parameters remain completely unchanged
        # This is automatic since we never compute gradients for them
        
        raise NotImplementedError("TODO: Implement ANIL.adapt")


class ANILWrapper(nn.Module):
    """
    Wrapper to automatically split any model into backbone + head for ANIL.
    
    This wrapper analyzes model architecture and automatically identifies
    the final classification layer as the "head" while treating everything
    else as the frozen "backbone".
    """
    
    def __init__(self, model: nn.Module, head_layer_names: Optional[List[str]] = None):
        """
        Initialize automatic backbone/head splitting.
        
        Args:
            model: Model to wrap for ANIL
            head_layer_names: Optional list of parameter names to treat as head.
                            If None, automatically detects common patterns.
        """
        super().__init__()
        
        # TODO: STEP 1 - Store original model
        # self.model = model
        
        # TODO: STEP 2 - Automatic head detection
        # if head_layer_names is None:
        #     # Common patterns for final classification layers
        #     head_patterns = ['classifier', 'fc', 'head', 'linear', 'output']
        #     head_layer_names = []
        #     for name, param in model.named_parameters():
        #         for pattern in head_patterns:
        #             if pattern in name.lower():
        #                 head_layer_names.append(name)
        #                 break
        
        # TODO: STEP 3 - Create parameter lists
        # self.head_layer_names = head_layer_names
        # self._update_parameter_lists()
        
        raise NotImplementedError("TODO: Implement ANILWrapper.__init__")
    
    def _update_parameter_lists(self):
        """Update head and backbone parameter lists based on current model state.""" 
        # TODO: Categorize parameters into head vs backbone
        # self.head_params = []
        # self.backbone_params = []
        # for name, param in self.model.named_parameters():
        #     if name in self.head_layer_names:
        #         self.head_params.append(param)
        #     else:
        #         self.backbone_params.append(param)
        
        raise NotImplementedError("TODO: Implement parameter categorization")
    
    def get_head_parameters(self) -> List[nn.Parameter]:
        """Return parameters that will be adapted during inner loop."""
        # TODO: Return list of head parameters for gradient computation
        # return self.head_params
        
        raise NotImplementedError("TODO: Return head parameters")
    
    def get_backbone_parameters(self) -> List[nn.Parameter]:
        """Return parameters that remain frozen during inner loop."""
        # TODO: Return list of backbone parameters (frozen during adaptation)
        # return self.backbone_params
        
        raise NotImplementedError("TODO: Return backbone parameters")
    
    def forward(self, *args, **kwargs):
        """Forward pass through wrapped model."""
        # TODO: Forward through the original model
        # return self.model(*args, **kwargs)
        
        raise NotImplementedError("TODO: Implement ANILWrapper.forward")


def anil_update(model: nn.Module, head_params: List[nn.Parameter], 
               head_lr: float, gradients: List[torch.Tensor]):
    """
    Apply ANIL parameter updates (head-only).
    
    Args:
        model: Model being updated
        head_params: List of head parameters to update
        head_lr: Learning rate for head updates
        gradients: Gradients for head parameters
        
    Returns:
        None (updates model in-place)
    """
    # TODO: STEP 1 - Apply updates only to head parameters
    # for param, grad_val in zip(head_params, gradients):
    #     if grad_val is not None:
    #         param.data = param.data - head_lr * grad_val
    
    # TODO: STEP 2 - Backbone parameters are never touched
    # This is the core insight of ANIL - massive computational savings
    # by avoiding gradients/updates for the majority of parameters
    
    raise NotImplementedError("TODO: Implement anil_update function")