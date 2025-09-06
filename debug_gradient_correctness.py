"""
Debug script to isolate MAML gradient correctness issues.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call

# Simple 1-layer linear model for debugging
class SimpleLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1, bias=False)
        # Initialize with known weight
        self.linear.weight.data.fill_(1.0)
    
    def forward(self, x):
        return self.linear(x).squeeze(-1)

def manual_maml_step(model, support_x, support_y, query_x, query_y, inner_lr=0.1):
    """
    Manual MAML implementation for debugging.
    """
    print("🔍 Manual MAML step:")
    print(f"   Initial weight: {model.linear.weight.item():.4f}")
    
    # Get initial parameters
    params = {k: v for k, v in model.named_parameters()}
    buffers = {k: v for k, v in model.named_buffers()}
    
    # Support forward pass
    support_pred = functional_call(model, (params, buffers), (support_x,))
    support_loss = F.mse_loss(support_pred, support_y)
    print(f"   Support loss: {support_loss.item():.4f}")
    
    # Support gradients
    grads = torch.autograd.grad(support_loss, tuple(params.values()), 
                               create_graph=True, allow_unused=False)
    
    # Adapted parameters  
    new_params = {}
    for (k, p), g in zip(params.items(), grads):
        new_params[k] = p - inner_lr * g
    
    print(f"   Adapted weight: {new_params['linear.weight'].item():.4f}")
    
    # Query forward pass with adapted parameters
    query_pred = functional_call(model, (new_params, buffers), (query_x,))
    query_loss = F.mse_loss(query_pred, query_y)
    
    print(f"   Query loss: {query_loss.item():.4f}")
    
    return query_loss

def test_simple_gradient():
    """Test gradients on simplest possible case."""
    print("🧪 Testing simple linear regression case...")
    
    model = SimpleLinear()
    
    # Simple data: y = 2*x (true function)
    support_x = torch.tensor([[1.0]])
    support_y = torch.tensor([2.0]) 
    query_x = torch.tensor([[2.0]])
    query_y = torch.tensor([4.0])
    
    print(f"Data: support ({support_x.item():.1f} -> {support_y.item():.1f}), query ({query_x.item():.1f} -> {query_y.item():.1f})")
    
    # Manual MAML
    query_loss = manual_maml_step(model, support_x, support_y, query_x, query_y, inner_lr=0.5)
    
    # Compute meta-gradient
    meta_grad = torch.autograd.grad(query_loss, model.parameters())[0]
    
    print(f"📊 Meta-gradient: {meta_grad.item():.4f}")
    
    # Finite difference validation
    print("\n🔢 Finite difference validation:")
    epsilon = 1e-5
    original_weight = model.linear.weight.item()
    
    # Forward finite difference
    model.linear.weight.data += epsilon
    loss_plus = manual_maml_step(model, support_x, support_y, query_x, query_y, inner_lr=0.5)
    model.linear.weight.data = torch.tensor([[original_weight]])  # Reset
    
    loss_baseline = manual_maml_step(model, support_x, support_y, query_x, query_y, inner_lr=0.5)
    
    finite_diff_grad = (loss_plus - loss_baseline) / epsilon
    print(f"   Finite diff gradient: {finite_diff_grad.item():.4f}")
    
    # Compare
    error = abs(meta_grad.item() - finite_diff_grad.item()) / (abs(finite_diff_grad.item()) + 1e-8)
    print(f"📊 Relative error: {error:.1%}")
    
    if error > 0.25:
        print("❌ GRADIENT ERROR > 25% - MAML implementation is broken!")
        return False
    else:
        print("✅ Gradients match within tolerance")
        return True

if __name__ == "__main__":
    test_simple_gradient()