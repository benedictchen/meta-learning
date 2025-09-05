from __future__ import annotations
from typing import Dict, Tuple
import torch, torch.nn as nn, torch.nn.functional as F
from torch.func import functional_call

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
