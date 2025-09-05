from __future__ import annotations
import torch, torch.nn as nn, torch.nn.functional as F
from typing import Tuple

def inner_adapt_and_eval(model: nn.Module, loss_fn, support: Tuple[torch.Tensor, torch.Tensor],
                         query: Tuple[torch.Tensor, torch.Tensor], inner_lr: float = 0.4, first_order: bool = False):
    (x_s, y_s), (x_q, y_q) = support, query
    loss_s = loss_fn(model(x_s), y_s)
    grads = torch.autograd.grad(loss_s, tuple(model.parameters()), create_graph=not first_order)
    adapted = [p - inner_lr * g for p,g in zip(model.parameters(), grads)]
    idx = 0
    def fwd(x):
        nonlocal idx; idx = 0; out = x
        for m in model.modules():
            if isinstance(m, nn.Linear):
                W = adapted[idx]; b = adapted[idx+1]; idx += 2
                out = F.linear(out, W, b)
            elif isinstance(m, nn.Conv2d):
                W = adapted[idx]; b = adapted[idx+1]; idx += 2
                out = F.conv2d(out, W, b, stride=m.stride, padding=m.padding)
            elif isinstance(m, nn.ReLU):
                out = F.relu(out, inplace=False)
        return out
    logits_q = fwd(x_q)
    return loss_fn(logits_q, y_q)

def meta_outer_step(model: nn.Module, loss_fn, meta_batch, inner_lr=0.4, first_order=False, optimizer=None):
    meta_losses = []
    for task in meta_batch:
        loss_q = inner_adapt_and_eval(model, loss_fn, task['support'], task['query'], inner_lr, first_order)
        meta_losses.append(loss_q)
    meta_loss = torch.stack(meta_losses).mean()
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True); meta_loss.backward(); optimizer.step()
    return meta_loss
