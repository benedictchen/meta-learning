import torch
import torch.nn as nn
import torch.nn.functional as F

class Tiny(nn.Module):
    def __init__(self, D=5, C=3):
        super().__init__()
        self.linear = nn.Linear(D, C)
    def forward(self, x):
        return self.linear(x)

def inner_adapt_and_eval(model, loss_fn, support, query, inner_lr=0.4, first_order=False):
    (x_s, y_s), (x_q, y_q) = support, query
    loss_s = loss_fn(model(x_s), y_s)
    grads = torch.autograd.grad(loss_s, tuple(model.parameters()), create_graph=not first_order)
    adapted = [p - inner_lr * g for p,g in zip(model.parameters(), grads)]
    # minimal functional forward
    W, b = adapted[0], adapted[1]
    logits_q = F.linear(x_q, W, b)
    return loss_fn(logits_q, y_q)

def test_second_order_path_and_no_inplace():
    torch.manual_seed(0)
    model = Tiny()
    loss_fn = nn.CrossEntropyLoss()
    x_s = torch.randn(6, 5); y_s = torch.randint(0, 3, (6,))
    x_q = torch.randn(4, 5); y_q = torch.randint(0, 3, (4,))
    loss_q = inner_adapt_and_eval(model, loss_fn, (x_s,y_s), (x_q,y_q), inner_lr=0.5, first_order=False)
    loss_q.backward()
    assert all(p.grad is not None for p in model.parameters())

def test_first_order_path_exists_but_no_second_order_graph():
    torch.manual_seed(1)
    model = Tiny()
    loss_fn = nn.CrossEntropyLoss()
    x_s = torch.randn(6, 5); y_s = torch.randint(0, 3, (6,))
    x_q = torch.randn(4, 5); y_q = torch.randint(0, 3, (4,))
    loss_q = inner_adapt_and_eval(model, loss_fn, (x_s,y_s), (x_q,y_q), inner_lr=0.5, first_order=True)
    loss_q.backward()
    assert all(p.grad is not None for p in model.parameters())