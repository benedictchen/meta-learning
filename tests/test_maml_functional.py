import torch, torch.nn as nn, torch.nn.functional as F
from meta_learning.algos.maml import inner_adapt_and_eval

class Tiny(nn.Module):
    def __init__(self): super().__init__(); self.fc = nn.Linear(4, 3)
    def forward(self, x): return self.fc(x)

def test_maml_second_order_runs():
    torch.manual_seed(0)
    mdl = Tiny()
    xs = torch.randn(6,4); ys = torch.tensor([0,0,0,1,1,2])
    xq = torch.randn(5,4); yq = torch.tensor([0,1,1,2,2])
    loss = inner_adapt_and_eval(mdl, F.cross_entropy, (xs, ys), (xq, yq), inner_lr=0.1, first_order=False)
    assert loss.requires_grad
