import torch
import torch.nn.functional as F

def test_matching_attention_over_supports_and_onehot_mix():
    torch.manual_seed(0)
    Ns, C, D, Nq = 6, 2, 5, 4
    z_s = torch.randn(Ns, D); y_s = torch.tensor([0,0,0,1,1,1])
    z_q = torch.randn(Nq, D)
    attn_logits = 10.0 * (F.normalize(z_q, dim=-1) @ F.normalize(z_s, dim=-1).T)  # [Nq,Ns]
    attn = F.softmax(attn_logits, dim=1)
    assert torch.allclose(attn.sum(dim=1), torch.ones(Nq))  # sums to 1 over supports
    onehot = torch.zeros(Ns, C).scatter_(1, y_s[:,None], 1)
    probs = attn @ onehot
    assert probs.shape == (Nq, C)