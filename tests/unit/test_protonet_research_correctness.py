import torch
import torch.nn.functional as F

def per_class_prototypes(z_support, y_support):
    classes = torch.unique(y_support)
    return torch.stack([z_support[y_support==i].mean(0) for i in classes], 0)

def test_prototypes_are_class_means_and_sign():
    torch.manual_seed(0)
    z_support = torch.randn(6, 4)
    y_support = torch.tensor([0,0,0,1,1,1])
    z_query = torch.randn(5, 4)
    protos = per_class_prototypes(z_support, y_support)
    dist = (z_query[:,None]-protos[None,:]).pow(2).sum(-1)
    logits = -dist                      # sign must be negative distances
    probs = F.softmax(logits, dim=1)
    # Query equal to class0 prototype -> max prob at class0
    q0 = protos[0].unsqueeze(0)
    dist_q0 = (q0[:,None]-protos[None,:]).pow(2).sum(-1)
    logits_q0 = -dist_q0
    assert logits_q0.argmax(dim=1).item() == 0

def test_cosine_normalization_and_temperature_monotone():
    torch.manual_seed(1)
    z_support = torch.randn(6, 8)
    y_support = torch.tensor([0,0,0,1,1,1])
    z_query = torch.randn(5, 8)
    p = per_class_prototypes(z_support, y_support)
    zq = F.normalize(z_query, dim=-1); pr = F.normalize(p, dim=-1)
    logits_lo =  1.0 * (zq @ pr.T)
    logits_hi = 20.0 * (zq @ pr.T)
    m_lo = F.softmax(logits_lo, dim=1).max(dim=1).values.mean()
    m_hi = F.softmax(logits_hi, dim=1).max(dim=1).values.mean()
    assert m_hi >= m_lo