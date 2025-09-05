import torch, torch.nn.functional as F
from meta_learning.core.episode import Episode, remap_labels
from meta_learning.algos.protonet import ProtoHead

def test_episode_validate_and_remap():
    ys = torch.tensor([10,10,10,42,42,42]); yq = torch.tensor([10,42,42,10])
    ys_m, yq_m = remap_labels(ys, yq)
    ep = Episode(torch.randn(6,5), ys_m, torch.randn(4,5), yq_m)
    ep.validate(expect_n_classes=2)

def test_protonet_invariants():
    torch.manual_seed(0)
    z_support = torch.randn(6,4); y_support = torch.tensor([0,0,0,1,1,1])
    p0 = z_support[y_support==0].mean(0); q0 = p0.unsqueeze(0)
    head = ProtoHead("sqeuclidean", tau=1.0)
    logits = head(z_support, y_support, q0)
    assert logits.argmax(1).item()==0
    z_query = torch.randn(5,4)
    lo = ProtoHead("cosine", tau=1.0); hi = ProtoHead("cosine", tau=20.0)
    m_lo = F.softmax(lo(z_support, y_support, z_query),1).max(1).values.mean()
    m_hi = F.softmax(hi(z_support, y_support, z_query),1).max(1).values.mean()
    assert m_hi >= m_lo
