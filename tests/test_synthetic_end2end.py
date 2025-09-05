from meta_learning.core.seed import seed_all
from meta_learning.data import SyntheticFewShotDataset, make_episodes
from meta_learning.algos.protonet import ProtoHead
from meta_learning.eval import evaluate

def test_end_to_end_synthetic():
    seed_all(1234)
    ds = SyntheticFewShotDataset(n_classes=30, dim=16, noise=0.1)
    head = ProtoHead()
    eps = list(make_episodes(ds, 5, 1, 5, 50))
    res = evaluate(lambda e: head(e.support_x, e.support_y, e.query_x), eps, outdir=None)
    assert 0.0 <= res["mean_acc"] <= 1.0 and res["episodes"]==50
