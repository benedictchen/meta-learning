import torch
import torch.nn.functional as F
from meta_learning.algos.protonet.proto_head import ProtoHead

def test_prototypes_are_class_means():
    """Test that prototypes are computed as class means per Snell et al. 2017."""
    torch.manual_seed(0)
    z_support = torch.randn(6, 4)
    y_support = torch.tensor([0,0,0, 1,1,1])
    z_query = torch.randn(5, 4)
    
    head = ProtoHead(distance="sqeuclidean", tau=1.0)
    logits = head(z_support, y_support, z_query)
    
    # Test that a query equal to class 0 prototype gets classified as class 0
    p0 = z_support[y_support==0].mean(0)
    q0 = p0.unsqueeze(0)
    test_logits = head(z_support, y_support, q0)
    
    assert test_logits.argmax(dim=1).item() == 0

def test_cosine_normalization_and_temperature():
    """Test cosine distance normalization and temperature scaling."""
    torch.manual_seed(1)
    z_support = torch.randn(6, 8)
    y_support = torch.tensor([0,0,0, 1,1,1])
    z_query = torch.randn(5, 8)
    
    head_lo = ProtoHead(distance="cosine", tau=1.0)
    head_hi = ProtoHead(distance="cosine", tau=20.0)
    
    p_lo = F.softmax(head_lo(z_support, y_support, z_query), dim=1)
    p_hi = F.softmax(head_hi(z_support, y_support, z_query), dim=1)
    
    # Higher temperature should produce more confident predictions
    assert (p_hi.max(dim=1).values.mean() >= p_lo.max(dim=1).values.mean())

def test_squared_euclidean_distance_computation():
    """Test that squared Euclidean distance computation is mathematically correct."""
    torch.manual_seed(2)
    z_support = torch.randn(4, 3)  # 2 classes, 2 samples each
    y_support = torch.tensor([0, 0, 1, 1])
    z_query = torch.randn(2, 3)
    
    head = ProtoHead(distance="sqeuclidean", tau=1.0)
    logits = head(z_support, y_support, z_query)
    
    # Manually compute expected prototypes and distances
    proto_0 = z_support[y_support == 0].mean(0)
    proto_1 = z_support[y_support == 1].mean(0)
    
    # Check first query point distance to first prototype
    expected_dist_00 = torch.sum((z_query[0] - proto_0) ** 2)
    
    # Extract distance from logits (logits = -tau * distance)
    actual_dist_00 = -logits[0, 0] / head._tau
    
    assert torch.allclose(expected_dist_00, actual_dist_00, atol=1e-6)

def test_label_remapping():
    """Test that arbitrary support labels get remapped correctly."""
    torch.manual_seed(3)
    z_support = torch.randn(6, 4)
    y_support = torch.tensor([7, 7, 7, 23, 23, 23])  # Non-sequential labels
    z_query = torch.randn(3, 4)
    
    head = ProtoHead(distance="sqeuclidean", tau=1.0)
    logits = head(z_support, y_support, z_query)
    
    # Should produce 2 classes (remapped to 0, 1)
    assert logits.shape == (3, 2)
    
    # Test that prototype for remapped class 0 (original 7) is class mean
    proto_7 = z_support[y_support == 7].mean(0)
    test_query = proto_7.unsqueeze(0)
    test_logits = head(z_support, y_support, test_query)
    
    # Should classify as class 0 (remapped from 7)
    assert test_logits.argmax(dim=1).item() == 0