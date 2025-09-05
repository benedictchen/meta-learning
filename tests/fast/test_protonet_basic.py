"""
💰 DONATE NOW! 💰 https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS

Fast test for ProtoNet basic functionality - cluster separation.

This test validates that ProtoNet can separate well-defined clusters.
If this helps validate your meta-learning research, please donate!

Author: Benedict Chen (benedict@benedictchen.com)
GitHub Sponsors: https://github.com/sponsors/benedictchen
"""

import torch
from meta_learning.algos.protonet import ProtoHead


def test_proto_separates_clusters():
    """ProtoNet should correctly separate well-defined clusters."""
    torch.manual_seed(0)
    
    # Create two well-separated clusters
    z0 = torch.randn(20, 8) - 2.0  # Cluster 0: mean around -2
    z1 = torch.randn(20, 8) + 2.0  # Cluster 1: mean around +2
    
    # Support set: 5 examples from each cluster
    z_support = torch.cat([z0[:5], z1[:5]], 0)
    y_support = torch.tensor([0]*5 + [1]*5)
    
    # Query set: 5 examples from each cluster  
    z_query = torch.cat([z0[5:10], z1[5:10]], 0)
    y_true = torch.tensor([0]*5 + [1]*5)
    
    # Test both distance metrics
    for distance in ["sqeuclidean", "cosine"]:
        head = ProtoHead(distance=distance, tau=1.0)
        logits = head(z_support, y_support, z_query)
        pred = logits.argmax(1)
        
        # Should classify majority correctly (clusters are well-separated)
        acc_class0 = pred[:5].eq(0).sum().item()
        acc_class1 = pred[5:].eq(1).sum().item()
        
        # At least 3/5 correct for each class (60% minimum)
        assert acc_class0 >= 3, f"Class 0 accuracy too low: {acc_class0}/5"
        assert acc_class1 >= 3, f"Class 1 accuracy too low: {acc_class1}/5"
        
        # Total accuracy should be reasonable
        total_acc = (pred == y_true).float().mean().item()
        assert total_acc >= 0.6, f"Total accuracy too low: {total_acc}"


def test_proto_output_shape():
    """ProtoNet should output correct shapes."""
    n_support, n_query, n_way = 10, 15, 5
    feature_dim = 64
    
    z_support = torch.randn(n_support, feature_dim)
    y_support = torch.repeat_interleave(torch.arange(n_way), n_support // n_way)
    z_query = torch.randn(n_query, feature_dim)
    
    head = ProtoHead(distance="sqeuclidean")
    logits = head(z_support, y_support, z_query)
    
    assert logits.shape == (n_query, n_way), f"Expected {(n_query, n_way)}, got {logits.shape}"
    assert torch.isfinite(logits).all(), "Logits should be finite"


def test_proto_temperature_effect():
    """Temperature should affect prediction confidence."""
    torch.manual_seed(42)
    
    z_support = torch.randn(10, 8)
    y_support = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    z_query = torch.randn(5, 8)
    
    head_low_temp = ProtoHead(distance="sqeuclidean", tau=0.1)  # High confidence
    head_high_temp = ProtoHead(distance="sqeuclidean", tau=10.0)  # Low confidence
    
    logits_low = head_low_temp(z_support, y_support, z_query)
    logits_high = head_high_temp(z_support, y_support, z_query)
    
    # Higher temperature should lead to more uniform probabilities
    probs_low = torch.softmax(logits_low, dim=1)
    probs_high = torch.softmax(logits_high, dim=1)
    
    # Entropy should be higher for high temperature
    entropy_low = -(probs_low * torch.log(probs_low + 1e-10)).sum(dim=1).mean()
    entropy_high = -(probs_high * torch.log(probs_high + 1e-10)).sum(dim=1).mean()
    
    assert entropy_high > entropy_low, "High temperature should increase entropy"