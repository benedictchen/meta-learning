"""
Basic functionality tests for the meta-learning package.
Tests core imports and basic operations.
"""

import pytest
import torch
import meta_learning as ml


def test_package_imports():
    """Test that the package imports correctly."""
    assert hasattr(ml, '__version__')
    assert ml.__version__ == "2.3.0"


def test_conv4_creation():
    """Test Conv4 backbone creation."""
    model = ml.Conv4(out_dim=64, p_drop=0.0)
    assert isinstance(model, torch.nn.Module)
    
    # Test forward pass
    x = torch.randn(2, 3, 84, 84)  # Batch of 2 RGB images
    features = model(x)
    assert features.shape == (2, 64)


def test_synthetic_dataset():
    """Test synthetic dataset creation."""
    from meta_learning.data import SyntheticFewShotDataset
    
    dataset = SyntheticFewShotDataset(n_classes=20, dim=32, image_mode=True)
    
    # Test episode sampling
    xs, ys, xq, yq = dataset.sample_support_query(n_way=5, k_shot=1, m_query=5)
    assert xs.shape == (5, 3, 16, 16)  # 5-way 1-shot in image format (calculated size)
    assert ys.shape == (5,)
    assert xq.shape == (25, 3, 16, 16)  # 5 queries per class  
    assert yq.shape == (25,)


def test_episode_creation():
    """Test episode sampling."""
    from meta_learning.data import SyntheticFewShotDataset, make_episodes
    
    dataset = SyntheticFewShotDataset(n_classes=20, dim=32, image_mode=True)
    
    episodes = list(make_episodes(
        dataset=dataset,
        n_way=5,
        k_shot=1,
        m_query=5,
        episodes=2
    ))
    
    assert len(episodes) == 2
    
    episode = episodes[0]
    assert episode.support_x.shape == (5, 3, 16, 16)   # 5-way 1-shot in image format
    assert episode.support_y.shape == (5,)
    assert episode.query_x.shape == (25, 3, 16, 16)    # 5 queries per class
    assert episode.query_y.shape == (25,)


def test_protohead_forward():
    """Test ProtoHead forward pass."""
    from meta_learning.algos.protonet import ProtoHead
    
    # Create model (ProtoHead doesn't take encoder, it works on features)
    model = ProtoHead(distance="sqeuclidean")
    
    # Create encoder separately 
    encoder = ml.Conv4(out_dim=64)
    
    # Create simple episode with features
    support_x = torch.randn(5, 3, 84, 84)  # 5-way 1-shot images
    support_y = torch.tensor([0, 1, 2, 3, 4])
    query_x = torch.randn(10, 3, 84, 84)
    
    # Extract features
    z_support = encoder(support_x)  # (5, 64)
    z_query = encoder(query_x)      # (10, 64)
    
    # Forward pass through ProtoHead
    logits = model(z_support, support_y, z_query)
    assert logits.shape == (10, 5)  # 10 queries, 5 classes
    
    # Check that probabilities sum to 1
    probs = torch.softmax(logits, dim=1)
    assert torch.allclose(probs.sum(dim=1), torch.ones(10), atol=1e-6)


def test_cli_imports():
    """Test that CLI module imports successfully."""
    import meta_learning.cli
    assert hasattr(meta_learning.cli, 'main')


if __name__ == "__main__":
    pytest.main([__file__])