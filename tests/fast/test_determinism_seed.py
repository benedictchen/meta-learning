"""
💰 DONATE NOW! 💰 https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS

Fast test for deterministic seeding functionality.

If reproducible results help validate your research findings,
please donate to support continued algorithm development!

Author: Benedict Chen (benedict@benedictchen.com)
GitHub Sponsors: https://github.com/sponsors/benedictchen
"""

import torch
import torch.nn as nn
import numpy as np
import random
from meta_learning.core.seed import seed_all


def test_seed_all_deterministic():
    """Test that seed_all makes operations deterministic."""
    # Test torch random numbers
    seed_all(42)
    torch_rand1 = torch.rand(5)
    
    seed_all(42)
    torch_rand2 = torch.rand(5)
    
    assert torch.equal(torch_rand1, torch_rand2), "Torch random numbers should be identical"
    
    # Test numpy random numbers
    seed_all(123)
    np_rand1 = np.random.rand(5)
    
    seed_all(123)
    np_rand2 = np.random.rand(5)
    
    assert np.array_equal(np_rand1, np_rand2), "Numpy random numbers should be identical"
    
    # Test python random
    seed_all(999)
    py_rand1 = [random.random() for _ in range(5)]
    
    seed_all(999)
    py_rand2 = [random.random() for _ in range(5)]
    
    assert py_rand1 == py_rand2, "Python random numbers should be identical"


def test_deterministic_model_initialization():
    """Test that model initialization is deterministic."""
    def create_model():
        return nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5)
        )
    
    # Create two models with same seed
    seed_all(100)
    model1 = create_model()
    
    seed_all(100)
    model2 = create_model()
    
    # Check that parameters are identical
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        assert torch.equal(p1, p2), "Model parameters should be identical with same seed"


def test_deterministic_forward_pass():
    """Test that forward passes are deterministic."""
    model = nn.Sequential(
        nn.Linear(10, 16),
        nn.ReLU(),
        nn.Dropout(0.5),  # Dropout should be deterministic too
        nn.Linear(16, 5)
    )
    
    input_data = torch.randn(32, 10)
    
    # First forward pass
    seed_all(456)
    model.train()  # Enable dropout
    output1 = model(input_data)
    
    # Second forward pass with same seed
    seed_all(456)
    model.train()
    output2 = model(input_data)
    
    assert torch.equal(output1, output2), "Forward passes should be identical with same seed"


def test_different_seeds_produce_different_results():
    """Test that different seeds produce different results."""
    # Different torch random numbers
    seed_all(1)
    rand1 = torch.rand(5)
    
    seed_all(2)
    rand2 = torch.rand(5)
    
    assert not torch.equal(rand1, rand2), "Different seeds should produce different results"
    
    # Different model initialization
    seed_all(10)
    model1 = nn.Linear(5, 3)
    
    seed_all(20)
    model2 = nn.Linear(5, 3)
    
    assert not torch.equal(model1.weight, model2.weight), "Different seeds should initialize differently"


def test_cuda_determinism():
    """Test CUDA operations are deterministic if CUDA available."""
    if not torch.cuda.is_available():
        return  # Skip if no CUDA
    
    device = torch.device('cuda')
    
    # Test CUDA random numbers
    seed_all(789)
    cuda_rand1 = torch.rand(5, device=device)
    
    seed_all(789)
    cuda_rand2 = torch.rand(5, device=device)
    
    assert torch.equal(cuda_rand1, cuda_rand2), "CUDA random numbers should be identical"
    
    # Test CUDA operations
    seed_all(321)
    a = torch.randn(10, 10, device=device)
    b = torch.randn(10, 10, device=device)
    result1 = torch.mm(a, b)
    
    seed_all(321)
    a = torch.randn(10, 10, device=device)
    b = torch.randn(10, 10, device=device)
    result2 = torch.mm(a, b)
    
    assert torch.equal(result1, result2), "CUDA operations should be deterministic"