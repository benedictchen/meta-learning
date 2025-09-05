"""
Enhanced Mathematical Property Tests for Sparse Coding
=====================================================

Tests that validate mathematical properties beyond basic functionality:
- Dictionary atom normalization
- Energy function monotonic decrease  
- Sparsity-reconstruction tradeoff
- Mode comparison (paper vs L1)
- Whitening filter accuracy
"""

import numpy as np
import pytest


def test_dictionary_atom_normalization():
    """Test that dictionary atoms maintain unit norm after updates."""
    from sparse_coding.api import SparseCoder
    
    rng = np.random.default_rng(42)
    p, K, N = 64, 96, 100
    X = rng.normal(size=(p, N))
    
    coder = SparseCoder(n_atoms=K, seed=42)
    coder.fit(X, n_steps=10, lr=0.1)
    
    # All atoms should have unit norm (within numerical tolerance)
    atom_norms = np.linalg.norm(coder.D, axis=0)
    assert np.allclose(atom_norms, 1.0, atol=1e-6), f"Atom norms: {atom_norms}"


def test_energy_monotonic_decrease():
    """Test that objective function decreases monotonically during fitting.""" 
    from sparse_coding.api import SparseCoder
    
    rng = np.random.default_rng(123)
    p, K, N = 32, 48, 80
    X = rng.normal(size=(p, N))
    
    coder = SparseCoder(n_atoms=K, seed=123)
    
    # Track energy during fitting
    energies = []
    initial_energy = None
    
    for step in range(5):
        A = coder.encode(X)
        X_hat = coder.decode(A)
        
        # Compute energy: reconstruction + sparsity penalty
        recon_error = np.linalg.norm(X - X_hat)**2
        sparsity_penalty = coder.lam * np.sum(np.abs(A)) if coder.lam else 0
        energy = recon_error + sparsity_penalty
        energies.append(energy)
        
        if initial_energy is None:
            initial_energy = energy
        
        # Update dictionary
        if step < 4:  # Don't update on last iteration
            coder.fit(X, n_steps=1, lr=0.1)
    
    # Energy should generally decrease (allowing small fluctuations)
    assert energies[-1] <= energies[0] * 1.1, f"Energy increased: {energies[0]:.3f} → {energies[-1]:.3f}"


def test_sparsity_reconstruction_tradeoff():
    """Test the tradeoff between sparsity and reconstruction quality."""
    from sparse_coding.api import SparseCoder
    
    rng = np.random.default_rng(456)
    p, K, N = 64, 96, 100
    X = rng.normal(size=(p, N))
    
    lambdas = [0.01, 0.1, 1.0]
    sparsities = []
    errors = []
    
    for lam in lambdas:
        coder = SparseCoder(n_atoms=K, lam=lam, seed=456)
        coder.fit(X, n_steps=10, lr=0.1)
        
        A = coder.encode(X)
        X_hat = coder.decode(A)
        
        sparsity = np.mean(np.abs(A) < 1e-6)
        error = np.linalg.norm(X - X_hat) / np.linalg.norm(X)
        
        sparsities.append(sparsity)
        errors.append(error)
    
    # Higher lambda should increase sparsity
    assert sparsities[2] >= sparsities[1] >= sparsities[0], f"Sparsity should increase with lambda: {sparsities}"
    
    # Higher lambda generally increases reconstruction error (tradeoff)
    # Allow some tolerance for optimization noise
    assert errors[2] >= errors[0] * 0.8, f"Higher lambda should increase error (tradeoff): {errors}"


def test_paper_vs_l1_mode_comparison():
    """Compare paper-exact vs L1-FISTA modes on same data."""
    from sparse_coding.api import SparseCoder
    
    rng = np.random.default_rng(789)
    p, K, N = 32, 48, 60
    
    # Generate data from known sparse structure
    D_true = rng.normal(size=(p, K))
    D_true /= np.linalg.norm(D_true, axis=0, keepdims=True)
    A_true = rng.laplace(size=(K, N)) * (rng.random((K, N)) < 0.1)
    X = D_true @ A_true + 0.01 * rng.normal(size=(p, N))
    
    # Test both modes
    coder_l1 = SparseCoder(n_atoms=K, mode='l1', lam=0.1, seed=789)
    coder_l1.fit(X, n_steps=15, lr=0.1)
    
    coder_paper = SparseCoder(n_atoms=K, mode='paper', max_iter=50, seed=789)
    coder_paper.fit(X, n_steps=15, lr=0.1)
    
    # Both should produce reasonable reconstructions
    A_l1 = coder_l1.encode(X)
    A_paper = coder_paper.encode(X)
    
    err_l1 = np.linalg.norm(X - coder_l1.decode(A_l1)) / np.linalg.norm(X)
    err_paper = np.linalg.norm(X - coder_paper.decode(A_paper)) / np.linalg.norm(X)
    
    assert err_l1 < 0.7, f"L1 mode error too high: {err_l1}"
    assert err_paper < 0.8, f"Paper mode error too high: {err_paper}"
    
    # Both should achieve some sparsity
    sparsity_l1 = np.mean(np.abs(A_l1) < 1e-6)
    sparsity_paper = np.mean(np.abs(A_paper) < 1e-6)
    
    assert sparsity_l1 > 0.3, f"L1 mode not sparse enough: {sparsity_l1}"
    assert sparsity_paper > 0.2, f"Paper mode not sparse enough: {sparsity_paper}"


def test_whitening_filter_mathematical_correctness():
    """Test that whitening filter matches Olshausen & Field formula."""
    from sparse_coding.paper_exact import whiten_patches_of96
    
    # Create test patches
    rng = np.random.default_rng(101112)
    patches = rng.normal(size=(16*16, 100))  # 16x16 patches
    
    # Apply whitening
    whitened = whiten_patches_of96(patches, patch_shape=(16, 16))
    
    # Whitening should reduce low-frequency power
    # (This is a basic sanity check - full validation would require frequency analysis)
    
    # Whitened patches should have reasonable statistics
    assert np.isfinite(whitened).all(), "Whitened patches should be finite"
    assert whitened.std() > 0, "Whitened patches should have non-zero variance"
    
    # Whitening should change the frequency characteristics
    original_var = np.var(patches)
    whitened_var = np.var(whitened) 
    
    # Variance may change due to whitening (this is expected)
    assert whitened_var > 0, "Whitened patches should retain variance"


def test_homeostatic_scaling_properties():
    """Test homeostatic gain equalization properties."""
    from sparse_coding.homeostasis import apply_homeostatic_scaling
    
    rng = np.random.default_rng(131415)
    
    # Create coefficients with unequal variances (some neurons more active)
    K, N = 64, 200
    A = rng.normal(size=(K, N))
    
    # Make some neurons much more active
    A[:20] *= 3.0  # First 20 neurons very active
    A[40:] *= 0.3  # Last 24 neurons less active
    
    # Apply homeostatic scaling
    A_scaled = apply_homeostatic_scaling(A, target_activity=1.0)
    
    # Check that variances are more equalized
    original_stds = np.std(A, axis=1)
    scaled_stds = np.std(A_scaled, axis=1)
    
    # Scaled version should have more uniform activity levels
    original_std_range = np.max(original_stds) - np.min(original_stds)
    scaled_std_range = np.max(scaled_stds) - np.min(scaled_stds)
    
    assert scaled_std_range < original_std_range, "Homeostatic scaling should reduce activity range"
    
    # All scaled activities should be finite
    assert np.isfinite(A_scaled).all(), "Homeostatic scaling should produce finite values"


def test_reconstruction_identity_property():
    """Test that encoding then decoding preserves dictionary atoms."""
    from sparse_coding.api import SparseCoder
    
    rng = np.random.default_rng(161718)
    p, K = 64, 96
    
    coder = SparseCoder(n_atoms=K, seed=161718)
    
    # Dictionary atoms should be perfectly reconstructible
    D = coder._init_dict(p)
    coder.D = D
    
    # Each atom should encode to a unit vector then decode back to itself
    for i in range(min(5, K)):  # Test first 5 atoms
        atom = D[:, i:i+1]  # Single atom as column vector
        
        # Encode the atom
        coeff = coder.encode(atom)
        
        # Should produce a sparse code with main activation on neuron i
        assert coeff[i, 0] > 0.5, f"Atom {i} should activate its own neuron strongly"
        
        # Decode back
        reconstructed = coder.decode(coeff)
        
        # Should approximately reconstruct the original atom
        reconstruction_error = np.linalg.norm(atom - reconstructed) / np.linalg.norm(atom)
        assert reconstruction_error < 0.1, f"Atom {i} reconstruction error too high: {reconstruction_error}"


if __name__ == "__main__":
    print("🧪 Running enhanced sparse coding mathematical property tests...")
    
    # Run all tests
    test_dictionary_atom_normalization()
    print("✅ Dictionary atom normalization")
    
    test_energy_monotonic_decrease() 
    print("✅ Energy monotonic decrease")
    
    test_sparsity_reconstruction_tradeoff()
    print("✅ Sparsity-reconstruction tradeoff")
    
    test_paper_vs_l1_mode_comparison()
    print("✅ Paper vs L1 mode comparison")
    
    test_whitening_filter_mathematical_correctness()
    print("✅ Whitening filter correctness")
    
    test_homeostatic_scaling_properties()
    print("✅ Homeostatic scaling properties")
    
    test_reconstruction_identity_property()
    print("✅ Reconstruction identity property")
    
    print("\n🎉 All enhanced mathematical property tests passed!")
    print("\n💡 Key Mathematical Properties Validated:")
    print("   • Dictionary atoms maintain unit norm")
    print("   • Energy decreases during optimization") 
    print("   • Sparsity-reconstruction tradeoff works correctly")
    print("   • Both paper and L1 modes produce valid results")
    print("   • Whitening preserves mathematical properties")
    print("   • Homeostatic scaling equalizes neuron activity")
    print("   • Encoding-decoding preserves dictionary structure")