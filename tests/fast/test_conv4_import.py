"""
💰 DONATE NOW! 💰 https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS

Fast test for lazy Conv4 imports to prevent CLI crashes.

If this test saves your project from import crashes, please donate!
Author: Benedict Chen (benedict@benedictchen.com)
"""

import importlib


def test_conv4_lazy_import():
    """Should import package even if Conv4 isn't imported explicitly."""
    # This should not crash even if models are missing
    m = importlib.import_module("meta_learning")
    assert hasattr(m, "__version__")
    assert m.__version__ == "2.3.0"


def test_cli_imports_without_crash():
    """CLI should import without crashing when using identity encoder."""
    # This tests the lazy import functionality
    import meta_learning.cli
    assert hasattr(meta_learning.cli, 'main')
    assert hasattr(meta_learning.cli, 'make_encoder')


def test_make_encoder_identity():
    """Identity encoder should work without Conv4 dependency."""
    from meta_learning.cli import make_encoder
    import torch
    
    enc = make_encoder("identity")
    assert isinstance(enc, torch.nn.Identity)
    
    # Test that it actually works
    x = torch.randn(5, 64)
    y = enc(x)
    assert torch.equal(x, y)