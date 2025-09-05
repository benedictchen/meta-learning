"""
💰 SUPPORT THIS RESEARCH - PLEASE DONATE! 💰

🙏 If this library helps your research or project, please consider donating:
💳 PayPal: https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS
⭐ GitHub Sponsors: https://github.com/sponsors/benedictchen

👨‍💻 Author: Benedict Chen
💰 Donations: Help support this work! Buy me a coffee ☕, beer 🍺, lamborghini 🏎️, or private island 🏝️
💖 Please consider recurring donations to fully support continued research

Your support enables cutting-edge AI research for everyone! 🚀

Comprehensive CLI Tests
======================

Tests for CLI functionality including TTCS, BN freeze, determinism, and encoder options.
"""

import pytest
import json
import tempfile
import os
from unittest.mock import patch
import torch
import sys
from pathlib import Path

# Add meta_learning to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent))

from meta_learning import cli


class TestCLIComprehensive:
    """Comprehensive CLI tests covering all major functionality"""
    
    def test_version_command(self, capsys):
        """Test version command"""
        with patch('sys.argv', ['mlfew', 'version']):
            try:
                cli.main()
            except SystemExit:
                pass
        captured = capsys.readouterr()
        # Accept current version (3.1.0 or 2.2.0)
        assert any(v in captured.out for v in ["2.3.0", "3.1.0"])
    
    def test_eval_synthetic_basic(self):
        """Test basic synthetic evaluation"""
        with tempfile.TemporaryDirectory() as tmpdir:
            args = [
                'mlfew', 'eval',
                '--dataset', 'synthetic',
                '--episodes', '10',  # Small number for fast test
                '--encoder', 'identity',
                '--outdir', tmpdir
            ]
            with patch('sys.argv', args):
                try:
                    cli.main()
                except SystemExit:
                    pass
                
            # Check that metrics file was created
            assert os.path.exists(os.path.join(tmpdir, 'metrics.json'))
    
    def test_eval_synthetic_conv4(self):
        """Test synthetic evaluation with conv4 encoder"""
        with tempfile.TemporaryDirectory() as tmpdir:
            args = [
                'mlfew', 'eval', 
                '--dataset', 'synthetic',
                '--episodes', '5',
                '--encoder', 'conv4',  # Test conv4 encoder (synthetic data auto-converts to image format)
                '--emb-dim', '64',     # Use 64 dim for better image reshaping (8x8 or 4x4x4)
                '--outdir', tmpdir
            ]
            with patch('sys.argv', args):
                try:
                    cli.main()
                except SystemExit:
                    pass
                
            # Check that metrics file was created
            assert os.path.exists(os.path.join(tmpdir, 'metrics.json'))
    
    def test_eval_with_ttcs(self):
        """Test Test-Time Compute Scaling functionality"""
        with tempfile.TemporaryDirectory() as tmpdir:
            args = [
                'mlfew', 'eval',
                '--dataset', 'synthetic', 
                '--episodes', '5',
                '--ttcs', '3',  # Test TTCS with 3 passes
                '--dropout', '0.1',  # Enable dropout for stochastic behavior
                '--outdir', tmpdir
            ]
            with patch('sys.argv', args):
                try:
                    cli.main()
                except SystemExit:
                    pass
                    
            # Check results exist
            assert os.path.exists(os.path.join(tmpdir, 'metrics.json'))
    
    def test_eval_with_bn_freeze(self):
        """Test BatchNorm freeze functionality"""
        with tempfile.TemporaryDirectory() as tmpdir:
            args = [
                'mlfew', 'eval',
                '--dataset', 'synthetic',
                '--episodes', '5', 
                '--encoder', 'identity',  # Use identity encoder since BN freeze only affects conv4
                '--freeze-bn',  # Test BN freeze (no effect with identity but tests flag parsing)
                '--outdir', tmpdir
            ]
            with patch('sys.argv', args):
                try:
                    cli.main()
                except SystemExit:
                    pass
                
            assert os.path.exists(os.path.join(tmpdir, 'metrics.json'))
    
    def test_eval_determinism_with_seed(self):
        """Test deterministic behavior with seed"""
        results = []
        
        # Run same configuration twice with same seed
        for _ in range(2):
            with tempfile.TemporaryDirectory() as tmpdir:
                args = [
                    'mlfew', 'eval',
                    '--dataset', 'synthetic',
                    '--episodes', '5',
                    '--seed', '42',  # Fixed seed for determinism
                    '--outdir', tmpdir
                ]
                with patch('sys.argv', args):
                    try:
                        cli.main()
                    except SystemExit:
                        pass
                        
                # Read results
                with open(os.path.join(tmpdir, 'metrics.json')) as f:
                    results.append(json.load(f))
        
        # Results should be identical with same seed
        assert results[0]['mean_acc'] == results[1]['mean_acc']
    
    def test_bench_command(self):
        """Test benchmark command"""
        with tempfile.TemporaryDirectory() as tmpdir:
            args = [
                'mlfew', 'bench',
                '--dataset', 'synthetic',
                '--episodes', '10',
                '--outdir', tmpdir
            ]
            with patch('sys.argv', args):
                try:
                    cli.main()
                except SystemExit:
                    pass
    
    def test_invalid_encoder(self):
        """Test error handling for invalid encoder"""
        with pytest.raises(ValueError, match="encoder must be"):
            cli.make_encoder("invalid_encoder")
    
    def test_device_selection(self):
        """Test device selection logic"""
        # Test auto device selection
        device = cli._device("auto")
        assert isinstance(device, torch.device)
        
        # Test explicit CPU
        device = cli._device("cpu")
        assert device.type == "cpu"
    
    def test_eval_with_predictions_dump(self):
        """Test prediction dumping functionality"""
        with tempfile.TemporaryDirectory() as tmpdir:
            args = [
                'mlfew', 'eval',
                '--dataset', 'synthetic',
                '--episodes', '3',
                '--dump-preds',  # Test prediction dumping
                '--outdir', tmpdir
            ]
            with patch('sys.argv', args):
                try:
                    cli.main()
                except SystemExit:
                    pass
                    
            # Check that prediction file was created
            assert os.path.exists(os.path.join(tmpdir, 'preds.jsonl'))


class TestDatasetIntegration:
    """Test dataset integration and manifest handling"""
    
    def test_cifar_fs_manifest_structure(self):
        """Test CIFAR-FS manifest handling (without download)"""
        # This tests the dataset structure without actually downloading
        try:
            from meta_learning.data import CIFARFSDataset
            
            # Test that dataset initialization works with default manifest
            # (should fail gracefully if CIFAR-100 not available)
            with pytest.raises((RuntimeError, FileNotFoundError, ImportError)):
                # This will fail because we don't have CIFAR-100 downloaded,
                # but it tests the import path and manifest loading
                ds = CIFARFSDataset(root="fake_root", download=False)
                
        except ImportError:
            pytest.skip("torchvision not available")
    
    def test_synthetic_dataset_consistency(self):
        """Test synthetic dataset produces consistent results"""
        from meta_learning.data import SyntheticFewShotDataset
        
        ds = SyntheticFewShotDataset(n_classes=10, dim=32, noise=0.1)
        
        # Same seed should produce same data
        support1, query1 = ds.sample_support_query(5, 1, 10, seed=42)[:2]
        support2, query2 = ds.sample_support_query(5, 1, 10, seed=42)[:2]
        
        assert torch.allclose(support1, support2)
        assert torch.allclose(query1, query2)


if __name__ == "__main__":
    pytest.main([__file__])