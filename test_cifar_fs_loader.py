#!/usr/bin/env python3
"""
Test CIFAR-FS Dataset Loader
============================

Quick test to verify CIFAR-FS loader functionality.
"""

import sys
import os
import torch
from pathlib import Path
import tempfile

# Add the src directory to Python path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from meta_learning.meta_learning_modules.dataset_management import DatasetManager, DatasetInfo
    print("✅ Successfully imported DatasetManager")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

def test_cifar_fs_loader():
    """Test CIFAR-FS loader with synthetic fallback."""
    print("\n🧪 Testing CIFAR-FS Dataset Loader...")
    
    try:
        # Initialize dataset manager
        manager = DatasetManager(max_cache_size_gb=1.0)
        
        # Create fake dataset info for CIFAR-FS
        dataset_info = DatasetInfo(
            name="CIFAR-FS",
            urls=["https://example.com/cifar-fs.tar.gz"],
            description="CIFAR Few-Shot dataset",
            checksums=["abc123"],
            file_size=100*1024*1024,  # 100MB
            n_classes=100,
            n_samples=60000,
            image_size=(32, 32, 3)
        )
        
        # Create temporary cache directory
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir)
            
            # Test the loader (will use synthetic fallback since no real data)
            dataset = manager._load_cifar_fs(cache_dir, dataset_info)
            
            # Verify dataset structure
            assert len(dataset) > 0, "Dataset should not be empty"
            
            # Get a sample
            sample_data, sample_label = dataset[0]
            
            # Verify data format
            assert isinstance(sample_data, torch.Tensor), "Sample data should be tensor"
            assert isinstance(sample_label, torch.Tensor), "Sample label should be tensor"
            assert sample_data.shape == torch.Size([3, 32, 32]), f"Expected [3, 32, 32], got {sample_data.shape}"
            assert 0 <= sample_data.min() <= 1.0, f"Data should be normalized to [0,1], min: {sample_data.min()}"
            assert 0 <= sample_data.max() <= 1.0, f"Data should be normalized to [0,1], max: {sample_data.max()}"
            
            print(f"✅ Dataset loaded: {len(dataset)} samples")
            print(f"✅ Sample shape: {sample_data.shape}")
            print(f"✅ Label range: {sample_label.item()}")
            print(f"✅ Data range: [{sample_data.min():.3f}, {sample_data.max():.3f}]")
            
            return True
            
    except Exception as e:
        print(f"❌ CIFAR-FS loader test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_cifar_fs_loader()
    if success:
        print("🎉 CIFAR-FS loader is working!")
    else:
        print("❌ CIFAR-FS loader needs fixes")
    sys.exit(0 if success else 1)