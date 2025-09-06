#!/usr/bin/env python3
"""
Test Lightning Integration
=========================

Quick test to verify Lightning integration functionality.
"""

import sys
import os
import torch
import torch.nn as nn

# Add the src directory to Python path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from meta_learning.frameworks.lightning_integration import MetaLearningLightningModule
    from meta_learning.shared.types import Episode
    print("✅ Successfully imported Lightning integration")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

def test_lightning_module():
    """Test Lightning module basic functionality."""
    print("\n🧪 Testing Lightning Module...")
    
    try:
        # Create simple model
        model = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 5)  # 5-way classification
        )
        
        # Initialize Lightning module
        lightning_module = MetaLearningLightningModule(
            model=model,
            algorithm="maml",
            lr=0.01,
            meta_lr=0.001,
            num_inner_steps=3
        )
        
        # Test forward pass
        test_input = torch.randn(32, 10)
        output = lightning_module(test_input)
        assert output.shape == (32, 5), f"Expected shape (32, 5), got {output.shape}"
        print("✅ Forward pass working")
        
        # Create synthetic episode
        support_x = torch.randn(25, 10)  # 5-way, 5-shot
        support_y = torch.repeat_interleave(torch.arange(5), 5)
        query_x = torch.randn(15, 10)    # 3 queries per class
        query_y = torch.repeat_interleave(torch.arange(5), 3)
        
        episode = Episode(support_x, support_y, query_x, query_y)
        
        # Test training step
        loss = lightning_module.training_step(episode, 0)
        assert isinstance(loss, torch.Tensor), "Training step should return tensor"
        assert loss.requires_grad, "Loss should require gradients"
        print("✅ Training step working")
        
        # Test validation step
        val_loss = lightning_module.validation_step(episode, 0)
        assert isinstance(val_loss, torch.Tensor), "Validation step should return tensor"
        print("✅ Validation step working")
        
        # Test optimizer configuration
        opt_config = lightning_module.configure_optimizers()
        assert "optimizer" in opt_config, "Should return optimizer"
        assert "lr_scheduler" in opt_config, "Should return scheduler"
        print("✅ Optimizer configuration working")
        
        # Test algorithm-specific methods
        prototypes = lightning_module._compute_prototypes(support_x, support_y)
        assert prototypes.shape == (5, 5), f"Expected prototypes shape (5, 5), got {prototypes.shape}"
        print("✅ Prototype computation working")
        
        query_logits = lightning_module._classify_queries(query_x, prototypes)
        assert query_logits.shape == (15, 5), f"Expected logits shape (15, 5), got {query_logits.shape}"
        print("✅ Query classification working")
        
        return True
        
    except Exception as e:
        print(f"❌ Lightning integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_lightning_module()
    if success:
        print("🎉 Lightning integration is working!")
    else:
        print("❌ Lightning integration needs fixes")
    sys.exit(0 if success else 1)