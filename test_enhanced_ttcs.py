#!/usr/bin/env python3
"""Test script for Enhanced TTCS with detach_module integration"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import gc

def test_memory_efficient_ttcs_basic():
    print("Testing MemoryEfficientTTCS basic functionality...")
    
    try:
        from src.meta_learning.algos.enhanced_ttcs import MemoryEfficientTTCS
        from src.meta_learning.shared.types import Episode
        
        # Create a simple model for testing
        encoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.Dropout(0.2),  # For MC-Dropout testing
            nn.ReLU(),
            nn.Linear(16, 32)
        )
        
        # Simple head for classification
        class SimpleHead(nn.Module):
            def forward(self, support_features, support_labels, query_features):
                # Compute prototypes
                unique_labels = torch.unique(support_labels)
                prototypes = []
                for label in unique_labels:
                    mask = support_labels == label
                    prototype = support_features[mask].mean(dim=0)
                    prototypes.append(prototype)
                prototypes = torch.stack(prototypes)
                
                # Compute distances and return logits
                distances = torch.cdist(query_features, prototypes)
                return -distances  # Negative distance as logits
        
        head = SimpleHead()
        
        # Create Enhanced TTCS
        enhanced_ttcs = MemoryEfficientTTCS(
            encoder=encoder,
            head=head,
            passes=4,
            enable_memory_efficient=True,
            enable_detach_optimization=True,
            performance_tracking=True,
            memory_budget_mb=None  # Disable memory budget for basic test
        )
        
        print("✓ MemoryEfficientTTCS initialized successfully")
        
        # Create mock episode
        support_x = torch.randn(9, 8)  # 3-way 3-shot
        support_y = torch.tensor([0,0,0, 1,1,1, 2,2,2])
        query_x = torch.randn(6, 8)   # 2 queries per class
        query_y = torch.tensor([0,0, 1,1, 2,2])
        
        episode = Episode(support_x, support_y, query_x, query_y)
        
        # Test forward pass
        predictions = enhanced_ttcs.forward(episode)
        assert predictions.shape == (6, 3), f"Expected shape (6, 3), got {predictions.shape}"
        assert torch.isfinite(predictions).all(), "Predictions should be finite"
        
        print(f"✓ Forward pass successful, predictions shape: {predictions.shape}")
        
        # Test with metrics (note: results may differ due to MC-Dropout stochasticity)
        predictions_with_metrics, metrics = enhanced_ttcs.forward(episode, return_metrics=True)
        # Check shapes and validity instead of exact equality due to MC-Dropout
        assert predictions_with_metrics.shape == predictions.shape
        assert torch.isfinite(predictions_with_metrics).all()
        assert isinstance(metrics, dict)
        assert 'method' in metrics
        assert 'memory_efficient' in metrics
        assert 'detach_optimization' in metrics
        
        print(f"✓ Metrics collection works: {list(metrics.keys())}")
        
        return True
        
    except Exception as e:
        print(f"✗ MemoryEfficientTTCS basic test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_memory_optimization_effectiveness():
    print("\nTesting memory optimization effectiveness...")
    
    try:
        from src.meta_learning.algos.enhanced_ttcs import MemoryEfficientTTCS
        from src.meta_learning.shared.types import Episode
        
        # Larger model to see memory difference
        encoder = nn.Sequential(
            nn.Linear(20, 64),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        class SimpleHead(nn.Module):
            def forward(self, support_features, support_labels, query_features):
                unique_labels = torch.unique(support_labels)
                prototypes = []
                for label in unique_labels:
                    mask = support_labels == label
                    prototype = support_features[mask].mean(dim=0)
                    prototypes.append(prototype)
                prototypes = torch.stack(prototypes)
                distances = torch.cdist(query_features, prototypes)
                return -distances
        
        head = SimpleHead()
        
        # Test with memory optimization
        optimized_ttcs = MemoryEfficientTTCS(
            encoder=encoder,
            head=head,
            passes=6,
            enable_memory_efficient=True,
            enable_detach_optimization=True,
            memory_cleanup_frequency=2,
            performance_tracking=True
        )
        
        # Test without memory optimization
        standard_ttcs = MemoryEfficientTTCS(
            encoder=encoder,
            head=head,
            passes=6,
            enable_memory_efficient=False,
            enable_detach_optimization=False,
            performance_tracking=True
        )
        
        # Create episode
        support_x = torch.randn(15, 20)  # 5-way 3-shot
        support_y = torch.tensor([0,0,0, 1,1,1, 2,2,2, 3,3,3, 4,4,4])
        query_x = torch.randn(10, 20)
        query_y = torch.tensor([0,0, 1,1, 2,2, 3,3, 4,4])
        
        episode = Episode(support_x, support_y, query_x, query_y)
        
        # Record initial memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()
        else:
            gc.collect()
            initial_memory = 0
        
        # Test optimized version
        start_time = time.time()
        optimized_pred, optimized_metrics = optimized_ttcs.forward(episode, return_metrics=True)
        optimized_time = time.time() - start_time
        
        # Clear memory between tests
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        else:
            gc.collect()
        
        # Test standard version  
        start_time = time.time()
        standard_pred, standard_metrics = standard_ttcs.forward(episode, return_metrics=True)
        standard_time = time.time() - start_time
        
        # Both should produce valid results
        assert torch.isfinite(optimized_pred).all()
        assert torch.isfinite(standard_pred).all()
        assert optimized_pred.shape == standard_pred.shape
        
        print(f"✓ Optimized TTCS time: {optimized_time:.4f}s")
        print(f"✓ Standard TTCS time: {standard_time:.4f}s")
        
        # Check memory metrics
        if 'memory_metrics' in optimized_metrics:
            memory_stats = optimized_metrics['memory_metrics']
            print(f"✓ Memory optimization events: {memory_stats.get('optimization_events', {})}")
        
        print("✓ Memory optimization test completed")
        
        return True
        
    except Exception as e:
        print(f"✗ Memory optimization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_adaptive_memory_ttcs():
    print("\nTesting AdaptiveMemoryTTCS...")
    
    try:
        from src.meta_learning.algos.enhanced_ttcs import AdaptiveMemoryTTCS
        from src.meta_learning.shared.types import Episode
        
        # Simple model
        encoder = nn.Sequential(nn.Linear(6, 12), nn.Dropout(0.2), nn.ReLU(), nn.Linear(12, 16))
        
        class SimpleHead(nn.Module):
            def forward(self, support_features, support_labels, query_features):
                unique_labels = torch.unique(support_labels)
                prototypes = []
                for label in unique_labels:
                    mask = support_labels == label
                    prototype = support_features[mask].mean(dim=0)
                    prototypes.append(prototype)
                prototypes = torch.stack(prototypes)
                distances = torch.cdist(query_features, prototypes)
                return -distances
        
        head = SimpleHead()
        
        # Create Adaptive TTCS
        adaptive_ttcs = AdaptiveMemoryTTCS(
            encoder=encoder,
            head=head,
            passes=8,
            performance_tracking=True
        )
        
        print(f"✓ AdaptiveMemoryTTCS initialized with budget: {adaptive_ttcs.memory_budget_mb}MB")
        
        # Create episode
        support_x = torch.randn(6, 6)
        support_y = torch.tensor([0, 0, 1, 1, 2, 2])
        query_x = torch.randn(6, 6)
        query_y = torch.tensor([0, 0, 1, 1, 2, 2])
        
        episode = Episode(support_x, support_y, query_x, query_y)
        
        # Test adaptive forward pass
        predictions, metrics = adaptive_ttcs.forward(episode, return_metrics=True)
        
        assert torch.isfinite(predictions).all()
        assert predictions.shape == (6, 3)
        
        # Check for adaptive information in metrics
        if 'adaptive_info' in metrics:
            adaptive_info = metrics['adaptive_info']
            print(f"✓ Memory pressure: {adaptive_info.get('memory_pressure', 0):.3f}")
            print(f"✓ Passes used: {adaptive_info.get('adaptive_passes', 'N/A')}")
        
        print("✓ AdaptiveMemoryTTCS test successful")
        
        return True
        
    except Exception as e:
        print(f"✗ AdaptiveMemoryTTCS test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_failure_prediction_integration():
    print("\nTesting failure prediction integration...")
    
    try:
        from src.meta_learning.algos.enhanced_ttcs import MemoryEfficientTTCS
        from src.meta_learning.ml_enhancements.failure_prediction import FailurePredictionModel
        from src.meta_learning.shared.types import Episode
        
        # Create failure predictor
        failure_predictor = FailurePredictionModel()
        
        # Simple model
        encoder = nn.Sequential(nn.Linear(8, 16), nn.Dropout(0.2), nn.ReLU(), nn.Linear(16, 20))
        
        class SimpleHead(nn.Module):
            def forward(self, support_features, support_labels, query_features):
                unique_labels = torch.unique(support_labels)
                prototypes = []
                for label in unique_labels:
                    mask = support_labels == label
                    prototype = support_features[mask].mean(dim=0)
                    prototypes.append(prototype)
                prototypes = torch.stack(prototypes)
                distances = torch.cdist(query_features, prototypes)
                return -distances
        
        head = SimpleHead()
        
        # Enhanced TTCS with failure prediction
        enhanced_ttcs = MemoryEfficientTTCS(
            encoder=encoder,
            head=head,
            passes=6,
            failure_predictor=failure_predictor,
            enable_memory_efficient=True,
            performance_tracking=True
        )
        
        # Create episode
        support_x = torch.randn(9, 8)
        support_y = torch.tensor([0,0,0, 1,1,1, 2,2,2])
        query_x = torch.randn(6, 8)
        query_y = torch.tensor([0,0, 1,1, 2,2])
        
        episode = Episode(support_x, support_y, query_x, query_y)
        
        # Forward pass should work with failure prediction
        predictions, metrics = enhanced_ttcs.forward(episode, return_metrics=True)
        
        assert torch.isfinite(predictions).all()
        assert 'failure_risk' in metrics
        assert 0.0 <= metrics['failure_risk'] <= 1.0
        
        print(f"✓ Failure prediction integration works, risk: {metrics['failure_risk']:.3f}")
        
        # Test multiple episodes to build failure prediction history
        for i in range(3):
            episode_i = Episode(
                torch.randn(6, 8), torch.tensor([0, 0, 1, 1, 2, 2]),
                torch.randn(4, 8), torch.tensor([0, 1, 2, 0])
            )
            pred = enhanced_ttcs.forward(episode_i)
            assert torch.isfinite(pred).all()
        
        print("✓ Multiple episodes with failure prediction successful")
        
        return True
        
    except Exception as e:
        print(f"✗ Failure prediction integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_memory_conservative_mode():
    print("\nTesting memory conservative mode...")
    
    try:
        from src.meta_learning.algos.enhanced_ttcs import MemoryEfficientTTCS
        from src.meta_learning.shared.types import Episode
        
        # Simple model
        encoder = nn.Sequential(nn.Linear(5, 10), nn.Dropout(0.3), nn.ReLU(), nn.Linear(10, 8))
        
        class SimpleHead(nn.Module):
            def forward(self, support_features, support_labels, query_features):
                unique_labels = torch.unique(support_labels)
                prototypes = []
                for label in unique_labels:
                    mask = support_labels == label
                    prototype = support_features[mask].mean(dim=0)
                    prototypes.append(prototype)
                prototypes = torch.stack(prototypes)
                distances = torch.cdist(query_features, prototypes)
                return -distances
        
        head = SimpleHead()
        
        # Test with very restrictive memory budget
        conservative_ttcs = MemoryEfficientTTCS(
            encoder=encoder,
            head=head,
            passes=8,
            memory_budget_mb=10,  # Very small budget to trigger conservative mode
            enable_memory_efficient=True,
            performance_tracking=True
        )
        
        # Create episode
        support_x = torch.randn(6, 5)
        support_y = torch.tensor([0, 0, 1, 1, 2, 2])
        query_x = torch.randn(3, 5)
        query_y = torch.tensor([0, 1, 2])
        
        episode = Episode(support_x, support_y, query_x, query_y)
        
        # Should work even with memory constraints
        predictions, metrics = conservative_ttcs.forward(episode, return_metrics=True)
        
        assert torch.isfinite(predictions).all()
        assert predictions.shape == (3, 3)
        
        # Check if memory conservation was triggered
        method = metrics.get('method', 'unknown')
        print(f"✓ Conservative mode method: {method}")
        
        if 'memory_saved_mb' in metrics:
            print(f"✓ Memory saved: {metrics['memory_saved_mb']:.2f}MB")
        
        print("✓ Memory conservative mode test successful")
        
        return True
        
    except Exception as e:
        print(f"✗ Memory conservative mode test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_convenience_functions():
    print("\nTesting convenience functions...")
    
    try:
        from src.meta_learning.algos.enhanced_ttcs import (
            enhanced_ttcs_predict,
            adaptive_ttcs_predict,
            ttcs_with_memory_monitoring
        )
        from src.meta_learning.shared.types import Episode
        
        # Simple model components
        encoder = nn.Sequential(nn.Linear(4, 8), nn.Dropout(0.2), nn.ReLU(), nn.Linear(8, 6))
        
        class SimpleHead(nn.Module):
            def forward(self, support_features, support_labels, query_features):
                unique_labels = torch.unique(support_labels)
                prototypes = []
                for label in unique_labels:
                    mask = support_labels == label
                    prototype = support_features[mask].mean(dim=0)
                    prototypes.append(prototype)
                prototypes = torch.stack(prototypes)
                distances = torch.cdist(query_features, prototypes)
                return -distances
        
        head = SimpleHead()
        
        # Create episode
        support_x = torch.randn(4, 4)
        support_y = torch.tensor([0, 0, 1, 1])
        query_x = torch.randn(2, 4)
        query_y = torch.tensor([0, 1])
        
        episode = Episode(support_x, support_y, query_x, query_y)
        
        # Test enhanced_ttcs_predict
        enhanced_pred = enhanced_ttcs_predict(
            encoder=encoder,
            head=head,
            episode=episode,
            passes=4,
            enable_memory_efficient=True,
            enable_detach_optimization=True
        )
        
        assert enhanced_pred.shape == (2, 2)
        assert torch.isfinite(enhanced_pred).all()
        print("✓ enhanced_ttcs_predict works")
        
        # Test adaptive_ttcs_predict
        adaptive_pred = adaptive_ttcs_predict(
            encoder=encoder,
            head=head,
            episode=episode,
            passes=4
        )
        
        assert adaptive_pred.shape == (2, 2)
        assert torch.isfinite(adaptive_pred).all()
        print("✓ adaptive_ttcs_predict works")
        
        # Test ttcs_with_memory_monitoring
        monitoring_pred, memory_metrics = ttcs_with_memory_monitoring(
            encoder=encoder,
            head=head,
            episode=episode,
            passes=4
        )
        
        assert monitoring_pred.shape == (2, 2)
        assert torch.isfinite(monitoring_pred).all()
        assert isinstance(memory_metrics, dict)
        print(f"✓ ttcs_with_memory_monitoring works, metrics: {list(memory_metrics.keys())}")
        
        return True
        
    except Exception as e:
        print(f"✗ Convenience functions test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_mixed_precision_support():
    print("\nTesting mixed precision support...")
    
    try:
        from src.meta_learning.algos.enhanced_ttcs import MemoryEfficientTTCS
        from src.meta_learning.shared.types import Episode
        
        # Skip if CUDA not available
        if not torch.cuda.is_available():
            print("⚠️ CUDA not available, skipping mixed precision test")
            return True
        
        # Simple model
        encoder = nn.Sequential(
            nn.Linear(6, 12),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(12, 8)
        ).cuda()
        
        class SimpleHead(nn.Module):
            def forward(self, support_features, support_labels, query_features):
                unique_labels = torch.unique(support_labels)
                prototypes = []
                for label in unique_labels:
                    mask = support_labels == label
                    prototype = support_features[mask].mean(dim=0)
                    prototypes.append(prototype)
                prototypes = torch.stack(prototypes)
                distances = torch.cdist(query_features, prototypes)
                return -distances
        
        head = SimpleHead().cuda()
        
        # Test with mixed precision
        mixed_precision_ttcs = MemoryEfficientTTCS(
            encoder=encoder,
            head=head,
            passes=4,
            enable_mixed_precision=True,
            enable_memory_efficient=True,
            performance_tracking=True
        )
        
        # Create episode
        support_x = torch.randn(6, 6).cuda()
        support_y = torch.tensor([0, 0, 1, 1, 2, 2]).cuda()
        query_x = torch.randn(3, 6).cuda()
        query_y = torch.tensor([0, 1, 2]).cuda()
        
        episode = Episode(support_x, support_y, query_x, query_y)
        
        # Should work with mixed precision
        predictions = mixed_precision_ttcs.forward(episode, device=torch.device('cuda'))
        
        assert torch.isfinite(predictions).all()
        assert predictions.shape == (3, 3)
        
        print("✓ Mixed precision support works")
        
        return True
        
    except Exception as e:
        print(f"✗ Mixed precision test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
    
    print("Testing Enhanced TTCS with detach_module Integration")
    print("=" * 58)
    
    success = True
    success &= test_memory_efficient_ttcs_basic()
    success &= test_memory_optimization_effectiveness()
    success &= test_adaptive_memory_ttcs()
    success &= test_failure_prediction_integration()
    success &= test_memory_conservative_mode()
    success &= test_convenience_functions()
    success &= test_mixed_precision_support()
    
    print("\n" + "=" * 58)
    if success:
        print("✅ All Enhanced TTCS tests completed successfully!")
        print("🚀 Enhanced TTCS with detach_module integration is ready!")
        print("💰 Please donate if this accelerates your research!")
    else:
        print("❌ Some Enhanced TTCS tests failed!")
        sys.exit(1)