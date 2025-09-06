#!/usr/bin/env python3
"""Test script for Enhanced MAML with clone_module integration"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

def test_enhanced_maml_basic():
    print("Testing Enhanced MAML basic functionality...")
    
    try:
        from src.meta_learning.algos.enhanced_maml import EnhancedMAML
        from src.meta_learning.shared.types import Episode
        
        # Create a simple model for testing
        model = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 3)
        )
        
        # Create Enhanced MAML
        enhanced_maml = EnhancedMAML(
            model=model,
            inner_lr=0.01,
            inner_steps=3,
            use_enhanced_cloning=True,
            performance_tracking=True
        )
        
        print("✓ Enhanced MAML initialized successfully")
        
        # Create mock episode
        support_x = torch.randn(9, 8)  # 3-way 3-shot
        support_y = torch.tensor([0,0,0, 1,1,1, 2,2,2])
        query_x = torch.randn(6, 8)   # 2 queries per class
        query_y = torch.tensor([0,0, 1,1, 2,2])
        
        episode = Episode(support_x, support_y, query_x, query_y)
        
        # Test forward pass
        predictions = enhanced_maml.forward(episode)
        assert predictions.shape == (6, 3), f"Expected shape (6, 3), got {predictions.shape}"
        assert torch.isfinite(predictions).all(), "Predictions should be finite"
        
        print(f"✓ Forward pass successful, predictions shape: {predictions.shape}")
        
        # Test with metrics
        predictions_with_metrics, metrics = enhanced_maml.forward(episode, return_metrics=True)
        assert torch.equal(predictions, predictions_with_metrics)
        assert isinstance(metrics, dict)
        assert 'adaptation_time' in metrics
        assert 'method' in metrics
        
        print(f"✓ Metrics collection works: {list(metrics.keys())}")
        
        return True
        
    except Exception as e:
        print(f"✗ Enhanced MAML basic test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_enhanced_vs_standard_cloning():
    print("\nTesting Enhanced vs Standard cloning comparison...")
    
    try:
        from src.meta_learning.algos.enhanced_maml import EnhancedMAML
        from src.meta_learning.shared.types import Episode
        
        # Simple model
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
        
        # Create episode
        support_x = torch.randn(4, 4)
        support_y = torch.tensor([0, 0, 1, 1])
        query_x = torch.randn(2, 4)
        query_y = torch.tensor([0, 1])
        episode = Episode(support_x, support_y, query_x, query_y)
        
        # Test enhanced cloning
        enhanced_maml = EnhancedMAML(
            model=model,
            inner_lr=0.05,
            inner_steps=2,
            use_enhanced_cloning=True
        )
        
        start_time = time.time()
        enhanced_predictions = enhanced_maml.forward(episode)
        enhanced_time = time.time() - start_time
        
        # Test standard cloning fallback
        standard_maml = EnhancedMAML(
            model=model,
            inner_lr=0.05,
            inner_steps=2,
            use_enhanced_cloning=False
        )
        
        start_time = time.time()
        standard_predictions = standard_maml.forward(episode)
        standard_time = time.time() - start_time
        
        # Both should produce valid results
        assert torch.isfinite(enhanced_predictions).all()
        assert torch.isfinite(standard_predictions).all()
        assert enhanced_predictions.shape == standard_predictions.shape
        
        print(f"✓ Enhanced cloning time: {enhanced_time:.4f}s")
        print(f"✓ Standard cloning time: {standard_time:.4f}s")
        print("✓ Both cloning methods produce valid results")
        
        return True
        
    except Exception as e:
        print(f"✗ Cloning comparison test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_failure_prediction_integration():
    print("\nTesting failure prediction integration...")
    
    try:
        from src.meta_learning.algos.enhanced_maml import EnhancedMAML
        from src.meta_learning.ml_enhancements.failure_prediction import FailurePredictionModel
        from src.meta_learning.shared.types import Episode
        
        # Create failure predictor
        failure_predictor = FailurePredictionModel()
        
        # Simple model
        model = nn.Sequential(nn.Linear(6, 10), nn.ReLU(), nn.Linear(10, 2))
        
        # Enhanced MAML with failure prediction
        enhanced_maml = EnhancedMAML(
            model=model,
            inner_lr=0.01,
            inner_steps=3,
            failure_predictor=failure_predictor,
            performance_tracking=True
        )
        
        # Create episode
        support_x = torch.randn(6, 6)
        support_y = torch.tensor([0, 0, 0, 1, 1, 1])
        query_x = torch.randn(4, 6)
        query_y = torch.tensor([0, 0, 1, 1])
        episode = Episode(support_x, support_y, query_x, query_y)
        
        # Forward pass should work with failure prediction
        predictions, metrics = enhanced_maml.forward(episode, return_metrics=True)
        
        assert torch.isfinite(predictions).all()
        assert 'failure_risk' in metrics
        assert 0.0 <= metrics['failure_risk'] <= 1.0
        
        print(f"✓ Failure prediction integration works, risk: {metrics['failure_risk']:.3f}")
        
        # Test multiple episodes to build failure prediction history
        for i in range(3):
            episode_i = Episode(
                torch.randn(6, 6), torch.tensor([0, 0, 0, 1, 1, 1]),
                torch.randn(4, 6), torch.tensor([0, 0, 1, 1])
            )
            pred = enhanced_maml.forward(episode_i)
            assert torch.isfinite(pred).all()
        
        print("✓ Multiple episodes with failure prediction successful")
        
        return True
        
    except Exception as e:
        print(f"✗ Failure prediction integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_performance_monitoring():
    print("\nTesting performance monitoring...")
    
    try:
        from src.meta_learning.algos.enhanced_maml import EnhancedMAML
        from src.meta_learning.shared.types import Episode
        
        model = nn.Sequential(nn.Linear(5, 12), nn.ReLU(), nn.Linear(12, 3))
        
        enhanced_maml = EnhancedMAML(
            model=model,
            performance_tracking=True
        )
        
        # Run several episodes to collect metrics
        for i in range(3):
            support_x = torch.randn(9, 5)
            support_y = torch.tensor([0,0,0, 1,1,1, 2,2,2])
            query_x = torch.randn(3, 5) 
            query_y = torch.tensor([0, 1, 2])
            
            episode = Episode(support_x, support_y, query_x, query_y)
            predictions = enhanced_maml.forward(episode)
            assert torch.isfinite(predictions).all()
        
        # Get performance metrics
        metrics = enhanced_maml.get_performance_metrics()
        
        assert isinstance(metrics, dict)
        assert len(metrics) > 0
        
        expected_keys = [
            'use_enhanced_cloning', 'memory_efficient', 
            'inner_lr', 'inner_steps', 'has_failure_predictor'
        ]
        for key in expected_keys:
            assert key in metrics, f"Missing metric key: {key}"
        
        print(f"✓ Performance metrics collected: {list(metrics.keys())}")
        print(f"✓ Enhanced cloning enabled: {metrics['use_enhanced_cloning']}")
        print(f"✓ Memory efficient: {metrics['memory_efficient']}")
        
        # Test metrics reset
        enhanced_maml.reset_metrics()
        reset_metrics = enhanced_maml.get_performance_metrics()
        
        # Should still have configuration but not performance data
        assert 'use_enhanced_cloning' in reset_metrics
        print("✓ Metrics reset successful")
        
        return True
        
    except Exception as e:
        print(f"✗ Performance monitoring test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_batch_adaptation():
    print("\nTesting batch adaptation...")
    
    try:
        from src.meta_learning.algos.enhanced_maml import EnhancedMAML
        from src.meta_learning.shared.types import Episode
        
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
        
        enhanced_maml = EnhancedMAML(model=model, inner_steps=2)
        
        # Create multiple episodes
        episodes = []
        for i in range(3):
            support_x = torch.randn(4, 4)
            support_y = torch.tensor([0, 0, 1, 1])
            query_x = torch.randn(2, 4)
            query_y = torch.tensor([0, 1])
            
            episode = Episode(support_x, support_y, query_x, query_y)
            episodes.append(episode)
        
        # Batch adaptation
        batch_predictions = enhanced_maml.batch_adapt(episodes)
        
        assert len(batch_predictions) == 3
        for pred in batch_predictions:
            assert pred.shape == (2, 2)  # 2 queries, 2 classes
            assert torch.isfinite(pred).all()
        
        print(f"✓ Batch adaptation successful for {len(episodes)} episodes")
        
        return True
        
    except Exception as e:
        print(f"✗ Batch adaptation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_convenience_functions():
    print("\nTesting convenience functions...")
    
    try:
        from src.meta_learning.algos.enhanced_maml import (
            enhanced_inner_adapt_and_eval, 
            enhanced_meta_outer_step
        )
        from src.meta_learning.shared.types import Episode
        
        model = nn.Sequential(nn.Linear(6, 8), nn.ReLU(), nn.Linear(8, 2))
        
        # Test enhanced_inner_adapt_and_eval
        support_x = torch.randn(4, 6)
        support_y = torch.tensor([0, 0, 1, 1])
        query_x = torch.randn(2, 6)
        query_y = torch.tensor([0, 1])
        
        episode = Episode(support_x, support_y, query_x, query_y)
        
        predictions = enhanced_inner_adapt_and_eval(
            model=model,
            episode=episode,
            inner_lr=0.01,
            inner_steps=2,
            use_enhanced_cloning=True
        )
        
        assert predictions.shape == (2, 2)
        assert torch.isfinite(predictions).all()
        print("✓ enhanced_inner_adapt_and_eval works")
        
        # Test enhanced_meta_outer_step
        episodes = [episode]  # Single episode for simplicity
        
        meta_loss, meta_metrics = enhanced_meta_outer_step(
            model=model,
            episodes=episodes,
            inner_lr=0.01,
            inner_steps=2,
            meta_lr=0.001,
            use_enhanced_cloning=True
        )
        
        assert isinstance(meta_loss, float)
        assert isinstance(meta_metrics, dict)
        assert 'meta_loss' in meta_metrics
        assert 'num_episodes' in meta_metrics
        
        print(f"✓ enhanced_meta_outer_step works, meta_loss: {meta_loss:.4f}")
        print(f"✓ Meta metrics: {list(meta_metrics.keys())}")
        
        return True
        
    except Exception as e:
        print(f"✗ Convenience functions test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_memory_efficiency():
    print("\nTesting memory efficiency options...")
    
    try:
        from src.meta_learning.algos.enhanced_maml import EnhancedMAML
        from src.meta_learning.shared.types import Episode
        
        # Larger model to test memory efficiency
        model = nn.Sequential(
            nn.Linear(10, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 3)
        )
        
        # Test memory efficient mode
        efficient_maml = EnhancedMAML(
            model=model,
            memory_efficient=True,
            use_enhanced_cloning=True
        )
        
        # Test standard mode
        standard_maml = EnhancedMAML(
            model=model,
            memory_efficient=False,
            use_enhanced_cloning=True
        )
        
        # Create episode
        support_x = torch.randn(9, 10)
        support_y = torch.tensor([0,0,0, 1,1,1, 2,2,2])
        query_x = torch.randn(6, 10)
        query_y = torch.tensor([0,0, 1,1, 2,2])
        
        episode = Episode(support_x, support_y, query_x, query_y)
        
        # Both should work
        efficient_pred = efficient_maml.forward(episode)
        standard_pred = standard_maml.forward(episode)
        
        assert efficient_pred.shape == standard_pred.shape
        assert torch.isfinite(efficient_pred).all()
        assert torch.isfinite(standard_pred).all()
        
        print("✓ Both memory efficient and standard modes work")
        print(f"✓ Prediction shapes: {efficient_pred.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Memory efficiency test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
    
    print("Testing Enhanced MAML with Clone Module Integration")
    print("=" * 55)
    
    success = True
    success &= test_enhanced_maml_basic()
    success &= test_enhanced_vs_standard_cloning()
    success &= test_failure_prediction_integration()
    success &= test_performance_monitoring()
    success &= test_batch_adaptation()
    success &= test_convenience_functions()
    success &= test_memory_efficiency()
    
    print("\n" + "=" * 55)
    if success:
        print("✅ All Enhanced MAML tests completed successfully!")
        print("🚀 Enhanced MAML with clone_module integration is ready!")
    else:
        print("❌ Some Enhanced MAML tests failed!")
        sys.exit(1)