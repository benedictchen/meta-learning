#!/usr/bin/env python3
"""Test script for Learnable Optimizer integration with Failure Prediction"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def test_failure_prediction_integration():
    print("Testing Failure Prediction Integration...")
    
    # Set seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    try:
        from src.meta_learning.optimization.learnable_optimizer import LearnableOptimizer, ScaleTransform
        from src.meta_learning.ml_enhancements.failure_prediction import FailurePredictionModel
        from src.meta_learning.shared.types import Episode
        
        print("✓ All imports successful")
        
        # Create failure prediction model
        failure_predictor = FailurePredictionModel()
        
        # Create a simple model
        model = nn.Linear(8, 3)
        
        # Create learnable optimizer with failure prediction
        transform = ScaleTransform(per_element=False, init_scale=1.0)
        optimizer = LearnableOptimizer(
            model=model,
            transform=transform,
            lr=0.05,
            meta_lr=0.01,
            failure_prediction_model=failure_predictor,
            adaptive_lr=True
        )
        
        print("✓ LearnableOptimizer with failure prediction created")
        
        # Create mock episode data
        support_x = torch.randn(15, 8)  # Support set
        support_y = torch.randint(0, 3, (15,))
        query_x = torch.randn(5, 8)    # Query set
        query_y = torch.randint(0, 3, (5,))
        
        # Create Episode object
        episode = Episode(
            support_x=support_x,
            support_y=support_y,
            query_x=query_x,
            query_y=query_y
        )
        
        print("✓ Mock episode created")
        
        # Set current episode for failure prediction
        optimizer.set_current_episode(episode)
        
        # Test failure prediction without history (should return low risk)
        initial_lr = optimizer.get_learning_rate()
        failure_risk = optimizer.predict_and_adjust_learning_rate()
        new_lr = optimizer.get_learning_rate()
        
        print(f"✓ Initial failure risk prediction: {failure_risk:.4f}")
        print(f"✓ Learning rate adjustment: {initial_lr:.4f} -> {new_lr:.4f}")
        
        # Test optimization steps with failure prediction
        losses = []
        for step in range(5):
            def closure():
                optimizer.zero_grad()
                logits = model(support_x)
                loss = F.cross_entropy(logits, support_y)
                loss.backward()
                return loss
            
            # Predict and adjust learning rate before step
            step_failure_risk = optimizer.predict_and_adjust_learning_rate()
            
            # Take optimization step
            loss = optimizer.step(closure)
            losses.append(loss.item())
            
            # Update failure prediction model with outcome
            # Consider it a success if loss is decreasing or reasonable
            success = len(losses) < 2 or losses[-1] <= losses[-2]
            optimizer.update_failure_prediction_model(success)
            
            print(f"  Step {step+1}: loss={loss:.4f}, risk={step_failure_risk:.4f}, lr={optimizer.get_learning_rate():.4f}")
        
        print("✓ Optimization with failure prediction completed")
        
        # Test failure prediction metrics
        fp_metrics = optimizer.get_failure_prediction_metrics()
        print(f"✓ Failure prediction metrics: {fp_metrics}")
        
        # Verify metrics structure
        expected_keys = [
            'current_failure_risk', 'average_failure_risk', 'max_failure_risk',
            'min_failure_risk', 'lr_adjustment_ratio', 'adaptive_lr_enabled',
            'has_failure_prediction'
        ]
        
        for key in expected_keys:
            assert key in fp_metrics, f"Missing key: {key}"
        
        assert fp_metrics['has_failure_prediction'] == True
        assert fp_metrics['adaptive_lr_enabled'] == True
        assert 0.0 <= fp_metrics['current_failure_risk'] <= 1.0
        
        print("✓ All failure prediction metrics validated")
        
        # Test optimizer without failure prediction (should still work)
        optimizer_no_fp = LearnableOptimizer(
            model=nn.Linear(4, 2),
            lr=0.1,
            adaptive_lr=False,
            failure_prediction_model=None
        )
        
        # Should return 0 risk when no failure prediction
        no_fp_risk = optimizer_no_fp.predict_and_adjust_learning_rate()
        assert no_fp_risk == 0.0
        
        no_fp_metrics = optimizer_no_fp.get_failure_prediction_metrics()
        assert no_fp_metrics == {}
        
        print("✓ No failure prediction mode works correctly")
        
        # Test learning rate bounds
        optimizer.set_learning_rate(0.001)  # Set to minimum
        optimizer.failure_risk_history = [0.9]  # High risk
        risk = optimizer.predict_and_adjust_learning_rate()
        
        # Should not go below minimum
        assert optimizer.get_learning_rate() >= 0.001
        print(f"✓ Learning rate bounds respected: {optimizer.get_learning_rate():.6f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Failure prediction integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_adaptive_learning_scenarios():
    """Test different failure risk scenarios."""
    print("\nTesting adaptive learning rate scenarios...")
    
    try:
        from src.meta_learning.optimization.learnable_optimizer import LearnableOptimizer, ScaleTransform
        from src.meta_learning.ml_enhancements.failure_prediction import FailurePredictionModel
        from src.meta_learning.shared.types import Episode
        
        # Mock failure predictor that returns controllable risk levels
        class MockFailurePredictor:
            def __init__(self, risk_level):
                self.risk_level = risk_level
                self.feature_history = []
                self.failure_history = []
            
            def predict_failure_risk(self, episode, algorithm_state):
                return self.risk_level
            
            def update_with_outcome(self, episode, algorithm_state, failed):
                pass
        
        model = nn.Linear(4, 2)
        base_lr = 0.1
        
        # Test high risk scenario (should reduce learning rate)
        high_risk_predictor = MockFailurePredictor(0.8)
        optimizer_high = LearnableOptimizer(
            model=model,
            lr=base_lr,
            failure_prediction_model=high_risk_predictor,
            adaptive_lr=True
        )
        
        # Create dummy episode
        episode = Episode(
            support_x=torch.randn(5, 4),
            support_y=torch.randint(0, 2, (5,)),
            query_x=torch.randn(3, 4),
            query_y=torch.randint(0, 2, (3,))
        )
        optimizer_high.set_current_episode(episode)
        
        risk = optimizer_high.predict_and_adjust_learning_rate()
        high_risk_lr = optimizer_high.get_learning_rate()
        
        assert risk == 0.8
        assert high_risk_lr < base_lr  # Should reduce learning rate
        print(f"✓ High risk scenario: LR {base_lr} -> {high_risk_lr:.4f} (risk: {risk})")
        
        # Test low risk scenario (should increase learning rate)
        low_risk_predictor = MockFailurePredictor(0.05)
        optimizer_low = LearnableOptimizer(
            model=nn.Linear(4, 2),
            lr=base_lr,
            failure_prediction_model=low_risk_predictor,
            adaptive_lr=True
        )
        optimizer_low.set_current_episode(episode)
        
        risk = optimizer_low.predict_and_adjust_learning_rate()
        low_risk_lr = optimizer_low.get_learning_rate()
        
        assert risk == 0.05
        assert low_risk_lr > base_lr  # Should increase learning rate
        print(f"✓ Low risk scenario: LR {base_lr} -> {low_risk_lr:.4f} (risk: {risk})")
        
        # Test medium risk scenario (should moderately adjust)
        medium_risk_predictor = MockFailurePredictor(0.5)
        optimizer_med = LearnableOptimizer(
            model=nn.Linear(4, 2),
            lr=base_lr,
            failure_prediction_model=medium_risk_predictor,
            adaptive_lr=True
        )
        optimizer_med.set_current_episode(episode)
        
        risk = optimizer_med.predict_and_adjust_learning_rate()
        med_risk_lr = optimizer_med.get_learning_rate()
        
        assert risk == 0.5
        assert med_risk_lr < base_lr  # Should reduce learning rate moderately
        print(f"✓ Medium risk scenario: LR {base_lr} -> {med_risk_lr:.4f} (risk: {risk})")
        
        return True
        
    except Exception as e:
        print(f"✗ Adaptive learning scenarios test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
    
    success = test_failure_prediction_integration()
    if success:
        success = test_adaptive_learning_scenarios()
    
    if success:
        print("\n✅ All Failure Prediction Integration tests completed successfully!")
    else:
        print("\n❌ Some Failure Prediction Integration tests failed!")
        sys.exit(1)