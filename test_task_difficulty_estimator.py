#!/usr/bin/env python3
"""
Test Task Difficulty Estimator Implementation
=============================================

Quick test to verify if the Task Difficulty Estimator is working.
"""

import sys
import os
import torch

# Add the src directory to Python path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from meta_learning.algorithms.task_difficulty_estimator import FewShotTaskDifficultyEstimator
    from meta_learning.shared.types import Episode
    print("✅ Successfully imported FewShotTaskDifficultyEstimator and Episode")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

def test_task_difficulty_estimator():
    """Test basic task difficulty estimation functionality."""
    print("\n🧪 Testing Task Difficulty Estimator...")
    
    try:
        # Initialize estimator
        estimator = FewShotTaskDifficultyEstimator()
        
        # Create synthetic few-shot episode
        n_way = 5
        k_shot = 3
        n_query = 10
        
        # Create simple separable task (should be easier)
        support_x = torch.randn(n_way * k_shot, 64)
        support_y = torch.repeat_interleave(torch.arange(n_way), k_shot)
        query_x = torch.randn(n_query, 64)
        query_y = torch.randint(0, n_way, (n_query,))
        
        episode_easy = Episode(support_x, support_y, query_x, query_y)
        
        # Estimate difficulty
        difficulty_easy = estimator.estimate_episode_difficulty(episode_easy)
        print(f"✅ Easy task difficulty: {difficulty_easy:.4f}")
        
        # Create hard task (overlapping classes)
        support_x_hard = 0.1 * torch.randn(n_way * k_shot, 64)  # Very similar
        episode_hard = Episode(support_x_hard, support_y, query_x, query_y)
        
        difficulty_hard = estimator.estimate_episode_difficulty(episode_hard)
        print(f"✅ Hard task difficulty: {difficulty_hard:.4f}")
        
        # Verify difficulty makes sense
        assert 0.0 <= difficulty_easy <= 1.0, f"Easy difficulty out of range: {difficulty_easy}"
        assert 0.0 <= difficulty_hard <= 1.0, f"Hard difficulty out of range: {difficulty_hard}"
        
        print("✅ Task difficulty estimation working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Task difficulty estimation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_task_difficulty_estimator()
    if success:
        print("🎉 Task Difficulty Estimator is working!")
    else:
        print("❌ Task Difficulty Estimator needs fixes")
    sys.exit(0 if success else 1)