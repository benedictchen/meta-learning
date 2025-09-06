#!/usr/bin/env python3
"""Test script for hardness_metric function"""

import torch
import numpy as np

def test_hardness_metric():
    print("Testing hardness_metric function...")
    
    # Set seeds for reproducibility  
    np.random.seed(42)
    torch.manual_seed(42)
    
    try:
        from src.meta_learning.evaluation.task_analysis import hardness_metric, TaskDifficultyAnalyzer
        from src.meta_learning.shared.types import Episode
        
        print("✓ All imports successful")
        
        # Test 1: Easy task (well-separated classes)
        # Create two classes with clear separation
        class_0 = torch.zeros(10, 4) + torch.randn(10, 4) * 0.1  # Centered at origin
        class_1 = torch.ones(10, 4) * 5 + torch.randn(10, 4) * 0.1  # Centered at (5,5,5,5)
        
        easy_support_x = torch.cat([class_0, class_1], dim=0)
        easy_support_y = torch.cat([torch.zeros(10, dtype=torch.long), 
                                   torch.ones(10, dtype=torch.long)], dim=0)
        
        easy_episode = Episode(
            support_x=easy_support_x,
            support_y=easy_support_y,
            query_x=torch.randn(4, 4),
            query_y=torch.randint(0, 2, (4,))
        )
        
        easy_hardness = hardness_metric(easy_episode, num_classes=2)
        print(f"✓ Easy task hardness: {easy_hardness:.4f}")
        assert 0.0 <= easy_hardness <= 1.0
        
        # Test 2: Hard task (overlapping classes)
        # Create two classes with significant overlap
        class_0_hard = torch.randn(10, 4) * 2.0  # Wide spread around origin
        class_1_hard = torch.randn(10, 4) * 2.0 + 1.0  # Wide spread around (1,1,1,1)
        
        hard_support_x = torch.cat([class_0_hard, class_1_hard], dim=0)
        hard_support_y = torch.cat([torch.zeros(10, dtype=torch.long), 
                                   torch.ones(10, dtype=torch.long)], dim=0)
        
        hard_episode = Episode(
            support_x=hard_support_x,
            support_y=hard_support_y,
            query_x=torch.randn(4, 4),
            query_y=torch.randint(0, 2, (4,))
        )
        
        hard_hardness = hardness_metric(hard_episode, num_classes=2)
        print(f"✓ Hard task hardness: {hard_hardness:.4f}")
        assert 0.0 <= hard_hardness <= 1.0
        
        # Hard task should have higher hardness than easy task
        assert hard_hardness > easy_hardness, f"Hard task ({hard_hardness}) should be harder than easy task ({easy_hardness})"
        print(f"✓ Hardness ordering correct: hard ({hard_hardness:.4f}) > easy ({easy_hardness:.4f})")
        
        # Test 3: Edge cases
        # Single class
        single_class_episode = Episode(
            support_x=torch.randn(5, 4),
            support_y=torch.zeros(5, dtype=torch.long),
            query_x=torch.randn(2, 4),
            query_y=torch.zeros(2, dtype=torch.long)
        )
        
        single_hardness = hardness_metric(single_class_episode, num_classes=1)
        assert single_hardness == 0.0
        print("✓ Single class task has zero hardness")
        
        # Empty episode
        empty_episode = Episode(
            support_x=torch.empty(0, 4),
            support_y=torch.empty(0, dtype=torch.long),
            query_x=torch.randn(2, 4),
            query_y=torch.randint(0, 2, (2,))
        )
        
        empty_hardness = hardness_metric(empty_episode, num_classes=2)
        assert empty_hardness == 0.0
        print("✓ Empty episode has zero hardness")
        
        # Test 4: TaskDifficultyAnalyzer integration
        analyzer = TaskDifficultyAnalyzer()
        
        difficulty_report = analyzer.analyze_episode(easy_episode)
        print(f"✓ Difficulty report keys: {list(difficulty_report.keys())}")
        
        expected_keys = ['hardness', 'num_classes', 'support_size', 'feature_dim']
        for key in expected_keys:
            assert key in difficulty_report, f"Missing key: {key}"
        
        assert difficulty_report['hardness'] == easy_hardness
        print("✓ TaskDifficultyAnalyzer integration works")
        
        # Test 5: Three-class scenario
        class_0_3 = torch.zeros(8, 4) + torch.randn(8, 4) * 0.1
        class_1_3 = torch.ones(8, 4) * 3 + torch.randn(8, 4) * 0.1  
        class_2_3 = torch.ones(8, 4) * -3 + torch.randn(8, 4) * 0.1
        
        three_class_x = torch.cat([class_0_3, class_1_3, class_2_3], dim=0)
        three_class_y = torch.cat([torch.zeros(8, dtype=torch.long),
                                  torch.ones(8, dtype=torch.long),
                                  torch.full((8,), 2, dtype=torch.long)], dim=0)
        
        three_class_episode = Episode(
            support_x=three_class_x,
            support_y=three_class_y,
            query_x=torch.randn(6, 4),
            query_y=torch.randint(0, 3, (6,))
        )
        
        three_class_hardness = hardness_metric(three_class_episode, num_classes=3)
        print(f"✓ Three-class task hardness: {three_class_hardness:.4f}")
        assert 0.0 <= three_class_hardness <= 1.0
        
        print("🎉 All hardness_metric tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Hardness metric test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
    
    success = test_hardness_metric()
    if success:
        print("\n✅ All hardness metric tests completed successfully!")
    else:
        print("\n❌ Some hardness metric tests failed!")
        sys.exit(1)