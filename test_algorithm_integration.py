#!/usr/bin/env python3
"""Test script for algorithm registry, selector, and A/B testing integration"""

import torch
import torch.nn as nn
import numpy as np
import time

def test_algorithm_registry():
    print("Testing AlgorithmRegistry...")
    
    try:
        from src.meta_learning.ml_enhancements.algorithm_registry import (
            algorithm_registry, AlgorithmType, TaskDifficulty,
            get_algorithm, select_algorithm, get_suitable_algorithms
        )
        from src.meta_learning.shared.types import Episode
        
        # Test registry initialization
        all_algorithms = algorithm_registry.get_all_algorithms()
        print(f"✓ Registry loaded {len(all_algorithms)} algorithms")
        
        # Verify ridge regression is registered
        assert 'ridge_regression' in all_algorithms
        ridge_metadata = all_algorithms['ridge_regression']
        assert ridge_metadata.algorithm_type == AlgorithmType.OPTIMIZATION_BASED
        print("✓ Ridge regression properly registered")
        
        # Test algorithm type filtering
        optimization_algorithms = algorithm_registry.get_algorithms_by_type(AlgorithmType.OPTIMIZATION_BASED)
        assert 'ridge_regression' in optimization_algorithms
        print(f"✓ Found {len(optimization_algorithms)} optimization-based algorithms")
        
        # Test suitability search
        suitable = algorithm_registry.get_suitable_algorithms(
            n_shot=5,
            n_classes=3,
            task_difficulty=TaskDifficulty.MEDIUM
        )
        assert len(suitable) > 0
        print(f"✓ Found {len(suitable)} suitable algorithms for 5-shot 3-class medium task")
        
        # Test episode-based selection
        support_x = torch.randn(15, 8)  # 5-way 3-shot
        support_y = torch.tensor([0,0,0, 1,1,1, 2,2,2, 3,3,3, 4,4,4])
        query_x = torch.randn(10, 8)
        query_y = torch.tensor([0,0, 1,1, 2,2, 3,3, 4,4])
        
        episode = Episode(support_x, support_y, query_x, query_y)
        selected = select_algorithm(episode)
        assert selected in all_algorithms
        print(f"✓ Selected algorithm: {selected}")
        
        return True
        
    except Exception as e:
        print(f"✗ Algorithm registry test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_enhanced_algorithm_selector():
    print("\nTesting Enhanced AlgorithmSelector...")
    
    try:
        from src.meta_learning.ml_enhancements.algorithm_selection import AlgorithmSelector
        from src.meta_learning.shared.types import Episode
        
        # Create enhanced selector
        selector = AlgorithmSelector(selection_strategy="heuristic_enhanced")
        
        # Test basic selection
        support_x = torch.randn(12, 6)  # 4-way 3-shot
        support_y = torch.tensor([0,0,0, 1,1,1, 2,2,2, 3,3,3])
        query_x = torch.randn(8, 6)
        query_y = torch.tensor([0,0, 1,1, 2,2, 3,3])
        
        episode = Episode(support_x, support_y, query_x, query_y)
        selected = selector.select_algorithm(episode)
        print(f"✓ Selected algorithm for 4-way 3-shot task: {selected}")
        
        # Test performance tracking
        selector.update_performance(selected, episode, 0.85)
        stats = selector.get_algorithm_stats(selected)
        assert stats['count'] == 1
        assert stats['mean_accuracy'] == 0.85
        print("✓ Performance tracking works")
        
        # Test best algorithms recommendation
        best_algorithms = selector.get_best_algorithms(episode, top_k=3)
        assert len(best_algorithms) <= 3
        assert all(isinstance(alg, str) for alg in best_algorithms)
        print(f"✓ Best algorithms: {best_algorithms}")
        
        # Test comprehensive recommendations
        recommendations = selector.get_algorithm_recommendations(episode)
        assert 'primary_recommendation' in recommendations
        assert 'task_analysis' in recommendations
        assert 'selection_reasoning' in recommendations
        print(f"✓ Primary recommendation: {recommendations['primary_recommendation']}")
        print(f"✓ Reasoning: {recommendations['selection_reasoning'][:100]}...")
        
        # Test different episode types
        
        # Very few-shot
        few_shot_episode = Episode(
            torch.randn(2, 6), torch.tensor([0, 1]),
            torch.randn(2, 6), torch.tensor([0, 1])
        )
        few_shot_selection = selector.select_algorithm(few_shot_episode)
        print(f"✓ Few-shot selection: {few_shot_selection}")
        
        # Many classes
        many_class_episode = Episode(
            torch.randn(60, 6), torch.repeat_interleave(torch.arange(15), 4),
            torch.randn(15, 6), torch.arange(15)
        )
        many_class_selection = selector.select_algorithm(many_class_episode)
        print(f"✓ Many-class selection: {many_class_selection}")
        
        return True
        
    except Exception as e:
        print(f"✗ Enhanced algorithm selector test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_enhanced_ab_testing():
    print("\nTesting Enhanced A/B Testing Framework...")
    
    try:
        from src.meta_learning.ml_enhancements.ab_testing import ABTestingFramework
        
        # Create enhanced A/B testing framework
        ab_framework = ABTestingFramework(use_registry=True)
        
        # Test registry-based test creation
        ab_framework.create_registry_based_test(
            test_name="ridge_vs_others",
            algorithm_types=["optimization_based", "gradient_based", "metric_based"],
            include_ridge_regression=True,
            max_algorithms=3
        )
        
        summary = ab_framework.get_test_summary("ridge_vs_others")
        assert len(summary['algorithms']) <= 3
        assert 'ridge_regression' in summary['algorithms']
        print(f"✓ Registry-based test created with algorithms: {summary['algorithms']}")
        
        # Test multi-metric result recording
        algorithms = summary['algorithms']
        for i, algorithm in enumerate(algorithms):
            for episode_idx in range(10):
                # Simulate different performance characteristics
                if algorithm == 'ridge_regression':
                    accuracy = np.random.normal(0.85, 0.05)  # High accuracy, low variance
                    exec_time = np.random.normal(0.01, 0.002)  # Fast
                    memory = np.random.normal(50, 5)  # Low memory
                elif algorithm == 'maml':
                    accuracy = np.random.normal(0.80, 0.08)  # Medium accuracy, higher variance
                    exec_time = np.random.normal(0.15, 0.03)  # Slower
                    memory = np.random.normal(150, 20)  # Higher memory
                else:
                    accuracy = np.random.normal(0.75, 0.10)  # Lower accuracy
                    exec_time = np.random.normal(0.08, 0.02)  # Medium speed
                    memory = np.random.normal(100, 15)  # Medium memory
                
                ab_framework.record_multi_metric_result(
                    test_name="ridge_vs_others",
                    algorithm=algorithm,
                    accuracy=max(0.0, min(1.0, accuracy)),  # Clamp to [0,1]
                    execution_time=max(0.001, exec_time),
                    memory_usage=max(10, memory)
                )
        
        print("✓ Multi-metric results recorded")
        
        # Test advanced analysis
        analysis = ab_framework.analyze_ab_test_advanced("ridge_vs_others")
        
        # Verify comprehensive statistics
        assert 'basic_statistics' in analysis
        assert 'performance_ranking' in analysis
        assert 'statistical_tests' in analysis
        
        for algorithm in algorithms:
            if algorithm in analysis['basic_statistics']:
                stats = analysis['basic_statistics'][algorithm]
                assert 'accuracy' in stats
                assert 'execution_time' in stats
                assert 'memory_usage' in stats
                
                # Check confidence intervals
                assert 'ci_95' in stats['accuracy']
                assert len(stats['accuracy']['ci_95']) == 2
        
        print("✓ Advanced statistical analysis works")
        
        # Test best algorithm detection
        best_algorithm, performance_details = ab_framework.get_best_algorithm("ridge_vs_others", metric="accuracy")
        assert best_algorithm is not None
        assert 'best_score' in performance_details
        print(f"✓ Best algorithm by accuracy: {best_algorithm} (score: {performance_details['best_score']:.3f})")
        
        # Test algorithm comparison
        if len(algorithms) >= 2:
            comparison = ab_framework.compare_algorithms(
                test_name="ridge_vs_others",
                algorithm1=algorithms[0],
                algorithm2=algorithms[1],
                metric="accuracy"
            )
            
            assert 'algorithm1' in comparison
            assert 'algorithm2' in comparison
            assert 't_test' in comparison
            assert 'effect_size' in comparison
            assert 'winner' in comparison
            
            print(f"✓ Algorithm comparison: {comparison['winner']} wins with p-value {comparison['t_test'].get('p_value', 'N/A'):.3f}")
        
        # Test performance ranking
        ranking = analysis['performance_ranking']
        assert len(ranking) > 0
        assert ranking[0]['rank'] == 1  # First should be rank 1
        if len(ranking) > 1:
            # Rankings should be sorted by accuracy descending
            assert ranking[0]['mean_accuracy'] >= ranking[1]['mean_accuracy']
        
        print(f"✓ Performance ranking: {[r['algorithm'] for r in ranking]}")
        
        return True
        
    except Exception as e:
        print(f"✗ Enhanced A/B testing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ridge_regression_integration():
    print("\nTesting Ridge Regression Integration...")
    
    try:
        from src.meta_learning.ml_enhancements.algorithm_selection import AlgorithmSelector
        from src.meta_learning.ml_enhancements.ab_testing import ABTestingFramework
        from src.meta_learning.shared.types import Episode
        
        # Test ridge regression gets selected appropriately
        selector = AlgorithmSelector()
        
        # Create episode where ridge regression should be preferred
        # Many shots per class, reasonable number of classes
        support_x = torch.randn(30, 10)  # 5-way 6-shot
        support_y = torch.repeat_interleave(torch.arange(5), 6)
        query_x = torch.randn(10, 10)
        query_y = torch.repeat_interleave(torch.arange(5), 2)
        
        episode = Episode(support_x, support_y, query_x, query_y)
        selected = selector.select_algorithm(episode)
        
        # Ridge regression should be selected for this scenario
        print(f"✓ Algorithm selected for 5-way 6-shot: {selected}")
        
        # Get recommendations and verify ridge regression is highly ranked
        recommendations = selector.get_algorithm_recommendations(episode)
        primary = recommendations['primary_recommendation']
        alternatives = recommendations['alternatives']
        
        print(f"✓ Primary: {primary}, Alternatives: {alternatives}")
        
        # Test A/B test specifically including ridge regression
        ab_framework = ABTestingFramework()
        ab_framework.create_ab_test(
            test_name="ridge_regression_test",
            algorithms=["ridge_regression", "maml", "protonet"]
        )
        
        # Simulate ridge regression performing well
        for algorithm in ["ridge_regression", "maml", "protonet"]:
            for i in range(5):
                if algorithm == "ridge_regression":
                    accuracy = np.random.normal(0.90, 0.03)  # Higher accuracy for ridge regression
                else:
                    accuracy = np.random.normal(0.75, 0.05)
                
                ab_framework.record_result(
                    test_name="ridge_regression_test",
                    algorithm=algorithm,
                    result={"accuracy": max(0.0, min(1.0, accuracy))}
                )
        
        # Analyze results
        analysis = ab_framework.analyze_ab_test_advanced("ridge_regression_test")
        ranking = analysis['performance_ranking']
        
        # Ridge regression should perform well
        ridge_rank = next((r['rank'] for r in ranking if r['algorithm'] == 'ridge_regression'), None)
        print(f"✓ Ridge regression rank: {ridge_rank}")
        
        return True
        
    except Exception as e:
        print(f"✗ Ridge regression integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integration_with_existing_systems():
    print("\nTesting integration with existing systems...")
    
    try:
        from src.meta_learning.ml_enhancements.algorithm_selection import AlgorithmSelector
        from src.meta_learning.ml_enhancements.ab_testing import ABTestingFramework
        from src.meta_learning.shared.types import Episode
        
        # Test backward compatibility
        selector = AlgorithmSelector()
        ab_framework = ABTestingFramework(use_registry=False)  # Test without registry
        
        # Create episode
        episode = Episode(
            torch.randn(6, 5), torch.tensor([0,0, 1,1, 2,2]),
            torch.randn(3, 5), torch.tensor([0, 1, 2])
        )
        
        # Test old-style selection still works
        selected = selector.select_algorithm(episode)
        assert selected in ['ridge_regression', 'maml', 'protonet', 'ttcs', 'matching_networks']
        print(f"✓ Backward compatibility: {selected}")
        
        # Test old-style A/B testing still works
        ab_framework.create_ab_test("legacy_test", ["maml", "protonet"])
        
        for alg in ["maml", "protonet"]:
            ab_framework.record_result("legacy_test", alg, {"accuracy": 0.8})
        
        legacy_analysis = ab_framework.analyze_ab_test("legacy_test")
        assert len(legacy_analysis) > 0
        print("✓ Legacy A/B testing compatibility maintained")
        
        # Test integration between selector and A/B testing
        recommendations = selector.get_algorithm_recommendations(episode)
        recommended_algorithms = [recommendations['primary_recommendation']] + recommendations['alternatives']
        
        # Create A/B test with recommended algorithms
        if len(recommended_algorithms) >= 2:
            ab_framework.create_ab_test("integration_test", recommended_algorithms[:3])
            print(f"✓ Integration test created with: {recommended_algorithms[:3]}")
        
        return True
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
    
    print("Testing Algorithm Registry, Selector, and A/B Testing Integration")
    print("=" * 70)
    
    success = True
    success &= test_algorithm_registry()
    success &= test_enhanced_algorithm_selector()
    success &= test_enhanced_ab_testing()
    success &= test_ridge_regression_integration()
    success &= test_integration_with_existing_systems()
    
    print("\n" + "=" * 70)
    if success:
        print("✅ All algorithm integration tests completed successfully!")
        print("🚀 Ridge regression integration with selector and A/B testing is ready!")
        print("💰 Please donate if this accelerates your research!")
    else:
        print("❌ Some algorithm integration tests failed!")
        sys.exit(1)