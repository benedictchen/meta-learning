"""
Test integration of Matching Networks with Algorithm Selector and A/B Testing.
Verifies that matching networks is properly registered and accessible.
"""

import pytest
import torch
from unittest.mock import Mock, patch

from meta_learning.ml_enhancements.algorithm_registry import algorithm_registry, AlgorithmType
from meta_learning.ml_enhancements.algorithm_selection import AlgorithmSelector
from meta_learning.ml_enhancements.ab_testing import ABTestingFramework
from meta_learning.shared.types import Episode


class TestMatchingNetworksIntegration:
    """Test Matching Networks integration with ML enhancements."""
    
    def test_matching_networks_registry_registration(self):
        """Test that matching networks is properly registered in the algorithm registry."""
        # Check if matching networks is in the registry
        all_algorithms = algorithm_registry.get_all_algorithms()
        assert "matching_networks" in all_algorithms
        
        # Get the algorithm metadata
        algorithm_metadata = algorithm_registry.get_algorithm_metadata("matching_networks")
        assert algorithm_metadata is not None
        assert algorithm_metadata.name == "matching_networks"
        assert "matching" in algorithm_metadata.description.lower() or "attention" in algorithm_metadata.description.lower()
        assert algorithm_metadata.implementation_module == "meta_learning.algorithms.matching_networks"
        assert algorithm_metadata.implementation_class == "MatchingNetworks"
    
    def test_algorithm_selector_matching_networks_support(self):
        """Test that AlgorithmSelector can work with matching networks."""
        selector = AlgorithmSelector()
        
        # Create a simple episode
        support_x = torch.randn(15, 64)  # 5-way 3-shot
        support_y = torch.arange(5).repeat_interleave(3)
        query_x = torch.randn(25, 64)    # 5-way 5-query
        query_y = torch.arange(5).repeat_interleave(5)
        episode = Episode(support_x, support_y, query_x, query_y)
        
        # Test heuristic selection - should not crash
        algorithm = selector.select_heuristic(episode)
        all_algorithms = algorithm_registry.get_all_algorithms()
        assert algorithm in all_algorithms
        
        # Test feature extraction - should work with any algorithm
        features = selector.extract_features(episode)
        assert isinstance(features, dict)
        assert "n_way" in features
        assert "k_shot" in features
    
    def test_ab_testing_framework_matching_networks_support(self):
        """Test that ABTestingFramework can set up tests with matching networks."""
        ab_framework = ABTestingFramework(use_registry=True)
        
        # Create A/B test with matching networks
        algorithms = ["protonet", "matching_networks"]
        ab_framework.create_ab_test(
            test_name="protonet_vs_matching_networks",
            algorithms=algorithms
        )
        
        assert "protonet_vs_matching_networks" in ab_framework.test_groups
        test_config = ab_framework.test_groups["protonet_vs_matching_networks"]
        assert test_config["algorithms"] == algorithms
        assert len(test_config["allocation_ratio"]) == 2
        
        # Verify allocation ratio sums to 1
        assert abs(sum(test_config["allocation_ratio"]) - 1.0) < 1e-6
    
    def test_algorithm_registry_create_algorithm_matching_networks(self):
        """Test creating matching networks instance through registry."""
        # This tests the registration integration
        all_algorithms = algorithm_registry.get_all_algorithms()
        assert "matching_networks" in all_algorithms
        
        # Try to get matching networks algorithm metadata
        algo_metadata = algorithm_registry.get_algorithm_metadata("matching_networks")
        assert algo_metadata.algorithm_type == AlgorithmType.METRIC_BASED
        assert "attention" in algo_metadata.description.lower() or "matching" in algo_metadata.description.lower()
        assert algo_metadata.default_config["embed_dim"] == 64  # Default parameter
    
    def test_performance_tracking_with_matching_networks(self):
        """Test that performance can be tracked for matching networks."""
        selector = AlgorithmSelector()
        
        # Create test episode
        support_x = torch.randn(10, 32)  # 5-way 2-shot
        support_y = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
        query_x = torch.randn(15, 32)
        query_y = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4])
        episode = Episode(support_x, support_y, query_x, query_y)
        
        # Track performance for matching networks
        algorithm = "matching_networks"
        accuracy = 0.85
        time_taken = 0.5
        
        selector.update_performance(algorithm, episode, accuracy, time_taken)
        
        # Check that performance was recorded
        assert algorithm in selector.performance_history
        assert len(selector.performance_history[algorithm]) == 1
        
        perf_record = selector.performance_history[algorithm][0]
        assert perf_record["accuracy"] == accuracy
        assert perf_record["time"] == time_taken
        assert "features" in perf_record
    
    def test_algorithm_recommendation_including_matching_networks(self):
        """Test that algorithm recommendation can suggest matching networks."""
        selector = AlgorithmSelector()
        
        # Add some performance history that would favor matching networks
        support_x = torch.randn(20, 64)  # Large feature dim
        support_y = torch.arange(5).repeat(4)  # 5-way 4-shot
        query_x = torch.randn(25, 64)
        query_y = torch.arange(5).repeat(5)
        episode = Episode(support_x, support_y, query_x, query_y)
        
        # Simulate matching networks performing well on complex tasks
        selector.update_performance("matching_networks", episode, 0.92, 0.8)
        selector.update_performance("protonet", episode, 0.78, 0.3)
        
        # Test performance-based selection 
        selected = selector.select_performance_based(episode)
        # Should prefer the algorithm with better accuracy
        # (This might be matching_networks based on our fake data)
        all_algorithms = algorithm_registry.get_all_algorithms()
        assert selected in all_algorithms
    
    def test_ab_test_with_matching_networks_execution(self):
        """Test running A/B test scenarios with matching networks."""
        ab_framework = ABTestingFramework(use_registry=True)
        
        # Set up test
        algorithms = ["protonet", "matching_networks"]  
        ab_framework.create_ab_test("test_matching", algorithms)
        
        # Simulate assignment (should not crash)
        assignment = ab_framework.assign_algorithm("test_matching", {"n_way": 5, "k_shot": 3})
        assert assignment in algorithms
        
        # Record some results
        ab_framework.record_result(
            "test_matching", 
            assignment, 
            {"accuracy": 0.85, "time": 0.5}
        )
        
        # Should have recorded data
        assert "test_matching" in ab_framework.test_groups
        test_data = ab_framework.test_groups["test_matching"]
        assert len(test_data["results"]) == 1


class TestMatchingNetworksAvailability:
    """Test that matching networks is available and accessible."""
    
    def test_algorithm_registry_has_matching_networks(self):
        """Test that matching networks appears in algorithm registry."""
        algorithms = algorithm_registry.get_all_algorithms()
        assert "matching_networks" in algorithms
        
    def test_matching_networks_algorithm_info_structure(self):
        """Test the structure of matching networks algorithm info."""
        metadata = algorithm_registry.get_algorithm_metadata("matching_networks")
        
        # Required fields
        assert hasattr(metadata, "name")
        assert hasattr(metadata, "algorithm_type")
        assert hasattr(metadata, "description")
        assert hasattr(metadata, "default_config")
        assert hasattr(metadata, "implementation_module")
        assert hasattr(metadata, "implementation_class")
        
        # Specific values
        assert metadata.name == "matching_networks"
        assert metadata.algorithm_type == AlgorithmType.METRIC_BASED
        assert "attention" in metadata.description.lower() or "matching" in metadata.description.lower()
    
    def test_matching_networks_default_parameters(self):
        """Test that matching networks has reasonable default parameters."""
        metadata = algorithm_registry.get_algorithm_metadata("matching_networks")
        params = metadata.default_config
        
        # Should have some reasonable defaults
        assert "embed_dim" in params
        assert isinstance(params["embed_dim"], int)
        assert params["embed_dim"] > 0
        
        # May have other parameters like attention settings
        if "use_attention" in params:
            assert isinstance(params["use_attention"], bool)
            
        if "fce" in params:
            assert isinstance(params["fce"], bool)