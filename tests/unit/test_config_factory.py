"""
Tests for Configuration Factory Solutions
========================================

Tests all configuration factory functions and comprehensive configuration system.
"""

import pytest
from meta_learning.meta_learning_modules.config_factory import (
    ComprehensiveMetaLearningConfig,
    create_all_fixme_solutions_config,
    create_research_accurate_config,
    create_performance_optimized_config,
    create_specific_solution_config,
    create_modular_config,
    create_comprehensive_component_config,
    create_educational_config,
    get_available_solutions,
    validate_config,
    print_solution_summary
)


class TestComprehensiveMetaLearningConfig:
    """Test the master configuration class."""
    
    def test_config_creation(self):
        """Test creating comprehensive configuration."""
        config = ComprehensiveMetaLearningConfig()
        
        # Test default values
        assert config.global_seed == 42
        assert config.device == "auto"
        assert config.verbose == True
        
        # Test optional configurations are None by default
        assert config.test_time_compute is None
        assert config.prototypical is None
        assert config.uncertainty is None
        
    def test_config_customization(self):
        """Test customizing configuration values."""
        config = ComprehensiveMetaLearningConfig(
            global_seed=123,
            device="cuda",
            verbose=False
        )
        
        assert config.global_seed == 123
        assert config.device == "cuda"
        assert config.verbose == False


class TestFactoryFunctions:
    """Test configuration factory functions."""
    
    def test_create_all_fixme_solutions_config(self):
        """Test factory function that enables ALL research solutions."""
        config = create_all_fixme_solutions_config()
        
        # Should have all major configurations
        assert config.test_time_compute is not None
        assert config.prototypical is not None
        assert config.matching is not None
        assert config.relation is not None
        assert config.continual_meta is not None
        assert config.online_meta is not None
        assert config.maml is not None
        assert config.evaluation is not None
        
        # New component configurations should be enabled
        assert config.uncertainty is not None
        assert config.hierarchical is not None
        assert config.task_adaptive is not None
        assert config.dataset_loading is not None
        assert config.task_difficulty is not None
        
        # Check specific settings
        assert config.test_time_compute.compute_strategy == "hybrid"
        assert config.test_time_compute.use_process_reward == True
        assert config.test_time_compute.use_test_time_training == True
        assert config.test_time_compute.use_chain_of_thought == True
        
        assert config.uncertainty.method == "monte_carlo_dropout"
        assert config.hierarchical.method == "multi_level"
        assert config.task_adaptive.method == "attention_based"
        assert config.dataset_loading.method == "torchmeta"
        assert config.task_difficulty.method == "intra_class_variance"
        
    def test_create_research_accurate_config(self):
        """Test factory function for research accuracy."""
        config = create_research_accurate_config()
        
        # Should prioritize research accuracy
        assert config.test_time_compute.compute_strategy == "snell2024"
        assert config.prototypical.use_original_implementation == True
        assert config.continual_meta.ewc_method == "diagonal"
        assert config.maml.maml_variant == "maml"
        assert config.evaluation.confidence_interval_method == "t_distribution"
        
    def test_create_performance_optimized_config(self):
        """Test factory function for performance optimization."""
        config = create_performance_optimized_config()
        
        # Should prioritize performance
        assert config.test_time_compute.compute_strategy == "basic"
        assert config.test_time_compute.max_compute_budget == 100
        assert config.prototypical.protonet_variant == "simple"
        assert config.maml.maml_variant == "fomaml"
        assert config.evaluation.num_episodes == 300  # Reduced for speed
        
    def test_create_specific_solution_config(self):
        """Test factory function for specific solutions."""
        solutions = [
            "monte_carlo_dropout",
            "attention_based_adaptation", 
            "torchmeta_loading",
            "intra_class_difficulty"
        ]
        
        config = create_specific_solution_config(solutions)
        
        # Should enable only specified solutions
        assert config.uncertainty is not None
        assert config.uncertainty.method == "monte_carlo_dropout"
        
        assert config.task_adaptive is not None
        assert config.task_adaptive.method == "attention_based"
        
        assert config.dataset_loading is not None
        assert config.dataset_loading.method == "torchmeta"
        
        assert config.task_difficulty is not None
        assert config.task_difficulty.method == "intra_class_variance"
        
    def test_create_specific_solution_config_unknown(self):
        """Test specific solution config with unknown solution."""
        solutions = ["unknown_solution", "monte_carlo_dropout"]
        
        # Should handle unknown solutions gracefully (print warning)
        config = create_specific_solution_config(solutions)
        
        # Should still configure known solutions
        assert config.uncertainty is not None
        assert config.uncertainty.method == "monte_carlo_dropout"
        
    def test_create_modular_config(self):
        """Test modular configuration factory."""
        config = create_modular_config(
            test_time_compute="hybrid",
            few_shot_method="prototypical",
            continual_method="ewc",
            maml_variant="fomaml",
            evaluation_method="bootstrap"
        )
        
        assert config.test_time_compute.compute_strategy == "hybrid"
        assert config.prototypical is not None
        assert config.continual_meta.memory_consolidation_method == "ewc"
        assert config.maml.maml_variant == "fomaml"
        assert config.evaluation.confidence_interval_method == "bootstrap"
        
    def test_create_comprehensive_component_config(self):
        """Test comprehensive component configuration factory."""
        config = create_comprehensive_component_config()
        
        # Should have all new components configured
        assert config.uncertainty is not None
        assert config.hierarchical is not None
        assert config.task_adaptive is not None
        assert config.dataset_loading is not None
        assert config.task_difficulty is not None
        
        # Should have basic configs for completeness
        assert config.prototypical is not None
        assert config.evaluation is not None
        
        # Check specific settings
        assert config.uncertainty.method == "monte_carlo_dropout"
        assert config.hierarchical.method == "multi_level"
        assert config.task_adaptive.method == "attention_based"
        
    def test_create_educational_config(self):
        """Test educational configuration factory."""
        config = create_educational_config()
        
        # Should be simplified but working
        assert config.test_time_compute.compute_strategy == "basic"
        assert config.test_time_compute.max_compute_budget == 50
        assert config.prototypical.protonet_variant == "simple"
        assert config.maml.inner_steps == 3
        assert config.evaluation.num_episodes == 100


class TestConfigurationValidation:
    """Test configuration validation system."""
    
    def test_validate_config_no_issues(self):
        """Test validation with no issues."""
        config = create_educational_config()
        issues = validate_config(config)
        
        assert 'warnings' in issues
        assert 'errors' in issues
        assert isinstance(issues['warnings'], list)
        assert isinstance(issues['errors'], list)
        
    def test_validate_config_warnings(self):
        """Test validation with potential warnings."""
        config = create_all_fixme_solutions_config()
        issues = validate_config(config)
        
        # Should detect potential performance implications
        assert isinstance(issues['warnings'], list)
        
        # Check for specific warnings about expensive operations
        warning_messages = ' '.join(issues['warnings'])
        if 'exact Fisher Information' in warning_messages:
            assert 'expensive' in warning_messages.lower()


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_get_available_solutions(self):
        """Test getting available solutions."""
        solutions = get_available_solutions()
        
        assert isinstance(solutions, dict)
        
        # Should have all major modules
        expected_modules = [
            "test_time_compute",
            "few_shot_learning",
            "uncertainty_components",
            "hierarchical_components", 
            "adaptive_components",
            "dataset_loading",
            "difficulty_estimation",
            "continual_meta_learning",
            "maml_variants",
            "utils"
        ]
        
        for module in expected_modules:
            assert module in solutions
            assert isinstance(solutions[module], list)
            assert len(solutions[module]) > 0
            
    def test_print_solution_summary(self):
        """Test printing solution summary."""
        # Should not raise any errors
        try:
            print_solution_summary()
            success = True
        except Exception:
            success = False
            
        assert success


class TestConfigurationIntegration:
    """Integration tests for configuration system."""
    
    def test_all_factory_functions_work(self):
        """Test that all factory functions produce valid configs."""
        factories = [
            create_all_fixme_solutions_config,
            create_research_accurate_config,
            create_performance_optimized_config,
            create_comprehensive_component_config,
            create_educational_config,
        ]
        
        for factory in factories:
            config = factory()
            assert isinstance(config, ComprehensiveMetaLearningConfig)
            
            # Validation should not raise errors
            issues = validate_config(config)
            assert isinstance(issues, dict)
            
    def test_specific_solutions_comprehensive(self):
        """Test specific solutions with comprehensive set."""
        all_solutions = get_available_solutions()
        
        # Collect all solution names
        solution_names = []
        for module_solutions in all_solutions.values():
            solution_names.extend(module_solutions)
            
        # Test with subset of solutions
        test_solutions = solution_names[:5]  # First 5 solutions
        config = create_specific_solution_config(test_solutions)
        
        assert isinstance(config, ComprehensiveMetaLearningConfig)
        
    def test_modular_config_combinations(self):
        """Test various modular configuration combinations."""
        test_combinations = [
            {
                "test_time_compute": "basic",
                "few_shot_method": "prototypical",
                "evaluation_method": "t_distribution"
            },
            {
                "test_time_compute": "hybrid",
                "continual_method": "ewc",
                "maml_variant": "maml"
            },
            {
                "few_shot_method": "matching",
                "evaluation_method": "bootstrap"
            }
        ]
        
        for params in test_combinations:
            config = create_modular_config(**params)
            assert isinstance(config, ComprehensiveMetaLearningConfig)
            
            issues = validate_config(config)
            assert isinstance(issues, dict)
            
    def test_config_attribute_consistency(self):
        """Test that configuration attributes are consistent."""
        config = create_all_fixme_solutions_config()
        
        # Test that configurations have expected attributes
        if config.uncertainty is not None:
            assert hasattr(config.uncertainty, 'method')
            assert hasattr(config.uncertainty, 'num_samples')
            
        if config.hierarchical is not None:
            assert hasattr(config.hierarchical, 'method')
            assert hasattr(config.hierarchical, 'num_levels')
            
        if config.task_adaptive is not None:
            assert hasattr(config.task_adaptive, 'method')
            assert hasattr(config.task_adaptive, 'attention_heads')
            
        if config.dataset_loading is not None:
            assert hasattr(config.dataset_loading, 'method')
            assert hasattr(config.dataset_loading, 'fallback_to_synthetic')
            
        if config.task_difficulty is not None:
            assert hasattr(config.task_difficulty, 'method')
            assert hasattr(config.task_difficulty, 'fallback_method')
            
    def test_config_serialization_compatibility(self):
        """Test that configurations can be converted to dict (for serialization)."""
        config = create_all_fixme_solutions_config()
        
        # Should be able to access as dict-like
        config_dict = config.__dict__
        assert isinstance(config_dict, dict)
        
        # Should have expected keys
        expected_keys = [
            'test_time_compute', 'prototypical', 'uncertainty', 
            'hierarchical', 'task_adaptive', 'dataset_loading', 
            'task_difficulty', 'global_seed', 'device', 'verbose'
        ]
        
        for key in expected_keys:
            assert key in config_dict