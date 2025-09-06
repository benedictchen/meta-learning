"""
Tests for the warnings system module.
"""

import pytest
import warnings
from unittest.mock import patch

from meta_learning.warnings_system import (
    WarningLevel, ConfigurationWarning, MetaLearningWarnings,
    get_warning_system, warn_if_suboptimal_config
)


class TestWarningLevel:
    """Test warning level enumeration."""
    
    def test_warning_levels(self):
        """Test warning level values."""
        assert WarningLevel.INFO.value == "info"
        assert WarningLevel.WARNING.value == "warning"
        assert WarningLevel.ERROR.value == "error"


class TestConfigurationWarning:
    """Test configuration warning dataclass."""
    
    def test_configuration_warning_creation(self):
        """Test creating configuration warnings."""
        warning = ConfigurationWarning(
            level=WarningLevel.WARNING,
            category="test_category",
            message="Test message",
            suggestion="Test suggestion",
            parameter="test_param",
            value=42
        )
        
        assert warning.level == WarningLevel.WARNING
        assert warning.category == "test_category"
        assert warning.message == "Test message"
        assert warning.suggestion == "Test suggestion"
        assert warning.parameter == "test_param"
        assert warning.value == 42
    
    def test_configuration_warning_minimal(self):
        """Test creating minimal configuration warning."""
        warning = ConfigurationWarning(
            level=WarningLevel.INFO,
            category="minimal",
            message="Minimal warning"
        )
        
        assert warning.level == WarningLevel.INFO
        assert warning.category == "minimal"
        assert warning.message == "Minimal warning"
        assert warning.suggestion is None
        assert warning.parameter is None
        assert warning.value is None


class TestMetaLearningWarnings:
    """Test meta-learning warnings system."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.warning_system = MetaLearningWarnings(enabled=True)
    
    def test_initialization(self):
        """Test warning system initialization."""
        assert self.warning_system.enabled == True
        assert len(self.warning_system._warned_configurations) == 0
    
    def test_disable_enable(self):
        """Test disabling and enabling warnings."""
        self.warning_system.disable()
        assert self.warning_system.enabled == False
        
        self.warning_system.enable()
        assert self.warning_system.enabled == True
    
    def test_reset(self):
        """Test resetting warning history."""
        self.warning_system._warned_configurations.add("test_key")
        assert len(self.warning_system._warned_configurations) == 1
        
        self.warning_system.reset()
        assert len(self.warning_system._warned_configurations) == 0


class TestFewShotWarnings:
    """Test few-shot configuration warnings."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.warning_system = MetaLearningWarnings(enabled=True)
    
    def test_challenging_few_shot_warning(self):
        """Test warning for challenging few-shot configuration."""
        warnings_list = self.warning_system.warn_if_suboptimal_few_shot(15, 1)
        
        assert len(warnings_list) == 1
        warning = warnings_list[0]
        assert warning.level == WarningLevel.WARNING
        assert warning.category == "few_shot_difficulty"
        assert "15-way 1-shot is very challenging" in warning.message
    
    def test_high_n_way_warning(self):
        """Test warning for high n_way."""
        warnings_list = self.warning_system.warn_if_suboptimal_few_shot(25, 2)
        
        assert len(warnings_list) == 1
        warning = warnings_list[0]
        assert warning.level == WarningLevel.WARNING
        assert warning.category == "high_n_way"
        assert "n_way=25 is unusually high" in warning.message
    
    def test_high_k_shot_warning(self):
        """Test warning for high k_shot."""
        warnings_list = self.warning_system.warn_if_suboptimal_few_shot(5, 15)
        
        assert len(warnings_list) == 1
        warning = warnings_list[0]
        assert warning.level == WarningLevel.INFO
        assert warning.category == "high_k_shot"
        assert "k_shot=15 is higher than typical" in warning.message
    
    def test_small_query_set_warning(self):
        """Test warning for small query set."""
        warnings_list = self.warning_system.warn_if_suboptimal_few_shot(5, 2, n_query=3)
        
        assert len(warnings_list) == 1
        warning = warnings_list[0]
        assert warning.level == WarningLevel.WARNING
        assert warning.category == "small_query_set"
        assert "n_query=3 may give unreliable accuracy estimates" in warning.message
    
    def test_multiple_warnings(self):
        """Test multiple warnings for same configuration."""
        warnings_list = self.warning_system.warn_if_suboptimal_few_shot(25, 15, n_query=2)
        
        # Should have multiple warnings
        assert len(warnings_list) >= 2
        categories = [w.category for w in warnings_list]
        assert "high_n_way" in categories
        assert "high_k_shot" in categories
        assert "small_query_set" in categories
    
    def test_no_warnings_optimal_config(self):
        """Test no warnings for optimal configuration."""
        warnings_list = self.warning_system.warn_if_suboptimal_few_shot(5, 2, n_query=15)
        
        assert len(warnings_list) == 0


class TestDistanceConfigWarnings:
    """Test distance configuration warnings."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.warning_system = MetaLearningWarnings(enabled=True)
    
    def test_low_cosine_temperature_warning(self):
        """Test warning for low temperature with cosine distance."""
        warnings_list = self.warning_system.warn_if_suboptimal_distance_config("cosine", 0.05)
        
        assert len(warnings_list) == 1
        warning = warnings_list[0]
        assert warning.level == WarningLevel.WARNING
        assert warning.category == "low_temperature"
        assert "tau=0.05 is very low for cosine distance" in warning.message
    
    def test_high_cosine_temperature_warning(self):
        """Test warning for high temperature with cosine distance."""
        warnings_list = self.warning_system.warn_if_suboptimal_distance_config("cosine", 15.0)
        
        assert len(warnings_list) == 1
        warning = warnings_list[0]
        assert warning.level == WarningLevel.INFO
        assert warning.category == "high_temperature"
        assert "tau=15.0 is very high for cosine distance" in warning.message
    
    def test_very_low_euclidean_temperature_error(self):
        """Test error for extremely low temperature with squared Euclidean."""
        warnings_list = self.warning_system.warn_if_suboptimal_distance_config("sqeuclidean", 0.005)
        
        assert len(warnings_list) == 1
        warning = warnings_list[0]
        assert warning.level == WarningLevel.ERROR
        assert warning.category == "numerical_instability"
        assert "tau=0.005 is extremely low" in warning.message
    
    def test_high_euclidean_temperature_warning(self):
        """Test warning for high temperature with squared Euclidean."""
        warnings_list = self.warning_system.warn_if_suboptimal_distance_config("sqeuclidean", 150.0)
        
        assert len(warnings_list) == 1
        warning = warnings_list[0]
        assert warning.level == WarningLevel.WARNING
        assert warning.category == "high_temperature"
        assert "tau=150.0 is very high for squared Euclidean" in warning.message
    
    def test_optimal_distance_config(self):
        """Test no warnings for optimal distance configuration."""
        warnings_list = self.warning_system.warn_if_suboptimal_distance_config("cosine", 1.0)
        assert len(warnings_list) == 0
        
        warnings_list = self.warning_system.warn_if_suboptimal_distance_config("sqeuclidean", 2.0)
        assert len(warnings_list) == 0


class TestMAMLConfigWarnings:
    """Test MAML configuration warnings."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.warning_system = MetaLearningWarnings(enabled=True)
    
    def test_high_inner_lr_warning(self):
        """Test warning for high inner learning rate."""
        warnings_list = self.warning_system.warn_if_suboptimal_maml_config(0.2, 3, 0.001)
        
        assert len(warnings_list) == 1
        warning = warnings_list[0]
        assert warning.level == WarningLevel.WARNING
        assert warning.category == "high_inner_lr"
        assert "inner_lr=0.2 is very high for MAML" in warning.message
    
    def test_low_inner_lr_warning(self):
        """Test warning for low inner learning rate."""
        warnings_list = self.warning_system.warn_if_suboptimal_maml_config(1e-6, 3, 0.001)
        
        assert len(warnings_list) == 1
        warning = warnings_list[0]
        assert warning.level == WarningLevel.WARNING
        assert warning.category == "low_inner_lr"
        assert "inner_lr=1e-06 is very low for MAML" in warning.message
    
    def test_high_inner_steps_warning(self):
        """Test warning for high number of inner steps."""
        warnings_list = self.warning_system.warn_if_suboptimal_maml_config(0.01, 15, 0.001)
        
        assert len(warnings_list) == 1
        warning = warnings_list[0]
        assert warning.level == WarningLevel.INFO
        assert warning.category == "many_inner_steps"
        assert "inner_steps=15 is higher than typical" in warning.message
    
    def test_no_adaptation_error(self):
        """Test error for no inner steps."""
        warnings_list = self.warning_system.warn_if_suboptimal_maml_config(0.01, 0, 0.001)
        
        assert len(warnings_list) == 1
        warning = warnings_list[0]
        assert warning.level == WarningLevel.ERROR
        assert warning.category == "no_adaptation"
        assert "inner_steps=0 means no adaptation" in warning.message
    
    def test_lr_ratio_warning(self):
        """Test warning for high learning rate ratio."""
        warnings_list = self.warning_system.warn_if_suboptimal_maml_config(0.001, 3, 0.05)
        
        assert len(warnings_list) == 1
        warning = warnings_list[0]
        assert warning.level == WarningLevel.WARNING
        assert warning.category == "lr_ratio_imbalance"
        assert "outer_lr/inner_lr ratio is very high" in warning.message
    
    def test_optimal_maml_config(self):
        """Test no warnings for optimal MAML configuration."""
        warnings_list = self.warning_system.warn_if_suboptimal_maml_config(0.01, 3, 0.001)
        assert len(warnings_list) == 0


class TestModelConfigWarnings:
    """Test model configuration warnings."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.warning_system = MetaLearningWarnings(enabled=True)
    
    def test_batchnorm_warning(self):
        """Test warning for BatchNorm layers."""
        model_info = {"has_batchnorm": True, "parameter_count": 1000}
        warnings_list = self.warning_system.warn_if_suboptimal_model_config(model_info)
        
        assert len(warnings_list) == 1
        warning = warnings_list[0]
        assert warning.level == WarningLevel.WARNING
        assert warning.category == "batchnorm_episodic"
        assert "Model contains BatchNorm layers" in warning.message
    
    def test_no_parameters_warning(self):
        """Test warning for model with no parameters."""
        model_info = {"has_batchnorm": False, "parameter_count": 0}
        warnings_list = self.warning_system.warn_if_suboptimal_model_config(model_info)
        
        assert len(warnings_list) == 1
        warning = warnings_list[0]
        assert warning.level == WarningLevel.WARNING
        assert warning.category == "no_parameters"
        assert "Model has no trainable parameters" in warning.message
    
    def test_large_model_warning(self):
        """Test warning for large model."""
        model_info = {"has_batchnorm": False, "parameter_count": 20000000}
        warnings_list = self.warning_system.warn_if_suboptimal_model_config(model_info)
        
        assert len(warnings_list) == 1
        warning = warnings_list[0]
        assert warning.level == WarningLevel.INFO
        assert warning.category == "large_model"
        assert "20,000,000 parameters" in warning.message
    
    def test_optimal_model_config(self):
        """Test no warnings for optimal model configuration."""
        model_info = {"has_batchnorm": False, "parameter_count": 50000}
        warnings_list = self.warning_system.warn_if_suboptimal_model_config(model_info)
        assert len(warnings_list) == 0


class TestWarningOnce:
    """Test warn_once functionality."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.warning_system = MetaLearningWarnings(enabled=True)
    
    @patch('warnings.warn')
    def test_warn_once_first_time(self, mock_warn):
        """Test that warning is emitted the first time."""
        self.warning_system.warn_once("test_key", "Test message")
        mock_warn.assert_called_once()
    
    @patch('warnings.warn')
    def test_warn_once_subsequent_times(self, mock_warn):
        """Test that warning is not emitted subsequent times."""
        self.warning_system.warn_once("test_key", "Test message")
        self.warning_system.warn_once("test_key", "Test message")
        
        # Should only be called once
        assert mock_warn.call_count == 1
    
    @patch('warnings.warn')
    def test_warn_once_disabled(self, mock_warn):
        """Test that warning is not emitted when disabled."""
        self.warning_system.disable()
        self.warning_system.warn_once("test_key", "Test message")
        mock_warn.assert_not_called()


class TestGlobalWarningSystem:
    """Test global warning system access."""
    
    def test_get_warning_system(self):
        """Test getting global warning system."""
        warning_system = get_warning_system()
        assert isinstance(warning_system, MetaLearningWarnings)
        
        # Should return same instance
        warning_system2 = get_warning_system()
        assert warning_system is warning_system2


class TestConvenienceFunction:
    """Test convenience warning function."""
    
    def test_warn_if_suboptimal_config_few_shot(self):
        """Test convenience function with few-shot config."""
        warnings_list = warn_if_suboptimal_config(n_way=15, k_shot=1)
        
        assert len(warnings_list) == 1
        assert warnings_list[0].category == "few_shot_difficulty"
    
    def test_warn_if_suboptimal_config_distance(self):
        """Test convenience function with distance config."""
        warnings_list = warn_if_suboptimal_config(distance="cosine", tau=0.05)
        
        assert len(warnings_list) == 1
        assert warnings_list[0].category == "low_temperature"
    
    def test_warn_if_suboptimal_config_maml(self):
        """Test convenience function with MAML config."""
        warnings_list = warn_if_suboptimal_config(
            inner_lr=0.2, inner_steps=3, outer_lr=0.001
        )
        
        assert len(warnings_list) == 1
        assert warnings_list[0].category == "high_inner_lr"
    
    def test_warn_if_suboptimal_config_model(self):
        """Test convenience function with model config."""
        model_info = {"has_batchnorm": True, "parameter_count": 1000}
        warnings_list = warn_if_suboptimal_config(model_info=model_info)
        
        assert len(warnings_list) == 1
        assert warnings_list[0].category == "batchnorm_episodic"
    
    def test_warn_if_suboptimal_config_multiple(self):
        """Test convenience function with multiple configs."""
        warnings_list = warn_if_suboptimal_config(
            n_way=15, k_shot=1,
            distance="cosine", tau=0.05,
            inner_lr=0.2, inner_steps=3, outer_lr=0.001
        )
        
        # Should have warnings from multiple categories
        assert len(warnings_list) >= 3
        categories = [w.category for w in warnings_list]
        assert "few_shot_difficulty" in categories
        assert "low_temperature" in categories
        assert "high_inner_lr" in categories