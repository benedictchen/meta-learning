"""
Meta-Learning Validation Package
================================

This package provides research validation tools and paper compliance checking
for meta-learning implementations.

MODULAR STRUCTURE:
- paper_validators/: Research paper validation and reference system
- honest_status_reporter.py: Implementation status reporting
- research_accuracy_validator.py: Research accuracy validation
- paper_metadata.py: Paper metadata management
"""

# Import core validation functions from the sibling validation.py module
# These need to be re-exported since the validation/ directory shadows validation.py
import importlib.util
import os

# Get the path to the validation.py file
validation_py_path = os.path.join(os.path.dirname(__file__), '..', 'validation.py')
validation_py_path = os.path.abspath(validation_py_path)

# Load the validation module directly
spec = importlib.util.spec_from_file_location("meta_learning_validation", validation_py_path)
validation_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validation_module)

# Import the required functions
ValidationError = validation_module.ValidationError
ConfigurationWarning = validation_module.ConfigurationWarning
validate_episode_tensors = validation_module.validate_episode_tensors
validate_few_shot_configuration = validation_module.validate_few_shot_configuration
validate_distance_metric = validation_module.validate_distance_metric
validate_temperature_parameter = validation_module.validate_temperature_parameter  
validate_learning_rate = validation_module.validate_learning_rate

# Import validation components as they become available
from .paper_validators.paper_reference import (
    PaperBenchmarkResult,
    PaperEquation, 
    ResearchPaperReference,
    create_maml_paper_reference,
    create_protonet_paper_reference,
    create_meta_sgd_paper_reference,
    create_lora_paper_reference
)

__all__ = [
    # Core validation functions (re-exported from validation.py)
    'ValidationError', 
    'ConfigurationWarning', 
    'validate_episode_tensors',
    'validate_few_shot_configuration', 
    'validate_distance_metric', 
    'validate_temperature_parameter', 
    'validate_learning_rate',
    
    # Paper validation components
    'PaperBenchmarkResult',
    'PaperEquation',
    'ResearchPaperReference', 
    'create_maml_paper_reference',
    'create_protonet_paper_reference',
    'create_meta_sgd_paper_reference',
    'create_lora_paper_reference'
]

# TODO: Add other validation components as they are implemented
# from .paper_validators.validation_utils import ValidationHelpers
# from .paper_validators.maml_validator import MAMLPaperValidator