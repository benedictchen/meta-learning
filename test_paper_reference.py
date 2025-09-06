#!/usr/bin/env python3
"""
Test Paper Reference System Implementation
==========================================

Step 2: Confirm paper reference functionality is accurate before removing TODOs
Tests all implemented paper reference system methods for research accuracy.
"""

import sys
import os
sys.path.insert(0, 'src')

import logging
from src.meta_learning.validation.paper_validators.paper_reference import (
    PaperBenchmarkResult, 
    PaperEquation, 
    ResearchPaperReference,
    create_maml_paper_reference,
    create_protonet_paper_reference,
    create_meta_sgd_paper_reference,
    create_lora_paper_reference
)

def test_paper_benchmark_result_validation():
    """Test PaperBenchmarkResult dataclass validation."""
    print("🧪 Testing PaperBenchmarkResult validation...")
    
    # Test valid benchmark result
    valid_result = PaperBenchmarkResult(
        dataset_name="omniglot",
        task_config="5way_1shot", 
        reported_accuracy=98.7,
        confidence_interval=(98.4, 99.0),
        additional_metrics={"std_dev": 0.15}
    )
    
    print(f"✅ Valid result: {valid_result.dataset_name} {valid_result.task_config} = {valid_result.reported_accuracy}%")
    
    # Test accuracy range validation
    try:
        invalid_result = PaperBenchmarkResult(
            dataset_name="test",
            task_config="5way_1shot",
            reported_accuracy=150.0  # Invalid: > 100%
        )
        assert False, "Should have raised ValueError for invalid accuracy"
    except ValueError as e:
        print(f"✅ Accuracy validation working: {e}")
    
    # Test confidence interval validation
    try:
        invalid_ci = PaperBenchmarkResult(
            dataset_name="test",
            task_config="5way_1shot",
            reported_accuracy=50.0,
            confidence_interval=(60.0, 70.0)  # Invalid: accuracy outside interval
        )
        assert False, "Should have raised ValueError for invalid confidence interval"
    except ValueError as e:
        print(f"✅ Confidence interval validation working: {e}")
    
    print("✅ PaperBenchmarkResult validation test PASSED")

def test_paper_equation_validation():
    """Test PaperEquation dataclass validation.""" 
    print("🧪 Testing PaperEquation validation...")
    
    # Test valid equation
    valid_eq = PaperEquation(
        name="inner_update",
        latex_formula="θ' = θ - α * ∇_θ L(f_θ)",
        description="Inner loop parameter update",
        equation_number="Equation 1",
        page_reference=3
    )
    
    print(f"✅ Valid equation: {valid_eq.name} = {valid_eq.latex_formula}")
    
    # Test required field validation
    try:
        invalid_eq = PaperEquation(
            name="",  # Invalid: empty name
            latex_formula="θ = θ - α",
            description="Test equation"
        )
        assert False, "Should have raised ValueError for empty name"
    except ValueError as e:
        print(f"✅ Required field validation working: {e}")
    
    print("✅ PaperEquation validation test PASSED")

def test_research_paper_reference_basic():
    """Test basic ResearchPaperReference functionality."""
    print("🧪 Testing ResearchPaperReference basic operations...")
    
    # Create paper reference
    paper_ref = ResearchPaperReference(
        paper_title="Test Meta-Learning Paper",
        authors="Smith et al.",
        year=2024,
        venue="ICLR",
        arxiv_id="2024.12345"
    )
    
    # Verify basic attributes
    assert paper_ref.paper_title == "Test Meta-Learning Paper"
    assert paper_ref.authors == "Smith et al."
    assert paper_ref.year == 2024
    assert paper_ref.venue == "ICLR"
    assert paper_ref.arxiv_id == "2024.12345"
    
    # Verify citation generation
    expected_citation = "Smith et al. (2024). Test Meta-Learning Paper, ICLR"
    assert paper_ref.citation == expected_citation
    print(f"✅ Citation generated: {paper_ref.citation}")
    
    # Verify empty collections
    assert len(paper_ref.equations) == 0
    assert len(paper_ref.benchmark_results) == 0
    print("✅ Empty collections initialized correctly")
    
    print("✅ ResearchPaperReference basic test PASSED")

def test_equation_management():
    """Test equation addition and retrieval."""
    print("🧪 Testing equation management...")
    
    paper_ref = ResearchPaperReference(
        paper_title="Test Paper",
        authors="Author et al.", 
        year=2024
    )
    
    # Add equation
    test_equation = PaperEquation(
        name="test_update",
        latex_formula="x = x + Δx",
        description="Simple update rule"
    )
    paper_ref.add_equation(test_equation)
    
    # Verify addition
    assert len(paper_ref.equations) == 1
    assert "test_update" in paper_ref.list_equations()
    print(f"✅ Added equation: {test_equation.name}")
    
    # Retrieve equation
    retrieved_eq = paper_ref.get_equation("test_update")
    assert retrieved_eq is not None
    assert retrieved_eq.name == "test_update"
    assert retrieved_eq.latex_formula == "x = x + Δx"
    print(f"✅ Retrieved equation: {retrieved_eq.latex_formula}")
    
    # Test non-existent equation
    missing_eq = paper_ref.get_equation("nonexistent")
    assert missing_eq is None
    print("✅ Missing equation returns None correctly")
    
    print("✅ Equation management test PASSED")

def test_benchmark_result_management():
    """Test benchmark result addition and retrieval."""
    print("🧪 Testing benchmark result management...")
    
    paper_ref = ResearchPaperReference(
        paper_title="Test Paper",
        authors="Author et al.",
        year=2024
    )
    
    # Add benchmark result
    test_result = PaperBenchmarkResult(
        dataset_name="test_dataset", 
        task_config="5way_1shot",
        reported_accuracy=75.5,
        confidence_interval=(74.0, 77.0)
    )
    paper_ref.add_benchmark_result(test_result)
    
    # Verify addition
    assert len(paper_ref.benchmark_results) == 1
    result_keys = paper_ref.list_benchmark_results()
    assert "test_dataset_5way_1shot" in result_keys
    print(f"✅ Added benchmark result: {result_keys[0]}")
    
    # Retrieve result
    retrieved_result = paper_ref.get_benchmark_result("test_dataset", "5way_1shot")
    assert retrieved_result is not None
    assert retrieved_result.reported_accuracy == 75.5
    assert retrieved_result.confidence_interval == (74.0, 77.0)
    print(f"✅ Retrieved result: {retrieved_result.reported_accuracy}%")
    
    # Test non-existent result
    missing_result = paper_ref.get_benchmark_result("nonexistent", "5way_1shot")
    assert missing_result is None
    print("✅ Missing result returns None correctly")
    
    print("✅ Benchmark result management test PASSED")

def test_serialization():
    """Test dictionary serialization and deserialization."""
    print("🧪 Testing serialization...")
    
    # Create paper with data
    paper_ref = ResearchPaperReference(
        paper_title="Serialization Test Paper",
        authors="Test et al.",
        year=2024,
        venue="TestConf"
    )
    
    # Add equation and result
    paper_ref.add_equation(PaperEquation(
        name="test_eq",
        latex_formula="y = mx + b", 
        description="Linear equation"
    ))
    
    paper_ref.add_benchmark_result(PaperBenchmarkResult(
        dataset_name="test_data",
        task_config="test_config",
        reported_accuracy=80.0
    ))
    
    # Test serialization
    paper_dict = paper_ref.to_dict()
    assert paper_dict['paper_title'] == "Serialization Test Paper"
    assert paper_dict['authors'] == "Test et al."
    assert paper_dict['year'] == 2024
    assert 'equations' in paper_dict
    assert 'benchmark_results' in paper_dict
    print("✅ Serialization to dictionary successful")
    
    # Test deserialization
    restored_ref = ResearchPaperReference.from_dict(paper_dict)
    assert restored_ref.paper_title == "Serialization Test Paper"
    assert restored_ref.authors == "Test et al."
    assert len(restored_ref.equations) == 1
    assert len(restored_ref.benchmark_results) == 1
    
    # Verify equation restoration
    restored_eq = restored_ref.get_equation("test_eq")
    assert restored_eq is not None
    assert restored_eq.latex_formula == "y = mx + b"
    
    # Verify result restoration
    restored_result = restored_ref.get_benchmark_result("test_data", "test_config")
    assert restored_result is not None
    assert restored_result.reported_accuracy == 80.0
    
    print("✅ Deserialization from dictionary successful")
    print("✅ Serialization test PASSED")

def test_maml_paper_reference():
    """Test MAML paper reference factory function."""
    print("🧪 Testing MAML paper reference...")
    
    maml_ref = create_maml_paper_reference()
    
    # Verify paper metadata
    assert maml_ref.paper_title == "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks"
    assert maml_ref.authors == "Finn et al."
    assert maml_ref.year == 2017
    assert maml_ref.venue == "ICML"
    print(f"✅ MAML paper: {maml_ref.citation}")
    
    # Verify equations
    equations = maml_ref.list_equations()
    assert "inner_update" in equations
    assert "meta_objective" in equations 
    assert "meta_gradient" in equations
    print(f"✅ MAML equations: {len(equations)} equations loaded")
    
    # Verify specific equation
    inner_eq = maml_ref.get_equation("inner_update")
    assert inner_eq is not None
    assert "θ_i' = θ - α * ∇_θ L_Ti(f_θ)" in inner_eq.latex_formula
    print(f"✅ Inner update equation: {inner_eq.latex_formula}")
    
    # Verify benchmark results
    results = maml_ref.list_benchmark_results()
    assert "omniglot_5way_1shot" in results
    assert "omniglot_5way_5shot" in results
    assert "miniimagenet_5way_1shot" in results
    assert "miniimagenet_5way_5shot" in results
    print(f"✅ MAML benchmarks: {len(results)} results loaded")
    
    # Verify specific result
    omniglot_1shot = maml_ref.get_benchmark_result("omniglot", "5way_1shot")
    assert omniglot_1shot is not None
    assert omniglot_1shot.reported_accuracy == 98.7
    assert omniglot_1shot.confidence_interval == (98.4, 99.0)
    print(f"✅ Omniglot 5-way 1-shot: {omniglot_1shot.reported_accuracy}%")
    
    print("✅ MAML paper reference test PASSED")

def test_other_paper_references():
    """Test other paper reference factory functions."""
    print("🧪 Testing other paper references...")
    
    # Test ProtoNet reference
    protonet_ref = create_protonet_paper_reference()
    assert protonet_ref.authors == "Snell et al."
    assert protonet_ref.year == 2017
    assert len(protonet_ref.list_equations()) > 0
    print(f"✅ ProtoNet reference: {protonet_ref.citation}")
    
    # Test Meta-SGD reference
    meta_sgd_ref = create_meta_sgd_paper_reference()
    assert meta_sgd_ref.authors == "Li et al."
    assert meta_sgd_ref.year == 2017
    assert len(meta_sgd_ref.list_equations()) > 0
    print(f"✅ Meta-SGD reference: {meta_sgd_ref.citation}")
    
    # Test LoRA reference
    lora_ref = create_lora_paper_reference()
    assert lora_ref.authors == "Hu et al."
    assert lora_ref.year == 2021
    assert len(lora_ref.list_equations()) > 0
    print(f"✅ LoRA reference: {lora_ref.citation}")
    
    print("✅ Other paper references test PASSED")

def test_logging_functionality():
    """Test logging functionality in paper references."""
    print("🧪 Testing logging functionality...")
    
    # Set up logging to capture messages
    import io
    log_capture = io.StringIO()
    handler = logging.StreamHandler(log_capture)
    handler.setLevel(logging.DEBUG)
    
    paper_ref = ResearchPaperReference(
        paper_title="Logging Test",
        authors="Logger et al.",
        year=2024
    )
    
    # Add handler to paper's logger
    paper_ref.logger.addHandler(handler)
    paper_ref.logger.setLevel(logging.DEBUG)
    
    # Add equation - should log debug message
    test_eq = PaperEquation(
        name="log_test",
        latex_formula="log(x) = y",
        description="Logging test equation"
    )
    paper_ref.add_equation(test_eq)
    
    # Add benchmark result - should log debug message
    test_result = PaperBenchmarkResult(
        dataset_name="log_dataset",
        task_config="log_config", 
        reported_accuracy=85.0
    )
    paper_ref.add_benchmark_result(test_result)
    
    # Check if logging occurred
    log_contents = log_capture.getvalue()
    assert "Added equation 'log_test'" in log_contents
    assert "Added benchmark result 'log_dataset_log_config'" in log_contents
    
    print("✅ Logging messages captured correctly")
    print("✅ Logging functionality test PASSED")

def main():
    """Run all paper reference system tests."""
    print("🚀 Starting Paper Reference System Implementation Tests")
    print("=" * 60)
    
    try:
        test_paper_benchmark_result_validation()
        test_paper_equation_validation()
        test_research_paper_reference_basic()
        test_equation_management()
        test_benchmark_result_management()
        test_serialization()
        test_maml_paper_reference()
        test_other_paper_references()
        test_logging_functionality()
        
        print("=" * 60)
        print("🎉 ALL PAPER REFERENCE SYSTEM TESTS PASSED!")
        print("✅ Implementation follows research validation requirements")
        print("✅ Mathematical accuracy and data integrity confirmed")
        print("✅ Ready to remove TODO comments")
        
        return True
        
    except Exception as e:
        print("=" * 60)
        print(f"❌ TEST FAILED: {e}")
        print("❌ Implementation needs fixes before removing TODOs")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)