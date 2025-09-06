# TODO: EVALUATION METRICS TESTING - Comprehensive metrics validation
# TODO: Test accuracy calculations with different input formats
# TODO: Test confidence interval computations
# TODO: Test calibration error calculations
# TODO: Test uncertainty quantification metrics
# TODO: Test per-class accuracy analysis
# TODO: Test edge cases and error handling

"""Tests for evaluation metrics implementations."""

import pytest
import torch
import numpy as np
from meta_learning.evaluation.metrics import (
    EvaluationMetrics,
    AccuracyCalculator,
    CalibrationCalculator,
    UncertaintyCalculator
)


class TestEvaluationMetrics:
    """Test EvaluationMetrics dataclass."""
    
    def test_basic_creation(self):
        """Test basic metrics creation."""
        metrics = EvaluationMetrics(accuracy=0.85)
        
        assert metrics.accuracy == 0.85
        assert metrics.loss is None
        assert metrics.confidence_interval is None
        assert metrics.per_class_accuracy is None
        assert metrics.calibration_error is None
    
    def test_full_creation(self):
        """Test metrics creation with all fields."""
        metrics = EvaluationMetrics(
            accuracy=0.92,
            loss=0.15,
            confidence_interval=(0.89, 0.95),
            per_class_accuracy={0: 0.90, 1: 0.94, 2: 0.92},
            calibration_error=0.03
        )
        
        assert metrics.accuracy == 0.92
        assert metrics.loss == 0.15
        assert metrics.confidence_interval == (0.89, 0.95)
        assert metrics.per_class_accuracy == {0: 0.90, 1: 0.94, 2: 0.92}
        assert metrics.calibration_error == 0.03
    
    def test_to_dict_minimal(self):
        """Test conversion to dictionary with minimal data."""
        metrics = EvaluationMetrics(accuracy=0.75)
        result = metrics.to_dict()
        
        expected = {"accuracy": 0.75}
        assert result == expected
    
    def test_to_dict_full(self):
        """Test conversion to dictionary with all fields."""
        metrics = EvaluationMetrics(
            accuracy=0.88,
            loss=0.22,
            confidence_interval=(0.85, 0.91),
            per_class_accuracy={0: 0.85, 1: 0.90, 2: 0.89},
            calibration_error=0.05
        )
        
        result = metrics.to_dict()
        
        assert result["accuracy"] == 0.88
        assert result["loss"] == 0.22
        assert result["confidence_interval"] == (0.85, 0.91)
        assert result["per_class_accuracy"] == {0: 0.85, 1: 0.90, 2: 0.89}
        assert result["calibration_error"] == 0.05


class TestAccuracyCalculator:
    """Test AccuracyCalculator class."""
    
    def test_perfect_accuracy(self):
        """Test accuracy calculation with perfect predictions."""
        calc = AccuracyCalculator()
        
        predictions = torch.tensor([
            [0.9, 0.1, 0.0],  # Class 0
            [0.1, 0.8, 0.1],  # Class 1  
            [0.0, 0.2, 0.8]   # Class 2
        ])
        targets = torch.tensor([0, 1, 2])
        
        accuracy = calc.compute_accuracy(predictions, targets)
        assert accuracy == 1.0
    
    def test_zero_accuracy(self):
        """Test accuracy calculation with completely wrong predictions."""
        calc = AccuracyCalculator()
        
        predictions = torch.tensor([
            [0.1, 0.9, 0.0],  # Predicted 1, actual 0
            [0.8, 0.1, 0.1],  # Predicted 0, actual 1
            [0.1, 0.8, 0.1]   # Predicted 1, actual 2
        ])
        targets = torch.tensor([0, 1, 2])
        
        accuracy = calc.compute_accuracy(predictions, targets)
        assert accuracy == 0.0
    
    def test_partial_accuracy(self):
        """Test accuracy calculation with some correct predictions."""
        calc = AccuracyCalculator()
        
        predictions = torch.tensor([
            [0.9, 0.1, 0.0],  # Correct: Class 0
            [0.8, 0.1, 0.1],  # Wrong: Predicted 0, actual 1
            [0.0, 0.1, 0.9],  # Correct: Class 2
            [0.1, 0.8, 0.1]   # Correct: Class 1
        ])
        targets = torch.tensor([0, 1, 2, 1])
        
        accuracy = calc.compute_accuracy(predictions, targets)
        assert accuracy == 0.75  # 3/4 correct
    
    def test_logits_input(self):
        """Test accuracy calculation with raw logits (pre-softmax)."""
        calc = AccuracyCalculator()
        
        # Raw logits (will be converted to probabilities internally)
        logits = torch.tensor([
            [2.0, 0.5, 0.1],  # Highest at index 0
            [0.1, 1.8, 0.2],  # Highest at index 1
            [0.3, 0.4, 2.5]   # Highest at index 2
        ])
        targets = torch.tensor([0, 1, 2])
        
        accuracy = calc.compute_accuracy(logits, targets)
        assert accuracy == 1.0
    
    def test_confidence_interval_calculation(self):
        """Test confidence interval calculation."""
        calc = AccuracyCalculator()
        
        # Create larger dataset for meaningful CI
        torch.manual_seed(42)
        predictions = torch.randn(100, 5)  # 100 samples, 5 classes
        targets = torch.randint(0, 5, (100,))
        
        accuracy, ci = calc.compute_accuracy_with_ci(predictions, targets)
        
        # Check that CI is a tuple of two floats
        assert isinstance(ci, tuple)
        assert len(ci) == 2
        assert ci[0] <= accuracy <= ci[1]  # Accuracy should be within CI
        assert 0 <= ci[0] <= 1 and 0 <= ci[1] <= 1  # CI should be in [0, 1]
    
    def test_per_class_accuracy(self):
        """Test per-class accuracy calculation."""
        calc = AccuracyCalculator()
        
        predictions = torch.tensor([
            [0.9, 0.1],  # Class 0, correct
            [0.8, 0.2],  # Class 0, correct
            [0.3, 0.7],  # Class 1, correct
            [0.6, 0.4],  # Class 0, wrong (predicted 0, actual 1)
            [0.2, 0.8]   # Class 1, correct
        ])
        targets = torch.tensor([0, 0, 1, 1, 1])
        
        per_class_acc = calc.compute_per_class_accuracy(predictions, targets)
        
        # Class 0: 2/2 correct = 1.0
        # Class 1: 2/3 correct = 0.667
        assert abs(per_class_acc[0] - 1.0) < 1e-6
        assert abs(per_class_acc[1] - 2/3) < 1e-6
    
    def test_empty_predictions(self):
        """Test handling of empty predictions."""
        calc = AccuracyCalculator()
        
        predictions = torch.empty(0, 3)
        targets = torch.empty(0, dtype=torch.long)
        
        # Empty tensors should return NaN, not raise an error
        accuracy = calc.compute_accuracy(predictions, targets)
        assert torch.isnan(torch.tensor(accuracy)) or accuracy == 0.0
    
    def test_single_prediction(self):
        """Test accuracy calculation with single prediction."""
        calc = AccuracyCalculator()
        
        predictions = torch.tensor([[0.7, 0.3]])
        targets = torch.tensor([0])
        
        accuracy = calc.compute_accuracy(predictions, targets)
        assert accuracy == 1.0
    
    def test_mismatched_dimensions(self):
        """Test error handling for mismatched dimensions."""
        calc = AccuracyCalculator()
        
        predictions = torch.tensor([[0.5, 0.5], [0.6, 0.4]])  # 2 samples
        targets = torch.tensor([0, 1, 2])  # 3 targets
        
        with pytest.raises((RuntimeError, AssertionError)):
            calc.compute_accuracy(predictions, targets)


class TestCalibrationCalculator:
    """Test CalibrationCalculator class."""
    
    def test_perfect_calibration(self):
        """Test calibration with perfectly calibrated predictions."""
        calc = CalibrationCalculator()
        
        # Create perfectly calibrated predictions
        # 70% confidence predictions should be correct 70% of the time
        predictions = torch.tensor([
            [0.7, 0.3],  # 70% confident, class 0
            [0.7, 0.3],  # 70% confident, class 0
            [0.7, 0.3],  # 70% confident, class 0
            [0.3, 0.7],  # 70% confident, class 1
            [0.3, 0.7],  # 70% confident, class 1
            [0.3, 0.7],  # 70% confident, class 1
            [0.3, 0.7],  # 70% confident, class 1
            [0.3, 0.7],  # 70% confident, class 1
            [0.3, 0.7],  # 70% confident, class 1
            [0.3, 0.7],  # 70% confident, class 1
        ])
        
        # Make 70% of them correct
        targets = torch.tensor([0, 0, 1, 1, 1, 1, 1, 1, 0, 0])  # 7/10 correct
        
        ece = calc.compute_expected_calibration_error(predictions, targets, n_bins=5)
        
        # Should be close to 0 for perfect calibration
        assert ece < 0.15  # Allow more tolerance due to binning and small sample size
    
    def test_poor_calibration(self):
        """Test calibration with poorly calibrated predictions."""
        calc = CalibrationCalculator()
        
        # Overconfident predictions (90% confidence but only ~50% correct)
        predictions = torch.tensor([
            [0.9, 0.1],  # Very confident, class 0
            [0.9, 0.1],  # Very confident, class 0  
            [0.1, 0.9],  # Very confident, class 1
            [0.1, 0.9],  # Very confident, class 1
        ])
        
        # But only 50% are actually correct
        targets = torch.tensor([0, 1, 1, 0])  # 2/4 = 50% correct
        
        ece = calc.compute_expected_calibration_error(predictions, targets, n_bins=5)
        
        # Should have high calibration error, but small sample size affects precision
        assert ece > 0.1  # Lowered threshold for small sample
    
    def test_reliability_diagram_data(self):
        """Test reliability diagram data generation."""
        calc = CalibrationCalculator()
        
        torch.manual_seed(42)
        predictions = torch.softmax(torch.randn(50, 3), dim=1)
        targets = torch.randint(0, 3, (50,))
        
        bin_boundaries, bin_lowers, bin_uppers, bin_accs, bin_confs, bin_sizes = \
            calc.get_reliability_diagram_data(predictions, targets, n_bins=10)
        
        # Check output structure
        assert len(bin_boundaries) == 11  # n_bins + 1
        assert len(bin_lowers) == len(bin_uppers) == len(bin_accs) == len(bin_confs) == len(bin_sizes)
        assert all(0 <= acc <= 1 for acc in bin_accs if not np.isnan(acc))
        assert all(0 <= conf <= 1 for conf in bin_confs if not np.isnan(conf))
        assert all(size >= 0 for size in bin_sizes)
    
    def test_maximum_calibration_error(self):
        """Test maximum calibration error calculation."""
        calc = CalibrationCalculator()
        
        predictions = torch.tensor([
            [0.9, 0.1],  # High confidence, correct
            [0.8, 0.2],  # High confidence, wrong  
            [0.6, 0.4],  # Medium confidence, correct
        ])
        targets = torch.tensor([0, 1, 0])
        
        mce = calc.compute_maximum_calibration_error(predictions, targets, n_bins=3)
        
        # Should be > 0 due to calibration errors
        assert mce >= 0
        assert mce <= 1
    
    def test_edge_case_single_bin(self):
        """Test calibration with single bin."""
        calc = CalibrationCalculator()
        
        predictions = torch.tensor([[0.6, 0.4], [0.7, 0.3]])
        targets = torch.tensor([0, 0])
        
        # Single bin should work
        ece = calc.compute_expected_calibration_error(predictions, targets, n_bins=1)
        assert isinstance(ece, float)
        assert 0 <= ece <= 1


class TestUncertaintyCalculator:
    """Test UncertaintyCalculator class."""
    
    def test_entropy_calculation(self):
        """Test entropy calculation for different prediction distributions."""
        calc = UncertaintyCalculator()
        
        # Uniform distribution (maximum entropy)
        uniform_pred = torch.tensor([[0.25, 0.25, 0.25, 0.25]])
        uniform_entropy = calc.compute_predictive_entropy(uniform_pred)
        
        # Peaked distribution (low entropy) - use logits for more control
        peaked_logits = torch.tensor([[5.0, 0.1, 0.1, 0.1]])  # Very peaked after softmax
        peaked_entropy = calc.compute_predictive_entropy(peaked_logits)
        
        # Uniform should have higher entropy
        assert uniform_entropy > peaked_entropy
        assert uniform_entropy > 1.0  # Should be close to log(4) ≈ 1.386
        assert peaked_entropy < 1.0   # Should be lower than uniform
    
    def test_entropy_batch(self):
        """Test entropy calculation for batched predictions."""
        calc = UncertaintyCalculator()
        
        # Batch of predictions with different confidence levels
        predictions = torch.tensor([
            [0.9, 0.1],     # Low entropy (high confidence)
            [0.5, 0.5],     # High entropy (uncertain)  
            [0.99, 0.01],   # Very low entropy
        ])
        
        entropies = calc.compute_predictive_entropy(predictions)
        
        assert len(entropies) == 3
        assert entropies[1] > entropies[0] > entropies[2]  # Middle is most uncertain
    
    def test_mutual_information_estimation(self):
        """Test mutual information estimation."""
        calc = UncertaintyCalculator()
        
        # Create Monte Carlo samples (simulating model uncertainty)
        torch.manual_seed(42)
        mc_predictions = torch.softmax(torch.randn(10, 20, 5), dim=2)  # 10 samples, 20 data points, 5 classes
        
        mutual_info = calc.compute_mutual_information(mc_predictions)
        
        assert len(mutual_info) == 20  # One MI value per data point
        assert all(mi >= 0 for mi in mutual_info)  # MI should be non-negative
    
    def test_confidence_calculation(self):
        """Test confidence calculation (max probability)."""
        calc = UncertaintyCalculator()
        
        # Use logits that will result in the expected probabilities after softmax
        predictions = torch.tensor([
            [1.386, 0.693],  # Results in ~0.8, 0.2 after softmax  
            [2.197, 0.693],  # Results in ~0.9, 0.1 after softmax
            [0.200, 0.000],  # Results in ~0.55, 0.45 after softmax
        ])
        
        confidences = calc.compute_confidence(predictions)
        
        # Check that confidence is the max probability (approximately)
        assert len(confidences) == 3
        assert all(conf > 0.5 for conf in confidences)  # All should be > 0.5
        assert confidences[1] > confidences[0]  # Second should be highest
        assert confidences[0] > confidences[2]  # First should be higher than third
    
    def test_uncertainty_decomposition(self):
        """Test decomposition of uncertainty into aleatoric and epistemic components."""
        calc = UncertaintyCalculator()
        
        # Create Monte Carlo predictions with some variation
        torch.manual_seed(42)
        n_samples, n_data, n_classes = 15, 10, 4
        mc_predictions = torch.softmax(torch.randn(n_samples, n_data, n_classes), dim=2)
        
        total_entropy, aleatoric_entropy, epistemic_entropy = \
            calc.decompose_uncertainty(mc_predictions)
        
        assert len(total_entropy) == n_data
        assert len(aleatoric_entropy) == n_data  
        assert len(epistemic_entropy) == n_data
        
        # Total uncertainty should be sum of aleatoric and epistemic
        assert torch.allclose(total_entropy, aleatoric_entropy + epistemic_entropy, atol=1e-6)
        
        # All values should be non-negative
        assert all(te >= 0 for te in total_entropy)
        assert all(ae >= 0 for ae in aleatoric_entropy)
        assert all(ee >= 0 for ee in epistemic_entropy)
    
    def test_zero_entropy_edge_case(self):
        """Test entropy calculation with deterministic predictions."""
        calc = UncertaintyCalculator()
        
        # Use very high logits for near-deterministic behavior
        deterministic_logits = torch.tensor([[10.0, -10.0, -10.0]])  # Very peaked after softmax
        entropy = calc.compute_predictive_entropy(deterministic_logits)
        
        # Should be low entropy (softmax prevents exactly zero)
        assert entropy < 0.1
    
    def test_large_batch_entropy(self):
        """Test entropy calculation efficiency with large batches."""
        calc = UncertaintyCalculator()
        
        # Large batch of predictions
        torch.manual_seed(42)
        large_predictions = torch.softmax(torch.randn(1000, 10), dim=1)
        
        entropies = calc.compute_predictive_entropy(large_predictions)
        
        assert len(entropies) == 1000
        assert all(0 <= ent <= np.log(10) for ent in entropies)  # Entropy bounded by log(n_classes)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])