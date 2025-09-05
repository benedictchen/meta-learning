"""
Episode Contract for Meta-Learning
==================================

Author: Benedict Chen (benedict@benedictchen.com)

Enforces strict contracts for few-shot learning episodes with runtime
validation to catch API violations and mathematical inconsistencies.

Research Critical Issues:
1. N-way/K-shot/M-query parameters must be mathematically consistent
2. Class labels must be remapped to contiguous [0, N-1] range
3. Support/query sets must have identical class distributions
4. No data leakage between support and query sets
5. Batch dimensions must be consistent for vectorized operations

This module provides dataclasses with built-in validation that catch
violations at runtime, preventing silent failures that corrupt results.

References:
- Vinyals et al. (2016): "Matching Networks" - episode structure definition
- Snell et al. (2017): "Prototypical Networks" - support/query split requirements
- Finn et al. (2017): "MAML" - inner/outer loop data requirements
"""

import torch
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Dict, Any, Union
import warnings
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class EpisodeValidationLevel(Enum):
    """Validation strictness levels for episode contracts."""
    STRICT = "strict"        # All validations enforced, raise on any violation
    LENIENT = "lenient"     # Log warnings for violations, continue execution
    DISABLED = "disabled"   # No validation (for performance-critical code)


@dataclass
class EpisodeContract:
    """
    Contract for few-shot learning episodes with runtime validation.
    
    Enforces mathematical consistency and API correctness for meta-learning.
    All tensor shapes and label ranges are validated at construction.
    """
    
    # Episode parameters
    n_way: int                          # Number of classes in episode
    k_shot: int                         # Number of support examples per class
    m_query: int                        # Number of query examples per class
    
    # Episode data
    support_x: torch.Tensor             # Support inputs [N*K, ...]
    support_y: torch.Tensor             # Support labels [N*K], values in [0, N-1]
    query_x: torch.Tensor               # Query inputs [N*M, ...]
    query_y: torch.Tensor               # Query labels [N*M], values in [0, N-1]
    
    # Optional metadata
    episode_id: str = "unknown"
    original_classes: Optional[List[int]] = None  # Original class IDs before remapping
    data_source: Optional[str] = None
    validation_level: EpisodeValidationLevel = EpisodeValidationLevel.STRICT
    
    # Private fields for validation state
    _validated: bool = field(default=False, init=False)
    _validation_errors: List[str] = field(default_factory=list, init=False)
    
    def __post_init__(self):
        """Validate episode contract immediately after construction."""
        if self.validation_level != EpisodeValidationLevel.DISABLED:
            self._validate_episode()
            
        if self.validation_level == EpisodeValidationLevel.STRICT and self._validation_errors:
            error_msg = f"Episode contract violations in {self.episode_id}:\n" + \
                       "\n".join(f"  - {error}" for error in self._validation_errors)
            raise ValueError(error_msg)
        elif self._validation_errors:
            for error in self._validation_errors:
                logger.warning(f"Episode {self.episode_id}: {error}")
                
        self._validated = True
        
    def _validate_episode(self):
        """Comprehensive episode validation."""
        self._validation_errors = []
        
        # 1. Validate basic parameter consistency
        self._validate_parameters()
        
        # 2. Validate tensor shapes
        self._validate_tensor_shapes()
        
        # 3. Validate label consistency
        self._validate_label_consistency()
        
        # 4. Validate class distribution
        self._validate_class_distribution()
        
        # 5. Validate data leakage prevention
        self._validate_no_data_leakage()
        
        # 6. Validate tensor properties
        self._validate_tensor_properties()
        
    def _validate_parameters(self):
        """Validate basic episode parameters."""
        if self.n_way <= 0:
            self._validation_errors.append(f"n_way must be positive, got {self.n_way}")
            
        if self.k_shot <= 0:
            self._validation_errors.append(f"k_shot must be positive, got {self.k_shot}")
            
        if self.m_query <= 0:
            self._validation_errors.append(f"m_query must be positive, got {self.m_query}")
            
        # Common mistake: confusing total shots with shots per class
        if self.k_shot > 100:
            self._validation_errors.append(
                f"k_shot={self.k_shot} seems very high. "
                f"k_shot should be shots PER CLASS, not total shots."
            )
            
    def _validate_tensor_shapes(self):
        """Validate tensor shape consistency."""
        # Support set shape validation
        expected_support_size = self.n_way * self.k_shot
        if self.support_x.shape[0] != expected_support_size:
            self._validation_errors.append(
                f"support_x batch size {self.support_x.shape[0]} != "
                f"n_way * k_shot = {expected_support_size}"
            )
            
        if self.support_y.shape[0] != expected_support_size:
            self._validation_errors.append(
                f"support_y size {self.support_y.shape[0]} != "
                f"n_way * k_shot = {expected_support_size}"
            )
            
        # Query set shape validation
        expected_query_size = self.n_way * self.m_query
        if self.query_x.shape[0] != expected_query_size:
            self._validation_errors.append(
                f"query_x batch size {self.query_x.shape[0]} != "
                f"n_way * m_query = {expected_query_size}"
            )
            
        if self.query_y.shape[0] != expected_query_size:
            self._validation_errors.append(
                f"query_y size {self.query_y.shape[0]} != "
                f"n_way * m_query = {expected_query_size}"
            )
            
        # Feature dimension consistency
        if len(self.support_x.shape) != len(self.query_x.shape):
            self._validation_errors.append(
                f"support_x and query_x have different numbers of dimensions: "
                f"{len(self.support_x.shape)} vs {len(self.query_x.shape)}"
            )
        elif self.support_x.shape[1:] != self.query_x.shape[1:]:
            self._validation_errors.append(
                f"support_x and query_x have different feature shapes: "
                f"{self.support_x.shape[1:]} vs {self.query_x.shape[1:]}"
            )
            
    def _validate_label_consistency(self):
        """Validate label ranges and consistency."""
        # Labels must be in [0, N-1] range
        support_labels = self.support_y.unique().sort()[0]
        query_labels = self.query_y.unique().sort()[0]
        
        expected_labels = torch.arange(self.n_way)
        
        if not torch.equal(support_labels, expected_labels):
            self._validation_errors.append(
                f"support_y labels {support_labels.tolist()} != "
                f"expected range [0, {self.n_way-1}]"
            )
            
        if not torch.equal(query_labels, expected_labels):
            self._validation_errors.append(
                f"query_y labels {query_labels.tolist()} != "
                f"expected range [0, {self.n_way-1}]"
            )
            
        # Support and query must have same classes
        if not torch.equal(support_labels, query_labels):
            self._validation_errors.append(
                f"support and query have different classes: "
                f"support={support_labels.tolist()}, query={query_labels.tolist()}"
            )
            
    def _validate_class_distribution(self):
        """Validate balanced class distribution."""
        # Check support set balance
        support_counts = torch.bincount(self.support_y)
        if len(support_counts) != self.n_way:
            self._validation_errors.append(
                f"support set has {len(support_counts)} classes, expected {self.n_way}"
            )
        elif not torch.all(support_counts == self.k_shot):
            self._validation_errors.append(
                f"support set is unbalanced: {support_counts.tolist()}, "
                f"expected {self.k_shot} examples per class"
            )
            
        # Check query set balance
        query_counts = torch.bincount(self.query_y)
        if len(query_counts) != self.n_way:
            self._validation_errors.append(
                f"query set has {len(query_counts)} classes, expected {self.n_way}"
            )
        elif not torch.all(query_counts == self.m_query):
            self._validation_errors.append(
                f"query set is unbalanced: {query_counts.tolist()}, "
                f"expected {self.m_query} examples per class"
            )
            
    def _validate_no_data_leakage(self):
        """Validate no data leakage between support and query."""
        # This is a basic check - more sophisticated checks would compare
        # actual data content, but that's computationally expensive
        
        # Check that we don't have exact same tensor references
        if self.support_x is self.query_x:
            self._validation_errors.append("support_x and query_x are the same tensor object")
            
        if self.support_y is self.query_y:
            self._validation_errors.append("support_y and query_y are the same tensor object")
            
        # Check device consistency
        if self.support_x.device != self.query_x.device:
            self._validation_errors.append(
                f"support_x and query_x on different devices: "
                f"{self.support_x.device} vs {self.query_x.device}"
            )
            
    def _validate_tensor_properties(self):
        """Validate tensor properties and common issues."""
        # Check for NaN or infinite values
        if torch.isnan(self.support_x).any():
            self._validation_errors.append("support_x contains NaN values")
            
        if torch.isnan(self.query_x).any():
            self._validation_errors.append("query_x contains NaN values")
            
        if torch.isinf(self.support_x).any():
            self._validation_errors.append("support_x contains infinite values")
            
        if torch.isinf(self.query_x).any():
            self._validation_errors.append("query_x contains infinite values")
            
        # Check for reasonable value ranges
        if self.support_x.abs().max() > 1e6:
            self._validation_errors.append(
                f"support_x has very large values (max={self.support_x.abs().max():.2e})"
            )
            
        # Check label data types
        if not torch.is_integer(self.support_y):
            self._validation_errors.append(f"support_y must be integer type, got {self.support_y.dtype}")
            
        if not torch.is_integer(self.query_y):
            self._validation_errors.append(f"query_y must be integer type, got {self.query_y.dtype}")
    
    def validate_prediction_output(self, predictions: torch.Tensor) -> bool:
        """
        Validate that model predictions match episode contract.
        
        Args:
            predictions: Model predictions [N*M, N] or [N*M,]
            
        Returns:
            True if predictions are valid
        """
        errors = []
        
        # Check batch dimension
        if predictions.shape[0] != self.query_y.shape[0]:
            errors.append(
                f"Predictions batch size {predictions.shape[0]} != "
                f"query batch size {self.query_y.shape[0]}"
            )
            
        # Check class dimension (if logits)
        if len(predictions.shape) == 2:
            if predictions.shape[1] != self.n_way:
                errors.append(
                    f"Predictions class dimension {predictions.shape[1]} != "
                    f"n_way {self.n_way}"
                )
        elif len(predictions.shape) == 1:
            # Predicted class indices
            if predictions.max() >= self.n_way or predictions.min() < 0:
                errors.append(
                    f"Predicted classes {predictions.unique().tolist()} "
                    f"outside valid range [0, {self.n_way-1}]"
                )
                
        if errors:
            if self.validation_level == EpisodeValidationLevel.STRICT:
                raise ValueError(f"Prediction validation errors:\n" + 
                               "\n".join(f"  - {error}" for error in errors))
            else:
                for error in errors:
                    logger.warning(f"Episode {self.episode_id} prediction: {error}")
                return False
                
        return True
    
    def get_episode_summary(self) -> Dict[str, Any]:
        """Get summary statistics for the episode."""
        return {
            'episode_id': self.episode_id,
            'n_way': self.n_way,
            'k_shot': self.k_shot,
            'm_query': self.m_query,
            'support_shape': list(self.support_x.shape),
            'query_shape': list(self.query_x.shape),
            'support_classes': self.support_y.unique().sort()[0].tolist(),
            'query_classes': self.query_y.unique().sort()[0].tolist(),
            'device': str(self.support_x.device),
            'dtype': str(self.support_x.dtype),
            'validated': self._validated,
            'validation_errors': len(self._validation_errors),
            'data_source': self.data_source,
            'original_classes': self.original_classes
        }
    
    @classmethod
    def from_raw_data(cls,
                     support_x: torch.Tensor,
                     support_y: torch.Tensor,
                     query_x: torch.Tensor,
                     query_y: torch.Tensor,
                     episode_id: str = "from_raw",
                     validation_level: EpisodeValidationLevel = EpisodeValidationLevel.STRICT) -> 'EpisodeContract':
        """
        Create episode contract from raw data with automatic parameter inference.
        
        Automatically infers n_way, k_shot, m_query from the data.
        """
        # Infer parameters from data
        unique_support = support_y.unique()
        n_way = len(unique_support)
        
        support_counts = torch.bincount(support_y)
        k_shot = support_counts[0].item()  # Assume balanced
        
        query_counts = torch.bincount(query_y)
        m_query = query_counts[0].item()  # Assume balanced
        
        return cls(
            n_way=n_way,
            k_shot=k_shot,
            m_query=m_query,
            support_x=support_x,
            support_y=support_y,
            query_x=query_x,
            query_y=query_y,
            episode_id=episode_id,
            validation_level=validation_level
        )
    
    def to_device(self, device: torch.device) -> 'EpisodeContract':
        """Move episode to specified device."""
        return EpisodeContract(
            n_way=self.n_way,
            k_shot=self.k_shot,
            m_query=self.m_query,
            support_x=self.support_x.to(device),
            support_y=self.support_y.to(device),
            query_x=self.query_x.to(device),
            query_y=self.query_y.to(device),
            episode_id=self.episode_id,
            original_classes=self.original_classes,
            data_source=self.data_source,
            validation_level=self.validation_level
        )
    
    def __repr__(self) -> str:
        return (f"EpisodeContract({self.episode_id}: "
                f"{self.n_way}-way {self.k_shot}-shot "
                f"{self.m_query}-query, "
                f"validated={self._validated})")


def create_episode_contract(n_way: int,
                          k_shot: int, 
                          m_query: int,
                          support_x: torch.Tensor,
                          support_y: torch.Tensor,
                          query_x: torch.Tensor,
                          query_y: torch.Tensor,
                          episode_id: str = "created",
                          validation_level: EpisodeValidationLevel = EpisodeValidationLevel.STRICT) -> EpisodeContract:
    """
    Factory function to create validated episode contracts.
    
    Convenience function with cleaner parameter ordering.
    """
    return EpisodeContract(
        n_way=n_way,
        k_shot=k_shot,
        m_query=m_query,
        support_x=support_x,
        support_y=support_y,
        query_x=query_x,
        query_y=query_y,
        episode_id=episode_id,
        validation_level=validation_level
    )


def validate_episode_batch(episodes: List[EpisodeContract]) -> Dict[str, Any]:
    """
    Validate a batch of episodes for consistency.
    
    Ensures all episodes in a batch have compatible parameters
    for batched processing.
    """
    if not episodes:
        return {'valid': True, 'errors': []}
        
    errors = []
    
    # Check parameter consistency
    reference = episodes[0]
    for i, episode in enumerate(episodes[1:], 1):
        if episode.n_way != reference.n_way:
            errors.append(f"Episode {i} has n_way={episode.n_way}, expected {reference.n_way}")
            
        if episode.k_shot != reference.k_shot:
            errors.append(f"Episode {i} has k_shot={episode.k_shot}, expected {reference.k_shot}")
            
        if episode.m_query != reference.m_query:
            errors.append(f"Episode {i} has m_query={episode.m_query}, expected {reference.m_query}")
            
        # Check shape consistency
        if episode.support_x.shape[1:] != reference.support_x.shape[1:]:
            errors.append(f"Episode {i} has different feature shape")
            
        if episode.support_x.device != reference.support_x.device:
            errors.append(f"Episode {i} on different device")
            
    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'batch_size': len(episodes),
        'parameters': {
            'n_way': reference.n_way,
            'k_shot': reference.k_shot,
            'm_query': reference.m_query
        }
    }


if __name__ == "__main__":
    # Test episode contract functionality
    print("Episode Contract for Meta-Learning Test")
    print("=" * 50)
    
    # Test valid episode creation
    n_way, k_shot, m_query = 5, 3, 2
    
    # Create valid episode data
    support_x = torch.randn(n_way * k_shot, 128)
    support_y = torch.repeat_interleave(torch.arange(n_way), k_shot)
    query_x = torch.randn(n_way * m_query, 128)
    query_y = torch.repeat_interleave(torch.arange(n_way), m_query)
    
    try:
        episode = create_episode_contract(
            n_way, k_shot, m_query,
            support_x, support_y, query_x, query_y,
            episode_id="test_valid"
        )
        print(f"✓ Valid episode created: {episode}")
        
        # Test prediction validation
        predictions = torch.randn(n_way * m_query, n_way)
        valid_preds = episode.validate_prediction_output(predictions)
        print(f"✓ Prediction validation: {valid_preds}")
        
    except ValueError as e:
        print(f"✗ Unexpected validation error: {e}")
    
    # Test invalid episode creation
    print("\nTesting invalid episode detection...")
    
    # Invalid: wrong support size
    try:
        bad_support_x = torch.randn(10, 128)  # Wrong size
        episode = create_episode_contract(
            n_way, k_shot, m_query,
            bad_support_x, support_y, query_x, query_y,
            episode_id="test_invalid"
        )
        print("✗ Should have failed validation")
    except ValueError as e:
        print(f"✓ Caught invalid episode: {str(e)[:100]}...")
    
    # Test from_raw_data
    episode_inferred = EpisodeContract.from_raw_data(
        support_x, support_y, query_x, query_y,
        episode_id="inferred"
    )
    print(f"✓ Inferred parameters: {episode_inferred.n_way}-way {episode_inferred.k_shot}-shot")
    
    # Test episode summary
    summary = episode_inferred.get_episode_summary()
    print(f"✓ Episode summary: {len(summary)} fields")
    
    print("\n✓ Episode contract tests completed")