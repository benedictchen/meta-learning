"""
💰 SUPPORT THIS RESEARCH - PLEASE DONATE! 💰

🙏 If this algorithm registry helps your research, please donate:
💳 PayPal: https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS
⭐ GitHub Sponsors: https://github.com/sponsors/benedictchen

👨‍💻 Author: Benedict Chen
💰 Donations: Help support this work! Buy me a coffee ☕, beer 🍺, lamborghini 🏎️, or private island 🏝️
💖 Please consider recurring donations to fully support continued research

Your support enables cutting-edge AI research for everyone! 🚀

Algorithm Registry for Meta-Learning - Modular Algorithm Management
=================================================================

This module provides a centralized registry for all meta-learning algorithms,
making it easy to add new algorithms and integrate them with selection and
A/B testing frameworks without modifying existing files.

🎯 **Key Features**:
- Centralized algorithm registration and discovery
- Modular algorithm metadata management
- Automatic integration with selection and A/B testing
- Performance characteristic definitions
- Extensible plugin-style architecture

💰 Please donate if this accelerates your research!
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass
from enum import Enum
import torch
import torch.nn as nn
from ..shared.types import Episode


class AlgorithmType(Enum):
    """Categories of meta-learning algorithms."""
    GRADIENT_BASED = "gradient_based"  # MAML, FOMAML, etc.
    METRIC_BASED = "metric_based"      # ProtoNet, Matching Networks, etc.
    OPTIMIZATION_BASED = "optimization_based"  # Ridge Regression, Closed-form solvers
    HYBRID = "hybrid"                  # TTCS, Enhanced algorithms
    MEMORY_BASED = "memory_based"      # Memory-augmented networks


class TaskDifficulty(Enum):
    """Task difficulty levels for algorithm selection."""
    VERY_EASY = "very_easy"      # >95% confidence, simple patterns
    EASY = "easy"               # 80-95% confidence
    MEDIUM = "medium"           # 60-80% confidence
    HARD = "hard"               # 40-60% confidence
    VERY_HARD = "very_hard"     # <40% confidence


@dataclass
class AlgorithmMetadata:
    """Metadata describing algorithm characteristics and performance."""
    name: str
    algorithm_type: AlgorithmType
    description: str
    
    # Performance characteristics
    min_shots_recommended: int
    max_shots_recommended: int
    best_task_difficulties: List[TaskDifficulty]
    computational_complexity: str  # O(n), O(n^2), etc.
    memory_complexity: str
    
    # Suitability factors
    good_for_few_shot: bool
    good_for_many_shot: bool
    good_for_many_classes: bool
    requires_gradients: bool
    supports_mixed_precision: bool
    
    # Implementation details
    implementation_module: str
    implementation_class: str
    default_config: Dict[str, Any]
    
    # Performance hints for algorithm selector
    selection_priority: float  # 0.0-1.0, higher is better
    fallback_algorithms: List[str]


class AlgorithmRegistry:
    """
    Centralized registry for meta-learning algorithms.
    
    Provides a plugin-style architecture for registering algorithms
    and their metadata, making it easy to extend the system without
    modifying existing code.
    """
    
    def __init__(self):
        """Initialize the algorithm registry."""
        self._algorithms: Dict[str, AlgorithmMetadata] = {}
        self._algorithm_instances: Dict[str, Any] = {}
        self._selection_strategies: Dict[str, Callable] = {}
        
        # Register built-in algorithms
        self._register_builtin_algorithms()
        self._register_builtin_strategies()
    
    def register_algorithm(self, metadata: AlgorithmMetadata):
        """
        Register a new algorithm with the registry.
        
        Args:
            metadata: Complete metadata describing the algorithm
        """
        self._algorithms[metadata.name] = metadata
        # Clear cached instances to force reinitialization
        if metadata.name in self._algorithm_instances:
            del self._algorithm_instances[metadata.name]
    
    def get_algorithm_metadata(self, name: str) -> Optional[AlgorithmMetadata]:
        """Get metadata for a registered algorithm."""
        return self._algorithms.get(name)
    
    def get_all_algorithms(self) -> Dict[str, AlgorithmMetadata]:
        """Get all registered algorithm metadata."""
        return self._algorithms.copy()
    
    def get_algorithms_by_type(self, algorithm_type: AlgorithmType) -> Dict[str, AlgorithmMetadata]:
        """Get all algorithms of a specific type."""
        return {
            name: metadata for name, metadata in self._algorithms.items()
            if metadata.algorithm_type == algorithm_type
        }
    
    def get_suitable_algorithms(
        self,
        n_shot: int,
        n_classes: int,
        task_difficulty: TaskDifficulty,
        require_gradients: bool = False,
        max_algorithms: int = 3
    ) -> List[AlgorithmMetadata]:
        """
        Get algorithms suitable for given task characteristics.
        
        Args:
            n_shot: Number of shots per class
            n_classes: Number of classes
            task_difficulty: Estimated task difficulty
            require_gradients: Whether gradients are available
            max_algorithms: Maximum number of algorithms to return
            
        Returns:
            List of suitable algorithms sorted by priority
        """
        suitable = []
        
        for metadata in self._algorithms.values():
            # Check shot count suitability
            if n_shot < metadata.min_shots_recommended or n_shot > metadata.max_shots_recommended:
                continue
            
            # Check task difficulty
            if task_difficulty not in metadata.best_task_difficulties:
                continue
            
            # Check gradient requirements
            if require_gradients and not metadata.requires_gradients:
                continue
            
            # Check class count suitability
            if n_classes > 10 and not metadata.good_for_many_classes:
                continue
            if n_shot <= 3 and not metadata.good_for_few_shot:
                continue
            
            suitable.append(metadata)
        
        # Sort by selection priority
        suitable.sort(key=lambda x: x.selection_priority, reverse=True)
        
        return suitable[:max_algorithms]
    
    def create_algorithm_instance(self, name: str, **kwargs) -> Any:
        """
        Create an instance of a registered algorithm.
        
        Args:
            name: Algorithm name
            **kwargs: Override default configuration
            
        Returns:
            Algorithm instance
        """
        if name not in self._algorithms:
            raise ValueError(f"Algorithm '{name}' not registered")
        
        # Use cached instance if available and no config overrides
        if not kwargs and name in self._algorithm_instances:
            return self._algorithm_instances[name]
        
        metadata = self._algorithms[name]
        
        # Import the module dynamically
        module_parts = metadata.implementation_module.split('.')
        module = __import__(metadata.implementation_module, fromlist=[module_parts[-1]])
        
        # Get the class
        algorithm_class = getattr(module, metadata.implementation_class)
        
        # Merge default config with overrides
        config = metadata.default_config.copy()
        config.update(kwargs)
        
        # Create instance
        instance = algorithm_class(**config)
        
        # Cache if no overrides
        if not kwargs:
            self._algorithm_instances[name] = instance
        
        return instance
    
    def register_selection_strategy(self, name: str, strategy: Callable):
        """Register a new algorithm selection strategy."""
        self._selection_strategies[name] = strategy
    
    def select_algorithm(
        self,
        episode: Episode,
        strategy: str = "heuristic_enhanced",
        **kwargs
    ) -> str:
        """
        Select best algorithm for an episode using a registered strategy.
        
        Args:
            episode: Episode to analyze
            strategy: Selection strategy to use
            **kwargs: Additional arguments for the strategy
            
        Returns:
            Name of selected algorithm
        """
        if strategy not in self._selection_strategies:
            raise ValueError(f"Selection strategy '{strategy}' not registered")
        
        return self._selection_strategies[strategy](self, episode, **kwargs)
    
    def _register_builtin_algorithms(self):
        """Register built-in algorithms."""
        
        # MAML
        self.register_algorithm(AlgorithmMetadata(
            name="maml",
            algorithm_type=AlgorithmType.GRADIENT_BASED,
            description="Model-Agnostic Meta-Learning",
            min_shots_recommended=3,
            max_shots_recommended=20,
            best_task_difficulties=[TaskDifficulty.MEDIUM, TaskDifficulty.HARD],
            computational_complexity="O(k*n)",  # k=inner steps, n=parameters
            memory_complexity="O(n)",
            good_for_few_shot=True,
            good_for_many_shot=True,
            good_for_many_classes=True,
            requires_gradients=True,
            supports_mixed_precision=True,
            implementation_module="meta_learning.algos.maml",
            implementation_class="DualModeMAML",
            default_config={"inner_lr": 0.01, "inner_steps": 5},
            selection_priority=0.8,
            fallback_algorithms=["protonet", "ridge_regression"]
        ))
        
        # Prototypical Networks
        self.register_algorithm(AlgorithmMetadata(
            name="protonet",
            algorithm_type=AlgorithmType.METRIC_BASED,
            description="Prototypical Networks",
            min_shots_recommended=1,
            max_shots_recommended=50,
            best_task_difficulties=[TaskDifficulty.EASY, TaskDifficulty.MEDIUM],
            computational_complexity="O(n*m)",  # n=support, m=query
            memory_complexity="O(c*d)",  # c=classes, d=feature_dim
            good_for_few_shot=True,
            good_for_many_shot=True,
            good_for_many_classes=True,
            requires_gradients=False,
            supports_mixed_precision=True,
            implementation_module="meta_learning.algos.protonet",
            implementation_class="ProtoHead",
            default_config={},
            selection_priority=0.7,
            fallback_algorithms=["ridge_regression"]
        ))
        
        # Test-Time Compute Scaling
        self.register_algorithm(AlgorithmMetadata(
            name="ttcs",
            algorithm_type=AlgorithmType.HYBRID,
            description="Test-Time Compute Scaling",
            min_shots_recommended=1,
            max_shots_recommended=10,
            best_task_difficulties=[TaskDifficulty.HARD, TaskDifficulty.VERY_HARD],
            computational_complexity="O(p*n*m)",  # p=passes, n=support, m=query
            memory_complexity="O(p*d)",
            good_for_few_shot=True,
            good_for_many_shot=False,
            good_for_many_classes=False,
            requires_gradients=False,
            supports_mixed_precision=True,
            implementation_module="meta_learning.algos.ttcs",
            implementation_class="TestTimeComputeScaler",
            default_config={"passes": 8, "combine": "mean_prob"},
            selection_priority=0.6,
            fallback_algorithms=["protonet", "maml"]
        ))
        
        # Ridge Regression (NEW)
        self.register_algorithm(AlgorithmMetadata(
            name="ridge_regression",
            algorithm_type=AlgorithmType.OPTIMIZATION_BASED,
            description="Ridge Regression with Woodbury Matrix Identity",
            min_shots_recommended=1,
            max_shots_recommended=100,
            best_task_difficulties=[TaskDifficulty.VERY_EASY, TaskDifficulty.EASY, TaskDifficulty.MEDIUM],
            computational_complexity="O(d^3)",  # d=feature_dim for matrix inversion
            memory_complexity="O(d^2)",
            good_for_few_shot=True,
            good_for_many_shot=True,
            good_for_many_classes=True,
            requires_gradients=False,
            supports_mixed_precision=True,
            implementation_module="meta_learning.algorithms.ridge_regression",
            implementation_class="RidgeRegression",
            default_config={"lambda_reg": 1e-4, "use_woodbury": True},
            selection_priority=0.9,  # High priority - very stable
            fallback_algorithms=["protonet"]
        ))
        
        # Matching Networks (if available)
        self.register_algorithm(AlgorithmMetadata(
            name="matching_networks",
            algorithm_type=AlgorithmType.METRIC_BASED,
            description="Matching Networks with Attention",
            min_shots_recommended=1,
            max_shots_recommended=20,
            best_task_difficulties=[TaskDifficulty.MEDIUM, TaskDifficulty.HARD],
            computational_complexity="O(n*m*d)",
            memory_complexity="O(n*d)",
            good_for_few_shot=True,
            good_for_many_shot=False,
            good_for_many_classes=False,
            requires_gradients=False,
            supports_mixed_precision=True,
            implementation_module="meta_learning.algorithms.matching_networks",
            implementation_class="MatchingNetworks",
            default_config={"temperature": 1.0},
            selection_priority=0.5,
            fallback_algorithms=["protonet", "ridge_regression"]
        ))
    
    def _register_builtin_strategies(self):
        """Register built-in selection strategies."""
        
        def heuristic_enhanced_strategy(registry, episode: Episode, **kwargs):
            """Enhanced heuristic strategy with ridge regression priority."""
            n_support = len(episode.support_y)
            n_classes = len(torch.unique(episode.support_y))
            n_query = len(episode.query_y)
            
            # Estimate task difficulty based on support size and complexity
            if n_support >= 20:
                difficulty = TaskDifficulty.EASY
            elif n_support >= 10:
                difficulty = TaskDifficulty.MEDIUM
            elif n_support >= 5:
                difficulty = TaskDifficulty.HARD
            else:
                difficulty = TaskDifficulty.VERY_HARD
            
            # Get suitable algorithms
            suitable = registry.get_suitable_algorithms(
                n_shot=n_support // n_classes,
                n_classes=n_classes,
                task_difficulty=difficulty,
                max_algorithms=1
            )
            
            if suitable:
                return suitable[0].name
            
            # Fallback logic
            if n_support < 5:
                return "ttcs"
            elif n_classes > 10:
                return "protonet"
            else:
                return "ridge_regression"  # Prefer stable ridge regression
        
        def performance_based_strategy(registry, episode: Episode, **kwargs):
            """Select based on historical performance (placeholder)."""
            # This would use performance history in a real implementation
            return heuristic_enhanced_strategy(registry, episode, **kwargs)
        
        self.register_selection_strategy("heuristic_enhanced", heuristic_enhanced_strategy)
        self.register_selection_strategy("performance_based", performance_based_strategy)


# Global registry instance
algorithm_registry = AlgorithmRegistry()


# Convenience functions
def register_algorithm(metadata: AlgorithmMetadata):
    """Register an algorithm with the global registry."""
    algorithm_registry.register_algorithm(metadata)


def get_algorithm(name: str, **kwargs) -> Any:
    """Create an algorithm instance from the global registry."""
    return algorithm_registry.create_algorithm_instance(name, **kwargs)


def select_algorithm(episode: Episode, strategy: str = "heuristic_enhanced") -> str:
    """Select the best algorithm for an episode."""
    return algorithm_registry.select_algorithm(episode, strategy)


def get_suitable_algorithms(n_shot: int, n_classes: int, task_difficulty: TaskDifficulty) -> List[str]:
    """Get names of algorithms suitable for given task characteristics."""
    suitable = algorithm_registry.get_suitable_algorithms(n_shot, n_classes, task_difficulty)
    return [metadata.name for metadata in suitable]