"""
Comprehensive Tests for Meta-Learning Package

Tests all major components of the meta-learning package to ensure
functionality and research accuracy.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple

# Import meta-learning components
import sys
sys.path.insert(0, '../src')
from meta_learning import (
    TestTimeComputeScaler,
    MAMLLearner,
    PrototypicalNetworks,
    OnlineMetaLearner,
    MetaLearningDataset,
    few_shot_accuracy,
    adaptation_speed
)
from meta_learning.meta_learning_modules import (
    TestTimeComputeConfig,
    MAMLConfig,
    PrototypicalConfig,
    OnlineMetaConfig,
    TaskConfiguration
)


class SimpleClassifier(nn.Module):
    """Simple classifier for testing."""
    
    def __init__(self, input_dim: int = 784, hidden_dim: int = 64, output_dim: int = 5):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.network(x.view(x.size(0), -1))


@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    torch.manual_seed(42)
    
    # Generate synthetic few-shot learning data
    n_samples = 200
    input_dim = 784  # 28x28 images
    n_classes = 10
    
    # Create synthetic data with clear class structure
    data = []
    labels = []
    
    for class_id in range(n_classes):
        # Generate class-specific mean
        class_mean = torch.randn(input_dim) * 0.5
        
        # Generate samples around the class mean
        for _ in range(n_samples // n_classes):
            sample = class_mean + torch.randn(input_dim) * 0.1
            data.append(sample)
            labels.append(class_id)
    
    return torch.stack(data), torch.tensor(labels)


@pytest.fixture
def simple_model():
    """Simple model for testing."""
    return SimpleClassifier()


class TestTestTimeComputeScaler:
    """Test TestTimeComputeScaler functionality."""
    
    def test_initialization(self):
        """Test proper initialization."""
        # Create proper few-shot model with backbone and simplified config
        backbone = SimpleClassifier(input_dim=784, hidden_dim=64, output_dim=64)  # Output to embedding_dim
        proto_config = PrototypicalConfig(
            embedding_dim=64,
            num_classes=5,
            multi_scale_features=False,  # Disable advanced features
            adaptive_prototypes=False,
            use_original_implementation=True  # Use baseline implementation for testing
        )
        few_shot_model = PrototypicalNetworks(backbone, proto_config)
        
        config = TestTimeComputeConfig(max_compute_budget=100)
        scaler = TestTimeComputeScaler(few_shot_model, config)
        
        assert scaler.base_model is few_shot_model
        assert scaler.config.max_compute_budget == 100
        assert len(scaler.compute_history) == 0
    
    def test_scale_compute(self, sample_data):
        """Test compute scaling functionality."""
        data, labels = sample_data
        
        # Create few-shot task
        n_support = 25
        n_query = 15
        support_data = data[:n_support]
        support_labels = labels[:n_support]
        query_data = data[n_support:n_support+n_query]
        
        # Create proper few-shot model with backbone and simplified config
        backbone = SimpleClassifier(input_dim=784, hidden_dim=64, output_dim=64)  # Output to embedding_dim
        proto_config = PrototypicalConfig(
            embedding_dim=64,
            num_classes=5,
            multi_scale_features=False,  # Disable advanced features
            adaptive_prototypes=False,
            use_original_implementation=True  # Use baseline implementation for testing
        )
        few_shot_model = PrototypicalNetworks(backbone, proto_config)
        
        config = TestTimeComputeConfig(
            max_compute_budget=50,
            min_compute_steps=5,
            confidence_threshold=0.9
        )
        scaler = TestTimeComputeScaler(few_shot_model, config)
        
        # Test scaling
        predictions, metrics = scaler.scale_compute(
            support_data, support_labels, query_data
        )
        
        # Validate outputs
        assert predictions.shape == (n_query, len(torch.unique(support_labels)))
        assert "compute_used" in metrics
        assert "final_confidence" in metrics
        assert metrics["compute_used"] >= config.min_compute_steps
        assert metrics["compute_used"] <= config.max_compute_budget
    
    def test_difficulty_estimation(self, simple_model, sample_data):
        """Test task difficulty estimation."""
        data, labels = sample_data
        
        support_data = data[:25]
        support_labels = labels[:25]
        query_data = data[25:40]
        
        scaler = TestTimeComputeScaler(simple_model)
        
        # Test difficulty estimation
        difficulty = scaler._estimate_difficulty(support_data, support_labels, query_data)
        
        assert 0.0 <= difficulty <= 1.0
        assert isinstance(difficulty, float)


class TestMAMLLearner:
    """Test MAML learner functionality."""
    
    def test_initialization(self, simple_model):
        """Test proper initialization."""
        config = MAMLConfig(inner_lr=0.01, inner_steps=3)
        maml = MAMLLearner(simple_model, config)
        
        assert maml.model is simple_model
        assert maml.config.inner_lr == 0.01
        assert maml.config.inner_steps == 3
    
    def test_meta_test(self, simple_model, sample_data):
        """Test meta-testing functionality."""
        data, labels = sample_data
        
        # Create 5-way 5-shot task
        n_way = 5
        k_shot = 5
        n_query = 15
        
        # Sample balanced task
        task_classes = torch.unique(labels)[:n_way]
        support_data, support_labels = [], []
        query_data, query_labels = [], []
        
        for new_label, class_id in enumerate(task_classes):
            class_indices = (labels == class_id).nonzero().flatten()
            selected = class_indices[:k_shot + n_query//n_way]
            
            # Support set
            for i in range(k_shot):
                support_data.append(data[selected[i]])
                support_labels.append(new_label)
            
            # Query set
            for i in range(k_shot, len(selected)):
                query_data.append(data[selected[i]])
                query_labels.append(new_label)
        
        support_data = torch.stack(support_data)
        support_labels = torch.tensor(support_labels)
        query_data = torch.stack(query_data)
        query_labels = torch.tensor(query_labels)
        
        config = MAMLConfig(inner_steps=3)
        maml = MAMLLearner(simple_model, config)
        
        # Test meta-testing
        results = maml.meta_test(support_data, support_labels, query_data, query_labels)
        
        # Validate results
        assert "predictions" in results
        assert "accuracy" in results
        assert "loss" in results
        assert 0.0 <= results["accuracy"] <= 1.0
    
    def test_adaptation(self, simple_model, sample_data):
        """Test task adaptation."""
        data, labels = sample_data
        
        support_data = data[:25]
        support_labels = labels[:25]
        
        config = MAMLConfig(inner_steps=5, inner_lr=0.01)
        maml = MAMLLearner(simple_model, config)
        
        # Test adaptation
        adapted_params, adaptation_info = maml._adapt_to_task(support_data, support_labels)
        
        # Validate adaptation
        assert len(adapted_params) > 0
        assert "steps" in adaptation_info
        assert adaptation_info["steps"] <= config.inner_steps
        assert "final_loss" in adaptation_info


class TestPrototypicalNetworks:
    """Test Prototypical Networks functionality."""
    
    def test_initialization(self, simple_model):
        """Test proper initialization."""
        config = PrototypicalConfig(
            embedding_dim=64,
            multi_scale_features=True
        )
        proto_net = PrototypicalNetworks(simple_model, config)
        
        assert proto_net.backbone is simple_model
        assert proto_net.config.multi_scale_features is True
    
    def test_forward_pass(self, simple_model, sample_data):
        """Test forward pass functionality."""
        data, labels = sample_data
        
        # Create few-shot task
        n_way = 5
        k_shot = 5
        n_query = 15
        
        # Sample task
        task_classes = torch.unique(labels)[:n_way]
        support_data, support_labels = [], []
        query_data, query_labels = [], []
        
        for new_label, class_id in enumerate(task_classes):
            class_indices = (labels == class_id).nonzero().flatten()
            selected = class_indices[:k_shot + n_query//n_way]
            
            for i in range(k_shot):
                support_data.append(data[selected[i]])
                support_labels.append(new_label)
            
            for i in range(k_shot, len(selected)):
                query_data.append(data[selected[i]])
                query_labels.append(new_label)
        
        support_data = torch.stack(support_data)
        support_labels = torch.tensor(support_labels)
        query_data = torch.stack(query_data)
        
        config = PrototypicalConfig(embedding_dim=5)  # Match model output
        proto_net = PrototypicalNetworks(simple_model, config)
        
        # Forward pass
        results = proto_net.forward(support_data, support_labels, query_data)
        
        # Validate results
        assert "logits" in results
        assert "probabilities" in results
        assert "distances" in results
        assert results["logits"].shape == (len(query_data), n_way)
        assert torch.allclose(results["probabilities"].sum(dim=-1), torch.ones(len(query_data)))
    
    def test_prototype_computation(self, simple_model, sample_data):
        """Test prototype computation."""
        data, labels = sample_data
        
        support_data = data[:25]
        support_labels = labels[:25]
        
        proto_net = PrototypicalNetworks(simple_model)
        
        # Extract features
        with torch.no_grad():
            features = simple_model(support_data)
        
        # Compute prototypes
        prototypes = proto_net._compute_prototypes(features, support_labels)
        
        n_classes = len(torch.unique(support_labels))
        assert prototypes.shape == (n_classes, features.shape[-1])


class TestOnlineMetaLearner:
    """Test Online Meta-Learner functionality."""
    
    def test_initialization(self, simple_model):
        """Test proper initialization."""
        config = OnlineMetaConfig(
            memory_size=100,
            experience_replay=True
        )
        online_learner = OnlineMetaLearner(simple_model, config)
        
        assert online_learner.model is simple_model
        assert online_learner.config.memory_size == 100
        assert len(online_learner.experience_memory) == 0
    
    def test_learn_task(self, simple_model, sample_data):
        """Test online task learning."""
        data, labels = sample_data
        
        # Create few-shot task
        n_support = 15
        n_query = 10
        support_data = data[:n_support]
        support_labels = labels[:n_support]
        query_data = data[n_support:n_support+n_query]
        query_labels = labels[n_support:n_support+n_query]
        
        config = OnlineMetaConfig(memory_size=50)
        online_learner = OnlineMetaLearner(simple_model, config)
        
        # Learn task
        results = online_learner.learn_task(
            support_data, support_labels, query_data, query_labels
        )
        
        # Validate results
        assert "task_id" in results
        assert "query_accuracy" in results
        assert "meta_loss" in results
        assert 0.0 <= results["query_accuracy"] <= 1.0
        assert len(online_learner.experience_memory) > 0
    
    def test_continual_learning(self, simple_model, sample_data):
        """Test continual learning across multiple tasks."""
        data, labels = sample_data
        
        config = OnlineMetaConfig(memory_size=100, experience_replay=True)
        online_learner = OnlineMetaLearner(simple_model, config)
        
        # Learn multiple tasks
        task_results = []
        for i in range(3):
            # Create task with different data slice
            start_idx = i * 20
            support_data = data[start_idx:start_idx+15]
            support_labels = labels[start_idx:start_idx+15]
            query_data = data[start_idx+15:start_idx+25]
            query_labels = labels[start_idx+15:start_idx+25]
            
            results = online_learner.learn_task(
                support_data, support_labels, query_data, query_labels,
                task_id=f"task_{i}"
            )
            task_results.append(results)
        
        # Validate continual learning
        assert len(task_results) == 3
        assert online_learner.task_count == 3
        assert len(online_learner.experience_memory) > 0


class TestMetaLearningDataset:
    """Test MetaLearningDataset functionality."""
    
    def test_initialization(self, sample_data):
        """Test dataset initialization."""
        data, labels = sample_data
        
        config = TaskConfiguration(n_way=5, k_shot=3, q_query=10)
        dataset = MetaLearningDataset(data, labels, config)
        
        assert len(dataset) == config.num_tasks
        assert dataset.num_classes == len(torch.unique(labels))
        assert len(dataset.class_to_indices) == dataset.num_classes
    
    def test_task_sampling(self, sample_data):
        """Test task sampling functionality."""
        data, labels = sample_data
        
        config = TaskConfiguration(n_way=5, k_shot=3, q_query=10)
        dataset = MetaLearningDataset(data, labels, config)
        
        # Sample a task
        task = dataset.sample_task(task_idx=0)
        
        # Validate task structure
        assert "support" in task
        assert "query" in task
        assert "task_classes" in task
        assert "metadata" in task
        
        # Validate support set
        assert task["support"]["data"].shape[0] == config.n_way * config.k_shot
        assert task["support"]["labels"].shape[0] == config.n_way * config.k_shot
        
        # Validate query set
        assert task["query"]["data"].shape[0] == config.n_way * config.q_query
        assert task["query"]["labels"].shape[0] == config.n_way * config.q_query
        
        # Validate labels are in correct range
        assert task["support"]["labels"].max() < config.n_way
        assert task["query"]["labels"].max() < config.n_way
    
    def test_difficulty_estimation(self, sample_data):
        """Test class difficulty estimation."""
        data, labels = sample_data
        
        dataset = MetaLearningDataset(data, labels)
        
        # Validate difficulty estimation
        assert len(dataset.class_difficulties) == dataset.num_classes
        for class_id, difficulty in dataset.class_difficulties.items():
            assert 0.0 <= difficulty <= 1.0


class TestUtilities:
    """Test utility functions."""
    
    def test_few_shot_accuracy(self):
        """Test few-shot accuracy computation."""
        # Create test predictions and targets
        predictions = torch.tensor([
            [0.8, 0.1, 0.1],  # Predicted class 0, actual 0 ✓
            [0.2, 0.7, 0.1],  # Predicted class 1, actual 1 ✓
            [0.1, 0.2, 0.7],  # Predicted class 2, actual 0 ✗
        ])
        targets = torch.tensor([0, 1, 0])
        
        # Test overall accuracy
        accuracy = few_shot_accuracy(predictions, targets)
        expected_accuracy = 2.0 / 3.0  # 2 correct out of 3
        assert abs(accuracy - expected_accuracy) < 1e-6
        
        # Test per-class accuracy
        overall_acc, per_class_acc = few_shot_accuracy(predictions, targets, return_per_class=True)
        assert abs(overall_acc - expected_accuracy) < 1e-6
        assert len(per_class_acc) == 2  # Classes 0 and 1
    
    def test_adaptation_speed(self):
        """Test adaptation speed measurement."""
        # Test convergent loss curve
        convergent_losses = [1.0, 0.8, 0.6, 0.45, 0.44, 0.43, 0.43]
        steps, final_loss = adaptation_speed(convergent_losses, convergence_threshold=0.02)
        
        assert steps < len(convergent_losses)  # Should converge early
        assert final_loss < convergent_losses[0]  # Loss should decrease
        
        # Test non-convergent loss curve
        non_convergent_losses = [1.0, 0.8, 0.6, 0.4, 0.2]
        steps, final_loss = adaptation_speed(non_convergent_losses, convergence_threshold=0.01)
        
        assert steps == len(non_convergent_losses)  # Should use all steps
        assert final_loss == non_convergent_losses[-1]


@pytest.mark.integration
class TestIntegration:
    """Integration tests combining multiple components."""
    
    def test_end_to_end_meta_learning(self, sample_data):
        """Test complete meta-learning pipeline."""
        data, labels = sample_data
        
        # Create dataset
        config = TaskConfiguration(n_way=3, k_shot=5, q_query=10)
        dataset = MetaLearningDataset(data, labels, config)
        
        # Create model
        model = SimpleClassifier(input_dim=784, output_dim=config.n_way)
        
        # Create MAML learner
        maml_config = MAMLConfig(inner_steps=3, inner_lr=0.01)
        maml = MAMLLearner(model, maml_config)
        
        # Sample task
        task = dataset.sample_task(task_idx=0)
        
        # Run meta-test
        results = maml.meta_test(
            task["support"]["data"],
            task["support"]["labels"],
            task["query"]["data"],
            task["query"]["labels"]
        )
        
        # Validate end-to-end functionality
        assert "accuracy" in results
        assert "predictions" in results
        assert 0.0 <= results["accuracy"] <= 1.0
        assert results["predictions"].shape[0] == len(task["query"]["data"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])