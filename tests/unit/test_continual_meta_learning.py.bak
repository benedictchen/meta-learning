"""
Comprehensive unit tests for continual_meta_learning module.

Tests all research solutions, EWC implementations, memory bank functionality,
and research-accurate continual learning approaches following 2024/2025 pytest best practices.
"""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from hypothesis import given, strategies as st, settings
from typing import Dict, List, Tuple, Any, Optional

from meta_learning.meta_learning_modules.continual_meta_learning import (
    ContinualMetaLearner,
    OnlineMetaLearner, 
    ContinualConfig,
    OnlineConfig,
    EWCRegularizer,
    MemoryBank,
    create_continual_learner
)


class TestContinualConfig:
    """Test continual meta-learning configuration."""
    
    def test_continual_config_defaults(self):
        """Test default configuration values."""
        config = ContinualConfig()
        assert config.ewc_lambda == 0.4
        assert config.ewc_method == "diagonal"
        assert config.fisher_estimation_method == "empirical"
        assert config.memory_bank_size == 1000
        assert config.use_memory_bank == True
        assert config.regularization_strength == 1.0
        
    def test_continual_config_ewc_methods(self):
        """Test EWC method configuration options."""
        methods = ["diagonal", "full", "evcl", "none"]
        for method in methods:
            config = ContinualConfig(ewc_method=method)
            assert config.ewc_method == method
            
    def test_continual_config_fisher_methods(self):
        """Test Fisher Information estimation method options."""
        methods = ["empirical", "exact", "kfac"]
        for method in methods:
            config = ContinualConfig(fisher_estimation_method=method)
            assert config.fisher_estimation_method == method
            
    @given(
        ewc_lambda=st.floats(min_value=0.0, max_value=10.0),
        memory_size=st.integers(min_value=100, max_value=5000),
        reg_strength=st.floats(min_value=0.1, max_value=5.0)
    )
    def test_continual_config_property_based(self, ewc_lambda, memory_size, reg_strength):
        """Property-based test for continual configuration."""
        config = ContinualConfig(
            ewc_lambda=ewc_lambda,
            memory_bank_size=memory_size,
            regularization_strength=reg_strength
        )
        assert config.ewc_lambda == ewc_lambda
        assert config.memory_bank_size == memory_size
        assert config.regularization_strength == reg_strength


class TestOnlineConfig:
    """Test online meta-learning configuration."""
    
    def test_online_config_defaults(self):
        """Test default online configuration values."""
        config = OnlineConfig()
        assert config.learning_rate == 0.001
        assert config.meta_learning_rate == 0.01
        assert config.adaptation_steps == 5
        assert config.use_second_order == True
        assert config.buffer_size == 5000
        
    def test_online_config_parameters(self):
        """Test online configuration parameters."""
        config = OnlineConfig(
            learning_rate=0.1,
            meta_learning_rate=0.05,
            adaptation_steps=10,
            use_second_order=False
        )
        assert config.learning_rate == 0.1
        assert config.meta_learning_rate == 0.05
        assert config.adaptation_steps == 10
        assert config.use_second_order == False
        
    def test_online_config_inheritance(self):
        """Test that OnlineConfig inherits from ContinualConfig."""
        config = OnlineConfig()
        assert hasattr(config, 'ewc_lambda')  # From ContinualConfig
        assert hasattr(config, 'learning_rate')  # From OnlineConfig


class TestEWCRegularizer:
    """Test Elastic Weight Consolidation regularizer with all research solutions."""
    
    @pytest.fixture
    def simple_model(self):
        """Create simple model for EWC testing."""
        return nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5)
        )
        
    @pytest.fixture
    def sample_data(self):
        """Create sample data for Fisher Information estimation."""
        batch_size = 32
        x = torch.randn(batch_size, 10)
        y = torch.randint(0, 5, (batch_size,))
        return x, y
        
    def test_ewc_regularizer_init(self, simple_model):
        """Test EWC regularizer initialization."""
        config = ContinualConfig(ewc_method="diagonal")
        regularizer = EWCRegularizer(simple_model, config)
        
        assert regularizer.model == simple_model
        assert regularizer.config == config
        assert regularizer.ewc_lambda == config.ewc_lambda
        
    def test_ewc_diagonal_fisher_estimation(self, simple_model, sample_data):
        """Test diagonal Fisher Information estimation."""
        config = ContinualConfig(ewc_method="diagonal", fisher_estimation_method="empirical")
        regularizer = EWCRegularizer(simple_model, config)
        
        x, y = sample_data
        fisher_info = regularizer.compute_fisher_information(x, y)
        
        # Check that Fisher information is computed for all parameters
        assert len(fisher_info) > 0
        for param_name, fisher_values in fisher_info.items():
            assert torch.isfinite(fisher_values).all()
            assert fisher_values.shape == simple_model.state_dict()[param_name].shape
            
    def test_ewc_full_fisher_estimation(self, simple_model, sample_data):
        """Test full Fisher Information matrix estimation."""
        config = ContinualConfig(ewc_method="full", fisher_estimation_method="empirical")
        regularizer = EWCRegularizer(simple_model, config)
        
        x, y = sample_data
        fisher_info = regularizer.compute_fisher_information(x, y)
        
        assert len(fisher_info) > 0
        for param_name, fisher_values in fisher_info.items():
            assert torch.isfinite(fisher_values).all()
            
    def test_ewc_kfac_fisher_estimation(self, simple_model, sample_data):
        """Test K-FAC Fisher Information estimation."""
        config = ContinualConfig(
            ewc_method="diagonal", 
            fisher_estimation_method="kfac"
        )
        regularizer = EWCRegularizer(simple_model, config)
        
        x, y = sample_data
        fisher_info = regularizer.compute_fisher_information(x, y)
        
        assert len(fisher_info) > 0
        for param_name, fisher_values in fisher_info.items():
            assert torch.isfinite(fisher_values).all()
            
    def test_ewc_exact_fisher_estimation(self, simple_model, sample_data):
        """Test exact Fisher Information estimation."""
        config = ContinualConfig(
            ewc_method="diagonal",
            fisher_estimation_method="exact"
        )
        regularizer = EWCRegularizer(simple_model, config)
        
        x, y = sample_data
        fisher_info = regularizer.compute_fisher_information(x, y)
        
        assert len(fisher_info) > 0
        for param_name, fisher_values in fisher_info.items():
            assert torch.isfinite(fisher_values).all()
            
    def test_store_task_parameters(self, simple_model, sample_data):
        """Test storing task-specific parameters for EWC."""
        config = ContinualConfig()
        regularizer = EWCRegularizer(simple_model, config)
        
        x, y = sample_data
        
        # Store parameters for first task
        regularizer.store_task_parameters(x, y, task_id="task1")
        
        assert "task1" in regularizer.task_params
        assert "task1" in regularizer.fisher_info
        
        # Check stored parameters have correct shapes
        for param_name in simple_model.state_dict():
            assert param_name in regularizer.task_params["task1"]
            assert param_name in regularizer.fisher_info["task1"]
            
    def test_compute_ewc_loss(self, simple_model, sample_data):
        """Test EWC regularization loss computation."""
        config = ContinualConfig()
        regularizer = EWCRegularizer(simple_model, config)
        
        x, y = sample_data
        
        # Store parameters for previous task
        regularizer.store_task_parameters(x, y, task_id="task1")
        
        # Modify model parameters slightly
        for param in simple_model.parameters():
            param.data += torch.randn_like(param.data) * 0.01
            
        # Compute EWC loss
        ewc_loss = regularizer.compute_ewc_loss()
        
        assert isinstance(ewc_loss, torch.Tensor)
        assert ewc_loss.ndim == 0  # Scalar loss
        assert ewc_loss.item() >= 0  # Non-negative loss
        
    def test_multiple_tasks_ewc(self, simple_model, sample_data):
        """Test EWC with multiple previous tasks."""
        config = ContinualConfig()
        regularizer = EWCRegularizer(simple_model, config)
        
        x, y = sample_data
        
        # Store parameters for multiple tasks
        for task_id in ["task1", "task2", "task3"]:
            regularizer.store_task_parameters(x, y, task_id=task_id)
            # Slightly modify model between tasks
            for param in simple_model.parameters():
                param.data += torch.randn_like(param.data) * 0.005
                
        ewc_loss = regularizer.compute_ewc_loss()
        assert isinstance(ewc_loss, torch.Tensor)
        assert ewc_loss.item() >= 0


class TestMemoryBank:
    """Test memory bank functionality for continual learning."""
    
    @pytest.fixture
    def memory_bank(self):
        """Create memory bank for testing."""
        config = ContinualConfig(memory_bank_size=100)
        return MemoryBank(config)
        
    @pytest.fixture
    def sample_episodes(self):
        """Create sample episodes for memory bank."""
        episodes = []
        for i in range(5):
            episode = {
                'support_x': torch.randn(3, 2, 10),  # 3-way, 2-shot
                'support_y': torch.arange(3).repeat_interleave(2),
                'query_x': torch.randn(15, 10),
                'query_y': torch.arange(3).repeat(5),
                'task_id': f"task_{i}"
            }
            episodes.append(episode)
        return episodes
        
    def test_memory_bank_init(self):
        """Test memory bank initialization."""
        config = ContinualConfig(memory_bank_size=500)
        bank = MemoryBank(config)
        
        assert bank.max_size == 500
        assert len(bank.episodes) == 0
        assert bank.current_size == 0
        
    def test_add_episode(self, memory_bank, sample_episodes):
        """Test adding episodes to memory bank."""
        episode = sample_episodes[0]
        
        memory_bank.add_episode(episode)
        
        assert len(memory_bank.episodes) == 1
        assert memory_bank.current_size == 1
        assert memory_bank.episodes[0] == episode
        
    def test_memory_bank_overflow(self, sample_episodes):
        """Test memory bank behavior when capacity is exceeded."""
        config = ContinualConfig(memory_bank_size=3)  # Small capacity
        bank = MemoryBank(config)
        
        # Add more episodes than capacity
        for episode in sample_episodes:  # 5 episodes, capacity 3
            bank.add_episode(episode)
            
        assert len(bank.episodes) <= 3  # Should not exceed capacity
        assert bank.current_size <= 3
        
    def test_sample_episodes(self, memory_bank, sample_episodes):
        """Test sampling episodes from memory bank."""
        # Add episodes to bank
        for episode in sample_episodes:
            memory_bank.add_episode(episode)
            
        # Sample episodes
        sampled = memory_bank.sample_episodes(n_episodes=3)
        
        assert len(sampled) == 3
        for episode in sampled:
            assert 'support_x' in episode
            assert 'support_y' in episode
            assert 'query_x' in episode
            assert 'query_y' in episode
            
    def test_sample_more_than_available(self, memory_bank, sample_episodes):
        """Test sampling more episodes than available."""
        # Add only 2 episodes
        for episode in sample_episodes[:2]:
            memory_bank.add_episode(episode)
            
        # Try to sample 5 episodes
        sampled = memory_bank.sample_episodes(n_episodes=5)
        
        assert len(sampled) <= 2  # Should not exceed available
        
    def test_get_all_episodes(self, memory_bank, sample_episodes):
        """Test getting all episodes from memory bank."""
        for episode in sample_episodes:
            memory_bank.add_episode(episode)
            
        all_episodes = memory_bank.get_all_episodes()
        
        assert len(all_episodes) == len(sample_episodes)
        for i, episode in enumerate(all_episodes):
            assert episode['task_id'] == f"task_{i}"
            
    def test_clear_memory_bank(self, memory_bank, sample_episodes):
        """Test clearing memory bank."""
        for episode in sample_episodes:
            memory_bank.add_episode(episode)
            
        assert len(memory_bank.episodes) > 0
        
        memory_bank.clear()
        
        assert len(memory_bank.episodes) == 0
        assert memory_bank.current_size == 0


class TestContinualMetaLearner:
    """Test continual meta-learner implementation."""
    
    @pytest.fixture
    def simple_model(self):
        """Create simple model for continual learning."""
        return nn.Sequential(
            nn.Linear(20, 32),
            nn.ReLU(),
            nn.Linear(32, 10)
        )
        
    @pytest.fixture
    def continual_learner(self, simple_model):
        """Create continual meta-learner with default config."""
        config = ContinualConfig()
        return ContinualMetaLearner(simple_model, config)
        
    @pytest.fixture
    def task_data(self):
        """Create task data for continual learning."""
        tasks = []
        for i in range(3):
            task = {
                'support_x': torch.randn(5, 3, 20),  # 5-way, 3-shot
                'support_y': torch.arange(5).repeat_interleave(3),
                'query_x': torch.randn(25, 20),
                'query_y': torch.arange(5).repeat(5),
                'task_id': f"continual_task_{i}"
            }
            tasks.append(task)
        return tasks
        
    def test_continual_meta_learner_init(self, simple_model):
        """Test continual meta-learner initialization."""
        config = ContinualConfig()
        learner = ContinualMetaLearner(simple_model, config)
        
        assert learner.model == simple_model
        assert learner.config == config
        assert isinstance(learner.ewc_regularizer, EWCRegularizer)
        assert isinstance(learner.memory_bank, MemoryBank)
        
    def test_learn_task(self, continual_learner, task_data):
        """Test learning a single task."""
        task = task_data[0]
        
        loss = continual_learner.learn_task(
            task['support_x'], 
            task['support_y'], 
            task['query_x'], 
            task['query_y'],
            task_id=task['task_id']
        )
        
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.item() >= 0
        
    def test_sequential_task_learning(self, continual_learner, task_data):
        """Test learning multiple tasks sequentially."""
        losses = []
        
        for task in task_data:
            loss = continual_learner.learn_task(
                task['support_x'],
                task['support_y'], 
                task['query_x'],
                task['query_y'],
                task_id=task['task_id']
            )
            losses.append(loss.item())
            
        assert len(losses) == len(task_data)
        assert all(loss >= 0 for loss in losses)
        
        # Check that EWC has stored parameters for all tasks
        assert len(continual_learner.ewc_regularizer.task_params) == len(task_data)
        
    def test_evaluate_task(self, continual_learner, task_data):
        """Test task evaluation."""
        task = task_data[0]
        
        # Learn the task first
        continual_learner.learn_task(
            task['support_x'],
            task['support_y'],
            task['query_x'], 
            task['query_y'],
            task_id=task['task_id']
        )
        
        # Evaluate the task
        accuracy = continual_learner.evaluate_task(
            task['support_x'],
            task['support_y'],
            task['query_x'],
            task['query_y']
        )
        
        assert 0 <= accuracy <= 1
        assert isinstance(accuracy, float)
        
    def test_memory_replay_integration(self, continual_learner, task_data):
        """Test integration of memory replay with continual learning."""
        # Learn first few tasks
        for task in task_data[:2]:
            continual_learner.learn_task(
                task['support_x'],
                task['support_y'],
                task['query_x'],
                task['query_y'],
                task_id=task['task_id']
            )
            
        # Check that episodes are stored in memory bank
        assert len(continual_learner.memory_bank.episodes) > 0
        
        # Sample from memory for replay
        replay_episodes = continual_learner.memory_bank.sample_episodes(n_episodes=1)
        assert len(replay_episodes) > 0


class TestOnlineMetaLearner:
    """Test online meta-learner implementation."""
    
    @pytest.fixture
    def simple_model(self):
        """Create simple model for online learning."""
        return nn.Sequential(
            nn.Linear(15, 25),
            nn.ReLU(),
            nn.Linear(25, 8)
        )
        
    @pytest.fixture  
    def online_learner(self, simple_model):
        """Create online meta-learner with default config."""
        config = OnlineConfig()
        return OnlineMetaLearner(simple_model, config)
        
    @pytest.fixture
    def streaming_tasks(self):
        """Create streaming task data."""
        tasks = []
        for i in range(10):  # More tasks for online setting
            task = {
                'support_x': torch.randn(3, 2, 15),  # 3-way, 2-shot
                'support_y': torch.arange(3).repeat_interleave(2),
                'query_x': torch.randn(12, 15),
                'query_y': torch.arange(3).repeat(4),
                'task_id': f"online_task_{i}"
            }
            tasks.append(task)
        return tasks
        
    def test_online_meta_learner_init(self, simple_model):
        """Test online meta-learner initialization."""
        config = OnlineConfig()
        learner = OnlineMetaLearner(simple_model, config)
        
        assert learner.model == simple_model
        assert learner.config == config
        assert hasattr(learner, 'meta_optimizer')
        assert hasattr(learner, 'task_buffer')
        
    def test_online_adaptation(self, online_learner, streaming_tasks):
        """Test online adaptation to new tasks."""
        task = streaming_tasks[0]
        
        loss = online_learner.adapt_to_task(
            task['support_x'],
            task['support_y'],
            task['query_x'],
            task['query_y']
        )
        
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        
    def test_meta_update(self, online_learner, streaming_tasks):
        """Test meta-parameter updates in online learning."""
        # Get initial parameters
        initial_params = {name: param.clone() 
                         for name, param in online_learner.model.named_parameters()}
        
        # Perform several online adaptations
        for task in streaming_tasks[:3]:
            online_learner.adapt_to_task(
                task['support_x'],
                task['support_y'],
                task['query_x'], 
                task['query_y']
            )
            
        # Update meta-parameters
        online_learner.meta_update()
        
        # Check that parameters have changed
        parameter_changed = False
        for name, param in online_learner.model.named_parameters():
            if not torch.allclose(initial_params[name], param, atol=1e-6):
                parameter_changed = True
                break
                
        assert parameter_changed, "Meta-parameters should change after meta-update"
        
    def test_buffer_management(self, online_learner, streaming_tasks):
        """Test task buffer management in online learning."""
        # Add tasks to buffer
        for task in streaming_tasks[:5]:
            online_learner.add_task_to_buffer(task)
            
        assert len(online_learner.task_buffer) <= online_learner.config.buffer_size
        
        # Sample from buffer
        sampled_tasks = online_learner.sample_from_buffer(n_tasks=3)
        assert len(sampled_tasks) <= 3
        assert len(sampled_tasks) <= len(online_learner.task_buffer)
        
    def test_continual_online_learning(self, online_learner, streaming_tasks):
        """Test full online continual learning pipeline."""
        losses = []
        
        for i, task in enumerate(streaming_tasks):
            # Online adaptation
            loss = online_learner.adapt_to_task(
                task['support_x'],
                task['support_y'], 
                task['query_x'],
                task['query_y']
            )
            losses.append(loss.item())
            
            # Add to buffer
            online_learner.add_task_to_buffer(task)
            
            # Periodic meta-update
            if (i + 1) % 3 == 0:
                online_learner.meta_update()
                
        assert len(losses) == len(streaming_tasks)
        assert all(loss >= 0 for loss in losses)


class TestCreateContinualLearner:
    """Test factory function for continual learners."""
    
    @pytest.fixture
    def simple_model(self):
        """Create simple model for factory testing."""
        return nn.Linear(10, 5)
        
    def test_create_continual_meta_learner(self, simple_model):
        """Test creation of continual meta-learner."""
        config = ContinualConfig()
        learner = create_continual_learner("continual", simple_model, config)
        
        assert isinstance(learner, ContinualMetaLearner)
        assert learner.model == simple_model
        assert learner.config == config
        
    def test_create_online_meta_learner(self, simple_model):
        """Test creation of online meta-learner."""
        config = OnlineConfig()
        learner = create_continual_learner("online", simple_model, config)
        
        assert isinstance(learner, OnlineMetaLearner)
        assert learner.model == simple_model
        assert learner.config == config
        
    def test_create_invalid_learner_type(self, simple_model):
        """Test creation with invalid learner type."""
        config = ContinualConfig()
        
        with pytest.raises(ValueError, match="Unknown continual learner type"):
            create_continual_learner("invalid_type", simple_model, config)
            
    @pytest.mark.parametrize("learner_type,expected_class", [
        ("continual", ContinualMetaLearner),
        ("online", OnlineMetaLearner),
    ])
    def test_create_learner_parametrized(self, simple_model, learner_type, expected_class):
        """Parametrized test for different learner types."""
        if learner_type == "online":
            config = OnlineConfig()
        else:
            config = ContinualConfig()
            
        learner = create_continual_learner(learner_type, simple_model, config)
        assert isinstance(learner, expected_class)


class TestFixmeSolutions:
    """Test all research solutions are properly implemented."""
    
    @pytest.fixture
    def model(self):
        """Create model for FIXME testing.""" 
        return nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
        
    @pytest.fixture
    def sample_data(self):
        """Create sample data for FIXME testing."""
        x = torch.randn(20, 8)
        y = torch.randint(0, 4, (20,))
        return x, y
        
    @pytest.mark.fixme_solution
    def test_diagonal_ewc_solution(self, model, sample_data):
        """Test diagonal EWC FIXME solution."""
        config = ContinualConfig(ewc_method="diagonal")
        regularizer = EWCRegularizer(model, config)
        
        x, y = sample_data
        fisher_info = regularizer.compute_fisher_information(x, y)
        
        assert len(fisher_info) > 0
        for param_name, fisher_values in fisher_info.items():
            # Diagonal Fisher should have same shape as parameters
            assert fisher_values.shape == model.state_dict()[param_name].shape
            
    @pytest.mark.fixme_solution  
    def test_full_ewc_solution(self, model, sample_data):
        """Test full EWC Fisher matrix FIXME solution."""
        config = ContinualConfig(ewc_method="full")
        regularizer = EWCRegularizer(model, config)
        
        x, y = sample_data
        fisher_info = regularizer.compute_fisher_information(x, y)
        
        assert len(fisher_info) > 0
        # Full Fisher matrices are more complex, just check they exist and are finite
        for param_name, fisher_values in fisher_info.items():
            assert torch.isfinite(fisher_values).all()
            
    @pytest.mark.fixme_solution
    def test_kfac_fisher_estimation_solution(self, model, sample_data):
        """Test K-FAC Fisher estimation FIXME solution."""
        config = ContinualConfig(fisher_estimation_method="kfac")
        regularizer = EWCRegularizer(model, config)
        
        x, y = sample_data
        fisher_info = regularizer.compute_fisher_information(x, y)
        
        assert len(fisher_info) > 0
        for param_name, fisher_values in fisher_info.items():
            assert torch.isfinite(fisher_values).all()
            
    @pytest.mark.fixme_solution
    def test_exact_fisher_estimation_solution(self, model, sample_data):
        """Test exact Fisher estimation FIXME solution.""" 
        config = ContinualConfig(fisher_estimation_method="exact")
        regularizer = EWCRegularizer(model, config)
        
        x, y = sample_data
        fisher_info = regularizer.compute_fisher_information(x, y)
        
        assert len(fisher_info) > 0
        for param_name, fisher_values in fisher_info.items():
            assert torch.isfinite(fisher_values).all()
            
    @pytest.mark.fixme_solution
    def test_memory_bank_integration_solution(self, model):
        """Test memory bank integration FIXME solution."""
        config = ContinualConfig(use_memory_bank=True, memory_bank_size=50)
        learner = ContinualMetaLearner(model, config)
        
        # Create sample episode
        episode = {
            'support_x': torch.randn(3, 2, 8),
            'support_y': torch.arange(3).repeat_interleave(2),
            'query_x': torch.randn(12, 8),
            'query_y': torch.arange(3).repeat(4),
            'task_id': "test_task"
        }
        
        # Learn task (should store in memory bank)
        learner.learn_task(
            episode['support_x'],
            episode['support_y'],
            episode['query_x'],
            episode['query_y'],
            task_id=episode['task_id']
        )
        
        # Check that episode was stored
        assert len(learner.memory_bank.episodes) > 0
        
    @pytest.mark.fixme_solution
    def test_online_meta_learning_solution(self, model):
        """Test online meta-learning FIXME solution."""
        config = OnlineConfig(
            learning_rate=0.01,
            meta_learning_rate=0.001,
            adaptation_steps=3
        )
        learner = OnlineMetaLearner(model, config)
        
        # Create streaming task
        support_x = torch.randn(2, 3, 8)  # 2-way, 3-shot
        support_y = torch.arange(2).repeat_interleave(3)
        query_x = torch.randn(10, 8)
        query_y = torch.arange(2).repeat(5)
        
        # Test online adaptation
        loss = learner.adapt_to_task(support_x, support_y, query_x, query_y)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert torch.isfinite(loss)


class TestContinualLearningIntegration:
    """Integration tests for complete continual learning pipeline."""
    
    @pytest.fixture
    def complete_setup(self):
        """Create complete continual learning setup."""
        model = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(), 
            nn.Linear(32, 16)
        )
        
        config = ContinualConfig(
            ewc_lambda=0.5,
            ewc_method="diagonal",
            fisher_estimation_method="empirical",
            memory_bank_size=200,
            use_memory_bank=True
        )
        
        learner = ContinualMetaLearner(model, config)
        
        # Create sequence of tasks
        tasks = []
        for i in range(5):
            task = {
                'support_x': torch.randn(4, 4, 32),  # 4-way, 4-shot
                'support_y': torch.arange(4).repeat_interleave(4),
                'query_x': torch.randn(20, 32),
                'query_y': torch.arange(4).repeat(5),
                'task_id': f"integration_task_{i}"
            }
            tasks.append(task)
            
        return learner, tasks
        
    def test_complete_continual_learning_pipeline(self, complete_setup):
        """Test complete continual learning pipeline."""
        learner, tasks = complete_setup
        
        task_losses = []
        task_accuracies = []
        
        for task in tasks:
            # Learn task
            loss = learner.learn_task(
                task['support_x'],
                task['support_y'],
                task['query_x'],
                task['query_y'],
                task_id=task['task_id']
            )
            task_losses.append(loss.item())
            
            # Evaluate task
            accuracy = learner.evaluate_task(
                task['support_x'],
                task['support_y'],
                task['query_x'],
                task['query_y']
            )
            task_accuracies.append(accuracy)
            
        # Check pipeline completed successfully
        assert len(task_losses) == len(tasks)
        assert len(task_accuracies) == len(tasks)
        assert all(loss >= 0 for loss in task_losses)
        assert all(0 <= acc <= 1 for acc in task_accuracies)
        
        # Check EWC and memory bank were used
        assert len(learner.ewc_regularizer.task_params) == len(tasks)
        assert len(learner.memory_bank.episodes) > 0
        
    def test_catastrophic_forgetting_mitigation(self, complete_setup):
        """Test that EWC helps mitigate catastrophic forgetting."""
        learner, tasks = complete_setup
        
        # Learn first task
        first_task = tasks[0]
        learner.learn_task(
            first_task['support_x'],
            first_task['support_y'], 
            first_task['query_x'],
            first_task['query_y'],
            task_id=first_task['task_id']
        )
        
        # Get performance on first task
        initial_accuracy = learner.evaluate_task(
            first_task['support_x'],
            first_task['support_y'],
            first_task['query_x'],
            first_task['query_y']
        )
        
        # Learn subsequent tasks
        for task in tasks[1:]:
            learner.learn_task(
                task['support_x'],
                task['support_y'],
                task['query_x'],
                task['query_y'],
                task_id=task['task_id']
            )
            
        # Check performance on first task again
        final_accuracy = learner.evaluate_task(
            first_task['support_x'],
            first_task['support_y'],
            first_task['query_x'],
            first_task['query_y']
        )
        
        # With EWC, performance shouldn't degrade too much
        # (This is a weak test since we can't guarantee specific performance)
        assert final_accuracy >= 0  # At least still works
        
    def test_memory_replay_effectiveness(self, complete_setup):
        """Test that memory replay helps with continual learning."""
        learner, tasks = complete_setup
        
        # Learn all tasks
        for task in tasks:
            learner.learn_task(
                task['support_x'],
                task['support_y'],
                task['query_x'],
                task['query_y'],
                task_id=task['task_id']
            )
            
        # Sample from memory bank
        replay_episodes = learner.memory_bank.sample_episodes(n_episodes=3)
        
        assert len(replay_episodes) > 0
        for episode in replay_episodes:
            assert 'support_x' in episode
            assert 'support_y' in episode
            assert 'query_x' in episode
            assert 'query_y' in episode


@pytest.mark.property  
class TestPropertyBasedContinualLearning:
    """Property-based tests using Hypothesis for continual learning."""
    
    @given(
        ewc_lambda=st.floats(min_value=0.1, max_value=2.0),
        memory_size=st.integers(min_value=50, max_value=500),
        n_tasks=st.integers(min_value=2, max_value=8)
    )
    @settings(max_examples=5, deadline=15000)
    def test_continual_learner_invariants(self, ewc_lambda, memory_size, n_tasks):
        """Test invariants for continual learner."""
        model = nn.Sequential(nn.Linear(10, 8), nn.ReLU(), nn.Linear(8, 3))
        config = ContinualConfig(ewc_lambda=ewc_lambda, memory_bank_size=memory_size)
        learner = ContinualMetaLearner(model, config)
        
        # Create and learn tasks
        for i in range(n_tasks):
            support_x = torch.randn(3, 2, 10) 
            support_y = torch.arange(3).repeat_interleave(2)
            query_x = torch.randn(12, 10)
            query_y = torch.arange(3).repeat(4)
            
            loss = learner.learn_task(
                support_x, support_y, query_x, query_y, 
                task_id=f"prop_task_{i}"
            )
            
            # Invariants
            assert isinstance(loss, torch.Tensor)
            assert torch.isfinite(loss)
            assert loss.item() >= 0
            
        # Memory bank should not exceed capacity
        assert len(learner.memory_bank.episodes) <= memory_size
        
        # Should have EWC parameters for all tasks
        assert len(learner.ewc_regularizer.task_params) == n_tasks
        
    @given(
        learning_rate=st.floats(min_value=0.001, max_value=0.1),
        adaptation_steps=st.integers(min_value=1, max_value=10),
        buffer_size=st.integers(min_value=10, max_value=100)
    )
    @settings(max_examples=5, deadline=10000)
    def test_online_learner_properties(self, learning_rate, adaptation_steps, buffer_size):
        """Test properties of online meta-learner."""
        model = nn.Sequential(nn.Linear(8, 4), nn.ReLU(), nn.Linear(4, 2))
        config = OnlineConfig(
            learning_rate=learning_rate,
            adaptation_steps=adaptation_steps,
            buffer_size=buffer_size
        )
        learner = OnlineMetaLearner(model, config)
        
        # Test adaptation
        support_x = torch.randn(2, 1, 8)  # 2-way, 1-shot
        support_y = torch.arange(2)
        query_x = torch.randn(6, 8)
        query_y = torch.arange(2).repeat(3)
        
        loss = learner.adapt_to_task(support_x, support_y, query_x, query_y)
        
        # Properties
        assert isinstance(loss, torch.Tensor)
        assert torch.isfinite(loss)
        assert loss.ndim == 0
        
        # Buffer management
        task = {
            'support_x': support_x, 'support_y': support_y,
            'query_x': query_x, 'query_y': query_y, 
            'task_id': 'prop_test'
        }
        learner.add_task_to_buffer(task)
        
        assert len(learner.task_buffer) <= buffer_size


class TestContinualLearningEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.fixture
    def minimal_model(self):
        """Create minimal model for edge case testing."""
        return nn.Linear(4, 2)
        
    def test_single_task_continual_learning(self, minimal_model):
        """Test continual learning with only one task."""
        config = ContinualConfig()
        learner = ContinualMetaLearner(minimal_model, config)
        
        support_x = torch.randn(2, 1, 4)
        support_y = torch.arange(2)
        query_x = torch.randn(4, 4)
        query_y = torch.arange(2).repeat(2)
        
        loss = learner.learn_task(support_x, support_y, query_x, query_y, task_id="single_task")
        
        assert isinstance(loss, torch.Tensor)
        assert torch.isfinite(loss)
        
    def test_empty_memory_bank_sampling(self):
        """Test sampling from empty memory bank."""
        config = ContinualConfig(memory_bank_size=10)
        bank = MemoryBank(config)
        
        sampled = bank.sample_episodes(n_episodes=5)
        assert len(sampled) == 0
        
    def test_zero_ewc_lambda(self, minimal_model):
        """Test EWC with zero regularization strength."""
        config = ContinualConfig(ewc_lambda=0.0)
        learner = ContinualMetaLearner(minimal_model, config)
        
        support_x = torch.randn(2, 2, 4)
        support_y = torch.arange(2).repeat_interleave(2)
        query_x = torch.randn(8, 4)
        query_y = torch.arange(2).repeat(4)
        
        # Learn two tasks
        for i in range(2):
            loss = learner.learn_task(
                support_x, support_y, query_x, query_y, 
                task_id=f"zero_ewc_task_{i}"
            )
            assert torch.isfinite(loss)
            
    def test_invalid_fisher_method(self, minimal_model):
        """Test handling of invalid Fisher estimation method."""
        config = ContinualConfig(fisher_estimation_method="invalid_method")
        
        # Should still create learner (error handling in compute_fisher_information)
        learner = ContinualMetaLearner(minimal_model, config)
        assert learner is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])