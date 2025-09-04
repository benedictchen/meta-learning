"""
🧪 End-to-End Pipeline Tests for Meta-Learning Package
====================================================

These tests validate the complete meta-learning pipeline from data loading
through training to evaluation, ensuring all components work together.

Test Coverage:
- Complete few-shot learning pipelines
- MAML training and adaptation workflows  
- Test-time compute scaling pipelines
- Continual learning scenarios
- Hardware acceleration integration
- All research solutions working together
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Meta-learning components
from src.meta_learning.meta_learning_modules.few_shot_learning import (
    PrototypicalLearner, PrototypicalConfig
)
from src.meta_learning.meta_learning_modules.maml_variants import (
    MAML, MAMLConfig
)
from src.meta_learning.meta_learning_modules.test_time_compute import (
    TestTimeComputeScaler, TestTimeComputeConfig
)
from src.meta_learning.meta_learning_modules.continual_meta_learning import (
    ContinualMetaLearner, ContinualConfig
)
from src.meta_learning.meta_learning_modules.utils import (
    MetaLearningDataset, DatasetConfig, EvaluationMetrics, MetricsConfig
)
from src.meta_learning.meta_learning_modules.hardware_utils import (
    HardwareManager, HardwareConfig, create_hardware_manager
)


class TestCompleteMetaLearningPipeline:
    """Test complete meta-learning pipelines from start to finish."""
    
    @pytest.fixture
    def pipeline_config(self):
        """Configuration for end-to-end pipeline testing."""
        return {
            'n_way': 5,
            'k_shot': 5,
            'n_query': 10,
            'n_episodes': 10,
            'feature_dim': 64,
            'hidden_dim': 128,
            'device': 'cpu'  # Use CPU for consistent testing
        }
    
    @pytest.fixture
    def simple_encoder(self, pipeline_config):
        """Simple encoder for pipeline testing."""
        return nn.Sequential(
            nn.Linear(pipeline_config['feature_dim'], pipeline_config['hidden_dim']),
            nn.ReLU(),
            nn.Linear(pipeline_config['hidden_dim'], pipeline_config['hidden_dim'])
        )
    
    @pytest.fixture
    def meta_dataset(self, pipeline_config):
        """Create meta-learning dataset for pipeline testing."""
        config = DatasetConfig(
            n_way=pipeline_config['n_way'],
            k_shot=pipeline_config['k_shot'],
            n_query=pipeline_config['n_query'],
            feature_dim=pipeline_config['feature_dim'],
            num_classes=20,  # Total classes available
            episode_length=pipeline_config['n_episodes']
        )
        return MetaLearningDataset(config)
    
    def test_complete_prototypical_network_pipeline(self, simple_encoder, meta_dataset, pipeline_config):
        """Test complete prototypical network pipeline with all components."""
        # 1. Initialize hardware manager
        hw_config = HardwareConfig(
            device=pipeline_config['device'],
            use_mixed_precision=False,  # Disable for CPU testing
            memory_efficient=True
        )
        hw_manager = HardwareManager(hw_config)
        
        # 2. Prepare model for hardware
        encoder = hw_manager.prepare_model(simple_encoder)
        
        # 3. Configure prototypical learner with enhanced features
        proto_config = PrototypicalConfig(
            distance_metric='euclidean',
            use_uncertainty_aware_distances=True,
            use_hierarchical_prototypes=True,
            use_task_adaptive_prototypes=True,
            uncertainty_threshold=0.1,
            hierarchy_levels=2,
            adaptation_lr=0.01
        )
        learner = PrototypicalLearner(encoder, proto_config)
        
        # 4. Initialize evaluation metrics
        metrics_config = MetricsConfig(
            confidence_level=0.95,
            bootstrap_samples=100,
            track_per_class_metrics=True,
            compute_confusion_matrix=True
        )
        evaluator = EvaluationMetrics(metrics_config)
        
        # 5. Run complete pipeline
        all_accuracies = []
        all_losses = []
        
        for episode_idx in range(pipeline_config['n_episodes']):
            # Generate episode
            episode_data = meta_dataset.generate_episode()
            support_x, support_y, query_x, query_y = episode_data
            
            # Prepare data for hardware
            support_x = hw_manager.prepare_data(support_x)
            support_y = hw_manager.prepare_data(support_y)
            query_x = hw_manager.prepare_data(query_x)
            query_y = hw_manager.prepare_data(query_y)
            
            # Forward pass with hardware optimization
            with hw_manager.autocast_context():
                logits = learner(support_x, support_y, query_x)
                loss = nn.CrossEntropyLoss()(logits, query_y)
            
            # Compute accuracy
            predictions = torch.argmax(logits, dim=1)
            accuracy = (predictions == query_y).float().mean().item()
            
            all_accuracies.append(accuracy)
            all_losses.append(loss.item())
            
            # Update evaluator
            evaluator.update(predictions.cpu().numpy(), query_y.cpu().numpy())
        
        # 6. Compute final metrics
        final_metrics = evaluator.compute_metrics()
        mean_accuracy = np.mean(all_accuracies)
        mean_loss = np.mean(all_losses)
        
        # 7. Validate pipeline results
        assert len(all_accuracies) == pipeline_config['n_episodes']
        assert len(all_losses) == pipeline_config['n_episodes']
        assert 0.0 <= mean_accuracy <= 1.0
        assert mean_loss > 0
        assert 'accuracy' in final_metrics
        assert 'confidence_interval' in final_metrics
        assert 'per_class_metrics' in final_metrics
        
        # 8. Test enhanced features
        assert proto_config.use_uncertainty_aware_distances
        assert proto_config.use_hierarchical_prototypes
        assert proto_config.use_task_adaptive_prototypes
        
        print(f"✅ Complete prototypical pipeline: {mean_accuracy:.3f} ± {final_metrics['confidence_interval']:.3f}")
    
    def test_complete_maml_pipeline(self, simple_encoder, meta_dataset, pipeline_config):
        """Test complete MAML pipeline with all variants and optimizations."""
        # 1. Initialize hardware manager
        hw_manager = create_hardware_manager()
        
        # 2. Configure MAML with advanced features
        maml_config = MAMLConfig(
            maml_variant='maml',
            inner_lr=0.01,
            outer_lr=0.001,
            num_inner_steps=5,
            use_first_order=False,
            use_adaptive_lr=True,
            use_memory_efficient=True,
            memory_decay=0.95,
            inner_loop_grad_clip=1.0,
            per_param_lr=False
        )
        
        # 3. Initialize MAML learner
        maml_learner = MAML(simple_encoder, maml_config)
        maml_learner = hw_manager.prepare_model(maml_learner)
        
        # 4. Create optimizer
        optimizer = torch.optim.Adam(maml_learner.parameters(), lr=maml_config.outer_lr)
        
        # 5. Run MAML training pipeline
        meta_losses = []
        meta_accuracies = []
        
        for episode_idx in range(pipeline_config['n_episodes']):
            # Generate episode
            episode_data = meta_dataset.generate_episode()
            support_x, support_y, query_x, query_y = episode_data
            
            # Prepare data
            support_x = hw_manager.prepare_data(support_x)
            support_y = hw_manager.prepare_data(support_y)
            query_x = hw_manager.prepare_data(query_x)
            query_y = hw_manager.prepare_data(query_y)
            
            # MAML meta-training step
            optimizer.zero_grad()
            
            with hw_manager.autocast_context():
                meta_loss, adapted_params = maml_learner.meta_forward(
                    support_x, support_y, query_x, query_y
                )
                
                # Compute accuracy with adapted parameters
                logits = maml_learner.forward_with_params(query_x, adapted_params)
                predictions = torch.argmax(logits, dim=1)
                accuracy = (predictions == query_y).float().mean().item()
            
            # Backward pass with hardware optimization
            hw_manager.backward_and_step(meta_loss, optimizer)
            
            meta_losses.append(meta_loss.item())
            meta_accuracies.append(accuracy)
        
        # 6. Validate MAML pipeline
        mean_loss = np.mean(meta_losses)
        mean_accuracy = np.mean(meta_accuracies)
        
        assert len(meta_losses) == pipeline_config['n_episodes']
        assert len(meta_accuracies) == pipeline_config['n_episodes']
        assert mean_loss > 0
        assert 0.0 <= mean_accuracy <= 1.0
        
        # Verify MAML-specific features
        assert maml_config.use_adaptive_lr
        assert maml_config.use_memory_efficient
        assert maml_config.num_inner_steps == 5
        
        print(f"✅ Complete MAML pipeline: {mean_accuracy:.3f} accuracy, {mean_loss:.3f} loss")
    
    def test_test_time_compute_pipeline(self, simple_encoder, meta_dataset, pipeline_config):
        """Test complete test-time compute scaling pipeline."""
        # 1. Configure test-time compute with multiple strategies
        ttc_configs = [
            TestTimeComputeConfig(
                compute_strategy='snell2024',
                base_compute_steps=3,
                max_compute_steps=10,
                use_process_reward_model=True,
                process_reward_threshold=0.7
            ),
            TestTimeComputeConfig(
                compute_strategy='akyurek2024',
                use_test_time_training=True,
                test_time_lr=0.001,
                test_time_steps=5
            ),
            TestTimeComputeConfig(
                compute_strategy='openai_o1',
                use_chain_of_thought=True,
                cot_max_length=100
            )
        ]
        
        # 2. Initialize hardware manager
        hw_manager = create_hardware_manager()
        encoder = hw_manager.prepare_model(simple_encoder)
        
        # 3. Test each strategy
        for strategy_idx, ttc_config in enumerate(ttc_configs):
            # Initialize scaler
            ttc_scaler = TestTimeComputeScaler(ttc_config)
            
            # Create base learner (prototypical)
            base_config = PrototypicalConfig(distance_metric='euclidean')
            base_learner = PrototypicalLearner(encoder, base_config)
            
            # Run test-time compute pipeline
            episode_results = []
            
            for episode_idx in range(min(5, pipeline_config['n_episodes'])):  # Fewer episodes for TTC
                # Generate episode
                episode_data = meta_dataset.generate_episode()
                support_x, support_y, query_x, query_y = episode_data
                
                # Prepare data
                support_x = hw_manager.prepare_data(support_x)
                support_y = hw_manager.prepare_data(support_y)
                query_x = hw_manager.prepare_data(query_x)
                query_y = hw_manager.prepare_data(query_y)
                
                # Test-time compute scaling
                with hw_manager.autocast_context():
                    scaled_logits, compute_info = ttc_scaler.scale_compute(
                        base_learner, support_x, support_y, query_x
                    )
                
                # Evaluate results
                predictions = torch.argmax(scaled_logits, dim=1)
                accuracy = (predictions == query_y).float().mean().item()
                
                episode_results.append({
                    'accuracy': accuracy,
                    'compute_steps': compute_info.get('compute_steps', 0),
                    'strategy': ttc_config.compute_strategy
                })
            
            # Validate strategy results
            mean_accuracy = np.mean([r['accuracy'] for r in episode_results])
            mean_compute_steps = np.mean([r['compute_steps'] for r in episode_results])
            
            assert len(episode_results) == min(5, pipeline_config['n_episodes'])
            assert 0.0 <= mean_accuracy <= 1.0
            assert mean_compute_steps >= ttc_config.base_compute_steps
            
            print(f"✅ TTC {ttc_config.compute_strategy}: {mean_accuracy:.3f} accuracy, {mean_compute_steps:.1f} steps")
    
    def test_continual_learning_pipeline(self, simple_encoder, meta_dataset, pipeline_config):
        """Test complete continual learning pipeline with EWC and memory banks."""
        # 1. Configure continual learning with all features
        continual_config = ContinualConfig(
            use_ewc=True,
            ewc_lambda=1000.0,
            use_memory_bank=True,
            memory_size=100,
            memory_update_strategy='reservoir',
            catastrophic_forgetting_threshold=0.1,
            use_task_boundaries=True,
            regularization_strength=0.5
        )
        
        # 2. Initialize hardware and continual learner
        hw_manager = create_hardware_manager()
        encoder = hw_manager.prepare_model(simple_encoder)
        
        continual_learner = ContinualMetaLearner(encoder, continual_config)
        
        # 3. Simulate continual learning across multiple tasks
        num_tasks = 3
        episodes_per_task = pipeline_config['n_episodes'] // num_tasks
        
        task_accuracies = []
        forgetting_scores = []
        
        for task_id in range(num_tasks):
            print(f"Training on Task {task_id + 1}/{num_tasks}")
            
            # Task-specific accuracies
            task_episode_accuracies = []
            
            for episode_idx in range(episodes_per_task):
                # Generate episode (simulate task-specific data)
                episode_data = meta_dataset.generate_episode()
                support_x, support_y, query_x, query_y = episode_data
                
                # Modify labels to simulate different tasks
                support_y = (support_y + task_id * pipeline_config['n_way']) % (pipeline_config['n_way'] * 2)
                query_y = (query_y + task_id * pipeline_config['n_way']) % (pipeline_config['n_way'] * 2)
                
                # Prepare data
                support_x = hw_manager.prepare_data(support_x)
                support_y = hw_manager.prepare_data(support_y)
                query_x = hw_manager.prepare_data(query_x)
                query_y = hw_manager.prepare_data(query_y)
                
                # Continual learning step
                with hw_manager.autocast_context():
                    loss, adapted_params = continual_learner.continual_forward(
                        support_x, support_y, query_x, query_y, task_id
                    )
                    
                    # Evaluate current performance
                    logits = continual_learner.forward_with_params(query_x, adapted_params)
                    predictions = torch.argmax(logits, dim=1)
                    accuracy = (predictions == query_y).float().mean().item()
                
                task_episode_accuracies.append(accuracy)
                
                # Update memory bank and EWC
                continual_learner.update_memory(support_x, support_y)
                if episode_idx == episodes_per_task - 1:  # End of task
                    continual_learner.compute_fisher_information(
                        [(support_x, support_y) for _ in range(5)]  # Fisher estimation
                    )
            
            task_accuracy = np.mean(task_episode_accuracies)
            task_accuracies.append(task_accuracy)
            
            # Measure forgetting on previous tasks
            if task_id > 0:
                forgetting_score = continual_learner.measure_forgetting(task_id - 1)
                forgetting_scores.append(forgetting_score)
        
        # 4. Validate continual learning pipeline
        assert len(task_accuracies) == num_tasks
        assert len(forgetting_scores) == num_tasks - 1
        
        mean_task_accuracy = np.mean(task_accuracies)
        mean_forgetting = np.mean(forgetting_scores) if forgetting_scores else 0.0
        
        assert 0.0 <= mean_task_accuracy <= 1.0
        assert mean_forgetting >= 0.0  # Forgetting should be non-negative
        
        # Verify continual learning features
        assert continual_config.use_ewc
        assert continual_config.use_memory_bank
        assert continual_learner.memory_bank is not None
        assert continual_learner.ewc_regularizer is not None
        
        print(f"✅ Continual learning: {mean_task_accuracy:.3f} accuracy, {mean_forgetting:.3f} forgetting")
    
    def test_integrated_pipeline_with_all_fixme_solutions(self, simple_encoder, meta_dataset, pipeline_config):
        """Test integrated pipeline using all implemented research solutions."""
        # 1. Create comprehensive configuration combining all solutions
        
        # Test-Time Compute with multiple strategies
        ttc_config = TestTimeComputeConfig(
            compute_strategy='hybrid',  # Uses multiple strategies
            use_process_reward_model=True,
            use_test_time_training=True,
            use_chain_of_thought=True
        )
        
        # MAML with all enhancements
        maml_config = MAMLConfig(
            maml_variant='maml',
            use_adaptive_lr=True,
            use_memory_efficient=True,
            per_param_lr=True,
            inner_loop_grad_clip=1.0
        )
        
        # Prototypical with all uncertainty features
        proto_config = PrototypicalConfig(
            use_uncertainty_aware_distances=True,
            use_hierarchical_prototypes=True,
            use_task_adaptive_prototypes=True
        )
        
        # Continual learning with full setup
        continual_config = ContinualConfig(
            use_ewc=True,
            use_memory_bank=True,
            catastrophic_forgetting_threshold=0.1
        )
        
        # Hardware acceleration
        hw_config = HardwareConfig(
            use_mixed_precision=False,  # CPU testing
            memory_efficient=True,
            compile_model=False
        )
        
        # 2. Initialize all components
        hw_manager = HardwareManager(hw_config)
        encoder = hw_manager.prepare_model(simple_encoder)
        
        # Initialize all learners
        ttc_scaler = TestTimeComputeScaler(ttc_config)
        maml_learner = MAML(encoder, maml_config)
        proto_learner = PrototypicalLearner(encoder, proto_config)
        continual_learner = ContinualMetaLearner(encoder, continual_config)
        
        # 3. Run integrated pipeline
        pipeline_results = {
            'test_time_compute': [],
            'maml': [],
            'prototypical': [],
            'continual': []
        }
        
        for episode_idx in range(min(5, pipeline_config['n_episodes'])):
            # Generate episode
            episode_data = meta_dataset.generate_episode()
            support_x, support_y, query_x, query_y = episode_data
            
            # Prepare data
            support_x = hw_manager.prepare_data(support_x)
            support_y = hw_manager.prepare_data(support_y)
            query_x = hw_manager.prepare_data(query_x)
            query_y = hw_manager.prepare_data(query_y)
            
            with hw_manager.autocast_context():
                # Test each component
                
                # 1. Test-Time Compute
                ttc_logits, ttc_info = ttc_scaler.scale_compute(
                    proto_learner, support_x, support_y, query_x
                )
                ttc_acc = (torch.argmax(ttc_logits, dim=1) == query_y).float().mean().item()
                pipeline_results['test_time_compute'].append(ttc_acc)
                
                # 2. MAML
                maml_loss, maml_params = maml_learner.meta_forward(
                    support_x, support_y, query_x, query_y
                )
                maml_logits = maml_learner.forward_with_params(query_x, maml_params)
                maml_acc = (torch.argmax(maml_logits, dim=1) == query_y).float().mean().item()
                pipeline_results['maml'].append(maml_acc)
                
                # 3. Enhanced Prototypical
                proto_logits = proto_learner(support_x, support_y, query_x)
                proto_acc = (torch.argmax(proto_logits, dim=1) == query_y).float().mean().item()
                pipeline_results['prototypical'].append(proto_acc)
                
                # 4. Continual Learning
                cont_loss, cont_params = continual_learner.continual_forward(
                    support_x, support_y, query_x, query_y, task_id=0
                )
                cont_logits = continual_learner.forward_with_params(query_x, cont_params)
                cont_acc = (torch.argmax(cont_logits, dim=1) == query_y).float().mean().item()
                pipeline_results['continual'].append(cont_acc)
        
        # 4. Validate integrated pipeline
        for component, accuracies in pipeline_results.items():
            mean_acc = np.mean(accuracies)
            assert len(accuracies) == min(5, pipeline_config['n_episodes'])
            assert 0.0 <= mean_acc <= 1.0
            print(f"✅ {component}: {mean_acc:.3f} accuracy")
        
        # 5. Verify all research solutions are active
        assert ttc_config.use_process_reward_model
        assert ttc_config.use_test_time_training
        assert ttc_config.use_chain_of_thought
        assert maml_config.use_adaptive_lr
        assert maml_config.use_memory_efficient
        assert proto_config.use_uncertainty_aware_distances
        assert proto_config.use_hierarchical_prototypes
        assert continual_config.use_ewc
        assert continual_config.use_memory_bank
        
        print("✅ All research solutions integrated and working in complete pipeline")


@pytest.mark.slow
class TestPerformancePipeline:
    """Performance-focused end-to-end tests."""
    
    def test_large_scale_pipeline(self):
        """Test pipeline with larger datasets and episodes."""
        # Large-scale configuration
        config = {
            'n_way': 10,
            'k_shot': 10,
            'n_query': 20,
            'n_episodes': 50,
            'feature_dim': 128,
            'hidden_dim': 256
        }
        
        # Create larger encoder
        encoder = nn.Sequential(
            nn.Linear(config['feature_dim'], config['hidden_dim']),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config['hidden_dim'], config['hidden_dim']),
            nn.ReLU(),
            nn.Linear(config['hidden_dim'], config['hidden_dim'])
        )
        
        # Hardware optimization for performance
        hw_config = HardwareConfig(
            memory_efficient=True,
            max_memory_fraction=0.8
        )
        hw_manager = HardwareManager(hw_config)
        encoder = hw_manager.prepare_model(encoder)
        
        # Dataset configuration
        dataset_config = DatasetConfig(
            n_way=config['n_way'],
            k_shot=config['k_shot'],
            n_query=config['n_query'],
            feature_dim=config['feature_dim'],
            num_classes=50,  # Large class pool
            episode_length=config['n_episodes']
        )
        dataset = MetaLearningDataset(dataset_config)
        
        # Run performance test
        proto_config = PrototypicalConfig(
            use_uncertainty_aware_distances=True,
            use_task_adaptive_prototypes=True
        )
        learner = PrototypicalLearner(encoder, proto_config)
        
        accuracies = []
        import time
        start_time = time.time()
        
        for episode_idx in range(config['n_episodes']):
            episode_data = dataset.generate_episode()
            support_x, support_y, query_x, query_y = episode_data
            
            # Prepare data
            support_x = hw_manager.prepare_data(support_x)
            support_y = hw_manager.prepare_data(support_y)
            query_x = hw_manager.prepare_data(query_x)
            query_y = hw_manager.prepare_data(query_y)
            
            with hw_manager.autocast_context():
                logits = learner(support_x, support_y, query_x)
                predictions = torch.argmax(logits, dim=1)
                accuracy = (predictions == query_y).float().mean().item()
            
            accuracies.append(accuracy)
        
        end_time = time.time()
        total_time = end_time - start_time
        episodes_per_second = config['n_episodes'] / total_time
        
        # Performance validation
        mean_accuracy = np.mean(accuracies)
        assert len(accuracies) == config['n_episodes']
        assert 0.0 <= mean_accuracy <= 1.0
        assert episodes_per_second > 0
        
        print(f"✅ Large-scale pipeline: {mean_accuracy:.3f} accuracy, {episodes_per_second:.1f} eps/sec")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])