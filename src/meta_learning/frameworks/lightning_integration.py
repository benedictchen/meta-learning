"""
PyTorch Lightning Integration Module
===================================

PyTorch Lightning integration for meta-learning algorithms,
enabling distributed training, automatic logging, checkpointing, and modern ML workflows.

Core functionality implemented:
- MetaLearningLightningModule with MAML, ProtoNet, Meta-SGD support
- Training/validation steps with proper metric logging
- Optimizer configuration with learning rate scheduling
- Algorithm-specific adaptation methods

RESEARCH FOUNDATIONS:
- Falcon et al. (2019): PyTorch Lightning framework design
- Modern MLOps best practices for research reproducibility
- Distributed meta-learning training patterns
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Union, Callable
from abc import ABC, abstractmethod

try:
    import lightning as L
    from lightning.pytorch import LightningModule, Trainer
    from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
    from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
    LIGHTNING_AVAILABLE = True
except ImportError:
    # Graceful fallback when Lightning not installed
    class LightningModule:
        pass
    L = None
    Trainer = None
    LIGHTNING_AVAILABLE = False

from ..shared.types import Episode
from ..algorithms import inner_adapt_and_eval, meta_outer_step, ProtoHead
from ..eval import evaluate


class MetaLearningLightningModule(LightningModule):
    """
    Base Lightning module for meta-learning algorithms.
    
    This class provides common functionality for all meta-learning algorithms
    including episode handling, metric logging, and distributed training support.
    """
    
    def __init__(self, model: nn.Module, algorithm: str = "maml", 
                 learning_rate: float = 1e-3, meta_lr: float = 1e-3,
                 **kwargs):
        """
        Initialize meta-learning Lightning module.
        
        Args:
            model: Base neural network model
            algorithm: Meta-learning algorithm ("maml", "protonet", "meta_sgd", etc.)
            learning_rate: Inner loop learning rate
            meta_lr: Meta-learning (outer loop) learning rate
            **kwargs: Additional algorithm-specific parameters
        """
        super().__init__()
        
        # Store model and algorithm configuration
        self.model = model
        self.algorithm = algorithm
        self.learning_rate = learning_rate 
        self.meta_lr = meta_lr
        self.algorithm_kwargs = kwargs
        
        # Save hyperparameters for Lightning automatic logging
        self.save_hyperparameters(ignore=['model'])
        
        # Initialize algorithm-specific components
        # Based on algorithm type, set up the appropriate meta-learning components:
        if algorithm == "maml":
            # MAML requires gradient computation through inner loop
            self.inner_steps = kwargs.get('inner_steps', 5)
            self.first_order = kwargs.get('first_order', False)
        elif algorithm == "protonet":
            # Prototypical Networks use distance-based classification
            self.distance_metric = kwargs.get('distance_metric', 'euclidean')
            self.temperature = kwargs.get('temperature', 1.0)
        elif algorithm == "meta_sgd":
            # Meta-SGD learns per-parameter learning rates
            self.learnable_lrs = kwargs.get('learnable_lrs', True)
        
        # Initialize metrics tracking
        try:
            import torchmetrics
            self.train_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=kwargs.get('num_classes', 5))
            self.val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=kwargs.get('num_classes', 5))
        except ImportError:
            self.train_accuracy = None
            self.val_accuracy = None
    
    def forward(self, x):
        """Forward pass through the model."""
        return self.model(x)
    
    def training_step(self, batch: Episode, batch_idx: int) -> torch.Tensor:
        """
        Training step for meta-learning episode.
        
        Args:
            batch: Episode with support and query data
            batch_idx: Batch index
            
        Returns:
            Loss tensor for backpropagation
        """
        
        # Extract episode data
        support_data, support_labels = batch.support_x, batch.support_y
        query_data, query_labels = batch.query_x, batch.query_y
        
        # Route to algorithm-specific training
        if self.algorithm == "maml":
            # MAML inner loop adaptation
            adapted_model = self._maml_inner_loop(support_data, support_labels)
            query_logits = adapted_model(query_data)
            loss = torch.nn.functional.cross_entropy(query_logits, query_labels)
        elif self.algorithm == "protonet":
            # Prototypical Networks
            prototypes = self._compute_prototypes(support_data, support_labels)
            query_logits = self._classify_queries(query_data, prototypes)
            loss = torch.nn.functional.cross_entropy(query_logits, query_labels)
        elif self.algorithm == "meta_sgd":
            # Meta-SGD with learnable learning rates
            adapted_model = self._meta_sgd_adapt(support_data, support_labels)
            query_logits = adapted_model(query_data)
            loss = torch.nn.functional.cross_entropy(query_logits, query_labels)
        else:
            # Fallback to basic adaptation
            from ..algorithms import inner_adapt_and_eval
            loss, query_logits = inner_adapt_and_eval(
                self.model, support_data, support_labels, query_data, query_labels,
                num_steps=getattr(self, 'inner_steps', 5), lr=self.learning_rate, 
                first_order=getattr(self, 'first_order', False)
            )
        
        # Log metrics
        if self.train_accuracy is not None:
            accuracy = self.train_accuracy(query_logits, query_labels)
            self.log('train_accuracy', accuracy, on_step=True, on_epoch=True, prog_bar=True)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch: Episode, batch_idx: int) -> torch.Tensor:
        """
        Validation step for meta-learning episode.
        
        Args:
            batch: Episode with support and query data
            batch_idx: Batch index
            
        Returns:
            Loss tensor
        """
        # Similar to training_step but with validation metrics
        # Key difference: Use torch.no_grad() for inner loop adaptation in validation
        # to avoid computing gradients through the adaptation process
        
        # Extract episode data
        support_data, support_labels = batch.support_x, batch.support_y
        query_data, query_labels = batch.query_x, batch.query_y
        
        # Validation uses no_grad for inner adaptation to avoid computing gradients
        with torch.no_grad():
            # Route to algorithm-specific validation
            if self.algorithm == "maml":
                adapted_model = self._maml_inner_loop(support_data, support_labels)
                query_logits = adapted_model(query_data)
                loss = torch.nn.functional.cross_entropy(query_logits, query_labels)
            elif self.algorithm == "protonet":
                prototypes = self._compute_prototypes(support_data, support_labels)
                query_logits = self._classify_queries(query_data, prototypes)
                loss = torch.nn.functional.cross_entropy(query_logits, query_labels)
            elif self.algorithm == "meta_sgd":
                adapted_model = self._meta_sgd_adapt(support_data, support_labels)
                query_logits = adapted_model(query_data)
                loss = torch.nn.functional.cross_entropy(query_logits, query_labels)
            else:
                # Fallback to basic adaptation
                from ..eval import evaluate
                loss, accuracy = evaluate(
                    self.model, batch, num_inner_steps=getattr(self, 'inner_steps', 5),
                    inner_lr=self.learning_rate, first_order=getattr(self, 'first_order', False)
                )
                # For consistency, compute logits for accuracy calculation
                query_logits = self.model(query_data)
        
        # Log validation metrics
        if self.val_accuracy is not None:
            accuracy = self.val_accuracy(query_logits, query_labels)
            self.log('val_accuracy', accuracy, on_step=False, on_epoch=True, prog_bar=True)
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        """Configure optimizers for meta-learning."""
        
        # Set up meta-optimizer (outer loop)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.meta_lr)
        
        # Configure learning rate scheduling
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            }
        }
    
    def _maml_inner_loop(self, support_data: torch.Tensor, 
                        support_labels: torch.Tensor) -> nn.Module:
        """MAML inner loop adaptation using existing core functionality."""
        
        # Use existing MAML adaptation functionality
        from ..core.utils import clone_module
        from torch.autograd import grad
        import torch.nn.functional as F
        
        # Clone model for adaptation
        adapted_model = clone_module(self.model)
        
        # Inner loop adaptation
        for step in range(getattr(self, 'inner_steps', 5)):
            # Forward pass
            support_logits = adapted_model(support_data)
            inner_loss = F.cross_entropy(support_logits, support_labels)
            
            # Compute gradients
            grads = grad(inner_loss, adapted_model.parameters(), 
                        retain_graph=(step < self.num_inner_steps - 1),
                        create_graph=not self.first_order)
            
            # Update parameters
            with torch.no_grad():
                for param, grad_val in zip(adapted_model.parameters(), grads):
                    if grad_val is not None:
                        param.data = param.data - self.learning_rate * grad_val
        
        return adapted_model
    
    def _compute_prototypes(self, support_data: torch.Tensor, 
                           support_labels: torch.Tensor) -> torch.Tensor:
        """Compute prototypes for Prototypical Networks."""
        n_classes = support_labels.unique().numel()
        support_features = self.model(support_data)
        
        prototypes = torch.zeros(n_classes, support_features.size(-1), 
                                device=support_data.device)
        
        for class_id in range(n_classes):
            class_mask = (support_labels == class_id)
            if class_mask.sum() > 0:
                prototypes[class_id] = support_features[class_mask].mean(dim=0)
        
        return prototypes
    
    def _classify_queries(self, query_data: torch.Tensor, 
                         prototypes: torch.Tensor) -> torch.Tensor:
        """Classify queries using prototypes."""
        query_features = self.model(query_data)
        
        # Compute distances to prototypes (negative for logits)
        distances = torch.cdist(query_features, prototypes)
        logits = -distances
        
        return logits
    
    def _meta_sgd_adapt(self, support_data: torch.Tensor, 
                       support_labels: torch.Tensor) -> nn.Module:
        """Meta-SGD adaptation with learnable learning rates."""
        # Use existing Meta-SGD functionality
        from ..algorithms.meta_sgd import meta_sgd_update
        from ..core.utils import clone_module
        
        adapted_model = clone_module(self.model)
        
        # Simplified Meta-SGD adaptation
        for step in range(self.num_inner_steps):
            support_logits = adapted_model(support_data)
            inner_loss = torch.nn.functional.cross_entropy(support_logits, support_labels)
            
            # Use Meta-SGD update with learnable learning rates
            adapted_model = meta_sgd_update(adapted_model, inner_loss)
        
        return adapted_model


class MAMLLightningModule(MetaLearningLightningModule):
    """
    PyTorch Lightning wrapper for MAML algorithm.
    
    Wraps our existing MAML implementation in algos.maml with Lightning features
    for distributed training, logging, and checkpointing.
    """
    
    def __init__(self, model: nn.Module, inner_lr: float = 1e-2, 
                 meta_lr: float = 1e-3, inner_steps: int = 5,
                 first_order: bool = False, **kwargs):
        """
        Initialize MAML Lightning module.
        
        Args:
            model: Base model for meta-learning
            inner_lr: Inner loop learning rate
            meta_lr: Meta-learning rate  
            inner_steps: Number of inner loop gradient steps
            first_order: Use first-order approximation
        """
        super().__init__(
            model=model, algorithm="maml", learning_rate=inner_lr,
            meta_lr=meta_lr, inner_steps=inner_steps, first_order=first_order,
            **kwargs
        )
        
        # Initialize MAML-specific parameters
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps  
        self.first_order = first_order
        
        # Set up inner loop optimizer
        # We'll create this dynamically during training for each episode
        self.inner_optimizer_class = torch.optim.SGD
    
    def training_step(self, batch: Episode, batch_idx: int) -> torch.Tensor:
        """MAML training step with inner/outer loop optimization."""
        # Inner loop adaptation on support set
        # Use our existing MAML implementation from core
        support_data, support_labels = batch.support_x, batch.support_y
        query_data, query_labels = batch.query_x, batch.query_y
        
        # Use the parent class _maml_inner_loop method
        adapted_model = self._maml_inner_loop(support_data, support_labels)
        
        # Outer loop evaluation on query set
        query_logits = adapted_model(query_data)
        query_loss = torch.nn.functional.cross_entropy(query_logits, query_labels)
        
        # Meta-gradient computation
        # The Lightning framework handles automatic differentiation for the outer loop
        
        # Log metrics
        if self.train_accuracy is not None:
            accuracy = self.train_accuracy(query_logits, query_labels)
            self.log('train_accuracy', accuracy, on_step=True, on_epoch=True, prog_bar=True)
        
        self.log('train_loss', query_loss, on_step=True, on_epoch=True, prog_bar=True)
        return query_loss


class ProtoNetLightningModule(MetaLearningLightningModule):
    """
    PyTorch Lightning wrapper for Prototypical Networks.
    
    Wraps our existing ProtoHead implementation with Lightning features.
    """
    
    def __init__(self, model: nn.Module, distance_metric: str = 'euclidean',
                 temperature: float = 1.0, **kwargs):
        """
        Initialize Prototypical Networks Lightning module.
        
        Args:
            model: Feature extraction backbone
            distance_metric: Distance metric for prototype matching
            temperature: Temperature for softmax
        """
        super().__init__(
            model=model, algorithm="protonet", 
            distance_metric=distance_metric, temperature=temperature,
            **kwargs
        )
        
        # Initialize ProtoNet components
        try:
            from ..algorithms import ProtoHead
            self.proto_head = ProtoHead(distance=distance_metric, tau=temperature)
        except (ImportError, TypeError):
            # Fallback if ProtoHead not available or parameters don't match
            self.proto_head = None
        self.distance_metric = distance_metric
        self.temperature = temperature
    
    def training_step(self, batch: Episode, batch_idx: int) -> torch.Tensor:
        """Prototypical Networks training step."""
        # Extract features using backbone
        support_data, support_labels = batch.support_x, batch.support_y
        query_data, query_labels = batch.query_x, batch.query_y
        
        support_features = self.model(support_data)
        query_features = self.model(query_data)
        
        # Compute prototypes and classify queries
        if self.proto_head is not None:
            logits = self.proto_head(support_features, support_labels, query_features)
        else:
            # Fallback implementation
            prototypes = self._compute_prototypes(support_data, support_labels)
            logits = self._classify_queries(query_data, prototypes)
        
        loss = torch.nn.functional.cross_entropy(logits, query_labels)
        
        # Log metrics
        if self.train_accuracy is not None:
            accuracy = self.train_accuracy(logits, query_labels)
            self.log('train_accuracy', accuracy, on_step=True, on_epoch=True, prog_bar=True)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss


class LightningTrainerFactory:
    """
    Factory for creating configured Lightning trainers for meta-learning.
    
    Provides pre-configured trainers for different meta-learning scenarios
    with appropriate callbacks, loggers, and distributed settings.
    """
    
    @staticmethod
    def create_meta_learning_trainer(
        max_epochs: int = 100,
        gpus: Optional[Union[int, List[int]]] = None,
        distributed_strategy: Optional[str] = None,
        logger_type: str = "tensorboard",
        experiment_name: str = "meta_learning",
        checkpoint_dir: str = "./checkpoints",
        early_stopping: bool = True,
        **trainer_kwargs
    ):
        """
        Create pre-configured Lightning trainer for meta-learning.
        
        Args:
            max_epochs: Maximum training epochs
            gpus: GPU configuration
            distributed_strategy: Distributed training strategy
            logger_type: Logger type ("tensorboard", "wandb")
            experiment_name: Experiment name for logging
            checkpoint_dir: Directory for checkpoints
            early_stopping: Enable early stopping
            **trainer_kwargs: Additional trainer arguments
            
        Returns:
            Configured Lightning trainer
        """
        if not LIGHTNING_AVAILABLE:
            raise ImportError("PyTorch Lightning not installed. Install with: pip install lightning")
        
        # Configure logger
        if logger_type == "tensorboard":
            logger = TensorBoardLogger("tb_logs", name=experiment_name)
        elif logger_type == "wandb":
            try:
                logger = WandbLogger(project=experiment_name)
            except ImportError:
                print("Warning: wandb not installed, falling back to tensorboard")
                logger = TensorBoardLogger("tb_logs", name=experiment_name)
        else:
            logger = None
        
        # Set up callbacks
        callbacks = []
        
        # Checkpointing
        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename=f"{experiment_name}-{{epoch:02d}}-{{val_loss:.2f}}",
            save_top_k=3,
            monitor="val_loss",
            mode="min"
        )
        callbacks.append(checkpoint_callback)
        
        # Early stopping
        if early_stopping:
            early_stop_callback = EarlyStopping(
                monitor="val_loss",
                min_delta=0.00,
                patience=10,
                verbose=False,
                mode="min"
            )
            callbacks.append(early_stop_callback)
        
        # Configure trainer
        trainer = Trainer(
            max_epochs=max_epochs,
            devices=gpus,
            strategy=distributed_strategy,
            logger=logger,
            callbacks=callbacks,
            **trainer_kwargs
        )
        return trainer


def convert_to_lightning_module(model: nn.Module, algorithm: str, 
                              **algorithm_kwargs) -> MetaLearningLightningModule:
    """
    Convert existing meta-learning model to Lightning module.
    
    This function wraps our existing implementations in Lightning modules
    without modifying the core algorithms.
    
    Args:
        model: Existing meta-learning model
        algorithm: Algorithm type ("maml", "protonet", "meta_sgd")
        **algorithm_kwargs: Algorithm-specific parameters
        
    Returns:
        Lightning module wrapping the existing model
    """
    # Route to appropriate Lightning module
    if algorithm.lower() == "maml":
        return MAMLLightningModule(model, **algorithm_kwargs)
    elif algorithm.lower() == "protonet":
        return ProtoNetLightningModule(model, **algorithm_kwargs)
    elif algorithm.lower() == "meta_sgd":
        # Generic wrapper until Meta-SGD specific implementation is complete
        return MetaLearningLightningModule(model, algorithm, **algorithm_kwargs)
    else:
        # Generic wrapper for other algorithms
        return MetaLearningLightningModule(model, algorithm, **algorithm_kwargs)


class DistributedMetaLearning:
    """
    Utilities for distributed meta-learning training.
    
    Handles episode distribution, gradient synchronization, and other
    distributed training concerns specific to meta-learning.
    """
    
    @staticmethod
    def setup_distributed_episodes(episodes: List[Episode], 
                                 world_size: int, rank: int) -> List[Episode]:
        """
        Distribute episodes across multiple processes/GPUs.
        
        Args:
            episodes: List of training episodes
            world_size: Number of processes
            rank: Current process rank
            
        Returns:
            Episodes assigned to current process
        """
        # Partition episodes across processes
        # Ensure each process gets a balanced subset of episodes
        episodes_per_process = len(episodes) // world_size
        start_idx = rank * episodes_per_process
        end_idx = start_idx + episodes_per_process if rank < world_size - 1 else len(episodes)
        return episodes[start_idx:end_idx]
    
    @staticmethod
    def synchronize_meta_gradients(model: nn.Module, world_size: int) -> None:
        """
        Synchronize meta-gradients across distributed processes.
        
        Args:
            model: Model whose gradients need synchronization
            world_size: Number of processes
        """
        # Average gradients across all processes
        # This is critical for meta-learning where gradient computation
        # happens through inner loop adaptation
        
        try:
            import torch.distributed as dist
            
            for param in model.parameters():
                if param.grad is not None:
                    # Average gradients across all processes
                    dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                    param.grad.data /= world_size
                    
        except ImportError:
            # Handle case where distributed training is not available
            print("Warning: torch.distributed not available, skipping gradient synchronization")
            pass
        
        # Handle meta-learning specific gradient patterns
        # Meta-learning gradients have different characteristics than
        # standard deep learning due to second-order derivatives
        # Additional synchronization could be added here for higher-order gradients


# Backwards compatibility check
if not LIGHTNING_AVAILABLE:
    # Provide stub implementations when Lightning is not available
    class MetaLearningLightningModule:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch Lightning not installed. Install with: pip install lightning")
    
    class MAMLLightningModule(MetaLearningLightningModule):
        pass
    
    class ProtoNetLightningModule(MetaLearningLightningModule):
        pass