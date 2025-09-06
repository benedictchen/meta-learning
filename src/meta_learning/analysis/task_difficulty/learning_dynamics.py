"""
Learning dynamics analysis for task difficulty assessment.

This module analyzes learning dynamics during few-shot adaptation to assess
task difficulty. It measures convergence properties, gradient behavior, and
loss landscape characteristics.

Key measures:
- Convergence rate during adaptation
- Gradient variance across training steps
- Loss landscape smoothness via parameter perturbations

Research foundations:
- Li et al. (2018): Visualizing the Loss Landscape of Neural Nets
- Garipov et al. (2018): Loss Surfaces, Mode Connectivity, and Fast Ensembling
- Fort & Ganguli (2019): Emergent properties of the local geometry of neural loss landscapes
"""

from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import copy


class LearningDynamicsAnalyzer:
    """Analyze learning dynamics to assess task difficulty."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def convergence_rate(
        self, 
        model: nn.Module,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        n_steps: int = 10,
        lr: float = 0.01
    ) -> float:
        """
        Measure convergence rate during few-shot adaptation.
        
        Runs gradient descent on the support set and measures how quickly
        the loss decreases. Slower convergence indicates higher task difficulty.
        
        Args:
            model: Neural network model
            support_x: Support set features [N_support, ...]
            support_y: Support set labels [N_support]
            n_steps: Number of gradient steps
            lr: Learning rate for adaptation
            
        Returns:
            Difficulty score (0=fast convergence, 1=slow/no convergence)
        """
        model.eval()
        
        # Clone model for temporary training
        temp_model = self._clone_model(model)
        temp_model.train()
        
        optimizer = torch.optim.SGD(temp_model.parameters(), lr=lr)
        
        losses = []
        for step in range(n_steps):
            optimizer.zero_grad()
            logits = temp_model(support_x)
            loss = F.cross_entropy(logits, support_y)
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
        
        if len(losses) < 2:
            return 0.5
            
        # Measure convergence rate as relative improvement
        initial_loss = losses[0]
        final_loss = losses[-1]
        
        if initial_loss == 0:
            return 0.0
            
        improvement_rate = (initial_loss - final_loss) / initial_loss
        
        # Convert to difficulty (slower convergence = higher difficulty)
        # Sigmoid transformation to handle negative improvements (divergence)
        difficulty = 1 / (1 + np.exp(5 * (improvement_rate - 0.2)))
        
        return float(difficulty)
    
    def gradient_variance(
        self,
        model: nn.Module, 
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        n_samples: int = 10
    ) -> float:
        """
        Measure gradient variance across multiple forward passes.
        
        Computes gradients multiple times on the same data to measure
        variance in gradient magnitudes. Higher variance indicates
        more unstable/difficult optimization.
        
        Args:
            model: Neural network model
            support_x: Support set features [N_support, ...]
            support_y: Support set labels [N_support]  
            n_samples: Number of gradient computations
            
        Returns:
            Difficulty score (0=stable gradients, 1=high variance)
        """
        model.eval()
        
        gradients = []
        for _ in range(n_samples):
            model.zero_grad()
            logits = model(support_x)
            loss = F.cross_entropy(logits, support_y)
            loss.backward()
            
            # Collect gradient norm
            grad_norm = 0.0
            for param in model.parameters():
                if param.grad is not None:
                    grad_norm += torch.norm(param.grad).item() ** 2
            
            gradients.append(grad_norm ** 0.5)
        
        if len(gradients) < 2:
            return 0.0
            
        # Compute coefficient of variation
        mean_grad = np.mean(gradients)
        std_grad = np.std(gradients)
        
        if mean_grad == 0:
            return 0.0
            
        cv = std_grad / mean_grad
        
        # Normalize to [0, 1] range
        difficulty = min(1.0, cv / 2.0)  # Assume CV > 2 is maximum difficulty
        
        return difficulty
    
    def loss_landscape_smoothness(
        self,
        model: nn.Module,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        n_perturbations: int = 20,
        perturbation_scale: float = 0.01
    ) -> float:
        """
        Measure loss landscape smoothness around current parameters.
        
        Adds small random perturbations to model parameters and measures
        how much the loss changes. Rougher landscapes (high sensitivity)
        indicate higher optimization difficulty.
        
        Args:
            model: Neural network model
            support_x: Support set features [N_support, ...]
            support_y: Support set labels [N_support]
            n_perturbations: Number of random perturbations
            perturbation_scale: Standard deviation of perturbations
            
        Returns:
            Difficulty score (0=smooth landscape, 1=rough landscape)
        """
        model.eval()
        
        # Get baseline loss
        baseline_logits = model(support_x)
        baseline_loss = F.cross_entropy(baseline_logits, support_y).item()
        
        loss_variations = []
        
        with torch.no_grad():
            original_params = [p.clone() for p in model.parameters()]
            
            for _ in range(n_perturbations):
                # Add random perturbation
                for param, orig_param in zip(model.parameters(), original_params):
                    noise = torch.randn_like(param) * perturbation_scale
                    param.data = orig_param + noise
                
                # Compute perturbed loss
                perturbed_logits = model(support_x)
                perturbed_loss = F.cross_entropy(perturbed_logits, support_y).item()
                
                loss_variations.append(abs(perturbed_loss - baseline_loss))
            
            # Restore original parameters
            for param, orig_param in zip(model.parameters(), original_params):
                param.data = orig_param.clone()
        
        if not loss_variations:
            return 0.5
            
        # Higher variance = rougher landscape = higher difficulty
        variance = np.var(loss_variations)
        difficulty = min(1.0, variance / 0.1)  # Normalize assuming max variance of 0.1
        
        return difficulty
    
    def adaptation_stability(
        self,
        model: nn.Module,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        n_runs: int = 5,
        n_steps: int = 10,
        lr: float = 0.01
    ) -> float:
        """
        Measure stability of adaptation across multiple runs.
        
        Runs adaptation multiple times with different random initializations
        and measures how consistent the final performance is.
        
        Args:
            model: Neural network model
            support_x: Support set features [N_support, ...]
            support_y: Support set labels [N_support]
            n_runs: Number of adaptation runs
            n_steps: Steps per adaptation run
            lr: Learning rate
            
        Returns:
            Difficulty score (0=stable adaptation, 1=unstable adaptation)
        """
        final_losses = []
        
        for run in range(n_runs):
            # Clone model and add small random noise to break symmetry
            temp_model = self._clone_model(model)
            
            # Add small random perturbation to initial parameters
            with torch.no_grad():
                for param in temp_model.parameters():
                    param.data += torch.randn_like(param) * 1e-4
            
            temp_model.train()
            optimizer = torch.optim.SGD(temp_model.parameters(), lr=lr)
            
            # Run adaptation
            for step in range(n_steps):
                optimizer.zero_grad()
                logits = temp_model(support_x)
                loss = F.cross_entropy(logits, support_y)
                loss.backward()
                optimizer.step()
            
            # Record final loss
            temp_model.eval()
            with torch.no_grad():
                final_logits = temp_model(support_x)
                final_loss = F.cross_entropy(final_logits, support_y).item()
                final_losses.append(final_loss)
        
        if len(final_losses) < 2:
            return 0.5
            
        # Higher variance in final losses = less stable adaptation
        mean_loss = np.mean(final_losses)
        std_loss = np.std(final_losses)
        
        if mean_loss == 0:
            return 0.0
            
        # Coefficient of variation as stability measure
        cv = std_loss / mean_loss
        instability = min(1.0, cv)  # Cap at 1.0
        
        return instability
    
    def gradient_norm_trajectory(
        self,
        model: nn.Module,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        n_steps: int = 10,
        lr: float = 0.01
    ) -> Dict[str, float]:
        """
        Analyze gradient norm trajectory during adaptation.
        
        Tracks how gradient norms change during adaptation, which can
        indicate optimization difficulty and convergence properties.
        
        Args:
            model: Neural network model
            support_x: Support set features [N_support, ...]
            support_y: Support set labels [N_support]
            n_steps: Number of adaptation steps
            lr: Learning rate
            
        Returns:
            Dictionary with trajectory analysis metrics
        """
        temp_model = self._clone_model(model)
        temp_model.train()
        optimizer = torch.optim.SGD(temp_model.parameters(), lr=lr)
        
        grad_norms = []
        losses = []
        
        for step in range(n_steps):
            optimizer.zero_grad()
            logits = temp_model(support_x)
            loss = F.cross_entropy(logits, support_y)
            loss.backward()
            
            # Compute gradient norm
            total_norm = 0.0
            for param in temp_model.parameters():
                if param.grad is not None:
                    total_norm += param.grad.data.norm(2).item() ** 2
            grad_norm = total_norm ** 0.5
            
            grad_norms.append(grad_norm)
            losses.append(loss.item())
            
            optimizer.step()
        
        if len(grad_norms) < 2:
            return {'gradient_instability': 0.5}
        
        # Analyze gradient norm trajectory
        grad_norms = np.array(grad_norms)
        
        # Gradient explosion detection
        max_grad = np.max(grad_norms)
        mean_grad = np.mean(grad_norms)
        explosion_ratio = max_grad / (mean_grad + 1e-8)
        
        # Gradient vanishing detection  
        min_grad = np.min(grad_norms)
        vanishing_ratio = min_grad / (mean_grad + 1e-8)
        
        # Overall gradient stability
        grad_cv = np.std(grad_norms) / (np.mean(grad_norms) + 1e-8)
        
        return {
            'gradient_instability': min(1.0, grad_cv),
            'explosion_risk': min(1.0, explosion_ratio / 10.0),
            'vanishing_risk': max(0.0, 1.0 - vanishing_ratio * 10.0),
            'mean_gradient_norm': float(mean_grad),
            'final_gradient_norm': float(grad_norms[-1])
        }
    
    def compute_all_dynamics_measures(
        self,
        model: nn.Module,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        n_steps: int = 10,
        lr: float = 0.01
    ) -> Dict[str, float]:
        """
        Compute all learning dynamics measures.
        
        Args:
            model: Neural network model
            support_x: Support set features [N_support, ...]
            support_y: Support set labels [N_support]
            n_steps: Number of adaptation steps
            lr: Learning rate
            
        Returns:
            Dictionary of dynamics measures
        """
        measures = {}
        
        try:
            measures['convergence_rate'] = self.convergence_rate(
                model, support_x, support_y, n_steps, lr
            )
        except Exception as e:
            self.logger.warning(f"Error computing convergence rate: {e}")
            measures['convergence_rate'] = 0.5
            
        try:
            measures['gradient_variance'] = self.gradient_variance(
                model, support_x, support_y
            )
        except Exception as e:
            self.logger.warning(f"Error computing gradient variance: {e}")
            measures['gradient_variance'] = 0.5
            
        try:
            measures['loss_landscape_smoothness'] = self.loss_landscape_smoothness(
                model, support_x, support_y
            )
        except Exception as e:
            self.logger.warning(f"Error computing landscape smoothness: {e}")
            measures['loss_landscape_smoothness'] = 0.5
            
        try:
            measures['adaptation_stability'] = self.adaptation_stability(
                model, support_x, support_y, n_steps=n_steps, lr=lr
            )
        except Exception as e:
            self.logger.warning(f"Error computing adaptation stability: {e}")
            measures['adaptation_stability'] = 0.5
            
        try:
            trajectory_measures = self.gradient_norm_trajectory(
                model, support_x, support_y, n_steps, lr
            )
            measures.update(trajectory_measures)
        except Exception as e:
            self.logger.warning(f"Error computing gradient trajectory: {e}")
            measures['gradient_instability'] = 0.5
            
        return measures
    
    def _clone_model(self, model: nn.Module) -> nn.Module:
        """Create a deep copy of the model."""
        return copy.deepcopy(model)


def analyze_learning_dynamics(
    model: nn.Module,
    support_x: torch.Tensor,
    support_y: torch.Tensor,
    **kwargs
) -> Dict[str, float]:
    """
    Convenience function to analyze learning dynamics.
    
    Args:
        model: Neural network model
        support_x: Support set features [N_support, ...]
        support_y: Support set labels [N_support]
        **kwargs: Additional arguments for analysis
        
    Returns:
        Dictionary of dynamics measures
    """
    analyzer = LearningDynamicsAnalyzer()
    return analyzer.compute_all_dynamics_measures(model, support_x, support_y, **kwargs)


if __name__ == "__main__":
    # Simple demonstration
    import torch.nn as nn
    
    torch.manual_seed(42)
    
    # Create a simple model
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 3)
    )
    
    # Create synthetic support data
    support_x = torch.randn(15, 10)  # 3-way, 5-shot
    support_y = torch.repeat_interleave(torch.arange(3), 5)
    
    analyzer = LearningDynamicsAnalyzer()
    
    print("Learning Dynamics Analysis:")
    measures = analyzer.compute_all_dynamics_measures(model, support_x, support_y)
    
    for measure, value in measures.items():
        print(f"  {measure}: {value:.3f}")
    
    print(f"\nOverall learning difficulty: {np.mean(list(measures.values())):.3f}")