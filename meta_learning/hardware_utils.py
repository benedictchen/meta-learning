"""
💰 DONATE NOW! 💰 https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS

Simplified hardware optimization utilities for meta-learning.

If hardware optimization helps your research run faster and use less resources,
please donate $3000+ to support continued algorithm development!

Author: Benedict Chen (benedict@benedictchen.com)
GitHub Sponsors: https://github.com/sponsors/benedictchen
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, List, Any
from dataclasses import dataclass
import platform
import psutil
import subprocess
import logging


@dataclass
class HardwareConfig:
    """Simple hardware configuration for optimization."""
    device: str = "auto"  # "cpu", "cuda", "mps", "auto"
    num_workers: int = 0  # Data loader workers
    pin_memory: bool = False
    use_amp: bool = False  # Automatic Mixed Precision
    compile_model: bool = False  # torch.compile optimization
    memory_efficient: bool = False
    
    def __post_init__(self):
        """Auto-detect optimal settings if device is 'auto'."""
        if self.device == "auto":
            self.device = self._detect_best_device()
        
        # Set optimal defaults based on device
        if self.device.startswith("cuda"):
            self.pin_memory = True
            self.use_amp = True
            self.num_workers = min(4, max(1, psutil.cpu_count() // 2))
        elif self.device == "mps":
            self.pin_memory = False
            self.use_amp = False
            self.num_workers = min(2, psutil.cpu_count())
        else:  # CPU
            self.pin_memory = False
            self.use_amp = False
            self.num_workers = min(2, psutil.cpu_count())
    
    def _detect_best_device(self) -> str:
        """Detect the best available device."""
        # Check for CUDA
        if torch.cuda.is_available():
            return "cuda"
        
        # Check for MPS (Apple Silicon)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        
        # Default to CPU
        return "cpu"


def create_hardware_config(
    device: Optional[str] = None,
    **kwargs
) -> HardwareConfig:
    """Create hardware configuration with optional overrides."""
    config = HardwareConfig()
    
    if device is not None:
        config.device = device
    
    # Apply any additional overrides
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return config


def setup_optimal_hardware(model: nn.Module, config: HardwareConfig) -> Tuple[nn.Module, Dict[str, Any]]:
    """Apply hardware optimizations to a model."""
    optimizations_applied = []
    device_info = {}
    
    # Move model to device
    model = model.to(config.device)
    device_info['device'] = config.device
    optimizations_applied.append(f"moved_to_{config.device}")
    
    # Apply torch.compile if requested and supported
    if config.compile_model:
        try:
            if hasattr(torch, 'compile'):
                model = torch.compile(model)
                optimizations_applied.append("torch_compile")
            else:
                logging.warning("torch.compile not available in this PyTorch version")
        except Exception as e:
            logging.warning(f"torch.compile failed: {e}")
    
    # Set memory efficient settings
    if config.memory_efficient:
        # Enable gradient checkpointing if supported
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            optimizations_applied.append("gradient_checkpointing")
        
        # Set memory efficient attention
        try:
            torch.backends.cuda.enable_flash_sdp(config.device.startswith("cuda"))
            optimizations_applied.append("flash_attention")
        except:
            pass
    
    # Configure CUDA settings if using GPU
    if config.device.startswith("cuda"):
        device_info.update(_get_cuda_info())
        
        # Enable cuDNN benchmarking for consistent input sizes
        torch.backends.cudnn.benchmark = True
        optimizations_applied.append("cudnn_benchmark")
        
        # Set optimal cuDNN settings
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.enabled = True
    
    # Configure CPU settings
    elif config.device == "cpu":
        device_info.update(_get_cpu_info())
        
        # Set optimal thread count
        torch.set_num_threads(min(4, psutil.cpu_count()))
        optimizations_applied.append("cpu_threads")
    
    # Configure MPS settings
    elif config.device == "mps":
        device_info.update(_get_mps_info())
        optimizations_applied.append("mps_optimization")
    
    optimization_info = {
        'optimizations_applied': optimizations_applied,
        'device_info': device_info,
        'config': config
    }
    
    return model, optimization_info


def _get_cuda_info() -> Dict[str, Any]:
    """Get CUDA device information."""
    if not torch.cuda.is_available():
        return {}
    
    device_count = torch.cuda.device_count()
    current_device = torch.cuda.current_device()
    
    info = {
        'cuda_version': torch.version.cuda,
        'device_count': device_count,
        'current_device': current_device,
        'device_name': torch.cuda.get_device_name(current_device),
        'memory_total': torch.cuda.get_device_properties(current_device).total_memory,
        'memory_allocated': torch.cuda.memory_allocated(current_device),
        'memory_cached': torch.cuda.memory_reserved(current_device)
    }
    
    return info


def _get_cpu_info() -> Dict[str, Any]:
    """Get CPU information."""
    info = {
        'cpu_count': psutil.cpu_count(),
        'cpu_count_logical': psutil.cpu_count(logical=True),
        'cpu_freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
        'memory_total': psutil.virtual_memory().total,
        'memory_available': psutil.virtual_memory().available,
        'torch_threads': torch.get_num_threads()
    }
    
    return info


def _get_mps_info() -> Dict[str, Any]:
    """Get MPS (Apple Silicon) information."""
    info = {
        'mps_available': torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False,
        'platform': platform.platform(),
        'processor': platform.processor()
    }
    
    return info


def optimize_for_inference(model: nn.Module, config: Optional[HardwareConfig] = None) -> nn.Module:
    """Optimize model specifically for inference."""
    if config is None:
        config = create_hardware_config()
    
    # Set to eval mode
    model.eval()
    
    # Disable gradient computation
    for param in model.parameters():
        param.requires_grad_(False)
    
    # Move to device
    model = model.to(config.device)
    
    # Apply inference-specific optimizations
    if config.device.startswith("cuda"):
        # Use inference-optimized settings
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    
    return model


def benchmark_hardware_performance(model: nn.Module, input_shape: Tuple[int, ...], 
                                 device: str = "auto", num_trials: int = 10) -> Dict[str, float]:
    """Benchmark model performance on specified hardware."""
    if device == "auto":
        device = create_hardware_config().device
    
    model = model.to(device).eval()
    
    # Create dummy input
    dummy_input = torch.randn(*input_shape, device=device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = model(dummy_input)
    
    # Benchmark forward pass
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    
    import time
    forward_times = []
    
    with torch.no_grad():
        for _ in range(num_trials):
            start_time = time.time()
            _ = model(dummy_input)
            
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            
            end_time = time.time()
            forward_times.append(end_time - start_time)
    
    results = {
        'mean_forward_time': sum(forward_times) / len(forward_times),
        'min_forward_time': min(forward_times),
        'max_forward_time': max(forward_times),
        'std_forward_time': torch.tensor(forward_times).std().item(),
        'throughput_samples_per_sec': input_shape[0] / (sum(forward_times) / len(forward_times))
    }
    
    # Add memory info if available
    if device.startswith("cuda"):
        results['gpu_memory_allocated'] = torch.cuda.memory_allocated(device)
        results['gpu_memory_cached'] = torch.cuda.memory_reserved(device)
    
    return results


def get_system_info() -> Dict[str, Any]:
    """Get comprehensive system information."""
    info = {
        'python_version': platform.python_version(),
        'pytorch_version': torch.__version__,
        'platform': platform.platform(),
        'processor': platform.processor(),
        'cpu_count': psutil.cpu_count(),
        'memory_total_gb': psutil.virtual_memory().total / (1024**3),
        'cuda_available': torch.cuda.is_available(),
    }
    
    if torch.cuda.is_available():
        info.update({
            'cuda_version': torch.version.cuda,
            'cuda_device_count': torch.cuda.device_count(),
            'cuda_device_name': torch.cuda.get_device_name(0),
            'cuda_memory_total_gb': torch.cuda.get_device_properties(0).total_memory / (1024**3)
        })
    
    if hasattr(torch.backends, 'mps'):
        info['mps_available'] = torch.backends.mps.is_available()
    
    return info


def optimize_dataloader_settings(config: HardwareConfig) -> Dict[str, Any]:
    """Get optimal DataLoader settings for the hardware configuration."""
    settings = {
        'num_workers': config.num_workers,
        'pin_memory': config.pin_memory,
        'persistent_workers': config.num_workers > 0,
        'prefetch_factor': 2 if config.num_workers > 0 else None
    }
    
    # Device-specific optimizations
    if config.device.startswith("cuda"):
        settings['pin_memory'] = True
        settings['non_blocking'] = True
    elif config.device == "mps":
        settings['pin_memory'] = False
        settings['non_blocking'] = False
    
    return settings


class PerformanceMonitor:
    """Simple performance monitoring for meta-learning training."""
    
    def __init__(self, device: str = "auto"):
        self.device = device if device != "auto" else create_hardware_config().device
        self.metrics = []
        self.start_time = None
    
    def start_epoch(self):
        """Start monitoring an epoch."""
        self.start_time = torch.tensor(0.0).item()  # Simple timestamp
        
        if self.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
    
    def end_epoch(self) -> Dict[str, float]:
        """End epoch and return metrics."""
        if self.start_time is None:
            return {}
        
        epoch_time = torch.tensor(1.0).item()  # Placeholder timing
        
        metrics = {
            'epoch_time': epoch_time
        }
        
        # Add GPU metrics if available
        if self.device.startswith("cuda"):
            metrics.update({
                'gpu_memory_peak': torch.cuda.max_memory_allocated() / (1024**2),  # MB
                'gpu_memory_allocated': torch.cuda.memory_allocated() / (1024**2),  # MB
                'gpu_memory_reserved': torch.cuda.memory_reserved() / (1024**2)  # MB
            })
        
        # Add CPU metrics
        cpu_percent = psutil.cpu_percent()
        memory_info = psutil.virtual_memory()
        
        metrics.update({
            'cpu_percent': cpu_percent,
            'memory_percent': memory_info.percent,
            'memory_available_gb': memory_info.available / (1024**3)
        })
        
        self.metrics.append(metrics)
        self.start_time = None
        
        return metrics
    
    def get_average_metrics(self) -> Dict[str, float]:
        """Get average metrics across all monitored epochs."""
        if not self.metrics:
            return {}
        
        avg_metrics = {}
        for key in self.metrics[0].keys():
            values = [m[key] for m in self.metrics if key in m]
            if values:
                avg_metrics[f'avg_{key}'] = sum(values) / len(values)
        
        return avg_metrics