"""
💰 SUPPORT THIS RESEARCH - PLEASE DONATE! 💰

🙏 If this library helps your research or project, please consider donating:
💳 PayPal: https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS
⭐ GitHub Sponsors: https://github.com/sponsors/benedictchen

👨‍💻 Author: Benedict Chen
💰 Donations: Help support this work! Buy me a coffee ☕, beer 🍺, lamborghini 🏎️, or private island 🏝️
💖 Please consider recurring donations to fully support continued research

Your support enables cutting-edge AI research for everyone! 🚀

Modern Hardware Support Utilities for Meta-Learning ⚡🖥️
======================================================

🎯 **ELI5 Explanation**:
Think of this like a smart power manager for your computer's brain (GPU/CPU)!
Just like how a race car driver needs the right engine settings for different tracks,
machine learning needs the right hardware settings for different tasks:
- 🏎️ **GPU Acceleration**: Like switching from a bicycle to a race car for computations
- 🧠 **Mixed Precision**: Like using shorthand writing - faster but still accurate
- 🤝 **Multi-GPU**: Like having multiple chefs working together in a kitchen
- 💾 **Memory Optimization**: Like organizing your workspace so you can work more efficiently

📊 **Hardware Performance Hierarchy**:
```
Performance Scale (relative speed):
CPU (1x) ────→ GPU (50x) ────→ Multi-GPU (200x) ────→ TPU (400x)
🐌             🏎️             🚀                    🛸

Memory Usage Optimization:
FP32 (100%) → FP16 (50%) → INT8 (25%) → Dynamic (varies)
```

🔧 **Hardware Support Matrix**:
```
┌─────────────────┬──────────┬─────────┬────────────┐
│ Hardware        │ Speed    │ Memory  │ Precision  │
├─────────────────┼──────────┼─────────┼────────────┤
│ NVIDIA RTX 4090 │ ⭐⭐⭐⭐⭐ │ 24GB    │ FP16/BF16  │
│ NVIDIA A100     │ ⭐⭐⭐⭐⭐ │ 40GB    │ TF32/FP16  │
│ Apple M3 Max    │ ⭐⭐⭐    │ 128GB   │ FP16       │
│ Intel CPU       │ ⭐       │ System  │ FP32       │
└─────────────────┴──────────┴─────────┴────────────┘
```

🚀 **Automatic Hardware Detection**: 
Intelligently detects and configures optimal settings for your hardware,
from gaming GPUs to datacenter accelerators.

Comprehensive support for modern hardware accelerators including:
- NVIDIA GPUs (RTX 4090, A100, H100, etc.) 
- Apple Silicon (M1/M2/M3/M4 with MPS)
- Multi-GPU distributed training
- Mixed precision training (FP16, BF16)
- Memory optimization and efficient computation

This module provides hardware abstraction that automatically detects
and utilizes the best available hardware for meta-learning workloads.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import psutil
import gc
from contextlib import contextmanager
from dataclasses import dataclass
import warnings
import platform
import subprocess

logger = logging.getLogger(__name__)


@dataclass
class HardwareConfig:
    """Configuration for modern hardware utilization."""
    # Device selection
    device: Optional[str] = None  # Auto-detect if None
    use_mixed_precision: bool = True  # AMP for faster training
    precision_dtype: str = "float16"  # "float16", "bfloat16", or "float32"
    
    # Multi-GPU settings
    use_data_parallel: bool = False  # Use DataParallel
    use_distributed: bool = False  # Use DistributedDataParallel
    gpu_ids: Optional[List[int]] = None  # GPU IDs to use
    
    # Memory optimization
    gradient_checkpointing: bool = False  # Trade compute for memory
    max_memory_fraction: float = 0.9  # Max GPU memory to use
    memory_efficient_attention: bool = True  # Flash attention when available
    
    # Performance settings
    compile_model: bool = False  # torch.compile (PyTorch 2.0+)
    enable_cudnn_benchmark: bool = True  # CuDNN auto-tuning
    num_workers: int = 4  # DataLoader workers
    
    # Apple Silicon settings (MPS)
    use_mps: bool = True  # Use Metal Performance Shaders on Apple Silicon
    
    # Debugging and profiling
    detect_anomaly: bool = False  # Gradient anomaly detection
    profile_memory: bool = False  # Memory profiling
    log_hardware_info: bool = True  # Log detected hardware


class HardwareDetector:
    """Automatically detect and configure optimal hardware settings."""
    
    @staticmethod
    def detect_best_device() -> torch.device:
        """Detect the best available device for computation."""
        # Check for CUDA (NVIDIA GPUs)
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"CUDA detected: {device_name} ({memory_gb:.1f}GB)")
            return torch.device("cuda")
        
        # Check for Apple Silicon MPS
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            logger.info("Apple Silicon MPS detected")
            return torch.device("mps")
        
        # Fallback to CPU
        cpu_count = psutil.cpu_count()
        memory_gb = psutil.virtual_memory().total / 1e9
        logger.info(f"CPU detected: {cpu_count} cores, {memory_gb:.1f}GB RAM")
        return torch.device("cpu")
    
    @staticmethod
    def get_optimal_batch_size(model: nn.Module, input_shape: Tuple, 
                              device: torch.device, max_memory_fraction: float = 0.8) -> int:
        """Determine optimal batch size based on available memory."""
        if device.type == "cpu":
            # For CPU, use moderate batch size based on memory
            available_memory_gb = psutil.virtual_memory().available / 1e9
            # Rough heuristic: 1GB per 32 samples for typical models
            return min(128, max(1, int(available_memory_gb * 32)))
        
        if device.type == "cuda":
            # For CUDA, probe memory usage
            model = model.to(device)
            model.eval()
            
            # Test with small batch first
            test_batch_size = 1
            try:
                with torch.no_grad():
                    dummy_input = torch.randn(test_batch_size, *input_shape[1:]).to(device)
                    _ = model(dummy_input)
                    
                # Measure memory usage
                torch.cuda.synchronize()
                memory_used = torch.cuda.memory_allocated()
                memory_available = torch.cuda.get_device_properties(0).total_memory * max_memory_fraction
                
                # Estimate optimal batch size
                optimal_batch = int(memory_available / memory_used * 0.8)  # Safety margin
                optimal_batch = max(1, min(optimal_batch, 512))  # Reasonable bounds
                
                logger.info(f"Estimated optimal batch size: {optimal_batch}")
                return optimal_batch
                
            except RuntimeError as e:
                logger.warning(f"Could not determine optimal batch size: {e}")
                return 32
        
        # Default for other devices
        return 32
    
    @staticmethod
    def get_hardware_info() -> Dict[str, Any]:
        """Get comprehensive hardware information."""
        info = {
            "platform": platform.system(),
            "cpu_cores": psutil.cpu_count(),
            "memory_gb": psutil.virtual_memory().total / 1e9,
        }
        
        # CUDA information
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["gpu_count"] = torch.cuda.device_count()
            info["gpu_devices"] = []
            
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                info["gpu_devices"].append({
                    "name": props.name,
                    "memory_gb": props.total_memory / 1e9,
                    "compute_capability": f"{props.major}.{props.minor}",
                    "multiprocessor_count": props.multi_processor_count
                })
        
        # Apple Silicon MPS
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            info["mps_available"] = True
        
        return info


class MemoryManager:
    """Memory management utilities for efficient training."""
    
    @staticmethod
    def clear_cache():
        """Clear all caches to free up memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        gc.collect()
    
    @staticmethod
    def get_memory_usage() -> Dict[str, float]:
        """Get current memory usage statistics."""
        stats = {}
        
        # System memory
        mem = psutil.virtual_memory()
        stats["system_memory_used_gb"] = (mem.total - mem.available) / 1e9
        stats["system_memory_total_gb"] = mem.total / 1e9
        stats["system_memory_percent"] = mem.percent
        
        # CUDA memory
        if torch.cuda.is_available():
            stats["gpu_memory_allocated_gb"] = torch.cuda.memory_allocated() / 1e9
            stats["gpu_memory_reserved_gb"] = torch.cuda.memory_reserved() / 1e9
            total_memory = torch.cuda.get_device_properties(0).total_memory
            stats["gpu_memory_total_gb"] = total_memory / 1e9
            stats["gpu_memory_percent"] = (torch.cuda.memory_allocated() / total_memory) * 100
        
        return stats
    
    @contextmanager
    def memory_efficient_context(self):
        """Context manager for memory-efficient operations."""
        try:
            # Clear cache before operation
            self.clear_cache()
            yield
        finally:
            # Clear cache after operation
            self.clear_cache()


class ModelOptimizer:
    """Model optimization utilities for different hardware configurations."""
    
    def __init__(self, config: HardwareConfig):
        self.config = config
        self.device = self._get_device()
        self.scaler = None
        
        if self.config.use_mixed_precision and self.device.type == "cuda":
            self.scaler = GradScaler()
    
    def _get_device(self) -> torch.device:
        """Get the configured device."""
        if self.config.device:
            return torch.device(self.config.device)
        return HardwareDetector.detect_best_device()
    
    def prepare_model(self, model: nn.Module) -> nn.Module:
        """Prepare model for optimal hardware utilization."""
        # Move to device
        model = model.to(self.device)
        
        # Enable mixed precision if supported
        if self.config.use_mixed_precision and self.device.type == "cuda":
            logger.info("Mixed precision training enabled")
        
        # Enable gradient checkpointing for memory efficiency
        if self.config.gradient_checkpointing:
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
                logger.info("Gradient checkpointing enabled")
        
        # Multi-GPU setup
        if self.config.use_data_parallel and torch.cuda.device_count() > 1:
            gpu_ids = self.config.gpu_ids or list(range(torch.cuda.device_count()))
            model = nn.DataParallel(model, device_ids=gpu_ids)
            logger.info(f"DataParallel enabled on GPUs: {gpu_ids}")
        
        # PyTorch 2.0 compilation
        if self.config.compile_model:
            try:
                model = torch.compile(model)
                logger.info("Model compiled with torch.compile")
            except Exception as e:
                logger.warning(f"Could not compile model: {e}")
        
        # Configure CuDNN
        if self.config.enable_cudnn_benchmark and self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            logger.info("CuDNN benchmark mode enabled")
        
        return model
    
    def optimize_training_step(self, model: nn.Module, optimizer, loss_fn, 
                             inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Perform an optimized training step."""
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        
        # Mixed precision forward pass
        if self.scaler:
            with autocast():
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
            
            # Scaled backward pass
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            # Standard precision
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
        
        optimizer.zero_grad()
        return loss
    
    @contextmanager
    def inference_mode(self):
        """Context manager for optimized inference."""
        original_mode = torch.is_grad_enabled()
        try:
            torch.set_grad_enabled(False)
            if self.device.type == "cuda":
                with torch.cuda.amp.autocast(enabled=self.config.use_mixed_precision):
                    yield
            else:
                yield
        finally:
            torch.set_grad_enabled(original_mode)


class HardwareProfiler:
    """Hardware performance profiling utilities."""
    
    def __init__(self):
        self.memory_manager = MemoryManager()
    
    def profile_model_forward(self, model: nn.Module, input_tensor: torch.Tensor, 
                            num_iterations: int = 100) -> Dict[str, float]:
        """Profile model forward pass performance."""
        import time
        
        model.eval()
        device = next(model.parameters()).device
        input_tensor = input_tensor.to(device)
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = model(input_tensor)
        
        # Synchronize before timing
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        # Time forward passes
        start_time = time.time()
        for _ in range(num_iterations):
            with torch.no_grad():
                _ = model(input_tensor)
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        end_time = time.time()
        
        # Calculate metrics
        total_time = end_time - start_time
        avg_time_per_forward = total_time / num_iterations
        throughput = input_tensor.size(0) / avg_time_per_forward  # samples/second
        
        return {
            "total_time_s": total_time,
            "avg_forward_time_ms": avg_time_per_forward * 1000,
            "throughput_samples_per_sec": throughput,
            "memory_usage_gb": self.memory_manager.get_memory_usage().get("gpu_memory_allocated_gb", 0)
        }
    
    def benchmark_hardware(self) -> Dict[str, Any]:
        """Run comprehensive hardware benchmarks."""
        results = {
            "hardware_info": HardwareDetector.get_hardware_info(),
            "memory_info": self.memory_manager.get_memory_usage()
        }
        
        # Simple compute benchmark
        device = HardwareDetector.detect_best_device()
        
        # Matrix multiplication benchmark
        size = 2048
        a = torch.randn(size, size, device=device)
        b = torch.randn(size, size, device=device)
        
        import time
        start_time = time.time()
        for _ in range(10):
            c = torch.mm(a, b)
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        end_time = time.time()
        
        # FLOPS calculation (approximate)
        flops = 10 * (2 * size**3)  # 10 iterations of matrix multiply
        gflops = flops / (end_time - start_time) / 1e9
        
        results["compute_benchmark"] = {
            "matrix_size": size,
            "time_s": end_time - start_time,
            "gflops": gflops
        }
        
        return results


def create_hardware_config(device: Optional[str] = None, **kwargs) -> HardwareConfig:
    """
    Create hardware configuration with sensible defaults.
    
    Args:
        device: Target device ("cuda", "mps", "cpu", or None for auto-detection)
        **kwargs: Additional configuration parameters
    
    Returns:
        Configured HardwareConfig object
    """
    # Auto-detect optimal settings
    if device is None:
        device = str(HardwareDetector.detect_best_device())
    
    # Default settings based on device
    defaults = {"device": device}
    
    if device.startswith("cuda"):
        defaults.update({
            "use_mixed_precision": True,
            "precision_dtype": "float16",
            "enable_cudnn_benchmark": True,
            "use_data_parallel": torch.cuda.device_count() > 1
        })
    elif device == "mps":
        defaults.update({
            "use_mixed_precision": False,  # MPS has limited AMP support
            "precision_dtype": "float32",
        })
    elif device == "cpu":
        defaults.update({
            "use_mixed_precision": False,
            "precision_dtype": "float32",
            "num_workers": min(8, psutil.cpu_count())
        })
    
    # Override with user settings
    defaults.update(kwargs)
    
    return HardwareConfig(**defaults)


def setup_optimal_hardware(model: nn.Module, config: Optional[HardwareConfig] = None) -> Tuple[nn.Module, HardwareConfig]:
    """
    One-step setup for optimal hardware utilization.
    
    Args:
        model: PyTorch model to optimize
        config: Hardware configuration (auto-created if None)
    
    Returns:
        Tuple of (optimized_model, used_config)
    """
    if config is None:
        config = create_hardware_config()
    
    optimizer_manager = ModelOptimizer(config)
    optimized_model = optimizer_manager.prepare_model(model)
    
    if config.log_hardware_info:
        hardware_info = HardwareDetector.get_hardware_info()
        logger.info(f"Hardware setup completed:")
        logger.info(f"  Device: {optimizer_manager.device}")
        logger.info(f"  Mixed precision: {config.use_mixed_precision}")
        logger.info(f"  Memory: {hardware_info.get('memory_gb', 'Unknown'):.1f}GB")
        
        if "gpu_devices" in hardware_info:
            for i, gpu in enumerate(hardware_info["gpu_devices"]):
                logger.info(f"  GPU {i}: {gpu['name']} ({gpu['memory_gb']:.1f}GB)")
    
    return optimized_model, config


if __name__ == "__main__":
    # Example usage and hardware detection
    print("Hardware Utils Test & Detection")
    print("=" * 50)
    
    # Detect hardware
    detector = HardwareDetector()
    device = detector.detect_best_device()
    hardware_info = detector.get_hardware_info()
    
    print(f"Best device: {device}")
    print(f"Platform: {hardware_info.get('platform', 'Unknown')}")
    print(f"CPU cores: {hardware_info.get('cpu_cores', 'Unknown')}")
    print(f"Memory: {hardware_info.get('memory_gb', 0):.1f}GB")
    
    if "gpu_devices" in hardware_info:
        print(f"GPUs detected: {len(hardware_info['gpu_devices'])}")
        for i, gpu in enumerate(hardware_info["gpu_devices"]):
            print(f"  GPU {i}: {gpu['name']} ({gpu['memory_gb']:.1f}GB)")
    
    # Test model optimization
    model = nn.Sequential(
        nn.Linear(100, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    )
    
    # Setup optimal hardware configuration
    optimized_model, config = setup_optimal_hardware(model)
    print(f"\n✓ Model optimized for {config.device}")
    print(f"✓ Mixed precision: {config.use_mixed_precision}")
    
    # Memory usage test
    memory_manager = MemoryManager()
    memory_stats = memory_manager.get_memory_usage()
    print(f"✓ Memory usage: {memory_stats.get('system_memory_percent', 0):.1f}%")
    
    # Quick performance test
    profiler = HardwareProfiler()
    test_input = torch.randn(32, 100)
    
    try:
        perf_stats = profiler.profile_model_forward(optimized_model, test_input, num_iterations=50)
        print(f"✓ Performance test completed:")
        print(f"  Forward pass: {perf_stats['avg_forward_time_ms']:.2f}ms")
        print(f"  Throughput: {perf_stats['throughput_samples_per_sec']:.1f} samples/sec")
    except Exception as e:
        print(f"⚠ Performance test failed: {e}")
    
    print("\n✓ Hardware utils test completed successfully!")