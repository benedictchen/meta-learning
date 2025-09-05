"""
💰 DONATE NOW! 💰 https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS


If hardware optimization helps your research run faster and use less resources,
Please donate $3000+ to support continued algorithm development!

Author: Benedict Chen (benedict@benedictchen.com)


PLEASE CONSIDER DONATING IF THIS LIBRARY HELPED YOU IN ANY WAY!

PayPal: https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS
GitHub Sponsors: https://github.com/sponsors/benedictchen
"""

from __future__ import annotations
import contextlib
import os
import re
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from typing import Dict, Optional, Tuple, List, Any, Union
from dataclasses import dataclass
import platform
import psutil
import logging


@dataclass
class HardwareConfig:
    """💰 DONATE $5000+ for hardware optimization breakthroughs! 💰
    
    Simple Usage:
        config = HardwareConfig()  # Auto-detects best settings
        
    Advanced Usage:
        config = HardwareConfig(
            device="cuda:0",
            deterministic=True,           # Reproducible results  
            mixed_precision=True,         # Real AMP with context managers
            compile_mode="reduce-overhead", # Optimized torch.compile
            per_process_memory_fraction=0.8,  # Proper device-aware memory
            enable_profiling=True,
            batch_size_estimator=custom_estimator  # Your own batch size logic
        )
    """
    # === BASIC FEATURES (Simple API) ===
    device: str = "auto"              # "cpu", "cuda", "mps", "cuda:0", "auto"  
    deterministic: bool = False       # Enable reproducible behavior (disables benchmark)
    mixed_precision: bool = False     # Enable AMP with proper autocast/scaler
    compile_model: bool = False       # Enable torch.compile (conservative default)
    
    # === ADVANCED FEATURES (Our enhanced capabilities) ===
    # Torch Compile Control
    compile_mode: str = "reduce-overhead"  # "default", "reduce-overhead", "max-autotune"
    compile_backend: Optional[str] = None  # Custom backend for torch.compile
    
    # Memory Management (Device-Aware)
    per_process_memory_fraction: Optional[float] = None  # Fraction of GPU memory to use
    memory_monitoring: bool = False    # Track memory usage with proper device indexing
    memory_cleanup: bool = False       # Automatic memory cleanup
    
    # Performance & Batch Size
    batch_size_optimization: bool = False  # Auto-suggest batch sizes
    batch_size_estimator: Optional[callable] = None  # Custom batch size estimation function
    bytes_per_sample: Optional[int] = None  # For better batch size estimation
    
    # Profiling & Analysis  
    enable_profiling: bool = False     # Enable performance profiling
    profile_memory: bool = False       # Profile memory usage
    log_hardware_info: bool = False    # Log hardware summary
    
    # DataLoader Optimization
    dataloader_optimization: bool = False  # Optimize data loading
    num_workers: int = 0               # Data loader workers (auto-set)
    pin_memory: bool = False           # Pin memory for CUDA (auto-set)
    
    # Legacy Compatibility
    use_amp: bool = False              # Legacy name for mixed_precision (auto-synced)
    
    def __post_init__(self):
        """Auto-detect and validate settings."""
        # Resolve device
        if self.device == "auto":
            self.device = self._detect_best_device()
        
        # Sync legacy compatibility
        if self.use_amp and not self.mixed_precision:
            self.mixed_precision = self.use_amp
        elif self.mixed_precision:
            self.use_amp = self.mixed_precision
        
        # Set optimal defaults based on resolved device
        device_type = self._get_device_type(self.device)
        
        if device_type == "cuda":
            self.pin_memory = True
            self.num_workers = min(4, max(1, psutil.cpu_count() // 2))
            
            # Conservative mixed precision for CUDA (most compatible)
            if self.mixed_precision is None:
                self.mixed_precision = True
                
        elif device_type == "mps":
            self.pin_memory = False
            self.num_workers = min(2, psutil.cpu_count())
            
            # MPS has limited AMP support - capability check instead of blanket ban
            if self.mixed_precision:
                try:
                    # Test if MPS autocast is available
                    with autocast(device_type="mps"):
                        pass
                    # If we get here, MPS autocast works
                except Exception:
                    # Fall back to no AMP for MPS
                    self.mixed_precision = False
                    self.use_amp = False
                    
        else:  # CPU
            self.pin_memory = False
            self.mixed_precision = False
            self.use_amp = False
            self.num_workers = min(2, psutil.cpu_count())
    
    def _detect_best_device(self) -> str:
        """Detect the best available device."""
        if torch.cuda.is_available():
            return "cuda:0"  # Return specific device index
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    
    def _get_device_type(self, device_str: str) -> str:
        """Extract device type from device string."""
        if device_str.startswith("cuda"):
            return "cuda"
        elif device_str == "mps":
            return "mps"
        else:
            return "cpu"
    
    def get_device_index(self) -> Optional[int]:
        """Extract CUDA device index from device string."""
        if not self.device.startswith("cuda"):
            return None
        match = re.search(r"cuda:(\d+)", self.device)
        return int(match.group(1)) if match else 0


# === CRITICAL UTILITY FUNCTIONS ===

@contextlib.contextmanager
def device_guard(device_str: str):
    """Set current CUDA device within context and restore afterward.
    
    """
    if device_str.startswith("cuda"):
        match = re.search(r"cuda:(\d+)", device_str)
        idx = int(match.group(1)) if match else 0
        
        if torch.cuda.is_available():
            prev_device = torch.cuda.current_device()
            torch.cuda.set_device(idx)
            try:
                yield torch.device(device_str)
            finally:
                torch.cuda.set_device(prev_device)
        else:
            yield torch.device("cpu")
    else:
        yield torch.device(device_str)


def apply_determinism(enable: bool) -> None:
    """Enable deterministic behavior when requested.
    
    """
    if not enable:
        return
    
    # Set environment variables for deterministic CUDA operations
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


@contextlib.contextmanager
def maybe_autocast(config: HardwareConfig, device_type: Optional[str] = None):
    """Proper autocast context manager.
    
    Usage:
        with maybe_autocast(config, "cuda"):
            output = model(input)
    """
    if not config.mixed_precision:
        yield
        return
    
    # Auto-detect device type if not provided
    if device_type is None:
        if config.device.startswith("cuda"):
            device_type = "cuda"
        elif config.device == "mps":
            device_type = "mps"
        else:
            device_type = "cpu"
    
    # Only use autocast for supported device types
    if device_type in ["cuda", "mps"]:
        try:
            with autocast(device_type=device_type):
                yield
        except Exception:
            # Fallback if autocast fails
            yield
    else:
        yield


def create_grad_scaler(config: HardwareConfig) -> GradScaler:
    """Create properly configured GradScaler.
    
    """
    # Enable scaler only for CUDA mixed precision
    enabled = (
        config.mixed_precision and 
        config.device.startswith("cuda") and 
        torch.cuda.is_available()
    )
    return GradScaler(enabled=enabled)


def maybe_compile_model(model: nn.Module, config: HardwareConfig) -> Tuple[nn.Module, bool, Optional[str]]:
    """Safely compile model with proper error handling.
    
    
    Returns:
        (compiled_model, was_compiled, compile_mode_used)
    """
    if not config.compile_model or not hasattr(torch, "compile"):
        return model, False, None
    
    try:
        compile_kwargs = {"mode": config.compile_mode}
        if config.compile_backend:
            compile_kwargs["backend"] = config.compile_backend
            
        compiled_model = torch.compile(model, **compile_kwargs)
        
        if config.log_hardware_info:
            logging.info(f"torch.compile enabled with mode='{config.compile_mode}', backend='{config.compile_backend or 'default'}'")
            
        return compiled_model, True, config.compile_mode
        
    except Exception as e:
        if config.log_hardware_info:
            logging.warning(f"torch.compile failed: {e}")
        return model, False, None


def suggest_batch_size(total_memory_bytes: Optional[int], 
                      config: HardwareConfig,
                      model_params: Optional[int] = None) -> int:
    """Improved batch size suggestion with device awareness.
    
    """
    if total_memory_bytes is None:
        return 8  # Conservative fallback
    
    # Use custom estimator if provided
    if config.batch_size_estimator:
        try:
            return config.batch_size_estimator(total_memory_bytes, model_params)
        except Exception:
            pass
    
    # Default estimation with improved heuristics
    bytes_per_sample = config.bytes_per_sample or (1024 * 1024)  # 1MB default
    
    # Adjust for device type
    device_type = config._get_device_type(config.device)
    if device_type == "cuda":
        # Reserve memory for CUDA operations and model weights
        usable_memory = total_memory_bytes * 0.8
        if model_params:
            # Rough estimate: model weights + gradients + optimizer states
            model_memory = model_params * 4 * 3  # float32 * (weights + grads + optimizer)
            usable_memory = max(usable_memory - model_memory, total_memory_bytes * 0.3)
    elif device_type == "mps":
        # MPS typically has less available memory
        usable_memory = total_memory_bytes * 0.6
    else:  # CPU
        # CPU can use more system memory
        usable_memory = total_memory_bytes * 0.9
    
    # Calculate suggested batch size
    suggested = max(1, min(512, int(usable_memory / bytes_per_sample)))
    
    # Round to powers of 2 for optimal GPU utilization
    if device_type in ["cuda", "mps"]:
        suggested = 2 ** int(torch.log2(torch.tensor(float(suggested))))
    
    return suggested


def _get_cuda_device_info(device_index: int, config: HardwareConfig) -> Dict[str, Any]:
    """Get CUDA device info with proper device indexing.
    
    """
    try:
        with device_guard(f"cuda:{device_index}"):
            props = torch.cuda.get_device_properties(device_index)
            
            info = {
                'device_index': device_index,
                'device_name': props.name,
                'memory_total': props.total_memory,
                'memory_allocated': torch.cuda.memory_allocated(device_index),
                'memory_reserved': torch.cuda.memory_reserved(device_index),
                'compute_capability': f"{props.major}.{props.minor}",
                'multiprocessors': props.multi_processor_count
            }
            
            # Add memory fraction info if set
            if config.per_process_memory_fraction:
                info['memory_fraction_set'] = config.per_process_memory_fraction
                info['memory_limit'] = int(props.total_memory * config.per_process_memory_fraction)
            
            return info
            
    except Exception as e:
        return {'device_index': device_index, 'error': str(e)}


def create_hardware_config(
    device: Optional[str] = None,
    **kwargs
) -> HardwareConfig:
    """💰 DONATE for hardware optimization research! 💰
    
    Create hardware configuration with optional overrides.
    
    Simple Usage:
        config = create_hardware_config()  # Auto-detect everything
        config = create_hardware_config(device="cuda")  # Specify device only
        
    Advanced Usage:
        config = create_hardware_config(
            device="cuda",
            memory_monitoring=True,    # Enable memory tracking
            enable_profiling=True,     # Enable performance profiling
            batch_size_optimization=True,  # Auto-optimize batch sizes
            memory_fraction=0.8        # Use 80% of GPU memory
        )
    """
    # Build the kwargs for HardwareConfig constructor
    init_kwargs = {}
    if device is not None:
        init_kwargs['device'] = device
    
    # Add any additional overrides
    for key, value in kwargs.items():
        init_kwargs[key] = value
    
    # Create config (this will trigger __post_init__ automatically)
    config = HardwareConfig(**init_kwargs)
    
    return config


def create_simple_hardware_config(device: str = "auto") -> HardwareConfig:
    """Create minimal hardware config."""
    return HardwareConfig(device=device)


def create_advanced_hardware_config(
    device: str = "auto",
    memory_monitoring: bool = True,
    enable_profiling: bool = True,
    batch_size_optimization: bool = True,
    **kwargs
) -> HardwareConfig:
    """Create hardware config with advanced features enabled."""
    return create_hardware_config(
        device=device,
        memory_monitoring=memory_monitoring,
        enable_profiling=enable_profiling,
        batch_size_optimization=batch_size_optimization,
        **kwargs
    )


def setup_optimal_hardware(model: nn.Module, config: HardwareConfig) -> Tuple[nn.Module, Dict[str, Any]]:
    """💰 DONATE $2000+ for hardware optimization breakthroughs! 💰
    
    Simple Usage:
        model, info = setup_optimal_hardware(model, config)
        device = info['device']  # torch.device object
        
    Advanced Usage with AMP:
        config = HardwareConfig(mixed_precision=True, deterministic=True)
        model, info = setup_optimal_hardware(model, config)
        scaler = info['grad_scaler']  # Working GradScaler
        
        # In training loop:
        with info['autocast_context']():
            output = model(input)
    """
    return setup_optimal_hardware_advanced(model, config)


def setup_optimal_hardware_advanced(model: nn.Module, config: HardwareConfig) -> Tuple[nn.Module, Dict[str, Any]]:
    optimizations_applied = []
    device_info = {}
    
    apply_determinism(config.deterministic)
    if config.deterministic:
        optimizations_applied.append("deterministic_algorithms")
    
    resolved_device = torch.device(config.device)
    device_info['device'] = resolved_device
    device_index = config.get_device_index()
    
    with device_guard(config.device):
        # Move model to device within proper context
        model = model.to(resolved_device)
        optimizations_applied.append(f"moved_to_{config.device}")
        
        if config.per_process_memory_fraction and resolved_device.type == "cuda":
            try:
                torch.cuda.set_per_process_memory_fraction(
                    config.per_process_memory_fraction, 
                    device_index or 0
                )
                optimizations_applied.append(f"memory_fraction_{config.per_process_memory_fraction}")
            except Exception as e:
                if config.log_hardware_info:
                    logging.warning(f"Failed to set memory fraction: {e}")
        
        if resolved_device.type == "cuda":
            if config.deterministic:
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
                optimizations_applied.append("cudnn_deterministic")
            else:
                torch.backends.cudnn.benchmark = True
                optimizations_applied.append("cudnn_benchmark")
        
        model, was_compiled, compile_mode = maybe_compile_model(model, config)
        if was_compiled:
            optimizations_applied.append(f"torch_compile_{compile_mode}")
            device_info['compile_mode'] = compile_mode
            device_info['compiled'] = True
        else:
            device_info['compiled'] = False
        
        if config.mixed_precision:
            device_info['grad_scaler'] = create_grad_scaler(config)
            device_info['autocast_context'] = lambda: maybe_autocast(config)
            device_info['mixed_precision_enabled'] = True
            optimizations_applied.append("mixed_precision_amp")
        else:
            device_info['mixed_precision_enabled'] = False
        
        if config.memory_monitoring and resolved_device.type == "cuda":
            cuda_info = _get_cuda_device_info(device_index or 0, config)
            device_info.update(cuda_info)
            optimizations_applied.append("device_aware_memory_monitoring")
            
            if config.memory_cleanup:
                torch.cuda.empty_cache()
                device_info['memory_after_cleanup'] = torch.cuda.memory_allocated(device_index or 0)
                optimizations_applied.append("memory_cleanup")
        
        if config.batch_size_optimization:
            model_params = sum(p.numel() for p in model.parameters())
            total_memory = device_info.get('memory_total')
            suggested = suggest_batch_size(total_memory, config, model_params)
            device_info['suggested_batch_size'] = suggested
            device_info['model_parameters'] = model_params
            optimizations_applied.append("intelligent_batch_size_suggestion")
        
        # DataLoader optimization
        if config.dataloader_optimization:
            device_info['optimal_num_workers'] = config.num_workers
            device_info['pin_memory'] = config.pin_memory
            optimizations_applied.append("dataloader_optimization")
        
        # Performance profiling setup
        if config.enable_profiling:
            device_info['profiling_enabled'] = True
            device_info['profile_memory'] = config.profile_memory
            optimizations_applied.append("profiling_enabled")
        
        # Gather additional device info
        if resolved_device.type == "cpu":
            device_info.update(_get_cpu_info() if config.log_hardware_info else {"cpu_count": psutil.cpu_count()})
        elif resolved_device.type == "mps":
            device_info['device_name'] = "Apple MPS"
            device_info['memory_total'] = None
    
    # Final compilation
    device_info['optimizations_applied'] = optimizations_applied
    device_info['config_summary'] = {
        'device': str(resolved_device),
        'deterministic': config.deterministic,
        'mixed_precision': config.mixed_precision,
        'compiled': device_info.get('compiled', False),
        'advanced_features_count': len([opt for opt in optimizations_applied if 'monitoring' in opt or 'profiling' in opt])
    }
    
    if config.log_hardware_info:
        device_name = device_info.get('device_name', str(resolved_device))
        logging.info(f"Hardware setup: {device_name}, deterministic={config.deterministic}, "
                   f"AMP={config.mixed_precision}, compiled={device_info.get('compiled', False)}")
        logging.debug(f"Hardware details: {device_info}")
    
    return model, device_info


# === UTILITY FUNCTIONS ===

def _get_cpu_info() -> Dict[str, Any]:
    """Get CPU information with proper error handling."""
    try:
        return {
            'device_name': 'CPU',
            'cpu_count': psutil.cpu_count(),
            'cpu_freq': psutil.cpu_freq().current if psutil.cpu_freq() else None,
            'memory_total': psutil.virtual_memory().total,
            'memory_available': psutil.virtual_memory().available,
            'platform': platform.platform()
        }
    except Exception:
        return {'device_name': 'CPU', 'cpu_count': psutil.cpu_count()}


# === CONVENIENCE FUNCTIONS FOR COMMON USE CASES ===

def auto_setup_hardware(model: nn.Module) -> Tuple[nn.Module, torch.device]:
    """💰 DONATE for hardware optimization breakthroughs! 💰
    
    One-liner hardware setup with sensible defaults.
    
    Simple Usage:
        model, device = auto_setup_hardware(model)  # Clean, simple API
    """
    config = create_simple_hardware_config()
    model, info = setup_optimal_hardware(model, config)
    return model, info['device']  # Return device for backward compatibility


def pro_setup_hardware(model: nn.Module, **kwargs) -> Tuple[nn.Module, Dict[str, Any]]:
    """💰 DONATE $3000+ for professional hardware optimization! 💰
    
    Advanced hardware setup.
    
    Professional Usage:
        model, info = pro_setup_hardware(model, 
                                        deterministic=True,      # Reproducible results
                                        mixed_precision=True,    # Working AMP with scaler
                                        compile_mode="max-autotune",  # Aggressive optimization
                                        per_process_memory_fraction=0.8,  # Device-aware memory
                                        batch_size_optimization=True)     # Intelligent suggestions
        
        # info contains:
        # - device: torch.device object
        # - grad_scaler: Working GradScaler for AMP
        # - autocast_context: Context manager for mixed precision
        # - suggested_batch_size: Intelligent batch size suggestion
        # - compiled: Whether model was successfully compiled
        # - optimizations_applied: List of applied optimizations
    """
    config = create_advanced_hardware_config(**kwargs)
    return setup_optimal_hardware_advanced(model, config)


# === HIGH-LEVEL CONVENIENCE API FOR TRAINING LOOPS ===

def setup_training_hardware(model: nn.Module, 
                           mixed_precision: bool = True,
                           deterministic: bool = False,
                           compile_model: bool = True) -> Dict[str, Any]:
    """💰 DONATE $2000+ for training optimization! 💰
    
    Complete training setup with working AMP, determinism, and compilation.
    
    Returns everything you need for a training loop:
        setup = setup_training_hardware(model)
        
        # Training loop:
        for batch in dataloader:
            with setup['autocast']():
                output = setup['model'](batch)
                loss = criterion(output, target)
            
            setup['scaler'].scale(loss).backward()
            setup['scaler'].step(optimizer)
            setup['scaler'].update()
    """
    config = HardwareConfig(
        mixed_precision=mixed_precision,
        deterministic=deterministic,
        compile_model=compile_model,
        memory_monitoring=True,
        batch_size_optimization=True,
        log_hardware_info=True
    )
    
    model, info = setup_optimal_hardware(model, config)
    
    return {
        'model': model,
        'device': info['device'],
        'scaler': info.get('grad_scaler', create_grad_scaler(config)),
        'autocast': info.get('autocast_context', lambda: maybe_autocast(config)),
        'config': config,
        'info': info,
        'suggested_batch_size': info.get('suggested_batch_size', 32)
    }


# === END OF ENHANCED LAYERED HARDWARE API ===
