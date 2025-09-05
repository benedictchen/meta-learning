# Reproducibility

Ensuring reproducible meta-learning research with deterministic guarantees.

## Research Guarantees

This package provides:
- **Deterministic seeding** helpers for PyTorch and NumPy
- **Episode contract** validation to prevent data leakage
- **Normalization leakage guards** to prevent train/test contamination  
- **CI/CD pipeline** ensuring unit and property tests pass on CPU environments

## Environment Control

### Dependency Management
- Lock your environment with `requirements*.txt` or Poetry/uv lockfile
- For absolute reproducibility, use the provided `Dockerfile` or VS Code devcontainer
- Pin PyTorch version to avoid numerical differences across releases

### Deterministic Configuration
```python
from meta_learning.utils.determinism import setup_deterministic_environment

# Set global random seeds
setup_deterministic_environment(seed=42)

# For CUDA determinism (slower but reproducible)
setup_deterministic_environment(seed=42, strict_determinism=True)
```

## Command Recording

Every benchmark run should record:
- **Config hash**: Resolved configuration parameters and their hash
- **Random seeds**: All seeds used (torch, numpy, random, cuda)
- **Environment**: `pip freeze` output and system information
- **Metrics**: Mean ± 95% CI with exact episode count
- **Hardware**: GPU model, CUDA version, driver version

Example metadata:
```json
{
  "config_hash": "a1b2c3d4...",
  "seeds": {"torch": 42, "numpy": 42, "random": 42},
  "episodes": 10000,
  "result": {"mean": 0.498, "ci95": 0.003},
  "environment": "torch==2.2.2, numpy==1.26.4",
  "hardware": "Tesla V100, CUDA 12.1"
}
```

## Research Standards

Following established few-shot learning evaluation protocols:
- **Minimum 10,000 episodes** for statistical significance
- **95% confidence intervals** using proper t-distribution
- **Cross-validation** across multiple random seeds
- **Identical data splits** using standard benchmarks (miniImageNet, tieredImageNet)
- **Fair comparison** using same backbone architectures and hyperparameters