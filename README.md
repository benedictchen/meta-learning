# Meta-Learning Toolkit

<div align="center">

[![PyPI version](https://badge.fury.io/py/meta-learning-toolkit.svg)](https://pypi.org/project/meta-learning-toolkit/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: Custom](https://img.shields.io/badge/License-Custom-red.svg)](LICENSE)
[![Tests](https://github.com/benedictchen/meta-learning-toolkit/actions/workflows/test.yml/badge.svg)](https://github.com/benedictchen/meta-learning-toolkit/actions)
[![Documentation](https://img.shields.io/badge/docs-included-blue)](https://pypi.org/project/meta-learning-toolkit/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

**Production-ready meta-learning algorithms with research-accurate implementations**

*Based on foundational research in meta-learning and few-shot learning*

[📚 Documentation](#-documentation) •
[🚀 Quick Start](#-60-second-quickstart) •
[💻 CLI Tool](#-cli-tool) •
[🎯 Algorithms](#-algorithms-implemented) •
[❤️ Support](#️-support-this-research)

</div>

---

## 🧠 What is Meta-Learning?

Meta-learning, or "learning to learn," enables AI systems to rapidly adapt to new tasks with minimal examples. Instead of training from scratch on each task, meta-learning algorithms develop learning strategies that generalize across tasks.

**Key Insight**: Train on many tasks → Learn to learn → Rapidly adapt to new tasks

## ✨ Why This Toolkit?

This toolkit provides breakthrough meta-learning algorithms not available elsewhere:

- 🔥 **Test-Time Compute Scaling** - World-first public implementation (2024 breakthrough)
- 🧪 **Research-Accurate MAML** - All 5 variants with proper second-order gradients
- 🛠️ **Research Patches** - Critical BatchNorm fixes for few-shot learning
- 📊 **Professional Evaluation** - Statistical rigor with 95% confidence intervals
- ⚙️ **Production Ready** - CLI tools, comprehensive documentation, modern packaging

## 🚀 60-Second Quickstart

### Installation

```bash
pip install meta-learning-toolkit
```

### Basic Usage: Test-Time Compute Scaling (2024 Breakthrough)

```python
from algorithms.test_time_compute_scaler import TestTimeComputeScaler
from algorithms.test_time_compute_config import TestTimeComputeConfig

# 1. Configure test-time compute scaling
config = TestTimeComputeConfig(
    max_compute_budget=100,
    confidence_threshold=0.95,
    use_process_reward=True
)

# 2. Create the scaler
scaler = TestTimeComputeScaler(config)

# 3. Use scaler for improved few-shot performance  
results = scaler.scale_compute(
    task_data=your_few_shot_task,
    base_model=your_model
)

print(f"Improved accuracy: {results['accuracy']:.3f}")
```

### MAML: Research-Accurate Implementation

```python
import torch.nn as nn
from algorithms.maml_research_accurate import ResearchMAML, MAMLConfig, MAMLVariant

# 1. Create your model
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(), 
    nn.Linear(256, 5)  # 5-way classification
)

# 2. Configure MAML
config = MAMLConfig(
    variant=MAMLVariant.MAML,  # or FOMAML, ANIL, BOIL, REPTILE
    inner_lr=0.01,
    inner_steps=5
)

# 3. Create MAML learner
maml = ResearchMAML(model, config)

# 4. Now ready for few-shot meta-learning!
print("MAML ready for training on meta-learning tasks")
```

### Research Patches and Evaluation

```python
# Apply research-accurate BatchNorm fixes
from research_patches.batch_norm_policy import apply_episodic_bn_policy
from research_patches.determinism_hooks import setup_deterministic_environment

# Fix BatchNorm for few-shot learning  
model_fixed = apply_episodic_bn_policy(model, policy="group_norm")

# Ensure reproducible research
setup_deterministic_environment(seed=42)

# Professional evaluation with confidence intervals
from evaluation.few_shot_evaluation_harness import FewShotEvaluationHarness
harness = FewShotEvaluationHarness()
```

**That's it!** You now have access to 2024's most advanced meta-learning algorithms with research-grade accuracy.

## 💻 CLI Tool

The `mlfew` command provides benchmarking and evaluation:

```bash
# Check version
mlfew version

# Run benchmarks on few-shot tasks
mlfew bench --dataset omniglot --n-way 5 --k-shot 1 --episodes 1000

# Evaluate with specific parameters
mlfew eval --dataset miniimagenet --n-way 5 --k-shot 5 --device cuda
```

## 📊 Supported Datasets

| Dataset | Classes | Samples/Class | Paper | Status |
|---------|---------|---------------|-------|---------|
| **CIFAR-FS** | 100 classes | 600 | Bertinetto et al. 2018 | ✅ Built-in |
| **Synthetic** | Configurable | Configurable | N/A | ✅ Built-in |

*Note: This package focuses on breakthrough algorithms. For additional datasets, easily integrate with torchvision or other dataset libraries.*

## 🧪 Algorithms Implemented

| Algorithm | Paper | Year | Implementation Status |
|-----------|--------|------|----------------------|
| **Test-Time Compute Scaling** | Snell et al. | 2024 | ✅ **World-first public implementation** |
| **MAML (All Variants)** | Finn et al. | 2017 | ✅ Research-accurate: MAML, FOMAML, ANIL, BOIL, Reptile |
| **BatchNorm Research Patches** | Various | 2017-2024 | ✅ Episode-aware policies for few-shot learning |
| **Evaluation Harness** | Research Standard | N/A | ✅ 95% confidence intervals, statistical rigor |

## 🔬 Research Accuracy

All implementations follow exact mathematical formulations from original papers:

### MAML (Research-Accurate)
```
Inner adaptation: θ'_i = θ - α * ∇_θ L_{T_i}^{train}(f_θ)
Meta-update: θ ← θ - β * ∇_θ Σ_i L_{T_i}^{test}(f_{θ'_i})
Second-order gradients: create_graph=True (preserved)
Functional updates: No in-place mutations
```

### Test-Time Compute Scaling
```
Compute allocation: C(t) = f(confidence, budget, time)
Process rewards: R_step = quality_estimation(step_output)
Solution selection: argmax_s Σ_i R_i * w_i
```

**Research-critical fixes**: Proper gradient computation, episodic BatchNorm, deterministic environments.

## 🚢 Installation Options

### Option 1: PyPI (Recommended)
```bash
pip install meta-learning-toolkit
```

### Option 2: Development Install
```bash
git clone https://github.com/benedictchen/meta-learning-toolkit
cd meta-learning-toolkit
pip install -e .[dev,test,datasets,visualization]
```

## 🧑‍💻 Requirements

- **Python**: 3.9+
- **PyTorch**: 2.0+  
- **Core**: `numpy`, `scipy`, `scikit-learn`, `tqdm`, `rich`, `pyyaml`
- **Optional**: `matplotlib`, `seaborn`, `wandb` (for visualization)
- **Development**: `pytest`, `ruff`, `mypy`, `pre-commit`

## 📚 Documentation

Complete documentation is included in the package:

- 🚀 **Quick Start**: Examples in this README
- 📖 **API Reference**: Comprehensive docstrings in all modules
- 💡 **Examples**: Working code examples throughout documentation
- 🔬 **Research**: Mathematical formulations and research foundations in docstrings

## 🧪 Testing

Test suite with expanding coverage:

```bash
# Run all tests
pytest

# Run specific test categories
pytest -m "not slow"          # Skip slow tests
pytest -m "regression"        # Mathematical correctness

# With coverage report
pytest --cov=src/meta_learning --cov-report=html
```

## 📄 License

Custom Non-Commercial License - See [LICENSE](LICENSE) for details.

**TL;DR**: Free for research and educational use. Commercial use requires permission.

## 🎓 Citation

If this toolkit helps your research, please cite:

```bibtex
@software{chen2025metalearning,
  title={Meta-Learning Toolkit: Production-Ready Few-Shot Learning},
  author={Chen, Benedict},
  year={2025},
  url={https://github.com/benedictchen/meta-learning-toolkit},
  version={2.0.0}
}
```

## ❤️ Support This Research

This toolkit is developed and maintained by [Benedict Chen](mailto:benedict@benedictchen.com). If it helps your research or projects, please consider:

<div align="center">

[![Sponsor](https://img.shields.io/badge/Sponsor-❤️-red?style=for-the-badge&logo=github)](https://github.com/sponsors/benedictchen)
[![PayPal](https://img.shields.io/badge/PayPal-💙-blue?style=for-the-badge&logo=paypal)](https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS)

</div>

- ⭐ **Star the repository**
- 💳 **[Donate via PayPal](https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS)**
- 💖 **[Sponsor on GitHub](https://github.com/sponsors/benedictchen)**
- 🐦 **Share on social media**
- 📝 **Cite in your papers**

*Your support enables continued development of cutting-edge AI research tools!*

---

<div align="center">

**Built with ❤️ by [Benedict Chen](mailto:benedict@benedictchen.com)**

*Turning research papers into production-ready code*

</div>