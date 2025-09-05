# Meta-Learning Toolkit

<div align="center">

[![PyPI version](https://badge.fury.io/py/meta-learning-toolkit.svg)](https://pypi.org/project/meta-learning-toolkit/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: Custom](https://img.shields.io/badge/License-Custom-red.svg)](LICENSE)
[![Tests](https://github.com/benedictchen/meta-learning-toolkit/actions/workflows/test.yml/badge.svg)](https://github.com/benedictchen/meta-learning-toolkit/actions)
[![Documentation](https://img.shields.io/badge/docs-mkdocs-blue)](https://meta-learning-toolkit.readthedocs.io)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

**Production-ready meta-learning algorithms with research-accurate implementations**

*Based on 30+ foundational papers spanning 1987-2025*

[📚 Documentation](https://meta-learning-toolkit.readthedocs.io) •
[🚀 Quick Start](#-60-second-quickstart) •
[💻 CLI Tool](#-cli-tool) •
[🎯 Examples](#-examples) •
[❤️ Support](#️-support-this-research)

</div>

---

## 🧠 What is Meta-Learning?

Meta-learning, or "learning to learn," enables AI systems to rapidly adapt to new tasks with minimal examples. Instead of training from scratch on each task, meta-learning algorithms develop learning strategies that generalize across tasks.

**Key Insight**: Train on many tasks → Learn to learn → Rapidly adapt to new tasks

## ✨ Why This Toolkit?

Unlike existing libraries ([learn2learn](https://github.com/learnables/learn2learn), [torchmeta](https://github.com/tristandeleu/pytorch-meta), [higher](https://github.com/facebookresearch/higher)), this toolkit provides:

- ✅ **Test-Time Compute Scaling** - First public implementation (2024 breakthrough)
- ✅ **Research-Accurate Math** - Fixes common bugs in distance metrics, gradients  
- ✅ **Production CLI** - Professional `mlfew` command-line interface
- ✅ **Complete Documentation** - Mathematical foundations + working examples
- ✅ **Modern Architecture** - Clean API, type hints, comprehensive tests

## 🚀 60-Second Quickstart

### Installation

```bash
pip install meta-learning-toolkit
```

### Basic Usage

```python
import meta_learning as ml
import torch.nn as nn

# 1. Create a feature extractor
feature_extractor = nn.Sequential(
    nn.Conv2d(1, 64, 3, padding=1),
    nn.ReLU(),
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten()
)

# 2. Create Prototypical Networks model
model = ml.ProtoHead(feature_extractor)

# 3. Load dataset and sample episode
dataset = ml.get_dataset("omniglot", split="train")
support_x, support_y, query_x, query_y = ml.make_episode(
    dataset, n_way=5, k_shot=1, n_query=15
)

# 4. Run few-shot learning
logits = model(support_x, support_y, query_x)
accuracy = (logits.argmax(-1) == query_y).float().mean()

print(f"5-way 1-shot accuracy: {accuracy:.3f}")
```

**That's it!** You just ran few-shot learning with Prototypical Networks.

## 💻 CLI Tool

The `mlfew` command provides a complete workflow:

```bash
# Train a model
mlfew fit --dataset omniglot --algorithm protonet --n-way 5 --k-shot 1

# Evaluate performance  
mlfew eval --model checkpoints/protonet_omniglot.pt --dataset omniglot

# Run benchmarks
mlfew benchmark --datasets omniglot,miniimagenet --algorithms protonet,maml
```

## 📊 Supported Datasets

All datasets include automatic downloading, checksum verification, and canonical splits:

| Dataset | Classes | Samples/Class | Paper | Auto-Download |
|---------|---------|---------------|-------|---------------|
| **Omniglot** | 1,623 characters | 20 | Lake et al. 2015 | ✅ |
| **miniImageNet** | 100 classes | 600 | Vinyals et al. 2016 | ⚠️ Manual* |
| **CIFAR-FS** | 100 classes | 600 | Bertinetto et al. 2018 | ✅ |

*Manual download required due to ImageNet licensing. Automatic CIFAR-10 proxy provided.

## 🧪 Algorithms Implemented

| Algorithm | Paper | Year | Implementation Status |
|-----------|--------|------|----------------------|
| **Prototypical Networks** | Snell et al. | 2017 | ✅ Research-accurate |
| **MAML** | Finn et al. | 2017 | ✅ Second-order gradients |
| **Test-Time Compute Scaling** | 2024 Research | 2024 | ✅ **First public impl** |
| **Multi-Scale ProtoNet** | Enhanced | 2024 | ✅ Complete |
| **Online Meta-Learning** | Finn et al. | 2019 | ✅ Continual learning |

## 🔬 Research Accuracy

All implementations follow exact mathematical formulations from original papers:

### Prototypical Networks
```
Prototype computation: c_k = (1/|S_k|) Σ f_φ(x_i) for (x_i, y_i) ∈ S_k
Classification: p(y=k|x) = exp(-d(f_φ(x), c_k)) / Σ_k' exp(-d(f_φ(x), c_k'))
Distance: d(·,·) = ||· - ·||₂² (squared Euclidean)
```

### MAML
```
Inner update: φ_i = θ - α∇_θ L_Ti(f_θ)  
Outer update: θ ← θ - β∇_θ Σ_Ti L_Ti(f_φi)
Gradients: Second-order (create_graph=True)
```

**Common bugs fixed**: Wrong distance signs, missing second-order gradients, BatchNorm episodic leakage.

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

Complete documentation available at: **https://meta-learning-toolkit.readthedocs.io**

- 🚀 [Quick Start Guide](https://meta-learning-toolkit.readthedocs.io/quickstart/)
- 📖 [API Reference](https://meta-learning-toolkit.readthedocs.io/api/)
- 💡 [Examples & Tutorials](https://meta-learning-toolkit.readthedocs.io/examples/)
- 🔬 [Research & Theory](https://meta-learning-toolkit.readthedocs.io/research/)

## 🧪 Testing

Comprehensive test suite with 90%+ coverage:

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