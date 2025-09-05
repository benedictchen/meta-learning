# Meta-Learning Toolkit

<div align="center">

![Meta-Learning Banner](https://img.shields.io/badge/Meta--Learning-Toolkit-blue?style=for-the-badge&logo=brain)

[![PyPI version](https://badge.fury.io/py/meta-learning-toolkit.svg)](https://pypi.org/project/meta-learning-toolkit/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: Custom](https://img.shields.io/badge/License-Custom-red.svg)](LICENSE)
[![Tests](https://github.com/benedictchen/meta-learning-toolkit/actions/workflows/test.yml/badge.svg)](https://github.com/benedictchen/meta-learning-toolkit/actions)
[![Documentation](https://img.shields.io/badge/docs-mkdocs-blue)](https://meta-learning-toolkit.readthedocs.io)

**Production-ready meta-learning algorithms with research-accurate implementations**

[Get Started](quickstart.md){ .md-button .md-button--primary }
[View Examples](examples/basic.md){ .md-button }
[GitHub](https://github.com/benedictchen/meta-learning-toolkit){ .md-button }

</div>

---

## 🚀 Key Features

- **Research-Accurate**: Implementations based on 30+ foundational papers (1987-2025)
- **Production-Ready**: Professional CLI, comprehensive testing, and CI/CD
- **Easy to Use**: Simple API with `fit_episode()` and `make_episode()` functions
- **Well-Documented**: Extensive documentation with mathematical foundations
- **Extensible**: Modular architecture for custom algorithms

## 🧠 Algorithms Implemented

| Algorithm | Paper | Year | Status |
|-----------|--------|------|---------|
| **Prototypical Networks** | Snell et al. | 2017 | ✅ Complete |
| **MAML** | Finn et al. | 2017 | ✅ Complete |
| **Test-Time Compute Scaling** | 2024 Research | 2024 | ✅ **First Implementation** |
| **Few-Shot Learning** | Multi-Scale | 2024 | ✅ Complete |
| **Online Meta-Learning** | Continual | 2019 | ✅ Complete |

## ⚡ 60-Second Quickstart

```python
import meta_learning as ml

# Load a dataset
dataset = ml.get_dataset("omniglot", split="train")

# Sample an episode
support_x, support_y, query_x, query_y = ml.make_episode(
    dataset, n_way=5, k_shot=1, n_query=15
)

# Use Prototypical Networks
model = ml.ProtoHead(feature_extractor)
logits = ml.fit_episode(model, support_x, support_y, query_x)

# Compute accuracy  
accuracy = (logits.argmax(-1) == query_y).float().mean()
print(f"Accuracy: {accuracy:.3f}")
```

Or via CLI:

```bash
# Install
pip install meta-learning-toolkit

# Train a model
mlfew fit --dataset omniglot --algorithm protonet --n-way 5 --k-shot 1

# Evaluate
mlfew eval --model checkpoints/protonet_omniglot.pt --dataset omniglot

# Benchmark
mlfew benchmark --datasets omniglot,miniimagenet --algorithms protonet,maml
```

## 📊 Supported Datasets

- **Omniglot** (Lake et al. 2015) - 1,623 handwritten characters
- **miniImageNet** (Vinyals et al. 2016) - 100 ImageNet classes  
- **CIFAR-FS** (Bertinetto et al. 2018) - CIFAR-100 few-shot splits
- **tieredImageNet** (Ren et al. 2018) - Hierarchical ImageNet subset

All datasets include:
- ✅ Automatic downloading
- ✅ Checksum verification  
- ✅ Canonical train/val/test splits
- ✅ Standard preprocessing

## 🎯 Research Foundation

This toolkit implements algorithms from foundational meta-learning papers:

!!! info "Mathematical Rigor"
    All implementations follow the exact mathematical formulations from original papers, with proper citations and research context.

### Prototypical Networks (Snell et al. 2017)

$$c_k = \frac{1}{|S_k|} \sum_{(x_i, y_i) \in S_k} f_\phi(x_i)$$

$$p_\phi(y=k|x) = \frac{\exp(-d(f_\phi(x), c_k))}{\sum_{k'} \exp(-d(f_\phi(x), c_{k'}))}$$

### MAML (Finn et al. 2017)

$$\theta_i' = \theta - \alpha \nabla_\theta \mathcal{L}_{\mathcal{T}_i}(f_\theta)$$

$$\theta \leftarrow \theta - \beta \nabla_\theta \sum_{\mathcal{T}_i \sim p(\mathcal{T})} \mathcal{L}_{\mathcal{T}_i}(f_{\theta_i'})$$

## 💡 What Makes This Special

Unlike existing libraries ([learn2learn](https://github.com/learnables/learn2learn), [torchmeta](https://github.com/tristandeleu/pytorch-meta), [higher](https://github.com/facebookresearch/higher)), this toolkit provides:

- ✅ **Test-Time Compute Scaling** - Available nowhere else publicly
- ✅ **Research-Accurate Math** - Fixes common implementation bugs
- ✅ **Production CLI** - Professional command-line interface  
- ✅ **Complete Documentation** - Mathematical foundations + code examples
- ✅ **Modern Architecture** - Clean API, type hints, comprehensive tests

## 🏗️ Built For Researchers

Created by [Benedict Chen](mailto:benedict@benedictchen.com), this toolkit emerged from frustration with existing meta-learning libraries that had:

- ❌ Bloated, hard-to-understand implementations
- ❌ Missing critical algorithms (especially 2024 breakthroughs)
- ❌ Mathematical errors and research inaccuracies  
- ❌ Poor documentation and examples
- ❌ No production-ready tooling

## 🎓 Citation

If this toolkit helps your research, please consider:

```bibtex
@software{chen2025metalearning,
  title={Meta-Learning Toolkit: Production-Ready Implementations},
  author={Chen, Benedict},
  year={2025},
  url={https://github.com/benedictchen/meta-learning-toolkit}
}
```

## ❤️ Support This Research

This toolkit is developed and maintained by [Benedict Chen](mailto:benedict@benedictchen.com). If it helps your research or projects:

- 💳 [Donate via PayPal](https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS)
- 💖 [Sponsor on GitHub](https://github.com/sponsors/benedictchen)
- ⭐ Star the repository
- 📝 Share with colleagues

*Your support enables continued development of cutting-edge AI research tools!*

---

<div align="center">

**Ready to get started?**

[Installation Guide](installation.md){ .md-button .md-button--primary }
[Quickstart Tutorial](quickstart.md){ .md-button }
[View Examples](examples/basic.md){ .md-button }

</div>