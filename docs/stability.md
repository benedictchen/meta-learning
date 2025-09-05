# API Stability Index

This document defines the stability guarantees for different parts of the meta-learning library.

## Stability Levels

### 🟢 **Stable** (v1.0+)
- **Guaranteed**: No breaking changes without major version bump
- **Deprecation**: 6+ month notice with migration path  
- **Documentation**: Complete with examples
- **Testing**: 100% coverage, property tests, performance budgets

### 🟡 **Beta** (v0.3.0+)  
- **Likely stable**: Minor changes possible with deprecation warning
- **Deprecation**: 3+ month notice with migration path
- **Documentation**: Complete API docs
- **Testing**: >90% coverage

### 🔴 **Alpha** (v0.1.0+)
- **Experimental**: Breaking changes without notice
- **Use cases**: Research, experimentation, feedback
- **Documentation**: Basic usage examples
- **Testing**: Basic functionality tests

### ⚫ **Internal**
- **Private APIs**: Subject to change without notice
- **Import path**: Contains `_internal` or starts with `_`
- **Usage**: Not recommended for external use

## Current API Stability (v0.3.0)

### Core Algorithms
| API | Level | Notes |
|-----|-------|-------|
| `ProtoNet` | 🟢 Stable | Main class, fit_meta(), evaluate() |
| `MAML` | 🟢 Stable | Main class, inner_adapt_and_eval() |
| `ProtoNet.forward()` | 🟡 Beta | Signature may change |
| `MAML.inner_loop()` | 🔴 Alpha | Implementation details |

### Episode Management  
| API | Level | Notes |
|-----|-------|-------|
| `make_episodes()` | 🟢 Stable | Core episode creation |
| `validate_episode()` | 🟡 Beta | Input validation |
| `EpisodeDataset` | 🟡 Beta | May add methods |
| `episode_stats()` | 🔴 Alpha | Metrics computation |

### Datasets
| API | Level | Notes |
|-----|-------|-------|  
| `MiniImageNet` | 🟢 Stable | Standard benchmark |
| `CIFARFS` | 🟢 Stable | Standard benchmark |
| `Omniglot` | 🟡 Beta | Less mature |
| `TieredImageNet` | 🔴 Alpha | Experimental |

### Backbones
| API | Level | Notes |
|-----|-------|-------|
| `Conv4` | 🟢 Stable | Standard architecture |
| `ResNet12` | 🟢 Stable | Standard architecture |
| `load_backbone()` | 🟡 Beta | May add parameters |
| `ViT` | 🔴 Alpha | Experimental |

### Heads & Components
| API | Level | Notes |
|-----|-------|-------|
| `ProtoHead` | 🟡 Beta | Core prototypical head |
| `LinearHead` | 🟡 Beta | Simple classification |
| `AttentionHead` | 🔴 Alpha | Research feature |
| `TransformerHead` | 🔴 Alpha | Experimental |

### Utilities
| API | Level | Notes |
|-----|-------|-------|
| `check_data_leakage()` | 🟡 Beta | Validation utility |
| `compute_confidence_interval()` | 🟡 Beta | Statistics |
| `profile_memory()` | 🔴 Alpha | Debugging |

### Internal APIs
| API | Level | Notes |
|-----|-------|-------|
| `meta_learning._internal.*` | ⚫ Internal | Implementation details |
| `meta_learning.*.training_step()` | ⚫ Internal | Private training |
| `meta_learning.*._*` | ⚫ Internal | Private methods |

## Migration Guarantees

### Breaking Changes
- **Stable APIs**: No breaking changes without major version bump
- **Beta APIs**: Minimum 3 months deprecation notice
- **Alpha APIs**: Best-effort compatibility notice  

### Deprecation Process
1. **Warning Phase** (1-2 releases): DeprecationWarning with clear migration path
2. **Error Phase** (1 release): Raise error with migration instructions  
3. **Removal Phase** (next major): Complete removal

### Example Migration
```python
# v0.2.0 - Working  
task = make_task(dataset, n_way=5, n_shot=1)

# v0.3.0 - Deprecated with warning
task = make_task(dataset, n_way=5, n_shot=1)  # DeprecationWarning
episode = make_episodes(dataset, n_way=5, n_shot=1)  # New API

# v0.4.0 - Error with guidance
task = make_task(dataset, n_way=5, n_shot=1)  # RuntimeError with migration

# v1.0.0 - Removed
episode = make_episodes(dataset, n_way=5, n_shot=1)  # Only new API
```

## Version Planning

### v0.4.0 (Q1 2025)
- Promote `ProtoHead`, `LinearHead` to Stable
- Add performance budgets to CI
- Complete mypy --strict compliance  

### v1.0.0 (Q2 2025)  
- All core APIs promoted to Stable
- Plugin system finalized
- Complete benchmarking suite
- Signed releases with SBOM

### Post-1.0.0
- Semantic versioning strictly followed
- Long-term support releases
- Enterprise stability guarantees

## Feedback & Contributions

- **Stability concerns**: Open issue with "stability" label
- **API suggestions**: Discussion in GitHub Discussions
- **Breaking change proposals**: RFC process required

For questions about API stability, see [GitHub Discussions](https://github.com/user/meta-learning/discussions) or email maintainers@meta-learning.dev.