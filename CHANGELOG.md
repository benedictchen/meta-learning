# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - Enterprise Enhancements

### Added
- Professional benchmarking suite with statistical confidence intervals
- Property-based testing using Hypothesis for mathematical invariants
- Docker containerization with multi-stage builds (CPU/GPU/dev variants)
- VS Code devcontainer configuration for consistent development environment
- Comprehensive CI/CD pipeline with automated testing and releases
- Documentation framework with reproducibility guidelines
- Enterprise-grade requirements management (requirements.txt, requirements-dev.txt)

### Fixed
- **CRITICAL**: All `create_graph=False` → `create_graph=True` in MAML implementations
- **CRITICAL**: Meta-learning gradient flow now mathematically correct (21 fixes)
- Second-order optimization gradients properly preserved
- Gradient computation breaks in test-time compute scaling resolved

### Changed
- Enhanced Dockerfile with security hardening and multi-stage builds
- Improved CI pipeline with comprehensive linting and type checking
- Added statistical rigor to benchmark evaluation protocols

## [0.1.0] - Initial Research Implementation

### Added
- Paper-faithful implementations of ProtoNet & MAML algorithms
- Research guardrails (BatchNorm policy, episode contract, determinism, leakage checks)  
- Modular architecture with 67,244+ lines of organized research code
- Complete test suites for mathematical correctness validation
- CLI interface and configuration management
- Documentation skeleton with research references

### Research Papers Implemented
- **MAML**: Finn et al. (2017) "Model-Agnostic Meta-Learning for Fast Adaptation"
- **Prototypical Networks**: Snell et al. (2017) "Prototypical Networks for Few-shot Learning"
- **Matching Networks**: Vinyals et al. (2016) "Matching Networks for One Shot Learning"
- **Test-Time Compute Scaling**: Advanced meta-learning optimization techniques