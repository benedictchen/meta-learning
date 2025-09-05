# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.3.1] - 2025-01-09

### Fixed

#### 🔧 Mathematical Correctness Improvements

- **BREAKING**: Fixed ProtoNet temperature scaling direction (tau now correctly softens predictions)
  - Previous: `logits = -tau * dist` (wrong direction - higher tau made predictions harder)
  - Current: `logits = -dist / tau` (correct - higher tau makes predictions softer)
  
- **Fixed**: Unified temperature semantics across distance metrics
  - Both `sqeuclidean` and `cosine` distances now use consistent temperature scaling
  - Temperature parameter `tau`: higher values → higher entropy → less confident predictions
  - Maintains research accuracy with proper mathematical foundations

- **Fixed**: TTCS confidence tracking bug
  - Previous: Used `exp(logits)` for confidence (not normalized, could exceed 1.0)
  - Current: Uses `softmax(logits)` for proper probability normalization

- **Fixed**: Principled diversity weighting in TTCS
  - Previous: Heuristic weighting based on prediction variance
  - Current: Uses KL divergence to mean distribution for theoretically grounded weighting

- **Added**: Input validation for temperature parameter
  - `cosine_logits()` now validates `tau > 0` to prevent mathematical errors
  - Raises `ValueError` for invalid temperature values

#### 🧪 Test Suite Improvements

- **Added**: Comprehensive cosine temperature entropy tests
- **Added**: Mathematical smoke tests for end-to-end validation
- **Fixed**: Flaky TTCS tests with multi-seed averaging
- **Fixed**: Hard-coded version strings replaced with regex patterns
- **Improved**: BatchNorm test robustness for MC-Dropout scenarios

### Documentation

- **Added**: Temperature scaling documentation with unified semantics
- **Updated**: README with Prototypical Networks temperature examples
- **Improved**: Mathematical foundations in docstrings

### Notes

This release addresses critical mathematical correctness issues identified through comprehensive code review. The temperature scaling fix is a breaking change but ensures proper research accuracy. Users should verify that higher `tau` values now correctly produce softer (less confident) predictions.

## [2.3.0] - 2025-01-08

### Added
- Initial PyPI release with comprehensive meta-learning algorithms
- Test-Time Compute Scaling (TTCS) implementation
- Research-accurate MAML variants
- Advanced few-shot learning components
- Production-ready CI/CD pipeline

### Features
- 59/59 passing tests with expanding coverage
- Modern src layout following 2024 best practices
- Professional documentation and examples
- PyPI-ready package distribution