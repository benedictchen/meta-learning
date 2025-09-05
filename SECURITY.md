# Security Policy

## Supported Versions

| Version | Supported          | Security Updates |
| ------- | ------------------ | ---------------- |
| 0.3.x   | :white_check_mark: | Active           |
| 0.2.x   | :x:                | End of Life      |
| 0.1.x   | :x:                | End of Life      |

## Reporting a Vulnerability

We take security seriously. If you discover a security vulnerability, please report it responsibly:

### 🔒 Private Disclosure (Preferred)

**Email**: security@meta-learning.dev
**PGP**: See [public key](#pgp-public-key) below

Please include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact assessment
- Suggested fix (if available)

### 📧 Response Timeline

- **Initial response**: Within 48 hours
- **Triage completion**: Within 7 days  
- **Fix timeline**: Based on severity (see below)

### 🚨 Severity Levels

| Severity | Response Time | Examples |
|----------|---------------|----------|
| **Critical** | 24-48 hours | Remote code execution, privilege escalation |
| **High** | 7 days | Data leakage, authentication bypass |
| **Medium** | 30 days | DoS, information disclosure |
| **Low** | 90 days | Minor information leakage |

## Security Measures

### 🛡️ Supply Chain Security

- **Dependency scanning**: Weekly automated scans with `pip-audit` and `safety`
- **Signed releases**: All releases signed with Sigstore
- **SBOM**: Software Bill of Materials included with releases
- **SLSA**: Level 3 provenance for build reproducibility
- **Pinned dependencies**: Exact versions in `requirements-lock.txt`

### 🔐 Code Security

- **Static analysis**: Bandit security linting in CI
- **Input validation**: All user inputs validated with clear error messages
- **Memory safety**: Careful handling of tensors and numpy arrays
- **Privilege separation**: Docker runs as non-root user

### 🏗️ Build Security

- **Reproducible builds**: Deterministic wheel building
- **Multi-stage Docker**: Minimal production images
- **Image scanning**: Trivy scans for container vulnerabilities
- **Signature verification**: All artifacts signed and verifiable

## Security Best Practices

### For Users

```python
# ✅ Good: Validate input datasets
from meta_learning.utils.validation import validate_episode, check_data_leakage

episode = load_episode(untrusted_source)
validate_episode(episode)  # Throws on invalid data
check_data_leakage(episode['support_y'], episode['query_y'])

# ❌ Bad: Direct usage of untrusted data
model.fit_meta(untrusted_episodes)  # Could contain adversarial examples
```

```python
# ✅ Good: Use memory monitoring
from meta_learning.utils import monitor_memory

with monitor_memory(max_mb=2048):
    model.fit_meta(episodes)  # Automatically prevents memory exhaustion

# ❌ Bad: Uncontrolled memory usage
model.fit_meta(huge_episodes)  # Could cause OOM
```

### For Developers

```python
# ✅ Good: Input validation
def fit_meta(self, episodes, n_epochs=100):
    if not isinstance(n_epochs, int) or n_epochs <= 0:
        raise ValueError(f"n_epochs must be positive integer, got {n_epochs}")
    # ... rest of method

# ❌ Bad: No validation
def fit_meta(self, episodes, n_epochs=100):
    for epoch in range(n_epochs):  # Could crash with invalid input
        # ...
```

## Known Security Considerations

### 🎯 Model Poisoning

**Risk**: Adversarial examples in training episodes can degrade model performance.

**Mitigation**:
```python
from meta_learning.security import detect_adversarial_episodes

# Screen episodes before training
safe_episodes = detect_adversarial_episodes(episodes, threshold=0.95)
model.fit_meta(safe_episodes)
```

### 🔍 Data Leakage

**Risk**: Test data accidentally included in training episodes.

**Mitigation**:
```python
from meta_learning.utils.validation import check_data_leakage

# Automatic validation
check_data_leakage(train_classes, test_classes)  # Throws if overlap detected
```

### 💾 Memory Exhaustion  

**Risk**: Large episodes can cause out-of-memory crashes.

**Mitigation**:
```python
# Built-in memory monitoring
model.fit_meta(episodes, memory_limit_mb=4096)  # Automatic batching
```

### 🏗️ Deserialization Attacks

**Risk**: Pickle files from untrusted sources can execute arbitrary code.

**Mitigation**:
```python
# ✅ Good: Use safe formats
import json
with open('config.json') as f:
    config = json.load(f)

# ❌ Bad: Avoid pickle from untrusted sources  
import pickle
with open('untrusted.pkl', 'rb') as f:
    data = pickle.load(f)  # DANGEROUS
```

## Vulnerability Disclosure History

### CVE-2024-XXXX (Hypothetical)
- **Reported**: 2024-01-15
- **Fixed**: 2024-01-20 (v0.3.1)
- **Severity**: Medium
- **Description**: Information disclosure in episode validation
- **Mitigation**: Upgrade to v0.3.1+

## Security Contact

- **Email**: security@meta-learning.dev
- **Response SLA**: 48 hours
- **Encryption**: PGP key available

### PGP Public Key

```
-----BEGIN PGP PUBLIC KEY BLOCK-----
[PGP key would go here in production]
-----END PGP PUBLIC KEY BLOCK-----
```

## Bug Bounty Program

We currently do not offer a formal bug bounty program, but we recognize security researchers who report vulnerabilities responsibly:

- **Hall of Fame**: Public recognition on our security page
- **Direct communication**: With the development team
- **Early access**: To security fixes and beta releases

---

*Last updated: 2024-12-05*
*Next review: 2025-03-05*