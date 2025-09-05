# Multi-stage Dockerfile for meta-learning
# Supports both CPU and CUDA variants with reproducible builds

ARG PYTHON_VERSION=3.9
ARG TORCH_VERSION=2.1.0
ARG CUDA_VERSION=11.8

FROM python:${PYTHON_VERSION}-slim as base

# Metadata for supply chain security
LABEL maintainer="meta-learning-team"
LABEL version="0.3.0"
LABEL description="Industrial-grade meta-learning library"
LABEL org.opencontainers.image.source="https://github.com/user/meta-learning"
LABEL org.opencontainers.image.licenses="MIT"

# Security: Create non-root user
RUN groupadd -r metalearning && useradd -r -g metalearning metalearning

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements-lock.txt ./
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-lock.txt

# Copy source code
COPY . .

# Install package in development mode
RUN pip install -e .

# Set ownership
RUN chown -R metalearning:metalearning /app

# Switch to non-root user
USER metalearning

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import meta_learning; print('OK')" || exit 1

# Default command
CMD ["python", "-c", "import meta_learning; meta_learning.check_performance_env()"]

# CPU variant (default)
FROM base as cpu
ENV TORCH_DEVICE=cpu
ENV OMP_NUM_THREADS=4

# GPU variant with CUDA
FROM base as cuda
ARG CUDA_VERSION
ENV TORCH_DEVICE=cuda
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Install CUDA-compatible PyTorch
RUN pip uninstall -y torch torchvision && \
    pip install --no-cache-dir \
    torch==${TORCH_VERSION}+cu$(echo ${CUDA_VERSION} | sed 's/\.//') \
    torchvision==0.16.0+cu$(echo ${CUDA_VERSION} | sed 's/\.//') \
    --index-url https://download.pytorch.org/whl/cu$(echo ${CUDA_VERSION} | sed 's/\.//')

# Development variant with additional tools  
FROM base as dev
USER root

# Install development tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    vim \
    tmux \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Install development Python packages
RUN pip install --no-cache-dir \
    jupyter \
    jupyterlab \
    ipython \
    tensorboard

# Expose Jupyter port
EXPOSE 8888

USER metalearning

# Jupyter command for development
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]