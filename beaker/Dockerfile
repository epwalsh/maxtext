# FROM nvcr.io/nvidia/jax:25.08-py3
FROM ghcr.io/nvidia/jax:maxtext-2025-08-30

# Install direct dependencies, but not source code.
WORKDIR /stage
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    rm -rf *

WORKDIR /runtime
