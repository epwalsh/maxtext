FROM nvcr.io/nvidia/jax:25.08-py3
#FROM ghcr.io/nvidia/jax:maxtext-2025-08-30

# Install uv.
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Install direct dependencies, but not source code.
WORKDIR /stage
COPY requirements.txt .
RUN pip uninstall -y jax-cuda13-pjrt jax-cuda13-plugin && \
    /root/.local/bin/uv pip install  --system --break-system-packages -r requirements.txt && \
    rm -rf *

WORKDIR /runtime
