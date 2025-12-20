# syntax=docker/dockerfile:1

# Stage 1: Builder (Compiles dependencies)
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/root/miniforge3/bin:${PATH}"

# Install build tools
RUN apt-get update && apt-get install -y \
    git wget unzip build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Miniforge
RUN wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh \
    && bash /tmp/miniforge.sh -b -p /root/miniforge3 \
    && rm /tmp/miniforge.sh

# Create Conda Environment
# (Only re-runs if default.yml changes)
COPY environments/default.yml /tmp/default.yml
RUN mamba env create -f /tmp/default.yml && mamba clean -afy

# Set up environment for pip installs
ENV PATH="/root/miniforge3/envs/sam3d-objects/bin:$PATH"
ENV CONDA_DEFAULT_ENV=sam3d-objects
ENV PIP_EXTRA_INDEX_URL="https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121"
ENV PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"
ENV MAX_JOBS=1
ENV CUDA_HOME=/usr/local/cuda
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"

WORKDIR /workspace/build

# --- OPTIMIZED PIP INSTALL SECTION ---
# Key Fix: Files are copied and installed individually.
# We use --mount=type=cache to save downloaded wheels between builds.

# 1. Base Requirements (Most stable)
COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# 2. Heavy 3D Libs (Slowest to compile, rarely change)
COPY requirements.p3d.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.p3d.txt

# 3. Inference Requirements (Medium stability)
COPY requirements.inference.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.inference.txt

# 4. Dev Requirements (Highly volatile)
# If you change this file, only THIS step re-runs.
COPY requirements.dev.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.dev.txt

# 5. Project Metadata & Tools
COPY pyproject.toml ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install 'huggingface-hub[cli]<1.0' runpod requests

# Apply Hydra Patch (Modifies site-packages)
# This must happen after packages are installed
COPY patching ./patching
RUN python ./patching/hydra

# Stage 2: Downloader (Downloads weights in parallel)
FROM python:3.11-slim AS downloader
ARG HF_TOKEN
RUN pip install 'huggingface-hub[cli]<1.0'

# Download checkpoints
RUN if [ -n "$HF_TOKEN" ]; then \
    echo "Downloading checkpoints..." && \
    huggingface-cli download \
        --token $HF_TOKEN \
        --repo-type model \
        --include "checkpoints/*" \
        --local-dir /tmp/sam3d_download \
        facebook/sam-3d-objects; \
    else \
    echo "No token provided, creating empty dir"; \
    mkdir -p /tmp/sam3d_download/checkpoints; \
    fi

# Stage 3: Final Runtime (Lightweight & Clean)
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/root/miniforge3/envs/sam3d-objects/bin:$PATH"
ENV CONDA_DEFAULT_ENV=sam3d-objects
# Ensure CUDA libraries are in the path
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}"

# Install Runtime System Dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libasound2 \
    libcairo2 \
    vim \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy Conda Environment from Builder
COPY --from=builder /root/miniforge3 /root/miniforge3

# Copy Checkpoints from Downloader
WORKDIR /workspace/sam-3d-objects
RUN mkdir -p checkpoints/hf
COPY --from=downloader /tmp/sam3d_download/checkpoints/ ./checkpoints/hf/

# Copy Source Code (Most frequent changes happen here)
COPY . .

# Install Project (Editable mode)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -e .

CMD ["python", "-u", "handler.py"]