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

# Install Python Dependencies
COPY pyproject.toml requirements.txt requirements.dev.txt requirements.inference.txt requirements.p3d.txt ./
RUN pip install -r requirements.txt
RUN pip install -r requirements.p3d.txt
RUN pip install -r requirements.inference.txt
RUN pip install -r requirements.dev.txt
RUN pip install 'huggingface-hub[cli]<1.0' runpod requests

# Apply Hydra Patch (Modifies site-packages in the env)
COPY patching ./patching
RUN python ./patching/hydra

# Stage 2: Downloader (Downloads weights in parallel)
FROM python:3.11-slim AS downloader
ARG HF_TOKEN
RUN pip install 'huggingface-hub[cli]<1.0'

# Download to a temporary directory
# We download the 'checkpoints' folder from the repo
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

# Install Runtime System Dependencies
# We need these for OpenCV, PyTorch3D rendering, etc.
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
# Copy the contents of the downloaded 'checkpoints' folder into 'checkpoints/hf'
COPY --from=downloader /tmp/sam3d_download/checkpoints/ ./checkpoints/hf/

# Copy Source Code
COPY . .

# Install Project (Editable mode)
# This is fast and necessary for the code to be importable
RUN pip install -e .

CMD ["python", "-u", "handler.py"]
