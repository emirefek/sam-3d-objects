# Base image with CUDA 12.1 support (Required for PyTorch3D and project dependencies)
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Set environment variables to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
# Update PATH for Miniforge
ENV PATH="/root/miniforge3/bin:${PATH}"

# 1. Install System Dependencies
# libgl1-mesa-glx, libglib2.0-0 etc. are required for OpenCV and rendering
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libasound2 \
    libcairo2 \
    git \
    wget \
    unzip \
    build-essential \
    vim \
    && rm -rf /var/lib/apt/lists/*

# 2. Install Miniforge (comes with Mamba and Conda-Forge configured)
# Using Miniforge instead of Miniconda avoids the "conda install mamba" step which often fails
RUN wget \
    https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
    -O /tmp/miniforge.sh \
    && bash /tmp/miniforge.sh -b -p /root/miniforge3 \
    && rm /tmp/miniforge.sh

# 3. Set up Conda Environment
# Copy environment file
COPY environments/default.yml /tmp/default.yml

# Create environment using mamba
RUN mamba env create -f /tmp/default.yml && \
    mamba clean -afy

# Activate environment by adding it to PATH
# This ensures all subsequent commands run inside the conda environment
ENV PATH="/root/miniforge3/envs/sam3d-objects/bin:$PATH"
ENV CONDA_DEFAULT_ENV=sam3d-objects

# 4. Install Dependencies (Cached Layer)
WORKDIR /workspace/sam-3d-objects

# Copy only dependency files first to leverage Docker cache
# This ensures that if you change your code (handler.py, etc.), 
# Docker won't re-run the heavy dependency installation steps.
COPY pyproject.toml requirements.txt requirements.dev.txt requirements.inference.txt requirements.p3d.txt ./

# Set environment variables for installation sources
ENV PIP_EXTRA_INDEX_URL="https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121"
ENV PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"
# Limit compilation parallelism to avoid OOM (Resource Exhausted) errors
ENV MAX_JOBS=1
# Ensure CUDA is found for PyTorch3D build
ENV CUDA_HOME=/usr/local/cuda
# Set CUDA architectures for gsplat and other CUDA extensions
# 8.0 for A100, 8.6 for A6000/RTX3090, 8.9 for L40/RTX4090, 9.0 for H100
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"

# Install dependencies separately.
RUN pip install -r requirements.txt
RUN pip install -r requirements.p3d.txt
RUN pip install -r requirements.inference.txt
RUN pip install -r requirements.dev.txt
RUN pip install 'huggingface-hub[cli]<1.0' runpod requests

# 5. Download Checkpoints (Bake into image)
# Requires passing --build-arg HF_TOKEN=your_token during build
# Placed here to leverage cache (so code changes don't trigger re-download)
ARG HF_TOKEN
RUN if [ -n "$HF_TOKEN" ]; then \
    echo "Baking checkpoints into image..." && \
    mkdir -p checkpoints/hf && \
    huggingface-cli download \
        --token $HF_TOKEN \
        --repo-type model \
        --include "checkpoints/*" \
        --local-dir /tmp/sam3d_download \
        facebook/sam-3d-objects && \
    mv /tmp/sam3d_download/checkpoints/* checkpoints/hf/ && \
    rm -rf /tmp/sam3d_download; \
    else \
    echo "HF_TOKEN not provided. Checkpoints will be downloaded at runtime (slower startup)."; \
    fi

# 6. Copy Project Files
COPY . .

# 7. Install Project
# Install the package itself in editable mode
RUN pip install -e .

# 8. Patch Hydra (Required fix mentioned in setup.md)
RUN python ./patching/hydra

# 9. Copy Handler
COPY handler.py .

# Set the default command to run the handler
CMD ["python", "-u", "handler.py"]
