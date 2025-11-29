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
ENV PATH /root/miniforge3/envs/sam3d-objects/bin:$PATH
ENV CONDA_DEFAULT_ENV sam3d-objects

# 4. Copy Project Files
WORKDIR /workspace/sam-3d-objects
COPY . .

# 5. Install Python Dependencies
# Set environment variables for installation sources
ENV PIP_EXTRA_INDEX_URL="https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121"
ENV PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"

# Install packages in editable mode
RUN pip install -e '.[dev]' && \
    pip install -e '.[p3d]' && \
    pip install -e '.[inference]' && \
    pip install 'huggingface-hub[cli]<1.0' && \
    pip install runpod requests

# 6. Patch Hydra (Required fix mentioned in setup.md)
RUN python ./patching/hydra

# 7. Copy Handler
COPY handler.py .

# Set the default command to run the handler
CMD ["python", "-u", "handler.py"]
