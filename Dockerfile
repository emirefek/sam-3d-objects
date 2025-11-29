# Base image with CUDA 12.1 support (Required for PyTorch3D and project dependencies)
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Set environment variables to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/root/miniconda3/bin:${PATH}"

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

# 2. Install Miniconda
RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /root/miniconda3 \
    && rm /tmp/miniconda.sh

# 3. Set up Conda Environment
# Copy environment file
COPY environments/default.yml /tmp/default.yml

# Install mamba and create environment
# We use mamba for faster resolution as recommended in the docs
RUN conda install -n base -c conda-forge mamba -y && \
    mamba env create -f /tmp/default.yml && \
    conda clean -afy

# Activate environment by adding it to PATH
# This ensures all subsequent commands run inside the conda environment
ENV PATH /root/miniconda3/envs/sam3d-objects/bin:$PATH
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
    pip install 'huggingface-hub[cli]<1.0'

# 6. Patch Hydra (Required fix mentioned in setup.md)
RUN python ./patching/hydra

# Set the default command to bash
CMD ["/bin/bash"]
