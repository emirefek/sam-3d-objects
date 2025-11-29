#!/bin/bash

# Stop on error
set -e

echo "Starting setup for SAM 3D Objects on RunPod..."

# 1. System Dependencies
echo "Installing system dependencies..."
apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libasound2 \
    libcairo2 \
    git \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# 2. Conda/Mamba Setup
# Check if mamba is installed, if not try conda, if not install micromamba
if ! command -v mamba &> /dev/null; then
    if command -v conda &> /dev/null; then
        echo "Mamba not found, using conda..."
        CMD=conda
    else
        echo "Conda not found. Please use a RunPod template with Conda/Mamba installed (e.g. PyTorch templates)."
        exit 1
    fi
else
    CMD=mamba
fi

# Initialize conda for shell interaction
eval "$($CMD shell.bash hook)"

# 3. Create Environment
echo "Creating conda environment from environments/default.yml..."
# Check if env exists
if $CMD env list | grep -q "sam3d-objects"; then
    echo "Environment sam3d-objects already exists. Updating..."
    $CMD env update -f environments/default.yml --prune
else
    $CMD env create -f environments/default.yml
fi

# Activate environment
echo "Activating environment..."
conda activate sam3d-objects

# 4. Install Python Dependencies
echo "Installing Python dependencies..."

# Set environment variables for installation
export PIP_EXTRA_INDEX_URL="https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121"
export PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"

# Install packages
echo "Installing core dependencies..."
pip install -e '.[dev]'

echo "Installing PyTorch3D dependencies..."
pip install -e '.[p3d]'

echo "Installing Inference dependencies..."
pip install -e '.[inference]'

# 5. Patch Hydra
echo "Patching Hydra..."
python ./patching/hydra

# 6. Install Hugging Face CLI for checkpoints
pip install 'huggingface-hub[cli]<1.0'

echo "Setup complete!"
echo "To start using the environment, run: conda activate sam3d-objects"
echo "Don't forget to set your HF_TOKEN and download checkpoints as described in README_RUNPOD.md"
