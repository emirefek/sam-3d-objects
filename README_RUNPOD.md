# Running SAM 3D Objects on RunPod

This guide will help you set up and run SAM 3D Objects on a RunPod instance.

## 1. Select a Pod

*   **GPU**: Choose a GPU with at least **32GB VRAM** (e.g., A100, A6000, H100). The model requires significant memory.
*   **Template**: Select a template with **PyTorch 2.x** and **CUDA 12.1** support.
    *   Recommended: `RunPod PyTorch 2.2.0` (or newer) which usually includes Conda/Mamba.
    *   Ensure the CUDA version matches the requirements (CUDA 12.1 is used in the setup script).

## 2. Initial Setup

Once your pod is running and you have connected via SSH or the Web Terminal:

1.  **Upload the Code**: You can git clone the repository or upload your local files.
    ```bash
    git clone https://github.com/facebookresearch/sam-3d-objects.git
    cd sam-3d-objects
    ```
    *(If you uploaded your modified code, just `cd` into the folder)*

2.  **Run the Setup Script**:
    I have provided a `setup_runpod.sh` script to automate the installation.
    ```bash
    chmod +x setup_runpod.sh
    ./setup_runpod.sh
    ```
    *This script will install system dependencies, create the conda environment, install python packages, and patch hydra.*

## 3. Download Checkpoints

You need to download the model weights from Hugging Face.

1.  **Set Hugging Face Token**:
    You need an access token from Hugging Face with access to the `facebook/sam-3d-objects` repository.
    Set it as an environment variable (you can also add this to your RunPod environment variables):
    ```bash
    export HF_TOKEN="your_huggingface_token_here"
    ```

2.  **Download Weights**:
    ```bash
    conda activate sam3d-objects
    
    TAG=hf
    huggingface-cli download \
      --token $HF_TOKEN \
      --repo-type model \
      --local-dir checkpoints/${TAG}-download \
      --max-workers 4 \
      facebook/sam-3d-objects
    
    mkdir -p checkpoints/${TAG}
    mv checkpoints/${TAG}-download/checkpoints/* checkpoints/${TAG}/
    rm -rf checkpoints/${TAG}-download
    ```

## 4. Run the Demo

Now you can run the export example we created:

```bash
conda activate sam3d-objects
python export_mesh_example.py
```

This will generate `model.glb`, `model.obj`, and potentially `model.fbx` (if supported) in the current directory.

## 5. (Optional) Using Custom Docker Image

If you prefer to use a custom Docker image instead of setting up the environment every time:

1.  **Build the Image** (Locally or on a build server):
    ```bash
    docker build -t yourusername/sam3d-objects:v1 .
    ```

2.  **Push to Docker Hub**:
    ```bash
    docker push yourusername/sam3d-objects:v1
    ```

3.  **Use on RunPod**:
    *   When creating a new Pod, select "Custom Template".
    *   Enter your image name: `yourusername/sam3d-objects:v1`.
    *   Set Container Disk Size to at least 20GB.
    *   Start the pod. The environment will be pre-installed.
    *   You will still need to download the checkpoints (Step 3) as they are too large to bake into the image.

## Troubleshooting

*   **CUDA Errors**: If you see errors related to CUDA, ensure you selected a Pod with CUDA 12.1 support.
*   **Memory Errors**: If the process is killed or you get OOM errors, you might need a GPU with more VRAM (e.g., 40GB or 80GB A100).
*   **Pytorch3D Issues**: The setup script installs Pytorch3D in a specific way to avoid compatibility issues. If it fails, try running `pip install -e '.[p3d]'` again manually.
