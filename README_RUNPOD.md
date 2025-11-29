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

## 5. Using Custom Docker Image (Recommended)

Since the installation process involves compiling heavy CUDA dependencies (like PyTorch3D), building the image on RunPod's cloud build often times out (limit is 15 mins). **It is highly recommended to build the image locally and push it to Docker Hub.**

### Prerequisites
*   **Docker Desktop** installed on your computer.
*   A **Docker Hub** account.

### Build and Push Instructions

1.  **Login to Docker Hub**:
    ```bash
    docker login
    ```

2.  **Build the Image**:
    *Important: Since RunPod uses Linux x86_64 servers, you MUST specify the platform if you are building on a Mac (M1/M2/M3) or Windows.*
    
    Replace `yourusername` with your Docker Hub username.
    ```bash
    # This might take 20-40 minutes depending on your computer
    docker build --platform linux/amd64 -t yourusername/sam3d-objects:v1 .
    ```

3.  **Push to Docker Hub**:
    ```bash
    docker push yourusername/sam3d-objects:v1
    ```

4.  **Use on RunPod**:
    *   Start a new Pod.
    *   Select **"Custom Template"**.
    *   **Container Image**: `yourusername/sam3d-objects:v1`
    *   **Container Disk Size**: At least **20 GB** (The image is large).
    *   **Volume Disk Size**: At least **20 GB** (For checkpoints and data).
    *   Launch the pod.

Once the pod is running, the environment is already set up. You just need to download the checkpoints (Step 3 above).

## 6. Deploying as Serverless Endpoint

To deploy this as a scalable Serverless Endpoint on RunPod:

1.  **Prepare the Image**:
    Follow the "Build and Push" instructions in Step 5. Ensure your image is on Docker Hub.

2.  **Create Endpoint**:
    *   Go to RunPod Console > Serverless > New Endpoint.
    *   **Container Image**: `yourusername/sam3d-objects:v1`
    *   **Container Disk**: 20GB+
    *   **Environment Variables**:
        *   `HF_TOKEN`: Your Hugging Face token (Required if checkpoints are not baked in).
    
3.  **Optimization (Cold Starts)**:
    *   The first request will be slow because it downloads the model (if not baked in) and loads it into memory.
    *   To speed this up, you can use **Network Volumes** to store the checkpoints persistently, so they don't need to be downloaded every time a new worker starts.
    *   Or, you can bake the checkpoints into the Docker image during build (requires passing token as build arg, which is complex but fastest for startup).

4.  **Usage**:
    Send a POST request to your endpoint URL:
    ```json
    {
      "input": {
        "image_url": "https://your-image-url.com/image.png",
        "format": "glb"
      }
    }
    ```
    The response will contain the base64 encoded mesh file.

## Troubleshooting

*   **CUDA Errors**: If you see errors related to CUDA, ensure you selected a Pod with CUDA 12.1 support.
*   **Memory Errors**: If the process is killed or you get OOM errors, you might need a GPU with more VRAM (e.g., 40GB or 80GB A100).
*   **Pytorch3D Issues**: The setup script installs Pytorch3D in a specific way to avoid compatibility issues. If it fails, try running `pip install -e '.[p3d]'` again manually.
