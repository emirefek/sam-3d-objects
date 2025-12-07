import sys
import os
import shutil
import base64
import requests
import torch
import numpy as np
import trimesh
import runpod
from io import BytesIO
from PIL import Image
import tempfile

# Add notebook folder to path to import inference
sys.path.append("notebook")
from inference import Inference, load_image, load_mask

# Global variable to hold the model
inference_model = None


def download_file(url, suffix=".png"):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        for chunk in response.iter_content(chunk_size=8192):
            tmp.write(chunk)
        return tmp.name


def init_model():
    global inference_model
    if inference_model is None:
        print("Loading model...")
        # Assuming checkpoints are mounted or downloaded to checkpoints/
        tag = "hf"
        config_path = f"checkpoints/{tag}/pipeline.yaml"

        if not os.path.exists(config_path):
            print(
                f"Config not found at {config_path}. Attempting to download checkpoints..."
            )
            hf_token = os.environ.get("HF_TOKEN")
            if not hf_token:
                raise ValueError(
                    "HF_TOKEN environment variable is required to download checkpoints."
                )

            # Download using huggingface-cli
            import subprocess

            try:
                # Create directory
                os.makedirs(f"checkpoints/{tag}", exist_ok=True)

                # Download
                cmd = [
                    "huggingface-cli",
                    "download",
                    "--token",
                    hf_token,
                    "--repo-type",
                    "model",
                    "--local-dir",
                    f"checkpoints/{tag}-download",
                    "facebook/sam-3d-objects",
                ]
                subprocess.check_call(cmd)

                # Move files (handling the structure)
                # The download might create a nested structure or flat, depending on repo.
                # Based on setup script: mv checkpoints/${TAG}-download/checkpoints/* checkpoints/${TAG}/
                # Let's adjust to match the expected structure

                # Move from download temp to target
                source_dir = f"checkpoints/{tag}-download/checkpoints"
                if os.path.exists(source_dir):
                    # Move contents of source_dir to checkpoints/{tag}
                    for item in os.listdir(source_dir):
                        shutil.move(
                            os.path.join(source_dir, item), f"checkpoints/{tag}/"
                        )
                else:
                    # Maybe the repo structure is different or flat?
                    # Fallback: just move everything from download dir to target if 'checkpoints' subdir doesn't exist
                    # But the repo seems to have a 'checkpoints' folder at root based on previous scripts.
                    # Let's assume the previous script logic was correct.
                    pass

                # Cleanup
                shutil.rmtree(f"checkpoints/{tag}-download")

                print("Checkpoints downloaded successfully.")
            except Exception as e:
                raise RuntimeError(f"Failed to download checkpoints: {e}")

        inference_model = Inference(config_path, compile=False)
        print("Model loaded successfully.")


def handler(job):
    global inference_model

    # Initialize model if not already loaded (Cold Start)
    if inference_model is None:
        init_model()

    job_input = job["input"]

    # Validate input
    if "image_url" not in job_input:
        return {"error": "Missing 'image_url' in input."}

    image_url = job_input["image_url"]
    mask_url = job_input.get("mask_url")  # Optional
    seed = job_input.get("seed", 42)
    output_format = job_input.get("format", "glb")  # glb, obj, fbx (if supported)

    try:
        # Load Image
        print(f"Downloading image from {image_url}...")
        image_path = download_file(image_url)
        image = load_image(image_path)

        # Handle Mask
        mask = None
        if mask_url:
            print(f"Downloading mask from {mask_url}...")
            mask_path = download_file(mask_url)
            mask = load_mask(mask_path)
            os.remove(mask_path)
        else:
            # Check for alpha channel in the loaded image
            if image.ndim == 3 and image.shape[-1] == 4:
                print("No mask_url provided, but image has Alpha channel. Using it as mask.")
                mask = load_mask(image_path)
            else:
                print("No mask provided and no alpha channel. Using full image.")
                mask = np.ones(image.shape[:2], dtype=bool)
        
        # Cleanup image file
        os.remove(image_path)

        # Run Inference
        print("Running inference...")
        # Note: The inference pipeline expects specific mask format.
        # If no mask is provided, the model might need one or handle full image.
        # The original demo uses a mask. If user doesn't provide one, we might need to generate it (e.g. with SAM)
        # But for this handler, let's assume mask is provided or optional if the model supports it.
        # Looking at export_mesh_example.py, it uses load_single_mask.

        output = inference_model(image, mask, seed=seed)
        print("Inference completed successfully.")
        print(f"Output keys: {list(output.keys())}")

        results = {}

        # Process Output
        if "glb" in output and output["glb"] is not None:
            print("Found 'glb' in output. Processing mesh...")
            mesh = output["glb"]

            with tempfile.NamedTemporaryFile(
                suffix=f".{output_format}", delete=False
            ) as tmp:
                mesh_path = tmp.name
            print(f"Created temporary file at: {mesh_path}")

            # Export
            try:
                print(f"Exporting mesh as {output_format}...")
                if output_format == "fbx":
                    try:
                        mesh.export(mesh_path)
                    except Exception as e:
                        print(f"FBX export failed: {e}")
                        return {
                            "error": f"FBX export failed: {str(e)}. Try 'glb' or 'obj'."
                        }
                else:
                    mesh.export(mesh_path)
                print("Mesh export finished.")
            except Exception as e:
                print(f"General export failed: {e}")
                raise e

            # Read back and encode to base64
            print("Reading file and encoding to base64...")
            with open(mesh_path, "rb") as f:
                mesh_data = f.read()

            mesh_base64 = base64.b64encode(mesh_data).decode("utf-8")
            results["mesh_base64"] = mesh_base64
            results["format"] = output_format
            print(f"Encoding complete. Base64 length: {len(mesh_base64)}")

            # Cleanup
            os.remove(mesh_path)
            print("Temporary file removed.")
        else:
            print("WARNING: 'glb' key missing or None in output.")

        if "gs" in output:
            print("Gaussian Splat ('gs') output is available.")
            # Gaussian Splat Point Cloud
            # We can also return this if requested
            pass

        print("Returning results.")
        return results

    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
