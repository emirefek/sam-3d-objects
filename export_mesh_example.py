import sys
import os
import trimesh

# Add notebook folder to path to import inference
sys.path.append("notebook")
from inference import Inference, load_image, load_single_mask

def main():
    # Load model
    tag = "hf"
    config_path = f"checkpoints/{tag}/pipeline.yaml"
    
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        return

    print("Loading model...")
    inference = Inference(config_path, compile=False)

    # Load image (using the example from demo.py)
    image_path = "notebook/images/shutterstock_stylish_kidsroom_1640806567/image.png"
    mask_path = "notebook/images/shutterstock_stylish_kidsroom_1640806567"
    mask_index = 14
    
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    print("Loading image and mask...")
    image = load_image(image_path)
    mask = load_single_mask(mask_path, index=mask_index)

    # Run model
    print("Running inference...")
    # The pipeline returns a dictionary with 'gs' (Gaussian Splats) and 'glb' (Trimesh object)
    # by default because decode_formats=["gaussian", "mesh"] in InferencePipeline
    output = inference(image, mask, seed=42)

    # Export Gaussian Splat (Point Cloud)
    if "gs" in output:
        print("Saving Gaussian Splat to splat.ply...")
        output["gs"].save_ply("splat.ply")

    # Export Mesh
    if "glb" in output and output["glb"] is not None:
        mesh = output["glb"]
        print(f"Mesh generated with {len(mesh.vertices)} vertices and {len(mesh.faces)} faces.")
        
        # Save as GLB
        print("Saving mesh to model.glb...")
        mesh.export("model.glb")
        
        # Save as OBJ
        print("Saving mesh to model.obj...")
        mesh.export("model.obj")
        
        # Try saving as FBX (might require extra dependencies or might not work directly with trimesh)
        try:
            print("Attempting to save mesh to model.fbx...")
            mesh.export("model.fbx")
            print("Successfully saved model.fbx")
        except Exception as e:
            print(f"Could not export directly to FBX using trimesh: {e}")
            print("You can convert the generated model.glb or model.obj to .fbx using Blender or an online converter.")
            
    else:
        print("No mesh found in output. Ensure decode_formats includes 'mesh'.")

if __name__ == "__main__":
    main()
