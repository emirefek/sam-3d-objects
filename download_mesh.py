import argparse
import requests
import base64
import os
import time
import sys

def get_job_status(endpoint_id, request_id, api_key):
    url = f"https://api.runpod.ai/v2/{endpoint_id}/status/{request_id}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()

def main():
    parser = argparse.ArgumentParser(description="Download mesh from RunPod job result.")
    parser.add_argument("request_id", help="The ID of the job to retrieve.")
    parser.add_argument("--endpoint_id", required=True, help="The RunPod Endpoint ID (e.g., vllm-xyz).")
    parser.add_argument("--api_key", default=os.environ.get("RUNPOD_API_KEY"), help="RunPod API Key (or set RUNPOD_API_KEY env var).")
    parser.add_argument("--output", "-o", default="output.glb", help="Output filename (default: output.glb).")
    
    args = parser.parse_args()
    
    if not args.api_key:
        print("Error: API Key is required. Provide --api_key or set RUNPOD_API_KEY environment variable.")
        sys.exit(1)

    print(f"Checking status for job {args.request_id} on endpoint {args.endpoint_id}...")
    
    while True:
        try:
            data = get_job_status(args.endpoint_id, args.request_id, args.api_key)
        except requests.exceptions.RequestException as e:
            print(f"API Request failed: {e}")
            sys.exit(1)

        status = data.get("status")
        
        if status == "COMPLETED":
            print("Job completed!")
            output = data.get("output", {})
            
            # Check for handler-level errors
            if isinstance(output, dict) and "error" in output:
                print(f"Job returned an error: {output['error']}")
                sys.exit(1)
                
            mesh_base64 = output.get("mesh_base64")
            file_format = output.get("format", "glb")
            
            if not mesh_base64:
                print("Error: No 'mesh_base64' found in output.")
                print(f"Full output: {output}")
                sys.exit(1)
            
            # Determine output filename
            filename = args.output
            # If user didn't specify a custom name (used default), update extension based on format
            if args.output == "output.glb" and file_format != "glb":
                filename = f"output.{file_format}"
            
            print(f"Decoding and saving to {filename}...")
            try:
                mesh_data = base64.b64decode(mesh_base64)
                with open(filename, "wb") as f:
                    f.write(mesh_data)
                print(f"Successfully saved {len(mesh_data)} bytes to {filename}")
            except Exception as e:
                print(f"Failed to decode/save file: {e}")
                sys.exit(1)
            
            break
            
        elif status in ["FAILED", "TIMED_OUT"]:
            print(f"Job failed with status: {status}")
            print(f"Error: {data.get('error')}")
            sys.exit(1)
            
        else:
            print(f"Status: {status}. Waiting...")
            time.sleep(2)

if __name__ == "__main__":
    main()
