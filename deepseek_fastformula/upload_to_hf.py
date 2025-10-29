"""
Upload trained model to HuggingFace Hub

Usage:
    python upload_to_hf.py --model_path ./my_model --repo_name angelchencn/deepseek-coder-fastformula
"""

import argparse
from huggingface_hub import HfApi, create_repo
import os

def upload_model(model_path, repo_name, private=False):
    """Upload model to HuggingFace Hub"""
    
    print(f"Uploading model from {model_path} to {repo_name}")
    
    # Create repository if it doesn't exist
    try:
        create_repo(repo_name, private=private, exist_ok=True)
        print(f"Repository created/verified: https://huggingface.co/{repo_name}")
    except Exception as e:
        print(f"Error creating repo: {e}")
        return
    
    # Upload all files in the model directory
    api = HfApi()
    
    try:
        api.upload_folder(
            folder_path=model_path,
            repo_id=repo_name,
            repo_type="model",
        )
        print(f"\n✅ Model uploaded successfully!")
        print(f"🔗 View at: https://huggingface.co/{repo_name}")
        
    except Exception as e:
        print(f"❌ Error uploading model: {e}")


def main():
    parser = argparse.ArgumentParser(description="Upload model to HuggingFace Hub")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--repo_name", type=str, required=True, help="HuggingFace repo name (username/model-name)")
    parser.add_argument("--private", action="store_true", help="Make repository private")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model path {args.model_path} does not exist")
        return
    
    upload_model(args.model_path, args.repo_name, args.private)


if __name__ == "__main__":
    main()